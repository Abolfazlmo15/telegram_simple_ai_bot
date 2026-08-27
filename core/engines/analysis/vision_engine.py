"""Vision/Image processing engine – tries OpenRouter (free verified) then Hugging Face via proxy."""
import logging
import asyncio
import base64
import io
import time
from typing import Dict, Tuple, Optional, Any, List, Callable, Coroutine

import httpx
from PIL import Image

from core.config import Config
from core.managers.vision_model_manager import VisionModelManager
from core.utils.image_processor import ImageProcessor
from core.managers.user_data_manager import UserDataManager
from core.managers.proxy_manager import ProxyManager

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Custom exception for restart
# ------------------------------------------------------------------
class RestartSearchException(Exception):
    pass

# ------------------------------------------------------------------
# Retry helper with exponential backoff
# ------------------------------------------------------------------
async def _retry_async(func, *args, max_retries=2, base_delay=0.5, **kwargs):
    """Retry an async function with exponential backoff for network errors."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + (0.1 * attempt)
                await asyncio.sleep(delay)
                continue
            raise
        except Exception:
            # Non‑network errors are re‑raised immediately
            raise
    raise last_exc


class VisionEngine:
    def __init__(self, user_data_manager: Optional[UserDataManager] = None,
                 proxy_manager: Optional[ProxyManager] = None):
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.model_manager = VisionModelManager(proxy_manager=proxy_manager)
        # Increased max_size and quality for better analysis
        self.image_processor = ImageProcessor(max_size=512, quality=85)
        self._client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False
        self.user_data_manager = user_data_manager
        self.proxy_manager = proxy_manager

        self._hf_token = Config.HUGGINGFACE_TOKEN
        self._fallback_timeout = Config.VISION_FALLBACK_TIMEOUT

        self._model_failures: Dict[str, float] = {}
        self._blacklist_ttl = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS
        self._non_retryable_errors = {404, 400, 402, 429}

        # Reliable HF models (used as fallback)
        self._hf_reliable_models = Config.HUGGINGFACE_VISION_MODELS

        # Timeout configuration
        self.search_timeout = Config.VISION_SEARCH_TIMEOUT_SECONDS
        self.restart_timeout = Config.GLOBAL_RESTART_TIMEOUT_SECONDS

        logger.info("🔷 VisionEngine initialized (OpenRouter free verified → Hugging Face via proxy)")

    async def initialize(self) -> bool:
        logger.info("🔷 VisionEngine.initialize: START")
        try:
            self.model_manager.start()
            logger.info("🔷 VisionModelManager started")
            timeout = httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=10.0, pool=10.0)

            # Build client with proxy if available
            client_kwargs = {
                "timeout": timeout,
                "limits": httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE,
                                       max_keepalive_connections=5)
            }
            proxy_url = None
            if self.proxy_manager:
                proxy_url = self.proxy_manager.get_proxy()
                if proxy_url:
                    client_kwargs["proxy"] = proxy_url
                    logger.info(f"🔷 VisionEngine using proxy: {proxy_url}")

            try:
                self._client = httpx.AsyncClient(**client_kwargs, http2=True)
                logger.info("🔷 Main HTTP client created (HTTP/2)")
            except Exception:
                logger.warning("HTTP/2 failed, falling back to HTTP/1.1")
                self._client = httpx.AsyncClient(**client_kwargs, http2=False)
                logger.info("🔷 Main HTTP client created (HTTP/1.1)")

            self.is_initialized = True
            logger.info("🔷 VisionEngine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ VisionEngine.initialize failed: {e}", exc_info=True)
            return False

    async def shutdown(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self.model_manager.stop()
        self.is_initialized = False
        logger.info("🔷 VisionEngine shutdown complete")

    def _is_model_blacklisted(self, model: str) -> bool:
        if model in self._model_failures:
            elapsed = time.time() - self._model_failures[model]
            if elapsed < self._blacklist_ttl:
                return True
            else:
                del self._model_failures[model]
        return False

    def _mark_model_failure(self, model: str):
        self._model_failures[model] = time.time()
        logger.info(f"🚫 Blacklisted {model} for {self._blacklist_ttl}s")

    def _clear_blacklist(self):
        self._model_failures.clear()
        logger.info("🧹 VisionEngine blacklist cleared (restart)")

    async def process(
        self,
        input_data: Any,
        context: Optional[Dict] = None,
        status_callback: Optional[Callable[[str, bool], Coroutine]] = None
    ) -> Tuple[str, str, int]:
        """Process an image (bytes or PIL Image) and return description."""
        logger.info("🔷 VisionEngine.process: START")
        if not self.is_initialized:
            raise RuntimeError("Vision engine not initialized")

        if asyncio.current_task().cancelled():
            raise asyncio.CancelledError

        user_id = context.get('user_id') if context else None
        query_text = context.get('query_text', '').strip() if context else ''
        priority_list = context.get('priority_list') if context else None
        logger.info(f"🔷 user_id: {user_id}, query: {query_text[:50] if query_text else '(empty)'}...")

        # --- Process the image ---
        try:
            pil_image = await self.image_processor.process_image(input_data)
            logger.info(f"🔷 Image processed: {pil_image.size[0]}x{pil_image.size[1]}")
        except Exception as e:
            logger.error(f"❌ Image processing failed: {e}", exc_info=True)
            # Return a user-friendly error instead of raising
            return (
                "❌ *Image Processing Failed*\n\n"
                "The image could not be processed. Please try a different image.\n\n"
                f"_Error: {str(e) or 'Unknown error'}_",
                "image-processing-error",
                0
            )

        # Convert to base64 for OpenRouter
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # --- Timeout & restart logic ---
        start_time = time.time()
        timer_task = asyncio.create_task(
            self._search_timer(
                status_callback,
                search_timeout=self.search_timeout,
                restart_timeout=self.restart_timeout,
                start_time=start_time
            )
        )

        try:
            while True:
                if asyncio.current_task().cancelled():
                    raise asyncio.CancelledError

                # 1. Try OpenRouter
                dynamic_models = self.model_manager.get_available_models()
                verified_free_models = [
                    "google/gemini-2.0-flash-exp:free",
                    "meta-llama/llama-3.2-11b-vision-instruct:free",
                    "qwen/qwen-2-vl-7b-instruct:free",
                    "openai/gpt-4o-mini:free"
                ]
                model_list = self._get_model_list(priority_list, dynamic_models, verified_free_models)
                logger.info(f"🔷 OpenRouter model list (first 5): {model_list[:5] if model_list else 'EMPTY'}")

                if model_list and self.api_key and len(self.api_key) > 10:
                    result = await self._try_openrouter_models(model_list, base64_image, query_text)
                    if result:
                        timer_task.cancel()
                        return result
                else:
                    logger.warning("🔷 OpenRouter skipped: no models or invalid API key")

                # 2. Fallback: Hugging Face
                result = await self._try_huggingface(pil_image, query_text)
                if result:
                    timer_task.cancel()
                    return result

                # 3. Check if restart was triggered – safely
                if timer_task.done():
                    exc = timer_task.exception()
                    if exc and isinstance(exc, RestartSearchException):
                        logger.info("Restart triggered, clearing blacklist and refreshing model list.")
                        self._clear_blacklist()
                        try:
                            self.model_manager.get_available_models(force_refresh=True)
                        except Exception:
                            pass
                        continue
                else:
                    # Timer still running, no restart yet
                    break

            # All providers failed → PIL metadata fallback
            logger.warning("All providers failed. Using PIL metadata fallback.")
            width, height = pil_image.size
            mode = pil_image.mode
            format_name = pil_image.format or "Unknown"
            response_text = (
                f"📷 *Image Analysis Unavailable*\n\n"
                f"• Format: {format_name}\n"
                f"• Size: {width} x {height} pixels\n"
                f"• Mode: {mode}\n\n"
                "I couldn't get an AI description. Please check:\n"
                "• Your OpenRouter API key (must be valid)\n"
                "• Your Hugging Face token (if using HF)\n"
                "• Your internet connection\n\n"
                "_Try again in a moment._"
            )
            timer_task.cancel()
            return response_text, "pil-fallback", len(response_text) // 4

        except asyncio.CancelledError:
            timer_task.cancel()
            logger.info("VisionEngine.process cancelled, re-raising")
            raise
        except Exception as e:
            timer_task.cancel()
            logger.error(f"❌ VisionEngine.process unexpected error: {e}", exc_info=True)
            # Return a friendly error instead of raising
            return (
                f"❌ *Vision Error*\n\nSomething went wrong while analyzing the image.\n\n"
                f"_Error: {str(e) or 'Unknown error'}_\n\nPlease try again later.",
                "vision-error",
                0
            )
        finally:
            if not timer_task.done():
                timer_task.cancel()

    async def _search_timer(
        self,
        status_callback: Optional[Callable[[str, bool], Coroutine]],
        search_timeout: int,
        restart_timeout: int,
        start_time: float
    ):
        """Background timer to show progress and trigger restart."""
        try:
            elapsed = 0
            while elapsed < search_timeout:
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time
                if elapsed >= search_timeout:
                    break
            if status_callback:
                await status_callback("🖼️ *AI Models Scarce, Please Wait ...*", edit=True)
                logger.info("⏳ Sent 'wait' message for vision engine")

            while elapsed < restart_timeout:
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time
                if elapsed >= restart_timeout:
                    break

            if status_callback:
                await status_callback("🔄 *Restarting search...*", edit=True)
            raise RestartSearchException("Restart search due to timeout")
        except asyncio.CancelledError:
            pass
        except RestartSearchException:
            raise
        except Exception as e:
            logger.error(f"Vision timer error: {e}")

    # ---------- HUGGING FACE (fallback) ----------
    async def _try_huggingface(self, pil_image: Image.Image, query_text: str) -> Optional[Tuple[str, str, int]]:
        if not self._hf_token or len(self._hf_token) <= 10:
            logger.info("🔷 Hugging Face token not set. Skipping.")
            return None

        logger.info("🔷 Trying Hugging Face (fallback models via proxy)...")
        for model in self._hf_reliable_models:
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError

            if self._is_model_blacklisted(model):
                logger.info(f"Skipping blacklisted HF model: {model}")
                continue

            try:
                logger.info(f"🔷 HF model: {model}")
                response, model_used, tokens = await _retry_async(
                    self._call_huggingface,
                    pil_image, query_text, model,
                    max_retries=2
                )
                if response and len(response) > 5:
                    logger.info(f"✅ Hugging Face succeeded with {model_used}")
                    return response, model_used, tokens
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                logger.warning(f"HF {model} network error after retries: {e}. Blacklisting.")
                self._mark_model_failure(model)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 402:
                    logger.warning(f"HF {model} requires payment (402). Skipping.")
                elif e.response.status_code in self._non_retryable_errors:
                    logger.warning(f"HF {model} non-retryable error {e.response.status_code}. Blacklisting.")
                    self._mark_model_failure(model)
                else:
                    logger.warning(f"HF {model} HTTP error: {e}")
            except Exception as e:
                logger.warning(f"HF {model} failed: {e}")
                self._mark_model_failure(model)
        return None

    async def _call_huggingface(self, pil_image: Image.Image, query: str, model: str) -> Tuple[str, str, int]:
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self._hf_token}"}

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        resp = await self._client.post(
            url,
            headers=headers,
            content=image_bytes,
            timeout=self._fallback_timeout
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "generated_text" in data[0]:
                response_text = data[0]["generated_text"]
            else:
                response_text = str(data[0])
        else:
            response_text = str(data)

        tokens_used = len(response_text) // 4
        return response_text, f"huggingface-{model}", tokens_used

    # ---------- OPENROUTER (primary) ----------
    def _get_model_list(self, priority_list: Optional[List[str]], dynamic_models: List[str], fallback_models: List[str]) -> List[str]:
        ordered = []
        seen = set()
        if priority_list:
            for m in priority_list:
                if m not in seen and not self._is_model_blacklisted(m):
                    ordered.append(m)
                    seen.add(m)
        for m in dynamic_models:
            if m not in seen and not self._is_model_blacklisted(m):
                ordered.append(m)
                seen.add(m)
        for m in fallback_models:
            if m not in seen and not self._is_model_blacklisted(m):
                ordered.append(m)
                seen.add(m)
        if not ordered:
            logger.warning("OpenRouter model list is empty after filtering, using fallback.")
            return list(dict.fromkeys(fallback_models))
        return ordered

    async def _try_openrouter_models(self, model_list: List[str], base64_image: str, query: str) -> Optional[Tuple[str, str, int]]:
        logger.info("🔷 Trying OpenRouter (free models) – primary...")
        system_prompt = self._build_vision_prompt(query)

        # If API key is missing, skip immediately
        if not self.api_key or len(self.api_key) < 10:
            logger.warning("🔷 OpenRouter API key missing or invalid. Skipping.")
            return None

        for model in model_list:
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError

            for attempt in range(Config.HTTP_MAX_RETRIES):
                try:
                    logger.info(f"🔷 OR model: {model} (attempt {attempt+1})")
                    response, tokens_used = await _retry_async(
                        self._call_openrouter,
                        model, base64_image, query, system_prompt,
                        max_retries=2
                    )
                    if response and len(response) > 5:
                        logger.info(f"✅ OpenRouter succeeded with {model} (tokens: {tokens_used})")
                        return response, model, tokens_used
                    else:
                        logger.warning(f"OpenRouter model {model} returned empty/short response")
                        self._mark_model_failure(model)
                        break
                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    logger.warning(f"OR {model} network error: {e}")
                    if attempt == Config.HTTP_MAX_RETRIES - 1:
                        self._mark_model_failure(model)
                    else:
                        await asyncio.sleep(0.5)
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    if status in self._non_retryable_errors:
                        logger.warning(f"Non-retryable error {status} for {model}, blacklisting.")
                        self._mark_model_failure(model)
                        break
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
                    else:
                        self._mark_model_failure(model)
                except Exception as e:
                    logger.warning(f"OR {model} attempt {attempt+1} failed: {e}")
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
                    else:
                        self._mark_model_failure(model)
        return None

    async def _call_openrouter(self, model: str, base64_image: str, query: str, system_prompt: str) -> Tuple[str, int]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.BOT_REPO_URL,
            "X-Title": Config.BOT_NAME
        }
        user_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
        if query and query.lower() not in ["describe this image", "describe the image", "what is in this image"]:
            user_content.append({"type": "text", "text": query})

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        payload = {"model": model, "messages": messages, "max_tokens": 1500}

        resp = await self._client.post(self.base_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        response_text = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return response_text, tokens_used

    # ---------- PROMPT BUILDER ----------
    def _build_vision_prompt(self, user_query: str) -> str:
        generic_phrases = [
            "describe this image", "describe the image",
            "what is in this image", "what is in the image",
            "image analysis", "analyze this image", "analyze the image"
        ]
        if not user_query or user_query.lower() in generic_phrases:
            return (
                "You are an expert visual analyst. Provide a detailed, accurate description of the image.\n"
                "If the image contains text, transcribe it accurately. Describe objects, people, colors, layout, and context.\n"
                "Keep the description clear, well-structured, and use plain text (no markdown)."
            )
        else:
            return (
                "You are an expert visual analyst. The user has asked a specific question about the image.\n"
                "Use the image content to answer the user's question accurately and thoroughly.\n"
                "If the question is about an opinion (e.g., 'what is beautiful'), give your subjective analysis based on the image.\n"
                "Provide a clear, well-structured response in plain text (no markdown).\n\n"
                f"User question: {user_query}"
            )

    def get_engine_info(self) -> Dict:
        return {
            "type": "VisionEngine",
            "initialized": self.is_initialized,
            "providers": ["openrouter", "huggingface"],
            "hf_token": bool(self._hf_token),
            "openrouter_key": bool(self.api_key),
            "blacklisted_models": list(self._model_failures.keys())
        }