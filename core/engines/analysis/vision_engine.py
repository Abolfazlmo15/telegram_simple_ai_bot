"""Vision/Image processing engine – tries Hugging Face (many models) then OpenRouter (dynamic)."""
import logging
import asyncio
import base64
import json
import time
from typing import Dict, Tuple, Optional, Any, List
import httpx
from PIL import Image
import io
from core.config import Config
from core.managers.vision_model_manager import VisionModelManager
from core.utils.image_processor import ImageProcessor
from core.managers.user_data_manager import UserDataManager

logger = logging.getLogger(__name__)


class VisionEngine:
    def __init__(self, user_data_manager: Optional[UserDataManager] = None):
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.model_manager = VisionModelManager()
        self.image_processor = ImageProcessor(max_size=128, quality=60)
        self._client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False
        self.user_data_manager = user_data_manager

        # Hugging Face token
        self._hf_token = Config.HUGGINGFACE_TOKEN
        self._fallback_timeout = Config.VISION_FALLBACK_TIMEOUT

        # OpenRouter blacklisting (for models that fail)
        self._model_failures: Dict[str, float] = {}
        self._blacklist_ttl = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS
        self._non_retryable_errors = {404, 400, 402, 429}

        # We'll use the main client for all calls

        logger.info("🔷 VisionEngine initialized (HF models → OpenRouter dynamic)")

    async def initialize(self) -> bool:
        logger.info("🔷 VisionEngine.initialize: START")
        try:
            self.model_manager.start()
            logger.info("🔷 VisionModelManager started")
            timeout = httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=10.0, pool=10.0)
            try:
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=True,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
                logger.info("🔷 Main HTTP client created (HTTP/2)")
            except Exception:
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=False,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
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

    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Tuple[str, str, int]:
        logger.info("🔷 VisionEngine.process: START")
        if not self.is_initialized:
            raise RuntimeError("Vision engine not initialized")

        user_id = context.get('user_id') if context else None
        username = context.get('username') if context else None
        query_text = context.get('query_text', '').strip() if context else ''
        priority_list = context.get('priority_list') if context else None
        logger.info(f"🔷 user_id: {user_id}, query: {query_text[:50] if query_text else '(empty)'}...")

        try:
            processed_image = await self.image_processor.process_image(input_data)
            logger.info(f"🔷 Image processed: {processed_image.size[0]}x{processed_image.size[1]}")
            pil_image = processed_image
        except Exception as e:
            logger.error(f"❌ Image processing failed: {e}", exc_info=True)
            raise ValueError(f"Could not process image: {e}")

        # Convert to bytes and base64 once
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # ============================================================
        # 1. TRY HUGGING FACE (many models, ordered by reliability)
        # ============================================================
        if self._hf_token and len(self._hf_token) > 10:
            logger.info("🔷 Trying Hugging Face (many models)...")
            hf_models = Config.HUGGINGFACE_VISION_MODELS
            for model in hf_models:
                if self._is_model_blacklisted(model):
                    logger.info(f"Skipping blacklisted HF model: {model}")
                    continue
                try:
                    logger.info(f"🔷 HF model: {model}")
                    response, model_used, tokens = await self._call_huggingface(pil_image, query_text, model)
                    if response and len(response) > 5:
                        logger.info(f"✅ Hugging Face succeeded with {model_used}")
                        return response, model_used, tokens
                except httpx.ConnectError:
                    logger.warning(f"HF {model}: Connection error (DNS/network).")
                    # If we get a ConnectError, it's likely a DNS issue; break to avoid wasting time on all models.
                    # But we'll continue to next model because maybe the next one has a different endpoint? Usually DNS is global.
                    # We'll try a few more before breaking.
                    continue
                except Exception as e:
                    logger.warning(f"HF {model} failed: {e}")
                    # Mark as failed to avoid retrying
                    self._mark_model_failure(model)
        else:
            logger.warning("🔷 Hugging Face token not set. Skipping.")

        # ============================================================
        # 2. TRY OPENROUTER (dynamic + fallback list)
        # ============================================================
        logger.info("🔷 Trying OpenRouter (dynamic models)...")
        try:
            # Get dynamic models from manager
            dynamic_models = self.model_manager.get_available_models()
            # Combine with hardcoded fallback (if any)
            fallback_models = Config.OPENROUTER_VISION_MODELS
            # Build model list: user priority -> dynamic -> fallback
            model_list = self._get_model_list(priority_list, dynamic_models, fallback_models)
            if model_list:
                result = await self._try_openrouter_models(model_list, base64_image, query_text)
                if result:
                    response, model, tokens = result
                    logger.info(f"✅ OpenRouter succeeded with {model}")
                    return response, model, tokens
        except Exception as e:
            logger.warning(f"OpenRouter failed: {e}")

        # ============================================================
        # 3. ULTIMATE FALLBACK: PIL METADATA
        # ============================================================
        logger.warning("All providers failed. Using PIL metadata fallback.")
        width, height = pil_image.size
        mode = pil_image.mode
        format_name = pil_image.format or "Unknown"
        response_text = (
            f"📷 *Image Metadata*\n\n"
            f"• Format: {format_name}\n"
            f"• Size: {width} x {height} pixels\n"
            f"• Mode: {mode}\n\n"
            "Could not get AI description. Please check your Hugging Face token or OpenRouter API key."
        )
        return response_text, "pil-fallback", len(response_text) // 4

    # ---------- HUGGING FACE ----------
    async def _call_huggingface(self, pil_image: Image.Image, query: str, model: str) -> Tuple[str, str, int]:
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self._hf_token}"}

        # Send image as bytes with proper content type
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        # Try raw image (most models accept this)
        resp = await self._client.post(
            url,
            headers=headers,
            content=image_bytes,
            timeout=self._fallback_timeout
        )
        resp.raise_for_status()
        data = resp.json()

        # Parse response
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "generated_text" in data[0]:
                response_text = data[0]["generated_text"]
            else:
                response_text = str(data[0])
        else:
            response_text = str(data)

        tokens_used = len(response_text) // 4
        return response_text, f"huggingface-{model}", tokens_used

    # ---------- OPENROUTER ----------
    def _get_model_list(self, priority_list: Optional[List[str]], dynamic_models: List[str], fallback_models: List[str]) -> List[str]:
        ordered = []
        seen = set()

        # 1. User priority (if any)
        if priority_list:
            for m in priority_list:
                if m not in seen and not self._is_model_blacklisted(m):
                    ordered.append(m)
                    seen.add(m)

        # 2. Dynamic models from manager (live)
        for m in dynamic_models:
            if m not in seen and not self._is_model_blacklisted(m):
                ordered.append(m)
                seen.add(m)

        # 3. Hardcoded fallback
        for m in fallback_models:
            if m not in seen and not self._is_model_blacklisted(m):
                ordered.append(m)
                seen.add(m)

        return ordered

    async def _try_openrouter_models(self, model_list: List[str], base64_image: str, query: str) -> Optional[Tuple[str, str, int]]:
        # Build system prompt (generic description)
        system_prompt = self._build_vision_prompt(query)

        for model in model_list:
            for attempt in range(Config.HTTP_MAX_RETRIES):
                try:
                    logger.info(f"🔷 OR model: {model} (attempt {attempt+1})")
                    response, tokens_used = await self._call_openrouter(model, base64_image, query, system_prompt)
                    if response and len(response) > 5:
                        return (response, model, tokens_used)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in self._non_retryable_errors:
                        self._mark_model_failure(model)
                        break
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
                except Exception as e:
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
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
            "providers": ["huggingface", "openrouter"],
            "hf_token": bool(self._hf_token),
            "openrouter_key": bool(self.api_key),
            "blacklisted_models": list(self._model_failures.keys())
        }