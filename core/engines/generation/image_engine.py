"""Image generation engine with multi-tier fallback (pollinations → Hugging Face → OpenRouter) and blacklisting."""
import logging
import asyncio
import time
import httpx
from typing import Dict, Tuple, Optional, Any, List, Callable, Coroutine
from urllib.parse import quote
from core.config import Config
from core.managers.user_data_manager import UserDataManager
from core.managers.image_model_manager import ImageModelManager
from core.utils.network import retry_async

logger = logging.getLogger(__name__)

# Custom exception for restart
class RestartSearchException(Exception):
    pass


class ImageGenerationEngine:
    """
    Multi-tier image generation engine.
    Tiers (in order): pollinations, huggingface, openrouter.
    Each tier has its own set of models; the engine falls back to the next tier on failure.
    """
    def __init__(self, user_data_manager: Optional[UserDataManager] = None):
        self.user_data_manager = user_data_manager
        self.api_key = Config.OPENROUTER_API_KEY
        self.image_base_url = Config.OPENROUTER_IMAGE_GENERATION_URL
        self.pollinations_base_url = Config.POLLINATIONS_IMAGE_URL
        self.huggingface_token = Config.HUGGINGFACE_TOKEN
        self.huggingface_url = Config.HUGGINGFACE_IMAGE_URL
        self._client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False

        # Tier definitions
        self.tiers = Config.IMAGE_GENERATION_PRIORITY

        # Model lists per tier
        self.pollinations_models = Config.POLLINATIONS_MODELS
        self.huggingface_models = Config.HUGGINGFACE_MODELS
        self.openrouter_models = Config.OPENROUTER_IMAGE_MODELS

        # Blacklisting
        self._model_failures: Dict[str, float] = {}
        self._blacklist_ttl = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS
        self._non_retryable_errors = {404, 400, 402, 429}

        # Image generation defaults
        self.default_size = Config.IMAGE_GENERATION_SIZE
        self.default_quality = Config.IMAGE_GENERATION_QUALITY

        # Timeout configuration
        self.search_timeout = Config.VISION_SEARCH_TIMEOUT_SECONDS
        self.restart_timeout = Config.GLOBAL_RESTART_TIMEOUT_SECONDS

        # Max prompt length for Pollinations (URL length limit)
        self.pollinations_max_prompt_length = 2000

        self.model_manager = ImageModelManager()
        logger.info("🖼️ ImageGenerationEngine initialized with multi-tier fallback, blacklisting & timeout/restart")

    async def initialize(self) -> bool:
        try:
            self.model_manager.start()
            timeout = httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=30.0, pool=10.0)
            try:
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=True,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
                logger.info("🖼️ ImageGenerationEngine HTTP/2 client created")
            except Exception:
                logger.warning("HTTP/2 failed for image engine, falling back to HTTP/1.1")
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=False,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
                logger.info("🖼️ ImageGenerationEngine HTTP/1.1 client created")
            self.is_initialized = True
            logger.info(f"🖼️ Available image generation tiers: {self.tiers}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize ImageGenerationEngine: {e}", exc_info=True)
            return False

    async def shutdown(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self.model_manager.stop()
        self.is_initialized = False
        logger.info("🖼️ ImageGenerationEngine shutdown complete")

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
        logger.info(f"🚫 Blacklisted image model {model} for {self._blacklist_ttl}s")

    def _clear_blacklist(self):
        self._model_failures.clear()
        logger.info("🧹 ImageEngine blacklist cleared (restart)")

    def _get_model_list(self, tier: str, preferred_models: Optional[List[str]] = None) -> List[str]:
        if tier == "pollinations":
            base = self.pollinations_models
        elif tier == "huggingface":
            base = self.huggingface_models
        elif tier == "openrouter":
            base = self.openrouter_models
        else:
            return []

        if preferred_models:
            ordered = []
            for m in preferred_models:
                if m in base and not self._is_model_blacklisted(m):
                    ordered.append(m)
            for m in base:
                if m not in ordered and not self._is_model_blacklisted(m):
                    ordered.append(m)
            return ordered
        else:
            return [m for m in base if not self._is_model_blacklisted(m)]

    async def generate(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        status_callback: Optional[Callable[[str, bool], Coroutine]] = None
    ) -> Tuple[bytes, str, int]:
        if not self.is_initialized:
            raise RuntimeError("Image generation engine not initialized")
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Empty prompt provided for image generation")

        user_id = context.get('user_id') if context else None
        username = context.get('username') if context else None
        preferred_models = context.get('recommended_models', []) if context else []
        priority_tiers = await self._get_tier_priority(user_id, username)
        size = Config.IMAGE_GENERATION_SIZE
        quality = Config.IMAGE_GENERATION_QUALITY

        last_error = None
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
                restart_triggered = False
                for tier in priority_tiers:
                    logger.info(f"🖼️ Trying tier: {tier}")
                    models = self._get_model_list(tier, preferred_models)
                    if not models:
                        logger.warning(f"No available models for tier {tier}")
                        continue

                    # For Pollinations, truncate prompt to avoid URL length issues
                    if tier == "pollinations" and len(prompt) > self.pollinations_max_prompt_length:
                        truncated_prompt = prompt[:self.pollinations_max_prompt_length]
                        logger.info(f"📝 Truncated prompt for Pollinations to {self.pollinations_max_prompt_length} chars")
                        result = await self._try_tier_models(tier, models, truncated_prompt, size, quality)
                    else:
                        result = await self._try_tier_models(tier, models, prompt, size, quality)

                    if result:
                        image_bytes, model = result
                        logger.info(f"✅ Image generated via {tier} using {model}")
                        timer_task.cancel()
                        if user_id and self.user_data_manager:
                            try:
                                matrix_info = await self.user_data_manager.save_image_matrix(user_id, username, image_bytes)
                                await self.user_data_manager.add_generated_image_to_history(
                                    user_id=user_id,
                                    username=username,
                                    prompt=prompt,
                                    response="Image generated successfully",
                                    matrix_file=matrix_info['file'],
                                    width=matrix_info['width'],
                                    height=matrix_info['height'],
                                    model_used=f"{tier}:{model}",
                                    response_time=0.0,
                                    tokens_used=0
                                )
                            except Exception as e:
                                logger.error(f"Failed to save generated image to history: {e}")
                        return image_bytes, f"{tier}:{model}", len(image_bytes)
                    else:
                        logger.warning(f"All models in tier {tier} failed")

                # Check if timer triggered restart safely
                if timer_task.done():
                    try:
                        exc = timer_task.exception()
                        if exc and isinstance(exc, RestartSearchException):
                            restart_triggered = True
                    except Exception:
                        pass

                if restart_triggered:
                    logger.info("Restart triggered, clearing blacklist and refreshing model list.")
                    self._clear_blacklist()
                    self.model_manager._fetch_and_update_models()
                    continue
                else:
                    break

            raise Exception(f"All image generation tiers failed. Last error: {last_error}")
        except Exception as e:
            timer_task.cancel()
            raise
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
        try:
            elapsed = 0
            while elapsed < search_timeout:
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time
                if elapsed >= search_timeout:
                    break
            if status_callback:
                await status_callback("🖼️ *AI Models Scarce, Please Wait ...*", edit=True)
                logger.info("⏳ Sent 'wait' message for image generation")

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
            logger.error(f"Image generation timer error: {e}")

    async def _get_tier_priority(self, user_id: Optional[int], username: Optional[str]) -> List[str]:
        if user_id and self.user_data_manager:
            try:
                return await self.user_data_manager.get_image_generation_priority(user_id, username)
            except Exception:
                pass
        return self.tiers

    async def _try_tier_models(self, tier: str, models: List[str], prompt: str, size: str, quality: str) -> Optional[Tuple[bytes, str]]:
        parallel_count = min(Config.PARALLEL_MODEL_ATTEMPTS, len(models))
        attempts = models[:parallel_count]
        tasks = []
        for model in attempts:
            if tier == "pollinations":
                task = asyncio.create_task(self._generate_pollinations(model, prompt, size, quality))
            elif tier == "huggingface":
                task = asyncio.create_task(self._generate_huggingface(model, prompt))
            elif tier == "openrouter":
                task = asyncio.create_task(self._generate_openrouter(model, prompt, size, quality))
            else:
                continue
            tasks.append((model, task))

        pending = {task for _, task in tasks}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                model = next(m for m, t in tasks if t == task)
                try:
                    result = task.result()
                    if result:
                        return result, model
                except Exception as e:
                    logger.warning(f"Model {model} in tier {tier} failed: {e}")
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in self._non_retryable_errors:
                        self._mark_model_failure(model)

        # Sequential fallback
        remaining = models[parallel_count:]
        for model in remaining:
            try:
                logger.info(f"🖼️ Trying sequential model: {model}")
                if tier == "pollinations":
                    result = await self._generate_pollinations(model, prompt, size, quality)
                elif tier == "huggingface":
                    result = await self._generate_huggingface(model, prompt)
                elif tier == "openrouter":
                    result = await self._generate_openrouter(model, prompt, size, quality)
                else:
                    continue
                if result:
                    return result, model
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in self._non_retryable_errors:
                    self._mark_model_failure(model)
        return None

    # ---------- Pollinations ----------
    async def _generate_pollinations(self, model: str, prompt: str, size: str, quality: str) -> Optional[bytes]:
        width, height = size.split('x') if 'x' in size else ('1024', '1024')
        params = {
            "model": model,
            "width": width,
            "height": height,
            "nologo": "true",
            "seed": "random"
        }
        # Ensure prompt is not too long (Pollinations may reject long URLs)
        if len(prompt) > 2000:
            prompt = prompt[:2000]
        url = f"{self.pollinations_base_url}/{quote(prompt)}"
        try:
            response = await retry_async(
                self._client.get,
                url,
                params=params,
                max_attempts=2
            )
            response.raise_for_status()
            content = response.content
            if content and len(content) > 100:
                return content
            else:
                logger.warning(f"Pollinations: Model {model} returned empty/short response")
                return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                logger.warning(f"Pollinations: Model {model} requires payment (402)")
            elif e.response.status_code == 404:
                logger.warning(f"Pollinations: Model {model} not found (404)")
            else:
                logger.warning(f"Pollinations: Model {model} error: {e}")
            raise
        except Exception as e:
            logger.warning(f"Pollinations: Model {model} error: {e}")
            raise

    # ---------- Hugging Face ----------
    async def _generate_huggingface(self, model: str, prompt: str) -> Optional[bytes]:
        if not self.huggingface_token:
            raise Exception("Hugging Face token not configured")
        url = f"{self.huggingface_url}/{model}"
        headers = {"Authorization": f"Bearer {self.huggingface_token}"}
        payload = {"inputs": prompt}
        try:
            response = await retry_async(
                self._client.post,
                url,
                headers=headers,
                json=payload,
                max_attempts=2
            )
            response.raise_for_status()
            content = response.content
            if response.headers.get('content-type', '').startswith('image/'):
                return content
            else:
                logger.warning(f"Hugging Face: Model {model} returned non-image response")
                return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                logger.warning(f"Hugging Face: Model {model} requires payment (402)")
            else:
                logger.warning(f"Hugging Face: Model {model} error: {e}")
            raise
        except Exception as e:
            logger.warning(f"Hugging Face: Model {model} error: {e}")
            raise

    # ---------- OpenRouter ----------
    async def _generate_openrouter(self, model: str, prompt: str, size: str, quality: str) -> Optional[bytes]:
        if not self.api_key:
            raise Exception("OpenRouter API key not configured")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.BOT_REPO_URL,
            "X-Title": Config.BOT_NAME
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "width": int(size.split('x')[0]),
            "height": int(size.split('x')[1]) if 'x' in size else 1024,
            "quality": quality,
            "response_format": "url"
        }
        try:
            response = await retry_async(
                self._client.post,
                self.image_base_url,
                headers=headers,
                json=payload,
                max_attempts=2
            )
            response.raise_for_status()
            data = response.json()
            image_url = data.get('data', [{}])[0].get('url')
            if not image_url:
                logger.warning(f"OpenRouter: No image URL in response for model {model}")
                return None
            img_response = await retry_async(
                self._client.get,
                image_url,
                max_attempts=2
            )
            img_response.raise_for_status()
            return img_response.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                logger.warning(f"OpenRouter: Model {model} requires payment (402)")
            elif e.response.status_code == 400:
                logger.warning(f"OpenRouter: Model {model} bad request (400)")
            else:
                logger.warning(f"OpenRouter: Model {model} error: {e}")
            raise
        except Exception as e:
            logger.warning(f"OpenRouter: Model {model} error: {e}")
            raise

    def get_engine_info(self) -> Dict:
        return {
            "type": "ImageGenerationEngine",
            "initialized": self.is_initialized,
            "tiers": self.tiers,
            "available_models": {
                "pollinations": self.pollinations_models,
                "huggingface": self.huggingface_models,
                "openrouter": self.openrouter_models,
            },
            "blacklisted_models": list(self._model_failures.keys())
        }