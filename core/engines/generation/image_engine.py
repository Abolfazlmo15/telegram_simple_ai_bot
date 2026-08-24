"""Image generation engine with multi-tier fallback and style-aware model selection.
Supports dynamic priority ordering per user."""
import logging
import asyncio
import base64
from typing import Dict, Tuple, Optional, Any, List
import httpx
from core.config import Config
from core.managers.user_data_manager import UserDataManager
from core.managers.image_model_manager import ImageModelManager

logger = logging.getLogger(__name__)


class ImageGenerationEngine:
    """
    Multi-tier image generation engine with style-aware model selection.
    Priority: Configurable via user preferences (default: Config.IMAGE_GENERATION_PRIORITY).
    """

    def __init__(self, user_data_manager: Optional[UserDataManager] = None):
        self.user_data_manager = user_data_manager
        self.model_manager = ImageModelManager()
        self._client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False
        self.image_size = Config.IMAGE_GENERATION_SIZE

        # Tier configurations
        self.tiers = {
            "pollinations": {
                "enabled": True,
                "url": Config.POLLINATIONS_IMAGE_URL,
                "models": Config.POLLINATIONS_MODELS,
                "api_key": None,
                "priority": 1,
            },
            "huggingface": {
                "enabled": bool(Config.HUGGINGFACE_API_KEY),
                "url": Config.HUGGINGFACE_IMAGE_URL,
                "models": Config.HUGGINGFACE_MODELS,
                "api_key": Config.HUGGINGFACE_API_KEY,
                "priority": 2,
            },
            "openrouter": {
                "enabled": bool(Config.OPENROUTER_API_KEY),
                "url": Config.OPENROUTER_IMAGE_GENERATION_URL,
                "models": Config.OPENROUTER_IMAGE_MODELS,
                "api_key": Config.OPENROUTER_API_KEY,
                "priority": 3,
            }
        }

        logger.info("🖼️ ImageGenerationEngine initialized with multi-tier fallback")

    async def initialize(self) -> bool:
        try:
            self.model_manager.start()
            timeout = httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=60.0,
                pool=10.0
            )
            try:
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=True,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
            except Exception:
                logger.warning("HTTP/2 failed for image engine, falling back to HTTP/1.1")
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=False,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )

            available = [name for name, tier in self.tiers.items() if tier["enabled"]]
            logger.info(f"🖼️ Available image generation tiers: {available}")

            self.is_initialized = True
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

    async def generate(self, prompt: str, context: Optional[Dict] = None) -> Tuple[bytes, str, int]:
        """Generate image with style-aware model selection and multi-tier fallback."""
        if not self.is_initialized:
            raise RuntimeError("Image generation engine not initialized")

        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Empty prompt provided for image generation")

        # ============================================================
        # 1. Get user's priority order (if available)
        # ============================================================
        user_id = context.get('user_id') if context else None
        username = context.get('username') if context else None

        priority_tiers = None
        if self.user_data_manager and user_id:
            priority_tiers = await self.user_data_manager.get_image_generation_priority(user_id, username)

        # If no user priority, use config default
        if not priority_tiers:
            priority_tiers = Config.IMAGE_GENERATION_PRIORITY

        logger.info(f"🖼️ Using priority order: {priority_tiers}")

        # ============================================================
        # 2. Extract style and negative prompt from context
        # ============================================================
        detected_style = context.get('detected_style', 'no_style') if context else 'no_style'
        negative_prompt = context.get('negative_prompt', '') if context else ''
        recommended_models = context.get('recommended_models', []) if context else []

        if detected_style == 'no_style':
            detected_style = self._detect_style(prompt)

        if not recommended_models:
            recommended_models = self._get_preferred_models_for_style(detected_style)

        logger.info(f"🖼️ Detected style: {detected_style}")
        logger.info(f"🖼️ Preferred models: {recommended_models[:3]}...")
        if negative_prompt:
            logger.info(f"🚫 Using negative prompt: {negative_prompt[:50]}...")

        # ============================================================
        # 3. Try each tier in user-specified priority order
        # ============================================================
        last_error = None
        for tier_name in priority_tiers:
            tier = self.tiers.get(tier_name)
            if not tier or not tier["enabled"]:
                continue

            logger.info(f"🖼️ Trying tier: {tier_name}")

            try:
                # Build model list: for OpenRouter, prioritize recommended models
                if tier_name == "openrouter":
                    models_to_try = recommended_models + tier["models"]
                    models_to_try = list(dict.fromkeys(models_to_try))  # Remove duplicates
                else:
                    models_to_try = tier["models"]

                image_bytes, model_used = await self._call_tier_with_models(
                    tier_name, tier, prompt, models_to_try, negative_prompt
                )

                if image_bytes:
                    logger.info(f"✅ Image generated via {tier_name} using {model_used} (style: {detected_style})")

                    if self.user_data_manager and context and 'user_id' in context:
                        await self._save_to_history(context, prompt, image_bytes, f"{tier_name}:{model_used}")

                    return image_bytes, f"{tier_name}:{model_used}", len(image_bytes)

            except Exception as e:
                logger.warning(f"⚠️ Tier {tier_name} failed: {e}")
                last_error = e
                continue

        raise Exception(f"All image generation tiers failed. Last error: {last_error}")

    async def _call_tier_with_models(self, tier_name: str, tier: Dict, prompt: str,
                                     models: List[str], negative_prompt: str = "") -> Tuple[Optional[bytes], str]:
        """Call a specific tier with a given list of models."""
        if tier_name == "pollinations":
            return await self._call_pollinations_with_models(tier, prompt, models)
        elif tier_name == "huggingface":
            return await self._call_huggingface_with_models(tier, prompt, models, negative_prompt)
        elif tier_name == "openrouter":
            return await self._call_openrouter_with_models(tier, prompt, models, negative_prompt)
        else:
            raise ValueError(f"Unknown tier: {tier_name}")

    # ============================================================
    # TIER 1: Pollinations.ai (FREE)
    # ============================================================
    async def _call_pollinations_with_models(self, tier: Dict, prompt: str, models: List[str]) -> Tuple[Optional[bytes], str]:
        for model in models:
            try:
                url = f"{tier['url']}/{self._url_encode(prompt)}"
                params = {
                    "model": model,
                    "width": 1024,
                    "height": 1024,
                    "nologo": "true",
                    "seed": "random"
                }

                resp = await self._client.get(url, params=params, timeout=30.0)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                if "image" in content_type or resp.content[:4] in [b'\xFF\xD8\xFF', b'\x89PNG']:
                    logger.info(f"✅ Pollinations: Success with model {model}")
                    return resp.content, model

            except Exception as e:
                logger.warning(f"Pollinations: Model {model} error: {e}")
                continue

        return None, "none"

    # ============================================================
    # TIER 2: Hugging Face Inference API (FREE tier)
    # ============================================================
    async def _call_huggingface_with_models(self, tier: Dict, prompt: str, models: List[str],
                                            negative_prompt: str = "") -> Tuple[Optional[bytes], str]:
        if not tier["api_key"]:
            return None, "none"

        headers = {
            "Authorization": f"Bearer {tier['api_key']}",
            "Content-Type": "application/json"
        }

        negative = negative_prompt or "blurry, low quality, distorted"

        for model in models:
            try:
                url = f"{tier['url']}/{model}"
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "negative_prompt": negative,
                        "num_inference_steps": 20,
                        "guidance_scale": 7.5,
                    }
                }

                resp = await self._client.post(url, headers=headers, json=payload, timeout=60.0)

                if resp.status_code == 200:
                    if resp.content[:4] in [b'\xFF\xD8\xFF', b'\x89PNG']:
                        logger.info(f"✅ Hugging Face: Success with model {model}")
                        return resp.content, model

                elif resp.status_code == 503:
                    logger.info(f"Hugging Face: Model {model} is loading, waiting...")
                    await asyncio.sleep(2)
                    retry_resp = await self._client.post(url, headers=headers, json=payload, timeout=60.0)
                    if retry_resp.status_code == 200 and retry_resp.content[:4] in [b'\xFF\xD8\xFF', b'\x89PNG']:
                        return retry_resp.content, model

                elif resp.status_code in (402, 429):
                    logger.warning(f"Hugging Face: Rate limit or payment required for {model}")
                    break

            except Exception as e:
                logger.warning(f"Hugging Face: Model {model} error: {e}")
                continue

        return None, "none"

    # ============================================================
    # TIER 3: OpenRouter (PAID) - Style-Aware
    # ============================================================
    async def _call_openrouter_with_models(self, tier: Dict, prompt: str, models: List[str],
                                           negative_prompt: str = "") -> Tuple[Optional[bytes], str]:
        if not tier["api_key"]:
            return None, "none"

        headers = {
            "Authorization": f"Bearer {tier['api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.BOT_REPO_URL,
            "X-Title": Config.BOT_NAME
        }

        for model in models:
            try:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "n": 1,
                    "size": self.image_size
                }
                if negative_prompt and "dall-e" not in model.lower():
                    payload["negative_prompt"] = negative_prompt

                resp = await self._client.post(tier["url"], headers=headers, json=payload, timeout=60.0)

                if resp.status_code == 200:
                    data = resp.json()
                    if "data" in data and len(data["data"]) > 0:
                        image_data = data["data"][0]
                        if "b64_json" in image_data:
                            logger.info(f"✅ OpenRouter: Success with model {model}")
                            return base64.b64decode(image_data["b64_json"]), model
                        elif "url" in image_data:
                            img_resp = await self._client.get(image_data["url"])
                            img_resp.raise_for_status()
                            return img_resp.content, model

                elif resp.status_code == 402:
                    logger.warning(f"OpenRouter: Model {model} requires payment (402)")
                    continue

            except Exception as e:
                logger.warning(f"OpenRouter: Model {model} error: {e}")
                continue

        return None, "none"

    # ============================================================
    # STYLE DETECTION & MODEL MAPPING (fallback)
    # ============================================================
    def _detect_style(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        for style, keywords in Config.STYLE_KEYWORDS.items():
            if not keywords:
                continue
            for keyword in keywords:
                if keyword in prompt_lower:
                    return style
        return "no_style"

    def _get_preferred_models_for_style(self, style: str) -> List[str]:
        style_models = Config.STYLE_MODEL_MAP.get(style, [])
        if style_models:
            return style_models
        return ["black-forest-labs/flux.2-pro", "google/gemini-2.5-flash-image"]

    def _url_encode(self, text: str) -> str:
        import urllib.parse
        return urllib.parse.quote(text)

    async def _save_to_history(self, context: Dict, prompt: str, image_bytes: bytes, source: str):
        try:
            user_id = context['user_id']
            username = context.get('username')

            matrix_info = await self.user_data_manager.save_image_matrix(
                user_id, username, image_bytes
            )

            await self.user_data_manager.add_generated_image_to_history(
                user_id=user_id,
                username=username,
                prompt=prompt,
                response=f"Image generated via {source}",
                matrix_file=matrix_info['file'],
                width=matrix_info['width'],
                height=matrix_info['height'],
                model_used=source,
                response_time=0.0,
                tokens_used=0
            )
        except Exception as e:
            logger.error(f"Failed to save generated image to history: {e}")

    def get_engine_info(self) -> Dict:
        return {
            "type": "ImageGenerationEngine",
            "initialized": self.is_initialized,
            "tiers": {
                name: {
                    "enabled": tier["enabled"],
                    "priority": tier["priority"],
                    "models": tier["models"][:3]
                }
                for name, tier in self.tiers.items()
            },
            "default_priority": Config.IMAGE_GENERATION_PRIORITY,
            "image_size": self.image_size,
            "style_model_manager": self.model_manager.get_engine_info() if self.model_manager.is_running else None
        }