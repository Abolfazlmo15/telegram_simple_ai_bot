"""Vision/Image processing engine with dynamic priority."""
import logging
import asyncio
import base64
import json
from typing import Dict, Tuple, Optional, Any, List
import httpx
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
        self.image_processor = ImageProcessor(max_size=512, quality=60)
        self._client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False
        self.user_data_manager = user_data_manager
        logger.info("🔷 VisionEngine __init__ done")

    async def initialize(self) -> bool:
        logger.info("🔷 VisionEngine.initialize: START")
        try:
            self.model_manager.start()
            logger.info("🔷 VisionModelManager started")
            timeout = httpx.Timeout(
                connect=10.0,
                read=Config.HTTP_TIMEOUT,
                write=10.0,
                pool=10.0
            )
            try:
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=True,
                    limits=httpx.Limits(
                        max_connections=Config.CONNECTION_POOL_SIZE,
                        max_keepalive_connections=5
                    )
                )
                logger.info("🔷 Vision engine HTTP client created (HTTP/2 attempt)")
            except Exception as e:
                logger.warning(f"🔷 HTTP/2 failed: {e}. Falling back to HTTP/1.1.")
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=False,
                    limits=httpx.Limits(
                        max_connections=Config.CONNECTION_POOL_SIZE,
                        max_keepalive_connections=5
                    )
                )
                logger.info("🔷 Vision engine HTTP client created (HTTP/1.1)")
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

    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Tuple[str, str, int]:
        logger.info("🔷 VisionEngine.process: START")
        if not self.is_initialized:
            raise RuntimeError("Vision engine not initialized")

        user_id = context.get('user_id') if context else None
        username = context.get('username') if context else None
        query_text = context.get('query_text', '').strip() if context else ''
        logger.info(f"🔷 user_id: {user_id}, query_text: {query_text[:50] if query_text else '(empty)'}...")

        # Process image
        try:
            processed_image = await self.image_processor.process_image(input_data)
            base64_image = self.image_processor.encode_to_base64(processed_image)
            logger.info(f"🔷 Image processed, base64 length: {len(base64_image)} chars")
        except Exception as e:
            logger.error(f"❌ Image processing failed: {e}", exc_info=True)
            raise ValueError(f"Could not process image: {e}")

        # Build system prompt
        system_prompt = self._build_vision_prompt(query_text)
        logger.info(f"🔷 System prompt built (length: {len(system_prompt)})")

        # Get dynamic models
        model_list = self.model_manager.get_available_models()
        logger.info(f"🔷 Vision models: {len(model_list)} found")

        # Load user priority for vision
        if self.user_data_manager and user_id:
            priority_models = await self.user_data_manager.get_user_model_priority(user_id, username, engine="vision")
            if priority_models:
                logger.info(f"User {user_id} has vision priority list, reordering models")
                ordered = []
                for p in priority_models:
                    if p in model_list and p not in ordered:
                        ordered.append(p)
                for m in model_list:
                    if m not in ordered:
                        ordered.append(m)
                model_list = ordered
                logger.info(f"Reordered models: {model_list[:3]}...")

        last_error = None
        for model in model_list:
            for attempt in range(Config.HTTP_MAX_RETRIES):
                try:
                    logger.info(f"🔷 Trying vision model: {model} (attempt {attempt+1}/{Config.HTTP_MAX_RETRIES})")
                    response, tokens_used = await self._call_vision_api(
                        model, base64_image, query_text, system_prompt
                    )
                    logger.info(f"✅ Vision model {model} succeeded (tokens: {tokens_used})")
                    return response, model, tokens_used
                except Exception as e:
                    logger.warning(f"🔴 Model {model} attempt {attempt+1} failed: {e}")
                    last_error = e
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(1)
                    else:
                        break
        logger.error(f"❌ All vision models failed. Last error: {last_error}")
        raise Exception(f"All vision models failed. Last error: {last_error}")

    def _build_vision_prompt(self, user_query: str) -> str:
        generic_phrases = [
            "describe this image",
            "describe the image",
            "what is in this image",
            "what is in the image",
            "image analysis",
            "analyze this image",
            "analyze the image"
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

    async def _call_vision_api(self, model: str, base64_image: str,
                               query: str, system_prompt: str) -> Tuple[str, int]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.BOT_REPO_URL,
            "X-Title": Config.BOT_NAME
        }
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        generic_phrases = ["describe this image", "describe the image", "what is in this image"]
        if query and query.lower() not in generic_phrases:
            user_content.append({"type": "text", "text": query})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1500
        }
        logger.info(f"🔶 Payload size: {len(json.dumps(payload))} bytes")

        resp = await self._client.post(self.base_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        response_text = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return response_text, tokens_used

    def get_engine_info(self) -> Dict:
        return {
            "type": "VisionEngine",
            "initialized": self.is_initialized,
            "model_manager": "running" if self.model_manager.is_running else "stopped"
        }