import os
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
import httpx
from core.config import Config
from core.prompt_library import PromptLibrary, PromptCategory
from core.user_data_manager import UserDataManager
from core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, user_data_manager: UserDataManager):
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.user_data_manager = user_data_manager
        self.prompt_library = PromptLibrary()

        # Initialize Dynamic Model Manager
        self.model_manager = ModelManager()
        self.model_manager.start()

        self._client: Optional[httpx.AsyncClient] = None

        logger.info("LLM Client initialized with Dynamic Model Manager")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            try:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=10.0, pool=10.0),
                    http2=True,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
                logger.info("HTTP/2 connection pool initialized.")
            except ImportError:
                logger.warning("h2 package not installed. Falling back to HTTP/1.1.")
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=10.0, pool=10.0),
                    http2=False,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self.model_manager.stop()

    def _estimate_complexity(self, prompt: str) -> str:
        words = prompt.split()
        complex_keywords = ['code', 'script', 'explain', 'analyze', 'python', 'javascript', 'architecture']
        if len(words) > 40 or any(kw in prompt.lower() for kw in complex_keywords):
            return "smart"
        return "fast"

    async def ask(self, prompt: str, history: List[Dict], user_id: Optional[int] = None) -> Tuple[str, str, str]:
        category = self.prompt_library.detect_category(prompt)
        prompt_template = self.prompt_library.get_prompt(category)

        if user_id is not None:
            cached = self.user_data_manager.get_cached_response(prompt)
            if cached:
                response, cached_category, timestamp = cached
                logger.info(f"Using cached response for user {user_id}")
                return response, "cache", cached_category

        complexity = self._estimate_complexity(prompt)
        all_models = self.model_manager.get_smart_models() if complexity == "smart" else self.model_manager.get_fast_models()

        last_error = None
        client = await self._get_client()

        for attempt in range(Config.HTTP_MAX_RETRIES):
            for model in all_models:
                try:
                    logger.info(f"Trying model: {model} (attempt {attempt + 1}, complexity: {complexity})")
                    response = await self._call_api(
                        client, model, prompt, history,
                        prompt_template.system_message,
                        prompt_template.temperature,
                        prompt_template.max_tokens
                    )
                    logger.info(f"✅ Model {model} succeeded")
                    if user_id is not None:
                        self.user_data_manager.save_to_cache(prompt, response, category.value)
                    return response, model, category.value
                except Exception as e:
                    logger.warning(f"Model {model} failed (attempt {attempt + 1}): {e}")
                    last_error = e
                    continue

        logger.error(f"All models failed after {Config.HTTP_MAX_RETRIES} attempts")
        raise Exception(f"All models failed. Last error: {last_error}")

    async def _call_api(self, client: httpx.AsyncClient, model: str, prompt: str, history: List[Dict],
                        system_prompt: str, temperature: float, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.BOT_REPO_URL,
            "X-Title": Config.BOT_NAME
        }

        # 🧠 SHARED BRAIN: Inject continuity instruction if there is history
        continuity_note = ""
        if history:
            continuity_note = (
                "\n\nIMPORTANT CONTINUITY RULE: You are continuing an ongoing conversation. "
                "Previous messages in this history were generated by another AI model. "
                "You MUST maintain strict continuity with the established context, facts, and tone. "
                "Do not restart the conversation, introduce yourself again, or act as if this is the first message."
            )

        full_system_prompt = system_prompt + continuity_note

        messages = [{"role": "system", "content": full_system_prompt}]
        messages.extend(history[-Config.MAX_HISTORY_MESSAGES:])
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        for attempt in range(Config.HTTP_MAX_RETRIES):
            try:
                resp = await client.post(self.base_url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.ConnectTimeout:
                if attempt == Config.HTTP_MAX_RETRIES - 1: raise
                await asyncio.sleep(2 ** attempt)
            except httpx.ReadTimeout:
                if attempt == Config.HTTP_MAX_RETRIES - 1: raise
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"API error: {e}")
                raise