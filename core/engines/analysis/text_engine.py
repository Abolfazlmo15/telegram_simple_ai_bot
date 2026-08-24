"""Text processing engine with token tracking and retry."""
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
import httpx
from core.config import Config
from core.utils.prompt_library import PromptLibrary
from core.managers.user_data_manager import UserDataManager
from core.managers.model_manager import ModelManager

logger = logging.getLogger(__name__)


class TextEngine:
    def __init__(self, user_data_manager: UserDataManager):
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.user_data_manager = user_data_manager
        self.prompt_library = PromptLibrary()
        self.model_manager = ModelManager()
        self._client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False
        logger.info("Text engine initialized")

    async def initialize(self) -> bool:
        try:
            self.model_manager.start()
            try:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=10.0, pool=10.0),
                    http2=True,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
                logger.info("Text engine initialized successfully (HTTP/2).")
            except Exception:
                logger.warning("h2 package not installed or HTTP/2 failed. Falling back to HTTP/1.1.")
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=10.0, pool=10.0),
                    http2=False,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
                logger.info("Text engine initialized successfully (HTTP/1.1).")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize text engine: {e}")
            return False

    async def shutdown(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self.model_manager.stop()
        self.is_initialized = False
        logger.info("Text engine shutdown complete")

    async def process(self, input_data: str, context: Optional[Dict] = None) -> Tuple[str, str, int]:
        if not self.is_initialized:
            raise RuntimeError("Text engine not initialized")

        user_id = context.get('user_id') if context else None
        username = context.get('username') if context else None
        history = context.get('history', []) if context else []
        skip_cache = context.get('skip_cache', False) if context else False

        # Check cache only if not forced to skip
        if not skip_cache and user_id is not None:
            cached = self.user_data_manager.get_cached_response(input_data)
            if cached:
                response, cached_category, timestamp = cached
                logger.info(f"Cache hit for user {user_id}")
                return response, "cache", 0  # tokens unknown from cache

        complexity = self._estimate_complexity(input_data)
        base_models = self.model_manager.get_smart_models() if complexity == "smart" else self.model_manager.get_fast_models()

        # Load user priority (text engine)
        priority_models = None
        if user_id is not None:
            priority_models = await self.user_data_manager.get_user_model_priority(user_id, username, engine="text")
            if priority_models:
                logger.info(f"User {user_id} has text priority list, reordering models")
                ordered = []
                for p in priority_models:
                    if p in base_models and p not in ordered:
                        ordered.append(p)
                for m in base_models:
                    if m not in ordered:
                        ordered.append(m)
                all_models = ordered
            else:
                all_models = base_models
        else:
            all_models = base_models

        category = self.prompt_library.detect_category(input_data)
        prompt_template = self.prompt_library.get_prompt(category)
        if len(input_data.split()) < 5 and category.value == "casual_conversation":
            prompt_template.max_tokens = min(prompt_template.max_tokens, 300)

        last_error = None
        for model in all_models:
            # Retry the same model up to Config.HTTP_MAX_RETRIES times on transient errors
            for attempt in range(Config.HTTP_MAX_RETRIES):
                try:
                    logger.info(f"Trying model: {model} (attempt {attempt+1}/{Config.HTTP_MAX_RETRIES})")
                    response, tokens_used = await self._call_api(
                        model, input_data, history,
                        prompt_template.system_message,
                        prompt_template.temperature,
                        prompt_template.max_tokens
                    )
                    logger.info(f"✅ Model {model} succeeded (tokens: {tokens_used})")
                    if user_id is not None:
                        self.user_data_manager.save_to_cache(input_data, response, category.value)
                    return response, model, tokens_used
                except Exception as e:
                    logger.warning(f"Model {model} attempt {attempt+1} failed: {e}")
                    last_error = e
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(1)  # short delay before retry
                    else:
                        # All retries for this model exhausted
                        break
            # Move to next model after all retries failed
        logger.error("All models failed")
        raise Exception(f"All text models failed. Last error: {last_error}")

    def _estimate_complexity(self, prompt: str) -> str:
        words = prompt.split()
        complex_keywords = ['code', 'script', 'explain', 'analyze', 'python', 'javascript', 'architecture']
        if len(words) > 40 or any(kw in prompt.lower() for kw in complex_keywords):
            return "smart"
        return "fast"

    async def _call_api(self, model: str, prompt: str, history: List[Dict],
                       system_prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.BOT_REPO_URL,
            "X-Title": Config.BOT_NAME
        }
        messages = [{"role": "system", "content": system_prompt}]
        max_context = getattr(Config, 'MAX_CONTEXT_MESSAGES', 3)
        recent_entries = history[-max_context:] if history else []
        for entry in recent_entries:
            if entry.get('type') == 'text' and 'message' in entry:
                messages.append({"role": "user", "content": entry['message']})
                if 'response' in entry:
                    messages.append({"role": "assistant", "content": entry['response']})
            elif entry.get('type') == 'image':
                # For image entries, we can include the query and response as context
                if 'query' in entry:
                    messages.append({"role": "user", "content": entry['query']})
                if 'response' in entry:
                    messages.append({"role": "assistant", "content": entry['response']})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        resp = await self._client.post(self.base_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        response_text = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return response_text, tokens_used

    def get_engine_info(self) -> Dict:
        return {
            "type": "TextEngine",
            "initialized": self.is_initialized,
            "model_manager": "running" if self.model_manager.is_running else "stopped"
        }