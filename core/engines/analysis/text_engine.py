"""Text processing engine with token tracking, retry, parallel testing (FIRST_COMPLETED), and blacklisting."""
import logging
import asyncio
import time
from typing import List, Dict, Tuple, Optional, Any
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

        # ============================================================
        # PERFORMANCE: Blacklisting & Failure Tracking
        # ============================================================
        self._model_failures: Dict[str, float] = {}  # model_id -> failure_timestamp
        self._blacklist_ttl = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS
        self._non_retryable_errors = {404, 400, 402, 429}  # Don't retry these

        logger.info("Text engine initialized with parallel testing (FIRST_COMPLETED) & blacklisting")

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
        logger.info(f"🚫 Blacklisted model {model} for {self._blacklist_ttl}s")

    def _get_model_list(self, priority_list: Optional[List[str]], fallback_list: List[str]) -> List[str]:
        ordered = []
        seen = set()

        if priority_list:
            for m in priority_list:
                if m not in seen and not self._is_model_blacklisted(m):
                    ordered.append(m)
                    seen.add(m)

        for m in fallback_list:
            if m not in seen and not self._is_model_blacklisted(m):
                ordered.append(m)
                seen.add(m)

        if not ordered:
            logger.warning("All models blacklisted, returning full fallback list.")
            return list(dict.fromkeys(fallback_list))

        return ordered

    async def process(self, input_data: str, context: Optional[Dict] = None) -> Tuple[str, str, int]:
        if not self.is_initialized:
            raise RuntimeError("Text engine not initialized")

        user_id = context.get('user_id') if context else None
        username = context.get('username') if context else None
        history = context.get('history', []) if context else []
        skip_cache = context.get('skip_cache', False) if context else False
        priority_list = context.get('priority_list') if context else None

        # Check cache
        if not skip_cache and user_id is not None:
            cached = self.user_data_manager.get_cached_response(input_data)
            if cached:
                response, cached_category, timestamp = cached
                logger.info(f"Cache hit for user {user_id}")
                return response, "cache", 0

        complexity = self._estimate_complexity(input_data)
        base_models = self.model_manager.get_smart_models() if complexity == "smart" else self.model_manager.get_fast_models()

        # Build the final ordered model list using user priority + fallback
        model_list = self._get_model_list(priority_list, base_models)
        logger.info(f"📋 Using model list (first 5): {model_list[:5]}...")

        category = self.prompt_library.detect_category(input_data)
        prompt_template = self.prompt_library.get_prompt(category)
        if len(input_data.split()) < 5 and category.value == "casual_conversation":
            prompt_template.max_tokens = min(prompt_template.max_tokens, 300)

        last_error = None

        # ============================================================
        # PERFORMANCE: Parallel Testing with FIRST_COMPLETED
        # ============================================================
        if Config.ENABLE_PARALLEL_MODEL_TESTING and len(model_list) > 1:
            result = await self._try_models_parallel_first_completed(
                model_list[:Config.PARALLEL_MODEL_ATTEMPTS],
                input_data, history, prompt_template
            )
            if result:
                response, model, tokens_used, error = result
                if response is not None:
                    logger.info(f"✅ Parallel test succeeded on {model} (tokens: {tokens_used})")
                    if user_id is not None:
                        self.user_data_manager.save_to_cache(input_data, response, category.value)
                    return response, model, tokens_used
                elif error:
                    # Mark the failed model
                    self._mark_model_failure(model)
                    last_error = error

            # If parallel failed, fall through to sequential for remaining models
            remaining_models = model_list[Config.PARALLEL_MODEL_ATTEMPTS:]
            logger.info(f"Parallel failed, falling back to sequential for {len(remaining_models)} models")
        else:
            remaining_models = model_list

        # ============================================================
        # SEQUENTIAL FALLBACK (with smart retries)
        # ============================================================
        for model in remaining_models:
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
                except httpx.HTTPStatusError as e:
                    # Non-retryable status codes: blacklist immediately and break
                    if e.response.status_code in self._non_retryable_errors:
                        logger.warning(f"Non-retryable error {e.response.status_code} for {model}, blacklisting.")
                        self._mark_model_failure(model)
                        break  # skip retries for this model
                    else:
                        logger.warning(f"Model {model} attempt {attempt+1} failed: {e}")
                        last_error = e
                        if attempt < Config.HTTP_MAX_RETRIES - 1:
                            await asyncio.sleep(0.5)
                        else:
                            self._mark_model_failure(model)
                            break
                except Exception as e:
                    logger.warning(f"Model {model} attempt {attempt+1} failed: {e}")
                    last_error = e
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
                    else:
                        self._mark_model_failure(model)
                        break

        logger.error("All models failed")
        raise Exception(f"All text models failed. Last error: {last_error}")

    async def _try_models_parallel_first_completed(self, models: List[str], prompt: str, history: List[Dict],
                                                   prompt_template) -> Optional[Tuple[Optional[str], str, int, Optional[Exception]]]:
        """Test multiple models concurrently, returning the first successful result."""
        tasks = {}
        for model in models:
            task = asyncio.create_task(
                self._call_api_with_error_info(model, prompt, history, prompt_template)
            )
            tasks[task] = model

        # Wait for the first task to complete (success or failure)
        pending = set(tasks.keys())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                model = tasks[task]
                try:
                    response, tokens = task.result()
                    if response is not None:
                        # Cancel remaining tasks
                        for t in pending:
                            t.cancel()
                        return (response, model, tokens, None)
                except Exception as e:
                    # This model failed, continue waiting
                    last_error = e
                    # If all models failed, we'll catch it later
            # If we have no pending tasks (all failed), break
            if not pending:
                break

        # All failed, return None
        return None

    async def _call_api_with_error_info(self, model: str, prompt: str, history: List[Dict],
                                        prompt_template) -> Tuple[str, int]:
        """Wrapper for _call_api that propagates exceptions."""
        try:
            return await self._call_api(
                model, prompt, history,
                prompt_template.system_message,
                prompt_template.temperature,
                prompt_template.max_tokens
            )
        except Exception as e:
            raise e

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
            "model_manager": "running" if self.model_manager.is_running else "stopped",
            "parallel_testing": Config.ENABLE_PARALLEL_MODEL_TESTING,
            "blacklist_ttl": self._blacklist_ttl,
            "blacklisted_models": list(self._model_failures.keys())
        }