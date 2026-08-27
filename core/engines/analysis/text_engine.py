"""Text processing engine with token tracking, retry, parallel testing, and timeout/restart logic."""
import logging
import asyncio
import time
from typing import List, Dict, Tuple, Optional, Any, Callable, Coroutine
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

        # Blacklisting
        self._model_failures: Dict[str, float] = {}
        self._blacklist_ttl = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS
        self._non_retryable_errors = {404, 400, 402, 429}

        # Timeout configuration
        self.search_timeout = Config.TEXT_SEARCH_TIMEOUT_SECONDS
        self.restart_timeout = Config.GLOBAL_RESTART_TIMEOUT_SECONDS

        logger.info("Text engine initialized with parallel testing, timeout, and restart logic")

    async def initialize(self) -> bool:
        try:
            self.model_manager.start()
            try:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=10.0, pool=10.0),
                    http2=True,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
                logger.info("Text engine initialized (HTTP/2).")
            except Exception:
                logger.warning("HTTP/2 failed, falling back to HTTP/1.1.")
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=10.0, pool=10.0),
                    http2=False,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
                logger.info("Text engine initialized (HTTP/1.1).")
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
            if time.time() - self._model_failures[model] < self._blacklist_ttl:
                return True
            else:
                del self._model_failures[model]
        return False

    def _mark_model_failure(self, model: str):
        self._model_failures[model] = time.time()
        logger.info(f"🚫 Blacklisted model {model} for {self._blacklist_ttl}s")

    def _clear_blacklist(self):
        self._model_failures.clear()
        logger.info("🧹 TextEngine blacklist cleared (restart)")

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

    async def process(
        self,
        input_data: str,
        context: Optional[Dict] = None,
        status_callback: Optional[Callable[[str, bool], Coroutine]] = None
    ) -> Tuple[str, str, int]:
        """
        Process text input with timeout and restart logic.
        `status_callback` is an async function that updates the placeholder.
        """
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
                if response and len(response.strip()) > 5 and not response.startswith(("🌐", "❌", "⚠️")):
                    logger.info(f"Cache hit for user {user_id}")
                    return response, "cache", 0

        complexity = self._estimate_complexity(input_data)
        base_models = self.model_manager.get_smart_models() if complexity == "smart" else self.model_manager.get_fast_models()

        # Build model list
        model_list = self._get_model_list(priority_list, base_models)
        logger.info(f"📋 Using model list (first 5): {model_list[:5]}...")

        category = self.prompt_library.detect_category(input_data)
        prompt_template = self.prompt_library.get_prompt(category)
        if len(input_data.split()) < 5 and category.value == "casual_conversation":
            prompt_template.max_tokens = min(prompt_template.max_tokens, 300)

        # ============================================================
        # TIMEOUT & RESTART LOGIC – fixed using timer task exception
        # ============================================================
        start_time = time.time()

        # Create a timer task that will raise RestartSearchException after the timeout
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
                # Main model search loop
                result = await self._execute_model_search(
                    model_list, input_data, history, prompt_template
                )
                if result:
                    response, model, tokens_used = result
                    timer_task.cancel()
                    # Save to cache
                    if user_id is not None:
                        self.user_data_manager.save_to_cache(input_data, response, category.value)
                    return response, model, tokens_used

                # If we get here, all models failed. Check if restart was triggered – safely.
                if timer_task.done():
                    exc = timer_task.exception()
                    if exc and isinstance(exc, RestartSearchException):
                        logger.info("Restart triggered, clearing blacklist and refreshing model list.")
                        self._clear_blacklist()
                        # Re-fetch model list (with fresh blacklist)
                        base_models = self.model_manager.get_smart_models() if complexity == "smart" else self.model_manager.get_fast_models()
                        model_list = self._get_model_list(priority_list, base_models)
                        continue
                else:
                    # Timer still running, no restart
                    break

            # If loop exits without success
            raise Exception("All text models failed. No restart triggered.")
        except Exception as e:
            timer_task.cancel()
            raise e
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
        """Background timer for search feedback and restart."""
        try:
            # Wait for search timeout to show the wait message
            elapsed = 0
            while elapsed < search_timeout:
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time
                if elapsed >= search_timeout:
                    break
            if status_callback:
                await status_callback("🤖 *AI Models Scarce, Please Wait ...*", edit=True)
                logger.info("⏳ Sent 'wait' message for text engine")

            # Wait for restart timeout
            while elapsed < restart_timeout:
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time
                if elapsed >= restart_timeout:
                    break

            # Trigger restart
            if status_callback:
                await status_callback("🔄 *Restarting search...*", edit=True)
            raise RestartSearchException("Restart search due to timeout")
        except asyncio.CancelledError:
            # Timer cancelled – success or another condition
            pass
        except RestartSearchException:
            # Reraise to be caught by process loop
            raise
        except Exception as e:
            logger.error(f"Search timer error: {e}")

    async def _execute_model_search(
        self,
        model_list: List[str],
        prompt: str,
        history: List[Dict],
        prompt_template
    ) -> Optional[Tuple[str, str, int]]:
        """Execute the model search with parallel and sequential fallback."""
        last_error = None

        # Parallel testing
        if Config.ENABLE_PARALLEL_MODEL_TESTING and len(model_list) > 1:
            result = await self._try_models_parallel_first_completed(
                model_list[:Config.PARALLEL_MODEL_ATTEMPTS],
                prompt, history, prompt_template
            )
            if result:
                response, model, tokens_used, error = result
                if response is not None:
                    logger.info(f"✅ Parallel test succeeded on {model} (tokens: {tokens_used})")
                    return response, model, tokens_used
                elif error:
                    self._mark_model_failure(model)
                    last_error = error
            remaining_models = model_list[Config.PARALLEL_MODEL_ATTEMPTS:]
            logger.info(f"Parallel failed, falling back to sequential for {len(remaining_models)} models")
        else:
            remaining_models = model_list

        # Sequential fallback with retries
        for model in remaining_models:
            for attempt in range(Config.HTTP_MAX_RETRIES):
                try:
                    logger.info(f"Trying model: {model} (attempt {attempt+1}/{Config.HTTP_MAX_RETRIES})")
                    response, tokens_used = await self._call_api(
                        model, prompt, history,
                        prompt_template.system_message,
                        prompt_template.temperature,
                        prompt_template.max_tokens
                    )
                    if response and len(response.strip()) > 5:
                        logger.info(f"✅ Model {model} succeeded (tokens: {tokens_used})")
                        return response, model, tokens_used
                    else:
                        logger.warning(f"Model {model} returned empty/short response")
                        self._mark_model_failure(model)
                        break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in self._non_retryable_errors:
                        logger.warning(f"Non-retryable error {e.response.status_code} for {model}, blacklisting.")
                        self._mark_model_failure(model)
                        break
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
        return None

    # ---------- existing helper methods (unchanged) ----------
    async def _try_models_parallel_first_completed(self, models: List[str], prompt: str, history: List[Dict],
                                                   prompt_template) -> Optional[Tuple[Optional[str], str, int, Optional[Exception]]]:
        tasks = {}
        for model in models:
            task = asyncio.create_task(
                self._call_api(model, prompt, history,
                               prompt_template.system_message,
                               prompt_template.temperature,
                               prompt_template.max_tokens)
            )
            tasks[task] = model
        pending = set(tasks.keys())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                model = tasks[task]
                try:
                    response, tokens = task.result()
                    if response is not None and len(response.strip()) > 5:
                        for t in pending:
                            t.cancel()
                        return (response, model, tokens, None)
                except Exception as e:
                    # This model failed
                    continue
            if not pending:
                break
        return None

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

# Custom exception for restart
class RestartSearchException(Exception):
    pass