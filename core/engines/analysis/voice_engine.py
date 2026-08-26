"""Voice/Speech-to-Text processing engine using local Whisper with OpenRouter fallback and priority support."""
import logging
import asyncio
import time
import httpx
from typing import Dict, Tuple, Optional, Any, List, Callable, Coroutine
import io
from core.config import Config
from core.managers.user_data_manager import UserDataManager
from core.managers.proxy_manager import ProxyManager

logger = logging.getLogger(__name__)

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not installed. Voice engine will rely on OpenRouter fallback.")

# Custom exception for restart
class RestartSearchException(Exception):
    pass


class VoiceEngine:
    def __init__(self, user_data_manager: Optional[UserDataManager] = None,
                 proxy_manager: Optional[ProxyManager] = None):
        self.user_data_manager = user_data_manager
        self.proxy_manager = proxy_manager
        self.is_initialized = False
        self.model = None
        self.model_name = "base"

        self.openrouter_models = Config.DEFAULT_VOICE_ENGINE_PRIORITY
        self._model_failures: Dict[str, float] = {}
        self._blacklist_ttl = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS
        self._non_retryable_errors = {404, 400, 402, 429}
        self._client = None

        # Timeout configuration
        self.search_timeout = Config.VOICE_SEARCH_TIMEOUT_SECONDS
        self.restart_timeout = Config.GLOBAL_RESTART_TIMEOUT_SECONDS

        logger.info("🔊 VoiceEngine __init__ done (with priority & blacklisting, timeout/restart, proxy)")

    async def initialize(self) -> bool:
        logger.info("🔊 VoiceEngine.initialize: START")

        if WHISPER_AVAILABLE:
            try:
                logger.info(f"🔊 Loading Whisper model: {self.model_name}...")
                self.model = await asyncio.to_thread(whisper.load_model, self.model_name)
                self.is_initialized = True
                logger.info(f"🔊 Whisper model '{self.model_name}' loaded successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper model: {e}", exc_info=True)
        else:
            logger.warning("Whisper module not installed. Will use OpenRouter fallback only.")

        try:
            timeout = httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=10.0, pool=10.0)
            client_kwargs = {"timeout": timeout}
            if self.proxy_manager:
                proxy_url = self.proxy_manager.get_proxy()
                if proxy_url:
                    client_kwargs["proxy"] = proxy_url
                    logger.info(f"🔊 VoiceEngine using proxy: {proxy_url}")

            try:
                self._client = httpx.AsyncClient(**client_kwargs, http2=True)
                logger.info("🔊 VoiceEngine OpenRouter client with HTTP/2")
            except Exception:
                logger.warning("HTTP/2 failed, falling back to HTTP/1.1")
                self._client = httpx.AsyncClient(**client_kwargs, http2=False)
                logger.info("🔊 VoiceEngine OpenRouter client with HTTP/1.1")

            self.is_initialized = True
            logger.info("🔊 VoiceEngine initialized with OpenRouter fallback client and proxy.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize VoiceEngine: {e}")
            return False

    async def shutdown(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self.is_initialized = False
        self.model = None
        logger.info("🔊 VoiceEngine shutdown complete")

    def _is_model_blacklisted(self, model: str) -> bool:
        if model in self._model_failures:
            if time.time() - self._model_failures[model] < self._blacklist_ttl:
                return True
            else:
                del self._model_failures[model]
        return False

    def _mark_model_failure(self, model: str):
        self._model_failures[model] = time.time()
        logger.info(f"🚫 Blacklisted OpenRouter STT model {model} for {self._blacklist_ttl}s")

    def _clear_blacklist(self):
        self._model_failures.clear()
        logger.info("🧹 VoiceEngine blacklist cleared (restart)")

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
            logger.warning("All STT models blacklisted, returning full fallback list.")
            return list(dict.fromkeys(fallback_list))
        return ordered

    async def transcribe(
        self,
        audio_bytes: bytes,
        context: Optional[Dict] = None,
        status_callback: Optional[Callable[[str, bool], Coroutine]] = None
    ) -> Tuple[str, str, int]:
        if not self.is_initialized:
            raise RuntimeError("Voice engine not initialized")

        # Check cancellation
        if asyncio.current_task().cancelled():
            raise asyncio.CancelledError

        logger.info(f"🔊 Transcribing audio (size: {len(audio_bytes)} bytes)")

        # ============================================================
        # PRIORITY 1: Local Whisper
        # ============================================================
        if self.model:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                result = await asyncio.to_thread(
                    self.model.transcribe,
                    tmp_path,
                    language=None,
                    task="transcribe",
                    fp16=False
                )
                import os
                os.unlink(tmp_path)
                transcription = result.get("text", "").strip()
                tokens_used = len(transcription) // 4
                logger.info(f"🔊 Local Whisper transcription: {transcription[:50]}...")
                return transcription, f"whisper-{self.model_name}", tokens_used
            except Exception as e:
                logger.error(f"❌ Local Whisper failed: {e}. Falling back to OpenRouter.")

        # ============================================================
        # PRIORITY 2: OpenRouter Fallback with timeout/restart
        # ============================================================
        if not self._client:
            raise RuntimeError("No transcription method available.")

        priority_list = context.get('priority_list') if context else None
        model_list = self._get_model_list(priority_list, self.openrouter_models)
        logger.info(f"📋 Trying OpenRouter STT models: {model_list[:3]}...")

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
                # Check cancellation
                if asyncio.current_task().cancelled():
                    raise asyncio.CancelledError

                result = await self._try_stt_models(model_list, audio_bytes)
                if result:
                    timer_task.cancel()
                    return result

                # Check if restart was triggered
                if timer_task.exception() and isinstance(timer_task.exception(), RestartSearchException):
                    logger.info("Restart triggered, clearing blacklist and refreshing model list.")
                    self._clear_blacklist()
                    model_list = self._get_model_list(priority_list, self.openrouter_models)
                    # Continue loop
                else:
                    break

            raise Exception("All transcription methods failed. No restart triggered.")
        except asyncio.CancelledError:
            timer_task.cancel()
            raise
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
                await status_callback("🔊 *AI Models Scarce, Please Wait ...*", edit=True)
                logger.info("⏳ Sent 'wait' message for voice engine")

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
            logger.error(f"Voice timer error: {e}")

    async def _try_stt_models(self, model_list: List[str], audio_bytes: bytes) -> Optional[Tuple[str, str, int]]:
        # Parallel testing for OpenRouter models
        if Config.ENABLE_PARALLEL_MODEL_TESTING and len(model_list) > 1:
            result = await self._try_stt_parallel_first_completed(
                model_list[:Config.PARALLEL_MODEL_ATTEMPTS],
                audio_bytes
            )
            if result:
                response, model, tokens, error = result
                if response is not None:
                    logger.info(f"✅ OpenRouter STT parallel success on {model}")
                    return response, model, tokens
                elif error:
                    self._mark_model_failure(model)
            remaining_models = model_list[Config.PARALLEL_MODEL_ATTEMPTS:]
            logger.info(f"Parallel STT failed, falling back to sequential for {len(remaining_models)} models")
        else:
            remaining_models = model_list

        # Sequential fallback
        for model in remaining_models:
            # Check cancellation
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError

            for attempt in range(Config.HTTP_MAX_RETRIES):
                try:
                    logger.info(f"🔊 Trying OpenRouter STT: {model} (attempt {attempt+1})")
                    response, tokens = await self._call_openrouter_stt(model, audio_bytes)
                    logger.info(f"✅ OpenRouter STT {model} succeeded")
                    return response, model, tokens
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in self._non_retryable_errors:
                        logger.warning(f"Non-retryable error {e.response.status_code} for {model}, blacklisting.")
                        self._mark_model_failure(model)
                        break
                    else:
                        logger.warning(f"OpenRouter STT {model} attempt {attempt+1} failed: {e}")
                        if attempt < Config.HTTP_MAX_RETRIES - 1:
                            await asyncio.sleep(0.5)
                        else:
                            self._mark_model_failure(model)
                            break
                except Exception as e:
                    logger.warning(f"OpenRouter STT {model} attempt {attempt+1} failed: {e}")
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
                    else:
                        self._mark_model_failure(model)
                        break
        return None

    async def _try_stt_parallel_first_completed(self, models: List[str], audio_bytes: bytes) -> Optional[Tuple[Optional[str], str, int, Optional[Exception]]]:
        tasks = {}
        for model in models:
            task = asyncio.create_task(self._call_openrouter_stt_with_errors(model, audio_bytes))
            tasks[task] = model

        pending = set(tasks.keys())
        while pending:
            # Check cancellation during parallel wait
            if asyncio.current_task().cancelled():
                for t in pending:
                    t.cancel()
                raise asyncio.CancelledError

            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                model = tasks[task]
                try:
                    text, tokens = task.result()
                    if text is not None:
                        for t in pending:
                            t.cancel()
                        return (text, model, tokens, None)
                except Exception as e:
                    continue
            if not pending:
                break
        return None

    async def _call_openrouter_stt_with_errors(self, model: str, audio_bytes: bytes) -> Tuple[str, int]:
        try:
            return await self._call_openrouter_stt(model, audio_bytes)
        except Exception as e:
            raise e

    async def _call_openrouter_stt(self, model: str, audio_bytes: bytes) -> Tuple[str, int]:
        headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
            "HTTP-Referer": Config.BOT_REPO_URL,
            "X-Title": Config.BOT_NAME
        }

        files = {
            "audio": ("audio.ogg", audio_bytes, "audio/ogg"),
            "model": (None, model),
            "response_format": (None, "text")
        }

        try:
            resp = await self._client.post(
                "https://openrouter.ai/api/v1/audio/transcriptions",
                headers=headers,
                files=files,
                timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
            transcription = data.get("text", "").strip() if isinstance(data, dict) else str(data).strip()
            tokens_used = len(transcription) // 4
            return transcription, tokens_used
        except Exception as e:
            logger.error(f"OpenRouter STT call failed for {model}: {e}")
            raise

    def get_engine_info(self) -> Dict:
        return {
            "type": "VoiceEngine",
            "initialized": self.is_initialized,
            "whisper_model": self.model_name if self.model else None,
            "openrouter_fallback_enabled": self._client is not None,
            "blacklisted_models": list(self._model_failures.keys())
        }