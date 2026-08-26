"""Voice generation engine using OpenRouter TTS models with gTTS fallback and priority support."""
import logging
import asyncio
import time
from typing import Dict, Tuple, Optional, Any, List
import httpx
from io import BytesIO
from core.config import Config
from core.managers.user_data_manager import UserDataManager
from core.utils.markdown_stripper import MarkdownStripper

logger = logging.getLogger(__name__)

# Try to import gTTS for fallback
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS not installed. Fallback TTS will be disabled.")


class VoiceGenerationEngine:
    """
    Text-to-Speech generation engine.
    Uses OpenRouter TTS models with automatic fallback to gTTS.
    Supports markdown stripping and voice preferences.
    """
    def __init__(self, user_data_manager: Optional[UserDataManager] = None):
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.user_data_manager = user_data_manager
        self._client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False

        # ============================================================
        # FIXED: Correct TTS models (using config defaults)
        # ============================================================
        self.models = Config.DEFAULT_VOICE_GEN_PRIORITY

        self.default_voice = Config.TTS_DEFAULT_VOICE
        self.max_text_length = Config.TTS_MAX_TEXT_LENGTH
        self.use_gtts_fallback = GTTS_AVAILABLE
        self.response_format = "mp3"
        self.markdown_stripper = MarkdownStripper()

        # Voice style mapping
        self.voice_style_map = {
            "neutral": "neutral", "happy": "happy", "serious": "serious",
            "excited": "excited", "sad": "sad", "angry": "angry",
            "whisper": "whisper", "loud": "loud"
        }

        # ============================================================
        # PERFORMANCE: Blacklisting & failure tracking
        # ============================================================
        self._model_failures: Dict[str, float] = {}
        self._blacklist_ttl = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS
        self._non_retryable_errors = {404, 400, 402, 429}

        logger.info("🔊 VoiceGenerationEngine __init__ done (with priority & blacklisting)")

    async def initialize(self) -> bool:
        try:
            timeout = httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=30.0, pool=10.0)
            try:
                self._client = httpx.AsyncClient(timeout=timeout, http2=True, limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5))
                logger.info("🔊 VoiceGenerationEngine HTTP/2 client created")
            except Exception:
                logger.warning("HTTP/2 failed, falling back to HTTP/1.1")
                self._client = httpx.AsyncClient(timeout=timeout, http2=False, limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5))
                logger.info("🔊 VoiceGenerationEngine HTTP/1.1 client created")
            self.is_initialized = True
            logger.info("🔊 VoiceGenerationEngine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize VoiceGenerationEngine: {e}", exc_info=True)
            return False

    async def shutdown(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self.is_initialized = False
        logger.info("🔊 VoiceGenerationEngine shutdown complete")

    def _is_model_blacklisted(self, model: str) -> bool:
        if model in self._model_failures:
            if time.time() - self._model_failures[model] < self._blacklist_ttl:
                return True
            else:
                del self._model_failures[model]
        return False

    def _mark_model_failure(self, model: str):
        self._model_failures[model] = time.time()
        logger.info(f"🚫 Blacklisted TTS model {model} for {self._blacklist_ttl}s")

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
            logger.warning("All TTS models blacklisted, returning full fallback list.")
            return list(dict.fromkeys(fallback_list))
        return ordered

    async def generate(self, text: str, context: Optional[Dict] = None) -> Tuple[bytes, str, int]:
        if not self.is_initialized:
            raise RuntimeError("Voice generation engine not initialized")
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided for TTS")

        clean_text = self.markdown_stripper.strip_for_tts(text)
        if len(clean_text) > self.max_text_length:
            clean_text = clean_text[:self.max_text_length] + "..."

        voice_speed = 1.0
        voice_style = "neutral"
        if context:
            voice_speed = context.get('voice_speed', 1.0)
            voice_style = context.get('voice_style', 'neutral')
        voice_speed = max(0.5, min(2.0, voice_speed))

        priority_list = context.get('priority_list') if context else None
        model_list = self._get_model_list(priority_list, self.models)
        logger.info(f"📋 TTS model list (first 3): {model_list[:3]}...")

        last_error = None

        # Parallel testing for TTS
        if Config.ENABLE_PARALLEL_MODEL_TESTING and len(model_list) > 1:
            result = await self._try_tts_parallel_first_completed(
                model_list[:Config.PARALLEL_MODEL_ATTEMPTS],
                clean_text, voice_speed, voice_style
            )
            if result:
                audio, model, size, error = result
                if audio is not None:
                    logger.info(f"✅ TTS parallel success on {model}")
                    if self.user_data_manager and context and 'user_id' in context:
                        await self._save_voice_to_history(context, text, audio, model)
                    return audio, model, size
                elif error:
                    self._mark_model_failure(model)
                    last_error = error

            remaining_models = model_list[Config.PARALLEL_MODEL_ATTEMPTS:]
            logger.info(f"Parallel TTS failed, falling back to sequential for {len(remaining_models)} models")
        else:
            remaining_models = model_list

        # Sequential fallback
        for model in remaining_models:
            for attempt in range(Config.HTTP_MAX_RETRIES):
                try:
                    logger.info(f"🔊 Trying TTS model: {model} (attempt {attempt+1})")
                    audio_bytes = await self._call_tts_api(model, clean_text, voice_speed, voice_style)
                    logger.info(f"✅ TTS model {model} succeeded")
                    if self.user_data_manager and context and 'user_id' in context:
                        await self._save_voice_to_history(context, text, audio_bytes, model)
                    return audio_bytes, model, len(audio_bytes)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in self._non_retryable_errors:
                        logger.warning(f"Non-retryable error {e.response.status_code} for {model}, blacklisting.")
                        self._mark_model_failure(model)
                        break
                    else:
                        logger.warning(f"⚠️ TTS model {model} attempt {attempt+1} failed: {e}")
                        last_error = e
                        if attempt < Config.HTTP_MAX_RETRIES - 1:
                            await asyncio.sleep(0.5)
                        else:
                            self._mark_model_failure(model)
                            break
                except Exception as e:
                    logger.warning(f"⚠️ TTS model {model} attempt {attempt+1} failed: {e}")
                    last_error = e
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
                    else:
                        self._mark_model_failure(model)
                        break

        # Fallback to gTTS
        if self.use_gtts_fallback:
            logger.info("🔊 Falling back to gTTS for TTS")
            try:
                audio_bytes = await self._call_gtts(clean_text, voice_speed)
                logger.info("✅ gTTS fallback succeeded")
                return audio_bytes, "gtts", len(audio_bytes)
            except Exception as e:
                logger.error(f"❌ gTTS fallback failed: {e}")
                last_error = e

        raise Exception(f"All TTS models and fallbacks failed. Last error: {last_error}")

    async def _try_tts_parallel_first_completed(self, models: List[str], text: str, speed: float, style: str) -> Optional[Tuple[Optional[bytes], str, int, Optional[Exception]]]:
        tasks = {}
        for model in models:
            task = asyncio.create_task(self._call_tts_api_with_errors(model, text, speed, style))
            tasks[task] = model

        pending = set(tasks.keys())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                model = tasks[task]
                try:
                    audio_bytes = task.result()
                    if audio_bytes is not None:
                        for t in pending:
                            t.cancel()
                        return (audio_bytes, model, len(audio_bytes), None)
                except Exception as e:
                    # Continue waiting for other models
                    pass
            if not pending:
                break

        return None

    async def _call_tts_api_with_errors(self, model: str, text: str, speed: float, style: str) -> bytes:
        try:
            return await self._call_tts_api(model, text, speed, style)
        except Exception as e:
            raise e

    async def _call_tts_api(self, model: str, text: str, speed: float, style: str) -> bytes:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.BOT_REPO_URL,
            "X-Title": Config.BOT_NAME
        }
        voice = self._get_voice_for_model(model)
        payload = {"model": model, "input": text, "voice": voice, "response_format": self.response_format}
        if speed != 1.0:
            payload["speed"] = speed

        resp = await self._client.post(Config.OPENROUTER_TTS_URL, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.content

    def _get_voice_for_model(self, model: str) -> str:
        model_lower = model.lower()
        if "deepgram" in model_lower or "flux-tts" in model_lower:
            return "flux-alexis-en"
        if "openai" in model_lower:
            return self.default_voice
        return self.default_voice

    async def _call_gtts(self, text: str, speed: float = 1.0) -> bytes:
        def _generate():
            tts = gTTS(text=text, lang='en', slow=(speed < 1.0))
            buffer = BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer.read()
        return await asyncio.to_thread(_generate)

    async def _save_voice_to_history(self, context: Dict, text: str, audio_bytes: bytes, model: str):
        try:
            user_id = context['user_id']
            username = context.get('username')
            audio_path = await self.user_data_manager.save_audio_file(user_id, username, audio_bytes)
            await self.user_data_manager.add_generated_voice_to_history(
                user_id=user_id, username=username, prompt=text, response="Voice generated successfully",
                audio_file=audio_path, model_used=model, response_time=0.0, tokens_used=0
            )
        except Exception as e:
            logger.error(f"Failed to save voice to history: {e}")

    def get_engine_info(self) -> Dict:
        return {
            "type": "VoiceGenerationEngine",
            "initialized": self.is_initialized,
            "models": self.models,
            "gtts_fallback": self.use_gtts_fallback,
            "blacklisted_models": list(self._model_failures.keys())
        }