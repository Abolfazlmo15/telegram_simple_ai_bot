"""Voice generation engine using OpenRouter TTS models with gTTS fallback and priority support."""
import logging
import asyncio
import time
from typing import Dict, Tuple, Optional, Any, List, Callable, Coroutine
import httpx
from io import BytesIO
from core.config import Config
from core.managers.user_data_manager import UserDataManager
from core.managers.proxy_manager import ProxyManager
from core.utils.markdown_stripper import MarkdownStripper

logger = logging.getLogger(__name__)

# Try to import gTTS for fallback
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS not installed. Fallback TTS will be disabled.")


# ------------------------------------------------------------------
# Retry helper (exponential backoff)
# ------------------------------------------------------------------
async def _retry_async(func, *args, max_retries=3, base_delay=0.5, **kwargs):
    """Retry an async function with exponential backoff and jitter."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + (0.1 * attempt)
                await asyncio.sleep(delay)
                continue
            raise
        except Exception as e:
            # Non‑network errors are re‑raised immediately
            raise
    raise last_exc


class VoiceGenerationEngine:
    """
    Text-to-Speech generation engine.
    Uses OpenRouter TTS models with automatic fallback to gTTS.
    Supports markdown stripping and voice preferences.
    Splits long text into chunks to avoid audio truncation.
    """
    def __init__(self, user_data_manager: Optional[UserDataManager] = None,
                 proxy_manager: Optional[ProxyManager] = None):
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.user_data_manager = user_data_manager
        self.proxy_manager = proxy_manager
        self._client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False

        # Correct TTS models (using config defaults)
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

        # Blacklisting & failure tracking
        self._model_failures: Dict[str, float] = {}
        self._blacklist_ttl = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS
        self._non_retryable_errors = {404, 400, 402, 429}

        # Chunking settings
        self.chunk_char_limit = 400  # Approximate safe limit per request to avoid truncation

        logger.info("🔊 VoiceGenerationEngine __init__ done (with priority & blacklisting, proxy, chunking)")

    async def initialize(self) -> bool:
        try:
            timeout = httpx.Timeout(connect=10.0, read=Config.HTTP_TIMEOUT, write=30.0, pool=10.0)
            client_kwargs = {"timeout": timeout}
            if self.proxy_manager:
                proxy_url = self.proxy_manager.get_proxy()
                if proxy_url:
                    client_kwargs["proxy"] = proxy_url
                    logger.info(f"🔊 VoiceGenerationEngine using proxy: {proxy_url}")

            try:
                self._client = httpx.AsyncClient(
                    **client_kwargs,
                    http2=True,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
                logger.info("🔊 VoiceGenerationEngine HTTP/2 client created")
            except Exception:
                logger.warning("HTTP/2 failed, falling back to HTTP/1.1")
                self._client = httpx.AsyncClient(
                    **client_kwargs,
                    http2=False,
                    limits=httpx.Limits(max_connections=Config.CONNECTION_POOL_SIZE, max_keepalive_connections=5)
                )
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

    async def generate(self, text: str, context: Optional[Dict] = None,
                       status_callback: Optional[Callable[[str, bool], Coroutine]] = None) -> Tuple[bytes, str, int]:
        """
        Generate TTS audio from text.
        Splits long text into chunks to avoid audio truncation.
        """
        if not self.is_initialized:
            raise RuntimeError("Voice generation engine not initialized")

        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided for TTS")

        clean_text = self.markdown_stripper.strip_for_tts(text)
        if len(clean_text) == 0:
            clean_text = "I have nothing to say."
            logger.warning("Text became empty after markdown stripping, using fallback")

        # Overall cap
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

        # ============================================================
        # CHUNKING LOGIC
        # ============================================================
        chunks = self._split_text_into_chunks(clean_text, self.chunk_char_limit)
        if len(chunks) == 1:
            # Short text, generate as single request
            return await self._generate_single_chunk(chunks[0], model_list, voice_speed, voice_style, context, status_callback)

        # Multiple chunks – generate each and combine
        logger.info(f"🔊 Splitting text into {len(chunks)} chunks")
        audio_parts = []
        used_model = None
        total_size = 0

        for idx, chunk in enumerate(chunks):
            if status_callback:
                await status_callback(f"🔊 Generating voice part {idx+1}/{len(chunks)}...", edit=True)

            try:
                chunk_audio, model, size = await self._generate_single_chunk(
                    chunk, model_list, voice_speed, voice_style, context, status_callback=None
                )
                audio_parts.append(chunk_audio)
                total_size += size
                if used_model is None:
                    used_model = model
                # Slight delay between chunks to avoid rate limiting
                if idx < len(chunks) - 1:
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"❌ Failed to generate chunk {idx+1}: {e}")
                # If a chunk fails, propagate the error
                raise

        # Combine all audio bytes (simple concatenation works for MP3)
        combined_audio = b''.join(audio_parts)
        logger.info(f"🔊 Combined {len(audio_parts)} chunks into {len(combined_audio)} bytes")

        # Save to history if needed (only once)
        if self.user_data_manager and context and 'user_id' in context:
            await self._save_voice_to_history(context, text, combined_audio, used_model or "chunked")

        return combined_audio, used_model or "chunked", total_size

    async def _generate_single_chunk(self, text: str, model_list: List[str],
                                     speed: float, style: str, context: Optional[Dict],
                                     status_callback: Optional[Callable[[str, bool], Coroutine]] = None) -> Tuple[bytes, str, int]:
        """
        Generate audio for a single text chunk using the model fallback chain.
        """
        last_error = None

        # Parallel testing for TTS
        if Config.ENABLE_PARALLEL_MODEL_TESTING and len(model_list) > 1:
            result = await self._try_tts_parallel_first_completed(
                model_list[:Config.PARALLEL_MODEL_ATTEMPTS],
                text, speed, style
            )
            if result:
                audio, model, size, error = result
                if audio is not None:
                    logger.info(f"✅ TTS parallel success on {model}")
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
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError

            for attempt in range(Config.HTTP_MAX_RETRIES):
                try:
                    logger.info(f"🔊 Trying TTS model: {model} (attempt {attempt+1})")
                    audio_bytes = await _retry_async(
                        self._call_tts_api,
                        model, text, speed, style,
                        max_retries=2
                    )
                    if audio_bytes and len(audio_bytes) > 0:
                        logger.info(f"✅ TTS model {model} succeeded")
                        return audio_bytes, model, len(audio_bytes)
                    else:
                        logger.warning(f"TTS model {model} returned empty audio")
                        self._mark_model_failure(model)
                        break
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
                audio_bytes = await self._call_gtts(text, speed)
                if audio_bytes and len(audio_bytes) > 0:
                    logger.info("✅ gTTS fallback succeeded")
                    return audio_bytes, "gtts", len(audio_bytes)
            except Exception as e:
                logger.error(f"❌ gTTS fallback failed: {e}")
                last_error = e

        raise Exception(f"All TTS models and fallbacks failed. Last error: {last_error}")

    def _split_text_into_chunks(self, text: str, max_chars: int) -> List[str]:
        """
        Split text into chunks of roughly max_chars, preferring sentence boundaries.
        """
        if len(text) <= max_chars:
            return [text]

        # Try to split by sentence delimiters: . ! ?
        sentences = []
        import re
        # Split on punctuation followed by space or end
        parts = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0
        for part in parts:
            part_len = len(part)
            if current_len + part_len + 1 <= max_chars:
                current.append(part)
                current_len += part_len + 1
            else:
                if current:
                    chunks.append(' '.join(current))
                current = [part]
                current_len = part_len + 1
        if current:
            chunks.append(' '.join(current))

        # If a single chunk is still too long, force split by words
        if not chunks or max(len(c) for c in chunks) > max_chars:
            # Fallback: split by words
            words = text.split()
            chunks = []
            current_chunk = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 <= max_chars:
                    current_chunk.append(word)
                    current_len += len(word) + 1
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_len = len(word) + 1
            if current_chunk:
                chunks.append(' '.join(current_chunk))

        # Ensure no empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks if chunks else [text[:max_chars]]

    # ------------------------------------------------------------------
    # Parallel and API helpers (unchanged)
    # ------------------------------------------------------------------
    async def _try_tts_parallel_first_completed(self, models: List[str], text: str, speed: float, style: str) -> Optional[Tuple[Optional[bytes], str, int, Optional[Exception]]]:
        tasks = {}
        for model in models:
            task = asyncio.create_task(self._call_tts_api_with_errors(model, text, speed, style))
            tasks[task] = model

        pending = set(tasks.keys())
        while pending:
            if asyncio.current_task().cancelled():
                for t in pending:
                    t.cancel()
                raise asyncio.CancelledError

            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                model = tasks[task]
                try:
                    audio_bytes = task.result()
                    if audio_bytes is not None and len(audio_bytes) > 0:
                        for t in pending:
                            t.cancel()
                        return (audio_bytes, model, len(audio_bytes), None)
                except Exception as e:
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

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": self.response_format
        }
        if speed != 1.0:
            payload["speed"] = speed

        logger.debug(f"TTS payload: model={model}, voice={voice}, text_len={len(text)}")

        resp = await self._client.post(
            Config.OPENROUTER_TTS_URL,
            headers=headers,
            json=payload,
            timeout=30.0
        )
        resp.raise_for_status()
        content = resp.content
        if not content or len(content) < 100:
            logger.warning(f"TTS response too small ({len(content)} bytes) for model {model}")
        return content

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
                user_id=user_id,
                username=username,
                prompt=text,
                response="Voice generated successfully",
                audio_file=audio_path,
                model_used=model,
                response_time=0.0,
                tokens_used=0
            )
        except Exception as e:
            logger.error(f"Failed to save voice to history: {e}")

    def get_engine_info(self) -> Dict:
        return {
            "type": "VoiceGenerationEngine",
            "initialized": self.is_initialized,
            "models": self.models,
            "gtts_fallback": self.use_gtts_fallback,
            "chunk_limit": self.chunk_char_limit,
            "blacklisted_models": list(self._model_failures.keys())
        }