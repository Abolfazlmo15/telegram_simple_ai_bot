"""Voice generation engine using OpenRouter TTS models with gTTS fallback.
Supports markdown stripping, voice style, and speed preferences."""
import logging
import asyncio
from typing import Dict, Tuple, Optional, Any
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
        # TTS models via OpenRouter - using verified working models
        self.models = [
            "openai/gpt-4o-mini-tts-2025-12-15",  # OpenAI TTS (recommended)
            "deepgram/flux-tts:free",              # Free tier TTS
            "openai/tts-1",
            "openai/tts-1-hd",
        ]
        # Voice options depend on the model
        self.default_voice = Config.TTS_DEFAULT_VOICE
        self.max_text_length = Config.TTS_MAX_TEXT_LENGTH
        self.use_gtts_fallback = GTTS_AVAILABLE
        self.response_format = "mp3"  # mp3, pcm, wav

        # ============================================================
        # NEW: Markdown stripper
        # ============================================================
        self.markdown_stripper = MarkdownStripper()

        # Voice style mapping for different content types
        self.voice_style_map = {
            "neutral": "neutral",
            "happy": "happy",
            "serious": "serious",
            "excited": "excited",
            "sad": "sad",
            "angry": "angry",
            "whisper": "whisper",
            "loud": "loud",
        }

        logger.info("🔊 VoiceGenerationEngine __init__ done")

    async def initialize(self) -> bool:
        """Initialize the HTTP client."""
        try:
            timeout = httpx.Timeout(
                connect=10.0,
                read=Config.HTTP_TIMEOUT,
                write=30.0,
                pool=10.0
            )
            # Try HTTP/2 first
            try:
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=True,
                    limits=httpx.Limits(
                        max_connections=Config.CONNECTION_POOL_SIZE,
                        max_keepalive_connections=5
                    )
                )
                logger.info("🔊 VoiceGenerationEngine HTTP/2 client created")
            except Exception:
                logger.warning("HTTP/2 failed, falling back to HTTP/1.1")
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    http2=False,
                    limits=httpx.Limits(
                        max_connections=Config.CONNECTION_POOL_SIZE,
                        max_keepalive_connections=5
                    )
                )
                logger.info("🔊 VoiceGenerationEngine HTTP/1.1 client created")
            self.is_initialized = True
            logger.info("🔊 VoiceGenerationEngine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize VoiceGenerationEngine: {e}", exc_info=True)
            return False

    async def shutdown(self) -> None:
        """Shutdown the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self.is_initialized = False
        logger.info("🔊 VoiceGenerationEngine shutdown complete")

    async def generate(self, text: str, context: Optional[Dict] = None) -> Tuple[bytes, str, int]:
        """
        Generate audio from text.
        Returns: (audio_bytes, model_used, size_bytes)
        """
        if not self.is_initialized:
            raise RuntimeError("Voice generation engine not initialized")

        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided for TTS")

        # ============================================================
        # NEW: Strip markdown and emojis before TTS
        # ============================================================
        clean_text = self.markdown_stripper.strip_for_tts(text)
        logger.info(f"🔊 Cleaned text for TTS: {clean_text[:50]}...")

        # Truncate if too long
        if len(clean_text) > self.max_text_length:
            clean_text = clean_text[:self.max_text_length] + "..."

        # ============================================================
        # NEW: Get voice preferences from context
        # ============================================================
        voice_speed = 1.0
        voice_style = "neutral"

        if context:
            voice_speed = context.get('voice_speed', 1.0)
            voice_style = context.get('voice_style', 'neutral')

        # Clamp speed to valid range
        voice_speed = max(0.5, min(2.0, voice_speed))
        if voice_style not in self.voice_style_map:
            voice_style = "neutral"

        logger.info(f"🔊 Generating speech for: {clean_text[:50]}... (speed: {voice_speed}, style: {voice_style})")

        # Try OpenRouter models first
        last_error = None
        for model in self.models:
            for attempt in range(Config.HTTP_MAX_RETRIES):
                try:
                    logger.info(f"🔊 Trying TTS model: {model} (attempt {attempt+1})")
                    audio_bytes = await self._call_tts_api(model, clean_text, voice_speed, voice_style)
                    logger.info(f"✅ TTS model {model} succeeded")
                    # Save audio to user's history
                    if self.user_data_manager and context and 'user_id' in context:
                        user_id = context['user_id']
                        username = context.get('username')
                        audio_path = await self.user_data_manager.save_audio_file(
                            user_id, username, audio_bytes
                        )
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
                    return audio_bytes, model, len(audio_bytes)
                except Exception as e:
                    logger.warning(f"⚠️ TTS model {model} attempt {attempt+1} failed: {e}")
                    last_error = e
                    if attempt < Config.HTTP_MAX_RETRIES - 1:
                        await asyncio.sleep(1)
                    continue

        # Fallback to gTTS
        if self.use_gtts_fallback:
            logger.info("🔊 Falling back to gTTS for TTS")
            try:
                audio_bytes = await self._call_gtts(clean_text, voice_speed)
                logger.info("✅ gTTS fallback succeeded")
                # Save to history as gTTS
                if self.user_data_manager and context and 'user_id' in context:
                    user_id = context['user_id']
                    username = context.get('username')
                    audio_path = await self.user_data_manager.save_audio_file(
                        user_id, username, audio_bytes
                    )
                    await self.user_data_manager.add_generated_voice_to_history(
                        user_id=user_id,
                        username=username,
                        prompt=text,
                        response="Voice generated successfully (gTTS)",
                        audio_file=audio_path,
                        model_used="gtts",
                        response_time=0.0,
                        tokens_used=0
                    )
                return audio_bytes, "gtts", len(audio_bytes)
            except Exception as e:
                logger.error(f"❌ gTTS fallback failed: {e}")
                last_error = e

        raise Exception(f"All TTS models and fallbacks failed. Last error: {last_error}")

    async def _call_tts_api(self, model: str, text: str, speed: float, style: str) -> bytes:
        """
        Call OpenRouter TTS API.

        The endpoint is OpenAI-compatible and returns raw audio bytes.
        Required fields: model, input, voice, response_format
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.BOT_REPO_URL,
            "X-Title": Config.BOT_NAME
        }

        # Determine which voice to use based on the model
        voice = self._get_voice_for_model(model)

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": self.response_format,  # REQUIRED
        }

        # Speed is optional (0.5 to 2.0)
        if speed != 1.0:
            payload["speed"] = speed

        logger.info(f"🔊 Calling TTS API with model: {model}, voice: {voice}, speed: {speed}")

        resp = await self._client.post(
            Config.OPENROUTER_TTS_URL,
            headers=headers,
            json=payload
        )
        resp.raise_for_status()

        # The response is raw audio bytes, not JSON
        return resp.content

    def _get_voice_for_model(self, model: str) -> str:
        """Return the appropriate voice for each model."""
        model_lower = model.lower()

        # Deepgram/Flux TTS voices
        if "deepgram" in model_lower or "flux-tts" in model_lower:
            return "flux-alexis-en"

        # OpenAI TTS voices: alloy, echo, fable, onyx, nova, shimmer
        if "openai" in model_lower:
            return self.default_voice  # "alloy"

        # Default fallback
        return self.default_voice

    async def _call_gtts(self, text: str, speed: float = 1.0) -> bytes:
        """Fallback using gTTS (free, no API key)."""
        def _generate():
            tts = gTTS(text=text, lang='en', slow=(speed < 1.0))
            buffer = BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer.read()

        return await asyncio.to_thread(_generate)

    def get_engine_info(self) -> Dict:
        """Return engine information."""
        return {
            "type": "VoiceGenerationEngine",
            "initialized": self.is_initialized,
            "models": self.models,
            "gtts_fallback": self.use_gtts_fallback,
            "default_voice": self.default_voice,
            "response_format": self.response_format,
            "max_text_length": self.max_text_length,
            "markdown_stripper": True,
            "voice_styles": list(self.voice_style_map.keys())
        }