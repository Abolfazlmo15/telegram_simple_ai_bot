"""Voice/Speech-to-Text processing engine using local Whisper."""
import logging
import asyncio
from typing import Dict, Tuple, Optional, Any, List
import io
from core.config import Config
from core.managers.user_data_manager import UserDataManager

logger = logging.getLogger(__name__)

# Try to import whisper – if not available, engine will be disabled
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not installed. Voice engine will be disabled.")


class VoiceEngine:
    def __init__(self, user_data_manager: Optional[UserDataManager] = None):
        self.user_data_manager = user_data_manager
        self.is_initialized = False
        self.model = None
        self.model_name = "base"  # can be "tiny", "base", "small", "medium", "large"
        logger.info("🔊 VoiceEngine __init__ done")

    async def initialize(self) -> bool:
        logger.info("🔊 VoiceEngine.initialize: START")
        if not WHISPER_AVAILABLE:
            logger.error("❌ Whisper module not installed – voice engine disabled")
            return False

        try:
            # Load the model in a thread to avoid blocking the event loop
            logger.info(f"🔊 Loading Whisper model: {self.model_name}...")
            self.model = await asyncio.to_thread(whisper.load_model, self.model_name)
            self.is_initialized = True
            logger.info(f"🔊 Whisper model '{self.model_name}' loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}", exc_info=True)
            return False

    async def shutdown(self) -> None:
        self.is_initialized = False
        self.model = None
        logger.info("🔊 VoiceEngine shutdown complete")

    async def transcribe(self, audio_bytes: bytes, context: Optional[Dict] = None) -> Tuple[str, str, int]:
        """
        Transcribe audio using local Whisper.
        Returns: (transcription_text, model_used, tokens_used)
        """
        if not self.is_initialized:
            raise RuntimeError("Voice engine not initialized")

        logger.info(f"🔊 Transcribing audio (size: {len(audio_bytes)} bytes)")

        # Whisper expects a file path or a file-like object; we'll write to temp or use BytesIO
        # But whisper can accept a file-like object if we pass it correctly.
        # Using a temporary file is safer.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Run transcription in a thread
            result = await asyncio.to_thread(
                self.model.transcribe,
                tmp_path,
                language=None,           # auto-detect
                task="transcribe",       # or "translate"
                fp16=False               # use FP32 for CPU
            )
            transcription = result.get("text", "").strip()
            # Estimate tokens: rough ~4 chars per token
            tokens_used = len(transcription) // 4
            logger.info(f"🔊 Transcription: {transcription[:50]}...")
            return transcription, f"whisper-{self.model_name}", tokens_used
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}", exc_info=True)
            raise
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def get_engine_info(self) -> Dict:
        return {
            "type": "VoiceEngine",
            "initialized": self.is_initialized,
            "model": self.model_name if self.is_initialized else None
        }