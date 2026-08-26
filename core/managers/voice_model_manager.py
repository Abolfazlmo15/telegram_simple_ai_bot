"""Dynamic manager for voice/speech-to-text models."""
import logging
import threading
import time
import httpx
from typing import List, Optional
from core.config import Config

logger = logging.getLogger(__name__)


class VoiceModelManager:
    """
    Manages available speech-to-text models from OpenRouter.
    Auto-updates every 10 minutes with live model availability.
    """

    def __init__(self):
        self.available_models: List[str] = []
        self.is_running = False
        self.update_interval = 600  # 10 minutes
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Known free Whisper models (fallback)
        self.known_voice_models = [
            "openai/whisper-large-v3-turbo:free",
            "openai/whisper-large-v3:free",
            "openai/whisper-large-v2:free",
            "openai/whisper-small:free",
            "openai/whisper-medium:free",
            "openai/whisper-base:free",
        ]
        logger.info("🔊 VoiceModelManager initialized")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._fetch_and_update_models()
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        logger.info("Voice model manager started (updates every 10 minutes)")

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Voice model manager stopped")

    def _background_loop(self):
        while self.is_running:
            time.sleep(self.update_interval)
            if self.is_running:
                self._fetch_and_update_models()

    def _fetch_and_update_models(self):
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            with httpx.Client(timeout=15.0) as client:
                resp = client.get("https://openrouter.ai/api/v1/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()

                all_models = data.get("data", [])
                voice_models = []

                # Filter for Whisper models specifically
                for model in all_models:
                    model_id = model.get("id", "").lower()
                    if "whisper" in model_id:
                        voice_models.append(model_id)

                # Also add known models that might have been missed
                for known in self.known_voice_models:
                    if known not in voice_models:
                        base = known.split(':')[0]
                        if any(base in m.get("id", "") for m in all_models):
                            voice_models.append(known)

                with self._lock:
                    if voice_models:
                        self.available_models = voice_models
                        logger.info(f"Voice models updated: {len(voice_models)} models available")
                    else:
                        self.available_models = self.known_voice_models.copy()
                        logger.warning("No voice models detected, using known fallback models")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                logger.warning("⚠️ OpenRouter voice models require payment (402). Using fallback models.")
            else:
                logger.error(f"Failed to fetch voice models: {e}")
            with self._lock:
                if not self.available_models:
                    self.available_models = self.known_voice_models.copy()
        except Exception as e:
            logger.error(f"Failed to fetch voice models: {e}")
            with self._lock:
                if not self.available_models:
                    self.available_models = self.known_voice_models.copy()

    def get_available_models(self) -> List[str]:
        with self._lock:
            return self.available_models.copy() if self.available_models else self.known_voice_models.copy()