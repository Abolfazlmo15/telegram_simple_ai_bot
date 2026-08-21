"""Dynamic manager for vision-capable models."""
import logging
import threading
import time
import httpx
from typing import List
from core.config import Config

logger = logging.getLogger(__name__)


class VisionModelManager:
    """
    Manages available vision-capable models from OpenRouter.
    Auto-updates every 10 minutes with live model availability.
    """

    def __init__(self):
        self.available_models: List[str] = []
        self.is_running = False
        self.update_interval = 600  # 10 minutes
        self._lock = threading.Lock()

        # Expanded list of known free vision models (verified on OpenRouter)
        self.known_vision_models = [
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "meta-llama/llama-3.2-90b-vision-instruct:free",
            "qwen/qwen-2-vl-7b-instruct:free",
            "qwen/qwen-2-vl-72b-instruct:free",
            "google/gemini-flash-1.5:free",
            "google/gemini-pro-1.5:free",
            "openai/gpt-4o-mini:free",
            "mistral/pixtral-12b:free"
        ]

    def start(self):
        """Start the background model checker."""
        if self.is_running:
            return
        self.is_running = True
        self._fetch_and_update_models()
        self.thread = threading.Thread(target=self._background_loop, daemon=True)
        self.thread.start()
        logger.info("Vision model manager started (updates every 10 minutes)")

    def stop(self):
        """Stop the background model checker."""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        logger.info("Vision model manager stopped")

    def _background_loop(self):
        """Background loop for periodic updates."""
        while self.is_running:
            time.sleep(self.update_interval)
            if self.is_running:
                self._fetch_and_update_models()

    def _fetch_and_update_models(self):
        """Fetch available vision models from OpenRouter."""
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
                vision_models = []

                # Keywords that indicate vision capability (case-insensitive)
                vision_keywords = ['vision', 'llava', 'pixtral', 'gemini', 'gpt-4o', 'qwen-vl', 'qwen2-vl', 'multimodal']

                for model in all_models:
                    model_id = model.get("id", "")
                    model_id_lower = model_id.lower()
                    # Check if any keyword appears
                    if any(kw in model_id_lower for kw in vision_keywords):
                        vision_models.append(model_id)

                # Also add any known model that exists but wasn't caught
                for known in self.known_vision_models:
                    if known not in vision_models:
                        # Check if the base model exists (e.g., "meta-llama/llama-3.2-11b-vision-instruct:free" may appear without ":free")
                        base = known.split(':')[0]
                        if any(base in m.get("id", "") for m in all_models):
                            vision_models.append(known)

                with self._lock:
                    if vision_models:
                        self.available_models = vision_models
                        logger.info(f"Vision models updated: {len(vision_models)} models available")
                    else:
                        # Fallback to known models
                        self.available_models = self.known_vision_models
                        logger.warning("No vision models detected, using known fallback models")

        except Exception as e:
            logger.error(f"Failed to fetch vision models: {e}")
            with self._lock:
                if not self.available_models:
                    self.available_models = self.known_vision_models

    def get_available_models(self) -> List[str]:
        """Get list of available vision models."""
        with self._lock:
            return self.available_models.copy() if self.available_models else self.known_vision_models.copy()