"""Dynamic manager for OpenRouter image generation models."""
import logging
import threading
import time
import httpx
from typing import List, Optional
from core.config import Config

logger = logging.getLogger(__name__)


class ImageModelManager:
    """
    Manages available image generation models from OpenRouter.
    Auto-updates every 10 minutes with live model availability.
    Fetches from /api/v1/images/models (dedicated image models endpoint).
    """

    def __init__(self):
        self.available_models: List[str] = []
        self.is_running = False
        self.update_interval = 600  # 10 minutes
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Fallback models if API fetch fails (verified working models)
        self.fallback_models = [
            "black-forest-labs/flux.2-pro",
            "google/gemini-2.5-flash-image",
            "openai/gpt-5-image",
            "bytedance-seed/seedream-4.5",
        ]
        logger.info("🔷 ImageModelManager initialized")

    def start(self):
        """Start the background model checker."""
        if self.is_running:
            return
        self.is_running = True
        self._fetch_and_update_models()
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        logger.info("🔷 OpenRouter Image Model Manager started (updates every 10 minutes)")

    def stop(self):
        """Stop the background model checker."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("🔷 OpenRouter Image Model Manager stopped")

    def _background_loop(self):
        """Background loop for periodic updates."""
        while self.is_running:
            time.sleep(self.update_interval)
            if self.is_running:
                self._fetch_and_update_models()

    def _fetch_and_update_models(self):
        """Fetch available image generation models from OpenRouter."""
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            with httpx.Client(timeout=15.0) as client:
                # Use the dedicated image models endpoint
                resp = client.get("https://openrouter.ai/api/v1/images/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()

                image_models = []
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    # Only include models that output images
                    architecture = model.get("architecture", {})
                    output_modalities = architecture.get("output_modalities", [])
                    if "image" in output_modalities:
                        image_models.append(model_id)

                with self._lock:
                    if image_models:
                        self.available_models = image_models
                        logger.info(f"🖼️ OpenRouter image models updated: {len(image_models)} models available")
                    else:
                        self.available_models = self.fallback_models.copy()
                        logger.warning("⚠️ No OpenRouter image models detected, using fallback models")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                logger.warning("⚠️ OpenRouter image models require payment (402). Using fallback models.")
            else:
                logger.error(f"Failed to fetch OpenRouter image models: {e}")
            with self._lock:
                if not self.available_models:
                    self.available_models = self.fallback_models.copy()
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter image models: {e}")
            with self._lock:
                if not self.available_models:
                    self.available_models = self.fallback_models.copy()

    def get_available_models(self) -> List[str]:
        """Get list of available OpenRouter image generation models."""
        with self._lock:
            return self.available_models.copy() if self.available_models else self.fallback_models.copy()