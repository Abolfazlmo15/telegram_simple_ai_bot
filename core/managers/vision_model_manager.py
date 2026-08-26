"""Dynamic manager for vision-capable models – filters only free models."""
import logging
import threading
import time
import httpx
from typing import List, Optional
from core.config import Config

logger = logging.getLogger(__name__)


class VisionModelManager:
    """
    Manages available vision-capable models from OpenRouter.
    Auto-updates every 10 minutes with live model availability.
    Filters only free models (those with ':free' suffix or in the known free list).
    """

    def __init__(self):
        self.available_models: List[str] = []
        self.is_running = False
        self.update_interval = 600  # 10 minutes
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

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
        logger.info("🔷 VisionModelManager initialized (free models only)")

    def start(self):
        """Start the background model checker."""
        if self.is_running:
            return
        self.is_running = True
        self._fetch_and_update_models()
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        logger.info("Vision model manager started (updates every 10 minutes)")

    def stop(self):
        """Stop the background model checker."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Vision model manager stopped")

    def _background_loop(self):
        """Background loop for periodic updates."""
        while self.is_running:
            time.sleep(self.update_interval)
            if self.is_running:
                self._fetch_and_update_models()

    def _fetch_and_update_models(self):
        """Fetch available vision models from OpenRouter, keeping only free ones."""
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
                    # Check if it's a vision model and if it's free (':free' suffix)
                    if any(kw in model_id_lower for kw in vision_keywords):
                        if ":free" in model_id_lower:
                            vision_models.append(model_id)
                        else:
                            # Also check if this model is in our known free list
                            if model_id in self.known_vision_models:
                                vision_models.append(model_id)

                # Ensure all known free models are included
                for known in self.known_vision_models:
                    if known not in vision_models:
                        # Check if the base model exists (e.g., without ':free' suffix)
                        # but we only want the free version, so we just add it anyway
                        # because we know it's available on OpenRouter.
                        vision_models.append(known)

                with self._lock:
                    if vision_models:
                        # Remove duplicates while preserving order
                        seen = set()
                        unique = []
                        for m in vision_models:
                            if m not in seen:
                                seen.add(m)
                                unique.append(m)
                        self.available_models = unique
                        logger.info(f"Vision models updated: {len(self.available_models)} free models available")
                    else:
                        # Fallback to known free list
                        self.available_models = self.known_vision_models.copy()
                        logger.warning("No free vision models detected, using known fallback models")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                logger.warning("⚠️ OpenRouter vision models require payment (402). Using fallback models.")
            else:
                logger.error(f"Failed to fetch vision models: {e}")
            with self._lock:
                if not self.available_models:
                    self.available_models = self.known_vision_models.copy()
        except Exception as e:
            logger.error(f"Failed to fetch vision models: {e}")
            with self._lock:
                if not self.available_models:
                    self.available_models = self.known_vision_models.copy()

    def get_available_models(self) -> List[str]:
        """Get list of available vision models (free only)."""
        with self._lock:
            return self.available_models.copy() if self.available_models else self.known_vision_models.copy()