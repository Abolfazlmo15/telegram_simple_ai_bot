import os
import json
import time
import logging
import threading
import httpx
from typing import List
from core.config import Config

logger = logging.getLogger(__name__)


class ModelAvailabilityManager:
    def __init__(self):
        self.cache_file = "available_models.json"
        self.available_fast_models = []
        self.available_smart_models = []
        self.is_running = False
        self.check_interval = 300  # Check every 5 minutes

        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "fast" in data and "smart" in data:
                        self.available_fast_models = data["fast"]
                        self.available_smart_models = data["smart"]
                        logger.info("Loaded available models from cache.")
                        return
            except Exception as e:
                logger.warning(f"Failed to load model cache, using defaults: {e}")

        # Fallback to config defaults
        self._set_defaults()

    def _set_defaults(self):
        self.available_fast_models = Config.FAST_MODELS.copy()
        self.available_smart_models = Config.SMART_MODELS.copy()
        self._save_cache()

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "fast": self.available_fast_models,
                    "smart": self.available_smart_models
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model cache: {e}")

    def get_available_models(self, complexity: str) -> List[str]:
        """Returns the prioritized list of available models for the given complexity."""
        if complexity == "smart":
            return self.available_smart_models if self.available_smart_models else Config.SMART_MODELS
        return self.available_fast_models if self.available_fast_models else Config.FAST_MODELS

    def report_model_failure(self, model: str, complexity: str):
        """Immediately mark a model as unavailable to prevent immediate retries."""
        if complexity == "smart" and model in self.available_smart_models:
            self.available_smart_models.remove(model)
            self._save_cache()
            logger.info(f"Temporarily removed {model} from available smart models due to failure.")
        elif complexity == "fast" and model in self.available_fast_models:
            self.available_fast_models.remove(model)
            self._save_cache()
            logger.info(f"Temporarily removed {model} from available fast models due to failure.")

    def start_background_checker(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._background_check_loop, daemon=True)
        self.thread.start()
        logger.info("Model availability background checker started.")

    def stop_background_checker(self):
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        logger.info("Model availability background checker stopped.")

    def _background_check_loop(self):
        while self.is_running:
            try:
                self._check_openrouter_models()
            except Exception as e:
                logger.error(f"Error in background model checker: {e}")

            # Sleep in small increments to allow clean shutdown
            for _ in range(self.check_interval):
                if not self.is_running:
                    break
                time.sleep(1)

    def _check_openrouter_models(self):
        logger.info("Checking OpenRouter model availability...")
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            with httpx.Client(timeout=15.0) as client:
                resp = client.get("https://openrouter.ai/api/v1/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()

                available_model_ids = {m["id"] for m in data.get("data", [])}

                # Filter and sort FAST models based on Config priority (list comprehension preserves order)
                new_fast = [model for model in Config.FAST_MODELS if model in available_model_ids]

                # Filter and sort SMART models based on Config priority
                new_smart = [model for model in Config.SMART_MODELS if model in available_model_ids]

                # Only update if we found at least some models, to prevent wiping cache on API blip
                if new_fast or new_smart:
                    self.available_fast_models = new_fast if new_fast else self.available_fast_models
                    self.available_smart_models = new_smart if new_smart else self.available_smart_models
                    self._save_cache()
                    logger.info(
                        f"Updated available models. Fast: {len(self.available_fast_models)}, Smart: {len(self.available_smart_models)}")
                else:
                    logger.warning("OpenRouter returned no matching models. Keeping previous cache.")

        except Exception as e:
            logger.error(f"Failed to check OpenRouter models: {e}")