import logging
import threading
import time
import httpx
from typing import List, Dict
from core.config import Config

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.fast_models: List[str] = []
        self.smart_models: List[str] = []
        self.is_running = False
        self.update_interval = 600  # 10 minutes in seconds
        self._lock = threading.Lock()

        # ============================================================
        # PERFORMANCE: Centralized blacklisting for text models
        # ============================================================
        self._model_failures: Dict[str, float] = {}
        self._blacklist_ttl = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS
        self._fallback_models = Config.FALLBACK_MODELS
        # Models that are known to be invalid (hard-coded blacklist)
        self._static_blacklist = [
            "nousresearch/hermes-4-70b",
            "nousresearch/hermes-4-405b",
            "deepseek/deepseek-v4-flash-vision-exp",  # known 404
        ]

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._fetch_and_update_models()
        self.thread = threading.Thread(target=self._background_loop, daemon=True)
        self.thread.start()
        logger.info("Dynamic Model Manager started (updates every 10 minutes, with blacklisting).")

    def stop(self):
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        logger.info("Dynamic Model Manager stopped.")

    def _background_loop(self):
        while self.is_running:
            time.sleep(self.update_interval)
            if self.is_running:
                self._fetch_and_update_models()

    def _is_blacklisted(self, model: str) -> bool:
        # Check static blacklist
        if model in self._static_blacklist:
            return True
        # Check dynamic failure blacklist
        if model in self._model_failures:
            if time.time() - self._model_failures[model] < self._blacklist_ttl:
                return True
            else:
                # TTL expired, remove from blacklist
                del self._model_failures[model]
        return False

    def mark_failure(self, model: str):
        """Public method for engines to mark a model as failed."""
        self._model_failures[model] = time.time()
        logger.info(f"🚫 Manager blacklisted {model} for {self._blacklist_ttl}s")

    def _filter_blacklisted(self, models: List[str]) -> List[str]:
        """Remove blacklisted models from a list, keeping order."""
        filtered = [m for m in models if not self._is_blacklisted(m)]
        if len(filtered) < len(models):
            logger.info(f"Filtered out {len(models) - len(filtered)} blacklisted models.")
        return filtered

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

                all_model_ids = [m["id"] for m in data.get("data", [])]
                logger.info(f"Fetched {len(all_model_ids)} total models from OpenRouter.")

                # ============================================================
                # PRIORITY ORDER: DeepSeek > Qwen > Llama > Uncensored > Other Free
                # ============================================================
                priority_deepseek, priority_qwen, priority_llama, priority_uncensored, priority_other_free = [], [], [], [], []

                for model_id in all_model_ids:
                    mid = model_id.lower()
                    if "deepseek" in mid:
                        priority_deepseek.append(model_id)
                    elif "qwen" in mid:
                        priority_qwen.append(model_id)
                    elif "llama-3" in mid or "llama-4" in mid:
                        priority_llama.append(model_id)
                    elif "dolphin" in mid or "hermes" in mid or "openchat" in mid:
                        priority_uncensored.append(model_id)
                    elif ":free" in mid:
                        priority_other_free.append(model_id)

                # Combine in priority order, remove duplicates
                seen = set()
                dynamic_list = []
                for model_list in [priority_deepseek, priority_qwen, priority_llama,
                                   priority_uncensored, priority_other_free]:
                    for model_id in model_list:
                        if model_id not in seen:
                            seen.add(model_id)
                            dynamic_list.append(model_id)

                if not dynamic_list:
                    dynamic_list = self._fallback_models.copy()
                    logger.warning("Dynamic fetch returned no prioritized models. Using config defaults.")
                else:
                    logger.info(f"Dynamic models updated: {len(dynamic_list)} prioritized. (DeepSeek: {len(priority_deepseek)}, Qwen: {len(priority_qwen)})")

                # ============================================================
                # Apply blacklist filtering (static + dynamic failures)
                # ============================================================
                filtered_list = self._filter_blacklisted(dynamic_list)
                if not filtered_list:
                    logger.warning("All dynamic models blacklisted! Falling back to config defaults.")
                    filtered_list = self._fallback_models.copy()

                with self._lock:
                    self.fast_models = filtered_list
                    self.smart_models = filtered_list

        except Exception as e:
            logger.error(f"Failed to fetch dynamic models: {e}")
            with self._lock:
                if not self.fast_models:
                    self.fast_models = self._fallback_models.copy()
                    self.smart_models = self._fallback_models.copy()

    def get_fast_models(self) -> List[str]:
        with self._lock:
            return self.fast_models if self.fast_models else self._fallback_models.copy()

    def get_smart_models(self) -> List[str]:
        with self._lock:
            return self.smart_models if self.smart_models else self._fallback_models.copy()