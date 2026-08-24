import logging
import threading
import time
import httpx
from typing import List
from core.config import Config

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.fast_models: List[str] = []
        self.smart_models: List[str] = []
        self.is_running = False
        self.update_interval = 600  # 10 minutes in seconds
        self._lock = threading.Lock()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        # Fetch immediately on startup
        self._fetch_and_update_models()
        # Start background thread
        self.thread = threading.Thread(target=self._background_loop, daemon=True)
        self.thread.start()
        logger.info("Dynamic Model Manager started (updates every 10 minutes).")

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

                # 🚫 BLACKLIST: Models known to frequently return 422/400 on OpenRouter
                blacklist = ["nousresearch/hermes-4-70b", "nousresearch/hermes-4-405b"]

                all_model_ids = [m["id"] for m in data.get("data", []) if m["id"] not in blacklist]
                logger.info(f"Fetched {len(all_model_ids)} total models from OpenRouter (blacklist applied).")

                # ============================================================
                # NEW PRIORITY ORDER:
                # 1. DeepSeek (top priority)
                # 2. Qwen (second priority)
                # 3. Llama (third priority)
                # 4. Uncensored/Compliant (Dolphin, Hermes, OpenChat)
                # 5. Any other free models
                # ============================================================
                priority_deepseek = []
                priority_qwen = []
                priority_llama = []
                priority_uncensored = []
                priority_other_free = []

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

                # Fallback to config defaults if API returns nothing useful
                if not dynamic_list:
                    dynamic_list = Config.FALLBACK_MODELS
                    logger.warning("Dynamic fetch returned no prioritized models. Using config defaults.")
                else:
                    logger.info(f"Dynamic models updated: {len(dynamic_list)} prioritized models ready. "
                                f"(DeepSeek: {len(priority_deepseek)}, Qwen: {len(priority_qwen)})")

                with self._lock:
                    self.fast_models = dynamic_list
                    self.smart_models = dynamic_list

        except Exception as e:
            logger.error(f"Failed to fetch dynamic models: {e}")
            # On failure, ensure we at least have the config defaults
            with self._lock:
                if not self.fast_models:
                    self.fast_models = Config.FALLBACK_MODELS
                    self.smart_models = Config.FALLBACK_MODELS

    def get_fast_models(self) -> List[str]:
        with self._lock:
            return self.fast_models if self.fast_models else Config.FALLBACK_MODELS

    def get_smart_models(self) -> List[str]:
        with self._lock:
            return self.smart_models if self.smart_models else Config.FALLBACK_MODELS