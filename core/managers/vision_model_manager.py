"""Dynamic manager for vision-capable models – filters only free models, uses proxy."""
import logging
import threading
import time
import httpx
from typing import List, Optional
from core.config import Config
from core.managers.proxy_manager import ProxyManager

logger = logging.getLogger(__name__)


class VisionModelManager:
    """
    Manages available vision-capable models from OpenRouter.
    Auto-updates every 10 minutes with live model availability.
    Filters only free models (those with ':free' suffix or in the known free list).
    Uses the proxy from ProxyManager if available.
    """

    def __init__(self, proxy_manager: Optional[ProxyManager] = None):
        self.available_models: List[str] = []
        self.is_running = False
        self.update_interval = 600  # 10 minutes
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.proxy_manager = proxy_manager

        # Expanded list of known free vision models (verified manually)
        self.known_vision_models = [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "qwen/qwen-2-vl-7b-instruct:free",
            "qwen/qwen-2-vl-72b-instruct:free",
            "google/gemini-flash-1.5:free",
            "google/gemini-pro-1.5:free",
            "openai/gpt-4o-mini:free",
            "mistral/pixtral-12b:free",
            "llava-hf/llava-1.5-7b-hf:free",
            "llava-hf/llava-1.5-13b-hf:free",
            "HuggingFaceM4/idefics2-8b:free",
        ]
        self._last_fetch_time = 0
        self._fetch_lock = threading.Lock()
        logger.info("🔷 VisionModelManager initialized (free models only, with proxy)")

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

    def _get_client(self) -> httpx.Client:
        """Create an HTTP client with proxy if available."""
        client_kwargs = {"timeout": 15.0}
        if self.proxy_manager:
            proxy_url = self.proxy_manager.get_proxy()
            if proxy_url:
                client_kwargs["proxy"] = proxy_url
        return httpx.Client(**client_kwargs)

    def _fetch_and_update_models(self):
        """Fetch available vision models from OpenRouter, keeping only free ones."""
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            with self._get_client() as client:
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
                    # Check if it is vision-capable and free
                    if any(kw in model_id_lower for kw in vision_keywords):
                        if ":free" in model_id_lower:
                            vision_models.append(model_id)
                        else:
                            # If it's in our known list, include even without :free
                            if model_id in self.known_vision_models:
                                vision_models.append(model_id)

                # Ensure known models are present (some may not be detected)
                for known in self.known_vision_models:
                    if known not in vision_models:
                        vision_models.append(known)

                with self._lock:
                    if vision_models:
                        seen = set()
                        unique = []
                        for m in vision_models:
                            if m not in seen:
                                seen.add(m)
                                unique.append(m)
                        self.available_models = unique
                        logger.info(f"Vision models updated: {len(self.available_models)} free models available")
                    else:
                        self.available_models = self.known_vision_models.copy()
                        logger.warning("No free vision models detected, using known fallback models")
                self._last_fetch_time = time.time()

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

    def get_available_models(self, force_refresh: bool = False) -> List[str]:
        """
        Get list of available vision models (free only).
        If force_refresh is True, fetch fresh models from OpenRouter.
        """
        if force_refresh:
            logger.info("🔄 Force refreshing vision models from OpenRouter")
            self._fetch_and_update_models()
        with self._lock:
            return self.available_models.copy() if self.available_models else self.known_vision_models.copy()

    def get_last_fetch_time(self) -> float:
        """Get timestamp of last successful model fetch."""
        return self._last_fetch_time