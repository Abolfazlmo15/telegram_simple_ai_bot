"""
Health checker for monitoring model availability and API endpoints.
Runs background checks on configured models and providers to proactively
skip unhealthy endpoints before they cause request failures.
"""
import logging
import threading
import time
import random
import httpx
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from core.config import Config
from core.utils.network import retry_sync, is_retryable_exception

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of a model or endpoint."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ModelHealth:
    """Health record for a single model."""
    model_id: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_checked: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    failure_reason: str = ""
    response_time_ms: float = 0.0
    checked_count: int = 0


class HealthChecker:
    """
    Background health checker for models and API endpoints.
    Provides:
    - Periodic health checks on configured models
    - Automatic marking of unhealthy models
    - Health status queries for engines
    - Integration with model managers for blacklisting
    """

    def __init__(self):
        self._models: Dict[str, ModelHealth] = {}
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._client: Optional[httpx.Client] = None

        # Configuration
        self.check_interval_seconds = Config.HEALTH_CHECK_INTERVAL_SECONDS
        self.failure_threshold = 3
        self.success_threshold = 2
        self.cooldown_seconds = Config.MODEL_FAILURE_BLACKLIST_TTL_SECONDS

        # Retry settings for individual health checks
        self.health_check_retries = 2
        self.health_check_timeout = 10.0

        logger.info(f"🏥 HealthChecker initialized (interval: {self.check_interval_seconds}s)")

    def start(self):
        """Start the background health check thread."""
        if self._is_running:
            return

        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0),
            limits=httpx.Limits(max_connections=10)
        )

        self._seed_models()

        self._is_running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="HealthChecker")
        self._thread.start()
        logger.info("🏥 HealthChecker started")

    def stop(self):
        """Stop the background health check thread."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._client:
            self._client.close()
            self._client = None
        logger.info("🏥 HealthChecker stopped")

    def _seed_models(self):
        """Seed initial models from configuration."""
        model_ids = set()

        # Text models
        model_ids.update(Config.FALLBACK_MODELS)

        # Vision models (known free)
        model_ids.update([
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "google/gemini-flash-1.5:free",
            "qwen/qwen-2-vl-7b-instruct:free",
        ])

        # TTS models
        model_ids.update(Config.DEFAULT_VOICE_GEN_PRIORITY)

        # STT models
        model_ids.update(Config.DEFAULT_VOICE_ENGINE_PRIORITY)

        # Image generation models (OpenRouter)
        model_ids.update([
            "black-forest-labs/flux.2-pro",
            "google/gemini-2.5-flash-image",
            "openai/gpt-5-image",
        ])

        # Track provider endpoints
        self._track_provider("openrouter", "https://openrouter.ai/api/v1/models")
        self._track_provider("pollinations", "https://image.pollinations.ai/prompt/test")
        self._track_provider("huggingface", "https://api-inference.huggingface.co/models")

        for model_id in model_ids:
            if model_id and model_id not in self._models:
                self._models[model_id] = ModelHealth(model_id=model_id)

        logger.info(f"🏥 Seeded {len(self._models)} models for health monitoring")

    def _track_provider(self, name: str, endpoint: str):
        if name not in self._models:
            self._models[name] = ModelHealth(model_id=name)

    def _worker_loop(self):
        logger.info("🏥 HealthChecker worker loop started")
        while self._is_running:
            try:
                self._run_checks()
            except Exception as e:
                logger.error(f"🏥 Health check error: {e}", exc_info=True)

            # Sleep with small random jitter to avoid thundering herd
            sleep_time = self.check_interval_seconds + random.uniform(-2, 2)
            sleep_time = max(10, sleep_time)
            for _ in range(int(sleep_time)):
                if not self._is_running:
                    break
                time.sleep(1)
        logger.info("🏥 HealthChecker worker loop ended")

    def _run_checks(self):
        with self._lock:
            models_to_check = list(self._models.items())

        for model_id, health in models_to_check:
            try:
                status, latency, reason = self._check_model_with_retry(model_id)
                health.checked_count += 1
                health.last_checked = datetime.now()
                health.response_time_ms = latency

                if status == HealthStatus.HEALTHY:
                    health.consecutive_successes += 1
                    health.consecutive_failures = 0
                    health.last_success = datetime.now()
                    health.failure_reason = ""
                    if health.consecutive_successes >= self.success_threshold:
                        health.status = HealthStatus.HEALTHY
                else:
                    health.consecutive_failures += 1
                    health.consecutive_successes = 0
                    health.last_failure = datetime.now()
                    health.failure_reason = reason or "Unknown failure"
                    if health.consecutive_failures >= self.failure_threshold:
                        health.status = HealthStatus.UNHEALTHY
                        # If a model becomes unhealthy, we can log and optionally extend blacklist
                        logger.warning(f"🏥 Model {model_id} marked UNHEALTHY after {health.consecutive_failures} failures")

            except Exception as e:
                logger.warning(f"🏥 Health check failed for {model_id}: {e}")
                with self._lock:
                    health = self._models.get(model_id)
                    if health:
                        health.consecutive_failures += 1
                        health.consecutive_successes = 0
                        health.last_failure = datetime.now()
                        health.failure_reason = str(e)
                        if health.consecutive_failures >= self.failure_threshold:
                            health.status = HealthStatus.UNHEALTHY

    def _check_model_with_retry(self, model_id: str) -> Tuple[HealthStatus, float, str]:
        """Check a model with built‑in retry for transient errors."""
        last_error = None
        for attempt in range(self.health_check_retries):
            try:
                return self._check_model(model_id)
            except Exception as e:
                last_error = e
                if attempt < self.health_check_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
        return HealthStatus.UNHEALTHY, 0.0, str(last_error)

    def _check_model(self, model_id: str) -> Tuple[HealthStatus, float, str]:
        # Provider endpoints
        if model_id == "openrouter":
            return self._check_openrouter()
        if model_id == "pollinations":
            return self._check_pollinations()
        if model_id == "huggingface":
            return self._check_huggingface()

        # OpenRouter model check
        if any(x in model_id.lower() for x in ["deepseek", "qwen", "llama", "gemini", "gpt"]):
            return self._check_openrouter_model(model_id)

        return HealthStatus.UNKNOWN, 0.0, "Not checked"

    def _check_openrouter(self) -> Tuple[HealthStatus, float, str]:
        try:
            start = time.time()
            resp = self._client.get(
                "https://openrouter.ai/api/v1/models",
                timeout=self.health_check_timeout,
                headers={"Authorization": f"Bearer {Config.OPENROUTER_API_KEY}"}
            )
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                return HealthStatus.HEALTHY, latency, ""
            elif resp.status_code == 402:
                return HealthStatus.DEGRADED, latency, "Payment required"
            else:
                return HealthStatus.UNHEALTHY, latency, f"HTTP {resp.status_code}"
        except Exception as e:
            return HealthStatus.UNHEALTHY, 0.0, str(e)

    def _check_pollinations(self) -> Tuple[HealthStatus, float, str]:
        try:
            start = time.time()
            resp = self._client.get(
                "https://image.pollinations.ai/prompt/health_test",
                timeout=self.health_check_timeout,
                params={"width": 64, "height": 64}
            )
            latency = (time.time() - start) * 1000
            if resp.status_code == 200 and "image" in resp.headers.get("content-type", ""):
                return HealthStatus.HEALTHY, latency, ""
            return HealthStatus.UNHEALTHY, latency, f"HTTP {resp.status_code}"
        except Exception as e:
            return HealthStatus.UNHEALTHY, 0.0, str(e)

    def _check_huggingface(self) -> Tuple[HealthStatus, float, str]:
        if not Config.HUGGINGFACE_TOKEN:
            return HealthStatus.UNHEALTHY, 0.0, "No token configured"
        try:
            start = time.time()
            resp = self._client.get(
                "https://api-inference.huggingface.co/models",
                timeout=self.health_check_timeout,
                headers={"Authorization": f"Bearer {Config.HUGGINGFACE_TOKEN}"}
            )
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                return HealthStatus.HEALTHY, latency, ""
            return HealthStatus.UNHEALTHY, latency, f"HTTP {resp.status_code}"
        except Exception as e:
            return HealthStatus.UNHEALTHY, 0.0, str(e)

    def _check_openrouter_model(self, model_id: str) -> Tuple[HealthStatus, float, str]:
        try:
            start = time.time()
            resp = self._client.get(
                "https://openrouter.ai/api/v1/models",
                timeout=self.health_check_timeout,
                headers={"Authorization": f"Bearer {Config.OPENROUTER_API_KEY}"}
            )
            latency = (time.time() - start) * 1000

            if resp.status_code != 200:
                return HealthStatus.UNHEALTHY, latency, f"API HTTP {resp.status_code}"

            data = resp.json()
            model_ids = [m.get("id") for m in data.get("data", [])]
            exact_match = any(m == model_id for m in model_ids)
            is_free = ":free" in model_id

            if exact_match:
                return HealthStatus.HEALTHY, latency, ""
            elif is_free:
                return HealthStatus.DEGRADED, latency, "Model not in list but may work"
            else:
                return HealthStatus.UNHEALTHY, latency, "Model not found"
        except Exception as e:
            return HealthStatus.UNHEALTHY, 0.0, str(e)

    # ---------- PUBLIC API ----------
    def is_healthy(self, model_id: str) -> bool:
        with self._lock:
            health = self._models.get(model_id)
            if not health:
                return True
            return health.status != HealthStatus.UNHEALTHY

    def get_health(self, model_id: str) -> Optional[ModelHealth]:
        with self._lock:
            return self._models.get(model_id)

    def get_unhealthy_models(self) -> List[str]:
        with self._lock:
            return [m for m, h in self._models.items() if h.status == HealthStatus.UNHEALTHY]

    def get_health_status(self) -> Dict[str, str]:
        with self._lock:
            return {m: h.status.value for m, h in self._models.items()}

    def register_model(self, model_id: str):
        with self._lock:
            if model_id not in self._models:
                self._models[model_id] = ModelHealth(model_id=model_id)
                logger.info(f"🏥 Registered model: {model_id}")

    def clear_cache(self) -> None:
        with self._lock:
            for health in self._models.values():
                health.status = HealthStatus.UNKNOWN
                health.consecutive_failures = 0
                health.consecutive_successes = 0
                health.failure_reason = ""
                health.last_checked = None
                health.last_success = None
                health.last_failure = None
            logger.info("🏥 HealthChecker cache cleared (all health records reset)")

    def get_info(self) -> Dict:
        with self._lock:
            total = len(self._models)
            healthy = sum(1 for h in self._models.values() if h.status == HealthStatus.HEALTHY)
            unhealthy = sum(1 for h in self._models.values() if h.status == HealthStatus.UNHEALTHY)
            unknown = sum(1 for h in self._models.values() if h.status == HealthStatus.UNKNOWN)

            return {
                "type": "HealthChecker",
                "running": self._is_running,
                "total_models": total,
                "healthy": healthy,
                "unhealthy": unhealthy,
                "unknown": unknown,
                "check_interval": self.check_interval_seconds
            }