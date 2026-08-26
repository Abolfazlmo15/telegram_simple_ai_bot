"""Manages primary and backup proxies with persistence, expiry, and fallback."""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from core.config import Config

logger = logging.getLogger(__name__)


class ProxyManager:
    """
    Manages a list of proxies with:
    - Primary proxy (always used first)
    - Backup proxies (used aggressively on any network error)
    - Automatic expiry (proxies older than 12 hours are removed)
    - Persistence across bot restarts (saved to JSON)
    - **Fallback to default Telegram API if all proxies fail**
    - **Aggressive failover**: any network error triggers proxy rotation immediately
    """

    def __init__(self, storage_file: str = Config.PROXY_STORAGE_FILE):
        self.storage_file = Path(storage_file)
        self.primary: str = Config.WORKER_URL
        self.backups: List[Dict[str, any]] = []
        self.primary_fail_start: Optional[float] = None
        self.current_proxy: str = self.primary
        self._default_fallback = "https://api.telegram.org"  # Always available as last resort

        # ============================================================
        # AGGRESSIVE FAILOVER: track consecutive failures per proxy
        # ============================================================
        self._proxy_failure_counts: Dict[str, int] = {}
        self._proxy_failure_threshold = 2  # After 2 failures, mark as bad
        self._proxy_cooldown_seconds = 60  # Wait 60s before retrying a bad proxy

        self._load()

        logger.info(f"ProxyManager initialized: primary={self.primary}, backups={len(self.backups)}")

    def _load(self):
        """Load proxies from storage file."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.primary = data.get('primary', self.primary)
                self.backups = data.get('backups', [])
                self._prune_expired()
                logger.info(f"Loaded {len(self.backups)} backups from storage")
            except Exception as e:
                logger.error(f"Failed to load proxies: {e}")
                self.backups = []

    def _save(self):
        """Save proxies to storage file."""
        try:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'primary': self.primary,
                    'backups': self.backups
                }, f, indent=2)
            logger.debug("Proxies saved")
        except Exception as e:
            logger.error(f"Failed to save proxies: {e}")

    def _prune_expired(self):
        """Remove backups older than 12 hours."""
        now = datetime.now()
        max_age = timedelta(hours=Config.BACKUP_PROXY_MAX_AGE_HOURS)
        original_count = len(self.backups)
        self.backups = [
            b for b in self.backups
            if b.get('last_used')
            and datetime.fromisoformat(b['last_used']) > now - max_age
        ]
        if len(self.backups) < original_count:
            logger.info(f"Pruned {original_count - len(self.backups)} expired backups")
            self._save()

    def mark_primary_failure(self):
        """Mark that the primary proxy just failed."""
        if self.primary_fail_start is None:
            self.primary_fail_start = time.time()
            logger.info(f"Primary proxy failure recorded at {datetime.now().isoformat()}")
        # Increment failure count for primary
        self._proxy_failure_counts[self.primary] = self._proxy_failure_counts.get(self.primary, 0) + 1
        logger.debug(f"Primary failure count: {self._proxy_failure_counts[self.primary]}")

    def reset_primary_failure(self):
        """Reset the primary failure timer (used when primary works)."""
        if self.primary_fail_start is not None:
            self.primary_fail_start = None
            logger.info("Primary proxy failure timer reset")
        # Reset failure count on success
        self._proxy_failure_counts[self.primary] = 0

    def mark_proxy_failure(self, proxy_url: str):
        """Mark a specific proxy as failed (aggressive failover)."""
        self._proxy_failure_counts[proxy_url] = self._proxy_failure_counts.get(proxy_url, 0) + 1
        if proxy_url == self.primary:
            self.mark_primary_failure()
        logger.debug(f"Proxy {proxy_url} failure count: {self._proxy_failure_counts[proxy_url]}")

    def is_proxy_bad(self, proxy_url: str) -> bool:
        """Check if a proxy has been marked as bad (too many failures)."""
        count = self._proxy_failure_counts.get(proxy_url, 0)
        if count >= self._proxy_failure_threshold:
            # Check if cooldown has passed
            last_fail_time = self._proxy_failure_counts.get(f"{proxy_url}_last_fail", 0)
            if time.time() - last_fail_time > self._proxy_cooldown_seconds:
                # Reset count after cooldown
                self._proxy_failure_counts[proxy_url] = 0
                return False
            return True
        return False

    def get_proxy(self) -> str:
        """
        Return the current proxy to use.
        Aggressively falls back to backups on ANY failure.
        Falls back to default Telegram API if no proxy works.
        """
        # Check if primary is marked as bad
        if self.is_proxy_bad(self.primary):
            logger.info("Primary proxy marked as bad (too many failures), switching...")
            # Force primary failure state
            if self.primary_fail_start is None:
                self.primary_fail_start = time.time()

        # If primary is working or hasn't failed enough, use primary
        if self.primary_fail_start is None:
            self.current_proxy = self.primary
            return self.primary

        # Check if primary has been failing for 5+ minutes
        elapsed = time.time() - self.primary_fail_start
        if elapsed < (Config.BACKUP_PROXY_TIMEOUT_MINUTES * 60):
            # Even if marked bad, we still try primary until timeout
            self.current_proxy = self.primary
            return self.primary

        # Primary has failed for 5+ minutes – try a backup
        for backup in self.backups:
            url = backup.get('url', '')
            if not url:
                continue
            # Skip if this backup is marked as bad
            if self.is_proxy_bad(url):
                logger.debug(f"Skipping bad backup proxy: {url}")
                continue
            # Check last_used age
            last_used = backup.get('last_used')
            if last_used:
                last_used_time = datetime.fromisoformat(last_used).timestamp()
                if time.time() - last_used_time < (Config.BACKUP_PROXY_TIMEOUT_MINUTES * 60):
                    continue
            backup['last_used'] = datetime.now().isoformat()
            self.current_proxy = url
            self._save()
            logger.info(f"Switched to backup proxy: {self.current_proxy}")
            return self.current_proxy

        # No available backup – fall back to default Telegram API
        logger.warning("No available backups, falling back to default Telegram API.")
        self.current_proxy = self._default_fallback
        return self._default_fallback

    def add_backup(self, url: str):
        """Add a new backup proxy."""
        for b in self.backups:
            if b['url'] == url:
                return
        self.backups.append({'url': url, 'last_used': None})
        self._save()
        logger.info(f"Added backup proxy: {url}")

    def mark_success(self, proxy_used: str):
        """Called when a proxy successfully completes a request."""
        if proxy_used == self.primary:
            self.reset_primary_failure()
        # Reset failure counts for this proxy on success
        self._proxy_failure_counts[proxy_used] = 0
        self._proxy_failure_counts[f"{proxy_used}_last_fail"] = 0
        logger.debug(f"Proxy {proxy_used} marked as successful")

    def mark_failure(self, proxy_used: str):
        """
        Called when a proxy fails a request.
        Aggressively increments failure count and may trigger failover.
        """
        self.mark_proxy_failure(proxy_used)
        # If this is the current proxy and it's now bad, force a switch next time
        if proxy_used == self.current_proxy and self.is_proxy_bad(proxy_used):
            logger.info(f"Current proxy {proxy_used} is now bad, will switch on next request")
            if proxy_used == self.primary:
                self.mark_primary_failure()

    def get_all_proxies(self) -> List[str]:
        """Get all known proxies (primary + backups + fallback)."""
        return [self.primary] + [b['url'] for b in self.backups] + [self._default_fallback]

    def clear_cache(self) -> None:
        """Clear in-memory state (keeps storage file)."""
        self.primary_fail_start = None
        self.current_proxy = self.primary
        self._proxy_failure_counts.clear()
        logger.info("ProxyManager cache cleared (failure timer and counts reset)")

    def get_info(self) -> Dict[str, Any]:
        """Return information about the manager."""
        return {
            "type": "ProxyManager",
            "primary": self.primary,
            "backups": len(self.backups),
            "current_proxy": self.current_proxy,
            "primary_failing": self.primary_fail_start is not None,
            "storage_file": str(self.storage_file),
            "default_fallback": self._default_fallback,
            "failure_counts": dict(self._proxy_failure_counts)
        }