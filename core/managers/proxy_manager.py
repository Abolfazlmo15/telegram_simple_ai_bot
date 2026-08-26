"""Manages primary and backup proxies with persistence, expiry, and fallback."""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from core.config import Config

logger = logging.getLogger(__name__)


class ProxyManager:
    """
    Manages a list of proxies with:
    - Primary proxy (always used first)
    - Backup proxies (only used if primary fails for 5+ minutes)
    - Automatic expiry (proxies older than 12 hours are removed)
    - Persistence across bot restarts (saved to JSON)
    - **Fallback to default Telegram API if all proxies fail**
    """

    def __init__(self, storage_file: str = Config.PROXY_STORAGE_FILE):
        self.storage_file = Path(storage_file)
        self.primary: str = Config.WORKER_URL
        self.backups: List[Dict[str, any]] = []
        self.primary_fail_start: Optional[float] = None
        self.current_proxy: str = self.primary
        self._default_fallback = "https://api.telegram.org"  # Always available as last resort
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

    def reset_primary_failure(self):
        """Reset the primary failure timer (used when primary works)."""
        if self.primary_fail_start is not None:
            self.primary_fail_start = None
            logger.info("Primary proxy failure timer reset")

    def get_proxy(self) -> str:
        """
        Return the current proxy to use.
        Falls back to default Telegram API if no proxy works.
        """
        # If primary is working or hasn't failed long enough, use primary
        if self.primary_fail_start is None:
            self.current_proxy = self.primary
            return self.primary

        # Check if primary has been failing for 5+ minutes
        elapsed = time.time() - self.primary_fail_start
        if elapsed < (Config.BACKUP_PROXY_TIMEOUT_MINUTES * 60):
            self.current_proxy = self.primary
            return self.primary

        # Primary has failed for 5+ minutes – try a backup
        for backup in self.backups:
            last_used = backup.get('last_used')
            if last_used:
                last_used_time = datetime.fromisoformat(last_used).timestamp()
                if time.time() - last_used_time < (Config.BACKUP_PROXY_TIMEOUT_MINUTES * 60):
                    continue
            backup['last_used'] = datetime.now().isoformat()
            self.current_proxy = backup['url']
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
        # For backups, update their last_used (already done when selected)

    def get_all_proxies(self) -> List[str]:
        """Get all known proxies (primary + backups + fallback)."""
        return [self.primary] + [b['url'] for b in self.backups] + [self._default_fallback]

    def clear_cache(self) -> None:
        """Clear in-memory state (keeps storage file)."""
        self.primary_fail_start = None
        self.current_proxy = self.primary
        logger.info("ProxyManager cache cleared (failure timer reset)")

    def get_info(self) -> Dict[str, Any]:
        """Return information about the manager."""
        return {
            "type": "ProxyManager",
            "primary": self.primary,
            "backups": len(self.backups),
            "current_proxy": self.current_proxy,
            "primary_failing": self.primary_fail_start is not None,
            "storage_file": str(self.storage_file),
            "default_fallback": self._default_fallback
        }