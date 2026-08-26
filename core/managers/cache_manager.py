"""
Unified cache manager with TTL support for various data types.
Provides in-memory caching with expiration, LRU eviction, and persistence support.
"""
import logging
import time
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock

from core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds


class CacheManager:
    """
    Unified in-memory cache manager with TTL support.

    Features:
    - TTL-based expiration per entry
    - Size limits with LRU eviction
    - Namespace support for different cache contexts
    - Optional persistence to disk
    - Thread-safe (using RLock)
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300, persistence_dir: Optional[str] = None):
        """
        Initialize the cache manager.

        Args:
            max_size: Maximum number of entries across all namespaces
            default_ttl: Default TTL in seconds (300 = 5 minutes)
            persistence_dir: Optional directory for persistence
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = RLock()
        self._persistence_dir: Optional[Path] = None

        if persistence_dir:
            self._persistence_dir = Path(persistence_dir)
            self._persistence_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistence()

        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        logger.info(f"📦 CacheManager initialized (max_size: {max_size}, default_ttl: {default_ttl}s)")

    def _get_key(self, namespace: str, key: str) -> str:
        """Generate a composite key with namespace."""
        return f"{namespace}:{key}"

    def _evict_if_needed(self):
        """Evict oldest entries if cache is at capacity."""
        if len(self._cache) < self.max_size:
            return

        # Sort by last_accessed (LRU)
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed
        )

        # Evict oldest 20% to avoid frequent evictions
        to_evict = max(1, len(sorted_items) // 5)
        for key, _ in sorted_items[:to_evict]:
            del self._cache[key]
            self._eviction_count += 1

        logger.debug(f"📦 Evicted {to_evict} entries (total evictions: {self._eviction_count})")

    def set(self, namespace: str, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """
        Set a value in the cache.

        Args:
            namespace: Cache namespace (e.g., "responses", "preferences")
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds (uses default if None)
        """
        with self._lock:
            composite_key = self._get_key(namespace, key)

            # Check if we need to evict
            if composite_key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_if_needed()

            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

            self._cache[composite_key] = CacheEntry(
                value=value,
                ttl_seconds=ttl,
                timestamp=time.time(),
                last_accessed=time.time()
            )

    def get(self, namespace: str, key: str, default: Any = None) -> Optional[Any]:
        """
        Get a value from the cache.

        Returns the value if found and not expired, otherwise returns default.
        """
        with self._lock:
            composite_key = self._get_key(namespace, key)
            entry = self._cache.get(composite_key)

            if entry is None:
                self._miss_count += 1
                return default

            if entry.is_expired():
                del self._cache[composite_key]
                self._miss_count += 1
                return default

            # Update access metadata
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._hit_count += 1

            return entry.value

    def get_or_set(self, namespace: str, key: str, value_factory: Callable[[], Any],
                   ttl_seconds: Optional[float] = None) -> Any:
        """
        Get a value, or compute and set it if not present.

        Args:
            namespace: Cache namespace
            key: Cache key
            value_factory: Function that returns the value to cache
            ttl_seconds: TTL in seconds (uses default if None)

        Returns:
            The cached or computed value
        """
        value = self.get(namespace, key)
        if value is not None:
            return value

        value = value_factory()
        self.set(namespace, key, value, ttl_seconds)
        return value

    def delete(self, namespace: str, key: str) -> bool:
        """Delete a value from the cache."""
        with self._lock:
            composite_key = self._get_key(namespace, key)
            if composite_key in self._cache:
                del self._cache[composite_key]
                return True
            return False

    def clear_namespace(self, namespace: str) -> int:
        """
        Clear all entries in a namespace.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            prefix = f"{namespace}:"
            to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for k in to_delete:
                del self._cache[k]
            return len(to_delete)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hit_count = 0
            self._miss_count = 0
            self._eviction_count = 0
            logger.info("📦 Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = len(self._cache)
            total_requests = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total_requests * 100) if total_requests > 0 else 0

            return {
                "total_entries": total,
                "max_size": self.max_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": f"{hit_rate:.1f}%",
                "eviction_count": self._eviction_count
            }

    # ---------- PERSISTENCE ----------
    def _load_persistence(self):
        """Load cache from persistence file."""
        if not self._persistence_dir:
            return

        persist_file = self._persistence_dir / "cache.json"
        if not persist_file.exists():
            return

        try:
            with open(persist_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for key, entry_data in data.items():
                # Only load if not expired
                ttl = entry_data.get('ttl_seconds', self.default_ttl)
                timestamp = entry_data.get('timestamp', time.time())
                if ttl is not None and time.time() - timestamp > ttl:
                    continue

                self._cache[key] = CacheEntry(
                    value=entry_data['value'],
                    timestamp=timestamp,
                    ttl_seconds=ttl,
                    access_count=0,
                    last_accessed=time.time()
                )

            logger.info(f"📦 Loaded {len(self._cache)} entries from persistence")
        except Exception as e:
            logger.error(f"Failed to load persistence: {e}")

    def save_persistence(self) -> bool:
        """Save cache to persistence file."""
        if not self._persistence_dir:
            return False

        persist_file = self._persistence_dir / "cache.json"
        try:
            data = {}
            with self._lock:
                for key, entry in self._cache.items():
                    # Only save if not expired
                    if entry.is_expired():
                        continue
                    data[key] = {
                        'value': entry.value,
                        'timestamp': entry.timestamp,
                        'ttl_seconds': entry.ttl_seconds
                    }

            with open(persist_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"📦 Saved {len(data)} entries to persistence")
            return True
        except Exception as e:
            logger.error(f"Failed to save persistence: {e}")
            return False

    def clear_cache(self) -> None:
        """
        Alias for clear() for API consistency with other managers.
        """
        self.clear()

    def get_info(self) -> Dict[str, Any]:
        """Return information about the cache manager."""
        with self._lock:
            return {
                "type": "CacheManager",
                "total_entries": len(self._cache),
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
                "persistence_enabled": self._persistence_dir is not None,
                "hit_rate": self.get_stats()["hit_rate"]
            }