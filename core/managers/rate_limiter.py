"""
Simple in-memory rate limiter with sliding window per user.
"""
import time
import asyncio
import logging
from collections import defaultdict
from typing import Tuple

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter with sliding window per user."""

    def __init__(self, max_requests: int = 12, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check(self, user_id: int) -> Tuple[bool, int]:
        """
        Check if user can make a request.

        Args:
            user_id: Telegram user ID

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        async with self._lock:
            now = time.time()
            timestamps = self._history[user_id]
            cutoff = now - self.window_seconds

            # Remove old timestamps
            self._history[user_id] = [t for t in timestamps if t > cutoff]

            if len(self._history[user_id]) >= self.max_requests:
                logger.debug(f"Rate limit hit for user {user_id}")
                return False, 0

            # Add current request
            self._history[user_id].append(now)
            remaining = self.max_requests - len(self._history[user_id])
            return True, remaining

    def get_remaining(self, user_id: int) -> int:
        """Get remaining requests for a user."""
        now = time.time()
        cutoff = now - self.window_seconds
        timestamps = self._history.get(user_id, [])
        valid = [t for t in timestamps if t > cutoff]
        return max(0, self.max_requests - len(valid))

    def clear_cache(self) -> None:
        """Clear all rate limiting history (resets all users)."""
        self._history.clear()
        logger.info("🔄 RateLimiter cache cleared (all user histories reset)")

    def get_info(self) -> dict:
        """Return information about the rate limiter."""
        return {
            "type": "RateLimiter",
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "active_users": len(self._history)
        }