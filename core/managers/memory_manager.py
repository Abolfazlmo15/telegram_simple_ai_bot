"""
Structured memory management for users.
Handles short-term, long-term memory, summarization, and semantic retrieval.
"""
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from core.config import Config

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages user memory with:
    - Short-term memory: Last N interactions (default 20)
    - Long-term memory: Summarized conversations with topics
    - Semantic retrieval: Search by meaning (keyword-based for now)
    - Memory expiration: Old memories are compressed or removed
    """

    def __init__(self, base_dir: str = Config.USER_DATA_DIR, max_short_term: int = 20):
        self.base_dir = Path(base_dir)
        self.max_short_term = max_short_term
        self._short_term: Dict[int, List[Dict]] = {}
        self._long_term: Dict[int, List[Dict]] = {}
        self._topics: Dict[int, List[str]] = {}
        self._lock = None  # asyncio.Lock would be used in production
        logger.info(f"🧠 MemoryManager initialized (max_short_term: {max_short_term})")

    async def add_interaction(self, user_id: int, message: str, response: str,
                              category: str = "unknown", metadata: Optional[Dict] = None) -> None:
        """
        Add a new interaction to the user's short-term memory.
        """
        if user_id not in self._short_term:
            self._short_term[user_id] = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response,
            "category": category,
            "metadata": metadata or {}
        }

        self._short_term[user_id].append(entry)

        # Trim to max_short_term
        if len(self._short_term[user_id]) > self.max_short_term:
            # Move older entries to long-term memory
            await self._compress_short_term(user_id)

    async def _compress_short_term(self, user_id: int) -> None:
        """
        Move older short-term entries to long-term memory (summarization).
        """
        if user_id not in self._short_term:
            return

        # Take the oldest half of short-term (or keep last 10)
        short = self._short_term[user_id]
        keep_count = min(10, self.max_short_term // 2)
        keep = short[-keep_count:]
        to_compress = short[:-keep_count]

        if not to_compress:
            return

        # Generate a summary for long-term memory
        summary = await self._generate_summary(to_compress)

        if user_id not in self._long_term:
            self._long_term[user_id] = []

        self._long_term[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "entry_count": len(to_compress),
            "compressed_from": to_compress[0].get("timestamp", ""),
            "compressed_to": to_compress[-1].get("timestamp", "")
        })

        # Keep only last 20 long-term entries
        if len(self._long_term[user_id]) > 20:
            self._long_term[user_id] = self._long_term[user_id][-20:]

        # Update short-term
        self._short_term[user_id] = keep

        logger.info(f"🧠 Compressed {len(to_compress)} entries for user {user_id}")

    async def _generate_summary(self, entries: List[Dict]) -> str:
        """
        Generate a summary of conversation entries.
        For now, this is a simple concatenation with truncation.
        Later, this can be enhanced with an LLM.
        """
        messages = [e.get("message", "") for e in entries if e.get("message")]
        if not messages:
            return "No significant conversation."

        # Simple summarization: take first and last few messages
        if len(messages) <= 3:
            text = " ".join(messages)
        else:
            text = f"{messages[0]}\n...\n{messages[-1]}"

        # Truncate to reasonable length
        if len(text) > 500:
            text = text[:500] + "..."

        return text

    async def get_context(self, user_id: int, limit: int = 5) -> List[Dict]:
        """
        Get the most recent interactions for context (short-term + long-term summary).
        """
        context = []

        # Short-term
        short = self._short_term.get(user_id, [])
        context.extend(short[-limit:])

        # Long-term summaries (if short-term is empty or small)
        if len(context) < limit:
            long = self._long_term.get(user_id, [])
            for entry in reversed(long):
                summary = entry.get("summary", "")
                if summary:
                    context.insert(0, {
                        "type": "summary",
                        "timestamp": entry.get("timestamp", ""),
                        "content": summary,
                        "entry_count": entry.get("entry_count", 0)
                    })
                if len(context) >= limit:
                    break

        return context

    async def get_short_term(self, user_id: int) -> List[Dict]:
        """Get short-term memory entries."""
        return self._short_term.get(user_id, [])

    async def get_long_term(self, user_id: int) -> List[Dict]:
        """Get long-term memory entries."""
        return self._long_term.get(user_id, [])

    async def search_memory(self, user_id: int, query: str) -> List[Dict]:
        """
        Search both short-term and long-term memory for the query.
        """
        results = []
        query_lower = query.lower()

        # Search short-term
        for entry in self._short_term.get(user_id, []):
            text = entry.get("message", "") + " " + entry.get("response", "")
            if query_lower in text.lower():
                results.append({**entry, "source": "short_term"})

        # Search long-term summaries
        for entry in self._long_term.get(user_id, []):
            summary = entry.get("summary", "")
            if query_lower in summary.lower():
                results.append({**entry, "source": "long_term", "content": summary})

        return results

    async def clear_memory(self, user_id: int) -> None:
        """Clear all memory for a user."""
        if user_id in self._short_term:
            del self._short_term[user_id]
        if user_id in self._long_term:
            del self._long_term[user_id]
        if user_id in self._topics:
            del self._topics[user_id]
        logger.info(f"🧹 Cleared memory for user {user_id}")

    async def is_memory_enabled(self, user_id: int) -> bool:
        """Check if memory is enabled (by default, always true)."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Return information about the manager."""
        return {
            "type": "MemoryManager",
            "max_short_term": self.max_short_term,
            "active_users": len(self._short_term),
            "long_term_users": len(self._long_term)
        }