"""
Topic detection and tracking for conversations.
"""
import logging
import re
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from core.config import Config

logger = logging.getLogger(__name__)


class TopicManager:
    """
    Detects and tracks conversation topics per user.
    Topics are identified using keyword matching and simple heuristics.
    """

    # Topic keyword mappings
    TOPIC_KEYWORDS = {
        "coding": ["code", "python", "javascript", "programming", "function", "class", "debug", "error", "fix", "script"],
        "business": ["business", "strategy", "market", "profit", "revenue", "investment", "startup", "company"],
        "education": ["learn", "teach", "explain", "tutorial", "study", "course", "lesson", "educate", "understand"],
        "creative": ["write", "story", "poem", "creative", "fiction", "narrative", "character", "plot"],
        "science": ["science", "research", "experiment", "data", "analysis", "statistics", "physics", "chemistry", "biology"],
        "health": ["health", "fitness", "exercise", "diet", "nutrition", "medical", "disease", "symptom"],
        "technology": ["tech", "software", "hardware", "cloud", "ai", "machine learning", "data science", "cyber"],
        "travel": ["travel", "trip", "vacation", "journey", "tourist", "destination", "flight", "hotel"],
        "food": ["food", "cook", "recipe", "meal", "restaurant", "cuisine", "dinner", "lunch"],
        "personal": ["family", "friend", "life", "relationship", "love", "happy", "sad", "feeling"],
        "image_generation": ["image", "picture", "photo", "generate", "create", "draw", "art", "style", "anime", "realistic"],
        "voice": ["say", "speak", "voice", "audio", "talk", "tell", "read", "narrate", "tts"],
    }

    def __init__(self):
        self._user_topics: Dict[int, List[Dict]] = {}
        self._topic_counts: Dict[int, Dict[str, int]] = {}
        self._current_topic: Dict[int, str] = {}
        logger.info("📊 TopicManager initialized")

    async def add_message(self, user_id: int, text: str) -> Optional[str]:
        """
        Detect topic from a message and update user's topic history.
        Returns the detected topic, if any.
        """
        detected = self._detect_topic(text)
        if detected:
            # Update topic history
            if user_id not in self._user_topics:
                self._user_topics[user_id] = []
                self._topic_counts[user_id] = defaultdict(int)

            self._user_topics[user_id].append({
                "timestamp": datetime.now().isoformat(),
                "topic": detected,
                "text": text[:100]  # Truncate
            })

            self._topic_counts[user_id][detected] += 1
            self._current_topic[user_id] = detected

            # Trim history to last 50 topics
            if len(self._user_topics[user_id]) > 50:
                self._user_topics[user_id] = self._user_topics[user_id][-50:]

            logger.info(f"📊 Detected topic for user {user_id}: {detected}")
            return detected

        return None

    def _detect_topic(self, text: str) -> Optional[str]:
        """
        Detect the topic from text using keyword matching.
        """
        text_lower = text.lower()
        scores = defaultdict(int)

        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[topic] += 1

        if not scores:
            return None

        # Return the topic with the highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    async def get_current_topic(self, user_id: int) -> Optional[str]:
        """Get the current topic for a user."""
        return self._current_topic.get(user_id)

    async def get_topic_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get recent topics for a user."""
        return self._user_topics.get(user_id, [])[-limit:]

    async def get_topic_counts(self, user_id: int) -> Dict[str, int]:
        """Get topic frequency counts for a user."""
        return dict(self._topic_counts.get(user_id, {}))

    async def get_most_common_topics(self, user_id: int, n: int = 3) -> List[Tuple[str, int]]:
        """Get the most common topics for a user."""
        counts = self._topic_counts.get(user_id, {})
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_counts[:n]

    async def clear_topics(self, user_id: int) -> None:
        """Clear all topic data for a user."""
        if user_id in self._user_topics:
            del self._user_topics[user_id]
        if user_id in self._topic_counts:
            del self._topic_counts[user_id]
        if user_id in self._current_topic:
            del self._current_topic[user_id]
        logger.info(f"📊 Cleared topics for user {user_id}")

    def clear_cache(self) -> None:
        """Clear all in-memory topic data."""
        self._user_topics.clear()
        self._topic_counts.clear()
        self._current_topic.clear()
        logger.info("📊 TopicManager cache cleared")

    def get_info(self) -> Dict[str, Any]:
        """Return information about the manager."""
        return {
            "type": "TopicManager",
            "active_users": len(self._user_topics),
            "topics": list(self.TOPIC_KEYWORDS.keys())
        }