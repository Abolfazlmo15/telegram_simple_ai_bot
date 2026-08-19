import os
import json
import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from core.config import Config

logger = logging.getLogger(__name__)


class UserDataManager:
    """
    Enhanced user data manager with conversation history, profiles, and smart caching.
    Stores data in JSON files within the users directory.
    """

    def __init__(self, base_dir: str = Config.USER_DATA_DIR, cache_file: str = Config.CACHE_FILE):
        self.base_dir = Path(base_dir)
        self.cache_file = Path(cache_file)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._lock = asyncio.Lock()
        self.cache = self._load_cache()

        self.recent_conversations: Dict[int, List[Dict]] = {}
        self.max_cached_conversations = Config.MAX_CACHED_CONVERSATIONS

        logger.info(f"User data manager initialized (dir: {self.base_dir})")

    def _get_default_user_data(self, user_id: int, username: Optional[str] = None) -> dict:
        """Helper to generate default user data structure."""
        return {
            "user_id": user_id,
            "username": username,
            "created_at": datetime.now().isoformat(),
            "history": [],
            "stats": {
                "total_requests": 0,
                "total_tokens": 0,
                "total_conversations": 1,
                "avg_response_time": 0.0,
                "avg_session_duration": 0.0,
                "quality_score": 0.0
            },
            "profile": {
                "preferred_category": None,
                "preferred_response_length": "medium",
                "common_topics": [],
                "activity_pattern": {}
            },
            "sessions": []
        }

    def _load_cache(self) -> dict:
        """Load response cache from disk with corruption protection"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)

                if not isinstance(cache, dict):
                    logger.warning("Cache file is corrupted (not a dict). Resetting cache.")
                    self.cache_file.unlink()
                    return {}

                current_time = datetime.now().timestamp()
                valid_cache = {}
                for key, entry in cache.items():
                    if isinstance(entry, dict) and current_time - entry.get('timestamp', 0) < Config.CACHE_TTL_SECONDS:
                        valid_cache[key] = entry

                if len(valid_cache) != len(cache):
                    with open(self.cache_file, 'w', encoding='utf-8') as f:
                        json.dump(valid_cache, f, indent=2)

                return valid_cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Resetting cache.")
                try:
                    self.cache_file.unlink()
                except Exception:
                    pass
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _get_user_file(self, user_id: int) -> Path:
        """Get path to user's data file"""
        user_dir = self.base_dir / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "user_data.json"

    async def load_user_data(self, user_id: int, username: Optional[str] = None) -> dict:
        """Load or create user data."""
        async with self._lock:
            user_file = self._get_user_file(user_id)
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if username and data.get('username') != username:
                        data['username'] = username
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load user {user_id} data: {e}")

            default_data = self._get_default_user_data(user_id, username)
            await self._save_user_data(user_id, default_data)
            return default_data

    async def _save_user_data(self, user_id: int, data: dict):
        """Save user data to disk (Assumes lock is already held by caller)"""
        user_file = self._get_user_file(user_id)
        try:
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save user {user_id} data: {e}")

    async def add_message_to_history(self, user_id: int, message: str,
                                     response: str, category: str = "unknown",
                                     response_time: float = 0.0, tokens_used: int = 0):
        """Add a message-response pair to user's history. (FIXED: No nested lock calls)"""
        async with self._lock:
            user_file = self._get_user_file(user_id)
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = self._get_default_user_data(user_id)
            else:
                data = self._get_default_user_data(user_id)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "response": response,
                "category": category,
                "response_time": response_time,
                "tokens_used": tokens_used
            }

            data['history'].append(entry)

            if len(data['history']) > Config.MAX_HISTORY_MESSAGES * 2:
                data['history'] = data['history'][-Config.MAX_HISTORY_MESSAGES * 2:]

            data['stats']['total_requests'] += 1
            data['stats']['total_tokens'] += tokens_used

            total = data['stats']['total_requests']
            current_avg = data['stats']['avg_response_time']
            data['stats']['avg_response_time'] = ((current_avg * (total - 1)) + response_time) / total

            if category != "unknown":
                profile = data['profile']
                if profile['preferred_category'] is None:
                    profile['preferred_category'] = category
                else:
                    category_count = sum(1 for h in data['history'] if h.get('category') == category)
                    if category_count > len(data['history']) * 0.4:
                        profile['preferred_category'] = category

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if user_id not in self.recent_conversations:
                self.recent_conversations[user_id] = []
            self.recent_conversations[user_id].append(entry)

            if len(self.recent_conversations[user_id]) > self.max_cached_conversations:
                self.recent_conversations[user_id] = self.recent_conversations[user_id][-self.max_cached_conversations:]

    def get_cached_response(self, text: str) -> Optional[Tuple[str, str, float]]:
        """Get cached response with similarity matching."""
        normalized = text.strip().lower()
        key = hashlib.md5(normalized.encode('utf-8')).hexdigest()

        if key in self.cache:
            entry = self.cache[key]
            if isinstance(entry, dict):
                logger.info(f"Cache hit (exact): {text[:50]}...")
                return entry['response'], entry.get('category', 'unknown'), entry['timestamp']

        for cached_key, cached_entry in self.cache.items():
            if not isinstance(cached_entry, dict):
                continue

            cached_text = cached_entry.get('text', '').lower()
            words1 = set(normalized.split())
            words2 = set(cached_text.split())

            if len(words1) < 3 or len(words2) < 3:
                continue

            intersection = words1 & words2
            union = words1 | words2
            similarity = len(intersection) / len(union)

            if similarity >= Config.CACHE_SIMILARITY_THRESHOLD:
                logger.info(f"Cache hit (similar {similarity:.2f}): {text[:50]}...")
                return cached_entry['response'], cached_entry.get('category', 'unknown'), cached_entry['timestamp']

        return None

    def save_to_cache(self, text: str, response: str, category: str = "unknown"):
        """Save response to cache."""
        normalized = text.strip().lower()
        key = hashlib.md5(normalized.encode('utf-8')).hexdigest()

        self.cache[key] = {
            "text": text,
            "response": response,
            "category": category,
            "timestamp": datetime.now().timestamp()
        }

        self._save_cache()

    def get_user_profile(self, user_id: int) -> Optional[dict]:
        """Get user's profile data"""
        user_file = self._get_user_file(user_id)
        if not user_file.exists():
            return None

        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('profile', {})
        except Exception as e:
            logger.error(f"Failed to load profile for user {user_id}: {e}")
            return None

    def get_recent_conversations(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get user's recent conversations from memory cache"""
        if user_id not in self.recent_conversations:
            return []
        return self.recent_conversations[user_id][-limit:]

    async def start_new_session(self, user_id: int):
        """Start a new session for user (FIXED: No nested lock calls)"""
        async with self._lock:
            user_file = self._get_user_file(user_id)
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = self._get_default_user_data(user_id)
            else:
                data = self._get_default_user_data(user_id)

            session = {
                "start_time": datetime.now().isoformat(),
                "messages": 0,
                "end_time": None
            }

            data['sessions'].append(session)
            data['stats']['total_conversations'] += 1

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    async def end_session(self, user_id: int):
        """End current session for user (FIXED: No nested lock calls)"""
        async with self._lock:
            user_file = self._get_user_file(user_id)
            if not user_file.exists():
                return

            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                return

            sessions = data.get('sessions', [])
            if sessions:
                for session in reversed(sessions):
                    if session.get('end_time') is None:
                        session['end_time'] = datetime.now().isoformat()
                        session['duration_minutes'] = (
                                                              datetime.fromisoformat(session['end_time']) -
                                                              datetime.fromisoformat(session['start_time'])
                                                      ).total_seconds() / 60
                        break

                with open(user_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)