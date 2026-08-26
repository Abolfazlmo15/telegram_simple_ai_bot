import os
import json
import asyncio
import hashlib
import logging
import struct
import zlib
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from PIL import Image
import io
from core.config import Config

logger = logging.getLogger(__name__)


class UserDataManager:
    """
    Enhanced user data manager with:
    - User directories named: {user_id}-{username} (auto‑renamed on username change)
    - info.json with user metadata and profile photo path
    - Incremental stats (total_requests, total_images, total_tokens, avg_response_time)
    - History with text, image, voice entries and generated content
    - Search history by keyword
    - Preference management via PreferenceManager
    - Unified priority system for all engines (text, vision, voice, voice_gen)
    - In-memory caching with TTL for performance
    """

    def __init__(self, base_dir: str = Config.USER_DATA_DIR, cache_file: str = Config.CACHE_FILE):
        self.base_dir = Path(base_dir)
        self.cache_file = Path(cache_file)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._lock = asyncio.Lock()
        self.cache = self._load_cache()
        self.recent_conversations: Dict[int, List[Dict]] = {}
        self.max_cached_conversations = Config.MAX_CACHED_CONVERSATIONS

        # ========== Preference Manager ==========
        from core.managers.preference_manager import PreferenceManager
        self.preference_manager = PreferenceManager(self)

        # ============================================================
        # PERFORMANCE: In-memory TTL Caches
        # ============================================================
        self._priority_cache: Dict[str, Tuple[Optional[List[str]], float]] = {}  # key: f"{user_id}_{engine}" -> (list, timestamp)
        self._pref_cache: Dict[int, Tuple[Dict, float]] = {}  # user_id -> (prefs, timestamp)
        self._tiers_cache: Optional[Tuple[Dict[str, bool], float]] = None  # cache for get_available_tiers
        self._cache_ttl = 60  # 60 seconds TTL to balance freshness and performance

        logger.info(f"User data manager initialized (dir: {self.base_dir}) with TTL caching (TTL: {self._cache_ttl}s)")

    # ---------- USER DIRECTORY & INFO ----------
    def _get_user_dir(self, user_id: int, username: Optional[str]) -> Path:
        """Return the user's directory path, creating it if needed."""
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)

        existing = None
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"{user_id}-"):
                existing = d
                break

        if username:
            new_name = f"{user_id}-{username}"
            if existing and existing.name != new_name:
                new_path = self.base_dir / new_name
                if not new_path.exists():
                    existing.rename(new_path)
                    logger.info(f"Renamed user dir: {existing.name} -> {new_name}")
                    existing = new_path
                else:
                    import shutil
                    shutil.rmtree(existing)
                    existing = new_path
            elif not existing:
                existing = self.base_dir / new_name
                existing.mkdir(parents=True, exist_ok=True)
        else:
            if not existing:
                existing = self.base_dir / str(user_id)
                existing.mkdir(parents=True, exist_ok=True)

        return existing

    def _get_user_info_file(self, user_id: int, username: Optional[str]) -> Path:
        return self._get_user_dir(user_id, username) / "info.json"

    def _get_user_data_file(self, user_id: int, username: Optional[str]) -> Path:
        return self._get_user_dir(user_id, username) / "user_data.json"

    # ========== PREFERENCE MANAGER METHODS ==========
    async def get_preferences(self, user_id: int, username: Optional[str] = None) -> Dict[str, Any]:
        """Get user preferences via PreferenceManager."""
        # Check in-memory cache first
        if user_id in self._pref_cache:
            prefs, timestamp = self._pref_cache[user_id]
            if time.time() - timestamp < self._cache_ttl:
                return prefs
        prefs = await self.preference_manager.get_preferences(user_id, username)
        self._pref_cache[user_id] = (prefs, time.time())
        return prefs

    async def save_preferences(self, user_id: int, username: Optional[str],
                               preferences: Dict[str, Any]) -> bool:
        """Save user preferences via PreferenceManager."""
        result = await self.preference_manager.save_preferences(user_id, username, preferences)
        if result:
            self._pref_cache[user_id] = (preferences, time.time())
        return result

    async def get_preference(self, user_id: int, key: str,
                             username: Optional[str] = None) -> Any:
        """Get a single preference value."""
        return await self.preference_manager.get_preference(user_id, key, username)

    async def set_preference(self, user_id: int, key: str, value: Any,
                             username: Optional[str] = None) -> bool:
        """Set a single preference value."""
        return await self.preference_manager.set_preference(user_id, key, value, username)

    async def get_response_mode(self, user_id: int,
                                username: Optional[str] = None) -> str:
        """Get user's preferred response mode (text/voice/auto)."""
        return await self.preference_manager.get_response_mode(user_id, username)

    async def set_response_mode(self, user_id: int, mode: str,
                                username: Optional[str] = None) -> bool:
        """Set user's response mode."""
        return await self.preference_manager.set_response_mode(user_id, mode, username)

    async def get_response_style(self, user_id: int,
                                 username: Optional[str] = None) -> str:
        """Get user's preferred response style (concise/detailed/balanced)."""
        return await self.preference_manager.get_response_style(user_id, username)

    async def set_response_style(self, user_id: int, style: str,
                                 username: Optional[str] = None) -> bool:
        """Set user's response style."""
        return await self.preference_manager.set_response_style(user_id, style, username)

    async def get_preferred_models(self, user_id: int,
                                   username: Optional[str] = None) -> List[str]:
        """Get user's preferred models in priority order."""
        return await self.preference_manager.get_preferred_models(user_id, username)

    async def set_preferred_models(self, user_id: int, models: List[str],
                                   username: Optional[str] = None) -> bool:
        """Set user's preferred models."""
        return await self.preference_manager.set_preferred_models(user_id, models, username)

    async def get_preferred_style(self, user_id: int,
                                  username: Optional[str] = None) -> str:
        """Get user's preferred artistic style."""
        return await self.preference_manager.get_preferred_style(user_id, username)

    async def set_preferred_style(self, user_id: int, style: str,
                                  username: Optional[str] = None) -> bool:
        """Set user's preferred artistic style."""
        return await self.preference_manager.set_preference(user_id, "preferred_style", style, username)

    async def get_custom_instructions(self, user_id: int,
                                      username: Optional[str] = None) -> str:
        """Get user's custom instructions."""
        return await self.preference_manager.get_custom_instructions(user_id, username)

    async def set_custom_instructions(self, user_id: int, instructions: str,
                                      username: Optional[str] = None) -> bool:
        """Set user's custom instructions."""
        return await self.preference_manager.set_preference(user_id, "custom_instructions", instructions, username)

    async def get_voice_speed(self, user_id: int,
                              username: Optional[str] = None) -> float:
        """Get user's preferred voice speed."""
        return await self.preference_manager.get_voice_speed(user_id, username)

    async def set_voice_speed(self, user_id: int, speed: float,
                              username: Optional[str] = None) -> bool:
        """Set user's preferred voice speed."""
        return await self.preference_manager.set_preference(user_id, "voice_speed", speed, username)

    async def get_voice_style(self, user_id: int,
                              username: Optional[str] = None) -> str:
        """Get user's preferred voice style."""
        return await self.preference_manager.get_voice_style(user_id, username)

    async def set_voice_style(self, user_id: int, style: str,
                              username: Optional[str] = None) -> bool:
        """Set user's preferred voice style."""
        return await self.preference_manager.set_preference(user_id, "voice_style", style, username)

    async def is_memory_enabled(self, user_id: int,
                                username: Optional[str] = None) -> bool:
        """Check if memory is enabled for the user."""
        return await self.preference_manager.get_preference(user_id, "memory_enabled", username) or True

    # ---------- USER DATA & HISTORY ----------
    def _get_default_user_data(self, user_id: int, username: Optional[str] = None) -> dict:
        return {
            "user_id": user_id,
            "username": username,
            "created_at": datetime.now().isoformat(),
            "history": [],
            "stats": {
                "total_requests": 0,
                "total_tokens": 0,
                "total_conversations": 1,
                "total_images": 0,
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

    async def load_user_info(self, user_id: int, username: Optional[str],
                             first_name: str = "", last_name: str = "",
                             bio: str = "", phone_number: str = "") -> dict:
        """Load or create info.json with user metadata."""
        async with self._lock:
            info_file = self._get_user_info_file(user_id, username)
            if info_file.exists():
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            else:
                data = {}

            data.update({
                "user_id": user_id,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
                "bio": bio,
                "phone_number": phone_number,
                "last_updated": datetime.now().isoformat()
            })
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return data

    async def save_profile_photo(self, user_id: int, username: Optional[str], photo_bytes: bytes) -> Optional[str]:
        """Save the user's profile photo to their directory."""
        try:
            user_dir = self._get_user_dir(user_id, username)
            photo_path = user_dir / "profile_photo.jpg"
            with open(photo_path, 'wb') as f:
                f.write(photo_bytes)
            info_file = user_dir / "info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['profile_photo_path'] = str(photo_path)
                with open(info_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return str(photo_path)
        except Exception as e:
            logger.error(f"Failed to save profile photo for {user_id}: {e}")
            return None

    async def load_user_data(self, user_id: int, username: Optional[str] = None) -> dict:
        async with self._lock:
            user_file = self._get_user_data_file(user_id, username)
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
            await self._save_user_data(user_id, username, default_data)
            return default_data

    async def _save_user_data(self, user_id: int, username: Optional[str], data: dict):
        user_file = self._get_user_data_file(user_id, username)
        try:
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save user {user_id} data: {e}")

    async def add_message_to_history(self, user_id: int, username: Optional[str], message: str,
                                     response: str, category: str = "unknown",
                                     response_time: float = 0.0, tokens_used: int = 0):
        async with self._lock:
            user_file = self._get_user_data_file(user_id, username)
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = self._get_default_user_data(user_id, username)
            else:
                data = self._get_default_user_data(user_id, username)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "text",
                "message": message,
                "response": response,
                "category": category,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "username": username
            }
            data['history'].append(entry)
            if len(data['history']) > Config.MAX_HISTORY_MESSAGES * 2:
                data['history'] = data['history'][-Config.MAX_HISTORY_MESSAGES * 2:]

            stats = data['stats']
            stats['total_requests'] = stats.get('total_requests', 0) + 1
            stats['total_tokens'] = stats.get('total_tokens', 0) + tokens_used
            total = stats['total_requests']
            current_avg = stats['avg_response_time']
            stats['avg_response_time'] = ((current_avg * (total - 1)) + response_time) / total

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if user_id not in self.recent_conversations:
                self.recent_conversations[user_id] = []
            self.recent_conversations[user_id].append(entry)
            if len(self.recent_conversations[user_id]) > self.max_cached_conversations:
                self.recent_conversations[user_id] = self.recent_conversations[user_id][-self.max_cached_conversations:]

    async def add_image_to_history(self, user_id: int, username: Optional[str],
                                   query_text: str, response: str,
                                   matrix_file: str, width: int, height: int,
                                   response_time: float = 0.0, tokens_used: int = 0):
        async with self._lock:
            user_file = self._get_user_data_file(user_id, username)
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = self._get_default_user_data(user_id, username)
            else:
                data = self._get_default_user_data(user_id, username)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "image",
                "query": query_text,
                "response": response,
                "matrix_file": matrix_file,
                "width": width,
                "height": height,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "username": username
            }
            data['history'].append(entry)
            if len(data['history']) > Config.MAX_HISTORY_MESSAGES * 2:
                data['history'] = data['history'][-Config.MAX_HISTORY_MESSAGES * 2:]

            stats = data['stats']
            stats['total_requests'] = stats.get('total_requests', 0) + 1
            stats['total_images'] = stats.get('total_images', 0) + 1
            stats['total_tokens'] = stats.get('total_tokens', 0) + tokens_used
            total = stats['total_requests']
            current_avg = stats['avg_response_time']
            stats['avg_response_time'] = ((current_avg * (total - 1)) + response_time) / total

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if user_id not in self.recent_conversations:
                self.recent_conversations[user_id] = []
            self.recent_conversations[user_id].append(entry)
            if len(self.recent_conversations[user_id]) > self.max_cached_conversations:
                self.recent_conversations[user_id] = self.recent_conversations[user_id][-self.max_cached_conversations:]

    # ---------- GENERATION HISTORY ----------
    async def add_generated_image_to_history(self, user_id: int, username: Optional[str],
                                             prompt: str, response: str,
                                             matrix_file: str, width: int, height: int,
                                             model_used: str, response_time: float, tokens_used: int):
        """Add a generated image entry to user history."""
        async with self._lock:
            user_file = self._get_user_data_file(user_id, username)
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = self._get_default_user_data(user_id, username)
            else:
                data = self._get_default_user_data(user_id, username)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "generated_image",
                "prompt": prompt,
                "response": response,
                "matrix_file": matrix_file,
                "width": width,
                "height": height,
                "model_used": model_used,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "username": username
            }
            data['history'].append(entry)
            if len(data['history']) > Config.MAX_HISTORY_MESSAGES * 2:
                data['history'] = data['history'][-Config.MAX_HISTORY_MESSAGES * 2:]

            stats = data['stats']
            stats['total_requests'] = stats.get('total_requests', 0) + 1
            stats['total_images'] = stats.get('total_images', 0) + 1
            stats['total_tokens'] = stats.get('total_tokens', 0) + tokens_used
            total = stats['total_requests']
            current_avg = stats['avg_response_time']
            stats['avg_response_time'] = ((current_avg * (total - 1)) + response_time) / total

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if user_id not in self.recent_conversations:
                self.recent_conversations[user_id] = []
            self.recent_conversations[user_id].append(entry)
            if len(self.recent_conversations[user_id]) > self.max_cached_conversations:
                self.recent_conversations[user_id] = self.recent_conversations[user_id][-self.max_cached_conversations:]

    async def add_generated_voice_to_history(self, user_id: int, username: Optional[str],
                                             prompt: str, response: str,
                                             audio_file: str, model_used: str,
                                             response_time: float, tokens_used: int):
        """Add a generated voice entry to user history."""
        async with self._lock:
            user_file = self._get_user_data_file(user_id, username)
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = self._get_default_user_data(user_id, username)
            else:
                data = self._get_default_user_data(user_id, username)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "generated_voice",
                "prompt": prompt,
                "response": response,
                "audio_file": audio_file,
                "model_used": model_used,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "username": username
            }
            data['history'].append(entry)
            if len(data['history']) > Config.MAX_HISTORY_MESSAGES * 2:
                data['history'] = data['history'][-Config.MAX_HISTORY_MESSAGES * 2:]

            stats = data['stats']
            stats['total_requests'] = stats.get('total_requests', 0) + 1
            stats['total_tokens'] = stats.get('total_tokens', 0) + tokens_used
            total = stats['total_requests']
            current_avg = stats['avg_response_time']
            stats['avg_response_time'] = ((current_avg * (total - 1)) + response_time) / total

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if user_id not in self.recent_conversations:
                self.recent_conversations[user_id] = []
            self.recent_conversations[user_id].append(entry)
            if len(self.recent_conversations[user_id]) > self.max_cached_conversations:
                self.recent_conversations[user_id] = self.recent_conversations[user_id][-self.max_cached_conversations:]

    # ---------- SEARCH HISTORY ----------
    async def search_history(self, user_id: int, query: str, limit: int = 5) -> List[Dict]:
        """Search user's history for entries containing the query (case‑insensitive)."""
        user_data = await self.load_user_data(user_id)
        history = user_data.get('history', [])
        results = []
        query_lower = query.lower()
        for entry in reversed(history):
            text = entry.get('message', '') + ' ' + entry.get('query', '') + ' ' + entry.get('prompt', '')
            if query_lower in text.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    # ---------- CACHE ----------
    def _load_cache(self) -> dict:
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
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_cached_response(self, text: str) -> Optional[Tuple[str, str, float]]:
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
        error_indicators = ["🌐", "❌", "⚠️", "Connection Issue", "Rate Limit", "Error", "Timed Out"]
        if any(indicator in response for indicator in error_indicators):
            logger.info(f"Not caching error response: {response[:50]}...")
            return

        # Also reject empty or too-short responses
        if not response or len(response.strip()) < 5:
            logger.info("Not caching empty or too-short response")
            return

        normalized = text.strip().lower()
        key = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        self.cache[key] = {
            "text": text,
            "response": response,
            "category": category,
            "timestamp": datetime.now().timestamp()
        }
        self._save_cache()

    # ---------- USER STATS ----------
    async def get_user_stats(self, user_id: int, username: Optional[str] = None) -> Optional[Dict]:
        """Return pre‑computed stats from user_data.json."""
        try:
            user_file = self._get_user_data_file(user_id, username)
            if not user_file.exists():
                return None
            with open(user_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            stats = data.get('stats', {})
            sessions = data.get('sessions', [])
            session_duration = 0
            for session in sessions:
                if session.get('end_time'):
                    start = datetime.fromisoformat(session['start_time'])
                    end = datetime.fromisoformat(session['end_time'])
                    session_duration += (end - start).total_seconds() / 60
            return {
                "total_messages": stats.get('total_requests', 0),
                "total_images": stats.get('total_images', 0),
                "total_tokens": stats.get('total_tokens', 0),
                "avg_response_time": stats.get('avg_response_time', 0.0),
                "session_duration": f"{session_duration:.1f} minutes" if session_duration > 0 else "N/A"
            }
        except Exception as e:
            logger.error(f"Failed to get stats for user {user_id}: {e}")
            return None

    # ---------- IMAGE MATRIX STORAGE ----------
    async def save_image_matrix(self, user_id: int, username: Optional[str], image_bytes: bytes) -> Dict:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            pixels = list(img.getdata())

            # RLE encode
            rle_data = bytearray()
            if pixels:
                current_pixel = pixels[0]
                run_length = 1
                for p in pixels[1:]:
                    if p == current_pixel and run_length < 255:
                        run_length += 1
                    else:
                        rle_data.append(run_length)
                        rle_data.extend(current_pixel)
                        current_pixel = p
                        run_length = 1
                rle_data.append(run_length)
                rle_data.extend(current_pixel)

            compressed = zlib.compress(rle_data, level=9)

            user_dir = self._get_user_dir(user_id, username)
            matrix_dir = user_dir / Config.IMAGE_MATRIX_DIR
            matrix_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = matrix_dir / f"{timestamp}.rgbz"

            with open(filename, 'wb') as f:
                f.write(b'RGBZ')
                f.write(struct.pack('<HH', width, height))
                f.write(struct.pack('<I', len(compressed)))
                f.write(compressed)

            original_size = len(image_bytes)
            compressed_size = filename.stat().st_size

            return {
                "file": str(filename),
                "width": width,
                "height": height,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to save image matrix for user {user_id}: {e}")
            raise

    def prune_pictures(self, user_id: int, username: Optional[str], max_images: int = 5):
        user_dir = self._get_user_dir(user_id, username)
        matrix_dir = user_dir / Config.IMAGE_MATRIX_DIR
        if matrix_dir.exists():
            files = sorted(matrix_dir.glob("*.rgbz"), key=lambda p: p.stat().st_mtime, reverse=True)
            if len(files) > max_images:
                for f in files[max_images:]:
                    f.unlink()
                    logger.info(f"Deleted old matrix file: {f}")

    # ---------- PRIORITY ----------
    async def get_user_model_priority(self, user_id: int, username: Optional[str], engine: str = "text") -> Optional[List[str]]:
        """
        Get user's model priority list for a specific engine.
        engine: "text", "vision", "voice", "voice_gen"
        """
        cache_key = f"{user_id}_{engine}"
        if cache_key in self._priority_cache:
            priority_list, timestamp = self._priority_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return priority_list

        user_dir = self._get_user_dir(user_id, username)
        priority_file = user_dir / "model_priority.json"
        result = None
        if priority_file.exists():
            try:
                with open(priority_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    result = data.get(engine, [])
                elif isinstance(data, list) and engine == "text":
                    # Backward compatibility: old format
                    result = data
            except Exception as e:
                logger.error(f"Failed to load priority for {user_id} ({engine}): {e}")

        self._priority_cache[cache_key] = (result, time.time())
        return result

    async def save_model_priority(self, user_id: int, username: Optional[str],
                                  priority_list: List[str], engine: str = "text"):
        """
        Save user's model priority list for a specific engine.
        engine: "text", "vision", "voice", "voice_gen"
        """
        user_dir = self._get_user_dir(user_id, username)
        priority_file = user_dir / "model_priority.json"
        try:
            existing = {}
            if priority_file.exists():
                with open(priority_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
            existing[engine] = priority_list
            with open(priority_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2)
            # Update cache
            self._priority_cache[f"{user_id}_{engine}"] = (priority_list, time.time())
            logger.info(f"Saved {engine} priority for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to save {engine} priority for {user_id}: {e}")

    async def clear_user_data(self, user_id: int, username: Optional[str]) -> bool:
        user_dir = self._get_user_dir(user_id, username)
        try:
            if user_dir.exists():
                import shutil
                shutil.rmtree(user_dir)
                logger.info(f"Cleared data for user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to clear data for user {user_id}: {e}")
        return False

    # ---------- SESSIONS ----------
    async def start_new_session(self, user_id: int, username: Optional[str]):
        async with self._lock:
            user_file = self._get_user_data_file(user_id, username)
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = self._get_default_user_data(user_id, username)
            else:
                data = self._get_default_user_data(user_id, username)

            session = {
                "start_time": datetime.now().isoformat(),
                "messages": 0,
                "end_time": None
            }
            data['sessions'].append(session)
            data['stats']['total_conversations'] = data['stats'].get('total_conversations', 0) + 1

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    async def end_session(self, user_id: int, username: Optional[str]):
        async with self._lock:
            user_file = self._get_user_data_file(user_id, username)
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

    # ---------- VOICE/AUDIO METHODS ----------
    async def save_audio_file(self, user_id: int, username: Optional[str], audio_bytes: bytes) -> str:
        """Save the audio file to user's voices directory and return the file path."""
        user_dir = self._get_user_dir(user_id, username)
        audio_dir = user_dir / "voices"
        audio_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = audio_dir / f"{timestamp}.ogg"
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        return str(filename)

    def prune_voices(self, user_id: int, username: Optional[str], max_files: int = 5):
        """Delete oldest voice files, keep only the last `max_files`."""
        user_dir = self._get_user_dir(user_id, username)
        audio_dir = user_dir / "voices"
        if audio_dir.exists():
            files = sorted(audio_dir.glob("*.ogg"), key=lambda p: p.stat().st_mtime, reverse=True)
            if len(files) > max_files:
                for f in files[max_files:]:
                    f.unlink()
                    logger.info(f"Deleted old voice file: {f}")

    async def add_voice_to_history(self, user_id: int, username: Optional[str],
                                   transcription: str, response: str,
                                   audio_file: str,
                                   response_time: float = 0.0, tokens_used: int = 0):
        """Add a voice transcription entry to user history."""
        async with self._lock:
            user_file = self._get_user_data_file(user_id, username)
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = self._get_default_user_data(user_id, username)
            else:
                data = self._get_default_user_data(user_id, username)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "voice",
                "transcription": transcription,
                "response": response,
                "audio_file": audio_file,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "username": username
            }
            data['history'].append(entry)
            if len(data['history']) > Config.MAX_HISTORY_MESSAGES * 2:
                data['history'] = data['history'][-Config.MAX_HISTORY_MESSAGES * 2:]

            stats = data['stats']
            stats['total_requests'] = stats.get('total_requests', 0) + 1
            stats['total_tokens'] = stats.get('total_tokens', 0) + tokens_used
            total = stats['total_requests']
            current_avg = stats['avg_response_time']
            stats['avg_response_time'] = ((current_avg * (total - 1)) + response_time) / total

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if user_id not in self.recent_conversations:
                self.recent_conversations[user_id] = []
            self.recent_conversations[user_id].append(entry)
            if len(self.recent_conversations[user_id]) > self.max_cached_conversations:
                self.recent_conversations[user_id] = self.recent_conversations[user_id][-self.max_cached_conversations:]

    # ---------- IMAGE GENERATION PRIORITY ----------
    async def get_image_generation_priority(self, user_id: int, username: Optional[str] = None) -> List[str]:
        """
        Get the user's preferred image generation tier priority order.
        Returns a list of tier names in order of preference.
        """
        user_dir = self._get_user_dir(user_id, username)
        priority_file = user_dir / "image_priority.json"

        default_priority = Config.IMAGE_GENERATION_PRIORITY

        if priority_file.exists():
            try:
                with open(priority_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    return data
                elif isinstance(data, dict) and 'priority' in data:
                    return data['priority']
            except Exception as e:
                logger.error(f"Failed to load image priority for user {user_id}: {e}")

        return default_priority

    async def save_image_generation_priority(self, user_id: int, username: Optional[str],
                                             priority_list: List[str]) -> bool:
        """
        Save the user's preferred image generation tier priority order.
        """
        user_dir = self._get_user_dir(user_id, username)
        priority_file = user_dir / "image_priority.json"

        try:
            data = {
                "priority": priority_list,
                "last_updated": datetime.now().isoformat(),
                "user_id": user_id
            }
            with open(priority_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved image priority for user {user_id}: {priority_list}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image priority for user {user_id}: {e}")
            return False

    async def get_available_tiers(self) -> Dict[str, bool]:
        """Get all available image generation tiers and their status."""
        # Use cached value if fresh
        if self._tiers_cache:
            tiers, timestamp = self._tiers_cache
            if time.time() - timestamp < self._cache_ttl:
                return tiers

        tiers = {
            "pollinations": True,  # Always available
            "huggingface": bool(Config.HUGGINGFACE_TOKEN),  # Fixed: use HUGGINGFACE_TOKEN
            "openrouter": bool(Config.OPENROUTER_API_KEY),
        }
        self._tiers_cache = (tiers, time.time())
        return tiers

    # ---------- RESPONSE MODE (Legacy wrapper) ----------
    async def get_response_mode_legacy(self, user_id: int, username: Optional[str] = None) -> str:
        """
        Legacy method – use get_response_mode() instead.
        Kept for backward compatibility.
        """
        return await self.get_response_mode(user_id, username)

    async def set_response_mode_legacy(self, user_id: int, username: Optional[str], mode: str) -> bool:
        """
        Legacy method – use set_response_mode() instead.
        Kept for backward compatibility.
        """
        return await self.set_response_mode(user_id, mode, username)

    def get_info(self) -> Dict[str, Any]:
        """Return information about the manager."""
        return {
            "type": "UserDataManager",
            "base_dir": str(self.base_dir),
            "cache_size": len(self.cache),
            "recent_conversations": len(self.recent_conversations),
            "preference_manager": self.preference_manager.get_info() if hasattr(self.preference_manager, 'get_info') else {}
        }