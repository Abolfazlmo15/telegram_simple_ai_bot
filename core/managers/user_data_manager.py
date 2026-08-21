import os
import json
import asyncio
import hashlib
import logging
import struct
import zlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List
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
    - History with text and image entries (matrix file references)
    - Search history by keyword
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

    # ---------- USER DIRECTORY & INFO ----------
    def _get_user_dir(self, user_id: int, username: Optional[str]) -> Path:
        """Return the user's directory path, creating it if needed."""
        # Ensure base directory exists before iterating
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)

        # Find existing directory with this user_id
        existing = None
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"{user_id}-"):
                existing = d
                break

        # If username provided and existing dir doesn't match, rename it
        if username:
            new_name = f"{user_id}-{username}"
            if existing and existing.name != new_name:
                new_path = self.base_dir / new_name
                if not new_path.exists():
                    existing.rename(new_path)
                    logger.info(f"Renamed user dir: {existing.name} -> {new_name}")
                    existing = new_path
                else:
                    # New path exists – remove old (shouldn't happen)
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

            # Update with new info
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
            # Update info.json with photo path
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

            # Update stats incrementally
            stats = data['stats']
            stats['total_requests'] = stats.get('total_requests', 0) + 1
            stats['total_tokens'] = stats.get('total_tokens', 0) + tokens_used
            total = stats['total_requests']
            current_avg = stats['avg_response_time']
            stats['avg_response_time'] = ((current_avg * (total - 1)) + response_time) / total

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Update cache
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

    # ---------- SEARCH HISTORY ----------
    async def search_history(self, user_id: int, query: str, limit: int = 5) -> List[Dict]:
        """Search user's history for entries containing the query (case‑insensitive)."""
        user_data = await self.load_user_data(user_id)
        history = user_data.get('history', [])
        results = []
        query_lower = query.lower()
        for entry in reversed(history):  # newest first
            text = entry.get('message', '') + ' ' + entry.get('query', '')
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
        user_dir = self._get_user_dir(user_id, username)
        priority_file = user_dir / "model_priority.json"
        if priority_file.exists():
            try:
                with open(priority_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list) and engine == "text":
                    return data
                elif isinstance(data, dict):
                    return data.get(engine, []) if engine in data else None
                else:
                    return None
            except Exception as e:
                logger.error(f"Failed to load priority for {user_id} ({engine}): {e}")
        return None

    async def save_model_priority(self, user_id: int, username: Optional[str],
                                  priority_list: List[str], engine: str = "text"):
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
            logger.error(f"Failed to clear data for {user_id}: {e}")
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