"""
Manages user preferences with persistent JSON storage.
Preferences are stored in a separate file per user: {user_dir}/preferences.json
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from core.config import Config

logger = logging.getLogger(__name__)


class PreferenceManager:
    """
    Manages user preferences for the bot.

    Preferences stored:
    - response_mode: "text" | "voice" | "auto"
    - response_style: "concise" | "detailed" | "balanced"
    - preferred_models: List[str] (model IDs in priority order)
    - preferred_style: str (artistic style preference for images)
    - custom_instructions: str (user-defined instructions)
    - default_negatives: List[str] (default negative prompts)
    - max_response_length: int
    - voice_speed: float (0.5 - 2.0)
    - voice_style: str (e.g., "neutral", "happy", "serious")
    - memory_enabled: bool
    - last_updated: str (timestamp)
    """

    DEFAULT_PREFERENCES = {
        "response_mode": "text",  # "text" | "voice" | "auto"
        "response_style": "balanced",  # "concise" | "detailed" | "balanced"
        "preferred_models": [],  # List of model IDs in priority order
        "preferred_style": "no_style",  # Artistic style preference
        "custom_instructions": "",  # User-defined instructions
        "default_negatives": [
            "blurry, low quality, distorted",
            "bad anatomy, deformed, ugly"
        ],
        "max_response_length": 2000,  # Max tokens for responses
        "voice_speed": 1.0,  # 0.5 - 2.0
        "voice_style": "neutral",  # "neutral", "happy", "serious", "excited"
        "memory_enabled": True,
        "last_updated": ""
    }

    def __init__(self, user_data_manager):
        """
        Initialize the preference manager.

        Args:
            user_data_manager: UserDataManager instance for directory management
        """
        self.user_data_manager = user_data_manager
        self._cache: Dict[int, Dict] = {}
        logger.info("📋 PreferenceManager initialized")

    async def get_preferences(self, user_id: int, username: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user preferences. Returns default preferences if none exist.
        """
        if user_id in self._cache:
            return self._cache[user_id].copy()

        user_dir = self._get_user_dir(user_id, username)
        pref_file = user_dir / "preferences.json"

        if pref_file.exists():
            try:
                with open(pref_file, 'r', encoding='utf-8') as f:
                    prefs = json.load(f)
                # Ensure all default keys exist
                for key, default_val in self.DEFAULT_PREFERENCES.items():
                    if key not in prefs:
                        prefs[key] = default_val
                self._cache[user_id] = prefs
                return prefs.copy()
            except Exception as e:
                logger.error(f"Failed to load preferences for user {user_id}: {e}")

        # Return default preferences
        default_prefs = self.DEFAULT_PREFERENCES.copy()
        default_prefs["last_updated"] = datetime.now().isoformat()
        self._cache[user_id] = default_prefs
        await self._save_preferences(user_id, username, default_prefs)
        return default_prefs.copy()

    async def save_preferences(self, user_id: int, username: Optional[str],
                               preferences: Dict[str, Any]) -> bool:
        """
        Save user preferences.

        Args:
            user_id: User ID
            username: Username (optional)
            preferences: Dictionary of preferences to save

        Returns:
            True if successful, False otherwise
        """
        try:
            prefs = await self.get_preferences(user_id, username)
            # Update with new preferences
            for key, value in preferences.items():
                if key in self.DEFAULT_PREFERENCES:
                    prefs[key] = value
            prefs["last_updated"] = datetime.now().isoformat()
            self._cache[user_id] = prefs
            return await self._save_preferences(user_id, username, prefs)
        except Exception as e:
            logger.error(f"Failed to save preferences for user {user_id}: {e}")
            return False

    async def get_preference(self, user_id: int, key: str,
                             username: Optional[str] = None) -> Any:
        """
        Get a single preference value.
        """
        prefs = await self.get_preferences(user_id, username)
        return prefs.get(key, self.DEFAULT_PREFERENCES.get(key))

    async def set_preference(self, user_id: int, key: str, value: Any,
                             username: Optional[str] = None) -> bool:
        """
        Set a single preference value.
        """
        if key not in self.DEFAULT_PREFERENCES:
            logger.warning(f"Unknown preference key: {key}")
            return False
        return await self.save_preferences(user_id, username, {key: value})

    async def get_response_mode(self, user_id: int,
                                username: Optional[str] = None) -> str:
        """Get user's preferred response mode."""
        return await self.get_preference(user_id, "response_mode", username)

    async def set_response_mode(self, user_id: int, mode: str,
                                username: Optional[str] = None) -> bool:
        """Set user's response mode."""
        if mode not in ("text", "voice", "auto"):
            return False
        return await self.set_preference(user_id, "response_mode", mode, username)

    async def get_response_style(self, user_id: int,
                                 username: Optional[str] = None) -> str:
        """Get user's preferred response style."""
        return await self.get_preference(user_id, "response_style", username)

    async def set_response_style(self, user_id: int, style: str,
                                 username: Optional[str] = None) -> bool:
        """Set user's response style."""
        if style not in ("concise", "detailed", "balanced"):
            return False
        return await self.set_preference(user_id, "response_style", style, username)

    async def get_preferred_models(self, user_id: int,
                                   username: Optional[str] = None) -> List[str]:
        """Get user's preferred models in priority order."""
        return await self.get_preference(user_id, "preferred_models", username) or []

    async def set_preferred_models(self, user_id: int, models: List[str],
                                   username: Optional[str] = None) -> bool:
        """Set user's preferred models."""
        return await self.set_preference(user_id, "preferred_models", models, username)

    async def get_preferred_style(self, user_id: int,
                                  username: Optional[str] = None) -> str:
        """Get user's preferred artistic style."""
        return await self.get_preference(user_id, "preferred_style", username)

    async def get_custom_instructions(self, user_id: int,
                                      username: Optional[str] = None) -> str:
        """Get user's custom instructions."""
        return await self.get_preference(user_id, "custom_instructions", username) or ""

    async def get_voice_speed(self, user_id: int,
                              username: Optional[str] = None) -> float:
        """Get user's preferred voice speed."""
        return await self.get_preference(user_id, "voice_speed", username) or 1.0

    async def get_voice_style(self, user_id: int,
                              username: Optional[str] = None) -> str:
        """Get user's preferred voice style."""
        return await self.get_preference(user_id, "voice_style", username) or "neutral"

    def _get_user_dir(self, user_id: int, username: Optional[str]) -> Path:
        """
        Get user directory path.
        This mirrors the logic in UserDataManager._get_user_dir()
        """
        base_dir = self.user_data_manager.base_dir
        if username:
            dir_name = f"{user_id}-{username}"
        else:
            dir_name = str(user_id)
        user_dir = base_dir / dir_name
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    async def _save_preferences(self, user_id: int, username: Optional[str],
                                preferences: Dict) -> bool:
        """Save preferences to disk."""
        try:
            user_dir = self._get_user_dir(user_id, username)
            pref_file = user_dir / "preferences.json"
            with open(pref_file, 'w', encoding='utf-8') as f:
                json.dump(preferences, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save preferences for user {user_id}: {e}")
            return False

    async def clear_cache(self, user_id: int) -> None:
        """Clear cached preferences for a user."""
        if user_id in self._cache:
            del self._cache[user_id]

    def get_info(self) -> Dict[str, Any]:
        """Return information about the manager."""
        return {
            "type": "PreferenceManager",
            "cached_users": len(self._cache),
            "default_preferences": len(self.DEFAULT_PREFERENCES)
        }