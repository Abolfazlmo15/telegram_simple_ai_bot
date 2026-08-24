"""
Manages per-user conversation state including:
- Current response mode (text or voice)
- Whether voice mode is active
- Conversation history for context
"""
import logging
import json
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from core.config import Config

logger = logging.getLogger(__name__)


class ConversationMode(Enum):
    """Response mode for the bot."""
    TEXT = "text"
    VOICE = "voice"
    AUTO = "auto"  # Decide based on input type


class ConversationState:
    """
    Tracks conversation state per user.
    """

    def __init__(self, storage_dir: str = "users"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._states: Dict[int, Dict] = {}
        self._lock = None  # Asyncio lock would be used in production
        logger.info("🧠 ConversationState initialized")

    def _get_state_file(self, user_id: int) -> Path:
        """Get the state file path for a user."""
        return self.storage_dir / f"conv_state_{user_id}.json"

    async def get_mode(self, user_id: int) -> ConversationMode:
        """Get the user's current conversation mode."""
        state = await self._load_state(user_id)
        mode_str = state.get("mode", "text")
        try:
            return ConversationMode(mode_str)
        except ValueError:
            return ConversationMode.TEXT

    async def set_mode(self, user_id: int, mode: ConversationMode) -> None:
        """Set the user's conversation mode."""
        state = await self._load_state(user_id)
        state["mode"] = mode.value
        state["last_updated"] = datetime.now().isoformat()
        await self._save_state(user_id, state)
        logger.info(f"🗣️ User {user_id} mode set to: {mode.value}")

    async def is_voice_mode_active(self, user_id: int) -> bool:
        """Check if voice mode is active for the user."""
        state = await self._load_state(user_id)
        # Voice mode is active if:
        # 1. Mode is VOICE, OR
        # 2. Mode is AUTO and the last interaction was voice
        mode = state.get("mode", "text")

        if mode == "voice":
            return True
        if mode == "auto":
            # Check if last interaction was voice
            return state.get("last_input_type") == "voice"
        return False

    async def record_input(self, user_id: int, input_type: str, text: str = "") -> None:
        """Record that the user sent a message (for context)."""
        state = await self._load_state(user_id)
        state["last_input_type"] = input_type  # "text", "voice", "image"
        state["last_input_text"] = text[:200]  # Store truncated for context
        state["last_interaction"] = datetime.now().isoformat()

        # Update conversation history
        if "history" not in state:
            state["history"] = []
        state["history"].append({
            "timestamp": datetime.now().isoformat(),
            "type": input_type,
            "text": text[:100]
        })
        # Keep last 10 interactions
        if len(state["history"]) > 10:
            state["history"] = state["history"][-10:]

        await self._save_state(user_id, state)

    async def get_conversation_context(self, user_id: int) -> Dict:
        """Get the user's conversation context."""
        return await self._load_state(user_id)

    async def _load_state(self, user_id: int) -> Dict:
        """Load state from disk or return default."""
        if user_id in self._states:
            return self._states[user_id]

        state_file = self._get_state_file(user_id)
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self._states[user_id] = state
                return state
            except Exception as e:
                logger.error(f"Failed to load state for user {user_id}: {e}")

        # Default state
        default = {
            "mode": "text",
            "last_input_type": "text",
            "last_input_text": "",
            "last_interaction": datetime.now().isoformat(),
            "history": [],
            "voice_session_active": False,
            "voice_session_start": None
        }
        self._states[user_id] = default
        return default

    async def _save_state(self, user_id: int, state: Dict) -> None:
        """Save state to disk."""
        state_file = self._get_state_file(user_id)
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            self._states[user_id] = state
        except Exception as e:
            logger.error(f"Failed to save state for user {user_id}: {e}")

    async def reset_mode(self, user_id: int) -> None:
        """Reset the user's mode to text."""
        await self.set_mode(user_id, ConversationMode.TEXT)

    async def clear_state(self, user_id: int) -> None:
        """Clear the user's state."""
        if user_id in self._states:
            del self._states[user_id]
        state_file = self._get_state_file(user_id)
        if state_file.exists():
            state_file.unlink()
        logger.info(f"🧹 Cleared conversation state for user {user_id}")

    def get_info(self) -> Dict:
        """Return information about the state manager."""
        return {
            "type": "ConversationState",
            "active_users": len(self._states),
            "storage_dir": str(self.storage_dir)
        }