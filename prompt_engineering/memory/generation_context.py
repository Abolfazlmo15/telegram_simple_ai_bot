"""
Stores generation context for each user:
- Last generated image
- Last prompt used
- Model used
- Style detected
- Corrections applied
- Generation history with timestamps
"""
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from core.config import Config

logger = logging.getLogger(__name__)


class GenerationContext:
    """
    Manages generation context per user.
    Stores: prompt, image bytes (or reference), model, style, corrections.
    Supports multiple generations with history tracking.
    """

    def __init__(self, storage_dir: str = "users", max_history: int = 10):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._contexts: Dict[int, Dict[str, Any]] = {}
        self._history: Dict[int, List[Dict[str, Any]]] = {}
        self._lock = None  # Asyncio lock would be used in production
        self.max_history = max_history
        logger.info(f"🧠 GenerationContext initialized (max_history: {max_history})")

    async def store_generation(self, user_id: int, prompt: str, image_bytes: bytes,
                               model_used: str, style: str, source: str) -> None:
        """
        Store a new generation in context and add to history.
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "image_bytes": image_bytes,  # Store bytes in memory for quick access
            "model_used": model_used,
            "style": style,
            "source": source,
            "corrections": [],
            "refined_prompt": prompt,
            "original_prompt": prompt
        }

        self._contexts[user_id] = context

        # Add to history
        if user_id not in self._history:
            self._history[user_id] = []

        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model_used": model_used,
            "style": style,
            "source": source
        }
        self._history[user_id].append(history_entry)

        # Trim history
        if len(self._history[user_id]) > self.max_history:
            self._history[user_id] = self._history[user_id][-self.max_history:]

        # Also save to disk for persistence
        await self._save_to_disk(user_id, context)

        logger.info(f"🧠 Stored generation for user {user_id} (history: {len(self._history[user_id])})")

    async def get_last_generation(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get the last generation context for a user."""
        if user_id not in self._contexts:
            # Try to load from disk
            context = await self._load_from_disk(user_id)
            if context:
                self._contexts[user_id] = context
            else:
                return None

        context = self._contexts.get(user_id)
        if not context:
            return None

        # Check if context is too old (e.g., > 1 hour)
        timestamp = datetime.fromisoformat(context["timestamp"])
        if datetime.now() - timestamp > timedelta(hours=1):
            # Still return it but note it's old
            context["_is_old"] = True

        return context.copy()

    async def get_generation_history(self, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent generation history for a user."""
        if user_id not in self._history:
            # Try to load from disk
            await self._load_history_from_disk(user_id)

        history = self._history.get(user_id, [])
        return history[-limit:] if history else []

    async def has_recent_generation(self, user_id: int, max_age_seconds: int = 600) -> bool:
        """Check if the user has a recent generation (within max_age_seconds)."""
        context = await self.get_last_generation(user_id)
        if not context:
            return False

        timestamp = datetime.fromisoformat(context["timestamp"])
        age_seconds = (datetime.now() - timestamp).total_seconds()

        return age_seconds < max_age_seconds

    async def update_generation(self, user_id: int, prompt: str, image_bytes: bytes,
                                model_used: str, style: str, source: str) -> None:
        """Update an existing generation context."""
        if user_id not in self._contexts:
            await self.store_generation(user_id, prompt, image_bytes, model_used, style, source)
            return

        context = self._contexts[user_id]

        # Store the previous prompt as a correction
        if "corrections" not in context:
            context["corrections"] = []

        context["corrections"].append({
            "timestamp": datetime.now().isoformat(),
            "old_prompt": context["prompt"],
            "new_prompt": prompt,
            "reason": "correction"
        })

        context["timestamp"] = datetime.now().isoformat()
        context["prompt"] = prompt
        context["image_bytes"] = image_bytes
        context["model_used"] = model_used
        context["style"] = style
        context["source"] = source

        await self._save_to_disk(user_id, context)
        logger.info(f"🧠 Updated generation for user {user_id}")

    async def get_correction_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get correction history for a user."""
        context = await self.get_last_generation(user_id)
        if not context:
            return []
        return context.get("corrections", [])

    async def clear_context(self, user_id: int) -> None:
        """Clear the generation context for a user."""
        if user_id in self._contexts:
            del self._contexts[user_id]
        if user_id in self._history:
            del self._history[user_id]

        disk_file = self.storage_dir / f"gen_context_{user_id}.json"
        if disk_file.exists():
            disk_file.unlink()

        history_file = self.storage_dir / f"gen_history_{user_id}.json"
        if history_file.exists():
            history_file.unlink()

        logger.info(f"🧠 Cleared context for user {user_id}")

    async def _save_to_disk(self, user_id: int, context: Dict[str, Any]) -> None:
        """Save context to disk (without image bytes to save space)."""
        disk_context = {k: v for k, v in context.items() if k != "image_bytes"}
        disk_context["has_image_bytes"] = "image_bytes" in context

        disk_file = self.storage_dir / f"gen_context_{user_id}.json"
        try:
            with open(disk_file, 'w', encoding='utf-8') as f:
                json.dump(disk_context, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save generation context for user {user_id}: {e}")

        # Save history separately
        history_file = self.storage_dir / f"gen_history_{user_id}.json"
        try:
            if user_id in self._history and self._history[user_id]:
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(self._history[user_id], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save generation history for user {user_id}: {e}")

    async def _load_from_disk(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Load context from disk."""
        disk_file = self.storage_dir / f"gen_context_{user_id}.json"
        if not disk_file.exists():
            return None

        try:
            with open(disk_file, 'r', encoding='utf-8') as f:
                context = json.load(f)
            context["image_bytes"] = None
            return context
        except Exception as e:
            logger.error(f"Failed to load generation context for user {user_id}: {e}")
            return None

    async def _load_history_from_disk(self, user_id: int) -> None:
        """Load history from disk."""
        history_file = self.storage_dir / f"gen_history_{user_id}.json"
        if not history_file.exists():
            return

        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                self._history[user_id] = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load generation history for user {user_id}: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Return information about the context manager."""
        return {
            "type": "GenerationContext",
            "active_users": len(self._contexts),
            "history_users": len(self._history),
            "max_history": self.max_history,
            "storage_dir": str(self.storage_dir)
        }