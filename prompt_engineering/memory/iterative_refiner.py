"""
Handles multiple refinement iterations for image generation.
Tracks the history of refinements and merges them into a final prompt.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from prompt_engineering.base.base_memory import BaseMemory

logger = logging.getLogger(__name__)


class IterativeRefiner(BaseMemory):
    """
    Manages iterative refinement of prompts.

    Features:
    - Tracks all refinements applied to a prompt
    - Stores history of each iteration
    - Merges multiple corrections into a cohesive prompt
    - Can revert to previous versions
    - Supports rollback
    """

    def __init__(self, max_history: int = 10):
        super().__init__()
        self.max_history = max_history
        self._refinement_history: Dict[int, List[Dict[str, Any]]] = {}
        self._prompt_versions: Dict[int, List[str]] = {}
        logger.info("🔄 IterativeRefiner initialized")

    async def store(self, key: str, value: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a refinement step.
        Key is the user_id (as string).
        """
        user_id = int(key) if isinstance(key, str) else key
        timestamp = datetime.now().isoformat()

        entry = {
            "timestamp": timestamp,
            "prompt": value if isinstance(value, str) else str(value),
            "context": context or {},
            "version": len(self._prompt_versions.get(user_id, [])) + 1
        }

        if user_id not in self._refinement_history:
            self._refinement_history[user_id] = []
            self._prompt_versions[user_id] = []

        self._refinement_history[user_id].append(entry)
        self._prompt_versions[user_id].append(entry["prompt"])

        # Trim history if it exceeds max
        if len(self._refinement_history[user_id]) > self.max_history:
            self._refinement_history[user_id] = self._refinement_history[user_id][-self.max_history:]
            self._prompt_versions[user_id] = self._prompt_versions[user_id][-self.max_history:]

        logger.info(f"🔄 Stored refinement for user {user_id} (version {entry['version']})")

    async def retrieve(self, key: str, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Retrieve the latest prompt for a user.
        """
        user_id = int(key) if isinstance(key, str) else key
        if user_id in self._prompt_versions and self._prompt_versions[user_id]:
            return self._prompt_versions[user_id][-1]
        return None

    async def delete(self, key: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Clear all refinement history for a user.
        """
        user_id = int(key) if isinstance(key, str) else key
        if user_id in self._refinement_history:
            del self._refinement_history[user_id]
            del self._prompt_versions[user_id]
            logger.info(f"🔄 Cleared refinement history for user {user_id}")
            return True
        return False

    async def exists(self, key: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a user has refinement history.
        """
        user_id = int(key) if isinstance(key, str) else key
        return user_id in self._refinement_history and bool(self._refinement_history[user_id])

    async def get_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get the full refinement history for a user."""
        return self._refinement_history.get(user_id, [])

    async def get_latest(self, user_id: int) -> Optional[str]:
        """Get the latest refined prompt."""
        return await self.retrieve(str(user_id))

    async def get_version(self, user_id: int, version: int) -> Optional[str]:
        """Get a specific version of the prompt."""
        if user_id in self._prompt_versions and version <= len(self._prompt_versions[user_id]):
            return self._prompt_versions[user_id][version - 1]
        return None

    async def rollback(self, user_id: int, versions_back: int = 1) -> Optional[str]:
        """
        Rollback to a previous version.
        versions_back: number of versions to go back (1 = previous version).
        """
        if user_id not in self._prompt_versions or len(self._prompt_versions[user_id]) <= 1:
            logger.warning(f"🔄 Cannot rollback for user {user_id}: not enough history")
            return None

        if versions_back > len(self._prompt_versions[user_id]) - 1:
            versions_back = len(self._prompt_versions[user_id]) - 1

        target_index = len(self._prompt_versions[user_id]) - versions_back - 1
        if target_index < 0:
            target_index = 0

        # Remove all versions after target_index
        self._prompt_versions[user_id] = self._prompt_versions[user_id][:target_index + 1]
        self._refinement_history[user_id] = self._refinement_history[user_id][:target_index + 1]

        latest = self._prompt_versions[user_id][-1]
        logger.info(f"🔄 Rolled back user {user_id} to version {target_index + 1}")
        return latest

    async def merge_corrections(self, user_id: int, original: str, corrections: List[str]) -> str:
        """
        Merge multiple correction suggestions into a single refined prompt.
        """
        if not corrections:
            return original

        # Start with original
        refined = original

        for correction in corrections:
            # Apply each correction (simple append for now)
            # More sophisticated merging could be done with LLM
            if correction not in refined:
                refined = f"{refined}, {correction}"

        # Remove duplicates in a simple way
        parts = refined.split(', ')
        seen = set()
        unique_parts = []
        for part in parts:
            if part.lower() not in seen:
                seen.add(part.lower())
                unique_parts.append(part)

        return ', '.join(unique_parts)

    def get_info(self) -> Dict[str, Any]:
        """Return information about the refiner."""
        return {
            "name": self.name,
            "type": "IterativeRefiner",
            "max_history": self.max_history,
            "users_with_history": len(self._refinement_history)
        }