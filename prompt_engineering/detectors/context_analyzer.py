"""
Analyzes surrounding context to improve intent and style detection.
Uses conversation history to understand what the user is referring to.
"""
import logging
from typing import Dict, Any, Optional, List
from prompt_engineering.base.base_detector import BaseDetector
from core.managers.user_data_manager import UserDataManager

logger = logging.getLogger(__name__)


class ContextAnalyzer(BaseDetector):
    """
    Analyzes conversation context to improve detection accuracy.

    Features:
    - Detects if the user is referring to a previous generation
    - Extracts context from conversation history
    - Identifies follow-up questions and corrections
    """

    def __init__(self, user_data_manager: Optional[UserDataManager] = None):
        super().__init__()
        self.user_data_manager = user_data_manager
        self._context_cache = {}
        logger.info("📖 ContextAnalyzer initialized")

    async def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the context of the user's message.

        Returns:
            {
                "has_context": bool,
                "context_type": "follow_up" | "correction" | "new_topic" | "unknown",
                "referenced_type": "image" | "voice" | "text" | None,
                "referenced_prompt": str,
                "message_type": "question" | "statement" | "command",
                "conversation_age": int,  # seconds since last interaction
                "previous_topics": list,
                "is_related_to_previous": bool
            }
        """
        if not context:
            context = {}

        user_id = context.get('user_id')
        history = context.get('history', [])

        if not user_id:
            return {
                "has_context": False,
                "context_type": "unknown",
                "referenced_type": None,
                "referenced_prompt": "",
                "message_type": "statement",
                "conversation_age": 0,
                "previous_topics": [],
                "is_related_to_previous": False
            }

        # Analyze the context
        result = {
            "has_context": False,
            "context_type": "unknown",
            "referenced_type": None,
            "referenced_prompt": "",
            "message_type": "statement",
            "conversation_age": 0,
            "previous_topics": [],
            "is_related_to_previous": False
        }

        # Check if the user is referring to a previous generation
        if history:
            result["has_context"] = True
            last_entries = history[-5:]

            # Find the most recent generation entry
            for entry in reversed(last_entries):
                entry_type = entry.get('type', '')
                if entry_type in ('generated_image', 'generated_voice'):
                    result["referenced_type"] = entry_type.replace('generated_', '')
                    result["referenced_prompt"] = entry.get('prompt', '')
                    result["previous_topics"].append(result["referenced_type"])
                    result["has_context"] = True
                    break

            # Check if this is a follow-up or correction
            if result["has_context"]:
                text_lower = text.lower()
                correction_keywords = [
                    "no", "not", "didn't", "didnt", "without", "missing",
                    "add", "remove", "change", "instead", "but", "however",
                    "actually", "re do", "redo", "retry", "try again",
                    "wrong", "incorrect", "bad", "poor", "more", "less",
                    "better", "improve", "fix", "correct"
                ]

                # Determine if it's a correction or follow-up
                if any(kw in text_lower for kw in correction_keywords):
                    result["context_type"] = "correction"
                else:
                    result["context_type"] = "follow_up"

                result["is_related_to_previous"] = True

        # Determine message type
        text_lower = text.lower()
        if any(text_lower.startswith(q) for q in
               ["what", "how", "why", "when", "where", "who", "which", "can", "does", "is", "are"]):
            result["message_type"] = "question"
        elif any(text_lower.startswith(c) for c in
                 ["generate", "create", "make", "produce", "draw", "paint", "show", "give", "tell", "say", "speak"]):
            result["message_type"] = "command"
        else:
            result["message_type"] = "statement"

        # Calculate conversation age
        if history:
            last_timestamp = history[-1].get('timestamp', '')
            if last_timestamp:
                try:
                    from datetime import datetime
                    last_time = datetime.fromisoformat(last_timestamp)
                    result["conversation_age"] = int((datetime.now() - last_time).total_seconds())
                except Exception:
                    result["conversation_age"] = 0

        return result

    async def is_correction(self, text: str, context: Optional[Dict] = None) -> bool:
        """Check if the user is correcting a previous generation."""
        result = await self.detect(text, context)
        return result.get("context_type") == "correction" and result.get("is_related_to_previous", False)

    async def get_previous_generation_prompt(self, user_id: int) -> Optional[str]:
        """Get the prompt of the most recent generation for a user."""
        if not self.user_data_manager:
            return None

        user_data = await self.user_data_manager.load_user_data(user_id)
        history = user_data.get('history', [])

        for entry in reversed(history):
            if entry.get('type') in ('generated_image', 'generated_voice'):
                return entry.get('prompt', '')

        return None

    def get_info(self) -> Dict[str, Any]:
        """Return information about the analyzer."""
        return {
            "name": self.name,
            "type": "ContextAnalyzer",
            "features": ["conversation_history", "correction_detection", "topic_tracking"]
        }