"""
Detects when a user wants to switch conversation mode (text ↔ voice).
Uses natural language understanding without commands.
Expanded with more patterns and LLM fallback.
"""
import logging
import re
from typing import Dict, Any, Optional, Tuple
from prompt_engineering.base.base_detector import BaseDetector
from prompt_engineering.state.conversation_state import ConversationMode

logger = logging.getLogger(__name__)


class ModeDetector(BaseDetector):
    """
    Detects natural language mode switches.

    Examples:
    - "talk to me" → switch to voice
    - "talk in text" → switch to text
    - "speak to me" → switch to voice
    - "tell me in voice" → switch to voice
    - "answer in voice" → switch to voice
    - "type it" → switch to text
    - "just text" → switch to text
    - "let's talk in voice" → switch to voice
    - "I want to talk" → switch to voice
    - "voice mode" → switch to voice
    - "I'd rather hear it" → switch to voice
    - "just text is fine" → switch to text
    - "speak it out loud" → switch to voice
    """

    def __init__(self):
        super().__init__()

        # ============================================================
        # EXPANDED VOICE SWITCH PATTERNS
        # ============================================================
        self.voice_switch_patterns = [
            # Direct commands
            r"talk\s+to\s+me",
            r"speak\s+to\s+me",
            r"talk\s+in\s+voice",
            r"speak\s+in\s+voice",
            r"say\s+it\s+in\s+voice",
            r"tell\s+me\s+in\s+voice",
            r"answer\s+in\s+voice",
            r"respond\s+in\s+voice",
            r"voice\s+mode",
            r"voice\s+response",
            r"voice\s+chat",
            r"audio\s+mode",
            r"speak\s+mode",

            # Natural expressions
            r"let's\s+talk",
            r"let\'s\s+talk",
            r"i\s+want\s+to\s+talk",
            r"i'd\s+like\s+to\s+talk",
            r"can\s+we\s+talk",
            r"talk\s+with\s+me",
            r"speak\s+with\s+me",

            # Variations
            r"say\s+this",
            r"speak\s+this",
            r"read\s+this",
            r"narrate\s+this",
            r"convert\s+to\s+voice",
            r"text\s+to\s+speech",
            r"tts",

            # NEW: More natural variations
            r"i'd\s+rather\s+hear\s+it",
            r"i\s+would\s+rather\s+hear\s+it",
            r"speak\s+it\s+out\s+loud",
            r"say\s+it\s+out\s+loud",
            r"read\s+it\s+to\s+me",
            r"tell\s+me\s+verbally",
            r"respond\s+orally",
            r"give\s+me\s+an\s+audio\s+response",
            r"give\s+me\s+a\s+voice\s+response",
            r"can\s+you\s+speak\s+that",
            r"can\s+you\s+say\s+that",
            r"please\s+speak",
            r"please\s+say\s+it",
            r"speak\s+it\s+to\s+me",
            r"say\s+it\s+to\s+me",
            r"i\s+prefer\s+voice",
            r"i\s+prefer\s+hearing",
        ]

        # ============================================================
        # EXPANDED TEXT SWITCH PATTERNS
        # ============================================================
        self.text_switch_patterns = [
            # Direct commands
            r"type\s+it",
            r"type\s+this",
            r"just\s+text",
            r"text\s+mode",
            r"text\s+response",
            r"answer\s+in\s+text",
            r"respond\s+in\s+text",
            r"tell\s+me\s+in\s+text",
            r"say\s+it\s+in\s+text",
            r"write\s+it",
            r"write\s+this",
            r"just\s+write",
            r"text\s+chat",

            # Negations
            r"no\s+voice",
            r"stop\s+voice",
            r"switch\s+to\s+text",
            r"don't\s+speak",
            r"dont\s+speak",
            r"never\s+mind\s+voice",

            # NEW: More natural variations
            r"i'd\s+rather\s+read\s+it",
            r"i\s+would\s+rather\s+read\s+it",
            r"just\s+type\s+it",
            r"just\s+write\s+it",
            r"don't\s+say\s+it",
            r"dont\s+say\s+it",
            r"text\s+only",
            r"no\s+speech",
            r"keep\s+it\s+in\s+text",
            r"prefer\s+text",
            r"prefer\s+to\s+read",
            r"i\s+prefer\s+reading",
            r"i\s+prefer\s+text",
            r"stop\s+talking",
            r"enough\s+voice",
            r"back\s+to\s+text",
        ]

        # Compile regex patterns for efficiency
        self.voice_regexes = [re.compile(p, re.IGNORECASE) for p in self.voice_switch_patterns]
        self.text_regexes = [re.compile(p, re.IGNORECASE) for p in self.text_switch_patterns]

        # LLM fallback (will be set by base_engine)
        self._text_engine = None
        self._use_llm_fallback = True

        logger.info(f"🗣️ ModeDetector initialized (voice: {len(self.voice_switch_patterns)}, text: {len(self.text_switch_patterns)})")

    async def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect if the user wants to switch conversation mode.

        Returns:
            {
                "detected": bool,
                "target_mode": "voice" | "text" | None,
                "confidence": float,
                "detected_phrase": str,
                "is_mode_switch": bool
            }
        """
        if not text or not text.strip():
            return {
                "detected": False,
                "target_mode": None,
                "confidence": 0.0,
                "detected_phrase": "",
                "is_mode_switch": False
            }

        text_lower = text.lower()
        user_id = context.get('user_id') if context else None

        # ============================================================
        # LAYER 1: Check for voice mode switch patterns (regex)
        # ============================================================
        for regex in self.voice_regexes:
            match = regex.search(text_lower)
            if match:
                # Check if it's actually saying "generate voice" vs "switch to voice"
                if "image" in text_lower or "picture" in text_lower:
                    if not any(phrase in text_lower for phrase in ["say this", "speak this", "read this"]):
                        continue

                logger.info(f"🗣️ Voice mode switch detected (regex): {match.group()}")
                return {
                    "detected": True,
                    "target_mode": "voice",
                    "confidence": 0.9,
                    "detected_phrase": match.group(),
                    "is_mode_switch": True,
                    "full_text": text
                }

        # ============================================================
        # LAYER 2: Check for text mode switch patterns (regex)
        # ============================================================
        for regex in self.text_regexes:
            match = regex.search(text_lower)
            if match:
                logger.info(f"🗣️ Text mode switch detected (regex): {match.group()}")
                return {
                    "detected": True,
                    "target_mode": "text",
                    "confidence": 0.9,
                    "detected_phrase": match.group(),
                    "is_mode_switch": True,
                    "full_text": text
                }

        # ============================================================
        # LAYER 3: LLM Fallback for ambiguous cases
        # ============================================================
        if self._use_llm_fallback and self._text_engine and self._text_engine.is_initialized:
            llm_result = await self._detect_with_llm(text, context)
            if llm_result and llm_result.get('detected'):
                logger.info(f"🗣️ Mode switch detected via LLM: {llm_result.get('target_mode')}")
                return llm_result

        # ============================================================
        # LAYER 4: Voice message context
        # ============================================================
        if context and context.get('input_type') == 'voice':
            return {
                "detected": False,
                "target_mode": None,
                "confidence": 0.0,
                "detected_phrase": "",
                "is_mode_switch": False,
                "from_voice_message": True
            }

        return {
            "detected": False,
            "target_mode": None,
            "confidence": 0.0,
            "detected_phrase": "",
            "is_mode_switch": False
        }

    async def _detect_with_llm(self, text: str, context: Optional[Dict]) -> Optional[Dict]:
        """
        Use LLM to detect mode switches in ambiguous cases.
        """
        if not self._text_engine:
            return None

        detection_prompt = f"""Analyze the user's request and determine if they want to change the response mode to voice or text.

User request: "{text}"

Rules:
1. If the user wants to hear the response in voice, output: VOICE
2. If the user wants to read the response in text, output: TEXT
3. If the user is NOT asking to change mode, output: NONE

Output ONLY one of: VOICE, TEXT, or NONE. No explanation, no quotes.
"""

        try:
            response, _, _ = await self._text_engine.process(
                detection_prompt,
                context={'skip_cache': True}
            )

            result = response.strip().upper()
            if "VOICE" in result:
                return {
                    "detected": True,
                    "target_mode": "voice",
                    "confidence": 0.7,
                    "detected_phrase": text[:50],
                    "is_mode_switch": True
                }
            elif "TEXT" in result:
                return {
                    "detected": True,
                    "target_mode": "text",
                    "confidence": 0.7,
                    "detected_phrase": text[:50],
                    "is_mode_switch": True
                }
        except Exception as e:
            logger.warning(f"LLM mode detection failed: {e}")

        return None

    def set_text_engine(self, text_engine) -> None:
        """Set the text engine for LLM fallback."""
        self._text_engine = text_engine

    def set_llm_fallback(self, enabled: bool) -> None:
        """Enable or disable LLM fallback."""
        self._use_llm_fallback = enabled

    def get_info(self) -> Dict[str, Any]:
        """Return information about the detector."""
        return {
            "name": self.name,
            "type": "ModeDetector",
            "voice_patterns": len(self.voice_switch_patterns),
            "text_patterns": len(self.text_switch_patterns),
            "llm_fallback": self._text_engine is not None and self._use_llm_fallback
        }