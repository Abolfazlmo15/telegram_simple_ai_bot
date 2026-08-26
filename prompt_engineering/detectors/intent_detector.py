"""
Multi-layer intent detector for the bot.
Detects whether the user wants:
- Image generation
- Voice generation
- Text analysis (default)
- Correction/refinement of a previous generation
- Mode change (voice/text)
"""
import logging
import re
from typing import Dict, Any, Optional, Tuple
from core.config import Config
from prompt_engineering.base.base_detector import BaseDetector
from prompt_engineering.memory.generation_context import GenerationContext

logger = logging.getLogger(__name__)


class IntentDetector(BaseDetector):
    """
    Multi-layer intent detector.
    Uses keyword matching, pattern detection, and context analysis.
    """

    def __init__(self, generation_context: Optional[GenerationContext] = None):
        super().__init__()
        self.generation_context = generation_context
        self.keywords = Config.IMAGE_GENERATION_KEYWORDS
        self.voice_keywords = [
            "say this", "speak this", "read this aloud", "tell me this",
            "voice this", "audio of", "speak the text", "say it",
            "narrate this", "convert to speech", "text to speech", "tts",
            "say that", "speak that", "read that"
        ]
        self.correction_keywords = [
            "no", "not", "didn't", "didnt", "without", "missing", "add", "remove",
            "change", "instead", "but", "however", "actually", "re do", "redo",
            "retry", "try again", "generate again", "make it", "do it",
            "wrong", "incorrect", "bad", "poor", "ugly", "terrible",
            "more", "less", "bigger", "smaller", "brighter", "darker",
            "better", "improve", "fix", "correct"
        ]

        # ========== Mode change detection ==========
        self.mode_keywords = {
            "voice": [
                "voice mode", "talk in voice", "speak to me", "voice only",
                "respond in voice", "answer in voice", "voice response",
                "let's talk in voice", "switch to voice", "voice chat",
                "audio mode", "speak mode"
            ],
            "text": [
                "text mode", "talk in text", "text only", "respond in text",
                "answer in text", "text response", "let's talk in text",
                "switch to text", "text chat", "type mode"
            ]
        }
        self._last_intent_cache = {}
        # Confidence threshold for accepting a detection
        self.min_confidence = 0.6
        logger.info("🔍 IntentDetector initialized")

    async def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect the user's intent.

        Returns:
            {
                "intent": "image_generation" | "voice_generation" | "correction" | "text_analysis" | "mode_change" | "unknown",
                "confidence": float,
                "detected_phrase": str,
                "extracted_prompt": str,
                "is_correction": bool,
                "mode": "voice" | "text" | None  # only for mode_change
            }
        """
        if not text or not text.strip():
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "detected_phrase": "",
                "extracted_prompt": text or "",
                "is_correction": False
            }

        text_lower = text.lower()
        user_id = context.get('user_id') if context else None

        # ============================================================
        # 1. Mode change detection (high priority)
        # ============================================================
        mode = self._detect_mode_change(text_lower)
        if mode:
            logger.info(f"🔊 Detected mode change to: {mode}")
            return {
                "intent": "mode_change",
                "confidence": 0.9,
                "detected_phrase": text[:100],
                "extracted_prompt": "",
                "is_correction": False,
                "mode": mode
            }

        # ============================================================
        # 2. Correction detection (only if there is a recent generation)
        # ============================================================
        is_correction, correction_type, correction_text = await self._detect_correction(text, context)
        if is_correction and self.generation_context and user_id:
            # Ensure there is a recent generation (within 10 minutes)
            has_recent = await self.generation_context.has_recent_generation(user_id, max_age_seconds=600)
            if has_recent:
                original = await self.generation_context.get_last_generation(user_id)
                if original:
                    logger.info(f"🔍 Detected correction: {correction_type} - {correction_text}")
                    return {
                        "intent": "image_generation",
                        "confidence": 0.95,
                        "detected_phrase": text[:100],
                        "extracted_prompt": correction_text or text,
                        "is_correction": True,
                        "correction_type": correction_type,
                        "original_prompt": original.get("prompt", ""),
                        "original_model": original.get("model_used", "")
                    }

        # ============================================================
        # 3. Image generation detection (with extra validation)
        # ============================================================
        image_detection = self._detect_image_generation(text)
        if image_detection and image_detection.get("confidence", 0) >= self.min_confidence:
            prompt = image_detection.get("extracted_prompt", text)
            # Ensure the prompt is not just the keyword itself (e.g., "generate" alone)
            if len(prompt.split()) >= 2:
                return {
                    "intent": "image_generation",
                    "confidence": image_detection["confidence"],
                    "detected_phrase": image_detection.get("detected_phrase", ""),
                    "extracted_prompt": prompt,
                    "is_correction": False
                }

        # ============================================================
        # 4. Voice generation detection
        # ============================================================
        voice_detection = self._detect_voice_generation(text)
        if voice_detection and voice_detection.get("confidence", 0) >= self.min_confidence:
            prompt = voice_detection.get("extracted_prompt", text)
            if len(prompt.split()) >= 2:
                return {
                    "intent": "voice_generation",
                    "confidence": voice_detection["confidence"],
                    "detected_phrase": voice_detection.get("detected_phrase", ""),
                    "extracted_prompt": prompt,
                    "is_correction": False
                }

        # ============================================================
        # 5. Pattern matching fallback
        # ============================================================
        detected = self._detect_patterns(text)
        if detected and detected.get("confidence", 0) >= self.min_confidence:
            return detected

        # ============================================================
        # 6. Default: text analysis
        # ============================================================
        return {
            "intent": "text_analysis",
            "confidence": 0.7,
            "detected_phrase": "",
            "extracted_prompt": text,
            "is_correction": False
        }

    def _detect_mode_change(self, text: str) -> Optional[str]:
        """Check if the user wants to change response mode."""
        for mode, phrases in self.mode_keywords.items():
            for phrase in phrases:
                if phrase in text:
                    return mode
        return None

    async def _detect_correction(self, text: str, context: Optional[Dict] = None) -> Tuple[bool, str, str]:
        """Detect if the user is correcting a previous generation."""
        text_lower = text.lower()
        has_correction_keyword = any(kw in text_lower for kw in self.correction_keywords)
        if not has_correction_keyword:
            return False, "", ""

        # Additional check: the message should be relatively short and likely a correction
        if len(text.split()) > 20:
            # Long messages are less likely to be pure corrections
            return False, "", ""

        user_id = context.get('user_id') if context else None
        if user_id and self.generation_context:
            has_previous = await self.generation_context.has_recent_generation(user_id, max_age_seconds=600)
            if not has_previous:
                return False, "", ""

        correction_type = "refinement"
        if any(kw in text_lower for kw in ["no", "not", "didn't", "didnt", "without", "missing", "wrong", "incorrect"]):
            correction_type = "remove_or_fix"
        elif any(kw in text_lower for kw in ["add", "more", "include", "also", "too"]):
            correction_type = "add"
        elif any(kw in text_lower for kw in ["change", "instead", "better", "improve", "fix"]):
            correction_type = "replace"

        # Remove common filler words to get the actual correction
        correction_text = text
        filler_phrases = ["the image", "the picture", "it", "that", "this"]
        for phrase in filler_phrases:
            correction_text = re.sub(rf'\b{phrase}\b', '', correction_text, flags=re.IGNORECASE)
        correction_text = correction_text.strip()
        if not correction_text:
            correction_text = text

        return True, correction_type, correction_text

    def _detect_image_generation(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect image generation intent with extra validation."""
        text_lower = text.lower()
        for kw in self.keywords:
            if kw in text_lower:
                # Extract prompt after the keyword
                prompt = self._extract_prompt_from_keyword(text, kw)
                if prompt and len(prompt.split()) >= 2:
                    # Ensure the prompt contains some descriptive content (not just "something")
                    if not any(vague in prompt.lower() for vague in Config.VAGUE_PROMPT_INDICATORS):
                        return {
                            "intent": "image_generation",
                            "confidence": 0.9,
                            "detected_phrase": kw,
                            "extracted_prompt": prompt
                        }
        return None

    def _detect_voice_generation(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect voice generation intent."""
        text_lower = text.lower()
        for kw in self.voice_keywords:
            if kw in text_lower:
                prompt = self._extract_prompt_from_keyword(text, kw)
                if prompt and len(prompt.split()) >= 2:
                    return {
                        "intent": "voice_generation",
                        "confidence": 0.9,
                        "detected_phrase": kw,
                        "extracted_prompt": prompt
                    }
        return None

    def _extract_prompt_from_keyword(self, text: str, keyword: str) -> str:
        """Extract the prompt after the keyword."""
        idx = text.lower().find(keyword.lower())
        if idx != -1:
            prompt = text[idx + len(keyword):].strip()
            prompt = prompt.lstrip(':,;.- ').strip()
            if prompt:
                return prompt
        return text

    def _detect_patterns(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect using regex patterns."""
        text_lower = text.lower()
        patterns = [
            (r'image of (.+)', 'image_generation'),
            (r'picture of (.+)', 'image_generation'),
            (r'photo of (.+)', 'image_generation'),
            (r'visual of (.+)', 'image_generation'),
        ]
        for pattern, intent in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                prompt = match.group(1).strip()
                if prompt and len(prompt.split()) >= 2:
                    return {
                        "intent": intent,
                        "confidence": 0.8,
                        "detected_phrase": match.group(0),
                        "extracted_prompt": prompt,
                        "is_correction": False
                    }
        return None

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "IntentDetector",
            "keyword_count": len(self.keywords),
            "voice_keyword_count": len(self.voice_keywords),
            "correction_keyword_count": len(self.correction_keywords),
            "mode_keyword_count": sum(len(v) for v in self.mode_keywords.values())
        }