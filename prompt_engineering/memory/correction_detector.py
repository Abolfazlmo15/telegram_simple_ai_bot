"""
Detects if the user is correcting a previous generation.
Merges corrections intelligently with the original prompt.
"""
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from prompt_engineering.base.base_detector import BaseDetector
from prompt_engineering.memory.generation_context import GenerationContext

logger = logging.getLogger(__name__)


class CorrectionDetector(BaseDetector):
    """
    Detects and interprets user corrections to previous generations.

    Features:
    - Identifies correction keywords
    - Determines the type of correction (add, remove, change, fix)
    - Extracts the specific element to correct
    - Intelligently merges corrections with the original prompt
    """

    def __init__(self, generation_context: Optional[GenerationContext] = None):
        super().__init__()
        self.generation_context = generation_context

        # Correction keywords categorized by type
        self.correction_patterns = {
            "remove": [
                "no", "not", "don't", "dont", "without", "remove", "delete",
                "get rid of", "take out", "exclude", "missing", "didn't",
                "didnt", "absence", "lack", "forget", "forgot"
            ],
            "add": [
                "add", "include", "put", "place", "insert", "more", "also",
                "extra", "additional", "plus", "with", "and", "together",
                "make it", "make her", "make him", "make the"
            ],
            "change": [
                "change", "instead", "different", "rather", "replace",
                "swap", "switch", "alter", "modify", "adjust", "edit"
            ],
            "enhance": [
                "better", "improve", "enhance", "upgrade", "refine",
                "polish", "perfect", "fix", "correct", "make it"
            ],
            "style": [
                "style", "look", "appearance", "vibe", "aesthetic",
                "mood", "tone", "atmosphere", "feeling"
            ]
        }

        # Common correction phrases
        self.correction_phrases = [
            "the image you generated", "the picture you made", "the image you created",
            "you didn't", "you forgot", "you missed", "there is no", "there isn't",
            "i meant", "i wanted", "i said", "i asked for", "i requested",
            "the image", "the picture", "that image", "that picture"
        ]

        # Phrases that indicate the user wants to keep the original content
        self.keep_phrases = [
            "but keep", "keep the", "keep it", "retain", "same",
            "don't change", "don't remove", "keep everything else"
        ]

        logger.info("🔍 CorrectionDetector initialized")

    async def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the text to detect if it's a correction and what needs to be changed.

        Returns:
            {
                "is_correction": bool,
                "correction_type": "remove" | "add" | "change" | "enhance" | "style" | "unknown",
                "target": str,               # What the user wants to change
                "new_value": str,            # What they want to change it to
                "confidence": float,
                "original_prompt": str,      # The prompt being corrected
                "suggestion": str            # Merged prompt with correction applied
            }
        """
        if not text or not text.strip():
            return self._empty_result()

        text_lower = text.lower()

        # Check if the user is referring to a previous generation
        has_reference = any(phrase in text_lower for phrase in self.correction_phrases)
        has_keyword = self._has_correction_keyword(text_lower)

        if not has_reference and not has_keyword:
            return self._empty_result()

        # Get the original prompt from context or memory
        original_prompt = ""
        user_id = context.get('user_id') if context else None

        if self.generation_context and user_id:
            last_gen = await self.generation_context.get_last_generation(user_id)
            if last_gen:
                original_prompt = last_gen.get('prompt', '')

        if not original_prompt and context and context.get('history'):
            history = context.get('history', [])
            for entry in reversed(history):
                if entry.get('type') in ('generated_image', 'generated_voice'):
                    original_prompt = entry.get('prompt', '')
                    if original_prompt:
                        break

        # Determine correction type
        correction_type = self._determine_correction_type(text_lower)

        # Extract what the user wants to change and what to change it to
        target = self._extract_target(text)
        new_value = self._extract_new_value(text)

        # Generate an intelligent suggestion that merges the correction with the original
        suggestion = self._generate_intelligent_suggestion(
            original_prompt,
            correction_type,
            text,
            target,
            new_value
        )

        # If suggestion is empty or same as original, try a fallback
        if not suggestion or suggestion == original_prompt:
            suggestion = self._generate_fallback_suggestion(original_prompt, text)

        return {
            "is_correction": True,
            "correction_type": correction_type,
            "target": target,
            "new_value": new_value,
            "confidence": 0.85,
            "original_prompt": original_prompt,
            "suggestion": suggestion
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return an empty result."""
        return {
            "is_correction": False,
            "correction_type": "unknown",
            "target": "",
            "new_value": "",
            "confidence": 0.0,
            "original_prompt": "",
            "suggestion": ""
        }

    def _has_correction_keyword(self, text: str) -> bool:
        """Check if the text contains any correction keyword."""
        for patterns in self.correction_patterns.values():
            for pattern in patterns:
                if pattern in text:
                    return True
        return False

    def _determine_correction_type(self, text: str) -> str:
        """Determine the type of correction."""
        # Check for specific patterns first
        if re.search(r'(?:no|not|don\'t|dont|without)\s+(?:the\s+)?(\w+)', text, re.IGNORECASE):
            return "remove"
        if re.search(r'(?:add|include|more)\s+(\w+)', text, re.IGNORECASE):
            return "add"
        if re.search(r'(?:change|replace|instead)\s+(\w+)', text, re.IGNORECASE):
            return "change"
        if re.search(r'(?:better|improve|enhance|fix|correct)', text, re.IGNORECASE):
            return "enhance"
        if re.search(r'(?:style|look|appearance|aesthetic)', text, re.IGNORECASE):
            return "style"

        for corr_type, patterns in self.correction_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return corr_type
        return "unknown"

    def _extract_target(self, text: str) -> str:
        """Extract what the user wants to change."""
        # Patterns like "remove the X", "add more Y", "change Z"
        patterns = [
            (r'remove\s+(?:the\s+)?([\w\s]+)', 'remove'),
            (r'delete\s+(?:the\s+)?([\w\s]+)', 'remove'),
            (r'add\s+(?:more\s+)?([\w\s]+)', 'add'),
            (r'include\s+(?:the\s+)?([\w\s]+)', 'add'),
            (r'change\s+(?:the\s+)?([\w\s]+)', 'change'),
            (r'without\s+(?:the\s+)?([\w\s]+)', 'remove'),
            (r'(?:no|not)\s+(?:the\s+)?([\w\s]+)', 'remove'),
            (r'more\s+([\w\s]+)', 'add'),
            (r'less\s+([\w\s]+)', 'remove'),
            (r'better\s+([\w\s]+)', 'enhance'),
            (r'improve\s+(?:the\s+)?([\w\s]+)', 'enhance'),
        ]

        for pattern, corr_type in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                target = match.group(1).strip()
                if target:
                    return target

        # If no pattern matches, try to extract the subject
        words = text.split()
        important_words = ['people', 'person', 'woman', 'man', 'child', 'sea', 'ocean', 'beach',
                          'sky', 'sun', 'moon', 'stars', 'building', 'house', 'tree', 'flower',
                          'animal', 'dog', 'cat', 'bird', 'fish', 'body', 'face', 'hair', 'eyes']
        for word in words:
            if word.lower() in important_words:
                return word

        return ""

    def _extract_new_value(self, text: str) -> str:
        """Extract what the user wants the new value to be."""
        # Patterns like "change X to Y", "replace X with Y"
        patterns = [
            r'change\s+[\w\s]+\s+(?:to|into)\s+([\w\s]+)',
            r'replace\s+[\w\s]+\s+(?:with|by)\s+([\w\s]+)',
            r'(?:add|include)\s+([\w\s]+)',
            r'more\s+([\w\s]+)',
            r'better\s+([\w\s]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no pattern, extract the last few words
        words = text.split()
        if len(words) > 2:
            important = ['with', 'like', 'as', 'in', 'on', 'at']
            for i, word in enumerate(words):
                if word.lower() in important and i < len(words) - 1:
                    return ' '.join(words[i+1:])

        # If correction starts with "make", extract what comes after
        if text.lower().startswith("make"):
            rest = text[len("make"):].strip()
            if rest:
                return rest

        return ""

    def _generate_intelligent_suggestion(self, original: str, correction_type: str,
                                         text: str, target: str, new_value: str) -> str:
        """
        Generate an intelligent suggestion by merging the correction with the original prompt.
        Uses the LLM via PromptRefiner if available, otherwise falls back to basic string operations.
        """
        if not original:
            # If there's no original, just use the correction text
            return text

        # Try to use LLM if available (will be set by base_engine)
        if hasattr(self, '_refiner') and self._refiner:
            try:
                merged = self._merge_with_llm(original, text, correction_type)
                if merged:
                    return merged
            except Exception as e:
                logger.warning(f"LLM merge failed: {e}")

        # Fallback to intelligent string merging
        return self._merge_manually(original, correction_type, text, target, new_value)

    def _merge_with_llm(self, original: str, correction: str, correction_type: str) -> str:
        """Use LLM to intelligently merge correction with original."""
        import asyncio
        # This will be set by base_engine during initialization
        if not hasattr(self, '_refiner') or not self._refiner:
            return ""

        # The refiner needs to be passed in via a method
        # We'll handle this through the base_engine
        return ""

    def _merge_manually(self, original: str, correction_type: str,
                        text: str, target: str, new_value: str) -> str:
        """
        Manually merge the correction with the original prompt.
        This is the fallback when LLM isn't available.
        """
        if not original:
            return text

        prompt = original

        # Clean up the correction text by removing filler words
        clean_correction = text

        # Remove common filler phrases
        filler_phrases = [
            "make it", "make her", "make him", "make the", "make it look",
            "generate an image of", "generate a picture of", "generate image of",
            "create an image of", "create a picture of", "create image of",
            "i want", "i'd like", "please", "can you", "could you",
            "the image", "the picture", "that image", "that picture"
        ]

        for phrase in filler_phrases:
            clean_correction = re.sub(rf'\b{phrase}\b', '', clean_correction, flags=re.IGNORECASE)
        clean_correction = clean_correction.strip()

        # If correction is just "more natural" type things, add as comma-separated
        if correction_type in ("enhance", "style"):
            if clean_correction:
                prompt = f"{prompt}, {clean_correction}"
            return prompt

        # For "add" corrections, append the new value
        if correction_type == "add":
            if new_value:
                prompt = f"{prompt}, {new_value}"
            elif clean_correction:
                prompt = f"{prompt}, {clean_correction}"
            return prompt

        # For "remove" corrections, try to remove the target
        if correction_type == "remove" and target:
            # Remove phrases containing the target
            pattern = re.compile(r'[,;:.!?]?\s*' + re.escape(target) + r'[,;:.!?]?\s*', re.IGNORECASE)
            prompt = re.sub(pattern, '', prompt)
            prompt = re.sub(r'\s+', ' ', prompt)
            prompt = prompt.strip(' ,.;:')
            return prompt

        # For "change" corrections, try to replace the target
        if correction_type == "change" and target and new_value:
            prompt = prompt.replace(target, new_value)
            if clean_correction and clean_correction not in prompt:
                prompt = f"{prompt}, {clean_correction}"
            return prompt

        # Default: just append the cleaned correction
        if clean_correction and clean_correction not in prompt:
            prompt = f"{prompt}, {clean_correction}"

        return prompt

    def _generate_fallback_suggestion(self, original: str, text: str) -> str:
        """Generate a fallback suggestion when the main logic fails."""
        if not original:
            return text

        # If the correction is a simple "make it more X", just append it
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ["more", "better", "improve", "enhance"]):
            # Extract the key phrase after the verb
            words = text.split()
            if len(words) > 2:
                # Try to find the key modifier
                important = ["realistic", "natural", "beautiful", "detailed", "vibrant",
                           "colorful", "bright", "dark", "moody", "dramatic", "cinematic",
                           "anime", "cartoon", "pixel", "watercolor", "oil", "sketch"]
                for word in words:
                    if word.lower() in important:
                        return f"{original}, more {word.lower()}"

        # Default: append the entire correction as a modifier
        return f"{original}, {text}"

    async def get_info(self) -> Dict[str, Any]:
        """Return information about the detector."""
        return {
            "name": self.name,
            "type": "CorrectionDetector",
            "correction_patterns": len(self.correction_patterns),
            "correction_phrases": len(self.correction_phrases)
        }

    def set_refiner(self, refiner) -> None:
        """Set the refiner for LLM-based merging."""
        self._refiner = refiner