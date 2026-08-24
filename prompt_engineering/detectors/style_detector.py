"""
Style detection engine that identifies artistic styles from user prompts.
Supports automatic mapping to appropriate image generation models.
"""
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from core.config import Config
from prompt_engineering.base.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class StyleDetector(BaseDetector):
    """
    Detects artistic styles from user prompts using multi-layer analysis.

    Style detection is performed through:
    1. Keyword matching (from Config.STYLE_KEYWORDS)
    2. Compound phrase detection (e.g., "anime style", "like a painting")
    3. Style confidence scoring
    4. Conflict resolution (if multiple styles detected)
    """

    def __init__(self):
        super().__init__()
        self.style_keywords = Config.STYLE_KEYWORDS
        self.style_model_map = Config.STYLE_MODEL_MAP
        self._style_aliases = self._build_style_aliases()
        self._compound_style_patterns = self._build_compound_patterns()
        logger.info("🎨 StyleDetector initialized")

    def _build_style_aliases(self) -> Dict[str, str]:
        """Build aliases for style variations (e.g., 'photo' → 'realistic')."""
        aliases = {}
        for style, keywords in self.style_keywords.items():
            for kw in keywords:
                # Map variations to the main style
                aliases[kw] = style
                aliases[kw + " style"] = style
                aliases["in " + kw + " style"] = style
                aliases[kw + " look"] = style
                aliases[kw + " aesthetic"] = style
        return aliases

    def _build_compound_patterns(self) -> List[Tuple[str, str]]:
        """Build regex patterns for detecting compound style phrases."""
        patterns = []

        # "like a [style] painting"
        for style in self.style_keywords.keys():
            if style != "no_style" and self.style_keywords.get(style):
                # This would be too many patterns; we'll use dynamic detection
                pass

        # Common compound phrases
        compounds = [
            (r"like\s+a\s+(\w+)\s+(painting|drawing|sketch|photo|picture|render|art|illustration)", "painting"),
            (r"in\s+the\s+style\s+of\s+(\w+)", "style_of"),
            (r"(\w+)\s+style\s+(\w+)", "style_with_subject"),
            (r"digital\s+(\w+)\s+art", "digital_art"),
            (r"(\w+)\s+aesthetic", "aesthetic"),
        ]

        return patterns

    async def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect the primary artistic style from the user's prompt.

        Returns:
            {
                "style": str,                    # Detected style (e.g., "anime", "realistic")
                "confidence": float,             # Confidence score (0.0 - 1.0)
                "all_styles": dict,              # All detected styles with confidence
                "primary_style": str,            # Same as "style"
                "secondary_styles": list,        # Other styles detected
                "style_keywords": list,          # Keywords that triggered the detection
                "recommended_models": list       # Models recommended for this style
            }
        """
        if not text or not text.strip():
            return {
                "style": "no_style",
                "confidence": 0.0,
                "all_styles": {},
                "primary_style": "no_style",
                "secondary_styles": [],
                "style_keywords": [],
                "recommended_models": Config.STYLE_MODEL_MAP.get("no_style", [])
            }

        text_lower = text.lower()
        detected_styles = defaultdict(float)
        style_keywords_found = defaultdict(list)

        # Layer 1: Direct keyword matching
        for style, keywords in self.style_keywords.items():
            if style == "no_style":
                continue
            for kw in keywords:
                if kw in text_lower:
                    detected_styles[style] += 0.8
                    style_keywords_found[style].append(kw)

        # Layer 2: Check for compound phrases using regex
        compound_matches = self._detect_compound_phrases(text_lower)
        for style, confidence in compound_matches:
            detected_styles[style] += confidence * 1.2  # Bonus for compound phrases
            style_keywords_found[style].append("compound_phrase")

        # Layer 3: Check for style aliases
        for phrase, style in self._style_aliases.items():
            if phrase in text_lower:
                detected_styles[style] += 0.6
                style_keywords_found[style].append(phrase)

        # Layer 4: Context-based detection (if conversation history is available)
        if context and context.get('history'):
            history_style = await self._analyze_context_history(context['history'])
            if history_style:
                detected_styles[history_style] += 0.3

        # Layer 5: Image generation specific indicators
        image_indicators = self._detect_image_indicators(text_lower)
        for style, confidence in image_indicators:
            detected_styles[style] += confidence

        # Normalize confidence scores (cap at 1.0)
        for style in detected_styles:
            detected_styles[style] = min(detected_styles[style], 1.0)

        # Filter out styles with very low confidence
        filtered_styles = {
            style: confidence for style, confidence in detected_styles.items()
            if confidence >= 0.2
        }

        # If no styles detected, return no_style
        if not filtered_styles:
            return {
                "style": "no_style",
                "confidence": 0.0,
                "all_styles": {},
                "primary_style": "no_style",
                "secondary_styles": [],
                "style_keywords": [],
                "recommended_models": Config.STYLE_MODEL_MAP.get("no_style", [])
            }

        # Sort by confidence (highest first)
        sorted_styles = sorted(filtered_styles.items(), key=lambda x: x[1], reverse=True)
        primary_style = sorted_styles[0][0]
        primary_confidence = sorted_styles[0][1]
        secondary_styles = [style for style, _ in sorted_styles[1:]]

        # Get recommended models for the primary style
        recommended_models = self._get_recommended_models(primary_style)

        return {
            "style": primary_style,
            "confidence": primary_confidence,
            "all_styles": dict(sorted_styles),
            "primary_style": primary_style,
            "secondary_styles": secondary_styles,
            "style_keywords": style_keywords_found.get(primary_style, []),
            "recommended_models": recommended_models
        }

    def _detect_compound_phrases(self, text: str) -> List[Tuple[str, float]]:
        """Detect compound phrases that indicate a style."""
        results = []

        # Check for "like a [something] painting/drawing/etc"
        pattern = r"(?:like|as|similar to)\s+a\s+(\w+)\s+(?:painting|drawing|sketch|photo|picture|render|art|illustration)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            style = self._map_style_alias(match)
            if style:
                results.append((style, 0.7))

        # Check for "[style] style"
        pattern = r"(\w+)\s+style"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            style = self._map_style_alias(match)
            if style:
                results.append((style, 0.6))

        # Check for "in the style of [artist/style]"
        pattern = r"in\s+the\s+style\s+of\s+(\w+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            style = self._map_style_alias(match)
            if style:
                results.append((style, 0.8))

        return results

    def _map_style_alias(self, word: str) -> Optional[str]:
        """Map a word or phrase to a known style."""
        word_lower = word.lower()

        # Direct mapping from config style keywords
        for style, keywords in self.style_keywords.items():
            if style == "no_style":
                continue
            for kw in keywords:
                if word_lower == kw or word_lower in kw:
                    return style

        # Common word-to-style mappings
        word_map = {
            "anime": "anime",
            "manga": "anime",
            "cartoon": "anime",
            "real": "realistic",
            "photorealistic": "realistic",
            "photo": "realistic",
            "pixel": "pixel",
            "8bit": "pixel",
            "painting": "oil_painting",
            "oil": "oil_painting",
            "watercolor": "watercolor",
            "sketch": "sketch",
            "pencil": "sketch",
            "3d": "3d",
            "render": "3d",
            "cgi": "3d",
            "cyberpunk": "cyberpunk",
            "neon": "cyberpunk",
            "fantasy": "fantasy",
            "magical": "fantasy",
            "minimalist": "minimalist",
            "abstract": "abstract",
            "vintage": "vintage",
            "retro": "vintage",
            "gothic": "dark",
            "dark": "dark",
            "cinematic": "cinematic",
            "surreal": "surreal",
            "popart": "pop_art",
            "lowpoly": "low_poly",
            "vector": "vector",
        }

        return word_map.get(word_lower)

    async def _analyze_context_history(self, history: List[Dict]) -> Optional[str]:
        """Analyze the conversation history for style consistency."""
        style_counts = defaultdict(int)

        for entry in history[-5:]:  # Only check last 5 messages
            if entry.get('type') == 'generated_image':
                # If the user previously generated an image, check its style
                style = entry.get('model_used', '').split(':')[-1]
                if style:
                    style_counts[style] += 1

        if style_counts:
            return max(style_counts.items(), key=lambda x: x[1])[0]

        return None

    def _detect_image_indicators(self, text: str) -> List[Tuple[str, float]]:
        """Detect style indicators from image-related phrases."""
        results = []

        # Common phrases that indicate specific styles
        indicators = {
            "photorealistic": ("realistic", 0.9),
            "hyperrealistic": ("realistic", 0.9),
            "like a photo": ("realistic", 0.7),
            "real life": ("realistic", 0.8),
            "anime style": ("anime", 0.9),
            "cartoon style": ("anime", 0.7),
            "manga style": ("anime", 0.8),
            "pixelated": ("pixel", 0.8),
            "retro pixel": ("pixel", 0.7),
            "oil on canvas": ("oil_painting", 0.9),
            "watercolor painting": ("watercolor", 0.9),
            "pencil sketch": ("sketch", 0.9),
            "3d render": ("3d", 0.9),
        }

        for phrase, (style, confidence) in indicators.items():
            if phrase in text:
                results.append((style, confidence))

        return results

    def _get_recommended_models(self, style: str) -> List[str]:
        """Get recommended models for a given style."""
        # First try to get from config
        models = Config.STYLE_MODEL_MAP.get(style, [])

        # If no models in config, use generic
        if not models:
            models = ["black-forest-labs/flux.2-pro"]

        return models

    def get_info(self) -> Dict[str, Any]:
        """Return information about the detector."""
        return {
            "name": self.name,
            "type": "StyleDetector",
            "supported_styles": list(self.style_keywords.keys()),
            "style_count": len(self.style_keywords),
            "alias_count": len(self._style_aliases)
        }