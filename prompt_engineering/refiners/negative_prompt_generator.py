"""
Generates negative prompts based on the user's prompt and style.
Negative prompts tell the image generation model what NOT to include.
"""
import logging
from typing import Dict, Any, Optional, List
from prompt_engineering.base.base_refiner import BaseRefiner
from core.config import Config

logger = logging.getLogger(__name__)


class NegativePromptGenerator(BaseRefiner):
    """
    Generates negative prompts for image generation.

    Negative prompts help image models avoid common issues and undesired elements.
    """

    def __init__(self):
        super().__init__()
        self._init_negative_prompts()
        logger.info("🚫 NegativePromptGenerator initialized")

    def _init_negative_prompts(self):
        """Initialize negative prompt templates."""

        # Base negative prompts (always included)
        self.base_negative = [
            "blurry, low quality, distorted, deformed, ugly",
            "bad anatomy, bad proportions, extra limbs, missing limbs",
            "disfigured, mutilated, malformed, grotesque",
            "bad lighting, overexposed, underexposed, washed out"
        ]

        # Style-specific negative prompts
        self.style_negative = {
            "realistic": [
                "cartoon, anime, stylized, low poly, vector art",
                "unrealistic, artificial, fake, plastic-looking"
            ],
            "anime": [
                "realistic, photorealistic, 3d render, live-action",
                "western comic, western animation"
            ],
            "oil_painting": [
                "digital art, pixel art, low resolution, cartoon",
                "vector art, flat design, modern digital"
            ],
            "watercolor": [
                "oil painting, acrylic, digital art, harsh lines",
                "saturated, oversaturated, too vibrant"
            ],
            "sketch": [
                "digital art, rendered, painted, colored",
                "realistic, photorealistic, 3d"
            ],
            "pixel": [
                "smooth, realistic, anti-aliased, high resolution",
                "3d render, photorealistic, modern"
            ],
            "cyberpunk": [
                "bright, sunny, cheerful, vibrant, colorful",
                "clean, organized, orderly"
            ],
            "fantasy": [
                "sci-fi, modern, realistic, contemporary",
                "urban, industrial, futuristic"
            ]
        }

        # Composition-specific negative prompts
        self.composition_negative = {
            "portrait": [
                "profile view, side view, turned away, back view",
                "too far, distant, full body, extreme wide"
            ],
            "landscape": [
                "close-up, detailed, macro, tight crop",
                "cluttered, busy, disorganized"
            ],
            "action": [
                "static, still, frozen, posed, unnatural",
                "boring, dull, uninteresting"
            ]
        }

    async def refine(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a negative prompt based on the input prompt and context.

        Returns:
            A string containing the negative prompt.
        """
        if not prompt or not prompt.strip():
            return ", ".join(self.base_negative)

        negative_parts = []

        # Step 1: Add base negative prompts
        negative_parts.extend(self.base_negative)

        # Step 2: Add style-specific negatives
        style = await self._detect_style(prompt, context)
        if style in self.style_negative:
            negative_parts.extend(self.style_negative[style])

        # Step 3: Add composition-specific negatives
        composition = self._detect_composition(prompt)
        if composition in self.composition_negative:
            negative_parts.extend(self.composition_negative[composition])

        # Step 4: Add content-specific negatives based on the prompt
        content_negatives = self._generate_content_negatives(prompt)
        negative_parts.extend(content_negatives)

        # Combine all parts, limit to reasonable length
        combined = ", ".join(negative_parts)

        # Truncate if too long (Telegram has message limits)
        if len(combined) > 1000:
            combined = combined[:1000]

        return combined

    async def _detect_style(self, text: str, context: Optional[Dict]) -> str:
        """Detect the style from the prompt."""
        text_lower = text.lower()

        for style, keywords in Config.STYLE_KEYWORDS.items():
            if style == "no_style":
                continue
            for kw in keywords:
                if kw in text_lower:
                    return style

        if context and context.get('style'):
            return context['style']

        return "no_style"

    def _detect_composition(self, text: str) -> Optional[str]:
        """Detect the composition type from the prompt."""
        text_lower = text.lower()

        indicators = {
            "portrait": ["portrait", "face", "headshot", "close-up", "closeup"],
            "landscape": ["landscape", "scenery", "view", "panoramic", "wide"],
            "action": ["action", "moving", "running", "fighting", "jumping"]
        }

        for comp, keywords in indicators.items():
            if any(kw in text_lower for kw in keywords):
                return comp

        return None

    def _generate_content_negatives(self, prompt: str) -> List[str]:
        """Generate content-specific negative prompts."""
        prompt_lower = prompt.lower()
        negatives = []

        # Check for specific content that might cause issues
        if any(word in prompt_lower for word in ["person", "people", "man", "woman", "child"]):
            negatives.append("bad anatomy, missing limbs, extra limbs")

        if any(word in prompt_lower for word in ["hand", "hands", "finger", "fingers"]):
            negatives.append("bad hands, extra fingers, deformed hands")

        if "face" in prompt_lower or "faces" in prompt_lower:
            negatives.append("deformed face, mismatched eyes, bad facial features")

        if "animal" in prompt_lower or "animals" in prompt_lower:
            negatives.append("bad animal anatomy, unnatural poses, deformed")

        return negatives

    def get_info(self) -> Dict[str, Any]:
        """Return information about the generator."""
        return {
            "name": self.name,
            "type": "NegativePromptGenerator",
            "style_negatives": len(self.style_negative),
            "composition_negatives": len(self.composition_negative)
        }