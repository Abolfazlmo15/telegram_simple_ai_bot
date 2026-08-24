"""
Applies professional templates to prompts based on style and content type.
"""
import logging
import random
from typing import Dict, Any, Optional, List, Tuple
from prompt_engineering.base.base_refiner import BaseRefiner
from core.config import Config

logger = logging.getLogger(__name__)


class TemplateApplier(BaseRefiner):
    """
    Applies professional templates to prompts for better image generation results.

    Features:
    - Style-specific templates
    - Composition templates (portrait, landscape, action, etc.)
    - Quality enhancement templates
    - Mood and lighting templates
    """

    def __init__(self):
        super().__init__()
        self._init_templates()
        logger.info("📋 TemplateApplier initialized")

    def _init_templates(self):
        """Initialize all templates."""

        # Style-specific templates
        self.style_templates = {
            "realistic": [
                "photorealistic, natural lighting, shallow depth of field, 8k resolution",
                "hyperrealistic, detailed, sharp focus, professional photography",
                "realistic, cinematic lighting, high detail, lifelike textures",
                "photorealistic, natural colors, soft shadows, high definition"
            ],
            "anime": [
                "anime style, cel shading, vibrant colors, studio ghibli inspired",
                "manga style, clean lines, expressive characters, dramatic poses",
                "anime aesthetic, colorful, detailed background, character focus",
                "japanese animation style, dynamic composition, saturated colors"
            ],
            "oil_painting": [
                "oil painting, textured brushstrokes, rich colors, canvas texture",
                "masterpiece oil painting, impasto technique, renaissance style",
                "oil on canvas, classical painting, detailed, museum quality",
                "oil painting, dramatic lighting, thick brushwork, old masters style"
            ],
            "watercolor": [
                "watercolor painting, soft washes, flowing colors, artistic",
                "watercolor art, delicate, transparent washes, paper texture",
                "aquarelle style, soft edges, color bleeding, artistic"
            ],
            "sketch": [
                "pencil sketch, drawn, shading, line art, detailed drawing",
                "charcoal sketch, textured, monochrome, artistic shading",
                "hand-drawn sketch, precise lines, cross-hatching, detailed",
                "pencil drawing, realistic shading, fine art"
            ],
            "pixel": [
                "pixel art, 8-bit style, retro game aesthetic, blocky",
                "pixel art, 16-bit style, colorful, nostalgic, game sprite",
                "retro pixel art, game graphics, vibrant, chunky pixels"
            ],
            "cyberpunk": [
                "cyberpunk, neon lights, dark atmosphere, futuristic city",
                "cyberpunk aesthetic, glowing neon, rain, high-tech low-life",
                "futuristic cyberpunk, holographic, chrome, dark moody"
            ],
            "fantasy": [
                "fantasy, magical, enchanted, epic, mythical, high fantasy",
                "fantasy art, dragons, magic, mystical landscape, detailed",
                "epic fantasy, legendary, otherworldly, grand scale"
            ],
            "dark": [
                "dark, moody, atmospheric, gothic, brooding",
                "dark aesthetic, shadows, intense, dramatic, haunting"
            ],
            "cinematic": [
                "cinematic, dramatic lighting, film grain, epic composition",
                "movie scene, cinematic shot, wide angle, emotional",
                "cinematic style, dramatic, high contrast, professional"
            ],
            "surreal": [
                "surreal, dreamlike, impossible, Dali inspired, symbolic",
                "surrealist art, dreamy, melting, impossible geometry, fantasy"
            ],
            "3d": [
                "3d render, CGI, ray tracing, realistic materials, high poly",
                "3d art, blender render, detailed textures, realistic lighting",
                "3d modeling, CGI art, photorealistic, rendered"
            ],
            "vector": [
                "vector art, flat design, clean lines, minimalist, modern",
                "vector illustration, crisp, geometric, bold colors, simplified"
            ],
            "minimalist": [
                "minimalist, simple, clean, elegant, minimal detail",
                "minimalist art, negative space, simple composition, modern"
            ],
            "abstract": [
                "abstract art, geometric, colorful, modern, non-representational",
                "abstract painting, shapes, colors, emotional, freeform"
            ],
            "no_style": [
                "high quality, detailed, professional, masterpiece",
                "beautiful, well-composed, vibrant, stunning",
                "intricate details, sharp focus, rich colors, expert craft"
            ]
        }

        # Composition templates
        self.composition_templates = {
            "portrait": [
                "close-up portrait, facial expression visible, shallow depth of field",
                "portrait photography, centered composition, soft lighting, bokeh background",
                "headshot, professional, clear eyes, natural expression"
            ],
            "landscape": [
                "landscape, wide-angle, dramatic sky, distant horizon, layers",
                "scenic view, sweeping landscape, natural beauty, depth",
                "panoramic, vast, atmospheric, stunning scenery"
            ],
            "action": [
                "action shot, dynamic movement, motion blur, energy, excitement",
                "frozen action, peak moment, dramatic, powerful",
                "dynamic pose, action scene, tension, high impact"
            ],
            "closeup": [
                "extreme close-up, fine details, macro, texture, intimacy",
                "close-up detail, visible texture, focused, immersive"
            ]
        }

        # Mood templates
        self.mood_templates = {
            "bright": [
                "bright, sunny, warm colors, cheerful, uplifting, vibrant",
                "bright lighting, joyful atmosphere, colorful, energetic"
            ],
            "dark": [
                "dark, moody, shadows, intense, dramatic, brooding atmosphere",
                "low-key lighting, chiaroscuro, mysterious, atmospheric"
            ],
            "warm": [
                "warm tones, golden hour, cozy, inviting, comfortable",
                "warm lighting, sunset colors, friendly atmosphere"
            ],
            "cool": [
                "cool tones, blue hues, serene, calm, peaceful, refreshing",
                "cool lighting, crisp, clear, tranquil"
            ],
            "romantic": [
                "romantic atmosphere, soft lighting, dreamy, emotional, intimate",
                "romantic scene, gentle colors, soft focus, warm"
            ],
            "dramatic": [
                "dramatic, intense, high contrast, powerful, striking",
                "dramatic lighting, bold, impactful, stunning"
            ]
        }

        # Quality templates
        self.quality_templates = [
            "8k resolution, highly detailed, sharp focus",
            "4k, ultra HD, crisp, clear, professional quality",
            "masterpiece, award-winning, exceptional quality, museum-grade",
            "high-end, premium, exquisite, refined, polished",
            "world-class, extraordinary, breathtaking, magnificent"
        ]

    async def refine(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply templates to the prompt.
        """
        if not text or not text.strip():
            return text

        prompt = text

        # Step 1: Detect style from context or prompt
        style = await self._detect_style(prompt, context)

        # Step 2: Apply style-specific template
        style_template = self._get_style_template(style)
        if style_template:
            prompt = f"{prompt}, {style_template}"

        # Step 3: Apply quality template (if the prompt is short)
        if len(prompt.split()) < 15:
            quality_template = self._get_quality_template()
            prompt = f"{prompt}, {quality_template}"

        # Step 4: Apply composition templates based on content
        composition = self._detect_composition(prompt)
        if composition:
            comp_template = self._get_composition_template(composition)
            if comp_template:
                prompt = f"{prompt}, {comp_template}"

        return prompt

    async def _detect_style(self, text: str, context: Optional[Dict]) -> str:
        """Detect the style from the prompt."""
        text_lower = text.lower()

        # Check for style keywords
        for style, keywords in Config.STYLE_KEYWORDS.items():
            if style == "no_style":
                continue
            for kw in keywords:
                if kw in text_lower:
                    return style

        # Check if style is in context
        if context and context.get('style'):
            return context['style']

        return "no_style"

    def _get_style_template(self, style: str) -> Optional[str]:
        """Get a random template for the specified style."""
        templates = self.style_templates.get(style, self.style_templates.get("no_style", []))
        if templates:
            return random.choice(templates)
        return None

    def _get_quality_template(self) -> str:
        """Get a random quality template."""
        return random.choice(self.quality_templates)

    def _detect_composition(self, text: str) -> Optional[str]:
        """Detect the composition type from the prompt."""
        text_lower = text.lower()

        # Check for composition indicators
        indicators = {
            "portrait": ["portrait", "face", "headshot", "close-up", "closeup"],
            "landscape": ["landscape", "scenery", "view", "panoramic", "wide"],
            "action": ["action", "moving", "running", "fighting", "jumping", "dynamic"],
            "closeup": ["macro", "extreme close-up", "detailed", "texture"]
        }

        for comp, keywords in indicators.items():
            if any(kw in text_lower for kw in keywords):
                return comp

        return None

    def _get_composition_template(self, composition: str) -> Optional[str]:
        """Get a random template for the specified composition."""
        templates = self.composition_templates.get(composition, [])
        if templates:
            return random.choice(templates)
        return None

    def get_info(self) -> Dict[str, Any]:
        """Return information about the applier."""
        return {
            "name": self.name,
            "type": "TemplateApplier",
            "style_templates": len(self.style_templates),
            "composition_templates": len(self.composition_templates),
            "mood_templates": len(self.mood_templates),
            "quality_templates": len(self.quality_templates)
        }