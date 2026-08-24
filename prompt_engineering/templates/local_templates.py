"""
Hardcoded fallback templates for prompt engineering.
Used when GitHub is unavailable.
Contains comprehensive templates for various styles and compositions.
"""
import logging
from typing import Dict, Any, List, Optional
import random

logger = logging.getLogger(__name__)


class LocalTemplates:
    """
    Local fallback templates for image generation prompts.
    """

    def __init__(self):
        self._templates = self._build_templates()
        logger.info(f"📋 LocalTemplates initialized with {len(self._templates.get('styles', {}))} styles")

    def _build_templates(self) -> Dict[str, Any]:
        """Build all templates."""
        return {
            "styles": self._build_style_templates(),
            "templates": self._build_general_templates(),
            "quality": self._build_quality_templates(),
            "composition": self._build_composition_templates(),
            "mood": self._build_mood_templates()
        }

    def _build_style_templates(self) -> Dict[str, List[str]]:
        """Style-specific templates."""
        return {
            "realistic": [
                "photorealistic, natural lighting, shallow depth of field, 8k resolution, highly detailed",
                "hyperrealistic, detailed, sharp focus, professional photography, Canon EOS R5, 85mm lens",
                "realistic, cinematic lighting, high detail, lifelike textures, Unreal Engine 5 render",
                "photorealistic, natural colors, soft shadows, high definition, 4k, masterpiece"
            ],
            "anime": [
                "anime style, cel shading, vibrant colors, studio ghibli inspired, detailed background",
                "manga style, clean lines, expressive characters, dramatic poses, color anime aesthetic",
                "anime aesthetic, colorful, detailed background, character focus, Japanese animation",
                "japanese animation style, dynamic composition, saturated colors, soft lighting"
            ],
            "oil_painting": [
                "oil painting, textured brushstrokes, rich colors, canvas texture, baroque style",
                "masterpiece oil painting, impasto technique, renaissance style, high detail, dramatic",
                "oil on canvas, classical painting, detailed, museum quality, Rembrandt lighting",
                "oil painting, dramatic lighting, thick brushwork, old masters style, vintage"
            ],
            "watercolor": [
                "watercolor painting, soft washes, flowing colors, artistic, paper texture",
                "watercolor art, delicate, transparent washes, paper texture, ethereal, dreamy",
                "aquarelle style, soft edges, color bleeding, artistic, creative",
                "watercolor, soft, muted colors, paper texture, gentle, atmospheric"
            ],
            "sketch": [
                "pencil sketch, drawn, shading, line art, detailed drawing, high quality",
                "charcoal sketch, textured, monochrome, artistic shading, dramatic",
                "hand-drawn sketch, precise lines, cross-hatching, detailed, fine art",
                "pencil drawing, realistic shading, fine art, detailed, professional"
            ],
            "pixel": [
                "pixel art, 8-bit style, retro game aesthetic, blocky, colorful, nostalgic",
                "pixel art, 16-bit style, colorful, nostalgic, game sprite, detailed",
                "retro pixel art, game graphics, vibrant, chunky pixels, SNES style",
                "pixel art, nostalgic, colorful, detailed, 1990s video game"
            ],
            "cyberpunk": [
                "cyberpunk, neon lights, dark atmosphere, futuristic city, rainy, high-tech",
                "cyberpunk aesthetic, glowing neon, rain, high-tech low-life, dystopian",
                "futuristic cyberpunk, holographic, chrome, dark moody, sci-fi",
                "cyberpunk, neon, dark, futuristic, city, rain, reflection, detailed"
            ],
            "fantasy": [
                "fantasy, magical, enchanted, epic, mythical, high fantasy, detailed, vibrant",
                "fantasy art, dragons, magic, mystical landscape, detailed, epic, beautiful",
                "epic fantasy, legendary, otherworldly, grand scale, majestic, stunning",
                "fantasy, magical, elves, dragons, castles, enchanted forest, vibrant"
            ],
            "dark": [
                "dark, moody, atmospheric, gothic, brooding, intense, dramatic",
                "dark aesthetic, shadows, intense, dramatic, haunting, eerie",
                "dark, gothic, moody, atmospheric, mysterious, shadowy, cinematic"
            ],
            "cinematic": [
                "cinematic, dramatic lighting, film grain, epic composition, movie scene",
                "movie scene, cinematic shot, wide angle, emotional, dramatic, epic",
                "cinematic style, dramatic, high contrast, professional, Hollywood film",
                "cinematic, epic, emotional, dramatic, movie poster, wide shot"
            ],
            "surreal": [
                "surreal, dreamlike, impossible, Dali inspired, symbolic, visionary",
                "surrealist art, dreamy, melting, impossible geometry, fantasy, mind-bending",
                "surreal, dream, unreal, psychedelic, creative, abstract, symbolic"
            ],
            "3d": [
                "3d render, CGI, ray tracing, realistic materials, high poly, detailed",
                "3d art, blender render, detailed textures, realistic lighting, animated",
                "3d modeling, CGI art, photorealistic, rendered, modern, high quality",
                "3d, realistic, high-quality, render, cinema 4d, advanced"
            ],
            "vector": [
                "vector art, flat design, clean lines, minimalist, modern, colorful",
                "vector illustration, crisp, geometric, bold colors, simplified, graphic",
                "vector art, flat, minimal, clean, vibrant, professional, graphical"
            ],
            "minimalist": [
                "minimalist, simple, clean, elegant, minimal detail, modern",
                "minimalist art, negative space, simple composition, modern, refined"
            ],
            "abstract": [
                "abstract art, geometric, colorful, modern, non-representational, dynamic",
                "abstract painting, shapes, colors, emotional, freeform, expressive"
            ],
            "no_style": [
                "high quality, detailed, professional, masterpiece, stunning",
                "beautiful, well-composed, vibrant, stunning, impressive",
                "intricate details, sharp focus, rich colors, expert craft, breathtaking"
            ]
        }

    def _build_general_templates(self) -> Dict[str, str]:
        """General named templates."""
        return {
            "epic_landscape": "epic landscape, majestic, breathtaking, wide angle, dramatic lighting, natural beauty, stunning scenery",
            "portrait_closeup": "close-up portrait, facial expression, emotion, shallow depth of field, soft lighting, detailed eyes",
            "action_shot": "action shot, dynamic movement, motion blur, energy, excitement, powerful, dramatic",
            "macro_detail": "extreme close-up, fine details, macro, texture, intimacy, focus, sharp",
            "wide_shot": "wide shot, expansive, panoramic, vast, sweeping, landscape, scenic"
        }

    def _build_quality_templates(self) -> List[str]:
        """Quality enhancement templates."""
        return [
            "8k resolution, highly detailed, sharp focus, professional quality",
            "4k, ultra HD, crisp, clear, professional quality, masterpiece",
            "masterpiece, award-winning, exceptional quality, museum-grade, stunning",
            "high-end, premium, exquisite, refined, polished, world-class",
            "world-class, extraordinary, breathtaking, magnificent, stunning",
            "high quality, detailed, beautiful, vibrant, professional, 4k"
        ]

    def _build_composition_templates(self) -> Dict[str, List[str]]:
        """Composition-specific templates."""
        return {
            "portrait": [
                "close-up portrait, facial expression visible, shallow depth of field, bokeh background",
                "portrait photography, centered composition, soft lighting, bokeh background, eye contact",
                "headshot, professional, clear eyes, natural expression, high quality"
            ],
            "landscape": [
                "landscape, wide-angle, dramatic sky, distant horizon, layers, depth",
                "scenic view, sweeping landscape, natural beauty, depth, epic",
                "panoramic, vast, atmospheric, stunning scenery, professional photography"
            ],
            "action": [
                "action shot, dynamic movement, motion blur, energy, excitement, powerful",
                "frozen action, peak moment, dramatic, powerful, intense",
                "dynamic pose, action scene, tension, high impact, dramatic"
            ],
            "closeup": [
                "extreme close-up, fine details, macro, texture, intimacy, focus",
                "close-up detail, visible texture, focused, immersive, sharp"
            ]
        }

    def _build_mood_templates(self) -> Dict[str, List[str]]:
        """Mood-specific templates."""
        return {
            "bright": [
                "bright, sunny, warm colors, cheerful, uplifting, vibrant, energetic",
                "bright lighting, joyful atmosphere, colorful, energetic, happy"
            ],
            "dark": [
                "dark, moody, shadows, intense, dramatic, brooding atmosphere, mysterious",
                "low-key lighting, chiaroscuro, mysterious, atmospheric, dark"
            ],
            "warm": [
                "warm tones, golden hour, cozy, inviting, comfortable, intimate",
                "warm lighting, sunset colors, friendly atmosphere, comforting"
            ],
            "cool": [
                "cool tones, blue hues, serene, calm, peaceful, refreshing, tranquil",
                "cool lighting, crisp, clear, tranquil, serene"
            ],
            "romantic": [
                "romantic atmosphere, soft lighting, dreamy, emotional, intimate, warm",
                "romantic scene, gentle colors, soft focus, warm, loving"
            ],
            "dramatic": [
                "dramatic, intense, high contrast, powerful, striking, bold",
                "dramatic lighting, bold, impactful, stunning, emotional"
            ]
        }

    def get_all_templates(self) -> Dict[str, Any]:
        """Return all templates."""
        return self._templates

    def get_style_templates(self, style: str) -> List[str]:
        """Get style-specific templates."""
        return self._templates.get('styles', {}).get(style, [])

    def get_quality_template(self) -> str:
        """Get a random quality template."""
        templates = self._templates.get('quality', [])
        if templates:
            return random.choice(templates)
        return "high quality, detailed, professional"

    def get_composition_template(self, composition_type: str) -> Optional[str]:
        """Get a composition template."""
        templates = self._templates.get('composition', {}).get(composition_type, [])
        if templates:
            return random.choice(templates)
        return None

    def get_info(self) -> Dict[str, Any]:
        """Return information about the templates."""
        return {
            "type": "LocalTemplates",
            "styles": len(self._templates.get('styles', {})),
            "total_templates": len(self._templates.get('templates', {})),
            "quality_templates": len(self._templates.get('quality', [])),
            "composition_types": len(self._templates.get('composition', {}))
        }