"""
Core prompt refinement logic for image generation.
Refines user prompts to be more specific and detailed for image generation models.
Preserves ALL user-specific details while adding professional enhancements.
Ensures the refined prompt is clean and does NOT contain the LLM's thinking process.
"""
import logging
from typing import Dict, Any, Optional, List

from core.config import Config
from prompt_engineering.base.base_refiner import BaseRefiner
from prompt_engineering.detectors.style_detector import StyleDetector
from prompt_engineering.templates.template_manager import TemplateManager
from core.engines.analysis.text_engine import TextEngine
from core.managers.user_data_manager import UserDataManager

logger = logging.getLogger(__name__)


class PromptRefiner(BaseRefiner):
    """
    Refines prompts for image generation while preserving all user-specific details.

    Features:
    - Preserves ALL user-specific details (NEVER changes what the user said)
    - ONLY adds detail where the prompt is vague
    - Detects and preserves style preferences
    - NO filtering or censorship
    - Multiple refinement levels (basic, advanced, LLM-based)
    - Post-processing to strip the LLM's thinking process
    """

    def __init__(self, user_data_manager: Optional[UserDataManager] = None,
                 text_engine: Optional[TextEngine] = None):
        super().__init__()
        self.user_data_manager = user_data_manager
        self.text_engine = text_engine
        self.style_detector = StyleDetector()
        self.template_manager = TemplateManager()
        self._cache = {}
        self._refinement_level = "advanced"  # basic, advanced, llm

        # Phrases that indicate the LLM output is its reasoning, not the prompt
        self.thinking_phrases = [
            "thinking process", "analyze user input", "identify what needs keeping",
            "rules summary", "core request", "constraints", "output only",
            "here's a thinking process", "let me think", "step 1", "step 2",
            "1.", "2.", "3.", "analyze", "refine", "preserve", "enhance"
        ]

        logger.info("🔧 PromptRefiner initialized")

    async def refine(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Refine a prompt for image generation.

        Args:
            prompt: The raw user prompt
            context: Optional context including user_id, history, etc.

        Returns:
            Refined prompt string
        """
        if not prompt or not prompt.strip():
            return "a beautiful scene"

        # Step 1: Check if the prompt needs refinement at all
        if not self._needs_refinement(prompt):
            logger.info(f"Prompt is already detailed enough: {prompt[:50]}...")
            return prompt

        # Step 2: Detect style from the prompt
        style_info = await self.style_detector.detect(prompt, context)
        detected_style = style_info.get("style", "no_style")
        logger.info(f"🎨 Detected style: {detected_style}")

        # Step 3: Apply style-specific template
        template = self.template_manager.get_template_for_style(detected_style)
        if template:
            prompt = f"{prompt} {template}"

        # Step 4: Use LLM for intelligent refinement if available
        if self.text_engine and self.text_engine.is_initialized:
            refined = await self._refine_with_llm(prompt, context)
            if refined:
                # Post-process: strip thinking process if present
                cleaned = self._clean_refined_prompt(refined)
                if cleaned and cleaned != refined:
                    logger.info(f"🧹 Cleaned thinking process from refined prompt")
                return cleaned if cleaned else prompt

        # Step 5: Fallback to basic refinement
        return self._refine_basic(prompt, style_info)

    def _needs_refinement(self, prompt: str) -> bool:
        """
        Determine if the prompt needs refinement.
        Returns False if the prompt is already detailed enough.
        """
        prompt_lower = prompt.lower()
        words = prompt.split()

        # Very short prompts need refinement
        if len(words) < 5:
            return True

        # If the prompt has specific details, it might not need refinement
        specific_indicators = [
            "with", "in", "on", "at", "style", "background", "lighting",
            "color", "mood", "composition", "angle", "perspective", "close up",
            "wide shot", "portrait", "landscape", "detailed", "intricate",
            "specific", "exact", "precise", "particular"
        ]

        # Count specific details
        specific_count = sum(1 for indicator in specific_indicators if indicator in prompt_lower)

        # If there are multiple specific details, it's probably detailed enough
        if specific_count >= 3:
            return False

        # If the prompt contains a complex description (many words, comma-separated)
        if len(words) > 15 and ',' in prompt:
            return False

        return True

    async def _refine_with_llm(self, prompt: str, context: Optional[Dict]) -> Optional[str]:
        """Refine the prompt using the LLM with a strict instruction to output only the refined prompt."""
        # Strict instruction to force the LLM to output ONLY the refined prompt
        refinement_prompt = f"""You are an expert prompt engineer for image generation.

The user has provided a prompt: "{prompt}"

Your task is to refine this prompt to make it more specific and detailed for an image generation model.

CRITICAL RULES:
1. NEVER change the core request. Keep all specific details the user mentioned EXACTLY as stated.
   - If the user said "a woman", keep "a woman"
   - If they said "sea", keep "sea" - don't change it to "ocean"
   - If they said "realistic", keep "realistic"
2. ONLY add detail where the prompt is vague. If the user was specific, keep it.
3. DO NOT filter, censor, or remove ANYTHING. The user has complete freedom.
4. Add reasonable enhancements for clarity: describe mood, lighting, composition, quality.
5. Keep the refined prompt clear, concise, and directly usable for image generation.

IMPORTANT: Output ONLY the refined prompt text. NO explanation, NO thinking process, NO markdown, NO quotes around it.
The refined prompt must be a single, clear sentence or paragraph starting with the original core request.
Do NOT include any phrases like "Here's a thinking process", "Analyze user input", "Step 1", etc.
Just output the refined prompt text and nothing else."""

        try:
            response, _, _ = await self.text_engine.process(
                refinement_prompt,
                context={'skip_cache': True}
            )

            # Clean up the response
            refined = response.strip()
            refined = refined.strip('"').strip("'")

            # Remove any markdown artifacts
            refined = refined.replace('```', '').strip()

            # If the response is empty or too short, return None
            if not refined or len(refined) < 5:
                logger.warning("LLM returned empty or too short response")
                return None

            logger.info(f"✅ LLM refinement successful: {refined[:50]}...")
            return refined

        except Exception as e:
            logger.error(f"❌ LLM refinement failed: {e}")
            return None

    def _clean_refined_prompt(self, refined: str) -> str:
        """
        Post-process the refined prompt to remove any thinking process content.
        If the prompt looks like a thinking process, try to extract the actual prompt.
        """
        # Check if it contains thinking process phrases
        lower = refined.lower()
        contains_thinking = any(phrase in lower for phrase in self.thinking_phrases)

        if not contains_thinking:
            return refined

        logger.warning("Refined prompt contains thinking process phrases, attempting to clean")

        # Try to find the actual prompt: look for sentences after common markers
        # Often the thinking process ends with "Refined prompt:" or similar
        markers = [
            "refined prompt:", "final prompt:", "output:", "here is the refined prompt:",
            "refined: ", "prompt: ", "final refined prompt:"
        ]
        for marker in markers:
            if marker in refined.lower():
                parts = refined.lower().split(marker, 1)
                if len(parts) > 1:
                    extracted = parts[1].strip()
                    if extracted and len(extracted) > 10:
                        logger.info(f"Extracted prompt after marker '{marker}': {extracted[:50]}...")
                        return extracted

        # If no marker, try to find the first sentence that looks like a prompt
        # Look for the first sentence that is not an instruction
        lines = refined.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip lines that are clearly part of the thinking process
            if any(phrase in line.lower() for phrase in ["step", "analyze", "identify", "constraints", "rules"]):
                continue
            # If the line is long enough and doesn't contain thinking indicators, use it
            if len(line) > 20:
                return line

        # If all else fails, return the original refined (maybe it wasn't that bad)
        return refined

    def _refine_basic(self, prompt: str, style_info: Dict[str, Any]) -> str:
        """
        Basic refinement without LLM.
        Adds quality modifiers and style-specific enhancements.
        """
        refined = prompt

        # Add quality modifiers if the prompt is short
        if len(prompt.split()) < 10:
            quality_modifiers = [
                "high quality", "detailed", "professional",
                "masterpiece", "intricate details", "sharp focus"
            ]
            refined += f", {', '.join(quality_modifiers[:2])}"

        # Add style-specific modifiers
        style = style_info.get("style", "no_style")
        style_modifiers = {
            "realistic": "photorealistic, natural lighting, shallow depth of field",
            "anime": "anime style, cel shading, vibrant colors, studio ghibli inspired",
            "oil_painting": "oil painting, textured brushstrokes, rich colors, canvas texture",
            "watercolor": "watercolor painting, soft washes, flowing colors",
            "sketch": "pencil sketch, drawn, shading, line art",
            "pixel": "pixel art, 8-bit style, retro game aesthetic",
            "cyberpunk": "cyberpunk, neon lights, dark atmosphere, futuristic",
            "fantasy": "fantasy, magical, enchanted, epic, mythical",
            "dark": "dark, moody, atmospheric, gothic",
            "cinematic": "cinematic, dramatic lighting, film grain, epic composition",
            "surreal": "surreal, dreamlike, impossible, Dali inspired",
            "3d": "3d render, CGI, ray tracing, realistic materials",
            "vector": "vector art, flat design, clean lines, minimalist",
        }

        if style in style_modifiers and style != "no_style":
            refined += f", {style_modifiers[style]}"
        elif style == "no_style" and len(prompt.split()) < 8:
            # Generic enhancement for short prompts
            refined += ", high quality, well-composed, beautiful lighting"

        return refined

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("🔧 PromptRefiner cache cleared")

    def get_info(self) -> Dict[str, Any]:
        """Return information about the refiner."""
        return {
            "name": self.name,
            "type": "PromptRefiner",
            "supports_llm_refinement": self.text_engine is not None,
            "style_support": len(Config.STYLE_KEYWORDS),
            "refinement_level": self._refinement_level
        }