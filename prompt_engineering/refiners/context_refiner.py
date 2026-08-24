"""
Context-aware prompt refinement for image generation.
Uses memory, preferences, and topic information to refine prompts.
"""
import logging
from typing import Dict, Any, Optional, List
from core.config import Config
from prompt_engineering.base.base_refiner import BaseRefiner
from prompt_engineering.refiners.prompt_refiner import PromptRefiner
from core.managers.memory_manager import MemoryManager
from core.managers.topic_manager import TopicManager

logger = logging.getLogger(__name__)


class ContextRefiner(BaseRefiner):
    """
    Refines prompts using context from memory, preferences, and topics.

    Features:
    - Injects relevant conversation history into the prompt
    - Uses user preferences for style and length
    - Applies topic-specific enhancements
    - Preserves all user-specific details
    - Multiple refinement layers with fallback
    """

    def __init__(self, memory_manager: Optional[MemoryManager] = None,
                 topic_manager: Optional[TopicManager] = None,
                 prompt_refiner: Optional[PromptRefiner] = None):
        super().__init__()
        self.memory_manager = memory_manager
        self.topic_manager = topic_manager
        self.prompt_refiner = prompt_refiner
        self._cache = {}
        logger.info("🔧 ContextRefiner initialized")

    async def refine(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Refine a prompt using context from memory, preferences, and topics.

        Args:
            prompt: The raw user prompt
            context: Context including user_id, history, preferences, etc.

        Returns:
            Refined prompt string
        """
        if not prompt or not prompt.strip():
            return "a beautiful scene"

        user_id = context.get('user_id') if context else None
        preferences = context.get('preferences', {}) if context else {}
        history = context.get('history', []) if context else []

        refined_prompt = prompt

        # ============================================================
        # STEP 1: Check if prompt needs refinement
        # ============================================================
        if not self._needs_refinement(prompt):
            logger.info(f"📝 Prompt is already detailed enough: {prompt[:50]}...")
            return prompt

        # ============================================================
        # STEP 2: Inject memory context (if available)
        # ============================================================
        memory_context = await self._get_memory_context(user_id)
        if memory_context:
            refined_prompt = f"{refined_prompt}\n\nContext from our conversation: {memory_context}"
            logger.info(f"📖 Added memory context: {memory_context[:100]}...")

        # ============================================================
        # STEP 3: Apply user preferences
        # ============================================================
        refined_prompt = await self._apply_preferences(refined_prompt, preferences)

        # ============================================================
        # STEP 4: Apply topic-specific enhancements
        # ============================================================
        current_topic = await self._get_current_topic(user_id)
        if current_topic:
            refined_prompt = await self._apply_topic_enhancement(refined_prompt, current_topic)
            logger.info(f"📊 Applied topic enhancement: {current_topic}")

        # ============================================================
        # STEP 5: Apply quality enhancements for short prompts
        # ============================================================
        if len(refined_prompt.split()) < 10:
            refined_prompt = await self._apply_quality_enhancements(refined_prompt)

        # ============================================================
        # STEP 6: Use prompt_refiner for final refinement (if available)
        # ============================================================
        if self.prompt_refiner:
            refined_prompt = await self.prompt_refiner.refine(
                refined_prompt,
                context={'user_id': user_id, 'history': history}
            )

        return refined_prompt

    def _needs_refinement(self, prompt: str) -> bool:
        """Check if the prompt needs refinement."""
        words = prompt.split()
        if len(words) < 5:
            return True
        if len(words) > 15 and ',' in prompt:
            return False
        # Check for vague indicators
        vague_indicators = ["something", "anything", "nice", "cool", "good"]
        if any(ind in prompt.lower() for ind in vague_indicators):
            return True
        return True

    async def _get_memory_context(self, user_id: Optional[int]) -> Optional[str]:
        """Get relevant memory context for the user."""
        if not user_id or not self.memory_manager:
            return None

        # Check cache first
        cache_key = f"memory_{user_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get recent interactions
        short_term = await self.memory_manager.get_short_term(user_id)
        if not short_term:
            return None

        # Extract the last few meaningful messages
        contexts = []
        for entry in short_term[-3:]:
            msg = entry.get('message', '')
            if msg and len(msg) > 3:
                contexts.append(msg)

        if contexts:
            result = "Previously you asked about: " + ", ".join(contexts[-2:])
            self._cache[cache_key] = result
            return result

        return None

    async def _get_current_topic(self, user_id: Optional[int]) -> Optional[str]:
        """Get the current topic for the user."""
        if not user_id or not self.topic_manager:
            return None
        return await self.topic_manager.get_current_topic(user_id)

    async def _apply_preferences(self, prompt: str, preferences: Dict) -> str:
        """Apply user preferences to the prompt."""
        response_style = preferences.get('response_style', 'balanced')
        preferred_style = preferences.get('preferred_style', 'no_style')
        custom_instructions = preferences.get('custom_instructions', '')

        # Apply response style
        if response_style == 'concise':
            prompt = f"{prompt} (be concise and direct)"
        elif response_style == 'detailed':
            prompt = f"{prompt} (be thorough and detailed)"
        # balanced: no modification

        # Apply preferred artistic style
        if preferred_style and preferred_style != 'no_style':
            prompt = f"{prompt} in {preferred_style} style"

        # Apply custom instructions
        if custom_instructions:
            prompt = f"{prompt}\n\nAdditional instructions: {custom_instructions}"

        return prompt

    async def _apply_topic_enhancement(self, prompt: str, topic: str) -> str:
        """Apply topic-specific enhancements to the prompt."""
        topic_enhancements = {
            'coding': "with clean, well-structured code and clear explanations",
            'creative': "with vivid imagery and engaging storytelling",
            'business': "with professional, strategic insights and clear analysis",
            'education': "with clear, step-by-step explanations and helpful examples",
            'science': "with accurate, data-driven insights and scientific rigor",
            'health': "with evidence-based information and practical advice",
            'technology': "with cutting-edge technical insights and practical applications",
            'travel': "with vivid descriptions and practical travel tips",
            'food': "with mouth-watering descriptions and practical cooking tips",
            'personal': "with warmth, empathy, and genuine understanding",
            'image_generation': "with high-quality, well-composed visual detail",
            'voice': "with clear, natural speech and appropriate tone"
        }

        enhancement = topic_enhancements.get(topic)
        if enhancement and enhancement not in prompt:
            return f"{prompt}, {enhancement}"
        return prompt

    async def _apply_quality_enhancements(self, prompt: str) -> str:
        """Apply quality enhancements for short prompts."""
        quality_modifiers = [
            "high quality", "detailed", "professional",
            "masterpiece", "intricate details", "sharp focus"
        ]
        # Add only if not already present
        added = []
        for modifier in quality_modifiers:
            if modifier not in prompt.lower():
                added.append(modifier)
            if len(added) >= 2:
                break
        if added:
            prompt = f"{prompt}, {', '.join(added)}"
        return prompt

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("📖 ContextRefiner cache cleared")

    def get_info(self) -> Dict[str, Any]:
        """Return information about the refiner."""
        return {
            "name": self.name,
            "type": "ContextRefiner",
            "supports_memory": self.memory_manager is not None,
            "supports_topics": self.topic_manager is not None,
            "supports_preferences": True,
            "cache_size": len(self._cache)
        }