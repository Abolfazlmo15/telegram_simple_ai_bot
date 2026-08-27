"""Base engine that routes to the appropriate engine based on input type and intent.
Integrated with the new Prompt Engineering system and Conversation State.
Uses Preferences for user-specific behavior.
"""
import logging
import re
import asyncio
from typing import Dict, Tuple, Optional, Any, Union, Callable, Coroutine
from PIL import Image
from core.config import Config
from core.managers.user_data_manager import UserDataManager
from core.managers.preference_manager import PreferenceManager
from core.managers.proxy_manager import ProxyManager

# Import prompt engineering modules
from prompt_engineering.detectors import IntentDetector, PromptExtractor, StyleDetector, ContextAnalyzer
from prompt_engineering.refiners import PromptRefiner, TemplateApplier, NegativePromptGenerator
from prompt_engineering.memory import GenerationContext, CorrectionDetector, IterativeRefiner, ConversationState, ConversationMode, ModeDetector
from prompt_engineering.templates import TemplateManager
from prompt_engineering.formatters import TelegramFormatter

# Import document engine (new)
from core.engines.analysis.document_engine import DocumentEngine

logger = logging.getLogger(__name__)


class BaseEngine:
    def __init__(self, user_data_manager: UserDataManager, proxy_manager: Optional[ProxyManager] = None):
        self.user_data_manager = user_data_manager
        self.proxy_manager = proxy_manager
        self.text_engine = None
        self.vision_engine = None
        self.voice_engine = None
        self.image_generation_engine = None
        self.voice_generation_engine = None
        self.document_engine = None          # <-- new
        self.is_initialized = False

        # ========== Prompt Engineering Components ==========
        self.generation_context = GenerationContext(storage_dir=Config.USER_DATA_DIR)
        self.conversation_state = ConversationState(storage_dir=Config.USER_DATA_DIR)
        self.intent_detector = IntentDetector(self.generation_context)
        self.mode_detector = ModeDetector()
        self.prompt_extractor = PromptExtractor()
        self.style_detector = StyleDetector()
        self.template_manager = TemplateManager()
        self.prompt_refiner = PromptRefiner()
        self.template_applier = TemplateApplier()
        self.negative_prompt_generator = NegativePromptGenerator()
        self.correction_detector = CorrectionDetector(self.generation_context)
        self.iterative_refiner = IterativeRefiner()
        self.telegram_formatter = TelegramFormatter()

        # ========== Preference Manager ==========
        self.preference_manager = self.user_data_manager.preference_manager

        # Set refiner on correction detector for intelligent merging
        self.correction_detector._refiner = self.prompt_refiner

        logger.info("Base Engine (Router) initialized with Prompt Engineering, Conversation State, Preferences, and Proxy integration")

    async def initialize(self) -> bool:
        try:
            logger.info("Initialising all engines...")
            from core.engines.analysis.text_engine import TextEngine
            from core.engines.analysis.vision_engine import VisionEngine
            from core.engines.analysis.voice_engine import VoiceEngine

            self.text_engine = TextEngine(self.user_data_manager)
            text_success = await self.text_engine.initialize()
            if not text_success:
                logger.error("Failed to initialise text engine")
                return False

            self.vision_engine = VisionEngine(self.user_data_manager, proxy_manager=self.proxy_manager)
            vision_success = await self.vision_engine.initialize()
            if not vision_success:
                logger.warning("Vision engine failed to initialise – continuing without it")

            self.voice_engine = VoiceEngine(self.user_data_manager, proxy_manager=self.proxy_manager)
            voice_success = await self.voice_engine.initialize()
            if not voice_success:
                logger.warning("Voice (STT) engine failed to initialise – voice messages will be unavailable")

            from core.engines.generation.image_engine import ImageGenerationEngine
            from core.engines.generation.voice_engine import VoiceGenerationEngine

            self.image_generation_engine = ImageGenerationEngine(self.user_data_manager)
            image_gen_success = await self.image_generation_engine.initialize()
            if not image_gen_success:
                logger.warning("Image generation engine failed to initialise")

            self.voice_generation_engine = VoiceGenerationEngine(self.user_data_manager, proxy_manager=self.proxy_manager)
            voice_gen_success = await self.voice_generation_engine.initialize()
            if not voice_gen_success:
                logger.warning("Voice generation engine failed to initialise")

            # ========== Initialize Prompt Engineering Components ==========
            await self.template_manager.initialize()
            self.prompt_refiner.text_engine = self.text_engine

            # ========== Initialize Document Engine ==========
            self.document_engine = DocumentEngine(self.user_data_manager, text_engine=self.text_engine)
            logger.info("✅ Document engine initialised")

            self.is_initialized = True
            logger.info("✅ All engines and prompt engineering components initialised successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialise engines: {e}", exc_info=True)
            return False

    async def shutdown(self) -> None:
        logger.info("Shutting down all engines...")
        if self.text_engine:
            await self.text_engine.shutdown()
        if self.vision_engine:
            await self.vision_engine.shutdown()
        if self.voice_engine:
            await self.voice_engine.shutdown()
        if self.image_generation_engine:
            await self.image_generation_engine.shutdown()
        if self.voice_generation_engine:
            await self.voice_generation_engine.shutdown()
        self.is_initialized = False
        logger.info("All engines shutdown complete")

    async def process(
        self,
        input_data: Any,
        context: Optional[Dict] = None,
        status_callback: Optional[Callable[[str, bool], Coroutine]] = None
    ) -> Tuple[str, str, int]:
        """
        Process input data with optional status callback for timeout/feedback.
        Supports cancellation via asyncio.CancelledError.

        Args:
            input_data: Text, bytes (image/audio/document), or PIL Image.
            context: Context dict with user_id, history, preferences, etc.
            status_callback: Async callback function (msg, edit) for updating status messages.

        Returns:
            Tuple of (response_data, model_used, metadata)
        """
        if not self.is_initialized:
            raise RuntimeError("Base engine not initialised. Call initialize() first.")

        # Check for cancellation before processing
        if asyncio.current_task().cancelled():
            raise asyncio.CancelledError

        # ============================================================
        # FETCH ALL PRIORITIES ONCE AT THE START
        # ============================================================
        user_id = context.get('user_id') if context else None
        username = context.get('username') if context else None

        text_priority = None
        vision_priority = None
        voice_priority = None
        voice_gen_priority = None
        preferences = {}

        if user_id:
            preferences = await self.preference_manager.get_preferences(user_id, username)
            text_priority = await self.user_data_manager.get_user_model_priority(user_id, username, engine="text")
            vision_priority = await self.user_data_manager.get_user_model_priority(user_id, username, engine="vision")
            voice_priority = await self.user_data_manager.get_user_model_priority(user_id, username, engine="voice")
            voice_gen_priority = await self.user_data_manager.get_user_model_priority(user_id, username, engine="voice_gen")
            logger.info(f"📋 Loaded preferences for user {user_id}: mode={preferences.get('response_mode')}, style={preferences.get('response_style')}")

        # Inject priorities into context for sub‑engines
        if context is None:
            context = {}
        context['text_priority'] = text_priority
        context['vision_priority'] = vision_priority
        context['voice_priority'] = voice_priority
        context['voice_gen_priority'] = voice_gen_priority
        context['preferences'] = preferences

        try:
            if isinstance(input_data, str):
                text = input_data.strip()
                if not text:
                    raise ValueError("Empty input text")

                history = context.get('history', []) if context else []
                skip_cache = context.get('skip_cache', False) if context else False

                # ============================================================
                # STEP 1: Detect if this is a mode switch (voice ↔ text)
                # ============================================================
                mode_result = await self.mode_detector.detect(
                    text,
                    context={'user_id': user_id, 'input_type': 'text'}
                )

                is_mode_switch = mode_result.get('is_mode_switch', False)
                target_mode = mode_result.get('target_mode')

                if is_mode_switch and target_mode and user_id:
                    if target_mode == 'voice':
                        await self.conversation_state.set_mode(user_id, ConversationMode.VOICE)
                        await self.preference_manager.set_response_mode(user_id, "voice", username)
                        logger.info(f"🗣️ User {user_id} switched to VOICE mode (preference saved)")
                        return f"🗣️ *Voice mode activated!* I'll speak my responses from now on.\n\nTo switch back, just say 'type it' or 'text mode'.", "mode_switch_voice", 0
                    elif target_mode == 'text':
                        await self.conversation_state.set_mode(user_id, ConversationMode.TEXT)
                        await self.preference_manager.set_response_mode(user_id, "text", username)
                        logger.info(f"📝 User {user_id} switched to TEXT mode (preference saved)")
                        return f"📝 *Text mode activated!* I'll type my responses from now on.\n\nTo switch to voice, say 'talk to me' or 'voice mode'.", "mode_switch_text", 0

                # ============================================================
                # STEP 2: Record the input for context
                # ============================================================
                if user_id:
                    await self.conversation_state.record_input(user_id, 'text', text)

                # ============================================================
                # STEP 3: Analyze intent
                # ============================================================
                intent_result = await self.intent_detector.detect(
                    text,
                    context={'user_id': user_id, 'history': history}
                )
                intent = intent_result.get('intent', 'text_analysis')
                is_correction = intent_result.get('is_correction', False)

                # ============================================================
                # STEP 4: Extract the actual prompt
                # ============================================================
                extract_result = await self.prompt_extractor.detect(text)
                extracted_prompt = extract_result.get('extracted_prompt', text)

                # ============================================================
                # STEP 5: Handle correction or refine prompt
                # ============================================================
                if is_correction and user_id:
                    correction_type = intent_result.get('correction_type', 'unknown')
                    original_prompt = intent_result.get('original_prompt', '')
                    suggestion = intent_result.get('suggestion', '')

                    if original_prompt and suggestion:
                        logger.info(f"🔄 Applying correction: {correction_type}")
                        await self.iterative_refiner.store(
                            str(user_id),
                            suggestion,
                            context={'correction_type': correction_type, 'original': original_prompt}
                        )
                        refined_prompt = suggestion
                        logger.info(f"📝 Corrected prompt: {refined_prompt[:100]}...")
                    else:
                        refined_prompt = extracted_prompt
                else:
                    if intent == 'image_generation':
                        # Refine prompt – ensure we get a clean prompt, not the thinking process
                        refined_prompt = await self.prompt_refiner.refine(
                            extracted_prompt,
                            context={'user_id': user_id, 'history': history, 'preferences': preferences}
                        )
                        # If the refined prompt is suspiciously long or contains "thinking process",
                        # fallback to the original prompt
                        if len(refined_prompt) > 3000 or "thinking" in refined_prompt.lower()[:200]:
                            logger.warning("Refined prompt seems to contain thinking process, using original prompt.")
                            refined_prompt = extracted_prompt
                        logger.info(f"📝 Refined prompt: {refined_prompt[:100]}...")
                    else:
                        refined_prompt = extracted_prompt

                # ============================================================
                # STEP 6: Determine response mode
                # ============================================================
                response_mode = ConversationMode.TEXT
                if user_id:
                    pref_mode = await self.preference_manager.get_response_mode(user_id, username)
                    if pref_mode == "voice":
                        response_mode = ConversationMode.VOICE
                    elif pref_mode == "text":
                        response_mode = ConversationMode.TEXT
                    elif pref_mode == "auto":
                        state_mode = await self.conversation_state.get_mode(user_id)
                        response_mode = state_mode
                    else:
                        response_mode = await self.conversation_state.get_mode(user_id)

                if context and context.get('input_type') == 'voice':
                    response_mode = ConversationMode.VOICE
                    if user_id:
                        await self.conversation_state.set_mode(user_id, ConversationMode.VOICE)

                # ============================================================
                # STEP 7: Route to appropriate engine (with status_callback)
                # ============================================================
                if intent == 'image_generation':
                    logger.info(f"🎨 Detected image generation request")

                    preferred_style = await self.preference_manager.get_preferred_style(user_id, username) if user_id else "no_style"
                    if preferred_style and preferred_style != "no_style":
                        logger.info(f"🎨 Using user's preferred style: {preferred_style}")

                    style_result = await self.style_detector.detect(
                        refined_prompt,
                        context={'user_id': user_id, 'history': history}
                    )
                    detected_style = style_result.get('primary_style', 'no_style')
                    if preferred_style and preferred_style != "no_style" and detected_style == "no_style":
                        detected_style = preferred_style
                        logger.info(f"🎨 Applied preferred style: {detected_style}")

                    recommended_models = style_result.get('recommended_models', [])
                    negative_prompt = await self.negative_prompt_generator.refine(
                        refined_prompt,
                        context={'style': detected_style}
                    )

                    gen_context = {
                        'user_id': user_id,
                        'username': username,
                        'history': history,
                        'skip_cache': skip_cache,
                        'detected_style': detected_style,
                        'recommended_models': recommended_models,
                        'negative_prompt': negative_prompt,
                        'refined_prompt': refined_prompt,
                        'original_prompt': text,
                        'is_correction': is_correction,
                        'preferences': preferences
                    }

                    # Pass status_callback to image generation engine
                    image_bytes, model, size = await self.image_generation_engine.generate(
                        refined_prompt,
                        context=gen_context,
                        status_callback=status_callback
                    )

                    if user_id:
                        await self.generation_context.store_generation(
                            user_id=user_id,
                            prompt=refined_prompt,
                            image_bytes=image_bytes,
                            model_used=model,
                            style=detected_style,
                            source="image_generation"
                        )

                    return image_bytes, f"gen_image:{model}", size

                elif intent == 'voice_generation':
                    logger.info(f"🔊 Detected voice generation request (explicit)")
                    voice_text = self._extract_voice_text(text, refined_prompt)
                    logger.info(f"🔊 Speaking: {voice_text[:50]}...")

                    voice_speed = await self.preference_manager.get_voice_speed(user_id, username) if user_id else 1.0
                    voice_style = await self.preference_manager.get_voice_style(user_id, username) if user_id else "neutral"

                    gen_context = {
                        'user_id': user_id,
                        'username': username,
                        'voice_speed': voice_speed,
                        'voice_style': voice_style,
                        'preferences': preferences,
                        'priority_list': voice_gen_priority
                    }

                    # Pass status_callback to voice generation engine
                    audio_bytes, model, size = await self.voice_generation_engine.generate(
                        voice_text,
                        context=gen_context,
                        status_callback=status_callback
                    )

                    if user_id:
                        await self.generation_context.store_generation(
                            user_id=user_id,
                            prompt=voice_text,
                            image_bytes=None,
                            model_used=model,
                            style="voice",
                            source="voice_generation"
                        )

                    return audio_bytes, f"gen_voice:{model}", size

                else:
                    # ============================================================
                    # STEP 8: Text analysis – with status_callback
                    # ============================================================
                    logger.debug(f"Routing to TextEngine (analysis) in {response_mode.value} mode")

                    response_style = await self.preference_manager.get_response_style(user_id, username) if user_id else "balanced"
                    custom_instructions = await self.preference_manager.get_custom_instructions(user_id, username) if user_id else ""

                    text_context = context.copy() if context else {}
                    text_context['preferences'] = preferences
                    text_context['response_style'] = response_style
                    text_context['custom_instructions'] = custom_instructions
                    text_context['priority_list'] = text_priority

                    # Pass status_callback to text engine
                    response_text, model, tokens = await self.text_engine.process(
                        text,
                        text_context,
                        status_callback=status_callback
                    )

                    # ============================================================
                    # STEP 9: Convert to voice if in voice mode (with validation)
                    # ============================================================
                    if response_mode == ConversationMode.VOICE and self.voice_generation_engine and self.voice_generation_engine.is_initialized:
                        try:
                            if response_text and len(response_text.strip()) > 5:
                                logger.info(f"🔊 Converting text response to voice for user {user_id}")
                                voice_speed = await self.preference_manager.get_voice_speed(user_id, username) if user_id else 1.0
                                voice_style = await self.preference_manager.get_voice_style(user_id, username) if user_id else "neutral"

                                tts_context = {
                                    'user_id': user_id,
                                    'username': username,
                                    'voice_speed': voice_speed,
                                    'voice_style': voice_style,
                                    'priority_list': voice_gen_priority
                                }
                                # Pass status_callback to voice generation
                                audio_bytes, voice_model, size = await self.voice_generation_engine.generate(
                                    response_text,
                                    context=tts_context,
                                    status_callback=status_callback
                                )
                                return audio_bytes, f"gen_voice_conversation:{voice_model}", size
                            else:
                                logger.warning(f"Response text is empty or too short, skipping TTS")
                                return response_text, model, tokens
                        except Exception as e:
                            logger.error(f"❌ Voice conversion failed, falling back to text: {e}")
                            return response_text, model, tokens
                    else:
                        return response_text, model, tokens

            elif isinstance(input_data, bytes):
                input_type = context.get('input_type', 'image') if context else 'image'

                # --- Document handling (new) ---
                if input_type == 'document' and self.document_engine:
                    logger.debug("Routing to DocumentEngine")
                    # Inject required info into context
                    doc_context = context.copy() if context else {}
                    # Ensure we have file_extension and caption
                    return await self.document_engine.process(input_data, doc_context)

                elif input_type == 'audio' and self.voice_engine:
                    logger.debug("Routing to VoiceEngine (STT)")
                    # Pass status_callback to voice engine transcribe
                    transcription, model, tokens = await self.voice_engine.transcribe(
                        input_data,
                        context,
                        status_callback=status_callback
                    )
                    if user_id:
                        await self.conversation_state.record_input(user_id, 'voice', transcription)
                    voice_context = context.copy() if context else {}
                    voice_context['input_type'] = 'voice'
                    voice_context['priority_list'] = voice_priority
                    # Recursively process with status_callback
                    return await self.process(transcription, voice_context, status_callback)

                else:
                    # Default: treat as image (vision)
                    logger.debug("Routing to VisionEngine (bytes)")
                    context['priority_list'] = vision_priority
                    return await self.vision_engine.process(
                        input_data,
                        context,
                        status_callback=status_callback
                    )

            elif isinstance(input_data, Image.Image):
                logger.debug("Routing to VisionEngine (PIL Image)")
                context['priority_list'] = vision_priority
                return await self.vision_engine.process(
                    input_data,
                    context,
                    status_callback=status_callback
                )

            else:
                logger.warning(f"Unknown input type {type(input_data)}, treating as text")
                return await self.text_engine.process(str(input_data), context, status_callback)

        except asyncio.CancelledError:
            logger.info("BaseEngine.process cancelled, propagating")
            raise
        except Exception as e:
            logger.error(f"Error in base engine routing: {e}", exc_info=True)
            raise

    # ========== VOICE TEXT EXTRACTION ==========
    def _extract_voice_text(self, original: str, extracted: str) -> str:
        if extracted and extracted != original and extracted.strip():
            return extracted

        voice_triggers = [
            "say this", "speak this", "read this aloud", "tell me this",
            "voice this", "audio of", "speak the text", "say it",
            "narrate this", "convert to speech", "text to speech", "tts",
            "say that", "speak that", "read that", "talk to me", "tell me"
        ]

        text_lower = original.lower()
        for trigger in voice_triggers:
            if trigger in text_lower:
                rest = original.lower().replace(trigger, "").strip()
                rest = rest.lstrip(":,.!? ").strip()
                rest = rest.strip('"').strip("'").strip()
                if rest:
                    return rest

        return original

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "base_engine": {
                "initialized": self.is_initialized,
                "type": "Router"
            },
            "text_engine": self.text_engine.get_engine_info() if self.text_engine else None,
            "vision_engine": self.vision_engine.get_engine_info() if self.vision_engine else None,
            "voice_engine": self.voice_engine.get_engine_info() if self.voice_engine else None,
            "document_engine": self.document_engine.get_engine_info() if self.document_engine else None,
            "image_generation_engine": self.image_generation_engine.get_engine_info() if self.image_generation_engine else None,
            "voice_generation_engine": self.voice_generation_engine.get_engine_info() if self.voice_generation_engine else None,
            "conversation_state": self.conversation_state.get_info() if hasattr(self.conversation_state, 'get_info') else {},
            "preference_manager": self.preference_manager.get_info() if hasattr(self.preference_manager, 'get_info') else {},
            "prompt_engineering": {
                "template_manager": self.template_manager.get_info() if hasattr(self.template_manager, 'get_info') else {},
                "generation_context": self.generation_context.get_info() if hasattr(self.generation_context, 'get_info') else {}
            }
        }