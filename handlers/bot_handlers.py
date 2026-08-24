import logging
import time
import asyncio
import re
import httpx
from telegram import Update
from telegram.ext import ContextTypes
from telegram.error import NetworkError, TimedOut
from core.config import Config
from core.engines.base_engine import BaseEngine
from core.engines.analysis.voice_engine import VoiceEngine
from core.managers.rate_limiter import RateLimiter
from core.managers.user_data_manager import UserDataManager
from core.managers.proxy_manager import ProxyManager
from core.analytics.analytics_engine import AnalyticsEngine
from core.utils.response_formatter import ResponseFormatter

# ============================================================
# NEW: Import memory, topic, and preference managers
# ============================================================
from core.managers.memory_manager import MemoryManager
from core.managers.topic_manager import TopicManager

# Import TelegramFormatter from prompt_engineering
from prompt_engineering.formatters import TelegramFormatter
from prompt_engineering.refiners.context_refiner import ContextRefiner

logger = logging.getLogger(__name__)


class BotHandlers:
    def __init__(self, engine: BaseEngine, voice_engine: VoiceEngine,
                 rate_limiter: RateLimiter, user_data_manager: UserDataManager,
                 analytics_engine: AnalyticsEngine, proxy_manager: ProxyManager):
        self.engine = engine
        self.voice_engine = voice_engine
        self.rate_limiter = rate_limiter
        self.user_data_manager = user_data_manager
        self.analytics_engine = analytics_engine
        self.proxy_manager = proxy_manager
        self.formatter = ResponseFormatter()
        self.telegram_formatter = TelegramFormatter()

        # ============================================================
        # NEW: Initialize memory and topic managers
        # ============================================================
        self.memory_manager = MemoryManager(
            base_dir=Config.USER_DATA_DIR,
            max_short_term=Config.MEMORY_MAX_SHORT_TERM
        )
        self.topic_manager = TopicManager()

        # Initialize ContextRefiner with the managers
        self.context_refiner = ContextRefiner(
            memory_manager=self.memory_manager,
            topic_manager=self.topic_manager,
            prompt_refiner=self.engine.prompt_refiner
        )

        # Set the text engine on the mode detector for LLM fallback
        self.engine.mode_detector.set_text_engine(self.engine.text_engine)

        logger.info("📋 BotHandlers initialized with Memory, Topic, and ContextRefiner")

    # ========== COMMAND HANDLERS ==========
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        await self.user_data_manager.start_new_session(user.id, user.username)

        await self.user_data_manager.load_user_info(
            user.id, user.username,
            first_name=user.first_name or "",
            last_name=user.last_name or "",
            bio=user.bio or "",
            phone_number=user.phone_number or ""
        )
        if user.get_profile_photos:
            photos = await user.get_profile_photos(limit=1)
            if photos.photos:
                file = await photos.photos[0][-1].get_file()
                photo_bytes = await file.download_as_bytearray()
                await self.user_data_manager.save_profile_photo(user.id, user.username, photo_bytes)

        text = self.formatter.format_response(
            f"👋 *Welcome, {user.first_name}!*\n\n"
            f"I'm your AI assistant powered by advanced language models.\n\n"
            f"*What I can help with:*\n"
            f"• 💻 Coding & Technical Questions\n"
            f"• 📊 Data Analysis & Insights\n"
            f"• 📚 Learning & Explanations\n"
            f"• 💼 Business & Professional Advice\n"
            f"• ✨ Creative Writing\n"
            f"• 🖼️ *Image Analysis* - Just send me a photo!\n"
            f"• 🎤 *Voice Messages* - Send me a voice note!\n"
            f"• 🎨 *Image Generation* - Say 'generate an image of ...'\n"
            f"• 🔊 *Voice Generation* - Say 'say this ...' or 'speak this ...'\n"
            f"• 🗣️ *Voice Mode* - Say 'talk to me' to switch to voice responses, or 'type it' for text\n"
            f"• 🧠 *Memory* - I remember our conversations to give better answers\n\n"
            f"_Just type your question, send an image, a voice message, or a generation command._"
        )
        await self._send_chunked_message(update, text)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = self.formatter.format_response(
            f"📖 *Available Commands:*\n\n"
            f"/start - Start the bot\n"
            f"/help - Show this help message\n"
            f"/about - Learn about the bot\n"
            f"/status - Check your usage\n"
            f"/clear - Clear your history\n"
            f"/prioritize_text_engine - Set priority for text models\n"
            f"/prioritize_vision_engine - Set priority for vision models\n"
            f"/image_priority - Set priority for image generation tiers\n\n"
            f"💡 *Voice Mode:*\n"
            f"• Say 'talk to me' to switch to voice responses\n"
            f"• Say 'type it' or 'text mode' to switch back to text\n"
            f"• Send a voice message to automatically respond in voice\n\n"
            f"💡 *Memory:*\n"
            f"• I remember recent conversations to give better answers\n"
            f"• You can ask 'remember what we talked about?'\n\n"
            f"💡 *Generation Tips:*\n"
            f"• To generate an image, say: 'generate an image of ...'\n"
            f"• To generate voice, say: 'say this ...' or 'speak this ...'\n\n"
            f"⌨️ *Markdown Support:*\n"
            f"Use `backticks` for code\n"
            f"Use *asterisks* for bold\n"
            f"Use _underscores_ for italic"
        )
        await self._send_chunked_message(update, text)

    async def about(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = self.formatter.format_response(
            f"🧠 *About This Bot*\n\n"
            f"This is an advanced AI chatbot featuring:\n\n"
            f"• 🚀 *Fast Response Times* - Optimized with HTTP/2\n"
            f"• 💾 *Smart Caching* - Instant answers to common questions\n"
            f"• 📊 *Analytics* - Continuous improvement\n"
            f"• 💬 *Context Awareness* - Remembers conversation history\n"
            f"• 🧠 *Memory* - Long-term and short-term memory for better context\n"
            f"• 🎯 *Model Priority* - Customize your AI experience\n"
            f"• 🖼️ *Vision Capabilities* - Image analysis and description\n"
            f"• 🎤 *Voice Transcription* - Send a voice note, get a reply\n"
            f"• 🎨 *Image Generation* - Generate images from text\n"
            f"• 🔊 *Voice Generation* - Text-to-speech\n"
            f"• 🗣️ *Voice Mode* - Talk to me and I'll respond in voice\n"
            f"• 📊 *Topic Tracking* - I understand what topics we're discussing\n\n"
            f"*Powered by:*\n"
            f"• DeepSeek, Qwen, and Llama Vision models\n"
            f"• Whisper for speech-to-text\n"
            f"• Stable Diffusion & DALL-E for image generation\n"
            f"• OpenRouter API\n"
            f"• Python & Telegram Bot API\n\n"
            f"Source: [`{Config.BOT_REPO_URL}`]({Config.BOT_REPO_URL})"
        )
        await self._send_chunked_message(update, text)

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        username = update.effective_user.username
        remaining = self.rate_limiter.get_remaining(user_id)
        stats = await self.user_data_manager.get_user_stats(user_id, username)

        # Get memory stats
        short_term = await self.memory_manager.get_short_term(user_id)
        long_term = await self.memory_manager.get_long_term(user_id)

        text = f" *Your Status*\n\n"
        text += f"*Rate Limit:*\n"
        text += f"Remaining: {remaining}/{self.rate_limiter.max_requests}\n"
        text += f"Window: {self.rate_limiter.window_seconds}s\n\n"

        if stats:
            text += f"*Usage Stats:*\n"
            text += f"Total messages: {stats.get('total_messages', 0)}\n"
            text += f"Total images: {stats.get('total_images', 0)}\n"
            text += f"Total tokens used: {stats.get('total_tokens', 0)}\n"
            text += f"Avg. response time: {stats.get('avg_response_time', 0.0):.2f}s\n"
            text += f"Session duration: {stats.get('session_duration', 'N/A')}\n"
        else:
            text += f"*Usage Stats:*\n"
            text += f"No data available yet.\n"

        text += f"\n*Memory Stats:*\n"
        text += f"Short-term entries: {len(short_term)}\n"
        text += f"Long-term summaries: {len(long_term)}\n"

        text = self.formatter.format_response(text)
        await self._send_chunked_message(update, text)

    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        success = await self.user_data_manager.clear_user_data(user.id, user.username)
        if success:
            await self.memory_manager.clear_memory(user.id)
            await self.topic_manager.clear_topics(user.id)
            await update.message.reply_text(
                "*🗑️ History Cleared*\n\n"
                "Your conversation history and data have been cleared successfully.",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
        else:
            await update.message.reply_text(
                "❌ *Error*\n\n"
                "Failed to clear history. Please try again.",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )

    # ========== PRIORITY COMMANDS ==========
    async def prioritize_text_engine(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._start_priority_setup(update, context, engine="text")

    async def prioritize_vision_engine(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._start_priority_setup(update, context, engine="vision")

    async def _start_priority_setup(self, update: Update, context: ContextTypes.DEFAULT_TYPE, engine: str) -> None:
        user = update.effective_user
        context.user_data.clear()

        if engine == "text":
            available_models = self.engine.text_engine.model_manager.get_fast_models()
        elif engine == "vision":
            available_models = self.engine.vision_engine.model_manager.get_available_models()
        else:
            await update.message.reply_text("❌ Unknown engine type.")
            return

        if not available_models:
            await update.message.reply_text(
                "❌ *Error*\n\n"
                "No models available right now. Please try again later.",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
            return

        context.user_data['setting_priority'] = True
        context.user_data['priority_list'] = []
        context.user_data['available_models'] = available_models
        context.user_data['current_step'] = 1
        context.user_data['engine'] = engine

        model_list = "\n".join([f"{i + 1}. `{m}`" for i, m in enumerate(available_models[:10])])
        if len(available_models) > 10:
            model_list += f"\n... and {len(available_models) - 10} more"

        text = (
            f"🎯 *{engine.title()} Engine Priority Setup*\n\n"
            f"*Available models:*\n{model_list}\n\n"
            f"Let's set your priority. Which model should be *#1*?\n\n"
            f"_Type the exact model name (e.g., deepseek/deepseek-chat:free)_\n"
            f"_Or type /cancel to cancel_"
        )
        await update.message.reply_text(text, parse_mode="Markdown", reply_to_message_id=update.message.message_id)

    async def handle_priority_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not context.user_data.get('setting_priority'):
            return

        user = update.effective_user
        user_text = update.message.text.strip()
        available = context.user_data.get('available_models', [])
        priority_list = context.user_data.get('priority_list', [])
        step = context.user_data.get('current_step', 1)
        engine = context.user_data.get('engine', 'text')

        if user_text.lower() in ['cancel', 'stop', '/cancel']:
            context.user_data.clear()
            await update.message.reply_text(
                "❌ *Priority setup cancelled.*",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
            return

        if user_text.lower() == 'done':
            if len(priority_list) == 0:
                await update.message.reply_text(
                    "⚠️ *No models selected.*\n\n"
                    "Please select at least one model before finishing.",
                    parse_mode="Markdown",
                    reply_to_message_id=update.message.message_id
                )
                return

            await self.user_data_manager.save_model_priority(user.id, user.username, priority_list, engine=engine)
            context.user_data.clear()

            final_list = "\n".join([f"{i + 1}. `{m}`" for i, m in enumerate(priority_list)])
            text = (
                f"✅ *Priority Saved for {engine.title()} Engine!*\n\n"
                f"*Your new model order:*\n{final_list}\n\n"
                f"The bot will now use these models in this order for {engine} queries."
            )
            await update.message.reply_text(text, parse_mode="Markdown", reply_to_message_id=update.message.message_id)
            return

        if user_text not in available:
            await update.message.reply_text(
                f"️ *Invalid model.*\n\n"
                f"`{user_text}` not found in available models.\n"
                f"Please choose from the list.",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
            return

        if user_text in priority_list:
            await update.message.reply_text(
                "️ *Already selected.*\n\n"
                "Please choose a different model.",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
            return

        priority_list.append(user_text)
        context.user_data['priority_list'] = priority_list
        context.user_data['current_step'] = step + 1

        if len(priority_list) == len(available):
            await self.user_data_manager.save_model_priority(user.id, user.username, priority_list, engine=engine)
            context.user_data.clear()

            final_list = "\n".join([f"{i + 1}. `{m}`" for i, m in enumerate(priority_list)])
            text = (
                f"✅ *Priority Saved for {engine.title()} Engine!*\n\n"
                f"*Your new model order:*\n{final_list}\n\n"
                f"The bot will now use these models in this order for {engine} queries."
            )
            await update.message.reply_text(text, parse_mode="Markdown", reply_to_message_id=update.message.message_id)
        else:
            next_step = len(priority_list) + 1
            remaining = [m for m in available if m not in priority_list]
            remaining_str = "\n".join([f"- `{m}`" for m in remaining[:10]])
            if len(remaining) > 10:
                remaining_str += f"\n... and {len(remaining) - 10} more"

            text = (
                f"✅ Added *{user_text}* as #{step}.\n\n"
                f"Which model should be *#{next_step}*?\n\n"
                f"*Remaining:*\n{remaining_str}\n\n"
                f"_Type 'done' to finish with current list_"
            )
            await update.message.reply_text(text, parse_mode="Markdown", reply_to_message_id=update.message.message_id)

    # ========== IMAGE GENERATION PRIORITY ==========
    async def prioritize_image_generation_method(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Set priority order for image generation tiers."""
        user = update.effective_user
        user_id = user.id
        username = user.username

        # Get available tiers
        available_tiers = await self.user_data_manager.get_available_tiers()
        enabled_tiers = [name for name, enabled in available_tiers.items() if enabled]

        if not enabled_tiers:
            await update.message.reply_text(
                "❌ *No image generation tiers available.*\n\n"
                "Please check your API keys and try again.",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
            return

        # Get current priority
        current_priority = await self.user_data_manager.get_image_generation_priority(user_id, username)

        # Build the message
        tier_names = {
            "pollinations": "🖼️ Pollinations.ai (Free)",
            "huggingface": "🤗 Hugging Face (Free tier)",
            "openrouter": "🔗 OpenRouter (Paid)"
        }

        current_list = "\n".join([f"{i + 1}. {tier_names.get(t, t)}" for i, t in enumerate(current_priority)])

        available_list = "\n".join([f"• {tier_names.get(t, t)}" for t in enabled_tiers])

        text = f"🎯 *Image Generation Priority Setup*\n\n"
        text += f"*Current priority order:*\n{current_list}\n\n"
        text += f"*Available tiers:*\n{available_list}\n\n"
        text += f"To change the priority, send a list of tier names in order of preference.\n\n"
        text += f"*Example:*\n"
        text += f"`pollinations, huggingface, openrouter`\n\n"
        text += f"*Or:*\n"
        text += f"`huggingface, pollinations`\n\n"
        text += f"_Type /cancel to cancel._"

        context.user_data['setting_image_priority'] = True
        await update.message.reply_text(text, parse_mode="Markdown", reply_to_message_id=update.message.message_id)

    async def handle_image_priority_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the user's image priority input."""
        if not context.user_data.get('setting_image_priority'):
            return

        user = update.effective_user
        user_text = update.message.text.strip()

        if user_text.lower() in ['cancel', 'stop', '/cancel']:
            context.user_data.clear()
            await update.message.reply_text(
                "❌ *Priority setup cancelled.*",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
            return

        # Parse the priority list
        parts = re.split(r'[,;\n]', user_text)
        priority_list = []

        for part in parts:
            tier = part.strip().lower()
            if tier in self.engine.image_generation_engine.tiers:
                priority_list.append(tier)
            elif tier == "pollinations":
                priority_list.append("pollinations")
            elif tier == "huggingface":
                priority_list.append("huggingface")
            elif tier == "openrouter":
                priority_list.append("openrouter")

        # Remove duplicates while preserving order
        seen = set()
        priority_list = [x for x in priority_list if not (x in seen or seen.add(x))]

        # Validate
        available_tiers = await self.user_data_manager.get_available_tiers()
        valid_tiers = [name for name, enabled in available_tiers.items() if enabled]

        invalid = [t for t in priority_list if t not in valid_tiers]
        if invalid:
            await update.message.reply_text(
                f"❌ *Invalid tiers:* `{', '.join(invalid)}`\n\n"
                f"Please choose from: `{', '.join(valid_tiers)}`",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
            return

        if not priority_list:
            await update.message.reply_text(
                "❌ *No valid tiers selected.*\n\n"
                f"Please choose from: `{', '.join(valid_tiers)}`",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
            return

        # Add any missing valid tiers to the end
        for tier in valid_tiers:
            if tier not in priority_list:
                priority_list.append(tier)

        # Save the priority
        success = await self.user_data_manager.save_image_generation_priority(user.id, user.username, priority_list)

        if success:
            tier_names = {
                "pollinations": "Pollinations.ai (Free)",
                "huggingface": "Hugging Face (Free tier)",
                "openrouter": "OpenRouter (Paid)"
            }
            new_list = "\n".join([f"{i + 1}. {tier_names.get(t, t)}" for i, t in enumerate(priority_list)])

            text = f"✅ *Priority saved!*\n\n"
            text += f"*New priority order:*\n{new_list}\n\n"
            text += f"The bot will now try tiers in this order for image generation."

            await update.message.reply_text(text, parse_mode="Markdown", reply_to_message_id=update.message.message_id)
        else:
            await update.message.reply_text(
                "❌ *Failed to save priority.*\n\n"
                "Please try again.",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )

        context.user_data.clear()

    # ========== MESSAGE HANDLERS ==========
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Check if we're in priority setup mode
        if context.user_data.get('setting_image_priority'):
            await self.handle_image_priority_input(update, context)
            return

        if context.user_data.get('setting_priority'):
            await self.handle_priority_input(update, context)
            return

        if not self.engine.is_initialized:
            await update.message.reply_text("️ Bot is still initializing. Please wait.", reply_to_message_id=update.message.message_id)
            return

        user = update.effective_user
        user_id = user.id
        username = user.username
        user_text = update.message.text

        allowed, remaining = await self.rate_limiter.check(user_id)
        if not allowed:
            await self._handle_rate_limit(update, remaining)
            return

        if not user_text:
            await update.message.reply_text("Please send me a text message.", reply_to_message_id=update.message.message_id)
            return

        # Check if user is asking about past conversation (memory)
        memory_keywords = ["remember", "before", "previous", "past", "earlier", "last time", "what did we", "what did you", "search history"]
        if any(kw in user_text.lower() for kw in memory_keywords):
            history_results = await self.user_data_manager.search_history(user_id, user_text)
            if history_results:
                context_text = "I found this in our conversation history:\n\n"
                for entry in history_results[:3]:
                    if entry.get('type') == 'text':
                        context_text += f"• You: {entry.get('message', '')}\n"
                        context_text += f"  Me: {entry.get('response', '')}\n\n"
                    elif entry.get('type') == 'image':
                        context_text += f"• You sent an image with query: {entry.get('query', '')}\n"
                        context_text += f"  Me: {entry.get('response', '')}\n\n"
                    elif entry.get('type') == 'voice':
                        context_text += f"• You sent a voice message: {entry.get('transcription', '')}\n"
                        context_text += f"  Me: {entry.get('response', '')}\n\n"
                    elif entry.get('type') in ('generated_image', 'generated_voice'):
                        context_text += f"• You generated: {entry.get('prompt', '')}\n"
                        context_text += f"  Result: {entry.get('response', '')}\n\n"
                context_text += "Is that what you were looking for?"
                await update.message.reply_text(context_text, reply_to_message_id=update.message.message_id)
                return

        skip_cache = False
        if update.message.reply_to_message:
            skip_cache = True
        follow_up_words = ["image", "picture", "photo", "previous", "last", "that", "this"]
        if any(word in user_text.lower() for word in follow_up_words):
            skip_cache = True

        placeholder = await update.message.reply_text("🤔", reply_to_message_id=update.message.message_id)

        try:
            await update.message.chat.send_action(action="typing")
        except Exception:
            pass

        start_time = time.time()
        try:
            user_data = await self.user_data_manager.load_user_data(user_id, username)
            history = user_data.get('history', [])

            # ============================================================
            # NEW: Update memory and topic tracking
            # ============================================================
            # Add user message to memory
            await self.memory_manager.add_interaction(
                user_id=user_id,
                message=user_text,
                response="",  # Will be updated after response
                category="text"
            )

            # Detect and track topic
            detected_topic = await self.topic_manager.add_message(user_id, user_text)
            if detected_topic:
                logger.info(f"📊 Topic detected: {detected_topic}")

            # Get memory context for the engine
            memory_context = await self.memory_manager.get_context(user_id, limit=5)

            # ============================================================
            # NEW: Get user preferences
            # ============================================================
            preferences = await self.user_data_manager.get_preferences(user_id, username)
            custom_instructions = await self.user_data_manager.get_custom_instructions(user_id, username)

            # ============================================================
            # Process via base engine – pass memory and preferences
            # ============================================================
            result, model_used, metadata = await self.engine.process(
                user_text,
                context={
                    'user_id': user_id,
                    'username': username,
                    'history': history,
                    'skip_cache': skip_cache,
                    'input_type': 'text',
                    'preferences': preferences,
                    'custom_instructions': custom_instructions,
                    'memory_context': memory_context,
                    'current_topic': detected_topic
                }
            )

            response_time = time.time() - start_time

            # Check if the result is a generation (bytes) or text
            if isinstance(result, bytes):
                # Check for image generation
                if model_used.startswith("gen_image:"):
                    logger.info(f"📸 Generated image for user {user_id} using {model_used}")
                    # Get the refined prompt from the generation context or use the original
                    last_gen = await self.engine.generation_context.get_last_generation(user_id)
                    prompt = last_gen.get('prompt', user_text) if last_gen else user_text
                    style = last_gen.get('style', 'no_style') if last_gen else 'no_style'

                    # Format caption with safe Markdown (no HTML details)
                    caption = self.telegram_formatter.format_generated_image_caption(
                        prompt=prompt,
                        style=style,
                        model=model_used,
                        source="AI"
                    )

                    await placeholder.delete()

                    # Send image with Markdown caption
                    await update.message.reply_photo(
                        photo=result,
                        caption=caption,
                        parse_mode="Markdown",
                        reply_to_message_id=update.message.message_id
                    )

                    # If prompt is long, send the full prompt in a separate message
                    if len(prompt) > 150:
                        full_prompt_msg = self.telegram_formatter.format_full_prompt_message(prompt)
                        await update.message.reply_text(
                            full_prompt_msg,
                            parse_mode="Markdown",
                            reply_to_message_id=update.message.message_id
                        )

                # Check for voice generation (both explicit and conversation mode)
                elif model_used.startswith("gen_voice") or model_used.startswith("gen_voice_conversation"):
                    logger.info(f"🔊 Voice response for user {user_id} using {model_used}")
                    await placeholder.delete()
                    await update.message.reply_voice(
                        voice=result,
                        caption=f"🔊 *Voice Response*\n\n_Listen above_",
                        parse_mode="Markdown",
                        reply_to_message_id=update.message.message_id
                    )

                else:
                    # Unexpected bytes
                    logger.warning(f"Received bytes but model {model_used} not recognized as generation")
                    await placeholder.edit_text(
                        "❌ *Unexpected response format.*\n\n"
                        "Please try again.",
                        parse_mode="Markdown"
                    )
            else:
                # It's text response (analysis or mode switch)
                formatted_response = self.formatter.format_response(result)
                await placeholder.edit_text(formatted_response, parse_mode=Config.TELEGRAM_PARSE_MODE)

                # Save text history only if it's not a mode switch message
                if not model_used.startswith("mode_switch"):
                    # Update memory with the response
                    await self.memory_manager.add_interaction(
                        user_id=user_id,
                        message=user_text,
                        response=result,
                        category="text"
                    )

                    asyncio.create_task(
                        self.user_data_manager.add_message_to_history(
                            user_id=user_id,
                            username=username,
                            message=user_text,
                            response=result,
                            category="text",
                            response_time=response_time,
                            tokens_used=metadata
                        )
                    )

                    self.analytics_engine.record_message(
                        user_id=user_id,
                        category="text",
                        response_time=response_time
                    )
                    logger.info(f"Message processed for user {user_id} in {response_time:.2f}s using {model_used} (tokens: {metadata})")
                else:
                    logger.info(f"Mode switch message for user {user_id}: {result[:50]}...")

        except NetworkError as e:
            logger.error(f"Network error for user {user_id}: {e}")
            await placeholder.edit_text(
                "🌐 *Connection Issue*\n\n"
                "I'm having trouble connecting to the internet. "
                "Please check your connection and try again.",
                parse_mode="Markdown"
            )
        except TimedOut as e:
            logger.error(f"Timeout for user {user_id}: {e}")
            await placeholder.edit_text(
                "⏱️ *Request Timed Out*\n\n"
                "The request took too long to process. "
                "Please try again in a moment.",
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}", exc_info=True)
            await placeholder.edit_text(
                "❌ *Something Went Wrong*\n\n"
                "I encountered an error while processing your message. "
                "Please try again in a moment.",
                parse_mode="Markdown"
            )

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.info("🔵 handle_photo: START")
        if not self.engine.is_initialized:
            await update.message.reply_text("⚠️ Bot initializing...", reply_to_message_id=update.message.message_id)
            return

        user = update.effective_user
        user_id = user.id
        username = user.username
        logger.info(f"🔵 user_id: {user_id}, username: {username}")

        allowed, remaining = await self.rate_limiter.check(user_id)
        if not allowed:
            await self._handle_rate_limit(update, remaining)
            return

        photo = update.message.photo[-1]
        file = await photo.get_file()

        full_file_url = file.file_path
        logger.info(f"📷 Full file URL: {full_file_url}")

        if "/file/" in full_file_url:
            relative_path = full_file_url.split("/file/", 1)[1]
        else:
            relative_path = full_file_url

        proxy_base = self.proxy_manager.get_proxy().rstrip('/')
        proxy_file_url = f"{proxy_base}/file/{relative_path}"
        logger.info(f"🔵 Proxy file URL: {proxy_file_url}")

        image_bytes = None
        proxy_used = None
        try:
            timeout = httpx.Timeout(120.0, connect=20.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for attempt in range(5):
                    try:
                        logger.info(f"🔵 Download attempt {attempt + 1}/5 via proxy")
                        resp = await client.get(proxy_file_url)
                        resp.raise_for_status()
                        image_bytes = resp.content
                        logger.info(f"📦 Downloaded image size: {len(image_bytes)} bytes")
                        proxy_used = proxy_base
                        self.proxy_manager.mark_success(proxy_used)
                        break
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1} failed: {e}")
                        if attempt == 4:
                            if proxy_base == self.proxy_manager.primary:
                                self.proxy_manager.mark_primary_failure()
                            raise
                        await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"❌ All proxy download attempts failed: {e}")
            await update.message.reply_text("❌ Failed to download image. Please try again.", reply_to_message_id=update.message.message_id)
            return

        matrix_info = None
        try:
            matrix_info = await self.user_data_manager.save_image_matrix(user_id, username, image_bytes)
            logger.info(f"💾 Image matrix saved: {matrix_info['file']} ({matrix_info['width']}x{matrix_info['height']}) ratio: {matrix_info['compression_ratio']:.2f}x")
            self.user_data_manager.prune_pictures(user_id, username, max_images=5)
        except Exception as e:
            logger.error(f"Failed to save image matrix: {e}")

        query_text = update.message.caption or ""
        logger.info(f"🔵 query_text: {query_text[:50] if query_text else '(empty)'}...")

        placeholder = await update.message.reply_text("🖼️", reply_to_message_id=update.message.message_id)

        start_time = time.time()
        try:
            logger.info("🔵 Calling engine.process with image bytes")
            response, model_used, tokens_used = await self.engine.process(
                bytes(image_bytes),
                context={'query_text': query_text, 'user_id': user_id, 'username': username, 'input_type': 'image'}
            )
            response_time = time.time() - start_time
            logger.info(f"✅ Vision response using {model_used} (tokens: {tokens_used})")

            if matrix_info:
                await self.user_data_manager.add_image_to_history(
                    user_id=user_id,
                    username=username,
                    query_text=query_text,
                    response=response,
                    matrix_file=matrix_info['file'],
                    width=matrix_info['width'],
                    height=matrix_info['height'],
                    response_time=response_time,
                    tokens_used=tokens_used
                )

            await placeholder.edit_text(response)
            logger.info("🔵 handle_photo: SUCCESS")
        except Exception as e:
            logger.error(f"❌ Vision error: {e}", exc_info=True)
            await placeholder.edit_text(f"❌ *Image analysis failed*\n\n`{str(e)[:200]}`", parse_mode="Markdown")

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.info("🔊 handle_voice: START")
        if not self.engine.is_initialized or not self.voice_engine.is_initialized:
            await update.message.reply_text("⚠️ Bot initializing...", reply_to_message_id=update.message.message_id)
            return

        user = update.effective_user
        user_id = user.id
        username = user.username

        allowed, remaining = await self.rate_limiter.check(user_id)
        if not allowed:
            await self._handle_rate_limit(update, remaining)
            return

        voice = update.message.voice
        file = await voice.get_file()

        full_file_url = file.file_path
        logger.info(f"🔊 Full file URL: {full_file_url}")

        if "/file/" in full_file_url:
            relative_path = full_file_url.split("/file/", 1)[1]
        else:
            relative_path = full_file_url

        proxy_base = self.proxy_manager.get_proxy().rstrip('/')
        proxy_file_url = f"{proxy_base}/file/{relative_path}"
        logger.info(f"🔊 Proxy file URL: {proxy_file_url}")

        audio_bytes = None
        proxy_used = None
        try:
            timeout = httpx.Timeout(120.0, connect=20.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for attempt in range(5):
                    try:
                        logger.info(f"🔊 Download attempt {attempt + 1}/5 via proxy")
                        resp = await client.get(proxy_file_url)
                        resp.raise_for_status()
                        audio_bytes = resp.content
                        logger.info(f"📦 Downloaded audio size: {len(audio_bytes)} bytes")
                        proxy_used = proxy_base
                        self.proxy_manager.mark_success(proxy_used)
                        break
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1} failed: {e}")
                        if attempt == 4:
                            if proxy_base == self.proxy_manager.primary:
                                self.proxy_manager.mark_primary_failure()
                            raise
                        await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"❌ All proxy download attempts failed: {e}")
            await update.message.reply_text("❌ Failed to download voice message. Please try again.", reply_to_message_id=update.message.message_id)
            return

        # Save the audio file
        audio_file_path = await self.user_data_manager.save_audio_file(user_id, username, audio_bytes)
        self.user_data_manager.prune_voices(user_id, username, max_files=5)
        logger.info(f"🔊 Audio saved to {audio_file_path}")

        placeholder = await update.message.reply_text("🔊", reply_to_message_id=update.message.message_id)

        start_time = time.time()
        try:
            # Transcribe the audio
            transcription, voice_model, tokens_used = await self.voice_engine.transcribe(
                audio_bytes,
                context={'user_id': user_id, 'username': username}
            )
            logger.info(f"🔊 Transcription: {transcription[:50]}...")

            # Update memory with the transcription
            await self.memory_manager.add_interaction(
                user_id=user_id,
                message=f"[Voice message] {transcription}",
                response="",
                category="voice"
            )

            # Process the transcription through the main engine
            user_data = await self.user_data_manager.load_user_data(user_id, username)
            history = user_data.get('history', [])

            # Get memory context and preferences
            memory_context = await self.memory_manager.get_context(user_id, limit=5)
            preferences = await self.user_data_manager.get_preferences(user_id, username)

            result, model_used, metadata = await self.engine.process(
                transcription,
                context={
                    'user_id': user_id,
                    'username': username,
                    'history': history,
                    'skip_cache': True,
                    'input_type': 'voice',
                    'preferences': preferences,
                    'memory_context': memory_context
                }
            )

            response_time = time.time() - start_time
            total_tokens = tokens_used + (metadata if isinstance(metadata, int) else 0)

            # Handle response
            if isinstance(result, bytes) and (model_used.startswith("gen_voice") or model_used.startswith("gen_voice_conversation")):
                # Voice response
                logger.info(f"🔊 Voice response to voice message using {model_used}")
                await placeholder.delete()
                await update.message.reply_voice(
                    voice=result,
                    caption=f"🔊 *Voice Response*\n\n_Listen above_",
                    parse_mode="Markdown",
                    reply_to_message_id=update.message.message_id
                )
                # Update memory with the response
                await self.memory_manager.add_interaction(
                    user_id=user_id,
                    message=f"[Voice message] {transcription}",
                    response="[Voice response generated]",
                    category="voice"
                )
            else:
                # Text response (fallback or when mode is text)
                formatted_response = self.formatter.format_response(result if isinstance(result, str) else transcription)
                await placeholder.edit_text(formatted_response, parse_mode=Config.TELEGRAM_PARSE_MODE)

                # Save voice history with text response
                await self.user_data_manager.add_voice_to_history(
                    user_id=user_id,
                    username=username,
                    transcription=transcription,
                    response=result if isinstance(result, str) else "Voice response generated",
                    audio_file=audio_file_path,
                    response_time=response_time,
                    tokens_used=total_tokens
                )

                # Update memory with the response
                await self.memory_manager.add_interaction(
                    user_id=user_id,
                    message=f"[Voice message] {transcription}",
                    response=result if isinstance(result, str) else "Voice response generated",
                    category="voice"
                )

            self.analytics_engine.record_message(
                user_id=user_id,
                category="voice",
                response_time=response_time
            )

            logger.info(f"🔊 Voice handled in {response_time:.2f}s using {voice_model} + {model_used} (tokens: {total_tokens})")

        except Exception as e:
            logger.error(f"❌ Voice error: {e}", exc_info=True)
            await placeholder.edit_text(f"❌ *Voice processing failed*\n\n`{str(e)[:200]}`", parse_mode="Markdown")

    # ========== UTILITY ==========
    async def _send_chunked_message(self, update: Update, text: str):
        chunks = self.formatter.prepare_for_sending(text)
        for i, chunk in enumerate(chunks):
            try:
                await update.message.reply_text(
                    chunk,
                    parse_mode=Config.TELEGRAM_PARSE_MODE,
                    reply_to_message_id=update.message.message_id
                )
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Markdown parse failed, sending plain text: {e}")
                try:
                    await update.message.reply_text(
                        chunk,
                        reply_to_message_id=update.message.message_id
                    )
                except Exception as fallback_error:
                    logger.error(f"Plain text fallback also failed: {fallback_error}")

    async def _handle_rate_limit(self, update: Update, remaining: int):
        text = self.formatter.format_response(
            f"⏳ *Rate Limit Exceeded*\n\n"
            f"Remaining requests: {remaining}\n"
            f"Window: {self.rate_limiter.window_seconds} seconds\n\n"
            f"_Please wait a moment and try again._"
        )
        await self._send_chunked_message(update, text)