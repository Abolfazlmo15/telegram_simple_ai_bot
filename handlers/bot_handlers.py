import logging
import time
import asyncio
import re
from typing import Optional, Dict, Tuple

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import ContextTypes, CallbackQueryHandler
from telegram.error import NetworkError, TimedOut
from core.config import Config
from core.engines.base_engine import BaseEngine
from core.engines.analysis.voice_engine import VoiceEngine
from core.managers.rate_limiter import RateLimiter
from core.managers.user_data_manager import UserDataManager
from core.managers.proxy_manager import ProxyManager
from core.analytics.analytics_engine import AnalyticsEngine
from core.utils.network import retry_async
from core.utils.response_formatter import ResponseFormatter

# ============================================================
# Import memory, topic, health, and cache managers
# ============================================================
from core.managers.memory_manager import MemoryManager
from core.managers.topic_manager import TopicManager
from core.managers.health_checker import HealthChecker
from core.managers.cache_manager import CacheManager

# Import TelegramFormatter from prompt_engineering
from prompt_engineering.formatters import TelegramFormatter
from prompt_engineering.refiners.context_refiner import ContextRefiner

logger = logging.getLogger(__name__)


class BotHandlers:
    def __init__(self, engine: BaseEngine, voice_engine: VoiceEngine,
                 rate_limiter: RateLimiter, user_data_manager: UserDataManager,
                 analytics_engine: AnalyticsEngine, proxy_manager: ProxyManager,
                 health_checker: HealthChecker, cache_manager: CacheManager):
        self.engine = engine
        self.voice_engine = voice_engine
        self.rate_limiter = rate_limiter
        self.user_data_manager = user_data_manager
        self.analytics_engine = analytics_engine
        self.proxy_manager = proxy_manager
        self.health_checker = health_checker
        self.cache_manager = cache_manager
        self.formatter = ResponseFormatter()
        self.telegram_formatter = TelegramFormatter()

        # ============================================================
        # Initialize memory and topic managers
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

        # ============================================================
        # Active tasks dictionary for cancellation
        # Key: (user_id, placeholder_message_id) -> asyncio.Task
        # ============================================================
        self._active_tasks: Dict[Tuple[int, int], asyncio.Task] = {}

        logger.info("📋 BotHandlers initialized with Memory, Topic, ContextRefiner, HealthChecker, CacheManager, and Cancellation support")

    # ============================================================
    # CANCEL HANDLER
    # ============================================================
    async def cancel_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the cancel button callback."""
        query: CallbackQuery = update.callback_query
        await query.answer()

        data = query.data
        if not data.startswith("cancel_"):
            return

        try:
            _, user_id_str, msg_id_str = data.split("_")
            user_id = int(user_id_str)
            msg_id = int(msg_id_str)
        except ValueError:
            await query.edit_message_text("❌ Invalid cancellation request.")
            return

        task_key = (user_id, msg_id)
        task = self._active_tasks.pop(task_key, None)

        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Task cancellation error: {e}")

            await query.edit_message_text(
                "🛑 *Processing Cancelled*\n\nYour request has been stopped.",
                parse_mode="Markdown"
            )
        else:
            await query.edit_message_text(
                "✅ *Request Already Completed*\n\nNo action needed.",
                parse_mode="Markdown"
            )

    # ============================================================
    # /PRIORITIZE COMMAND AND CALLBACK
    # ============================================================
    async def prioritize_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send the priority selection keyboard."""
        keyboard = [
            [InlineKeyboardButton("📝 Text Analysis", callback_data="prioritize_text")],
            [InlineKeyboardButton("👁️ Vision Analysis", callback_data="prioritize_vision")],
            [InlineKeyboardButton("🎤 Voice STT", callback_data="prioritize_voice")],
            [InlineKeyboardButton("🔊 Voice Generation", callback_data="prioritize_voice_gen")],
            [InlineKeyboardButton("🖼️ Image Generation", callback_data="prioritize_image_gen")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_priority")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🎯 *Choose the engine you want to prioritize:*\n\n"
            "Select an option below to set your preferred model order for that engine.",
            parse_mode="Markdown",
            reply_markup=reply_markup
        )

    async def priority_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle priority selection from the inline keyboard."""
        query: CallbackQuery = update.callback_query
        data = query.data

        if data == "cancel_priority":
            await query.answer()
            await query.edit_message_text("❌ Priority setup cancelled.")
            return

        if not data.startswith("prioritize_"):
            await query.answer()
            return

        engine = data.replace("prioritize_", "")
        # Map to engine names used in the priority setup
        engine_map = {
            "text": "text",
            "vision": "vision",
            "voice": "voice",
            "voice_gen": "voice_gen",
            "image_gen": "image_gen"
        }

        if engine not in engine_map:
            await query.answer("Unknown engine.")
            return

        engine_key = engine_map[engine]

        # Answer callback and close the keyboard
        await query.answer(f"Setting priority for {engine}")

        # Get the chat_id from the callback
        chat_id = query.message.chat_id
        message_id = query.message.message_id

        # Delete the selection keyboard
        await query.edit_message_text(
            f"⚙️ *Setting priority for {engine}...*\n\nPlease wait.",
            parse_mode="Markdown"
        )

        # Now start the priority setup by sending a new message with the instructions.
        # We'll reuse the existing priority setup logic but we need to simulate a message.
        # We'll send a new message and call the internal helper.
        await self._send_priority_setup_message(chat_id, engine_key, context)

    async def _send_priority_setup_message(self, chat_id: int, engine: str, context: ContextTypes.DEFAULT_TYPE):
        """Send the priority setup instructions for a given engine."""
        # We need the available models list
        if engine == "text":
            available_models = self.engine.text_engine.model_manager.get_fast_models()
        elif engine == "vision":
            available_models = self.engine.vision_engine.model_manager.get_available_models()
        elif engine == "voice":
            available_models = self.engine.voice_engine.openrouter_models
        elif engine == "voice_gen":
            available_models = self.engine.voice_generation_engine.models
        elif engine == "image_gen":
            # For image generation, we use the priority method (tiers), not model list
            # We'll handle it separately via the existing method
            # Redirect to the existing command
            await context.bot.send_message(
                chat_id=chat_id,
                text="🖼️ *Image Generation Priority Setup*\n\n"
                     "Please use the `/image_gen_priority` command for image generation tiers.\n\n"
                     "The `/prioritize` menu currently supports model priority for text, vision, voice STT, and voice generation.\n\n"
                     "_We are working to integrate image generation tiers into this menu soon._",
                parse_mode="Markdown"
            )
            return
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Unknown engine type.",
                parse_mode="Markdown"
            )
            return

        if not available_models:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ *Error*\n\nNo models available right now. Please try again later.",
                parse_mode="Markdown"
            )
            return

        # Store priority setup state in context user_data
        # Since this is a callback, we need to store in the user_data
        # We'll use the context.user_data dict (it's shared across all messages for this user)
        # But we need to ensure we have the right chat_id; we can store under a key.
        # We'll use a combination: set context.user_data['setting_priority'] = True etc.
        # However, the main handler `handle_message` checks context.user_data for these flags.
        # So we can set them here.
        context.user_data['setting_priority'] = True
        context.user_data['priority_list'] = []
        context.user_data['available_models'] = available_models
        context.user_data['current_step'] = 1
        context.user_data['engine'] = engine

        engine_display_names = {
            "text": "Text Analysis",
            "vision": "Vision Analysis",
            "voice": "Voice STT (Speech-to-Text)",
            "voice_gen": "Voice Generation (TTS)"
        }
        display_name = engine_display_names.get(engine, engine.title())

        model_list = "\n".join([f"{i + 1}. `{m}`" for i, m in enumerate(available_models[:10])])
        if len(available_models) > 10:
            model_list += f"\n... and {len(available_models) - 10} more"

        text = (
            f"🎯 *{display_name} Priority Setup*\n\n"
            f"*Available models:*\n{model_list}\n\n"
            f"Let's set your priority. Which model should be *#1*?\n\n"
            f"_Type the exact model name (e.g., deepseek/deepseek-chat:free)_\n"
            f"_Or type /cancel to cancel_"
        )

        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown"
        )

        # Also send a cancel option
        cancel_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel Priority Setup", callback_data="cancel_priority_setup")]
        ])
        await context.bot.send_message(
            chat_id=chat_id,
            text="_Click below to cancel the priority setup._",
            parse_mode="Markdown",
            reply_markup=cancel_keyboard
        )

    # ============================================================
    # /MODE COMMAND AND CALLBACK
    # ============================================================
    async def mode_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send the mode selection keyboard."""
        keyboard = [
            [InlineKeyboardButton("🗣️ Voice Mode", callback_data="mode_voice")],
            [InlineKeyboardButton("📝 Text Mode", callback_data="mode_text")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_mode")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🎤 *Select your preferred response mode:*\n\n"
            "• *Voice Mode* – I will respond with voice messages.\n"
            "• *Text Mode* – I will respond with text.\n\n"
            "_You can also switch modes by saying 'talk to me' or 'type it'._",
            parse_mode="Markdown",
            reply_markup=reply_markup
        )

    async def mode_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle mode selection from the inline keyboard."""
        query: CallbackQuery = update.callback_query
        data = query.data

        if data == "cancel_mode":
            await query.answer()
            await query.edit_message_text("❌ Mode selection cancelled.")
            return

        if data == "mode_voice":
            await query.answer("Switching to Voice Mode")
            user_id = query.from_user.id
            username = query.from_user.username

            # Set mode in conversation state
            from prompt_engineering.state.conversation_state import ConversationMode
            await self.engine.conversation_state.set_mode(user_id, ConversationMode.VOICE)
            await self.user_data_manager.set_response_mode(user_id, "voice", username)

            await query.edit_message_text(
                "🗣️ *Voice Mode Activated!*\n\n"
                "I will now speak my responses. To switch back to text, use /mode or say 'text mode'.",
                parse_mode="Markdown"
            )

        elif data == "mode_text":
            await query.answer("Switching to Text Mode")
            user_id = query.from_user.id
            username = query.from_user.username

            from prompt_engineering.state.conversation_state import ConversationMode
            await self.engine.conversation_state.set_mode(user_id, ConversationMode.TEXT)
            await self.user_data_manager.set_response_mode(user_id, "text", username)

            await query.edit_message_text(
                "📝 *Text Mode Activated!*\n\n"
                "I will now respond with text. To switch to voice, use /mode or say 'voice mode'.",
                parse_mode="Markdown"
            )

        else:
            await query.answer("Unknown option.")

    # ============================================================
    # EXISTING PRIORITY COMMANDS (kept for backward compatibility)
    # ============================================================
    async def prioritize_text_engine(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._start_priority_setup(update, context, engine="text")

    async def prioritize_vision_engine(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._start_priority_setup(update, context, engine="vision")

    async def prioritize_voice_engine(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._start_priority_setup(update, context, engine="voice")

    async def prioritize_voice_generation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._start_priority_setup(update, context, engine="voice_gen")

    async def prioritize_image_generation_method(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Set priority order for image generation tiers (existing method)."""
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

    async def _start_priority_setup(self, update: Update, context: ContextTypes.DEFAULT_TYPE, engine: str) -> None:
        """Legacy entry point – redirects to the new helper."""
        chat_id = update.effective_chat.id
        await self._send_priority_setup_message(chat_id, engine, context)

    # ============================================================
    # HANDLE PRIORITY INPUT FROM USER (text messages)
    # ============================================================
    async def handle_priority_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle user's text input during priority setup."""
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
            display_name = {
                "text": "Text Analysis",
                "vision": "Vision Analysis",
                "voice": "Voice STT",
                "voice_gen": "Voice Generation"
            }.get(engine, engine.title())

            text = (
                f"✅ *Priority Saved for {display_name}!*\n\n"
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
            display_name = {
                "text": "Text Analysis",
                "vision": "Vision Analysis",
                "voice": "Voice STT",
                "voice_gen": "Voice Generation"
            }.get(engine, engine.title())

            text = (
                f"✅ *Priority Saved for {display_name}!*\n\n"
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

    # ============================================================
    # HANDLE IMAGE PRIORITY INPUT (existing)
    # ============================================================
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

    # ============================================================
    # COMMAND HANDLERS (start, help, about, status, clear)
    # ============================================================
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
            f"/prioritize - Set model priorities for all engines\n"
            f"/mode - Switch between Voice and Text modes\n\n"
            f"🎯 *Priority Commands (advanced):*\n"
            f"/text_engine_priority - Set priority for Text models\n"
            f"/vision_engine_priority - Set priority for Vision models\n"
            f"/voice_engine_priority - Set priority for Voice STT models\n"
            f"/voice_gen_priority - Set priority for Voice Generation (TTS) models\n"
            f"/image_gen_priority - Set priority for Image Generation tiers\n\n"
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
            f"• 🚀 *Fast Response Times* - Optimized with HTTP/2, parallel testing, and health checks\n"
            f"• 💾 *Smart Caching* - Instant answers to common questions\n"
            f"• 📊 *Analytics* - Continuous improvement\n"
            f"• 💬 *Context Awareness* - Remembers conversation history\n"
            f"• 🧠 *Memory* - Long-term and short-term memory for better context\n"
            f"• 🎯 *Model Priority* - Customize your AI experience for all engines\n"
            f"• 🖼️ *Vision Capabilities* - Image analysis and description\n"
            f"• 🎤 *Voice Transcription* - Send a voice note, get a reply\n"
            f"• 🎨 *Image Generation* - Generate images from text\n"
            f"• 🔊 *Voice Generation* - Text-to-speech\n"
            f"• 🗣️ *Voice Mode* - Talk to me and I'll respond in voice\n"
            f"• 📊 *Topic Tracking* - I understand what topics we're discussing\n"
            f"• 🏥 *Health Checking* - Auto-detects failing models to skip them\n\n"
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

        short_term = await self.memory_manager.get_short_term(user_id)
        long_term = await self.memory_manager.get_long_term(user_id)

        text_priority = await self.user_data_manager.get_user_model_priority(user_id, username, "text")
        vision_priority = await self.user_data_manager.get_user_model_priority(user_id, username, "vision")
        voice_priority = await self.user_data_manager.get_user_model_priority(user_id, username, "voice")
        voice_gen_priority = await self.user_data_manager.get_user_model_priority(user_id, username, "voice_gen")
        image_priority = await self.user_data_manager.get_image_generation_priority(user_id, username)

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

        text += f"\n*Priority Stats:*\n"
        text += f"• Text: {text_priority[0] if text_priority else 'Default'}\n"
        text += f"• Vision: {vision_priority[0] if vision_priority else 'Default'}\n"
        text += f"• Voice STT: {voice_priority[0] if voice_priority else 'Default'}\n"
        text += f"• Voice Gen: {voice_gen_priority[0] if voice_gen_priority else 'Default'}\n"
        text += f"• Image Gen: {image_priority[0] if image_priority else 'Default'}"

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

    # ============================================================
    # MESSAGE HANDLERS (text, photo, voice)
    # ============================================================
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

        # Send placeholder with cancel button
        placeholder = await update.message.reply_text(
            "🤔",
            reply_to_message_id=update.message.message_id
        )
        placeholder_msg_id = placeholder.message_id

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{user_id}_{placeholder_msg_id}")]
        ])
        await placeholder.edit_reply_markup(reply_markup=keyboard)

        async def process_task():
            try:
                await self._process_text_message(update, context, user_id, username, user_text, skip_cache, placeholder, keyboard)
            except asyncio.CancelledError:
                logger.info(f"Task cancelled for user {user_id}")
                raise
            except Exception as e:
                logger.error(f"Task error for user {user_id}: {e}", exc_info=True)
                try:
                    await placeholder.edit_text(
                        "❌ *Something Went Wrong*\n\n"
                        "I encountered an error while processing your message. "
                        "Please try again in a moment.",
                        parse_mode="Markdown",
                        reply_markup=None
                    )
                except Exception:
                    pass
            finally:
                self._active_tasks.pop((user_id, placeholder_msg_id), None)

        task = asyncio.create_task(process_task())
        self._active_tasks[(user_id, placeholder_msg_id)] = task

    async def _process_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                    user_id: int, username: str, user_text: str,
                                    skip_cache: bool, placeholder, keyboard):
        try:
            await update.message.chat.send_action(action="typing")
        except Exception:
            pass

        start_time = time.time()
        try:
            user_data = await self.user_data_manager.load_user_data(user_id, username)
            history = user_data.get('history', [])

            await self.memory_manager.add_interaction(
                user_id=user_id,
                message=user_text,
                response="",
                category="text"
            )

            detected_topic = await self.topic_manager.add_message(user_id, user_text)
            if detected_topic:
                logger.info(f"📊 Topic detected: {detected_topic}")

            memory_context = await self.memory_manager.get_context(user_id, limit=5)

            preferences = await self.user_data_manager.get_preferences(user_id, username)
            custom_instructions = await self.user_data_manager.get_custom_instructions(user_id, username)

            async def status_callback(msg: str, edit: bool = True):
                if edit:
                    try:
                        await placeholder.edit_text(msg, parse_mode="Markdown", reply_markup=keyboard)
                    except Exception as e:
                        logger.warning(f"Failed to edit placeholder: {e}")

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
                },
                status_callback=status_callback
            )

            response_time = time.time() - start_time

            if isinstance(result, bytes):
                if model_used.startswith("gen_image:"):
                    logger.info(f"📸 Generated image for user {user_id} using {model_used}")
                    last_gen = await self.engine.generation_context.get_last_generation(user_id)
                    prompt = last_gen.get('prompt', user_text) if last_gen else user_text
                    style = last_gen.get('style', 'no_style') if last_gen else 'no_style'

                    caption = self.telegram_formatter.format_generated_image_caption(
                        prompt=prompt,
                        style=style,
                        model=model_used,
                        source="AI"
                    )

                    await placeholder.delete()
                    self._active_tasks.pop((user_id, placeholder.message_id), None)

                    await update.message.reply_photo(
                        photo=result,
                        caption=caption,
                        parse_mode="Markdown",
                        reply_to_message_id=update.message.message_id
                    )

                    if len(prompt) > 150:
                        full_prompt_msg = self.telegram_formatter.format_full_prompt_message(prompt)
                        await update.message.reply_text(
                            full_prompt_msg,
                            parse_mode="Markdown",
                            reply_to_message_id=update.message.message_id
                        )

                elif model_used.startswith("gen_voice") or model_used.startswith("gen_voice_conversation"):
                    logger.info(f"🔊 Voice response for user {user_id} using {model_used}")
                    await placeholder.delete()
                    self._active_tasks.pop((user_id, placeholder.message_id), None)
                    await update.message.reply_voice(
                        voice=result,
                        caption=f"🔊 *Voice Response*\n\n_Listen above_",
                        parse_mode="Markdown",
                        reply_to_message_id=update.message.message_id
                    )

                else:
                    logger.warning(f"Received bytes but model {model_used} not recognized as generation")
                    await placeholder.edit_text(
                        "❌ *Unexpected response format.*\n\n"
                        "Please try again.",
                        parse_mode="Markdown",
                        reply_markup=None
                    )
                    self._active_tasks.pop((user_id, placeholder.message_id), None)
            else:
                if not result or not result.strip():
                    result = "I couldn't generate a response. Please try again."
                formatted_response = self.formatter.format_response(result)

                await placeholder.edit_text(formatted_response, parse_mode=Config.TELEGRAM_PARSE_MODE, reply_markup=None)
                self._active_tasks.pop((user_id, placeholder.message_id), None)

                if not model_used.startswith("mode_switch"):
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

        except asyncio.CancelledError:
            logger.info(f"Processing cancelled for user {user_id}")
            raise
        except NetworkError as e:
            logger.error(f"Network error for user {user_id}: {e}")
            self.proxy_manager.mark_failure(self.proxy_manager.current_proxy)
            await placeholder.edit_text(
                "🌐 *Connection Issue*\n\n"
                "I'm having trouble connecting to the internet. "
                "Please check your connection and try again.",
                parse_mode="Markdown",
                reply_markup=None
            )
            self._active_tasks.pop((user_id, placeholder.message_id), None)
        except TimedOut as e:
            logger.error(f"Timeout for user {user_id}: {e}")
            self.proxy_manager.mark_failure(self.proxy_manager.current_proxy)
            await placeholder.edit_text(
                "⏱️ *Request Timed Out*\n\n"
                "The request took too long to process. "
                "Please try again in a moment.",
                parse_mode="Markdown",
                reply_markup=None
            )
            self._active_tasks.pop((user_id, placeholder.message_id), None)
        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}", exc_info=True)
            await placeholder.edit_text(
                "❌ *Something Went Wrong*\n\n"
                "I encountered an error while processing your message. "
                "Please try again in a moment.",
                parse_mode="Markdown",
                reply_markup=None
            )
            self._active_tasks.pop((user_id, placeholder.message_id), None)

    # ============================================================
    # PHOTO HANDLER
    # ============================================================
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.info("🔵 handle_photo: START")
        if not self.engine.is_initialized:
            await update.message.reply_text("⚠️ Bot initializing...", reply_to_message_id=update.message.message_id)
            return

        user = update.effective_user
        user_id = user.id
        username = user.username

        allowed, remaining = await self.rate_limiter.check(user_id)
        if not allowed:
            await self._handle_rate_limit(update, remaining)
            return

        photo = update.message.photo[-1]
        file = await photo.get_file()
        full_file_url = file.file_path

        if "/file/" in full_file_url:
            relative_path = full_file_url.split("/file/", 1)[1]
        else:
            relative_path = full_file_url

        proxy_base = self.proxy_manager.get_proxy().rstrip('/')
        proxy_file_url = f"{proxy_base}/file/{relative_path}"

        try:
            image_bytes = await self._download_media(proxy_file_url)
            if image_bytes is None:
                await update.message.reply_text("❌ Failed to download image. Please try again.", reply_to_message_id=update.message.message_id)
                return
        except Exception as e:
            logger.error(f"❌ All proxy download attempts failed: {e}")
            self.proxy_manager.mark_failure(proxy_base)
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

        placeholder = await update.message.reply_text(
            "🖼️",
            reply_to_message_id=update.message.message_id
        )
        placeholder_msg_id = placeholder.message_id

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{user_id}_{placeholder_msg_id}")]
        ])
        await placeholder.edit_reply_markup(reply_markup=keyboard)

        async def process_photo_task():
            try:
                await self._process_photo(update, user_id, username, query_text, image_bytes, matrix_info, placeholder, keyboard)
            except asyncio.CancelledError:
                logger.info(f"Photo task cancelled for user {user_id}")
                raise
            except Exception as e:
                logger.error(f"Photo task error for user {user_id}: {e}", exc_info=True)
                try:
                    await placeholder.edit_text(
                        "❌ *Image analysis failed*\n\nPlease try again.",
                        parse_mode="Markdown",
                        reply_markup=None
                    )
                except Exception:
                    pass
            finally:
                self._active_tasks.pop((user_id, placeholder_msg_id), None)

        task = asyncio.create_task(process_photo_task())
        self._active_tasks[(user_id, placeholder_msg_id)] = task

    async def _process_photo(self, update, user_id, username, query_text, image_bytes, matrix_info, placeholder, keyboard):
        async def status_callback(msg: str, edit: bool = True):
            if edit:
                try:
                    await placeholder.edit_text(msg, parse_mode="Markdown", reply_markup=keyboard)
                except Exception as e:
                    logger.warning(f"Failed to edit placeholder: {e}")

        start_time = time.time()
        try:
            logger.info("🔵 Calling engine.process with image bytes")
            vision_priority = await self.user_data_manager.get_user_model_priority(user_id, username, "vision")
            response, model_used, tokens_used = await self.engine.process(
                bytes(image_bytes),
                context={
                    'query_text': query_text,
                    'user_id': user_id,
                    'username': username,
                    'input_type': 'image',
                    'priority_list': vision_priority
                },
                status_callback=status_callback
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

            await placeholder.edit_text(response, parse_mode="Markdown", reply_markup=None)
            self._active_tasks.pop((user_id, placeholder.message_id), None)
            logger.info("🔵 handle_photo: SUCCESS")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ Vision error: {e}", exc_info=True)
            await placeholder.edit_text(f"❌ *Image analysis failed*\n\n`{str(e)[:200]}`", parse_mode="Markdown", reply_markup=None)
            self._active_tasks.pop((user_id, placeholder.message_id), None)

    # ============================================================
    # VOICE HANDLER
    # ============================================================
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

        if "/file/" in full_file_url:
            relative_path = full_file_url.split("/file/", 1)[1]
        else:
            relative_path = full_file_url

        proxy_base = self.proxy_manager.get_proxy().rstrip('/')
        proxy_file_url = f"{proxy_base}/file/{relative_path}"

        try:
            audio_bytes = await self._download_media(proxy_file_url)
            if audio_bytes is None:
                await update.message.reply_text("❌ Failed to download voice message. Please try again.", reply_to_message_id=update.message.message_id)
                return
        except Exception as e:
            logger.error(f"❌ All proxy download attempts failed: {e}")
            self.proxy_manager.mark_failure(proxy_base)
            await update.message.reply_text("❌ Failed to download voice message. Please try again.", reply_to_message_id=update.message.message_id)
            return

        audio_file_path = await self.user_data_manager.save_audio_file(user_id, username, audio_bytes)
        self.user_data_manager.prune_voices(user_id, username, max_files=5)

        placeholder = await update.message.reply_text(
            "🔊",
            reply_to_message_id=update.message.message_id
        )
        placeholder_msg_id = placeholder.message_id

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{user_id}_{placeholder_msg_id}")]
        ])
        await placeholder.edit_reply_markup(reply_markup=keyboard)

        async def process_voice_task():
            try:
                await self._process_voice(update, user_id, username, audio_bytes, audio_file_path, placeholder, keyboard)
            except asyncio.CancelledError:
                logger.info(f"Voice task cancelled for user {user_id}")
                raise
            except Exception as e:
                logger.error(f"Voice task error for user {user_id}: {e}", exc_info=True)
                try:
                    await placeholder.edit_text(
                        "❌ *Voice processing failed*\n\nPlease try again.",
                        parse_mode="Markdown",
                        reply_markup=None
                    )
                except Exception:
                    pass
            finally:
                self._active_tasks.pop((user_id, placeholder_msg_id), None)

        task = asyncio.create_task(process_voice_task())
        self._active_tasks[(user_id, placeholder_msg_id)] = task

    async def _process_voice(self, update, user_id, username, audio_bytes, audio_file_path, placeholder, keyboard):
        async def status_callback(msg: str, edit: bool = True):
            if edit:
                try:
                    await placeholder.edit_text(msg, parse_mode="Markdown", reply_markup=keyboard)
                except Exception as e:
                    logger.warning(f"Failed to edit placeholder: {e}")

        start_time = time.time()
        try:
            voice_priority = await self.user_data_manager.get_user_model_priority(user_id, username, "voice")
            transcription, voice_model, tokens_used = await self.voice_engine.transcribe(
                audio_bytes,
                context={'user_id': user_id, 'username': username, 'priority_list': voice_priority},
                status_callback=status_callback
            )
            logger.info(f"🔊 Transcription: {transcription[:50]}...")

            await self.memory_manager.add_interaction(
                user_id=user_id,
                message=f"[Voice message] {transcription}",
                response="",
                category="voice"
            )

            user_data = await self.user_data_manager.load_user_data(user_id, username)
            history = user_data.get('history', [])

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
                },
                status_callback=status_callback
            )

            response_time = time.time() - start_time
            total_tokens = tokens_used + (metadata if isinstance(metadata, int) else 0)

            if isinstance(result, bytes) and (model_used.startswith("gen_voice") or model_used.startswith("gen_voice_conversation")):
                logger.info(f"🔊 Voice response to voice message using {model_used}")
                await placeholder.delete()
                self._active_tasks.pop((user_id, placeholder.message_id), None)
                await update.message.reply_voice(
                    voice=result,
                    caption=f"🔊 *Voice Response*\n\n_Listen above_",
                    parse_mode="Markdown",
                    reply_to_message_id=update.message.message_id
                )
                await self.memory_manager.add_interaction(
                    user_id=user_id,
                    message=f"[Voice message] {transcription}",
                    response="[Voice response generated]",
                    category="voice"
                )
            else:
                formatted_response = self.formatter.format_response(result if isinstance(result, str) else transcription)
                await placeholder.edit_text(formatted_response, parse_mode=Config.TELEGRAM_PARSE_MODE, reply_markup=None)
                self._active_tasks.pop((user_id, placeholder.message_id), None)

                await self.user_data_manager.add_voice_to_history(
                    user_id=user_id,
                    username=username,
                    transcription=transcription,
                    response=result if isinstance(result, str) else "Voice response generated",
                    audio_file=audio_file_path,
                    response_time=response_time,
                    tokens_used=total_tokens
                )

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

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ Voice error: {e}", exc_info=True)
            await placeholder.edit_text(f"❌ *Voice processing failed*\n\n`{str(e)[:200]}`", parse_mode="Markdown", reply_markup=None)
            self._active_tasks.pop((user_id, placeholder.message_id), None)

    # ============================================================
    # UTILITY METHODS
    # ============================================================
    async def _download_media(self, url: str, max_attempts: int = 5) -> Optional[bytes]:
        last_error = None
        for attempt in range(max_attempts):
            try:
                async def download():
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        return resp.content

                content = await retry_async(
                    download,
                    max_attempts=2,
                    base_delay=0.5,
                    on_retry=lambda e, a: logger.warning(f"Download retry {a}: {e}")
                )
                if content:
                    logger.info(f"📦 Downloaded media size: {len(content)} bytes")
                    self.proxy_manager.mark_success(self.proxy_manager.current_proxy)
                    return content
            except Exception as e:
                last_error = e
                logger.warning(f"Download attempt {attempt+1} failed: {e}")
                self.proxy_manager.mark_failure(self.proxy_manager.current_proxy)
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                continue
        logger.error(f"All download attempts failed: {last_error}")
        return None

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