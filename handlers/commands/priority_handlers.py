import logging
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class PriorityHandlers:
    """Handlers for /prioritize command and all priority-related actions."""

    # ========== /prioritize command ==========
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
        """Handle priority selection and the Done button."""
        query: CallbackQuery = update.callback_query
        data = query.data

        # Handle cancellation from priority selection
        if data == "cancel_priority":
            await query.answer()
            await query.edit_message_text("❌ Priority setup cancelled.")
            return

        # Handle the Done button from priority setup
        if data == "priority_done":
            await query.answer()
            user_id = query.from_user.id
            username = query.from_user.username

            priority_list = context.user_data.get('priority_list', [])
            engine = context.user_data.get('engine', 'text')

            if not priority_list:
                await query.edit_message_text(
                    "⚠️ *No models selected.*\n\n"
                    "Please select at least one model before finishing.",
                    parse_mode="Markdown"
                )
                return

            # Save and clear
            await self.user_data_manager.save_model_priority(user_id, username, priority_list, engine=engine)
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
            await query.edit_message_text(text, parse_mode="Markdown")
            return

        # Handle engine selection
        if not data.startswith("prioritize_"):
            await query.answer()
            return

        engine = data.replace("prioritize_", "")
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

        # Close the selection keyboard and show a waiting message
        await query.answer(f"Setting priority for {engine}")
        await query.edit_message_text(
            f"⚙️ *Setting priority for {engine}...*\n\nPlease wait.",
            parse_mode="Markdown"
        )

        # Start the priority setup by sending a new message with instructions
        chat_id = query.message.chat_id
        await self._send_priority_setup_message(chat_id, engine_key, context)

    async def _send_priority_setup_message(self, chat_id: int, engine: str, context: ContextTypes.DEFAULT_TYPE):
        """Send the priority setup instructions with a Done button."""
        if engine == "text":
            available_models = self.engine.text_engine.model_manager.get_fast_models()
        elif engine == "vision":
            available_models = self.engine.vision_engine.model_manager.get_available_models()
        elif engine == "voice":
            available_models = self.engine.voice_engine.openrouter_models
        elif engine == "voice_gen":
            available_models = self.engine.voice_generation_engine.models
        elif engine == "image_gen":
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

        # Store setup state in context.user_data
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
            f"_When you have added all desired models, click the Done button below._"
        )

        # Keyboard with Done button and Cancel button
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Done", callback_data="priority_done")],
            [InlineKeyboardButton("❌ Cancel Priority Setup", callback_data="cancel_priority_setup")]
        ])

        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    # ========== Individual priority commands (legacy) ==========
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

        current_priority = await self.user_data_manager.get_image_generation_priority(user_id, username)

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

    # ========== User text input during priority setup ==========
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

        # ============================================================
        # We still support typing 'done' for backward compatibility
        # ============================================================
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

        # Validate model
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

        # Add model to priority list
        priority_list.append(user_text)
        context.user_data['priority_list'] = priority_list
        context.user_data['current_step'] = step + 1

        # Check if all models are selected
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
            return

        # Send next prompt WITH the Done button
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

        # ============================================================
        # CRITICAL: Include the keyboard with Done button
        # ============================================================
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Done", callback_data="priority_done")],
            [InlineKeyboardButton("❌ Cancel Priority Setup", callback_data="cancel_priority_setup")]
        ])

        await update.message.reply_text(
            text,
            parse_mode="Markdown",
            reply_to_message_id=update.message.message_id,
            reply_markup=keyboard
        )

    # ========== Image priority input (legacy) ==========
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

        seen = set()
        priority_list = [x for x in priority_list if not (x in seen or seen.add(x))]

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

        for tier in valid_tiers:
            if tier not in priority_list:
                priority_list.append(tier)

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