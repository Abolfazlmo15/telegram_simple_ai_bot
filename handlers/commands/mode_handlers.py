import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import ContextTypes
from prompt_engineering.state.conversation_state import ConversationMode

logger = logging.getLogger(__name__)


class ModeHandlers:
    """Handlers for /mode command and mode switching."""

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

            await self.engine.conversation_state.set_mode(user_id, ConversationMode.TEXT)
            await self.user_data_manager.set_response_mode(user_id, "text", username)

            await query.edit_message_text(
                "📝 *Text Mode Activated!*\n\n"
                "I will now respond with text. To switch to voice, use /mode or say 'voice mode'.",
                parse_mode="Markdown"
            )

        else:
            await query.answer("Unknown option.")