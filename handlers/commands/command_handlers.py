import logging
from telegram import Update
from telegram.ext import ContextTypes

from core.config import Config

logger = logging.getLogger(__name__)


class CommandHandlers:
    """Handlers for /start, /help, /about, /status, /clear."""

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        await self.user_data_manager.start_new_session(user.id, user.username)

        # Use getattr to safely access bio (may not exist)
        bio = getattr(user, 'bio', '')
        first_name = getattr(user, 'first_name', '')
        last_name = getattr(user, 'last_name', '')
        phone_number = getattr(user, 'phone_number', '')

        await self.user_data_manager.load_user_info(
            user.id, user.username,
            first_name=first_name,
            last_name=last_name,
            bio=bio,
            phone_number=phone_number
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