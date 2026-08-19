import logging
import time
import asyncio
from telegram import Update
from telegram.ext import ContextTypes
from telegram.error import NetworkError, TimedOut
from core.config import Config
from core.llm_client import LLMClient
from core.rate_limiter import RateLimiter
from core.user_data_manager import UserDataManager
from core.analytics_engine import AnalyticsEngine
from utils.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)


class BotHandlers:
    """
    Enhanced bot handlers with:
    - Response formatting and chunking
    - Background analytics integration
    - Smart caching
    - Error handling
    """

    def __init__(self, llm_client: LLMClient, rate_limiter: RateLimiter,
                 user_data_manager: UserDataManager, analytics_engine: AnalyticsEngine):
        self.llm_client = llm_client
        self.rate_limiter = rate_limiter
        self.user_data_manager = user_data_manager
        self.analytics_engine = analytics_engine
        self.formatter = ResponseFormatter()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        user = update.effective_user

        # Start new session
        await self.user_data_manager.start_new_session(user.id)

        text = self.formatter.format_response(
            f"👋 *Welcome, {user.first_name}!*\n\n"
            f"I'm your AI assistant powered by advanced language models.\n\n"
            f"*What I can help with:*\n"
            f"• 💻 Coding & Technical Questions\n"
            f"• 📊 Data Analysis & Insights\n"
            f"• 📚 Learning & Explanations\n"
            f"•  Business & Professional Advice\n"
            f"• ✨ Creative Writing\n\n"
            f"_Just type your question and I'll do my best to help!_"
        )

        await self._send_chunked_message(update, text)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        text = self.formatter.format_response(
            f"📖 *Available Commands:*\n\n"
            f"/start - Start the bot\n"
            f"/help - Show this help message\n"
            f"/about - Learn about the bot\n"
            f"/status - Check your usage\n\n"
            f"💡 *Tips:*\n"
            f"• Be specific in your questions\n"
            f"• Include context for better answers\n"
            f"• Use code blocks for programming questions\n\n"
            f"⌨️ *Markdown Support:*\n"
            f"Use `backticks` for code\n"
            f"Use *asterisks* for bold\n"
            f"Use _underscores_ for italic"
        )

        await self._send_chunked_message(update, text)

    async def about(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /about command"""
        text = self.formatter.format_response(
            f"🧠 *About This Bot*\n\n"
            f"This is an advanced AI chatbot featuring:\n\n"
            f"• 🚀 *Fast Response Times* - Optimized with HTTP/2\n"
            f"•  *Smart Caching* - Instant answers to common questions\n"
            f"• 📊 *Analytics* - Continuous improvement\n"
            f"• 💬 *Context Awareness* - Remembers conversation history\n\n"
            f"*Powered by:*\n"
            f"• DeepSeek & Qwen models\n"
            f"• OpenRouter API\n"
            f"• Python & Telegram Bot API\n\n"
            f"Source: {Config.BOT_REPO_URL}"
        )

        await self._send_chunked_message(update, text)

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command"""
        user_id = update.effective_user.id
        remaining = self.rate_limiter.get_remaining(user_id)

        # Get user analytics
        analytics = self.analytics_engine.get_user_analytics(user_id)

        text = f" *Your Status*\n\n"
        text += f"*Rate Limit:*\n"
        text += f"Remaining: {remaining}/{self.rate_limiter.max_requests}\n"
        text += f"Window: {self.rate_limiter.window_seconds}s\n\n"

        if analytics:
            text += f"*Your Stats:*\n"
            text += f"Total messages: {analytics.total_messages}\n"
            text += f"Preferred category: {analytics.favorite_categories}\n"
            text += f"Most active hours: {analytics.most_active_hours}\n"

        text = self.formatter.format_response(text)
        await self._send_chunked_message(update, text)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages with full processing pipeline"""
        user_id = update.effective_user.id
        user_text = update.message.text

        start_time = time.time()

        # Check rate limit
        allowed, remaining = await self.rate_limiter.check(user_id)
        if not allowed:
            await self._handle_rate_limit(update, remaining)
            return

        if not user_text:
            await update.message.reply_text("Please send me a text message.")
            return

        # Send typing action
        try:
            await update.message.chat.send_action(action="typing")
        except Exception:
            pass  # Ignore typing action errors

        try:
            # Load user data
            user_data = await self.user_data_manager.load_user_data(
                user_id,
                update.effective_user.username
            )
            history = user_data.get('history', [])

            # Get response from LLM (with caching)
            response, model_used, category = await self.llm_client.ask(
                user_text,
                history,
                user_id=user_id
            )

            # Calculate response time
            response_time = time.time() - start_time

            # Format response
            formatted_response = self.formatter.format_response(response)

            # Send chunks
            await self._send_chunked_message(update, formatted_response)

            # Update user history (async, non-blocking)
            asyncio.create_task(
                self.user_data_manager.add_message_to_history(
                    user_id=user_id,
                    message=user_text,
                    response=response,
                    category=category,
                    response_time=response_time,
                    tokens_used=0  # Would need to parse from API response
                )
            )

            # Record analytics (async, non-blocking)
            self.analytics_engine.record_message(
                user_id=user_id,
                category=category,
                response_time=response_time
            )

            logger.info(f"Message processed for user {user_id} in {response_time:.2f}s using {model_used}")

        except NetworkError as e:
            logger.error(f"Network error for user {user_id}: {e}")
            error_text = self.formatter.create_error_response("network", "Connection issue. Please try again.")
            await self._send_chunked_message(update, error_text)

        except TimedOut as e:
            logger.error(f"Timeout for user {user_id}: {e}")
            error_text = self.formatter.create_error_response("timeout", "Request timed out. Please try again.")
            await self._send_chunked_message(update, error_text)

        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}", exc_info=True)
            error_text = self.formatter.create_error_response("unknown", str(e))
            await self._send_chunked_message(update, error_text)

    async def _send_chunked_message(self, update: Update, text: str):
        """Send message in chunks if needed"""
        chunks = self.formatter.prepare_for_sending(text)

        for i, chunk in enumerate(chunks):
            try:
                await update.message.reply_text(
                    chunk,
                    parse_mode=Config.TELEGRAM_PARSE_MODE
                )
                # Small delay between chunks to avoid rate limiting
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to send chunk {i}: {e}")
                # Fallback to plain text
                try:
                    await update.message.reply_text(chunk)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")

    async def _handle_rate_limit(self, update: Update, remaining: int):
        """Handle rate limit exceeded"""
        text = self.formatter.format_response(
            f"⏳ *Rate Limit Exceeded*\n\n"
            f"Remaining requests: {remaining}\n"
            f"Window: {self.rate_limiter.window_seconds} seconds\n\n"
            f"_Please wait a moment and try again._"
        )
        await self._send_chunked_message(update, text)