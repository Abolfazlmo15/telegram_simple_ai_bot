import logging
import time
import asyncio
from pathlib import Path
from datetime import datetime
import httpx
from telegram import Update
from telegram.ext import ContextTypes
from telegram.error import NetworkError, TimedOut
from core.config import Config
from core.engines.base_engine import BaseEngine
from core.managers.rate_limiter import RateLimiter
from core.managers.user_data_manager import UserDataManager
from core.managers.proxy_manager import ProxyManager
from core.analytics.analytics_engine import AnalyticsEngine
from core.utils.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)


class BotHandlers:
    def __init__(self, engine: BaseEngine, rate_limiter: RateLimiter,
                 user_data_manager: UserDataManager, analytics_engine: AnalyticsEngine,
                 proxy_manager: ProxyManager):
        self.engine = engine
        self.rate_limiter = rate_limiter
        self.user_data_manager = user_data_manager
        self.analytics_engine = analytics_engine
        self.proxy_manager = proxy_manager
        self.formatter = ResponseFormatter()

    # ---------- COMMAND HANDLERS ----------
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        await self.user_data_manager.start_new_session(user.id, user.username)

        # Save user info
        await self.user_data_manager.load_user_info(
            user.id, user.username,
            first_name=user.first_name or "",
            last_name=user.last_name or "",
            bio=user.bio or "",
            phone_number=user.phone_number or ""
        )
        # Save profile photo if available
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
            f"• 🖼️ *Image Analysis* - Just send me a photo!\n\n"
            f"_Just type your question or send an image and I'll do my best to help!_"
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
            f"/prioritize_vision_engine - Set priority for vision models\n\n"
            f"💡 *Tips:*\n"
            f"\n• Be specific in your questions\n"
            f"\n• Include context for better answers\n"
            f"\n• Use code blocks for programming questions\n\n"
            f"\n\n⌨️ *Markdown Support:*\n"
            f"Use `backticks` for code\n"
            f"Use *asterisks* for bold\n"
            f"Use _underscores_ for italic"
        )
        await self._send_chunked_message(update, text)

    async def about(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = self.formatter.format_response(
            f"🧠 *About This Bot*\n\n"
            f"This is an advanced AI chatbot featuring:\n\n"
            f"\n• 🚀 *Fast Response Times* - Optimized with HTTP/2\n"
            f"\n• 💾 *Smart Caching* - Instant answers to common questions\n"
            f"\n• 📊 *Analytics* - Continuous improvement\n"
            f"\n• 💬 *Context Awareness* - Remembers conversation history\n"
            f"\n• 🎯 *Model Priority* - Customize your AI experience\n"
            f"\n• 🖼️ *Vision Capabilities* - Image analysis and description\n\n"
            f"*Powered by:*\n"
            f"\n• DeepSeek, Qwen, and Llama Vision models\n"
            f"\n• OpenRouter API\n"
            f"\n• Python & Telegram Bot API\n\n"
            f"Source: [`{Config.BOT_REPO_URL}`]({Config.BOT_REPO_URL})"
        )
        await self._send_chunked_message(update, text)

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        username = update.effective_user.username
        remaining = self.rate_limiter.get_remaining(user_id)
        stats = await self.user_data_manager.get_user_stats(user_id, username)

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

        text = self.formatter.format_response(text)
        await self._send_chunked_message(update, text)

    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        success = await self.user_data_manager.clear_user_data(user.id, user.username)
        if success:
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

    # ---------- PRIORITY COMMANDS ----------
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

    # ---------- MESSAGE HANDLERS ----------
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

        # Check if user is asking about past conversation
        memory_keywords = ["remember", "before", "previous", "past", "earlier", "last time", "what did we", "what did you", "search history"]
        if any(kw in user_text.lower() for kw in memory_keywords):
            # Try to find relevant history
            history_results = await self.user_data_manager.search_history(user_id, user_text)
            if history_results:
                # Build a context response
                context_text = "I found this in our conversation history:\n\n"
                for entry in history_results[:3]:
                    if entry.get('type') == 'text':
                        context_text += f"• You: {entry.get('message', '')}\n"
                        context_text += f"  Me: {entry.get('response', '')}\n\n"
                    elif entry.get('type') == 'image':
                        context_text += f"• You sent an image with query: {entry.get('query', '')}\n"
                        context_text += f"  Me: {entry.get('response', '')}\n\n"
                context_text += "Is that what you were looking for?"
                await update.message.reply_text(context_text, reply_to_message_id=update.message.message_id)
                return

        # Determine if we should skip cache
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

            response, model_used, tokens_used = await self.engine.process(
                user_text,
                context={'user_id': user_id, 'username': username, 'history': history, 'skip_cache': skip_cache}
            )

            response_time = time.time() - start_time
            formatted_response = self.formatter.format_response(response)
            await placeholder.edit_text(formatted_response, parse_mode=Config.TELEGRAM_PARSE_MODE)

            asyncio.create_task(
                self.user_data_manager.add_message_to_history(
                    user_id=user_id,
                    username=username,
                    message=user_text,
                    response=response,
                    category="text",
                    response_time=response_time,
                    tokens_used=tokens_used
                )
            )

            self.analytics_engine.record_message(
                user_id=user_id,
                category="text",
                response_time=response_time
            )
            logger.info(f"Message processed for user {user_id} in {response_time:.2f}s using {model_used} (tokens: {tokens_used})")

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

        # Save matrix
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
                context={'query_text': query_text, 'user_id': user_id, 'username': username}
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

    # ---------- UTILITY ----------
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