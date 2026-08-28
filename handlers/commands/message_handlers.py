import logging
import time
import asyncio
from typing import Optional
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.error import NetworkError, TimedOut
from core.config import Config

logger = logging.getLogger(__name__)


class MessageHandlers:
    """Handlers for text, photo, voice, and document messages."""

    # ========== Text message handler ==========
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Check priority setup modes
        if context.user_data.get('setting_image_priority'):
            await self.handle_image_priority_input(update, context)
            return

        if context.user_data.get('setting_priority'):
            await self.handle_priority_input(update, context)
            return

        if not self.engine.is_initialized:
            await update.message.reply_text("️ Bot is still initializing. Please wait.",
                                            reply_to_message_id=update.message.message_id)
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
            await update.message.reply_text("Please send me a text message.",
                                            reply_to_message_id=update.message.message_id)
            return

        # Memory search
        memory_keywords = ["remember", "before", "previous", "past", "earlier", "last time", "what did we",
                           "what did you", "search history"]
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

        # ============================================================
        # INLINE PROCESSING – NO BACKGROUND TASK
        # The webhook will wait for this to complete.
        # ============================================================
        try:
            await self._process_text_message(
                update, context, user_id, username, user_text,
                skip_cache, placeholder, keyboard
            )
        except asyncio.CancelledError:
            # This should not happen with inline processing, but handle it anyway
            logger.info(f"Processing cancelled for user {user_id}")
            await placeholder.edit_text(
                "🛑 *Processing Cancelled*",
                parse_mode="Markdown",
                reply_markup=None
            )
        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}", exc_info=True)
            await placeholder.edit_text(
                "❌ *Something Went Wrong*\n\n"
                "I encountered an error while processing your message. "
                "Please try again in a moment.",
                parse_mode="Markdown",
                reply_markup=None
            )
        finally:
            # Remove from active tasks if it was ever stored (though it's not)
            self._active_tasks.pop((user_id, placeholder_msg_id), None)

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

                await placeholder.edit_text(formatted_response, parse_mode=Config.TELEGRAM_PARSE_MODE,
                                            reply_markup=None)
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
                    logger.info(
                        f"Message processed for user {user_id} in {response_time:.2f}s using {model_used} (tokens: {metadata})")
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

    # ========== Photo handler ==========
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
                await update.message.reply_text("❌ Failed to download image. Please try again.",
                                                reply_to_message_id=update.message.message_id)
                return
        except Exception as e:
            logger.error(f"❌ All proxy download attempts failed: {e}")
            self.proxy_manager.mark_failure(proxy_base)
            await update.message.reply_text("❌ Failed to download image. Please try again.",
                                            reply_to_message_id=update.message.message_id)
            return

        matrix_info = None
        try:
            matrix_info = await self.user_data_manager.save_image_matrix(user_id, username, image_bytes)
            logger.info(
                f"💾 Image matrix saved: {matrix_info['file']} ({matrix_info['width']}x{matrix_info['height']}) ratio: {matrix_info['compression_ratio']:.2f}x")
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

        # Inline processing
        try:
            await self._process_photo(update, user_id, username, query_text, image_bytes, matrix_info, placeholder,
                                      keyboard)
        except asyncio.CancelledError:
            logger.info(f"Photo task cancelled for user {user_id}")
            await placeholder.edit_text(
                "🛑 *Processing Cancelled*",
                parse_mode="Markdown",
                reply_markup=None
            )
        except Exception as e:
            logger.error(f"Photo task error for user {user_id}: {e}", exc_info=True)
            await placeholder.edit_text(
                "❌ *Image analysis failed*\n\nPlease try again.",
                parse_mode="Markdown",
                reply_markup=None
            )
        finally:
            self._active_tasks.pop((user_id, placeholder_msg_id), None)

    async def _process_photo(self, update, user_id, username, query_text, image_bytes, matrix_info, placeholder,
                             keyboard):
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
            await placeholder.edit_text(f"❌ *Image analysis failed*\n\n`{str(e)[:200]}`", parse_mode="Markdown",
                                        reply_markup=None)
            self._active_tasks.pop((user_id, placeholder.message_id), None)

    # ========== Voice handler ==========
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
                await update.message.reply_text("❌ Failed to download voice message. Please try again.",
                                                reply_to_message_id=update.message.message_id)
                return
        except Exception as e:
            logger.error(f"❌ All proxy download attempts failed: {e}")
            self.proxy_manager.mark_failure(proxy_base)
            await update.message.reply_text("❌ Failed to download voice message. Please try again.",
                                            reply_to_message_id=update.message.message_id)
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

        # Inline processing
        try:
            await self._process_voice(update, user_id, username, audio_bytes, audio_file_path, placeholder,
                                      keyboard)
        except asyncio.CancelledError:
            logger.info(f"Voice task cancelled for user {user_id}")
            await placeholder.edit_text(
                "🛑 *Processing Cancelled*",
                parse_mode="Markdown",
                reply_markup=None
            )
        except Exception as e:
            logger.error(f"Voice task error for user {user_id}: {e}", exc_info=True)
            await placeholder.edit_text(
                "❌ *Voice processing failed*\n\nPlease try again.",
                parse_mode="Markdown",
                reply_markup=None
            )
        finally:
            self._active_tasks.pop((user_id, placeholder_msg_id), None)

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

            if isinstance(result, bytes) and (
                    model_used.startswith("gen_voice") or model_used.startswith("gen_voice_conversation")):
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
                formatted_response = self.formatter.format_response(
                    result if isinstance(result, str) else transcription)
                await placeholder.edit_text(formatted_response, parse_mode=Config.TELEGRAM_PARSE_MODE,
                                            reply_markup=None)
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

            logger.info(
                f"🔊 Voice handled in {response_time:.2f}s using {voice_model} + {model_used} (tokens: {total_tokens})")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ Voice error: {e}", exc_info=True)
            await placeholder.edit_text(f"❌ *Voice processing failed*\n\n`{str(e)[:200]}`", parse_mode="Markdown",
                                        reply_markup=None)
            self._active_tasks.pop((user_id, placeholder.message_id), None)

    # ========== Document handler (new) ==========
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle PDF and DOCX document uploads."""
        logger.info("📄 handle_document: START")
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

        document = update.message.document
        file_name = document.file_name or "document"
        file_extension = ""
        if file_name and '.' in file_name:
            file_extension = file_name.split('.')[-1].lower()
            if not file_extension.startswith('.'):
                file_extension = f".{file_extension}"

        # Only support PDF and DOCX
        if file_extension not in ('.pdf', '.docx'):
            await update.message.reply_text(
                f"❌ *Unsupported file type* – only PDF and DOCX are supported.\n"
                f"Received: `{file_extension or 'unknown'}`",
                parse_mode="Markdown",
                reply_to_message_id=update.message.message_id
            )
            return

        # Download file
        file = await document.get_file()
        full_file_url = file.file_path

        if "/file/" in full_file_url:
            relative_path = full_file_url.split("/file/", 1)[1]
        else:
            relative_path = full_file_url

        proxy_base = self.proxy_manager.get_proxy().rstrip('/')
        proxy_file_url = f"{proxy_base}/file/{relative_path}"

        try:
            file_bytes = await self._download_media(proxy_file_url)
            if file_bytes is None:
                await update.message.reply_text("❌ Failed to download document. Please try again.",
                                                reply_to_message_id=update.message.message_id)
                return
        except Exception as e:
            logger.error(f"❌ All proxy download attempts failed: {e}")
            self.proxy_manager.mark_failure(proxy_base)
            await update.message.reply_text("❌ Failed to download document. Please try again.",
                                            reply_to_message_id=update.message.message_id)
            return

        # Show placeholder
        placeholder = await update.message.reply_text(
            "📄",
            reply_to_message_id=update.message.message_id
        )
        placeholder_msg_id = placeholder.message_id

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{user_id}_{placeholder_msg_id}")]
        ])
        await placeholder.edit_reply_markup(reply_markup=keyboard)

        try:
            await self._process_document(update, user_id, username, file_bytes, file_extension,
                                         update.message.caption or "", placeholder, keyboard)
        except asyncio.CancelledError:
            logger.info(f"Document task cancelled for user {user_id}")
            await placeholder.edit_text(
                "🛑 *Processing Cancelled*",
                parse_mode="Markdown",
                reply_markup=None
            )
        except Exception as e:
            logger.error(f"Document task error for user {user_id}: {e}", exc_info=True)
            await placeholder.edit_text(
                "❌ *Document processing failed*\n\nPlease try again.",
                parse_mode="Markdown",
                reply_markup=None
            )
        finally:
            self._active_tasks.pop((user_id, placeholder_msg_id), None)

    async def _process_document(self, update, user_id, username, file_bytes, file_extension,
                                caption, placeholder, keyboard):
        """Process the document with the DocumentEngine."""

        async def status_callback(msg: str, edit: bool = True):
            if edit:
                try:
                    await placeholder.edit_text(msg, parse_mode="Markdown", reply_markup=keyboard)
                except Exception as e:
                    logger.warning(f"Failed to edit placeholder: {e}")

        start_time = time.time()
        try:
            doc_context = {
                'user_id': user_id,
                'username': username,
                'file_extension': file_extension,
                'caption': caption,
                'input_type': 'document',
                'preferences': await self.user_data_manager.get_preferences(user_id, username)
            }

            response, model_used, metadata = await self.engine.process(
                file_bytes,
                context=doc_context,
                status_callback=status_callback
            )

            response_time = time.time() - start_time
            logger.info(f"📄 Document processed using {model_used} (tokens: {metadata})")

            # Use MarkdownStripper for fallback
            from core.utils.markdown_stripper import MarkdownStripper
            stripper = MarkdownStripper()

            if len(response) > Config.DOCUMENT_MAX_TEXT_REPLY_CHARS:
                from telegram import InputFile
                import io
                txt_bytes = response.encode('utf-8')
                txt_io = io.BytesIO(txt_bytes)
                txt_io.name = "document_content.txt"
                await placeholder.delete()
                self._active_tasks.pop((user_id, placeholder.message_id), None)
                await update.message.reply_document(
                    document=InputFile(txt_io, filename="document_content.txt"),
                    caption="📄 *Extracted content (too long for inline)*",
                    parse_mode="Markdown",
                    reply_to_message_id=update.message.message_id
                )
            else:
                formatted = self.formatter.format_response(response)
                try:
                    await placeholder.edit_text(formatted, parse_mode=Config.TELEGRAM_PARSE_MODE, reply_markup=None)
                except Exception as e:
                    # Fallback to plain text if Markdown fails
                    logger.warning(f"Markdown parsing failed: {e}. Sending plain text.")
                    plain_text = stripper.strip(response, remove_emojis=False)
                    await placeholder.edit_text(plain_text, parse_mode=None, reply_markup=None)
                self._active_tasks.pop((user_id, placeholder.message_id), None)

            logger.info(f"📄 Document response sent for user {user_id} in {response_time:.2f}s")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ Document processing error: {e}", exc_info=True)
            await placeholder.edit_text(f"❌ *Document processing failed*\n\n`{str(e)[:200]}`", parse_mode="Markdown",
                                        reply_markup=None)
            self._active_tasks.pop((user_id, placeholder.message_id), None)