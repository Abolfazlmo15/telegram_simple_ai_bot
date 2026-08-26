import logging
import asyncio
from typing import Optional, Dict, Tuple, Any
import httpx
from telegram import Update
from telegram.ext import ContextTypes
from core.config import Config
from core.utils.network import retry_async
from core.utils.response_formatter import ResponseFormatter
from core.managers.memory_manager import MemoryManager
from core.managers.topic_manager import TopicManager
from core.managers.health_checker import HealthChecker
from core.managers.cache_manager import CacheManager
from prompt_engineering.formatters import TelegramFormatter
from prompt_engineering.refiners.context_refiner import ContextRefiner

logger = logging.getLogger(__name__)


class BaseHandler:
    """Base class with shared attributes and utilities."""

    def __init__(self, engine, voice_engine, rate_limiter, user_data_manager,
                 analytics_engine, proxy_manager, health_checker, cache_manager):
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

        # Memory and topic managers
        self.memory_manager = MemoryManager(
            base_dir=Config.USER_DATA_DIR,
            max_short_term=Config.MEMORY_MAX_SHORT_TERM
        )
        self.topic_manager = TopicManager()

        # ContextRefiner
        self.context_refiner = ContextRefiner(
            memory_manager=self.memory_manager,
            topic_manager=self.topic_manager,
            prompt_refiner=self.engine.prompt_refiner
        )

        # Mode detector LLM fallback
        self.engine.mode_detector.set_text_engine(self.engine.text_engine)

        # Active tasks dict for cancellation
        self._active_tasks: Dict[Tuple[int, int], asyncio.Task] = {}

        logger.info("BaseHandler initialized")

    # --- Utility methods ---

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
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
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