import os
import sys
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import NetworkError, TimedOut
from core.config import Config
from core.engines.base_engine import BaseEngine
from core.managers.rate_limiter import RateLimiter
from core.managers.user_data_manager import UserDataManager
from core.managers.proxy_manager import ProxyManager
from core.analytics.analytics_engine import AnalyticsEngine
from handlers.bot_handlers import BotHandlers

for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error
    if isinstance(error, (NetworkError, TimedOut)):
        logger.warning(f"⚠️ Network hiccup (auto-reconnecting): {error}")
    else:
        logger.error("Exception while handling an update:", exc_info=context.error)


def main() -> None:
    if not Config.validate():
        sys.exit(1)

    token = Config.TELEGRAM_TOKEN
    worker_url = Config.WORKER_URL

    logger.info("🚀 Initializing bot components...")

    user_manager = UserDataManager()
    analytics_engine = AnalyticsEngine()
    if Config.ANALYTICS_ENABLED:
        analytics_engine.start()
        logger.info("✅ Analytics engine started (background)")

    rate_limiter = RateLimiter(
        max_requests=Config.RATE_LIMIT_MAX_REQUESTS,
        window_seconds=Config.RATE_LIMIT_WINDOW_SECONDS
    )

    logger.info("🔄 Initializing Base Engine (Text + Vision)...")
    engine = BaseEngine(user_manager)
    engines_ready = asyncio.run(engine.initialize())
    if not engines_ready:
        logger.error("❌ Failed to initialize engines. Exiting.")
        sys.exit(1)

    logger.info("✅ Base Engine initialized successfully")

    proxy_manager = ProxyManager()

    handlers = BotHandlers(engine, rate_limiter, user_manager, analytics_engine, proxy_manager)

    bot_url = worker_url.rstrip('/') + '/bot'
    logger.info(f"🔗 Using proxy URL: {bot_url}")

    application = (
        Application.builder()
            .token(token)
            .base_url(bot_url)
            .connection_pool_size(10)
            .read_timeout(120)
            .write_timeout(120)
            .connect_timeout(60)
            .pool_timeout(60)
            .build()
    )

    logger.info("📋 Registering command handlers...")
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help_command))
    application.add_handler(CommandHandler("about", handlers.about))
    application.add_handler(CommandHandler("status", handlers.status))
    application.add_handler(CommandHandler("clear", handlers.clear_history))
    application.add_handler(CommandHandler("prioritize_text_engine", handlers.prioritize_text_engine))
    application.add_handler(CommandHandler("prioritize_vision_engine", handlers.prioritize_vision_engine))

    logger.info("📋 Registering message handlers...")
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))

    application.add_error_handler(error_handler)

    logger.info("🔄 Starting polling...")
    logger.info("✅ Bot is online and waiting for messages!")
    logger.info(f"⚡ Performance: HTTP/2 enabled (with fallback), 10 connection pool")
    logger.info(f"💾 Cache: TTL={Config.CACHE_TTL_SECONDS}s, similarity={Config.CACHE_SIMILARITY_THRESHOLD}")

    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=False,
            poll_interval=2.5,
            timeout=120
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        raise
    finally:
        logger.info("🛑 Shutting down...")
        analytics_engine.stop()
        asyncio.run(engine.shutdown())


if __name__ == "__main__":
    main()