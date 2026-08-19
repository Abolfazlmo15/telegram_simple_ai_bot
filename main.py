import os
import sys
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import NetworkError, TimedOut  # <-- ADDED IMPORT

from core.config import Config
from core.llm_client import LLMClient
from core.rate_limiter import RateLimiter
from core.user_data_manager import UserDataManager
from core.analytics_engine import AnalyticsEngine
from handlers.bot_handlers import BotHandlers

# =============================================================================
# CRITICAL FIX: Kill system proxies BEFORE importing anything else
# =============================================================================
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

# Logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by Updates, but suppress noisy network disconnects."""
    error = context.error

    # Catch normal network hiccups (Cloudflare drops, proxy blips)
    if isinstance(error, (NetworkError, TimedOut)):
        logger.warning(f"⚠️ Network hiccup (auto-reconnecting): {error}")
    else:
        # For actual code errors, log the full traceback
        logger.error("Exception while handling an update:", exc_info=context.error)
        if update and hasattr(update, 'effective_message') and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ Sorry, an error occurred while processing your message.\n"
                    "Please try again in a moment."
                )
            except Exception:
                pass


def main() -> None:
    """Main entry point with enhanced error handling"""

    # Validate configuration
    if not Config.validate():
        sys.exit(1)

    token = Config.TELEGRAM_TOKEN
    worker_url = Config.WORKER_URL

    # Initialize components
    logger.info("🚀 Initializing bot components...")

    # 1. User Data Manager
    user_manager = UserDataManager()

    # 2. Analytics Engine (background processing)
    analytics_engine = AnalyticsEngine()
    if Config.ANALYTICS_ENABLED:
        analytics_engine.start()
        logger.info("✅ Analytics engine started (background)")

    # 3. Rate Limiter
    rate_limiter = RateLimiter(
        max_requests=Config.RATE_LIMIT_MAX_REQUESTS,
        window_seconds=Config.RATE_LIMIT_WINDOW_SECONDS
    )

    # 4. LLM Client
    llm_client = LLMClient(user_manager)

    # 5. Bot Handlers
    handlers = BotHandlers(llm_client, rate_limiter, user_manager, analytics_engine)

    # Setup bot URL with Cloudflare Worker proxy
    bot_url = worker_url.rstrip('/') + '/bot'
    logger.info(f"🔗 Using proxy URL: {bot_url}")

    # Build application with explicit settings
    application = (
        Application.builder()
            .token(token)
            .base_url(bot_url)
            .connection_pool_size(10)
            .read_timeout(30)
            .write_timeout(30)
            .connect_timeout(30)
            .pool_timeout(30)
            .build()
    )

    # Register command handlers
    logger.info("📋 Registering command handlers...")
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help_command))
    application.add_handler(CommandHandler("about", handlers.about))
    application.add_handler(CommandHandler("status", handlers.status))

    # Register message handler
    logger.info("📋 Registering message handler...")
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))

    # Add error handler
    application.add_error_handler(error_handler)

    # Start polling
    logger.info("🔄 Starting polling...")
    logger.info("✅ Bot is online and waiting for messages!")
    logger.info(f"⚡ Performance: HTTP/2 enabled (with fallback), 10 connection pool")
    logger.info(f"🗄️ Cache: TTL={Config.CACHE_TTL_SECONDS}s, similarity={Config.CACHE_SIMILARITY_THRESHOLD}")

    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=False,
            poll_interval=2.5,
            timeout=30
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        raise
    finally:
        logger.info("🛑 Shutting down...")
        # Cleanup
        analytics_engine.stop()
        llm_client.model_manager.stop()
        asyncio.run(llm_client.close())


if __name__ == "__main__":
    main()