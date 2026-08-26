import os
import sys
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import NetworkError, TimedOut
from core.config import Config
from core.engines.base_engine import BaseEngine
from core.engines.analysis.voice_engine import VoiceEngine
from core.managers.rate_limiter import RateLimiter
from core.managers.user_data_manager import UserDataManager
from core.managers.proxy_manager import ProxyManager
from core.managers.health_checker import HealthChecker
from core.managers.cache_manager import CacheManager
from core.analytics.analytics_engine import AnalyticsEngine
from handlers.bot_handlers import BotHandlers

# Kill system proxies
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

    # ---------- Core Managers ----------
    user_manager = UserDataManager()

    # ---------- Health Checker ----------
    health_checker = HealthChecker()
    health_checker.start()
    logger.info("✅ Health checker started (background)")

    # ---------- Cache Manager ----------
    cache_manager = CacheManager(
        max_size=5000,
        default_ttl=300,
        persistence_dir="cache_data"
    )
    logger.info("✅ Cache manager initialized")

    # ---------- Analytics Engine ----------
    analytics_engine = AnalyticsEngine()
    if Config.ANALYTICS_ENABLED:
        analytics_engine.start()
        logger.info("✅ Analytics engine started (background)")

    # ---------- Rate Limiter ----------
    rate_limiter = RateLimiter(
        max_requests=Config.RATE_LIMIT_MAX_REQUESTS,
        window_seconds=Config.RATE_LIMIT_WINDOW_SECONDS
    )

    # ---------- Base Engine (Text + Vision + Voice) ----------
    logger.info("🔄 Initializing Base Engine (Text + Vision + Voice)...")
    engine = BaseEngine(user_manager)
    engines_ready = asyncio.run(engine.initialize())
    if not engines_ready:
        logger.error("❌ Failed to initialize base engines. Exiting.")
        health_checker.stop()
        analytics_engine.stop()
        sys.exit(1)
    logger.info("✅ Base Engine initialized successfully")

    # ---------- REUSE the voice engine from BaseEngine (AVOID DOUBLE LOAD) ----------
    voice_engine = engine.voice_engine
    voice_ready = voice_engine is not None and voice_engine.is_initialized
    if voice_ready:
        logger.info("✅ Voice Engine reused from Base Engine")
    else:
        logger.warning("⚠️ Voice engine not available – voice messages will be unavailable")

    # ---------- Proxy Manager ----------
    proxy_manager = ProxyManager()

    # ---------- Handlers ----------
    handlers = BotHandlers(
        engine=engine,
        voice_engine=voice_engine,
        rate_limiter=rate_limiter,
        user_data_manager=user_manager,
        analytics_engine=analytics_engine,
        proxy_manager=proxy_manager,
        health_checker=health_checker,
        cache_manager=cache_manager
    )

    # ---------- Telegram Application ----------
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

    # ---------- Command Handlers ----------
    logger.info("📋 Registering command handlers...")
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help_command))
    application.add_handler(CommandHandler("about", handlers.about))
    application.add_handler(CommandHandler("status", handlers.status))
    application.add_handler(CommandHandler("clear", handlers.clear_history))

    # New consolidated priority and mode commands
    application.add_handler(CommandHandler("prioritize", handlers.prioritize_command))
    application.add_handler(CommandHandler("mode", handlers.mode_command))

    # Existing individual priority commands (kept for backward compatibility)
    application.add_handler(CommandHandler("text_engine_priority", handlers.prioritize_text_engine))
    application.add_handler(CommandHandler("vision_engine_priority", handlers.prioritize_vision_engine))
    application.add_handler(CommandHandler("voice_engine_priority", handlers.prioritize_voice_engine))
    application.add_handler(CommandHandler("voice_gen_priority", handlers.prioritize_voice_generation))
    application.add_handler(CommandHandler("image_gen_priority", handlers.prioritize_image_generation_method))

    # ---------- Message Handlers ----------
    logger.info("📋 Registering message handlers...")
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))
    if voice_ready:
        application.add_handler(MessageHandler(filters.VOICE, handlers.handle_voice))
        logger.info("✅ Voice handler registered")
    else:
        logger.warning("⚠️ Voice handler NOT registered (engine unavailable)")

    # ============================================================
    # CALLBACK QUERY HANDLERS
    # ============================================================
    application.add_handler(CallbackQueryHandler(handlers.cancel_task, pattern="^cancel_"))
    application.add_handler(CallbackQueryHandler(handlers.priority_callback, pattern="^prioritize_"))
    application.add_handler(CallbackQueryHandler(handlers.mode_callback, pattern="^mode_"))
    logger.info("✅ All callback handlers registered (cancel, prioritize, mode)")

    # ---------- Error Handler ----------
    application.add_error_handler(error_handler)

    # ---------- Start Polling ----------
    logger.info("🔄 Starting polling...")
    logger.info("✅ Bot is online and waiting for messages!")
    logger.info(f"⚡ Performance: HTTP/2 enabled (with fallback), 10 connection pool")
    logger.info(f"💾 Cache: TTL={Config.CACHE_TTL_SECONDS}s, similarity={Config.CACHE_SIMILARITY_THRESHOLD}")
    logger.info(f"🏥 Health checker: interval={Config.HEALTH_CHECK_INTERVAL_SECONDS}s")
    logger.info(f"📦 Cache manager: max_size=5000, default_ttl=300s")

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

        # Stop background services
        analytics_engine.stop()
        health_checker.stop()

        # Save cache persistence
        cache_manager.save_persistence()

        # Shutdown engines
        asyncio.run(engine.shutdown())
        # voice_engine is already part of engine, so no separate shutdown needed

        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    main()