import os
import json
import asyncio
from flask import Flask, request, Response
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
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

app = Flask(__name__)

# Global application instance (initialized once)
_telegram_app = None
_handlers = None
_engine = None
_user_manager = None
_voice_engine = None
_rate_limiter = None
_analytics_engine = None
_proxy_manager = None
_health_checker = None
_cache_manager = None


def init_telegram_app():
    """Initialize all components once when the server starts."""
    global _telegram_app, _handlers, _engine, _user_manager, _voice_engine
    global _rate_limiter, _analytics_engine, _proxy_manager, _health_checker, _cache_manager

    if _telegram_app is not None:
        return  # Already initialized

    print("🚀 Initializing bot components...")

    # Core Managers
    _user_manager = UserDataManager()
    _health_checker = HealthChecker()
    _health_checker.start()
    _cache_manager = CacheManager(max_size=5000, default_ttl=300, persistence_dir="cache_data")
    _analytics_engine = AnalyticsEngine()
    if Config.ANALYTICS_ENABLED:
        _analytics_engine.start()
    _rate_limiter = RateLimiter(
        max_requests=Config.RATE_LIMIT_MAX_REQUESTS,
        window_seconds=Config.RATE_LIMIT_WINDOW_SECONDS
    )

    # Base Engine
    print("🔄 Initializing Base Engine...")
    _engine = BaseEngine(_user_manager)
    import asyncio
    engines_ready = asyncio.run(_engine.initialize())
    if not engines_ready:
        raise RuntimeError("Failed to initialize base engines")

    _voice_engine = _engine.voice_engine
    voice_ready = _voice_engine is not None and _voice_engine.is_initialized
    if voice_ready:
        print("✅ Voice Engine reused from Base Engine")
    else:
        print("⚠️ Voice engine not available")

    _proxy_manager = ProxyManager()

    # Handlers
    _handlers = BotHandlers(
        engine=_engine,
        voice_engine=_voice_engine,
        rate_limiter=_rate_limiter,
        user_data_manager=_user_manager,
        analytics_engine=_analytics_engine,
        proxy_manager=_proxy_manager,
        health_checker=_health_checker,
        cache_manager=_cache_manager
    )

    # Build Telegram Application (without updater for webhook)
    token = Config.TELEGRAM_TOKEN
    bot_url = Config.WORKER_URL.rstrip('/') + '/bot'

    _telegram_app = (
        Application.builder()
        .token(token)
        .base_url(bot_url)
        .connection_pool_size(10)
        .build()
    )

    # ---------- Register Handlers ----------
    # Command handlers
    _telegram_app.add_handler(CommandHandler("start", _handlers.start))
    _telegram_app.add_handler(CommandHandler("help", _handlers.help_command))
    _telegram_app.add_handler(CommandHandler("about", _handlers.about))
    _telegram_app.add_handler(CommandHandler("status", _handlers.status))
    _telegram_app.add_handler(CommandHandler("clear", _handlers.clear_history))

    # New consolidated priority and mode commands
    _telegram_app.add_handler(CommandHandler("prioritize", _handlers.prioritize_command))
    _telegram_app.add_handler(CommandHandler("mode", _handlers.mode_command))

    # Existing individual priority commands (kept for backward compatibility)
    _telegram_app.add_handler(CommandHandler("text_engine_priority", _handlers.prioritize_text_engine))
    _telegram_app.add_handler(CommandHandler("vision_engine_priority", _handlers.prioritize_vision_engine))
    _telegram_app.add_handler(CommandHandler("voice_engine_priority", _handlers.prioritize_voice_engine))
    _telegram_app.add_handler(CommandHandler("voice_gen_priority", _handlers.prioritize_voice_generation))
    _telegram_app.add_handler(CommandHandler("image_gen_priority", _handlers.prioritize_image_generation_method))

    # Message handlers
    _telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handlers.handle_message))
    _telegram_app.add_handler(MessageHandler(filters.PHOTO, _handlers.handle_photo))
    if voice_ready:
        _telegram_app.add_handler(MessageHandler(filters.VOICE, _handlers.handle_voice))

    # Callback query handlers
    _telegram_app.add_handler(CallbackQueryHandler(_handlers.cancel_task, pattern="^cancel_"))
    _telegram_app.add_handler(CallbackQueryHandler(_handlers.priority_callback, pattern="^prioritize_"))
    _telegram_app.add_handler(CallbackQueryHandler(_handlers.mode_callback, pattern="^mode_"))

    print("✅ Bot initialized successfully with all handlers (prioritize, mode, cancel)!")


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming Telegram updates via webhook."""
    try:
        # Initialize once
        init_telegram_app()

        # Get the update data
        data = request.get_data(as_text=True)
        if not data:
            return Response('OK', status=200)

        # Process the update asynchronously
        update = Update.de_json(json.loads(data), _telegram_app.bot)

        # Put the update into the application's update queue
        async def process_update():
            async with _telegram_app:
                await _telegram_app.process_update(update)

        asyncio.run(process_update())

        return Response('OK', status=200)

    except Exception as e:
        print(f"Error processing webhook: {e}")
        return Response('OK', status=200)  # Always return 200 to Telegram


@app.route('/', methods=['GET'])
def index():
    return "Telegram bot is running. Webhook endpoint: /webhook"


# Initialize on startup
init_telegram_app()