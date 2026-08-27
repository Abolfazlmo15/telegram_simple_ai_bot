import os
import sys
import json
import asyncio
import traceback
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
# Import the new combined BotHandlers
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
        print("✅ Bot already initialized, reusing.")
        return

    print("🚀 Initializing bot components...")

    # Core Managers
    print("📦 Creating UserDataManager...")
    _user_manager = UserDataManager()
    print("✅ UserDataManager created.")

    print("🏥 Creating HealthChecker...")
    _health_checker = HealthChecker()
    _health_checker.start()
    print("✅ HealthChecker started.")

    print("📦 Creating CacheManager...")
    _cache_manager = CacheManager(max_size=5000, default_ttl=300, persistence_dir="cache_data")
    print("✅ CacheManager created.")

    print("📊 Creating AnalyticsEngine...")
    _analytics_engine = AnalyticsEngine()
    if Config.ANALYTICS_ENABLED:
        _analytics_engine.start()
        print("✅ AnalyticsEngine started.")
    else:
        print("ℹ️ AnalyticsEngine disabled by config.")

    print("⏱️ Creating RateLimiter...")
    _rate_limiter = RateLimiter(
        max_requests=Config.RATE_LIMIT_MAX_REQUESTS,
        window_seconds=Config.RATE_LIMIT_WINDOW_SECONDS
    )
    print("✅ RateLimiter created.")

    # Base Engine
    print("🔄 Initializing Base Engine...")
    _engine = BaseEngine(_user_manager)
    import asyncio
    engines_ready = asyncio.run(_engine.initialize())
    if not engines_ready:
        raise RuntimeError("Failed to initialize base engines")
    print("✅ Base Engine initialized.")

    _voice_engine = _engine.voice_engine
    voice_ready = _voice_engine is not None and _voice_engine.is_initialized
    if voice_ready:
        print("✅ Voice Engine reused from Base Engine")
    else:
        print("⚠️ Voice engine not available")

    print("🔗 Creating ProxyManager...")
    _proxy_manager = ProxyManager()
    print("✅ ProxyManager created.")

    # Handlers
    print("📋 Creating BotHandlers...")
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
    print("✅ BotHandlers created.")

    # Build Telegram Application (without updater for webhook)
    token = Config.TELEGRAM_TOKEN
    bot_url = Config.WORKER_URL.rstrip('/') + '/bot'
    print(f"🔗 Telegram bot URL: {bot_url}")

    _telegram_app = (
        Application.builder()
        .token(token)
        .base_url(bot_url)
        .connection_pool_size(10)
        .build()
    )
    print("✅ Telegram Application built.")

    # ---------- Register Handlers ----------
    print("📝 Registering command handlers...")
    _telegram_app.add_handler(CommandHandler("start", _handlers.start))
    _telegram_app.add_handler(CommandHandler("help", _handlers.help_command))
    _telegram_app.add_handler(CommandHandler("about", _handlers.about))
    _telegram_app.add_handler(CommandHandler("status", _handlers.status))
    _telegram_app.add_handler(CommandHandler("clear", _handlers.clear_history))

    _telegram_app.add_handler(CommandHandler("prioritize", _handlers.prioritize_command))
    _telegram_app.add_handler(CommandHandler("mode", _handlers.mode_command))

    _telegram_app.add_handler(CommandHandler("text_engine_priority", _handlers.prioritize_text_engine))
    _telegram_app.add_handler(CommandHandler("vision_engine_priority", _handlers.prioritize_vision_engine))
    _telegram_app.add_handler(CommandHandler("voice_engine_priority", _handlers.prioritize_voice_engine))
    _telegram_app.add_handler(CommandHandler("voice_gen_priority", _handlers.prioritize_voice_generation))
    _telegram_app.add_handler(CommandHandler("image_gen_priority", _handlers.prioritize_image_generation_method))
    print("✅ Command handlers registered.")

    print("📝 Registering message handlers...")
    _telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handlers.handle_message))
    _telegram_app.add_handler(MessageHandler(filters.PHOTO, _handlers.handle_photo))
    if voice_ready:
        _telegram_app.add_handler(MessageHandler(filters.VOICE, _handlers.handle_voice))
        print("✅ Voice message handler registered.")
    else:
        print("⚠️ Voice handler NOT registered (engine unavailable)")
    print("✅ Message handlers registered.")

    print("📝 Registering callback query handlers...")
    _telegram_app.add_handler(CallbackQueryHandler(_handlers.cancel_task, pattern="^cancel_"))
    _telegram_app.add_handler(CallbackQueryHandler(_handlers.priority_callback, pattern="^prioritize_"))
    _telegram_app.add_handler(CallbackQueryHandler(_handlers.mode_callback, pattern="^mode_"))
    print("✅ Callback handlers registered.")

    print("✅ Bot initialized successfully with all handlers!")


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming Telegram updates via webhook."""
    print("🔔 Webhook called!")  # This will appear in the PythonAnywhere server log
    try:
        # Initialize once
        init_telegram_app()

        # Get the update data
        data = request.get_data(as_text=True)
        print(f"📦 Received data length: {len(data) if data else 0}")

        if not data:
            print("⚠️ Empty data received.")
            return Response('OK', status=200)

        # Parse the update
        try:
            update_data = json.loads(data)
            print(f"📩 Update data keys: {update_data.keys() if isinstance(update_data, dict) else 'not a dict'}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return Response('OK', status=200)

        update = Update.de_json(update_data, _telegram_app.bot)
        print(f"🆕 Update ID: {update.update_id if update else 'None'}")
        if update and update.message:
            print(f"💬 Message text: {update.message.text if update.message.text else '(not text)'}")
            print(f"👤 User: {update.message.from_user.id if update.message.from_user else 'Unknown'}")
        else:
            print("ℹ️ No message in this update.")

        # Process the update synchronously (blocking) so the webhook waits for reply
        print("⏳ Processing update...")
        async def process_update():
            async with _telegram_app:
                await _telegram_app.process_update(update)

        asyncio.run(process_update())
        print("✅ Update processed successfully.")

        return Response('OK', status=200)

    except Exception as e:
        print(f"❌ Webhook error: {e}")
        traceback.print_exc(file=sys.stdout)
        return Response('OK', status=200)  # Always return 200 to Telegram


@app.route('/', methods=['GET'])
def index():
    return "Telegram bot is running. Webhook endpoint: /webhook"


@app.route('/ping', methods=['GET'])
def ping():
    """Simple health check."""
    return "pong"


@app.route('/webhook_info', methods=['GET'])
def webhook_info():
    """Check current webhook status (requires bot token)."""
    import httpx
    token = Config.TELEGRAM_TOKEN
    try:
        resp = httpx.get(f"https://api.telegram.org/bot{token}/getWebhookInfo")
        return Response(resp.text, status=resp.status_code, content_type="application/json")
    except Exception as e:
        return Response(f'{{"error": "{str(e)}"}}', status=500, content_type="application/json")


# Initialize on startup
print("🚀 Initializing on startup...")
init_telegram_app()
print("✅ App ready.")