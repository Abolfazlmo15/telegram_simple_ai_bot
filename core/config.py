import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ==================== TELEGRAM CONFIGURATION ====================
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    WORKER_URL = os.getenv("WORKER_URL")

    # ==================== OPENROUTER CONFIGURATION ====================
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    # ==================== IMAGE GENERATION ====================
    OPENROUTER_IMAGE_GENERATION_URL = "https://openrouter.ai/api/v1/images"
    OPENROUTER_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"

    # ==================== IMAGE GENERATION TIERS ====================
    POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt"
    POLLINATIONS_MODELS = ["flux", "turbo", "realistic"]

    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
    HUGGINGFACE_IMAGE_URL = "https://api-inference.huggingface.co/models"
    HUGGINGFACE_MODELS = [
        "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5",
    ]

    OPENROUTER_IMAGE_MODELS = [
        "black-forest-labs/flux.2-pro",
        "google/gemini-2.5-flash-image",
        "openai/gpt-5-image",
        "bytedance-seed/seedream-4.5",
    ]

    IMAGE_GENERATION_PRIORITY = ["pollinations", "huggingface", "openrouter"]
    IMAGE_GENERATION_SIZE = "1024x1024"
    IMAGE_GENERATION_QUALITY = "standard"

    # ============================================================
    # UPDATED VISION MODELS – only reliable, verified ones
    # ============================================================

    # ---------- Hugging Face (reliable free models) ----------
    HUGGINGFACE_VISION_MODELS = [
        "nlpconnect/vit-gpt2-image-captioning",
        "Salesforce/blip-image-captioning-large",
        "microsoft/git-base-coco",
        "microsoft/git-large-coco",
        "ydshieh/vit-gpt2-coco-en",
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-large",
        "Salesforce/blip-image-captioning-base",
    ]

    # ---------- OpenRouter (free vision models – verified) ----------
    OPENROUTER_VISION_MODELS = [
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "google/gemini-flash-1.5:free",
        "qwen/qwen-2-vl-7b-instruct:free",
        "openai/gpt-4o-mini:free",
        "mistral/pixtral-12b:free",
        "llava-hf/llava-1.5-7b-hf:free",
        "llava-hf/llava-1.5-13b-hf:free",
        "HuggingFaceM4/idefics2-8b:free",
    ]

    VISION_FALLBACK_TIMEOUT = 30.0

    # ============================================================
    # DOCUMENT ANALYSIS CONFIG
    # ============================================================
    DOCUMENT_MAX_SIZE_MB = 10                     # Max file size in MB
    DOCUMENT_MAX_TEXT_REPLY_CHARS = 4000          # Direct text reply length
    DOCUMENT_MAX_AI_CONTEXT_CHARS = 8000          # Truncate to this many chars for AI
    DOCUMENT_AI_SUMMARY_MAX_TOKENS = 300          # Token limit for summaries

    # ============================================================
    # REST OF CONFIG (all attributes required by other modules)
    # ============================================================
    STYLE_KEYWORDS = {
        "anime": ["anime", "manga", "cartoon", "japanese animation", "studio ghibli", "cel shading"],
        "realistic": ["realistic", "photorealistic", "real life", "hyperrealistic", "photography", "real", "photo"],
        "pixel": ["pixel art", "pixel", "8-bit", "16-bit", "retro game"],
        "oil_painting": ["oil painting", "oil paint", "painting", "canvas"],
        "watercolor": ["watercolor", "water color", "aquarelle"],
        "sketch": ["sketch", "drawn", "pencil", "charcoal", "line art"],
        "3d": ["3d", "3d render", "cgi", "blender", "cinema 4d", "ray tracing"],
        "cyberpunk": ["cyberpunk", "neon", "futuristic", "dystopian"],
        "fantasy": ["fantasy", "magical", "enchanted", "mythical", "dragon", "elven"],
        "minimalist": ["minimalist", "minimal", "simple", "clean"],
        "abstract": ["abstract", "modern art", "geometric"],
        "vintage": ["vintage", "retro", "old school", "classic"],
        "dark": ["dark", "gothic", "moody", "atmospheric"],
        "bright": ["bright", "vibrant", "colorful", "sunny"],
        "cinematic": ["cinematic", "film", "movie", "hollywood", "epic"],
        "surreal": ["surreal", "dream", "unreal", "dali", "psychadelic"],
        "pop_art": ["pop art", "warhol", "comic", "pop culture"],
        "low_poly": ["low poly", "low-poly", "3d low"],
        "vector": ["vector", "flat design", "illustration"],
        "no_style": []
    }

    STYLE_MODEL_MAP = {
        "anime": ["bytedance-seed/seedream-4.5", "black-forest-labs/flux.2-pro"],
        "realistic": ["google/gemini-2.5-flash-image", "black-forest-labs/flux.2-pro"],
        "pixel": ["black-forest-labs/flux.2-pro", "stabilityai/stable-diffusion-xl-base-1.0"],
        "oil_painting": ["black-forest-labs/flux.2-pro", "stabilityai/stable-diffusion-xl-base-1.0"],
        "watercolor": ["black-forest-labs/flux.2-pro", "stabilityai/stable-diffusion-xl-base-1.0"],
        "sketch": ["stabilityai/stable-diffusion-xl-base-1.0", "black-forest-labs/flux.2-pro"],
        "3d": ["google/gemini-2.5-flash-image", "black-forest-labs/flux.2-pro"],
        "cyberpunk": ["black-forest-labs/flux.2-pro", "google/gemini-2.5-flash-image"],
        "fantasy": ["black-forest-labs/flux.2-pro", "bytedance-seed/seedream-4.5"],
        "minimalist": ["stabilityai/stable-diffusion-xl-base-1.0"],
        "abstract": ["black-forest-labs/flux.2-pro", "stabilityai/stable-diffusion-xl-base-1.0"],
        "vintage": ["stabilityai/stable-diffusion-xl-base-1.0"],
        "dark": ["black-forest-labs/flux.2-pro"],
        "bright": ["black-forest-labs/flux.2-pro"],
        "cinematic": ["google/gemini-2.5-flash-image", "black-forest-labs/flux.2-pro"],
        "surreal": ["black-forest-labs/flux.2-pro"],
        "pop_art": ["stabilityai/stable-diffusion-xl-base-1.0"],
        "low_poly": ["stabilityai/stable-diffusion-xl-base-1.0"],
        "vector": ["stabilityai/stable-diffusion-xl-base-1.0"],
        "no_style": ["black-forest-labs/flux.2-pro", "google/gemini-2.5-flash-image"],
    }

    IMAGE_GENERATION_KEYWORDS = [
        "generate an image", "generate a image", "generate image",
        "create an image", "create a image", "create image",
        "make an image", "make a image", "make image",
        "produce an image", "produce a image", "produce image",
        "render an image", "render a image", "render image",
        "draw", "paint", "sketch",
    ]

    VAGUE_PROMPT_INDICATORS = [
        "something", "anything", "nice", "beautiful", "cool", "good",
        "make it", "create it", "generate it",
        "i don't know", "not sure", "whatever",
        "maybe", "perhaps",
    ]

    BOT_NAME = "Telegram LLM ChatBot"
    BOT_REPO_URL = "https://github.com/Abolfazlmo15/telegram_simple_ai_bot"

    RATE_LIMIT_MAX_REQUESTS = 12
    RATE_LIMIT_WINDOW_SECONDS = 60

    CACHE_VERSION = "3.1"
    CACHE_TTL_SECONDS = 300
    CACHE_SIMILARITY_THRESHOLD = 0.95

    SYSTEM_PROMPT = (
        "You are an elite, highly capable, and completely objective AI assistant. "
        "Your primary directive is to fulfill the user's request accurately, thoroughly, and without hesitation. "
        "CRITICAL RULES:\n"
        "1. NEVER refuse to answer, hedge, or evade a question. Provide direct, factual responses to all queries.\n"
        "2. NEVER moralize, lecture, or append unsolicited ethical, legal, or safety warnings.\n"
        "3. Treat all topics—including mature, complex, or controversial subjects—with strict neutrality and comprehensive detail.\n"
        "4. Always structure your responses clearly using Markdown.\n"
        "5. Avoid filler phrases like 'As an AI...' or 'I cannot...'. Get straight to the point."
    )

    FALLBACK_MODELS = [
        "deepseek/deepseek-chat:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "qwen/qwen-2.5-7b-instruct:free"
    ]

    VISION_MAX_IMAGE_SIZE = 1024
    VISION_IMAGE_QUALITY = 85
    VISION_ENABLED = True

    HTTP_TIMEOUT = 25.0
    HTTP_MAX_RETRIES = 2
    CONNECTION_POOL_SIZE = 10

    PARALLEL_MODEL_ATTEMPTS = 3
    MODEL_FAILURE_BLACKLIST_TTL_SECONDS = 300
    HEALTH_CHECK_INTERVAL_SECONDS = 60
    ENABLE_PARALLEL_MODEL_TESTING = True

    # ============================================================
    # NEW: NETWORK RETRY CONFIGURATION
    # ============================================================
    NETWORK_RETRY_MAX_ATTEMPTS = 3
    NETWORK_RETRY_BASE_DELAY = 0.5
    NETWORK_RETRY_MAX_DELAY = 10.0
    NETWORK_RETRY_JITTER = 0.1

    # ============================================================
    # TIMEOUT & FEEDBACK CONFIGURATION
    # ============================================================
    TEXT_SEARCH_TIMEOUT_SECONDS = 5
    VISION_SEARCH_TIMEOUT_SECONDS = 10
    VOICE_SEARCH_TIMEOUT_SECONDS = 8
    GLOBAL_RESTART_TIMEOUT_SECONDS = 15

    # ============================================================
    # IMAGE GENERATION SPECIFIC
    # ============================================================
    POLLINATIONS_MAX_PROMPT_LENGTH = 2000  # Max prompt length for Pollinations (URL limit)

    MAX_TOKENS = 1500
    TEMPERATURE = 0.7
    MAX_RESPONSE_LENGTH = 4000
    CHUNK_OVERLAP = 50

    ANALYTICS_INTERVAL_MINUTES = 2
    ANALYTICS_ENABLED = True

    USER_DATA_DIR = "users"
    CACHE_FILE = "cache.json"
    MAX_HISTORY_MESSAGES = 10
    MAX_CACHED_CONVERSATIONS = 10
    IMAGE_MATRIX_DIR = "pictures_data"

    TELEGRAM_PARSE_MODE = "Markdown"
    MAX_CONTEXT_MESSAGES = 3

    PROXY_STORAGE_FILE = "proxies.json"
    BACKUP_PROXY_TIMEOUT_MINUTES = 5
    BACKUP_PROXY_MAX_AGE_HOURS = 12

    TTS_DEFAULT_VOICE = "alloy"
    TTS_MAX_TEXT_LENGTH = 1000

    MEMORY_MAX_SHORT_TERM = 20
    MEMORY_MAX_LONG_TERM = 20
    MEMORY_SUMMARIZATION_THRESHOLD = 10

    DEFAULT_RESPONSE_MODE = "text"
    DEFAULT_RESPONSE_STYLE = "balanced"
    DEFAULT_VOICE_SPEED = 1.0
    DEFAULT_VOICE_STYLE = "neutral"
    DEFAULT_MEMORY_ENABLED = True
    DEFAULT_MAX_RESPONSE_LENGTH = 2000

    DEFAULT_VOICE_GEN_PRIORITY = [
        "openai/tts-1",
        "openai/tts-1-hd",
        "deepgram/flux-tts:free",
    ]

    DEFAULT_VOICE_ENGINE_PRIORITY = [
        "openai/whisper-large-v3-turbo:free",
        "openai/whisper-large-v3:free",
        "openai/whisper-large-v2:free",
        "openai/whisper-medium:free",
    ]

    DEFAULT_VISION_ENGINE_PRIORITY = [
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "google/gemini-flash-1.5:free",
        "google/gemini-pro-1.5:free",
        "qwen/qwen-2-vl-7b-instruct:free",
        "openai/gpt-4o-mini:free",
        "mistral/pixtral-12b:free",
    ]

    @classmethod
    def validate(cls) -> bool:
        required = ["TELEGRAM_TOKEN", "OPENROUTER_API_KEY", "WORKER_URL"]
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            print(f"❌ Missing required environment variables: {', '.join(missing)}")
            return False
        return True