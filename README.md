# Telegram Simple AI Bot

A multi‑modal Telegram bot that handles text, images, voice messages, documents (PDF/DOCX), generates images, and speaks back. Built with Python, it uses OpenRouter to access a wide range of AI models and runs reliably on PythonAnywhere.

**Try it now:** [@simple_ai_de_chat_bot](https://t.me/simple_ai_de_chat_bot)

---

## ✨ What it does

- **Text chat** – answers questions, explains concepts, helps with code, and more.
- **Image analysis** – send a photo and ask what’s in it (vision‑capable models).
- **Voice messages** – transcribes audio with local Whisper + OpenRouter fallback, then replies.
- **Documents** – extracts text from PDF and DOCX files, summarises them, or answers your questions about them.
- **Image generation** – create images from text prompts (fallback chain: Pollinations.ai → Hugging Face → OpenRouter).
- **Text‑to‑speech** – converts text replies to voice when you’re in voice mode.
- **Voice mode** – say *“talk to me”* to switch to voice replies, or *“type it”* to go back to text.
- **Memory & context** – remembers recent conversations and can search your history.
- **User preferences** – set your preferred response style, voice speed, model priority, and more.
- **Rate limiting, caching, analytics** – keeps things fast and fair.

---

## 🧠 Architecture

```
core/
├── engines/          – text, vision, voice, document, image/voice generation
├── managers/         – user data, cache, proxy, rate limiter, health checker, etc.
├── analytics/        – background analytics engine
└── utils/            – helpers (image processing, markdown stripper, network utils)

handlers/             – Telegram message handlers (text, photo, voice, document)
prompt_engineering/   – intent detection, style detection, prompt refinement
tests/                – pytest suite (135+ passing tests)
```

- **BaseEngine** routes incoming messages to the right engine (text, vision, voice, document).
- Each engine has its own fallback logic and blacklisting for unreliable models.
- Prompt engineering modules handle intent detection, style detection, and iterative refinement.
- Background services: health checker, analytics, cache persistence.

---

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Abolfazlmo15/telegram_simple_ai_bot.git
cd telegram_simple_ai_bot
```

### 2. Set up environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
- `TELEGRAM_TOKEN` – your bot token from [@BotFather](https://t.me/botfather)
- `OPENROUTER_API_KEY` – your OpenRouter API key
- `WORKER_URL` – the base URL where your bot is hosted (for webhooks)

Optional:
- `HUGGINGFACE_TOKEN` – for Hugging Face fallback (image generation & vision)
- Other settings are in `core/config.py`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The bot will automatically install missing packages when it starts.

### 4. Run the bot

**Polling mode** (for local development):

```bash
python main.py
```

**Webhook mode** (for PythonAnywhere / production):

```bash
python app.py
```

---

## 🧪 Testing

The project has a pytest suite with over 135 passing tests.

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest -v
```

Tests cover:
- All engines (text, vision, voice, document, generation)
- Managers (cache, rate limiter, proxy, health checker)
- Prompt engineering (intent, style, correction, extraction)
- Handlers (message routing, priority setup, mode switching)

---

## ☁️ Deployment

The bot is designed to run on **PythonAnywhere** (free tier) with webhook support.

- Use `app.py` for the Flask webhook entry point.
- The repository includes a `Procfile` for compatibility.
- A GitHub Actions workflow (`renew_pythonanywhere.yml`) automatically logs in and renews the web app every month, so you don’t have to remember.

---

## 📦 Dependencies

Key libraries:
- `python-telegram-bot` – Telegram API wrapper
- `httpx` – async HTTP client (with HTTP/2 support)
- `Pillow` – image processing
- `pypdf` & `python-docx` – document extraction
- `gTTS` – text‑to‑speech fallback
- `openai` – for OpenRouter API compatibility
- `pytest` – testing

Full list in `requirements.txt`.

---

## 🤝 Contributing

Found a bug or want to improve something? Feel free to open an issue or a pull request. Keep it practical – the goal is a working, maintainable bot that doesn’t over‑promise.

---

## 📄 License

MIT – see [LICENSE](https://github.com/Abolfazlmo15/telegram_simple_ai_bot/blob/main/LICENSE) for details.

---

## 🙋‍♂️ About

Built by [Abolfazl Motamed](https://github.com/Abolfazlmo15). The bot is live at [@simple_ai_de_chat_bot](https://t.me/simple_ai_de_chat_bot) – feel free to try it out.
