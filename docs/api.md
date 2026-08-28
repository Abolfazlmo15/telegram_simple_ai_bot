# API Reference

## Core Engines
- `BaseEngine`: routes text/images/voice/documents to appropriate engines.
- `TextEngine`: handles text completions with caching, retries, blacklisting.
- `VisionEngine`: image analysis via OpenRouter → Hugging Face.
- `VoiceEngine`: STT via Whisper → OpenRouter fallback.
- `DocumentEngine`: PDF/DOCX extraction + AI Q&A/summarization.
- `ImageGenerationEngine`: multi‑tier image generation.
- `VoiceGenerationEngine`: TTS via OpenRouter → gTTS.

## Managers
- `UserDataManager`: user data, history, preferences.
- `CacheManager`: TTL‑based caching.
- `ProxyManager`: proxy rotation with failover.
- `RateLimiter`: per‑user rate limiting.
- `HealthChecker`: background model health monitoring.

## Prompt Engineering
- `IntentDetector`, `StyleDetector`, `CorrectionDetector`, `PromptExtractor`, etc.