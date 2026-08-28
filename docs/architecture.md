# Architecture Overview

## System Flow

The bot receives updates from Telegram via a webhook (Flask) or polling. The `BaseEngine` acts as a router, delegating to the appropriate engine based on input type (text, image, voice, document). Each engine performs its task and returns a response, which is sent back to the user.

```mermaid
graph TD
    A[Telegram Update] --> B[BaseEngine]
    B --> C{Input Type?}
    C -->|Text| D[TextEngine]
    C -->|Image| E[VisionEngine]
    C -->|Voice| F[VoiceEngine]
    C -->|Document| G[DocumentEngine]
    
    D --> H[OpenRouter API]
    E --> I[OpenRouter / Hugging Face]
    F --> J[Whisper / OpenRouter]
    G --> K[PDF/DOCX Extractor]
    K --> L[TextEngine for Q&A/Summary]
    
    H --> M[Response]
    I --> M
    J --> M
    L --> M
    M --> N[Telegram Reply]