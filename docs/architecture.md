graph TD
    A[Telegram Update] --> B[BaseEngine]
    B --> C{Input Type?}
    C -->|Text| D[TextEngine]
    C -->|Image| E[VisionEngine]
    C -->|Voice| F[VoiceEngine]
    C -->|Document| G[DocumentEngine]
    D --> H[OpenRouter API]
    E --> I[OpenRouter / HF]
    F --> J[Whisper / OpenRouter]
    G --> K[PDF/DOCX Extractor]
    K --> L[TextEngine for Q&A/Summary]
    H --> M[Response]
    I --> M
    J --> M
    L --> M
    M --> N[Telegram Reply]