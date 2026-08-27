"""Document analysis engine for PDF and DOCX files.
Extracts text, handles captions, and uses AI models for Q&A/summarization.
"""
import logging
import io
from typing import Optional, Tuple, Dict, Any

from core.config import Config
from core.managers.user_data_manager import UserDataManager

logger = logging.getLogger(__name__)

# Lightweight imports – only used when a document is processed
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
    logger.warning("pypdf not installed. PDF support will be unavailable.")

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None
    logger.warning("python-docx not installed. DOCX support will be unavailable.")


class DocumentEngine:
    """Extracts and optionally summarises/answers questions from PDF/DOCX files."""

    def __init__(self, user_data_manager: UserDataManager, text_engine=None):
        self.user_data_manager = user_data_manager
        self.text_engine = text_engine
        self.max_file_size_mb = Config.DOCUMENT_MAX_SIZE_MB
        self.max_text_reply_chars = Config.DOCUMENT_MAX_TEXT_REPLY_CHARS
        self.max_ai_context_chars = Config.DOCUMENT_MAX_AI_CONTEXT_CHARS
        self.summary_max_tokens = Config.DOCUMENT_AI_SUMMARY_MAX_TOKENS

        logger.info("📄 DocumentEngine initialized")

    async def process(
        self,
        file_bytes: bytes,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, int]:
        """
        Process a document file.

        Args:
            file_bytes: Raw file content.
            context: Must contain 'file_extension' ('.pdf', '.docx') and optionally
                     'caption' (user prompt) and 'user_id', etc.

        Returns:
            Tuple of (response_text, model_used, tokens_used)
            - response_text can be a plain text or a formatted string with file content.
        """
        if context is None:
            context = {}

        file_extension = context.get('file_extension', '').lower()
        caption = context.get('caption', '').strip()
        user_id = context.get('user_id')

        # Validate file size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return (
                f"❌ *File too large* – maximum {self.max_file_size_mb} MB allowed.\n"
                f"Current size: {file_size_mb:.1f} MB",
                "document_error",
                0
            )

        # Extract text
        if file_extension in ('.pdf', '.pdf'):
            extracted_text = await self._extract_pdf(file_bytes)
        elif file_extension in ('.docx', '.docx'):
            extracted_text = await self._extract_docx(file_bytes)
        else:
            return (
                f"❌ *Unsupported file type* – only PDF and DOCX are supported.\n"
                f"Received: {file_extension}",
                "document_error",
                0
            )

        if not extracted_text or len(extracted_text.strip()) == 0:
            return (
                "❌ *No text could be extracted* from the document.\n"
                "It may be scanned or contain only images.",
                "document_error",
                0
            )

        # Log success
        logger.info(f"📄 Extracted {len(extracted_text)} characters from document")

        # If user provided a caption, treat as a question/instruction
        if caption:
            return await self._handle_caption(extracted_text, caption, user_id, context)

        # No caption: return extracted text (or summary if too long)
        if len(extracted_text) <= self.max_text_reply_chars:
            return extracted_text, "document_extraction", len(extracted_text) // 4
        else:
            # Long document – generate a summary using AI
            summary = await self._generate_summary(extracted_text, user_id, context)
            if summary and summary != extracted_text:
                return summary, "document_summary", len(summary) // 4
            else:
                # Fallback: truncate and send a note
                truncated = extracted_text[:self.max_text_reply_chars] + "\n\n…(truncated)"
                return truncated, "document_extraction_truncated", len(truncated) // 4

    # ---------- Extraction ----------
    async def _extract_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF bytes using pypdf."""
        if PdfReader is None:
            raise RuntimeError("pypdf not installed")
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

    async def _extract_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX bytes using python-docx."""
        if DocxDocument is None:
            raise RuntimeError("python-docx not installed")
        try:
            doc = DocxDocument(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise

    # ---------- AI Interactions ----------
    async def _handle_caption(self, document_text: str, caption: str, user_id: int, context: dict) -> Tuple[str, str, int]:
        """
        User provided a caption – use AI to answer based on the document.
        If document is too long, truncate to a safe length.
        """
        if not self.text_engine or not self.text_engine.is_initialized:
            # Fallback: send extracted text with a note
            return (
                f"⚠️ *AI engine unavailable* – here is the extracted text instead.\n\n"
                f"{document_text[:self.max_text_reply_chars]}",
                "document_extraction_fallback",
                len(document_text) // 4
            )

        # Truncate document text to a manageable length for AI context
        if len(document_text) > self.max_ai_context_chars:
            truncated = document_text[:self.max_ai_context_chars] + "\n…(truncated)"
            logger.info(f"📄 Truncated document to {self.max_ai_context_chars} chars for AI")
        else:
            truncated = document_text

        # Build a prompt that includes the document and the user's question
        prompt = (
            f"Here is the content of a document:\n\n"
            f"---\n{truncated}\n---\n\n"
            f"Based on the document, please answer the following:\n{caption}"
        )

        # Use text_engine to process this combined prompt
        try:
            response, model, tokens = await self.text_engine.process(
                prompt,
                context={
                    'user_id': user_id,
                    'skip_cache': True,
                    'preferences': context.get('preferences', {})
                }
            )
            # If response is too long, we might need to truncate (but text_engine should handle it)
            return response, f"document_ai_{model}", tokens
        except Exception as e:
            logger.error(f"AI processing failed for document: {e}")
            # Fallback: return extracted text (truncated)
            return (
                f"❌ *AI processing failed* – here is the extracted text:\n\n"
                f"{document_text[:self.max_text_reply_chars]}",
                "document_extraction_fallback",
                len(document_text) // 4
            )

    async def _generate_summary(self, document_text: str, user_id: int, context: dict) -> Optional[str]:
        """Generate a summary of the document using AI."""
        if not self.text_engine or not self.text_engine.is_initialized:
            return None

        # Truncate if needed
        if len(document_text) > self.max_ai_context_chars:
            truncated = document_text[:self.max_ai_context_chars] + "\n…(truncated)"
        else:
            truncated = document_text

        prompt = (
            f"Summarise the following document in a clear, concise way (max {self.summary_max_tokens} tokens):\n\n"
            f"{truncated}"
        )

        try:
            summary, model, tokens = await self.text_engine.process(
                prompt,
                context={
                    'user_id': user_id,
                    'skip_cache': True,
                    'preferences': context.get('preferences', {})
                }
            )
            return summary
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return None

    # ---------- Engine Info ----------
    def get_engine_info(self) -> dict:
        return {
            "type": "DocumentEngine",
            "available": True,
            "supports_pdf": PdfReader is not None,
            "supports_docx": DocxDocument is not None,
            "max_file_size_mb": self.max_file_size_mb,
            "max_text_reply_chars": self.max_text_reply_chars,
            "ai_context_chars": self.max_ai_context_chars,
        }