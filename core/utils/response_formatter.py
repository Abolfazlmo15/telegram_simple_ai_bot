import logging
from typing import List, Tuple, Optional
from core.utils.response_config import (
    ResponseTemplates, FormattingRules,
    MarkdownStyles, EmojiSet
)

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Handles all aspects of response formatting, chunking, and validation.
    Ensures responses are Telegram-compatible and well-structured.
    """

    def __init__(self):
        self.emoji = EmojiSet()
        self.md = MarkdownStyles()

    def format_response(self, text: str, content_type: Optional[str] = None) -> str:
        """
        Main formatting method. Applies appropriate formatting based on content type.
        """
        if not text:
            return ""

        # Auto-detect content type if not provided
        if content_type is None:
            content_type = FormattingRules.detect_content_type(text)

        # Apply formatting based on type
        if content_type == "code":
            return self._format_code_response(text)
        elif content_type == "structured":
            return self._format_structured_response(text)
        else:
            return self._format_text_response(text)

    def _format_code_response(self, text: str) -> str:
        """Format code-heavy responses"""
        # Ensure code blocks are properly formatted
        if '```' not in text:
            # Try to detect code and wrap it
            lines = text.split('\n')
            code_lines = []
            text_lines = []
            in_code = False

            for line in lines:
                if line.strip().startswith(('def ', 'class ', 'import ', 'from ',
                                            'function ', 'const ', 'let ', 'var ',
                                            'if ', 'else ', 'for ', 'while ')):
                    in_code = True
                    code_lines.append(line)
                elif in_code and (line.strip() == '' or line.startswith(' ') or line.startswith('\t')):
                    code_lines.append(line)
                else:
                    if code_lines:
                        in_code = False
                        text_lines.append(self.md.code_block('\n'.join(code_lines), 'python'))
                        code_lines = []
                    text_lines.append(line)

            if code_lines:
                text_lines.append(self.md.code_block('\n'.join(code_lines), 'python'))

            text = '\n'.join(text_lines)

        return text

    def _format_structured_response(self, text: str) -> str:
        """Format structured responses (lists, sections)"""
        # Add emoji bullets if not present
        lines = text.split('\n')
        formatted_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('- ') or stripped.startswith('• '):
                formatted_lines.append(f"{self.emoji.arrow_right} {stripped[2:]}")
            elif stripped and not any(stripped.startswith(x) for x in ['*', '_', '`']):
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _format_text_response(self, text: str) -> str:
        """Format plain text responses with basic Markdown"""
        # Add subtle formatting for readability
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []

        for para in paragraphs:
            if para.strip():
                formatted_paragraphs.append(para.strip())

        return '\n\n'.join(formatted_paragraphs)

    def chunk_response(self, text: str) -> List[str]:
        """
        Split response into Telegram-compatible chunks.
        """
        if not FormattingRules.needs_chunking(text):
            return [text]
        return FormattingRules.split_into_chunks(text)

    def validate_and_fix(self, text: str) -> Tuple[str, List[str]]:
        """
        Validate Markdown and attempt fixes.
        """
        is_valid, errors = FormattingRules.validate_markdown(text)
        if is_valid:
            return text, []

        logger.warning(f"Markdown validation failed: {errors}")
        fixed_text = FormattingRules.fix_markdown(text)

        # Re-validate
        is_valid, remaining_errors = FormattingRules.validate_markdown(fixed_text)
        if remaining_errors:
            logger.error(f"Could not fix all Markdown errors: {remaining_errors}")

        return fixed_text, errors + remaining_errors

    def prepare_for_sending(self, text: str) -> List[str]:
        """
        Complete preparation pipeline: format, validate, chunk.
        """
        # Format
        formatted = self.format_response(text)

        # Validate and fix
        validated, warnings = self.validate_and_fix(formatted)
        if warnings:
            logger.warning(f"Markdown warnings: {warnings}")

        # Chunk if needed
        chunks = self.chunk_response(validated)
        return chunks

    def add_header_footer(self, text: str, header: Optional[str] = None,
                          footer: Optional[str] = None) -> str:
        """Add optional header and footer to response"""
        parts = []
        if header:
            parts.append(f"{header}\n")
        parts.append(text)
        if footer:
            parts.append(f"\n{footer}")
        return '\n'.join(parts)

    def create_error_response(self, error_type: str, details: str = "") -> str:
        """Create a formatted error response"""
        return ResponseTemplates.error_message(error_type, details)

    def create_success_response(self, message: str, icon: Optional[str] = None) -> str:
        """Create a formatted success response"""
        icon = icon or self.emoji.success
        return f"{icon} {message}"