from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EmojiSet:
    """Emoji mappings for different response types"""
    success: str = "✅"
    error: str = "❌"
    warning: str = "⚠️"
    info: str = "ℹ️"
    code: str = "💻"
    tip: str = "💡"
    question: str = "❓"
    check: str = "✔️"
    arrow_right: str = "➡️"
    arrow_down: str = "⬇️"
    star: str = "⭐"
    fire: str = "🔥"


class MarkdownStyles:
    """Standard Markdown (V1) styling helpers"""

    @staticmethod
    def bold(text: str) -> str:
        return f"*{text}*"

    @staticmethod
    def italic(text: str) -> str:
        return f"_{text}_"

    @staticmethod
    def code(text: str) -> str:
        return f"`{text}`"

    @staticmethod
    def code_block(text: str, language: str = "") -> str:
        return f"```{language}\n{text}\n```"

    @staticmethod
    def quote(text: str) -> str:
        lines = text.split('\n')
        return '\n'.join(f"> {line}" for line in lines)

    @staticmethod
    def link(text: str, url: str) -> str:
        return f"[{text}]({url})"


class ResponseTemplates:
    """Pre-defined response templates for common scenarios"""

    @staticmethod
    def chunk_indicator(current: int, total: int) -> str:
        """Indicator for multi-part messages"""
        return f"\n\n_(Part {current} of {total})_"

    @staticmethod
    def error_message(error_type: str, details: str = "") -> str:
        """Create a formatted error response"""
        emoji_map = {
            "network": "🌐",
            "api": "⚠️",
            "timeout": "⏱️",
            "unknown": "❌"
        }
        emoji = emoji_map.get(error_type, "❌")
        return (
            f"{emoji} *Error Occurred*\n\n"
            f"*Type:* {error_type.title()}\n"
            f"{details}\n\n"
            f"_Please try again in a moment._"
        )


class FormattingRules:
    """Rules and validators for response formatting"""

    MAX_CHUNK_SIZE = 4000
    CHUNK_OVERLAP = 50

    @staticmethod
    def needs_chunking(text: str) -> bool:
        """Check if text needs to be chunked"""
        return len(text) > FormattingRules.MAX_CHUNK_SIZE

    @staticmethod
    def split_into_chunks(text: str) -> List[str]:
        """Split long text into chunks while preserving code blocks and structure."""
        if len(text) <= FormattingRules.MAX_CHUNK_SIZE:
            return [text]

        chunks = []
        current_chunk = ""

        # Split by code blocks first
        parts = text.split('```')

        for i, part in enumerate(parts):
            if i % 2 == 1:  # Inside code block
                code_block = f"```{part}```"
                if len(current_chunk) + len(code_block) > FormattingRules.MAX_CHUNK_SIZE:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    chunks.append(code_block)
                else:
                    current_chunk += code_block
            else:  # Regular text
                lines = part.split('\n')
                for line in lines:
                    line_with_newline = line + '\n'
                    if len(current_chunk) + len(line_with_newline) > FormattingRules.MAX_CHUNK_SIZE:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = line_with_newline
                    else:
                        current_chunk += line_with_newline

        if current_chunk:
            chunks.append(current_chunk)

        # Add chunk indicators
        if len(chunks) > 1:
            chunks = [
                chunk + ResponseTemplates.chunk_indicator(i + 1, len(chunks))
                for i, chunk in enumerate(chunks)
            ]

        return chunks

    @staticmethod
    def validate_markdown(text: str) -> tuple:
        """Validate Markdown syntax and return (is_valid, errors)"""
        errors = []
        if text.count('*') % 2 != 0:
            errors.append("Unbalanced asterisks")
        if text.count('_') % 2 != 0:
            errors.append("Unbalanced underscores")
        if text.count('`') % 2 != 0:
            errors.append("Unbalanced backticks")
        return len(errors) == 0, errors

    @staticmethod
    def fix_markdown(text: str) -> str:
        """Attempt to fix common Markdown errors"""
        if text.count('*') % 2 != 0:
            text = text.rstrip('*') + '*'
        if text.count('_') % 2 != 0:
            text = text.rstrip('_') + '_'
        if text.count('`') % 2 != 0:
            text = text.rstrip('`') + '`'
        return text

    @staticmethod
    def detect_content_type(text: str) -> str:
        """Detect the primary content type of the response"""
        if '```' in text:
            return "code"
        elif text.count('\n') > 10 and any(x in text for x in ['•', '-', '1.', '2.']):
            return "structured"
        elif len(text) < 200:
            return "short"
        else:
            return "text"