"""
Telegram-specific formatters for output.
Provides clean captions with prompt preview (no unsupported HTML).
"""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class TelegramFormatter:
    """
    Formats responses for Telegram.

    Features:
    - Clean captions with prompt preview
    - Truncates long prompts
    - Uses MarkdownV2 for formatting
    - No unsupported HTML tags
    """

    def __init__(self):
        self.max_prompt_preview = 150  # Characters to show in preview
        self.max_caption_length = 1000  # Telegram's caption limit is 1024 characters
        self.max_message_length = 4096  # Telegram's message limit
        logger.info("📱 TelegramFormatter initialized")

    def format_caption(self, prompt: str, style: Optional[str] = None,
                       model: Optional[str] = None) -> str:
        """
        Format a caption for an image/voice message.
        Uses Markdown V2 compatible formatting.

        Returns:
            Formatted caption string (plain text with basic Markdown)
        """
        parts = []

        # Header
        header = "🖼️ *Generated Image*"
        parts.append(header)

        # Style
        if style and style != "no_style":
            parts.append(f"*Style:* {style}")

        # Model
        if model:
            # Extract just the model name (remove tier prefix)
            model_name = model.split(':')[-1] if ':' in model else model
            parts.append(f"*Model:* {model_name}")

        # Prompt preview
        preview = self._truncate_prompt(prompt)
        parts.append(f"\n*Prompt:* {preview}")

        # Note about full prompt
        if len(prompt) > self.max_prompt_preview:
            parts.append("\n_📝 Full prompt sent in a separate message_")

        return "\n".join(parts)

    def format_full_prompt_message(self, prompt: str) -> str:
        """Format a message containing the full prompt."""
        return f"📝 *Full Prompt:*\n\n```\n{prompt}\n```"

    def format_generated_image_caption(self, prompt: str, style: Optional[str] = None,
                                       model: Optional[str] = None, source: Optional[str] = None) -> str:
        """
        Format the caption for a generated image.
        """
        # Extract just the model name
        model_name = model.split(':')[-1] if model and ':' in model else model
        return self.format_caption(prompt, style, model_name)

    def format_generated_voice_caption(self, text: str, model: Optional[str] = None,
                                       source: Optional[str] = None) -> str:
        """Format the caption for a generated voice message."""
        preview = self._truncate_prompt(text)
        model_name = model.split(':')[-1] if model and ':' in model else model

        parts = ["🔊 *Generated Voice*"]
        if model_name:
            parts.append(f"*Model:* {model_name}")
        parts.append(f"\n*Text:* {preview}")

        if len(text) > self.max_prompt_preview:
            parts.append("\n_📝 Full text sent in a separate message_")

        return "\n".join(parts)

    def format_text_response(self, text: str) -> str:
        """Format a regular text response."""
        return text

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate a prompt for preview."""
        if len(prompt) <= self.max_prompt_preview:
            return prompt
        return prompt[:self.max_prompt_preview] + "..."

    def chunk_message(self, text: str, max_length: int = None) -> List[str]:
        """
        Split a long message into chunks for Telegram.
        """
        max_length = max_length or self.max_message_length
        chunks = []

        if len(text) <= max_length:
            return [text]

        lines = text.split('\n')
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def get_info(self) -> Dict[str, Any]:
        """Return information about the formatter."""
        return {
            "type": "TelegramFormatter",
            "max_caption_length": self.max_caption_length,
            "max_message_length": self.max_message_length,
            "max_prompt_preview": self.max_prompt_preview
        }