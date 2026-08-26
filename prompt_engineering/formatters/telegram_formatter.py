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
    - Truncates long prompts and model names
    - Uses MarkdownV2 compatible formatting
    - No unsupported HTML tags
    - Specialized formatters for priority commands
    """

    def __init__(self):
        self.max_prompt_preview = 150  # Characters to show in preview
        self.max_caption_length = 1000  # Telegram's caption limit is 1024 characters
        self.max_message_length = 4096  # Telegram's message limit
        self.max_model_name_length = 40  # Truncate long model names
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

        # Model (truncate if too long)
        if model:
            model_name = self._truncate_model_name(model)
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
        """Format the caption for a generated image."""
        model_name = self._truncate_model_name(model)
        return self.format_caption(prompt, style, model_name)

    def format_generated_voice_caption(self, text: str, model: Optional[str] = None,
                                       source: Optional[str] = None) -> str:
        """Format the caption for a generated voice message."""
        preview = self._truncate_prompt(text)
        model_name = self._truncate_model_name(model)

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

    # ============================================================
    # NEW: Priority & Status Formatters (for the new commands)
    # ============================================================

    def format_priority_setup_intro(self, engine_name: str, available_models: List[str],
                                    current_priority: List[str]) -> str:
        """
        Format the introduction message for setting a priority list.
        engine_name: e.g., "Voice Generation", "Vision Analysis"
        """
        available_preview = "\n".join([f"• `{m}`" for m in available_models[:10]])
        if len(available_models) > 10:
            available_preview += f"\n... and {len(available_models) - 10} more"

        current_preview = "\n".join([f"{i+1}. `{m}`" for i, m in enumerate(current_priority)]) if current_priority else "*(Default system order)*"

        text = f"🎯 *{engine_name} Priority Setup*\n\n"
        text += f"*Current Priority:*\n{current_preview}\n\n"
        text += f"*Available Models:*\n{available_preview}\n\n"
        text += "To set your priority, send the model IDs in your preferred order, separated by commas or new lines.\n"
        text += f"_Example:_ `{available_models[0] if available_models else 'model/id:free'}, {available_models[1] if len(available_models) > 1 else 'model/id:free'}`\n\n"
        text += "_Type /cancel to cancel._"
        return text

    def format_priority_saved(self, engine_name: str, priority_list: List[str]) -> str:
        """Format the confirmation message when a priority list is saved."""
        formatted_list = "\n".join([f"{i+1}. `{m}`" for i, m in enumerate(priority_list)])
        return f"✅ *{engine_name} Priority Saved!*\n\n*New Priority Order:*\n{formatted_list}\n\nThe bot will now use models in this order."

    def format_priority_cancelled(self) -> str:
        """Format the cancellation message."""
        return "❌ *Priority setup cancelled.*"

    def format_priority_invalid(self, invalid_models: List[str], valid_models: List[str]) -> str:
        """Format the error message for invalid model entries."""
        invalid_str = ", ".join([f"`{m}`" for m in invalid_models[:5]])
        valid_preview = ", ".join([f"`{m}`" for m in valid_models[:5]])
        return f"❌ *Invalid Models:* {invalid_str}\n\nPlease choose from the available list.\n*Valid examples:* {valid_preview}..."

    def format_status_with_priorities(self, user_id: int, stats: Dict,
                                      text_priority: List[str], vision_priority: List[str],
                                      voice_priority: List[str], image_priority: List[str]) -> str:
        """Format the /status command with all priority lists included."""
        text = f"📊 *User Status*\n\n"
        text += f"*User ID:* `{user_id}`\n"
        text += f"*Messages:* {stats.get('total_messages', 0)}\n"
        text += f"*Tokens Used:* {stats.get('total_tokens', 0)}\n\n"

        text += "⚙️ *Your Priority Orders:*\n"
        text += f"• *Text:* {text_priority[0] if text_priority else 'Default'}\n"
        text += f"• *Vision:* {vision_priority[0] if vision_priority else 'Default'}\n"
        text += f"• *Voice STT:* {voice_priority[0] if voice_priority else 'Default'}\n"
        text += f"• *Image Gen:* {image_priority[0] if image_priority else 'Default'}\n\n"
        text += "_Use specific commands to modify these._"
        return text

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate a prompt for preview."""
        if not prompt:
            return ""
        if len(prompt) <= self.max_prompt_preview:
            return prompt
        return prompt[:self.max_prompt_preview] + "..."

    def _truncate_model_name(self, model: Optional[str]) -> str:
        """Truncate a model name to avoid cluttering the caption."""
        if not model:
            return "Unknown"
        # Remove tier prefixes (e.g., "gen_image:", "pollinations:")
        if ':' in model:
            parts = model.split(':')
            if len(parts) > 1:
                model = parts[-1]  # Take the last part after the last colon
        if len(model) > self.max_model_name_length:
            return model[:self.max_model_name_length] + "..."
        return model

    def chunk_message(self, text: str, max_length: int = None) -> List[str]:
        """
        Split a long message into chunks for Telegram.
        Preserves markdown structure when possible.
        """
        max_length = max_length or self.max_message_length
        chunks = []

        if len(text) <= max_length:
            return [text]

        lines = text.split('\n')
        current_chunk = ""
        in_code_block = False

        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block

            if len(current_chunk) + len(line) + 1 > max_length:
                # If we are inside a code block, try to end it gracefully
                if in_code_block:
                    current_chunk += "\n```"
                    in_code_block = False
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