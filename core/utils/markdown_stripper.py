"""
Strips markdown, emojis, and other formatting for clean text-to-speech.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MarkdownStripper:
    """
    Utility to strip markdown and emojis from text for TTS.
    """

    # Markdown patterns
    MD_PATTERNS = [
        (r'\*\*(.+?)\*\*', r'\1'),           # Bold: **text**
        (r'\*(.+?)\*', r'\1'),               # Italic: *text*
        (r'__(.+?)__', r'\1'),               # Underline: __text__
        (r'_(.+?)_', r'\1'),                 # Italic: _text_
        (r'~~(.+?)~~', r'\1'),               # Strikethrough: ~~text~~
        (r'`(.+?)`', r'\1'),                 # Inline code: `text`
        (r'```[a-zA-Z]*\n(.+?)\n```', r'\1'), # Code block
        (r'\[(.+?)\]\(.+?\)', r'\1'),        # Link: [text](url)
        (r'^#+\s*(.+)', r'\1'),              # Headings: # text
        (r'>\s*(.+)', r'\1'),                # Blockquote: > text
        (r'^- (.+)', r'\1'),                 # List: - text
        (r'^\d+\. (.+)', r'\1'),             # List: 1. text
    ]

    # Emoji patterns (simple range)
    EMOJI_PATTERN = re.compile(
        r'[\U0001F600-\U0001F64F'     # Emoticons
        r'\U0001F300-\U0001F5FF'      # Misc Symbols and Pictographs
        r'\U0001F680-\U0001F6FF'      # Transport and Map
        r'\U0001F700-\U0001F77F'      # Alchemical Symbols
        r'\U0001F780-\U0001F7FF'      # Geometric Shapes
        r'\U0001F800-\U0001F8FF'      # Supplemental Arrows-C
        r'\U0001F900-\U0001F9FF'      # Supplemental Symbols and Pictographs
        r'\U0001FA00-\U0001FA6F'      # Chess Symbols
        r'\U0001FA70-\U0001FAFF'      # Symbols and Pictographs Extended-A
        r'\u2600-\u26FF'              # Misc Symbols
        r'\u2700-\u27BF'              # Dingbats
        r']+', re.UNICODE
    )

    def __init__(self):
        self.compiled_patterns = [(re.compile(pattern, re.DOTALL | re.MULTILINE), repl)
                                   for pattern, repl in self.MD_PATTERNS]
        logger.info("📝 MarkdownStripper initialized")

    def strip(self, text: str, remove_emojis: bool = True) -> str:
        """
        Strip markdown and optionally emojis from text.
        """
        if not text:
            return ""

        # Apply markdown patterns
        for pattern, repl in self.compiled_patterns:
            text = pattern.sub(repl, text)

        # Remove emojis if requested
        if remove_emojis:
            text = self.EMOJI_PATTERN.sub('', text)

        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def strip_for_tts(self, text: str) -> str:
        """
        Prepare text specifically for text-to-speech.
        Removes markdown and emojis.
        """
        return self.strip(text, remove_emojis=True)

    def strip_markdown_only(self, text: str) -> str:
        """
        Strip only markdown, keep emojis.
        """
        return self.strip(text, remove_emojis=False)

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text for TTS: remove markdown, emojis, and clean up.
        """
        return self.strip_for_tts(text)