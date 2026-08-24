"""
Smart prompt extractor that handles complex user messages.
Extracts the actual prompt from instructions like "generate an image in accordance to below: ..."
"""
import logging
import re
from typing import Dict, Any, Optional, Tuple
from prompt_engineering.base.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class PromptExtractor(BaseDetector):
    """
    Extracts the actual prompt from complex user messages.
    Handles:
    - Instructions with colons: "generate an image in accordance to below: ..."
    - Multi-line prompts
    - Quoted prompts: "generate an image of 'a woman smiling'"
    - Code block prompts: ```prompt```
    - Prompts after indicators: "based on:", "according to:", etc.
    """

    def __init__(self):
        super().__init__()
        self.indicators = [
            "in accordance to", "in accordance with", "according to", "based on",
            "below:", "as follows:", "following:", "this prompt:", "these details:",
            "from this:", "using this:", "with this description:", "for this:",
            "like this:", "such as:", "including:", "specifically:", "namely:"
        ]
        self.quote_patterns = [
            r'"([^"]+)"',  # Double quotes
            r"'([^']+)'",  # Single quotes
            r'“([^”]+)”',  # Smart double quotes
            r'‘([^’]+)’',  # Smart single quotes
            r'`([^`]+)`',  # Backticks
            r'```([^`]+)```',  # Code blocks
        ]
        logger.info("📝 PromptExtractor initialized")

    async def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract the actual prompt from the user's message.

        Returns:
            {
                "original_text": str,
                "extracted_prompt": str,
                "extraction_method": str,
                "confidence": float,
                "is_instruction": bool,
                "instruction_text": str
            }
        """
        if not text or not text.strip():
            return {
                "original_text": text or "",
                "extracted_prompt": text or "",
                "extraction_method": "empty",
                "confidence": 0.0,
                "is_instruction": False,
                "instruction_text": ""
            }

        original = text

        # Method 1: Check for quoted text (highest priority)
        quoted_prompt = self._extract_quoted(text)
        if quoted_prompt:
            return {
                "original_text": original,
                "extracted_prompt": quoted_prompt,
                "extraction_method": "quoted",
                "confidence": 0.95,
                "is_instruction": False,
                "instruction_text": ""
            }

        # Method 2: Check for code blocks
        code_block_prompt = self._extract_code_block(text)
        if code_block_prompt:
            return {
                "original_text": original,
                "extracted_prompt": code_block_prompt,
                "extraction_method": "code_block",
                "confidence": 0.95,
                "is_instruction": False,
                "instruction_text": ""
            }

        # Method 3: Check for indicators (e.g., "in accordance to below:")
        indicator_result = self._extract_after_indicator(text)
        if indicator_result:
            prompt, instruction = indicator_result
            return {
                "original_text": original,
                "extracted_prompt": prompt,
                "extraction_method": "indicator",
                "confidence": 0.9,
                "is_instruction": True,
                "instruction_text": instruction
            }

        # Method 4: Check if the entire text is an instruction with no clear prompt
        is_pure_instruction = self._is_pure_instruction(text)
        if is_pure_instruction:
            # The user gave an instruction but no actual prompt
            return {
                "original_text": original,
                "extracted_prompt": text,  # Use the whole text as fallback
                "extraction_method": "instruction_only",
                "confidence": 0.5,
                "is_instruction": True,
                "instruction_text": text
            }

        # Method 5: Default - use the entire text
        return {
            "original_text": original,
            "extracted_prompt": text,
            "extraction_method": "whole_text",
            "confidence": 0.6,
            "is_instruction": False,
            "instruction_text": ""
        }

    def _extract_quoted(self, text: str) -> Optional[str]:
        """Extract text between quotes."""
        for pattern in self.quote_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Return the longest match (probably the main prompt)
                return max(matches, key=len).strip()
        return None

    def _extract_code_block(self, text: str) -> Optional[str]:
        """Extract text from code blocks."""
        code_block_pattern = r'```[a-zA-Z]*\n([\s\S]+?)\n```'
        match = re.search(code_block_pattern, text)
        if match:
            return match.group(1).strip()

        # Also check for inline code blocks
        inline_pattern = r'`([^`]+)`'
        matches = re.findall(inline_pattern, text)
        if len(matches) > 1:
            # If there are multiple, return the longest
            return max(matches, key=len).strip()
        elif matches:
            return matches[0].strip()

        return None

    def _extract_after_indicator(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract text after an indicator like 'in accordance to below:'."""
        text_lower = text.lower()

        # Find the first indicator
        for indicator in self.indicators:
            if indicator in text_lower:
                idx = text_lower.find(indicator)
                if idx != -1:
                    # Extract the instruction text (before the indicator)
                    instruction = text[:idx + len(indicator)].strip()
                    # Extract the prompt (after the indicator)
                    prompt = text[idx + len(indicator):].strip()

                    # Clean up the prompt
                    prompt = prompt.lstrip(':,;.- \n').strip()

                    # Check if the prompt is in quotes after the indicator
                    quoted_prompt = self._extract_quoted(prompt)
                    if quoted_prompt:
                        prompt = quoted_prompt

                    # Check if there's a second indicator (sometimes people chain them)
                    if prompt and ':' in prompt[:50]:
                        # There might be a nested indicator
                        nested = self._extract_after_indicator(prompt)
                        if nested:
                            nested_prompt, nested_instruction = nested
                            if nested_prompt:
                                prompt = nested_prompt

                    return prompt, instruction

        return None

    def _is_pure_instruction(self, text: str) -> bool:
        """Check if the text is purely an instruction with no actual prompt."""
        text_lower = text.lower()

        # Check if the text starts with common instruction phrases
        instruction_starts = [
            "generate", "create", "make", "produce", "render",
            "draw", "paint", "sketch", "illustrate", "visualize",
            "i want", "i'd like", "can you", "could you", "please"
        ]

        # Check if the text contains any of these and is relatively short
        has_instruction = any(text_lower.startswith(phrase) for phrase in instruction_starts)
        has_indicator = any(indicator in text_lower for indicator in self.indicators)

        if has_instruction or has_indicator:
            # If the text is short or doesn't have a clear prompt, it's likely pure instruction
            return True

        return False

    def get_info(self) -> Dict[str, Any]:
        """Return information about the extractor."""
        return {
            "name": self.name,
            "type": "PromptExtractor",
            "indicator_count": len(self.indicators),
            "quote_patterns": len(self.quote_patterns)
        }