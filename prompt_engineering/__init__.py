"""Prompt Engineering module for intelligent intent detection and prompt refinement."""
from prompt_engineering.base import (
    BaseDetector,
    BaseRefiner,
    BaseMemory
)
from prompt_engineering.detectors import (
    IntentDetector,
    StyleDetector,
    PromptExtractor,
    ContextAnalyzer
)
from prompt_engineering.refiners import (
    PromptRefiner,
    TemplateApplier,
    NegativePromptGenerator
)
from prompt_engineering.templates import (
    TemplateManager,
    GithubFetcher,
    LocalTemplates
)
from prompt_engineering.memory import (
    GenerationContext,
    CorrectionDetector,
    IterativeRefiner,
    ConversationState,
    ConversationMode,
    ModeDetector
)
from prompt_engineering.formatters import (
    TelegramFormatter
)

__all__ = [
    'BaseDetector',
    'BaseRefiner',
    'BaseMemory',
    'IntentDetector',
    'StyleDetector',
    'PromptExtractor',
    'ContextAnalyzer',
    'PromptRefiner',
    'TemplateApplier',
    'NegativePromptGenerator',
    'TemplateManager',
    'GithubFetcher',
    'LocalTemplates',
    'GenerationContext',
    'CorrectionDetector',
    'IterativeRefiner',
    'ConversationState',
    'ConversationMode',
    'ModeDetector',
    'TelegramFormatter'
]