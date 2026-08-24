"""Base classes for the prompt engineering system."""
from prompt_engineering.base.base_detector import BaseDetector
from prompt_engineering.base.base_refiner import BaseRefiner
from prompt_engineering.base.base_memory import BaseMemory

__all__ = [
    'BaseDetector',
    'BaseRefiner',
    'BaseMemory'
]