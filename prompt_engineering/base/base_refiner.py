"""Base class for all refiners."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseRefiner(ABC):
    """Abstract base class for refiners."""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    async def refine(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Refine the input text.

        Args:
            text: The text to refine
            context: Optional context data

        Returns:
            Refined text
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Return information about the refiner."""
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }