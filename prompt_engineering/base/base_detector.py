"""Base class for all detectors."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseDetector(ABC):
    """Abstract base class for detectors."""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    async def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect information from text.

        Args:
            text: The input text to analyze
            context: Optional context data

        Returns:
            Dict containing detection results
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Return information about the detector."""
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }