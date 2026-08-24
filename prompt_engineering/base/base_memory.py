"""Base class for memory management."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime


class BaseMemory(ABC):
    """Abstract base class for memory management."""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    async def store(self, key: str, value: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Store a value with a key."""
        pass

    @abstractmethod
    async def retrieve(self, key: str, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Retrieve a value by key."""
        pass

    @abstractmethod
    async def delete(self, key: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Delete a value by key."""
        pass

    @abstractmethod
    async def exists(self, key: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a key exists."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Return information about the memory."""
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }