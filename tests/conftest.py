"""Shared fixtures for all tests."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from core.config import Config
from core.managers.user_data_manager import UserDataManager
from core.managers.model_manager import ModelManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_user_data_manager():
    """Mock UserDataManager."""
    manager = MagicMock(spec=UserDataManager)

    # ---------- SYNCHRONOUS methods ----------
    manager.get_cached_response = MagicMock(return_value=None)
    manager.save_to_cache = MagicMock()

    # ---------- ASYNCHRONOUS methods ----------
    manager.load_user_data = AsyncMock(return_value={"history": []})
    manager.get_user_model_priority = AsyncMock(return_value=None)
    manager.get_preferences = AsyncMock(return_value={})
    manager.get_custom_instructions = AsyncMock(return_value="")
    manager.get_response_mode = AsyncMock(return_value="text")
    manager.get_response_style = AsyncMock(return_value="balanced")
    manager.get_preferred_style = AsyncMock(return_value="no_style")
    manager.get_voice_speed = AsyncMock(return_value=1.0)
    manager.get_voice_style = AsyncMock(return_value="neutral")
    manager.save_model_priority = AsyncMock(return_value=True)
    manager.save_image_generation_priority = AsyncMock(return_value=True)
    manager.add_message_to_history = AsyncMock()
    manager.add_image_to_history = AsyncMock()
    manager.add_voice_to_history = AsyncMock()
    manager.search_history = AsyncMock(return_value=[])
    # Add any other async methods used in tests

    return manager


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager with fast and smart models."""
    manager = MagicMock(spec=ModelManager)
    manager.get_fast_models = MagicMock(return_value=["fast/model1", "fast/model2"])
    manager.get_smart_models = MagicMock(return_value=["smart/model1", "smart/model2"])
    manager.is_running = True
    manager.start = MagicMock()
    manager.stop = MagicMock()
    return manager


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient."""
    client = AsyncMock()
    client.post = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def sample_text_prompt():
    """Sample user text prompt."""
    return "What is the capital of France?"


@pytest.fixture
def sample_text_response():
    """Sample assistant response."""
    return "The capital of France is Paris."


@pytest.fixture
def mock_prompt_library():
    """Mock PromptLibrary."""
    with patch("core.utils.prompt_library.PromptLibrary") as mock:
        instance = mock.return_value
        instance.detect_category = MagicMock(return_value="casual_conversation")
        instance.get_prompt = MagicMock(return_value=MagicMock(
            system_message="You are a helpful assistant.",
            temperature=0.7,
            max_tokens=500
        ))
        yield instance