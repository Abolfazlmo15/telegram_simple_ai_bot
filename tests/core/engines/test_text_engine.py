"""Tests for TextEngine."""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import HTTPStatusError, Response, Request

from core.engines.analysis.text_engine import TextEngine, RestartSearchException
from core.config import Config


@pytest.fixture
def text_engine(mock_user_data_manager, mock_model_manager):
    """Create a TextEngine instance with mocked dependencies."""
    engine = TextEngine(mock_user_data_manager)
    engine.model_manager = mock_model_manager
    engine.is_initialized = True
    engine._client = AsyncMock()
    engine._blacklist_ttl = 5
    engine._model_failures = {}
    return engine


@pytest.mark.asyncio
async def test_text_engine_initialization(text_engine):
    """Test basic initialization."""
    assert text_engine.is_initialized is True
    assert text_engine.api_key == Config.OPENROUTER_API_KEY
    assert text_engine.base_url == Config.OPENROUTER_BASE_URL


def test_is_model_blacklisted(text_engine):
    """Test model blacklisting."""
    model = "test/model"
    assert text_engine._is_model_blacklisted(model) is False

    text_engine._mark_model_failure(model)
    assert text_engine._is_model_blacklisted(model) is True

    # Simulate TTL expiry
    text_engine._model_failures[model] = time.time() - 10
    assert text_engine._is_model_blacklisted(model) is False
    assert model not in text_engine._model_failures


def test_get_model_list_priority(text_engine):
    """Test model list prioritization with blacklisting."""
    priority = ["p1", "p2"]
    fallback = ["f1", "f2", "f3"]

    # No blacklist
    result = text_engine._get_model_list(priority, fallback)
    assert result == ["p1", "p2", "f1", "f2", "f3"]

    # Blacklist p2 and f1
    text_engine._mark_model_failure("p2")
    text_engine._mark_model_failure("f1")
    result = text_engine._get_model_list(priority, fallback)
    assert result == ["p1", "f2", "f3"]


@pytest.mark.asyncio
async def test_process_cache_hit(text_engine, mock_user_data_manager):
    """Test that process returns cached response if available."""
    user_text = "test prompt"
    cached_response = ("cached answer", "cached_category", 12345)

    # Use MagicMock (synchronous) instead of AsyncMock
    mock_user_data_manager.get_cached_response = MagicMock(return_value=cached_response)

    result, model, tokens = await text_engine.process(
        user_text,
        context={'user_id': 1}
    )

    assert result == "cached answer"
    assert model == "cache"
    assert tokens == 0
    mock_user_data_manager.get_cached_response.assert_called_once_with(user_text)

@pytest.mark.asyncio
async def test_process_api_call_success(text_engine, mock_user_data_manager):
    """Test successful API call."""
    user_text = "Hello, world!"
    # Use MagicMock (synchronous) for no cache hit
    mock_user_data_manager.get_cached_response = MagicMock(return_value=None)

    text_engine._execute_model_search = AsyncMock(return_value=("AI response", "test/model", 100))

    result, model, tokens = await text_engine.process(
        user_text,
        context={'user_id': 1}
    )

    assert result == "AI response"
    assert model == "test/model"
    assert tokens == 100
    mock_user_data_manager.save_to_cache.assert_called_once_with(user_text, "AI response", "casual_conversation")

@pytest.mark.asyncio
async def test_process_all_models_fail_no_restart(text_engine):
    """Test that process raises when all models fail and no restart triggered."""
    user_text = "test"
    text_engine._execute_model_search = AsyncMock(return_value=None)
    async def timer_side_effect(*args, **kwargs):
        return None
    text_engine._search_timer = AsyncMock(side_effect=timer_side_effect)

    with pytest.raises(Exception, match="All text models failed. No restart triggered."):
        await text_engine.process(user_text, context={'user_id': 1})


@pytest.mark.asyncio
async def test_process_restart_triggered(text_engine):
    """Test that process restarts when timer raises RestartSearchException."""
    user_text = "test"
    call_count = 0

    async def execute_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Yield control to event loop so timer task can run
            await asyncio.sleep(0)
            return None
        return ("success", "restarted/model", 200)

    text_engine._execute_model_search = AsyncMock(side_effect=execute_side_effect)

    # Make _search_timer raise RestartSearchException immediately
    async def timer_side_effect(*args, **kwargs):
        raise RestartSearchException("restart")

    text_engine._search_timer = AsyncMock(side_effect=timer_side_effect)

    text_engine._clear_blacklist = MagicMock()
    text_engine._get_model_list = MagicMock(return_value=["restarted/model"])

    result, model, tokens = await text_engine.process(user_text, context={'user_id': 1})

    assert result == "success"
    assert model == "restarted/model"
    assert tokens == 200
    assert text_engine._clear_blacklist.called
    assert text_engine._get_model_list.called

@pytest.mark.asyncio
async def test_process_network_error_handling(text_engine):
    """Test that network errors from _execute_model_search are propagated."""
    async def execute_side_effect(*args, **kwargs):
        raise Exception("Network error")

    text_engine._execute_model_search = AsyncMock(side_effect=execute_side_effect)
    # No restart – timer returns normally
    text_engine._search_timer = AsyncMock(return_value=None)

    with pytest.raises(Exception, match="Network error"):
        await text_engine.process("test", context={'user_id': 1})