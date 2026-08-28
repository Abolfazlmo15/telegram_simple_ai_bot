"""Tests for VisionEngine."""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import io

from core.engines.analysis.vision_engine import VisionEngine, RestartSearchException
from core.config import Config


@pytest.fixture
def mock_vision_model_manager():
    """Mock VisionModelManager."""
    manager = MagicMock()
    manager.start = MagicMock()
    manager.stop = MagicMock()
    manager.get_available_models = MagicMock(return_value=[
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free"
    ])
    return manager


@pytest.fixture
def mock_image_processor():
    """Mock ImageProcessor."""
    processor = MagicMock()
    processor.process_image = AsyncMock(return_value=Image.new('RGB', (100, 100)))
    return processor


@pytest.fixture
def vision_engine(mock_user_data_manager, mock_vision_model_manager, mock_image_processor):
    """Create a VisionEngine with mocked dependencies."""
    with patch("core.engines.analysis.vision_engine.ImageProcessor", return_value=mock_image_processor):
        engine = VisionEngine(mock_user_data_manager)
        engine.model_manager = mock_vision_model_manager
        engine._client = AsyncMock()
        engine.is_initialized = True
        engine.api_key = "test_api_key"
        engine._hf_token = "test_hf_token"
        engine._blacklist_ttl = 5
        engine._model_failures = {}
        return engine


@pytest.mark.asyncio
async def test_vision_engine_initialization(vision_engine):
    """Test basic initialization."""
    assert vision_engine.is_initialized is True
    assert vision_engine.api_key == "test_api_key"
    assert vision_engine._hf_token == "test_hf_token"


def test_is_model_blacklisted(vision_engine):
    """Test model blacklisting logic."""
    model = "test/model"
    assert vision_engine._is_model_blacklisted(model) is False

    vision_engine._mark_model_failure(model)
    assert vision_engine._is_model_blacklisted(model) is True

    # Expire TTL
    import time
    vision_engine._model_failures[model] = time.time() - 10
    assert vision_engine._is_model_blacklisted(model) is False


def test_get_model_list(vision_engine):
    """Test model list prioritization with priority, dynamic, and fallback models."""
    priority = ["p1", "p2"]
    dynamic = ["d1", "d2"]
    fallback = ["f1", "f2"]

    result = vision_engine._get_model_list(priority, dynamic, fallback)
    assert result == ["p1", "p2", "d1", "d2", "f1", "f2"]

    # Blacklist p2 and d1
    vision_engine._mark_model_failure("p2")
    vision_engine._mark_model_failure("d1")
    result = vision_engine._get_model_list(priority, dynamic, fallback)
    assert result == ["p1", "d2", "f1", "f2"]


@pytest.mark.asyncio
async def test_process_openrouter_success(vision_engine):
    """Test successful processing via OpenRouter."""
    image_bytes = b"fake_image_data"
    context = {'user_id': 1, 'query_text': 'What is in this image?'}

    vision_engine._try_openrouter_models = AsyncMock(return_value=("OpenRouter response", "openrouter/model", 50))

    result, model, tokens = await vision_engine.process(image_bytes, context)

    assert result == "OpenRouter response"
    assert model == "openrouter/model"
    assert tokens == 50
    vision_engine._try_openrouter_models.assert_called_once()


@pytest.mark.asyncio
async def test_process_openrouter_fails_huggingface_success(vision_engine):
    """Test fallback to Hugging Face when OpenRouter fails."""
    image_bytes = b"fake_image_data"
    context = {'user_id': 1, 'query_text': 'Describe this image'}

    vision_engine._try_openrouter_models = AsyncMock(return_value=None)
    vision_engine._try_huggingface = AsyncMock(return_value=("HF response", "huggingface-model", 30))

    result, model, tokens = await vision_engine.process(image_bytes, context)

    assert result == "HF response"
    assert model == "huggingface-model"
    assert tokens == 30


@pytest.mark.asyncio
async def test_process_all_providers_fail_fallback(vision_engine):
    """Test PIL metadata fallback when all providers fail."""
    image_bytes = b"fake_image_data"
    context = {'user_id': 1}

    vision_engine._try_openrouter_models = AsyncMock(return_value=None)
    vision_engine._try_huggingface = AsyncMock(return_value=None)

    # Timer does not restart
    async def timer_side_effect(*args, **kwargs):
        return None
    vision_engine._search_timer = AsyncMock(side_effect=timer_side_effect)

    result, model, tokens = await vision_engine.process(image_bytes, context)

    assert model == "pil-fallback"
    assert "Image Analysis Unavailable" in result
    assert tokens > 0


@pytest.mark.asyncio
async def test_process_timer_restart(vision_engine):
    """Test that restart logic works when timer raises RestartSearchException."""
    image_bytes = b"fake_image_data"
    context = {'user_id': 1}

    call_count = 0

    async def try_openrouter_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Yield control to event loop so timer task can run
            await asyncio.sleep(0)
            return None
        return ("Success after restart", "restart/model", 100)

    vision_engine._try_openrouter_models = AsyncMock(side_effect=try_openrouter_side_effect)
    vision_engine._try_huggingface = AsyncMock(return_value=None)

    async def timer_side_effect(*args, **kwargs):
        raise RestartSearchException("restart")

    vision_engine._search_timer = AsyncMock(side_effect=timer_side_effect)

    vision_engine._clear_blacklist = MagicMock()
    vision_engine.model_manager.get_available_models = MagicMock(return_value=["restart/model"])

    result, model, tokens = await vision_engine.process(image_bytes, context)

    assert result == "Success after restart"
    assert model == "restart/model"
    assert tokens == 100
    assert vision_engine._clear_blacklist.called

@pytest.mark.asyncio
async def test_process_image_processing_failure(vision_engine):
    """Test that image processing error returns friendly error."""
    vision_engine.image_processor.process_image = AsyncMock(side_effect=Exception("Processing error"))

    result, model, tokens = await vision_engine.process(b"data", {})

    assert model == "image-processing-error"
    assert "Image Processing Failed" in result


@pytest.mark.asyncio
async def test_try_openrouter_models_blacklisting(vision_engine):
    """Test that models are blacklisted on non-retryable errors."""
    base64_image = "fake_base64"
    query = "test"
    model_list = ["bad/model1", "bad/model2"]

    from httpx import HTTPStatusError, Response, Request

    def call_side_effect(*args, **kwargs):
        raise HTTPStatusError(
            message="Not Found",
            request=MagicMock(spec=Request),
            response=MagicMock(spec=Response, status_code=404)
        )

    vision_engine._call_openrouter = AsyncMock(side_effect=call_side_effect)

    result = await vision_engine._try_openrouter_models(model_list, base64_image, query)
    assert result is None
    assert vision_engine._is_model_blacklisted("bad/model1") is True
    assert vision_engine._is_model_blacklisted("bad/model2") is True