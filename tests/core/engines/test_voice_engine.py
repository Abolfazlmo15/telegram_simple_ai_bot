"""Tests for VoiceEngine."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.engines.analysis.voice_engine import VoiceEngine, RestartSearchException
from core.config import Config


@pytest.fixture
def voice_engine(mock_user_data_manager):
    """Create a VoiceEngine with mocked dependencies."""
    engine = VoiceEngine(mock_user_data_manager)
    engine.is_initialized = True
    engine._client = AsyncMock()
    engine.model = None  # No local Whisper by default
    engine._blacklist_ttl = 5
    engine._model_failures = {}
    engine.openrouter_models = ["model1", "model2", "model3"]
    return engine


@pytest.mark.asyncio
async def test_voice_engine_initialization(voice_engine):
    """Test basic initialization."""
    assert voice_engine.is_initialized is True


def test_is_model_blacklisted(voice_engine):
    """Test blacklisting."""
    model = "test/model"
    assert voice_engine._is_model_blacklisted(model) is False
    voice_engine._mark_model_failure(model)
    assert voice_engine._is_model_blacklisted(model) is True


def test_get_model_list(voice_engine):
    """Test model list prioritization."""
    priority = ["p1", "p2"]
    fallback = ["f1", "f2", "f3"]
    result = voice_engine._get_model_list(priority, fallback)
    assert result == ["p1", "p2", "f1", "f2", "f3"]

    # Blacklist p2 and f1
    voice_engine._mark_model_failure("p2")
    voice_engine._mark_model_failure("f1")
    result = voice_engine._get_model_list(priority, fallback)
    assert result == ["p1", "f2", "f3"]


@pytest.mark.asyncio
async def test_transcribe_local_whisper_success(voice_engine):
    """Test Whisper local transcription."""
    mock_model = MagicMock()
    mock_model.transcribe = MagicMock(return_value={"text": "Hello world"})

    voice_engine.model = mock_model
    voice_engine.model_name = "base"

    with patch("tempfile.NamedTemporaryFile") as mock_ntf, \
         patch("os.unlink") as mock_unlink:
        mock_ntf.return_value.__enter__.return_value.name = "/tmp/test.ogg"

        result, model, tokens = await voice_engine.transcribe(b"audio_data", {})

        assert result == "Hello world"
        assert model == "whisper-base"
        assert tokens > 0
        mock_unlink.assert_called_once_with("/tmp/test.ogg")


@pytest.mark.asyncio
async def test_transcribe_openrouter_fallback(voice_engine):
    """Test fallback to OpenRouter when Whisper fails."""
    voice_engine.model = None  # no local
    voice_engine._try_stt_models = AsyncMock(return_value=("Transcribed text", "openrouter/model", 20))

    result, model, tokens = await voice_engine.transcribe(b"audio_data", {})

    assert result == "Transcribed text"
    assert model == "openrouter/model"
    assert tokens == 20


@pytest.mark.asyncio
async def test_transcribe_all_fail_no_restart(voice_engine):
    """Test that transcribe raises when all models fail and no restart."""
    voice_engine._try_stt_models = AsyncMock(return_value=None)
    voice_engine._search_timer = AsyncMock(return_value=None)

    with pytest.raises(Exception, match="All transcription methods failed"):
        await voice_engine.transcribe(b"audio_data", {})


@pytest.mark.asyncio
async def test_transcribe_restart_triggered(voice_engine):
    """Test restart logic when timer raises RestartSearchException."""
    call_count = 0

    async def try_models_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Yield control to event loop so timer task can run
            await asyncio.sleep(0)
            return None
        return ("Success after restart", "restart/model", 30)

    voice_engine._try_stt_models = AsyncMock(side_effect=try_models_side_effect)

    async def timer_side_effect(*args, **kwargs):
        raise RestartSearchException("restart")

    voice_engine._search_timer = AsyncMock(side_effect=timer_side_effect)

    voice_engine._clear_blacklist = MagicMock()
    voice_engine._get_model_list = MagicMock(return_value=["restart/model"])

    result, model, tokens = await voice_engine.transcribe(b"audio_data", {})

    assert result == "Success after restart"
    assert model == "restart/model"
    assert tokens == 30
    assert voice_engine._clear_blacklist.called

@pytest.mark.asyncio
async def test_try_stt_models_parallel_and_sequential(voice_engine):
    """Test that parallel testing is attempted and falls back to sequential."""
    models = ["m1", "m2", "m3"]
    audio = b"data"

    # Disable parallel testing to force sequential fallback
    with patch("core.config.Config.ENABLE_PARALLEL_MODEL_TESTING", False):
        # Mock sequential to succeed on m2: m1 must fail both attempts
        # HTTP_MAX_RETRIES is 2, so we need 2 exceptions for m1
        voice_engine._call_openrouter_stt = AsyncMock(side_effect=[
            Exception("m1 failed"),
            Exception("m1 failed again"),  # second attempt for m1 fails
            ("transcription", 10),         # m2 succeeds
        ])

        result = await voice_engine._try_stt_models(models, audio)
        assert result is not None
        response, model, tokens = result
        assert response == "transcription"
        assert model == "m2"
        assert tokens == 10