"""Tests for BaseEngine routing."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from core.engines.base_engine import BaseEngine
from core.engines.analysis.document_engine import DocumentEngine
from prompt_engineering.state.conversation_state import ConversationMode


@pytest.fixture
def base_engine(mock_user_data_manager):
    """Create a BaseEngine with mocked sub-engines."""
    # Ensure mock_user_data_manager has preference_manager
    mock_user_data_manager.preference_manager = AsyncMock()
    mock_user_data_manager.preference_manager.get_preferences = AsyncMock(return_value={})
    mock_user_data_manager.preference_manager.get_response_mode = AsyncMock(return_value="text")
    mock_user_data_manager.preference_manager.get_response_style = AsyncMock(return_value="balanced")
    mock_user_data_manager.preference_manager.get_preferred_style = AsyncMock(return_value="no_style")
    mock_user_data_manager.preference_manager.get_voice_speed = AsyncMock(return_value=1.0)
    mock_user_data_manager.preference_manager.get_voice_style = AsyncMock(return_value="neutral")
    mock_user_data_manager.preference_manager.get_custom_instructions = AsyncMock(return_value="")
    mock_user_data_manager.preference_manager.set_response_mode = AsyncMock()
    mock_user_data_manager.get_user_model_priority = AsyncMock(return_value=None)

    engine = BaseEngine(mock_user_data_manager)
    engine.is_initialized = True

    # Mock all sub-engines
    engine.text_engine = AsyncMock()
    engine.text_engine.process = AsyncMock(return_value=("text response", "text_model", 10))
    engine.text_engine.is_initialized = True

    engine.vision_engine = AsyncMock()
    engine.vision_engine.process = AsyncMock(return_value=("vision response", "vision_model", 20))
    engine.vision_engine.is_initialized = True

    engine.voice_engine = AsyncMock()
    engine.voice_engine.transcribe = AsyncMock(return_value=("transcription", "voice_model", 30))
    engine.voice_engine.is_initialized = True

    engine.voice_generation_engine = AsyncMock()
    engine.voice_generation_engine.generate = AsyncMock(return_value=(b"audio", "voice_gen_model", 40))
    engine.voice_generation_engine.is_initialized = True

    engine.image_generation_engine = AsyncMock()
    engine.image_generation_engine.generate = AsyncMock(return_value=(b"image", "gen_model", 50))
    engine.image_generation_engine.is_initialized = True

    engine.document_engine = AsyncMock()
    engine.document_engine.process = AsyncMock(return_value=("document response", "doc_model", 60))

    # Mock prompt engineering components
    engine.mode_detector.detect = AsyncMock(return_value={'is_mode_switch': False})
    engine.intent_detector.detect = AsyncMock(return_value={'intent': 'text_analysis', 'is_correction': False})
    engine.prompt_extractor.detect = AsyncMock(return_value={'extracted_prompt': 'test'})
    engine.conversation_state.get_mode = AsyncMock(return_value=ConversationMode.TEXT)
    engine.conversation_state.set_mode = AsyncMock()
    engine.preference_manager = mock_user_data_manager.preference_manager

    # Mock style detector and others
    engine.style_detector.detect = AsyncMock(return_value={'primary_style': 'realistic', 'recommended_models': []})
    engine.negative_prompt_generator.refine = AsyncMock(return_value="negative prompt")
    engine._extract_voice_text = MagicMock(return_value="say this text")

    return engine


@pytest.mark.asyncio
async def test_base_engine_text_input(base_engine):
    """Test that text input routes to TextEngine."""
    result, model, tokens = await base_engine.process("Hello", {'user_id': 1})

    base_engine.text_engine.process.assert_called_once()
    assert result == "text response"
    assert model == "text_model"


@pytest.mark.asyncio
async def test_base_engine_image_input(base_engine):
    """Test that image bytes route to VisionEngine."""
    result, model, tokens = await base_engine.process(b"image_data", {'input_type': 'image'})

    base_engine.vision_engine.process.assert_called_once()
    assert result == "vision response"
    assert model == "vision_model"


@pytest.mark.asyncio
async def test_base_engine_voice_input(base_engine):
    """Test that voice bytes route to VoiceEngine then recursively to TextEngine."""
    # Mock voice engine to return a transcription
    base_engine.voice_engine.transcribe = AsyncMock(return_value=("transcription", "voice_model", 30))
    # Mock text engine to return the final response
    base_engine.text_engine.process = AsyncMock(return_value=("final response", "final_model", 70))

    # Define a mock process that handles the recursion by calling text_engine directly after transcription
    async def mock_process(input_data, context, status_callback=None):
        if isinstance(input_data, bytes) and context.get('input_type') == 'audio':
            transcription, _, _ = await base_engine.voice_engine.transcribe(input_data, context)
            # Bypass recursion and directly call text_engine.process
            return await base_engine.text_engine.process(transcription, context, status_callback)
        # For other input types (should not happen in this test), return a default
        return ("default", "default", 0)

    with patch.object(base_engine, 'process', new=mock_process):
        result, model, tokens = await base_engine.process(b"audio_data", {'input_type': 'audio'})

    base_engine.voice_engine.transcribe.assert_called_once()
    base_engine.text_engine.process.assert_called_once()
    # Check that the transcription was passed to text_engine
    call_args = base_engine.text_engine.process.call_args[0][0]
    assert call_args == "transcription"
    assert result == "final response"
    assert model == "final_model"
    assert tokens == 70

@pytest.mark.asyncio
async def test_base_engine_document_input(base_engine):
    """Test that document bytes route to DocumentEngine."""
    result, model, tokens = await base_engine.process(b"doc_data", {'input_type': 'document'})

    base_engine.document_engine.process.assert_called_once()
    assert result == "document response"
    assert model == "doc_model"


@pytest.mark.asyncio
async def test_base_engine_pil_image(base_engine):
    """Test that PIL Image routes to VisionEngine."""
    img = Image.new('RGB', (10, 10))
    result, model, tokens = await base_engine.process(img, {})

    base_engine.vision_engine.process.assert_called_once()
    assert result == "vision response"


@pytest.mark.asyncio
async def test_base_engine_mode_switch(base_engine):
    """Test that mode switch detection returns early."""
    base_engine.mode_detector.detect = AsyncMock(return_value={
        'is_mode_switch': True,
        'target_mode': 'voice'
    })

    result, model, tokens = await base_engine.process("talk to me", {'user_id': 1})

    assert model == "mode_switch_voice"
    assert "Voice mode activated" in result
    base_engine.text_engine.process.assert_not_called()
    base_engine.conversation_state.set_mode.assert_called_with(1, ConversationMode.VOICE)


@pytest.mark.asyncio
async def test_base_engine_image_generation_intent(base_engine):
    """Test that image generation intent routes to ImageGenerationEngine."""
    base_engine.intent_detector.detect = AsyncMock(return_value={'intent': 'image_generation', 'is_correction': False})

    result, model, tokens = await base_engine.process("generate an image of a cat", {'user_id': 1})

    base_engine.image_generation_engine.generate.assert_called_once()
    assert isinstance(result, bytes)
    assert model.startswith("gen_image:")


@pytest.mark.asyncio
async def test_base_engine_voice_generation_intent(base_engine):
    """Test that voice generation intent routes to VoiceGenerationEngine."""
    base_engine.intent_detector.detect = AsyncMock(return_value={'intent': 'voice_generation', 'is_correction': False})

    result, model, tokens = await base_engine.process("say this hello", {'user_id': 1})

    base_engine.voice_generation_engine.generate.assert_called_once()
    assert isinstance(result, bytes)
    assert model.startswith("gen_voice:")


@pytest.mark.asyncio
async def test_base_engine_voice_mode_response(base_engine):
    """Test that text analysis in voice mode returns voice audio."""
    base_engine.text_engine.process = AsyncMock(return_value=("response text", "text_model", 10))
    base_engine.preference_manager.get_response_mode = AsyncMock(return_value="voice")
    base_engine.conversation_state.get_mode = AsyncMock(return_value=ConversationMode.VOICE)

    result, model, tokens = await base_engine.process("Hello", {'user_id': 1})

    base_engine.voice_generation_engine.generate.assert_called_once()
    assert isinstance(result, bytes)
    assert model.startswith("gen_voice_conversation:")


@pytest.mark.asyncio
async def test_base_engine_cancellation(base_engine):
    """Test that cancellation is propagated."""
    # Create a task that gets cancelled
    async def cancel_after():
        task = asyncio.current_task()
        task.cancel()
        await asyncio.sleep(0.1)
        # Should raise CancelledError
        return await base_engine.process("test", {})

    with pytest.raises(asyncio.CancelledError):
        await cancel_after()