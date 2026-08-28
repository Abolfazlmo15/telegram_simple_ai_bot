"""Tests for IntentDetector."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from prompt_engineering.detectors.intent_detector import IntentDetector
from prompt_engineering.memory.generation_context import GenerationContext


@pytest.fixture
def intent_detector():
    """Create an IntentDetector instance."""
    detector = IntentDetector()
    return detector


@pytest.fixture
def intent_detector_with_context():
    """Create an IntentDetector with mocked GenerationContext."""
    gen_context = MagicMock(spec=GenerationContext)
    gen_context.has_recent_generation = AsyncMock(return_value=True)
    gen_context.get_last_generation = AsyncMock(return_value={
        "prompt": "original prompt",
        "model_used": "test/model"
    })
    detector = IntentDetector(gen_context)
    return detector


@pytest.mark.asyncio
async def test_detect_text_analysis(intent_detector):
    """Test that plain text returns text_analysis intent."""
    result = await intent_detector.detect("What is the weather?")
    assert result["intent"] == "text_analysis"
    assert result["is_correction"] is False
    assert result["confidence"] == 0.7
    assert result["extracted_prompt"] == "What is the weather?"


@pytest.mark.asyncio
async def test_detect_image_generation_keyword(intent_detector):
    """Test detection of image generation keywords."""
    text = "generate an image of a sunset"
    result = await intent_detector.detect(text)
    assert result["intent"] == "image_generation"
    assert result["confidence"] >= 0.9
    assert "sunset" in result["extracted_prompt"]
    assert result["is_correction"] is False


@pytest.mark.asyncio
async def test_detect_image_generation_vague_prompt(intent_detector):
    """Test that vague prompts are not detected as image generation."""
    text = "generate something"
    result = await intent_detector.detect(text)
    # Should fall back to text_analysis because prompt is too vague
    assert result["intent"] == "text_analysis"
    assert result["confidence"] == 0.7


@pytest.mark.asyncio
async def test_detect_image_generation_pattern(intent_detector):
    """Test detection using regex patterns."""
    text = "picture of a cat"
    result = await intent_detector.detect(text)
    assert result["intent"] == "image_generation"
    assert "cat" in result["extracted_prompt"]


@pytest.mark.asyncio
async def test_detect_voice_generation(intent_detector):
    """Test detection of voice generation keywords."""
    text = "say this hello world"
    result = await intent_detector.detect(text)
    assert result["intent"] == "voice_generation"
    assert result["confidence"] >= 0.9
    assert "hello world" in result["extracted_prompt"]


@pytest.mark.skip(reason="Mode change detection is handled by ModeDetector, not IntentDetector")
@pytest.mark.asyncio
async def test_detect_mode_change_voice(intent_detector):
    """Test detection of voice mode switch."""
    text = "talk to me"
    result = await intent_detector.detect(text)
    assert result["intent"] == "mode_change"
    assert result["mode"] == "voice"
    assert result["confidence"] >= 0.9

@pytest.mark.skip(reason="Mode change detection is handled by ModeDetector, not IntentDetector")
@pytest.mark.asyncio
async def test_detect_mode_change_text(intent_detector):
    """Test detection of text mode switch."""
    text = "type it"
    result = await intent_detector.detect(text)
    assert result["intent"] == "mode_change"
    assert result["mode"] == "text"
    assert result["confidence"] >= 0.9

@pytest.mark.asyncio
async def test_detect_correction_with_recent_generation(intent_detector_with_context):
    """Test correction detection when there is a recent generation."""
    text = "make it more colorful"
    result = await intent_detector_with_context.detect(
        text,
        context={'user_id': 1}
    )
    assert result["intent"] == "image_generation"
    assert result["is_correction"] is True
    # The correction type may vary; we just check it's not empty
    assert "correction_type" in result
    assert result["original_prompt"] == "original prompt"


@pytest.mark.asyncio
async def test_detect_correction_without_recent_generation(intent_detector):
    """Test that correction keywords without recent generation fall back."""
    # No generation_context set, so no correction detection
    text = "make it better"
    result = await intent_detector.detect(text, context={'user_id': 1})
    # Should be text_analysis because no context
    assert result["intent"] == "text_analysis"
    assert result["is_correction"] is False


@pytest.mark.asyncio
async def test_extract_prompt_from_keyword(intent_detector):
    """Test extraction of prompt after keyword."""
    text = "generate an image of a beautiful mountain lake"
    extracted = intent_detector._extract_prompt_from_keyword(
        text, "generate an image of"
    )
    assert extracted == "a beautiful mountain lake"

    # With punctuation
    text = "say this: hello world"
    extracted = intent_detector._extract_prompt_from_keyword(text, "say this")
    assert extracted == "hello world"


def test_detect_patterns(intent_detector):
    """Test regex pattern detection."""
    # Should detect via pattern
    result = intent_detector._detect_patterns("image of a dog")
    assert result is not None
    assert result["intent"] == "image_generation"
    assert "dog" in result["extracted_prompt"]