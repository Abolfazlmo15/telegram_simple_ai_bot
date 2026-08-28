"""Tests for StyleDetector."""
import pytest
from unittest.mock import MagicMock
from prompt_engineering.detectors.style_detector import StyleDetector
from core.config import Config


@pytest.fixture
def style_detector():
    """Create a StyleDetector instance."""
    return StyleDetector()


@pytest.mark.asyncio
async def test_detect_no_style(style_detector):
    """Test that a prompt with no style returns no_style."""
    result = await style_detector.detect("A cat sitting on a table")
    assert result["style"] == "no_style"
    assert result["confidence"] == 0.0
    assert result["primary_style"] == "no_style"
    assert result["secondary_styles"] == []


@pytest.mark.asyncio
async def test_detect_anime_style(style_detector):
    """Test detection of anime style."""
    result = await style_detector.detect("anime style, a beautiful girl")
    assert result["style"] == "anime"
    assert result["confidence"] >= 0.8
    assert result["primary_style"] == "anime"
    assert "anime" in result["style_keywords"]


@pytest.mark.asyncio
async def test_detect_realistic_style(style_detector):
    """Test detection of realistic style."""
    result = await style_detector.detect("realistic photography of a landscape")
    assert result["style"] == "realistic"
    assert result["confidence"] >= 0.8


@pytest.mark.asyncio
async def test_detect_compound_phrase(style_detector):
    """Test detection of compound phrases like 'like a painting'."""
    result = await style_detector.detect("like a watercolor painting")
    # Depending on the implementation, it might detect watercolor or oil_painting
    # We'll check that it detects some style and not no_style
    assert result["style"] != "no_style"
    assert result["confidence"] >= 0.6


@pytest.mark.asyncio
async def test_detect_multiple_styles_primary_selected(style_detector):
    """Test that when multiple styles are detected, primary is highest confidence."""
    # Prompt with both anime and realistic keywords
    result = await style_detector.detect("anime style realistic portrait")
    # anime and realistic both present; should pick one as primary
    assert result["primary_style"] in ["anime", "realistic"]
    # Secondary should contain the other
    assert len(result["secondary_styles"]) >= 1
    # Confidence should be > 0
    assert result["confidence"] > 0


@pytest.mark.asyncio
async def test_detect_with_context_history(style_detector):
    """Test that context history can influence style detection."""
    context = {
        'history': [
            {'type': 'generated_image', 'model_used': 'gen_image:anime'}
        ]
    }
    result = await style_detector.detect("another image please", context)
    # Now the context should yield "anime" style
    assert result["style"] == "anime"
    assert result["primary_style"] == "anime"

@pytest.mark.asyncio
async def test_recommended_models(style_detector):
    """Test that recommended_models are returned for a style."""
    result = await style_detector.detect("anime style")
    assert "recommended_models" in result
    models = result["recommended_models"]
    # Should be from Config.STYLE_MODEL_MAP['anime'] or default
    assert len(models) > 0


def test_style_aliases(style_detector):
    """Test that style aliases are properly built."""
    # 'photo' should map to 'realistic'
    assert style_detector._style_aliases.get("photo") == "realistic"
    assert style_detector._style_aliases.get("pixel art") == "pixel"