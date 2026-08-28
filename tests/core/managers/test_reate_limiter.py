"""Tests for RateLimiter."""
import pytest
import time
from core.managers.rate_limiter import RateLimiter


@pytest.fixture
def rate_limiter():
    """Create a RateLimiter instance."""
    return RateLimiter(max_requests=3, window_seconds=1)


@pytest.mark.asyncio
async def test_rate_limiter_initialization(rate_limiter):
    """Test basic initialization."""
    assert rate_limiter.max_requests == 3
    assert rate_limiter.window_seconds == 1
    assert len(rate_limiter._history) == 0


@pytest.mark.asyncio
async def test_rate_limiter_allow_request_within_limit(rate_limiter):
    """Test that requests within limit are allowed."""
    user_id = 123

    for i in range(3):
        allowed, remaining = await rate_limiter.check(user_id)
        assert allowed is True
        assert remaining == 3 - (i + 1)

    # Check history length
    assert len(rate_limiter._history[user_id]) == 3


@pytest.mark.asyncio
async def test_rate_limiter_block_after_limit(rate_limiter):
    """Test that requests exceeding limit are blocked."""
    user_id = 456

    # Fill to limit
    for _ in range(3):
        await rate_limiter.check(user_id)

    # Next request should be blocked
    allowed, remaining = await rate_limiter.check(user_id)
    assert allowed is False
    assert remaining == 0


@pytest.mark.asyncio
async def test_rate_limiter_expiry(rate_limiter):
    """Test that old entries expire after window."""
    user_id = 789

    # Add requests
    for _ in range(3):
        await rate_limiter.check(user_id)

    # Wait for window to expire
    time.sleep(1.1)

    # Should be allowed again
    allowed, remaining = await rate_limiter.check(user_id)
    assert allowed is True
    assert remaining == 2  # max 3 - 1 new request

    # Old entries should be cleared
    assert len(rate_limiter._history[user_id]) == 1


def test_get_remaining(rate_limiter):
    """Test get_remaining method."""
    user_id = 111
    # Initially full
    assert rate_limiter.get_remaining(user_id) == 3

    # After one request
    import asyncio
    asyncio.run(rate_limiter.check(user_id))
    assert rate_limiter.get_remaining(user_id) == 2


def test_clear_cache(rate_limiter):
    """Test clear_cache clears all history."""
    import asyncio
    asyncio.run(rate_limiter.check(1))
    asyncio.run(rate_limiter.check(2))
    assert len(rate_limiter._history) == 2

    rate_limiter.clear_cache()
    assert len(rate_limiter._history) == 0


def test_get_info(rate_limiter):
    """Test get_info returns correct info."""
    info = rate_limiter.get_info()
    assert info["type"] == "RateLimiter"
    assert info["max_requests"] == 3
    assert info["window_seconds"] == 1