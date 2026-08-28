"""Tests for CacheManager."""
import pytest
import time
from core.managers.cache_manager import CacheManager


@pytest.fixture
def cache_manager():
    """Create a CacheManager instance."""
    return CacheManager(max_size=5, default_ttl=2)


def test_cache_set_and_get(cache_manager):
    """Test basic set and get."""
    cache_manager.set("test_ns", "key1", "value1")
    value = cache_manager.get("test_ns", "key1")
    assert value == "value1"


def test_cache_get_missing(cache_manager):
    """Test get returns default for missing key."""
    value = cache_manager.get("test_ns", "missing")
    assert value is None

    value = cache_manager.get("test_ns", "missing", default="default")
    assert value == "default"


def test_cache_ttl_expiry(cache_manager):
    """Test that entries expire after TTL."""
    cache_manager.set("test_ns", "key", "value", ttl_seconds=1)
    assert cache_manager.get("test_ns", "key") == "value"

    time.sleep(1.1)
    assert cache_manager.get("test_ns", "key") is None


def test_cache_delete(cache_manager):
    """Test delete removes entry."""
    cache_manager.set("test_ns", "key", "value")
    assert cache_manager.get("test_ns", "key") == "value"

    deleted = cache_manager.delete("test_ns", "key")
    assert deleted is True
    assert cache_manager.get("test_ns", "key") is None

    deleted = cache_manager.delete("test_ns", "nonexistent")
    assert deleted is False


def test_cache_clear_namespace(cache_manager):
    """Test clear_namespace removes only entries in that namespace."""
    cache_manager.set("ns1", "a", "A")
    cache_manager.set("ns1", "b", "B")
    cache_manager.set("ns2", "c", "C")

    assert cache_manager.get("ns1", "a") == "A"
    count = cache_manager.clear_namespace("ns1")
    assert count == 2
    assert cache_manager.get("ns1", "a") is None
    assert cache_manager.get("ns2", "c") == "C"


def test_cache_clear(cache_manager):
    """Test clear removes all entries."""
    cache_manager.set("ns1", "a", "A")
    cache_manager.set("ns2", "b", "B")
    cache_manager.clear()
    assert cache_manager.get("ns1", "a") is None
    assert cache_manager.get("ns2", "b") is None


def test_cache_eviction_lru(cache_manager):
    """Test LRU eviction when max_size exceeded."""
    # max_size = 5
    for i in range(5):
        cache_manager.set("ns", f"key{i}", f"value{i}")

    # Cache is full, add another entry -> should evict oldest
    cache_manager.set("ns", "key5", "value5")

    # Oldest should be evicted (key0)
    assert cache_manager.get("ns", "key0") is None
    # Others should exist
    for i in range(1, 6):
        assert cache_manager.get("ns", f"key{i}") == f"value{i}"


def test_cache_get_or_set(cache_manager):
    """Test get_or_set computes and stores value if missing."""
    call_count = 0

    def compute():
        nonlocal call_count
        call_count += 1
        return "computed"

    # First call should compute
    value = cache_manager.get_or_set("ns", "key", compute)
    assert value == "computed"
    assert call_count == 1

    # Second call should return cached
    value = cache_manager.get_or_set("ns", "key", compute)
    assert value == "computed"
    assert call_count == 1


def test_cache_stats(cache_manager):
    """Test get_stats returns correct statistics."""
    cache_manager.get("ns", "missing")
    cache_manager.set("ns", "key", "value")
    cache_manager.get("ns", "key")

    stats = cache_manager.get_stats()
    assert stats["total_entries"] == 1
    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 1
    assert stats["hit_rate"] == "50.0%"
    assert stats["eviction_count"] == 0


def test_cache_info(cache_manager):
    """Test get_info returns manager info."""
    info = cache_manager.get_info()
    assert info["type"] == "CacheManager"
    assert info["max_size"] == 5
    assert info["default_ttl"] == 2
    assert info["persistence_enabled"] is False