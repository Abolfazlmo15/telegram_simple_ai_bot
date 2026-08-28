"""Tests for ProxyManager."""
import pytest
import time
from pathlib import Path
from core.managers.proxy_manager import ProxyManager


@pytest.fixture
def proxy_manager(tmp_path):
    """Create a ProxyManager with temporary storage file."""
    storage_file = tmp_path / "proxies.json"
    manager = ProxyManager(str(storage_file))
    # Set primary to a test URL (overriding Config)
    manager.primary = "https://test.primary.com"
    # Also update current_proxy to match
    manager.current_proxy = manager.primary
    manager._default_fallback = "https://api.telegram.org"
    return manager

def test_proxy_manager_initialization(proxy_manager):
    """Test basic initialization."""
    # After fixture setup, current_proxy should be the same as primary
    # The fixture now sets both primary and current_proxy to the test URL
    assert proxy_manager.primary == "https://test.primary.com"
    assert proxy_manager.backups == []
    assert proxy_manager.current_proxy == proxy_manager.primary


def test_proxy_manager_add_backup(proxy_manager):
    """Test adding backup proxies."""
    proxy_manager.add_backup("https://backup1.com")
    proxy_manager.add_backup("https://backup2.com")
    proxy_manager.add_backup("https://backup1.com")  # Duplicate

    assert len(proxy_manager.backups) == 2
    assert proxy_manager.backups[0]["url"] == "https://backup1.com"
    assert proxy_manager.backups[1]["url"] == "https://backup2.com"


def test_proxy_manager_get_proxy_primary_ok(proxy_manager):
    """Test get_proxy returns primary when it's not failing."""
    proxy = proxy_manager.get_proxy()
    assert proxy == proxy_manager.primary
    # Verify reset of failure state
    assert proxy_manager.primary_fail_start is None




def test_proxy_manager_backup_rotation(proxy_manager):
    """Test rotation through backups and fallback to default."""
    proxy_manager.add_backup("https://backup1.com")
    proxy_manager.add_backup("https://backup2.com")

    # Force primary failure (set timeout expired)
    proxy_manager.primary_fail_start = time.time() - 400
    # Also mark primary as bad
    proxy_manager._proxy_failure_counts[proxy_manager.primary] = 3

    # Should return backup1
    proxy = proxy_manager.get_proxy()
    assert proxy == "https://backup1.com"

    # Mark backup1 as bad too, then it should skip to backup2
    proxy_manager.mark_failure("https://backup1.com")
    # But need to set the failure count threshold
    proxy_manager._proxy_failure_counts["https://backup1.com"] = 3
    proxy = proxy_manager.get_proxy()
    assert proxy == "https://backup2.com"

    # Mark backup2 as bad, should fallback to default
    proxy_manager.mark_failure("https://backup2.com")
    proxy_manager._proxy_failure_counts["https://backup2.com"] = 3
    proxy = proxy_manager.get_proxy()
    assert proxy == "https://api.telegram.org"


def test_proxy_manager_mark_failure(proxy_manager):
    """Test marking failure increments count for primary proxy."""
    proxy_manager._proxy_failure_counts.clear()

    # First call: primary increments by 2 (once in mark_proxy_failure, once in mark_primary_failure)
    proxy_manager.mark_proxy_failure(proxy_manager.primary)
    assert proxy_manager._proxy_failure_counts[proxy_manager.primary] == 2

    # Second call: increments by 2 again, total 4
    proxy_manager.mark_proxy_failure(proxy_manager.primary)
    assert proxy_manager._proxy_failure_counts[proxy_manager.primary] == 4

def test_proxy_manager_mark_success(proxy_manager):
    """Test mark_success resets failure counts."""
    # Set failure count manually
    proxy_manager._proxy_failure_counts[proxy_manager.primary] = 2

    proxy_manager.mark_success(proxy_manager.primary)
    assert proxy_manager._proxy_failure_counts.get(proxy_manager.primary, 0) == 0
    assert proxy_manager.primary_fail_start is None



def test_proxy_manager_is_proxy_bad(proxy_manager):
    """Test is_proxy_bad checks failure count and cooldown."""
    # Set failure count and recent fail time to make it bad
    proxy_manager._proxy_failure_counts["test"] = 2
    proxy_manager._proxy_failure_counts["test_last_fail"] = time.time()
    assert proxy_manager.is_proxy_bad("test") is True

    # Simulate cooldown expired
    proxy_manager._proxy_failure_counts["test_last_fail"] = time.time() - 100
    assert proxy_manager.is_proxy_bad("test") is False
    assert proxy_manager._proxy_failure_counts.get("test", 0) == 0


def test_proxy_manager_get_all_proxies(proxy_manager):
    """Test get_all_proxies returns all known proxies."""
    proxy_manager.add_backup("https://backup1.com")
    all_proxies = proxy_manager.get_all_proxies()
    expected = ["https://test.primary.com", "https://backup1.com", "https://api.telegram.org"]
    assert all_proxies == expected


def test_proxy_manager_clear_cache(proxy_manager):
    """Test clear_cache resets in-memory state."""
    proxy_manager.mark_failure(proxy_manager.primary)
    proxy_manager.primary_fail_start = time.time()
    proxy_manager.current_proxy = "https://other.com"

    proxy_manager.clear_cache()
    assert proxy_manager.primary_fail_start is None
    assert proxy_manager.current_proxy == proxy_manager.primary
    assert len(proxy_manager._proxy_failure_counts) == 0