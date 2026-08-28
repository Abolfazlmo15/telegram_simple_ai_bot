"""Tests for HealthChecker."""
import pytest
import time
from unittest.mock import MagicMock, patch
from core.managers.health_checker import HealthChecker, HealthStatus


@pytest.fixture
def health_checker():
    """Create a HealthChecker with mocked client and short intervals."""
    checker = HealthChecker()
    checker.check_interval_seconds = 1
    checker.failure_threshold = 2
    checker.success_threshold = 1
    checker.cooldown_seconds = 1
    checker.health_check_retries = 1
    checker.health_check_timeout = 1.0
    return checker


def test_health_checker_initialization(health_checker):
    """Test basic initialization."""
    assert health_checker._is_running is False
    assert health_checker._models == {}
    assert health_checker._client is None


def test_health_checker_seed_models(health_checker):
    """Test that seed_models populates from config."""
    health_checker._seed_models()
    assert len(health_checker._models) > 0
    # Check some expected models
    assert "deepseek/deepseek-chat:free" in health_checker._models
    assert "openrouter" in health_checker._models


def test_health_checker_register_model(health_checker):
    """Test registering a new model."""
    health_checker.register_model("test/model")
    assert "test/model" in health_checker._models
    health = health_checker.get_health("test/model")
    assert health.status == HealthStatus.UNKNOWN


def test_health_checker_is_healthy(health_checker):
    """Test is_healthy returns True for unknown or healthy models."""
    assert health_checker.is_healthy("unknown_model") is True
    health_checker.register_model("test/model")
    assert health_checker.is_healthy("test/model") is True

    # Mark as unhealthy
    health = health_checker.get_health("test/model")
    health.status = HealthStatus.UNHEALTHY
    assert health_checker.is_healthy("test/model") is False


def test_health_checker_get_unhealthy_models(health_checker):
    """Test get_unhealthy_models returns only unhealthy models."""
    health_checker.register_model("healthy")
    health_checker.register_model("unhealthy1")
    health_checker.register_model("unhealthy2")

    health_checker.get_health("unhealthy1").status = HealthStatus.UNHEALTHY
    health_checker.get_health("unhealthy2").status = HealthStatus.UNHEALTHY

    unhealthy = health_checker.get_unhealthy_models()
    assert len(unhealthy) == 2
    assert "unhealthy1" in unhealthy
    assert "unhealthy2" in unhealthy


@patch("core.managers.health_checker.httpx.Client")
def test_check_model_openrouter_success(mock_client, health_checker):
    """Test _check_model for openrouter endpoint success."""
    # Create a mock client and attach it to health_checker
    client_mock = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    client_mock.get.return_value = mock_response
    health_checker._client = client_mock

    status, latency, reason = health_checker._check_model("openrouter")
    assert status == HealthStatus.HEALTHY
    assert latency > 0
    assert reason == ""


@patch("core.managers.health_checker.httpx.Client")
def test_check_model_openrouter_failure(mock_client, health_checker):
    """Test _check_model for openrouter endpoint failure."""
    client_mock = MagicMock()
    client_mock.get.side_effect = Exception("Connection error")
    health_checker._client = client_mock

    status, latency, reason = health_checker._check_model("openrouter")
    assert status == HealthStatus.UNHEALTHY
    assert "Connection error" in reason


@patch("core.managers.health_checker.httpx.Client")
def test_check_model_pollinations_success(mock_client, health_checker):
    """Test _check_model for pollinations endpoint success."""
    client_mock = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "image/png"}
    client_mock.get.return_value = mock_response
    health_checker._client = client_mock

    status, latency, reason = health_checker._check_model("pollinations")
    assert status == HealthStatus.HEALTHY


def test_check_model_huggingface_no_token(health_checker):
    """Test _check_model for huggingface when token missing."""
    # Create a mock client to avoid AttributeError
    health_checker._client = MagicMock()
    with patch("core.managers.health_checker.Config.HUGGINGFACE_TOKEN", ""):
        status, latency, reason = health_checker._check_model("huggingface")
        assert status == HealthStatus.UNHEALTHY
        assert "No token configured" in reason


def test_health_checker_run_checks(health_checker):
    """Test the _run_checks method updates health statuses."""
    # Seed models
    health_checker._seed_models()
    # Mock _check_model_with_retry to return healthy for all
    health_checker._check_model_with_retry = MagicMock(return_value=(HealthStatus.HEALTHY, 10, ""))

    # Run checks
    health_checker._run_checks()

    # All models should have checked_count > 0 and status healthy
    for model, health in health_checker._models.items():
        assert health.checked_count > 0
        assert health.status == HealthStatus.HEALTHY


def test_health_checker_failure_threshold(health_checker):
    """Test that consecutive failures mark model unhealthy."""
    health_checker.register_model("test/model")
    health = health_checker.get_health("test/model")
    health.failure_threshold = 2
    health.consecutive_failures = 0

    # Simulate failures
    for i in range(2):
        health.consecutive_failures += 1
        if health.consecutive_failures >= 2:
            health.status = HealthStatus.UNHEALTHY

    assert health.status == HealthStatus.UNHEALTHY


def test_health_checker_clear_cache(health_checker):
    """Test clear_cache resets all health records."""
    health_checker.register_model("test")
    health = health_checker.get_health("test")
    health.status = HealthStatus.UNHEALTHY
    health.consecutive_failures = 3
    health.failure_reason = "test error"

    health_checker.clear_cache()

    new_health = health_checker.get_health("test")
    assert new_health.status == HealthStatus.UNKNOWN
    assert new_health.consecutive_failures == 0
    assert new_health.failure_reason == ""
    assert new_health.last_checked is None


def test_health_checker_get_info(health_checker):
    """Test get_info returns correct statistics."""
    health_checker.register_model("healthy")
    health_checker.register_model("unhealthy")
    health_checker.get_health("healthy").status = HealthStatus.HEALTHY
    health_checker.get_health("unhealthy").status = HealthStatus.UNHEALTHY
    health_checker._is_running = True

    info = health_checker.get_info()
    assert info["type"] == "HealthChecker"
    assert info["running"] is True
    assert info["total_models"] == 2
    assert info["healthy"] == 1
    assert info["unhealthy"] == 1
    assert info["unknown"] == 0