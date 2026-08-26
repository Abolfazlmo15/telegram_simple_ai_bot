"""
Unified network retry utilities with exponential backoff and jitter.
Provides both async and sync retry wrappers for HTTP calls.
Supports proxy configuration.
"""
import asyncio
import random
import logging
from typing import Callable, Any, Optional, Type, Tuple, Union
from functools import wraps

from core.config import Config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Exceptions that should trigger a retry (network-related)
# ------------------------------------------------------------------
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    BrokenPipeError,
    ConnectionResetError,
)

try:
    import httpx
    RETRYABLE_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.RemoteProtocolError,
        httpx.TransportError,
    )
except ImportError:
    pass


def is_retryable_exception(exc: Exception) -> bool:
    """Check if an exception is retryable (network-related)."""
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True

    if hasattr(exc, 'response') and hasattr(exc.response, 'status_code'):
        if exc.response.status_code >= 500 or exc.response.status_code == 429:
            return True

    return False


def calculate_backoff(attempt: int, base_delay: float = 0.5, max_delay: float = 10.0, jitter: float = 0.1) -> float:
    """
    Calculate exponential backoff with jitter.

    Args:
        attempt: Current attempt number (0-indexed).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap.
        jitter: Random jitter factor (0.0 to 1.0).

    Returns:
        Delay in seconds.
    """
    delay = min(max_delay, base_delay * (2 ** attempt))
    if jitter > 0:
        delay = delay * (1 + random.uniform(-jitter, jitter))
    return max(0, delay)


def create_httpx_client(proxy: Optional[str] = None, timeout: Optional[float] = None,
                        http2: bool = True, limits: Optional[httpx.Limits] = None) -> httpx.Client:
    """
    Create a synchronous httpx Client with optional proxy.
    """
    client_kwargs = {}
    if proxy:
        client_kwargs["proxy"] = proxy
    if timeout:
        client_kwargs["timeout"] = timeout
    if limits:
        client_kwargs["limits"] = limits
    try:
        return httpx.Client(**client_kwargs, http2=http2)
    except Exception:
        logger.warning("HTTP/2 failed, falling back to HTTP/1.1 for sync client")
        return httpx.Client(**client_kwargs, http2=False)


def create_async_httpx_client(proxy: Optional[str] = None, timeout: Optional[float] = None,
                              http2: bool = True, limits: Optional[httpx.Limits] = None) -> httpx.AsyncClient:
    """
    Create an asynchronous httpx AsyncClient with optional proxy.
    """
    client_kwargs = {}
    if proxy:
        client_kwargs["proxy"] = proxy
    if timeout:
        client_kwargs["timeout"] = timeout
    if limits:
        client_kwargs["limits"] = limits
    try:
        return httpx.AsyncClient(**client_kwargs, http2=http2)
    except Exception:
        logger.warning("HTTP/2 failed, falling back to HTTP/1.1 for async client")
        return httpx.AsyncClient(**client_kwargs, http2=False)


async def retry_async(
    func: Callable,
    *args,
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    jitter: Optional[float] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    **kwargs
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to call.
        *args: Positional arguments for func.
        max_attempts: Maximum retry attempts (including first call).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap.
        jitter: Random jitter factor.
        on_retry: Optional callback called on each retry (exception, attempt).
        **kwargs: Keyword arguments for func.

    Returns:
        The result of the function call.

    Raises:
        The last exception encountered if all attempts fail.
    """
    max_attempts = max_attempts or Config.NETWORK_RETRY_MAX_ATTEMPTS
    base_delay = base_delay or Config.NETWORK_RETRY_BASE_DELAY
    max_delay = max_delay or Config.NETWORK_RETRY_MAX_DELAY
    jitter = jitter or Config.NETWORK_RETRY_JITTER

    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not is_retryable_exception(e):
                logger.debug(f"Non-retryable exception: {e}")
                raise

            if attempt == max_attempts - 1:
                break

            delay = calculate_backoff(attempt, base_delay, max_delay, jitter)
            if on_retry:
                try:
                    on_retry(e, attempt + 1)
                except Exception:
                    pass
            logger.warning(f"Retryable error (attempt {attempt+1}/{max_attempts}): {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)

    raise last_exception


def retry_sync(
    func: Callable,
    *args,
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    jitter: Optional[float] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    **kwargs
) -> Any:
    """
    Retry a synchronous function with exponential backoff.

    Args:
        func: Synchronous function to call.
        *args: Positional arguments for func.
        max_attempts: Maximum retry attempts (including first call).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap.
        jitter: Random jitter factor.
        on_retry: Optional callback called on each retry (exception, attempt).
        **kwargs: Keyword arguments for func.

    Returns:
        The result of the function call.

    Raises:
        The last exception encountered if all attempts fail.
    """
    import time

    max_attempts = max_attempts or Config.NETWORK_RETRY_MAX_ATTEMPTS
    base_delay = base_delay or Config.NETWORK_RETRY_BASE_DELAY
    max_delay = max_delay or Config.NETWORK_RETRY_MAX_DELAY
    jitter = jitter or Config.NETWORK_RETRY_JITTER

    last_exception = None

    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not is_retryable_exception(e):
                raise

            if attempt == max_attempts - 1:
                break

            delay = calculate_backoff(attempt, base_delay, max_delay, jitter)
            if on_retry:
                try:
                    on_retry(e, attempt + 1)
                except Exception:
                    pass
            logger.warning(f"Retryable error (attempt {attempt+1}/{max_attempts}): {e}. Retrying in {delay:.2f}s")
            time.sleep(delay)

    raise last_exception


# ------------------------------------------------------------------
# Convenience decorators
# ------------------------------------------------------------------
def retryable_async(max_attempts: Optional[int] = None, base_delay: Optional[float] = None):
    """
    Decorator for async functions to add retry logic.

    Usage:
        @retryable_async(max_attempts=3)
        async def fetch_data():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                func,
                *args,
                max_attempts=max_attempts,
                base_delay=base_delay,
                **kwargs
            )
        return wrapper
    return decorator


def retryable_sync(max_attempts: Optional[int] = None, base_delay: Optional[float] = None):
    """
    Decorator for synchronous functions to add retry logic.

    Usage:
        @retryable_sync(max_attempts=3)
        def fetch_data():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_sync(
                func,
                *args,
                max_attempts=max_attempts,
                base_delay=base_delay,
                **kwargs
            )
        return wrapper
    return decorator