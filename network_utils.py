"""
Network utilities for robust API communication.
Includes retry logic, exponential backoff, and circuit breaker pattern.
"""
import time
import requests
from typing import Optional, Callable, Any, Dict
from functools import wraps
from logger import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_status_codes: tuple = (408, 429, 500, 502, 503, 504),
        retryable_exceptions: tuple = (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        )
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff calculation
            retryable_status_codes: HTTP status codes that should trigger retry
            retryable_exceptions: Exception types that should trigger retry
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_status_codes = retryable_status_codes
        self.retryable_exceptions = retryable_exceptions


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests fail immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function call

        Raises:
            Exception: If circuit is OPEN or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self.state = "HALF_OPEN"
            else:
                raise Exception(
                    f"Circuit breaker is OPEN. "
                    f"Service unavailable until {self._get_recovery_time()}"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout

    def _get_recovery_time(self) -> str:
        """Get estimated recovery time."""
        if self.last_failure_time is None:
            return "unknown"
        recovery_time = self.last_failure_time + timedelta(seconds=self.recovery_timeout)
        return recovery_time.strftime("%H:%M:%S")

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            logger.info("Circuit breaker transitioning to CLOSED")
            self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            if self.state != "OPEN":
                logger.warning(
                    f"Circuit breaker transitioning to OPEN after "
                    f"{self.failure_count} failures"
                )
                self.state = "OPEN"


def calculate_backoff_delay(
    attempt: int,
    config: RetryConfig,
    jitter: bool = True
) -> float:
    """
    Calculate delay for exponential backoff.

    Args:
        attempt: Current retry attempt number (0-indexed)
        config: Retry configuration
        jitter: Whether to add random jitter to prevent thundering herd

    Returns:
        Delay in seconds
    """
    import random

    # Calculate exponential delay
    delay = min(
        config.initial_delay * (config.exponential_base ** attempt),
        config.max_delay
    )

    # Add jitter (±25% of delay) to prevent thundering herd problem
    if jitter:
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)  # Ensure non-negative


def is_retryable_error(
    error: Exception,
    response: Optional[requests.Response],
    config: RetryConfig
) -> bool:
    """
    Determine if an error is retryable.

    Args:
        error: Exception that occurred
        response: HTTP response if available
        config: Retry configuration

    Returns:
        True if error is retryable
    """
    # Check if exception type is retryable
    if isinstance(error, config.retryable_exceptions):
        return True

    # Check if HTTP status code is retryable
    if response is not None and response.status_code in config.retryable_status_codes:
        return True

    return False


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        config: Retry configuration (uses defaults if None)
        circuit_breaker: Optional circuit breaker instance

    Returns:
        Decorated function

    Example:
        @retry_with_backoff(config=RetryConfig(max_retries=3))
        def fetch_data():
            return requests.get("https://api.example.com/data")
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            response = None

            for attempt in range(config.max_retries + 1):
                try:
                    # Call through circuit breaker if provided
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Get response if it's an HTTP error
                    if isinstance(e, requests.exceptions.HTTPError):
                        response = getattr(e, 'response', None)

                    # Check if we should retry
                    if attempt < config.max_retries:
                        if is_retryable_error(e, response, config):
                            delay = calculate_backoff_delay(attempt, config)

                            logger.warning(
                                f"Attempt {attempt + 1}/{config.max_retries + 1} failed "
                                f"for {func.__name__}: {str(e)}. "
                                f"Retrying in {delay:.2f}s..."
                            )

                            time.sleep(delay)
                            continue
                        else:
                            # Non-retryable error, fail immediately
                            logger.error(
                                f"Non-retryable error in {func.__name__}: {str(e)}"
                            )
                            raise
                    else:
                        # Max retries exceeded
                        logger.error(
                            f"Max retries ({config.max_retries}) exceeded "
                            f"for {func.__name__}: {str(e)}"
                        )
                        raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def create_session_with_retries(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple = (408, 429, 500, 502, 503, 504)
) -> requests.Session:
    """
    Create a requests Session with built-in retry logic.

    Args:
        max_retries: Maximum number of retries
        backoff_factor: Factor for exponential backoff
        status_forcelist: HTTP status codes to retry on

    Returns:
        Configured requests.Session

    Example:
        session = create_session_with_retries()
        response = session.get("https://api.example.com/data")
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    logger.debug(
        f"Created session with retry strategy: "
        f"max_retries={max_retries}, backoff_factor={backoff_factor}"
    )

    return session


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Allows bursts up to bucket capacity while maintaining average rate.
    """

    def __init__(self, calls_per_minute: int = 60, burst_size: Optional[int] = None):
        """
        Initialize rate limiter.

        Args:
            calls_per_minute: Maximum calls allowed per minute
            burst_size: Maximum burst size (defaults to calls_per_minute)
        """
        self.calls_per_minute = calls_per_minute
        self.burst_size = burst_size or calls_per_minute
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
        self.rate = calls_per_minute / 60.0  # tokens per second

    def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens for API call.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False otherwise
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_if_needed(self, tokens: int = 1) -> float:
        """
        Wait if necessary to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        waited = 0.0

        while not self.acquire(tokens):
            sleep_time = 0.1  # Check every 100ms
            time.sleep(sleep_time)
            waited += sleep_time

        if waited > 0:
            logger.debug(f"Rate limited: waited {waited:.2f}s")

        return waited

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        self.tokens = min(
            self.burst_size,
            self.tokens + (elapsed * self.rate)
        )

        self.last_update = now
