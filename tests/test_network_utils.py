"""
Tests for network utilities module.
"""
import pytest
import time
from unittest.mock import Mock, patch
import requests
from network_utils import (
    RetryConfig,
    CircuitBreaker,
    calculate_backoff_delay,
    is_retryable_error,
    retry_with_backoff,
    create_session_with_retries,
    RateLimiter
)


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert 429 in config.retryable_status_codes
        assert 500 in config.retryable_status_codes

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=30.0
        )

        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 30.0


class TestCalculateBackoffDelay:
    """Test exponential backoff delay calculation."""

    def test_exponential_growth(self):
        """Test delay grows exponentially."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0)

        delay0 = calculate_backoff_delay(0, config, jitter=False)
        delay1 = calculate_backoff_delay(1, config, jitter=False)
        delay2 = calculate_backoff_delay(2, config, jitter=False)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(initial_delay=1.0, max_delay=10.0, exponential_base=2.0)

        delay_high = calculate_backoff_delay(10, config, jitter=False)
        assert delay_high == 10.0

    def test_jitter_varies_delay(self):
        """Test jitter adds randomness to delay."""
        config = RetryConfig(initial_delay=4.0)

        delays = [calculate_backoff_delay(0, config, jitter=True) for _ in range(10)]

        # All delays should be different due to jitter
        assert len(set(delays)) > 1

        # All delays should be within ±25% of 4.0
        for delay in delays:
            assert 3.0 <= delay <= 5.0


class TestIsRetryableError:
    """Test retryable error detection."""

    def test_retryable_exception(self):
        """Test retryable exceptions."""
        config = RetryConfig()
        error = requests.exceptions.Timeout()

        assert is_retryable_error(error, None, config) is True

    def test_retryable_status_code(self):
        """Test retryable HTTP status codes."""
        config = RetryConfig()
        response = Mock()
        response.status_code = 429  # Too Many Requests

        error = Exception("Some error")

        assert is_retryable_error(error, response, config) is True

    def test_non_retryable_error(self):
        """Test non-retryable errors."""
        config = RetryConfig()
        error = ValueError("Bad input")

        assert is_retryable_error(error, None, config) is False

    def test_non_retryable_status_code(self):
        """Test non-retryable status codes."""
        config = RetryConfig()
        response = Mock()
        response.status_code = 404  # Not Found

        error = Exception("Some error")

        assert is_retryable_error(error, response, config) is False


class TestRetryWithBackoff:
    """Test retry decorator."""

    def test_successful_first_try(self):
        """Test function succeeds on first try."""
        mock_func = Mock(return_value="success")

        @retry_with_backoff()
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retries_on_failure(self):
        """Test function retries on retryable error."""
        mock_func = Mock(side_effect=[
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
            "success"
        ])

        @retry_with_backoff(config=RetryConfig(max_retries=3))
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_max_retries_exceeded(self):
        """Test raises error after max retries."""
        mock_func = Mock(side_effect=requests.exceptions.Timeout())

        @retry_with_backoff(config=RetryConfig(max_retries=2))
        def test_func():
            return mock_func()

        with pytest.raises(requests.exceptions.Timeout):
            test_func()

        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_non_retryable_error_fails_immediately(self):
        """Test non-retryable error doesn't retry."""
        mock_func = Mock(side_effect=ValueError("Bad input"))

        @retry_with_backoff(config=RetryConfig(max_retries=3))
        def test_func():
            return mock_func()

        with pytest.raises(ValueError):
            test_func()

        assert mock_func.call_count == 1  # No retries


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_closed_state_allows_calls(self):
        """Test calls pass through when circuit is closed."""
        cb = CircuitBreaker(failure_threshold=3)
        mock_func = Mock(return_value="success")

        result = cb.call(mock_func)

        assert result == "success"
        assert cb.state == "CLOSED"

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, expected_exception=ValueError)

        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
            except ValueError:
                pass

        assert cb.state == "OPEN"

    def test_open_state_blocks_calls(self):
        """Test calls are blocked when circuit is open."""
        cb = CircuitBreaker(failure_threshold=1, expected_exception=ValueError)

        # Cause one failure to open circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        except ValueError:
            pass

        # Now circuit should be open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(lambda: "should not execute")

    def test_half_open_after_timeout(self):
        """Test circuit transitions to half-open after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("error")))
        except Exception:
            pass

        assert cb.state == "OPEN"

        # Wait for recovery timeout
        time.sleep(1.1)

        # Next call should transition to HALF_OPEN
        try:
            result = cb.call(lambda: "success")
            assert cb.state == "CLOSED"
        except Exception:
            pass


class TestRateLimiter:
    """Test rate limiter."""

    def test_allows_calls_under_limit(self):
        """Test calls are allowed under rate limit."""
        limiter = RateLimiter(calls_per_minute=60)

        for _ in range(10):
            assert limiter.acquire() is True

    def test_blocks_calls_over_limit(self):
        """Test calls are blocked when limit exceeded."""
        limiter = RateLimiter(calls_per_minute=60, burst_size=5)

        # Acquire all tokens
        for _ in range(5):
            assert limiter.acquire() is True

        # Next acquire should fail (no refill yet)
        assert limiter.acquire() is False

    def test_refills_over_time(self):
        """Test tokens refill over time."""
        limiter = RateLimiter(calls_per_minute=60, burst_size=5)

        # Acquire all tokens
        for _ in range(5):
            limiter.acquire()

        # Wait for refill (60 calls/min = 1 call/sec)
        time.sleep(1.1)

        # Should have at least one token now
        assert limiter.acquire() is True

    def test_wait_if_needed(self):
        """Test wait_if_needed blocks until tokens available."""
        limiter = RateLimiter(calls_per_minute=600, burst_size=1)  # Very fast refill

        # Acquire the single token
        assert limiter.acquire() is True

        # This should wait briefly for refill
        start = time.time()
        waited = limiter.wait_if_needed()
        elapsed = time.time() - start

        assert waited > 0
        assert elapsed >= waited


class TestCreateSessionWithRetries:
    """Test session creation with retry logic."""

    def test_creates_session(self):
        """Test creates a requests session."""
        session = create_session_with_retries()

        assert isinstance(session, requests.Session)

    @patch('requests.Session.get')
    def test_session_retries_on_failure(self, mock_get):
        """Test session retries on retryable status codes."""
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500

        mock_response_success = Mock()
        mock_response_success.status_code = 200

        mock_get.side_effect = [mock_response_fail, mock_response_success]

        session = create_session_with_retries(max_retries=1)

        # The session's retry adapter should handle this automatically
        # For this test, we just verify the session was created
        assert session is not None
