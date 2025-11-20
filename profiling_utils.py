"""
Profiling and performance monitoring utilities.
Includes decorators for timing, memory profiling, and performance metrics collection.
"""
import time
import functools
import tracemalloc
from typing import Callable, Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Container for performance metrics."""
    function_name: str
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)
    memory_used: Optional[float] = None  # MB
    success: bool = True
    error: Optional[str] = None


class PerformanceMonitor:
    """
    Collects and aggregates performance metrics.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self._metrics: List[PerformanceMetric] = []
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'errors': 0
        })

    def record(self, metric: PerformanceMetric) -> None:
        """
        Record a performance metric.

        Args:
            metric: Performance metric to record
        """
        self._metrics.append(metric)

        # Update statistics
        stats = self._stats[metric.function_name]
        stats['count'] += 1
        stats['total_time'] += metric.duration
        stats['min_time'] = min(stats['min_time'], metric.duration)
        stats['max_time'] = max(stats['max_time'], metric.duration)
        stats['avg_time'] = stats['total_time'] / stats['count']

        if not metric.success:
            stats['errors'] += 1

        logger.debug(
            f"Performance: {metric.function_name} took {metric.duration:.4f}s "
            f"(mem: {metric.memory_used:.2f}MB)" if metric.memory_used else ""
        )

    def get_stats(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics.

        Args:
            function_name: Optional function name to filter by

        Returns:
            Dictionary with performance statistics
        """
        if function_name:
            return dict(self._stats.get(function_name, {}))
        return {k: dict(v) for k, v in self._stats.items()}

    def get_recent_metrics(self, limit: int = 100) -> List[PerformanceMetric]:
        """
        Get recent performance metrics.

        Args:
            limit: Maximum number of metrics to return

        Returns:
            List of recent metrics
        """
        return self._metrics[-limit:]

    def get_slow_functions(self, threshold: float = 1.0) -> List[tuple]:
        """
        Get functions that exceed the time threshold.

        Args:
            threshold: Time threshold in seconds

        Returns:
            List of (function_name, avg_time) tuples
        """
        slow_functions = [
            (name, stats['avg_time'])
            for name, stats in self._stats.items()
            if stats['avg_time'] > threshold
        ]
        return sorted(slow_functions, key=lambda x: x[1], reverse=True)

    def clear(self) -> None:
        """Clear all metrics and statistics."""
        self._metrics.clear()
        self._stats.clear()
        logger.info("Performance metrics cleared")

    def summary(self) -> str:
        """
        Get a summary of performance statistics.

        Returns:
            Formatted string with performance summary
        """
        if not self._stats:
            return "No performance data collected"

        lines = ["Performance Summary:", "=" * 60]

        for func_name, stats in sorted(self._stats.items()):
            lines.append(f"\n{func_name}:")
            lines.append(f"  Calls: {stats['count']}")
            lines.append(f"  Avg time: {stats['avg_time']:.4f}s")
            lines.append(f"  Min time: {stats['min_time']:.4f}s")
            lines.append(f"  Max time: {stats['max_time']:.4f}s")
            lines.append(f"  Total time: {stats['total_time']:.4f}s")
            if stats['errors'] > 0:
                lines.append(f"  Errors: {stats['errors']}")

        return "\n".join(lines)


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def timed(func: Optional[Callable] = None, *, monitor: bool = True) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to decorate
        monitor: Whether to record metrics in performance monitor

    Returns:
        Decorated function

    Example:
        @timed
        def slow_function():
            time.sleep(1)
            return "done"

        @timed(monitor=False)
        def fast_function():
            return "quick"
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            error_msg = None
            success = True

            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration = time.time() - start_time

                # Log timing
                logger.debug(f"{f.__name__} took {duration:.4f}s")

                # Record in monitor if enabled
                if monitor:
                    metric = PerformanceMetric(
                        function_name=f.__name__,
                        duration=duration,
                        success=success,
                        error=error_msg
                    )
                    _performance_monitor.record(metric)

        return wrapper

    # Handle both @timed and @timed() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


def memory_profiled(func: Callable) -> Callable:
    """
    Decorator to measure function memory usage.

    Args:
        func: Function to decorate

    Returns:
        Decorated function

    Example:
        @memory_profiled
        def memory_intensive_function():
            data = [i for i in range(1000000)]
            return len(data)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        tracemalloc.start()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)

            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            duration = time.time() - start_time

            # Convert to MB
            current_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024

            logger.info(
                f"{func.__name__} - "
                f"Time: {duration:.4f}s, "
                f"Current memory: {current_mb:.2f}MB, "
                f"Peak memory: {peak_mb:.2f}MB"
            )

            # Record in monitor
            metric = PerformanceMetric(
                function_name=f"{func.__name__} (mem)",
                duration=duration,
                memory_used=peak_mb,
                success=True
            )
            _performance_monitor.record(metric)

            return result

        finally:
            tracemalloc.stop()

    return wrapper


class Timer:
    """
    Context manager for timing code blocks.

    Example:
        with Timer("database_query"):
            results = database.search_books()
    """

    def __init__(self, name: str, log: bool = True):
        """
        Initialize timer.

        Args:
            name: Name of the code block
            log: Whether to log the timing
        """
        self.name = name
        self.log = log
        self.start_time = None
        self.duration = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log duration."""
        self.duration = time.time() - self.start_time

        if self.log:
            logger.debug(f"{self.name} took {self.duration:.4f}s")

        # Record in monitor
        metric = PerformanceMetric(
            function_name=self.name,
            duration=self.duration,
            success=exc_type is None,
            error=str(exc_val) if exc_val else None
        )
        _performance_monitor.record(metric)


def profile_queries(enabled: bool = True):
    """
    Context manager for profiling database queries.

    Args:
        enabled: Whether profiling is enabled

    Example:
        with profile_queries():
            db.search_books("test")
            db.get_all_books()
    """
    class QueryProfiler:
        def __init__(self, enabled: bool):
            self.enabled = enabled
            self.queries: List[tuple] = []

        def __enter__(self):
            if self.enabled:
                logger.info("Query profiling started")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.enabled and self.queries:
                logger.info(f"Executed {len(self.queries)} queries")
                for query, duration in self.queries:
                    logger.debug(f"  {query[:100]}... ({duration:.4f}s)")

        def add_query(self, query: str, duration: float):
            """Add a query to the profile."""
            if self.enabled:
                self.queries.append((query, duration))

    return QueryProfiler(enabled)


def measure_throughput(func: Callable, iterations: int = 100, warmup: int = 10) -> Dict[str, float]:
    """
    Measure function throughput.

    Args:
        func: Function to measure
        iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Dictionary with throughput metrics

    Example:
        metrics = measure_throughput(lambda: expensive_function(), iterations=100)
        print(f"Ops/sec: {metrics['ops_per_sec']}")
    """
    logger.info(f"Measuring throughput for {func.__name__} ({iterations} iterations)")

    # Warmup
    for _ in range(warmup):
        func()

    # Actual measurement
    times = []
    for _ in range(iterations):
        start = time.time()
        func()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'ops_per_sec': 1.0 / avg_time if avg_time > 0 else 0,
        'total_time': sum(times)
    }


def compare_implementations(*funcs: Callable, iterations: int = 100) -> Dict[str, Dict]:
    """
    Compare performance of multiple implementations.

    Args:
        *funcs: Functions to compare
        iterations: Number of iterations per function

    Returns:
        Dictionary mapping function names to their metrics

    Example:
        results = compare_implementations(
            implementation_a,
            implementation_b,
            iterations=1000
        )
    """
    results = {}

    for func in funcs:
        metrics = measure_throughput(func, iterations=iterations)
        results[func.__name__] = metrics

        logger.info(
            f"{func.__name__}: {metrics['ops_per_sec']:.2f} ops/sec "
            f"(avg: {metrics['avg_time']:.4f}s)"
        )

    return results


def get_performance_report() -> str:
    """
    Get a comprehensive performance report.

    Returns:
        Formatted performance report
    """
    monitor = get_performance_monitor()
    lines = [
        "\n" + "=" * 70,
        "PERFORMANCE REPORT",
        "=" * 70,
        "",
        monitor.summary(),
        "",
        "Slow Functions (>1s average):",
        "-" * 70
    ]

    slow_funcs = monitor.get_slow_functions(threshold=1.0)
    if slow_funcs:
        for func_name, avg_time in slow_funcs:
            lines.append(f"  {func_name}: {avg_time:.4f}s")
    else:
        lines.append("  None")

    lines.append("=" * 70)

    return "\n".join(lines)
