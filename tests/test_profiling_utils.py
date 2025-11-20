"""
Tests for profiling and performance monitoring utilities.
"""
import pytest
import time
from profiling_utils import (
    PerformanceMetric,
    PerformanceMonitor,
    get_performance_monitor,
    timed,
    memory_profiled,
    Timer,
    measure_throughput,
    compare_implementations,
    get_performance_report
)


class TestPerformanceMetric:
    """Test performance metric dataclass."""

    def test_creation(self):
        """Test metric creation."""
        metric = PerformanceMetric(
            function_name="test_func",
            duration=1.5,
            memory_used=10.5,
            success=True
        )

        assert metric.function_name == "test_func"
        assert metric.duration == 1.5
        assert metric.memory_used == 10.5
        assert metric.success is True
        assert metric.error is None

    def test_with_error(self):
        """Test metric with error."""
        metric = PerformanceMetric(
            function_name="test_func",
            duration=0.5,
            success=False,
            error="Test error"
        )

        assert metric.success is False
        assert metric.error == "Test error"


class TestPerformanceMonitor:
    """Test performance monitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        assert len(monitor.get_recent_metrics()) == 0
        assert monitor.get_stats() == {}

    def test_record_metric(self):
        """Test recording a metric."""
        monitor = PerformanceMonitor()

        metric = PerformanceMetric(
            function_name="test_func",
            duration=1.0,
            success=True
        )

        monitor.record(metric)

        stats = monitor.get_stats("test_func")
        assert stats['count'] == 1
        assert stats['total_time'] == 1.0
        assert stats['avg_time'] == 1.0
        assert stats['min_time'] == 1.0
        assert stats['max_time'] == 1.0

    def test_record_multiple_metrics(self):
        """Test recording multiple metrics."""
        monitor = PerformanceMonitor()

        for duration in [1.0, 2.0, 3.0]:
            metric = PerformanceMetric(
                function_name="test_func",
                duration=duration,
                success=True
            )
            monitor.record(metric)

        stats = monitor.get_stats("test_func")
        assert stats['count'] == 3
        assert stats['total_time'] == 6.0
        assert stats['avg_time'] == 2.0
        assert stats['min_time'] == 1.0
        assert stats['max_time'] == 3.0

    def test_record_errors(self):
        """Test recording errors."""
        monitor = PerformanceMonitor()

        # Successful call
        monitor.record(PerformanceMetric(
            function_name="test_func",
            duration=1.0,
            success=True
        ))

        # Failed call
        monitor.record(PerformanceMetric(
            function_name="test_func",
            duration=0.5,
            success=False,
            error="Test error"
        ))

        stats = monitor.get_stats("test_func")
        assert stats['count'] == 2
        assert stats['errors'] == 1

    def test_get_recent_metrics(self):
        """Test getting recent metrics."""
        monitor = PerformanceMonitor()

        for i in range(10):
            monitor.record(PerformanceMetric(
                function_name=f"func_{i}",
                duration=1.0,
                success=True
            ))

        recent = monitor.get_recent_metrics(limit=5)
        assert len(recent) == 5

    def test_get_slow_functions(self):
        """Test getting slow functions."""
        monitor = PerformanceMonitor()

        # Fast function
        monitor.record(PerformanceMetric(
            function_name="fast_func",
            duration=0.5,
            success=True
        ))

        # Slow function
        for _ in range(3):
            monitor.record(PerformanceMetric(
                function_name="slow_func",
                duration=2.0,
                success=True
            ))

        slow_funcs = monitor.get_slow_functions(threshold=1.0)
        assert len(slow_funcs) == 1
        assert slow_funcs[0][0] == "slow_func"
        assert slow_funcs[0][1] == 2.0

    def test_clear(self):
        """Test clearing metrics."""
        monitor = PerformanceMonitor()

        monitor.record(PerformanceMetric(
            function_name="test_func",
            duration=1.0,
            success=True
        ))

        monitor.clear()

        assert len(monitor.get_recent_metrics()) == 0
        assert monitor.get_stats() == {}

    def test_summary(self):
        """Test summary generation."""
        monitor = PerformanceMonitor()

        monitor.record(PerformanceMetric(
            function_name="test_func",
            duration=1.0,
            success=True
        ))

        summary = monitor.summary()
        assert "Performance Summary" in summary
        assert "test_func" in summary
        assert "Calls: 1" in summary


class TestTimedDecorator:
    """Test timed decorator."""

    def test_basic_timing(self):
        """Test basic function timing."""
        monitor = PerformanceMonitor()

        @timed
        def test_function():
            time.sleep(0.1)
            return "done"

        result = test_function()

        assert result == "done"
        # Note: Global monitor is used, so we can't easily test stats here

    def test_timed_with_monitor_disabled(self):
        """Test timed decorator with monitoring disabled."""
        @timed(monitor=False)
        def test_function():
            time.sleep(0.1)
            return "done"

        result = test_function()
        assert result == "done"

    def test_timed_with_exception(self):
        """Test timed decorator with exception."""
        @timed
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()


class TestMemoryProfiledDecorator:
    """Test memory_profiled decorator."""

    def test_basic_memory_profiling(self):
        """Test basic memory profiling."""
        @memory_profiled
        def memory_function():
            data = [i for i in range(10000)]
            return len(data)

        result = memory_function()
        assert result == 10000


class TestTimerContextManager:
    """Test Timer context manager."""

    def test_basic_timing(self):
        """Test basic timing."""
        with Timer("test_block") as timer:
            time.sleep(0.1)

        assert timer.duration >= 0.1
        assert timer.duration < 0.2

    def test_timer_with_exception(self):
        """Test timer with exception."""
        try:
            with Timer("test_block") as timer:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert timer.duration is not None

    def test_timer_no_log(self):
        """Test timer without logging."""
        with Timer("test_block", log=False) as timer:
            time.sleep(0.05)

        assert timer.duration >= 0.05


class TestMeasureThroughput:
    """Test throughput measurement."""

    def test_basic_throughput(self):
        """Test basic throughput measurement."""
        def simple_function():
            return sum(range(100))

        metrics = measure_throughput(simple_function, iterations=10, warmup=2)

        assert 'avg_time' in metrics
        assert 'min_time' in metrics
        assert 'max_time' in metrics
        assert 'ops_per_sec' in metrics
        assert 'total_time' in metrics

        assert metrics['min_time'] <= metrics['avg_time'] <= metrics['max_time']
        assert metrics['ops_per_sec'] > 0

    def test_throughput_with_slow_function(self):
        """Test throughput with slower function."""
        def slow_function():
            time.sleep(0.01)
            return "done"

        metrics = measure_throughput(slow_function, iterations=5, warmup=1)

        assert metrics['avg_time'] >= 0.01
        assert metrics['ops_per_sec'] <= 100  # Max 100 ops/sec


class TestCompareImplementations:
    """Test implementation comparison."""

    def test_compare_two_implementations(self):
        """Test comparing two implementations."""
        def implementation_a():
            return sum(range(100))

        def implementation_b():
            return sum(i for i in range(100))

        results = compare_implementations(
            implementation_a,
            implementation_b,
            iterations=10
        )

        assert 'implementation_a' in results
        assert 'implementation_b' in results
        assert 'avg_time' in results['implementation_a']
        assert 'ops_per_sec' in results['implementation_a']

    def test_compare_single_implementation(self):
        """Test comparing single implementation."""
        def single_impl():
            return 42

        results = compare_implementations(single_impl, iterations=10)

        assert 'single_impl' in results
        assert results['single_impl']['ops_per_sec'] > 0


class TestGetPerformanceReport:
    """Test performance report generation."""

    def test_get_report(self):
        """Test getting performance report."""
        # Clear existing metrics
        monitor = get_performance_monitor()
        monitor.clear()

        # Add some metrics
        @timed
        def test_func():
            time.sleep(0.01)

        test_func()

        report = get_performance_report()

        assert "PERFORMANCE REPORT" in report
        assert "Performance Summary" in report
        assert "Slow Functions" in report


class TestGlobalMonitor:
    """Test global performance monitor."""

    def test_get_global_monitor(self):
        """Test getting global monitor."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        # Should be the same instance
        assert monitor1 is monitor2
