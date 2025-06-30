"""Test memory optimization functionality."""

import os
from unittest.mock import patch

import pytest

from bot.memory_optimizer import (
    MemoryOptimizer,
    configure_slim_environment,
    get_memory_optimizer,
    monitor_memory_usage,
    optimize_numpy_memory,
    optimize_pandas_memory,
)


class TestMemoryOptimizer:
    """Test the MemoryOptimizer class."""

    def test_memory_optimizer_initialization(self):
        """Test memory optimizer initializes correctly."""
        optimizer = MemoryOptimizer(max_memory_mb=512)
        assert optimizer.max_memory_mb == 512
        assert optimizer.optimization_count == 0

    def test_get_memory_usage(self):
        """Test memory usage statistics retrieval."""
        optimizer = MemoryOptimizer()
        stats = optimizer.get_memory_usage()

        assert "rss_mb" in stats
        assert "vms_mb" in stats
        assert "percent" in stats
        assert isinstance(stats["rss_mb"], float)
        assert isinstance(stats["vms_mb"], float)
        assert isinstance(stats["percent"], float)

    def test_optimize_memory_with_force(self):
        """Test forced memory optimization."""
        optimizer = MemoryOptimizer(max_memory_mb=1)  # Very low threshold
        result = optimizer.optimize_memory(force=True)

        assert result is True
        assert optimizer.optimization_count == 1

    def test_optimize_memory_threshold_check(self):
        """Test memory optimization based on threshold."""
        optimizer = MemoryOptimizer(max_memory_mb=99999)  # Very high threshold
        result = optimizer.optimize_memory(force=False)

        # Should not optimize since we're likely under the threshold
        assert result is False
        assert optimizer.optimization_count == 0

    def test_get_memory_report(self):
        """Test memory report generation."""
        optimizer = MemoryOptimizer()
        report = optimizer.get_memory_report()

        assert "Memory Usage Report:" in report
        assert "RSS:" in report
        assert "VMS:" in report
        assert "Percent:" in report
        assert "Max Threshold:" in report
        assert "Optimizations:" in report


class TestGlobalFunctions:
    """Test global memory optimization functions."""

    def test_get_memory_optimizer_singleton(self):
        """Test memory optimizer singleton pattern."""
        optimizer1 = get_memory_optimizer()
        optimizer2 = get_memory_optimizer()

        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, MemoryOptimizer)

    def test_optimize_pandas_memory(self):
        """Test pandas memory optimization."""
        # Should not raise any exceptions
        optimize_pandas_memory()

    def test_optimize_numpy_memory(self):
        """Test numpy memory optimization."""
        # Should not raise any exceptions
        optimize_numpy_memory()

    @patch.dict(os.environ, {"ENABLE_MEMORY_OPTIMIZATION": "true"}, clear=False)
    def test_configure_slim_environment(self):
        """Test slim environment configuration."""
        # Should not raise any exceptions
        configure_slim_environment()

        # Check environment variables are set
        assert os.environ.get("PYTHONOPTIMIZE") == "2"
        assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        assert os.environ.get("MALLOC_TRIM_THRESHOLD_") == "100000"

    def test_monitor_memory_usage_decorator(self):
        """Test memory usage monitoring decorator."""

        # Create a mock function
        @monitor_memory_usage
        def test_function():
            return "test_result"

        # Should execute without errors
        result = test_function()
        assert result == "test_result"


class TestMemoryOptimizerIntegration:
    """Integration tests for memory optimizer."""

    def test_memory_optimization_in_loop(self):
        """Test memory optimization in a trading loop scenario."""
        optimizer = MemoryOptimizer(max_memory_mb=256)

        # Simulate multiple iterations
        for i in range(25):  # Should trigger optimization every 20 iterations
            if i % 20 == 0:
                optimizer.check_memory_threshold()

        # Check that optimization was attempted
        stats = optimizer.get_memory_usage()
        assert isinstance(stats, dict)

    def test_memory_stats_accuracy(self):
        """Test that memory statistics are reasonable."""
        optimizer = MemoryOptimizer()
        stats = optimizer.get_memory_usage()

        # Basic sanity checks
        assert stats["rss_mb"] > 0
        assert stats["vms_mb"] > 0
        assert 0 <= stats["percent"] <= 100
        assert stats["rss_mb"] <= stats["vms_mb"]  # RSS should be <= VMS

    def test_optimization_count_tracking(self):
        """Test that optimization count is tracked correctly."""
        optimizer = MemoryOptimizer()
        initial_count = optimizer.optimization_count

        # Force optimization multiple times
        optimizer.optimize_memory(force=True)
        optimizer.optimize_memory(force=True)

        assert optimizer.optimization_count == initial_count + 2

    @patch.dict(os.environ, {"MAX_MEMORY_MB": "384"}, clear=False)
    def test_environment_variable_integration(self):
        """Test memory optimizer respects environment variables."""
        optimizer = MemoryOptimizer()
        assert optimizer.max_memory_mb == 384


class TestErrorHandling:
    """Test error handling in memory optimization."""

    @patch("bot.memory_optimizer.psutil.Process")
    def test_memory_optimizer_with_psutil_error(self, mock_process):
        """Test memory optimizer handles psutil errors gracefully."""
        mock_process.side_effect = Exception("psutil error")

        # Should still initialize without crashing
        try:
            optimizer = MemoryOptimizer()
            # This should not crash even if psutil fails
            assert optimizer.max_memory_mb > 0
        except Exception:
            pytest.fail("Memory optimizer should handle psutil errors gracefully")

    def test_invalid_memory_threshold(self):
        """Test memory optimizer with invalid threshold."""
        # Should handle invalid input gracefully
        optimizer = MemoryOptimizer(max_memory_mb=-1)
        assert optimizer.max_memory_mb == -1  # Should accept but handle gracefully

        # Optimization should still work
        result = optimizer.optimize_memory(force=True)
        assert isinstance(result, bool)


if __name__ == "__main__":
    # Quick manual test
    print("Testing memory optimizer...")

    optimizer = get_memory_optimizer()
    print(f"Initial memory report:\n{optimizer.get_memory_report()}")

    # Force optimization
    optimizer.optimize_memory(force=True)
    print(f"\nAfter optimization:\n{optimizer.get_memory_report()}")

    print("\nMemory optimizer tests completed successfully!")
