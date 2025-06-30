"""Memory optimization utilities for the trading bot."""

import gc
import logging
import os
import warnings

import psutil

# Suppress unnecessary warnings to reduce memory overhead
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Optimize memory usage for the trading bot."""

    def __init__(self, max_memory_mb: int | None = None):
        """Initialize memory optimizer.

        Args:
            max_memory_mb: Maximum memory usage in MB before optimization kicks in
        """
        self.max_memory_mb = max_memory_mb or int(os.getenv("MAX_MEMORY_MB", "512"))
        self.process = psutil.Process()
        self.optimization_count = 0

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
        }

    def optimize_memory(self, force: bool = False) -> bool:
        """Optimize memory usage if needed.

        Args:
            force: Force optimization regardless of current usage

        Returns:
            True if optimization was performed
        """
        memory_stats = self.get_memory_usage()
        current_memory = memory_stats["rss_mb"]

        if force or current_memory > self.max_memory_mb:
            logger.info(
                f"Memory optimization triggered: {current_memory:.1f}MB > {self.max_memory_mb}MB"
            )

            # Force garbage collection
            collected = gc.collect()

            # Get updated memory stats
            new_memory_stats = self.get_memory_usage()
            new_memory = new_memory_stats["rss_mb"]

            self.optimization_count += 1

            logger.info(
                f"Memory optimization #{self.optimization_count}: "
                f"{current_memory:.1f}MB -> {new_memory:.1f}MB "
                f"(freed {current_memory - new_memory:.1f}MB, collected {collected} objects)"
            )

            return True

        return False

    def check_memory_threshold(self) -> None:
        """Check if memory usage exceeds threshold and optimize if needed."""
        self.optimize_memory(force=False)

    def get_memory_report(self) -> str:
        """Get a detailed memory usage report."""
        stats = self.get_memory_usage()
        return (
            f"Memory Usage Report:\n"
            f"  RSS: {stats['rss_mb']:.1f} MB\n"
            f"  VMS: {stats['vms_mb']:.1f} MB\n"
            f"  Percent: {stats['percent']:.1f}%\n"
            f"  Max Threshold: {self.max_memory_mb} MB\n"
            f"  Optimizations: {self.optimization_count}"
        )


# Global memory optimizer instance
_memory_optimizer: MemoryOptimizer | None = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get the global memory optimizer instance."""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def optimize_pandas_memory() -> None:
    """Optimize pandas memory usage."""
    try:
        import pandas as pd

        # Reduce memory usage for pandas
        pd.set_option("mode.copy_on_write", True)
        pd.set_option("compute.use_bottleneck", True)
        pd.set_option("compute.use_numexpr", True)
        logger.debug("Pandas memory optimizations applied")
    except ImportError:
        pass


def optimize_numpy_memory() -> None:
    """Optimize numpy memory usage."""
    try:
        import numpy as np

        # Configure numpy for memory efficiency
        np.seterr(all="ignore")  # Suppress warnings to reduce overhead
        logger.debug("NumPy memory optimizations applied")
    except ImportError:
        pass


def configure_slim_environment() -> None:
    """Configure the environment for minimal memory usage."""
    # Set environment variables for memory optimization
    os.environ["PYTHONOPTIMIZE"] = "2"  # Enable Python optimizations
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # Don't write .pyc files
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"  # Aggressive memory trimming

    # Apply optimizations
    optimize_pandas_memory()
    optimize_numpy_memory()

    # Force initial garbage collection
    gc.collect()

    logger.info("Slim environment configuration applied")


def monitor_memory_usage(func):
    """Decorator to monitor memory usage of functions."""

    def wrapper(*args, **kwargs):
        optimizer = get_memory_optimizer()
        before_stats = optimizer.get_memory_usage()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            after_stats = optimizer.get_memory_usage()
            memory_diff = after_stats["rss_mb"] - before_stats["rss_mb"]

            if memory_diff > 10:  # Log if memory increased by more than 10MB
                logger.warning(
                    f"Function {func.__name__} increased memory by {memory_diff:.1f}MB"
                )

            # Auto-optimize if needed
            optimizer.check_memory_threshold()

    return wrapper


# Auto-configure when module is imported
if os.getenv("ENABLE_MEMORY_OPTIMIZATION", "true").lower() == "true":
    configure_slim_environment()
