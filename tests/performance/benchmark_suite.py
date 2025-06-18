"""
Performance benchmark suite for the AI Trading Bot.

This module provides comprehensive performance benchmarks for all critical components
including indicator calculations, LLM response times, market data processing,
memory usage profiling, and database operations.
"""

import asyncio
import logging
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import psutil

from bot.config import create_settings

# Import bot components
from bot.indicators.vumanchu import CipherA, CipherB, VuManChuIndicators
from bot.strategy.llm_agent import LLMAgent
from bot.trading_types import IndicatorData, MarketState, Position

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    description: str
    execution_time_ms: float
    memory_usage_mb: float | None = None
    cpu_percent: float | None = None
    iterations: int = 1
    throughput_per_sec: float | None = None
    peak_memory_mb: float | None = None
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_time_per_iteration_ms(self) -> float:
        """Average time per iteration in milliseconds."""
        return self.execution_time_ms / max(self.iterations, 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "execution_time_ms": self.execution_time_ms,
            "avg_time_per_iteration_ms": self.avg_time_per_iteration_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "cpu_percent": self.cpu_percent,
            "iterations": self.iterations,
            "throughput_per_sec": self.throughput_per_sec,
            "additional_metrics": self.additional_metrics,
        }


@dataclass
class BenchmarkSuite:
    """Main benchmark suite configuration."""

    name: str
    description: str
    results: list[BenchmarkResult] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def total_duration_seconds(self) -> float:
        """Total duration of the benchmark suite."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def get_summary(self) -> dict[str, Any]:
        """Get benchmark suite summary."""
        if not self.results:
            return {"message": "No benchmark results available"}

        return {
            "suite_name": self.name,
            "description": self.description,
            "total_benchmarks": len(self.results),
            "total_duration_seconds": self.total_duration_seconds,
            "results": [result.to_dict() for result in self.results],
        }


class PerformanceBenchmarks:
    """
    Comprehensive performance benchmark suite for the trading bot.

    Measures performance across all critical components with detailed
    memory, CPU, and timing metrics.
    """

    def __init__(self):
        """Initialize the benchmark suite."""
        self.settings = create_settings()
        self.process = psutil.Process()

        # Generate test data
        self.test_data = self._generate_test_data()
        self.large_test_data = self._generate_test_data(size=5000)

        # Initialize components for testing
        self.indicator_calc = VuManChuIndicators()
        self.cipher_a = CipherA()
        self.cipher_b = CipherB()

    def _generate_test_data(self, size: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing.

        Args:
            size: Number of candles to generate

        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(42)  # For reproducible results

        # Generate realistic price movements
        start_price = 50000.0
        price_changes = np.random.normal(0, 0.01, size)  # 1% std dev
        prices = [start_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # Minimum price floor

        # Create OHLCV data
        data = []
        for i, close in enumerate(prices):
            # Generate realistic OHLC from close price
            volatility = np.random.uniform(
                0.005, 0.02
            )  # 0.5% to 2% intraday volatility
            high = close * (1 + volatility * np.random.uniform(0, 1))
            low = close * (1 - volatility * np.random.uniform(0, 1))

            # Ensure OHLC relationships are valid
            if i == 0:
                open_price = close
            else:
                open_price = prices[i - 1] * (1 + np.random.normal(0, 0.002))

            high = max(high, close, open_price)
            low = min(low, close, open_price)

            volume = np.random.uniform(100000, 1000000)

            data.append(
                {
                    "timestamp": datetime.utcnow() - timedelta(minutes=size - i),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def run_all_benchmarks(self) -> BenchmarkSuite:
        """
        Run all performance benchmarks.

        Returns:
            BenchmarkSuite with all results
        """
        suite = BenchmarkSuite(
            name="AI Trading Bot Performance Benchmarks",
            description="Comprehensive performance testing of all trading bot components",
        )

        suite.start_time = datetime.utcnow()
        logger.info("Starting comprehensive performance benchmark suite...")

        try:
            # Indicator calculation benchmarks
            suite.add_result(self.benchmark_cipher_a_calculation())
            suite.add_result(self.benchmark_cipher_b_calculation())
            suite.add_result(self.benchmark_full_indicator_calculation())
            suite.add_result(self.benchmark_indicator_scaling())

            # LLM benchmarks
            suite.add_result(self.benchmark_llm_initialization())
            suite.add_result(self.benchmark_llm_prompt_processing())

            # Market data processing benchmarks
            suite.add_result(self.benchmark_dataframe_operations())
            suite.add_result(self.benchmark_data_validation())

            # Memory usage benchmarks
            suite.add_result(self.benchmark_memory_usage())
            suite.add_result(self.benchmark_memory_scaling())

            # Database operation benchmarks (simulated)
            suite.add_result(self.benchmark_data_serialization())
            suite.add_result(self.benchmark_data_deserialization())

            logger.info(f"Completed {len(suite.results)} benchmarks successfully")

        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}")
            raise
        finally:
            suite.end_time = datetime.utcnow()

        return suite

    def benchmark_cipher_a_calculation(self) -> BenchmarkResult:
        """Benchmark Cipher A indicator calculation performance."""
        return self._run_benchmark(
            name="Cipher A Calculation",
            description="Performance of Cipher A indicator calculation with EMA, RSI, and signals",
            func=lambda: self.cipher_a.calculate(self.test_data),
            iterations=100,
        )

    def benchmark_cipher_b_calculation(self) -> BenchmarkResult:
        """Benchmark Cipher B indicator calculation performance."""
        return self._run_benchmark(
            name="Cipher B Calculation",
            description="Performance of Cipher B indicator calculation with VWAP, MFI, and waves",
            func=lambda: self.cipher_b.calculate(self.test_data),
            iterations=100,
        )

    def benchmark_full_indicator_calculation(self) -> BenchmarkResult:
        """Benchmark full indicator calculation performance."""
        return self._run_benchmark(
            name="Full Indicator Suite",
            description="Performance of complete indicator calculation including all indicators",
            func=lambda: self.indicator_calc.calculate_all(self.test_data),
            iterations=50,
        )

    def benchmark_indicator_scaling(self) -> BenchmarkResult:
        """Benchmark indicator calculation with larger datasets."""

        def scaling_test():
            results = {}
            for size in [500, 1000, 2000, 5000]:
                data = self._generate_test_data(size)
                start_time = time.perf_counter()
                self.indicator_calc.calculate_all(data)
                end_time = time.perf_counter()
                results[f"size_{size}"] = (end_time - start_time) * 1000
            return results

        result = self._run_benchmark(
            name="Indicator Scaling",
            description="Indicator calculation performance across different data sizes",
            func=scaling_test,
            iterations=10,
        )

        return result

    def benchmark_llm_initialization(self) -> BenchmarkResult:
        """Benchmark LLM agent initialization time."""

        def init_llm():
            return LLMAgent(model_provider="openai", model_name="gpt-3.5-turbo")

        return self._run_benchmark(
            name="LLM Initialization",
            description="Time to initialize LLM agent and load prompt templates",
            func=init_llm,
            iterations=5,
        )

    def benchmark_llm_prompt_processing(self) -> BenchmarkResult:
        """Benchmark LLM prompt processing and response time."""
        llm_agent = LLMAgent(model_provider="openai", model_name="gpt-3.5-turbo")

        # Create test market state
        test_position = Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0"),
            timestamp=datetime.utcnow(),
        )

        test_indicators = IndicatorData(
            cipher_a_dot=1.0,
            cipher_b_wave=0.5,
            cipher_b_money_flow=55.0,
            rsi=45.0,
            ema_fast=50000.0,
            ema_slow=49900.0,
        )

        test_market_state = MarketState(
            symbol="BTC-USD",
            interval="1m",
            timestamp=datetime.utcnow(),
            current_price=Decimal("50000"),
            ohlcv_data=self.test_data.tail(10).to_dict("records"),
            indicators=test_indicators,
            current_position=test_position,
        )

        async def llm_analysis():
            return await llm_agent.analyze_market(test_market_state)

        def sync_llm_analysis():
            return asyncio.run(llm_analysis())

        return self._run_benchmark(
            name="LLM Analysis",
            description="Time for LLM to analyze market state and generate trading decision",
            func=sync_llm_analysis,
            iterations=10,
        )

    def benchmark_dataframe_operations(self) -> BenchmarkResult:
        """Benchmark common DataFrame operations."""

        def dataframe_ops():
            df = self.test_data.copy()

            # Common operations
            df["sma_20"] = df["close"].rolling(20).mean()
            df["volatility"] = df["close"].pct_change().rolling(20).std()
            df["high_low_pct"] = (df["high"] - df["low"]) / df["close"]

            # Resampling
            resampled = df.resample("5T").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )

            return len(resampled)

        return self._run_benchmark(
            name="DataFrame Operations",
            description="Performance of common DataFrame operations and resampling",
            func=dataframe_ops,
            iterations=100,
        )

    def benchmark_data_validation(self) -> BenchmarkResult:
        """Benchmark data validation performance."""

        def validate_data():
            df = self.test_data.copy()

            # Validation checks
            checks = [
                df["high"] >= df["low"],
                df["high"] >= df["close"],
                df["high"] >= df["open"],
                df["low"] <= df["close"],
                df["low"] <= df["open"],
                df["volume"] > 0,
                df["close"] > 0,
            ]

            return all(check.all() for check in checks)

        return self._run_benchmark(
            name="Data Validation",
            description="Performance of OHLCV data validation checks",
            func=validate_data,
            iterations=200,
        )

    def benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage during indicator calculations."""
        tracemalloc.start()

        def memory_test():
            # Calculate indicators multiple times
            for _ in range(10):
                result = self.indicator_calc.calculate_all(self.test_data)
                # Force garbage collection simulation
                del result

        start_time = time.perf_counter()
        memory_test()
        end_time = time.perf_counter()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return BenchmarkResult(
            name="Memory Usage",
            description="Memory consumption during indicator calculations",
            execution_time_ms=(end_time - start_time) * 1000,
            memory_usage_mb=current / 1024 / 1024,
            peak_memory_mb=peak / 1024 / 1024,
            iterations=10,
            additional_metrics={
                "current_memory_bytes": current,
                "peak_memory_bytes": peak,
            },
        )

    def benchmark_memory_scaling(self) -> BenchmarkResult:
        """Benchmark memory usage scaling with data size."""

        def memory_scaling_test():
            results = {}

            for size in [500, 1000, 2000, 5000]:
                data = self._generate_test_data(size)

                tracemalloc.start()
                self.indicator_calc.calculate_all(data)
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                results[f"size_{size}_current_mb"] = current / 1024 / 1024
                results[f"size_{size}_peak_mb"] = peak / 1024 / 1024

            return results

        result = self._run_benchmark(
            name="Memory Scaling",
            description="Memory usage scaling across different data sizes",
            func=memory_scaling_test,
            iterations=3,
        )

        return result

    def benchmark_data_serialization(self) -> BenchmarkResult:
        """Benchmark data serialization performance."""

        def serialize_data():
            # Simulate database serialization
            df_dict = self.test_data.to_dict("records")
            return len(df_dict)

        return self._run_benchmark(
            name="Data Serialization",
            description="Performance of converting DataFrame to serializable format",
            func=serialize_data,
            iterations=100,
        )

    def benchmark_data_deserialization(self) -> BenchmarkResult:
        """Benchmark data deserialization performance."""
        serialized_data = self.test_data.to_dict("records")

        def deserialize_data():
            df = pd.DataFrame(serialized_data)
            df.set_index("timestamp", inplace=True)
            return len(df)

        return self._run_benchmark(
            name="Data Deserialization",
            description="Performance of converting serialized data back to DataFrame",
            func=deserialize_data,
            iterations=100,
        )

    def _run_benchmark(
        self, name: str, description: str, func: Callable, iterations: int = 1
    ) -> BenchmarkResult:
        """
        Run a single benchmark function.

        Args:
            name: Benchmark name
            description: Benchmark description
            func: Function to benchmark
            iterations: Number of iterations to run

        Returns:
            BenchmarkResult with metrics
        """
        logger.info(f"Running benchmark: {name}")

        # Warm up
        try:
            func()
        except Exception:
            pass  # Ignore warm-up errors

        # Collect baseline metrics
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        execution_times = []
        cpu_percents = []

        for i in range(iterations):
            # Start monitoring
            process.cpu_percent()

            # Run the function
            start_time = time.perf_counter()
            try:
                result = func()
            except Exception as e:
                logger.warning(f"Benchmark iteration {i+1} failed: {e}")
                continue
            end_time = time.perf_counter()

            # Record metrics
            execution_time_ms = (end_time - start_time) * 1000
            execution_times.append(execution_time_ms)

            # CPU usage (may not be accurate for short operations)
            cpu_percent = process.cpu_percent()
            if cpu_percent > 0:
                cpu_percents.append(cpu_percent)

        # Calculate final metrics
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = final_memory - initial_memory

        statistics.mean(execution_times) if execution_times else 0
        total_execution_time = sum(execution_times)
        avg_cpu = statistics.mean(cpu_percents) if cpu_percents else None

        # Calculate throughput
        throughput = None
        if total_execution_time > 0:
            throughput = (
                iterations / total_execution_time
            ) * 1000  # operations per second

        additional_metrics = {
            "min_time_ms": min(execution_times) if execution_times else 0,
            "max_time_ms": max(execution_times) if execution_times else 0,
            "std_dev_ms": (
                statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            ),
            "successful_iterations": len(execution_times),
        }

        # Add function-specific metrics if returned
        if hasattr(result, "keys") and isinstance(result, dict):
            additional_metrics.update(result)

        return BenchmarkResult(
            name=name,
            description=description,
            execution_time_ms=total_execution_time,
            memory_usage_mb=memory_usage if memory_usage > 0 else None,
            cpu_percent=avg_cpu,
            iterations=len(execution_times),
            throughput_per_sec=throughput,
            additional_metrics=additional_metrics,
        )


def run_benchmark_suite() -> BenchmarkSuite:
    """
    Run the complete benchmark suite.

    Returns:
        BenchmarkSuite with all results
    """
    benchmarks = PerformanceBenchmarks()
    return benchmarks.run_all_benchmarks()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run benchmarks
    suite = run_benchmark_suite()

    # Print summary
    summary = suite.get_summary()
    print(f"\n{'='*60}")
    print("BENCHMARK SUITE RESULTS")
    print(f"{'='*60}")
    print(f"Suite: {summary['suite_name']}")
    print(f"Description: {summary['description']}")
    print(f"Total Benchmarks: {summary['total_benchmarks']}")
    print(f"Total Duration: {summary['total_duration_seconds']:.2f} seconds")
    print(f"{'='*60}")

    for result in summary["results"]:
        print(f"\n{result['name']}")
        print(f"  Description: {result['description']}")
        print(f"  Total Time: {result['execution_time_ms']:.2f} ms")
        print(f"  Avg Per Iteration: {result['avg_time_per_iteration_ms']:.2f} ms")
        print(f"  Iterations: {result['iterations']}")

        if result.get("throughput_per_sec"):
            print(f"  Throughput: {result['throughput_per_sec']:.2f} ops/sec")

        if result.get("memory_usage_mb"):
            print(f"  Memory Usage: {result['memory_usage_mb']:.2f} MB")

        if result.get("peak_memory_mb"):
            print(f"  Peak Memory: {result['peak_memory_mb']:.2f} MB")

        if result.get("cpu_percent"):
            print(f"  CPU Usage: {result['cpu_percent']:.1f}%")

        # Print additional metrics
        if result.get("additional_metrics"):
            print("  Additional Metrics:")
            for key, value in result["additional_metrics"].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.2f}")
                else:
                    print(f"    {key}: {value}")

    print(f"\n{'='*60}")
