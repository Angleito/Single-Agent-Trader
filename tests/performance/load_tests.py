"""
Load testing suite for the AI Trading Bot.

This module provides comprehensive load testing including simulated high-frequency
market data loads, concurrent trading decision stress tests, memory leak detection,
API rate limit testing, and error recovery under load.
"""

import asyncio
import logging
import random
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import psutil

from bot.config import create_settings
from bot.indicators.vumanchu import VuManChuIndicators

# Import bot components
from bot.risk import RiskManager
from bot.strategy.llm_agent import LLMAgent
from bot.types import IndicatorData, MarketData, MarketState, Position
from bot.validator import TradeValidator

logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Container for load test results."""

    test_name: str
    description: str
    duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    operations_per_second: float
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    error_rate_percent: float
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "description": self.description,
            "duration_seconds": self.duration_seconds,
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "operations_per_second": self.operations_per_second,
            "avg_response_time_ms": self.avg_response_time_ms,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "error_rate_percent": self.error_rate_percent,
            "additional_metrics": self.additional_metrics,
        }


@dataclass
class LoadTestConfig:
    """Configuration for load tests."""

    duration_seconds: int = 60
    concurrent_users: int = 10
    operations_per_second: int = 10
    ramp_up_seconds: int = 10
    ramp_down_seconds: int = 10
    memory_check_interval: int = 5
    error_threshold_percent: float = 5.0


class MarketDataSimulator:
    """Simulates high-frequency market data for load testing."""

    def __init__(self, symbol: str = "BTC-USD"):
        """Initialize the market data simulator."""
        self.symbol = symbol
        self.base_price = 50000.0
        self.current_price = self.base_price
        self.data_buffer = []
        self.running = False

    def generate_tick_data(self, count: int = 1000) -> list[MarketData]:
        """
        Generate synthetic tick data for testing.

        Args:
            count: Number of ticks to generate

        Returns:
            List of OHLCV data points
        """
        ticks = []
        current_time = datetime.utcnow()

        for i in range(count):
            # Simulate price movement
            price_change = random.normalvariate(0, 0.001)  # 0.1% volatility
            self.current_price *= 1 + price_change
            self.current_price = max(self.current_price, 1000)  # Price floor

            # Generate OHLC for the tick
            volatility = random.uniform(0.0001, 0.001)
            high = self.current_price * (1 + volatility)
            low = self.current_price * (1 - volatility)
            open_price = self.current_price
            close_price = self.current_price
            volume = random.uniform(1000, 10000)

            tick_data = MarketData(
                symbol=self.symbol,
                timestamp=current_time - timedelta(seconds=count - i),
                open=Decimal(str(open_price)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close_price)),
                volume=Decimal(str(volume)),
            )

            ticks.append(tick_data)

        return ticks

    async def start_streaming(self, frequency_hz: int = 10):
        """
        Start streaming market data at specified frequency.

        Args:
            frequency_hz: Data update frequency in Hz
        """
        self.running = True
        interval = 1.0 / frequency_hz

        while self.running:
            tick = self.generate_tick_data(1)[0]
            self.data_buffer.append(tick)

            # Keep buffer size manageable
            if len(self.data_buffer) > 1000:
                self.data_buffer = self.data_buffer[-500:]

            await asyncio.sleep(interval)

    def stop_streaming(self):
        """Stop the data stream."""
        self.running = False

    def get_latest_data(self, limit: int = 100) -> list[MarketData]:
        """Get the latest market data."""
        return self.data_buffer[-limit:] if self.data_buffer else []


class LoadTestSuite:
    """
    Comprehensive load testing suite for the trading bot.

    Tests system behavior under various load conditions including
    high-frequency data, concurrent operations, and sustained stress.
    """

    def __init__(self, config: LoadTestConfig | None = None):
        """Initialize the load test suite."""
        self.config = config or LoadTestConfig()
        self.settings = create_settings()
        self.process = psutil.Process()

        # Initialize components
        self.indicator_calc = VuManChuIndicators()
        self.llm_agent = LLMAgent()
        self.risk_manager = RiskManager(position_manager=None)
        self.validator = TradeValidator()

        # Test data
        self.market_simulator = MarketDataSimulator()
        self.test_results = []

    async def run_all_load_tests(self) -> list[LoadTestResult]:
        """
        Run all load tests.

        Returns:
            List of load test results
        """
        logger.info("Starting comprehensive load test suite...")

        tests = [
            self.test_high_frequency_data_processing,
            self.test_concurrent_indicator_calculations,
            self.test_concurrent_llm_decisions,
            self.test_memory_leak_detection,
            self.test_sustained_load,
            self.test_burst_load,
            self.test_error_recovery,
            self.test_resource_limits,
        ]

        results = []
        for test in tests:
            try:
                result = await test()
                results.append(result)
                logger.info(f"Completed test: {result.test_name}")
            except Exception as e:
                logger.error(f"Test failed: {test.__name__} - {e}")
                # Create a failure result
                failure_result = LoadTestResult(
                    test_name=test.__name__,
                    description=f"Test failed with error: {str(e)}",
                    duration_seconds=0,
                    total_operations=0,
                    successful_operations=0,
                    failed_operations=1,
                    operations_per_second=0,
                    avg_response_time_ms=0,
                    min_response_time_ms=0,
                    max_response_time_ms=0,
                    p95_response_time_ms=0,
                    p99_response_time_ms=0,
                    memory_usage_mb=0,
                    peak_memory_mb=0,
                    cpu_usage_percent=0,
                    error_rate_percent=100.0,
                )
                results.append(failure_result)

        return results

    async def test_high_frequency_data_processing(self) -> LoadTestResult:
        """Test processing of high-frequency market data."""
        logger.info("Starting high-frequency data processing test...")

        tracemalloc.start()
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        response_times = []
        successful_ops = 0
        failed_ops = 0

        # Generate high-frequency data stream
        data_points = self.market_simulator.generate_tick_data(5000)

        # Process data in batches
        batch_size = 100
        for i in range(0, len(data_points), batch_size):
            batch = data_points[i : i + batch_size]

            # Convert to DataFrame
            df_data = []
            for tick in batch:
                df_data.append(
                    {
                        "timestamp": tick.timestamp,
                        "open": float(tick.open),
                        "high": float(tick.high),
                        "low": float(tick.low),
                        "close": float(tick.close),
                        "volume": float(tick.volume),
                    }
                )

            df = pd.DataFrame(df_data)
            df.set_index("timestamp", inplace=True)

            # Time the indicator calculation
            op_start = time.perf_counter()
            try:
                self.indicator_calc.calculate_all(df)
                op_end = time.perf_counter()

                response_times.append((op_end - op_start) * 1000)
                successful_ops += 1
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}")
                failed_ops += 1

        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        duration = end_time - start_time

        return self._create_load_test_result(
            test_name="High-Frequency Data Processing",
            description="Processing high-frequency market data with indicator calculations",
            duration=duration,
            response_times=response_times,
            successful_ops=successful_ops,
            failed_ops=failed_ops,
            initial_memory=initial_memory,
            final_memory=final_memory,
            peak_memory=peak_memory / 1024 / 1024,
            additional_metrics={
                "data_points_processed": len(data_points),
                "batch_size": batch_size,
                "batches_processed": successful_ops,
            },
        )

    async def test_concurrent_indicator_calculations(self) -> LoadTestResult:
        """Test concurrent indicator calculations."""
        logger.info("Starting concurrent indicator calculations test...")

        tracemalloc.start()
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        # Generate test data
        test_data = []
        for _i in range(self.config.concurrent_users):
            data = self.market_simulator.generate_tick_data(500)
            df_data = []
            for tick in data:
                df_data.append(
                    {
                        "timestamp": tick.timestamp,
                        "open": float(tick.open),
                        "high": float(tick.high),
                        "low": float(tick.low),
                        "close": float(tick.close),
                        "volume": float(tick.volume),
                    }
                )

            df = pd.DataFrame(df_data)
            df.set_index("timestamp", inplace=True)
            test_data.append(df)

        response_times = []
        successful_ops = 0
        failed_ops = 0

        # Run concurrent calculations
        with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
            futures = []

            for df in test_data:
                future = executor.submit(self._calculate_indicators_with_timing, df)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    response_time = future.result()
                    response_times.append(response_time)
                    successful_ops += 1
                except Exception as e:
                    logger.warning(f"Concurrent calculation failed: {e}")
                    failed_ops += 1

        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        duration = end_time - start_time

        return self._create_load_test_result(
            test_name="Concurrent Indicator Calculations",
            description="Concurrent execution of indicator calculations",
            duration=duration,
            response_times=response_times,
            successful_ops=successful_ops,
            failed_ops=failed_ops,
            initial_memory=initial_memory,
            final_memory=final_memory,
            peak_memory=peak_memory / 1024 / 1024,
            additional_metrics={
                "concurrent_users": self.config.concurrent_users,
                "calculations_per_user": 1,
            },
        )

    async def test_concurrent_llm_decisions(self) -> LoadTestResult:
        """Test concurrent LLM decision making."""
        logger.info("Starting concurrent LLM decisions test...")

        tracemalloc.start()
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        # Create test market states
        test_states = []
        for _i in range(min(5, self.config.concurrent_users)):  # Limit LLM tests
            test_position = Position(
                symbol="BTC-USD",
                side="FLAT",
                size=Decimal("0"),
                timestamp=datetime.utcnow(),
            )

            test_indicators = IndicatorData(
                timestamp=datetime.utcnow(),
                cipher_a_dot=random.uniform(-2, 2),
                cipher_b_wave=random.uniform(-1, 1),
                cipher_b_money_flow=random.uniform(30, 70),
                rsi=random.uniform(20, 80),
                ema_fast=random.uniform(49000, 51000),
                ema_slow=random.uniform(48900, 50900),
            )

            market_state = MarketState(
                symbol="BTC-USD",
                interval="1m",
                timestamp=datetime.utcnow(),
                current_price=Decimal("50000"),
                ohlcv_data=[],
                indicators=test_indicators,
                current_position=test_position,
            )

            test_states.append(market_state)

        response_times = []
        successful_ops = 0
        failed_ops = 0

        # Run concurrent LLM decisions
        tasks = []
        for state in test_states:
            task = self._analyze_market_with_timing(state)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"LLM analysis failed: {result}")
                failed_ops += 1
            else:
                response_times.append(result)
                successful_ops += 1

        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        duration = end_time - start_time

        return self._create_load_test_result(
            test_name="Concurrent LLM Decisions",
            description="Concurrent LLM market analysis and decision making",
            duration=duration,
            response_times=response_times,
            successful_ops=successful_ops,
            failed_ops=failed_ops,
            initial_memory=initial_memory,
            final_memory=final_memory,
            peak_memory=peak_memory / 1024 / 1024,
            additional_metrics={
                "concurrent_llm_requests": len(test_states),
                "llm_provider": self.llm_agent.model_provider,
            },
        )

    async def test_memory_leak_detection(self) -> LoadTestResult:
        """Test for memory leaks over extended operation."""
        logger.info("Starting memory leak detection test...")

        tracemalloc.start()
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        memory_snapshots = []
        response_times = []
        successful_ops = 0
        failed_ops = 0

        # Run operations for extended period
        operation_count = 0
        test_duration = min(self.config.duration_seconds, 120)  # Limit to 2 minutes

        while (time.perf_counter() - start_time) < test_duration:
            # Generate fresh data for each iteration
            data = self.market_simulator.generate_tick_data(200)
            df_data = []
            for tick in data:
                df_data.append(
                    {
                        "timestamp": tick.timestamp,
                        "open": float(tick.open),
                        "high": float(tick.high),
                        "low": float(tick.low),
                        "close": float(tick.close),
                        "volume": float(tick.volume),
                    }
                )

            df = pd.DataFrame(df_data)
            df.set_index("timestamp", inplace=True)

            # Perform operation
            op_start = time.perf_counter()
            try:
                result = self.indicator_calc.calculate_all(df)
                # Explicitly delete to test cleanup
                del result
                del df

                op_end = time.perf_counter()
                response_times.append((op_end - op_start) * 1000)
                successful_ops += 1
            except Exception as e:
                logger.warning(f"Memory leak test operation failed: {e}")
                failed_ops += 1

            operation_count += 1

            # Take memory snapshot every 10 operations
            if operation_count % 10 == 0:
                current_mem = self.process.memory_info().rss / 1024 / 1024
                memory_snapshots.append(current_mem)

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)

        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        duration = end_time - start_time

        # Analyze memory growth
        memory_growth = final_memory - initial_memory
        memory_growth_rate = memory_growth / duration if duration > 0 else 0

        return self._create_load_test_result(
            test_name="Memory Leak Detection",
            description="Extended operation to detect memory leaks",
            duration=duration,
            response_times=response_times,
            successful_ops=successful_ops,
            failed_ops=failed_ops,
            initial_memory=initial_memory,
            final_memory=final_memory,
            peak_memory=peak_memory / 1024 / 1024,
            additional_metrics={
                "memory_growth_mb": memory_growth,
                "memory_growth_rate_mb_per_sec": memory_growth_rate,
                "memory_snapshots": memory_snapshots,
                "operations_completed": operation_count,
                "test_duration_seconds": duration,
            },
        )

    async def test_sustained_load(self) -> LoadTestResult:
        """Test system under sustained load."""
        logger.info("Starting sustained load test...")

        tracemalloc.start()
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        response_times = []
        successful_ops = 0
        failed_ops = 0

        # Run sustained operations
        target_ops_per_sec = self.config.operations_per_second
        test_duration = min(self.config.duration_seconds, 60)  # Limit to 1 minute

        async def sustained_operation():
            data = self.market_simulator.generate_tick_data(100)
            df_data = []
            for tick in data:
                df_data.append(
                    {
                        "timestamp": tick.timestamp,
                        "open": float(tick.open),
                        "high": float(tick.high),
                        "low": float(tick.low),
                        "close": float(tick.close),
                        "volume": float(tick.volume),
                    }
                )

            df = pd.DataFrame(df_data)
            df.set_index("timestamp", inplace=True)

            op_start = time.perf_counter()
            self.indicator_calc.calculate_all(df)
            op_end = time.perf_counter()

            return (op_end - op_start) * 1000

        # Schedule operations at regular intervals
        interval = 1.0 / target_ops_per_sec
        next_operation_time = start_time

        while (time.perf_counter() - start_time) < test_duration:
            current_time = time.perf_counter()

            if current_time >= next_operation_time:
                try:
                    response_time = await sustained_operation()
                    response_times.append(response_time)
                    successful_ops += 1
                except Exception as e:
                    logger.warning(f"Sustained load operation failed: {e}")
                    failed_ops += 1

                next_operation_time += interval
            else:
                # Sleep until next operation
                sleep_time = next_operation_time - current_time
                if sleep_time > 0:
                    await asyncio.sleep(min(sleep_time, 0.001))

        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        duration = end_time - start_time

        return self._create_load_test_result(
            test_name="Sustained Load",
            description="System performance under sustained load",
            duration=duration,
            response_times=response_times,
            successful_ops=successful_ops,
            failed_ops=failed_ops,
            initial_memory=initial_memory,
            final_memory=final_memory,
            peak_memory=peak_memory / 1024 / 1024,
            additional_metrics={
                "target_ops_per_sec": target_ops_per_sec,
                "actual_ops_per_sec": successful_ops / duration if duration > 0 else 0,
                "test_duration_seconds": duration,
            },
        )

    async def test_burst_load(self) -> LoadTestResult:
        """Test system under burst load conditions."""
        logger.info("Starting burst load test...")

        tracemalloc.start()
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        response_times = []
        successful_ops = 0
        failed_ops = 0

        # Generate burst operations
        burst_size = 50
        burst_interval = 5  # seconds between bursts
        num_bursts = 3

        for burst_num in range(num_bursts):
            logger.info(f"Executing burst {burst_num + 1}/{num_bursts}")

            # Create burst of operations
            tasks = []
            for _i in range(burst_size):
                task = self._create_burst_operation()
                tasks.append(task)

            # Execute burst concurrently
            burst_start = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            burst_end = time.perf_counter()

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Burst operation failed: {result}")
                    failed_ops += 1
                else:
                    response_times.append(result)
                    successful_ops += 1

            logger.info(
                f"Burst {burst_num + 1} completed in {burst_end - burst_start:.2f}s"
            )

            # Wait before next burst
            if burst_num < num_bursts - 1:
                await asyncio.sleep(burst_interval)

        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        duration = end_time - start_time

        return self._create_load_test_result(
            test_name="Burst Load",
            description="System performance under burst load conditions",
            duration=duration,
            response_times=response_times,
            successful_ops=successful_ops,
            failed_ops=failed_ops,
            initial_memory=initial_memory,
            final_memory=final_memory,
            peak_memory=peak_memory / 1024 / 1024,
            additional_metrics={
                "burst_size": burst_size,
                "num_bursts": num_bursts,
                "burst_interval": burst_interval,
                "total_operations": burst_size * num_bursts,
            },
        )

    async def test_error_recovery(self) -> LoadTestResult:
        """Test error recovery under load."""
        logger.info("Starting error recovery test...")

        tracemalloc.start()
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        response_times = []
        successful_ops = 0
        failed_ops = 0
        recovery_times = []

        # Inject errors randomly
        error_rate = 0.2  # 20% error rate

        for _i in range(100):
            op_start = time.perf_counter()

            try:
                # Randomly inject errors
                if random.random() < error_rate:
                    # Simulate various error conditions
                    error_type = random.choice(
                        ["invalid_data", "timeout", "resource_exhaustion"]
                    )

                    if error_type == "invalid_data":
                        # Test with invalid data
                        invalid_df = pd.DataFrame({"invalid": [1, 2, 3]})
                        self.indicator_calc.calculate_all(invalid_df)
                    elif error_type == "timeout":
                        # Simulate timeout
                        await asyncio.sleep(0.1)
                        raise TimeoutError("Simulated timeout")
                    else:
                        # Simulate resource exhaustion
                        raise MemoryError("Simulated memory error")
                else:
                    # Normal operation
                    data = self.market_simulator.generate_tick_data(100)
                    df_data = []
                    for tick in data:
                        df_data.append(
                            {
                                "timestamp": tick.timestamp,
                                "open": float(tick.open),
                                "high": float(tick.high),
                                "low": float(tick.low),
                                "close": float(tick.close),
                                "volume": float(tick.volume),
                            }
                        )

                    df = pd.DataFrame(df_data)
                    df.set_index("timestamp", inplace=True)
                    self.indicator_calc.calculate_all(df)

                op_end = time.perf_counter()
                response_times.append((op_end - op_start) * 1000)
                successful_ops += 1

            except Exception as e:
                # Measure recovery time
                recovery_start = time.perf_counter()

                # Simulate recovery logic
                await asyncio.sleep(0.01)  # Brief recovery delay

                recovery_end = time.perf_counter()
                recovery_times.append((recovery_end - recovery_start) * 1000)
                failed_ops += 1

                logger.debug(f"Error recovery for: {type(e).__name__}")

        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        duration = end_time - start_time

        return self._create_load_test_result(
            test_name="Error Recovery",
            description="System error recovery under load conditions",
            duration=duration,
            response_times=response_times,
            successful_ops=successful_ops,
            failed_ops=failed_ops,
            initial_memory=initial_memory,
            final_memory=final_memory,
            peak_memory=peak_memory / 1024 / 1024,
            additional_metrics={
                "injected_error_rate": error_rate,
                "recovery_times_ms": recovery_times,
                "avg_recovery_time_ms": (
                    statistics.mean(recovery_times) if recovery_times else 0
                ),
                "max_recovery_time_ms": max(recovery_times) if recovery_times else 0,
            },
        )

    async def test_resource_limits(self) -> LoadTestResult:
        """Test system behavior at resource limits."""
        logger.info("Starting resource limits test...")

        tracemalloc.start()
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        response_times = []
        successful_ops = 0
        failed_ops = 0

        # Gradually increase load until resource limits
        current_load = 1
        max_load = 20

        while current_load <= max_load:
            logger.info(f"Testing load level: {current_load}")

            # Create operations at current load level
            tasks = []
            for _i in range(current_load):
                # Create increasingly large datasets
                data_size = 500 * current_load
                task = self._create_resource_intensive_operation(data_size)
                tasks.append(task)

            # Execute operations
            load_start = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            load_end = time.perf_counter()

            # Check if system is still responsive
            load_time = (load_end - load_start) * 1000

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    failed_ops += 1
                else:
                    response_times.append(result)
                    successful_ops += 1

            # Check memory usage
            current_mem = self.process.memory_info().rss / 1024 / 1024
            if current_mem > initial_memory * 5:  # 5x memory increase
                logger.warning(f"Memory limit approached at load level {current_load}")
                break

            # Check response time degradation
            if load_time > 30000:  # 30 second threshold
                logger.warning(
                    f"Response time limit exceeded at load level {current_load}"
                )
                break

            current_load += 2

        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        duration = end_time - start_time

        return self._create_load_test_result(
            test_name="Resource Limits",
            description="System behavior at resource limits",
            duration=duration,
            response_times=response_times,
            successful_ops=successful_ops,
            failed_ops=failed_ops,
            initial_memory=initial_memory,
            final_memory=final_memory,
            peak_memory=peak_memory / 1024 / 1024,
            additional_metrics={
                "max_load_level_tested": current_load - 2,
                "memory_multiplier": (
                    final_memory / initial_memory if initial_memory > 0 else 0
                ),
                "load_progression": list(range(1, current_load, 2)),
            },
        )

    def _calculate_indicators_with_timing(self, df: pd.DataFrame) -> float:
        """Calculate indicators and return timing."""
        start_time = time.perf_counter()
        self.indicator_calc.calculate_all(df)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000

    async def _analyze_market_with_timing(self, market_state: MarketState) -> float:
        """Analyze market state and return timing."""
        start_time = time.perf_counter()
        await self.llm_agent.analyze_market(market_state)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000

    async def _create_burst_operation(self) -> float:
        """Create a burst operation for load testing."""
        data = self.market_simulator.generate_tick_data(50)
        df_data = []
        for tick in data:
            df_data.append(
                {
                    "timestamp": tick.timestamp,
                    "open": float(tick.open),
                    "high": float(tick.high),
                    "low": float(tick.low),
                    "close": float(tick.close),
                    "volume": float(tick.volume),
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)

        start_time = time.perf_counter()
        self.indicator_calc.calculate_all(df)
        end_time = time.perf_counter()

        return (end_time - start_time) * 1000

    async def _create_resource_intensive_operation(self, data_size: int) -> float:
        """Create a resource-intensive operation."""
        data = self.market_simulator.generate_tick_data(data_size)
        df_data = []
        for tick in data:
            df_data.append(
                {
                    "timestamp": tick.timestamp,
                    "open": float(tick.open),
                    "high": float(tick.high),
                    "low": float(tick.low),
                    "close": float(tick.close),
                    "volume": float(tick.volume),
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)

        start_time = time.perf_counter()

        # Perform multiple operations to increase resource usage
        for _ in range(3):
            result = self.indicator_calc.calculate_all(df)
            # Additional processing
            result["additional_calc"] = result["close"].rolling(50).mean()

        end_time = time.perf_counter()

        return (end_time - start_time) * 1000

    def _create_load_test_result(
        self,
        test_name: str,
        description: str,
        duration: float,
        response_times: list[float],
        successful_ops: int,
        failed_ops: int,
        initial_memory: float,
        final_memory: float,
        peak_memory: float,
        additional_metrics: dict[str, Any] | None = None,
    ) -> LoadTestResult:
        """Create a load test result."""
        total_ops = successful_ops + failed_ops

        # Calculate response time statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)

            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = (
                sorted_times[p95_index]
                if p95_index < len(sorted_times)
                else max_response_time
            )
            p99_response_time = (
                sorted_times[p99_index]
                if p99_index < len(sorted_times)
                else max_response_time
            )
        else:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
            p95_response_time = 0
            p99_response_time = 0

        # Calculate rates
        ops_per_second = successful_ops / duration if duration > 0 else 0
        error_rate = (failed_ops / total_ops * 100) if total_ops > 0 else 0

        # Memory usage
        memory_usage = final_memory - initial_memory

        # CPU usage (approximation)
        cpu_usage = self.process.cpu_percent()

        return LoadTestResult(
            test_name=test_name,
            description=description,
            duration_seconds=duration,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            operations_per_second=ops_per_second,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=peak_memory,
            cpu_usage_percent=cpu_usage,
            error_rate_percent=error_rate,
            additional_metrics=additional_metrics or {},
        )


async def run_load_tests(config: LoadTestConfig | None = None) -> list[LoadTestResult]:
    """
    Run the complete load test suite.

    Args:
        config: Load test configuration

    Returns:
        List of load test results
    """
    load_tests = LoadTestSuite(config)
    return await load_tests.run_all_load_tests()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        config = LoadTestConfig(
            duration_seconds=60, concurrent_users=5, operations_per_second=5
        )

        results = await run_load_tests(config)

        # Print summary
        print(f"\n{'='*80}")
        print("LOAD TEST RESULTS")
        print(f"{'='*80}")

        for result in results:
            print(f"\n{result.test_name}")
            print(f"  Description: {result.description}")
            print(f"  Duration: {result.duration_seconds:.2f}s")
            print(
                f"  Operations: {result.total_operations} (Success: {result.successful_operations}, Failed: {result.failed_operations})"
            )
            print(f"  Throughput: {result.operations_per_second:.2f} ops/sec")
            print(f"  Avg Response Time: {result.avg_response_time_ms:.2f}ms")
            print(f"  P95 Response Time: {result.p95_response_time_ms:.2f}ms")
            print(f"  P99 Response Time: {result.p99_response_time_ms:.2f}ms")
            print(f"  Memory Usage: {result.memory_usage_mb:.2f}MB")
            print(f"  Peak Memory: {result.peak_memory_mb:.2f}MB")
            print(f"  Error Rate: {result.error_rate_percent:.2f}%")

            if result.additional_metrics:
                print("  Additional Metrics:")
                for key, value in result.additional_metrics.items():
                    if isinstance(value, int | float):
                        print(f"    {key}: {value:.2f}")
                    else:
                        print(f"    {key}: {value}")

        print(f"\n{'='*80}")

    asyncio.run(main())
