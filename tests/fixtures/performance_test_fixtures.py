"""
Performance Test Fixtures for Trading Systems

This module provides comprehensive fixtures and datasets for performance testing,
load testing, and stress testing of trading system components including:
- High-frequency data streams
- Large-scale orderbook data
- Memory usage test datasets
- Latency measurement fixtures
- Concurrent trading scenarios
"""

import asyncio
import gc
import json
import random
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import psutil

from bot.types.market_data import OrderBook
from tests.fixtures.orderbook_mock_data import (
    OrderBookMockConfig,
    OrderBookMockGenerator,
)
from tests.fixtures.websocket_message_factory import (
    MessageFactoryConfig,
    WebSocketMessageFactory,
)


@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing."""

    # Test scale parameters
    small_scale_size: int = 1000
    medium_scale_size: int = 10000
    large_scale_size: int = 100000
    stress_scale_size: int = 1000000

    # Timing parameters
    high_frequency_interval_ms: int = 1  # 1ms between messages
    normal_frequency_interval_ms: int = 100  # 100ms between messages
    burst_size: int = 1000
    burst_interval_s: int = 60

    # Memory test parameters
    memory_growth_factor: float = 1.5
    memory_test_iterations: int = 100
    gc_frequency: int = 10  # Run GC every 10 iterations

    # Concurrency parameters
    max_concurrent_operations: int = 100
    thread_pool_size: int = 20

    # Latency measurement
    latency_samples: int = 10000
    percentiles: list[float] = field(default_factory=lambda: [50, 90, 95, 99, 99.9])

    # Data diversity
    symbol_count: int = 10
    orderbook_depth: int = 100

    # System monitoring
    monitor_cpu: bool = True
    monitor_memory: bool = True
    monitor_io: bool = True

    # Random seed
    random_seed: int | None = 42


class PerformanceDataGenerator:
    """Generator for performance testing datasets."""

    def __init__(self, config: PerformanceTestConfig = None):
        self.config = config or PerformanceTestConfig()
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        self.rng = np.random.default_rng(self.config.random_seed)

        # Initialize mock data generators
        self.orderbook_generator = OrderBookMockGenerator()
        self.websocket_factory = WebSocketMessageFactory()

    def generate_large_orderbook_dataset(self, scale: str = "large") -> list[OrderBook]:
        """Generate large dataset of orderbooks for memory/performance testing."""
        scale_sizes = {
            "small": self.config.small_scale_size,
            "medium": self.config.medium_scale_size,
            "large": self.config.large_scale_size,
            "stress": self.config.stress_scale_size,
        }

        size = scale_sizes.get(scale, self.config.large_scale_size)
        orderbooks = []

        # Generate diverse symbols
        symbols = [f"SYMBOL-{i:03d}" for i in range(self.config.symbol_count)]

        base_time = datetime.now(UTC)

        for i in range(size):
            # Rotate through symbols
            symbol = symbols[i % len(symbols)]

            # Create configuration for this orderbook
            config = OrderBookMockConfig(
                symbol=symbol,
                base_price=self.rng.uniform(100, 100000),
                depth_levels=self.config.orderbook_depth,
                random_seed=self.config.random_seed + i,
            )

            generator = OrderBookMockGenerator(config)
            timestamp = base_time + timedelta(milliseconds=i * 10)

            # Vary market conditions
            condition_type = i % 4
            if condition_type == 0:
                orderbook = generator.generate_normal_orderbook(timestamp)
            elif condition_type == 1:
                orderbook = generator.generate_volatile_orderbook(timestamp)
            elif condition_type == 2:
                orderbook = generator.generate_illiquid_orderbook(timestamp)
            else:
                orderbook = generator.generate_normal_orderbook(timestamp)

            orderbooks.append(orderbook)

            # Periodic garbage collection for large datasets
            if i % self.config.gc_frequency == 0:
                gc.collect()

        return orderbooks

    def generate_high_frequency_message_stream(
        self, duration_seconds: int = 60, frequency_ms: int = None
    ) -> Generator[dict[str, Any], None, None]:
        """Generate high-frequency message stream."""
        frequency_ms = frequency_ms or self.config.high_frequency_interval_ms

        start_time = time.time()
        end_time = start_time + duration_seconds
        message_count = 0

        symbols = [f"PERF-{i:02d}" for i in range(min(5, self.config.symbol_count))]

        # Initialize factory with performance symbols
        factory_config = MessageFactoryConfig(
            symbols=symbols, message_interval_ms=frequency_ms
        )
        factory = WebSocketMessageFactory(factory_config)

        while time.time() < end_time:
            symbol = random.choice(symbols)

            # Weight message types for high frequency
            message_type = random.choices(
                ["l2update", "trade", "ticker"],
                weights=[0.7, 0.25, 0.05],  # Mostly orderbook updates
                k=1,
            )[0]

            if message_type == "l2update":
                yield factory.generate_l2update_message(symbol)
            elif message_type == "trade":
                yield factory.generate_trade_message(symbol)
            elif message_type == "ticker":
                yield factory.generate_ticker_message(symbol)

            message_count += 1

            # Simulate timing (in real test, this would be handled by event loop)
            if frequency_ms > 0:
                time.sleep(frequency_ms / 1000.0)

    def generate_burst_load_test(self) -> dict[str, list[Any]]:
        """Generate burst load test scenario."""
        bursts = []

        for burst_id in range(10):  # 10 bursts
            burst_data = {
                "burst_id": burst_id,
                "timestamp": datetime.now(UTC)
                + timedelta(seconds=burst_id * self.config.burst_interval_s),
                "messages": [],
            }

            # Generate burst of messages
            for _ in range(self.config.burst_size):
                message = self.websocket_factory.generate_l2update_message()
                burst_data["messages"].append(message)

            bursts.append(burst_data)

        return {
            "bursts": bursts,
            "total_messages": len(bursts) * self.config.burst_size,
            "expected_duration": len(bursts) * self.config.burst_interval_s,
            "peak_rate": self.config.burst_size,  # messages per burst
        }

    def generate_memory_growth_dataset(self) -> dict[str, Any]:
        """Generate dataset for memory growth testing."""
        datasets = []

        for iteration in range(self.config.memory_test_iterations):
            # Increase data size each iteration
            size_multiplier = self.config.memory_growth_factor**iteration
            dataset_size = int(self.config.small_scale_size * size_multiplier)

            # Cap at reasonable size to avoid system issues
            dataset_size = min(dataset_size, self.config.stress_scale_size)

            dataset = {
                "iteration": iteration,
                "expected_size": dataset_size,
                "orderbooks": self.generate_large_orderbook_dataset("small"),
                "messages": [
                    self.websocket_factory.generate_l2update_message()
                    for _ in range(min(dataset_size, 10000))
                ],
                "timestamp": datetime.now(UTC),
            }

            datasets.append(dataset)

            # Force garbage collection periodically
            if iteration % self.config.gc_frequency == 0:
                gc.collect()

        return {
            "datasets": datasets,
            "total_iterations": self.config.memory_test_iterations,
            "growth_factor": self.config.memory_growth_factor,
            "max_size": max(d["expected_size"] for d in datasets),
        }

    def generate_concurrent_operation_scenarios(self) -> dict[str, Any]:
        """Generate scenarios for concurrent operation testing."""
        scenarios = []

        # Scenario 1: Concurrent orderbook processing
        scenarios.append(
            {
                "name": "concurrent_orderbook_processing",
                "type": "orderbook",
                "operations": [
                    {
                        "operation_id": i,
                        "orderbook": self.orderbook_generator.generate_normal_orderbook(),
                        "processing_type": random.choice(
                            ["validate", "update", "query"]
                        ),
                    }
                    for i in range(self.config.max_concurrent_operations)
                ],
            }
        )

        # Scenario 2: Concurrent message processing
        scenarios.append(
            {
                "name": "concurrent_message_processing",
                "type": "websocket",
                "operations": [
                    {
                        "operation_id": i,
                        "message": self.websocket_factory.generate_l2update_message(),
                        "processing_type": random.choice(
                            ["parse", "validate", "route"]
                        ),
                    }
                    for i in range(self.config.max_concurrent_operations)
                ],
            }
        )

        # Scenario 3: Mixed concurrent operations
        scenarios.append(
            {"name": "mixed_concurrent_operations", "type": "mixed", "operations": []}
        )

        # Generate mixed operations
        for i in range(self.config.max_concurrent_operations):
            op_type = random.choice(["orderbook", "message", "trade"])

            if op_type == "orderbook":
                operation = {
                    "operation_id": i,
                    "type": "orderbook",
                    "data": self.orderbook_generator.generate_normal_orderbook(),
                    "action": random.choice(["process", "validate", "store"]),
                }
            elif op_type == "message":
                operation = {
                    "operation_id": i,
                    "type": "message",
                    "data": self.websocket_factory.generate_trade_message(),
                    "action": random.choice(["parse", "route", "handle"]),
                }
            else:  # trade
                operation = {
                    "operation_id": i,
                    "type": "trade",
                    "data": {
                        "symbol": "BTC-USD",
                        "side": random.choice(["buy", "sell"]),
                        "quantity": random.uniform(0.1, 10.0),
                        "price": random.uniform(45000, 55000),
                    },
                    "action": random.choice(["execute", "validate", "settle"]),
                }

            scenarios[2]["operations"].append(operation)

        return {
            "scenarios": scenarios,
            "total_operations": sum(len(s["operations"]) for s in scenarios),
            "max_concurrent": self.config.max_concurrent_operations,
        }


class LatencyMeasurementFixtures:
    """Fixtures for latency measurement and testing."""

    def __init__(self, config: PerformanceTestConfig = None):
        self.config = config or PerformanceTestConfig()
        self.measurements = []

    def create_latency_test_functions(self) -> dict[str, Callable]:
        """Create functions for measuring latency of various operations."""

        def measure_orderbook_processing():
            """Measure orderbook processing latency."""
            generator = OrderBookMockGenerator()

            start_time = time.perf_counter()
            orderbook = generator.generate_normal_orderbook()

            # Simulate processing operations
            _ = orderbook.get_best_bid()
            _ = orderbook.get_best_ask()
            _ = orderbook.get_spread()
            _ = orderbook.get_mid_price()
            _ = orderbook.get_depth_imbalance(5)

            end_time = time.perf_counter()
            return (end_time - start_time) * 1000  # Convert to milliseconds

        def measure_message_parsing():
            """Measure WebSocket message parsing latency."""
            factory = WebSocketMessageFactory()

            start_time = time.perf_counter()
            message = factory.generate_l2update_message()

            # Simulate message processing
            _ = json.dumps(message)
            _ = json.loads(json.dumps(message))

            end_time = time.perf_counter()
            return (end_time - start_time) * 1000

        def measure_data_validation():
            """Measure data validation latency."""
            generator = OrderBookMockGenerator()

            start_time = time.perf_counter()
            orderbook = generator.generate_normal_orderbook()

            # Simulate validation
            try:
                # This calls the validation in the OrderBook model
                OrderBook.model_validate(orderbook.model_dump())
            except Exception:
                pass

            end_time = time.perf_counter()
            return (end_time - start_time) * 1000

        return {
            "orderbook_processing": measure_orderbook_processing,
            "message_parsing": measure_message_parsing,
            "data_validation": measure_data_validation,
        }

    def run_latency_measurements(self) -> dict[str, dict[str, float]]:
        """Run latency measurements and calculate statistics."""
        test_functions = self.create_latency_test_functions()
        results = {}

        for test_name, test_func in test_functions.items():
            measurements = []

            # Take multiple measurements
            for _ in range(self.config.latency_samples):
                try:
                    latency = test_func()
                    measurements.append(latency)
                except Exception:
                    # Record failed measurement
                    measurements.append(float("inf"))

            # Calculate statistics
            valid_measurements = [m for m in measurements if m != float("inf")]

            if valid_measurements:
                stats = {
                    "count": len(valid_measurements),
                    "failures": len(measurements) - len(valid_measurements),
                    "min": min(valid_measurements),
                    "max": max(valid_measurements),
                    "mean": np.mean(valid_measurements),
                    "std": np.std(valid_measurements),
                    "median": np.median(valid_measurements),
                }

                # Add percentiles
                for p in self.config.percentiles:
                    stats[f"p{p}"] = np.percentile(valid_measurements, p)

                results[test_name] = stats
            else:
                results[test_name] = {"error": "All measurements failed"}

        return results


class SystemResourceMonitor:
    """Monitor system resources during performance tests."""

    def __init__(self, config: PerformanceTestConfig = None):
        self.config = config or PerformanceTestConfig()
        self.measurements = []
        self.process = psutil.Process()

    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start resource monitoring."""
        self.monitoring = True
        self.start_time = time.time()

        async def monitor_loop():
            while getattr(self, "monitoring", False):
                measurement = self.take_measurement()
                self.measurements.append(measurement)
                await asyncio.sleep(interval_seconds)

        # Note: In actual usage, this would be run in an event loop
        return monitor_loop

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False

    def take_measurement(self) -> dict[str, Any]:
        """Take a single resource measurement."""
        timestamp = time.time()
        measurement = {
            "timestamp": timestamp,
            "elapsed": timestamp - getattr(self, "start_time", timestamp),
        }

        try:
            if self.config.monitor_cpu:
                measurement.update(
                    {
                        "cpu_percent": self.process.cpu_percent(),
                        "cpu_times": self.process.cpu_times()._asdict(),
                        "system_cpu_percent": psutil.cpu_percent(interval=None),
                    }
                )

            if self.config.monitor_memory:
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                system_memory = psutil.virtual_memory()

                measurement.update(
                    {
                        "memory_rss": memory_info.rss,
                        "memory_vms": memory_info.vms,
                        "memory_percent": memory_percent,
                        "system_memory_percent": system_memory.percent,
                        "system_memory_available": system_memory.available,
                    }
                )

            if self.config.monitor_io:
                io_counters = self.process.io_counters()
                measurement.update(
                    {
                        "io_read_count": io_counters.read_count,
                        "io_write_count": io_counters.write_count,
                        "io_read_bytes": io_counters.read_bytes,
                        "io_write_bytes": io_counters.write_bytes,
                    }
                )

        except psutil.AccessDenied:
            measurement["error"] = "Access denied to process information"
        except Exception as e:
            measurement["error"] = str(e)

        return measurement

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary statistics from measurements."""
        if not self.measurements:
            return {"error": "No measurements taken"}

        stats = {}

        # Extract numeric fields
        numeric_fields = [
            "cpu_percent",
            "memory_rss",
            "memory_vms",
            "memory_percent",
            "system_cpu_percent",
            "system_memory_percent",
            "elapsed",
        ]

        for field in numeric_fields:
            values = [
                m.get(field)
                for m in self.measurements
                if field in m and m[field] is not None
            ]

            if values:
                stats[field] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "final": values[-1] if values else None,
                }

        # Add summary information
        stats["summary"] = {
            "total_measurements": len(self.measurements),
            "measurement_duration": max(
                [m.get("elapsed", 0) for m in self.measurements], default=0
            ),
            "errors": len([m for m in self.measurements if "error" in m]),
        }

        return stats


class StressTestScenarios:
    """Generate stress test scenarios for system limits."""

    def __init__(self, config: PerformanceTestConfig = None):
        self.config = config or PerformanceTestConfig()
        self.data_generator = PerformanceDataGenerator(config)

    def generate_memory_stress_scenario(self) -> dict[str, Any]:
        """Generate scenario that stresses memory usage."""
        return {
            "name": "memory_stress_test",
            "description": "Test system behavior under memory pressure",
            "data": {
                "large_orderbooks": self.data_generator.generate_large_orderbook_dataset(
                    "stress"
                ),
                "message_queues": [
                    list(self.data_generator.generate_high_frequency_message_stream(10))
                    for _ in range(10)
                ],
                "memory_growth_data": self.data_generator.generate_memory_growth_dataset(),
            },
            "expected_behavior": {
                "memory_usage_pattern": "gradual_increase",
                "gc_frequency": "increased",
                "performance_degradation": "expected_at_high_usage",
            },
        }

    def generate_cpu_stress_scenario(self) -> dict[str, Any]:
        """Generate scenario that stresses CPU usage."""
        # Generate computationally intensive operations
        large_orderbooks = self.data_generator.generate_large_orderbook_dataset("large")

        # Create operations that require heavy computation
        operations = []
        for i, orderbook in enumerate(large_orderbooks[:1000]):  # Limit for CPU test
            operations.append(
                {
                    "operation_id": i,
                    "type": "heavy_computation",
                    "data": orderbook,
                    "operations": [
                        "calculate_all_spreads",
                        "depth_analysis",
                        "volume_profile",
                        "statistical_analysis",
                        "validation",
                    ],
                }
            )

        return {
            "name": "cpu_stress_test",
            "description": "Test system behavior under CPU pressure",
            "data": {
                "operations": operations,
                "concurrent_threads": self.config.thread_pool_size,
                "iterations_per_operation": 100,
            },
            "expected_behavior": {
                "cpu_usage_pattern": "sustained_high",
                "response_time": "increased_latency",
                "throughput": "may_decrease",
            },
        }

    def generate_io_stress_scenario(self) -> dict[str, Any]:
        """Generate scenario that stresses I/O operations."""
        return {
            "name": "io_stress_test",
            "description": "Test system behavior under I/O pressure",
            "data": {
                "file_operations": [
                    {
                        "operation": "write_large_dataset",
                        "data": self.data_generator.generate_large_orderbook_dataset(
                            "medium"
                        ),
                        "format": "json",
                    },
                    {
                        "operation": "read_large_dataset",
                        "iterations": 100,
                        "concurrent_reads": 10,
                    },
                    {
                        "operation": "frequent_small_writes",
                        "count": 10000,
                        "data_size": "small",
                    },
                ],
                "network_operations": [
                    {
                        "operation": "websocket_message_flood",
                        "messages": list(
                            self.data_generator.generate_high_frequency_message_stream(
                                30
                            )
                        ),
                        "concurrent_connections": 5,
                    }
                ],
            },
            "expected_behavior": {
                "io_wait_time": "increased",
                "disk_usage": "temporary_increase",
                "network_utilization": "high",
            },
        }


# Factory function for easy access
def create_performance_test_suite(
    config: PerformanceTestConfig = None,
) -> dict[str, Any]:
    """Create complete performance test suite."""
    config = config or PerformanceTestConfig()

    data_generator = PerformanceDataGenerator(config)
    latency_fixtures = LatencyMeasurementFixtures(config)
    resource_monitor = SystemResourceMonitor(config)
    stress_scenarios = StressTestScenarios(config)

    return {
        # Performance datasets
        "large_orderbooks": {
            "small": data_generator.generate_large_orderbook_dataset("small"),
            "medium": data_generator.generate_large_orderbook_dataset("medium"),
            "large": data_generator.generate_large_orderbook_dataset("large"),
        },
        # High frequency data
        "high_frequency_streams": {
            "short_burst": list(
                data_generator.generate_high_frequency_message_stream(5)
            ),
            "medium_stream": list(
                data_generator.generate_high_frequency_message_stream(30)
            ),
            "extended_stream": list(
                data_generator.generate_high_frequency_message_stream(120)
            ),
        },
        # Load testing
        "burst_load_test": data_generator.generate_burst_load_test(),
        "memory_growth_test": data_generator.generate_memory_growth_dataset(),
        "concurrent_scenarios": data_generator.generate_concurrent_operation_scenarios(),
        # Latency measurement
        "latency_test_functions": latency_fixtures.create_latency_test_functions(),
        "resource_monitor": resource_monitor,
        # Stress test scenarios
        "stress_scenarios": {
            "memory_stress": stress_scenarios.generate_memory_stress_scenario(),
            "cpu_stress": stress_scenarios.generate_cpu_stress_scenario(),
            "io_stress": stress_scenarios.generate_io_stress_scenario(),
        },
        # Configuration and metadata
        "config": config,
        "generation_timestamp": datetime.now(UTC),
        "test_summary": {
            "dataset_scales": ["small", "medium", "large", "stress"],
            "latency_samples": config.latency_samples,
            "max_concurrent_ops": config.max_concurrent_operations,
            "stress_scenarios": 3,
            "monitoring_capabilities": ["cpu", "memory", "io"],
        },
    }


# Export main classes and functions
__all__ = [
    "LatencyMeasurementFixtures",
    "PerformanceDataGenerator",
    "PerformanceTestConfig",
    "StressTestScenarios",
    "SystemResourceMonitor",
    "create_performance_test_suite",
]
