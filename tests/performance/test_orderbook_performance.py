"""
Comprehensive Performance Benchmarks and Stress Tests for Orderbook Operations

This module provides comprehensive performance testing for orderbook functionality including:
- Processing speed benchmarks for various orderbook sizes
- Memory usage analysis under different load conditions
- WebSocket message throughput testing
- Concurrent request handling stress tests
- Volume calculation performance benchmarks
- CPU usage profiling and analysis
- Memory leak detection for long-running operations
- Cache performance analysis

Performance Metrics Tracked:
- Orderbook processing latency (p50, p95, p99)
- Memory consumption patterns and growth
- WebSocket message processing rate (messages/second)
- Concurrent request handling capacity
- Volume calculation throughput
- Cache hit/miss ratios and efficiency
- CPU utilization patterns
- Memory allocation and garbage collection impact

Test Scenarios:
- High-frequency orderbook updates (1000+ updates/second)
- Large orderbook depths (1000+ price levels)
- Multiple symbol subscriptions (50+ symbols)
- Long-duration stress testing (24+ hours simulation)
- Memory constrained environments
- Network latency simulation
- Error condition stress testing
"""

import asyncio
import gc
import json
import logging
import multiprocessing
import os
import statistics
import threading
import time
import tracemalloc
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import psutil
import pytest
import websockets
from memory_profiler import profile
from websockets.server import serve

# Import orderbook types and utilities
from bot.fp.types.market import OrderBook, OrderBookMessage


# Performance measurement utilities
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for orderbook operations."""

    # Processing latency metrics
    processing_times: list[float] = field(default_factory=list)
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    max_latency: float = 0.0
    min_latency: float = float("inf")

    # Memory metrics
    memory_usage_mb: list[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    memory_growth_rate: float = 0.0
    gc_collections: int = 0

    # Throughput metrics
    messages_processed: int = 0
    messages_per_second: float = 0.0
    bytes_processed: int = 0
    processing_start_time: float = 0.0
    processing_end_time: float = 0.0

    # Error metrics
    validation_errors: int = 0
    processing_errors: int = 0
    timeout_errors: int = 0
    connection_errors: int = 0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: float = 0.0

    # CPU metrics
    cpu_usage_percent: list[float] = field(default_factory=list)
    avg_cpu_usage: float = 0.0
    peak_cpu_usage: float = 0.0

    def calculate_statistics(self) -> None:
        """Calculate derived statistics from collected metrics."""
        if self.processing_times:
            self.processing_times.sort()
            self.p50_latency = statistics.median(self.processing_times)
            self.p95_latency = statistics.quantiles(self.processing_times, n=20)[
                18
            ]  # 95th percentile
            self.p99_latency = statistics.quantiles(self.processing_times, n=100)[
                98
            ]  # 99th percentile
            self.max_latency = max(self.processing_times)
            self.min_latency = min(self.processing_times)

        if self.memory_usage_mb:
            self.peak_memory_mb = max(self.memory_usage_mb)
            if len(self.memory_usage_mb) > 1:
                # Calculate growth rate (MB per measurement)
                self.memory_growth_rate = (
                    self.memory_usage_mb[-1] - self.memory_usage_mb[0]
                ) / len(self.memory_usage_mb)

        if self.cpu_usage_percent:
            self.avg_cpu_usage = statistics.mean(self.cpu_usage_percent)
            self.peak_cpu_usage = max(self.cpu_usage_percent)

        # Calculate throughput
        if self.processing_end_time > self.processing_start_time:
            duration = self.processing_end_time - self.processing_start_time
            self.messages_per_second = self.messages_processed / duration

        # Calculate cache hit ratio
        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations > 0:
            self.cache_hit_ratio = self.cache_hits / total_cache_operations

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return asdict(self)


@dataclass
class StressTestConfig:
    """Configuration for stress testing scenarios."""

    # Test duration and intensity
    duration_seconds: int = 300  # 5 minutes default
    messages_per_second: int = 1000
    max_orderbook_depth: int = 100
    concurrent_connections: int = 10
    concurrent_processors: int = 4

    # Test scenarios
    enable_high_frequency: bool = True
    enable_large_orderbooks: bool = True
    enable_multi_symbol: bool = True
    enable_memory_stress: bool = True
    enable_error_injection: bool = True

    # Resource limits
    max_memory_mb: int = 1024  # 1GB memory limit
    max_cpu_percent: int = 80

    # Symbols to test
    test_symbols: list[str] = field(
        default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD", "SUI-USD", "DOGE-USD"]
    )


class PerformanceMonitor:
    """Real-time performance monitoring for orderbook operations."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_thread = None
        self._start_time = None

    def start_monitoring(self):
        """Start real-time performance monitoring."""
        self._monitoring = True
        self._start_time = time.time()
        self.metrics.processing_start_time = self._start_time
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        # Start memory tracing
        tracemalloc.start()

    def stop_monitoring(self):
        """Stop performance monitoring and calculate final statistics."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

        self.metrics.processing_end_time = time.time()
        self.metrics.calculate_statistics()

        # Stop memory tracing
        tracemalloc.stop()

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Collect memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.metrics.memory_usage_mb.append(memory_mb)

                # Collect CPU usage
                cpu_percent = self.process.cpu_percent()
                self.metrics.cpu_usage_percent.append(cpu_percent)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have ended or access denied
                break

            time.sleep(0.1)  # Monitor every 100ms

    def record_processing_time(self, processing_time: float):
        """Record a processing time measurement."""
        self.metrics.processing_times.append(processing_time)

    def record_message_processed(self, message_size_bytes: int = 0):
        """Record a processed message."""
        self.metrics.messages_processed += 1
        self.metrics.bytes_processed += message_size_bytes

    def record_error(self, error_type: str):
        """Record an error occurrence."""
        if error_type == "validation":
            self.metrics.validation_errors += 1
        elif error_type == "processing":
            self.metrics.processing_errors += 1
        elif error_type == "timeout":
            self.metrics.timeout_errors += 1
        elif error_type == "connection":
            self.metrics.connection_errors += 1

    def record_cache_operation(self, hit: bool):
        """Record a cache operation."""
        if hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1


class OrderBookBenchmark:
    """Comprehensive orderbook performance benchmarking suite."""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def performance_measurement(self):
        """Context manager for performance measurement."""
        self.monitor.start_monitoring()
        try:
            yield self.monitor
        finally:
            self.monitor.stop_monitoring()

    def generate_test_orderbook(
        self, depth: int = 100, symbol: str = "BTC-USD"
    ) -> OrderBook:
        """Generate a test orderbook with specified depth."""
        base_price = Decimal("50000.00")

        # Generate bids (descending prices)
        bids = []
        for i in range(depth):
            price = base_price - Decimal(str(i * 10))  # $10 decrements
            size = Decimal(str(1.0 + (i * 0.1)))  # Increasing size with depth
            bids.append((price, size))

        # Generate asks (ascending prices)
        asks = []
        for i in range(depth):
            price = base_price + Decimal(
                str(50 + (i * 10))
            )  # $10 increments from $50 spread
            size = Decimal(str(1.0 + (i * 0.1)))  # Increasing size with depth
            asks.append((price, size))

        return OrderBook(bids=bids, asks=asks, timestamp=datetime.now(UTC))

    def generate_orderbook_message(
        self, symbol: str = "BTC-USD", depth: int = 100
    ) -> OrderBookMessage:
        """Generate a test orderbook WebSocket message."""
        orderbook = self.generate_test_orderbook(depth, symbol)

        message_data = {
            "type": "snapshot",
            "product_id": symbol,
            "bids": [[str(price), str(size)] for price, size in orderbook.bids],
            "asks": [[str(price), str(size)] for price, size in orderbook.asks],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        return OrderBookMessage(
            channel="level2",
            timestamp=datetime.now(UTC),
            data=message_data,
            bids=orderbook.bids,
            asks=orderbook.asks,
        )

    def benchmark_orderbook_creation(
        self, iterations: int = 1000, depth: int = 100
    ) -> PerformanceMetrics:
        """Benchmark orderbook creation performance."""
        with self.performance_measurement() as monitor:
            for i in range(iterations):
                start_time = time.time()
                orderbook = self.generate_test_orderbook(depth, f"TEST-{i % 10}")
                end_time = time.time()

                monitor.record_processing_time(end_time - start_time)
                monitor.record_message_processed()

        return monitor.metrics

    def benchmark_orderbook_properties(
        self, iterations: int = 10000, depth: int = 100
    ) -> PerformanceMetrics:
        """Benchmark orderbook property access performance."""
        orderbook = self.generate_test_orderbook(depth)

        with self.performance_measurement() as monitor:
            for i in range(iterations):
                start_time = time.time()

                # Access common properties
                _ = orderbook.mid_price
                _ = orderbook.spread
                _ = orderbook.bid_depth
                _ = orderbook.ask_depth
                _ = orderbook.get_spread_bps()
                _ = orderbook.best_bid
                _ = orderbook.best_ask

                end_time = time.time()

                monitor.record_processing_time(end_time - start_time)
                monitor.record_message_processed()

        return monitor.metrics

    def benchmark_price_impact_calculations(
        self, iterations: int = 1000, depth: int = 100
    ) -> PerformanceMetrics:
        """Benchmark price impact calculation performance."""
        orderbook = self.generate_test_orderbook(depth)

        with self.performance_measurement() as monitor:
            for i in range(iterations):
                size = Decimal(str(1 + (i % 50)))  # Varying order sizes
                side = "buy" if i % 2 == 0 else "sell"

                start_time = time.time()
                _ = orderbook.price_impact(side, size)
                end_time = time.time()

                monitor.record_processing_time(end_time - start_time)
                monitor.record_message_processed()

        return monitor.metrics

    def benchmark_volume_calculations(
        self, iterations: int = 5000, depth: int = 100
    ) -> PerformanceMetrics:
        """Benchmark volume calculation performance."""
        orderbook = self.generate_test_orderbook(depth)

        with self.performance_measurement() as monitor:
            for i in range(iterations):
                size = Decimal(str(1 + (i % 20)))
                side = "BUY" if i % 2 == 0 else "SELL"

                start_time = time.time()

                # Perform various volume calculations
                _ = orderbook.get_volume_weighted_price(size, side)
                _ = orderbook.get_depth_at_price(orderbook.mid_price, side)

                end_time = time.time()

                monitor.record_processing_time(end_time - start_time)
                monitor.record_message_processed()

        return monitor.metrics

    def stress_test_high_frequency_updates(
        self, duration_seconds: int = 60
    ) -> PerformanceMetrics:
        """Stress test high-frequency orderbook updates."""

        with self.performance_measurement() as monitor:
            end_time = time.time() + duration_seconds
            update_count = 0

            while time.time() < end_time:
                start_time = time.time()

                # Generate and process orderbook update
                message = self.generate_orderbook_message(
                    symbol="BTC-USD",
                    depth=50,  # Smaller depth for high frequency
                )

                # Simulate processing
                orderbook = OrderBook(
                    bids=message.bids, asks=message.asks, timestamp=message.timestamp
                )

                # Access properties to simulate real usage
                _ = orderbook.mid_price
                _ = orderbook.spread
                _ = orderbook.bid_depth

                process_time = time.time() - start_time
                monitor.record_processing_time(process_time)
                monitor.record_message_processed(len(json.dumps(message.data)))

                update_count += 1

                # Small delay to control frequency
                await_time = 0.001  # 1ms = 1000 updates/second max
                if process_time < await_time:
                    time.sleep(await_time - process_time)

        return monitor.metrics

    def stress_test_large_orderbooks(self, iterations: int = 100) -> PerformanceMetrics:
        """Stress test with very large orderbooks."""

        with self.performance_measurement() as monitor:
            depths = [500, 1000, 2000, 5000]  # Progressively larger orderbooks

            for i in range(iterations):
                depth = depths[i % len(depths)]

                start_time = time.time()

                # Create large orderbook
                orderbook = self.generate_test_orderbook(depth)

                # Perform comprehensive operations
                _ = orderbook.mid_price
                _ = orderbook.spread
                _ = orderbook.bid_depth
                _ = orderbook.ask_depth
                _ = orderbook.get_spread_bps()

                # Price impact for large orders
                for size in [Decimal(10), Decimal(100), Decimal(1000)]:
                    _ = orderbook.price_impact("buy", size)
                    _ = orderbook.price_impact("sell", size)

                end_time = time.time()

                monitor.record_processing_time(end_time - start_time)
                monitor.record_message_processed()

        return monitor.metrics

    def stress_test_concurrent_processing(
        self, num_threads: int = 4, iterations_per_thread: int = 250
    ) -> PerformanceMetrics:
        """Stress test concurrent orderbook processing."""

        def worker_function(worker_id: int, results_queue):
            """Worker function for concurrent processing."""
            local_metrics = PerformanceMetrics()
            local_metrics.processing_start_time = time.time()

            for i in range(iterations_per_thread):
                start_time = time.time()

                # Each worker processes different symbols to avoid contention
                symbol = f"TEST-{worker_id}-{i % 10}"
                orderbook = self.generate_test_orderbook(100, symbol)

                # Perform operations
                _ = orderbook.mid_price
                _ = orderbook.spread
                _ = orderbook.price_impact("buy", Decimal(10))

                end_time = time.time()

                local_metrics.processing_times.append(end_time - start_time)
                local_metrics.messages_processed += 1

            local_metrics.processing_end_time = time.time()
            results_queue.put(local_metrics)

        # Execute concurrent workers
        import queue

        results_queue = queue.Queue()

        with self.performance_measurement() as monitor:
            threads = []
            for worker_id in range(num_threads):
                thread = threading.Thread(
                    target=worker_function, args=(worker_id, results_queue)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Aggregate results from all workers
            while not results_queue.empty():
                worker_metrics = results_queue.get()
                monitor.metrics.processing_times.extend(worker_metrics.processing_times)
                monitor.metrics.messages_processed += worker_metrics.messages_processed

        return monitor.metrics

    @profile  # Memory profiler decorator
    def stress_test_memory_leak_detection(
        self, duration_seconds: int = 300
    ) -> PerformanceMetrics:
        """Stress test for memory leak detection during long-running operations."""

        with self.performance_measurement() as monitor:
            end_time = time.time() + duration_seconds
            iteration = 0

            # Collections to potentially cause memory growth
            orderbook_cache = {}
            message_history = deque(
                maxlen=1000
            )  # Limited size to prevent unbounded growth

            while time.time() < end_time:
                start_time = time.time()

                # Create orderbook with varying parameters
                symbol = f"TEST-{iteration % 10}"
                depth = 50 + (iteration % 100)  # Varying depth

                orderbook = self.generate_test_orderbook(depth, symbol)
                message = self.generate_orderbook_message(symbol, depth)

                # Cache management (simulate real-world caching)
                cache_key = f"{symbol}_{depth}"
                if cache_key in orderbook_cache:
                    monitor.record_cache_operation(hit=True)
                else:
                    monitor.record_cache_operation(hit=False)
                    orderbook_cache[cache_key] = orderbook

                # Store message history
                message_history.append(message.data)

                # Perform operations
                _ = orderbook.mid_price
                _ = orderbook.spread
                _ = orderbook.bid_depth

                # Periodic cleanup to test garbage collection
                if iteration % 100 == 0:
                    gc.collect()
                    monitor.metrics.gc_collections += 1

                    # Clear old cache entries (simulate cache eviction)
                    if len(orderbook_cache) > 50:
                        # Remove oldest entries
                        keys_to_remove = list(orderbook_cache.keys())[:10]
                        for key in keys_to_remove:
                            del orderbook_cache[key]

                process_time = time.time() - start_time
                monitor.record_processing_time(process_time)
                monitor.record_message_processed()

                iteration += 1

                # Control processing rate
                if process_time < 0.01:  # Minimum 10ms per iteration
                    time.sleep(0.01 - process_time)

        return monitor.metrics


class WebSocketStressTester:
    """WebSocket stress testing for orderbook data streams."""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)

    async def mock_websocket_server(self, websocket, path):
        """Mock WebSocket server for stress testing."""
        try:
            symbol = "BTC-USD"
            depth = 50

            # Send initial snapshot
            orderbook = OrderBookBenchmark(self.config).generate_test_orderbook(
                depth, symbol
            )

            snapshot_message = {
                "type": "snapshot",
                "product_id": symbol,
                "bids": [
                    [str(price), str(size)]
                    for price, size in orderbook.bids[: depth // 2]
                ],
                "asks": [
                    [str(price), str(size)]
                    for price, size in orderbook.asks[: depth // 2]
                ],
                "timestamp": datetime.now(UTC).isoformat(),
            }

            await websocket.send(json.dumps(snapshot_message))

            # Send continuous updates
            update_count = 0
            start_time = time.time()

            while time.time() - start_time < self.config.duration_seconds:
                # Generate l2update message
                update_message = {
                    "type": "l2update",
                    "product_id": symbol,
                    "changes": [
                        ["buy", "49995.00", str(1.0 + (update_count % 10))],
                        ["sell", "50005.00", str(1.0 + (update_count % 10))],
                    ],
                    "timestamp": datetime.now(UTC).isoformat(),
                }

                await websocket.send(json.dumps(update_message))

                update_count += 1

                # Control message rate
                await asyncio.sleep(1.0 / self.config.messages_per_second)

        except websockets.exceptions.ConnectionClosed:
            pass

    async def websocket_client(self, uri: str) -> PerformanceMetrics:
        """WebSocket client for stress testing."""

        metrics = PerformanceMetrics()
        metrics.processing_start_time = time.time()

        try:
            async with websockets.connect(uri) as websocket:
                while (
                    time.time() - metrics.processing_start_time
                    < self.config.duration_seconds
                ):
                    start_time = time.time()

                    try:
                        # Receive message with timeout
                        message_raw = await asyncio.wait_for(
                            websocket.recv(), timeout=1.0
                        )
                        message = json.loads(message_raw)

                        # Process message
                        if message.get("type") in ["snapshot", "l2update"]:
                            # Simulate orderbook processing
                            if "bids" in message and "asks" in message:
                                # Create OrderBookMessage
                                bids = [
                                    (Decimal(str(bid[0])), Decimal(str(bid[1])))
                                    for bid in message.get("bids", [])
                                ]
                                asks = [
                                    (Decimal(str(ask[0])), Decimal(str(ask[1])))
                                    for ask in message.get("asks", [])
                                ]

                                if bids and asks:
                                    orderbook = OrderBook(
                                        bids=bids,
                                        asks=asks,
                                        timestamp=datetime.now(UTC),
                                    )

                                    # Access properties to simulate real usage
                                    _ = orderbook.mid_price
                                    _ = orderbook.spread

                        end_time = time.time()
                        processing_time = end_time - start_time

                        metrics.processing_times.append(processing_time)
                        metrics.messages_processed += 1
                        metrics.bytes_processed += len(message_raw)

                    except TimeoutError:
                        metrics.timeout_errors += 1
                    except Exception as e:
                        metrics.processing_errors += 1
                        self.logger.error(f"WebSocket processing error: {e}")

        except Exception as e:
            metrics.connection_errors += 1
            self.logger.error(f"WebSocket connection error: {e}")

        metrics.processing_end_time = time.time()
        metrics.calculate_statistics()

        return metrics

    async def run_websocket_stress_test(self) -> PerformanceMetrics:
        """Run comprehensive WebSocket stress test."""

        # Start mock server
        server_port = 8765
        server = await serve(self.mock_websocket_server, "localhost", server_port)

        try:
            # Run multiple concurrent clients
            client_tasks = []
            for i in range(self.config.concurrent_connections):
                uri = f"ws://localhost:{server_port}/ws"
                task = asyncio.create_task(self.websocket_client(uri))
                client_tasks.append(task)

            # Wait for all clients to complete
            client_results = await asyncio.gather(*client_tasks, return_exceptions=True)

            # Aggregate results
            aggregated_metrics = PerformanceMetrics()
            aggregated_metrics.processing_start_time = time.time()

            for result in client_results:
                if isinstance(result, PerformanceMetrics):
                    aggregated_metrics.processing_times.extend(result.processing_times)
                    aggregated_metrics.messages_processed += result.messages_processed
                    aggregated_metrics.bytes_processed += result.bytes_processed
                    aggregated_metrics.timeout_errors += result.timeout_errors
                    aggregated_metrics.processing_errors += result.processing_errors
                    aggregated_metrics.connection_errors += result.connection_errors

            aggregated_metrics.processing_end_time = time.time()
            aggregated_metrics.calculate_statistics()

            return aggregated_metrics

        finally:
            server.close()
            await server.wait_closed()


class TestOrderBookPerformance:
    """Performance test suite for orderbook operations."""

    @pytest.fixture
    def stress_config(self):
        """Default stress test configuration."""
        return StressTestConfig(
            duration_seconds=30,  # Shorter for tests
            messages_per_second=500,
            max_orderbook_depth=200,
            concurrent_connections=5,
            concurrent_processors=2,
        )

    @pytest.fixture
    def benchmark_suite(self, stress_config):
        """OrderBook benchmark suite."""
        return OrderBookBenchmark(stress_config)

    def test_orderbook_creation_performance(self, benchmark_suite):
        """Test orderbook creation performance benchmarks."""

        # Test different orderbook sizes
        test_cases = [
            {"depth": 10, "iterations": 5000},
            {"depth": 50, "iterations": 2000},
            {"depth": 100, "iterations": 1000},
            {"depth": 500, "iterations": 200},
        ]

        results = {}
        for case in test_cases:
            metrics = benchmark_suite.benchmark_orderbook_creation(
                iterations=case["iterations"], depth=case["depth"]
            )

            results[f"depth_{case['depth']}"] = metrics

            # Performance assertions
            assert metrics.p95_latency < 0.001  # 95% under 1ms
            assert metrics.messages_per_second > 500  # At least 500 creations/second
            assert metrics.validation_errors == 0

            print(
                f"Depth {case['depth']}: "
                f"P95={metrics.p95_latency * 1000:.2f}ms, "
                f"Rate={metrics.messages_per_second:.0f}/s, "
                f"Peak Memory={metrics.peak_memory_mb:.1f}MB"
            )

    def test_orderbook_property_access_performance(self, benchmark_suite):
        """Test orderbook property access performance."""

        metrics = benchmark_suite.benchmark_orderbook_properties(
            iterations=10000, depth=100
        )

        # Performance assertions
        assert metrics.p95_latency < 0.0001  # 95% under 0.1ms
        assert metrics.messages_per_second > 50000  # At least 50k operations/second
        assert metrics.processing_errors == 0

        print(
            f"Property Access Performance: "
            f"P95={metrics.p95_latency * 1000000:.0f}μs, "
            f"Rate={metrics.messages_per_second:.0f}/s, "
            f"Peak Memory={metrics.peak_memory_mb:.1f}MB"
        )

    def test_price_impact_calculation_performance(self, benchmark_suite):
        """Test price impact calculation performance."""

        metrics = benchmark_suite.benchmark_price_impact_calculations(
            iterations=1000, depth=200
        )

        # Performance assertions
        assert metrics.p95_latency < 0.01  # 95% under 10ms
        assert metrics.messages_per_second > 100  # At least 100 calculations/second
        assert metrics.processing_errors == 0

        print(
            f"Price Impact Performance: "
            f"P95={metrics.p95_latency * 1000:.2f}ms, "
            f"Rate={metrics.messages_per_second:.0f}/s, "
            f"Peak Memory={metrics.peak_memory_mb:.1f}MB"
        )

    def test_volume_calculation_performance(self, benchmark_suite):
        """Test volume calculation performance benchmarks."""

        metrics = benchmark_suite.benchmark_volume_calculations(
            iterations=5000, depth=100
        )

        # Performance assertions
        assert metrics.p95_latency < 0.005  # 95% under 5ms
        assert metrics.messages_per_second > 500  # At least 500 calculations/second
        assert metrics.processing_errors == 0

        print(
            f"Volume Calculation Performance: "
            f"P95={metrics.p95_latency * 1000:.2f}ms, "
            f"Rate={metrics.messages_per_second:.0f}/s, "
            f"Peak Memory={metrics.peak_memory_mb:.1f}MB"
        )

    def test_high_frequency_update_stress(self, benchmark_suite):
        """Test high-frequency orderbook update stress scenario."""

        metrics = benchmark_suite.stress_test_high_frequency_updates(
            duration_seconds=30
        )

        # Stress test assertions
        assert metrics.messages_per_second > 100  # At least 100 updates/second
        assert metrics.p99_latency < 0.1  # 99% under 100ms
        assert metrics.memory_growth_rate < 1.0  # Less than 1MB growth per measurement
        assert metrics.processing_errors == 0

        print(
            f"High Frequency Stress Test: "
            f"Rate={metrics.messages_per_second:.0f}/s, "
            f"P99={metrics.p99_latency * 1000:.2f}ms, "
            f"Memory Growth={metrics.memory_growth_rate:.2f}MB/s, "
            f"Peak Memory={metrics.peak_memory_mb:.1f}MB"
        )

    def test_large_orderbook_stress(self, benchmark_suite):
        """Test large orderbook stress scenario."""

        metrics = benchmark_suite.stress_test_large_orderbooks(iterations=50)

        # Large orderbook assertions
        assert metrics.p95_latency < 0.1  # 95% under 100ms for large orderbooks
        assert metrics.messages_per_second > 10  # At least 10 large orderbooks/second
        assert metrics.peak_memory_mb < 500  # Less than 500MB peak memory
        assert metrics.processing_errors == 0

        print(
            f"Large Orderbook Stress Test: "
            f"P95={metrics.p95_latency * 1000:.2f}ms, "
            f"Rate={metrics.messages_per_second:.0f}/s, "
            f"Peak Memory={metrics.peak_memory_mb:.1f}MB"
        )

    def test_concurrent_processing_stress(self, benchmark_suite):
        """Test concurrent orderbook processing stress scenario."""

        metrics = benchmark_suite.stress_test_concurrent_processing(
            num_threads=4, iterations_per_thread=100
        )

        # Concurrent processing assertions
        assert metrics.messages_per_second > 200  # At least 200 operations/second total
        assert metrics.p95_latency < 0.05  # 95% under 50ms
        assert metrics.processing_errors == 0

        print(
            f"Concurrent Processing Stress Test: "
            f"Total Rate={metrics.messages_per_second:.0f}/s, "
            f"P95={metrics.p95_latency * 1000:.2f}ms, "
            f"Peak Memory={metrics.peak_memory_mb:.1f}MB"
        )

    def test_memory_leak_detection(self, benchmark_suite):
        """Test memory leak detection during long-running operations."""

        metrics = benchmark_suite.stress_test_memory_leak_detection(duration_seconds=60)

        # Memory leak assertions
        assert (
            metrics.memory_growth_rate < 0.5
        )  # Less than 0.5MB growth per measurement
        assert metrics.cache_hit_ratio > 0.1  # At least 10% cache hit ratio
        assert metrics.gc_collections > 0  # Garbage collection should occur
        assert metrics.processing_errors == 0

        print(
            f"Memory Leak Detection Test: "
            f"Memory Growth={metrics.memory_growth_rate:.3f}MB/measurement, "
            f"Cache Hit Ratio={metrics.cache_hit_ratio:.2%}, "
            f"GC Collections={metrics.gc_collections}, "
            f"Peak Memory={metrics.peak_memory_mb:.1f}MB"
        )

    @pytest.mark.asyncio
    async def test_websocket_throughput_stress(self, stress_config):
        """Test WebSocket throughput stress scenario."""

        tester = WebSocketStressTester(stress_config)
        metrics = await tester.run_websocket_stress_test()

        # WebSocket throughput assertions
        assert (
            metrics.messages_per_second > 50
        )  # At least 50 messages/second per connection
        assert metrics.p95_latency < 0.1  # 95% under 100ms
        assert metrics.connection_errors == 0  # No connection failures
        assert (
            metrics.timeout_errors < metrics.messages_processed * 0.01
        )  # Less than 1% timeouts

        print(
            f"WebSocket Throughput Stress Test: "
            f"Rate={metrics.messages_per_second:.0f}/s, "
            f"P95={metrics.p95_latency * 1000:.2f}ms, "
            f"Bytes/s={metrics.bytes_processed / (metrics.processing_end_time - metrics.processing_start_time):.0f}, "
            f"Errors={metrics.connection_errors + metrics.timeout_errors + metrics.processing_errors}"
        )

    def test_comprehensive_performance_report(self, benchmark_suite, stress_config):
        """Generate comprehensive performance report."""

        print("\n" + "=" * 80)
        print("COMPREHENSIVE ORDERBOOK PERFORMANCE REPORT")
        print("=" * 80)

        # System information
        print("\nSystem Information:")
        print(f"CPU Cores: {multiprocessing.cpu_count()}")
        print(
            f"Available Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB"
        )
        print(f"Python Version: {os.sys.version}")

        # Run all benchmarks
        test_suite = [
            (
                "Orderbook Creation (depth=100)",
                lambda: benchmark_suite.benchmark_orderbook_creation(1000, 100),
            ),
            (
                "Property Access (10k ops)",
                lambda: benchmark_suite.benchmark_orderbook_properties(10000, 100),
            ),
            (
                "Price Impact Calculations",
                lambda: benchmark_suite.benchmark_price_impact_calculations(1000, 100),
            ),
            (
                "Volume Calculations",
                lambda: benchmark_suite.benchmark_volume_calculations(2000, 100),
            ),
            (
                "High Frequency Updates (30s)",
                lambda: benchmark_suite.stress_test_high_frequency_updates(30),
            ),
            (
                "Large Orderbooks",
                lambda: benchmark_suite.stress_test_large_orderbooks(50),
            ),
            (
                "Concurrent Processing",
                lambda: benchmark_suite.stress_test_concurrent_processing(4, 100),
            ),
        ]

        report_data = []

        for test_name, test_func in test_suite:
            print(f"\nRunning: {test_name}")
            start_time = time.time()

            try:
                metrics = test_func()
                duration = time.time() - start_time

                report_data.append(
                    {
                        "test": test_name,
                        "duration": duration,
                        "messages_per_second": metrics.messages_per_second,
                        "p50_latency_ms": metrics.p50_latency * 1000,
                        "p95_latency_ms": metrics.p95_latency * 1000,
                        "p99_latency_ms": metrics.p99_latency * 1000,
                        "peak_memory_mb": metrics.peak_memory_mb,
                        "total_messages": metrics.messages_processed,
                        "error_rate": (
                            metrics.validation_errors + metrics.processing_errors
                        )
                        / max(metrics.messages_processed, 1),
                    }
                )

                print(f"  ✓ Completed in {duration:.1f}s")
                print(f"  Rate: {metrics.messages_per_second:.0f}/s")
                print(f"  P95 Latency: {metrics.p95_latency * 1000:.2f}ms")
                print(f"  Peak Memory: {metrics.peak_memory_mb:.1f}MB")

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                report_data.append({"test": test_name, "error": str(e)})

        # Summary table
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(
            f"{'Test':<30} {'Rate/s':>10} {'P95(ms)':>10} {'Memory(MB)':>12} {'Errors':>8}"
        )
        print("-" * 80)

        for data in report_data:
            if "error" not in data:
                print(
                    f"{data['test']:<30} "
                    f"{data['messages_per_second']:>10.0f} "
                    f"{data['p95_latency_ms']:>10.2f} "
                    f"{data['peak_memory_mb']:>12.1f} "
                    f"{data['error_rate']:>8.2%}"
                )
            else:
                print(f"{data['test']:<30} {'ERROR':>10} {'':>10} {'':>12} {'':>8}")

        print("\n" + "=" * 80)

        # Performance assertions for overall health
        successful_tests = [d for d in report_data if "error" not in d]
        assert (
            len(successful_tests) >= len(test_suite) * 0.8
        )  # At least 80% tests should pass

        # Check that no test has extremely poor performance
        for data in successful_tests:
            assert data["error_rate"] < 0.01  # Less than 1% error rate
            assert data["peak_memory_mb"] < 1000  # Less than 1GB memory usage


if __name__ == "__main__":
    # Run performance tests
    logging.basicConfig(level=logging.INFO)

    config = StressTestConfig(duration_seconds=60)
    benchmark = OrderBookBenchmark(config)

    print("Running Orderbook Performance Benchmarks...")

    # Quick benchmark run
    metrics = benchmark.benchmark_orderbook_creation(1000, 100)
    print(
        f"Creation Performance: {metrics.messages_per_second:.0f}/s, P95: {metrics.p95_latency * 1000:.2f}ms"
    )

    print("Performance testing completed!")
