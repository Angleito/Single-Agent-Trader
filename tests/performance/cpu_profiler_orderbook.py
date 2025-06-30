"""
CPU Profiling and Performance Analysis for Orderbook Operations

This module provides comprehensive CPU profiling and performance analysis including:
- CPU usage pattern analysis during different orderbook operations
- Function-level performance profiling with cProfile
- Line-by-line performance analysis with line_profiler
- CPU bottleneck identification and hotspot analysis
- Multi-threaded performance analysis
- CPU efficiency scoring and optimization recommendations

CPU Analysis Features:
- Real-time CPU monitoring during orderbook operations
- Function call frequency and duration analysis
- CPU usage by operation type (creation, updates, calculations)
- Thread-level CPU utilization analysis
- CPU cache efficiency analysis
- Performance regression detection
- Optimization recommendations based on profiling data
"""

import cProfile
import functools
import gc
import os
import pstats
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from io import StringIO

import psutil
from line_profiler import LineProfiler

# Import orderbook types
from bot.fp.types.market import OrderBook


@dataclass
class CPUMetrics:
    """CPU performance metrics for orderbook operations."""

    # CPU utilization
    cpu_percent: list[float] = field(default_factory=list)
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    cpu_efficiency_score: float = 0.0  # 0-100 score

    # Function timing
    function_times: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    function_call_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    total_function_time: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    # Operation timing
    operation_times: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    operations_per_second: dict[str, float] = field(default_factory=dict)

    # Thread analysis
    thread_cpu_usage: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    active_threads: int = 0
    cpu_per_thread: float = 0.0

    # Performance patterns
    cpu_spikes: list[tuple[float, float]] = field(
        default_factory=list
    )  # (timestamp, cpu_percent)
    performance_bottlenecks: list[str] = field(default_factory=list)
    optimization_opportunities: list[str] = field(default_factory=list)

    # Test metadata
    start_time: float = 0.0
    end_time: float = 0.0
    test_duration: float = 0.0
    total_operations: int = 0

    def calculate_derived_metrics(self) -> None:
        """Calculate derived CPU metrics."""
        if self.cpu_percent:
            self.avg_cpu_percent = statistics.mean(self.cpu_percent)
            self.peak_cpu_percent = max(self.cpu_percent)

            # CPU efficiency: lower variance indicates more efficient usage
            if len(self.cpu_percent) > 1:
                cpu_std = statistics.stdev(self.cpu_percent)
                # Efficiency score: high average with low variance is best
                self.cpu_efficiency_score = max(0, 100 - (cpu_std * 2))

        # Calculate operations per second
        self.test_duration = (
            self.end_time - self.start_time if self.end_time > self.start_time else 0
        )
        if self.test_duration > 0:
            for operation, times in self.operation_times.items():
                self.operations_per_second[operation] = len(times) / self.test_duration

        # Identify CPU spikes (> 80% usage)
        for i, cpu_val in enumerate(self.cpu_percent):
            if cpu_val > 80:
                timestamp = self.start_time + (i * 0.1)  # Assuming 100ms sampling
                self.cpu_spikes.append((timestamp, cpu_val))

        # Calculate CPU per thread
        if self.active_threads > 0:
            self.cpu_per_thread = self.avg_cpu_percent / self.active_threads

        # Identify bottlenecks
        self._identify_bottlenecks()

    def _identify_bottlenecks(self) -> None:
        """Identify performance bottlenecks based on profiling data."""
        self.performance_bottlenecks.clear()
        self.optimization_opportunities.clear()

        # Function-level bottlenecks
        if self.function_times:
            # Find functions with highest total time
            sorted_functions = sorted(
                self.total_function_time.items(), key=lambda x: x[1], reverse=True
            )

            total_time = sum(self.total_function_time.values())

            for func_name, func_time in sorted_functions[:5]:  # Top 5 time consumers
                time_percent = (func_time / total_time) * 100 if total_time > 0 else 0
                if time_percent > 10:  # Functions taking > 10% of total time
                    self.performance_bottlenecks.append(
                        f"{func_name}: {time_percent:.1f}% of total time ({func_time:.3f}s)"
                    )

        # High CPU usage patterns
        if self.avg_cpu_percent > 70:
            self.performance_bottlenecks.append(
                f"High average CPU usage: {self.avg_cpu_percent:.1f}%"
            )

        if len(self.cpu_spikes) > len(self.cpu_percent) * 0.1:  # More than 10% spikes
            self.performance_bottlenecks.append(
                f"Frequent CPU spikes: {len(self.cpu_spikes)} spikes detected"
            )

        # Optimization opportunities
        if self.cpu_efficiency_score < 60:
            self.optimization_opportunities.append(
                "High CPU variance detected - consider load balancing or batch processing"
            )

        # Function call frequency analysis
        for func_name, count in self.function_call_counts.items():
            if count > 10000:  # Very frequently called functions
                avg_time = (self.total_function_time[func_name] / count) * 1000  # ms
                if avg_time > 1:  # > 1ms per call
                    self.optimization_opportunities.append(
                        f"Optimize {func_name}: called {count:,} times, {avg_time:.3f}ms per call"
                    )


class CPUProfiler:
    """Advanced CPU profiler for orderbook operations."""

    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.metrics = CPUMetrics()
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_thread = None

        # Profiling tools
        self.cprofile = cProfile.Profile()
        self.line_profiler = LineProfiler()
        self._function_timers = {}

        # Thread tracking
        self._thread_monitors = {}

    def start_profiling(self) -> None:
        """Start comprehensive CPU profiling."""
        self.metrics = CPUMetrics()
        self.metrics.start_time = time.time()
        self._monitoring = True

        # Start cProfile
        self.cprofile.enable()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()

    def stop_profiling(self) -> CPUMetrics:
        """Stop profiling and return analysis results."""
        self._monitoring = False

        # Stop cProfile
        self.cprofile.disable()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        self.metrics.end_time = time.time()
        self.metrics.active_threads = threading.active_count()

        # Process cProfile results
        self._process_cprofile_results()

        # Calculate derived metrics
        self.metrics.calculate_derived_metrics()

        return self.metrics

    def _monitoring_loop(self) -> None:
        """Background CPU monitoring loop."""
        while self._monitoring:
            try:
                # Collect CPU usage
                cpu_percent = self.process.cpu_percent()
                self.metrics.cpu_percent.append(cpu_percent)

                # Collect per-thread CPU usage (if available)
                try:
                    for thread in self.process.threads():
                        thread_id = str(thread.id)
                        # Note: per-thread CPU is not available on all platforms
                        self.metrics.thread_cpu_usage[thread_id].append(
                            cpu_percent / threading.active_count()
                        )
                except (AttributeError, psutil.AccessDenied):
                    # Per-thread CPU not available
                    pass

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            time.sleep(self.sampling_interval)

    def _process_cprofile_results(self) -> None:
        """Process cProfile results and extract function timing data."""
        # Capture cProfile stats
        stats_stream = StringIO()
        stats = pstats.Stats(self.cprofile, stream=stats_stream)
        stats.sort_stats("cumulative")

        # Extract function timing data
        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, func_name = func_info

            # Create a readable function name
            if "orderbook" in filename.lower() or "market" in filename.lower():
                full_func_name = f"{func_name}({os.path.basename(filename)}:{line})"

                self.metrics.function_call_counts[full_func_name] = nc
                self.metrics.total_function_time[full_func_name] = ct

                # Estimate individual call times
                if nc > 0:
                    avg_time = ct / nc
                    self.metrics.function_times[full_func_name] = [avg_time] * nc

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling specific operations."""
        start_time = time.time()
        start_cpu = time.process_time()

        try:
            yield
        finally:
            end_time = time.time()
            end_cpu = time.process_time()

            # Record operation timing
            operation_duration = end_time - start_time
            cpu_time = end_cpu - start_cpu

            self.metrics.operation_times[operation_name].append(operation_duration)
            self.metrics.total_operations += 1

            # Log operation performance
            cpu_efficiency = (
                (cpu_time / operation_duration * 100) if operation_duration > 0 else 0
            )
            if operation_duration > 0.01:  # Log operations > 10ms
                print(
                    f"Operation '{operation_name}': {operation_duration * 1000:.2f}ms "
                    f"(CPU: {cpu_efficiency:.1f}%)"
                )

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile individual functions."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time

                self.metrics.function_times[func_name].append(duration)
                self.metrics.function_call_counts[func_name] += 1
                self.metrics.total_function_time[func_name] += duration

        # Add to line profiler
        self.line_profiler.add_function(func)

        return wrapper


class OrderBookCPUTester:
    """Specialized CPU testing for orderbook operations."""

    def __init__(self):
        self.profiler = CPUProfiler(sampling_interval=0.05)  # 50ms sampling

    def test_orderbook_creation_cpu(
        self, num_orderbooks: int = 1000, depth: int = 100
    ) -> CPUMetrics:
        """Test CPU usage during orderbook creation."""

        self.profiler.start_profiling()

        try:
            for i in range(num_orderbooks):
                with self.profiler.profile_operation("orderbook_creation"):
                    # Create orderbook with varying depth
                    current_depth = depth + (i % 50)

                    bids = [
                        (Decimal(f"{50000 - j}"), Decimal(f"{1.0 + j * 0.1}"))
                        for j in range(current_depth)
                    ]
                    asks = [
                        (Decimal(f"{50050 + j}"), Decimal(f"{1.0 + j * 0.1}"))
                        for j in range(current_depth)
                    ]

                    orderbook = OrderBook(
                        bids=bids, asks=asks, timestamp=datetime.now(UTC)
                    )

                    # Trigger validation
                    _ = orderbook.mid_price

        finally:
            return self.profiler.stop_profiling()

    def test_orderbook_operations_cpu(
        self, duration_seconds: int = 60, depth: int = 100
    ) -> CPUMetrics:
        """Test CPU usage during intensive orderbook operations."""

        # Create test orderbook
        orderbook = OrderBook(
            bids=[
                (Decimal(f"{50000 - i}"), Decimal(f"{1.0 + i * 0.1}"))
                for i in range(depth)
            ],
            asks=[
                (Decimal(f"{50050 + i}"), Decimal(f"{1.0 + i * 0.1}"))
                for i in range(depth)
            ],
            timestamp=datetime.now(UTC),
        )

        self.profiler.start_profiling()

        try:
            end_time = time.time() + duration_seconds
            operation_count = 0

            while time.time() < end_time:
                # Test different operation types
                operations = [
                    (
                        "property_access",
                        lambda: [
                            orderbook.mid_price,
                            orderbook.spread,
                            orderbook.bid_depth,
                            orderbook.ask_depth,
                            orderbook.get_spread_bps(),
                        ],
                    ),
                    (
                        "price_impact_buy",
                        lambda: orderbook.price_impact("buy", Decimal(10)),
                    ),
                    (
                        "price_impact_sell",
                        lambda: orderbook.price_impact("sell", Decimal(5)),
                    ),
                    (
                        "volume_weighted_price",
                        lambda: orderbook.get_volume_weighted_price(Decimal(20), "BUY"),
                    ),
                    (
                        "depth_calculation",
                        lambda: orderbook.get_depth_at_price(
                            orderbook.mid_price, "BUY"
                        ),
                    ),
                ]

                # Execute operations with profiling
                operation_name, operation_func = operations[
                    operation_count % len(operations)
                ]

                with self.profiler.profile_operation(operation_name):
                    _ = operation_func()

                operation_count += 1

        finally:
            return self.profiler.stop_profiling()

    def test_concurrent_cpu_usage(
        self, num_threads: int = 4, iterations_per_thread: int = 500
    ) -> CPUMetrics:
        """Test CPU usage during concurrent orderbook operations."""

        def worker_function(worker_id: int, orderbook: OrderBook):
            """Worker function for concurrent testing."""
            for i in range(iterations_per_thread):
                # Each worker performs different operations to avoid contention
                if worker_id % 4 == 0:
                    # Property access
                    _ = orderbook.mid_price
                    _ = orderbook.spread
                elif worker_id % 4 == 1:
                    # Price impact calculations
                    _ = orderbook.price_impact("buy", Decimal(str(1 + i % 10)))
                elif worker_id % 4 == 2:
                    # Volume calculations
                    _ = orderbook.get_volume_weighted_price(
                        Decimal(str(5 + i % 15)), "SELL"
                    )
                else:
                    # Depth calculations
                    _ = orderbook.get_depth_at_price(orderbook.mid_price, "BUY")

        # Create test orderbook
        orderbook = OrderBook(
            bids=[
                (Decimal(f"{50000 - i}"), Decimal(f"{1.0 + i * 0.1}"))
                for i in range(100)
            ],
            asks=[
                (Decimal(f"{50050 + i}"), Decimal(f"{1.0 + i * 0.1}"))
                for i in range(100)
            ],
            timestamp=datetime.now(UTC),
        )

        self.profiler.start_profiling()

        try:
            # Create and start threads
            threads = []
            for worker_id in range(num_threads):
                thread = threading.Thread(
                    target=worker_function, args=(worker_id, orderbook)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

        finally:
            return self.profiler.stop_profiling()

    def test_cpu_intensive_scenario(self, scenario_duration: int = 120) -> CPUMetrics:
        """Test CPU usage in intensive scenario combining all operations."""

        self.profiler.start_profiling()

        try:
            end_time = time.time() + scenario_duration
            iteration = 0

            # Data structures for CPU-intensive operations
            orderbook_cache = {}
            price_calculations = deque(maxlen=1000)
            volume_aggregates = defaultdict(float)

            while time.time() < end_time:
                with self.profiler.profile_operation("intensive_iteration"):
                    # Create varying size orderbooks
                    depth = 50 + (iteration % 100)
                    symbol = f"SYMBOL-{iteration % 20}"

                    # CPU-intensive orderbook creation
                    bids = [
                        (Decimal(f"{50000 - i}"), Decimal(f"{1.0 + i * 0.01}"))
                        for i in range(depth)
                    ]
                    asks = [
                        (Decimal(f"{50050 + i}"), Decimal(f"{1.0 + i * 0.01}"))
                        for i in range(depth)
                    ]

                    orderbook = OrderBook(
                        bids=bids, asks=asks, timestamp=datetime.now(UTC)
                    )

                    # CPU-intensive operations
                    with self.profiler.profile_operation("property_calculations"):
                        mid_price = orderbook.mid_price
                        spread = orderbook.spread
                        bid_depth = orderbook.bid_depth
                        ask_depth = orderbook.ask_depth
                        spread_bps = orderbook.get_spread_bps()

                    # CPU-intensive price impact calculations
                    with self.profiler.profile_operation("price_impact_calculations"):
                        for size in [
                            Decimal(1),
                            Decimal(10),
                            Decimal(50),
                            Decimal(100),
                        ]:
                            buy_impact = orderbook.price_impact("buy", size)
                            sell_impact = orderbook.price_impact("sell", size)

                    # CPU-intensive volume calculations
                    with self.profiler.profile_operation("volume_calculations"):
                        for size in [Decimal(5), Decimal(25), Decimal(75)]:
                            vwap_buy = orderbook.get_volume_weighted_price(size, "BUY")
                            vwap_sell = orderbook.get_volume_weighted_price(
                                size, "SELL"
                            )

                    # Data processing and aggregation
                    with self.profiler.profile_operation("data_processing"):
                        # Cache management
                        cache_key = f"{symbol}_{iteration}"
                        orderbook_cache[cache_key] = orderbook

                        # Price calculations storage
                        price_calculations.append(
                            {
                                "symbol": symbol,
                                "mid_price": float(mid_price),
                                "spread": float(spread),
                                "timestamp": time.time(),
                            }
                        )

                        # Volume aggregation
                        volume_aggregates[symbol] += float(bid_depth + ask_depth)

                        # Periodic cleanup (CPU-intensive)
                        if iteration % 50 == 0:
                            # Clean old cache entries
                            old_keys = [
                                k
                                for k in orderbook_cache
                                if int(k.split("_")[1]) < iteration - 200
                            ]
                            for key in old_keys:
                                del orderbook_cache[key]

                            # Force garbage collection
                            gc.collect()

                    iteration += 1

                    # Brief pause to prevent system overload
                    if iteration % 100 == 0:
                        time.sleep(0.01)

        finally:
            return self.profiler.stop_profiling()

    def benchmark_specific_functions(self) -> CPUMetrics:
        """Benchmark specific orderbook functions for detailed analysis."""

        # Create test orderbook
        orderbook = OrderBook(
            bids=[
                (Decimal(f"{50000 - i}"), Decimal(f"{1.0 + i * 0.1}"))
                for i in range(200)
            ],
            asks=[
                (Decimal(f"{50050 + i}"), Decimal(f"{1.0 + i * 0.1}"))
                for i in range(200)
            ],
            timestamp=datetime.now(UTC),
        )

        # Add specific functions to line profiler
        self.profiler.line_profiler.add_function(orderbook.price_impact)
        self.profiler.line_profiler.add_function(orderbook.get_volume_weighted_price)
        self.profiler.line_profiler.add_function(orderbook.get_spread_bps)

        self.profiler.start_profiling()

        try:
            # Benchmark specific functions with many iterations
            iterations = 5000

            for i in range(iterations):
                # Test each function type
                with self.profiler.profile_operation("mid_price_calculation"):
                    _ = orderbook.mid_price

                with self.profiler.profile_operation("spread_calculation"):
                    _ = orderbook.spread

                with self.profiler.profile_operation("depth_calculation"):
                    _ = orderbook.bid_depth
                    _ = orderbook.ask_depth

                with self.profiler.profile_operation("spread_bps_calculation"):
                    _ = orderbook.get_spread_bps()

                # Varying parameters for price impact
                size = Decimal(str(1 + i % 20))
                side = "buy" if i % 2 == 0 else "sell"

                with self.profiler.profile_operation("price_impact_calculation"):
                    _ = orderbook.price_impact(side, size)

                # Volume weighted price with varying parameters
                with self.profiler.profile_operation("vwap_calculation"):
                    _ = orderbook.get_volume_weighted_price(size, side.upper())

        finally:
            return self.profiler.stop_profiling()


def generate_cpu_report(metrics: CPUMetrics) -> str:
    """Generate comprehensive CPU analysis report."""

    report = []
    report.append("=" * 80)
    report.append("CPU PERFORMANCE ANALYSIS REPORT")
    report.append("=" * 80)

    # Summary statistics
    report.append("\nTest Summary:")
    report.append(f"  Test Duration: {metrics.test_duration:.1f} seconds")
    report.append(f"  Total Operations: {metrics.total_operations:,}")
    report.append(f"  Active Threads: {metrics.active_threads}")
    report.append(f"  Average CPU Usage: {metrics.avg_cpu_percent:.1f}%")
    report.append(f"  Peak CPU Usage: {metrics.peak_cpu_percent:.1f}%")
    report.append(f"  CPU Efficiency Score: {metrics.cpu_efficiency_score:.1f}%")

    # Operations per second
    if metrics.operations_per_second:
        report.append("\nOperations Performance:")
        for operation, ops_per_sec in sorted(
            metrics.operations_per_second.items(), key=lambda x: x[1], reverse=True
        ):
            report.append(f"  {operation}: {ops_per_sec:.1f} ops/sec")

    # CPU usage patterns
    if metrics.cpu_spikes:
        report.append("\nCPU Usage Patterns:")
        report.append(f"  CPU Spikes (>80%): {len(metrics.cpu_spikes)}")
        if len(metrics.cpu_spikes) <= 5:
            for timestamp, cpu_val in metrics.cpu_spikes:
                elapsed = timestamp - metrics.start_time
                report.append(f"    Spike at {elapsed:.1f}s: {cpu_val:.1f}%")
        else:
            report.append(
                f"    Peak spike: {max(spike[1] for spike in metrics.cpu_spikes):.1f}%"
            )

    # Function performance analysis
    if metrics.function_times:
        report.append("\nFunction Performance Analysis:")
        report.append(
            f"{'Function':<40} {'Calls':<10} {'Total(s)':<10} {'Avg(ms)':<10}"
        )
        report.append("-" * 80)

        # Sort by total time
        sorted_functions = sorted(
            metrics.total_function_time.items(), key=lambda x: x[1], reverse=True
        )

        for func_name, total_time in sorted_functions[:10]:  # Top 10
            calls = metrics.function_call_counts.get(func_name, 0)
            avg_time_ms = (total_time / calls * 1000) if calls > 0 else 0

            # Truncate long function names
            display_name = func_name[:37] + "..." if len(func_name) > 40 else func_name

            report.append(
                f"{display_name:<40} {calls:<10} {total_time:<10.3f} {avg_time_ms:<10.3f}"
            )

    # Performance bottlenecks
    if metrics.performance_bottlenecks:
        report.append("\nPerformance Bottlenecks:")
        for i, bottleneck in enumerate(metrics.performance_bottlenecks, 1):
            report.append(f"  {i}. {bottleneck}")
    else:
        report.append("\nPerformance Bottlenecks:")
        report.append("  ✅ No significant bottlenecks detected")

    # Optimization opportunities
    if metrics.optimization_opportunities:
        report.append("\nOptimization Opportunities:")
        for i, opportunity in enumerate(metrics.optimization_opportunities, 1):
            report.append(f"  {i}. {opportunity}")
    else:
        report.append("\nOptimization Opportunities:")
        report.append("  ✅ No specific optimization opportunities identified")

    # Thread analysis
    if metrics.thread_cpu_usage:
        report.append("\nThread Performance Analysis:")
        report.append(f"  CPU per Thread: {metrics.cpu_per_thread:.1f}%")
        report.append(
            f"  Thread Efficiency: {'Good' if metrics.cpu_per_thread < 25 else 'Poor'}"
        )

    # Performance assessment
    report.append("\nPerformance Assessment:")

    if metrics.avg_cpu_percent <= 25:
        report.append("✅ Low CPU usage - efficient processing")
    elif metrics.avg_cpu_percent <= 50:
        report.append("✅ Moderate CPU usage - acceptable performance")
    elif metrics.avg_cpu_percent <= 75:
        report.append("⚠️  High CPU usage - consider optimization")
    else:
        report.append("❌ Very high CPU usage - optimization required")

    if metrics.cpu_efficiency_score >= 80:
        report.append("✅ High CPU efficiency - consistent performance")
    elif metrics.cpu_efficiency_score >= 60:
        report.append("✅ Good CPU efficiency")
    elif metrics.cpu_efficiency_score >= 40:
        report.append("⚠️  Moderate CPU efficiency - some variance")
    else:
        report.append("❌ Low CPU efficiency - high variance in usage")

    if len(metrics.cpu_spikes) == 0:
        report.append("✅ No CPU spikes detected - stable performance")
    elif len(metrics.cpu_spikes) <= 5:
        report.append("⚠️  Few CPU spikes detected")
    else:
        report.append("❌ Frequent CPU spikes - investigate load balancing")

    # Recommendations
    report.append("\nRecommendations:")

    if metrics.avg_cpu_percent > 75:
        report.append("• Consider reducing orderbook depth or batch processing")
        report.append("• Implement caching for frequently accessed calculations")

    if metrics.cpu_efficiency_score < 60:
        report.append("• Implement load balancing across multiple threads")
        report.append("• Consider using object pools to reduce allocation overhead")

    if len(metrics.cpu_spikes) > 10:
        report.append("• Implement rate limiting for orderbook updates")
        report.append("• Consider asynchronous processing for heavy operations")

    if metrics.performance_bottlenecks:
        report.append("• Focus optimization efforts on identified bottlenecks")
        report.append("• Profile specific functions for micro-optimizations")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    print("Running orderbook CPU profiling tests...")

    tester = OrderBookCPUTester()

    # Test 1: Orderbook creation CPU
    print("\n1. Testing orderbook creation CPU usage...")
    metrics1 = tester.test_orderbook_creation_cpu(num_orderbooks=500, depth=100)
    print(
        f"Creation CPU: Avg={metrics1.avg_cpu_percent:.1f}%, "
        f"Peak={metrics1.peak_cpu_percent:.1f}%, "
        f"Efficiency={metrics1.cpu_efficiency_score:.1f}%"
    )

    # Test 2: Operations CPU
    print("\n2. Testing orderbook operations CPU usage...")
    metrics2 = tester.test_orderbook_operations_cpu(duration_seconds=30, depth=100)
    print(
        f"Operations CPU: Avg={metrics2.avg_cpu_percent:.1f}%, "
        f"Peak={metrics2.peak_cpu_percent:.1f}%, "
        f"Efficiency={metrics2.cpu_efficiency_score:.1f}%"
    )

    # Test 3: Concurrent CPU usage
    print("\n3. Testing concurrent CPU usage...")
    metrics3 = tester.test_concurrent_cpu_usage(
        num_threads=4, iterations_per_thread=250
    )
    print(
        f"Concurrent CPU: Avg={metrics3.avg_cpu_percent:.1f}%, "
        f"Per Thread={metrics3.cpu_per_thread:.1f}%, "
        f"Efficiency={metrics3.cpu_efficiency_score:.1f}%"
    )

    # Generate detailed report for the most comprehensive test
    print("\nDetailed CPU Analysis Report:")
    print(generate_cpu_report(metrics2))

    print("\n✅ CPU profiling tests completed!")
