"""
Memory Profiling and Leak Detection for Orderbook Operations

This module provides specialized memory profiling tools for orderbook operations including:
- Memory usage analysis under different load conditions
- Memory leak detection for long-running streams
- Memory allocation pattern analysis
- Garbage collection impact measurement
- Memory optimization recommendations

Memory Analysis Features:
- Real-time memory monitoring during orderbook operations
- Memory growth pattern detection
- Peak memory usage tracking
- Memory fragmentation analysis
- Garbage collection frequency and impact
- Memory efficiency scoring
- Memory usage predictions under load
"""

import gc
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import psutil
from memory_profiler import profile

# Import orderbook types
from bot.fp.types.market import OrderBook, OrderBookMessage


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""

    timestamp: datetime
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    heap_mb: float  # Python heap memory in MB
    peak_mb: float  # Peak memory usage in MB
    available_mb: float  # Available system memory in MB
    gc_objects: int  # Number of objects tracked by GC
    gc_collections: dict[int, int]  # GC collections by generation

    # Memory allocation info from tracemalloc
    current_size: int = 0
    peak_size: int = 0
    trace_count: int = 0


@dataclass
class MemoryAnalysis:
    """Comprehensive memory analysis results."""

    snapshots: list[MemorySnapshot] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None

    # Memory growth analysis
    memory_growth_rate_mb_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    baseline_memory_mb: float = 0.0
    memory_efficiency_score: float = 0.0  # 0-100 score

    # Leak detection
    potential_leaks: list[dict[str, Any]] = field(default_factory=list)
    memory_leaks_detected: bool = False
    leak_growth_threshold_mb: float = 10.0  # Memory growth threshold for leak detection

    # GC analysis
    gc_pressure_score: float = 0.0  # 0-100 score (higher = more pressure)
    avg_gc_collections_per_minute: float = 0.0
    gc_impact_on_performance: float = 0.0

    # Optimization recommendations
    recommendations: list[str] = field(default_factory=list)

    def analyze_memory_patterns(self) -> None:
        """Analyze memory usage patterns and generate insights."""
        if len(self.snapshots) < 2:
            return

        # Calculate memory growth rate
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]

        duration_seconds = (
            last_snapshot.timestamp - first_snapshot.timestamp
        ).total_seconds()
        if duration_seconds > 0:
            memory_growth = last_snapshot.rss_mb - first_snapshot.rss_mb
            self.memory_growth_rate_mb_per_sec = memory_growth / duration_seconds

        # Peak memory
        self.peak_memory_mb = max(snapshot.rss_mb for snapshot in self.snapshots)
        self.baseline_memory_mb = first_snapshot.rss_mb

        # Memory efficiency (how well memory is being used)
        if self.peak_memory_mb > 0:
            avg_memory = sum(s.rss_mb for s in self.snapshots) / len(self.snapshots)
            self.memory_efficiency_score = (avg_memory / self.peak_memory_mb) * 100

        # Leak detection
        self._detect_memory_leaks()

        # GC analysis
        self._analyze_gc_pressure()

        # Generate recommendations
        self._generate_recommendations()

    def _detect_memory_leaks(self) -> None:
        """Detect potential memory leaks based on growth patterns."""
        if len(self.snapshots) < 10:  # Need enough data points
            return

        # Analyze memory growth trend
        recent_snapshots = self.snapshots[-10:]  # Last 10 snapshots
        memory_values = [s.rss_mb for s in recent_snapshots]

        # Simple linear regression to detect upward trend
        n = len(memory_values)
        x_vals = list(range(n))

        # Calculate slope
        sum_x = sum(x_vals)
        sum_y = sum(memory_values)
        sum_xy = sum(x * y for x, y in zip(x_vals, memory_values, strict=False))
        sum_x_squared = sum(x * x for x in x_vals)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)

        # If slope is positive and significant, potential leak
        if slope > 0.1:  # Growing by 0.1 MB per measurement
            leak_info = {
                "growth_rate_mb_per_measurement": slope,
                "total_growth_mb": memory_values[-1] - memory_values[0],
                "detected_at": datetime.now(UTC),
                "confidence": min(100, slope * 100),  # Simple confidence metric
            }

            self.potential_leaks.append(leak_info)

            if slope > self.leak_growth_threshold_mb / len(recent_snapshots):
                self.memory_leaks_detected = True

    def _analyze_gc_pressure(self) -> None:
        """Analyze garbage collection pressure and impact."""
        if not self.snapshots:
            return

        # Calculate total GC collections across all generations
        total_collections = defaultdict(int)
        for snapshot in self.snapshots:
            for gen, count in snapshot.gc_collections.items():
                total_collections[gen] += count

        # Calculate GC pressure score
        duration_minutes = len(self.snapshots) / 60  # Assuming 1 snapshot per second
        if duration_minutes > 0:
            # Weight higher generations more heavily
            weighted_collections = (
                total_collections[0] * 1
                + total_collections[1] * 3
                + total_collections[2] * 10
            )
            self.avg_gc_collections_per_minute = weighted_collections / duration_minutes
            self.gc_pressure_score = min(100, self.avg_gc_collections_per_minute * 2)

    def _generate_recommendations(self) -> None:
        """Generate memory optimization recommendations."""
        self.recommendations.clear()

        # High memory usage
        if self.peak_memory_mb > 500:  # More than 500MB
            self.recommendations.append(
                f"High peak memory usage detected ({self.peak_memory_mb:.1f}MB). "
                "Consider reducing orderbook depth or implementing memory-efficient data structures."
            )

        # Memory growth
        if self.memory_growth_rate_mb_per_sec > 0.1:  # Growing by 0.1MB per second
            self.recommendations.append(
                f"Memory growth detected ({self.memory_growth_rate_mb_per_sec:.3f}MB/s). "
                "Review object lifecycle and implement proper cleanup."
            )

        # Memory leaks
        if self.memory_leaks_detected:
            self.recommendations.append(
                "Potential memory leaks detected. Review object references and "
                "ensure proper cleanup of orderbook caches and message buffers."
            )

        # Low memory efficiency
        if self.memory_efficiency_score < 50:
            self.recommendations.append(
                f"Low memory efficiency ({self.memory_efficiency_score:.1f}%). "
                "Consider object pooling or more efficient data structures."
            )

        # High GC pressure
        if self.gc_pressure_score > 70:
            self.recommendations.append(
                f"High garbage collection pressure ({self.gc_pressure_score:.1f}). "
                "Reduce object allocation frequency or use object pooling."
            )

        # Memory fragmentation
        avg_vms = sum(s.vms_mb for s in self.snapshots) / len(self.snapshots)
        avg_rss = sum(s.rss_mb for s in self.snapshots) / len(self.snapshots)
        if avg_vms > avg_rss * 1.5:  # VMS significantly higher than RSS
            self.recommendations.append(
                "Memory fragmentation detected. Consider periodic memory compaction "
                "or using memory pools for frequently allocated objects."
            )


class MemoryProfiler:
    """Advanced memory profiler for orderbook operations."""

    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.analysis = MemoryAnalysis()
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_thread = None
        self._tracemalloc_enabled = False

    def start_profiling(self) -> None:
        """Start memory profiling."""
        self.analysis = MemoryAnalysis()
        self._monitoring = True

        # Start tracemalloc for detailed memory tracking
        tracemalloc.start()
        self._tracemalloc_enabled = True

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()

    def stop_profiling(self) -> MemoryAnalysis:
        """Stop profiling and return analysis results."""
        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        if self._tracemalloc_enabled:
            tracemalloc.stop()
            self._tracemalloc_enabled = False

        self.analysis.end_time = datetime.now(UTC)
        self.analysis.analyze_memory_patterns()

        return self.analysis

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                snapshot = self._take_memory_snapshot()
                self.analysis.snapshots.append(snapshot)
            except Exception as e:
                print(f"Memory monitoring error: {e}")

            time.sleep(self.sampling_interval)

    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        # System memory info
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()

        # GC info
        gc_stats = gc.get_stats()
        gc_collections = {i: stat["collections"] for i, stat in enumerate(gc_stats)}

        # Tracemalloc info
        current_size = 0
        peak_size = 0
        trace_count = 0

        if self._tracemalloc_enabled:
            current, peak = tracemalloc.get_traced_memory()
            current_size = current
            peak_size = peak

            # Count traces
            snapshot = tracemalloc.take_snapshot()
            trace_count = len(snapshot.traces)

        return MemorySnapshot(
            timestamp=datetime.now(UTC),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            heap_mb=current_size / 1024 / 1024,
            peak_mb=peak_size / 1024 / 1024,
            available_mb=system_memory.available / 1024 / 1024,
            gc_objects=len(gc.get_objects()),
            gc_collections=gc_collections,
            current_size=current_size,
            peak_size=peak_size,
            trace_count=trace_count,
        )

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling specific operations."""
        start_snapshot = self._take_memory_snapshot()

        try:
            yield
        finally:
            end_snapshot = self._take_memory_snapshot()

            # Log operation memory impact
            memory_delta = end_snapshot.rss_mb - start_snapshot.rss_mb
            print(f"Operation '{operation_name}' memory impact: {memory_delta:+.2f}MB")


class OrderBookMemoryTester:
    """Specialized memory testing for orderbook operations."""

    def __init__(self):
        self.profiler = MemoryProfiler(sampling_interval=0.05)  # 50ms sampling

    def test_orderbook_creation_memory(
        self, num_orderbooks: int = 1000, depth: int = 100
    ) -> MemoryAnalysis:
        """Test memory usage during orderbook creation."""

        self.profiler.start_profiling()

        try:
            orderbooks = []

            for i in range(num_orderbooks):
                # Create orderbook with varying depth
                current_depth = depth + (i % 50)  # Vary depth

                bids = [
                    (Decimal(f"{50000 - j}"), Decimal(f"{1.0 + j * 0.1}"))
                    for j in range(current_depth)
                ]
                asks = [
                    (Decimal(f"{50050 + j}"), Decimal(f"{1.0 + j * 0.1}"))
                    for j in range(current_depth)
                ]

                orderbook = OrderBook(bids=bids, asks=asks, timestamp=datetime.now(UTC))

                orderbooks.append(orderbook)

                # Periodic cleanup to test GC behavior
                if i % 100 == 0:
                    gc.collect()

                    # Remove some older orderbooks to test memory reclamation
                    if len(orderbooks) > 500:
                        orderbooks = orderbooks[-400:]  # Keep only recent 400

        finally:
            analysis = self.profiler.stop_profiling()

        return analysis

    def test_orderbook_operations_memory(
        self, duration_seconds: int = 60
    ) -> MemoryAnalysis:
        """Test memory usage during intensive orderbook operations."""

        self.profiler.start_profiling()

        try:
            # Create base orderbook
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

            end_time = time.time() + duration_seconds
            operation_count = 0

            # Cache for testing memory accumulation
            results_cache = deque(maxlen=1000)  # Limited size cache

            while time.time() < end_time:
                # Perform various operations
                operations = [
                    lambda: orderbook.mid_price,
                    lambda: orderbook.spread,
                    lambda: orderbook.bid_depth,
                    lambda: orderbook.ask_depth,
                    lambda: orderbook.get_spread_bps(),
                    lambda: orderbook.price_impact("buy", Decimal(10)),
                    lambda: orderbook.price_impact("sell", Decimal(5)),
                    lambda: orderbook.get_volume_weighted_price(Decimal(20), "BUY"),
                ]

                # Execute random operations
                operation = operations[operation_count % len(operations)]
                result = operation()
                results_cache.append(result)

                operation_count += 1

                # Periodic memory cleanup
                if operation_count % 1000 == 0:
                    gc.collect()

        finally:
            analysis = self.profiler.stop_profiling()

        return analysis

    def test_websocket_message_processing_memory(
        self, num_messages: int = 5000
    ) -> MemoryAnalysis:
        """Test memory usage during WebSocket message processing."""

        self.profiler.start_profiling()

        try:
            messages_cache = []
            processed_orderbooks = {}

            for i in range(num_messages):
                # Generate orderbook message
                symbol = f"TEST-{i % 10}"
                depth = 50 + (i % 50)

                # Create message data
                message_data = {
                    "type": "snapshot" if i % 100 == 0 else "l2update",
                    "product_id": symbol,
                    "bids": [
                        [f"{50000 - j}", f"{1.0 + j * 0.1}"] for j in range(depth)
                    ],
                    "asks": [
                        [f"{50050 + j}", f"{1.0 + j * 0.1}"] for j in range(depth)
                    ],
                    "timestamp": datetime.now(UTC).isoformat(),
                }

                # Process message
                bids = [
                    (Decimal(bid[0]), Decimal(bid[1])) for bid in message_data["bids"]
                ]
                asks = [
                    (Decimal(ask[0]), Decimal(ask[1])) for ask in message_data["asks"]
                ]

                orderbook_msg = OrderBookMessage(
                    channel="level2",
                    timestamp=datetime.now(UTC),
                    data=message_data,
                    bids=bids,
                    asks=asks,
                )

                # Create orderbook
                orderbook = OrderBook(
                    bids=bids, asks=asks, timestamp=orderbook_msg.timestamp
                )

                # Cache management (simulate real-world usage)
                cache_key = f"{symbol}_{i}"
                processed_orderbooks[cache_key] = orderbook
                messages_cache.append(orderbook_msg)

                # Periodic cleanup
                if i % 500 == 0:
                    # Remove old cache entries
                    old_keys = [
                        k
                        for k in processed_orderbooks
                        if int(k.split("_")[1]) < i - 1000
                    ]
                    for key in old_keys:
                        del processed_orderbooks[key]

                    # Limit message cache
                    if len(messages_cache) > 1000:
                        messages_cache = messages_cache[-500:]

                    gc.collect()

        finally:
            analysis = self.profiler.stop_profiling()

        return analysis

    @profile  # Memory profiler decorator
    def test_memory_intensive_scenario(
        self, scenario_duration: int = 120
    ) -> MemoryAnalysis:
        """Test memory usage in intensive scenario combining all operations."""

        self.profiler.start_profiling()

        try:
            # Multiple data structures to simulate real application
            orderbook_cache = {}
            message_buffer = deque(maxlen=5000)
            price_history = defaultdict(list)
            volume_aggregates = defaultdict(float)

            end_time = time.time() + scenario_duration
            iteration = 0

            while time.time() < end_time:
                symbol = f"SYMBOL-{iteration % 20}"
                depth = 100 + (iteration % 100)

                # Create large orderbook
                with self.profiler.profile_operation(f"create_orderbook_{iteration}"):
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

                # Perform intensive operations
                with self.profiler.profile_operation(f"operations_{iteration}"):
                    mid_price = orderbook.mid_price
                    spread = orderbook.spread

                    # Multiple price impact calculations
                    for size in [Decimal(1), Decimal(10), Decimal(100)]:
                        buy_impact = orderbook.price_impact("buy", size)
                        sell_impact = orderbook.price_impact("sell", size)

                # Store results (memory accumulation test)
                orderbook_cache[f"{symbol}_{iteration}"] = orderbook
                price_history[symbol].append(float(mid_price))
                volume_aggregates[symbol] += float(
                    orderbook.bid_depth + orderbook.ask_depth
                )

                # Add to message buffer
                message_buffer.append(
                    {
                        "symbol": symbol,
                        "mid_price": float(mid_price),
                        "spread": float(spread),
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

                # Periodic cleanup and memory management
                if iteration % 100 == 0:
                    # Clean old cache entries
                    old_keys = [
                        k
                        for k in orderbook_cache
                        if int(k.split("_")[1]) < iteration - 500
                    ]
                    for key in old_keys:
                        del orderbook_cache[key]

                    # Limit price history
                    for symbol_key in price_history:
                        if len(price_history[symbol_key]) > 1000:
                            price_history[symbol_key] = price_history[symbol_key][-500:]

                    # Force garbage collection
                    gc.collect()

                iteration += 1

                # Control processing rate
                time.sleep(0.01)  # 10ms delay

        finally:
            analysis = self.profiler.stop_profiling()

        return analysis


def generate_memory_report(analysis: MemoryAnalysis) -> str:
    """Generate a comprehensive memory analysis report."""

    report = []
    report.append("=" * 80)
    report.append("MEMORY ANALYSIS REPORT")
    report.append("=" * 80)

    # Summary statistics
    duration = (
        (analysis.end_time - analysis.start_time).total_seconds()
        if analysis.end_time
        else 0
    )
    report.append(f"\nTest Duration: {duration:.1f} seconds")
    report.append(f"Memory Snapshots: {len(analysis.snapshots)}")
    report.append(f"Baseline Memory: {analysis.baseline_memory_mb:.1f} MB")
    report.append(f"Peak Memory: {analysis.peak_memory_mb:.1f} MB")
    report.append(
        f"Memory Growth Rate: {analysis.memory_growth_rate_mb_per_sec:.3f} MB/s"
    )
    report.append(f"Memory Efficiency Score: {analysis.memory_efficiency_score:.1f}%")

    # Leak detection
    report.append(f"\n{'=' * 40} LEAK DETECTION {'=' * 40}")
    if analysis.memory_leaks_detected:
        report.append("⚠️  POTENTIAL MEMORY LEAKS DETECTED!")
        for i, leak in enumerate(analysis.potential_leaks, 1):
            report.append(f"  Leak #{i}:")
            report.append(
                f"    Growth Rate: {leak['growth_rate_mb_per_measurement']:.3f} MB/measurement"
            )
            report.append(f"    Total Growth: {leak['total_growth_mb']:.1f} MB")
            report.append(f"    Confidence: {leak['confidence']:.1f}%")
    else:
        report.append("✅ No memory leaks detected")

    # GC analysis
    report.append(f"\n{'=' * 40} GARBAGE COLLECTION {'=' * 40}")
    report.append(f"GC Pressure Score: {analysis.gc_pressure_score:.1f}%")
    report.append(
        f"Avg GC Collections/min: {analysis.avg_gc_collections_per_minute:.1f}"
    )

    if analysis.gc_pressure_score > 70:
        report.append("⚠️  High garbage collection pressure detected!")
    elif analysis.gc_pressure_score > 40:
        report.append("⚪ Moderate garbage collection pressure")
    else:
        report.append("✅ Low garbage collection pressure")

    # Memory usage timeline
    if len(analysis.snapshots) > 1:
        report.append(f"\n{'=' * 40} MEMORY TIMELINE {'=' * 40}")
        report.append("Time (s)    RSS (MB)    Heap (MB)   GC Objects")
        report.append("-" * 50)

        start_time = analysis.snapshots[0].timestamp
        for i, snapshot in enumerate(
            analysis.snapshots[:: max(1, len(analysis.snapshots) // 10)]
        ):
            elapsed = (snapshot.timestamp - start_time).total_seconds()
            report.append(
                f"{elapsed:8.1f}    {snapshot.rss_mb:8.1f}    {snapshot.heap_mb:9.1f}    {snapshot.gc_objects:10d}"
            )

    # Recommendations
    if analysis.recommendations:
        report.append(f"\n{'=' * 40} RECOMMENDATIONS {'=' * 40}")
        for i, rec in enumerate(analysis.recommendations, 1):
            report.append(f"{i}. {rec}")
    else:
        report.append(f"\n{'=' * 40} RECOMMENDATIONS {'=' * 40}")
        report.append("✅ No specific memory optimization recommendations")

    # Memory efficiency breakdown
    if analysis.snapshots:
        avg_rss = sum(s.rss_mb for s in analysis.snapshots) / len(analysis.snapshots)
        avg_heap = sum(s.heap_mb for s in analysis.snapshots) / len(analysis.snapshots)
        avg_objects = sum(s.gc_objects for s in analysis.snapshots) / len(
            analysis.snapshots
        )

        report.append(f"\n{'=' * 40} EFFICIENCY METRICS {'=' * 40}")
        report.append(f"Average RSS Memory: {avg_rss:.1f} MB")
        report.append(f"Average Heap Memory: {avg_heap:.1f} MB")
        report.append(f"Average GC Objects: {avg_objects:.0f}")
        report.append(
            f"Memory Utilization: {(avg_heap / avg_rss) * 100:.1f}%"
            if avg_rss > 0
            else "Memory Utilization: N/A"
        )

    report.append("\n" + "=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    print("Running orderbook memory profiling tests...")

    tester = OrderBookMemoryTester()

    # Test 1: Orderbook creation memory
    print("\n1. Testing orderbook creation memory usage...")
    analysis1 = tester.test_orderbook_creation_memory(num_orderbooks=500, depth=100)
    print(generate_memory_report(analysis1))

    # Test 2: Operations memory
    print("\n2. Testing orderbook operations memory usage...")
    analysis2 = tester.test_orderbook_operations_memory(duration_seconds=30)
    print(generate_memory_report(analysis2))

    # Test 3: WebSocket processing memory
    print("\n3. Testing WebSocket message processing memory...")
    analysis3 = tester.test_websocket_message_processing_memory(num_messages=2000)
    print(generate_memory_report(analysis3))

    print("\n✅ Memory profiling tests completed!")
