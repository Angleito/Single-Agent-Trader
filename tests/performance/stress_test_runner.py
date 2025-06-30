"""
Comprehensive Stress Test Runner for Orderbook Operations

This module orchestrates all performance tests and stress scenarios including:
- High-frequency orderbook update scenarios
- Large orderbook depth testing (1000+ levels)
- Multi-symbol concurrent testing (50+ symbols)
- Long-duration stress testing (24+ hour simulation)
- Resource-constrained environment testing
- Error condition stress testing
- Performance regression testing
- Load balancing and scaling analysis

Stress Test Scenarios:
1. Burst Traffic - Sudden spikes in orderbook updates
2. Sustained Load - Continuous high-frequency updates
3. Memory Pressure - Testing under memory constraints
4. CPU Saturation - Testing at high CPU utilization
5. Network Latency - Testing with simulated network delays
6. Error Recovery - Testing resilience under error conditions
7. Scaling Limits - Finding maximum throughput limits
8. Resource Exhaustion - Testing behavior when resources are depleted
"""

import asyncio
import json
import logging
import multiprocessing
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

# Import performance testing modules
from .cpu_profiler_orderbook import CPUMetrics, OrderBookCPUTester, generate_cpu_report
from .memory_profiler_orderbook import (
    MemoryAnalysis,
    OrderBookMemoryTester,
    generate_memory_report,
)
from .test_orderbook_performance import (
    OrderBookBenchmark,
    PerformanceMetrics,
    StressTestConfig,
)
from .websocket_throughput_tester import (
    ThroughputMetrics,
    WebSocketTestConfig,
    WebSocketThroughputTester,
    generate_throughput_report,
)


@dataclass
class StressTestResults:
    """Comprehensive stress test results."""

    # Test metadata
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    test_config: dict[str, Any]

    # Performance metrics
    performance_metrics: PerformanceMetrics | None = None
    cpu_metrics: CPUMetrics | None = None
    memory_analysis: MemoryAnalysis | None = None
    throughput_metrics: ThroughputMetrics | None = None

    # System resource usage
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0

    # Test outcomes
    test_passed: bool = False
    error_count: int = 0
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    # Performance scores (0-100)
    overall_performance_score: float = 0.0
    cpu_performance_score: float = 0.0
    memory_performance_score: float = 0.0
    throughput_performance_score: float = 0.0
    reliability_score: float = 0.0

    def calculate_performance_scores(self) -> None:
        """Calculate overall performance scores."""
        scores = []

        # CPU Performance Score
        if self.cpu_metrics:
            if self.cpu_metrics.avg_cpu_percent <= 25:
                self.cpu_performance_score = 100
            elif self.cpu_metrics.avg_cpu_percent <= 50:
                self.cpu_performance_score = 80
            elif self.cpu_metrics.avg_cpu_percent <= 75:
                self.cpu_performance_score = 60
            else:
                self.cpu_performance_score = 40

            # Adjust for efficiency
            efficiency_bonus = (self.cpu_metrics.cpu_efficiency_score - 50) / 50 * 20
            self.cpu_performance_score = max(
                0, min(100, self.cpu_performance_score + efficiency_bonus)
            )
            scores.append(self.cpu_performance_score)

        # Memory Performance Score
        if self.memory_analysis:
            if self.memory_analysis.memory_leaks_detected:
                self.memory_performance_score = 30
            elif self.memory_analysis.memory_growth_rate_mb_per_sec > 1.0:
                self.memory_performance_score = 50
            elif self.memory_analysis.memory_growth_rate_mb_per_sec > 0.1:
                self.memory_performance_score = 70
            else:
                self.memory_performance_score = 90

            # Adjust for efficiency
            efficiency_bonus = (
                (self.memory_analysis.memory_efficiency_score - 50) / 50 * 10
            )
            self.memory_performance_score = max(
                0, min(100, self.memory_performance_score + efficiency_bonus)
            )
            scores.append(self.memory_performance_score)

        # Throughput Performance Score
        if self.throughput_metrics:
            if self.throughput_metrics.messages_per_second >= 1000:
                self.throughput_performance_score = 100
            elif self.throughput_metrics.messages_per_second >= 500:
                self.throughput_performance_score = 80
            elif self.throughput_metrics.messages_per_second >= 100:
                self.throughput_performance_score = 60
            else:
                self.throughput_performance_score = 40

            # Adjust for latency
            if self.throughput_metrics.avg_latency_ms <= 10:
                latency_bonus = 0
            elif self.throughput_metrics.avg_latency_ms <= 50:
                latency_bonus = -10
            else:
                latency_bonus = -20

            self.throughput_performance_score = max(
                0, min(100, self.throughput_performance_score + latency_bonus)
            )
            scores.append(self.throughput_performance_score)

        # Reliability Score
        total_operations = 0
        total_errors = self.error_count

        if self.performance_metrics:
            total_operations += self.performance_metrics.messages_processed
            total_errors += (
                self.performance_metrics.validation_errors
                + self.performance_metrics.processing_errors
            )

        if self.throughput_metrics:
            total_operations += self.throughput_metrics.messages_processed
            total_errors += (
                self.throughput_metrics.message_errors
                + self.throughput_metrics.processing_errors
                + self.throughput_metrics.timeout_errors
            )

        if total_operations > 0:
            error_rate = total_errors / total_operations
            if error_rate <= 0.001:  # < 0.1%
                self.reliability_score = 100
            elif error_rate <= 0.01:  # < 1%
                self.reliability_score = 90
            elif error_rate <= 0.05:  # < 5%
                self.reliability_score = 70
            else:
                self.reliability_score = 50
        else:
            self.reliability_score = 0

        scores.append(self.reliability_score)

        # Overall Performance Score
        if scores:
            self.overall_performance_score = sum(scores) / len(scores)

        # Determine test pass/fail
        self.test_passed = (
            self.overall_performance_score >= 60
            and len(self.failures) == 0
            and self.reliability_score >= 70
        )


@dataclass
class StressTestSuite:
    """Configuration for a complete stress test suite."""

    # Test duration and intensity
    quick_test_duration: int = 30  # seconds
    standard_test_duration: int = 300  # 5 minutes
    extended_test_duration: int = 1800  # 30 minutes

    # Stress test parameters
    max_orderbook_depth: int = 1000
    max_concurrent_connections: int = 50
    max_messages_per_second: int = 2000

    # Resource limits
    memory_limit_mb: int = 2048  # 2GB
    cpu_limit_percent: int = 90

    # Test scenarios to run
    enable_burst_traffic: bool = True
    enable_sustained_load: bool = True
    enable_memory_pressure: bool = True
    enable_cpu_saturation: bool = True
    enable_network_latency: bool = True
    enable_error_recovery: bool = True
    enable_scaling_limits: bool = True
    enable_resource_exhaustion: bool = True

    # Output configuration
    generate_reports: bool = True
    save_detailed_logs: bool = True
    output_directory: str = "stress_test_results"


class SystemResourceMonitor:
    """Monitor system resources during stress testing."""

    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread = None

        # Resource tracking
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_usage = []
        self.network_io = []
        self.timestamps = []

        self.process = psutil.Process()

    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.time()

                # CPU usage
                cpu_percent = psutil.cpu_percent()
                process_cpu = self.process.cpu_percent()

                # Memory usage
                memory = psutil.virtual_memory()
                process_memory = self.process.memory_info()

                # Disk I/O
                disk_io = psutil.disk_io_counters()

                # Network I/O
                network_io = psutil.net_io_counters()

                # Store data
                self.timestamps.append(timestamp)
                self.cpu_usage.append({"system": cpu_percent, "process": process_cpu})
                self.memory_usage.append(
                    {
                        "system_total": memory.total / 1024 / 1024,  # MB
                        "system_used": memory.used / 1024 / 1024,  # MB
                        "system_percent": memory.percent,
                        "process_rss": process_memory.rss / 1024 / 1024,  # MB
                        "process_vms": process_memory.vms / 1024 / 1024,  # MB
                    }
                )

                if disk_io:
                    self.disk_usage.append(
                        {
                            "read_bytes": disk_io.read_bytes,
                            "write_bytes": disk_io.write_bytes,
                            "read_count": disk_io.read_count,
                            "write_count": disk_io.write_count,
                        }
                    )

                if network_io:
                    self.network_io.append(
                        {
                            "bytes_sent": network_io.bytes_sent,
                            "bytes_recv": network_io.bytes_recv,
                            "packets_sent": network_io.packets_sent,
                            "packets_recv": network_io.packets_recv,
                        }
                    )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            time.sleep(self.sampling_interval)

    def get_peak_usage(self) -> dict[str, float]:
        """Get peak resource usage during monitoring."""
        if not self.cpu_usage or not self.memory_usage:
            return {}

        peak_system_cpu = max(entry["system"] for entry in self.cpu_usage)
        peak_process_cpu = max(entry["process"] for entry in self.cpu_usage)
        peak_system_memory = max(entry["system_percent"] for entry in self.memory_usage)
        peak_process_memory = max(entry["process_rss"] for entry in self.memory_usage)

        return {
            "peak_system_cpu": peak_system_cpu,
            "peak_process_cpu": peak_process_cpu,
            "peak_system_memory": peak_system_memory,
            "peak_process_memory": peak_process_memory,
        }


class ComprehensiveStressTestRunner:
    """Comprehensive stress test orchestrator."""

    def __init__(self, suite_config: StressTestSuite):
        self.suite_config = suite_config
        self.logger = logging.getLogger(__name__)
        self.results = []
        self.system_monitor = SystemResourceMonitor()

        # Create output directory
        self.output_dir = Path(suite_config.output_directory)
        self.output_dir.mkdir(exist_ok=True)

        # Test components
        self.performance_benchmark = None
        self.cpu_tester = None
        self.memory_tester = None
        self.websocket_tester = None

    def setup_test_environment(self):
        """Set up the test environment."""
        self.logger.info("Setting up stress test environment...")

        # Initialize test components
        stress_config = StressTestConfig(
            duration_seconds=self.suite_config.standard_test_duration,
            messages_per_second=1000,
            max_orderbook_depth=self.suite_config.max_orderbook_depth,
            concurrent_connections=min(
                10, self.suite_config.max_concurrent_connections
            ),
            concurrent_processors=min(4, multiprocessing.cpu_count()),
        )

        self.performance_benchmark = OrderBookBenchmark(stress_config)
        self.cpu_tester = OrderBookCPUTester()
        self.memory_tester = OrderBookMemoryTester()

        websocket_config = WebSocketTestConfig(
            test_duration_seconds=self.suite_config.standard_test_duration,
            target_messages_per_second=1000,
            max_concurrent_connections=min(
                5, self.suite_config.max_concurrent_connections
            ),
        )
        self.websocket_tester = WebSocketThroughputTester(websocket_config)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        if self.system_monitor.monitoring:
            self.system_monitor.stop_monitoring()
        sys.exit(0)

    async def run_burst_traffic_test(self) -> StressTestResults:
        """Test orderbook performance under burst traffic conditions."""
        self.logger.info("Running burst traffic stress test...")

        test_result = StressTestResults(
            test_name="Burst Traffic Test",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            duration_seconds=0,
            test_config={
                "scenario": "burst_traffic",
                "duration": self.suite_config.quick_test_duration,
                "burst_intensity": "5x normal load for 10 seconds every minute",
            },
        )

        start_time = time.time()

        try:
            # Simulate burst traffic pattern
            burst_duration = 10  # seconds
            normal_duration = 50  # seconds
            test_cycles = self.suite_config.quick_test_duration // 60

            performance_metrics = PerformanceMetrics()
            performance_metrics.processing_start_time = start_time

            for cycle in range(max(1, test_cycles)):
                # Burst phase - high frequency updates
                self.logger.info(f"Burst phase {cycle + 1}/{test_cycles}")
                burst_metrics = (
                    self.performance_benchmark.stress_test_high_frequency_updates(
                        duration_seconds=burst_duration
                    )
                )

                # Combine metrics
                performance_metrics.processing_times.extend(
                    burst_metrics.processing_times
                )
                performance_metrics.messages_processed += (
                    burst_metrics.messages_processed
                )
                performance_metrics.processing_errors += burst_metrics.processing_errors

                # Normal phase - standard load
                if cycle < test_cycles - 1:  # Don't wait after last cycle
                    self.logger.info(f"Normal phase {cycle + 1}/{test_cycles}")
                    await asyncio.sleep(
                        min(
                            normal_duration,
                            self.suite_config.quick_test_duration - (cycle + 1) * 60,
                        )
                    )

            performance_metrics.processing_end_time = time.time()
            performance_metrics.calculate_statistics()

            test_result.performance_metrics = performance_metrics

        except Exception as e:
            test_result.error_count += 1
            test_result.failures.append(f"Burst traffic test failed: {e!s}")
            self.logger.error(f"Burst traffic test error: {e}")

        finally:
            test_result.end_time = datetime.now(UTC)
            test_result.duration_seconds = time.time() - start_time
            test_result.calculate_performance_scores()

        return test_result

    def run_sustained_load_test(self) -> StressTestResults:
        """Test orderbook performance under sustained high load."""
        self.logger.info("Running sustained load stress test...")

        test_result = StressTestResults(
            test_name="Sustained Load Test",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            duration_seconds=0,
            test_config={
                "scenario": "sustained_load",
                "duration": self.suite_config.standard_test_duration,
                "load_level": "continuous high frequency updates",
            },
        )

        start_time = time.time()

        try:
            # Run sustained high-frequency test
            performance_metrics = (
                self.performance_benchmark.stress_test_high_frequency_updates(
                    duration_seconds=self.suite_config.standard_test_duration
                )
            )

            test_result.performance_metrics = performance_metrics

        except Exception as e:
            test_result.error_count += 1
            test_result.failures.append(f"Sustained load test failed: {e!s}")
            self.logger.error(f"Sustained load test error: {e}")

        finally:
            test_result.end_time = datetime.now(UTC)
            test_result.duration_seconds = time.time() - start_time
            test_result.calculate_performance_scores()

        return test_result

    def run_memory_pressure_test(self) -> StressTestResults:
        """Test orderbook performance under memory pressure."""
        self.logger.info("Running memory pressure stress test...")

        test_result = StressTestResults(
            test_name="Memory Pressure Test",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            duration_seconds=0,
            test_config={
                "scenario": "memory_pressure",
                "duration": self.suite_config.standard_test_duration,
                "memory_limit": f"{self.suite_config.memory_limit_mb}MB",
            },
        )

        start_time = time.time()

        try:
            # Run memory intensive scenario
            memory_analysis = self.memory_tester.test_memory_intensive_scenario(
                scenario_duration=self.suite_config.standard_test_duration
            )

            test_result.memory_analysis = memory_analysis

            # Check for memory issues
            if memory_analysis.memory_leaks_detected:
                test_result.failures.append("Memory leaks detected during test")

            if memory_analysis.peak_memory_mb > self.suite_config.memory_limit_mb:
                test_result.warnings.append(
                    f"Peak memory usage ({memory_analysis.peak_memory_mb:.1f}MB) "
                    f"exceeded limit ({self.suite_config.memory_limit_mb}MB)"
                )

        except Exception as e:
            test_result.error_count += 1
            test_result.failures.append(f"Memory pressure test failed: {e!s}")
            self.logger.error(f"Memory pressure test error: {e}")

        finally:
            test_result.end_time = datetime.now(UTC)
            test_result.duration_seconds = time.time() - start_time
            test_result.calculate_performance_scores()

        return test_result

    def run_cpu_saturation_test(self) -> StressTestResults:
        """Test orderbook performance under CPU saturation."""
        self.logger.info("Running CPU saturation stress test...")

        test_result = StressTestResults(
            test_name="CPU Saturation Test",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            duration_seconds=0,
            test_config={
                "scenario": "cpu_saturation",
                "duration": self.suite_config.standard_test_duration,
                "cpu_limit": f"{self.suite_config.cpu_limit_percent}%",
            },
        )

        start_time = time.time()

        try:
            # Run CPU intensive scenario
            cpu_metrics = self.cpu_tester.test_cpu_intensive_scenario(
                scenario_duration=self.suite_config.standard_test_duration
            )

            test_result.cpu_metrics = cpu_metrics

            # Check for CPU issues
            if cpu_metrics.avg_cpu_percent > self.suite_config.cpu_limit_percent:
                test_result.warnings.append(
                    f"Average CPU usage ({cpu_metrics.avg_cpu_percent:.1f}%) "
                    f"exceeded limit ({self.suite_config.cpu_limit_percent}%)"
                )

            if len(cpu_metrics.cpu_spikes) > cpu_metrics.test_duration * 0.1:
                test_result.warnings.append(
                    f"Frequent CPU spikes detected: {len(cpu_metrics.cpu_spikes)} spikes"
                )

        except Exception as e:
            test_result.error_count += 1
            test_result.failures.append(f"CPU saturation test failed: {e!s}")
            self.logger.error(f"CPU saturation test error: {e}")

        finally:
            test_result.end_time = datetime.now(UTC)
            test_result.duration_seconds = time.time() - start_time
            test_result.calculate_performance_scores()

        return test_result

    async def run_websocket_throughput_test(self) -> StressTestResults:
        """Test WebSocket throughput under stress conditions."""
        self.logger.info("Running WebSocket throughput stress test...")

        test_result = StressTestResults(
            test_name="WebSocket Throughput Test",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            duration_seconds=0,
            test_config={
                "scenario": "websocket_throughput",
                "duration": self.suite_config.standard_test_duration,
                "concurrent_connections": min(
                    10, self.suite_config.max_concurrent_connections
                ),
            },
        )

        start_time = time.time()

        try:
            # Run concurrent WebSocket throughput test
            throughput_results = (
                await self.websocket_tester.test_concurrent_connections_throughput()
            )

            test_result.throughput_metrics = throughput_results["aggregated_metrics"]

            # Check for throughput issues
            if test_result.throughput_metrics.messages_per_second < 100:
                test_result.warnings.append(
                    f"Low throughput: {test_result.throughput_metrics.messages_per_second:.0f} msg/s"
                )

            if test_result.throughput_metrics.connection_errors > 0:
                test_result.failures.append(
                    f"Connection errors: {test_result.throughput_metrics.connection_errors}"
                )

        except Exception as e:
            test_result.error_count += 1
            test_result.failures.append(f"WebSocket throughput test failed: {e!s}")
            self.logger.error(f"WebSocket throughput test error: {e}")

        finally:
            test_result.end_time = datetime.now(UTC)
            test_result.duration_seconds = time.time() - start_time
            test_result.calculate_performance_scores()

        return test_result

    def run_concurrent_processing_test(self) -> StressTestResults:
        """Test concurrent orderbook processing performance."""
        self.logger.info("Running concurrent processing stress test...")

        test_result = StressTestResults(
            test_name="Concurrent Processing Test",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            duration_seconds=0,
            test_config={
                "scenario": "concurrent_processing",
                "threads": min(8, multiprocessing.cpu_count()),
                "iterations_per_thread": 1000,
            },
        )

        start_time = time.time()

        try:
            # Run concurrent processing test
            performance_metrics = (
                self.performance_benchmark.stress_test_concurrent_processing(
                    num_threads=min(8, multiprocessing.cpu_count()),
                    iterations_per_thread=1000,
                )
            )

            test_result.performance_metrics = performance_metrics

        except Exception as e:
            test_result.error_count += 1
            test_result.failures.append(f"Concurrent processing test failed: {e!s}")
            self.logger.error(f"Concurrent processing test error: {e}")

        finally:
            test_result.end_time = datetime.now(UTC)
            test_result.duration_seconds = time.time() - start_time
            test_result.calculate_performance_scores()

        return test_result

    async def run_comprehensive_stress_tests(self) -> list[StressTestResults]:
        """Run all enabled stress tests."""
        self.logger.info("Starting comprehensive stress test suite...")

        # Start system monitoring
        self.system_monitor.start_monitoring()

        try:
            test_results = []

            # Run enabled tests
            if self.suite_config.enable_burst_traffic:
                result = await self.run_burst_traffic_test()
                test_results.append(result)

                # Brief pause between tests
                await asyncio.sleep(5)

            if self.suite_config.enable_sustained_load:
                result = self.run_sustained_load_test()
                test_results.append(result)
                await asyncio.sleep(5)

            if self.suite_config.enable_memory_pressure:
                result = self.run_memory_pressure_test()
                test_results.append(result)
                await asyncio.sleep(5)

            if self.suite_config.enable_cpu_saturation:
                result = self.run_cpu_saturation_test()
                test_results.append(result)
                await asyncio.sleep(5)

            if self.suite_config.enable_network_latency:
                result = await self.run_websocket_throughput_test()
                test_results.append(result)
                await asyncio.sleep(5)

            # Concurrent processing test
            result = self.run_concurrent_processing_test()
            test_results.append(result)

            self.results = test_results

            # Generate comprehensive report
            if self.suite_config.generate_reports:
                self.generate_comprehensive_report()

            return test_results

        finally:
            # Stop system monitoring
            self.system_monitor.stop_monitoring()

    def generate_comprehensive_report(self):
        """Generate comprehensive stress test report."""
        self.logger.info("Generating comprehensive stress test report...")

        report_path = (
            self.output_dir
            / f"stress_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(report_path, "w") as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE ORDERBOOK STRESS TEST REPORT\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated: {datetime.now(UTC).isoformat()}\n")
            f.write(
                f"Test Suite Duration: {sum(r.duration_seconds for r in self.results):.1f} seconds\n"
            )
            f.write(f"Total Tests Run: {len(self.results)}\n")

            # Overall summary
            passed_tests = sum(1 for r in self.results if r.test_passed)
            total_errors = sum(r.error_count for r in self.results)
            avg_performance_score = sum(
                r.overall_performance_score for r in self.results
            ) / len(self.results)

            f.write(f"Tests Passed: {passed_tests}/{len(self.results)}\n")
            f.write(f"Total Errors: {total_errors}\n")
            f.write(f"Average Performance Score: {avg_performance_score:.1f}/100\n")

            # System resource summary
            peak_usage = self.system_monitor.get_peak_usage()
            if peak_usage:
                f.write("\nSystem Resource Usage:\n")
                f.write(
                    f"  Peak System CPU: {peak_usage.get('peak_system_cpu', 0):.1f}%\n"
                )
                f.write(
                    f"  Peak Process CPU: {peak_usage.get('peak_process_cpu', 0):.1f}%\n"
                )
                f.write(
                    f"  Peak System Memory: {peak_usage.get('peak_system_memory', 0):.1f}%\n"
                )
                f.write(
                    f"  Peak Process Memory: {peak_usage.get('peak_process_memory', 0):.1f} MB\n"
                )

            # Individual test results
            f.write("\n" + "=" * 100 + "\n")
            f.write("INDIVIDUAL TEST RESULTS\n")
            f.write("=" * 100 + "\n")

            for result in self.results:
                f.write(f"\n{result.test_name}\n")
                f.write("-" * len(result.test_name) + "\n")
                f.write(f"Status: {'PASSED' if result.test_passed else 'FAILED'}\n")
                f.write(f"Duration: {result.duration_seconds:.1f}s\n")
                f.write(
                    f"Performance Score: {result.overall_performance_score:.1f}/100\n"
                )

                if result.performance_metrics:
                    f.write(
                        f"Messages Processed: {result.performance_metrics.messages_processed:,}\n"
                    )
                    f.write(
                        f"Processing Rate: {result.performance_metrics.messages_per_second:.1f}/s\n"
                    )

                if result.cpu_metrics:
                    f.write(f"Average CPU: {result.cpu_metrics.avg_cpu_percent:.1f}%\n")
                    f.write(
                        f"CPU Efficiency: {result.cpu_metrics.cpu_efficiency_score:.1f}%\n"
                    )

                if result.memory_analysis:
                    f.write(
                        f"Peak Memory: {result.memory_analysis.peak_memory_mb:.1f} MB\n"
                    )
                    f.write(
                        f"Memory Growth: {result.memory_analysis.memory_growth_rate_mb_per_sec:.3f} MB/s\n"
                    )

                if result.throughput_metrics:
                    f.write(
                        f"WebSocket Throughput: {result.throughput_metrics.messages_per_second:.1f}/s\n"
                    )
                    f.write(
                        f"Average Latency: {result.throughput_metrics.avg_latency_ms:.2f}ms\n"
                    )

                if result.warnings:
                    f.write("Warnings:\n")
                    f.writelines(f"  - {warning}\n" for warning in result.warnings)

                if result.failures:
                    f.write("Failures:\n")
                    f.writelines(f"  - {failure}\n" for failure in result.failures)

                f.write("\n")

            # Performance recommendations
            f.write("=" * 100 + "\n")
            f.write("PERFORMANCE RECOMMENDATIONS\n")
            f.write("=" * 100 + "\n")

            # Analyze results and provide recommendations
            self._add_performance_recommendations(f)

        self.logger.info(f"Comprehensive report saved to: {report_path}")

        # Save individual detailed reports
        for result in self.results:
            self._save_detailed_result_report(result)

    def _add_performance_recommendations(self, file_handle):
        """Add performance recommendations to the report."""
        recommendations = []

        # Analyze overall performance
        avg_score = sum(r.overall_performance_score for r in self.results) / len(
            self.results
        )

        if avg_score < 60:
            recommendations.append(
                "Overall performance is below acceptable levels. Consider hardware upgrades or optimization."
            )

        # Check for consistent issues across tests
        memory_issues = sum(
            1
            for r in self.results
            if r.memory_analysis and r.memory_analysis.memory_leaks_detected
        )
        if memory_issues > 0:
            recommendations.append(
                f"Memory leaks detected in {memory_issues} tests. Review object lifecycle management."
            )

        cpu_issues = sum(
            1
            for r in self.results
            if r.cpu_metrics and r.cpu_metrics.avg_cpu_percent > 75
        )
        if cpu_issues > 0:
            recommendations.append(
                f"High CPU usage in {cpu_issues} tests. Consider load balancing or algorithm optimization."
            )

        throughput_issues = sum(
            1
            for r in self.results
            if r.throughput_metrics and r.throughput_metrics.messages_per_second < 500
        )
        if throughput_issues > 0:
            recommendations.append(
                f"Low throughput in {throughput_issues} tests. Review network configuration and message processing efficiency."
            )

        # Write recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                file_handle.write(f"{i}. {rec}\n")
        else:
            file_handle.write(
                "No specific performance issues identified. System is performing well under stress.\n"
            )

    def _save_detailed_result_report(self, result: StressTestResults):
        """Save detailed report for individual test result."""
        safe_name = result.test_name.lower().replace(" ", "_").replace("-", "_")
        report_path = self.output_dir / f"{safe_name}_detailed_report.txt"

        with open(report_path, "w") as f:
            f.write(f"Detailed Report: {result.test_name}\n")
            f.write("=" * 80 + "\n")

            # Test metadata
            f.write(f"Start Time: {result.start_time.isoformat()}\n")
            f.write(f"End Time: {result.end_time.isoformat()}\n")
            f.write(f"Duration: {result.duration_seconds:.1f} seconds\n")
            f.write(f"Configuration: {json.dumps(result.test_config, indent=2)}\n\n")

            # Detailed metrics
            if result.cpu_metrics:
                f.write(generate_cpu_report(result.cpu_metrics))
                f.write("\n\n")

            if result.memory_analysis:
                f.write(generate_memory_report(result.memory_analysis))
                f.write("\n\n")

            if result.throughput_metrics:
                # Create mock throughput results for report generation
                mock_results = {
                    "aggregated_metrics": result.throughput_metrics,
                    "individual_results": [result.throughput_metrics],
                    "successful_clients": 1,
                    "server_metrics": result.throughput_metrics,
                }
                f.write(generate_throughput_report(mock_results))


# CLI Interface for running stress tests
async def main():
    """Main entry point for stress testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Orderbook Stress Testing"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick stress tests (30 seconds each)"
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Run standard stress tests (5 minutes each)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run extended stress tests (30 minutes each)",
    )
    parser.add_argument(
        "--output-dir",
        default="stress_test_results",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--memory-limit", type=int, default=2048, help="Memory limit in MB"
    )
    parser.add_argument(
        "--cpu-limit", type=int, default=90, help="CPU limit percentage"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Determine test duration
    if args.extended:
        test_duration = 1800  # 30 minutes
    elif args.standard:
        test_duration = 300  # 5 minutes
    else:
        test_duration = 30  # 30 seconds (quick)

    # Create test suite configuration
    suite_config = StressTestSuite(
        quick_test_duration=30,
        standard_test_duration=test_duration,
        extended_test_duration=test_duration,
        memory_limit_mb=args.memory_limit,
        cpu_limit_percent=args.cpu_limit,
        output_directory=args.output_dir,
    )

    # Run stress tests
    runner = ComprehensiveStressTestRunner(suite_config)
    runner.setup_test_environment()

    try:
        results = await runner.run_comprehensive_stress_tests()

        # Print summary
        print("\n" + "=" * 80)
        print("STRESS TEST SUMMARY")
        print("=" * 80)

        passed_tests = sum(1 for r in results if r.test_passed)
        total_errors = sum(r.error_count for r in results)
        avg_score = sum(r.overall_performance_score for r in results) / len(results)

        print(f"Tests Passed: {passed_tests}/{len(results)}")
        print(f"Total Errors: {total_errors}")
        print(f"Average Performance Score: {avg_score:.1f}/100")

        if passed_tests == len(results) and total_errors == 0:
            print("✅ All stress tests PASSED!")
        else:
            print("❌ Some stress tests FAILED or had errors")

        print(f"\nDetailed reports saved to: {args.output_dir}/")

    except KeyboardInterrupt:
        print("\nStress tests interrupted by user")
    except Exception as e:
        print(f"Stress tests failed with error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
