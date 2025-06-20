#!/usr/bin/env python3
"""
WebSocket Load Testing
Tests system performance under high message load.
Monitors queue handling, memory usage, and connection stability.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import psutil
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestMetrics:
    """Metrics collected during load test."""

    messages_sent: int = 0
    messages_failed: int = 0
    messages_queued: int = 0

    min_latency: float = float("inf")
    max_latency: float = 0
    avg_latency: float = 0
    p95_latency: float = 0
    p99_latency: float = 0

    memory_start: float = 0
    memory_peak: float = 0
    memory_end: float = 0

    cpu_avg: float = 0
    cpu_peak: float = 0

    connection_failures: int = 0
    reconnections: int = 0

    test_duration: float = 0
    messages_per_second: float = 0


class WebSocketLoadTester:
    """Load test WebSocket connections and message handling."""

    def __init__(self, dashboard_url: str = "ws://dashboard-backend:8000/ws"):
        self.dashboard_url = dashboard_url
        self.metrics = LoadTestMetrics()
        self.latencies: list[float] = []
        self.memory_samples: list[float] = []
        self.cpu_samples: list[float] = []
        self.process = psutil.Process()
        self.monitoring = False

    async def monitor_resources(self, interval: float = 1.0):
        """Monitor CPU and memory usage."""
        self.monitoring = True

        while self.monitoring:
            try:
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_samples.append(memory_mb)

                # CPU usage
                cpu_percent = self.process.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)

                await asyncio.sleep(interval)
            except Exception as e:
                logger.exception("Resource monitoring error: %s", e)
                break

    async def send_burst_messages(
        self,
        websocket: websockets.WebSocketClientProtocol,
        count: int,
        delay: float = 0,
    ) -> tuple[int, int]:
        """Send a burst of messages."""
        sent = 0
        failed = 0

        for i in range(count):
            try:
                message = {
                    "type": "load_test",
                    "timestamp": datetime.utcnow().isoformat(),
                    "sequence": i,
                    "data": {
                        "test_id": f"load_{i}",
                        "payload": "x" * 1024,  # 1KB payload
                    },
                }

                start_time = time.time()
                await websocket.send(json.dumps(message))
                latency = time.time() - start_time

                self.latencies.append(latency)
                sent += 1

                if delay > 0:
                    await asyncio.sleep(delay)

            except Exception as e:
                logger.debug("Failed to send message %s: %s", i, e)
                failed += 1

        return sent, failed

    async def test_sustained_load(
        self, duration: int = 60, messages_per_second: int = 100
    ):
        """Test sustained message load."""
        logger.info("\nTesting sustained load: %s msg/s for %ss", messages_per_second, duration)

        # Start resource monitoring
        monitor_task = asyncio.create_task(self.monitor_resources())

        # Record start metrics
        self.metrics.memory_start = self.process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        try:
            async with websockets.connect(self.dashboard_url) as websocket:
                logger.info("Connected to dashboard")

                # Calculate message timing
                interval = 1.0 / messages_per_second
                total_messages = duration * messages_per_second

                # Send messages at specified rate
                sent, failed = await self.send_burst_messages(
                    websocket, total_messages, interval
                )

                self.metrics.messages_sent = sent
                self.metrics.messages_failed = failed

        except Exception as e:
            logger.exception("Connection error during sustained load: %s", e)
            self.metrics.connection_failures += 1

        # Stop monitoring
        self.monitoring = False
        await monitor_task

        # Calculate metrics
        self.metrics.test_duration = time.time() - start_time
        self.calculate_metrics()

        return self.metrics

    async def test_burst_load(
        self, burst_size: int = 1000, burst_count: int = 10, burst_interval: float = 5.0
    ):
        """Test burst message patterns."""
        logger.info("\nTesting burst load: %s bursts of %s messages", burst_count, burst_size)

        # Start resource monitoring
        monitor_task = asyncio.create_task(self.monitor_resources())

        self.metrics.memory_start = self.process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        try:
            async with websockets.connect(self.dashboard_url) as websocket:
                logger.info("Connected to dashboard")

                for burst_num in range(burst_count):
                    logger.info("Sending burst %s/%s", burst_num + 1, burst_count)

                    # Send burst with no delay between messages
                    sent, failed = await self.send_burst_messages(
                        websocket, burst_size, delay=0
                    )

                    self.metrics.messages_sent += sent
                    self.metrics.messages_failed += failed

                    # Wait between bursts
                    if burst_num < burst_count - 1:
                        await asyncio.sleep(burst_interval)

        except Exception as e:
            logger.exception("Connection error during burst load: %s", e)
            self.metrics.connection_failures += 1

        # Stop monitoring
        self.monitoring = False
        await monitor_task

        # Calculate metrics
        self.metrics.test_duration = time.time() - start_time
        self.calculate_metrics()

        return self.metrics

    async def test_concurrent_connections(
        self, connection_count: int = 10, messages_per_connection: int = 100
    ):
        """Test multiple concurrent connections."""
        logger.info("\nTesting %s concurrent connections", connection_count)

        # Start resource monitoring
        monitor_task = asyncio.create_task(self.monitor_resources())

        self.metrics.memory_start = self.process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        async def connection_worker(conn_id: int):
            """Worker for a single connection."""
            try:
                async with websockets.connect(self.dashboard_url) as websocket:
                    sent, failed = await self.send_burst_messages(
                        websocket, messages_per_connection, delay=0.01
                    )
                    return sent, failed
            except Exception as e:
                logger.exception("Connection %s failed: %s", conn_id, e)
                return 0, messages_per_connection

        # Run concurrent connections
        tasks = [connection_worker(i) for i in range(connection_count)]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        for sent, failed in results:
            self.metrics.messages_sent += sent
            self.metrics.messages_failed += failed

        # Stop monitoring
        self.monitoring = False
        await monitor_task

        # Calculate metrics
        self.metrics.test_duration = time.time() - start_time
        self.calculate_metrics()

        return self.metrics

    async def test_queue_overflow(self, overflow_factor: int = 2):
        """Test message queue overflow handling."""
        logger.info("\nTesting message queue overflow")

        # Get queue size from environment or use default
        queue_size = int(os.getenv("SYSTEM__WEBSOCKET_QUEUE_SIZE", "100"))
        messages_to_send = queue_size * overflow_factor

        logger.info("Queue size: %s, sending: %s messages", queue_size, messages_to_send)

        # Start resource monitoring
        monitor_task = asyncio.create_task(self.monitor_resources())

        self.metrics.memory_start = self.process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        # Create a connection but don't read messages (simulate slow consumer)
        try:
            async with websockets.connect(self.dashboard_url) as websocket:
                # Send messages rapidly without reading responses
                sent, failed = await self.send_burst_messages(
                    websocket, messages_to_send, delay=0
                )

                self.metrics.messages_sent = sent
                self.metrics.messages_failed = failed

                # Calculate how many should be queued
                self.metrics.messages_queued = min(queue_size, sent)

        except Exception as e:
            logger.exception("Connection error during overflow test: %s", e)
            self.metrics.connection_failures += 1

        # Stop monitoring
        self.monitoring = False
        await monitor_task

        # Calculate metrics
        self.metrics.test_duration = time.time() - start_time
        self.calculate_metrics()

        return self.metrics

    def calculate_metrics(self):
        """Calculate performance metrics."""
        # Latency metrics
        if self.latencies:
            self.metrics.min_latency = min(self.latencies)
            self.metrics.max_latency = max(self.latencies)
            self.metrics.avg_latency = np.mean(self.latencies)
            self.metrics.p95_latency = np.percentile(self.latencies, 95)
            self.metrics.p99_latency = np.percentile(self.latencies, 99)

        # Memory metrics
        if self.memory_samples:
            self.metrics.memory_peak = max(self.memory_samples)
            self.metrics.memory_end = self.memory_samples[-1]

        # CPU metrics
        if self.cpu_samples:
            self.metrics.cpu_avg = np.mean(self.cpu_samples)
            self.metrics.cpu_peak = max(self.cpu_samples)

        # Throughput
        if self.metrics.test_duration > 0:
            self.metrics.messages_per_second = (
                self.metrics.messages_sent / self.metrics.test_duration
            )

    def print_metrics(self, test_name: str):
        """Print test metrics."""
        logger.info("\n%s", '=' * 60)
        logger.info("%s - Results", test_name)
        logger.info("%s", '=' * 60)

        logger.info("\nMessage Statistics:")
        logger.info("  Messages Sent: %s", self.metrics.messages_sent:,)
        logger.info("  Messages Failed: %s", self.metrics.messages_failed:,)
        logger.info("  Success Rate: %s%", (self.metrics.messages_sent / (self.metrics.messages_sent + self.metrics.messages_failed) * 100):.1f)
        logger.info("  Throughput: %s msg/s", self.metrics.messages_per_second:.1f)

        logger.info("\nLatency Statistics (ms):")
        logger.info("  Min: %s", self.metrics.min_latency * 1000:.2f)
        logger.info("  Max: %s", self.metrics.max_latency * 1000:.2f)
        logger.info("  Avg: %s", self.metrics.avg_latency * 1000:.2f)
        logger.info("  P95: %s", self.metrics.p95_latency * 1000:.2f)
        logger.info("  P99: %s", self.metrics.p99_latency * 1000:.2f)

        logger.info("\nResource Usage:")
        logger.info("  Memory Start: %s MB", self.metrics.memory_start:.1f)
        logger.info("  Memory Peak: %s MB", self.metrics.memory_peak:.1f)
        logger.info("  Memory End: %s MB", self.metrics.memory_end:.1f)
        logger.info("  Memory Growth: %s MB", self.metrics.memory_end - self.metrics.memory_start:.1f)
        logger.info("  CPU Average: %s%", self.metrics.cpu_avg:.1f)
        logger.info("  CPU Peak: %s%", self.metrics.cpu_peak:.1f)

        logger.info("\nConnection Stats:")
        logger.info("  Connection Failures: %s", self.metrics.connection_failures)
        logger.info("  Test Duration: %ss", self.metrics.test_duration:.1f)

    async def run_all_load_tests(self):
        """Run all load tests."""
        logger.info("=" * 60)
        logger.info("WebSocket Load Test Suite")
        logger.info("=" * 60)

        all_results = {}

        # Test 1: Sustained moderate load
        logger.info("\n[Test 1/4] Sustained Moderate Load")
        self.reset_metrics()
        await self.test_sustained_load(duration=30, messages_per_second=50)
        self.print_metrics("Sustained Moderate Load")
        all_results["sustained_moderate"] = self.metrics_to_dict()

        # Wait between tests
        await asyncio.sleep(5)

        # Test 2: Burst load
        logger.info("\n[Test 2/4] Burst Load Pattern")
        self.reset_metrics()
        await self.test_burst_load(burst_size=500, burst_count=5, burst_interval=3)
        self.print_metrics("Burst Load Pattern")
        all_results["burst_load"] = self.metrics_to_dict()

        await asyncio.sleep(5)

        # Test 3: Concurrent connections
        logger.info("\n[Test 3/4] Concurrent Connections")
        self.reset_metrics()
        await self.test_concurrent_connections(
            connection_count=5, messages_per_connection=100
        )
        self.print_metrics("Concurrent Connections")
        all_results["concurrent_connections"] = self.metrics_to_dict()

        await asyncio.sleep(5)

        # Test 4: Queue overflow
        logger.info("\n[Test 4/4] Queue Overflow Handling")
        self.reset_metrics()
        await self.test_queue_overflow(overflow_factor=2)
        self.print_metrics("Queue Overflow Handling")
        all_results["queue_overflow"] = self.metrics_to_dict()

        # Save results
        self.save_results(all_results)

        # Print summary
        self.print_summary(all_results)

    def reset_metrics(self):
        """Reset metrics for new test."""
        self.metrics = LoadTestMetrics()
        self.latencies = []
        self.memory_samples = []
        self.cpu_samples = []

    def metrics_to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "messages_sent": self.metrics.messages_sent,
            "messages_failed": self.metrics.messages_failed,
            "success_rate": (
                self.metrics.messages_sent
                / (self.metrics.messages_sent + self.metrics.messages_failed)
                * 100
                if (self.metrics.messages_sent + self.metrics.messages_failed) > 0
                else 0
            ),
            "throughput": self.metrics.messages_per_second,
            "latency": {
                "min_ms": self.metrics.min_latency * 1000,
                "max_ms": self.metrics.max_latency * 1000,
                "avg_ms": self.metrics.avg_latency * 1000,
                "p95_ms": self.metrics.p95_latency * 1000,
                "p99_ms": self.metrics.p99_latency * 1000,
            },
            "memory": {
                "start_mb": self.metrics.memory_start,
                "peak_mb": self.metrics.memory_peak,
                "end_mb": self.metrics.memory_end,
                "growth_mb": self.metrics.memory_end - self.metrics.memory_start,
            },
            "cpu": {
                "avg_percent": self.metrics.cpu_avg,
                "peak_percent": self.metrics.cpu_peak,
            },
            "duration_seconds": self.metrics.test_duration,
        }

    def save_results(self, results: dict[str, Any]):
        """Save test results to file."""
        results_file = "tests/docker/results/load_test_results.json"

        with open(results_file, "w") as f:
            json.dump(
                {
                    "test_results": results,
                    "timestamp": datetime.utcnow().isoformat(),
                    "environment": {
                        "dashboard_url": self.dashboard_url,
                        "queue_size": os.getenv("SYSTEM__WEBSOCKET_QUEUE_SIZE", "100"),
                    },
                },
                f,
                indent=2,
            )

        logger.info("\nResults saved to: %s", results_file)

    def print_summary(self, results: dict[str, Any]):
        """Print test summary."""
        logger.info("\n" + "=" * 60)
        logger.info("Load Test Summary")
        logger.info("=" * 60)

        # Check for performance issues
        issues = []

        for test_name, metrics in results.items():
            logger.info("\n%s:", test_name)
            logger.info("  Success Rate: %s%", metrics['success_rate']:.1f)
            logger.info("  Throughput: %s msg/s", metrics['throughput']:.1f)
            logger.info("  Avg Latency: %s ms", metrics['latency']['avg_ms']:.2f)
            logger.info("  Memory Growth: %s MB", metrics['memory']['growth_mb']:.1f)

            # Check for issues
            if metrics["success_rate"] < 99:
                issues.append(
                    f"{test_name}: Low success rate ({metrics['success_rate']:.1f}%)"
                )

            if metrics["latency"]["p99_ms"] > 100:
                issues.append(
                    f"{test_name}: High P99 latency ({metrics['latency']['p99_ms']:.1f} ms)"
                )

            if metrics["memory"]["growth_mb"] > 50:
                issues.append(
                    f"{test_name}: High memory growth ({metrics['memory']['growth_mb']:.1f} MB)"
                )

        if issues:
            logger.warning("\nPerformance Issues Detected:")
            for issue in issues:
                logger.warning("  - %s", issue)
        else:
            logger.info("\n✅ All load tests passed with good performance!")

        # Exit code based on issues
        sys.exit(0 if not issues else 1)


async def main():
    """Main entry point."""
    import os

    # Get dashboard URL from environment
    dashboard_url = os.getenv(
        "SYSTEM__WEBSOCKET_DASHBOARD_URL", "ws://dashboard-backend:8000/ws"
    )

    tester = WebSocketLoadTester(dashboard_url)
    await tester.run_all_load_tests()


if __name__ == "__main__":
    # Check if numpy is available
    try:
        import numpy as np
    except ImportError:
        logger.exception("NumPy is required for load testing. Please install it:")
        logger.exception("  pip install numpy")
        sys.exit(1)

    asyncio.run(main())
