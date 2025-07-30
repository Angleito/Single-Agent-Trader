#!/usr/bin/env python3
"""
Performance Monitoring Script for AI Trading Bot

Tracks comprehensive performance metrics and outputs in Prometheus format:
- Trading bot response times
- Order execution latency
- Memory usage patterns
- CPU utilization
- Network I/O metrics
- Database query performance

Design principles:
- Minimal overhead using efficient sampling techniques
- Ring buffers for time-series data
- Atomic operations for thread safety
- Prometheus exposition format for easy integration
"""

import asyncio
import logging
import mmap
import os
import struct
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psutil
from aiohttp import web
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # Response time buckets for histogram (in seconds)
    RESPONSE_TIME_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    )

    # Latency buckets for order execution (in milliseconds)
    LATENCY_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

    # Memory size buckets (in MB)
    MEMORY_BUCKETS = (50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000)

    # Network I/O buckets (in KB/s)
    NETWORK_BUCKETS = (10, 50, 100, 500, 1000, 5000, 10000, 50000)

    # Query time buckets (in milliseconds)
    QUERY_BUCKETS = (0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000)


class HighPerformanceMonitor:
    """High-performance monitoring system with minimal overhead."""

    def __init__(self, port: int = 9090):
        self.port = port
        self.settings = Settings()
        self.start_time = time.time()

        # Create custom registry for clean metrics
        self.registry = CollectorRegistry()

        # Initialize Prometheus metrics
        self._init_prometheus_metrics()

        # Performance tracking with ring buffers
        self._response_times: deque[tuple[float, float]] = deque(maxlen=10000)
        self._order_latencies: deque[tuple[float, float]] = deque(maxlen=10000)
        self._query_times: deque[tuple[str, float, float]] = deque(maxlen=10000)

        # Memory-mapped files for zero-copy metrics sharing
        self._init_shared_memory()

        # Network I/O tracking
        self._last_net_io = psutil.net_io_counters()
        self._last_net_time = time.time()

        # CPU tracking
        self._cpu_monitor_thread = None
        self._stop_monitoring = threading.Event()

        # Database query tracking
        self._query_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
            }
        )

    def _init_prometheus_metrics(self):
        """Initialize all Prometheus metrics."""
        # Trading bot response times
        self.response_time_histogram = Histogram(
            "trading_bot_response_seconds",
            "Trading bot response time in seconds",
            buckets=PerformanceMetrics.RESPONSE_TIME_BUCKETS,
            registry=self.registry,
        )

        self.response_time_summary = Summary(
            "trading_bot_response_summary_seconds",
            "Trading bot response time summary",
            registry=self.registry,
        )

        # Order execution latency
        self.order_latency_histogram = Histogram(
            "order_execution_latency_milliseconds",
            "Order execution latency in milliseconds",
            buckets=PerformanceMetrics.LATENCY_BUCKETS,
            registry=self.registry,
        )

        self.order_counter = Counter(
            "orders_total",
            "Total number of orders",
            ["action", "status"],
            registry=self.registry,
        )

        # Memory metrics
        self.memory_usage_gauge = Gauge(
            "memory_usage_bytes",
            "Current memory usage in bytes",
            ["type"],
            registry=self.registry,
        )

        self.memory_usage_histogram = Histogram(
            "memory_usage_mb_histogram",
            "Memory usage distribution in MB",
            buckets=PerformanceMetrics.MEMORY_BUCKETS,
            registry=self.registry,
        )

        # CPU metrics
        self.cpu_usage_gauge = Gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            ["core"],
            registry=self.registry,
        )

        self.cpu_usage_summary = Summary(
            "cpu_usage_summary_percent", "CPU usage summary", registry=self.registry
        )

        # Network I/O metrics
        self.network_io_gauge = Gauge(
            "network_io_bytes_per_second",
            "Network I/O in bytes per second",
            ["direction", "interface"],
            registry=self.registry,
        )

        self.network_packets_counter = Counter(
            "network_packets_total",
            "Total network packets",
            ["direction", "interface"],
            registry=self.registry,
        )

        # Database query metrics
        self.db_query_histogram = Histogram(
            "database_query_duration_milliseconds",
            "Database query duration in milliseconds",
            ["operation", "table"],
            buckets=PerformanceMetrics.QUERY_BUCKETS,
            registry=self.registry,
        )

        self.db_query_counter = Counter(
            "database_queries_total",
            "Total database queries",
            ["operation", "table", "status"],
            registry=self.registry,
        )

        # System health metrics
        self.uptime_gauge = Gauge(
            "bot_uptime_seconds", "Bot uptime in seconds", registry=self.registry
        )

        self.health_status_gauge = Gauge(
            "bot_health_status",
            "Overall bot health status (1=healthy, 0=unhealthy)",
            registry=self.registry,
        )

    def _init_shared_memory(self):
        """Initialize memory-mapped files for zero-copy metrics."""
        try:
            # Create shared memory directory
            shm_dir = Path("/dev/shm/trading_bot_metrics")
            shm_dir.mkdir(exist_ok=True)

            # Create memory-mapped file for real-time metrics
            self.shm_path = shm_dir / f"perf_metrics_{os.getpid()}.dat"
            self.shm_size = 4096  # 4KB should be enough

            # Initialize with zeros
            with open(self.shm_path, "wb") as f:
                f.write(b"\x00" * self.shm_size)

            # Memory map the file
            self.shm_fd = os.open(str(self.shm_path), os.O_RDWR)
            self.shm_data = mmap.mmap(self.shm_fd, self.shm_size)

        except Exception as e:
            logger.warning(f"Failed to initialize shared memory: {e}")
            self.shm_data = None

    @contextmanager
    def measure_response_time(self, operation: str = "trading_decision"):
        """Context manager to measure response times."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._response_times.append((time.time(), duration))
            self.response_time_histogram.observe(duration)
            self.response_time_summary.observe(duration)

            # Update shared memory if available
            if self.shm_data:
                try:
                    self.shm_data.seek(0)
                    self.shm_data.write(struct.pack("d", duration))
                except Exception:
                    pass

    @contextmanager
    def measure_order_latency(self, order_type: str):
        """Context manager to measure order execution latency."""
        start_time = time.time()
        status = "success"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._order_latencies.append((time.time(), latency_ms))
            self.order_latency_histogram.observe(latency_ms)
            self.order_counter.labels(action=order_type, status=status).inc()

    @contextmanager
    def measure_query_time(self, operation: str, table: str = "memory"):
        """Context manager to measure database query times."""
        start_time = time.time()
        status = "success"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._query_times.append((f"{operation}:{table}", time.time(), duration_ms))

            # Update statistics
            stats = self._query_stats[f"{operation}:{table}"]
            stats["count"] += 1
            stats["total_time"] += duration_ms
            stats["min_time"] = min(stats["min_time"], duration_ms)
            stats["max_time"] = max(stats["max_time"], duration_ms)

            # Update Prometheus metrics
            self.db_query_histogram.labels(operation=operation, table=table).observe(
                duration_ms
            )
            self.db_query_counter.labels(
                operation=operation, table=table, status=status
            ).inc()

    def start_monitoring(self):
        """Start all monitoring tasks."""
        # Start CPU monitoring thread
        self._cpu_monitor_thread = threading.Thread(target=self._monitor_cpu_thread)
        self._cpu_monitor_thread.daemon = True
        self._cpu_monitor_thread.start()

        # Start async monitoring tasks
        asyncio.create_task(self._monitor_memory_loop())
        asyncio.create_task(self._monitor_network_loop())
        asyncio.create_task(self._update_system_metrics_loop())

        # Start Prometheus metrics server
        asyncio.create_task(self._start_metrics_server())

        logger.info(f"Performance monitoring started on port {self.port}")

    def _monitor_cpu_thread(self):
        """Monitor CPU usage in a separate thread."""
        while not self._stop_monitoring.is_set():
            try:
                # Overall CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage_gauge.labels(core="all").set(cpu_percent)
                self.cpu_usage_summary.observe(cpu_percent)

                # Per-core CPU usage
                cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
                for i, percent in enumerate(cpu_percents):
                    self.cpu_usage_gauge.labels(core=f"core{i}").set(percent)

            except Exception as e:
                logger.error(f"Error monitoring CPU: {e}")

            time.sleep(5)  # Update every 5 seconds

    async def _monitor_memory_loop(self):
        """Monitor memory usage patterns."""
        while True:
            try:
                # Get memory info
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()

                # Process-specific memory
                process = psutil.Process()
                process_memory = process.memory_info()

                # Update gauges
                self.memory_usage_gauge.labels(type="system_total").set(memory.total)
                self.memory_usage_gauge.labels(type="system_used").set(memory.used)
                self.memory_usage_gauge.labels(type="system_available").set(
                    memory.available
                )
                self.memory_usage_gauge.labels(type="swap_used").set(swap.used)
                self.memory_usage_gauge.labels(type="process_rss").set(
                    process_memory.rss
                )
                self.memory_usage_gauge.labels(type="process_vms").set(
                    process_memory.vms
                )

                # Update histogram
                process_mb = process_memory.rss / (1024 * 1024)
                self.memory_usage_histogram.observe(process_mb)

                # Detect memory leaks
                if hasattr(self, "_last_memory_check"):
                    memory_growth = process_memory.rss - self._last_memory_check
                    if memory_growth > 100 * 1024 * 1024:  # 100MB growth
                        logger.warning(
                            f"Potential memory leak detected: {memory_growth / 1024 / 1024:.1f}MB growth"
                        )

                self._last_memory_check = process_memory.rss

            except Exception as e:
                logger.error(f"Error monitoring memory: {e}")

            await asyncio.sleep(10)  # Update every 10 seconds

    async def _monitor_network_loop(self):
        """Monitor network I/O metrics."""
        while True:
            try:
                current_time = time.time()
                net_io = psutil.net_io_counters(pernic=True)

                # Calculate rates
                time_delta = current_time - self._last_net_time

                for interface, stats in net_io.items():
                    if interface in ["lo", "lo0"]:  # Skip loopback
                        continue

                    # Calculate bytes per second
                    if (
                        hasattr(self, "_last_net_stats")
                        and interface in self._last_net_stats
                    ):
                        last_stats = self._last_net_stats[interface]

                        bytes_sent_per_sec = (
                            stats.bytes_sent - last_stats.bytes_sent
                        ) / time_delta
                        bytes_recv_per_sec = (
                            stats.bytes_recv - last_stats.bytes_recv
                        ) / time_delta
                        packets_sent_per_sec = (
                            stats.packets_sent - last_stats.packets_sent
                        ) / time_delta
                        packets_recv_per_sec = (
                            stats.packets_recv - last_stats.packets_recv
                        ) / time_delta

                        # Update gauges
                        self.network_io_gauge.labels(
                            direction="sent", interface=interface
                        ).set(bytes_sent_per_sec)
                        self.network_io_gauge.labels(
                            direction="recv", interface=interface
                        ).set(bytes_recv_per_sec)

                        # Update counters
                        self.network_packets_counter.labels(
                            direction="sent", interface=interface
                        )._value._value = stats.packets_sent
                        self.network_packets_counter.labels(
                            direction="recv", interface=interface
                        )._value._value = stats.packets_recv

                self._last_net_stats = net_io
                self._last_net_time = current_time

            except Exception as e:
                logger.error(f"Error monitoring network: {e}")

            await asyncio.sleep(5)  # Update every 5 seconds

    async def _update_system_metrics_loop(self):
        """Update general system metrics."""
        while True:
            try:
                # Update uptime
                uptime = time.time() - self.start_time
                self.uptime_gauge.set(uptime)

                # Calculate health status based on recent metrics
                health_score = self._calculate_health_score()
                self.health_status_gauge.set(health_score)

                # Export database query statistics
                for query_key, stats in self._query_stats.items():
                    if stats["count"] > 0:
                        avg_time = stats["total_time"] / stats["count"]
                        logger.debug(
                            f"Query {query_key}: avg={avg_time:.2f}ms, min={stats['min_time']:.2f}ms, max={stats['max_time']:.2f}ms"
                        )

            except Exception as e:
                logger.error(f"Error updating system metrics: {e}")

            await asyncio.sleep(30)  # Update every 30 seconds

    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-1)."""
        try:
            score = 1.0

            # Check response times
            if self._response_times:
                recent_times = [t[1] for t in list(self._response_times)[-100:]]
                avg_response = sum(recent_times) / len(recent_times)
                if avg_response > 5.0:  # >5 seconds is unhealthy
                    score *= 0.5
                elif avg_response > 2.0:  # >2 seconds is degraded
                    score *= 0.8

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                score *= 0.5
            elif memory.percent > 80:
                score *= 0.8

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                score *= 0.6
            elif cpu_percent > 80:
                score *= 0.9

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5  # Default to degraded if calculation fails

    async def _start_metrics_server(self):
        """Start the Prometheus metrics HTTP server."""
        app = web.Application()
        app.router.add_get("/metrics", self._handle_metrics)
        app.router.add_get("/health", self._handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()

        logger.info(
            f"Prometheus metrics available at http://0.0.0.0:{self.port}/metrics"
        )

    async def _handle_metrics(self, request):
        """Handle Prometheus metrics endpoint."""
        try:
            metrics_data = generate_latest(self.registry)
            return web.Response(
                body=metrics_data,
                content_type=CONTENT_TYPE_LATEST,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return web.Response(status=500, text=str(e))

    async def _handle_health(self, request):
        """Handle health check endpoint."""
        health_score = self._calculate_health_score()
        status = (
            "healthy"
            if health_score > 0.7
            else "degraded"
            if health_score > 0.3
            else "unhealthy"
        )

        health_data = {
            "status": status,
            "score": health_score,
            "uptime_seconds": time.time() - self.start_time,
            "metrics": {
                "response_times_tracked": len(self._response_times),
                "orders_tracked": len(self._order_latencies),
                "queries_tracked": sum(s["count"] for s in self._query_stats.values()),
            },
        }

        return web.json_response(health_data)

    def cleanup(self):
        """Clean up resources."""
        self._stop_monitoring.set()

        if hasattr(self, "shm_data") and self.shm_data:
            self.shm_data.close()
            os.close(self.shm_fd)
            try:
                os.unlink(self.shm_path)
            except Exception:
                pass

        logger.info("Performance monitor cleaned up")


# Global monitor instance
_monitor: HighPerformanceMonitor | None = None


def get_monitor() -> HighPerformanceMonitor:
    """Get or create the global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = HighPerformanceMonitor()
    return _monitor


# Convenience decorators for easy integration
def monitor_response_time(operation: str = "trading_decision"):
    """Decorator to monitor function response time."""

    def decorator(func):
        def sync_wrapper(*args, **kwargs):
            with get_monitor().measure_response_time(operation):
                return func(*args, **kwargs)

        async def async_wrapper(*args, **kwargs):
            with get_monitor().measure_response_time(operation):
                return await func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def monitor_order_execution(order_type: str):
    """Decorator to monitor order execution latency."""

    def decorator(func):
        def sync_wrapper(*args, **kwargs):
            with get_monitor().measure_order_latency(order_type):
                return func(*args, **kwargs)

        async def async_wrapper(*args, **kwargs):
            with get_monitor().measure_order_latency(order_type):
                return await func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def monitor_db_query(operation: str, table: str = "memory"):
    """Decorator to monitor database query performance."""

    def decorator(func):
        def sync_wrapper(*args, **kwargs):
            with get_monitor().measure_query_time(operation, table):
                return func(*args, **kwargs)

        async def async_wrapper(*args, **kwargs):
            with get_monitor().measure_query_time(operation, table):
                return await func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


async def main():
    """Main entry point for standalone monitoring."""
    monitor = get_monitor()
    monitor.start_monitoring()

    logger.info("Performance monitoring started. Press Ctrl+C to stop.")
    logger.info(f"Metrics available at: http://localhost:{monitor.port}/metrics")
    logger.info(f"Health check at: http://localhost:{monitor.port}/health")

    try:
        # Keep running
        while True:
            await asyncio.sleep(60)

            # Log a summary every minute
            logger.info(
                f"Monitoring active - Uptime: {time.time() - monitor.start_time:.0f}s"
            )

    except KeyboardInterrupt:
        logger.info("Shutting down performance monitor...")
        monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
