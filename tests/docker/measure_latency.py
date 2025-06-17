#!/usr/bin/env python3
"""
WebSocket Performance Monitoring
Measures end-to-end latency and tracks system performance metrics.
"""

import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import docker
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""

    timestamp: datetime
    event_type: str
    bot_time: float
    dashboard_time: float
    frontend_time: float | None = None
    total_latency: float | None = None

    def calculate_total(self):
        """Calculate total end-to-end latency."""
        if self.frontend_time:
            self.total_latency = self.frontend_time
        else:
            self.total_latency = self.dashboard_time


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""

    latency_measurements: list[LatencyMeasurement] = field(default_factory=list)
    container_stats: dict[str, list[dict]] = field(default_factory=dict)
    message_rates: dict[str, float] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)

    def add_latency(self, measurement: LatencyMeasurement):
        """Add a latency measurement."""
        measurement.calculate_total()
        self.latency_measurements.append(measurement)

    def get_latency_stats(self) -> dict[str, dict[str, float]]:
        """Calculate latency statistics by event type."""
        stats = {}

        # Group by event type
        by_type = {}
        for m in self.latency_measurements:
            if m.event_type not in by_type:
                by_type[m.event_type] = []
            if m.total_latency:
                by_type[m.event_type].append(m.total_latency)

        # Calculate stats for each type
        for event_type, latencies in by_type.items():
            if latencies:
                stats[event_type] = {
                    "count": len(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": statistics.mean(latencies),
                    "median": statistics.median(latencies),
                    "p95": (
                        statistics.quantiles(latencies, n=20)[18]
                        if len(latencies) > 5
                        else max(latencies)
                    ),
                    "p99": (
                        statistics.quantiles(latencies, n=100)[98]
                        if len(latencies) > 10
                        else max(latencies)
                    ),
                }

        return stats


class PerformanceMonitor:
    """Monitor WebSocket performance and system metrics."""

    def __init__(self, dashboard_url: str = "ws://dashboard-backend:8000/ws"):
        self.dashboard_url = dashboard_url
        self.metrics = PerformanceMetrics()
        self.docker_client = None
        self.monitoring = False
        self.message_timestamps: dict[str, datetime] = {}

        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker client not available: {e}")

    async def inject_test_event(self, event_type: str) -> str:
        """Inject a test event and return its ID."""
        event_id = f"{event_type}_{int(time.time() * 1000)}"

        # Record timestamp
        self.message_timestamps[event_id] = datetime.utcnow()

        # Send test event through bot
        if event_type == "market_data":
            # Simulate market data update
            await self._trigger_market_update(event_id)
        elif event_type == "ai_decision":
            # Simulate decision trigger
            await self._trigger_decision(event_id)
        elif event_type == "indicator_update":
            # Simulate indicator calculation
            await self._trigger_indicators(event_id)

        return event_id

    async def _trigger_market_update(self, event_id: str):
        """Trigger a market data update."""
        # Execute command in bot container to trigger update
        try:
            bot_container = self.docker_client.containers.get("ai-trading-bot")
            bot_container.exec_run(
                f"python -c \"import json; print(json.dumps({{'event_id': '{event_id}', 'type': 'market_update', 'price': 50000.0}}));\"",
                stream=False,
            )
            logger.debug(f"Triggered market update: {event_id}")
        except Exception as e:
            logger.error(f"Failed to trigger market update: {e}")

    async def _trigger_decision(self, event_id: str):
        """Trigger an AI decision."""
        try:
            bot_container = self.docker_client.containers.get("ai-trading-bot")
            bot_container.exec_run(
                f"python -c \"import json; print(json.dumps({{'event_id': '{event_id}', 'type': 'trigger_decision'}}));\"",
                stream=False,
            )
            logger.debug(f"Triggered AI decision: {event_id}")
        except Exception as e:
            logger.error(f"Failed to trigger decision: {e}")

    async def _trigger_indicators(self, event_id: str):
        """Trigger indicator calculation."""
        try:
            bot_container = self.docker_client.containers.get("ai-trading-bot")
            bot_container.exec_run(
                f"python -c \"import json; print(json.dumps({{'event_id': '{event_id}', 'type': 'calculate_indicators'}}));\"",
                stream=False,
            )
            logger.debug(f"Triggered indicators: {event_id}")
        except Exception as e:
            logger.error(f"Failed to trigger indicators: {e}")

    async def monitor_messages(self, duration: int = 60):
        """Monitor WebSocket messages and measure latency."""
        end_time = datetime.utcnow() + timedelta(seconds=duration)

        try:
            async with websockets.connect(self.dashboard_url) as websocket:
                logger.info("Connected to dashboard WebSocket")

                while datetime.utcnow() < end_time:
                    try:
                        # Receive message
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        receive_time = datetime.utcnow()

                        # Parse message
                        data = json.loads(message)

                        # Track message rate
                        msg_type = data.get("type", "unknown")
                        if msg_type not in self.metrics.message_rates:
                            self.metrics.message_rates[msg_type] = 0
                        self.metrics.message_rates[msg_type] += 1

                        # Check for test events
                        event_id = data.get("event_id")
                        if event_id and event_id in self.message_timestamps:
                            # Calculate latency
                            sent_time = self.message_timestamps[event_id]
                            bot_latency = (receive_time - sent_time).total_seconds()

                            measurement = LatencyMeasurement(
                                timestamp=receive_time,
                                event_type=msg_type,
                                bot_time=bot_latency,
                                dashboard_time=bot_latency,  # Same for now
                            )

                            self.metrics.add_latency(measurement)
                            logger.info(f"Latency for {event_id}: {bot_latency:.3f}s")

                            # Clean up
                            del self.message_timestamps[event_id]

                    except TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        self.metrics.error_counts["message_processing"] = (
                            self.metrics.error_counts.get("message_processing", 0) + 1
                        )

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.metrics.error_counts["connection"] = (
                self.metrics.error_counts.get("connection", 0) + 1
            )

    async def monitor_containers(self, interval: float = 5.0):
        """Monitor Docker container statistics."""
        if not self.docker_client:
            logger.warning("Docker monitoring not available")
            return

        self.monitoring = True
        containers = ["ai-trading-bot", "dashboard-backend", "dashboard-frontend"]

        while self.monitoring:
            try:
                for container_name in containers:
                    try:
                        container = self.docker_client.containers.get(container_name)
                        stats = container.stats(stream=False)

                        # Parse stats
                        cpu_percent = self._calculate_cpu_percent(stats)
                        memory_usage = (
                            stats["memory_stats"]["usage"] / 1024 / 1024
                        )  # MB
                        memory_limit = (
                            stats["memory_stats"]["limit"] / 1024 / 1024
                        )  # MB

                        # Store stats
                        if container_name not in self.metrics.container_stats:
                            self.metrics.container_stats[container_name] = []

                        self.metrics.container_stats[container_name].append(
                            {
                                "timestamp": datetime.utcnow().isoformat(),
                                "cpu_percent": cpu_percent,
                                "memory_mb": memory_usage,
                                "memory_limit_mb": memory_limit,
                                "memory_percent": (memory_usage / memory_limit) * 100,
                            }
                        )

                    except Exception as e:
                        logger.debug(f"Error getting stats for {container_name}: {e}")

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Container monitoring error: {e}")
                await asyncio.sleep(interval)

    def _calculate_cpu_percent(self, stats: dict) -> float:
        """Calculate CPU percentage from Docker stats."""
        try:
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"]
                - stats["precpu_stats"]["system_cpu_usage"]
            )

            if system_delta > 0:
                cpu_percent = (
                    (cpu_delta / system_delta)
                    * len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
                    * 100
                )
                return round(cpu_percent, 2)
        except Exception:
            pass

        return 0.0

    async def run_latency_test(self, duration: int = 60, test_interval: int = 10):
        """Run latency test with periodic test events."""
        logger.info(f"Starting latency test for {duration} seconds")

        # Start monitoring tasks
        monitor_task = asyncio.create_task(self.monitor_messages(duration))
        container_task = asyncio.create_task(self.monitor_containers())

        # Inject test events periodically
        event_types = ["market_data", "ai_decision", "indicator_update"]
        test_count = 0

        start_time = time.time()
        while time.time() - start_time < duration:
            # Inject test event
            event_type = event_types[test_count % len(event_types)]
            event_id = await self.inject_test_event(event_type)
            logger.info(f"Injected test event: {event_id}")

            test_count += 1
            await asyncio.sleep(test_interval)

        # Wait for monitoring to complete
        await monitor_task

        # Stop container monitoring
        self.monitoring = False
        await container_task

        # Calculate final metrics
        self.calculate_message_rates(duration)

        return self.metrics

    def calculate_message_rates(self, duration: float):
        """Calculate message rates per second."""
        for msg_type, count in self.metrics.message_rates.items():
            self.metrics.message_rates[msg_type] = count / duration

    def print_results(self):
        """Print performance test results."""
        logger.info("\n" + "=" * 60)
        logger.info("Performance Test Results")
        logger.info("=" * 60)

        # Latency statistics
        latency_stats = self.metrics.get_latency_stats()

        if latency_stats:
            logger.info("\nLatency Statistics (seconds):")
            for event_type, stats in latency_stats.items():
                logger.info(f"\n{event_type}:")
                logger.info(f"  Count: {stats['count']}")
                logger.info(f"  Min: {stats['min']:.3f}s")
                logger.info(f"  Max: {stats['max']:.3f}s")
                logger.info(f"  Mean: {stats['mean']:.3f}s")
                logger.info(f"  Median: {stats['median']:.3f}s")
                logger.info(f"  P95: {stats['p95']:.3f}s")
                logger.info(f"  P99: {stats['p99']:.3f}s")
        else:
            logger.warning("No latency measurements collected")

        # Message rates
        logger.info("\nMessage Rates (per second):")
        for msg_type, rate in sorted(self.metrics.message_rates.items()):
            logger.info(f"  {msg_type}: {rate:.2f} msg/s")

        # Container statistics
        if self.metrics.container_stats:
            logger.info("\nContainer Resource Usage:")
            for container, stats_list in self.metrics.container_stats.items():
                if stats_list:
                    cpu_values = [s["cpu_percent"] for s in stats_list]
                    memory_values = [s["memory_mb"] for s in stats_list]

                    logger.info(f"\n{container}:")
                    logger.info(
                        f"  CPU - Avg: {statistics.mean(cpu_values):.1f}%, Max: {max(cpu_values):.1f}%"
                    )
                    logger.info(
                        f"  Memory - Avg: {statistics.mean(memory_values):.1f}MB, Max: {max(memory_values):.1f}MB"
                    )

        # Error summary
        if self.metrics.error_counts:
            logger.info("\nErrors:")
            for error_type, count in self.metrics.error_counts.items():
                logger.info(f"  {error_type}: {count}")
        else:
            logger.info("\nNo errors detected ✓")

    def save_results(self, filename: str = "performance_results.json"):
        """Save results to JSON file."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "latency_stats": self.metrics.get_latency_stats(),
            "message_rates": self.metrics.message_rates,
            "container_stats_summary": self._summarize_container_stats(),
            "error_counts": self.metrics.error_counts,
            "raw_measurements": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "event_type": m.event_type,
                    "total_latency": m.total_latency,
                }
                for m in self.metrics.latency_measurements
            ],
        }

        filepath = f"tests/docker/results/{filename}"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {filepath}")

    def _summarize_container_stats(self) -> dict:
        """Summarize container statistics."""
        summary = {}

        for container, stats_list in self.metrics.container_stats.items():
            if stats_list:
                cpu_values = [s["cpu_percent"] for s in stats_list]
                memory_values = [s["memory_mb"] for s in stats_list]

                summary[container] = {
                    "cpu": {
                        "avg": statistics.mean(cpu_values),
                        "max": max(cpu_values),
                        "min": min(cpu_values),
                    },
                    "memory": {
                        "avg": statistics.mean(memory_values),
                        "max": max(memory_values),
                        "min": min(memory_values),
                    },
                }

        return summary


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket Performance Monitor")
    parser.add_argument(
        "--duration", type=int, default=60, help="Test duration in seconds"
    )
    parser.add_argument(
        "--interval", type=int, default=10, help="Test event interval in seconds"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://dashboard-backend:8000/ws",
        help="Dashboard WebSocket URL",
    )

    args = parser.parse_args()

    # Create monitor
    monitor = PerformanceMonitor(args.url)

    # Run test
    metrics = await monitor.run_latency_test(
        duration=args.duration, test_interval=args.interval
    )

    # Print results
    monitor.print_results()

    # Save results
    monitor.save_results()

    # Exit with appropriate code
    if metrics.error_counts:
        sys.exit(1)

    # Check latency thresholds
    latency_stats = metrics.get_latency_stats()
    for event_type, stats in latency_stats.items():
        if stats["p99"] > 1.0:  # 1 second threshold
            logger.warning(f"High P99 latency for {event_type}: {stats['p99']:.3f}s")
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
