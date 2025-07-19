#!/usr/bin/env python3
"""
Comprehensive Stress Test Suite for Low-Resource Environment

This script conducts thorough stress testing of both the trading bot and Bluefin service
in a resource-constrained environment to verify stability and responsiveness under load.

Features:
- Memory-constrained environment simulation
- Concurrent load testing
- Resource monitoring and alerting
- Stability verification
- Performance degradation detection
- Recovery testing
"""

import asyncio
import json
import logging
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import psutil
import websockets

import docker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("stress_test.log")],
)
logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    timestamp: datetime
    memory_mb: float
    cpu_percent: float
    disk_io_read_mb: float = 0
    disk_io_write_mb: float = 0
    network_io_sent_mb: float = 0
    network_io_recv_mb: float = 0


@dataclass
class ServiceHealth:
    """Service health status."""

    service_name: str
    is_healthy: bool
    response_time_ms: float
    error_message: str | None = None
    additional_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Results of a stress test."""

    test_name: str
    description: str
    duration_seconds: float
    success: bool
    operations_completed: int
    operations_failed: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_memory_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    error_rate_percent: float
    resource_metrics: list[ResourceMetrics] = field(default_factory=list)
    service_health_history: list[ServiceHealth] = field(default_factory=list)
    additional_metrics: dict[str, Any] = field(default_factory=dict)


class LowResourceStressTester:
    """
    Comprehensive stress tester for low-resource environments.
    """

    def __init__(self, config_file: str | None = None):
        """Initialize the stress tester."""
        self.config = self._load_config(config_file)
        self.docker_client = docker.from_env()
        self.process = psutil.Process()
        self.start_time = time.time()

        # Resource monitoring
        self.resource_metrics: list[ResourceMetrics] = []
        self.monitoring_active = False

        # Test results
        self.test_results: list[StressTestResult] = []

        # Service endpoints
        self.bluefin_endpoint = self.config.get(
            "bluefin_endpoint", "http://localhost:8082"
        )
        self.dashboard_endpoint = self.config.get(
            "dashboard_endpoint", "http://localhost:8000"
        )
        self.websocket_endpoint = self.config.get(
            "websocket_endpoint", "ws://localhost:8000/ws"
        )

        # Memory limits for low-resource testing
        self.memory_limit_mb = self.config.get("memory_limit_mb", 512)
        self.cpu_limit_percent = self.config.get("cpu_limit_percent", 80)

    def _load_config(self, config_file: str | None) -> dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "test_duration_seconds": 300,  # 5 minutes
            "concurrent_users": 3,  # Low for resource constraints
            "operations_per_second": 5,
            "memory_limit_mb": 512,
            "cpu_limit_percent": 80,
            "resource_check_interval": 5,
            "stress_test_cycles": 3,
            "recovery_time_seconds": 30,
        }

        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                file_config = json.load(f)
                default_config.update(file_config)

        return default_config

    async def start_resource_monitoring(self):
        """Start continuous resource monitoring."""
        self.monitoring_active = True
        logger.info("Starting resource monitoring...")

        while self.monitoring_active:
            try:
                # Collect system metrics
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024

                cpu_percent = self.process.cpu_percent(interval=0.1)

                # Collect I/O metrics
                io_counters = psutil.disk_io_counters()
                network_counters = psutil.net_io_counters()

                metric = ResourceMetrics(
                    timestamp=datetime.now(UTC),
                    memory_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    disk_io_read_mb=(
                        io_counters.read_bytes / 1024 / 1024 if io_counters else 0
                    ),
                    disk_io_write_mb=(
                        io_counters.write_bytes / 1024 / 1024 if io_counters else 0
                    ),
                    network_io_sent_mb=(
                        network_counters.bytes_sent / 1024 / 1024
                        if network_counters
                        else 0
                    ),
                    network_io_recv_mb=(
                        network_counters.bytes_recv / 1024 / 1024
                        if network_counters
                        else 0
                    ),
                )

                self.resource_metrics.append(metric)

                # Check for resource limit violations
                if memory_mb > self.memory_limit_mb:
                    logger.warning(
                        f"Memory limit exceeded: {memory_mb:.1f}MB > {self.memory_limit_mb}MB"
                    )

                if cpu_percent > self.cpu_limit_percent:
                    logger.warning(
                        f"CPU limit exceeded: {cpu_percent:.1f}% > {self.cpu_limit_percent}%"
                    )

                await asyncio.sleep(self.config["resource_check_interval"])

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(5)

    def stop_resource_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        logger.info("Stopped resource monitoring")

    async def check_service_health(
        self, service_name: str, endpoint: str
    ) -> ServiceHealth:
        """Check health of a specific service."""
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                health_url = f"{endpoint}/health"
                async with session.get(health_url) as response:
                    response_time = (time.time() - start_time) * 1000

                    if response.status == 200:
                        health_data = await response.json()
                        return ServiceHealth(
                            service_name=service_name,
                            is_healthy=True,
                            response_time_ms=response_time,
                            additional_info=health_data,
                        )
                    return ServiceHealth(
                        service_name=service_name,
                        is_healthy=False,
                        response_time_ms=response_time,
                        error_message=f"HTTP {response.status}",
                    )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ServiceHealth(
                service_name=service_name,
                is_healthy=False,
                response_time_ms=response_time,
                error_message=str(e),
            )

    async def test_bluefin_service_load(self) -> StressTestResult:
        """Test Bluefin service under load."""
        logger.info("Starting Bluefin service load test...")

        start_time = time.time()
        response_times = []
        operations_completed = 0
        operations_failed = 0
        service_health_history = []

        # Start resource monitoring
        monitor_task = asyncio.create_task(self.start_resource_monitoring())

        try:
            # Test duration
            test_duration = self.config["test_duration_seconds"]
            operations_per_second = self.config["operations_per_second"]

            async def bluefin_operation():
                """Single Bluefin API operation."""
                operation_start = time.time()

                try:
                    async with aiohttp.ClientSession() as session:
                        # Test different endpoints
                        endpoints = [
                            f"{self.bluefin_endpoint}/health",
                            f"{self.bluefin_endpoint}/account/balance",
                            f"{self.bluefin_endpoint}/market/info",
                        ]

                        endpoint = random.choice(endpoints)
                        async with session.get(
                            endpoint, timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            await response.text()  # Consume response

                        operation_time = (time.time() - operation_start) * 1000
                        return operation_time

                except Exception as e:
                    logger.debug(f"Bluefin operation failed: {e}")
                    raise

            # Execute load test
            end_time = start_time + test_duration

            while time.time() < end_time:
                # Create batch of concurrent operations
                batch_size = min(
                    self.config["concurrent_users"], 3
                )  # Limit for low resources
                tasks = []

                for _ in range(batch_size):
                    task = bluefin_operation()
                    tasks.append(task)

                # Execute batch
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        if isinstance(result, Exception):
                            operations_failed += 1
                        else:
                            response_times.append(result)
                            operations_completed += 1

                    # Check service health periodically
                    if operations_completed % 10 == 0:
                        health = await self.check_service_health(
                            "bluefin-service", self.bluefin_endpoint
                        )
                        service_health_history.append(health)

                    # Rate limiting for low resources
                    await asyncio.sleep(1.0 / operations_per_second)

                except Exception as e:
                    logger.error(f"Batch execution error: {e}")
                    operations_failed += batch_size

        finally:
            self.stop_resource_monitoring()
            await monitor_task

        # Calculate metrics
        duration = time.time() - start_time
        error_rate = (
            operations_failed / max(operations_completed + operations_failed, 1)
        ) * 100

        # Response time statistics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = (
            statistics.quantiles(response_times, n=20)[18]
            if len(response_times) > 20
            else avg_response_time
        )
        p99_response_time = (
            statistics.quantiles(response_times, n=100)[98]
            if len(response_times) > 100
            else avg_response_time
        )

        # Resource statistics
        memory_values = [m.memory_mb for m in self.resource_metrics]
        cpu_values = [m.cpu_percent for m in self.resource_metrics]

        result = StressTestResult(
            test_name="Bluefin Service Load Test",
            description="High-frequency API calls to Bluefin service under resource constraints",
            duration_seconds=duration,
            success=error_rate < 10,  # Success if error rate < 10%
            operations_completed=operations_completed,
            operations_failed=operations_failed,
            avg_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            max_memory_mb=max(memory_values) if memory_values else 0,
            avg_cpu_percent=statistics.mean(cpu_values) if cpu_values else 0,
            peak_cpu_percent=max(cpu_values) if cpu_values else 0,
            error_rate_percent=error_rate,
            resource_metrics=self.resource_metrics.copy(),
            service_health_history=service_health_history,
            additional_metrics={
                "target_ops_per_second": operations_per_second,
                "actual_ops_per_second": operations_completed / duration,
                "concurrent_users": self.config["concurrent_users"],
            },
        )

        self.test_results.append(result)
        logger.info(
            f"Bluefin load test completed: {operations_completed} ops, {error_rate:.1f}% error rate"
        )
        return result

    async def test_trading_bot_websocket_load(self) -> StressTestResult:
        """Test trading bot WebSocket connections under load."""
        logger.info("Starting trading bot WebSocket load test...")

        start_time = time.time()
        response_times = []
        operations_completed = 0
        operations_failed = 0
        service_health_history = []

        # Reset resource monitoring
        self.resource_metrics.clear()
        monitor_task = asyncio.create_task(self.start_resource_monitoring())

        try:
            test_duration = self.config["test_duration_seconds"]

            async def websocket_operation():
                """Single WebSocket operation."""
                operation_start = time.time()

                try:
                    async with websockets.connect(
                        self.websocket_endpoint,
                        timeout=10,
                        ping_interval=None,  # Disable ping for performance
                    ) as websocket:
                        # Send test message
                        message = {
                            "type": "stress_test",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "data": {"test_payload": "x" * 512},  # 512 bytes
                        }

                        await websocket.send(json.dumps(message))

                        # Wait for response (with timeout)
                        try:
                            response = await asyncio.wait_for(
                                websocket.recv(), timeout=5.0
                            )
                            operation_time = (time.time() - operation_start) * 1000
                            return operation_time
                        except TimeoutError:
                            raise Exception("WebSocket response timeout")

                except Exception as e:
                    logger.debug(f"WebSocket operation failed: {e}")
                    raise

            # Execute load test with multiple connections
            end_time = start_time + test_duration
            connection_count = min(
                self.config["concurrent_users"], 2
            )  # Limit for WebSocket

            async def connection_worker():
                """Worker for sustained WebSocket operations."""
                connection_ops = 0
                connection_failures = 0

                while time.time() < end_time:
                    try:
                        response_time = await websocket_operation()
                        response_times.append(response_time)
                        connection_ops += 1

                        # Rate limiting
                        await asyncio.sleep(0.5)  # 2 ops per second per connection

                    except Exception:
                        connection_failures += 1
                        await asyncio.sleep(1)  # Longer delay on failure

                return connection_ops, connection_failures

            # Run concurrent connections
            worker_tasks = [connection_worker() for _ in range(connection_count)]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            # Aggregate results
            for result in results:
                if isinstance(result, Exception):
                    operations_failed += 10  # Estimate
                else:
                    ops, failures = result
                    operations_completed += ops
                    operations_failed += failures

            # Periodic health checks
            for i in range(0, int(test_duration), 30):
                await asyncio.sleep(min(30, test_duration - i))
                health = await self.check_service_health(
                    "dashboard-backend", self.dashboard_endpoint
                )
                service_health_history.append(health)

        finally:
            self.stop_resource_monitoring()
            await monitor_task

        # Calculate metrics
        duration = time.time() - start_time
        error_rate = (
            operations_failed / max(operations_completed + operations_failed, 1)
        ) * 100

        # Response time statistics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = (
            statistics.quantiles(response_times, n=20)[18]
            if len(response_times) > 20
            else avg_response_time
        )
        p99_response_time = (
            statistics.quantiles(response_times, n=100)[98]
            if len(response_times) > 100
            else avg_response_time
        )

        # Resource statistics
        memory_values = [m.memory_mb for m in self.resource_metrics]
        cpu_values = [m.cpu_percent for m in self.resource_metrics]

        result = StressTestResult(
            test_name="Trading Bot WebSocket Load Test",
            description="Multiple WebSocket connections with sustained message load",
            duration_seconds=duration,
            success=error_rate < 15,  # More lenient for WebSocket
            operations_completed=operations_completed,
            operations_failed=operations_failed,
            avg_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            max_memory_mb=max(memory_values) if memory_values else 0,
            avg_cpu_percent=statistics.mean(cpu_values) if cpu_values else 0,
            peak_cpu_percent=max(cpu_values) if cpu_values else 0,
            error_rate_percent=error_rate,
            resource_metrics=self.resource_metrics.copy(),
            service_health_history=service_health_history,
            additional_metrics={
                "concurrent_connections": connection_count,
                "messages_per_connection": (
                    operations_completed // connection_count
                    if connection_count > 0
                    else 0
                ),
            },
        )

        self.test_results.append(result)
        logger.info(
            f"WebSocket load test completed: {operations_completed} ops, {error_rate:.1f}% error rate"
        )
        return result

    async def test_memory_pressure_stability(self) -> StressTestResult:
        """Test system stability under memory pressure."""
        logger.info("Starting memory pressure stability test...")

        start_time = time.time()
        operations_completed = 0
        operations_failed = 0
        service_health_history = []
        memory_allocations = []

        # Reset resource monitoring
        self.resource_metrics.clear()
        monitor_task = asyncio.create_task(self.start_resource_monitoring())

        try:
            test_duration = (
                self.config["test_duration_seconds"] // 2
            )  # Shorter for memory test

            # Gradually increase memory pressure
            memory_pressure_mb = 0
            max_pressure_mb = min(
                200, self.memory_limit_mb // 2
            )  # Don't exceed half the limit

            end_time = start_time + test_duration

            while time.time() < end_time:
                # Increase memory pressure gradually
                if memory_pressure_mb < max_pressure_mb:
                    # Allocate 10MB chunks
                    chunk = bytearray(10 * 1024 * 1024)  # 10MB
                    memory_allocations.append(chunk)
                    memory_pressure_mb += 10

                    logger.info(f"Memory pressure: {memory_pressure_mb}MB")

                # Test service responsiveness under pressure
                try:
                    # Test Bluefin service
                    bluefin_health = await self.check_service_health(
                        "bluefin-service", self.bluefin_endpoint
                    )
                    service_health_history.append(bluefin_health)

                    # Test Dashboard
                    dashboard_health = await self.check_service_health(
                        "dashboard-backend", self.dashboard_endpoint
                    )
                    service_health_history.append(dashboard_health)

                    if bluefin_health.is_healthy and dashboard_health.is_healthy:
                        operations_completed += 1
                    else:
                        operations_failed += 1
                        logger.warning(
                            f"Service health check failed under memory pressure: {memory_pressure_mb}MB"
                        )

                except Exception as e:
                    operations_failed += 1
                    logger.error(f"Health check error under memory pressure: {e}")

                await asyncio.sleep(5)  # Check every 5 seconds

            # Gradually release memory pressure
            logger.info("Releasing memory pressure...")
            for i in range(len(memory_allocations) // 2):
                del memory_allocations[0]
                if i % 5 == 0:  # Check health during recovery
                    recovery_health = await self.check_service_health(
                        "bluefin-service", self.bluefin_endpoint
                    )
                    service_health_history.append(recovery_health)

        finally:
            # Clean up memory allocations
            memory_allocations.clear()
            self.stop_resource_monitoring()
            await monitor_task

        duration = time.time() - start_time
        error_rate = (
            operations_failed / max(operations_completed + operations_failed, 1)
        ) * 100

        # Resource statistics
        memory_values = [m.memory_mb for m in self.resource_metrics]
        cpu_values = [m.cpu_percent for m in self.resource_metrics]

        result = StressTestResult(
            test_name="Memory Pressure Stability Test",
            description="Service stability under increasing memory pressure",
            duration_seconds=duration,
            success=error_rate < 20,  # More lenient for pressure test
            operations_completed=operations_completed,
            operations_failed=operations_failed,
            avg_response_time_ms=0,  # Not measuring response times here
            p95_response_time_ms=0,
            p99_response_time_ms=0,
            max_memory_mb=max(memory_values) if memory_values else 0,
            avg_cpu_percent=statistics.mean(cpu_values) if cpu_values else 0,
            peak_cpu_percent=max(cpu_values) if cpu_values else 0,
            error_rate_percent=error_rate,
            resource_metrics=self.resource_metrics.copy(),
            service_health_history=service_health_history,
            additional_metrics={
                "max_memory_pressure_mb": max_pressure_mb,
                "memory_growth_rate": (
                    (max(memory_values) - min(memory_values)) / duration
                    if memory_values
                    else 0
                ),
            },
        )

        self.test_results.append(result)
        logger.info(f"Memory pressure test completed: {error_rate:.1f}% error rate")
        return result

    async def test_recovery_after_stress(self) -> StressTestResult:
        """Test system recovery after stress."""
        logger.info("Starting recovery test...")

        recovery_start = time.time()
        recovery_time = self.config["recovery_time_seconds"]

        # Wait for recovery period
        await asyncio.sleep(recovery_time)

        # Test service responsiveness after recovery
        post_recovery_health = []

        for _ in range(5):  # Multiple checks
            bluefin_health = await self.check_service_health(
                "bluefin-service", self.bluefin_endpoint
            )
            dashboard_health = await self.check_service_health(
                "dashboard-backend", self.dashboard_endpoint
            )

            post_recovery_health.extend([bluefin_health, dashboard_health])
            await asyncio.sleep(2)

        # Check if services recovered
        healthy_checks = sum(1 for h in post_recovery_health if h.is_healthy)
        total_checks = len(post_recovery_health)
        recovery_rate = (healthy_checks / total_checks) * 100

        duration = time.time() - recovery_start

        result = StressTestResult(
            test_name="Post-Stress Recovery Test",
            description="Service recovery and stability after stress testing",
            duration_seconds=duration,
            success=recovery_rate > 80,  # 80% of health checks should pass
            operations_completed=healthy_checks,
            operations_failed=total_checks - healthy_checks,
            avg_response_time_ms=statistics.mean(
                [h.response_time_ms for h in post_recovery_health]
            ),
            p95_response_time_ms=0,
            p99_response_time_ms=0,
            max_memory_mb=0,
            avg_cpu_percent=0,
            peak_cpu_percent=0,
            error_rate_percent=100 - recovery_rate,
            service_health_history=post_recovery_health,
            additional_metrics={
                "recovery_rate_percent": recovery_rate,
                "recovery_time_seconds": recovery_time,
            },
        )

        self.test_results.append(result)
        logger.info(f"Recovery test completed: {recovery_rate:.1f}% recovery rate")
        return result

    def check_docker_containers(self) -> dict[str, Any]:
        """Check Docker container health."""
        container_status = {}

        try:
            containers = self.docker_client.containers.list(all=True)

            for container in containers:
                if any(
                    name in container.name
                    for name in ["ai-trading-bot", "bluefin-service", "dashboard"]
                ):
                    status_info = {
                        "name": container.name,
                        "status": container.status,
                        "image": (
                            container.image.tags[0]
                            if container.image.tags
                            else "unknown"
                        ),
                        "created": container.attrs["Created"],
                        "health": "unknown",
                    }

                    # Get health check status if available
                    if "Health" in container.attrs.get("State", {}):
                        health_info = container.attrs["State"]["Health"]
                        status_info["health"] = health_info.get("Status", "unknown")

                    # Get resource usage
                    try:
                        stats = container.stats(stream=False)
                        memory_usage = (
                            stats["memory_stats"].get("usage", 0) / 1024 / 1024
                        )  # MB
                        cpu_percent = 0

                        if "cpu_stats" in stats and "precpu_stats" in stats:
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
                                    * len(
                                        stats["cpu_stats"]["cpu_usage"]["percpu_usage"]
                                    )
                                    * 100
                                )

                        status_info["memory_mb"] = memory_usage
                        status_info["cpu_percent"] = cpu_percent

                    except Exception as e:
                        logger.debug(f"Could not get stats for {container.name}: {e}")

                    container_status[container.name] = status_info

        except Exception as e:
            logger.error(f"Error checking Docker containers: {e}")

        return container_status

    async def run_comprehensive_stress_test(self) -> dict[str, Any]:
        """Run the complete stress test suite."""
        logger.info(
            "Starting comprehensive stress test suite for low-resource environment"
        )

        suite_start_time = time.time()

        # Pre-test checks
        logger.info("Performing pre-test system checks...")
        initial_container_status = self.check_docker_containers()

        # Test sequence
        test_sequence = [
            ("Bluefin Service Load", self.test_bluefin_service_load),
            ("WebSocket Load", self.test_trading_bot_websocket_load),
            ("Memory Pressure", self.test_memory_pressure_stability),
            ("Recovery", self.test_recovery_after_stress),
        ]

        # Execute tests
        for test_name, test_func in test_sequence:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'=' * 60}")

            try:
                await test_func()
                logger.info(f"✅ {test_name} completed")
            except Exception as e:
                logger.error(f"❌ {test_name} failed: {e}")
                # Create failure result
                failure_result = StressTestResult(
                    test_name=test_name,
                    description=f"Test failed with error: {e!s}",
                    duration_seconds=0,
                    success=False,
                    operations_completed=0,
                    operations_failed=1,
                    avg_response_time_ms=0,
                    p95_response_time_ms=0,
                    p99_response_time_ms=0,
                    max_memory_mb=0,
                    avg_cpu_percent=0,
                    peak_cpu_percent=0,
                    error_rate_percent=100,
                    additional_metrics={"error": str(e)},
                )
                self.test_results.append(failure_result)

            # Brief pause between tests
            await asyncio.sleep(10)

        # Post-test checks
        logger.info("Performing post-test system checks...")
        final_container_status = self.check_docker_containers()

        suite_duration = time.time() - suite_start_time

        # Generate comprehensive report
        report = {
            "test_suite": "Low-Resource Environment Stress Test",
            "timestamp": datetime.now(UTC).isoformat(),
            "duration_seconds": suite_duration,
            "total_tests": len(self.test_results),
            "successful_tests": sum(1 for r in self.test_results if r.success),
            "failed_tests": sum(1 for r in self.test_results if not r.success),
            "initial_container_status": initial_container_status,
            "final_container_status": final_container_status,
            "test_results": [self._result_to_dict(r) for r in self.test_results],
            "summary": self._generate_summary(),
        }

        return report

    def _result_to_dict(self, result: StressTestResult) -> dict[str, Any]:
        """Convert StressTestResult to dictionary."""
        return {
            "test_name": result.test_name,
            "description": result.description,
            "duration_seconds": result.duration_seconds,
            "success": result.success,
            "operations_completed": result.operations_completed,
            "operations_failed": result.operations_failed,
            "avg_response_time_ms": result.avg_response_time_ms,
            "p95_response_time_ms": result.p95_response_time_ms,
            "p99_response_time_ms": result.p99_response_time_ms,
            "max_memory_mb": result.max_memory_mb,
            "avg_cpu_percent": result.avg_cpu_percent,
            "peak_cpu_percent": result.peak_cpu_percent,
            "error_rate_percent": result.error_rate_percent,
            "additional_metrics": result.additional_metrics,
            "service_health_summary": {
                "total_checks": len(result.service_health_history),
                "healthy_checks": sum(
                    1 for h in result.service_health_history if h.is_healthy
                ),
                "avg_response_time_ms": (
                    statistics.mean(
                        [h.response_time_ms for h in result.service_health_history]
                    )
                    if result.service_health_history
                    else 0
                ),
            },
        }

    def _generate_summary(self) -> dict[str, Any]:
        """Generate test suite summary."""
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]

        all_response_times = []
        all_memory_values = []
        all_cpu_values = []

        for result in self.test_results:
            if result.avg_response_time_ms > 0:
                all_response_times.append(result.avg_response_time_ms)
            if result.max_memory_mb > 0:
                all_memory_values.append(result.max_memory_mb)
            if result.avg_cpu_percent > 0:
                all_cpu_values.append(result.avg_cpu_percent)

        return {
            "overall_success": len(failed_tests) == 0,
            "success_rate_percent": (
                (len(successful_tests) / len(self.test_results)) * 100
                if self.test_results
                else 0
            ),
            "avg_response_time_ms": (
                statistics.mean(all_response_times) if all_response_times else 0
            ),
            "max_memory_mb": max(all_memory_values) if all_memory_values else 0,
            "avg_cpu_percent": statistics.mean(all_cpu_values) if all_cpu_values else 0,
            "total_operations": sum(r.operations_completed for r in self.test_results),
            "total_failures": sum(r.operations_failed for r in self.test_results),
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        for result in self.test_results:
            if not result.success:
                recommendations.append(f"❌ {result.test_name}: {result.description}")

            if result.error_rate_percent > 10:
                recommendations.append(
                    f"⚠️ High error rate in {result.test_name}: {result.error_rate_percent:.1f}%"
                )

            if result.avg_response_time_ms > 1000:
                recommendations.append(
                    f"🐌 Slow response times in {result.test_name}: {result.avg_response_time_ms:.1f}ms avg"
                )

            if result.max_memory_mb > self.memory_limit_mb * 0.9:
                recommendations.append(
                    f"💾 High memory usage in {result.test_name}: {result.max_memory_mb:.1f}MB"
                )

            if result.peak_cpu_percent > self.cpu_limit_percent:
                recommendations.append(
                    f"🔥 High CPU usage in {result.test_name}: {result.peak_cpu_percent:.1f}%"
                )

        if not recommendations:
            recommendations.append(
                "✅ All tests passed successfully with good performance metrics"
            )
            recommendations.append(
                "🚀 System is stable and responsive under low-resource constraints"
            )

        return recommendations

    def save_report(self, report: dict[str, Any], filename: str = None):
        """Save the test report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stress_test_report_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Test report saved to: {filename}")
        return filename


async def main():
    """Main execution function."""
    print("🚀 AI Trading Bot - Low-Resource Stress Test Suite")
    print("=" * 60)

    # Check if running in appropriate environment
    if os.getenv("STRESS_TEST_ENV") != "low_resource":
        logger.warning("Environment variable STRESS_TEST_ENV not set to 'low_resource'")
        logger.warning("This test is designed for resource-constrained environments")

    # Initialize tester
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    tester = LowResourceStressTester(config_file)

    try:
        # Run comprehensive test suite
        report = await tester.run_comprehensive_stress_test()

        # Save report
        report_file = tester.save_report(report)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 STRESS TEST RESULTS SUMMARY")
        print("=" * 60)

        summary = report["summary"]
        print(f"✅ Overall Success: {summary['overall_success']}")
        print(f"📈 Success Rate: {summary['success_rate_percent']:.1f}%")
        print(f"⏱️ Avg Response Time: {summary['avg_response_time_ms']:.1f}ms")
        print(f"💾 Max Memory Usage: {summary['max_memory_mb']:.1f}MB")
        print(f"🔥 Avg CPU Usage: {summary['avg_cpu_percent']:.1f}%")
        print(f"🔢 Total Operations: {summary['total_operations']}")
        print(f"❌ Total Failures: {summary['total_failures']}")

        print("\n📝 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"  {rec}")

        print(f"\n📋 Full report saved to: {report_file}")

        # Exit code based on overall success
        sys.exit(0 if summary["overall_success"] else 1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
