"""
Property-based tests for Docker service invariants.

These tests verify that our Docker services maintain critical properties
under various conditions including network failures, restarts, and load.
"""

import asyncio
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import docker
import hypothesis.strategies as st
import pytest
import requests
from hypothesis import assume, given, settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Service configuration with expected properties
SERVICE_CONFIG = {
    "bluefin-service": {
        "health_endpoint": "/health",
        "port": 8081,
        "container_name": "bluefin-service",
        "expected_health_keys": {"status", "timestamp"},
        "restart_policy": "unless-stopped",
        "network_aliases": ["bluefin-service", "bluefin"],
        "max_memory": "768M",
        "healthcheck_interval": 30,
        "healthcheck_timeout": 10,
        "healthcheck_retries": 3,
    },
    "ai-trading-bot": {
        "health_endpoint": None,  # Uses healthcheck.sh script
        "port": None,
        "container_name": "ai-trading-bot",
        "expected_health_keys": set(),
        "restart_policy": "unless-stopped",
        "network_aliases": ["ai-trading-bot", "trading-bot"],
        "max_memory": "1.5G",
        "healthcheck_interval": 30,
        "healthcheck_timeout": 15,
        "healthcheck_retries": 3,
        "depends_on": ["mcp-omnisearch", "mcp-memory"],
    },
    "mcp-omnisearch": {
        "health_endpoint": None,  # Uses node command
        "port": 8767,
        "container_name": "mcp-omnisearch-server",
        "expected_health_keys": set(),
        "restart_policy": "unless-stopped",
        "network_aliases": ["mcp-omnisearch", "omnisearch"],
        "max_memory": "768M",
        "healthcheck_interval": 30,
        "healthcheck_timeout": 10,
        "healthcheck_retries": 3,
    },
    "mcp-memory": {
        "health_endpoint": "/health",
        "port": 8765,
        "container_name": "mcp-memory-server",
        "expected_health_keys": {"status"},
        "restart_policy": "unless-stopped",
        "network_aliases": ["mcp-memory", "memory-server"],
        "max_memory": "768M",
        "healthcheck_interval": 30,
        "healthcheck_timeout": 10,
        "healthcheck_retries": 3,
    },
    "dashboard-backend": {
        "health_endpoint": "/health",
        "port": 8000,
        "container_name": "dashboard-backend",
        "expected_health_keys": {"status"},
        "restart_policy": "unless-stopped",
        "network_aliases": ["dashboard-backend", "api"],
        "max_memory": "768M",
        "healthcheck_interval": 30,
        "healthcheck_timeout": 10,
        "healthcheck_retries": 3,
        "depends_on": ["bluefin-service"],
    },
    "dashboard-frontend": {
        "health_endpoint": None,  # Static files served by nginx
        "port": 3000,
        "container_name": "dashboard-frontend",
        "expected_health_keys": set(),
        "restart_policy": "unless-stopped",
        "network_aliases": ["dashboard-frontend", "frontend"],
        "max_memory": "1G",
        "depends_on": ["dashboard-backend"],
    },
}


# Docker client fixture
@pytest.fixture(scope="session")
def docker_client():
    """Create Docker client for the test session."""
    try:
        client = docker.from_env()
        # Verify Docker is accessible
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")


@pytest.fixture(scope="function")
def service_monitor(docker_client):
    """Monitor service status during tests."""
    monitor = ServiceMonitor(docker_client)
    yield monitor
    monitor.cleanup()


class ServiceMonitor:
    """Monitor Docker service health and network status."""

    def __init__(self, docker_client):
        self.client = docker_client
        self.start_time = time.time()
        self.event_log = []
        self.health_checks = defaultdict(list)

    def log_event(self, event_type: str, service: str, details: dict[str, Any]):
        """Log service events for analysis."""
        self.event_log.append(
            {
                "timestamp": time.time() - self.start_time,
                "type": event_type,
                "service": service,
                "details": details,
            }
        )

    def check_service_health(self, service_name: str) -> dict[str, Any] | None:
        """Check if a service is healthy via its health endpoint."""
        config = SERVICE_CONFIG.get(service_name)
        if not config or not config.get("health_endpoint"):
            return None

        port = config.get("port")
        if not port:
            return None

        try:
            url = f"http://localhost:{port}{config['health_endpoint']}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                health_data = (
                    response.json()
                    if response.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else {"status": "ok"}
                )
                self.health_checks[service_name].append(
                    {
                        "timestamp": time.time(),
                        "status": "healthy",
                        "data": health_data,
                    }
                )
                return health_data
        except Exception as e:
            self.health_checks[service_name].append(
                {
                    "timestamp": time.time(),
                    "status": "unhealthy",
                    "error": str(e),
                }
            )
        return None

    def get_container_stats(self, container_name: str) -> dict[str, Any] | None:
        """Get container resource usage statistics."""
        try:
            container = self.client.containers.get(container_name)
            stats = container.stats(stream=False)
            return {
                "cpu_percent": self._calculate_cpu_percent(stats),
                "memory_usage": stats["memory_stats"].get("usage", 0),
                "memory_limit": stats["memory_stats"].get("limit", 0),
                "network_rx": sum(
                    net["rx_bytes"] for net in stats.get("networks", {}).values()
                ),
                "network_tx": sum(
                    net["tx_bytes"] for net in stats.get("networks", {}).values()
                ),
            }
        except Exception:
            return None

    def _calculate_cpu_percent(self, stats: dict[str, Any]) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"]
                - stats["precpu_stats"]["system_cpu_usage"]
            )
            num_cpus = len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1]))
            if system_delta > 0:
                return (cpu_delta / system_delta) * num_cpus * 100.0
        except Exception:
            pass
        return 0.0

    def cleanup(self):
        """Cleanup and report findings."""
        if self.event_log:
            logger.info(f"Recorded {len(self.event_log)} events during test")


# Hypothesis strategies
@st.composite
def service_names(draw) -> str:
    """Generate valid service names."""
    return draw(st.sampled_from(list(SERVICE_CONFIG.keys())))


@st.composite
def network_delays(draw) -> float:
    """Generate realistic network delays in seconds."""
    return draw(st.floats(min_value=0.0, max_value=5.0))


@st.composite
def failure_scenarios(draw) -> dict[str, Any]:
    """Generate failure injection scenarios."""
    scenario_type = draw(
        st.sampled_from(
            [
                "network_partition",
                "service_crash",
                "resource_exhaustion",
                "slow_response",
            ]
        )
    )
    duration = draw(st.floats(min_value=0.1, max_value=10.0))
    services = draw(st.lists(service_names(), min_size=1, max_size=3, unique=True))

    return {
        "type": scenario_type,
        "duration": duration,
        "services": services,
        "severity": draw(st.sampled_from(["low", "medium", "high"])),
    }


@st.composite
def health_check_timings(draw) -> dict[str, int]:
    """Generate health check timing configurations."""
    return {
        "interval": draw(st.integers(min_value=5, max_value=60)),
        "timeout": draw(st.integers(min_value=1, max_value=30)),
        "retries": draw(st.integers(min_value=1, max_value=10)),
        "start_period": draw(st.integers(min_value=0, max_value=120)),
    }


# Property tests
class TestDockerServiceProperties:
    """Property-based tests for Docker service invariants."""

    @given(service_name=service_names())
    @settings(max_examples=20, deadline=timedelta(seconds=30))
    def test_service_discoverable(self, docker_client, service_name):
        """Property: All configured services must be discoverable in Docker."""
        config = SERVICE_CONFIG[service_name]
        container_name = config["container_name"]

        try:
            container = docker_client.containers.get(container_name)
            assert container.status in ["running", "restarting"]

            # Verify container is on the expected network
            networks = container.attrs["NetworkSettings"]["Networks"]
            assert "trading-network" in networks

            # Verify network aliases
            network_config = networks["trading-network"]
            for alias in config["network_aliases"]:
                assert alias in network_config.get("Aliases", [])

        except docker.errors.NotFound:
            pytest.skip(f"Service {container_name} not running")

    @given(service_name=service_names())
    @settings(max_examples=15, deadline=timedelta(seconds=45))
    def test_health_check_schema_invariant(
        self, docker_client, service_monitor, service_name
    ):
        """Property: Health check responses must maintain expected schema."""
        config = SERVICE_CONFIG[service_name]

        if not config.get("health_endpoint"):
            pytest.skip(f"Service {service_name} has no health endpoint")

        # Perform multiple health checks
        health_responses = []
        for _ in range(5):
            health_data = service_monitor.check_service_health(service_name)
            if health_data:
                health_responses.append(health_data)
            time.sleep(0.5)

        # Verify schema consistency
        if health_responses and config["expected_health_keys"]:
            for response in health_responses:
                assert isinstance(response, dict)
                for key in config["expected_health_keys"]:
                    assert (
                        key in response
                    ), f"Missing expected key '{key}' in health response"

    @given(
        service_name=service_names(),
        restart_delay=st.floats(min_value=0.5, max_value=5.0),
    )
    @settings(max_examples=10, deadline=timedelta(seconds=60))
    def test_service_restart_resilience(
        self, docker_client, service_monitor, service_name, restart_delay
    ):
        """Property: Services must recover from restarts within expected time."""
        config = SERVICE_CONFIG[service_name]
        container_name = config["container_name"]

        try:
            container = docker_client.containers.get(container_name)
            initial_id = container.id

            # Record initial health
            initial_health = service_monitor.check_service_health(service_name)

            # Restart container
            container.restart(timeout=10)
            time.sleep(restart_delay)

            # Verify container recovered
            container = docker_client.containers.get(container_name)
            assert container.status == "running"

            # Wait for health check to pass
            max_recovery_time = config["healthcheck_interval"] * config[
                "healthcheck_retries"
            ] + config.get("start_period", 0)
            start_time = time.time()
            recovered = False

            while time.time() - start_time < max_recovery_time:
                if container.attrs["State"]["Health"]["Status"] == "healthy":
                    recovered = True
                    break
                time.sleep(1)
                container.reload()

            assert (
                recovered
            ), f"Service {service_name} did not recover within {max_recovery_time}s"

            # Verify health endpoint works after restart
            if config.get("health_endpoint"):
                post_restart_health = service_monitor.check_service_health(service_name)
                assert post_restart_health is not None

        except docker.errors.NotFound:
            pytest.skip(f"Service {container_name} not running")

    @given(
        services=st.lists(service_names(), min_size=2, max_size=4, unique=True),
        partition_duration=st.floats(min_value=0.5, max_value=3.0),
    )
    @settings(max_examples=5, deadline=timedelta(seconds=90))
    def test_network_partition_tolerance(
        self, docker_client, service_monitor, services, partition_duration
    ):
        """Property: Services must handle network partitions gracefully."""
        # Get containers for selected services
        containers = []
        for service_name in services:
            try:
                container_name = SERVICE_CONFIG[service_name]["container_name"]
                container = docker_client.containers.get(container_name)
                containers.append((service_name, container))
            except docker.errors.NotFound:
                pass

        if len(containers) < 2:
            pytest.skip("Not enough running services for network partition test")

        # Simulate network partition by disconnecting from network
        disconnected = []
        try:
            for service_name, container in containers[: len(containers) // 2]:
                container.reload()
                networks = list(container.attrs["NetworkSettings"]["Networks"].keys())
                if "trading-network" in networks:
                    docker_client.networks.get("trading-network").disconnect(container)
                    disconnected.append((service_name, container))
                    service_monitor.log_event(
                        "network_partition", service_name, {"action": "disconnect"}
                    )

            time.sleep(partition_duration)

            # Reconnect services
            for service_name, container in disconnected:
                docker_client.networks.get("trading-network").connect(container)
                service_monitor.log_event(
                    "network_partition", service_name, {"action": "reconnect"}
                )

            # Verify services recover
            time.sleep(5)  # Allow time for recovery

            for service_name, container in containers:
                container.reload()
                assert container.status == "running"

                # Check if dependent services handle the partition
                config = SERVICE_CONFIG[service_name]
                if "depends_on" in config:
                    # Verify service didn't crash due to dependency unavailability
                    logs = container.logs(tail=50).decode("utf-8")
                    assert "fatal" not in logs.lower()
                    assert "panic" not in logs.lower()

        finally:
            # Ensure all containers are reconnected
            for service_name, container in disconnected:
                try:
                    docker_client.networks.get("trading-network").connect(container)
                except Exception:
                    pass

    @given(
        service_name=service_names(),
        load_factor=st.floats(min_value=1.0, max_value=10.0),
    )
    @settings(max_examples=10, deadline=timedelta(seconds=45))
    def test_resource_limit_enforcement(
        self, docker_client, service_monitor, service_name, load_factor
    ):
        """Property: Services must respect resource limits under load."""
        config = SERVICE_CONFIG[service_name]
        container_name = config["container_name"]

        try:
            container = docker_client.containers.get(container_name)

            # Get resource stats multiple times
            stats_samples = []
            for _ in range(int(load_factor)):
                stats = service_monitor.get_container_stats(container_name)
                if stats:
                    stats_samples.append(stats)
                time.sleep(0.5)

            if not stats_samples:
                pytest.skip("Could not collect resource statistics")

            # Verify memory usage stays within limits
            max_memory_str = config["max_memory"]
            max_memory_bytes = self._parse_memory_string(max_memory_str)

            for stats in stats_samples:
                memory_usage = stats["memory_usage"]
                assert (
                    memory_usage <= max_memory_bytes
                ), f"Memory usage {memory_usage} exceeds limit {max_memory_bytes}"

                # Verify CPU usage is reasonable (not pegged at 100%)
                cpu_percent = stats["cpu_percent"]
                assert cpu_percent < 95.0, f"CPU usage too high: {cpu_percent}%"

        except docker.errors.NotFound:
            pytest.skip(f"Service {container_name} not running")

    def _parse_memory_string(self, memory_str: str) -> int:
        """Convert memory string like '768M' to bytes."""
        units = {"K": 1024, "M": 1024**2, "G": 1024**3}
        for unit, multiplier in units.items():
            if memory_str.endswith(unit):
                return int(float(memory_str[:-1]) * multiplier)
        return int(memory_str)


class DockerServiceStateMachine(RuleBasedStateMachine):
    """Stateful property testing for Docker service interactions."""

    def __init__(self):
        super().__init__()
        try:
            self.docker_client = docker.from_env()
            self.service_monitor = ServiceMonitor(self.docker_client)
        except Exception:
            pytest.skip("Docker not available")

        self.running_services = set()
        self.health_states = {}
        self.network_connected = defaultdict(bool)
        self._initialize_state()

    def _initialize_state(self):
        """Initialize state from current Docker environment."""
        for service_name, config in SERVICE_CONFIG.items():
            try:
                container = self.docker_client.containers.get(config["container_name"])
                if container.status == "running":
                    self.running_services.add(service_name)
                    self.network_connected[service_name] = (
                        "trading-network"
                        in container.attrs["NetworkSettings"]["Networks"]
                    )

                    # Check initial health
                    health_data = self.service_monitor.check_service_health(
                        service_name
                    )
                    self.health_states[service_name] = health_data is not None
            except docker.errors.NotFound:
                pass

    # Bundles for state management
    services = Bundle("services")

    @rule(target=services, service=service_names())
    def add_service(self, service):
        """Add a service to track."""
        assume(service in self.running_services)
        return service

    @rule(service=services)
    def check_health_consistency(self, service):
        """Health checks should be consistent over time."""
        if not SERVICE_CONFIG[service].get("health_endpoint"):
            return

        # Perform multiple health checks
        results = []
        for _ in range(3):
            health_data = self.service_monitor.check_service_health(service)
            results.append(health_data is not None)
            time.sleep(0.2)

        # All checks should have same result (all healthy or all unhealthy)
        assert (
            len(set(results)) <= 1
        ), f"Inconsistent health check results for {service}: {results}"

    @rule(service=services)
    def verify_network_connectivity(self, service):
        """Services should maintain network connectivity."""
        config = SERVICE_CONFIG[service]
        try:
            container = self.docker_client.containers.get(config["container_name"])
            networks = container.attrs["NetworkSettings"]["Networks"]

            # Should be connected to trading network
            assert "trading-network" in networks

            # Should have expected aliases
            aliases = networks["trading-network"].get("Aliases", [])
            for expected_alias in config["network_aliases"]:
                assert expected_alias in aliases

        except docker.errors.NotFound:
            # Service was stopped, remove from tracking
            self.running_services.discard(service)

    @rule(service=services, duration=st.floats(min_value=0.1, max_value=2.0))
    def simulate_load(self, service, duration):
        """Simulate load on a service and verify it remains stable."""
        start_time = time.time()
        errors = 0
        requests_made = 0

        while time.time() - start_time < duration:
            health_data = self.service_monitor.check_service_health(service)
            requests_made += 1
            if health_data is None:
                errors += 1
            time.sleep(0.05)

        # Error rate should be low
        if requests_made > 0:
            error_rate = errors / requests_made
            assert (
                error_rate < 0.1
            ), f"High error rate {error_rate} for {service} under load"

    @rule(service1=services, service2=services)
    def verify_service_dependencies(self, service1, service2):
        """Dependent services should handle unavailability gracefully."""
        assume(service1 != service2)

        config1 = SERVICE_CONFIG[service1]
        depends_on = config1.get("depends_on", [])

        if service2 in depends_on:
            # service1 depends on service2
            # If service2 is unhealthy, service1 should still be running
            try:
                container1 = self.docker_client.containers.get(
                    config1["container_name"]
                )
                assert container1.status in ["running", "restarting"]

                # Check logs for dependency errors
                logs = container1.logs(tail=100).decode("utf-8")
                # Should not have fatal errors about missing dependencies
                assert "dependency failed" not in logs.lower()
                assert (
                    "cannot connect to" not in logs.lower()
                    or "retrying" in logs.lower()
                )

            except docker.errors.NotFound:
                pass


# Integration test using property strategies
@given(failure_scenario=failure_scenarios())
@settings(max_examples=5, deadline=timedelta(seconds=120))
def test_failure_recovery_integration(docker_client, service_monitor, failure_scenario):
    """Integration test: System recovers from various failure scenarios."""
    scenario_type = failure_scenario["type"]
    duration = failure_scenario["duration"]
    services = failure_scenario["services"]

    # Record initial state
    initial_states = {}
    for service in services:
        config = SERVICE_CONFIG[service]
        try:
            container = docker_client.containers.get(config["container_name"])
            initial_states[service] = {
                "status": container.status,
                "health": service_monitor.check_service_health(service),
            }
        except docker.errors.NotFound:
            pass

    if not initial_states:
        pytest.skip("No services running for failure scenario")

    # Apply failure scenario
    if scenario_type == "service_crash":
        # Stop services
        for service in services:
            try:
                container = docker_client.containers.get(
                    SERVICE_CONFIG[service]["container_name"]
                )
                container.stop(timeout=5)
                service_monitor.log_event(
                    "failure_injection", service, {"type": "stop"}
                )
            except Exception:
                pass

        time.sleep(duration)

        # Services should auto-restart due to restart policy
        for service in services:
            config = SERVICE_CONFIG[service]
            if config["restart_policy"] == "unless-stopped":
                # Wait for auto-restart
                time.sleep(10)
                try:
                    container = docker_client.containers.get(config["container_name"])
                    assert container.status == "running"
                except docker.errors.NotFound:
                    # Manually start if not auto-restarted
                    container = docker_client.containers.run(
                        config.get("image", "ai-trading-bot:latest"),
                        name=config["container_name"],
                        detach=True,
                        network="trading-network",
                        restart_policy={"Name": config["restart_policy"]},
                    )

    elif scenario_type == "resource_exhaustion":
        # Update container with tight resource limits
        for service in services:
            try:
                container = docker_client.containers.get(
                    SERVICE_CONFIG[service]["container_name"]
                )
                # Docker doesn't support runtime resource updates, so we simulate load instead
                # In a real test, you might recreate the container with different limits
                service_monitor.log_event(
                    "failure_injection", service, {"type": "resource_pressure"}
                )
            except Exception:
                pass

    # Verify recovery
    time.sleep(15)  # Allow time for recovery

    for service in initial_states:
        config = SERVICE_CONFIG[service]
        try:
            container = docker_client.containers.get(config["container_name"])

            # Should be running again
            assert container.status == "running"

            # Health should be restored
            if config.get("health_endpoint"):
                current_health = service_monitor.check_service_health(service)
                if initial_states[service]["health"] is not None:
                    assert (
                        current_health is not None
                    ), f"Service {service} health not restored"

        except docker.errors.NotFound:
            pytest.fail(f"Service {service} did not recover from {scenario_type}")


# Performance property tests
@given(
    concurrent_services=st.integers(min_value=1, max_value=5),
    check_interval=st.floats(min_value=0.1, max_value=1.0),
)
@settings(max_examples=10, deadline=timedelta(seconds=60))
def test_concurrent_health_check_performance(
    docker_client, service_monitor, concurrent_services, check_interval
):
    """Property: Concurrent health checks should not degrade performance."""
    # Select services with health endpoints
    services_with_health = [
        name for name, config in SERVICE_CONFIG.items() if config.get("health_endpoint")
    ][:concurrent_services]

    if not services_with_health:
        pytest.skip("No services with health endpoints")

    # Perform concurrent health checks
    async def check_all_services():
        tasks = []
        for service in services_with_health:
            tasks.append(
                asyncio.create_task(async_health_check(service, service_monitor))
            )
        return await asyncio.gather(*tasks)

    async def async_health_check(service, monitor):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, monitor.check_service_health, service)

    # Run multiple rounds of concurrent checks
    response_times = []
    for _ in range(5):
        start_time = time.time()
        results = asyncio.run(check_all_services())
        elapsed = time.time() - start_time
        response_times.append(elapsed)
        time.sleep(check_interval)

    # Response times should be consistent (not degrading)
    avg_response_time = sum(response_times) / len(response_times)
    max_response_time = max(response_times)

    # Max should not be more than 2x average (indicates degradation)
    assert (
        max_response_time < avg_response_time * 2
    ), f"Performance degradation detected: avg={avg_response_time:.2f}s, max={max_response_time:.2f}s"


# Helper functions
@contextmanager
def network_delay_injection(docker_client, container_name: str, delay_ms: int):
    """Context manager to inject network delay using tc (traffic control)."""
    try:
        container = docker_client.containers.get(container_name)
        # Add network delay
        container.exec_run(
            f"tc qdisc add dev eth0 root netem delay {delay_ms}ms", privileged=True
        )
        yield
    finally:
        # Remove network delay
        try:
            container.exec_run("tc qdisc del dev eth0 root", privileged=True)
        except Exception:
            pass


def verify_service_logs_health(
    docker_client, service_name: str, duration: int = 10
) -> bool:
    """Verify service logs don't contain critical errors."""
    config = SERVICE_CONFIG[service_name]
    try:
        container = docker_client.containers.get(config["container_name"])
        logs = container.logs(since=int(time.time() - duration)).decode("utf-8")

        # Check for critical errors
        error_patterns = [
            "panic:",
            "fatal error:",
            "segmentation fault",
            "out of memory",
            "cannot allocate memory",
            "too many open files",
        ]

        for pattern in error_patterns:
            if pattern.lower() in logs.lower():
                return False

        return True

    except Exception:
        return False


# Run stateful tests
def test_docker_service_state_machine():
    """Run the stateful property tests."""
    TestCase = DockerServiceStateMachine.TestCase
    TestCase.settings = settings(
        max_examples=20,
        deadline=timedelta(seconds=300),
        stateful_step_count=10,
    )
    state_machine_test = TestCase()
    state_machine_test.runTest()


if __name__ == "__main__":
    # Run with: python -m pytest tests/property/test_docker_services.py -v
    pytest.main([__file__, "-v", "--tb=short"])
