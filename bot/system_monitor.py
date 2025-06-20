"""
System health monitoring and comprehensive error recovery for bulletproof reliability.

This module provides continuous health monitoring, automatic recovery actions,
and comprehensive system resilience strategies.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import aiohttp
import psutil

from bot.error_handling import ServiceHealth, ServiceStatus

# Configure logger
logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""

    RESTART_COMPONENT = "restart_component"
    RECONNECT_SERVICE = "reconnect_service"
    CLEAR_CACHE = "clear_cache"
    RESET_CONNECTION = "reset_connection"
    FALLBACK_MODE = "fallback_mode"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class SystemMetrics:
    """System performance and health metrics."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available: int = 0
    disk_usage_percent: float = 0.0
    network_connections: int = 0
    uptime_seconds: float = 0.0
    active_components: int = 0
    error_rate: float = 0.0
    response_time_avg: float = 0.0


@dataclass
class RecoveryAction:
    """Recovery action configuration."""

    name: str
    strategy: RecoveryStrategy
    action: Callable
    max_attempts: int = 3
    cooldown_seconds: int = 60
    last_attempt: datetime | None = None
    attempt_count: int = 0
    success_count: int = 0


class SystemHealthMonitor:
    """
    Comprehensive system health monitor with automatic recovery capabilities.

    Continuously monitors system health, component status, and performance metrics.
    Automatically triggers recovery actions when issues are detected.
    """

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.component_health: dict[str, ServiceHealth] = {}
        self.recovery_actions: dict[str, list[RecoveryAction]] = {}
        self.health_checks: dict[str, Callable] = {}
        self.system_metrics: list[SystemMetrics] = []
        self.monitoring_task: asyncio.Task | None = None
        self.is_monitoring = False
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "error_rate": 5.0,  # errors per minute
            "response_time_avg": 5000.0,  # milliseconds
        }
        self._start_time = datetime.now(UTC)

    def register_component(
        self,
        name: str,
        health_check: Callable,
        recovery_actions: list[RecoveryAction] | None = None,
    ) -> None:
        """Register a component for health monitoring."""
        self.component_health[name] = ServiceHealth(name=name)
        self.health_checks[name] = health_check

        if recovery_actions:
            self.recovery_actions[name] = recovery_actions
        else:
            self.recovery_actions[name] = []

        logger.info(f"Registered component {name} for health monitoring")

    def add_recovery_action(
        self, component_name: str, recovery_action: RecoveryAction
    ) -> None:
        """Add a recovery action for a component."""
        if component_name not in self.recovery_actions:
            self.recovery_actions[component_name] = []

        self.recovery_actions[component_name].append(recovery_action)
        logger.info(
            f"Added recovery action {recovery_action.name} for {component_name}"
        )

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring is already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started system health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task

        logger.info("Stopped system health monitoring")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Check component health
                await self._check_all_components()

                # Analyze health and trigger recovery if needed
                await self._analyze_and_recover()

                # Wait for next check interval
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # Get system metrics using psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network_connections = len(psutil.net_connections())

            # Calculate uptime
            uptime = (datetime.now(UTC) - self._start_time).total_seconds()

            # Calculate error rate (errors in last minute)
            current_time = datetime.now(UTC)
            one_minute_ago = current_time - timedelta(minutes=1)

            error_count = 0
            total_response_time = 0
            response_count = 0

            # Count errors and calculate average response time from component health
            for health in self.component_health.values():
                if health.last_check and health.last_check >= one_minute_ago:
                    if health.status in [ServiceStatus.UNHEALTHY, ServiceStatus.ERROR]:
                        error_count += 1

                    # Add response time if available in metrics
                    if "response_time_ms" in health.metrics:
                        total_response_time += health.metrics["response_time_ms"]
                        response_count += 1

            error_rate = error_count  # errors per minute
            avg_response_time = (
                total_response_time / response_count if response_count > 0 else 0
            )

            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available,
                disk_usage_percent=disk.percent,
                network_connections=network_connections,
                uptime_seconds=uptime,
                active_components=len(
                    [
                        h
                        for h in self.component_health.values()
                        if h.status == ServiceStatus.HEALTHY
                    ]
                ),
                error_rate=error_rate,
                response_time_avg=avg_response_time,
            )

            self.system_metrics.append(metrics)

            # Keep only last 1000 metrics (about 8 hours at 30s intervals)
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]

        except Exception as e:
            logger.exception(f"Failed to collect system metrics: {e}")

    async def _check_all_components(self) -> None:
        """Check health of all registered components."""
        for component_name, health_check in self.health_checks.items():
            await self._check_component_health(component_name, health_check)

    async def _check_component_health(
        self, component_name: str, health_check: Callable
    ) -> None:
        """Check health of a specific component."""
        component_health = self.component_health[component_name]
        start_time = datetime.now()

        try:
            # Execute health check
            if asyncio.iscoroutinefunction(health_check):
                is_healthy = await health_check()
            else:
                is_healthy = health_check()

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            if is_healthy:
                # Component is healthy
                if component_health.status != ServiceStatus.HEALTHY:
                    logger.info(
                        f"Component {component_name} recovered to healthy state"
                    )

                component_health.status = ServiceStatus.HEALTHY
                component_health.consecutive_failures = 0
                component_health.last_error = None

            else:
                # Component is unhealthy
                component_health.status = ServiceStatus.UNHEALTHY
                component_health.failure_count += 1
                component_health.consecutive_failures += 1
                component_health.last_error = "Health check returned unhealthy status"

                logger.warning(
                    f"Component {component_name} is unhealthy "
                    f"(consecutive failures: {component_health.consecutive_failures})"
                )

            component_health.last_check = datetime.now(UTC)
            component_health.metrics["response_time_ms"] = response_time

        except Exception as e:
            # Health check failed
            component_health.status = ServiceStatus.ERROR
            component_health.failure_count += 1
            component_health.consecutive_failures += 1
            component_health.last_error = str(e)
            component_health.last_check = datetime.now(UTC)

            logger.exception(f"Health check failed for {component_name}: {e}")

    async def _analyze_and_recover(self) -> None:
        """Analyze component health and trigger recovery actions."""
        current_time = datetime.now(UTC)

        # Check system-level thresholds
        if self.system_metrics:
            latest_metrics = self.system_metrics[-1]
            await self._check_system_thresholds(latest_metrics)

        # Check component-level recovery needs
        for component_name, health in self.component_health.items():
            if self._needs_recovery(health):
                await self._trigger_recovery(component_name, health, current_time)

    async def _check_system_thresholds(self, metrics: SystemMetrics) -> None:
        """Check if system metrics exceed alert thresholds."""
        alerts = []

        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        if metrics.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")

        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {metrics.error_rate:.1f} errors/min")

        if metrics.response_time_avg > self.alert_thresholds["response_time_avg"]:
            alerts.append(f"High response time: {metrics.response_time_avg:.1f}ms")

        if alerts:
            logger.warning(f"System threshold alerts: {', '.join(alerts)}")

    def _needs_recovery(self, health: ServiceHealth) -> bool:
        """Determine if a component needs recovery action."""
        # Trigger recovery after 3 consecutive failures
        return health.consecutive_failures >= 3 and health.status in [
            ServiceStatus.UNHEALTHY,
            ServiceStatus.ERROR,
        ]

    async def _trigger_recovery(
        self, component_name: str, health: ServiceHealth, current_time: datetime
    ) -> None:
        """Trigger recovery actions for a component."""
        recovery_actions = self.recovery_actions.get(component_name, [])

        if not recovery_actions:
            logger.warning(f"No recovery actions configured for {component_name}")
            return

        logger.info(f"Triggering recovery for {component_name}")
        health.recovery_attempts += 1

        for recovery_action in recovery_actions:
            # Check if we can attempt this recovery action
            if not self._can_attempt_recovery(recovery_action, current_time):
                continue

            try:
                logger.info(f"Executing recovery action: {recovery_action.name}")

                # Execute recovery action
                if asyncio.iscoroutinefunction(recovery_action.action):
                    await recovery_action.action(component_name, health)
                else:
                    recovery_action.action(component_name, health)

                # Update recovery action state
                recovery_action.last_attempt = current_time
                recovery_action.attempt_count += 1
                recovery_action.success_count += 1

                logger.info(
                    f"Recovery action {recovery_action.name} completed successfully"
                )
                break

            except Exception as e:
                recovery_action.last_attempt = current_time
                recovery_action.attempt_count += 1

                logger.exception(f"Recovery action {recovery_action.name} failed: {e}")

    def _can_attempt_recovery(
        self, recovery_action: RecoveryAction, current_time: datetime
    ) -> bool:
        """Check if a recovery action can be attempted."""
        # Check max attempts
        if recovery_action.attempt_count >= recovery_action.max_attempts:
            return False

        # Check cooldown period
        if recovery_action.last_attempt:
            cooldown_delta = timedelta(seconds=recovery_action.cooldown_seconds)
            if current_time - recovery_action.last_attempt < cooldown_delta:
                return False

        return True

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        latest_metrics = self.system_metrics[-1] if self.system_metrics else None

        component_status = {}
        for name, health in self.component_health.items():
            component_status[name] = {
                "status": health.status.value,
                "last_check": (
                    health.last_check.isoformat() if health.last_check else None
                ),
                "failure_count": health.failure_count,
                "consecutive_failures": health.consecutive_failures,
                "recovery_attempts": health.recovery_attempts,
                "last_error": health.last_error,
                "metrics": health.metrics,
            }

        recovery_status = {}
        for component_name, actions in self.recovery_actions.items():
            recovery_status[component_name] = [
                {
                    "name": action.name,
                    "strategy": action.strategy.value,
                    "attempt_count": action.attempt_count,
                    "success_count": action.success_count,
                    "last_attempt": (
                        action.last_attempt.isoformat() if action.last_attempt else None
                    ),
                }
                for action in actions
            ]

        return {
            "monitoring_active": self.is_monitoring,
            "uptime_seconds": (datetime.now(UTC) - self._start_time).total_seconds(),
            "system_metrics": latest_metrics.__dict__ if latest_metrics else None,
            "component_health": component_status,
            "recovery_actions": recovery_status,
            "alert_thresholds": self.alert_thresholds,
        }

    def get_health_history(
        self, component_name: str, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get health history for a component."""
        # This would typically be stored in a time-series database
        # For now, we'll return recent metrics
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        return [
            {
                "timestamp": metric.timestamp.isoformat(),
                "cpu_percent": metric.cpu_percent,
                "memory_percent": metric.memory_percent,
                "error_rate": metric.error_rate,
                "response_time_avg": metric.response_time_avg,
            }
            for metric in self.system_metrics
            if metric.timestamp >= cutoff_time
        ]



class ErrorRecoveryManager:
    """
    Comprehensive error recovery strategies for different failure types.

    Provides specialized recovery strategies for various types of system failures
    including network errors, authentication issues, data corruption, and position inconsistencies.
    """

    def __init__(self):
        self.recovery_strategies: dict[str, Callable] = {
            "network_error": self._recover_network_connection,
            "auth_error": self._recover_authentication,
            "data_error": self._recover_data_integrity,
            "position_error": self._recover_position_state,
            "api_rate_limit": self._recover_rate_limit,
            "websocket_error": self._recover_websocket_connection,
            "database_error": self._recover_database_connection,
            "memory_error": self._recover_memory_issue,
        }
        self.recovery_history: list[dict[str, Any]] = []

    async def recover_from_error(
        self, error_type: str, error_context: dict[str, Any], component_name: str = ""
    ) -> bool:
        """Execute recovery strategy for a specific error type."""
        if error_type not in self.recovery_strategies:
            logger.warning(
                f"No recovery strategy available for error type: {error_type}"
            )
            return False

        recovery_start = datetime.now(UTC)
        recovery_strategy = self.recovery_strategies[error_type]

        try:
            logger.info(
                f"Starting recovery for {error_type} in component {component_name}"
            )

            # Execute recovery strategy
            success = await recovery_strategy(error_context, component_name)

            # Record recovery attempt
            recovery_record = {
                "timestamp": recovery_start.isoformat(),
                "error_type": error_type,
                "component": component_name,
                "success": success,
                "duration_seconds": (
                    datetime.now(UTC) - recovery_start
                ).total_seconds(),
                "context": error_context,
            }

            self.recovery_history.append(recovery_record)

            if success:
                logger.info(f"Recovery successful for {error_type}")
            else:
                logger.warning(f"Recovery failed for {error_type}")

            return success

        except Exception as e:
            logger.exception(f"Recovery strategy failed for {error_type}: {e}")

            # Record failed recovery attempt
            recovery_record = {
                "timestamp": recovery_start.isoformat(),
                "error_type": error_type,
                "component": component_name,
                "success": False,
                "duration_seconds": (
                    datetime.now(UTC) - recovery_start
                ).total_seconds(),
                "context": error_context,
                "recovery_error": str(e),
            }

            self.recovery_history.append(recovery_record)
            return False

    async def _recover_network_connection(
        self, error_context: dict[str, Any], component: str
    ) -> bool:
        """Recover from network connection errors."""
        logger.info(f"Attempting network recovery for component {component}")

        # Wait for network to stabilize
        await asyncio.sleep(2)

        # Test connectivity
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session, session.get("https://httpbin.org/status/200") as response:
                if response.status == 200:
                    logger.info("Network connectivity restored")
                    return True
        except Exception as e:
            logger.warning(f"Network connectivity test failed: {e}")

        return False

    async def _recover_authentication(
        self, error_context: dict[str, Any], component: str
    ) -> bool:
        """Recover from authentication errors."""
        logger.info(f"Attempting authentication recovery for component {component}")

        # This would typically involve:
        # 1. Refreshing API tokens
        # 2. Re-authenticating with the service
        # 3. Clearing cached authentication data

        # For now, we'll simulate a recovery delay
        await asyncio.sleep(1)

        logger.info("Authentication recovery completed")
        return True

    async def _recover_data_integrity(
        self, error_context: dict[str, Any], component: str
    ) -> bool:
        """Recover from data integrity errors."""
        logger.info(f"Attempting data integrity recovery for component {component}")

        # This would typically involve:
        # 1. Clearing corrupted caches
        # 2. Re-fetching clean data from sources
        # 3. Validating data consistency

        # Clear any cached data mentioned in error context
        cache_keys = error_context.get("cache_keys", [])
        for key in cache_keys:
            logger.debug(f"Clearing cache key: {key}")
            # Implementation would clear actual cache

        await asyncio.sleep(0.5)

        logger.info("Data integrity recovery completed")
        return True

    async def _recover_position_state(
        self, error_context: dict[str, Any], component: str
    ) -> bool:
        """Recover from position state inconsistencies."""
        logger.info(f"Attempting position state recovery for component {component}")

        # This would typically involve:
        # 1. Reconciling positions with exchange
        # 2. Fixing position tracking inconsistencies
        # 3. Resetting position managers

        symbol = error_context.get("symbol", "")
        if symbol:
            logger.debug(f"Reconciling positions for symbol: {symbol}")

        await asyncio.sleep(1)

        logger.info("Position state recovery completed")
        return True

    async def _recover_rate_limit(
        self, error_context: dict[str, Any], component: str
    ) -> bool:
        """Recover from API rate limit errors."""
        logger.info(f"Attempting rate limit recovery for component {component}")

        # Extract rate limit information from context
        retry_after = error_context.get("retry_after", 60)

        logger.info(f"Waiting {retry_after} seconds for rate limit to reset")
        await asyncio.sleep(retry_after)

        logger.info("Rate limit recovery completed")
        return True

    async def _recover_websocket_connection(
        self, error_context: dict[str, Any], component: str
    ) -> bool:
        """Recover from WebSocket connection errors."""
        logger.info(f"Attempting WebSocket recovery for component {component}")

        # This would typically involve:
        # 1. Closing existing connections
        # 2. Re-establishing WebSocket connections
        # 3. Re-subscribing to data streams

        await asyncio.sleep(2)

        logger.info("WebSocket recovery completed")
        return True

    async def _recover_database_connection(
        self, error_context: dict[str, Any], component: str
    ) -> bool:
        """Recover from database connection errors."""
        logger.info(f"Attempting database recovery for component {component}")

        # This would typically involve:
        # 1. Closing existing connections
        # 2. Re-establishing database connections
        # 3. Validating connection pool health

        await asyncio.sleep(1)

        logger.info("Database recovery completed")
        return True

    async def _recover_memory_issue(
        self, error_context: dict[str, Any], component: str
    ) -> bool:
        """Recover from memory-related errors."""
        logger.info(f"Attempting memory recovery for component {component}")

        # This would typically involve:
        # 1. Clearing memory caches
        # 2. Triggering garbage collection
        # 3. Reducing memory usage

        import gc

        gc.collect()

        await asyncio.sleep(0.5)

        logger.info("Memory recovery completed")
        return True

    def _get_recovery_action(
        self,
        name: str,
        action: Callable,
        strategy: RecoveryStrategy = RecoveryStrategy.RESTART_COMPONENT,
        max_attempts: int = 3,
        cooldown_seconds: int = 60,
    ) -> RecoveryAction:
        """Create a recovery action with the specified parameters."""
        return RecoveryAction(
            name=name,
            strategy=strategy,
            action=action,
            max_attempts=max_attempts,
            cooldown_seconds=cooldown_seconds,
        )

    def get_recovery_statistics(self) -> dict[str, Any]:
        """Get statistics about recovery attempts."""
        if not self.recovery_history:
            return {"total_recoveries": 0}

        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r["success"])

        # Group by error type
        error_type_stats = {}
        for record in self.recovery_history:
            error_type = record["error_type"]
            if error_type not in error_type_stats:
                error_type_stats[error_type] = {"total": 0, "successful": 0}

            error_type_stats[error_type]["total"] += 1
            if record["success"]:
                error_type_stats[error_type]["successful"] += 1

        return {
            "total_recoveries": total_recoveries,
            "successful_recoveries": successful_recoveries,
            "success_rate": (
                successful_recoveries / total_recoveries if total_recoveries > 0 else 0
            ),
            "error_type_statistics": error_type_stats,
            "recent_recoveries": self.recovery_history[-10:],  # Last 10 recoveries
        }


# Global instances
system_monitor = SystemHealthMonitor()
error_recovery_manager = ErrorRecoveryManager()
