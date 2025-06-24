"""
Functional System Monitor Adapter

This module provides functional programming enhancements to the existing
system monitoring and error recovery capabilities, using pure functions
and composable effects while preserving all existing APIs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Protocol

from ...system_monitor import (
    ErrorRecoveryManager,
    RecoveryStrategy,
    SystemHealthMonitor,
)
from ..effects.io import IO
from ..effects.monitoring import (
    HealthCheck,
    HealthStatus,
    SystemMetrics,
    batch_health_checks,
    collect_system_metrics,
    system_health_check,
)

logger = logging.getLogger(__name__)


class RecoveryResult(Enum):
    """Recovery operation results"""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class RecoveryPlan:
    """Immutable recovery plan with pure functions"""

    component: str
    issue_type: str
    strategies: list[RecoveryStrategy]
    priority: int
    estimated_duration: float
    prerequisites: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecoveryExecution:
    """Immutable record of recovery execution"""

    plan: RecoveryPlan
    result: RecoveryResult
    executed_at: datetime
    duration_seconds: float
    details: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass(frozen=True)
class HealthState:
    """Immutable health state snapshot"""

    timestamp: datetime
    component_health: dict[str, HealthCheck]
    system_metrics: SystemMetrics
    recovery_history: list[RecoveryExecution]
    overall_health_score: float
    critical_issues: list[str]


@dataclass(frozen=True)
class MonitoringConfig:
    """Immutable monitoring configuration"""

    check_interval_seconds: float = 30.0
    recovery_cooldown_seconds: float = 300.0
    max_recovery_attempts: int = 3
    critical_threshold: float = 70.0
    warning_threshold: float = 85.0


class HealthCheckProvider(Protocol):
    """Protocol for health check providers"""

    def check_health(self) -> IO[HealthCheck]:
        """Return health check result"""
        ...


class RecoveryProvider(Protocol):
    """Protocol for recovery providers"""

    def recover(self, context: dict[str, Any]) -> IO[RecoveryResult]:
        """Execute recovery with given context"""
        ...


class FunctionalSystemMonitor:
    """
    Functional system monitor that enhances existing system monitoring
    with pure functions, immutable state, and composable effects.
    """

    def __init__(
        self,
        legacy_monitor: SystemHealthMonitor | None = None,
        error_recovery: ErrorRecoveryManager | None = None,
        config: MonitoringConfig | None = None,
    ):
        """Initialize with optional legacy components and configuration"""
        self.legacy_monitor = legacy_monitor
        self.error_recovery = error_recovery
        self.config = config or MonitoringConfig()

        # Functional state
        self._health_providers: dict[str, HealthCheckProvider] = {}
        self._recovery_providers: dict[str, RecoveryProvider] = {}
        self._monitoring_history: list[HealthState] = []
        self._recovery_plans: dict[str, RecoveryPlan] = {}

        # Initialize default health checks and recovery plans
        self._initialize_default_providers()

    # ==============================================================================
    # Pure Health Check Functions
    # ==============================================================================

    def _initialize_default_providers(self) -> None:
        """Initialize default health check and recovery providers"""

        # System resource health checks
        self._health_providers.update(
            {
                "cpu": CPUHealthProvider(),
                "memory": MemoryHealthProvider(),
                "disk": DiskHealthProvider(),
                "network": NetworkHealthProvider(),
                "processes": ProcessHealthProvider(),
            }
        )

        # Recovery providers
        self._recovery_providers.update(
            {
                "memory_cleanup": MemoryRecoveryProvider(),
                "disk_cleanup": DiskRecoveryProvider(),
                "service_restart": ServiceRecoveryProvider(),
                "network_reset": NetworkRecoveryProvider(),
            }
        )

        # Default recovery plans
        self._recovery_plans.update(
            {
                "high_memory": RecoveryPlan(
                    component="memory",
                    issue_type="high_usage",
                    strategies=[
                        RecoveryStrategy.CLEAR_CACHE,
                        RecoveryStrategy.RESTART_COMPONENT,
                    ],
                    priority=1,
                    estimated_duration=30.0,
                ),
                "high_cpu": RecoveryPlan(
                    component="cpu",
                    issue_type="high_usage",
                    strategies=[
                        RecoveryStrategy.CIRCUIT_BREAKER,
                        RecoveryStrategy.FALLBACK_MODE,
                    ],
                    priority=2,
                    estimated_duration=15.0,
                ),
                "disk_full": RecoveryPlan(
                    component="disk",
                    issue_type="space_exhausted",
                    strategies=[RecoveryStrategy.CLEAR_CACHE],
                    priority=1,
                    estimated_duration=60.0,
                ),
                "network_failure": RecoveryPlan(
                    component="network",
                    issue_type="connectivity_lost",
                    strategies=[
                        RecoveryStrategy.RESET_CONNECTION,
                        RecoveryStrategy.RECONNECT_SERVICE,
                    ],
                    priority=1,
                    estimated_duration=45.0,
                ),
            }
        )

    def create_health_state(self) -> IO[HealthState]:
        """Create comprehensive health state snapshot"""

        def create_state() -> HealthState:
            # Collect system metrics
            sys_metrics = collect_system_metrics().run()

            # Run all health checks
            health_checks = self._run_all_health_checks().run()

            # Calculate overall health score
            score = self._calculate_health_score(health_checks)

            # Identify critical issues
            critical = self._identify_critical_issues(health_checks, sys_metrics)

            # Get recent recovery history
            recent_history = self._get_recent_recovery_history()

            return HealthState(
                timestamp=datetime.now(UTC),
                component_health=health_checks,
                system_metrics=sys_metrics,
                recovery_history=recent_history,
                overall_health_score=score,
                critical_issues=critical,
            )

        return IO(create_state)

    def _run_all_health_checks(self) -> IO[dict[str, HealthCheck]]:
        """Run all registered health checks"""

        def run_checks() -> dict[str, HealthCheck]:
            checks = []

            # Add system health check
            checks.append(system_health_check())

            # Add provider-based health checks
            for name, provider in self._health_providers.items():
                try:
                    check = provider.check_health()
                    checks.append(check)
                except Exception:
                    # Create failed health check
                    failed_check = IO(
                        lambda: HealthCheck(
                            component=name,
                            status=HealthStatus.UNHEALTHY,
                            timestamp=datetime.now(UTC),
                            details={"error": str(e)},
                        )
                    )
                    checks.append(failed_check)

            return batch_health_checks(checks).run()

        return IO(run_checks)

    def _calculate_health_score(self, health_checks: dict[str, HealthCheck]) -> float:
        """Calculate overall health score from component checks"""
        if not health_checks:
            return 0.0

        total_score = 0.0
        for check in health_checks.values():
            if check.status == HealthStatus.HEALTHY:
                total_score += 100.0
            elif check.status == HealthStatus.DEGRADED:
                total_score += 60.0
            else:  # UNHEALTHY
                total_score += 0.0

        return total_score / len(health_checks)

    def _identify_critical_issues(
        self, health_checks: dict[str, HealthCheck], system_metrics: SystemMetrics
    ) -> list[str]:
        """Identify critical issues requiring immediate attention"""
        issues = []

        # Check unhealthy components
        for name, check in health_checks.items():
            if check.status == HealthStatus.UNHEALTHY:
                issues.append(
                    f"Component '{name}' is unhealthy: {check.details.get('error', 'Unknown error')}"
                )

        # Check critical system metrics
        if system_metrics.cpu_percent > 95.0:
            issues.append(f"Critical CPU usage: {system_metrics.cpu_percent:.1f}%")

        if system_metrics.memory_percent > 95.0:
            issues.append(
                f"Critical memory usage: {system_metrics.memory_percent:.1f}%"
            )

        if system_metrics.disk_percent > 98.0:
            issues.append(f"Critical disk usage: {system_metrics.disk_percent:.1f}%")

        return issues

    def _get_recent_recovery_history(self) -> list[RecoveryExecution]:
        """Get recent recovery executions from history"""
        # In a real implementation, this would pull from persistent storage
        # For now, return empty list as placeholder
        return []

    # ==============================================================================
    # Pure Recovery Functions
    # ==============================================================================

    def create_recovery_plan(
        self, component: str, issue_type: str, health_state: HealthState
    ) -> IO[RecoveryPlan | None]:
        """Create a recovery plan based on current health state"""

        def create_plan() -> RecoveryPlan | None:
            # Check if we have a predefined plan
            plan_key = f"{component}_{issue_type}"
            if plan_key in self._recovery_plans:
                return self._recovery_plans[plan_key]

            # Check for generic component plans
            if component in self._recovery_plans:
                return self._recovery_plans[component]

            # Create dynamic plan based on issue type
            if issue_type == "high_usage":
                return RecoveryPlan(
                    component=component,
                    issue_type=issue_type,
                    strategies=[
                        RecoveryStrategy.CLEAR_CACHE,
                        RecoveryStrategy.RESTART_COMPONENT,
                    ],
                    priority=2,
                    estimated_duration=30.0,
                )
            if issue_type == "connectivity":
                return RecoveryPlan(
                    component=component,
                    issue_type=issue_type,
                    strategies=[
                        RecoveryStrategy.RESET_CONNECTION,
                        RecoveryStrategy.RECONNECT_SERVICE,
                    ],
                    priority=1,
                    estimated_duration=45.0,
                )
            if issue_type == "performance":
                return RecoveryPlan(
                    component=component,
                    issue_type=issue_type,
                    strategies=[
                        RecoveryStrategy.CIRCUIT_BREAKER,
                        RecoveryStrategy.FALLBACK_MODE,
                    ],
                    priority=3,
                    estimated_duration=20.0,
                )

            return None

        return IO(create_plan)

    def execute_recovery_plan(self, plan: RecoveryPlan) -> IO[RecoveryExecution]:
        """Execute a recovery plan and return the execution result"""

        def execute() -> RecoveryExecution:
            start_time = datetime.now(UTC)

            try:
                # Check prerequisites
                if not self._check_prerequisites(plan.prerequisites):
                    return RecoveryExecution(
                        plan=plan,
                        result=RecoveryResult.SKIPPED,
                        executed_at=start_time,
                        duration_seconds=0.0,
                        details={"reason": "Prerequisites not met"},
                    )

                # Execute recovery strategies in order
                success_count = 0
                total_strategies = len(plan.strategies)
                execution_details = {}

                for i, strategy in enumerate(plan.strategies):
                    try:
                        strategy_result = self._execute_strategy(
                            strategy, plan.component, {"plan": plan, "step": i}
                        )

                        if strategy_result:
                            success_count += 1
                            execution_details[f"strategy_{i}"] = "success"
                        else:
                            execution_details[f"strategy_{i}"] = "failed"

                    except Exception as e:
                        execution_details[f"strategy_{i}"] = f"error: {e!s}"

                # Determine overall result
                if success_count == total_strategies:
                    result = RecoveryResult.SUCCESS
                elif success_count > 0:
                    result = RecoveryResult.PARTIAL
                else:
                    result = RecoveryResult.FAILURE

                end_time = datetime.now(UTC)
                duration = (end_time - start_time).total_seconds()

                return RecoveryExecution(
                    plan=plan,
                    result=result,
                    executed_at=start_time,
                    duration_seconds=duration,
                    details=execution_details,
                )

            except Exception as e:
                end_time = datetime.now(UTC)
                duration = (end_time - start_time).total_seconds()

                return RecoveryExecution(
                    plan=plan,
                    result=RecoveryResult.FAILURE,
                    executed_at=start_time,
                    duration_seconds=duration,
                    error_message=str(e),
                )

        return IO(execute)

    def _check_prerequisites(self, prerequisites: list[str]) -> bool:
        """Check if recovery prerequisites are met"""
        # In a real implementation, this would check system state
        # For now, always return True
        return True

    def _execute_strategy(
        self, strategy: RecoveryStrategy, component: str, context: dict[str, Any]
    ) -> bool:
        """Execute a specific recovery strategy"""
        try:
            if strategy == RecoveryStrategy.CLEAR_CACHE:
                return self._clear_cache_strategy(component, context)
            if strategy == RecoveryStrategy.RESTART_COMPONENT:
                return self._restart_component_strategy(component, context)
            if strategy == RecoveryStrategy.RESET_CONNECTION:
                return self._reset_connection_strategy(component, context)
            if strategy == RecoveryStrategy.RECONNECT_SERVICE:
                return self._reconnect_service_strategy(component, context)
            if strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self._circuit_breaker_strategy(component, context)
            if strategy == RecoveryStrategy.FALLBACK_MODE:
                return self._fallback_mode_strategy(component, context)
            logger.warning(f"Unknown recovery strategy: {strategy}")
            return False
        except Exception as e:
            logger.error(f"Error executing strategy {strategy} for {component}: {e}")
            return False

    def _clear_cache_strategy(self, component: str, context: dict[str, Any]) -> bool:
        """Execute cache clearing strategy"""
        logger.info(f"Executing cache clear strategy for {component}")

        if component == "memory":
            import gc

            gc.collect()
            return True
        if component == "disk":
            # In real implementation, would clear temporary files
            logger.info("Clearing disk cache (simulated)")
            return True

        return True

    def _restart_component_strategy(
        self, component: str, context: dict[str, Any]
    ) -> bool:
        """Execute component restart strategy"""
        logger.info(f"Executing restart strategy for {component} (simulated)")
        # In real implementation, would restart specific services
        return True

    def _reset_connection_strategy(
        self, component: str, context: dict[str, Any]
    ) -> bool:
        """Execute connection reset strategy"""
        logger.info(f"Executing connection reset strategy for {component} (simulated)")
        # In real implementation, would reset network connections
        return True

    def _reconnect_service_strategy(
        self, component: str, context: dict[str, Any]
    ) -> bool:
        """Execute service reconnection strategy"""
        logger.info(
            f"Executing service reconnection strategy for {component} (simulated)"
        )
        # In real implementation, would reconnect to external services
        return True

    def _circuit_breaker_strategy(
        self, component: str, context: dict[str, Any]
    ) -> bool:
        """Execute circuit breaker strategy"""
        logger.info(f"Executing circuit breaker strategy for {component} (simulated)")
        # In real implementation, would enable circuit breaker patterns
        return True

    def _fallback_mode_strategy(self, component: str, context: dict[str, Any]) -> bool:
        """Execute fallback mode strategy"""
        logger.info(f"Executing fallback mode strategy for {component} (simulated)")
        # In real implementation, would switch to fallback behavior
        return True

    # ==============================================================================
    # Monitoring and Analysis Functions
    # ==============================================================================

    def analyze_health_trends(
        self, duration: timedelta = timedelta(hours=1)
    ) -> IO[dict[str, Any]]:
        """Analyze health trends over time"""

        def analyze() -> dict[str, Any]:
            cutoff_time = datetime.now(UTC) - duration
            recent_states = [
                state
                for state in self._monitoring_history
                if state.timestamp >= cutoff_time
            ]

            if not recent_states:
                return {"message": "No recent health data available"}

            # Health score trend
            health_scores = [state.overall_health_score for state in recent_states]
            health_trend = self._calculate_trend(health_scores)

            # Component stability analysis
            component_stability = {}
            all_components = set()
            for state in recent_states:
                all_components.update(state.component_health.keys())

            for component in all_components:
                statuses = []
                for state in recent_states:
                    if component in state.component_health:
                        status = state.component_health[component].status
                        statuses.append(1 if status == HealthStatus.HEALTHY else 0)

                if statuses:
                    stability = sum(statuses) / len(statuses) * 100
                    component_stability[component] = stability

            # Recovery effectiveness
            recoveries = []
            for state in recent_states:
                recoveries.extend(state.recovery_history)

            if recoveries:
                successful_recoveries = sum(
                    1 for r in recoveries if r.result == RecoveryResult.SUCCESS
                )
                recovery_success_rate = successful_recoveries / len(recoveries) * 100
            else:
                recovery_success_rate = 0.0

            return {
                "analysis_period": duration.total_seconds(),
                "health_trend": {
                    "direction": health_trend,
                    "current_score": health_scores[-1] if health_scores else 0,
                    "average_score": (
                        sum(health_scores) / len(health_scores) if health_scores else 0
                    ),
                    "min_score": min(health_scores) if health_scores else 0,
                    "max_score": max(health_scores) if health_scores else 0,
                },
                "component_stability": component_stability,
                "recovery_effectiveness": {
                    "total_recoveries": len(recoveries),
                    "success_rate": recovery_success_rate,
                    "most_common_issues": self._get_most_common_issues(recoveries),
                },
                "critical_incidents": len(
                    [state for state in recent_states if state.critical_issues]
                ),
            }

        return IO(analyze)

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"

        # Simple trend based on first vs last values
        change_percent = (
            ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        )

        if abs(change_percent) < 5:
            return "stable"
        if change_percent > 0:
            return "improving"
        return "declining"

    def _get_most_common_issues(
        self, recoveries: list[RecoveryExecution]
    ) -> list[dict[str, Any]]:
        """Get most common recovery issues"""
        issue_counts = {}
        for recovery in recoveries:
            issue_type = recovery.plan.issue_type
            if issue_type not in issue_counts:
                issue_counts[issue_type] = 0
            issue_counts[issue_type] += 1

        return [
            {"issue_type": issue, "count": count}
            for issue, count in sorted(
                issue_counts.items(), key=lambda x: x[1], reverse=True
            )
        ]

    # ==============================================================================
    # Integration and Lifecycle Functions
    # ==============================================================================

    def add_to_history(self, health_state: HealthState) -> IO[None]:
        """Add health state to monitoring history"""

        def add() -> None:
            self._monitoring_history.append(health_state)
            # Keep history bounded (last 1000 entries)
            if len(self._monitoring_history) > 1000:
                self._monitoring_history = self._monitoring_history[-1000:]

        return IO(add)

    async def start_functional_monitoring(
        self, interval_seconds: float | None = None
    ) -> None:
        """Start functional monitoring loop"""

        interval = interval_seconds or self.config.check_interval_seconds

        async def monitoring_loop():
            while True:
                try:
                    # Create health state
                    health_state = self.create_health_state().run()

                    # Add to history
                    self.add_to_history(health_state).run()

                    # Check for critical issues and trigger recovery
                    if health_state.critical_issues:
                        await self._handle_critical_issues(health_state)

                    # Log summary periodically
                    if len(self._monitoring_history) % 10 == 0:
                        logger.info(
                            f"Health monitoring: Score {health_state.overall_health_score:.1f}, "
                            f"{len(health_state.critical_issues)} critical issues, "
                            f"{len(health_state.component_health)} components checked"
                        )

                    await asyncio.sleep(interval)

                except Exception as e:
                    logger.error(f"Error in functional monitoring loop: {e}")
                    await asyncio.sleep(interval)

        # Start the monitoring loop
        asyncio.create_task(monitoring_loop())
        logger.info(f"Started functional health monitoring with {interval}s interval")

    async def _handle_critical_issues(self, health_state: HealthState) -> None:
        """Handle critical issues by triggering appropriate recovery plans"""

        for issue in health_state.critical_issues:
            try:
                # Determine component and issue type from issue description
                if "CPU" in issue:
                    component, issue_type = "cpu", "high_usage"
                elif "memory" in issue:
                    component, issue_type = "memory", "high_usage"
                elif "disk" in issue:
                    component, issue_type = "disk", "space_exhausted"
                elif "unhealthy" in issue:
                    # Extract component name
                    parts = issue.split("'")
                    component = parts[1] if len(parts) >= 2 else "unknown"
                    issue_type = "health_failure"
                else:
                    continue

                # Create and execute recovery plan
                plan = self.create_recovery_plan(
                    component, issue_type, health_state
                ).run()
                if plan:
                    logger.info(
                        f"Executing recovery plan for {component}: {issue_type}"
                    )
                    execution = self.execute_recovery_plan(plan).run()

                    if execution.result == RecoveryResult.SUCCESS:
                        logger.info(f"Recovery successful for {component}")
                    else:
                        logger.warning(
                            f"Recovery failed for {component}: {execution.error_message}"
                        )

            except Exception as e:
                logger.error(f"Error handling critical issue '{issue}': {e}")

    def get_latest_health_state(self) -> HealthState | None:
        """Get the most recent health state"""
        return self._monitoring_history[-1] if self._monitoring_history else None

    def export_health_summary(self) -> IO[dict[str, Any]]:
        """Export comprehensive health summary"""

        def export() -> dict[str, Any]:
            latest_state = self.get_latest_health_state()
            if not latest_state:
                return {"message": "No health data available"}

            return {
                "timestamp": latest_state.timestamp.isoformat(),
                "overall_health_score": latest_state.overall_health_score,
                "critical_issues_count": len(latest_state.critical_issues),
                "critical_issues": latest_state.critical_issues,
                "component_health": {
                    name: {
                        "status": check.status.value,
                        "response_time_ms": check.response_time_ms,
                        "details": check.details,
                    }
                    for name, check in latest_state.component_health.items()
                },
                "system_metrics": {
                    "cpu_percent": latest_state.system_metrics.cpu_percent,
                    "memory_percent": latest_state.system_metrics.memory_percent,
                    "memory_mb": latest_state.system_metrics.memory_mb,
                    "disk_percent": latest_state.system_metrics.disk_percent,
                    "uptime_seconds": latest_state.system_metrics.uptime_seconds,
                },
                "recent_recoveries": len(latest_state.recovery_history),
                "monitoring_history_size": len(self._monitoring_history),
            }

        return IO(export)


# ==============================================================================
# Health Check Provider Implementations
# ==============================================================================


class CPUHealthProvider(HealthCheckProvider):
    """CPU health check provider"""

    def check_health(self) -> IO[HealthCheck]:
        def check() -> HealthCheck:
            try:
                import psutil

                cpu_percent = psutil.cpu_percent(interval=0.1)

                if cpu_percent < 80:
                    status = HealthStatus.HEALTHY
                elif cpu_percent < 95:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.UNHEALTHY

                return HealthCheck(
                    component="cpu",
                    status=status,
                    timestamp=datetime.now(UTC),
                    details={"cpu_percent": cpu_percent},
                )
            except Exception as e:
                return HealthCheck(
                    component="cpu",
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(UTC),
                    details={"error": str(e)},
                )

        return IO(check)


class MemoryHealthProvider(HealthCheckProvider):
    """Memory health check provider"""

    def check_health(self) -> IO[HealthCheck]:
        def check() -> HealthCheck:
            try:
                import psutil

                memory = psutil.virtual_memory()

                if memory.percent < 80:
                    status = HealthStatus.HEALTHY
                elif memory.percent < 95:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.UNHEALTHY

                return HealthCheck(
                    component="memory",
                    status=status,
                    timestamp=datetime.now(UTC),
                    details={
                        "memory_percent": memory.percent,
                        "available_mb": memory.available / 1024 / 1024,
                    },
                )
            except Exception as e:
                return HealthCheck(
                    component="memory",
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(UTC),
                    details={"error": str(e)},
                )

        return IO(check)


class DiskHealthProvider(HealthCheckProvider):
    """Disk health check provider"""

    def check_health(self) -> IO[HealthCheck]:
        def check() -> HealthCheck:
            try:
                import psutil

                disk = psutil.disk_usage("/")

                if disk.percent < 85:
                    status = HealthStatus.HEALTHY
                elif disk.percent < 95:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.UNHEALTHY

                return HealthCheck(
                    component="disk",
                    status=status,
                    timestamp=datetime.now(UTC),
                    details={
                        "disk_percent": disk.percent,
                        "free_gb": disk.free / 1024 / 1024 / 1024,
                    },
                )
            except Exception as e:
                return HealthCheck(
                    component="disk",
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(UTC),
                    details={"error": str(e)},
                )

        return IO(check)


class NetworkHealthProvider(HealthCheckProvider):
    """Network health check provider"""

    def check_health(self) -> IO[HealthCheck]:
        def check() -> HealthCheck:
            try:
                import psutil

                connections = psutil.net_connections()
                active_connections = len(
                    [c for c in connections if c.status == "ESTABLISHED"]
                )

                # Simple heuristic: too many or too few connections might indicate issues
                if 5 <= active_connections <= 100:
                    status = HealthStatus.HEALTHY
                elif 1 <= active_connections <= 200:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.UNHEALTHY

                return HealthCheck(
                    component="network",
                    status=status,
                    timestamp=datetime.now(UTC),
                    details={"active_connections": active_connections},
                )
            except Exception as e:
                return HealthCheck(
                    component="network",
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(UTC),
                    details={"error": str(e)},
                )

        return IO(check)


class ProcessHealthProvider(HealthCheckProvider):
    """Process health check provider"""

    def check_health(self) -> IO[HealthCheck]:
        def check() -> HealthCheck:
            try:
                import psutil

                process_count = len(psutil.pids())

                # Simple heuristic for process count health
                if process_count < 500:
                    status = HealthStatus.HEALTHY
                elif process_count < 1000:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.UNHEALTHY

                return HealthCheck(
                    component="processes",
                    status=status,
                    timestamp=datetime.now(UTC),
                    details={"process_count": process_count},
                )
            except Exception as e:
                return HealthCheck(
                    component="processes",
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(UTC),
                    details={"error": str(e)},
                )

        return IO(check)


# ==============================================================================
# Recovery Provider Implementations
# ==============================================================================


class MemoryRecoveryProvider(RecoveryProvider):
    """Memory recovery provider"""

    def recover(self, context: dict[str, Any]) -> IO[RecoveryResult]:
        def recover_memory() -> RecoveryResult:
            try:
                import gc

                gc.collect()
                logger.info("Executed memory garbage collection")
                return RecoveryResult.SUCCESS
            except Exception as e:
                logger.error(f"Memory recovery failed: {e}")
                return RecoveryResult.FAILURE

        return IO(recover_memory)


class DiskRecoveryProvider(RecoveryProvider):
    """Disk recovery provider"""

    def recover(self, context: dict[str, Any]) -> IO[RecoveryResult]:
        def recover_disk() -> RecoveryResult:
            try:
                # In real implementation, would clean temporary files
                logger.info("Executed disk cleanup (simulated)")
                return RecoveryResult.SUCCESS
            except Exception as e:
                logger.error(f"Disk recovery failed: {e}")
                return RecoveryResult.FAILURE

        return IO(recover_disk)


class ServiceRecoveryProvider(RecoveryProvider):
    """Service recovery provider"""

    def recover(self, context: dict[str, Any]) -> IO[RecoveryResult]:
        def recover_service() -> RecoveryResult:
            try:
                # In real implementation, would restart services
                logger.info("Executed service restart (simulated)")
                return RecoveryResult.SUCCESS
            except Exception as e:
                logger.error(f"Service recovery failed: {e}")
                return RecoveryResult.FAILURE

        return IO(recover_service)


class NetworkRecoveryProvider(RecoveryProvider):
    """Network recovery provider"""

    def recover(self, context: dict[str, Any]) -> IO[RecoveryResult]:
        def recover_network() -> RecoveryResult:
            try:
                # In real implementation, would reset network connections
                logger.info("Executed network reset (simulated)")
                return RecoveryResult.SUCCESS
            except Exception as e:
                logger.error(f"Network recovery failed: {e}")
                return RecoveryResult.FAILURE

        return IO(recover_network)


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_functional_system_monitor(
    legacy_monitor: SystemHealthMonitor | None = None,
    error_recovery: ErrorRecoveryManager | None = None,
    config: MonitoringConfig | None = None,
) -> FunctionalSystemMonitor:
    """Factory function to create a functional system monitor"""
    return FunctionalSystemMonitor(legacy_monitor, error_recovery, config)


def enhance_existing_system_monitor(
    system_monitor: SystemHealthMonitor,
    error_recovery: ErrorRecoveryManager | None = None,
) -> FunctionalSystemMonitor:
    """Enhance an existing system monitor with functional capabilities"""
    return FunctionalSystemMonitor(system_monitor, error_recovery)
