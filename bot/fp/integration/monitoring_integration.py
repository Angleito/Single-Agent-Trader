"""
Functional Monitoring Integration

This module provides integration between the functional monitoring enhancements
and the existing imperative monitoring systems, preserving all existing APIs
while adding functional programming capabilities.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from bot.fp.adapters.performance_monitor_adapter import (
    FunctionalPerformanceMonitor,
    MonitoringSnapshot,
    enhance_existing_monitor,
)
from bot.fp.adapters.system_monitor_adapter import (
    FunctionalSystemMonitor,
    HealthState,
    enhance_existing_system_monitor,
)
from bot.fp.alerting.functional_alerting import (
    AlertEvent,
    AlertingEngine,
    AlertRule,
    AlertSeverity,
    NotificationChannel,
    NotificationConfig,
    create_alerting_engine,
    create_simple_threshold_rule,
)
from bot.fp.effects.io import IO
from bot.fp.effects.monitoring import MetricPoint, collect_system_metrics
from bot.performance_monitor import PerformanceMonitor, get_performance_monitor
from bot.system_monitor import SystemHealthMonitor, system_monitor

if TYPE_CHECKING:
    from bot.monitoring.performance_metrics import PerformanceMetricsCollector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntegrationConfig:
    """Configuration for monitoring integration"""

    enable_functional_performance: bool = True
    enable_functional_system: bool = True
    enable_functional_alerting: bool = True
    performance_monitor_interval: float = 30.0
    system_monitor_interval: float = 60.0
    alerting_evaluation_interval: float = 15.0
    preserve_legacy_apis: bool = True


@dataclass(frozen=True)
class MonitoringState:
    """Unified monitoring state snapshot"""

    timestamp: datetime
    performance_snapshot: MonitoringSnapshot | None
    system_health_state: HealthState | None
    active_alerts: list[AlertEvent]
    functional_metrics: list[MetricPoint]
    system_health_score: float
    integration_status: dict[str, Any] = field(default_factory=dict)


class UnifiedMonitoringSystem:
    """
    Unified monitoring system that integrates functional monitoring
    enhancements with existing imperative monitoring while preserving
    all existing APIs and adding new functional capabilities.
    """

    def __init__(self, config: IntegrationConfig | None = None):
        """Initialize the unified monitoring system"""
        self.config = config or IntegrationConfig()

        # Legacy monitoring components
        self.legacy_performance_monitor: PerformanceMonitor | None = None
        self.legacy_system_monitor: SystemHealthMonitor | None = None
        self.legacy_metrics_collector: PerformanceMetricsCollector | None = None

        # Functional monitoring components
        self.functional_performance: FunctionalPerformanceMonitor | None = None
        self.functional_system: FunctionalSystemMonitor | None = None
        self.alerting_engine: AlertingEngine | None = None

        # Integration state
        self.monitoring_state_history: list[MonitoringState] = []
        self.is_monitoring = False
        self.monitoring_tasks: list[asyncio.Task] = []

        # Initialize components
        self._initialize_monitoring_components()

    def _initialize_monitoring_components(self) -> None:
        """Initialize monitoring components based on configuration"""

        # Initialize legacy components if needed
        if self.config.preserve_legacy_apis:
            try:
                self.legacy_performance_monitor = get_performance_monitor()
                logger.info("Connected to legacy performance monitor")
            except Exception as e:
                logger.warning(f"Could not connect to legacy performance monitor: {e}")

            try:
                self.legacy_system_monitor = system_monitor
                logger.info("Connected to legacy system monitor")
            except Exception as e:
                logger.warning(f"Could not connect to legacy system monitor: {e}")

        # Initialize functional components
        if self.config.enable_functional_performance:
            self.functional_performance = (
                enhance_existing_monitor(self.legacy_performance_monitor)
                if self.legacy_performance_monitor
                else FunctionalPerformanceMonitor()
            )
            logger.info("Initialized functional performance monitor")

        if self.config.enable_functional_system:
            self.functional_system = (
                enhance_existing_system_monitor(self.legacy_system_monitor)
                if self.legacy_system_monitor
                else FunctionalSystemMonitor()
            )
            logger.info("Initialized functional system monitor")

        if self.config.enable_functional_alerting:
            self.alerting_engine = create_alerting_engine()
            self._setup_default_alert_rules()
            logger.info("Initialized functional alerting engine")

    def _setup_default_alert_rules(self) -> None:
        """Set up default alert rules for common monitoring scenarios"""
        if not self.alerting_engine:
            return

        # CPU usage alert
        cpu_rule = create_simple_threshold_rule(
            rule_id="high_cpu_usage",
            name="High CPU Usage",
            metric_name="cpu_percent",
            threshold=80.0,
            operator=">",
            severity=AlertSeverity.HIGH,
            channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE],
        )
        self.alerting_engine.add_alert_rule(cpu_rule).run()

        # Memory usage alert
        memory_rule = create_simple_threshold_rule(
            rule_id="high_memory_usage",
            name="High Memory Usage",
            metric_name="memory_percent",
            threshold=85.0,
            operator=">",
            severity=AlertSeverity.HIGH,
            channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE],
        )
        self.alerting_engine.add_alert_rule(memory_rule).run()

        # Disk usage alert
        disk_rule = create_simple_threshold_rule(
            rule_id="high_disk_usage",
            name="High Disk Usage",
            metric_name="disk_percent",
            threshold=90.0,
            operator=">",
            severity=AlertSeverity.CRITICAL,
            channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE],
        )
        self.alerting_engine.add_alert_rule(disk_rule).run()

        # Response time alert
        response_time_rule = create_simple_threshold_rule(
            rule_id="high_response_time",
            name="High Response Time",
            metric_name="response_time_avg",
            threshold=2000.0,
            operator=">",
            severity=AlertSeverity.MEDIUM,
            channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE],
        )
        self.alerting_engine.add_alert_rule(response_time_rule).run()

        logger.info("Set up default alert rules")

    # ==============================================================================
    # Unified Monitoring Operations
    # ==============================================================================

    def create_unified_monitoring_state(self) -> IO[MonitoringState]:
        """Create a unified monitoring state snapshot"""

        def create_state() -> MonitoringState:
            timestamp = datetime.now(UTC)

            # Collect performance snapshot
            performance_snapshot = None
            if self.functional_performance:
                try:
                    performance_snapshot = (
                        self.functional_performance.create_monitoring_snapshot().run()
                    )
                except Exception as e:
                    logger.exception(f"Error creating performance snapshot: {e}")

            # Collect system health state
            system_health_state = None
            if self.functional_system:
                try:
                    system_health_state = (
                        self.functional_system.create_health_state().run()
                    )
                except Exception as e:
                    logger.exception(f"Error creating system health state: {e}")

            # Collect active alerts
            active_alerts = []
            if self.alerting_engine:
                try:
                    active_alerts = self.alerting_engine.get_active_alerts().run()
                except Exception as e:
                    logger.exception(f"Error getting active alerts: {e}")

            # Collect functional metrics
            functional_metrics = []
            try:
                system_metrics = collect_system_metrics().run()
                functional_metrics.extend(
                    [
                        MetricPoint(
                            name="cpu_percent",
                            value=system_metrics.cpu_percent,
                            timestamp=timestamp,
                            unit="%",
                        ),
                        MetricPoint(
                            name="memory_percent",
                            value=system_metrics.memory_percent,
                            timestamp=timestamp,
                            unit="%",
                        ),
                        MetricPoint(
                            name="disk_percent",
                            value=system_metrics.disk_percent,
                            timestamp=timestamp,
                            unit="%",
                        ),
                    ]
                )
            except Exception as e:
                logger.exception(f"Error collecting functional metrics: {e}")

            # Calculate system health score
            system_health_score = 100.0
            if system_health_state:
                system_health_score = system_health_state.overall_health_score
            elif performance_snapshot:
                system_health_score = performance_snapshot.health_score_value

            # Integration status
            integration_status = {
                "functional_performance_enabled": self.functional_performance
                is not None,
                "functional_system_enabled": self.functional_system is not None,
                "alerting_enabled": self.alerting_engine is not None,
                "legacy_performance_connected": self.legacy_performance_monitor
                is not None,
                "legacy_system_connected": self.legacy_system_monitor is not None,
                "monitoring_active": self.is_monitoring,
                "active_tasks": len(self.monitoring_tasks),
            }

            return MonitoringState(
                timestamp=timestamp,
                performance_snapshot=performance_snapshot,
                system_health_state=system_health_state,
                active_alerts=active_alerts,
                functional_metrics=functional_metrics,
                system_health_score=system_health_score,
                integration_status=integration_status,
            )

        return IO(create_state)

    def process_metrics_with_alerting(
        self, metrics: list[MetricPoint]
    ) -> IO[list[AlertEvent]]:
        """Process metrics through the alerting engine"""

        def process_metrics() -> list[AlertEvent]:
            if not self.alerting_engine:
                return []

            try:
                return self.alerting_engine.process_metrics_batch(metrics).run()
            except Exception as e:
                logger.exception(f"Error processing metrics with alerting: {e}")
                return []

        return IO(process_metrics)

    def get_comprehensive_health_status(self) -> IO[dict[str, Any]]:
        """Get comprehensive health status from all monitoring components"""

        def get_status() -> dict[str, Any]:
            status = {
                "timestamp": datetime.now(UTC).isoformat(),
                "overall_health_score": 0.0,
                "component_health": {},
                "performance_metrics": {},
                "alert_summary": {},
                "integration_info": {},
            }

            # Get latest monitoring state
            try:
                latest_state = self.create_unified_monitoring_state().run()
                status["overall_health_score"] = latest_state.system_health_score

                # Performance metrics
                if latest_state.performance_snapshot:
                    status["performance_metrics"] = {
                        "health_score": latest_state.performance_snapshot.health_score_value,
                        "metrics_count": len(latest_state.performance_snapshot.metrics),
                        "alerts_count": len(latest_state.performance_snapshot.alerts),
                    }

                # System health
                if latest_state.system_health_state:
                    status["component_health"] = {
                        name: {
                            "status": check.status.value,
                            "response_time_ms": check.response_time_ms,
                        }
                        for name, check in latest_state.system_health_state.component_health.items()
                    }

                # Alert summary
                status["alert_summary"] = {
                    "active_alerts": len(latest_state.active_alerts),
                    "alert_severities": {
                        severity.value: sum(
                            1
                            for alert in latest_state.active_alerts
                            if alert.rule.severity == severity
                        )
                        for severity in AlertSeverity
                    },
                }

                # Integration info
                status["integration_info"] = latest_state.integration_status

            except Exception as e:
                logger.exception(f"Error getting comprehensive health status: {e}")
                status["error"] = str(e)

            return status

        return IO(get_status)

    # ==============================================================================
    # Legacy API Preservation
    # ==============================================================================

    def get_performance_summary(
        self, duration: timedelta | None = None
    ) -> dict[str, Any]:
        """Preserve legacy performance summary API with functional enhancements"""
        duration = duration or timedelta(minutes=10)

        try:
            # Try functional performance monitor first
            if self.functional_performance:
                analysis = self.functional_performance.analyze_performance(
                    duration
                ).run()
                return {
                    "timestamp": analysis.timestamp.isoformat(),
                    "health_score": analysis.health_score,
                    "bottlenecks": analysis.bottlenecks,
                    "recommendations": analysis.recommendations,
                    "metrics_summary": analysis.metrics_summary,
                    "trend_analysis": analysis.trend_analysis,
                    "source": "functional",
                }

            # Fallback to legacy monitor
            if self.legacy_performance_monitor:
                return {
                    **self.legacy_performance_monitor.get_performance_summary(duration),
                    "source": "legacy",
                }

            return {
                "error": "No performance monitor available",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.exception(f"Error getting performance summary: {e}")
            return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    def get_system_status(self) -> dict[str, Any]:
        """Preserve legacy system status API with functional enhancements"""
        try:
            # Try functional system monitor first
            if self.functional_system:
                health_summary = self.functional_system.export_health_summary().run()
                return {**health_summary, "source": "functional"}

            # Fallback to legacy monitor
            if self.legacy_system_monitor:
                return {
                    **self.legacy_system_monitor.get_system_status(),
                    "source": "legacy",
                }

            return {
                "error": "No system monitor available",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.exception(f"Error getting system status: {e}")
            return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    # ==============================================================================
    # Enhanced Functional APIs
    # ==============================================================================

    def add_custom_alert_rule(self, rule: AlertRule) -> bool:
        """Add a custom alert rule to the functional alerting engine"""
        if not self.alerting_engine:
            logger.warning("Alerting engine not available")
            return False

        try:
            self.alerting_engine.add_alert_rule(rule).run()
            logger.info(f"Added custom alert rule: {rule.name}")
            return True
        except Exception as e:
            logger.exception(f"Error adding alert rule: {e}")
            return False

    def configure_notification_channel(
        self, channel: NotificationChannel, config: NotificationConfig
    ) -> bool:
        """Configure a notification channel for alerts"""
        if not self.alerting_engine:
            logger.warning("Alerting engine not available")
            return False

        try:
            self.alerting_engine.configure_notification_channel(channel, config).run()
            logger.info(f"Configured notification channel: {channel.value}")
            return True
        except Exception as e:
            logger.exception(f"Error configuring notification channel: {e}")
            return False

    def get_monitoring_trends(
        self, duration: timedelta = timedelta(hours=24)
    ) -> dict[str, Any]:
        """Get monitoring trends analysis using functional components"""
        try:
            trends = {}

            if self.functional_system:
                system_trends = self.functional_system.analyze_health_trends(
                    duration
                ).run()
                trends["system_health"] = system_trends

            if self.functional_performance:
                # Get recent snapshots for trend analysis
                cutoff_time = datetime.now(UTC) - duration
                recent_snapshots = self.functional_performance.get_snapshots_since(
                    cutoff_time
                )

                if recent_snapshots:
                    health_scores = [s.health_score_value for s in recent_snapshots]
                    trends["performance"] = {
                        "health_score_trend": {
                            "current": health_scores[-1] if health_scores else 0,
                            "average": (
                                sum(health_scores) / len(health_scores)
                                if health_scores
                                else 0
                            ),
                            "min": min(health_scores) if health_scores else 0,
                            "max": max(health_scores) if health_scores else 0,
                            "sample_count": len(health_scores),
                        }
                    }

            if self.alerting_engine:
                alert_stats = self.alerting_engine.get_alert_statistics(duration).run()
                trends["alerting"] = alert_stats

            return {
                "analysis_period_hours": duration.total_seconds() / 3600,
                "timestamp": datetime.now(UTC).isoformat(),
                "trends": trends,
            }

        except Exception as e:
            logger.exception(f"Error getting monitoring trends: {e}")
            return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    # ==============================================================================
    # Monitoring Lifecycle Management
    # ==============================================================================

    async def start_unified_monitoring(self) -> None:
        """Start unified monitoring with all components"""
        if self.is_monitoring:
            logger.warning("Unified monitoring is already running")
            return

        self.is_monitoring = True

        # Start functional performance monitoring
        if self.functional_performance:
            task = asyncio.create_task(
                self.functional_performance.start_functional_monitoring(
                    self.config.performance_monitor_interval
                )
            )
            self.monitoring_tasks.append(task)

        # Start functional system monitoring
        if self.functional_system:
            task = asyncio.create_task(
                self.functional_system.start_functional_monitoring(
                    self.config.system_monitor_interval
                )
            )
            self.monitoring_tasks.append(task)

        # Start unified monitoring loop
        task = asyncio.create_task(self._unified_monitoring_loop())
        self.monitoring_tasks.append(task)

        # Start legacy monitoring if available
        if self.legacy_performance_monitor:
            try:
                await self.legacy_performance_monitor.start_monitoring()
            except Exception as e:
                logger.exception(f"Error starting legacy performance monitor: {e}")

        if self.legacy_system_monitor:
            try:
                await self.legacy_system_monitor.start_monitoring()
            except Exception as e:
                logger.exception(f"Error starting legacy system monitor: {e}")

        logger.info("Started unified monitoring system")

    async def stop_unified_monitoring(self) -> None:
        """Stop unified monitoring"""
        self.is_monitoring = False

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.monitoring_tasks.clear()

        # Stop legacy monitoring
        if self.legacy_performance_monitor:
            try:
                await self.legacy_performance_monitor.stop_monitoring()
            except Exception as e:
                logger.exception(f"Error stopping legacy performance monitor: {e}")

        if self.legacy_system_monitor:
            try:
                await self.legacy_system_monitor.stop_monitoring()
            except Exception as e:
                logger.exception(f"Error stopping legacy system monitor: {e}")

        logger.info("Stopped unified monitoring system")

    async def _unified_monitoring_loop(self) -> None:
        """Main unified monitoring loop"""
        while self.is_monitoring:
            try:
                # Create monitoring state snapshot
                monitoring_state = self.create_unified_monitoring_state().run()

                # Add to history
                self.monitoring_state_history.append(monitoring_state)

                # Keep history bounded
                if len(self.monitoring_state_history) > 1000:
                    self.monitoring_state_history = self.monitoring_state_history[
                        -1000:
                    ]

                # Process metrics through alerting
                if monitoring_state.functional_metrics:
                    alert_events = self.process_metrics_with_alerting(
                        monitoring_state.functional_metrics
                    ).run()

                    if alert_events:
                        logger.info(
                            f"Generated {len(alert_events)} alerts from metrics"
                        )

                # Log summary periodically
                if len(self.monitoring_state_history) % 10 == 0:
                    logger.info(
                        f"Unified monitoring: Health score {monitoring_state.system_health_score:.1f}, "
                        f"{len(monitoring_state.active_alerts)} active alerts, "
                        f"{len(monitoring_state.functional_metrics)} metrics collected"
                    )

                await asyncio.sleep(self.config.alerting_evaluation_interval)

            except Exception as e:
                logger.exception(f"Error in unified monitoring loop: {e}")
                await asyncio.sleep(self.config.alerting_evaluation_interval)

    def get_latest_monitoring_state(self) -> MonitoringState | None:
        """Get the most recent monitoring state"""
        return (
            self.monitoring_state_history[-1] if self.monitoring_state_history else None
        )

    def export_monitoring_data(
        self, format_type: str = "json", duration: timedelta | None = None
    ) -> str:
        """Export comprehensive monitoring data"""
        try:
            if duration:
                cutoff_time = datetime.now(UTC) - duration
                states = [
                    state
                    for state in self.monitoring_state_history
                    if state.timestamp >= cutoff_time
                ]
            else:
                states = self.monitoring_state_history

            if format_type.lower() == "json":
                import json

                export_data = {
                    "export_timestamp": datetime.now(UTC).isoformat(),
                    "total_states": len(states),
                    "config": {
                        "functional_performance": self.config.enable_functional_performance,
                        "functional_system": self.config.enable_functional_system,
                        "functional_alerting": self.config.enable_functional_alerting,
                        "preserve_legacy": self.config.preserve_legacy_apis,
                    },
                    "states": [
                        {
                            "timestamp": state.timestamp.isoformat(),
                            "health_score": state.system_health_score,
                            "metrics_count": len(state.functional_metrics),
                            "alerts_count": len(state.active_alerts),
                            "integration_status": state.integration_status,
                        }
                        for state in states
                    ],
                }
                return json.dumps(export_data, indent=2)

            return f"Unsupported export format: {format_type}"

        except Exception as e:
            logger.exception(f"Error exporting monitoring data: {e}")
            return f"Export failed: {e!s}"


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_unified_monitoring_system(
    config: IntegrationConfig | None = None,
) -> UnifiedMonitoringSystem:
    """Factory function to create a unified monitoring system"""
    return UnifiedMonitoringSystem(config)


def integrate_with_existing_monitoring(
    performance_monitor: PerformanceMonitor | None = None,
    system_monitor: SystemHealthMonitor | None = None,
    metrics_collector: PerformanceMetricsCollector | None = None,
) -> UnifiedMonitoringSystem:
    """Integrate functional monitoring with existing monitoring components"""

    # Create unified system
    unified_system = UnifiedMonitoringSystem()

    # Connect existing components
    if performance_monitor:
        unified_system.legacy_performance_monitor = performance_monitor
        unified_system.functional_performance = enhance_existing_monitor(
            performance_monitor
        )
        logger.info("Integrated with existing performance monitor")

    if system_monitor:
        unified_system.legacy_system_monitor = system_monitor
        unified_system.functional_system = enhance_existing_system_monitor(
            system_monitor
        )
        logger.info("Integrated with existing system monitor")

    if metrics_collector:
        unified_system.legacy_metrics_collector = metrics_collector
        logger.info("Connected to existing metrics collector")

    return unified_system


# ==============================================================================
# Global Integration Instance
# ==============================================================================

# Global unified monitoring system instance
_unified_monitoring: UnifiedMonitoringSystem | None = None


def get_unified_monitoring() -> UnifiedMonitoringSystem:
    """Get the global unified monitoring system instance"""
    global _unified_monitoring
    if _unified_monitoring is None:
        _unified_monitoring = create_unified_monitoring_system()
    return _unified_monitoring


def initialize_monitoring_integration(
    config: IntegrationConfig | None = None,
) -> UnifiedMonitoringSystem:
    """Initialize the global unified monitoring system"""
    global _unified_monitoring
    _unified_monitoring = create_unified_monitoring_system(config)
    return _unified_monitoring
