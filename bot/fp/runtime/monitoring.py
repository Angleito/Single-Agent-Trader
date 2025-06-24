"""
System Monitoring Runtime for Functional Trading Bot

This module provides runtime monitoring capabilities for system health,
performance metrics, and alerting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..effects.io import IO
from ..effects.logging import error, info, warn
from ..effects.monitoring import (
    AlertLevel,
    HealthStatus,
    create_alert,
    send_alert,
    health_check,
    record_gauge,
    record_histogram,
)
from ..effects.time import now


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""

    health_check_interval: timedelta = timedelta(minutes=1)
    metrics_collection_interval: timedelta = timedelta(seconds=30)
    alert_cooldown: timedelta = timedelta(minutes=5)
    memory_threshold_mb: float = 1000.0
    cpu_threshold_percent: float = 80.0


@dataclass
class SystemMetrics:
    """System metrics snapshot"""

    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    active_connections: int
    error_rate: float
    response_time_ms: float


class MonitoringRuntime:
    """System monitoring runtime"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.last_alerts: dict[str, datetime] = {}
        self.metrics_history: list[SystemMetrics] = []
        self.running = False

    def collect_system_metrics(self) -> IO[SystemMetrics]:
        """Collect current system metrics"""

        def collect():
            # Simulate metrics collection
            import random

            import psutil

            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=psutil.cpu_percent(),
                memory_mb=psutil.virtual_memory().used / 1024 / 1024,
                active_connections=random.randint(5, 20),
                error_rate=random.uniform(0, 5),
                response_time_ms=random.uniform(10, 100),
            )

        return IO(collect)

    def record_metrics(self, metrics: SystemMetrics) -> IO[None]:
        """Record metrics to monitoring system"""

        def record():
            # Record individual metrics
            record_gauge("system.cpu.percent", metrics.cpu_percent).run()
            record_gauge("system.memory.mb", metrics.memory_mb).run()
            record_gauge("system.connections.active", metrics.active_connections).run()
            record_gauge("system.error.rate", metrics.error_rate).run()
            record_histogram("system.response.time", metrics.response_time_ms).run()

            # Store in history (keep last 100 entries)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)

            info(
                "System metrics recorded",
                {
                    "cpu": metrics.cpu_percent,
                    "memory_mb": metrics.memory_mb,
                    "connections": metrics.active_connections,
                },
            ).run()

        return IO(record)

    def check_thresholds(self, metrics: SystemMetrics) -> IO[None]:
        """Check metrics against thresholds and alert if needed"""

        def check():
            current_time = now().run()

            # Check CPU threshold
            if metrics.cpu_percent > self.config.cpu_threshold_percent:
                self.send_alert_if_not_recent(
                    "high_cpu",
                    AlertLevel.WARNING,
                    f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    current_time,
                )

            # Check memory threshold
            if metrics.memory_mb > self.config.memory_threshold_mb:
                self.send_alert_if_not_recent(
                    "high_memory",
                    AlertLevel.WARNING,
                    f"High memory usage: {metrics.memory_mb:.1f}MB",
                    current_time,
                )

            # Check error rate
            if metrics.error_rate > 10:
                self.send_alert_if_not_recent(
                    "high_error_rate",
                    AlertLevel.CRITICAL,
                    f"High error rate: {metrics.error_rate:.1f}%",
                    current_time,
                )

        return IO(check)

    def send_alert_if_not_recent(
        self, alert_type: str, level: AlertLevel, message: str, current_time: datetime
    ) -> None:
        """Send alert if not sent recently"""
        last_alert = self.last_alerts.get(alert_type)

        if last_alert is None or current_time - last_alert > self.config.alert_cooldown:
            alert_obj = create_alert(level, message).run()
            send_alert(alert_obj).run()
            self.last_alerts[alert_type] = current_time

    def run_health_checks(self) -> IO[dict[str, HealthStatus]]:
        """Run all health checks"""

        def run_checks():
            checks = {
                "database": health_check("database").run(),
                "exchange_api": health_check("exchange_api").run(),
                "websocket": health_check("websocket").run(),
                "scheduler": health_check("scheduler").run(),
            }

            # Log health status
            unhealthy = [
                name
                for name, status in checks.items()
                if status != HealthStatus.HEALTHY
            ]

            if unhealthy:
                warn(
                    f"Unhealthy components: {', '.join(unhealthy)}",
                    {"unhealthy_components": unhealthy},
                ).run()
            else:
                info("All components healthy").run()

            return checks

        return IO(run_checks)

    async def monitoring_loop(self) -> None:
        """Main monitoring loop"""
        self.running = True
        info("Monitoring runtime started").run()

        try:
            last_health_check = datetime.min
            last_metrics_collection = datetime.min

            while self.running:
                current_time = now().run()

                # Collect metrics
                if (
                    current_time - last_metrics_collection
                    >= self.config.metrics_collection_interval
                ):
                    try:
                        metrics = self.collect_system_metrics().run()
                        self.record_metrics(metrics).run()
                        self.check_thresholds(metrics).run()
                        last_metrics_collection = current_time

                    except Exception as e:
                        error(f"Metrics collection failed: {e!s}").run()

                # Health checks
                if (
                    current_time - last_health_check
                    >= self.config.health_check_interval
                ):
                    try:
                        health_status = self.run_health_checks().run()

                        # Check for critical health issues
                        critical_issues = [
                            name
                            for name, status in health_status.items()
                            if status == HealthStatus.UNHEALTHY
                        ]

                        if critical_issues:
                            alert_obj = create_alert(
                                AlertLevel.CRITICAL,
                                f"Critical health issues: {', '.join(critical_issues)}",
                            ).run()
                            send_alert(alert_obj).run()

                        last_health_check = current_time

                    except Exception as e:
                        error(f"Health check failed: {e!s}").run()

                # Sleep until next check
                await asyncio.sleep(5.0)

        except Exception as e:
            error(f"Monitoring loop failed: {e!s}").run()
            raise

        finally:
            info("Monitoring runtime stopped").run()

    def stop(self) -> None:
        """Stop the monitoring runtime"""
        self.running = False
        info("Monitoring runtime stop requested").run()

    def get_latest_metrics(self) -> SystemMetrics | None:
        """Get the latest system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of recent metrics"""
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-10:]  # Last 10 samples

        return {
            "latest": {
                "cpu_percent": recent_metrics[-1].cpu_percent,
                "memory_mb": recent_metrics[-1].memory_mb,
                "connections": recent_metrics[-1].active_connections,
                "error_rate": recent_metrics[-1].error_rate,
            },
            "averages": {
                "cpu_percent": sum(m.cpu_percent for m in recent_metrics)
                / len(recent_metrics),
                "memory_mb": sum(m.memory_mb for m in recent_metrics)
                / len(recent_metrics),
                "response_time_ms": sum(m.response_time_ms for m in recent_metrics)
                / len(recent_metrics),
            },
            "sample_count": len(self.metrics_history),
            "last_updated": recent_metrics[-1].timestamp.isoformat(),
        }


# Global monitoring instance
_monitoring: MonitoringRuntime | None = None


def get_monitoring() -> MonitoringRuntime:
    """Get the global monitoring runtime"""
    global _monitoring
    if _monitoring is None:
        _monitoring = MonitoringRuntime(MonitoringConfig())
    return _monitoring
