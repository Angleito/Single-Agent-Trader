"""
Enhanced Functional Monitoring Effects for Trading Bot

This module provides comprehensive functional effects for metrics collection,
health checks, performance monitoring, alerting, and system observability.
All operations are pure functions that return IO effects for execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import psutil

from .io import IO

if TYPE_CHECKING:
    from collections.abc import Callable

A = TypeVar("A")
B = TypeVar("B")


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class MetricPoint:
    """Immutable metric data point"""

    name: str
    value: float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass(frozen=True)
class HealthCheck:
    """Immutable health check result"""

    component: str
    status: HealthStatus
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0


@dataclass(frozen=True)
class Alert:
    """Immutable alert event"""

    level: AlertLevel
    message: str
    component: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Span:
    """Immutable tracing span"""

    name: str
    start_time: float
    end_time: float | None = None
    tags: dict[str, str] = field(default_factory=dict)
    duration_ms: float | None = None

    def with_end_time(self, end_time: float) -> Span:
        """Return new span with end time and calculated duration"""
        duration = (end_time - self.start_time) * 1000
        return Span(
            name=self.name,
            start_time=self.start_time,
            end_time=end_time,
            tags=self.tags,
            duration_ms=duration,
        )


@dataclass(frozen=True)
class SystemMetrics:
    """Immutable system metrics snapshot"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    network_connections: int
    process_count: int
    uptime_seconds: float


# ==============================================================================
# Core Metrics Collection Effects
# ==============================================================================


def increment_counter(name: str, tags: dict[str, str] | None = None) -> IO[MetricPoint]:
    """Increment a counter metric and return the metric point"""

    def increment() -> MetricPoint:
        # In real implementation, this would interact with metrics backend
        return MetricPoint(
            name=name,
            value=1.0,
            timestamp=datetime.now(UTC),
            tags=tags or {},
            unit="count",
        )

    return IO(increment)


def record_gauge(
    name: str, value: float, tags: dict[str, str] | None = None, unit: str = ""
) -> IO[MetricPoint]:
    """Record a gauge metric and return the metric point"""

    def record() -> MetricPoint:
        return MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(UTC),
            tags=tags or {},
            unit=unit,
        )

    return IO(record)


def record_histogram(
    name: str, value: float, tags: dict[str, str] | None = None, unit: str = ""
) -> IO[MetricPoint]:
    """Record a histogram metric and return the metric point"""

    def record() -> MetricPoint:
        return MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(UTC),
            tags=tags or {},
            unit=unit,
        )

    return IO(record)


def record_timer(
    name: str, duration_ms: float, tags: dict[str, str] | None = None
) -> IO[MetricPoint]:
    """Record a timer metric and return the metric point"""

    def record() -> MetricPoint:
        return MetricPoint(
            name=name,
            value=duration_ms,
            timestamp=datetime.now(UTC),
            tags=tags or {},
            unit="ms",
        )

    return IO(record)


def record_multiple_metrics(metrics: list[MetricPoint]) -> IO[list[MetricPoint]]:
    """Record multiple metrics atomically"""

    def record() -> list[MetricPoint]:
        # In real implementation, this would batch metrics to backend
        return metrics

    return IO(record)


# ==============================================================================
# Health Check Effects
# ==============================================================================


def health_check(component: str, check_fn: Callable[[], bool]) -> IO[HealthCheck]:
    """Perform a health check on a component with custom check function"""

    def check() -> HealthCheck:
        start_time = time.perf_counter()
        try:
            is_healthy = check_fn()
            response_time = (time.perf_counter() - start_time) * 1000
            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
            return HealthCheck(
                component=component,
                status=status,
                timestamp=datetime.now(UTC),
                response_time_ms=response_time,
            )
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            return HealthCheck(
                component=component,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(UTC),
                details={"error": str(e)},
                response_time_ms=response_time,
            )

    return IO(check)


def system_health_check() -> IO[HealthCheck]:
    """Perform a system-wide health check"""

    def check() -> HealthCheck:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Determine health based on resource usage
            is_healthy = (
                cpu_percent < 90.0 and memory.percent < 90.0 and disk.percent < 95.0
            )

            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.DEGRADED

            return HealthCheck(
                component="system",
                status=status,
                timestamp=datetime.now(UTC),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                },
            )
        except Exception as e:
            return HealthCheck(
                component="system",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(UTC),
                details={"error": str(e)},
            )

    return IO(check)


def file_health_check(file_path: str | Path) -> IO[HealthCheck]:
    """Check if a file exists and is readable"""

    def check() -> HealthCheck:
        try:
            path = Path(file_path)
            is_healthy = path.exists() and path.is_file() and path.stat().st_size > 0
            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY

            return HealthCheck(
                component=f"file:{path.name}",
                status=status,
                timestamp=datetime.now(UTC),
                details={
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                },
            )
        except Exception as e:
            return HealthCheck(
                component=f"file:{Path(file_path).name}",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(UTC),
                details={"error": str(e)},
            )

    return IO(check)


# ==============================================================================
# System Metrics Collection Effects
# ==============================================================================


def collect_system_metrics() -> IO[SystemMetrics]:
    """Collect comprehensive system metrics"""

    def collect() -> SystemMetrics:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net_connections = len(psutil.net_connections())
            process_count = len(psutil.pids())
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time

            return SystemMetrics(
                timestamp=datetime.now(UTC),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / 1024 / 1024,
                disk_percent=disk.percent,
                network_connections=net_connections,
                process_count=process_count,
                uptime_seconds=uptime,
            )
        except Exception:
            # Return safe defaults on error
            return SystemMetrics(
                timestamp=datetime.now(UTC),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0,
                disk_percent=0.0,
                network_connections=0,
                process_count=0,
                uptime_seconds=0.0,
            )

    return IO(collect)


def collect_process_metrics(pid: int | None = None) -> IO[dict[str, MetricPoint]]:
    """Collect metrics for a specific process"""

    def collect() -> dict[str, MetricPoint]:
        try:
            process = psutil.Process(pid) if pid else psutil.Process()
            timestamp = datetime.now(UTC)

            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()

            return {
                "process_memory_rss": MetricPoint(
                    name="process_memory_rss",
                    value=memory_info.rss / 1024 / 1024,
                    timestamp=timestamp,
                    unit="MB",
                ),
                "process_memory_vms": MetricPoint(
                    name="process_memory_vms",
                    value=memory_info.vms / 1024 / 1024,
                    timestamp=timestamp,
                    unit="MB",
                ),
                "process_cpu_percent": MetricPoint(
                    name="process_cpu_percent",
                    value=cpu_percent,
                    timestamp=timestamp,
                    unit="%",
                ),
                "process_num_threads": MetricPoint(
                    name="process_num_threads",
                    value=float(process.num_threads()),
                    timestamp=timestamp,
                    unit="count",
                ),
                "process_num_fds": MetricPoint(
                    name="process_num_fds",
                    value=float(
                        process.num_fds() if hasattr(process, "num_fds") else 0
                    ),
                    timestamp=timestamp,
                    unit="count",
                ),
            }
        except Exception:
            # Return empty metrics on error
            return {}

    return IO(collect)


# ==============================================================================
# Tracing and Span Effects
# ==============================================================================


def start_span(name: str, tags: dict[str, str] | None = None) -> IO[Span]:
    """Start a tracing span"""

    def start() -> Span:
        return Span(name=name, start_time=time.perf_counter(), tags=tags or {})

    return IO(start)


def finish_span(span: Span) -> IO[Span]:
    """Finish a tracing span and calculate duration"""

    def finish() -> Span:
        end_time = time.perf_counter()
        return span.with_end_time(end_time)

    return IO(finish)


def span_to_metric(span: Span) -> IO[MetricPoint]:
    """Convert a finished span to a timing metric"""

    def convert() -> MetricPoint:
        duration = span.duration_ms or 0.0
        return MetricPoint(
            name=f"span.{span.name}.duration",
            value=duration,
            timestamp=datetime.now(UTC),
            tags=span.tags,
            unit="ms",
        )

    return IO(convert)


# ==============================================================================
# Alerting Effects
# ==============================================================================


def create_alert(
    level: AlertLevel,
    message: str,
    component: str,
    metric_name: str,
    current_value: float,
    threshold: float,
    tags: dict[str, str] | None = None,
) -> IO[Alert]:
    """Create an alert with complete context"""

    def create() -> Alert:
        return Alert(
            level=level,
            message=message,
            component=component,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.now(UTC),
            tags=tags or {},
        )

    return IO(create)


def send_alert(alert: Alert) -> IO[Alert]:
    """Send an alert through the notification system"""

    def send() -> Alert:
        # In real implementation, this would send to notification backends
        print(f"ALERT [{alert.level.value.upper()}] {alert.component}: {alert.message}")
        print(
            f"  Metric: {alert.metric_name} = {alert.current_value} (threshold: {alert.threshold})"
        )
        if alert.tags:
            print(f"  Tags: {alert.tags}")
        return alert

    return IO(send)


def alert_if_threshold_exceeded(
    metric: MetricPoint,
    threshold: float,
    component: str,
    message_template: str = "{metric_name} is {current_value} (threshold: {threshold})",
) -> IO[Alert | None]:
    """Create and send an alert if metric exceeds threshold"""

    def check_and_alert() -> Alert | None:
        if metric.value > threshold:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=message_template.format(
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=threshold,
                ),
                component=component,
                metric_name=metric.name,
                current_value=metric.value,
                threshold=threshold,
                timestamp=datetime.now(UTC),
                tags=metric.tags,
            )
            # Send the alert
            send_alert(alert).run()
            return alert
        return None

    return IO(check_and_alert)


# ==============================================================================
# Monitoring Combinators
# ==============================================================================


def monitor_with_metrics(operation_name: str) -> Callable[[IO[A]], IO[A]]:
    """Decorator that adds automatic metrics collection to an IO operation"""

    def decorator(io_operation: IO[A]) -> IO[A]:
        def monitored_operation() -> A:
            start_time = time.perf_counter()
            try:
                result = io_operation.run()
                duration = (time.perf_counter() - start_time) * 1000

                # Record success metrics
                record_timer(f"{operation_name}.duration", duration).run()
                increment_counter(f"{operation_name}.success").run()

                return result
            except Exception as e:
                duration = (time.perf_counter() - start_time) * 1000

                # Record error metrics
                record_timer(f"{operation_name}.duration", duration).run()
                increment_counter(
                    f"{operation_name}.error", {"error_type": type(e).__name__}
                ).run()

                raise

        return IO(monitored_operation)

    return decorator


def batch_health_checks(checks: list[IO[HealthCheck]]) -> IO[dict[str, HealthCheck]]:
    """Run multiple health checks and return results"""

    def run_checks() -> dict[str, HealthCheck]:
        results = {}
        for check in checks:
            try:
                result = check.run()
                results[result.component] = result
            except Exception as e:
                # Create failed health check for any exception
                results[f"unknown_component_{len(results)}"] = HealthCheck(
                    component="unknown",
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(UTC),
                    details={"error": str(e)},
                )
        return results

    return IO(run_checks)


def combine_metrics(*metric_effects: IO[MetricPoint]) -> IO[list[MetricPoint]]:
    """Combine multiple metric collection effects into one"""

    def collect_all() -> list[MetricPoint]:
        metrics = []
        for effect in metric_effects:
            try:
                metric = effect.run()
                metrics.append(metric)
            except Exception:
                # Skip failed metrics collection
                pass
        return metrics

    return IO(collect_all)


def derive_metric(
    source_metric: MetricPoint,
    derivation_fn: Callable[[float], float],
    new_name: str,
    new_unit: str = "",
) -> IO[MetricPoint]:
    """Derive a new metric from an existing one using a pure function"""

    def derive() -> MetricPoint:
        new_value = derivation_fn(source_metric.value)
        return MetricPoint(
            name=new_name,
            value=new_value,
            timestamp=source_metric.timestamp,
            tags=source_metric.tags,
            unit=new_unit,
        )

    return IO(derive)


# ==============================================================================
# Performance Analysis Effects
# ==============================================================================


def calculate_percentile(values: list[float], percentile: float) -> IO[float]:
    """Calculate percentile from a list of values"""

    def calculate() -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    return IO(calculate)


def calculate_statistics(values: list[float]) -> IO[dict[str, float]]:
    """Calculate comprehensive statistics from values"""

    def calculate() -> dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "std": 0}

        count = len(values)
        mean = sum(values) / count
        min_val = min(values)
        max_val = max(values)

        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in values) / count
        std_dev = variance**0.5

        return {
            "count": float(count),
            "mean": mean,
            "min": min_val,
            "max": max_val,
            "std": std_dev,
        }

    return IO(calculate)


def health_score(health_checks: dict[str, HealthCheck]) -> IO[float]:
    """Calculate overall health score from component health checks"""

    def calculate() -> float:
        if not health_checks:
            return 0.0

        total_components = len(health_checks)
        healthy_components = sum(
            1
            for check in health_checks.values()
            if check.status == HealthStatus.HEALTHY
        )

        return (healthy_components / total_components) * 100.0

    return IO(calculate)


# ==============================================================================
# Utility Functions for Monitoring
# ==============================================================================


def monitoring_context(name: str) -> Callable[[IO[A]], IO[A]]:
    """Create a monitoring context that tracks timing and health"""

    def context(operation: IO[A]) -> IO[A]:
        def monitored() -> A:
            # Start span
            span = start_span(f"context.{name}").run()

            try:
                # Execute operation
                result = operation.run()

                # Finish span and record metrics
                finished_span = finish_span(span).run()
                span_to_metric(finished_span).run()

                return result
            except Exception as e:
                # Record error and finish span
                finished_span = finish_span(span).run()
                span_to_metric(finished_span).run()
                increment_counter(
                    f"context.{name}.error", {"error_type": type(e).__name__}
                ).run()
                raise

        return IO(monitored)

    return context


def periodic_metrics_collection(
    metrics_fn: Callable[[], IO[list[MetricPoint]]], interval_seconds: float = 30.0
) -> IO[None]:
    """Set up periodic metrics collection (returns immediately)"""

    def setup() -> None:
        # In real implementation, this would set up a background task
        # For now, we just record the setup
        print(f"Setting up periodic metrics collection every {interval_seconds}s")

    return IO(setup)


def export_metrics_prometheus(metrics: list[MetricPoint]) -> IO[str]:
    """Export metrics in Prometheus format"""

    def export() -> str:
        lines = []
        for metric in metrics:
            # Add metric with labels
            if metric.tags:
                labels = ",".join(f'{k}="{v}"' for k, v in metric.tags.items())
                lines.append(f"{metric.name}{{{labels}}} {metric.value}")
            else:
                lines.append(f"{metric.name} {metric.value}")
        return "\n".join(lines)

    return IO(export)


def log_monitoring_summary(
    metrics: list[MetricPoint], health_checks: dict[str, HealthCheck]
) -> IO[None]:
    """Log a summary of monitoring data"""

    def log_summary() -> None:
        print(f"\n=== Monitoring Summary ({datetime.now(UTC).isoformat()}) ===")
        print(f"Metrics collected: {len(metrics)}")
        print(f"Health checks: {len(health_checks)}")

        # Health status summary
        healthy = sum(
            1 for hc in health_checks.values() if hc.status == HealthStatus.HEALTHY
        )
        degraded = sum(
            1 for hc in health_checks.values() if hc.status == HealthStatus.DEGRADED
        )
        unhealthy = sum(
            1 for hc in health_checks.values() if hc.status == HealthStatus.UNHEALTHY
        )

        print(f"Health: {healthy} healthy, {degraded} degraded, {unhealthy} unhealthy")

        # Recent metrics summary
        if metrics:
            print("Recent metrics:")
            for metric in metrics[-5:]:  # Show last 5 metrics
                print(f"  {metric.name}: {metric.value} {metric.unit}")

        print("=" * 50)

    return IO(log_summary)


def alert(
    level: AlertLevel,
    message: str,
    component: str = "system",
    metric_name: str = "alert",
    current_value: float = 0.0,
    threshold: float = 0.0,
    tags: dict[str, str] | None = None,
) -> IO[Alert]:
    """Create an alert - alias for create_alert for backward compatibility"""
    return create_alert(
        level, message, component, metric_name, current_value, threshold, tags
    )
