"""
Real-time performance monitoring for the AI Trading Bot.

This module provides comprehensive performance monitoring including real-time
metrics collection, latency tracking, resource usage monitoring, performance
alerting, and bottleneck identification.
"""

import asyncio
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, NamedTuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Performance alert levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    value: float
    timestamp: datetime
    unit: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert."""

    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            "level": self.level.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


class TimingContext(NamedTuple):
    """Context for timing operations."""

    name: str
    start_time: float
    tags: dict[str, str]


class PerformanceThresholds:
    """Performance threshold configuration."""

    def __init__(self):
        # Latency thresholds (milliseconds)
        self.indicator_calculation_ms = 100
        self.llm_response_ms = 5000
        self.market_data_processing_ms = 50
        self.trade_execution_ms = 1000

        # Throughput thresholds (operations per second)
        self.min_market_data_ops_per_sec = 10
        self.min_indicator_calculations_per_sec = 5

        # Resource thresholds
        self.max_memory_usage_mb = 2048
        self.max_cpu_usage_percent = 80
        self.max_memory_growth_rate_mb_per_min = 50

        # Error thresholds
        self.max_error_rate_percent = 5
        self.max_consecutive_errors = 10


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, max_history_size: int = 1000):
        """
        Initialize the metrics collector.

        Args:
            max_history_size: Maximum number of metrics to keep in memory
        """
        self.max_history_size = max_history_size
        self._metrics_history: dict[str, deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self._lock = threading.Lock()

        # Running statistics
        self._metric_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "count": 0.0,
                "sum": 0.0,
                "sum_squares": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }
        )

    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric."""
        with self._lock:
            self._metrics_history[metric.name].append(metric)

            # Update running statistics
            stats = self._metric_stats[metric.name]
            stats["count"] += 1.0
            stats["sum"] += metric.value
            stats["sum_squares"] += metric.value**2
            stats["min"] = min(stats["min"], metric.value)
            stats["max"] = max(stats["max"], metric.value)

    def get_metric_history(
        self, metric_name: str, duration: timedelta | None = None
    ) -> list[PerformanceMetric]:
        """Get metric history for a specific metric."""
        with self._lock:
            metrics = list(self._metrics_history[metric_name])

            if duration:
                cutoff_time = datetime.utcnow() - duration
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            return metrics

    def get_metric_statistics(
        self, metric_name: str, duration: timedelta | None = None
    ) -> dict[str, float]:
        """Get statistical summary for a metric."""
        metrics = self.get_metric_history(metric_name, duration)

        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "p95": float(np.percentile(values, 95)) if values else 0.0,
            "p99": float(np.percentile(values, 99)) if values else 0.0,
        }

    def get_all_metrics(self) -> dict[str, list[PerformanceMetric]]:
        """Get all metrics."""
        with self._lock:
            return {
                name: list(history) for name, history in self._metrics_history.items()
            }


class LatencyTracker:
    """Tracks operation latencies with timing contexts."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize the latency tracker."""
        self.metrics_collector = metrics_collector
        self._active_timings: dict[str, TimingContext] = {}
        self._lock = threading.Lock()

    @contextmanager
    def track_operation(self, operation_name: str, tags: dict[str, str] | None = None):
        """
        Context manager for tracking operation latency.

        Args:
            operation_name: Name of the operation
            tags: Additional tags for the metric

        Usage:
            with latency_tracker.track_operation("indicator_calculation"):
                calculate_indicators()
        """
        start_time = time.perf_counter()
        timing_id = str(id(threading.current_thread()))

        with self._lock:
            self._active_timings[timing_id] = TimingContext(
                name=operation_name, start_time=start_time, tags=tags or {}
            )

        try:
            yield
        finally:
            end_time = time.perf_counter()

            with self._lock:
                timing_context = self._active_timings.pop(timing_id, None)

            if timing_context:
                latency_ms = (end_time - timing_context.start_time) * 1000

                metric = PerformanceMetric(
                    name=f"latency.{timing_context.name}",
                    value=latency_ms,
                    timestamp=datetime.utcnow(),
                    unit="milliseconds",
                    tags=timing_context.tags,
                )

                self.metrics_collector.add_metric(metric)

    def track_async_operation(
        self, operation_name: str, tags: dict[str, str] | None = None
    ):
        """
        Decorator for tracking async operation latency.

        Args:
            operation_name: Name of the operation
            tags: Additional tags for the metric
        """

        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                with self.track_operation(operation_name, tags):
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    def track_sync_operation(
        self, operation_name: str, tags: dict[str, str] | None = None
    ):
        """
        Decorator for tracking sync operation latency.

        Args:
            operation_name: Name of the operation
            tags: Additional tags for the metric
        """

        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                with self.track_operation(operation_name, tags):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


class ResourceMonitor:
    """Monitors system resource usage."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize the resource monitor."""
        self.metrics_collector = metrics_collector
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_task: asyncio.Task[None] | None = None
        self._baseline_memory: float | None = None

    async def start_monitoring(self, interval_seconds: float = 5.0):
        """
        Start continuous resource monitoring.

        Args:
            interval_seconds: Monitoring interval
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._baseline_memory = self.process.memory_info().rss / 1024 / 1024
        self._monitor_task = asyncio.create_task(
            self._monitor_resources(interval_seconds)
        )
        logger.info("Started resource monitoring")

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task
        logger.info("Stopped resource monitoring")

    async def _monitor_resources(self, interval: float):
        """Internal resource monitoring loop."""
        while self._monitoring:
            try:
                timestamp = datetime.utcnow()

                # Memory metrics
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = self.process.memory_percent()

                self.metrics_collector.add_metric(
                    PerformanceMetric(
                        name="resource.memory_usage_mb",
                        value=memory_mb,
                        timestamp=timestamp,
                        unit="megabytes",
                    )
                )

                self.metrics_collector.add_metric(
                    PerformanceMetric(
                        name="resource.memory_percent",
                        value=memory_percent,
                        timestamp=timestamp,
                        unit="percent",
                    )
                )

                # Memory growth rate
                if self._baseline_memory:
                    growth_mb = memory_mb - self._baseline_memory
                    self.metrics_collector.add_metric(
                        PerformanceMetric(
                            name="resource.memory_growth_mb",
                            value=growth_mb,
                            timestamp=timestamp,
                            unit="megabytes",
                        )
                    )

                # CPU metrics
                cpu_percent = self.process.cpu_percent()
                if cpu_percent > 0:  # Only record non-zero CPU usage
                    self.metrics_collector.add_metric(
                        PerformanceMetric(
                            name="resource.cpu_percent",
                            value=cpu_percent,
                            timestamp=timestamp,
                            unit="percent",
                        )
                    )

                # System-wide metrics
                system_memory = psutil.virtual_memory()
                self.metrics_collector.add_metric(
                    PerformanceMetric(
                        name="resource.system_memory_percent",
                        value=system_memory.percent,
                        timestamp=timestamp,
                        unit="percent",
                    )
                )

                system_cpu = psutil.cpu_percent()
                self.metrics_collector.add_metric(
                    PerformanceMetric(
                        name="resource.system_cpu_percent",
                        value=system_cpu,
                        timestamp=timestamp,
                        unit="percent",
                    )
                )

                # Thread count
                thread_count = threading.active_count()
                self.metrics_collector.add_metric(
                    PerformanceMetric(
                        name="resource.thread_count",
                        value=thread_count,
                        timestamp=timestamp,
                        unit="count",
                    )
                )

            except Exception as e:
                logger.warning(f"Error monitoring resources: {e}")

            await asyncio.sleep(interval)


class AlertManager:
    """Manages performance alerts and notifications."""

    def __init__(self, thresholds: PerformanceThresholds):
        """Initialize the alert manager."""
        self.thresholds = thresholds
        self.alerts: deque[PerformanceAlert] = deque(maxlen=1000)
        self._alert_callbacks: list[Callable[[PerformanceAlert], None]] = []
        self._last_alert_times: dict[str, datetime] = defaultdict(lambda: datetime.min)
        self._alert_cooldown = timedelta(minutes=5)

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add a callback for alert notifications."""
        self._alert_callbacks.append(callback)

    def check_metric_thresholds(self, metric: PerformanceMetric):
        """Check metric against thresholds and generate alerts if needed."""
        alerts = []

        # Latency threshold checks
        if metric.name == "latency.indicator_calculation":
            if metric.value > self.thresholds.indicator_calculation_ms:
                alerts.append(
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"Indicator calculation latency high: {metric.value:.1f}ms",
                        metric.name,
                        metric.value,
                        self.thresholds.indicator_calculation_ms,
                        metric.tags,
                    )
                )

        elif metric.name == "latency.llm_response":
            if metric.value > self.thresholds.llm_response_ms:
                level = (
                    AlertLevel.CRITICAL
                    if metric.value > self.thresholds.llm_response_ms * 2
                    else AlertLevel.WARNING
                )
                alerts.append(
                    self._create_alert(
                        level,
                        f"LLM response latency high: {metric.value:.1f}ms",
                        metric.name,
                        metric.value,
                        self.thresholds.llm_response_ms,
                        metric.tags,
                    )
                )

        elif metric.name == "latency.market_data_processing":
            if metric.value > self.thresholds.market_data_processing_ms:
                alerts.append(
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"Market data processing latency high: {metric.value:.1f}ms",
                        metric.name,
                        metric.value,
                        self.thresholds.market_data_processing_ms,
                        metric.tags,
                    )
                )

        # Resource threshold checks
        elif metric.name == "resource.memory_usage_mb":
            if metric.value > self.thresholds.max_memory_usage_mb:
                level = (
                    AlertLevel.CRITICAL
                    if metric.value > self.thresholds.max_memory_usage_mb * 1.5
                    else AlertLevel.WARNING
                )
                alerts.append(
                    self._create_alert(
                        level,
                        f"Memory usage high: {metric.value:.1f}MB",
                        metric.name,
                        metric.value,
                        self.thresholds.max_memory_usage_mb,
                        metric.tags,
                    )
                )

        elif metric.name == "resource.cpu_percent":
            if metric.value > self.thresholds.max_cpu_usage_percent:
                alerts.append(
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"CPU usage high: {metric.value:.1f}%",
                        metric.name,
                        metric.value,
                        self.thresholds.max_cpu_usage_percent,
                        metric.tags,
                    )
                )

        # Process alerts
        for alert in alerts:
            self._process_alert(alert)

    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        tags: dict[str, str],
    ) -> PerformanceAlert:
        """Create a performance alert."""
        return PerformanceAlert(
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.utcnow(),
            tags=tags,
        )

    def _process_alert(self, alert: PerformanceAlert):
        """Process an alert with cooldown logic."""
        # Check cooldown period
        last_alert_time = self._last_alert_times[alert.metric_name]
        if datetime.utcnow() - last_alert_time < self._alert_cooldown:
            return

        # Record alert
        self.alerts.append(alert)
        self._last_alert_times[alert.metric_name] = alert.timestamp

        # Log alert
        log_level = (
            logging.WARNING if alert.level == AlertLevel.WARNING else logging.ERROR
        )
        logger.log(log_level, f"Performance Alert: {alert.message}")

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.exception(f"Alert callback failed: {e}")

    def get_recent_alerts(
        self, duration: timedelta = timedelta(hours=1)
    ) -> list[PerformanceAlert]:
        """Get recent alerts within the specified duration."""
        cutoff_time = datetime.utcnow() - duration
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]


class BottleneckAnalyzer:
    """Analyzes performance bottlenecks."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize the bottleneck analyzer."""
        self.metrics_collector = metrics_collector

    def analyze_bottlenecks(
        self, duration: timedelta = timedelta(minutes=10)
    ) -> dict[str, Any]:
        """
        Analyze performance bottlenecks over a time period.

        Args:
            duration: Time period to analyze

        Returns:
            Dictionary with bottleneck analysis
        """
        analysis: dict[str, Any] = {
            "analysis_period": duration.total_seconds(),
            "timestamp": datetime.utcnow(),
            "bottlenecks": [],
            "recommendations": [],
        }

        # Analyze latency patterns
        latency_metrics = [
            "latency.indicator_calculation",
            "latency.llm_response",
            "latency.market_data_processing",
            "latency.trade_execution",
        ]

        for metric_name in latency_metrics:
            stats = self.metrics_collector.get_metric_statistics(metric_name, duration)
            if stats and stats["count"] > 5:  # Require minimum sample size
                # Check for high latency
                if stats["p95"] > stats["mean"] * 2:
                    analysis["bottlenecks"].append(
                        {
                            "type": "latency_spike",
                            "metric": metric_name,
                            "p95_latency_ms": stats["p95"],
                            "mean_latency_ms": stats["mean"],
                            "severity": (
                                "high" if stats["p95"] > stats["mean"] * 3 else "medium"
                            ),
                        }
                    )

                # Check for high variability
                if stats["std_dev"] > stats["mean"] * 0.5:
                    analysis["bottlenecks"].append(
                        {
                            "type": "latency_variability",
                            "metric": metric_name,
                            "std_dev_ms": stats["std_dev"],
                            "mean_latency_ms": stats["mean"],
                            "coefficient_of_variation": stats["std_dev"]
                            / stats["mean"],
                        }
                    )

        # Analyze resource usage patterns
        resource_metrics = [
            "resource.memory_usage_mb",
            "resource.cpu_percent",
            "resource.memory_growth_mb",
        ]

        for metric_name in resource_metrics:
            stats = self.metrics_collector.get_metric_statistics(metric_name, duration)
            if stats and stats["count"] > 5:
                # Check for resource exhaustion
                if metric_name == "resource.memory_usage_mb" and stats["max"] > 1024:
                    analysis["bottlenecks"].append(
                        {
                            "type": "memory_usage",
                            "metric": metric_name,
                            "max_memory_mb": stats["max"],
                            "mean_memory_mb": stats["mean"],
                            "growth_trend": (
                                "increasing"
                                if stats["max"] > stats["mean"] * 1.2
                                else "stable"
                            ),
                        }
                    )

                elif metric_name == "resource.cpu_percent" and stats["mean"] > 70:
                    analysis["bottlenecks"].append(
                        {
                            "type": "cpu_usage",
                            "metric": metric_name,
                            "mean_cpu_percent": stats["mean"],
                            "max_cpu_percent": stats["max"],
                        }
                    )

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(
            analysis["bottlenecks"]
        )

        return analysis

    def _generate_recommendations(self, bottlenecks: list[dict[str, Any]]) -> list[str]:
        """Generate optimization recommendations based on bottlenecks."""
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "latency_spike":
                if "indicator_calculation" in bottleneck["metric"]:
                    recommendations.append(
                        "Consider optimizing indicator calculations with vectorization or caching"
                    )
                elif "llm_response" in bottleneck["metric"]:
                    recommendations.append(
                        "Consider reducing LLM prompt complexity or implementing response caching"
                    )
                elif "market_data_processing" in bottleneck["metric"]:
                    recommendations.append(
                        "Consider optimizing DataFrame operations or reducing data processing frequency"
                    )

            elif bottleneck["type"] == "memory_usage":
                recommendations.append(
                    "Implement data cleanup strategies and consider memory pooling for large datasets"
                )

            elif bottleneck["type"] == "cpu_usage":
                recommendations.append(
                    "Consider distributing CPU-intensive operations or optimizing algorithms"
                )

            elif bottleneck["type"] == "latency_variability":
                recommendations.append(
                    "Investigate sources of latency variance and implement consistent processing paths"
                )

        # Remove duplicates
        return list(set(recommendations))


class PerformanceMonitor:
    """
    Main performance monitoring system.

    Integrates all monitoring components to provide comprehensive
    performance tracking and alerting for the trading bot.
    """

    def __init__(self, thresholds: PerformanceThresholds | None = None):
        """Initialize the performance monitor."""
        self.thresholds = thresholds or PerformanceThresholds()
        self.metrics_collector = MetricsCollector()
        self.latency_tracker = LatencyTracker(self.metrics_collector)
        self.resource_monitor = ResourceMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.thresholds)
        self.bottleneck_analyzer = BottleneckAnalyzer(self.metrics_collector)

        # Connect metrics to alert checking by storing original method
        self._original_add_metric = self.metrics_collector.add_metric

        self._monitoring = False

    def add_metric(self, metric: PerformanceMetric) -> None:
        """Add metric with automatic alert checking."""
        self._original_add_metric(metric)
        self.alert_manager.check_metric_thresholds(metric)

    async def start_monitoring(self, resource_monitor_interval: float = 5.0):
        """
        Start comprehensive performance monitoring.

        Args:
            resource_monitor_interval: Resource monitoring interval in seconds
        """
        if self._monitoring:
            return

        self._monitoring = True
        await self.resource_monitor.start_monitoring(resource_monitor_interval)
        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        await self.resource_monitor.stop_monitoring()
        logger.info("Performance monitoring stopped")

    def get_performance_summary(
        self, duration: timedelta = timedelta(minutes=10)
    ) -> dict[str, Any]:
        """
        Get comprehensive performance summary.

        Args:
            duration: Time period to summarize

        Returns:
            Performance summary dictionary
        """
        summary: dict[str, Any] = {
            "timestamp": datetime.utcnow(),
            "period_minutes": duration.total_seconds() / 60,
            "latency_summary": {},
            "resource_summary": {},
            "recent_alerts": [],
            "bottleneck_analysis": {},
            "health_score": 0.0,
        }

        # Latency summary
        latency_metrics = [
            "latency.indicator_calculation",
            "latency.llm_response",
            "latency.market_data_processing",
        ]

        for metric in latency_metrics:
            stats = self.metrics_collector.get_metric_statistics(metric, duration)
            if stats:
                summary["latency_summary"][metric] = stats

        # Resource summary
        resource_metrics = [
            "resource.memory_usage_mb",
            "resource.cpu_percent",
            "resource.memory_growth_mb",
        ]

        for metric in resource_metrics:
            stats = self.metrics_collector.get_metric_statistics(metric, duration)
            if stats:
                summary["resource_summary"][metric] = stats

        # Recent alerts - convert to dict format
        recent_alerts = self.alert_manager.get_recent_alerts(duration)
        summary["recent_alerts"] = [
            {
                "level": alert.level.value,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat(),
                "tags": alert.tags,
            }
            for alert in recent_alerts
        ]

        # Bottleneck analysis
        summary["bottleneck_analysis"] = self.bottleneck_analyzer.analyze_bottlenecks(
            duration
        )

        # Calculate health score
        summary["health_score"] = self._calculate_health_score(summary)

        return summary

    def _calculate_health_score(self, summary: dict[str, Any]) -> float:
        """
        Calculate overall system health score (0-100).

        Args:
            summary: Performance summary

        Returns:
            Health score between 0 and 100
        """
        score = 100.0

        # Deduct for latency issues
        latency_summary = summary.get("latency_summary", {})
        for metric, stats in latency_summary.items():
            if (
                "indicator_calculation" in metric
                and stats.get("p95", 0) > self.thresholds.indicator_calculation_ms
            ):
                score -= 10
            elif (
                "llm_response" in metric
                and stats.get("p95", 0) > self.thresholds.llm_response_ms
            ):
                score -= 15
            elif (
                "market_data_processing" in metric
                and stats.get("p95", 0) > self.thresholds.market_data_processing_ms
            ):
                score -= 10

        # Deduct for resource issues
        resource_summary = summary.get("resource_summary", {})
        memory_stats = resource_summary.get("resource.memory_usage_mb", {})
        if memory_stats.get("max", 0) > self.thresholds.max_memory_usage_mb:
            score -= 20

        cpu_stats = resource_summary.get("resource.cpu_percent", {})
        if cpu_stats.get("mean", 0) > self.thresholds.max_cpu_usage_percent:
            score -= 15

        # Deduct for alerts
        alert_count = len(summary.get("recent_alerts", []))
        critical_alerts = sum(
            1
            for alert in summary.get("recent_alerts", [])
            if alert.get("level") == "critical"
        )
        score -= min(alert_count * 2, 20)  # Max 20 points for alerts
        score -= critical_alerts * 10  # Additional penalty for critical alerts

        # Deduct for bottlenecks
        bottleneck_count = len(
            summary.get("bottleneck_analysis", {}).get("bottlenecks", [])
        )
        score -= min(bottleneck_count * 5, 15)  # Max 15 points for bottlenecks

        return max(0.0, min(100.0, score))

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts."""
        self.alert_manager.add_alert_callback(callback)

    # Convenience methods for tracking operations
    def track_operation(self, operation_name: str, tags: dict[str, str] | None = None):
        """Track operation latency."""
        return self.latency_tracker.track_operation(operation_name, tags)

    def track_async_operation(
        self, operation_name: str, tags: dict[str, str] | None = None
    ):
        """Decorator for tracking async operations."""
        return self.latency_tracker.track_async_operation(operation_name, tags)

    def track_sync_operation(
        self, operation_name: str, tags: dict[str, str] | None = None
    ):
        """Decorator for tracking sync operations."""
        return self.latency_tracker.track_sync_operation(operation_name, tags)


# Global performance monitor instance
_performance_monitor: PerformanceMonitor | None = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def init_performance_monitoring(
    thresholds: PerformanceThresholds | None = None,
) -> PerformanceMonitor:
    """
    Initialize global performance monitoring.

    Args:
        thresholds: Performance thresholds configuration

    Returns:
        Performance monitor instance
    """
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(thresholds)
    return _performance_monitor


# Convenience decorators using global monitor
def track_async(operation_name: str, tags: dict[str, str] | None = None):
    """Convenience decorator for tracking async operations."""
    return get_performance_monitor().track_async_operation(operation_name, tags)


def track_sync(operation_name: str, tags: dict[str, str] | None = None):
    """Convenience decorator for tracking sync operations."""
    return get_performance_monitor().track_sync_operation(operation_name, tags)


@contextmanager
def track(operation_name: str, tags: dict[str, str] | None = None):
    """Convenience context manager for tracking operations."""
    with get_performance_monitor().track_operation(operation_name, tags):
        yield
