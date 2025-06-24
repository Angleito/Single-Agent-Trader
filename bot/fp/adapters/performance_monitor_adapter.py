"""
Functional Performance Monitor Adapter

This module provides a functional programming adapter for the existing
performance monitoring system, enhancing it with pure functions and
composable monitoring effects while preserving all existing APIs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..effects.io import IO
from ..effects.monitoring import (
    Alert,
    AlertLevel,
    HealthCheck,
    MetricPoint,
    SystemMetrics,
    alert_if_threshold_exceeded,
    batch_health_checks,
    collect_process_metrics,
    collect_system_metrics,
    combine_metrics,
    health_check,
    health_score,
    increment_counter,
    monitoring_context,
    record_gauge,
    record_histogram,
    record_timer,
    send_alert,
    system_health_check,
)
from ...performance_monitor import (
    PerformanceAlert,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceThresholds,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Immutable snapshot of monitoring state"""
    timestamp: datetime
    metrics: List[MetricPoint]
    health_checks: Dict[str, HealthCheck]
    alerts: List[Alert]
    system_metrics: SystemMetrics
    health_score_value: float


@dataclass(frozen=True)
class ThresholdConfig:
    """Immutable threshold configuration"""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    response_time_threshold: float = 2000.0
    error_rate_threshold: float = 5.0
    disk_threshold: float = 90.0


class MonitoringEffect(Protocol):
    """Protocol for monitoring effects"""
    
    def run(self) -> Any:
        """Execute the monitoring effect"""
        ...


@dataclass(frozen=True)
class PerformanceAnalysis:
    """Immutable performance analysis result"""
    timestamp: datetime
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    health_score: float
    trend_analysis: Dict[str, Any]
    metrics_summary: Dict[str, Any]


class FunctionalPerformanceMonitor:
    """
    Functional performance monitor that enhances the existing PerformanceMonitor
    with functional programming patterns while maintaining API compatibility.
    """

    def __init__(
        self, 
        legacy_monitor: Optional[PerformanceMonitor] = None,
        thresholds: Optional[ThresholdConfig] = None
    ):
        """Initialize with optional legacy monitor and thresholds"""
        self.legacy_monitor = legacy_monitor
        self.thresholds = thresholds or ThresholdConfig()
        self._monitoring_history: List[MonitoringSnapshot] = []
        self._max_history_size = 1000

    # ==============================================================================
    # Pure Functional Monitoring Operations
    # ==============================================================================

    def create_monitoring_snapshot(self) -> IO[MonitoringSnapshot]:
        """Create a complete monitoring snapshot using functional effects"""

        def create_snapshot() -> MonitoringSnapshot:
            # Collect system metrics
            sys_metrics = collect_system_metrics().run()
            
            # Collect process metrics
            process_metrics_dict = collect_process_metrics().run()
            process_metrics = list(process_metrics_dict.values())
            
            # Additional performance metrics
            additional_metrics = self._collect_performance_metrics().run()
            
            # Combine all metrics
            all_metrics = process_metrics + additional_metrics
            
            # Run health checks
            health_checks = self._run_all_health_checks().run()
            
            # Calculate health score
            score = health_score(health_checks).run()
            
            # Check for alerts
            alerts = self._check_thresholds(all_metrics, sys_metrics).run()
            
            return MonitoringSnapshot(
                timestamp=datetime.now(UTC),
                metrics=all_metrics,
                health_checks=health_checks,
                alerts=alerts,
                system_metrics=sys_metrics,
                health_score_value=score
            )

        return IO(create_snapshot)

    def _collect_performance_metrics(self) -> IO[List[MetricPoint]]:
        """Collect additional performance metrics"""

        def collect() -> List[MetricPoint]:
            metrics = []
            timestamp = datetime.now(UTC)
            
            # If legacy monitor exists, extract its metrics
            if self.legacy_monitor:
                try:
                    summary = self.legacy_monitor.get_performance_summary(timedelta(minutes=5))
                    
                    # Convert legacy metrics to functional metrics
                    if "latency_summary" in summary:
                        for metric_name, stats in summary["latency_summary"].items():
                            if stats and "mean" in stats:
                                metrics.append(MetricPoint(
                                    name=metric_name.replace(".", "_"),
                                    value=stats["mean"],
                                    timestamp=timestamp,
                                    unit="ms"
                                ))
                    
                    if "resource_summary" in summary:
                        for metric_name, stats in summary["resource_summary"].items():
                            if stats and "mean" in stats:
                                unit = "MB" if "memory" in metric_name else "%"
                                metrics.append(MetricPoint(
                                    name=metric_name.replace(".", "_"),
                                    value=stats["mean"],
                                    timestamp=timestamp,
                                    unit=unit
                                ))
                    
                    # Health score as metric
                    if "health_score" in summary:
                        metrics.append(MetricPoint(
                            name="system_health_score",
                            value=summary["health_score"],
                            timestamp=timestamp,
                            unit="score"
                        ))
                
                except Exception as e:
                    logger.debug(f"Could not extract legacy metrics: {e}")
            
            return metrics

        return IO(collect)

    def _run_all_health_checks(self) -> IO[Dict[str, HealthCheck]]:
        """Run comprehensive health checks"""

        checks = [
            system_health_check(),
            health_check("performance_monitor", lambda: self.legacy_monitor is not None),
            health_check("disk_space", self._check_disk_space),
            health_check("memory_usage", self._check_memory_usage),
        ]

        return batch_health_checks(checks)

    def _check_disk_space(self) -> bool:
        """Check if disk space is adequate"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return disk.percent < 95.0
        except Exception:
            return False

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is adequate"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90.0
        except Exception:
            return False

    def _check_thresholds(
        self, 
        metrics: List[MetricPoint], 
        system_metrics: SystemMetrics
    ) -> IO[List[Alert]]:
        """Check all metrics against thresholds and generate alerts"""

        def check_all() -> List[Alert]:
            alerts = []
            
            # Check system metrics thresholds
            if system_metrics.cpu_percent > self.thresholds.cpu_threshold:
                alert = alert_if_threshold_exceeded(
                    MetricPoint(
                        name="cpu_percent",
                        value=system_metrics.cpu_percent,
                        timestamp=system_metrics.timestamp,
                        unit="%"
                    ),
                    self.thresholds.cpu_threshold,
                    "system",
                    "High CPU usage: {current_value}% (threshold: {threshold}%)"
                ).run()
                if alert:
                    alerts.append(alert)
            
            if system_metrics.memory_percent > self.thresholds.memory_threshold:
                alert = alert_if_threshold_exceeded(
                    MetricPoint(
                        name="memory_percent",
                        value=system_metrics.memory_percent,
                        timestamp=system_metrics.timestamp,
                        unit="%"
                    ),
                    self.thresholds.memory_threshold,
                    "system",
                    "High memory usage: {current_value}% (threshold: {threshold}%)"
                ).run()
                if alert:
                    alerts.append(alert)
            
            if system_metrics.disk_percent > self.thresholds.disk_threshold:
                alert = alert_if_threshold_exceeded(
                    MetricPoint(
                        name="disk_percent",
                        value=system_metrics.disk_percent,
                        timestamp=system_metrics.timestamp,
                        unit="%"
                    ),
                    self.thresholds.disk_threshold,
                    "system",
                    "High disk usage: {current_value}% (threshold: {threshold}%)"
                ).run()
                if alert:
                    alerts.append(alert)
            
            # Check individual metrics
            for metric in metrics:
                if "response_time" in metric.name or "latency" in metric.name:
                    if metric.value > self.thresholds.response_time_threshold:
                        alert = alert_if_threshold_exceeded(
                            metric,
                            self.thresholds.response_time_threshold,
                            "performance",
                            "High response time: {current_value}ms (threshold: {threshold}ms)"
                        ).run()
                        if alert:
                            alerts.append(alert)
                
                elif "error_rate" in metric.name:
                    if metric.value > self.thresholds.error_rate_threshold:
                        alert = alert_if_threshold_exceeded(
                            metric,
                            self.thresholds.error_rate_threshold,
                            "system",
                            "High error rate: {current_value}% (threshold: {threshold}%)"
                        ).run()
                        if alert:
                            alerts.append(alert)
            
            return alerts

        return IO(check_all)

    # ==============================================================================
    # Performance Analysis Functions
    # ==============================================================================

    def analyze_performance(
        self, 
        duration: timedelta = timedelta(minutes=10)
    ) -> IO[PerformanceAnalysis]:
        """Analyze performance over a time period using functional approach"""

        def analyze() -> PerformanceAnalysis:
            cutoff_time = datetime.now(UTC) - duration
            
            # Get recent snapshots
            recent_snapshots = [
                snapshot for snapshot in self._monitoring_history
                if snapshot.timestamp >= cutoff_time
            ]
            
            if not recent_snapshots:
                # Return empty analysis if no data
                return PerformanceAnalysis(
                    timestamp=datetime.now(UTC),
                    bottlenecks=[],
                    recommendations=[],
                    health_score=0.0,
                    trend_analysis={},
                    metrics_summary={}
                )
            
            # Analyze bottlenecks
            bottlenecks = self._identify_bottlenecks(recent_snapshots)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(recent_snapshots, bottlenecks)
            
            # Calculate average health score
            avg_health_score = sum(s.health_score_value for s in recent_snapshots) / len(recent_snapshots)
            
            # Trend analysis
            trends = self._analyze_trends(recent_snapshots)
            
            # Metrics summary
            metrics_summary = self._summarize_metrics(recent_snapshots)
            
            return PerformanceAnalysis(
                timestamp=datetime.now(UTC),
                bottlenecks=bottlenecks,
                recommendations=recommendations,
                health_score=avg_health_score,
                trend_analysis=trends,
                metrics_summary=metrics_summary
            )

        return IO(analyze)

    def _identify_bottlenecks(self, snapshots: List[MonitoringSnapshot]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from snapshots"""
        bottlenecks = []
        
        if not snapshots:
            return bottlenecks
        
        # Analyze system metrics trends
        cpu_values = [s.system_metrics.cpu_percent for s in snapshots]
        memory_values = [s.system_metrics.memory_percent for s in snapshots]
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        
        if avg_cpu > self.thresholds.cpu_threshold:
            bottlenecks.append({
                "type": "cpu_usage",
                "severity": "high" if avg_cpu > self.thresholds.cpu_threshold * 1.2 else "medium",
                "average_value": avg_cpu,
                "threshold": self.thresholds.cpu_threshold,
                "description": f"Average CPU usage ({avg_cpu:.1f}%) exceeds threshold"
            })
        
        if avg_memory > self.thresholds.memory_threshold:
            bottlenecks.append({
                "type": "memory_usage",
                "severity": "high" if avg_memory > self.thresholds.memory_threshold * 1.1 else "medium",
                "average_value": avg_memory,
                "threshold": self.thresholds.memory_threshold,
                "description": f"Average memory usage ({avg_memory:.1f}%) exceeds threshold"
            })
        
        # Analyze response time patterns
        response_times = []
        for snapshot in snapshots:
            for metric in snapshot.metrics:
                if "response_time" in metric.name or "latency" in metric.name:
                    response_times.append(metric.value)
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            if avg_response_time > self.thresholds.response_time_threshold:
                bottlenecks.append({
                    "type": "response_time",
                    "severity": "high" if avg_response_time > self.thresholds.response_time_threshold * 2 else "medium",
                    "average_value": avg_response_time,
                    "threshold": self.thresholds.response_time_threshold,
                    "description": f"Average response time ({avg_response_time:.1f}ms) exceeds threshold"
                })
        
        return bottlenecks

    def _generate_recommendations(
        self, 
        snapshots: List[MonitoringSnapshot], 
        bottlenecks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "cpu_usage":
                if bottleneck["severity"] == "high":
                    recommendations.append("Critical: CPU usage is very high - consider scaling resources or optimizing CPU-intensive operations")
                else:
                    recommendations.append("Warning: CPU usage is elevated - monitor for sustained high usage patterns")
            
            elif bottleneck["type"] == "memory_usage":
                if bottleneck["severity"] == "high":
                    recommendations.append("Critical: Memory usage is very high - investigate memory leaks and implement cleanup strategies")
                else:
                    recommendations.append("Warning: Memory usage is elevated - monitor memory allocation patterns")
            
            elif bottleneck["type"] == "response_time":
                if bottleneck["severity"] == "high":
                    recommendations.append("Critical: Response times are very high - investigate performance bottlenecks immediately")
                else:
                    recommendations.append("Warning: Response times are elevated - consider optimizing slow operations")
        
        # General recommendations based on patterns
        if len(snapshots) > 5:
            alert_counts = [len(s.alerts) for s in snapshots]
            avg_alerts = sum(alert_counts) / len(alert_counts)
            
            if avg_alerts > 3:
                recommendations.append("High alert frequency detected - review alerting thresholds and address underlying issues")
        
        # Health score recommendations
        health_scores = [s.health_score_value for s in snapshots]
        avg_health = sum(health_scores) / len(health_scores)
        
        if avg_health < 70:
            recommendations.append("Overall system health is low - prioritize addressing unhealthy components")
        elif avg_health < 85:
            recommendations.append("System health could be improved - review component health checks")
        
        if not recommendations:
            recommendations.append("Performance metrics look good - continue monitoring for any changes")
        
        return recommendations

    def _analyze_trends(self, snapshots: List[MonitoringSnapshot]) -> Dict[str, Any]:
        """Analyze performance trends"""
        trends = {}
        
        if len(snapshots) < 2:
            return trends
        
        # CPU trend
        cpu_values = [s.system_metrics.cpu_percent for s in snapshots]
        cpu_trend = self._calculate_trend(cpu_values)
        trends["cpu_percent"] = {
            "direction": cpu_trend,
            "values": cpu_values,
            "change_percent": ((cpu_values[-1] - cpu_values[0]) / cpu_values[0] * 100) if cpu_values[0] != 0 else 0
        }
        
        # Memory trend
        memory_values = [s.system_metrics.memory_percent for s in snapshots]
        memory_trend = self._calculate_trend(memory_values)
        trends["memory_percent"] = {
            "direction": memory_trend,
            "values": memory_values,
            "change_percent": ((memory_values[-1] - memory_values[0]) / memory_values[0] * 100) if memory_values[0] != 0 else 0
        }
        
        # Health score trend
        health_values = [s.health_score_value for s in snapshots]
        health_trend = self._calculate_trend(health_values)
        trends["health_score"] = {
            "direction": health_trend,
            "values": health_values,
            "change_percent": ((health_values[-1] - health_values[0]) / health_values[0] * 100) if health_values[0] != 0 else 0
        }
        
        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation based on first vs last values
        change_percent = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        
        if abs(change_percent) < 5:
            return "stable"
        elif change_percent > 0:
            return "increasing"
        else:
            return "decreasing"

    def _summarize_metrics(self, snapshots: List[MonitoringSnapshot]) -> Dict[str, Any]:
        """Summarize metrics from snapshots"""
        summary = {}
        
        if not snapshots:
            return summary
        
        # System metrics summary
        cpu_values = [s.system_metrics.cpu_percent for s in snapshots]
        memory_values = [s.system_metrics.memory_percent for s in snapshots]
        
        summary["system"] = {
            "cpu_percent": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values)
            },
            "memory_percent": {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": sum(memory_values) / len(memory_values)
            }
        }
        
        # Metrics by type
        metric_groups = {}
        for snapshot in snapshots:
            for metric in snapshot.metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)
        
        summary["metrics"] = {}
        for name, values in metric_groups.items():
            summary["metrics"][name] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "count": len(values)
            }
        
        return summary

    # ==============================================================================
    # Integration with Legacy Monitor
    # ==============================================================================

    def track_operation(self, operation_name: str) -> Callable[[IO[Any]], IO[Any]]:
        """Functional wrapper for operation tracking"""
        return monitoring_context(operation_name)

    def record_custom_metric(self, name: str, value: float, unit: str = "") -> IO[MetricPoint]:
        """Record a custom metric functionally"""
        return record_gauge(name, value, unit=unit)

    def add_to_history(self, snapshot: MonitoringSnapshot) -> IO[None]:
        """Add snapshot to monitoring history"""

        def add() -> None:
            self._monitoring_history.append(snapshot)
            # Keep history size bounded
            if len(self._monitoring_history) > self._max_history_size:
                self._monitoring_history = self._monitoring_history[-self._max_history_size:]

        return IO(add)

    async def start_functional_monitoring(self, interval_seconds: float = 30.0) -> None:
        """Start functional monitoring loop"""
        
        async def monitoring_loop():
            while True:
                try:
                    # Create monitoring snapshot
                    snapshot = self.create_monitoring_snapshot().run()
                    
                    # Add to history
                    self.add_to_history(snapshot).run()
                    
                    # Send any alerts
                    for alert in snapshot.alerts:
                        send_alert(alert).run()
                    
                    # Log summary periodically (every 10 snapshots)
                    if len(self._monitoring_history) % 10 == 0:
                        logger.info(
                            f"Monitoring snapshot: {len(snapshot.metrics)} metrics, "
                            f"health score: {snapshot.health_score_value:.1f}, "
                            f"{len(snapshot.alerts)} alerts"
                        )
                    
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in functional monitoring loop: {e}")
                    await asyncio.sleep(interval_seconds)
        
        # Start the monitoring loop
        asyncio.create_task(monitoring_loop())
        logger.info(f"Started functional monitoring with {interval_seconds}s interval")

    def get_latest_snapshot(self) -> Optional[MonitoringSnapshot]:
        """Get the most recent monitoring snapshot"""
        return self._monitoring_history[-1] if self._monitoring_history else None

    def get_snapshots_since(self, since: datetime) -> List[MonitoringSnapshot]:
        """Get snapshots since a specific time"""
        return [
            snapshot for snapshot in self._monitoring_history
            if snapshot.timestamp >= since
        ]

    def export_functional_metrics(self) -> IO[Dict[str, Any]]:
        """Export functional metrics in a structured format"""

        def export() -> Dict[str, Any]:
            latest_snapshot = self.get_latest_snapshot()
            if not latest_snapshot:
                return {}
            
            return {
                "timestamp": latest_snapshot.timestamp.isoformat(),
                "health_score": latest_snapshot.health_score_value,
                "system_metrics": {
                    "cpu_percent": latest_snapshot.system_metrics.cpu_percent,
                    "memory_percent": latest_snapshot.system_metrics.memory_percent,
                    "memory_mb": latest_snapshot.system_metrics.memory_mb,
                    "disk_percent": latest_snapshot.system_metrics.disk_percent,
                    "network_connections": latest_snapshot.system_metrics.network_connections,
                    "uptime_seconds": latest_snapshot.system_metrics.uptime_seconds,
                },
                "metrics_count": len(latest_snapshot.metrics),
                "health_checks": {
                    name: {
                        "status": check.status.value,
                        "response_time_ms": check.response_time_ms
                    }
                    for name, check in latest_snapshot.health_checks.items()
                },
                "alerts_count": len(latest_snapshot.alerts),
                "history_size": len(self._monitoring_history)
            }

        return IO(export)


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_functional_monitor(
    legacy_monitor: Optional[PerformanceMonitor] = None,
    thresholds: Optional[ThresholdConfig] = None
) -> FunctionalPerformanceMonitor:
    """Factory function to create a functional performance monitor"""
    return FunctionalPerformanceMonitor(legacy_monitor, thresholds)


def enhance_existing_monitor(
    performance_monitor: PerformanceMonitor
) -> FunctionalPerformanceMonitor:
    """Enhance an existing performance monitor with functional capabilities"""
    return FunctionalPerformanceMonitor(performance_monitor)