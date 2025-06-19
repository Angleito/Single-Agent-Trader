"""
Enhanced Balance Operation Monitoring and Correlation System.

This module provides unified monitoring, error correlation, and performance tracking
for balance operations across both the Bluefin SDK service and client, with integrated
metrics collection and alerting capabilities.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .secure_logging import get_balance_operation_context

# Import new monitoring components
try:
    from ..monitoring.balance_alerts import get_balance_alert_manager
    from ..monitoring.balance_metrics import (
        get_balance_metrics_collector,
        record_operation_complete,
        record_operation_start,
        record_timeout,
    )
    ENHANCED_MONITORING_AVAILABLE = True
except ImportError:
    ENHANCED_MONITORING_AVAILABLE = False


@dataclass
class BalanceOperationEvent:
    """Represents a balance operation event for correlation tracking."""

    correlation_id: str
    component: str  # 'service' or 'client'
    operation: str
    status: str  # 'started', 'success', 'failed', 'timeout'
    timestamp: datetime
    duration_ms: float | None = None
    balance_amount: str | None = None
    error: str | None = None
    error_category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for balance operations."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_counts_by_category: dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


class BalanceOperationMonitor:
    """
    Centralized monitoring system for balance operations across services.

    Provides correlation tracking, performance metrics, and error analysis
    for both the SDK service and client components.
    """

    def __init__(self, max_events: int = 10000, metrics_window_minutes: int = 60):
        """
        Initialize the balance operation monitor.

        Args:
            max_events: Maximum number of events to keep in memory
            metrics_window_minutes: Time window for metrics calculation
        """
        self.max_events = max_events
        self.metrics_window_minutes = metrics_window_minutes

        # Event storage and correlation
        self.events: deque = deque(maxlen=max_events)
        self.active_operations: dict[str, BalanceOperationEvent] = {}
        self.completed_operations: dict[str, list[BalanceOperationEvent]] = defaultdict(
            list
        )

        # Performance metrics by component and operation
        self.metrics: dict[str, dict[str, PerformanceMetrics]] = defaultdict(
            lambda: defaultdict(PerformanceMetrics)
        )

        # Error correlation tracking
        self.error_patterns: dict[str, list[BalanceOperationEvent]] = defaultdict(list)
        self.cross_component_errors: list[dict[str, Any]] = []

        # Logger for this monitor
        self.logger = logging.getLogger(__name__)

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def record_operation_start(
        self,
        correlation_id: str,
        component: str,
        operation: str,
        metadata: dict[str, Any] = None,
    ) -> float:
        """
        Record the start of a balance operation.

        Args:
            correlation_id: Unique operation identifier
            component: Component name ('service' or 'client')
            operation: Operation name
            metadata: Additional operation metadata
            
        Returns:
            Start time for enhanced metrics collection
        """
        # Record in enhanced metrics system if available
        start_time = None
        if ENHANCED_MONITORING_AVAILABLE:
            try:
                start_time = record_operation_start(
                    operation=operation,
                    component=component,
                    correlation_id=correlation_id,
                    metadata=metadata
                )
            except Exception as e:
                self.logger.debug(f"Enhanced metrics recording failed: {e}")

        async with self._lock:
            event = BalanceOperationEvent(
                correlation_id=correlation_id,
                component=component,
                operation=operation,
                status="started",
                timestamp=datetime.now(UTC),
                metadata=metadata or {},
            )

            self.events.append(event)
            self.active_operations[correlation_id] = event

            self.logger.debug(
                f"Balance operation started: {component}.{operation}",
                extra=get_balance_operation_context(
                    correlation_id,
                    operation,
                    additional_context={
                        "component": component,
                        "event_type": "operation_start",
                        "metadata": metadata,
                    },
                ),
            )

        return start_time or time.perf_counter()

    async def record_operation_complete(
        self,
        correlation_id: str,
        status: str,
        balance_amount: str = None,
        error: str = None,
        error_category: str = None,
        metadata: dict[str, Any] = None,
        start_time: float | None = None,
    ) -> None:
        """
        Record the completion of a balance operation.

        Args:
            correlation_id: Unique operation identifier
            status: Final status ('success', 'failed', 'timeout')
            balance_amount: Balance amount if successful
            error: Error message if failed
            error_category: Category of error if failed
            metadata: Additional completion metadata
            start_time: Start time from record_operation_start for enhanced metrics
        """
        # Record in enhanced metrics system if available
        if ENHANCED_MONITORING_AVAILABLE and start_time is not None:
            try:
                # Parse balance amounts for enhanced metrics
                balance_before = None
                balance_after = None
                if balance_amount:
                    try:
                        balance_after = float(balance_amount.replace("$", "").replace(",", ""))
                    except (ValueError, AttributeError):
                        pass

                record_operation_complete(
                    operation="unknown",  # Will be determined from start_event
                    component="unknown",  # Will be determined from start_event
                    start_time=start_time,
                    success=(status == "success"),
                    balance_before=balance_before,
                    balance_after=balance_after,
                    error_type=error_category,
                    correlation_id=correlation_id,
                    metadata=metadata
                )
            except Exception as e:
                self.logger.debug(f"Enhanced metrics completion recording failed: {e}")

        async with self._lock:
            start_event = self.active_operations.pop(correlation_id, None)
            if not start_event:
                self.logger.warning(
                    f"No start event found for correlation_id: {correlation_id}",
                    extra={"correlation_id": correlation_id, "status": status},
                )
                return

            # Calculate duration
            now = datetime.now(UTC)
            duration_ms = (now - start_event.timestamp).total_seconds() * 1000

            # Update enhanced metrics with correct operation details if available
            if ENHANCED_MONITORING_AVAILABLE and start_time is not None:
                try:
                    balance_before = None
                    balance_after = None
                    if balance_amount:
                        try:
                            balance_after = float(balance_amount.replace("$", "").replace(",", ""))
                        except (ValueError, AttributeError):
                            pass

                    record_operation_complete(
                        operation=start_event.operation,
                        component=start_event.component,
                        start_time=start_time,
                        success=(status == "success"),
                        balance_before=balance_before,
                        balance_after=balance_after,
                        error_type=error_category,
                        correlation_id=correlation_id,
                        metadata=metadata
                    )
                except Exception as e:
                    self.logger.debug(f"Enhanced metrics final recording failed: {e}")

            # Create completion event
            completion_event = BalanceOperationEvent(
                correlation_id=correlation_id,
                component=start_event.component,
                operation=start_event.operation,
                status=status,
                timestamp=now,
                duration_ms=duration_ms,
                balance_amount=balance_amount,
                error=error,
                error_category=error_category,
                metadata={**start_event.metadata, **(metadata or {})},
            )

            self.events.append(completion_event)
            self.completed_operations[correlation_id] = [start_event, completion_event]

            # Update performance metrics
            await self._update_performance_metrics(completion_event)

            # Check for error patterns
            if status == "failed" and error_category:
                await self._track_error_pattern(completion_event)

            # Log completion
            self.logger.info(
                f"Balance operation completed: {start_event.component}.{start_event.operation}",
                extra=get_balance_operation_context(
                    correlation_id,
                    start_event.operation,
                    balance_amount=balance_amount,
                    duration_ms=duration_ms,
                    success=(status == "success"),
                    error_category=error_category,
                    additional_context={
                        "component": start_event.component,
                        "event_type": "operation_complete",
                        "final_status": status,
                        "metadata": completion_event.metadata,
                    },
                ),
            )

    async def _update_performance_metrics(self, event: BalanceOperationEvent) -> None:
        """Update performance metrics for the completed operation."""
        metrics = self.metrics[event.component][event.operation]

        metrics.total_operations += 1
        metrics.last_updated = event.timestamp

        if event.status == "success":
            metrics.successful_operations += 1
        else:
            metrics.failed_operations += 1

            # Track error categories
            if event.error_category:
                if event.error_category not in metrics.error_counts_by_category:
                    metrics.error_counts_by_category[event.error_category] = 0
                metrics.error_counts_by_category[event.error_category] += 1

        # Update response time metrics
        if event.duration_ms is not None:
            metrics.response_times.append(event.duration_ms)

            # Recalculate timing statistics
            times = list(metrics.response_times)
            if times:
                metrics.average_response_time = sum(times) / len(times)
                metrics.min_response_time = min(times)
                metrics.max_response_time = max(times)

                # Calculate percentiles
                sorted_times = sorted(times)
                if len(sorted_times) >= 20:  # Need enough samples for percentiles
                    p95_idx = int(len(sorted_times) * 0.95)
                    p99_idx = int(len(sorted_times) * 0.99)
                    metrics.p95_response_time = sorted_times[p95_idx]
                    metrics.p99_response_time = sorted_times[p99_idx]

    async def _track_error_pattern(self, event: BalanceOperationEvent) -> None:
        """Track error patterns for correlation analysis."""
        pattern_key = f"{event.component}:{event.operation}:{event.error_category}"
        self.error_patterns[pattern_key].append(event)

        # Keep only recent errors (last 100 per pattern)
        if len(self.error_patterns[pattern_key]) > 100:
            self.error_patterns[pattern_key] = self.error_patterns[pattern_key][-100:]

        # Check for cross-component correlation
        await self._check_cross_component_correlation(event)

    async def _check_cross_component_correlation(
        self, event: BalanceOperationEvent
    ) -> None:
        """Check for correlated errors across components."""
        # Look for errors in the other component within the last 30 seconds
        threshold_time = event.timestamp.timestamp() - 30  # 30 seconds ago

        other_component = "client" if event.component == "service" else "service"

        # Find recent errors in the other component
        recent_other_errors = [
            e
            for e in self.events
            if (
                e.component == other_component
                and e.status == "failed"
                and e.timestamp.timestamp() > threshold_time
            )
        ]

        if recent_other_errors:
            correlation = {
                "trigger_event": {
                    "correlation_id": event.correlation_id,
                    "component": event.component,
                    "operation": event.operation,
                    "error_category": event.error_category,
                    "timestamp": event.timestamp.isoformat(),
                },
                "correlated_errors": [
                    {
                        "correlation_id": e.correlation_id,
                        "component": e.component,
                        "operation": e.operation,
                        "error_category": e.error_category,
                        "timestamp": e.timestamp.isoformat(),
                        "time_diff_seconds": (
                            event.timestamp - e.timestamp
                        ).total_seconds(),
                    }
                    for e in recent_other_errors
                ],
                "detected_at": datetime.now(UTC).isoformat(),
            }

            self.cross_component_errors.append(correlation)

            # Keep only recent correlations (last 1000)
            if len(self.cross_component_errors) > 1000:
                self.cross_component_errors = self.cross_component_errors[-1000:]

            self.logger.warning(
                "Cross-component error correlation detected",
                extra={
                    "correlation_detected": True,
                    "trigger_component": event.component,
                    "correlated_component": other_component,
                    "correlation_count": len(recent_other_errors),
                    "correlation_details": correlation,
                },
            )

    def get_performance_summary(
        self, component: str = None, operation: str = None
    ) -> dict[str, Any]:
        """
        Get performance summary for balance operations.

        Args:
            component: Filter by component ('service' or 'client')
            operation: Filter by operation name

        Returns:
            Performance summary dictionary
        """
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_events": len(self.events),
            "active_operations": len(self.active_operations),
            "completed_operations": len(self.completed_operations),
            "metrics": {},
        }

        # Aggregate metrics
        if component and operation:
            # Specific component and operation
            if component in self.metrics and operation in self.metrics[component]:
                metrics = self.metrics[component][operation]
                summary["metrics"][f"{component}.{operation}"] = self._format_metrics(
                    metrics
                )
        elif component:
            # All operations for a component
            if component in self.metrics:
                for op, metrics in self.metrics[component].items():
                    summary["metrics"][f"{component}.{op}"] = self._format_metrics(
                        metrics
                    )
        else:
            # All components and operations
            for comp, operations in self.metrics.items():
                for op, metrics in operations.items():
                    summary["metrics"][f"{comp}.{op}"] = self._format_metrics(metrics)

        return summary

    def _format_metrics(self, metrics: PerformanceMetrics) -> dict[str, Any]:
        """Format performance metrics for output."""
        success_rate = 0.0
        if metrics.total_operations > 0:
            success_rate = (
                metrics.successful_operations / metrics.total_operations
            ) * 100

        return {
            "total_operations": metrics.total_operations,
            "successful_operations": metrics.successful_operations,
            "failed_operations": metrics.failed_operations,
            "success_rate_percent": round(success_rate, 2),
            "average_response_time_ms": round(metrics.average_response_time, 2),
            "min_response_time_ms": (
                round(metrics.min_response_time, 2)
                if metrics.min_response_time != float("inf")
                else 0
            ),
            "max_response_time_ms": round(metrics.max_response_time, 2),
            "p95_response_time_ms": round(metrics.p95_response_time, 2),
            "p99_response_time_ms": round(metrics.p99_response_time, 2),
            "error_counts_by_category": dict(metrics.error_counts_by_category),
            "last_updated": metrics.last_updated.isoformat(),
        }

    def get_error_analysis(self) -> dict[str, Any]:
        """Get error analysis and correlation report."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "error_patterns": {
                pattern: len(events) for pattern, events in self.error_patterns.items()
            },
            "cross_component_correlations": len(self.cross_component_errors),
            "recent_cross_component_errors": self.cross_component_errors[
                -10:
            ],  # Last 10
            "top_error_categories": self._get_top_error_categories(),
        }

    def _get_top_error_categories(self) -> list[dict[str, Any]]:
        """Get top error categories across all components."""
        error_totals = defaultdict(int)

        for component_metrics in self.metrics.values():
            for operation_metrics in component_metrics.values():
                for (
                    category,
                    count,
                ) in operation_metrics.error_counts_by_category.items():
                    error_totals[category] += count

        # Sort by count and return top 10
        sorted_errors = sorted(error_totals.items(), key=lambda x: x[1], reverse=True)
        return [
            {"category": category, "total_count": count}
            for category, count in sorted_errors[:10]
        ]

    async def cleanup_old_events(self) -> None:
        """Clean up old events and metrics outside the time window."""
        cutoff_time = datetime.now(UTC).timestamp() - (self.metrics_window_minutes * 60)

        async with self._lock:
            # Remove old events
            while self.events and self.events[0].timestamp.timestamp() < cutoff_time:
                self.events.popleft()

            # Clean up old error patterns
            for pattern_key in list(self.error_patterns.keys()):
                self.error_patterns[pattern_key] = [
                    event
                    for event in self.error_patterns[pattern_key]
                    if event.timestamp.timestamp() > cutoff_time
                ]
                if not self.error_patterns[pattern_key]:
                    del self.error_patterns[pattern_key]

            # Clean up old cross-component errors
            self.cross_component_errors = [
                correlation
                for correlation in self.cross_component_errors
                if datetime.fromisoformat(
                    correlation["detected_at"].replace("Z", "+00:00")
                ).timestamp()
                > cutoff_time
            ]


# Global monitor instance
_global_monitor: BalanceOperationMonitor | None = None


def get_balance_monitor() -> BalanceOperationMonitor:
    """Get the global balance operation monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = BalanceOperationMonitor()
    return _global_monitor


async def record_balance_operation_start(
    correlation_id: str,
    component: str,
    operation: str,
    metadata: dict[str, Any] = None,
) -> float:
    """Convenience function to record balance operation start."""
    monitor = get_balance_monitor()
    return await monitor.record_operation_start(correlation_id, component, operation, metadata)


async def record_balance_operation_complete(
    correlation_id: str,
    status: str,
    balance_amount: str = None,
    error: str = None,
    error_category: str = None,
    metadata: dict[str, Any] = None,
    start_time: float | None = None,
) -> None:
    """Convenience function to record balance operation completion."""
    monitor = get_balance_monitor()
    await monitor.record_operation_complete(
        correlation_id, status, balance_amount, error, error_category, metadata, start_time
    )


# Background cleanup task
async def start_monitoring_cleanup_task(interval_minutes: int = 15) -> None:
    """Start background task to clean up old monitoring data."""
    monitor = get_balance_monitor()

    while True:
        try:
            await asyncio.sleep(interval_minutes * 60)
            await monitor.cleanup_old_events()
        except asyncio.CancelledError:
            break
        except Exception as e:
            monitor.logger.error(f"Error in monitoring cleanup task: {e}")


def generate_monitoring_report() -> dict[str, Any]:
    """Generate comprehensive monitoring report with enhanced metrics."""
    monitor = get_balance_monitor()

    base_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "performance_summary": monitor.get_performance_summary(),
        "error_analysis": monitor.get_error_analysis(),
        "system_health": {
            "total_events_tracked": len(monitor.events),
            "active_operations": len(monitor.active_operations),
            "memory_usage_events": len(monitor.events),
            "memory_usage_patterns": sum(
                len(events) for events in monitor.error_patterns.values()
            ),
            "memory_usage_correlations": len(monitor.cross_component_errors),
        },
    }

    # Add enhanced metrics if available
    if ENHANCED_MONITORING_AVAILABLE:
        try:
            from ..monitoring.balance_alerts import get_balance_alert_manager
            from ..monitoring.balance_metrics import get_metrics_summary

            enhanced_metrics = get_metrics_summary()
            alert_manager = get_balance_alert_manager()
            alert_summary = alert_manager.get_alert_summary()

            base_report["enhanced_metrics"] = enhanced_metrics
            base_report["alert_summary"] = alert_summary
            base_report["monitoring_features"] = {
                "enhanced_metrics_enabled": True,
                "alerting_enabled": True,
                "prometheus_metrics_available": True
            }
        except Exception as e:
            logger.warning(f"Failed to include enhanced metrics in report: {e}")
            base_report["monitoring_features"] = {
                "enhanced_metrics_enabled": False,
                "alerting_enabled": False,
                "prometheus_metrics_available": False,
                "error": str(e)
            }
    else:
        base_report["monitoring_features"] = {
            "enhanced_metrics_enabled": False,
            "alerting_enabled": False,
            "prometheus_metrics_available": False,
            "reason": "Enhanced monitoring components not available"
        }

    return base_report


# Enhanced monitoring integration functions
def get_enhanced_metrics_summary() -> dict[str, Any]:
    """Get enhanced metrics summary if available."""
    if ENHANCED_MONITORING_AVAILABLE:
        try:
            from ..monitoring.balance_metrics import get_metrics_summary
            return get_metrics_summary()
        except Exception as e:
            logger.error(f"Failed to get enhanced metrics summary: {e}")
            return {"error": str(e)}
    return {"error": "Enhanced monitoring not available"}


def get_prometheus_metrics() -> list[str]:
    """Get Prometheus format metrics if available."""
    if ENHANCED_MONITORING_AVAILABLE:
        try:
            from ..monitoring.balance_metrics import get_prometheus_metrics
            return get_prometheus_metrics()
        except Exception as e:
            logger.error(f"Failed to get Prometheus metrics: {e}")
            return []
    return []


async def trigger_alert_evaluation() -> list:
    """Trigger alert evaluation if alerting is available."""
    if ENHANCED_MONITORING_AVAILABLE:
        try:
            from ..monitoring.balance_alerts import trigger_alert_evaluation
            return await trigger_alert_evaluation()
        except Exception as e:
            logger.error(f"Failed to trigger alert evaluation: {e}")
            return []
    return []


def enable_enhanced_monitoring() -> bool:
    """
    Enable enhanced monitoring and alerting if available.
    
    Returns:
        True if enhanced monitoring was successfully enabled
    """
    if ENHANCED_MONITORING_AVAILABLE:
        try:
            # Initialize enhanced components
            from ..monitoring.balance_alerts import get_balance_alert_manager
            from ..monitoring.balance_metrics import get_balance_metrics_collector

            metrics_collector = get_balance_metrics_collector()
            alert_manager = get_balance_alert_manager()

            logger.info("✅ Enhanced balance monitoring enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable enhanced monitoring: {e}")
            return False
    else:
        logger.warning("Enhanced monitoring components not available")
        return False
