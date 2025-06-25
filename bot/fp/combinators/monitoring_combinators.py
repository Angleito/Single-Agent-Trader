"""
Monitoring Combinators for Functional Trading Bot

This module provides functional combinators for composing complex monitoring
operations, health checks, and metric calculations. All combinators are pure
functions that return composed IO effects.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeVar

from bot.fp.effects.io import IO
from bot.fp.effects.monitoring import HealthCheck, HealthStatus, MetricPoint

if TYPE_CHECKING:
    from collections.abc import Callable

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


@dataclass(frozen=True)
class CompositeHealthCheck:
    """Immutable composite health check result"""

    name: str
    component_checks: dict[str, HealthCheck]
    overall_status: HealthStatus
    timestamp: datetime
    aggregation_method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricTransformation:
    """Immutable metric transformation result"""

    source_metrics: list[MetricPoint]
    result_metric: MetricPoint
    transformation_type: str
    timestamp: datetime
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MonitoringPipeline:
    """Immutable monitoring pipeline configuration"""

    name: str
    stages: list[str]
    config: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Basic Combinators
# ==============================================================================


def map_io[A, B](f: Callable[[A], B]) -> Callable[[IO[A]], IO[B]]:
    """Map a pure function over an IO operation"""

    def mapper(io_a: IO[A]) -> IO[B]:
        def mapped_operation() -> B:
            result_a = io_a.run()
            return f(result_a)

        return IO(mapped_operation)

    return mapper


def sequence_io[A](io_operations: list[IO[A]]) -> IO[list[A]]:
    """Sequence a list of IO operations into a single IO operation"""

    def sequenced_operation() -> list[A]:
        results = []
        for io_op in io_operations:
            result = io_op.run()
            results.append(result)
        return results

    return IO(sequenced_operation)


def parallel_io[A](io_operations: list[IO[A]]) -> IO[list[A]]:
    """Execute IO operations in parallel (simulation for sync operations)"""

    def parallel_operation() -> list[A]:
        # For actual parallel execution, this would use threading or asyncio
        # For simplicity, we execute sequentially but could be enhanced
        results = []
        for io_op in io_operations:
            try:
                result = io_op.run()
                results.append(result)
            except Exception as e:
                # Log error but continue with other operations
                print(f"Error in parallel operation: {e}")
        return results

    return IO(parallel_operation)


def chain_io(io_a: IO[A], f: Callable[[A], IO[B]]) -> IO[B]:
    """Chain IO operations with monadic bind"""

    def chained_operation() -> B:
        result_a = io_a.run()
        io_b = f(result_a)
        return io_b.run()

    return IO(chained_operation)


def filter_io[A](
    predicate: Callable[[A], bool],
) -> Callable[[IO[list[A]]], IO[list[A]]]:
    """Filter results of an IO operation"""

    def filter_operation(io_list: IO[list[A]]) -> IO[list[A]]:
        def filtered() -> list[A]:
            results = io_list.run()
            return [item for item in results if predicate(item)]

        return IO(filtered)

    return filter_operation


def fold_io(initial: B, combine: Callable[[B, A], B]) -> Callable[[IO[list[A]]], IO[B]]:
    """Fold over results of an IO operation"""

    def fold_operation(io_list: IO[list[A]]) -> IO[B]:
        def folded() -> B:
            results = io_list.run()
            return reduce(combine, results, initial)

        return IO(folded)

    return fold_operation


# ==============================================================================
# Health Check Combinators
# ==============================================================================


def combine_health_checks(
    checks: list[IO[HealthCheck]], aggregation: str = "worst"
) -> IO[CompositeHealthCheck]:
    """Combine multiple health checks with specified aggregation strategy"""

    def combine_checks() -> CompositeHealthCheck:
        check_results = {}
        statuses = []

        for check_io in checks:
            try:
                check_result = check_io.run()
                check_results[check_result.component] = check_result
                statuses.append(check_result.status)
            except Exception as e:
                # Create failed health check
                failed_check = HealthCheck(
                    component="unknown",
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(UTC),
                    details={"error": str(e)},
                )
                check_results[f"failed_{len(check_results)}"] = failed_check
                statuses.append(HealthStatus.UNHEALTHY)

        # Aggregate status based on strategy
        overall_status = _aggregate_health_statuses(statuses, aggregation)

        return CompositeHealthCheck(
            name="composite_health_check",
            component_checks=check_results,
            overall_status=overall_status,
            timestamp=datetime.now(UTC),
            aggregation_method=aggregation,
            metadata={
                "total_checks": len(check_results),
                "healthy_count": sum(1 for s in statuses if s == HealthStatus.HEALTHY),
                "degraded_count": sum(
                    1 for s in statuses if s == HealthStatus.DEGRADED
                ),
                "unhealthy_count": sum(
                    1 for s in statuses if s == HealthStatus.UNHEALTHY
                ),
            },
        )

    return IO(combine_checks)


def _aggregate_health_statuses(
    statuses: list[HealthStatus], method: str
) -> HealthStatus:
    """Aggregate health statuses using specified method"""
    if not statuses:
        return HealthStatus.UNHEALTHY

    if method == "worst":
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    if method == "best":
        if HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.UNHEALTHY

    if method == "majority":
        status_counts = defaultdict(int)
        for status in statuses:
            status_counts[status] += 1

        # Return most common status
        return max(status_counts.items(), key=lambda x: x[1])[0]

    if method == "threshold":
        # Require majority to be healthy
        healthy_count = sum(1 for s in statuses if s == HealthStatus.HEALTHY)
        if healthy_count >= len(statuses) * 0.6:
            return HealthStatus.HEALTHY
        if healthy_count >= len(statuses) * 0.3:
            return HealthStatus.DEGRADED
        return HealthStatus.UNHEALTHY

    # Default to worst
    return _aggregate_health_statuses(statuses, "worst")


def health_check_with_timeout(
    check: IO[HealthCheck], timeout_seconds: float
) -> IO[HealthCheck]:
    """Add timeout to a health check"""

    def timed_check() -> HealthCheck:
        start_time = time.perf_counter()

        try:
            # Simple timeout simulation (in real implementation would use threading/asyncio)
            result = check.run()

            duration = (time.perf_counter() - start_time) * 1000

            # Update response time
            return HealthCheck(
                component=result.component,
                status=result.status,
                timestamp=result.timestamp,
                details=result.details,
                response_time_ms=duration,
            )

        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            return HealthCheck(
                component="timeout_check",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(UTC),
                details={"error": str(e), "timeout_seconds": timeout_seconds},
                response_time_ms=duration,
            )

    return IO(timed_check)


def retry_health_check(
    check: IO[HealthCheck], max_retries: int = 3, delay_seconds: float = 1.0
) -> IO[HealthCheck]:
    """Retry a health check with exponential backoff"""

    def retried_check() -> HealthCheck:
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = check.run()
                if result.status != HealthStatus.UNHEALTHY:
                    return result
                last_error = result.details.get("error", "Check returned unhealthy")
            except Exception as e:
                last_error = str(e)

            if attempt < max_retries:
                # Exponential backoff simulation
                time.sleep(delay_seconds * (2**attempt))

        # All retries failed
        return HealthCheck(
            component="retry_check",
            status=HealthStatus.UNHEALTHY,
            timestamp=datetime.now(UTC),
            details={
                "error": f"All {max_retries + 1} attempts failed",
                "last_error": last_error,
            },
        )

    return IO(retried_check)


def conditional_health_check(
    condition: IO[bool], check_if_true: IO[HealthCheck], check_if_false: IO[HealthCheck]
) -> IO[HealthCheck]:
    """Conditionally execute health checks based on a condition"""

    def conditional_check() -> HealthCheck:
        condition_result = condition.run()
        if condition_result:
            return check_if_true.run()
        return check_if_false.run()

    return IO(conditional_check)


# ==============================================================================
# Metric Combinators
# ==============================================================================


def combine_metrics(
    metrics: list[IO[MetricPoint]], combination_type: str = "sum"
) -> IO[MetricPoint]:
    """Combine multiple metrics into a single metric"""

    def combine() -> MetricPoint:
        metric_results = []

        for metric_io in metrics:
            try:
                metric = metric_io.run()
                metric_results.append(metric)
            except Exception as e:
                print(f"Error collecting metric: {e}")

        if not metric_results:
            return MetricPoint(
                name="combined_metric_error",
                value=0.0,
                timestamp=datetime.now(UTC),
                unit="error",
            )

        # Combine values based on type
        values = [m.value for m in metric_results]

        if combination_type == "sum":
            combined_value = sum(values)
        elif combination_type == "average":
            combined_value = sum(values) / len(values)
        elif combination_type == "max":
            combined_value = max(values)
        elif combination_type == "min":
            combined_value = min(values)
        elif combination_type == "median":
            combined_value = statistics.median(values)
        else:
            combined_value = sum(values)  # Default to sum

        # Use the first metric's metadata as base
        base_metric = metric_results[0]

        return MetricPoint(
            name=f"combined_{combination_type}",
            value=combined_value,
            timestamp=datetime.now(UTC),
            tags={**base_metric.tags, "combination_type": combination_type},
            unit=base_metric.unit,
        )

    return IO(combine)


def transform_metric(
    metric: IO[MetricPoint],
    transformation: Callable[[float], float],
    new_name: str | None = None,
    new_unit: str | None = None,
) -> IO[MetricPoint]:
    """Transform a metric value using a pure function"""

    def transform() -> MetricPoint:
        source_metric = metric.run()
        transformed_value = transformation(source_metric.value)

        return MetricPoint(
            name=new_name or f"{source_metric.name}_transformed",
            value=transformed_value,
            timestamp=source_metric.timestamp,
            tags={**source_metric.tags, "transformed": "true"},
            unit=new_unit or source_metric.unit,
        )

    return IO(transform)


def rolling_window_metric(
    metric: IO[MetricPoint],
    history: list[MetricPoint],
    window_size: int,
    aggregation: str = "average",
) -> IO[MetricPoint]:
    """Calculate rolling window statistics for a metric"""

    def calculate_rolling() -> MetricPoint:
        current_metric = metric.run()

        # Add current metric to history and maintain window size
        updated_history = [*history, current_metric]
        if len(updated_history) > window_size:
            updated_history = updated_history[-window_size:]

        # Calculate aggregation
        values = [m.value for m in updated_history]

        if aggregation == "average":
            aggregated_value = sum(values) / len(values)
        elif aggregation == "sum":
            aggregated_value = sum(values)
        elif aggregation == "max":
            aggregated_value = max(values)
        elif aggregation == "min":
            aggregated_value = min(values)
        elif aggregation == "std":
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                aggregated_value = variance**0.5
            else:
                aggregated_value = 0.0
        else:
            aggregated_value = sum(values) / len(values)  # Default to average

        return MetricPoint(
            name=f"{current_metric.name}_rolling_{aggregation}",
            value=aggregated_value,
            timestamp=current_metric.timestamp,
            tags={
                **current_metric.tags,
                "window_size": str(window_size),
                "aggregation": aggregation,
            },
            unit=current_metric.unit,
        )

    return IO(calculate_rolling)


def rate_of_change_metric(
    current_metric: IO[MetricPoint],
    previous_metric: MetricPoint | None = None,
    _time_window_seconds: float = 60.0,
) -> IO[MetricPoint]:
    """Calculate rate of change for a metric"""

    def calculate_rate() -> MetricPoint:
        current = current_metric.run()

        if previous_metric is None:
            # No previous metric, rate is 0
            rate = 0.0
        else:
            # Calculate rate of change
            time_diff = (current.timestamp - previous_metric.timestamp).total_seconds()
            if time_diff > 0:
                value_diff = current.value - previous_metric.value
                rate = value_diff / time_diff
            else:
                rate = 0.0

        return MetricPoint(
            name=f"{current.name}_rate",
            value=rate,
            timestamp=current.timestamp,
            tags={**current.tags, "rate_calculation": "true"},
            unit=f"{current.unit}/s",
        )

    return IO(calculate_rate)


def threshold_metric(
    metric: IO[MetricPoint], threshold: float, comparison: str = ">"
) -> IO[MetricPoint]:
    """Create a binary metric based on threshold comparison"""

    def calculate_threshold() -> MetricPoint:
        source_metric = metric.run()

        if comparison == ">":
            threshold_exceeded = source_metric.value > threshold
        elif comparison == "<":
            threshold_exceeded = source_metric.value < threshold
        elif comparison == ">=":
            threshold_exceeded = source_metric.value >= threshold
        elif comparison == "<=":
            threshold_exceeded = source_metric.value <= threshold
        elif comparison == "==":
            threshold_exceeded = abs(source_metric.value - threshold) < 1e-9
        elif comparison == "!=":
            threshold_exceeded = abs(source_metric.value - threshold) >= 1e-9
        else:
            threshold_exceeded = False

        return MetricPoint(
            name=f"{source_metric.name}_threshold",
            value=1.0 if threshold_exceeded else 0.0,
            timestamp=source_metric.timestamp,
            tags={
                **source_metric.tags,
                "threshold": str(threshold),
                "comparison": comparison,
            },
            unit="binary",
        )

    return IO(calculate_threshold)


# ==============================================================================
# Advanced Combinators
# ==============================================================================


def monitoring_pipeline[A, B](
    stages: list[Callable[[A], IO[B]]],
) -> Callable[[A], IO[B]]:
    """Create a monitoring pipeline from a list of transformation stages"""

    def pipeline(initial_input: A) -> IO[B]:
        def execute_pipeline() -> B:
            current_value = initial_input

            for stage in stages:
                stage_io = stage(current_value)
                current_value = stage_io.run()

            return current_value

        return IO(execute_pipeline)

    return pipeline


def circuit_breaker_monitoring[A](
    operation: IO[A],
    failure_threshold: int = 5,
    recovery_timeout_seconds: float = 60.0,
    failure_tracker: dict[str, Any] | None = None,
) -> IO[A | None]:
    """Apply circuit breaker pattern to monitoring operations"""

    if failure_tracker is None:
        failure_tracker = {"failures": 0, "last_failure": None, "state": "closed"}

    def circuit_breaker_operation() -> A | None:
        current_time = datetime.now(UTC)

        # Check circuit breaker state
        if failure_tracker["state"] == "open":
            # Check if recovery timeout has passed
            if failure_tracker["last_failure"]:
                time_since_failure = (
                    current_time - failure_tracker["last_failure"]
                ).total_seconds()
                if time_since_failure >= recovery_timeout_seconds:
                    failure_tracker["state"] = "half_open"
                else:
                    # Circuit still open, return None
                    return None

        try:
            # Execute operation
            result = operation.run()

            # Success - reset failure count
            if failure_tracker["state"] == "half_open":
                failure_tracker["state"] = "closed"
            failure_tracker["failures"] = 0
            failure_tracker["last_failure"] = None

            return result

        except Exception as e:
            # Failure - increment counter
            failure_tracker["failures"] += 1
            failure_tracker["last_failure"] = current_time

            # Check if threshold exceeded
            if failure_tracker["failures"] >= failure_threshold:
                failure_tracker["state"] = "open"

            print(f"Circuit breaker recorded failure: {e}")
            return None

    return IO(circuit_breaker_operation)


def cached_monitoring(
    operation: IO[A],
    cache_duration_seconds: float = 60.0,
    cache_key: str = "default",
    cache_storage: dict[str, tuple[datetime, A]] | None = None,
) -> IO[A]:
    """Add caching to monitoring operations"""

    if cache_storage is None:
        cache_storage = {}

    def cached_operation() -> A:
        current_time = datetime.now(UTC)

        # Check cache
        if cache_key in cache_storage:
            cached_time, cached_value = cache_storage[cache_key]
            if (current_time - cached_time).total_seconds() < cache_duration_seconds:
                return cached_value

        # Cache miss or expired - execute operation
        result = operation.run()

        # Store in cache
        cache_storage[cache_key] = (current_time, result)

        return result

    return IO(cached_operation)


def fan_out_monitoring(
    input_operation: IO[A], transformations: list[Callable[[A], IO[B]]]
) -> IO[list[B]]:
    """Fan out a single input to multiple transformations"""

    def fan_out() -> list[B]:
        input_result = input_operation.run()
        results = []

        for transformation in transformations:
            try:
                transformed_io = transformation(input_result)
                result = transformed_io.run()
                results.append(result)
            except Exception as e:
                print(f"Error in fan-out transformation: {e}")

        return results

    return IO(fan_out)


def conditional_monitoring(
    condition: IO[bool], if_true: IO[A], if_false: IO[A]
) -> IO[A]:
    """Conditionally execute monitoring operations"""

    def conditional_execution() -> A:
        condition_result = condition.run()
        if condition_result:
            return if_true.run()
        return if_false.run()

    return IO(conditional_execution)


def monitoring_with_fallback(primary: IO[A], fallback: IO[A]) -> IO[A]:
    """Execute primary monitoring with fallback on failure"""

    def with_fallback() -> A:
        try:
            return primary.run()
        except Exception:
            return fallback.run()

    return IO(with_fallback)


# ==============================================================================
# Monitoring State Combinators
# ==============================================================================


def stateful_monitoring(
    operation: IO[A],
    state_updater: Callable[[B | None, A], B],
    initial_state: B | None = None,
    state_storage: dict[str, B] | None = None,
) -> IO[tuple[A, B]]:
    """Add stateful behavior to monitoring operations"""

    if state_storage is None:
        state_storage = {"state": initial_state}

    def stateful_operation() -> tuple[A, B]:
        # Execute operation
        result = operation.run()

        # Update state
        current_state = state_storage["state"]
        new_state = state_updater(current_state, result)
        state_storage["state"] = new_state

        return result, new_state

    return IO(stateful_operation)


def accumulative_monitoring(
    operation: IO[MetricPoint],
    accumulator: Callable[[list[MetricPoint], MetricPoint], list[MetricPoint]],
    history_storage: dict[str, list[MetricPoint]] | None = None,
) -> IO[list[MetricPoint]]:
    """Accumulate monitoring results over time"""

    if history_storage is None:
        history_storage = {"history": []}

    def accumulative_operation() -> list[MetricPoint]:
        # Execute operation
        new_metric = operation.run()

        # Update history
        current_history = history_storage["history"]
        updated_history = accumulator(current_history, new_metric)
        history_storage["history"] = updated_history

        return updated_history

    return IO(accumulative_operation)


# ==============================================================================
# Utility Functions
# ==============================================================================


def create_metric_aggregator(
    metric_names: list[str], aggregation_type: str = "average"
) -> Callable[[list[MetricPoint]], MetricPoint]:
    """Create a metric aggregation function"""

    def aggregator(metrics: list[MetricPoint]) -> MetricPoint:
        filtered_metrics = [m for m in metrics if m.name in metric_names]

        if not filtered_metrics:
            return MetricPoint(
                name="aggregated_metric",
                value=0.0,
                timestamp=datetime.now(UTC),
                unit="error",
            )

        values = [m.value for m in filtered_metrics]

        if aggregation_type == "sum":
            aggregated_value = sum(values)
        elif aggregation_type == "average":
            aggregated_value = sum(values) / len(values)
        elif aggregation_type == "max":
            aggregated_value = max(values)
        elif aggregation_type == "min":
            aggregated_value = min(values)
        else:
            aggregated_value = sum(values) / len(values)

        return MetricPoint(
            name=f"aggregated_{aggregation_type}",
            value=aggregated_value,
            timestamp=datetime.now(UTC),
            tags={
                "aggregation": aggregation_type,
                "source_count": str(len(filtered_metrics)),
            },
            unit=filtered_metrics[0].unit,
        )

    return aggregator


def create_health_checker(
    check_name: str, check_function: Callable[[], bool]
) -> IO[HealthCheck]:
    """Create a simple health check from a boolean function"""

    def health_check() -> HealthCheck:
        try:
            is_healthy = check_function()
            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY

            return HealthCheck(
                component=check_name, status=status, timestamp=datetime.now(UTC)
            )
        except Exception as e:
            return HealthCheck(
                component=check_name,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(UTC),
                details={"error": str(e)},
            )

    return IO(health_check)


def create_monitoring_scheduler[A](
    operations: list[IO[A]], interval_seconds: float = 30.0
) -> IO[None]:
    """Create a simple monitoring scheduler (returns immediately)"""

    def setup_scheduler() -> None:
        # In real implementation, this would set up background scheduling
        print(
            f"Setting up monitoring scheduler with {len(operations)} operations, interval: {interval_seconds}s"
        )

    return IO(setup_scheduler)


# ==============================================================================
# Composition Utilities
# ==============================================================================


def compose_monitoring_functions(*functions) -> Callable:
    """Compose multiple monitoring functions into a single function"""
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def pipe_monitoring_operations(*operations) -> Callable[[A], IO[Any]]:
    """Pipe monitoring operations in sequence"""

    def piped_operation(initial_input: A) -> IO[Any]:
        def execute_pipe() -> Any:
            current_value = initial_input

            for operation in operations:
                if callable(operation):
                    # If it's a function, apply it
                    current_value = operation(current_value)
                else:
                    # If it's an IO operation, run it
                    current_value = operation.run()

            return current_value

        return IO(execute_pipe)

    return piped_operation
