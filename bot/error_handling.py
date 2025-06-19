"""
Enterprise-grade error handling and recovery mechanisms for bulletproof system reliability.

This module implements comprehensive error boundaries, recovery patterns, and resilience
strategies to ensure the trading bot never fails completely.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from typing import Any, Protocol

import aiohttp

# Configure logger
logger = logging.getLogger(__name__)

# Import balance-specific exceptions for handling
try:
    from .exchange.base import (
        BalanceRetrievalError,
        BalanceServiceUnavailableError,
        BalanceTimeoutError,
        BalanceValidationError,
        InsufficientBalanceError,
    )
except ImportError:
    # Fallback if imports fail
    class BalanceRetrievalError(Exception):
        pass

    class BalanceServiceUnavailableError(BalanceRetrievalError):
        pass

    class BalanceTimeoutError(BalanceRetrievalError):
        pass

    class BalanceValidationError(BalanceRetrievalError):
        pass

    class InsufficientBalanceError(BalanceRetrievalError):
        pass


class FallbackCallable(Protocol):
    """Protocol for fallback callable functions."""

    def __call__(self, error: Exception, context: ErrorContext) -> Any: ...


class AsyncFallbackCallable(Protocol):
    """Protocol for async fallback callable functions."""

    async def __call__(self, error: Exception, context: ErrorContext) -> Any: ...


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ServiceStatus(Enum):
    """Service status states for health monitoring."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    ERROR = "error"


@dataclass
class ErrorContext:
    """Enhanced error context for comprehensive error tracking."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    component: str = ""
    operation: str = ""
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    traceback_info: str = ""
    context_data: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    should_retry: bool = False
    recovery_attempted: bool = False


@dataclass
class ServiceHealth:
    """Service health information for monitoring."""

    name: str
    status: ServiceStatus = ServiceStatus.HEALTHY
    last_check: datetime | None = None
    failure_count: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    recovery_attempts: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


class ErrorBoundary:
    """
    Error boundary pattern implementation for component isolation and fallback behavior.

    Prevents cascade failures by containing errors within component boundaries
    and executing fallback behaviors when failures occur.
    """

    def __init__(
        self,
        component_name: str,
        fallback_behavior: Callable[..., Any] | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        self.component_name = component_name
        self.fallback_behavior = fallback_behavior
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.severity = severity
        self.error_count = 0
        self.last_error: Exception | None = None
        self.error_history: list[ErrorContext] = []
        self._is_degraded = False

    async def __aenter__(self):
        logger.debug(f"Entering error boundary for {self.component_name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self._handle_error(exc_val)
            # Boundary contains the error - don't propagate
            return True
        return False

    async def _handle_error(self, exception: Exception) -> None:
        """Handle error within the boundary with comprehensive logging and recovery."""
        self.error_count += 1
        self.last_error = exception

        # Create enhanced error context
        error_context = ErrorContext(
            component=self.component_name,
            operation="boundary_contained_operation",
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=self.severity,
            traceback_info=traceback.format_exc(),
            context_data={
                "error_count": self.error_count,
                "is_degraded": self._is_degraded,
                "max_retries": self.max_retries,
            },
        )

        self.error_history.append(error_context)

        # Log error with full context
        logger.error(
            f"Error boundary triggered in {self.component_name}: {exception}",
            extra={"error_context": error_context.__dict__},
        )

        # Mark component as degraded if error count exceeds threshold
        if self.error_count >= self.max_retries:
            self._is_degraded = True
            logger.warning(f"Component {self.component_name} marked as degraded")

        # Execute fallback behavior if available
        if self.fallback_behavior:
            try:
                logger.info(f"Executing fallback behavior for {self.component_name}")
                await self._execute_fallback(exception, error_context)
            except Exception as fallback_error:
                logger.error(
                    f"Fallback failed for {self.component_name}: {fallback_error}",
                    extra={"original_error": str(exception)},
                )

    async def _execute_fallback(
        self, original_error: Exception, context: ErrorContext
    ) -> None:
        """Execute fallback behavior with error context."""
        if self.fallback_behavior is None:
            return

        if asyncio.iscoroutinefunction(self.fallback_behavior):
            await self.fallback_behavior(original_error, context)
        else:
            self.fallback_behavior(original_error, context)

    def is_degraded(self) -> bool:
        """Check if component is in degraded state."""
        return self._is_degraded

    def reset(self) -> None:
        """Reset error boundary state."""
        self.error_count = 0
        self._is_degraded = False
        self.error_history.clear()
        logger.info(f"Error boundary reset for {self.component_name}")


class TradeSaga:
    """
    Transaction saga pattern for trade operations with compensation actions.

    Ensures trade consistency by tracking all steps and providing automatic
    compensation (rollback) when operations fail.
    """

    def __init__(self, saga_name: str):
        self.saga_name = saga_name
        self.steps: list[tuple[Callable[..., Any], str]] = []
        self.compensation_actions: list[Callable[..., Any] | None] = []
        self.completed_steps: list[tuple[int, str, Any, datetime]] = []
        self.saga_id = f"{saga_name}_{int(datetime.now(UTC).timestamp() * 1000)}"
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.status = "pending"

    def add_step(
        self,
        action: Callable[..., Any],
        compensation: Callable[..., Any] | None = None,
        step_name: str = "",
    ) -> None:
        """Add a step with optional compensation action."""
        self.steps.append((action, step_name or f"step_{len(self.steps)}"))
        if compensation:
            self.compensation_actions.append(compensation)
        else:
            self.compensation_actions.append(None)

    async def execute(self) -> bool:
        """Execute the saga with automatic compensation on failure."""
        self.start_time = datetime.now(UTC)
        self.status = "executing"

        logger.info(f"Starting saga execution: {self.saga_id}")

        try:
            for i, (step_action, step_name) in enumerate(self.steps):
                logger.debug(f"Executing saga step {i}: {step_name}")

                # Execute step
                if asyncio.iscoroutinefunction(step_action):
                    result = await step_action()
                else:
                    result = step_action()

                # Record successful step
                self.completed_steps.append((i, step_name, result, datetime.now(UTC)))
                logger.debug(f"Saga step {i} completed successfully")

            self.status = "completed"
            self.end_time = datetime.now(UTC)
            logger.info(f"Saga {self.saga_id} completed successfully")
            return True

        except Exception as e:
            self.status = "failed"
            self.end_time = datetime.now(UTC)
            logger.error(
                f"Saga {self.saga_id} failed at step {len(self.completed_steps)}: {e}"
            )

            # Execute compensation actions
            await self._compensate()
            raise

    async def _compensate(self) -> None:
        """Execute compensation actions in reverse order."""
        logger.info(f"Starting compensation for saga {self.saga_id}")

        # Execute compensation actions in reverse order of completion
        for i in reversed(range(len(self.completed_steps))):
            step_index, step_name, result, timestamp = self.completed_steps[i]

            if step_index < len(self.compensation_actions):
                compensation_action = self.compensation_actions[step_index]

                if compensation_action:
                    try:
                        logger.debug(
                            f"Executing compensation for step {step_index}: {step_name}"
                        )

                        if asyncio.iscoroutinefunction(compensation_action):
                            await compensation_action(result)
                        else:
                            compensation_action(result)

                        logger.debug(f"Compensation for step {step_index} completed")

                    except Exception as comp_error:
                        logger.error(
                            f"Compensation failed for step {step_index} ({step_name}): {comp_error}",
                            extra={"saga_id": self.saga_id, "original_result": result},
                        )

        logger.info(f"Compensation completed for saga {self.saga_id}")

    def get_status(self) -> dict[str, Any]:
        """Get saga status and execution details."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "saga_id": self.saga_id,
            "saga_name": self.saga_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "total_steps": len(self.steps),
            "completed_steps": len(self.completed_steps),
            "step_details": [
                {
                    "step_index": idx,
                    "step_name": name,
                    "completed_at": timestamp.isoformat(),
                    "has_result": result is not None,
                }
                for idx, name, result, timestamp in self.completed_steps
            ],
        }


class GracefulDegradation:
    """
    Graceful degradation system with fallback strategies for service failures.

    Maintains system functionality by providing fallback strategies when
    primary services become unavailable or unreliable.
    """

    def __init__(self):
        self.degraded_services: set[str] = set()
        self.fallback_strategies: dict[str, Callable[..., Any]] = {}
        self.service_health: dict[str, ServiceHealth] = {}
        self.degradation_thresholds: dict[str, int] = {}

    def register_service(
        self,
        service_name: str,
        fallback_strategy: Callable[..., Any],
        degradation_threshold: int = 3,
    ) -> None:
        """Register a service with its fallback strategy."""
        self.fallback_strategies[service_name] = fallback_strategy
        self.service_health[service_name] = ServiceHealth(name=service_name)
        self.degradation_thresholds[service_name] = degradation_threshold

        logger.info(f"Registered service {service_name} with fallback strategy")

    async def execute_with_fallback(
        self, service_name: str, primary_action: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Execute action with automatic fallback on failure."""
        service_health = self.service_health.get(service_name)
        if not service_health:
            raise ValueError(f"Service {service_name} not registered")

        # Use fallback if service is degraded
        if service_name in self.degraded_services:
            logger.info(f"Service {service_name} is degraded, using fallback")
            return await self._execute_fallback(service_name, *args, **kwargs)

        try:
            # Attempt primary action
            if asyncio.iscoroutinefunction(primary_action):
                result = await primary_action(*args, **kwargs)
            else:
                result = primary_action(*args, **kwargs)

            # Service recovered - update status
            await self._mark_service_healthy(service_name)
            return result

        except Exception as e:
            # Service failed - update health and potentially degrade
            await self._handle_service_failure(service_name, e)

            # Use fallback if available
            if service_name in self.fallback_strategies:
                logger.warning(f"Service {service_name} failed, using fallback: {e}")
                return await self._execute_fallback(service_name, *args, **kwargs)
            else:
                # No fallback available - re-raise exception
                raise

    async def _execute_fallback(self, service_name: str, *args, **kwargs) -> Any:
        """Execute fallback strategy for a service."""
        fallback_strategy = self.fallback_strategies[service_name]

        try:
            if asyncio.iscoroutinefunction(fallback_strategy):
                return await fallback_strategy(*args, **kwargs)
            else:
                return fallback_strategy(*args, **kwargs)
        except Exception as fallback_error:
            logger.error(
                f"Fallback strategy failed for {service_name}: {fallback_error}"
            )
            raise

    async def _handle_service_failure(
        self, service_name: str, error: Exception
    ) -> None:
        """Handle service failure and update health status."""
        service_health = self.service_health[service_name]
        service_health.failure_count += 1
        service_health.consecutive_failures += 1
        service_health.last_error = str(error)
        service_health.last_check = datetime.now(UTC)

        threshold = self.degradation_thresholds.get(service_name, 3)

        if service_health.consecutive_failures >= threshold:
            self.degraded_services.add(service_name)
            service_health.status = ServiceStatus.DEGRADED
            logger.warning(
                f"Service {service_name} marked as degraded after {service_health.consecutive_failures} failures"
            )
        else:
            service_health.status = ServiceStatus.UNHEALTHY

    async def _mark_service_healthy(self, service_name: str) -> None:
        """Mark service as healthy and reset failure counters."""
        service_health = self.service_health[service_name]

        # Reset failure counters
        service_health.consecutive_failures = 0
        service_health.last_error = None
        service_health.last_check = datetime.now(UTC)
        service_health.status = ServiceStatus.HEALTHY

        # Remove from degraded services
        self.degraded_services.discard(service_name)

        if service_name in self.degraded_services:
            logger.info(f"Service {service_name} recovered from degraded state")

    def get_service_status(self, service_name: str) -> ServiceHealth | None:
        """Get health status for a specific service."""
        return self.service_health.get(service_name)

    def get_all_service_status(self) -> dict[str, ServiceHealth]:
        """Get health status for all registered services."""
        return self.service_health.copy()


class EnhancedExceptionHandler:
    """
    Enhanced exception context and logging with retry logic.

    Provides comprehensive exception tracking, analysis, and intelligent retry strategies.
    """

    def __init__(self):
        self.exception_history: list[ErrorContext] = []
        self.retry_strategies: dict[type, int] = {
            ConnectionError: 3,
            TimeoutError: 5,
            aiohttp.ClientError: 3,
            # Balance-specific retry strategies
            BalanceServiceUnavailableError: 4,  # Service issues may resolve quickly
            BalanceTimeoutError: 5,  # Network timeouts benefit from retries
            BalanceValidationError: 1,  # Data issues unlikely to resolve with retry
            BalanceRetrievalError: 2,  # Generic balance errors get limited retries
            InsufficientBalanceError: 0,  # Insufficient funds won't resolve with retry
        }
        self.retry_delays: dict[type, float] = {
            ConnectionError: 2.0,
            TimeoutError: 1.0,
            aiohttp.ClientError: 1.5,
            # Balance-specific retry delays
            BalanceServiceUnavailableError: 3.0,  # Longer delay for service recovery
            BalanceTimeoutError: 1.5,  # Standard delay for timeouts
            BalanceValidationError: 0.5,  # Quick retry for validation issues
            BalanceRetrievalError: 2.0,  # Standard delay for balance errors
            InsufficientBalanceError: 0.0,  # No retry for insufficient funds
        }

    def log_exception_with_context(
        self,
        exception: Exception,
        context: dict[str, Any],
        component: str = "",
        operation: str = "",
    ) -> ErrorContext:
        """Log exception with comprehensive context information."""
        error_context = ErrorContext(
            component=component,
            operation=operation,
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=self._classify_error_severity(exception),
            traceback_info=traceback.format_exc(),
            context_data=context,
            should_retry=self.should_retry_exception(exception),
        )

        self.exception_history.append(error_context)

        # Log structured exception data
        logger.error(
            f"Exception in {component}.{operation}: {exception}",
            extra={
                "error_context": error_context.__dict__,
                "exception_type": type(exception).__name__,
                "should_retry": error_context.should_retry,
            },
        )

        return error_context

    def should_retry_exception(self, exception: Exception) -> bool:
        """Determine if exception is retryable based on type and context."""
        # Check if this exception type has a retry strategy
        for exc_type, _ in self.retry_strategies.items():
            if isinstance(exception, exc_type):
                return True

        # Additional logic for specific error patterns
        error_message = str(exception).lower()
        retryable_patterns = [
            "connection reset",
            "timeout",
            "temporary failure",
            "rate limit",
            "service unavailable",
        ]

        return any(pattern in error_message for pattern in retryable_patterns)

    def get_retry_config(self, exception: Exception) -> tuple[int, float]:
        """Get retry configuration for an exception type."""
        for exc_type in self.retry_strategies:
            if isinstance(exception, exc_type):
                max_retries = self.retry_strategies[exc_type]
                delay = self.retry_delays.get(exc_type, 1.0)
                return max_retries, delay

        # Default retry config for retryable exceptions
        if self.should_retry_exception(exception):
            return 3, 1.0

        return 0, 0.0

    def _classify_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Classify error severity based on exception type and context."""
        critical_exceptions = [
            SystemExit,
            KeyboardInterrupt,
            MemoryError,
        ]

        high_severity_exceptions = [
            ValueError,
            TypeError,
            AttributeError,
            BalanceValidationError,  # Data corruption issues are high severity
        ]

        medium_severity_exceptions = [
            ConnectionError,
            TimeoutError,
            BalanceServiceUnavailableError,  # Service issues are concerning but recoverable
            BalanceTimeoutError,  # Timeout issues indicate performance problems
            BalanceRetrievalError,  # General balance issues need attention
        ]

        low_severity_exceptions = [
            InsufficientBalanceError,  # Expected business logic condition
        ]

        if any(isinstance(exception, exc_type) for exc_type in critical_exceptions):
            return ErrorSeverity.CRITICAL
        elif any(
            isinstance(exception, exc_type) for exc_type in high_severity_exceptions
        ):
            return ErrorSeverity.HIGH
        elif any(
            isinstance(exception, exc_type) for exc_type in medium_severity_exceptions
        ):
            return ErrorSeverity.MEDIUM
        elif any(
            isinstance(exception, exc_type) for exc_type in low_severity_exceptions
        ):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.LOW

    def get_exception_statistics(self) -> dict[str, Any]:
        """Get statistics about handled exceptions."""
        if not self.exception_history:
            return {"total_exceptions": 0}

        # Count exceptions by type
        exception_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}

        for error_context in self.exception_history:
            exc_type = error_context.error_type
            severity = error_context.severity.value

            exception_counts[exc_type] = exception_counts.get(exc_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_exceptions": len(self.exception_history),
            "exception_types": exception_counts,
            "severity_distribution": severity_counts,
            "recent_exceptions": [
                {
                    "timestamp": ctx.timestamp.isoformat(),
                    "component": ctx.component,
                    "error_type": ctx.error_type,
                    "severity": ctx.severity.value,
                }
                for ctx in self.exception_history[-10:]  # Last 10 exceptions
            ],
        }


class BalanceErrorHandler:
    """
    Specialized error handler for balance-related operations.

    Provides specific handling strategies for different types of balance errors,
    including intelligent retry logic and fallback behaviors.
    """

    def __init__(self, exception_handler: EnhancedExceptionHandler):
        self.exception_handler = exception_handler
        self.balance_error_counts: dict[str, int] = {}
        self.consecutive_insufficient_balance_errors = 0

    def handle_balance_error(
        self,
        error: Exception,
        operation_context: dict[str, Any],
        component: str = "balance_operations",
    ) -> dict[str, Any]:
        """
        Handle balance-specific errors with appropriate logging and recovery strategies.

        Args:
            error: The balance exception that occurred
            operation_context: Context about the operation that failed
            component: Component where the error occurred

        Returns:
            Dictionary with error handling recommendations
        """
        error_type = type(error).__name__
        self.balance_error_counts[error_type] = (
            self.balance_error_counts.get(error_type, 0) + 1
        )

        # Enhanced context for balance errors
        enhanced_context = {
            **operation_context,
            "balance_error_count": self.balance_error_counts[error_type],
            "total_balance_errors": sum(self.balance_error_counts.values()),
        }

        # Add balance-specific context if available
        if hasattr(error, "get_error_context"):
            enhanced_context.update(error.get_error_context())

        # Log the error with enhanced context
        error_context = self.exception_handler.log_exception_with_context(
            error, enhanced_context, component, "balance_operation"
        )

        # Generate handling recommendations
        recommendations = self._generate_handling_recommendations(error, error_context)

        # Special handling for consecutive insufficient balance errors
        if isinstance(error, InsufficientBalanceError):
            self.consecutive_insufficient_balance_errors += 1
            recommendations["consecutive_insufficient_balance"] = (
                self.consecutive_insufficient_balance_errors
            )

            if self.consecutive_insufficient_balance_errors >= 3:
                recommendations["action"] = "halt_trading"
                recommendations["reason"] = (
                    "Multiple consecutive insufficient balance errors detected"
                )
                logger.critical(
                    f"Critical: {self.consecutive_insufficient_balance_errors} consecutive "
                    f"insufficient balance errors. Trading should be halted."
                )
        else:
            self.consecutive_insufficient_balance_errors = 0

        return recommendations

    def _generate_handling_recommendations(
        self, error: Exception, error_context: ErrorContext
    ) -> dict[str, Any]:
        """Generate specific handling recommendations based on error type."""
        recommendations = {
            "error_type": type(error).__name__,
            "severity": error_context.severity.value,
            "should_retry": error_context.should_retry,
            "action": "continue",
        }

        if isinstance(error, InsufficientBalanceError):
            recommendations.update(
                {
                    "action": "check_balance",
                    "retry_strategy": "none",
                    "user_action_required": True,
                    "message": "Insufficient balance detected. Please check account funding.",
                }
            )

        elif isinstance(error, BalanceServiceUnavailableError):
            recommendations.update(
                {
                    "action": "retry_with_backoff",
                    "retry_strategy": "exponential_backoff",
                    "max_retries": 4,
                    "initial_delay": 3.0,
                    "message": "Balance service temporarily unavailable. Will retry with backoff.",
                }
            )

        elif isinstance(error, BalanceTimeoutError):
            recommendations.update(
                {
                    "action": "retry_with_reduced_timeout",
                    "retry_strategy": "linear_backoff",
                    "max_retries": 5,
                    "initial_delay": 1.5,
                    "message": "Balance request timed out. Will retry with adjusted timeout.",
                }
            )

        elif isinstance(error, BalanceValidationError):
            recommendations.update(
                {
                    "action": "log_and_fallback",
                    "retry_strategy": "single_retry",
                    "max_retries": 1,
                    "initial_delay": 0.5,
                    "message": "Balance data validation failed. May indicate API changes.",
                    "escalation_required": True,
                }
            )

        elif isinstance(error, BalanceRetrievalError):
            recommendations.update(
                {
                    "action": "retry_with_fallback",
                    "retry_strategy": "exponential_backoff",
                    "max_retries": 2,
                    "initial_delay": 2.0,
                    "message": "General balance retrieval error. Will retry with fallback.",
                }
            )

        return recommendations

    def get_balance_error_summary(self) -> dict[str, Any]:
        """Get summary of balance errors encountered."""
        return {
            "error_counts_by_type": self.balance_error_counts.copy(),
            "total_balance_errors": sum(self.balance_error_counts.values()),
            "consecutive_insufficient_balance": self.consecutive_insufficient_balance_errors,
            "critical_threshold_reached": self.consecutive_insufficient_balance_errors
            >= 3,
        }

    def reset_error_counts(self):
        """Reset error counters (useful for new trading sessions)."""
        self.balance_error_counts.clear()
        self.consecutive_insufficient_balance_errors = 0


# Retry decorator with exponential backoff
def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator for automatic retry with exponential backoff."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )

                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we can't use async sleep
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )

                    import time

                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global instances for use throughout the application
exception_handler = EnhancedExceptionHandler()
graceful_degradation = GracefulDegradation()
balance_error_handler = BalanceErrorHandler(exception_handler)
