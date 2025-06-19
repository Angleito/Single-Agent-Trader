"""
Bluefin Service Client for connecting to the isolated Bluefin SDK service.

This client communicates with the Bluefin service container that has the
actual Bluefin SDK installed, avoiding dependency conflicts.
"""

import asyncio
import logging
import os
import random
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import aiohttp
from aiohttp import ClientError, ClientTimeout, TCPConnector

from ..error_handling import (
    exception_handler,
)

logger = logging.getLogger(__name__)


class BluefinClientError(Exception):
    """Base exception for Bluefin client errors."""


class BluefinServiceConnectionError(BluefinClientError):
    """Exception raised when connection to Bluefin service fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        service_url: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.service_url = service_url


class BluefinServiceAuthError(BluefinClientError):
    """Exception raised when authentication with Bluefin service fails."""


class BluefinServiceRateLimitError(BluefinClientError):
    """Exception raised when Bluefin service rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class BluefinServiceDataError(BluefinClientError):
    """Exception raised when Bluefin service returns invalid data."""


class BluefinServiceClient:
    """
    Client for communicating with the Bluefin SDK service container.

    The service provides a REST API wrapper around the Bluefin SDK,
    allowing the main bot to interact with Bluefin without dependency
    conflicts.
    """

    def __init__(
        self,
        service_url: str = "http://bluefin-service:8080",
        api_key: str | None = None,
    ):
        """
        Initialize the Bluefin service client.

        Args:
            service_url: URL of the Bluefin service container
            api_key: API key for authentication (if not provided, will use
                BLUEFIN_SERVICE_API_KEY env var)
        """
        self.service_url = service_url
        self._session: aiohttp.ClientSession | None = None
        self._connected = False
        self._session_closed = False

        # Generate client ID for logging context
        self.client_id = f"bluefin-client-{int(time.time())}"

        # Enhanced connection and retry settings
        self._max_retries = 5  # Increased from 3
        self._base_retry_delay = 1.0  # Base delay for exponential backoff
        self._max_retry_delay = 60.0  # Maximum retry delay
        self._request_timeout = 30.0  # seconds
        self._jitter_factor = 0.1  # Add randomness to retry delays

        # Request timeout settings
        self.quick_timeout = ClientTimeout(total=5.0)  # For health checks
        self.normal_timeout = ClientTimeout(total=15.0)  # For standard API calls
        self.heavy_timeout = ClientTimeout(total=30.0)  # For large responses

        # Enhanced connection health tracking
        self.last_health_check = 0.0
        self.health_check_interval = (
            30  # Check health every 30 seconds (reduced from 60)
        )
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

        # Comprehensive connectivity monitoring
        self.connection_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "connection_attempts": 0,
            "successful_connections": 0,
            "last_successful_request": None,
            "last_failed_request": None,
            "response_times": [],  # Rolling window of response times
            "error_counts_by_type": {},
            "health_check_history": [],  # Recent health check results
        }

        # WebSocket monitoring (if applicable)
        self.websocket_status = {
            "connected": False,
            "last_ping": None,
            "reconnect_attempts": 0,
            "messages_received": 0,
            "messages_sent": 0,
        }

        # Enhanced circuit breaker state
        self.circuit_open = False
        self.circuit_open_until = 0.0
        self.circuit_failure_threshold = 3  # Reduced threshold for faster failover
        self.circuit_recovery_timeout = 60  # seconds

        # Balance operation health tracking
        self.balance_operation_failures = 0
        self.last_successful_balance_fetch = 0
        self.balance_staleness_threshold = 300  # 5 minutes

        # Connection pooling settings
        self._connection_pool_size = 10
        self._connection_pool_ttl = 300  # 5 minutes
        self._keep_alive_timeout = 30

        # Session reuse tracking
        self._session_created_at = 0.0
        self._session_max_age = 1800  # 30 minutes
        self._session_request_count = 0
        self._max_requests_per_session = 1000

        # Get API key from parameter or environment - MANDATORY
        self.api_key = api_key or os.getenv("BLUEFIN_SERVICE_API_KEY")
        if not self.api_key:
            error_msg = (
                "BLUEFIN_SERVICE_API_KEY is required for Bluefin service authentication"
            )
            logger.error(
                "Critical: API key missing - service will fail",
                extra={
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "auth_configured": False,
                    "required_env_var": "BLUEFIN_SERVICE_API_KEY",
                },
            )
            raise BluefinServiceAuthError(error_msg)

        # Validate API key format
        if len(self.api_key) < 16:
            error_msg = f"BLUEFIN_SERVICE_API_KEY appears too short (got {len(self.api_key)} chars, expected 16+)"
            logger.error(
                "API key validation failed",
                extra={
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "api_key_length": len(self.api_key),
                    "minimum_length": 16,
                },
            )
            raise BluefinServiceAuthError(error_msg)

        logger.info(
            "Bluefin service client initialized with valid authentication",
            extra={
                "client_id": self.client_id,
                "service_url": self.service_url,
                "auth_configured": True,
                "api_key_length": len(self.api_key),
            },
        )

        # Balance operation audit trail
        self.balance_operation_audit = []
        self.max_balance_audit_entries = 500

        # Performance tracking for balance operations
        self.balance_performance_metrics = {
            "total_balance_requests": 0,
            "successful_balance_requests": 0,
            "failed_balance_requests": 0,
            "average_balance_response_time": 0.0,
            "balance_response_times": [],
            "last_balance_request": None,
            "last_successful_balance": None,
            "balance_error_counts": {},
        }

        # Prepare headers with authentication
        self._headers = {
            "Content-Type": "application/json",
            "User-Agent": f"BluefinServiceClient/{self.client_id}",
            "Connection": "keep-alive",
            "Accept-Encoding": "gzip, deflate",
        }
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

    def _validate_parameters(self, **kwargs) -> bool:
        """Validate common request parameters."""
        for key, value in kwargs.items():
            if key == "symbol" and value:
                if not isinstance(value, str) or len(value) < 3:
                    logger.error(f"Invalid symbol parameter: {value}")
                    return False
            elif key == "limit" and value is not None:
                if not isinstance(value, int) or value < 1 or value > 1000:
                    logger.error(f"Invalid limit parameter: {value}")
                    return False
            elif key in ["startTime", "endTime"] and value:
                if not isinstance(value, int) or value <= 0:
                    logger.error(f"Invalid timestamp parameter {key}: {value}")
                    return False
        return True

    def _record_balance_operation_audit(
        self,
        correlation_id: str,
        operation: str,
        status: str,
        balance_amount: str = None,
        error: str = None,
        duration_ms: float = None,
        response_size: int = None,
        metadata: dict = None,
    ) -> None:
        """
        Record balance operation in client audit trail.

        Args:
            correlation_id: Unique identifier for the operation
            operation: Type of balance operation
            status: success/failed/timeout/retried
            balance_amount: Balance amount (if applicable)
            error: Error message (if failed)
            duration_ms: Operation duration in milliseconds
            response_size: Size of response in bytes
            metadata: Additional context data
        """
        audit_entry = {
            "correlation_id": correlation_id,
            "operation": operation,
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
            "client_id": self.client_id,
            "service_url": self.service_url,
            "balance_amount": balance_amount,
            "error": error,
            "duration_ms": duration_ms,
            "response_size_bytes": response_size,
            "metadata": metadata or {},
        }

        # Add to audit trail
        self.balance_operation_audit.append(audit_entry)

        # Keep only recent entries
        if len(self.balance_operation_audit) > self.max_balance_audit_entries:
            self.balance_operation_audit = self.balance_operation_audit[
                -self.max_balance_audit_entries :
            ]

        # Update performance metrics
        self._update_balance_performance_metrics(status, duration_ms, error)

        # Log audit entry
        logger.debug(
            f"Balance client audit: {operation} - {status}",
            extra={
                "audit_entry": audit_entry,
                "operation_type": "balance_client_audit",
            },
        )

    def _update_balance_performance_metrics(
        self, status: str, duration_ms: float = None, error: str = None
    ) -> None:
        """Update balance operation performance metrics."""
        self.balance_performance_metrics["total_balance_requests"] += 1
        self.balance_performance_metrics["last_balance_request"] = datetime.now(UTC)

        if status == "success":
            self.balance_performance_metrics["successful_balance_requests"] += 1
            self.balance_performance_metrics["last_successful_balance"] = datetime.now(
                UTC
            )

            if duration_ms is not None:
                self.balance_performance_metrics["balance_response_times"].append(
                    duration_ms
                )
                # Keep only recent response times
                if (
                    len(self.balance_performance_metrics["balance_response_times"])
                    > 100
                ):
                    self.balance_performance_metrics["balance_response_times"] = (
                        self.balance_performance_metrics[
                            "balance_response_times"
                        ][-100:]
                    )

                # Update average
                times = self.balance_performance_metrics["balance_response_times"]
                self.balance_performance_metrics["average_balance_response_time"] = sum(
                    times
                ) / len(times)
        else:
            self.balance_performance_metrics["failed_balance_requests"] += 1

            if error:
                error_type = (
                    type(error).__name__ if isinstance(error, Exception) else str(error)
                )
                if (
                    error_type
                    not in self.balance_performance_metrics["balance_error_counts"]
                ):
                    self.balance_performance_metrics["balance_error_counts"][
                        error_type
                    ] = 0
                self.balance_performance_metrics["balance_error_counts"][
                    error_type
                ] += 1

    def _get_balance_error_context(
        self,
        correlation_id: str,
        operation: str,
        error: Exception,
        duration_ms: float = None,
        endpoint: str = None,
        additional_context: dict = None,
    ) -> dict:
        """
        Generate comprehensive error context for balance operations.

        Args:
            correlation_id: Unique operation identifier
            operation: Operation name
            error: The exception that occurred
            duration_ms: Operation duration in milliseconds
            endpoint: API endpoint called
            additional_context: Additional context data

        Returns:
            Dictionary with comprehensive error context
        """
        context = {
            "correlation_id": correlation_id,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now(UTC).isoformat(),
            "client_id": self.client_id,
            "service_url": self.service_url,
            "endpoint": endpoint,
            "duration_ms": duration_ms,
            "success": False,
            "error_category": self._categorize_balance_error(error),
            "actionable_message": self._get_actionable_balance_error_message(error),
            "retry_recommended": self._should_retry_balance_error(error),
            "circuit_breaker_open": self._is_circuit_open(),
            "consecutive_failures": self.consecutive_failures,
        }

        if additional_context:
            context.update(additional_context)

        return context

    def _categorize_balance_error(self, error: Exception) -> str:
        """Categorize balance-specific errors."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if "balance" in error_str:
            return "balance_specific"
        elif "account" in error_str:
            return "account_related"
        elif "timeout" in error_str or "timeout" in error_type:
            return "network_timeout"
        elif "connection" in error_str or "connection" in error_type:
            return "network_connection"
        elif "authentication" in error_str or "auth" in error_str:
            return "authentication"
        elif "rate" in error_str and "limit" in error_str:
            return "rate_limiting"
        elif "502" in error_str or "503" in error_str or "504" in error_str:
            return "service_unavailable"
        else:
            return "unknown"

    def _get_actionable_balance_error_message(self, error: Exception) -> str:
        """Generate actionable error message for balance operations."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if "balance" in error_str:
            return (
                "Balance retrieval failed. Account may not be initialized on Bluefin."
            )
        elif "account" in error_str:
            return "Account access failed. Verify account exists and is properly configured."
        elif "timeout" in error_str:
            return "Request timed out. Check network connection and retry."
        elif "connection" in error_str:
            return (
                "Connection failed. Verify Bluefin service is running and accessible."
            )
        elif "authentication" in error_str or "auth" in error_str:
            return "Authentication failed. Check API key configuration."
        elif "rate" in error_str and "limit" in error_str:
            return "Rate limit exceeded. Wait before making another request."
        else:
            return "Balance operation failed. Check logs for details."

    def _should_retry_balance_error(self, error: Exception) -> bool:
        """Determine if balance error should be retried."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Don't retry authentication or validation errors
        if "authentication" in error_str or "auth" in error_str:
            return False
        if "invalid" in error_str and ("key" in error_str or "token" in error_str):
            return False

        # Retry network and service errors
        if any(
            keyword in error_str
            for keyword in ["timeout", "connection", "502", "503", "504"]
        ):
            return True

        # Retry rate limit errors (with delay)
        if "rate" in error_str and "limit" in error_str:
            return True

        # Default to retry for unknown errors
        return True

    def get_balance_audit_trail(self, limit: int = 50) -> list[dict]:
        """Get recent balance operation audit trail."""
        return self.balance_operation_audit[-limit:]

    def get_balance_performance_summary(self) -> dict:
        """Get balance operation performance summary."""
        metrics = self.balance_performance_metrics.copy()

        # Calculate success rate
        total = metrics["total_balance_requests"]
        successful = metrics["successful_balance_requests"]
        metrics["success_rate_percent"] = (successful / total * 100) if total > 0 else 0

        # Format timestamps
        if metrics["last_balance_request"]:
            metrics["last_balance_request"] = metrics[
                "last_balance_request"
            ].isoformat()
        if metrics["last_successful_balance"]:
            metrics["last_successful_balance"] = metrics[
                "last_successful_balance"
            ].isoformat()

        return metrics

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        if self.circuit_open and time.time() < self.circuit_open_until:
            return True
        elif self.circuit_open and time.time() >= self.circuit_open_until:
            # Circuit breaker timeout expired, attempt to close
            logger.info(
                "Circuit breaker timeout expired, attempting to close circuit",
                extra={"client_id": self.client_id},
            )
            self.circuit_open = False
            self.consecutive_failures = 0
        return False

    def reset_circuit_breaker(self, force: bool = False) -> bool:
        """
        Manual circuit breaker reset functionality.

        Args:
            force: If True, force reset even if not ready naturally

        Returns:
            True if reset was successful
        """
        current_time = time.time()
        old_state = self.circuit_open
        old_failures = self.consecutive_failures

        if force or current_time >= self.circuit_open_until:
            self.circuit_open = False
            self.circuit_open_until = 0.0
            self.consecutive_failures = 0

            logger.info(
                "Circuit breaker manually reset",
                extra={
                    "client_id": self.client_id,
                    "was_open": old_state,
                    "previous_failures": old_failures,
                    "force_reset": force,
                    "time_until_natural_reset": max(
                        0, self.circuit_open_until - current_time
                    ),
                },
            )
            return True
        else:
            remaining_time = self.circuit_open_until - current_time
            logger.warning(
                f"Circuit breaker reset not ready - {remaining_time:.1f}s remaining",
                extra={
                    "client_id": self.client_id,
                    "remaining_seconds": remaining_time,
                    "use_force_to_override": True,
                },
            )
            return False

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """
        Get current circuit breaker status for monitoring.

        Returns:
            Dictionary with circuit breaker state information
        """
        current_time = time.time()
        return {
            "is_open": self.circuit_open,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.circuit_failure_threshold,
            "recovery_timeout": self.circuit_recovery_timeout,
            "open_until": self.circuit_open_until,
            "seconds_until_reset": (
                max(0, self.circuit_open_until - current_time)
                if self.circuit_open
                else 0
            ),
            "can_reset_manually": not self.circuit_open
            or current_time >= self.circuit_open_until,
        }

    def _record_success(self):
        """Record successful operation - reset failure counters."""
        if self.consecutive_failures > 0:
            logger.info(
                f"Operation succeeded, resetting failure count from {self.consecutive_failures}",
                extra={"client_id": self.client_id},
            )
            self.consecutive_failures = 0

    def _record_failure(self):
        """Record failed operation - may trigger circuit breaker."""
        self.consecutive_failures += 1

        if self.consecutive_failures >= self.circuit_failure_threshold:
            self.circuit_open = True
            self.circuit_open_until = time.time() + self.circuit_recovery_timeout
            logger.warning(
                f"Circuit breaker opened due to {self.consecutive_failures} consecutive failures. "
                f"Will retry after {self.circuit_recovery_timeout}s",
                extra={"client_id": self.client_id},
            )

    def _calculate_retry_delay(self, attempt: int, operation: str = "default") -> float:
        """
        Calculate retry delay with enhanced exponential backoff and intelligent jitter.

        Args:
            attempt: Current attempt number (0-based)
            operation: Type of operation for context-aware delays

        Returns:
            Calculated delay in seconds
        """
        # Base exponential backoff: base * (2 ^ attempt)
        base_delay = self._base_retry_delay * (2**attempt)

        # Apply operation-specific multipliers
        operation_multipliers = {
            "health_check": 0.5,  # Faster retries for health checks
            "connect": 1.0,  # Standard delays for connection
            "api_call": 1.2,  # Slightly longer for API calls
            "heavy_operation": 1.5,  # Longer delays for heavy operations
        }

        multiplier = operation_multipliers.get(operation, 1.0)
        adjusted_delay = base_delay * multiplier

        # Cap at maximum delay
        capped_delay = min(adjusted_delay, self._max_retry_delay)

        # Add intelligent jitter
        # - More jitter for early attempts to spread out retries
        # - Less jitter for later attempts to be more predictable
        jitter_factor = self._jitter_factor * (
            1 - (attempt * 0.1)
        )  # Decrease jitter over time
        jitter_factor = max(jitter_factor, 0.05)  # Minimum jitter

        jitter = capped_delay * jitter_factor * random.uniform(0.5, 1.5)

        final_delay = capped_delay + jitter

        logger.debug(
            f"Calculated retry delay for {operation}",
            extra={
                "client_id": self.client_id,
                "attempt": attempt,
                "operation": operation,
                "base_delay": base_delay,
                "multiplier": multiplier,
                "capped_delay": capped_delay,
                "jitter": jitter,
                "final_delay": final_delay,
            },
        )

        return final_delay

    async def _retry_request(self, func, *args, **kwargs):
        """Execute request with enhanced exponential backoff retry and circuit breaker."""
        # Check circuit breaker first
        if self._is_circuit_open():
            logger.warning(
                "Circuit breaker is open, skipping request",
                extra={"client_id": self.client_id},
            )
            raise BluefinServiceConnectionError(
                "Circuit breaker is open - service temporarily unavailable",
                service_url=self.service_url,
            )

        last_exception = None

        for attempt in range(self._max_retries):
            try:
                result = await func(*args, **kwargs)
                self._record_success()
                return result
            except (TimeoutError, ClientError, aiohttp.ServerConnectionError) as e:
                last_exception = e
                self._record_failure()

                if attempt < self._max_retries - 1:
                    delay = self._calculate_retry_delay(attempt, "api_call")
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self._max_retries}), "
                        f"retrying in {delay:.2f}s: {e}",
                        extra={
                            "client_id": self.client_id,
                            "attempt": attempt + 1,
                            "max_retries": self._max_retries,
                            "delay": delay,
                            "error_type": type(e).__name__,
                        },
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Request failed after {self._max_retries} attempts: {e}",
                        extra={
                            "client_id": self.client_id,
                            "total_attempts": self._max_retries,
                            "error_type": type(e).__name__,
                        },
                    )
            except Exception as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "client_id": self.client_id,
                        "attempt": attempt + 1,
                        "non_retryable_error": True,
                        "function_name": (
                            func.__name__ if hasattr(func, "__name__") else "unknown"
                        ),
                    },
                    component="BluefinServiceClient",
                    operation="_retry_request",
                )
                last_exception = e
                break

        if last_exception:
            raise last_exception
        return None

    def _should_recreate_session(self) -> bool:
        """Check if session should be recreated based on age and usage."""
        if not self._session or self._session.closed:
            return True

        current_time = time.time()

        # Check session age
        if current_time - self._session_created_at > self._session_max_age:
            logger.debug(
                "Session exceeded max age, will recreate",
                extra={
                    "client_id": self.client_id,
                    "session_age": current_time - self._session_created_at,
                    "max_age": self._session_max_age,
                },
            )
            return True

        # Check request count
        if self._session_request_count >= self._max_requests_per_session:
            logger.debug(
                "Session exceeded max requests, will recreate",
                extra={
                    "client_id": self.client_id,
                    "request_count": self._session_request_count,
                    "max_requests": self._max_requests_per_session,
                },
            )
            return True

        return False

    async def _create_session(self) -> aiohttp.ClientSession:
        """Create a new HTTP session with connection pooling."""
        # Create TCP connector with connection pooling
        connector = TCPConnector(
            limit=self._connection_pool_size,
            limit_per_host=self._connection_pool_size,
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=self._keep_alive_timeout,
            enable_cleanup_closed=True,
        )

        session = aiohttp.ClientSession(
            headers=self._headers,
            timeout=self.normal_timeout,
            connector=connector,
            connector_owner=True,  # Session owns the connector
        )

        self._session_created_at = time.time()
        self._session_request_count = 0

        logger.debug(
            "Created new HTTP session with connection pooling",
            extra={
                "client_id": self.client_id,
                "pool_size": self._connection_pool_size,
                "keep_alive_timeout": self._keep_alive_timeout,
            },
        )

        return session

    async def _check_connection_health(self) -> bool:
        """Enhanced connection health check with intelligent reconnection and resilient error handling."""
        current_time = time.time()

        # Skip health check if done recently and connection appears healthy
        if (
            current_time - self.last_health_check < self.health_check_interval
            and self._connected
            and not self._should_recreate_session()
        ):
            return self._connected

        # Multiple retry attempts for health check resilience
        max_health_retries = 3
        health_retry_delay = 1.0

        for attempt in range(max_health_retries):
            try:
                # Recreate session if needed
                if self._should_recreate_session():
                    await self._ensure_session()

                # Quick health check with timeout
                if self._session is not None:
                    async with self._session.get(
                        f"{self.service_url}/health", timeout=self.quick_timeout
                    ) as resp:
                        self._connected = resp.status == 200
                        self.last_health_check = current_time

                    if self._connected:
                        self.consecutive_failures = 0
                        logger.debug(
                            f"Health check successful on attempt {attempt + 1}",
                            extra={
                                "client_id": self.client_id,
                                "status_code": resp.status,
                                "connected": self._connected,
                                "attempt": attempt + 1,
                            },
                        )
                        return True
                    elif attempt < max_health_retries - 1:
                        logger.debug(
                            f"Health check failed (status {resp.status}), retrying in {health_retry_delay}s",
                            extra={
                                "client_id": self.client_id,
                                "status_code": resp.status,
                                "attempt": attempt + 1,
                                "max_retries": max_health_retries,
                            },
                        )
                        await asyncio.sleep(health_retry_delay)
                        continue
                    else:
                        logger.warning(
                            f"Health check failed after {max_health_retries} attempts: status {resp.status}",
                            extra={
                                "client_id": self.client_id,
                                "status_code": resp.status,
                                "total_attempts": max_health_retries,
                            },
                        )
                else:
                    logger.warning(
                        f"Health check session unavailable on attempt {attempt + 1}",
                        extra={
                            "client_id": self.client_id,
                            "attempt": attempt + 1,
                        },
                    )
                    if attempt < max_health_retries - 1:
                        await asyncio.sleep(health_retry_delay)
                        continue

                # If we reach here, this attempt failed
                self._connected = False

            except (ClientError, TimeoutError, OSError) as e:
                if attempt < max_health_retries - 1:
                    logger.debug(
                        f"Health check network error on attempt {attempt + 1}, retrying: {e}",
                        extra={
                            "client_id": self.client_id,
                            "error_type": type(e).__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_health_retries,
                        },
                    )
                    await asyncio.sleep(health_retry_delay)
                    continue
                else:
                    # Final attempt failed
                    exception_handler.log_exception_with_context(
                        e,
                        {
                            "client_id": self.client_id,
                            "service_url": self.service_url,
                            "consecutive_failures": self.consecutive_failures,
                            "health_check_network_error": True,
                            "total_attempts": max_health_retries,
                        },
                        component="BluefinServiceClient",
                        operation="_check_connection_health",
                    )
                    self._connected = False
                    self.consecutive_failures += 1
                    return False

            except Exception as e:
                # For unexpected errors, don't retry - fail fast
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "client_id": self.client_id,
                        "unexpected_health_check_error": True,
                        "attempt": attempt + 1,
                    },
                    component="BluefinServiceClient",
                    operation="_check_connection_health",
                )
                self._connected = False
                self.consecutive_failures += 1
                return False

        # All attempts failed
        self._connected = False
        self.consecutive_failures += 1
        logger.warning(
            f"Health check failed after {max_health_retries} resilient attempts",
            extra={
                "client_id": self.client_id,
                "total_attempts": max_health_retries,
                "consecutive_failures": self.consecutive_failures,
            },
        )
        return False

    async def _ensure_session(self) -> None:
        """Ensure we have a valid session, creating one if needed."""
        if self._should_recreate_session():
            # Close existing session
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                    logger.debug(
                        "Closed existing session for recreation",
                        extra={"client_id": self.client_id},
                    )
                except Exception as e:
                    exception_handler.log_exception_with_context(
                        e,
                        {
                            "client_id": self.client_id,
                            "session_close_error": True,
                        },
                        component="BluefinServiceClient",
                        operation="_ensure_session",
                    )

            # Create new session
            self._session = await self._create_session()
            self._session_closed = False

    async def connect(self) -> bool:
        """
        Connect to the Bluefin service with enhanced reliability.

        Returns:
            True if connection successful
        """
        try:
            # Ensure we have a session
            await self._ensure_session()

            logger.debug(
                "Establishing connection to Bluefin service",
                extra={
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "session_age": time.time() - self._session_created_at,
                    "request_count": self._session_request_count,
                },
            )

            # Check service health (health endpoint doesn't require auth)
            health_url = f"{self.service_url}/health"
            logger.debug(
                "Checking service health",
                extra={
                    "client_id": self.client_id,
                    "health_url": health_url,
                    "operation": "health_check",
                },
            )

            if self._session is not None:
                async with self._session.get(
                    health_url, timeout=self.quick_timeout
                ) as resp:
                    self._session_request_count += 1

                    if resp.status == 200:
                        data = await resp.json()
                        self._connected = data.get("status") == "healthy"
                        self.last_health_check = time.time()
                        self._record_success()

                        logger.info(
                            "Successfully connected to Bluefin service",
                            extra={
                                "client_id": self.client_id,
                                "service_url": self.service_url,
                                "service_status": data.get("status"),
                                "service_initialized": data.get("initialized"),
                                "service_network": data.get("network"),
                                "connected": self._connected,
                            },
                        )
                        return self._connected
                    else:
                        error_text = await resp.text()
                        self._record_failure()
                        logger.error(
                            "Bluefin service health check failed",
                            extra={
                                "client_id": self.client_id,
                                "service_url": self.service_url,
                                "status_code": resp.status,
                                "error_response": error_text[:200],
                                "operation": "health_check",
                            },
                        )
                        return False
            else:
                self._record_failure()
                logger.error(
                    "Session is None - cannot perform health check",
                    extra={
                        "client_id": self.client_id,
                        "service_url": self.service_url,
                        "operation": "health_check",
                    },
                )
                return False

        except (ClientError, TimeoutError, OSError) as e:
            self._record_failure()
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "connection_network_error": True,
                },
                component="BluefinServiceClient",
                operation="connect",
            )
            return False
        except Exception as e:
            self._record_failure()
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "unexpected_connection_error": True,
                },
                component="BluefinServiceClient",
                operation="connect",
            )
            return False

    async def force_session_refresh(self) -> bool:
        """
        Force a session refresh, useful for recovering from connection issues.

        Returns:
            True if session was successfully refreshed
        """
        logger.info(
            "Forcing session refresh",
            extra={
                "client_id": self.client_id,
                "current_session_age": (
                    time.time() - self._session_created_at if self._session else 0
                ),
                "current_request_count": self._session_request_count,
            },
        )

        try:
            # Close existing session
            if self._session and not self._session.closed:
                await self._session.close()
                logger.debug("Closed existing session for refresh")

            # Create new session
            self._session = await self._create_session()
            self._session_closed = False

            logger.info(
                "Session refresh completed successfully",
                extra={"client_id": self.client_id},
            )
            return True

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "session_refresh_error": True,
                },
                component="BluefinServiceClient",
                operation="force_session_refresh",
            )
            return False

    async def get_session_stats(self) -> dict[str, Any]:
        """
        Get current session statistics for monitoring.

        Returns:
            Dictionary with session statistics
        """
        current_time = time.time()
        session_age = current_time - self._session_created_at if self._session else 0

        return {
            "has_session": self._session is not None,
            "session_closed": self._session.closed if self._session else True,
            "session_age_seconds": session_age,
            "session_max_age_seconds": self._session_max_age,
            "request_count": self._session_request_count,
            "max_requests_per_session": self._max_requests_per_session,
            "needs_recreation": self._should_recreate_session(),
            "pool_size": self._connection_pool_size,
            "keep_alive_timeout": self._keep_alive_timeout,
            "last_health_check": self.last_health_check,
            "consecutive_failures": self.consecutive_failures,
        }

    async def disconnect(self) -> None:
        """Disconnect from the Bluefin service with comprehensive cleanup."""
        logger.info(
            "Disconnecting from Bluefin service",
            extra={
                "client_id": self.client_id,
                "service_url": self.service_url,
                "was_connected": self._connected,
                "session_stats": await self.get_session_stats(),
            },
        )

        if self._session and not self._session.closed:
            try:
                # Wait for any pending requests to complete (with timeout)
                if hasattr(self._session, "_connector") and self._session._connector:
                    logger.debug("Waiting for pending requests to complete")
                    await asyncio.sleep(0.1)  # Brief wait for graceful closure

                await self._session.close()

                # Wait for session cleanup to complete
                await asyncio.sleep(0.1)

                logger.debug(
                    "HTTP session closed successfully",
                    extra={
                        "client_id": self.client_id,
                        "final_request_count": self._session_request_count,
                    },
                )
            except Exception as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "client_id": self.client_id,
                        "session_close_error": True,
                    },
                    component="BluefinServiceClient",
                    operation="disconnect",
                )
            finally:
                self._session = None
                self._session_closed = True
                self._session_created_at = 0.0
                self._session_request_count = 0

        # Reset connection state
        self._connected = False
        self.last_health_check = 0.0

        logger.info(
            "Successfully disconnected from Bluefin service",
            extra={"client_id": self.client_id},
        )

    async def get_account_data(self) -> dict[str, Any]:
        """
        Get account data including balances with comprehensive error handling.

        Returns:
            Account data dictionary
        """
        # Generate correlation ID for this balance operation
        correlation_id = str(uuid.uuid4())
        operation_start = time.time()

        if self._session is None or self._session.closed:
            error_msg = "Session not initialized or closed - call connect() first"

            # Record audit for session error
            self._record_balance_operation_audit(
                correlation_id,
                "get_account_data",
                "failed",
                error=error_msg,
                duration_ms=0,
                metadata={"error_stage": "session_check"},
            )

            logger.error(
                "Account data request failed - no session",
                extra={
                    "client_id": self.client_id,
                    "correlation_id": correlation_id,
                    "operation": "get_account_data",
                    "session_initialized": self._session is not None,
                    "session_closed": self._session.closed if self._session else True,
                    "error_category": "session_unavailable",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return {"error": error_msg}

        # Check connection health
        health_start = time.time()
        if not await self._check_connection_health():
            health_duration = time.time() - health_start
            error_msg = "Connection health check failed"

            # Record audit for health check failure
            self._record_balance_operation_audit(
                correlation_id,
                "get_account_data",
                "failed",
                error=error_msg,
                duration_ms=round(health_duration * 1000, 2),
                metadata={"error_stage": "health_check"},
            )

            logger.error(
                "Account data request failed - health check",
                extra={
                    "client_id": self.client_id,
                    "correlation_id": correlation_id,
                    "operation": "get_account_data",
                    "error_category": "health_check_failed",
                    "health_check_duration_ms": round(health_duration * 1000, 2),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return {"error": error_msg}

        logger.debug(
            "Requesting account data from Bluefin service",
            extra={
                "client_id": self.client_id,
                "correlation_id": correlation_id,
                "operation": "get_account_data",
                "endpoint": "/account",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        try:
            result = await self._make_request_with_retry(
                method="GET", endpoint="/account", operation="get_account_data"
            )

            # Record successful operation
            total_duration = time.time() - operation_start
            balance_amount = (
                result.get("balance", "0") if isinstance(result, dict) else "0"
            )

            self._record_balance_operation_audit(
                correlation_id,
                "get_account_data",
                "success",
                balance_amount=balance_amount,
                duration_ms=round(total_duration * 1000, 2),
                response_size=len(str(result)),
                metadata={
                    "has_balance": bool(balance_amount and balance_amount != "0"),
                    "has_address": (
                        bool(result.get("address"))
                        if isinstance(result, dict)
                        else False
                    ),
                    "response_type": type(result).__name__,
                },
            )

            logger.info(
                "Successfully retrieved account data",
                extra={
                    "client_id": self.client_id,
                    "correlation_id": correlation_id,
                    "operation": "get_account_data",
                    "duration_ms": round(total_duration * 1000, 2),
                    "has_balance": bool(balance_amount and balance_amount != "0"),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "success": True,
                },
            )

            return result

        except Exception as e:
            total_duration = time.time() - operation_start

            # Generate comprehensive error context
            error_context = self._get_balance_error_context(
                correlation_id,
                "get_account_data",
                e,
                round(total_duration * 1000, 2),
                "/account",
                additional_context={"error_stage": "api_request"},
            )

            # Record audit for failed operation
            self._record_balance_operation_audit(
                correlation_id,
                "get_account_data",
                "failed",
                error=str(e),
                duration_ms=round(total_duration * 1000, 2),
                metadata={
                    "error_stage": "api_request",
                    "exception_type": type(e).__name__,
                },
            )

            logger.error(
                "Account data request failed with exception",
                extra=error_context,
            )

            return {"error": f"Account data request failed: {e!s}"}

    async def get_user_positions(self) -> list[dict[str, Any]]:
        """
        Get current user positions with comprehensive error handling.

        Returns:
            List of position dictionaries
        """
        if self._session is None or self._session.closed:
            logger.error(
                "Positions request failed - no session",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_user_positions",
                    "session_initialized": self._session is not None,
                    "session_closed": self._session.closed if self._session else True,
                },
            )
            return []

        logger.debug(
            "Requesting user positions from Bluefin service",
            extra={
                "client_id": self.client_id,
                "operation": "get_user_positions",
                "endpoint": "/positions",
            },
        )

        try:
            result = await self._make_request_with_retry(
                method="GET", endpoint="/positions", operation="get_user_positions"
            )

            positions = result.get("positions", [])
            logger.info(
                "Successfully retrieved user positions",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_user_positions",
                    "positions_count": len(positions),
                },
            )
            return positions

        except (
            BluefinServiceAuthError,
            BluefinServiceRateLimitError,
            BluefinServiceConnectionError,
        ) as e:
            logger.error(
                "Service error getting positions",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_user_positions",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            return []
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "operation": "get_user_positions",
                    "unexpected_positions_error": True,
                },
                component="BluefinServiceClient",
                operation="get_user_positions",
            )
            return []

    async def place_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """
        Place an order through the Bluefin service with enhanced reliability.

        Args:
            order_data: Order details including symbol, side, quantity, etc.

        Returns:
            Order response dictionary
        """
        try:
            result = await self._make_request_with_retry(
                method="POST",
                endpoint="/orders",
                operation="place_order",
                json_data=order_data,
            )
            return {"status": "success", "order": result}

        except BluefinServiceAuthError as e:
            logger.error(
                f"Authentication failed placing order: {e}",
                extra={"client_id": self.client_id, "operation": "place_order"},
            )
            return {"status": "error", "message": "Authentication failed"}

        except BluefinServiceRateLimitError as e:
            logger.error(
                f"Rate limited placing order: {e}",
                extra={"client_id": self.client_id, "operation": "place_order"},
            )
            return {
                "status": "error",
                "message": f"Rate limit exceeded, retry after {e.retry_after}s",
                "retry_after": e.retry_after,
            }

        except BluefinServiceConnectionError as e:
            logger.error(
                f"Connection error placing order: {e}",
                extra={"client_id": self.client_id, "operation": "place_order"},
            )
            return {"status": "error", "message": f"Connection error: {e!s}"}

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "operation": "place_order",
                    "unexpected_order_error": True,
                    "has_order_data": bool(order_data),
                },
                component="BluefinServiceClient",
                operation="place_order",
            )
            return {"status": "error", "message": str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with enhanced reliability.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        try:
            await self._make_request_with_retry(
                method="DELETE",
                endpoint=f"/orders/{order_id}",
                operation="cancel_order",
            )
            return True

        except (
            BluefinServiceAuthError,
            BluefinServiceRateLimitError,
            BluefinServiceConnectionError,
        ) as e:
            logger.error(
                f"Service error canceling order {order_id}: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "cancel_order",
                    "order_id": order_id,
                    "error_type": type(e).__name__,
                },
            )
            return False

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "operation": "cancel_order",
                    "order_id": order_id,
                    "unexpected_cancel_error": True,
                },
                component="BluefinServiceClient",
                operation="cancel_order",
            )
            return False

    async def get_market_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Get market ticker data with enhanced reliability.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker data dictionary
        """
        try:
            result = await self._make_request_with_retry(
                method="GET",
                endpoint="/market/ticker",
                operation="get_market_ticker",
                params={"symbol": symbol},
            )
            return result

        except (
            BluefinServiceAuthError,
            BluefinServiceRateLimitError,
            BluefinServiceConnectionError,
        ) as e:
            logger.error(
                f"Service error getting ticker for {symbol}: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_market_ticker",
                    "symbol": symbol,
                    "error_type": type(e).__name__,
                },
            )
            return {"price": "0", "error": str(e)}

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "operation": "get_market_ticker",
                    "symbol": symbol,
                    "unexpected_ticker_error": True,
                },
                component="BluefinServiceClient",
                operation="get_market_ticker",
            )
            return {"price": "0", "error": str(e)}

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol with enhanced reliability.

        Args:
            symbol: Trading symbol
            leverage: Leverage value

        Returns:
            True if successful
        """
        try:
            await self._make_request_with_retry(
                method="POST",
                endpoint="/leverage",
                operation="set_leverage",
                json_data={"symbol": symbol, "leverage": leverage},
            )
            return True

        except (
            BluefinServiceAuthError,
            BluefinServiceRateLimitError,
            BluefinServiceConnectionError,
        ) as e:
            logger.error(
                f"Service error setting leverage for {symbol}: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "set_leverage",
                    "symbol": symbol,
                    "leverage": leverage,
                    "error_type": type(e).__name__,
                },
            )
            return False

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "operation": "set_leverage",
                    "symbol": symbol,
                    "leverage": leverage,
                    "unexpected_leverage_error": True,
                },
                component="BluefinServiceClient",
                operation="set_leverage",
            )
            return False

    async def get_candlestick_data(self, params: dict[str, Any]) -> list[list[Any]]:
        """
        Get historical candlestick data with enhanced validation and retry logic.

        Args:
            params: Parameters including symbol, interval, and limit

        Returns:
            List of candlestick arrays
            [timestamp, open, high, low, close, volume]
        """
        # Validate parameters
        if not self._validate_parameters(**params):
            logger.error("Invalid parameters provided")
            return []

        try:
            result = await self._make_request_with_retry(
                method="GET",
                endpoint="/market/candles",
                operation="get_candlestick_data",
                params=params,
            )

            candles = result.get("candles", [])

            # Validate candle data
            if self._validate_candle_data(candles):
                logger.info(
                    f"✅ Successfully retrieved {len(candles)} validated candles",
                    extra={
                        "client_id": self.client_id,
                        "operation": "get_candlestick_data",
                        "candle_count": len(candles),
                    },
                )
                return candles
            else:
                logger.warning(
                    "⚠️ Received invalid candle data from service",
                    extra={
                        "client_id": self.client_id,
                        "operation": "get_candlestick_data",
                        "raw_candle_count": len(candles),
                    },
                )
                return []

        except (BluefinServiceAuthError, BluefinServiceRateLimitError) as e:
            logger.error(
                f"Service error getting candlestick data: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_candlestick_data",
                    "error_type": type(e).__name__,
                },
            )
            return []
        except BluefinServiceConnectionError as e:
            logger.error(
                f"Connection error getting candlestick data: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_candlestick_data",
                    "error_type": type(e).__name__,
                },
            )
            return []
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "operation": "get_candlestick_data",
                    "params": params,
                    "unexpected_candlestick_error": True,
                },
                component="BluefinServiceClient",
                operation="get_candlestick_data",
            )
            return []

    async def _make_request_with_retry(
        self,
        method: str,
        endpoint: str,
        operation: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make HTTP request with enhanced retry logic and comprehensive error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            operation: Operation name for logging
            json_data: JSON data for POST requests
            params: Query parameters

        Returns:
            Response data dictionary

        Raises:
            BluefinServiceAuthError: When authentication fails
            BluefinServiceRateLimitError: When rate limited
            BluefinServiceConnectionError: When request fails
        """
        # Check circuit breaker first
        if self._is_circuit_open():
            logger.warning(
                "Circuit breaker is open, skipping request",
                extra={"client_id": self.client_id, "operation": operation},
            )
            raise BluefinServiceConnectionError(
                "Circuit breaker is open - service temporarily unavailable",
                service_url=self.service_url,
            )

        # Ensure we have a healthy connection
        if not await self._check_connection_health():
            logger.error(
                "Connection health check failed before request",
                extra={"client_id": self.client_id, "operation": operation},
            )
            raise BluefinServiceConnectionError(
                "Connection unhealthy", service_url=self.service_url
            )

        url = f"{self.service_url}{endpoint}"

        for attempt in range(self._max_retries):
            try:
                # Ensure session is valid before each attempt
                await self._ensure_session()

                logger.debug(
                    f"Making HTTP request (attempt {attempt + 1}/{self._max_retries})",
                    extra={
                        "client_id": self.client_id,
                        "method": method,
                        "url": url,
                        "operation": operation,
                        "attempt": attempt + 1,
                        "has_json_data": bool(json_data),
                        "has_params": bool(params),
                        "session_request_count": self._session_request_count,
                    },
                )

                request_kwargs: dict[str, Any] = {}
                if json_data is not None:
                    request_kwargs["json"] = json_data
                if params is not None:
                    request_kwargs["params"] = params

                # Choose appropriate timeout based on endpoint
                if "/market/candles" in endpoint:
                    timeout = self.heavy_timeout
                elif "/health" in endpoint:
                    timeout = self.quick_timeout
                else:
                    timeout = self.normal_timeout
                request_kwargs["timeout"] = timeout

                if self._session is not None:
                    async with self._session.request(
                        method, url, **request_kwargs
                    ) as resp:
                        self._session_request_count += 1

                        if resp.status == 200:
                            data = await resp.json()
                            self._record_success()
                            logger.debug(
                                "Request successful",
                                extra={
                                    "client_id": self.client_id,
                                    "operation": operation,
                                    "status_code": resp.status,
                                    "response_size": len(str(data)),
                                    "session_requests": self._session_request_count,
                                },
                            )
                            return data

                        elif resp.status == 401:
                            error_msg = (
                                "Authentication failed - check BLUEFIN_SERVICE_API_KEY"
                            )
                            logger.error(
                                "Authentication failed",
                                extra={
                                    "client_id": self.client_id,
                                    "operation": operation,
                                    "status_code": resp.status,
                                    "endpoint": endpoint,
                                },
                            )
                            raise BluefinServiceAuthError(error_msg)

                        elif resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", "60"))
                            error_msg = f"Rate limit exceeded, retry after {retry_after} seconds"
                            logger.warning(
                                "Rate limit exceeded",
                                extra={
                                    "client_id": self.client_id,
                                    "operation": operation,
                                    "status_code": resp.status,
                                    "retry_after": retry_after,
                                    "endpoint": endpoint,
                                },
                            )

                            # For rate limiting, wait and retry if we have attempts left
                            if attempt < self._max_retries - 1:
                                wait_time = min(retry_after, 30)  # Cap wait time at 30s
                                logger.info(
                                    f"Waiting {wait_time}s for rate limit reset",
                                    extra={
                                        "client_id": self.client_id,
                                        "operation": operation,
                                    },
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                raise BluefinServiceRateLimitError(
                                    error_msg, retry_after=retry_after
                                )

                        else:
                            error_text = await resp.text()
                            self._record_failure()

                            if attempt < self._max_retries - 1:
                                delay = self._calculate_retry_delay(attempt)
                                logger.warning(
                                    f"Request failed, retrying in {delay:.2f}s",
                                    extra={
                                        "client_id": self.client_id,
                                        "operation": operation,
                                        "status_code": resp.status,
                                        "error_response": error_text[:200],
                                        "attempt": attempt + 1,
                                        "max_retries": self._max_retries,
                                        "delay": delay,
                                    },
                                )
                                await asyncio.sleep(delay)
                                continue
                            else:
                                error_msg = f"Request failed: HTTP {resp.status}"
                                logger.error(
                                    "All retry attempts failed",
                                    extra={
                                        "client_id": self.client_id,
                                        "operation": operation,
                                        "status_code": resp.status,
                                        "error_response": error_text[:200],
                                        "total_attempts": attempt + 1,
                                    },
                                )
                                raise BluefinServiceConnectionError(
                                    error_msg,
                                    status_code=resp.status,
                                    service_url=self.service_url,
                                )

            except (
                BluefinServiceAuthError,
                BluefinServiceRateLimitError,
                BluefinServiceConnectionError,
            ):
                # Re-raise our custom exceptions without retry
                raise
            except aiohttp.ClientError as e:
                self._record_failure()

                if attempt < self._max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Network error, retrying in {delay:.2f}s",
                        extra={
                            "client_id": self.client_id,
                            "operation": operation,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "attempt": attempt + 1,
                            "max_retries": self._max_retries,
                            "delay": delay,
                        },
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    error_msg = (
                        f"Network error after {self._max_retries} retries: {e!s}"
                    )
                    logger.error(
                        "Network error - all retries exhausted",
                        extra={
                            "client_id": self.client_id,
                            "operation": operation,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "total_attempts": attempt + 1,
                        },
                    )
                    raise BluefinServiceConnectionError(
                        error_msg, service_url=self.service_url
                    ) from e
            except Exception as e:
                self._record_failure()
                error_msg = f"Unexpected error in {operation}: {e!s}"
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "client_id": self.client_id,
                        "operation": operation,
                        "attempt": attempt + 1,
                        "unexpected_request_error": True,
                        "method": method,
                        "endpoint": endpoint,
                    },
                    component="BluefinServiceClient",
                    operation="_make_request_with_retry",
                )
                raise BluefinServiceConnectionError(
                    error_msg, service_url=self.service_url
                ) from e

        # This should never be reached due to the logic above
        raise BluefinServiceConnectionError(
            f"Request failed after {self._max_retries} attempts",
            service_url=self.service_url,
        )

    def _validate_candle_data(self, candles: list[Any]) -> bool:
        """
        Validate candle data for consistency and correctness.

        Args:
            candles: List of candle arrays

        Returns:
            True if data is valid
        """
        if not candles:
            logger.warning("⚠️ Empty candle data")
            return False

        valid_count = 0
        total_candles = len(candles)

        for i, candle in enumerate(candles):
            if not isinstance(candle, list) or len(candle) < 6:
                logger.debug(f"⚠️ Invalid candle format at index {i}: {candle}")
                continue

            try:
                timestamp, open_p, high_p, low_p, close_p, volume = candle[0:6]

                # Convert and validate data types more robustly
                try:
                    # Handle timestamps as strings, ints, or floats
                    if isinstance(timestamp, str):
                        timestamp = float(timestamp)
                    timestamp = float(timestamp)

                    # Convert price values to float, handling strings
                    open_p = float(open_p) if open_p is not None else 0.0
                    high_p = float(high_p) if high_p is not None else 0.0
                    low_p = float(low_p) if low_p is not None else 0.0
                    close_p = float(close_p) if close_p is not None else 0.0
                    volume = float(volume) if volume is not None else 0.0

                except (ValueError, TypeError) as e:
                    logger.debug(f"⚠️ Type conversion failed for candle {i}: {e}")
                    continue

                # Basic sanity checks
                if timestamp <= 0:
                    logger.debug(f"⚠️ Invalid timestamp in candle {i}: {timestamp}")
                    continue

                # Validate price values are positive (but allow 0 volume)
                if not all(x > 0 for x in [open_p, high_p, low_p, close_p]):
                    logger.debug(
                        f"⚠️ Non-positive prices in candle {i}: O={open_p}, H={high_p}, L={low_p}, C={close_p}"
                    )
                    continue

                # Volume can be 0 but not negative
                if volume < 0:
                    logger.debug(f"⚠️ Negative volume in candle {i}: {volume}")
                    continue

                # Check for reasonable price ranges (avoid extreme outliers)
                prices = [open_p, high_p, low_p, close_p]
                max_price = max(prices)
                min_price = min(prices)
                avg_price = sum(prices) / 4

                # Try to fix obvious OHLC relationship errors from data source
                max_oc = max(open_p, close_p)
                min_oc = min(open_p, close_p)

                # Fix high if it's too low
                if high_p < max_oc:
                    logger.debug(
                        f"🔧 Fixing high price: {high_p} -> {max_oc} (candle {i})"
                    )
                    high_p = max_oc

                # Fix low if it's too high
                if low_p > min_oc:
                    logger.debug(
                        f"🔧 Fixing low price: {low_p} -> {min_oc} (candle {i})"
                    )
                    low_p = min_oc

                # Validate OHLC relationships with tolerance for floating point precision
                tolerance = max(
                    avg_price * 1e-6, 1e-8
                )  # 0.0001% of price or 1e-8, whichever is larger
                if not (
                    high_p >= max(open_p, close_p) - tolerance
                    and low_p <= min(open_p, close_p) + tolerance
                ):
                    logger.debug(
                        f"⚠️ Invalid OHLC relationship in candle {i}: O={open_p}, H={high_p}, L={low_p}, C={close_p}, tolerance={tolerance}"
                    )
                    continue
                if max_price - min_price > avg_price * 0.5:
                    logger.debug(
                        f"⚠️ Suspicious price spread in candle {i}: range={max_price-min_price:.6f}, avg={avg_price:.6f}"
                    )
                    continue

                valid_count += 1

            except (ValueError, IndexError, TypeError) as e:
                logger.debug(f"⚠️ Error validating candle {i}: {e}")
                continue

        validation_rate = valid_count / total_candles if total_candles else 0

        # Lower the threshold to 40% since we've identified data quality issues from the source
        if validation_rate < 0.4:
            logger.warning(
                f"⚠️ Low validation rate: {validation_rate:.1%} ({valid_count}/{total_candles} valid)"
            )

            # Analyze validation failure patterns
            failure_reasons = {
                "format": 0,
                "type_conversion": 0,
                "timestamp": 0,
                "negative_prices": 0,
                "negative_volume": 0,
                "ohlc_relationship": 0,
                "price_spread": 0,
            }

            for _, candle in enumerate(
                candles[:20]
            ):  # Check first 20 for pattern analysis
                if not isinstance(candle, list) or len(candle) < 6:
                    failure_reasons["format"] += 1
                    continue

                try:
                    timestamp, open_p, high_p, low_p, close_p, volume = candle[0:6]

                    # Check type conversion
                    try:
                        if isinstance(timestamp, str):
                            timestamp = float(timestamp)
                        timestamp = float(timestamp)
                        open_p = float(open_p) if open_p is not None else 0.0
                        high_p = float(high_p) if high_p is not None else 0.0
                        low_p = float(low_p) if low_p is not None else 0.0
                        close_p = float(close_p) if close_p is not None else 0.0
                        volume = float(volume) if volume is not None else 0.0
                    except (ValueError, TypeError):
                        failure_reasons["type_conversion"] += 1
                        continue

                    # Check timestamp
                    if timestamp <= 0:
                        failure_reasons["timestamp"] += 1
                        continue

                    # Check negative prices
                    if not all(x > 0 for x in [open_p, high_p, low_p, close_p]):
                        failure_reasons["negative_prices"] += 1
                        continue

                    # Check volume
                    if volume < 0:
                        failure_reasons["negative_volume"] += 1
                        continue

                    # Check OHLC relationships
                    prices_temp = [open_p, high_p, low_p, close_p]
                    avg_price_temp = sum(prices_temp) / 4
                    tolerance = max(avg_price_temp * 1e-6, 1e-8)
                    if not (
                        high_p >= max(open_p, close_p) - tolerance
                        and low_p <= min(open_p, close_p) + tolerance
                    ):
                        failure_reasons["ohlc_relationship"] += 1
                        continue

                    # Check price spread
                    prices = [open_p, high_p, low_p, close_p]
                    max_price = max(prices)
                    min_price = min(prices)
                    avg_price = sum(prices) / 4
                    if max_price - min_price > avg_price * 0.5:
                        failure_reasons["price_spread"] += 1
                        continue

                except Exception:
                    failure_reasons["format"] += 1

            logger.warning(f"📊 Validation failure analysis: {failure_reasons}")

            # Log a few sample candles with OHLC relationship validation details
            sample_count = 0
            for i, candle in enumerate(candles[:10]):
                if sample_count >= 5:
                    break
                try:
                    if isinstance(candle, list) and len(candle) >= 6:
                        timestamp, open_p, high_p, low_p, close_p, volume = candle[0:6]
                        # Convert values
                        open_p = float(open_p) if open_p is not None else 0.0
                        high_p = float(high_p) if high_p is not None else 0.0
                        low_p = float(low_p) if low_p is not None else 0.0
                        close_p = float(close_p) if close_p is not None else 0.0
                        volume = float(volume) if volume is not None else 0.0

                        # Check OHLC relationship
                        max_oc = max(open_p, close_p)
                        min_oc = min(open_p, close_p)
                        high_ok = high_p >= max_oc
                        low_ok = low_p <= min_oc

                        logger.warning(
                            f"Sample candle {i}: O={open_p:.8f}, H={high_p:.8f}, L={low_p:.8f}, C={close_p:.8f}, V={volume:.4f}"
                        )
                        logger.warning(
                            f"  OHLC check: H>={max_oc:.8f}? {high_ok} ({high_p:.8f} >= {max_oc:.8f}), L<={min_oc:.8f}? {low_ok} ({low_p:.8f} <= {min_oc:.8f})"
                        )
                        sample_count += 1
                except Exception as e:
                    exception_handler.log_exception_with_context(
                        e,
                        {
                            "client_id": self.client_id,
                            "candle_index": i,
                            "candle_validation_error": True,
                            "operation": "validate_sample_candle",
                        },
                        component="BluefinServiceClient",
                        operation="_validate_candle_data",
                    )
                    sample_count += 1

            return False

        logger.info(
            f"✅ Candle validation passed: {validation_rate:.1%} ({valid_count}/{total_candles} valid)"
        )
        return True

    # Comprehensive Connectivity Monitoring Methods

    def _record_request_metrics(
        self, success: bool, response_time: float = 0.0, error_type: str = None
    ) -> None:
        """Record metrics for a request."""
        from datetime import UTC, datetime

        self.connection_metrics["total_requests"] += 1

        if success:
            self.connection_metrics["successful_requests"] += 1
            self.connection_metrics["last_successful_request"] = datetime.now(UTC)

            # Record response time (keep rolling window of 100)
            self.connection_metrics["response_times"].append(response_time)
            if len(self.connection_metrics["response_times"]) > 100:
                self.connection_metrics["response_times"] = self.connection_metrics[
                    "response_times"
                ][-100:]
        else:
            self.connection_metrics["failed_requests"] += 1
            self.connection_metrics["last_failed_request"] = datetime.now(UTC)

            if error_type:
                if error_type not in self.connection_metrics["error_counts_by_type"]:
                    self.connection_metrics["error_counts_by_type"][error_type] = 0
                self.connection_metrics["error_counts_by_type"][error_type] += 1

    def _record_health_check_result(
        self, success: bool, response_time: float = 0.0, status_code: int = None
    ) -> None:
        """Record health check result for monitoring."""
        from datetime import UTC, datetime

        result = {
            "timestamp": datetime.now(UTC),
            "success": success,
            "response_time_ms": response_time * 1000,
            "status_code": status_code,
        }

        self.connection_metrics["health_check_history"].append(result)

        # Keep only last 50 health checks
        if len(self.connection_metrics["health_check_history"]) > 50:
            self.connection_metrics["health_check_history"] = self.connection_metrics[
                "health_check_history"
            ][-50:]

    async def get_connectivity_status(self) -> dict[str, Any]:
        """Get comprehensive connectivity status and metrics."""
        from datetime import UTC, datetime

        status = {
            "timestamp": datetime.now(UTC).isoformat(),
            "service_url": self.service_url,
            "client_id": self.client_id,
            "connection": {
                "connected": self._connected,
                "session_active": self._session is not None
                and not self._session.closed,
                "consecutive_failures": self.consecutive_failures,
                "last_health_check": (
                    datetime.fromtimestamp(self.last_health_check, UTC).isoformat()
                    if self.last_health_check > 0
                    else None
                ),
                "circuit_breaker_open": self._is_circuit_open(),
            },
            "metrics": {},
            "performance": {},
            "websocket": self.websocket_status.copy(),
            "health_checks": [],
        }

        # Connection metrics
        total_requests = self.connection_metrics["total_requests"]
        successful_requests = self.connection_metrics["successful_requests"]
        failed_requests = self.connection_metrics["failed_requests"]

        status["metrics"] = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (
                (successful_requests / total_requests * 100)
                if total_requests > 0
                else 0
            ),
            "error_rate": (
                (failed_requests / total_requests * 100) if total_requests > 0 else 0
            ),
            "connection_attempts": self.connection_metrics["connection_attempts"],
            "successful_connections": self.connection_metrics["successful_connections"],
            "last_successful_request": (
                self.connection_metrics["last_successful_request"].isoformat()
                if self.connection_metrics["last_successful_request"]
                else None
            ),
            "last_failed_request": (
                self.connection_metrics["last_failed_request"].isoformat()
                if self.connection_metrics["last_failed_request"]
                else None
            ),
            "error_counts_by_type": self.connection_metrics[
                "error_counts_by_type"
            ].copy(),
        }

        # Performance metrics
        response_times = self.connection_metrics["response_times"]
        if response_times:
            status["performance"] = {
                "average_response_time_ms": sum(response_times)
                / len(response_times)
                * 1000,
                "min_response_time_ms": min(response_times) * 1000,
                "max_response_time_ms": max(response_times) * 1000,
                "response_time_samples": len(response_times),
            }

            # Calculate percentiles if we have enough samples
            if len(response_times) >= 10:
                sorted_times = sorted(response_times)
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)

                status["performance"]["p95_response_time_ms"] = (
                    sorted_times[p95_index] * 1000
                )
                status["performance"]["p99_response_time_ms"] = (
                    sorted_times[p99_index] * 1000
                )
        else:
            status["performance"] = {
                "average_response_time_ms": 0,
                "response_time_samples": 0,
            }

        # Recent health checks
        status["health_checks"] = [
            {
                "timestamp": hc["timestamp"].isoformat(),
                "success": hc["success"],
                "response_time_ms": hc["response_time_ms"],
                "status_code": hc["status_code"],
            }
            for hc in self.connection_metrics["health_check_history"][
                -10:
            ]  # Last 10 checks
        ]

        return status

    async def run_comprehensive_connectivity_test(self) -> dict[str, Any]:
        """Run a comprehensive connectivity test with detailed results."""
        import time
        from datetime import UTC, datetime

        test_start = time.time()
        test_results = {
            "test_id": f"conn_test_{int(test_start)}",
            "timestamp": datetime.now(UTC).isoformat(),
            "service_url": self.service_url,
            "tests": {},
            "overall_status": "unknown",
            "recommendations": [],
        }

        # Test 1: Basic connectivity
        basic_test_start = time.time()
        try:
            # Ensure we have a session for testing
            await self._ensure_session()

            if self._session and not self._session.closed:
                async with self._session.get(
                    f"{self.service_url}/health", timeout=self.quick_timeout
                ) as resp:
                    response_time = time.time() - basic_test_start

                    test_results["tests"]["basic_connectivity"] = {
                        "status": "pass" if resp.status == 200 else "fail",
                        "response_time_ms": response_time * 1000,
                        "status_code": resp.status,
                        "details": "Basic HTTP connectivity test",
                    }
            else:
                test_results["tests"]["basic_connectivity"] = {
                    "status": "fail",
                    "response_time_ms": (time.time() - basic_test_start) * 1000,
                    "error": "No session available",
                    "details": "Could not establish session",
                }

        except Exception as e:
            test_results["tests"]["basic_connectivity"] = {
                "status": "fail",
                "response_time_ms": (time.time() - basic_test_start) * 1000,
                "error": str(e),
                "details": "Exception during basic connectivity test",
            }

        # Test 2: Detailed health check
        detailed_test_start = time.time()
        try:
            if self._session and not self._session.closed:
                async with self._session.get(
                    f"{self.service_url}/health/detailed", timeout=self.normal_timeout
                ) as resp:
                    response_time = time.time() - detailed_test_start

                    if resp.status == 200:
                        health_data = await resp.json()
                        test_results["tests"]["detailed_health"] = {
                            "status": "pass",
                            "response_time_ms": response_time * 1000,
                            "status_code": resp.status,
                            "health_data": health_data,
                            "details": "Detailed health endpoint test",
                        }
                    else:
                        test_results["tests"]["detailed_health"] = {
                            "status": "degraded",
                            "response_time_ms": response_time * 1000,
                            "status_code": resp.status,
                            "details": "Detailed health endpoint returned non-200 status",
                        }
            else:
                test_results["tests"]["detailed_health"] = {
                    "status": "fail",
                    "error": "No session available",
                    "details": "Could not test detailed health endpoint",
                }

        except Exception as e:
            test_results["tests"]["detailed_health"] = {
                "status": "fail",
                "response_time_ms": (time.time() - detailed_test_start) * 1000,
                "error": str(e),
                "details": "Exception during detailed health test",
            }

        # Test 3: Authentication test
        auth_test_start = time.time()
        try:
            if self._session and not self._session.closed:
                async with self._session.get(
                    f"{self.service_url}/account", timeout=self.normal_timeout
                ) as resp:
                    response_time = time.time() - auth_test_start

                    if resp.status == 200:
                        test_results["tests"]["authentication"] = {
                            "status": "pass",
                            "response_time_ms": response_time * 1000,
                            "status_code": resp.status,
                            "details": "Authentication working correctly",
                        }
                    elif resp.status == 401:
                        test_results["tests"]["authentication"] = {
                            "status": "fail",
                            "response_time_ms": response_time * 1000,
                            "status_code": resp.status,
                            "details": "Authentication failed - check API key",
                        }
                        test_results["recommendations"].append(
                            "Check BLUEFIN_SERVICE_API_KEY environment variable"
                        )
                    else:
                        test_results["tests"]["authentication"] = {
                            "status": "degraded",
                            "response_time_ms": response_time * 1000,
                            "status_code": resp.status,
                            "details": f"Unexpected status code: {resp.status}",
                        }
            else:
                test_results["tests"]["authentication"] = {
                    "status": "fail",
                    "error": "No session available",
                    "details": "Could not test authentication",
                }

        except Exception as e:
            test_results["tests"]["authentication"] = {
                "status": "fail",
                "response_time_ms": (time.time() - auth_test_start) * 1000,
                "error": str(e),
                "details": "Exception during authentication test",
            }

        # Test 4: Performance test (multiple quick requests)
        perf_test_start = time.time()
        try:
            if self._session and not self._session.closed:
                response_times = []
                success_count = 0

                for i in range(5):  # 5 quick requests
                    req_start = time.time()
                    try:
                        async with self._session.get(
                            f"{self.service_url}/health", timeout=self.quick_timeout
                        ) as resp:
                            req_time = time.time() - req_start
                            response_times.append(req_time)
                            if resp.status == 200:
                                success_count += 1
                    except Exception:
                        response_times.append(time.time() - req_start)

                avg_response_time = (
                    sum(response_times) / len(response_times) if response_times else 0
                )
                success_rate = success_count / 5 * 100

                test_results["tests"]["performance"] = {
                    "status": (
                        "pass"
                        if success_rate >= 80
                        else "degraded"
                        if success_rate >= 60
                        else "fail"
                    ),
                    "response_time_ms": avg_response_time * 1000,
                    "success_rate": success_rate,
                    "total_requests": 5,
                    "successful_requests": success_count,
                    "details": f"Performance test: {success_rate:.1f}% success rate",
                }

                if avg_response_time > 2.0:
                    test_results["recommendations"].append(
                        "High response times detected - check service performance"
                    )
                if success_rate < 100:
                    test_results["recommendations"].append(
                        "Some requests failed - check service stability"
                    )
            else:
                test_results["tests"]["performance"] = {
                    "status": "fail",
                    "error": "No session available",
                    "details": "Could not run performance test",
                }

        except Exception as e:
            test_results["tests"]["performance"] = {
                "status": "fail",
                "response_time_ms": (time.time() - perf_test_start) * 1000,
                "error": str(e),
                "details": "Exception during performance test",
            }

        # Determine overall status
        test_statuses = [
            test.get("status", "fail") for test in test_results["tests"].values()
        ]

        if all(status == "pass" for status in test_statuses):
            test_results["overall_status"] = "healthy"
        elif any(status == "pass" for status in test_statuses):
            test_results["overall_status"] = "degraded"
        else:
            test_results["overall_status"] = "unhealthy"

        # Add general recommendations
        if not test_results["recommendations"]:
            if test_results["overall_status"] == "healthy":
                test_results["recommendations"] = [
                    "All connectivity tests passed - service is operating normally"
                ]
            else:
                test_results["recommendations"].append(
                    "Review failed tests and check service configuration"
                )

        test_results["total_test_time_ms"] = (time.time() - test_start) * 1000

        return test_results

    def get_connection_health_summary(self) -> dict[str, Any]:
        """Get a concise connection health summary."""
        from datetime import UTC, datetime

        total_requests = self.connection_metrics["total_requests"]
        successful_requests = self.connection_metrics["successful_requests"]

        # Recent health checks (last 10)
        recent_health_checks = self.connection_metrics["health_check_history"][-10:]
        recent_success_rate = 0
        if recent_health_checks:
            recent_successes = sum(1 for hc in recent_health_checks if hc["success"])
            recent_success_rate = recent_successes / len(recent_health_checks) * 100

        # Average response time from recent samples
        response_times = self.connection_metrics["response_times"][
            -20:
        ]  # Last 20 samples
        avg_response_time = (
            sum(response_times) / len(response_times) * 1000 if response_times else 0
        )

        health_status = "healthy"
        if not self._connected or self.consecutive_failures >= 3:
            health_status = "unhealthy"
        elif recent_success_rate < 80 or avg_response_time > 5000:
            health_status = "degraded"

        return {
            "overall_health": health_status,
            "connected": self._connected,
            "consecutive_failures": self.consecutive_failures,
            "circuit_breaker_open": self._is_circuit_open(),
            "success_rate": (
                (successful_requests / total_requests * 100)
                if total_requests > 0
                else 0
            ),
            "recent_health_check_success_rate": recent_success_rate,
            "average_response_time_ms": avg_response_time,
            "last_health_check": (
                datetime.fromtimestamp(self.last_health_check, UTC).isoformat()
                if self.last_health_check > 0
                else None
            ),
            "total_requests": total_requests,
            "websocket_connected": self.websocket_status["connected"],
        }

    async def reset_connectivity_metrics(self) -> None:
        """Reset all connectivity metrics."""

        logger.info(
            "Resetting connectivity metrics", extra={"client_id": self.client_id}
        )

        self.connection_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "connection_attempts": 0,
            "successful_connections": 0,
            "last_successful_request": None,
            "last_failed_request": None,
            "response_times": [],
            "error_counts_by_type": {},
            "health_check_history": [],
        }

        self.websocket_status = {
            "connected": False,
            "last_ping": None,
            "reconnect_attempts": 0,
            "messages_received": 0,
            "messages_sent": 0,
        }

        # Reset circuit breaker
        self.reset_circuit_breaker(force=True)

    async def get_balance_operation_health(self) -> dict[str, Any]:
        """
        Get specific health metrics for balance operations.

        Returns:
            Dictionary with balance operation health information
        """
        current_time = time.time()

        # Check if service supports health endpoint
        service_health = None
        try:
            # Try to get circuit breaker status from service if it has the method
            if hasattr(self, "_service_client") and self._service_client:
                if hasattr(self._service_client, "get_circuit_breaker_status"):
                    service_health = self._service_client.get_circuit_breaker_status()
        except Exception as e:
            logger.debug(
                f"Could not get service health status: {e}",
                extra={"client_id": self.client_id},
            )

        return {
            "balance_operation_failures": getattr(
                self, "balance_operation_failures", 0
            ),
            "last_successful_balance_fetch": getattr(
                self, "last_successful_balance_fetch", 0
            ),
            "time_since_last_success": (
                current_time - getattr(self, "last_successful_balance_fetch", 0)
                if getattr(self, "last_successful_balance_fetch", 0) > 0
                else None
            ),
            "balance_staleness_threshold": getattr(
                self, "balance_staleness_threshold", 300
            ),
            "is_balance_stale": (
                current_time - getattr(self, "last_successful_balance_fetch", 0)
                > getattr(self, "balance_staleness_threshold", 300)
                if getattr(self, "last_successful_balance_fetch", 0) > 0
                else True
            ),
            "service_circuit_breaker": service_health,
            "client_circuit_breaker_open": self._is_circuit_open(),
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
