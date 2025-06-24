"""
Bluefin Service Client for connecting to the isolated Bluefin SDK service.

This client communicates with the Bluefin service container that has the
actual Bluefin SDK installed, avoiding dependency conflicts.
"""

import asyncio
import logging
import os
import secrets
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import aiohttp
from aiohttp import ClientError, ClientTimeout, TCPConnector

from bot.error_handling import exception_handler

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

    @staticmethod
    def _detect_docker_environment() -> bool:
        """Detect if we're running inside a Docker container."""
        # Check for Docker-specific files
        if os.path.exists("/.dockerenv"):
            return True

        # Check for Docker in cgroup
        try:
            with open("/proc/self/cgroup") as f:
                if "docker" in f.read():
                    return True
        except (FileNotFoundError, PermissionError):
            pass

        # Check for common Docker environment variables
        docker_env_vars = ["DOCKER_CONTAINER", "DOCKER_HOST", "KUBERNETES_SERVICE_HOST"]
        if any(os.getenv(var) for var in docker_env_vars):
            return True

        # Check hostname (often container ID in Docker)
        try:
            import socket

            hostname = socket.gethostname()
            # Docker container hostnames are typically 12-character hex strings
            if len(hostname) == 12 and all(c in "0123456789abcdef" for c in hostname):
                return True
        except Exception:
            pass

        return False

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
        # Detect if we're running inside Docker
        self.is_docker = self._detect_docker_environment()

        # Primary service URL (can be overridden)
        self.primary_service_url = service_url

        # Build fallback URLs based on environment
        if self.is_docker:
            # Inside Docker, prioritize Docker service name
            self.fallback_urls = [
                "http://bluefin-service:8080",  # Docker service name
                "http://host.docker.internal:8080",  # Docker Desktop host
                "http://172.17.0.1:8080",  # Default Docker bridge gateway
                "http://localhost:8080",  # Fallback to localhost
                "http://127.0.0.1:8080",  # Fallback to localhost IP
            ]
        else:
            # Outside Docker, prioritize localhost
            self.fallback_urls = [
                "http://localhost:8080",  # Local host
                "http://127.0.0.1:8080",  # Localhost IP
                "http://bluefin-service:8080",  # Try Docker service name
                "http://host.docker.internal:8080",  # Docker Desktop on macOS/Windows
            ]

        # Remove duplicates and put primary URL first
        self.service_urls = [self.primary_service_url]
        for url in self.fallback_urls:
            if url not in self.service_urls:
                self.service_urls.append(url)

        # Current active URL and tracking
        self.service_url = self.primary_service_url
        self.current_url_index = 0
        self.url_failure_counts: dict[str, int] = dict.fromkeys(self.service_urls, 0)
        self.last_successful_url = None
        self.service_discovery_complete = False

        logger.info(
            "Bluefin client initialized with service discovery",
            extra={
                "is_docker": self.is_docker,
                "primary_url": self.primary_service_url,
                "fallback_urls": self.fallback_urls[:3],  # Log first 3 for brevity
                "total_urls": len(self.service_urls),
            },
        )
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
        self.connection_metrics: dict[str, Any] = {
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

        # Enhanced circuit breaker state (per URL)
        self.circuit_states: dict[str, dict[str, Any]] = {}
        for url in self.service_urls:
            self.circuit_states[url] = {
                "open": False,
                "open_until": 0.0,
                "consecutive_failures": 0,
                "last_attempt": 0.0,
            }
        self.circuit_failure_threshold = 3  # Reduced threshold for faster failover
        self.circuit_recovery_timeout = 60  # seconds

        # Legacy circuit breaker properties for backward compatibility
        self.circuit_open = False
        self.circuit_open_until = 0.0

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

        # Get API key from parameter or environment - make it optional for graceful degradation
        self.api_key = api_key or os.getenv("BLUEFIN_SERVICE_API_KEY")
        if not self.api_key:
            # Generate a dummy API key for testing/development when service might not be running
            self.api_key = "dummy-api-key-for-optional-service"
            logger.warning(
                "BLUEFIN_SERVICE_API_KEY not provided - using dummy key. Service may not work properly.",
                extra={
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "auth_configured": False,
                    "required_env_var": "BLUEFIN_SERVICE_API_KEY",
                },
            )

        # Validate API key format only if it's not the dummy key
        if (
            self.api_key != "dummy-api-key-for-optional-service"
            and len(self.api_key) < 16
        ):
            logger.warning(
                "API key appears to be invalid (too short) - service may not authenticate properly",
                extra={
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "auth_configured": False,
                    "key_length": len(self.api_key),
                },
            )

        logger.info(
            "Bluefin service client initialized with valid authentication",
            extra={
                "client_id": self.client_id,
                "service_url": self.service_url,
                "auth_configured": True,
            },
        )

        # Balance operation audit trail
        self.balance_operation_audit: list[dict[str, Any]] = []
        self.max_balance_audit_entries = 500

        # Performance tracking for balance operations
        self.balance_performance_metrics: dict[str, Any] = {
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
        if self.api_key and self.api_key != "dummy-api-key-for-optional-service":
            self._headers["Authorization"] = f"Bearer {self.api_key}"

    def _validate_parameters(self, **kwargs) -> bool:
        """Validate common request parameters."""
        for key, value in kwargs.items():
            if key == "symbol" and value:
                if not isinstance(value, str) or len(value) < 3:
                    logger.error("Invalid symbol parameter: %s", value)
                    return False
            elif key == "limit" and value is not None:
                if not isinstance(value, int) or value < 1 or value > 1000:
                    logger.error("Invalid limit parameter: %s", value)
                    return False
            elif (
                key in ["startTime", "endTime"]
                and value
                and (not isinstance(value, int) or value <= 0)
            ):
                logger.error("Invalid timestamp parameter %s: %s", key, value)
                return False
        return True

    def _record_balance_operation_audit(
        self,
        correlation_id: str,
        operation: str,
        status: str,
        balance_amount: str | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
        response_size: int | None = None,
        metadata: dict[str, Any] | None = None,
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
            "Balance client audit: %s - %s",
            operation,
            status,
            extra={
                "audit_entry": audit_entry,
                "operation_type": "balance_client_audit",
            },
        )

    def _update_balance_performance_metrics(
        self, status: str, duration_ms: float | None = None, error: str | None = None
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
                self.balance_performance_metrics["balance_response_times"].append(  # type: ignore[index]
                    duration_ms
                )
                # Keep only recent response times
                if (
                    len(self.balance_performance_metrics["balance_response_times"])  # type: ignore[arg-type]
                    > 100
                ):
                    self.balance_performance_metrics["balance_response_times"] = (  # type: ignore[index]
                        self.balance_performance_metrics["balance_response_times"][  # type: ignore[index]
                            -100:
                        ]
                    )

                # Update average
                times = self.balance_performance_metrics["balance_response_times"]
                self.balance_performance_metrics["average_balance_response_time"] = sum(  # type: ignore[index]
                    times
                ) / len(
                    times
                )  # type: ignore[arg-type]
        else:
            self.balance_performance_metrics["failed_balance_requests"] += 1

            if error:
                error_type = (
                    type(error).__name__ if isinstance(error, Exception) else str(error)
                )
                if (
                    error_type
                    not in self.balance_performance_metrics["balance_error_counts"]  # type: ignore[operator]
                ):
                    self.balance_performance_metrics["balance_error_counts"][  # type: ignore[index]
                        error_type
                    ] = 0
                self.balance_performance_metrics["balance_error_counts"][  # type: ignore[index]
                    error_type
                ] += 1

    def _get_balance_error_context(
        self,
        correlation_id: str,
        operation: str,
        error: Exception,
        duration_ms: float | None = None,
        endpoint: str | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        if "account" in error_str:
            return "account_related"
        if "timeout" in error_str or "timeout" in error_type:
            return "network_timeout"
        if "connection" in error_str or "connection" in error_type:
            return "network_connection"
        if "authentication" in error_str or "auth" in error_str:
            return "authentication"
        if "rate" in error_str and "limit" in error_str:
            return "rate_limiting"
        if "502" in error_str or "503" in error_str or "504" in error_str:
            return "service_unavailable"
        return "unknown"

    def _get_actionable_balance_error_message(self, error: Exception) -> str:
        """Generate actionable error message for balance operations."""
        error_str = str(error).lower()
        type(error).__name__.lower()

        if "balance" in error_str:
            return (
                "Balance retrieval failed. Account may not be initialized on Bluefin."
            )
        if "account" in error_str:
            return "Account access failed. Verify account exists and is properly configured."
        if "timeout" in error_str:
            return "Request timed out. Check network connection and retry."
        if "connection" in error_str:
            return (
                "Connection failed. Verify Bluefin service is running and accessible."
            )
        if "authentication" in error_str or "auth" in error_str:
            return "Authentication failed. Check API key configuration."
        if "rate" in error_str and "limit" in error_str:
            return "Rate limit exceeded. Wait before making another request."
        return "Balance operation failed. Check logs for details."

    def _should_retry_balance_error(self, error: Exception) -> bool:
        """Determine if balance error should be retried."""
        error_str = str(error).lower()
        type(error).__name__.lower()

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

    def get_balance_audit_trail(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent balance operation audit trail."""
        return self.balance_operation_audit[-limit:]

    def get_balance_performance_summary(self) -> dict[str, Any]:
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

    def _is_url_circuit_open(self, url: str) -> bool:
        """Check if circuit breaker is open for a specific URL."""
        if url not in self.circuit_states:
            return False

        circuit = self.circuit_states[url]
        if circuit["open"] and time.time() < circuit["open_until"]:
            return True
        if circuit["open"] and time.time() >= circuit["open_until"]:
            # Circuit breaker timeout expired, attempt to close
            circuit["open"] = False
            circuit["consecutive_failures"] = 0
            logger.info(
                "Circuit breaker closed for URL",
                extra={"url": url, "client_id": self.client_id},
            )
        return False

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open for current URL (backward compatibility)."""
        return self._is_url_circuit_open(self.service_url)

    def _open_circuit_for_url(self, url: str) -> None:
        """Open circuit breaker for a specific URL."""
        if url in self.circuit_states:
            self.circuit_states[url]["open"] = True
            self.circuit_states[url]["open_until"] = (
                time.time() + self.circuit_recovery_timeout
            )
            logger.warning(
                "Circuit breaker opened for URL",
                extra={
                    "url": url,
                    "recovery_timeout": self.circuit_recovery_timeout,
                    "consecutive_failures": self.circuit_states[url][
                        "consecutive_failures"
                    ],
                },
            )

    def _record_url_failure(self, url: str) -> None:
        """Record a failure for a specific URL."""
        if url in self.circuit_states:
            self.circuit_states[url]["consecutive_failures"] += 1
            self.circuit_states[url]["last_attempt"] = time.time()

            if (
                self.circuit_states[url]["consecutive_failures"]
                >= self.circuit_failure_threshold
            ):
                self._open_circuit_for_url(url)

        if url in self.url_failure_counts:
            self.url_failure_counts[url] += 1

    def _record_url_success(self, url: str) -> None:
        """Record a success for a specific URL."""
        if url in self.circuit_states:
            self.circuit_states[url]["consecutive_failures"] = 0
            self.circuit_states[url]["open"] = False

        self.last_successful_url = url
        self.service_url = url  # Update current URL to the successful one

    async def _try_connect_to_url(self, url: str) -> bool:
        """Try to connect to a specific URL."""
        if self._is_url_circuit_open(url):
            logger.debug(
                "Skipping URL - circuit breaker open",
                extra={"url": url, "client_id": self.client_id},
            )
            return False

        try:
            logger.debug(
                "Attempting connection to URL",
                extra={"url": url, "client_id": self.client_id},
            )

            # Temporarily set service_url for the health check
            original_url = self.service_url
            self.service_url = url

            # Try health check
            health_url = f"{url}/health"

            if self._session is not None:
                async with self._session.get(
                    health_url, timeout=self.quick_timeout
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "healthy":
                            self._record_url_success(url)
                            logger.info(
                                "Successfully connected to Bluefin service",
                                extra={
                                    "url": url,
                                    "client_id": self.client_id,
                                    "service_status": data.get("status"),
                                    "service_initialized": data.get("initialized"),
                                    "service_network": data.get("network"),
                                },
                            )
                            return True

            # If we get here, connection failed
            self.service_url = original_url
            self._record_url_failure(url)
            return False

        except (ClientError, TimeoutError, OSError) as e:
            self.service_url = original_url
            self._record_url_failure(url)
            logger.debug(
                "Connection attempt failed for URL",
                extra={
                    "url": url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "client_id": self.client_id,
                },
            )
            return False

    async def _discover_service(self) -> bool:
        """Discover which service URL is available."""
        logger.info(
            "Starting service discovery",
            extra={"urls_to_try": self.service_urls, "client_id": self.client_id},
        )

        # Try last successful URL first if available
        if self.last_successful_url:
            if await self._try_connect_to_url(self.last_successful_url):
                self.service_discovery_complete = True
                return True

        # Try all URLs in order
        for url in self.service_urls:
            if url == self.last_successful_url:
                continue  # Already tried

            if await self._try_connect_to_url(url):
                self.service_discovery_complete = True
                return True

        # All URLs failed
        logger.error(
            "Service discovery failed - all URLs unavailable",
            extra={
                "urls_tried": self.service_urls,
                "failure_counts": self.url_failure_counts,
                "client_id": self.client_id,
            },
        )
        return False

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        # For backward compatibility, check current URL
        return self._is_url_circuit_open(self.service_url)

    def _old_is_circuit_open(self) -> bool:
        """Legacy circuit breaker check."""
        if self.circuit_open and time.time() < self.circuit_open_until:
            return True
        if self.circuit_open and time.time() >= self.circuit_open_until:
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
        remaining_time = self.circuit_open_until - current_time
        logger.warning(
            "Circuit breaker reset not ready - %.1fs remaining",
            remaining_time,
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
                "Operation succeeded, resetting failure count from %s",
                self.consecutive_failures,
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
                "Circuit breaker opened due to %s consecutive failures. Will retry after %ss",
                self.consecutive_failures,
                self.circuit_recovery_timeout,
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

        # Use cryptographically secure random for jitter calculation
        jitter = (
            capped_delay * jitter_factor * ((secrets.randbelow(100) + 50) / 100.0)
        )  # 0.5 to 1.5

        final_delay = capped_delay + jitter

        logger.debug(
            "Calculated retry delay for %s",
            operation,
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
            except (TimeoutError, ClientError, aiohttp.ServerConnectionError) as e:
                last_exception = e
                self._record_failure()

                if attempt < self._max_retries - 1:
                    delay = self._calculate_retry_delay(attempt, "api_call")
                    logger.warning(
                        "Request failed (attempt %s/%s), retrying in %.2fs: %s",
                        attempt + 1,
                        self._max_retries,
                        delay,
                        e,
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
                    logger.exception(
                        "Request failed after %s attempts",
                        self._max_retries,
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
            else:
                return result

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
                            "Health check successful on attempt %s",
                            attempt + 1,
                            extra={
                                "client_id": self.client_id,
                                "status_code": resp.status,
                                "connected": self._connected,
                                "attempt": attempt + 1,
                            },
                        )
                        return True
                    if attempt < max_health_retries - 1:
                        logger.debug(
                            "Health check failed (status %s), retrying in %ss",
                            resp.status,
                            health_retry_delay,
                            extra={
                                "client_id": self.client_id,
                                "status_code": resp.status,
                                "attempt": attempt + 1,
                                "max_retries": max_health_retries,
                            },
                        )
                        await asyncio.sleep(health_retry_delay)
                        continue
                    logger.warning(
                        "Health check failed after %s attempts: status %s",
                        max_health_retries,
                        resp.status,
                        extra={
                            "client_id": self.client_id,
                            "status_code": resp.status,
                            "total_attempts": max_health_retries,
                        },
                    )
                else:
                    logger.warning(
                        "Health check session unavailable on attempt %s",
                        attempt + 1,
                        extra={"client_id": self.client_id, "attempt": attempt + 1},
                    )
                    if attempt < max_health_retries - 1:
                        await asyncio.sleep(health_retry_delay)
                        continue

                # If we reach here, this attempt failed
                self._connected = False

            except (ClientError, TimeoutError, OSError) as e:
                if attempt < max_health_retries - 1:
                    logger.debug(
                        "Health check network error on attempt %s, retrying: %s",
                        attempt + 1,
                        e,
                        extra={
                            "client_id": self.client_id,
                            "error_type": type(e).__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_health_retries,
                        },
                    )
                    await asyncio.sleep(health_retry_delay)
                    continue
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
            "Health check failed after %s resilient attempts",
            max_health_retries,
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
                else:
                    logger.debug(
                        "Closed existing session for recreation",
                        extra={"client_id": self.client_id},
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
                    "primary_url": self.primary_service_url,
                    "current_url": self.service_url,
                    "discovery_complete": self.service_discovery_complete,
                    "session_age": time.time() - self._session_created_at,
                    "request_count": self._session_request_count,
                },
            )

            # If we haven't discovered a working service yet, do it now
            if not self.service_discovery_complete:
                if await self._discover_service():
                    self._connected = True
                    self.last_health_check = time.time()
                    self._record_success()
                    return True
                self._connected = False
                self._record_failure()
                raise BluefinServiceConnectionError(
                    "Service unavailable - all connection attempts failed",
                    service_url="All URLs failed",
                )

            # Otherwise, just check the current URL
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
                        self._record_url_success(self.service_url)

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

                    # Health check failed, try rediscovery
                    error_text = await resp.text()
                    self._record_failure()
                    self._record_url_failure(self.service_url)
                    logger.warning(
                        "Bluefin service health check failed, attempting rediscovery",
                        extra={
                            "client_id": self.client_id,
                            "service_url": self.service_url,
                            "status_code": resp.status,
                            "error_response": error_text[:200],
                            "operation": "health_check",
                        },
                    )

                    # Force rediscovery
                    self.service_discovery_complete = False
                    if await self._discover_service():
                        self._connected = True
                        return True

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
            self._record_url_failure(self.service_url)

            # Try rediscovery on connection errors
            logger.warning(
                "Connection error, attempting service rediscovery",
                extra={
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            self.service_discovery_complete = False
            if await self._discover_service():
                self._connected = True
                return True

            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "connection_network_error": True,
                    "all_urls_failed": True,
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
        else:
            return True

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
            else:
                logger.debug(
                    "HTTP session closed successfully",
                    extra={
                        "client_id": self.client_id,
                        "final_request_count": self._session_request_count,
                    },
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

            logger.exception(
                "Account data request failed with exception",
                extra=error_context,
            )

            return {"error": f"Account data request failed: {e!s}"}
        else:
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
            logger.exception(
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

        Supports both signed (live trading) and unsigned (paper trading) orders.

        Args:
            order_data: Order details including symbol, side, quantity, etc.
                       For live trading, includes signature, publicKey, orderHash, etc.

        Returns:
            Order response dictionary
        """
        # Log order signature status
        has_signature = all(
            key in order_data for key in ["signature", "publicKey", "orderHash"]
        )
        if has_signature:
            logger.debug(
                "🔐 Placing SIGNED order - Hash: %s, PublicKey: %s",
                order_data.get("orderHash", "")[:16] + "...",
                order_data.get("publicKey", "")[:16] + "...",
            )
        else:
            logger.debug("📊 Placing UNSIGNED order (paper trading)")

        try:
            result = await self._make_request_with_retry(
                method="POST",
                endpoint="/orders",
                operation="place_order",
                json_data=order_data,
            )

        except BluefinServiceAuthError:
            logger.exception(
                "Authentication failed placing order",
                extra={"client_id": self.client_id, "operation": "place_order"},
            )
            return {"status": "error", "message": "Authentication failed"}

        except BluefinServiceRateLimitError as e:
            logger.exception(
                "Rate limited placing order",
                extra={"client_id": self.client_id, "operation": "place_order"},
            )
            return {
                "status": "error",
                "message": f"Rate limit exceeded, retry after {e.retry_after}s",
                "retry_after": e.retry_after,
            }

        except BluefinServiceConnectionError as e:
            logger.exception(
                "Connection error placing order",
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
        else:
            return {"status": "success", "order": result}

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
            logger.exception(
                "Service error canceling order %s",
                order_id,
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
            return await self._make_request_with_retry(
                method="GET",
                endpoint="/market/ticker",
                operation="get_market_ticker",
                params={"symbol": symbol},
            )

        except (
            BluefinServiceAuthError,
            BluefinServiceRateLimitError,
            BluefinServiceConnectionError,
        ) as e:
            logger.exception(
                "Service error getting ticker for %s",
                symbol,
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
            logger.exception(
                "Service error setting leverage for %s",
                symbol,
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
                    "✅ Successfully retrieved %s validated candles",
                    len(candles),
                    extra={
                        "client_id": self.client_id,
                        "operation": "get_candlestick_data",
                        "candle_count": len(candles),
                    },
                )
                return candles
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
            logger.exception(
                "Service error getting candlestick data",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_candlestick_data",
                    "error_type": type(e).__name__,
                },
            )
            return []
        except BluefinServiceConnectionError as e:
            logger.exception(
                "Connection error getting candlestick data",
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
        # Ensure we have a working service URL
        if not self.service_discovery_complete:
            if not await self._discover_service():
                raise BluefinServiceConnectionError(
                    "Service unavailable - all connection attempts failed",
                    service_url="All URLs failed",
                )

        # Track URLs we've tried for this request
        urls_tried = set()
        last_error = None

        # Get list of URLs to try (current first, then others)
        urls_to_try = [self.service_url]
        for url in self.service_urls:
            if url not in urls_to_try and not self._is_url_circuit_open(url):
                urls_to_try.append(url)

        # Try each URL until one works
        for url_index, base_url in enumerate(urls_to_try):
            urls_tried.add(base_url)

            # Skip if circuit is open for this URL
            if self._is_url_circuit_open(base_url):
                logger.debug(
                    "Skipping URL due to open circuit",
                    extra={
                        "url": base_url,
                        "client_id": self.client_id,
                        "operation": operation,
                    },
                )
                continue

            url = f"{base_url}{endpoint}"

            # Attempt request with retries for this specific URL
            for attempt in range(self._max_retries):
                try:
                    # Log URL switch if not the first URL
                    if url_index > 0 and attempt == 0:
                        logger.info(
                            "Switching to fallback URL",
                            extra={
                                "fallback_url": base_url,
                                "previous_url": (
                                    urls_to_try[url_index - 1]
                                    if url_index > 0
                                    else None
                                ),
                                "client_id": self.client_id,
                                "operation": operation,
                            },
                        )

                    # Rest of the original request logic...
                    await self._ensure_session()

                    logger.debug(
                        "Making HTTP request (attempt %s/%s)",
                        attempt + 1,
                        self._max_retries,
                        extra={
                            "client_id": self.client_id,
                            "method": method,
                            "url": url,
                            "operation": operation,
                            "attempt": attempt + 1,
                            "url_index": url_index + 1,
                            "total_urls": len(urls_to_try),
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
                                self._record_url_success(base_url)

                                # Update service URL to successful one
                                if base_url != self.service_url:
                                    logger.info(
                                        "Updating primary service URL to successful fallback",
                                        extra={
                                            "new_url": base_url,
                                            "old_url": self.service_url,
                                            "client_id": self.client_id,
                                        },
                                    )
                                    self.service_url = base_url

                                logger.debug(
                                    "Request successful",
                                    extra={
                                        "client_id": self.client_id,
                                        "operation": operation,
                                        "status_code": resp.status,
                                        "response_size": len(str(data)),
                                        "session_requests": self._session_request_count,
                                        "service_url": base_url,
                                    },
                                )
                                return data

                            # Handle non-200 responses (keeping original logic)
                            if resp.status == 401:
                                error_msg = "Authentication failed - check BLUEFIN_SERVICE_API_KEY"
                                logger.error(
                                    "Authentication failed",
                                    extra={
                                        "client_id": self.client_id,
                                        "operation": operation,
                                        "status_code": resp.status,
                                        "endpoint": endpoint,
                                        "service_url": base_url,
                                    },
                                )
                                raise BluefinServiceAuthError(error_msg)

                            if resp.status == 429:
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
                                        "service_url": base_url,
                                    },
                                )

                                # For rate limiting, wait and retry if we have attempts left
                                if attempt < self._max_retries - 1:
                                    wait_time = min(
                                        retry_after, 30
                                    )  # Cap wait time at 30s
                                    logger.info(
                                        "Waiting %ss for rate limit reset",
                                        wait_time,
                                        extra={
                                            "client_id": self.client_id,
                                            "operation": operation,
                                        },
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                                raise BluefinServiceRateLimitError(
                                    error_msg, retry_after=retry_after
                                )

                            # Other error responses
                            error_text = await resp.text()
                            self._record_failure()
                            self._record_url_failure(base_url)

                            if attempt < self._max_retries - 1:
                                delay = self._calculate_retry_delay(attempt)
                                logger.warning(
                                    "Request failed, retrying in %.2fs",
                                    delay,
                                    extra={
                                        "client_id": self.client_id,
                                        "operation": operation,
                                        "status_code": resp.status,
                                        "error_response": error_text[:200],
                                        "attempt": attempt + 1,
                                        "max_retries": self._max_retries,
                                        "delay": delay,
                                        "service_url": base_url,
                                    },
                                )
                                await asyncio.sleep(delay)
                                continue

                            # All retries exhausted for this URL
                            last_error = BluefinServiceConnectionError(
                                f"Request failed: HTTP {resp.status}",
                                status_code=resp.status,
                                service_url=base_url,
                            )
                            break  # Try next URL

                except (
                    BluefinServiceAuthError,
                    BluefinServiceRateLimitError,
                ):
                    # Don't retry auth or specific service errors across URLs
                    raise

                except (ClientError, TimeoutError, OSError) as e:
                    self._record_failure()
                    self._record_url_failure(base_url)

                    if attempt < self._max_retries - 1:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(
                            "Network error, retrying in %.2fs",
                            delay,
                            extra={
                                "client_id": self.client_id,
                                "operation": operation,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "attempt": attempt + 1,
                                "max_retries": self._max_retries,
                                "delay": delay,
                                "service_url": base_url,
                            },
                        )
                        await asyncio.sleep(delay)
                        continue

                    # All retries exhausted for this URL
                    last_error = BluefinServiceConnectionError(
                        f"Network error: {e!s}", service_url=base_url
                    )
                    break  # Try next URL

                except Exception as e:
                    self._record_failure()
                    self._record_url_failure(base_url)
                    last_error = BluefinServiceConnectionError(
                        f"Unexpected error in {operation}: {e!s}", service_url=base_url
                    )
                    exception_handler.log_exception_with_context(
                        e,
                        {
                            "client_id": self.client_id,
                            "operation": operation,
                            "attempt": attempt + 1,
                            "unexpected_request_error": True,
                            "method": method,
                            "endpoint": endpoint,
                            "service_url": base_url,
                        },
                        component="BluefinServiceClient",
                        operation="_make_request_with_retry",
                    )
                    break  # Try next URL

        # All URLs failed
        logger.error(
            "All service URLs failed",
            extra={
                "client_id": self.client_id,
                "operation": operation,
                "urls_tried": list(urls_tried),
                "total_urls": len(self.service_urls),
                "last_error": str(last_error) if last_error else "Unknown",
            },
        )

        # Force rediscovery on next request
        self.service_discovery_complete = False

        if last_error:
            raise last_error
        raise BluefinServiceConnectionError(
            "Request failed - all URLs exhausted", service_url="All URLs failed"
        )

    # This section has been replaced by the new implementation above

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
            if self._is_valid_candle(candle, i):
                valid_count += 1

        return self._evaluate_validation_results(valid_count, total_candles, candles)

    def _is_valid_candle(self, candle: Any, index: int) -> bool:
        """Check if a single candle is valid.

        Args:
            candle: Single candle data
            index: Candle index for logging

        Returns:
            True if candle is valid
        """
        if not isinstance(candle, list) or len(candle) < 6:
            logger.debug("⚠️ Invalid candle format at index %s: %s", index, candle)
            return False

        try:
            timestamp, open_p, high_p, low_p, close_p, volume = candle[0:6]

            # Convert and validate data types
            converted_values = self._convert_candle_values(
                timestamp, open_p, high_p, low_p, close_p, volume
            )
            if not converted_values:
                logger.debug("⚠️ Type conversion failed for candle %s", index)
                return False

            timestamp, open_p, high_p, low_p, close_p, volume = converted_values

            # Run validation checks
            return self._validate_candle_values(
                timestamp, open_p, high_p, low_p, close_p, volume, index
            )

        except (ValueError, IndexError, TypeError) as e:
            logger.debug("⚠️ Error validating candle %s: %s", index, e)
            return False

    def _convert_candle_values(
        self,
        timestamp: Any,
        open_p: Any,
        high_p: Any,
        low_p: Any,
        close_p: Any,
        volume: Any,
    ) -> tuple[float, float, float, float, float, float] | None:
        """Convert candle values to proper types.

        Returns:
            Tuple of converted values or None if conversion fails
        """
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

        except (ValueError, TypeError):
            return None
        else:
            return timestamp, open_p, high_p, low_p, close_p, volume

    def _validate_candle_values(
        self,
        timestamp: float,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        volume: float,
        index: int,
    ) -> bool:
        """Validate converted candle values.

        Args:
            timestamp, open_p, high_p, low_p, close_p, volume: Converted candle values
            index: Candle index for logging

        Returns:
            True if all values are valid
        """
        # Basic sanity checks
        if timestamp <= 0:
            logger.debug("⚠️ Invalid timestamp in candle %s: %s", index, timestamp)
            return False

        # Validate price values are positive (but allow 0 volume)
        if not all(x > 0 for x in [open_p, high_p, low_p, close_p]):
            logger.debug(
                "⚠️ Non-positive prices in candle %s: O=%s, H=%s, L=%s, C=%s",
                index,
                open_p,
                high_p,
                low_p,
                close_p,
            )
            return False

        # Volume can be 0 but not negative
        if volume < 0:
            logger.debug("⚠️ Negative volume in candle %s: %s", index, volume)
            return False

        # Apply OHLC fixes and validate relationships
        return self._validate_ohlc_relationships(open_p, high_p, low_p, close_p, index)

    def _validate_ohlc_relationships(
        self, open_p: float, high_p: float, low_p: float, close_p: float, index: int
    ) -> bool:
        """Validate and fix OHLC relationships.

        Args:
            open_p, high_p, low_p, close_p: Price values
            index: Candle index for logging

        Returns:
            True if relationships are valid after fixes
        """
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
                "🔧 Fixing high price: %s -> %s (candle %s)", high_p, max_oc, index
            )
            high_p = max_oc

        # Fix low if it's too high
        if low_p > min_oc:
            logger.debug(
                "🔧 Fixing low price: %s -> %s (candle %s)", low_p, min_oc, index
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
                "⚠️ Invalid OHLC relationship in candle %s: O=%s, H=%s, L=%s, C=%s, tolerance=%s",
                index,
                open_p,
                high_p,
                low_p,
                close_p,
                tolerance,
            )
            return False

        if max_price - min_price > avg_price * 0.5:
            logger.debug(
                "⚠️ Suspicious price spread in candle %s: range=%.6f, avg=%.6f",
                index,
                max_price - min_price,
                avg_price,
            )
            return False

        return True

    def _evaluate_validation_results(
        self, valid_count: int, total_candles: int, candles: list[Any]
    ) -> bool:
        """Evaluate validation results and decide if data is acceptable.

        Args:
            valid_count: Number of valid candles
            total_candles: Total number of candles
            candles: Original candle data for failure analysis

        Returns:
            True if validation rate is acceptable
        """
        validation_rate = valid_count / total_candles if total_candles else 0

        # Lower the threshold to 40% since we've identified data quality issues from the source
        if validation_rate < 0.4:
            logger.warning(
                "⚠️ Low validation rate: %.1f%% (%s/%s valid)",
                validation_rate * 100,
                valid_count,
                total_candles,
            )

            # Analyze and log failure patterns
            self._analyze_validation_failures(candles)
            return False

        logger.info(
            "✅ Candle validation passed: %.1f%% (%s/%s valid)",
            validation_rate * 100,
            valid_count,
            total_candles,
        )
        return True

    def _analyze_validation_failures(self, candles: list[Any]) -> None:
        """Analyze and log validation failure patterns.

        Args:
            candles: List of candle data to analyze
        """
        failure_reasons = {
            "format": 0,
            "type_conversion": 0,
            "timestamp": 0,
            "negative_prices": 0,
            "negative_volume": 0,
            "ohlc_relationship": 0,
            "price_spread": 0,
        }

        # Analyze first 20 candles for pattern analysis
        for _, candle in enumerate(candles[:20]):
            failure_type = self._classify_candle_failure(candle)
            if failure_type:
                failure_reasons[failure_type] += 1

        logger.warning("📊 Validation failure analysis: %s", failure_reasons)
        self._log_sample_candles(candles[:10])

    def _classify_candle_failure(self, candle: Any) -> str | None:
        """Classify the type of failure for a candle.

        Args:
            candle: Candle data to classify

        Returns:
            Failure type string or None if no failure
        """
        # Early format validation
        if not isinstance(candle, list) or len(candle) < 6:
            return "format"

        try:
            timestamp, open_p, high_p, low_p, close_p, volume = candle[0:6]

            # Check type conversion
            converted_values = self._convert_candle_values(
                timestamp, open_p, high_p, low_p, close_p, volume
            )
            if not converted_values:
                return "type_conversion"

            timestamp, open_p, high_p, low_p, close_p, volume = converted_values

            # Perform all validations and collect failure type
            failure_type = None

            # Check timestamp
            if timestamp <= 0:
                failure_type = "timestamp"
            # Check negative prices
            elif not all(x > 0 for x in [open_p, high_p, low_p, close_p]):
                failure_type = "negative_prices"
            # Check volume
            elif volume < 0:
                failure_type = "negative_volume"
            else:
                # Check OHLC relationships
                prices_temp = [open_p, high_p, low_p, close_p]
                avg_price_temp = sum(prices_temp) / 4
                tolerance = max(avg_price_temp * 1e-6, 1e-8)
                if not (
                    high_p >= max(open_p, close_p) - tolerance
                    and low_p <= min(open_p, close_p) + tolerance
                ):
                    failure_type = "ohlc_relationship"
                else:
                    # Check price spread
                    max_price = max(prices_temp)
                    min_price = min(prices_temp)
                    if max_price - min_price > avg_price_temp * 0.5:
                        failure_type = "price_spread"

        except Exception:
            return "format"
        else:
            return failure_type

    def _log_sample_candles(self, sample_candles: list[Any]) -> None:
        """Log sample candles with OHLC validation details.

        Args:
            sample_candles: List of sample candles to log
        """
        sample_count = 0
        for i, candle in enumerate(sample_candles):
            if sample_count >= 5:
                break
            try:
                if isinstance(candle, list) and len(candle) >= 6:
                    timestamp, open_p, high_p, low_p, close_p, volume = candle[0:6]
                    # Convert values
                    converted_values = self._convert_candle_values(
                        timestamp, open_p, high_p, low_p, close_p, volume
                    )
                    if not converted_values:
                        continue

                    _, open_p, high_p, low_p, close_p, volume = converted_values

                    # Check OHLC relationship
                    max_oc = max(open_p, close_p)
                    min_oc = min(open_p, close_p)
                    high_ok = high_p >= max_oc
                    low_ok = low_p <= min_oc

                    logger.warning(
                        "Sample candle %s: O=%.8f, H=%.8f, L=%.8f, C=%.8f, V=%.4f",
                        i,
                        open_p,
                        high_p,
                        low_p,
                        close_p,
                        volume,
                    )
                    logger.warning(
                        "  OHLC check: H>=%.8f? %s (%.8f >= %.8f), L<=%.8f? %s (%.8f <= %.8f)",
                        max_oc,
                        high_ok,
                        high_p,
                        max_oc,
                        min_oc,
                        low_ok,
                        low_p,
                        min_oc,
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

    # Comprehensive Connectivity Monitoring Methods

    def _record_request_metrics(
        self, success: bool, response_time: float = 0.0, error_type: str | None = None
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
        self, success: bool, response_time: float = 0.0, status_code: int | None = None
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

    async def _run_basic_connectivity_test(
        self, _test_start_time: float
    ) -> dict[str, Any]:
        """Run basic connectivity test."""
        import time

        start_time = time.time()
        try:
            await self._ensure_session()

            if self._session and not self._session.closed:
                async with self._session.get(
                    f"{self.service_url}/health", timeout=self.quick_timeout
                ) as resp:
                    response_time = time.time() - start_time
                    return {
                        "status": "pass" if resp.status == 200 else "fail",
                        "response_time_ms": response_time * 1000,
                        "status_code": resp.status,
                        "details": "Basic HTTP connectivity test",
                    }
            else:
                return {
                    "status": "fail",
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "error": "No session available",
                    "details": "Could not establish session",
                }

        except Exception as e:
            return {
                "status": "fail",
                "response_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Exception during basic connectivity test",
            }

    async def _run_detailed_health_test(self) -> dict[str, Any]:
        """Run detailed health endpoint test."""
        import time

        start_time = time.time()
        try:
            if self._session and not self._session.closed:
                async with self._session.get(
                    f"{self.service_url}/health/detailed", timeout=self.normal_timeout
                ) as resp:
                    response_time = time.time() - start_time

                    if resp.status == 200:
                        health_data = await resp.json()
                        return {
                            "status": "pass",
                            "response_time_ms": response_time * 1000,
                            "status_code": resp.status,
                            "health_data": health_data,
                            "details": "Detailed health endpoint test",
                        }
                    return {
                        "status": "degraded",
                        "response_time_ms": response_time * 1000,
                        "status_code": resp.status,
                        "details": "Detailed health endpoint returned non-200 status",
                    }
            else:
                return {
                    "status": "fail",
                    "error": "No session available",
                    "details": "Could not test detailed health endpoint",
                }

        except Exception as e:
            return {
                "status": "fail",
                "response_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Exception during detailed health test",
            }

    async def _run_authentication_test(self) -> tuple[dict[str, Any], list[str]]:
        """Run authentication test. Returns (test_result, recommendations)."""
        import time

        start_time = time.time()
        recommendations = []

        try:
            if self._session and not self._session.closed:
                async with self._session.get(
                    f"{self.service_url}/account", timeout=self.normal_timeout
                ) as resp:
                    response_time = time.time() - start_time

                    if resp.status == 200:
                        return {
                            "status": "pass",
                            "response_time_ms": response_time * 1000,
                            "status_code": resp.status,
                            "details": "Authentication working correctly",
                        }, recommendations
                    if resp.status == 401:
                        recommendations.append(
                            "Check BLUEFIN_SERVICE_API_KEY environment variable"
                        )
                        return {
                            "status": "fail",
                            "response_time_ms": response_time * 1000,
                            "status_code": resp.status,
                            "details": "Authentication failed - check API key",
                        }, recommendations
                    return {
                        "status": "degraded",
                        "response_time_ms": response_time * 1000,
                        "status_code": resp.status,
                        "details": f"Unexpected status code: {resp.status}",
                    }, recommendations
            else:
                return {
                    "status": "fail",
                    "error": "No session available",
                    "details": "Could not test authentication",
                }, recommendations

        except Exception as e:
            return {
                "status": "fail",
                "response_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Exception during authentication test",
            }, recommendations

    async def _run_performance_test(self) -> tuple[dict[str, Any], list[str]]:
        """Run performance test with multiple requests. Returns (test_result, recommendations)."""
        import time

        start_time = time.time()
        recommendations = []

        try:
            if self._session and not self._session.closed:
                response_times = []
                success_count = 0

                for _i in range(5):  # 5 quick requests
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

                # Determine status based on success rate
                if success_rate >= 80:
                    status = "pass"
                elif success_rate >= 60:
                    status = "degraded"
                else:
                    status = "fail"

                # Add recommendations based on performance
                if avg_response_time > 2.0:
                    recommendations.append(
                        "High response times detected - check service performance"
                    )
                if success_rate < 100:
                    recommendations.append(
                        "Some requests failed - check service stability"
                    )

                return {
                    "status": status,
                    "response_time_ms": avg_response_time * 1000,
                    "success_rate": success_rate,
                    "total_requests": 5,
                    "successful_requests": success_count,
                    "details": f"Performance test: {success_rate:.1f}% success rate",
                }, recommendations
            return {
                "status": "fail",
                "error": "No session available",
                "details": "Could not run performance test",
            }, recommendations

        except Exception as e:
            return {
                "status": "fail",
                "response_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "details": "Exception during performance test",
            }, recommendations

    def _determine_overall_status(self, test_statuses: list[str]) -> str:
        """Determine overall status based on individual test results."""
        if all(status == "pass" for status in test_statuses):
            return "healthy"
        if any(status == "pass" for status in test_statuses):
            return "degraded"
        return "unhealthy"

    def _add_general_recommendations(self, test_results: dict[str, Any]) -> None:
        """Add general recommendations if none exist."""
        if not test_results["recommendations"]:
            if test_results["overall_status"] == "healthy":
                test_results["recommendations"] = [
                    "All connectivity tests passed - service is operating normally"
                ]
            else:
                test_results["recommendations"].append(
                    "Review failed tests and check service configuration"
                )

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

        # Run all connectivity tests
        test_results["tests"]["basic_connectivity"] = (
            await self._run_basic_connectivity_test(test_start)
        )
        test_results["tests"][
            "detailed_health"
        ] = await self._run_detailed_health_test()

        auth_result, auth_recommendations = await self._run_authentication_test()
        test_results["tests"]["authentication"] = auth_result
        test_results["recommendations"].extend(auth_recommendations)

        perf_result, perf_recommendations = await self._run_performance_test()
        test_results["tests"]["performance"] = perf_result
        test_results["recommendations"].extend(perf_recommendations)

        # Determine overall status
        test_statuses = [
            test.get("status", "fail") for test in test_results["tests"].values()
        ]
        test_results["overall_status"] = self._determine_overall_status(test_statuses)

        # Add general recommendations
        self._add_general_recommendations(test_results)

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
            if (
                hasattr(self, "_service_client")
                and self._service_client
                and hasattr(self._service_client, "get_circuit_breaker_status")
            ):
                service_health = self._service_client.get_circuit_breaker_status()
        except Exception as e:
            logger.debug(
                "Could not get service health status: %s",
                e,
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

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit."""
        await self.disconnect()
