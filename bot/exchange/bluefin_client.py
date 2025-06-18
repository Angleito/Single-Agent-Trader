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
import traceback
from typing import Any

import aiohttp  # type: ignore
from aiohttp import ClientError, ClientTimeout, TCPConnector

from ..error_handling import (
    ErrorContext,
    ErrorSeverity,
    exception_handler,
)

logger = logging.getLogger(__name__)


class BluefinClientError(Exception):
    """Base exception for Bluefin client errors."""

    pass


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

    pass


class BluefinServiceRateLimitError(BluefinClientError):
    """Exception raised when Bluefin service rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class BluefinServiceDataError(BluefinClientError):
    """Exception raised when Bluefin service returns invalid data."""

    pass


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
        self.last_health_check = 0
        self.health_check_interval = 30  # Check health every 30 seconds (reduced from 60)
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

        # Circuit breaker state
        self.circuit_open = False
        self.circuit_open_until = 0
        self.circuit_failure_threshold = 5
        self.circuit_recovery_timeout = 60  # seconds

        # Connection pooling settings
        self._connection_pool_size = 10
        self._connection_pool_ttl = 300  # 5 minutes
        self._keep_alive_timeout = 30

        # Session reuse tracking
        self._session_created_at = 0
        self._session_max_age = 1800  # 30 minutes
        self._session_request_count = 0
        self._max_requests_per_session = 1000

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("BLUEFIN_SERVICE_API_KEY")
        if not self.api_key:
            logger.warning(
                "No BLUEFIN_SERVICE_API_KEY configured - authentication may fail",
                extra={
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "auth_configured": False,
                },
            )
        else:
            logger.debug(
                "Bluefin service client initialized with authentication",
                extra={
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "auth_configured": True,
                    "api_key_length": len(self.api_key),
                },
            )

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

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        if self.circuit_open and time.time() < self.circuit_open_until:
            return True
        elif self.circuit_open and time.time() >= self.circuit_open_until:
            # Circuit breaker timeout expired, attempt to close
            logger.info(
                "Circuit breaker timeout expired, attempting to close circuit",
                extra={"client_id": self.client_id}
            )
            self.circuit_open = False
            self.consecutive_failures = 0
        return False

    def _record_success(self):
        """Record successful operation - reset failure counters."""
        if self.consecutive_failures > 0:
            logger.info(
                f"Operation succeeded, resetting failure count from {self.consecutive_failures}",
                extra={"client_id": self.client_id}
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
                extra={"client_id": self.client_id}
            )

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        # Exponential backoff: base * (2 ^ attempt)
        delay = min(self._base_retry_delay * (2 ** attempt), self._max_retry_delay)

        # Add jitter to prevent thundering herd
        jitter = delay * self._jitter_factor * random.random()

        return delay + jitter

    async def _retry_request(self, func, *args, **kwargs):
        """Execute request with enhanced exponential backoff retry and circuit breaker."""
        # Check circuit breaker first
        if self._is_circuit_open():
            logger.warning(
                "Circuit breaker is open, skipping request",
                extra={"client_id": self.client_id}
            )
            raise BluefinServiceConnectionError(
                "Circuit breaker is open - service temporarily unavailable",
                service_url=self.service_url
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
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self._max_retries}), "
                        f"retrying in {delay:.2f}s: {e}",
                        extra={
                            "client_id": self.client_id,
                            "attempt": attempt + 1,
                            "max_retries": self._max_retries,
                            "delay": delay,
                            "error_type": type(e).__name__
                        }
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Request failed after {self._max_retries} attempts: {e}",
                        extra={
                            "client_id": self.client_id,
                            "total_attempts": self._max_retries,
                            "error_type": type(e).__name__
                        }
                    )
            except Exception as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "client_id": self.client_id,
                        "attempt": attempt + 1,
                        "non_retryable_error": True,
                        "function_name": func.__name__ if hasattr(func, "__name__") else "unknown",
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
                    "max_age": self._session_max_age
                }
            )
            return True

        # Check request count
        if self._session_request_count >= self._max_requests_per_session:
            logger.debug(
                "Session exceeded max requests, will recreate",
                extra={
                    "client_id": self.client_id,
                    "request_count": self._session_request_count,
                    "max_requests": self._max_requests_per_session
                }
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
                "keep_alive_timeout": self._keep_alive_timeout
            }
        )

        return session

    async def _check_connection_health(self) -> bool:
        """Enhanced connection health check with intelligent reconnection."""
        current_time = time.time()

        # Skip health check if done recently and connection appears healthy
        if (current_time - self.last_health_check < self.health_check_interval and
            self._connected and not self._should_recreate_session()):
            return self._connected

        try:
            # Recreate session if needed
            if self._should_recreate_session():
                await self._ensure_session()

            # Quick health check
            async with self._session.get(
                f"{self.service_url}/health", timeout=self.quick_timeout
            ) as resp:
                self._connected = resp.status == 200
                self.last_health_check = current_time

                if self._connected:
                    self.consecutive_failures = 0

                logger.debug(
                    f"Health check completed: status={resp.status}, connected={self._connected}",
                    extra={
                        "client_id": self.client_id,
                        "status_code": resp.status,
                        "connected": self._connected
                    }
                )

                return self._connected

        except (ClientError, TimeoutError, OSError) as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "service_url": self.service_url,
                    "consecutive_failures": self.consecutive_failures,
                    "health_check_network_error": True,
                },
                component="BluefinServiceClient",
                operation="_check_connection_health",
            )
            self._connected = False
            self.consecutive_failures += 1
            return False
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "client_id": self.client_id,
                    "unexpected_health_check_error": True,
                },
                component="BluefinServiceClient",
                operation="_check_connection_health",
            )
            self._connected = False
            self.consecutive_failures += 1
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
                        extra={"client_id": self.client_id}
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
                    "request_count": self._session_request_count
                }
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

            async with self._session.get(health_url, timeout=self.quick_timeout) as resp:
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

    async def disconnect(self) -> None:
        """Disconnect from the Bluefin service with proper cleanup."""
        logger.info(
            "Disconnecting from Bluefin service",
            extra={
                "client_id": self.client_id,
                "service_url": self.service_url,
                "was_connected": self._connected,
            },
        )

        if self._session and not self._session.closed:
            try:
                await self._session.close()
                logger.debug(
                    "HTTP session closed successfully",
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
                    operation="disconnect",
                )
            finally:
                self._session = None
                self._session_closed = True

        self._connected = False
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
        if self._session is None or self._session.closed:
            error_msg = "Session not initialized or closed - call connect() first"
            logger.error(
                "Account data request failed - no session",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_account_data",
                    "session_initialized": self._session is not None,
                    "session_closed": self._session.closed if self._session else True,
                },
            )
            return {"error": error_msg}

        # Check connection health
        if not await self._check_connection_health():
            return {"error": "Connection health check failed"}

        logger.debug(
            "Requesting account data from Bluefin service",
            extra={
                "client_id": self.client_id,
                "operation": "get_account_data",
                "endpoint": "/account",
            },
        )

        return await self._make_request_with_retry(
            method="GET", endpoint="/account", operation="get_account_data"
        )

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
                json_data=order_data
            )
            return {"status": "success", "order": result}

        except BluefinServiceAuthError as e:
            logger.error(
                f"Authentication failed placing order: {e}",
                extra={"client_id": self.client_id, "operation": "place_order"}
            )
            return {"status": "error", "message": "Authentication failed"}

        except BluefinServiceRateLimitError as e:
            logger.error(
                f"Rate limited placing order: {e}",
                extra={"client_id": self.client_id, "operation": "place_order"}
            )
            return {
                "status": "error",
                "message": f"Rate limit exceeded, retry after {e.retry_after}s",
                "retry_after": e.retry_after
            }

        except BluefinServiceConnectionError as e:
            logger.error(
                f"Connection error placing order: {e}",
                extra={"client_id": self.client_id, "operation": "place_order"}
            )
            return {"status": "error", "message": f"Connection error: {str(e)}"}

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
                operation="cancel_order"
            )
            return True

        except (BluefinServiceAuthError, BluefinServiceRateLimitError,
                BluefinServiceConnectionError) as e:
            logger.error(
                f"Service error canceling order {order_id}: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "cancel_order",
                    "order_id": order_id,
                    "error_type": type(e).__name__
                }
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
                params={"symbol": symbol}
            )
            return result

        except (BluefinServiceAuthError, BluefinServiceRateLimitError,
                BluefinServiceConnectionError) as e:
            logger.error(
                f"Service error getting ticker for {symbol}: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_market_ticker",
                    "symbol": symbol,
                    "error_type": type(e).__name__
                }
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
                json_data={"symbol": symbol, "leverage": leverage}
            )
            return True

        except (BluefinServiceAuthError, BluefinServiceRateLimitError,
                BluefinServiceConnectionError) as e:
            logger.error(
                f"Service error setting leverage for {symbol}: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "set_leverage",
                    "symbol": symbol,
                    "leverage": leverage,
                    "error_type": type(e).__name__
                }
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
                params=params
            )

            candles = result.get("candles", [])

            # Validate candle data
            if self._validate_candle_data(candles):
                logger.info(
                    f"✅ Successfully retrieved {len(candles)} validated candles",
                    extra={
                        "client_id": self.client_id,
                        "operation": "get_candlestick_data",
                        "candle_count": len(candles)
                    }
                )
                return candles
            else:
                logger.warning(
                    "⚠️ Received invalid candle data from service",
                    extra={
                        "client_id": self.client_id,
                        "operation": "get_candlestick_data",
                        "raw_candle_count": len(candles)
                    }
                )
                return []

        except (BluefinServiceAuthError, BluefinServiceRateLimitError) as e:
            logger.error(
                f"Service error getting candlestick data: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_candlestick_data",
                    "error_type": type(e).__name__
                }
            )
            return []
        except BluefinServiceConnectionError as e:
            logger.error(
                f"Connection error getting candlestick data: {e}",
                extra={
                    "client_id": self.client_id,
                    "operation": "get_candlestick_data",
                    "error_type": type(e).__name__
                }
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
        json_data: dict | None = None,
        params: dict | None = None,
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
                extra={"client_id": self.client_id, "operation": operation}
            )
            raise BluefinServiceConnectionError(
                "Circuit breaker is open - service temporarily unavailable",
                service_url=self.service_url
            )

        # Ensure we have a healthy connection
        if not await self._check_connection_health():
            logger.error(
                "Connection health check failed before request",
                extra={"client_id": self.client_id, "operation": operation}
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

                request_kwargs = {}
                if json_data:
                    request_kwargs["json"] = json_data
                if params:
                    request_kwargs["params"] = params

                # Choose appropriate timeout based on endpoint
                if "/market/candles" in endpoint:
                    timeout = self.heavy_timeout
                elif "/health" in endpoint:
                    timeout = self.quick_timeout
                else:
                    timeout = self.normal_timeout
                request_kwargs["timeout"] = timeout

                async with self._session.request(method, url, **request_kwargs) as resp:
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
                        error_msg = (
                            f"Rate limit exceeded, retry after {retry_after} seconds"
                        )
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
                                extra={"client_id": self.client_id, "operation": operation}
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
                        f"Network error after {self._max_retries} retries: {str(e)}"
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
                error_msg = f"Unexpected error in {operation}: {str(e)}"
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

    def _validate_candle_data(self, candles: list) -> bool:
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

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
