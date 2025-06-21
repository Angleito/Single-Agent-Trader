#!/usr/bin/env python3
"""
Bluefin SDK Service - REST API wrapper for the Bluefin SDK.

This service runs in an isolated Docker container with the Bluefin SDK installed,
providing a REST API for the main bot to interact with Bluefin DEX.

IMPORTANT INTERVAL LIMITATIONS:
- Bluefin DEX only supports specific intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 1w, 1M
- Sub-minute intervals (15s, 30s, etc.) are NOT supported and will be converted to 1m with warnings
- This conversion results in granularity loss - 15s data becomes 1m aggregated data
- All interval conversions are logged with warnings to alert users of data granularity changes
"""

import asyncio
import logging
import os
import secrets
import sys
import time
import traceback
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from aiohttp import ClientError, ClientTimeout, web
from bluefin_v2_client import MARKET_SYMBOLS, BluefinClient, Networks
from bluefin_v2_client.interfaces import GetOrderbookRequest

# HTTP status codes
HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500

# Configuration constants
MIN_KEY_LENGTH = 64  # Minimum hex key length
MIN_MNEMONIC_WORDS = 12  # BIP39 minimum words
MAX_MNEMONIC_WORDS = 24  # BIP39 maximum words
MIN_WORD_LENGTH = 2  # Minimum mnemonic word length
DEFAULT_API_LIMIT = 50  # Default API request limit
DEFAULT_RETRY_LIMIT = 10  # Default retry attempts
DEFAULT_TIMEOUT_SECONDS = 10  # Default timeout
ORDERBOOK_PRECISION_LIMIT = 6  # Decimal precision for orderbook
VALIDATION_TIMEOUT = 20  # Validation timeout
LARGE_DECIMAL_LIMIT = 80  # Large decimal validation
LARGE_PERCENTAGE_LIMIT = 80  # Large percentage validation
DEFAULT_PRICE_PRECISION = 5  # Default price precision
DECIMAL_PRECISION_BASE = 64  # Base decimal precision
API_LIMIT_MAX = 1000  # Maximum API request limit
SCALE_MULTIPLIER_BASE = 10  # Base for scale calculations
ADDRESS_DISPLAY_MIN_LENGTH = 12  # Minimum address length for truncation

# Add parent directory to path to import secure logging and utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ErrorType(Enum):
    """Classification of different error types for retry logic."""

    TEMPORARY = "temporary"  # Retryable errors (network, timeout, rate limit)
    PERMANENT = "permanent"  # Non-retryable errors (auth, invalid params)
    UNKNOWN = "unknown"  # Unclassified errors


try:
    from bot.exchange.bluefin_endpoints import BluefinEndpointConfig, get_rest_api_url
    from bot.utils.secure_logging import create_secure_logger
    from bot.utils.symbol_utils import (
        BluefinSymbolConverter,
        InvalidSymbolError,
        SymbolConversionError,
        get_testnet_symbol_fallback,
        is_bluefin_symbol_supported,
    )

    logger = create_secure_logger(__name__, level=logging.INFO)
except ImportError:
    # Fallback to standard logging if secure logging not available
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Fallback endpoint configuration if centralized config not available
    def get_rest_api_url(network: str = "mainnet") -> str:
        """Fallback endpoint resolver when centralized config is not available."""
        if network.lower() == "testnet":
            return "https://dapi.api.sui-staging.bluefin.io"
        return "https://dapi.api.sui-prod.bluefin.io"

    # Create fallback symbol utilities
    class BluefinSymbolConverter:
        def __init__(self, market_symbols_enum=None):
            self.market_symbols_enum = market_symbols_enum

        def to_market_symbol(self, symbol: str):
            if not self.market_symbols_enum:
                raise BluefinMarketSymbolError("MARKET_SYMBOLS enum not available")
            base_symbol = symbol.split("-")[0].upper()
            if hasattr(self.market_symbols_enum, base_symbol):
                return getattr(self.market_symbols_enum, base_symbol)
            raise BluefinSymbolError(f"Unknown symbol: {symbol}")

        def validate_market_symbol(self, symbol: str) -> bool:
            try:
                self.to_market_symbol(symbol)
            except Exception:
                return False
            else:
                return True

        def from_market_symbol(self, market_symbol) -> str:
            # Convert market symbol back to string format
            if isinstance(market_symbol, str):
                return market_symbol
            return str(market_symbol)

    class SymbolConversionError(Exception):
        pass

    class InvalidSymbolError(Exception):
        pass

    # Fallback BluefinEndpointConfig class
    class BluefinEndpointConfig:
        @staticmethod
        def get_endpoints(network: str = "mainnet"):
            class Endpoints:
                def __init__(self, network: str):
                    if network.lower() == "testnet":
                        self.rest_api = "https://dapi.api.sui-staging.bluefin.io"
                        self.websocket_api = "wss://dapi.api.sui-staging.bluefin.io"
                        self.websocket_notifications = (
                            "wss://dapi.api.sui-staging.bluefin.io"
                        )
                    else:
                        self.rest_api = "https://dapi.api.sui-prod.bluefin.io"
                        self.websocket_api = "wss://dapi.api.sui-prod.bluefin.io"
                        self.websocket_notifications = (
                            "wss://dapi.api.sui-prod.bluefin.io"
                        )

            return Endpoints(network)


class BluefinServiceError(Exception):
    """Base exception for Bluefin service errors."""


class BluefinConnectionError(BluefinServiceError):
    """Exception raised when connection to Bluefin fails."""


class BluefinAPIError(BluefinServiceError):
    """Exception raised when Bluefin API returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: dict | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class BluefinDataError(BluefinServiceError):
    """Exception raised when Bluefin data is invalid or missing."""


class BluefinSymbolError(BluefinServiceError):
    """Exception raised when symbol conversion or validation fails."""


class BluefinMarketSymbolError(BluefinServiceError):
    """Exception raised when market symbols enum is not available."""


class BluefinDataFetchError(BluefinServiceError):
    """Exception raised when unable to fetch market data from any source."""


class BluefinSDKNotInitializedError(BluefinServiceError):
    """Exception raised when SDK is not properly initialized."""


class BluefinSDKService:
    """
    REST API service wrapping the Bluefin SDK.

    Provides comprehensive error handling and structured logging for all
    Bluefin DEX operations including account management, trading, and market data.
    """

    def __init__(self):
        self.client: BluefinClient | None = None
        self.private_key = os.getenv("BLUEFIN_PRIVATE_KEY")
        self.network = os.getenv("BLUEFIN_NETWORK", "mainnet")
        self.initialized = False

        # Request timeout settings
        self.quick_timeout = ClientTimeout(total=5.0)  # For health checks, tickers
        self.normal_timeout = ClientTimeout(total=15.0)  # For standard API calls
        self.heavy_timeout = ClientTimeout(
            total=30.0
        )  # For candle data, large responses

        # Retry configuration
        self.max_retries = 3
        self.base_retry_delay = 1.0

        # Enhanced circuit breaker for API failures
        self.failure_count = 0
        self.circuit_open_until = 0
        self.circuit_failure_threshold = 5
        self.circuit_recovery_timeout = 300  # 5 minutes
        self.circuit_half_open_max_calls = 3  # Max calls in half-open state
        self.circuit_half_open_calls = 0
        self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._cleanup_complete = False

        # Failure type tracking for better circuit breaker decisions
        self.failure_types = {
            "connection": 0,
            "timeout": 0,
            "server_error": 0,
            "rate_limit": 0,
            "auth": 0,
        }
        self.last_failure_reset = time.time()

        # Add service identification for logging context
        self.service_id = f"bluefin-sdk-{int(time.time())}"

        # Service health monitoring
        self.health_stats = {
            "start_time": time.time(),
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "last_request_time": 0.0,
            "last_success_time": 0.0,
            "last_failure_time": 0.0,
            "average_response_time": 0.0,
            "response_times": [],
            "max_response_times": 100,  # Keep last 100 response times for averaging
        }

        # Initialize symbol converter with market symbols
        self.symbol_converter = BluefinSymbolConverter(MARKET_SYMBOLS)

        # Enhanced retry configuration for balance operations
        self.balance_retry_config = {
            "max_retries": 4,
            "base_delay": 1.0,  # Start with 1 second
            "max_delay": 8.0,  # Cap at 8 seconds
            "backoff_factor": 2.0,  # Exponential backoff
            "jitter_factor": 0.1,  # Add randomness
        }

        # Balance staleness tracking
        self.last_successful_balance_fetch = 0
        self.balance_staleness_threshold = 300  # 5 minutes
        self.cached_balance_data = None
        self.balance_cache_timestamp = 0

        # Health check configuration
        self.health_check_interval = 30  # seconds
        self.last_health_check = 0
        self.consecutive_health_failures = 0
        self.max_health_failures = 3

        # Balance operation audit trail
        self.balance_audit_trail = []
        self.max_audit_entries = 1000

        logger.info(
            "Initializing Bluefin SDK Service",
            extra={
                "service_id": self.service_id,
                "network": self.network,
                "has_private_key": bool(self.private_key),
                "private_key_length": len(self.private_key) if self.private_key else 0,
            },
        )

    def _record_balance_audit(
        self,
        correlation_id: str,
        operation: str,
        status: str,
        balance_amount: str | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Record balance operation in audit trail.

        Args:
            correlation_id: Unique identifier for the operation
            operation: Type of balance operation
            status: success/failed/timeout
            balance_amount: Balance amount (if applicable)
            error: Error message (if failed)
            duration_ms: Operation duration in milliseconds
            metadata: Additional context data
        """
        audit_entry = {
            "correlation_id": correlation_id,
            "operation": operation,
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
            "service_id": self.service_id,
            "network": self.network,
            "balance_amount": balance_amount,
            "error": error,
            "duration_ms": duration_ms,
            "metadata": metadata or {},
        }

        # Add to audit trail
        self.balance_audit_trail.append(audit_entry)

        # Keep only recent entries
        if len(self.balance_audit_trail) > self.max_audit_entries:
            self.balance_audit_trail = self.balance_audit_trail[
                -self.max_audit_entries :
            ]

        # Log audit entry
        logger.info(
            "Balance audit: %s - %s",
            operation,
            status,
            extra={"audit_entry": audit_entry, "operation_type": "balance_audit"},
        )

    def get_balance_audit_trail(self, limit: int = 100) -> list[dict]:
        """
        Get recent balance operation audit trail.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent audit entries
        """
        return self.balance_audit_trail[-limit:]

    def _get_error_context(
        self,
        correlation_id: str,
        operation: str,
        error: Exception,
        duration_ms: float | None = None,
        additional_context: dict | None = None,
    ) -> dict:
        """
        Generate comprehensive error context for logging.

        Args:
            correlation_id: Unique operation identifier
            operation: Operation name
            error: The exception that occurred
            duration_ms: Operation duration in milliseconds
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
            "service_id": self.service_id,
            "network": self.network,
            "sdk_version": getattr(self.client, "version", "unknown"),
            "client_initialized": self.initialized,
            "duration_ms": duration_ms,
            "success": False,
            "error_category": self._categorize_error(error),
            "actionable_message": self._get_actionable_error_message(error),
        }

        if additional_context:
            context.update(additional_context)

        return context

    def _categorize_error(self, error: Exception) -> str:
        """
        Categorize error for better troubleshooting.

        Args:
            error: The exception to categorize

        Returns:
            Error category string
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if "'data'" in error_str or "keyerror" in error_type:
            return "api_response_structure"
        if "timeout" in error_str or "timeout" in error_type:
            return "network_timeout"
        if "connection" in error_str or "connection" in error_type:
            return "network_connection"
        if "authentication" in error_str or "auth" in error_str:
            return "authentication"
        if "rate" in error_str and "limit" in error_str:
            return "rate_limiting"
        if "balance" in error_str:
            return "balance_specific"
        if "sdk" in error_str:
            return "sdk_internal"
        return "unknown"

    def _get_actionable_error_message(self, error: Exception) -> str:
        """
        Generate actionable error message for users.

        Args:
            error: The exception to analyze

        Returns:
            Actionable error message
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if "'data'" in error_str or "keyerror" in error_type:
            return (
                "Account may not be initialized on Bluefin. Try depositing funds first."
            )
        if "timeout" in error_str:
            return "Network request timed out. Check internet connection and try again."
        if "connection" in error_str:
            return "Unable to connect to Bluefin network. Check network connectivity."
        if "authentication" in error_str or "auth" in error_str:
            return "Authentication failed. Verify private key configuration."
        if "rate" in error_str and "limit" in error_str:
            return "Rate limit exceeded. Please wait before making another request."
        return "An unexpected error occurred. Check logs for details."

    def _classify_error(self, error: Exception) -> ErrorType:
        """
        Classify an error as temporary, permanent, or unknown for retry logic.

        Args:
            error: The exception to classify

        Returns:
            ErrorType indicating whether the error is retryable
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Temporary errors (retryable)
        if any(
            keyword in error_str
            for keyword in [
                "timeout",
                "connection",
                "network",
                "temporary",
                "rate limit",
                "503",
                "502",
                "504",
                "429",
                "too many requests",
                "unavailable",
            ]
        ):
            return ErrorType.TEMPORARY

        if any(
            keyword in error_type
            for keyword in ["timeout", "connection", "network", "client"]
        ):
            return ErrorType.TEMPORARY

        # Permanent errors (not retryable)
        if any(
            keyword in error_str
            for keyword in [
                "unauthorized",
                "forbidden",
                "invalid",
                "authentication",
                "401",
                "403",
                "404",
                "400",
                "bad request",
            ]
        ):
            return ErrorType.PERMANENT

        # Special handling for Bluefin SDK errors
        if "'data'" in error_str or "keyerror" in error_type:
            # These are typically account not initialized - temporary
            return ErrorType.TEMPORARY

        # Default to unknown
        return ErrorType.UNKNOWN

    def _calculate_retry_delay(self, attempt: int, operation: str = "balance") -> float:
        """
        Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current attempt number (0-based)
            operation: Type of operation for context-aware delays

        Returns:
            Delay in seconds
        """
        config = self.balance_retry_config

        # Base exponential backoff
        delay = config["base_delay"] * (config["backoff_factor"] ** attempt)

        # Cap at maximum delay
        delay = min(delay, config["max_delay"])

        # Add jitter to prevent thundering herd
        jitter = (
            delay * config["jitter_factor"] * (secrets.randbits(32) / (2**32) * 2 - 1)
        )
        final_delay = max(0.1, delay + jitter)  # Minimum 100ms delay

        logger.debug(
            "Calculated retry delay for %s",
            operation,
            extra={
                "service_id": self.service_id,
                "attempt": attempt,
                "operation": operation,
                "base_delay": delay,
                "jitter": jitter,
                "final_delay": final_delay,
            },
        )

        return final_delay

    async def _check_service_health(self) -> bool:
        """
        Check service health before making balance requests.

        Returns:
            True if service is healthy, False otherwise
        """
        current_time = time.time()

        # Skip if checked recently
        if current_time - self.last_health_check < self.health_check_interval:
            return self.consecutive_health_failures < self.max_health_failures

        self.last_health_check = current_time

        try:
            # Simple health check - try to get public address
            if self.client:
                address = self.client.get_public_address()
                if address:
                    self.consecutive_health_failures = 0
                    logger.debug(
                        "Service health check passed",
                        extra={
                            "service_id": self.service_id,
                            "address_length": len(address),
                        },
                    )
                    return True

        except Exception as e:
            self.consecutive_health_failures += 1
            logger.warning(
                "Service health check error",
                extra={
                    "service_id": self.service_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "consecutive_failures": self.consecutive_health_failures,
                },
            )
            return False
        else:
            self.consecutive_health_failures += 1
            logger.warning(
                "Service health check failed",
                extra={
                    "service_id": self.service_id,
                    "consecutive_failures": self.consecutive_health_failures,
                    "max_failures": self.max_health_failures,
                },
            )
            return False

    def _is_balance_stale(self) -> bool:
        """
        Check if cached balance data is stale.

        Returns:
            True if balance data is stale or missing
        """
        current_time = time.time()

        # No cached data
        if not self.cached_balance_data or self.balance_cache_timestamp == 0:
            return True

        # Check staleness
        age = current_time - self.balance_cache_timestamp
        is_stale = age > self.balance_staleness_threshold

        if is_stale:
            logger.warning(
                "Balance data is stale",
                extra={
                    "service_id": self.service_id,
                    "age_seconds": age,
                    "threshold_seconds": self.balance_staleness_threshold,
                },
            )

        return is_stale

    async def _retry_balance_operation(self, operation_func, operation_name: str):
        """
        Execute a balance operation with exponential backoff retry logic.

        Args:
            operation_func: The function to execute
            operation_name: Name of the operation for logging

        Returns:
            The result of the operation

        Raises:
            Exception: If all retries are exhausted
        """
        config = self.balance_retry_config
        last_exception = None

        for attempt in range(config["max_retries"]):
            try:
                # Check circuit breaker
                if await self._is_circuit_open():
                    raise BluefinConnectionError("Circuit breaker is open")

                # Check service health before each attempt
                if not await self._check_service_health():
                    raise BluefinConnectionError("Service health check failed")

                # Execute the operation
                result = await operation_func()

                # Success - record metrics
                self.last_successful_balance_fetch = time.time()
                self._record_api_success()

                logger.debug(
                    "Balance operation %s succeeded",
                    operation_name,
                    extra={
                        "service_id": self.service_id,
                        "operation": operation_name,
                        "attempt": attempt + 1,
                    },
                )

                return result

            except Exception as e:
                last_exception = e
                error_type = self._classify_error(e)

                logger.warning(
                    "Balance operation %s failed",
                    operation_name,
                    extra={
                        "service_id": self.service_id,
                        "operation": operation_name,
                        "attempt": attempt + 1,
                        "error_type": error_type.value,
                        "error_class": type(e).__name__,
                        "error_message": str(e),
                    },
                )

                # Don't retry permanent errors
                if error_type == ErrorType.PERMANENT:
                    logger.exception(
                        "Permanent error in %s, not retrying",
                        operation_name,
                        extra={
                            "service_id": self.service_id,
                            "operation": operation_name,
                            "error_type": error_type.value,
                        },
                    )
                    break

                # If this is the last attempt, don't wait
                if attempt == config["max_retries"] - 1:
                    break

                # Calculate delay and wait
                delay = self._calculate_retry_delay(attempt, operation_name)
                logger.info(
                    "Retrying %s in %.2fs",
                    operation_name,
                    delay,
                    extra={
                        "service_id": self.service_id,
                        "operation": operation_name,
                        "attempt": attempt + 1,
                        "max_retries": config["max_retries"],
                        "delay": delay,
                    },
                )

                await asyncio.sleep(delay)

        # All retries exhausted
        self._record_api_failure("balance_operation")

        if last_exception:
            raise last_exception
        raise BluefinConnectionError(f"All retries exhausted for {operation_name}")

    def _get_market_symbol(self, symbol: str):
        """
        Convert symbol string to MARKET_SYMBOLS attribute using the symbol converter.

        Args:
            symbol: Symbol string like "BTC-PERP", "ETH-PERP", "SUI-PERP"

        Returns:
            The corresponding MARKET_SYMBOLS attribute

        Raises:
            BluefinDataError: When symbol cannot be resolved
        """
        logger.debug(
            "Converting symbol to MARKET_SYMBOLS attribute",
            extra={
                "service_id": self.service_id,
                "input_symbol": symbol,
                "symbol_type": type(symbol).__name__,
            },
        )

        try:
            # Check if symbol is supported on the current network first
            if self.network == "testnet" and not is_bluefin_symbol_supported(
                symbol, "testnet"
            ):
                # Use testnet fallback for unsupported symbols
                fallback_symbol = get_testnet_symbol_fallback(symbol)
                logger.warning(
                    "Symbol %s not available on testnet, using fallback: %s",
                    symbol,
                    fallback_symbol,
                    extra={
                        "service_id": self.service_id,
                        "original_symbol": symbol,
                        "fallback_symbol": fallback_symbol,
                        "network": self.network,
                    },
                )
                symbol = fallback_symbol

            # Use the symbol converter for robust conversion
            market_symbol = self.symbol_converter.to_market_symbol(symbol)

            logger.info(
                "Successfully converted symbol to MARKET_SYMBOLS attribute",
                extra={
                    "service_id": self.service_id,
                    "input_symbol": symbol,
                    "market_symbol": str(market_symbol),
                    "market_symbol_type": type(market_symbol).__name__,
                },
            )
            return market_symbol

        except (SymbolConversionError, InvalidSymbolError) as e:
            # Convert symbol utility exceptions to service exceptions
            error_msg = f"Unable to resolve market symbol '{symbol}': {e!s}"
            logger.exception(
                error_msg,
                extra={
                    "service_id": self.service_id,
                    "input_symbol": symbol,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise BluefinDataError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error processing symbol '{symbol}': {e!s}"
            logger.exception(
                "Unexpected error in symbol conversion",
                extra={
                    "service_id": self.service_id,
                    "input_symbol": symbol,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            raise BluefinDataError(error_msg) from e

    def _symbol_to_string(self, market_symbol) -> str:
        """
        Convert MARKET_SYMBOLS enum to string representation using the symbol converter.

        Args:
            market_symbol: MARKET_SYMBOLS enum value

        Returns:
            String representation of the symbol
        """
        try:
            return self.symbol_converter.from_market_symbol(market_symbol)
        except Exception:
            logger.exception("Failed to convert market symbol to string")
            return str(market_symbol)

    def _validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol can be converted to MARKET_SYMBOLS enum.

        Args:
            symbol: Symbol string to validate

        Returns:
            True if symbol can be converted, False otherwise
        """
        return self.symbol_converter.validate_market_symbol(symbol)

    def _get_market_symbol_value(self, symbol: str):
        """
        Convert symbol string to MARKET_SYMBOLS value (for SDK calls that expect the enum value).

        This method specifically handles the case where the Bluefin SDK expects the actual
        enum value (accessed via .value) rather than the enum object itself.

        FIXES IMPLEMENTED:
        - Comprehensive type checking to handle string, enum, and mixed types
        - Multiple fallback approaches for enum attribute access (.value, .name, str())
        - Safe getattr() usage with explicit exception handling
        - Graceful degradation to string conversion when enum access fails
        - Proper symbol normalization for SDK compatibility

        Args:
            symbol: Symbol string like "BTC-PERP", "ETH-PERP", "SUI-PERP"

        Returns:
            The corresponding symbol value suitable for SDK calls (string)

        Raises:
            BluefinDataError: When symbol cannot be resolved
        """
        logger.debug(
            "Converting symbol to MARKET_SYMBOLS value for SDK call",
            extra={
                "service_id": self.service_id,
                "input_symbol": symbol,
                "method": "_get_market_symbol_value",
            },
        )

        try:
            market_symbol = self._get_market_symbol(symbol)

            # Enhanced type checking with multiple fallback strategies
            # Strategy 1: Already a string - return as-is
            if isinstance(market_symbol, str):
                logger.debug(
                    "Market symbol is already a string value",
                    extra={
                        "service_id": self.service_id,
                        "input_symbol": symbol,
                        "market_symbol": market_symbol,
                        "strategy": "direct_string",
                    },
                )
                return market_symbol

            # Strategy 2: Enum-like object - try multiple attribute access methods
            symbol_value = None

            # Try .value attribute first (most common for enums)
            if hasattr(market_symbol, "value"):
                try:
                    symbol_value = getattr(market_symbol, "value", None)
                    if symbol_value is not None:
                        logger.debug(
                            "Extracted enum value from MARKET_SYMBOLS object",
                            extra={
                                "service_id": self.service_id,
                                "input_symbol": symbol,
                                "market_symbol": str(market_symbol),
                                "symbol_value": symbol_value,
                                "value_type": type(symbol_value).__name__,
                                "strategy": "enum_value_attribute",
                            },
                        )
                        return str(symbol_value)
                except (AttributeError, TypeError) as attr_err:
                    logger.debug(
                        "Failed to access .value attribute",
                        extra={
                            "service_id": self.service_id,
                            "input_symbol": symbol,
                            "error": str(attr_err),
                            "strategy": "enum_value_attribute_failed",
                        },
                    )

            # Try .name attribute as fallback
            if hasattr(market_symbol, "name"):
                try:
                    symbol_value = getattr(market_symbol, "name", None)
                    if symbol_value is not None:
                        logger.debug(
                            "Extracted enum name from MARKET_SYMBOLS object",
                            extra={
                                "service_id": self.service_id,
                                "input_symbol": symbol,
                                "market_symbol": str(market_symbol),
                                "symbol_value": symbol_value,
                                "strategy": "enum_name_attribute",
                            },
                        )
                        return str(symbol_value)
                except (AttributeError, TypeError) as attr_err:
                    logger.debug(
                        "Failed to access .name attribute",
                        extra={
                            "service_id": self.service_id,
                            "input_symbol": symbol,
                            "error": str(attr_err),
                            "strategy": "enum_name_attribute_failed",
                        },
                    )

            # Strategy 3: Direct string conversion as final fallback
            try:
                symbol_str = str(market_symbol)
                logger.debug(
                    "Using direct string conversion",
                    extra={
                        "service_id": self.service_id,
                        "input_symbol": symbol,
                        "market_symbol_type": type(market_symbol).__name__,
                        "converted_value": symbol_str,
                        "strategy": "direct_string_conversion",
                    },
                )

                # Clean up the string representation if it contains enum prefixes
                if "MARKET_SYMBOLS." in symbol_str:
                    symbol_str = symbol_str.replace("MARKET_SYMBOLS.", "")

                return symbol_str

            except Exception as str_err:
                logger.warning(
                    "Failed to convert market symbol to string",
                    extra={
                        "service_id": self.service_id,
                        "input_symbol": symbol,
                        "error": str(str_err),
                        "strategy": "direct_string_conversion_failed",
                    },
                )

        except Exception as e:
            if isinstance(e, BluefinDataError):
                logger.exception("Bluefin data error in symbol extraction")
                raise

            # If all else fails, provide a safe fallback
            logger.exception(
                "Critical error in symbol value extraction, using normalized symbol as fallback",
                extra={
                    "service_id": self.service_id,
                    "input_symbol": symbol,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "fallback_strategy": "normalized_symbol",
                },
            )

            # Try to normalize the symbol before returning as fallback
            try:
                # Extract base currency from symbol if it's in PERP format
                if "-PERP" in symbol.upper():
                    base_symbol = symbol.upper().split("-")[0]
                    logger.debug(
                        "Using normalized base symbol as fallback",
                        extra={
                            "service_id": self.service_id,
                            "original_symbol": symbol,
                            "fallback_value": base_symbol,
                        },
                    )
                    return base_symbol
                return symbol.upper()
            except Exception as fallback_err:
                logger.exception(
                    "Even fallback normalization failed, returning original symbol",
                    extra={
                        "service_id": self.service_id,
                        "original_symbol": symbol,
                        "fallback_error": str(fallback_err),
                    },
                )
                return symbol

    def _validate_interval(self, interval: str) -> bool:
        """Validate if interval is supported by Bluefin API.

        Returns True only for intervals that are natively supported by Bluefin.
        This does NOT include unsupported intervals that can be converted.
        """
        if not interval or not isinstance(interval, str):
            return False

        # Bluefin API officially supported intervals (from API documentation)
        # Note: Removed "3d" as it's not in the official API docs
        supported = {
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "1w",
            "1M",
        }
        return interval in supported

    def _validate_timestamp(self, timestamp: int) -> bool:
        """Validate timestamp is reasonable (not in future, not too old)."""
        if not isinstance(timestamp, int) or timestamp <= 0:
            return False

        current_time = int(time.time() * 1000)
        # Not in future (allow 1 minute tolerance)
        if timestamp > current_time + 60000:
            return False

        # Not too old (max 1 year)
        if timestamp < current_time - (365 * 24 * 60 * 60 * 1000):
            return False

        return True

    def _validate_limit(self, limit: int) -> bool:
        """Validate limit parameter."""
        return isinstance(limit, int) and 1 <= limit <= API_LIMIT_MAX

    def _get_circuit_state(self) -> str:
        """Get current circuit breaker state with transitions."""
        current_time = time.time()

        if self.circuit_state == "OPEN" and current_time >= self.circuit_open_until:
            # Transition to HALF_OPEN
            self.circuit_state = "HALF_OPEN"
            self.circuit_half_open_calls = 0
            logger.info(
                "Circuit breaker transitioning from OPEN to HALF_OPEN",
                extra={
                    "service_id": self.service_id,
                    "failure_count": self.failure_count,
                    "failure_types": self.failure_types,
                },
            )

        return self.circuit_state

    async def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open or half-open with call limit."""
        state = self._get_circuit_state()

        if state == "OPEN":
            return True
        if (
            state == "HALF_OPEN"
            and self.circuit_half_open_calls >= self.circuit_half_open_max_calls
        ):
            logger.warning(
                "Circuit breaker HALF_OPEN call limit exceeded",
                extra={"service_id": self.service_id},
            )
            return True

        return False

    def _record_api_failure(self, failure_type: str = "unknown"):
        """Record API failure for enhanced circuit breaker."""
        self.failure_count += 1

        # Track failure type for better diagnostics
        if failure_type in self.failure_types:
            self.failure_types[failure_type] += 1

        current_time = time.time()
        state = self._get_circuit_state()

        if state == "HALF_OPEN":
            # Failure in half-open state - go back to open
            self.circuit_state = "OPEN"
            self.circuit_open_until = current_time + self.circuit_recovery_timeout
            logger.warning(
                "Circuit breaker reopened due to failure in HALF_OPEN state",
                extra={
                    "service_id": self.service_id,
                    "failure_type": failure_type,
                    "total_failures": self.failure_count,
                    "failure_types": self.failure_types,
                },
            )
        elif state == "CLOSED" and self.failure_count >= self.circuit_failure_threshold:
            # Too many failures - open circuit
            self.circuit_state = "OPEN"
            self.circuit_open_until = current_time + self.circuit_recovery_timeout
            logger.warning(
                "Circuit breaker opened due to %s failures",
                self.failure_count,
                extra={
                    "service_id": self.service_id,
                    "failure_threshold": self.circuit_failure_threshold,
                    "recovery_timeout": self.circuit_recovery_timeout,
                    "failure_types": self.failure_types,
                    "primary_failure_type": max(
                        self.failure_types, key=self.failure_types.get
                    ),
                },
            )

    def _record_api_success(self):
        """Record API success, manage circuit breaker state."""
        state = self._get_circuit_state()

        if state == "HALF_OPEN":
            self.circuit_half_open_calls += 1

            # If we've had enough successful calls in half-open, close the circuit
            if self.circuit_half_open_calls >= self.circuit_half_open_max_calls:
                self.circuit_state = "CLOSED"
                self.failure_count = 0
                self.circuit_half_open_calls = 0
                # Reset failure type counters on successful recovery
                self.failure_types = dict.fromkeys(self.failure_types, 0)
                logger.info(
                    "Circuit breaker closed after successful recovery",
                    extra={"service_id": self.service_id},
                )
        elif self.failure_count > 0:
            # Gradual recovery in closed state
            old_count = self.failure_count
            self.failure_count = max(0, self.failure_count - 1)
            logger.debug(
                "API call succeeded, reducing failure count from %s to %s",
                old_count,
                self.failure_count,
                extra={"service_id": self.service_id},
            )

    def reset_circuit_breaker(self, force: bool = False) -> bool:
        """
        Manual circuit breaker reset functionality for the SDK service.

        Args:
            force: If True, force reset even if not ready naturally

        Returns:
            True if reset was successful
        """
        current_time = time.time()
        old_state = self.circuit_state
        old_failures = self.failure_count

        if force or (
            self.circuit_state == "OPEN" and current_time >= self.circuit_open_until
        ):
            self.circuit_state = "CLOSED"
            self.circuit_open_until = 0
            self.failure_count = 0
            self.circuit_half_open_calls = 0
            # Reset failure type counters
            self.failure_types = dict.fromkeys(self.failure_types, 0)

            logger.info(
                "SDK service circuit breaker manually reset",
                extra={
                    "service_id": self.service_id,
                    "previous_state": old_state,
                    "previous_failures": old_failures,
                    "force_reset": force,
                    "time_until_natural_reset": (
                        max(0, self.circuit_open_until - current_time)
                        if old_state == "OPEN"
                        else 0
                    ),
                },
            )
            return True
        if self.circuit_state == "OPEN":
            remaining_time = self.circuit_open_until - current_time
            logger.warning(
                "SDK circuit breaker reset not ready - %.1fs remaining",
                remaining_time,
                extra={
                    "service_id": self.service_id,
                    "remaining_seconds": remaining_time,
                    "current_state": self.circuit_state,
                    "use_force_to_override": True,
                },
            )
        else:
            logger.info(
                "SDK circuit breaker already in %s state",
                self.circuit_state,
                extra={
                    "service_id": self.service_id,
                    "current_state": self.circuit_state,
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
            "state": self.circuit_state,
            "failure_count": self.failure_count,
            "failure_threshold": self.circuit_failure_threshold,
            "recovery_timeout": self.circuit_recovery_timeout,
            "open_until": self.circuit_open_until,
            "seconds_until_reset": (
                max(0, self.circuit_open_until - current_time)
                if self.circuit_state == "OPEN"
                else 0
            ),
            "half_open_calls": self.circuit_half_open_calls,
            "half_open_max_calls": self.circuit_half_open_max_calls,
            "failure_types": dict(self.failure_types),
            "can_reset_manually": self.circuit_state != "OPEN"
            or current_time >= self.circuit_open_until,
        }

    def _record_request_start(self) -> float:
        """Record the start of a request for health monitoring."""
        request_start_time = time.time()
        self.health_stats["total_requests"] += 1
        self.health_stats["last_request_time"] = request_start_time
        return request_start_time

    def _record_request_end(self, start_time: float, success: bool):
        """Record the end of a request for health monitoring."""
        end_time = time.time()
        response_time = end_time - start_time

        # Update response time statistics
        self.health_stats["response_times"].append(response_time)
        if (
            len(self.health_stats["response_times"])
            > self.health_stats["max_response_times"]
        ):
            self.health_stats["response_times"].pop(0)  # Remove oldest

        # Calculate average response time
        if self.health_stats["response_times"]:
            self.health_stats["average_response_time"] = sum(
                self.health_stats["response_times"]
            ) / len(self.health_stats["response_times"])

        # Update success/failure counters
        if success:
            self.health_stats["successful_requests"] += 1
            self.health_stats["last_success_time"] = end_time
        else:
            self.health_stats["failed_requests"] += 1
            self.health_stats["last_failure_time"] = end_time

    def get_health_status(self) -> dict[str, Any]:
        """
        Get comprehensive health status for monitoring and diagnostics.

        Returns:
            Dictionary with health statistics and status information
        """
        current_time = time.time()
        uptime = current_time - self.health_stats["start_time"]

        # Calculate success rate
        total_requests = self.health_stats["total_requests"]
        success_rate = (
            (self.health_stats["successful_requests"] / total_requests * 100)
            if total_requests > 0
            else 0
        )

        # Calculate request rate (requests per minute)
        request_rate = (total_requests / uptime * 60) if uptime > 0 else 0

        return {
            "service_id": self.service_id,
            "network": self.network,
            "initialized": self.initialized,
            "has_client": bool(self.client),
            "uptime_seconds": uptime,
            "circuit_breaker": self.get_circuit_breaker_status(),
            "requests": {
                "total": total_requests,
                "successful": self.health_stats["successful_requests"],
                "failed": self.health_stats["failed_requests"],
                "success_rate_percent": round(success_rate, 2),
                "rate_per_minute": round(request_rate, 2),
            },
            "response_times": {
                "average_ms": round(
                    self.health_stats["average_response_time"] * 1000, 2
                ),
                "sample_count": len(self.health_stats["response_times"]),
                "latest_ms": (
                    round(self.health_stats["response_times"][-1] * 1000, 2)
                    if self.health_stats["response_times"]
                    else 0
                ),
            },
            "last_activity": {
                "request_time": self.health_stats["last_request_time"],
                "success_time": self.health_stats["last_success_time"],
                "failure_time": self.health_stats["last_failure_time"],
                "seconds_since_last_request": (
                    round(current_time - self.health_stats["last_request_time"], 1)
                    if self.health_stats["last_request_time"] > 0
                    else None
                ),
                "seconds_since_last_success": (
                    round(current_time - self.health_stats["last_success_time"], 1)
                    if self.health_stats["last_success_time"] > 0
                    else None
                ),
            },
            "status": (
                "healthy"
                if self.initialized
                and success_rate > DEFAULT_API_LIMIT
                and self.circuit_state == "CLOSED"
                else "degraded"
            ),
        }

    async def _retry_request(self, func, *args, **kwargs):
        """Execute request with enhanced exponential backoff retry and circuit breaker."""
        if await self._is_circuit_open():
            logger.warning(
                "Circuit breaker is %s, skipping request",
                self._get_circuit_state(),
                extra={"service_id": self.service_id},
            )
            return None

        last_exception = None
        state = self._get_circuit_state()

        # Track half-open calls
        if state == "HALF_OPEN":
            self.circuit_half_open_calls += 1

        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                self._record_api_success()
                return result
            except TimeoutError as e:
                last_exception = e
                self._record_api_failure("timeout")
                if attempt < self.max_retries - 1:
                    delay = (
                        self.base_retry_delay
                        * (2**attempt)
                        * (0.8 + secrets.randbits(16) / (2**16) * 0.4)
                    )
                    logger.warning(
                        "Timeout error (attempt %s/%s), retrying in %.2fs: %s",
                        attempt + 1,
                        self.max_retries,
                        delay,
                        e,
                        extra={"service_id": self.service_id},
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        "Timeout error after %s attempts",
                        self.max_retries,
                        extra={"service_id": self.service_id},
                    )
            except ClientError as e:
                last_exception = e

                # Classify the error type
                if "connection" in str(e).lower():
                    failure_type = "connection"
                elif "rate" in str(e).lower() or "429" in str(e):
                    failure_type = "rate_limit"
                elif "auth" in str(e).lower() or "401" in str(e) or "403" in str(e):
                    failure_type = "auth"
                elif any(code in str(e) for code in ["500", "502", "503", "504"]):
                    failure_type = "server_error"
                else:
                    failure_type = "connection"

                self._record_api_failure(failure_type)

                if attempt < self.max_retries - 1:
                    delay = (
                        self.base_retry_delay
                        * (2**attempt)
                        * (0.8 + secrets.randbits(16) / (2**16) * 0.4)
                    )
                    logger.warning(
                        "Client error (attempt %s/%s), retrying in %.2fs: %s",
                        attempt + 1,
                        self.max_retries,
                        delay,
                        e,
                        extra={
                            "service_id": self.service_id,
                            "failure_type": failure_type,
                        },
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        "Client error after %s attempts",
                        self.max_retries,
                        extra={
                            "service_id": self.service_id,
                            "failure_type": failure_type,
                        },
                    )
            except Exception as e:
                logger.exception(
                    "Non-retryable error",
                    extra={
                        "service_id": self.service_id,
                        "error_type": type(e).__name__,
                    },
                )
                self._record_api_failure("unknown")
                last_exception = e
                break

        if last_exception:
            raise last_exception
        return None

    def _validate_private_key(self) -> None:
        """Validate and normalize private key format.

        Raises:
            BluefinConnectionError: If private key is invalid
        """
        if not self.private_key:
            error_msg = "BLUEFIN_PRIVATE_KEY environment variable not set"
            logger.error(
                "Private key validation failed",
                extra={
                    "service_id": self.service_id,
                    "error_type": "missing_private_key",
                    "required_env_var": "BLUEFIN_PRIVATE_KEY",
                },
            )
            raise BluefinConnectionError(error_msg)

        key_validation_errors = []
        original_key = self.private_key.strip()
        words = original_key.split()

        if len(words) in [MIN_MNEMONIC_WORDS, MAX_MNEMONIC_WORDS]:
            self._validate_mnemonic_key(words, key_validation_errors)
        else:
            self._validate_hex_key(original_key, key_validation_errors)

        # Check for invalid patterns in hex keys
        if len(words) not in [12, 24]:
            self._check_invalid_hex_patterns(key_validation_errors)

        if key_validation_errors:
            self._raise_key_validation_error(key_validation_errors)

    def _validate_mnemonic_key(self, words: list[str], errors: list[str]) -> None:
        """Validate mnemonic private key format."""
        logger.info(
            "Detected %s-word mnemonic phrase, keeping original format for BluefinClient",
            len(words),
            extra={
                "service_id": self.service_id,
                "mnemonic_word_count": len(words),
                "format_type": "mnemonic_passthrough",
            },
        )

        if all(word.isalpha() and len(word) > MIN_WORD_LENGTH for word in words):
            logger.info(
                "Mnemonic validation passed, will pass directly to BluefinClient",
                extra={
                    "service_id": self.service_id,
                    "original_format": "mnemonic",
                    "client_format": "mnemonic",
                    "word_count": len(words),
                },
            )
            # Keep original mnemonic - BluefinClient handles BIP39 conversion internally
            self.private_key = " ".join(words)
        else:
            errors.append(
                "Invalid mnemonic format: words must be alphabetic and >2 characters"
            )

    def _validate_hex_key(self, original_key: str, errors: list[str]) -> None:
        """Validate hexadecimal private key format."""
        if len(original_key) < MIN_KEY_LENGTH:
            errors.append(
                f"Too short: got {len(original_key)} chars, expected {MIN_KEY_LENGTH}+ for hex or {MIN_MNEMONIC_WORDS}/{MAX_MNEMONIC_WORDS} words for mnemonic"
            )

        try:
            clean_key = original_key
            if clean_key.startswith(("0x", "0X")):
                clean_key = clean_key[2:]

            bytes.fromhex(clean_key)

            if clean_key != original_key:
                logger.debug(
                    "Cleaned private key by removing hex prefix",
                    extra={
                        "service_id": self.service_id,
                        "original_length": len(original_key),
                        "cleaned_length": len(clean_key),
                    },
                )
                self.private_key = clean_key
        except ValueError as hex_err:
            errors.append(f"Invalid hexadecimal format: {hex_err!s}")

    def _check_invalid_hex_patterns(self, errors: list[str]) -> None:
        """Check for common invalid hex key patterns."""
        if self.private_key.count("0") == len(self.private_key):
            errors.append("All zeros - invalid private key")
        elif self.private_key.count("f") == len(self.private_key.replace("F", "f")):
            errors.append("All 0xf values - likely invalid private key")

    def _raise_key_validation_error(self, errors: list[str]) -> None:
        """Raise validation error with detailed information."""
        error_msg = f"BLUEFIN_PRIVATE_KEY validation failed: {'; '.join(errors)}"
        logger.error(
            "Private key format validation failed",
            extra={
                "service_id": self.service_id,
                "error_type": "invalid_private_key_format",
                "key_length": len(self.private_key),
                "validation_errors": errors,
                "expected_format": "64+ hexadecimal characters (32+ bytes)",
            },
        )
        raise BluefinConnectionError(error_msg)

    def _resolve_network_config(self):
        """Resolve network configuration from string to Networks object.

        Returns:
            Network configuration object

        Raises:
            BluefinConnectionError: If network is invalid
        """
        try:
            network = (
                Networks["SUI_PROD"]
                if self.network == "mainnet"
                else Networks["SUI_STAGING"]
            )
            logger.debug(
                "Network configuration resolved",
                extra={
                    "service_id": self.service_id,
                    "network_name": self.network,
                    "network_config": str(network)[:100],  # Truncate for logging
                },
            )
            return network
        except KeyError as e:
            error_msg = f"Invalid network configuration '{self.network}'. Expected 'mainnet' or 'testnet'"
            logger.exception(
                "Network configuration failed",
                extra={
                    "service_id": self.service_id,
                    "error_type": "invalid_network",
                    "network_name": self.network,
                    "available_networks": list(Networks.keys()),
                    "original_error": str(e),
                },
            )
            raise BluefinConnectionError(error_msg) from e

    def _should_retry_init_error(self, error: Exception) -> bool:
        """Determine if initialization error is retryable.

        Args:
            error: Exception that occurred during initialization

        Returns:
            True if error should be retried, False otherwise
        """
        error_str = str(error).lower()

        # Retryable network/connection errors
        retryable_keywords = [
            "network",
            "connection",
            "timeout",
            "temporary",
            "503",
            "502",
            "500",
        ]
        if any(keyword in error_str for keyword in retryable_keywords):
            return True

        # Non-retryable authentication errors
        if "private key" in error_str or "authentication" in error_str:
            return False

        # Non-retryable validation errors
        if "invalid" in error_str and "key" in error_str:
            return False

        # Default to retry for unknown errors
        return True

    async def _initialize_client_with_retry(self, network) -> None:
        """Initialize Bluefin client with retry mechanism.

        Args:
            network: Network configuration object

        Raises:
            BluefinAPIError: If all initialization attempts fail
        """
        max_init_retries = 3
        init_retry_delay = 2.0

        for init_attempt in range(max_init_retries):
            logger.info(
                "Creating Bluefin client instance (attempt %s/%s)",
                init_attempt + 1,
                max_init_retries,
                extra={
                    "service_id": self.service_id,
                    "network": self.network,
                    "onchain_enabled": True,
                    "attempt": init_attempt + 1,
                },
            )

            try:
                self.client = BluefinClient(True, network, self.private_key)
                logger.debug(
                    "Bluefin client instance created successfully",
                    extra={
                        "service_id": self.service_id,
                        "client_type": type(self.client).__name__,
                        "attempt": init_attempt + 1,
                    },
                )

                logger.info(
                    "Initializing Bluefin client connection",
                    extra={
                        "service_id": self.service_id,
                        "initialization_step": "client_init",
                        "attempt": init_attempt + 1,
                    },
                )

                await self.client.init(True)

                logger.info(
                    "Bluefin client initialization completed successfully",
                    extra={
                        "service_id": self.service_id,
                        "initialization_successful": True,
                        "successful_attempt": init_attempt + 1,
                        "total_attempts": init_attempt + 1,
                    },
                )
                return  # Success - exit retry loop

            except Exception as e:
                if init_attempt < max_init_retries - 1:
                    if self._should_retry_init_error(e):
                        delay = init_retry_delay * (
                            2**init_attempt
                        )  # Exponential backoff
                        logger.warning(
                            "Client initialization failed (attempt %s/%s), retrying in %.1fs: %s",
                            init_attempt + 1,
                            max_init_retries,
                            delay,
                            e,
                            extra={
                                "service_id": self.service_id,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "attempt": init_attempt + 1,
                                "max_retries": max_init_retries,
                                "retry_delay": delay,
                            },
                        )
                        await asyncio.sleep(delay)
                        continue

                    # Non-retryable error
                    logger.exception(
                        "Non-retryable initialization error",
                        extra={
                            "service_id": self.service_id,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "attempt": init_attempt + 1,
                            "non_retryable": True,
                        },
                    )
                    raise BluefinAPIError(
                        f"Failed to initialize Bluefin client: {e!s}"
                    ) from e

                # Final attempt failed
                error_msg = f"Failed to initialize Bluefin client after {max_init_retries} attempts: {e!s}"
                logger.exception(
                    "All initialization attempts failed",
                    extra={
                        "service_id": self.service_id,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "total_attempts": max_init_retries,
                        "traceback": traceback.format_exc(),
                    },
                )
                raise BluefinAPIError(error_msg) from e

    async def initialize(self) -> bool:
        """
        Initialize the Bluefin SDK client with comprehensive error handling.

        Returns:
            True if initialization successful, False otherwise

        Raises:
            BluefinConnectionError: When connection setup fails
            BluefinAPIError: When SDK initialization fails
        """
        logger.info(
            "Starting Bluefin SDK initialization",
            extra={
                "service_id": self.service_id,
                "network": self.network,
                "initialization_attempt": True,
            },
        )

        try:
            self._validate_private_key()
            network = self._resolve_network_config()
            await self._initialize_client_with_retry(network)

            self.initialized = True
            logger.info(
                "Bluefin SDK service initialized successfully",
                extra={
                    "service_id": self.service_id,
                    "network": self.network,
                    "initialized": True,
                    "client_ready": bool(self.client),
                },
            )
            return True

        except (BluefinConnectionError, BluefinAPIError):
            logger.exception("Bluefin service error during initialization")
            raise
        except Exception as e:
            error_msg = f"Unexpected error during Bluefin SDK initialization: {e!s}"
            logger.exception(
                "Unexpected initialization error",
                extra={
                    "service_id": self.service_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            raise BluefinAPIError(error_msg) from e

    async def cleanup(self) -> None:
        """Cleanup service resources on shutdown."""
        if self._cleanup_complete:
            return

        logger.info("Cleaning up Bluefin SDK service resources...")
        try:
            if self.client:
                # Try to close any underlying HTTP sessions
                try:
                    # Check if the client has a session attribute to close
                    if (
                        hasattr(self.client, "session")
                        and self.client.session
                        and hasattr(self.client.session, "close")
                    ):
                        await self.client.session.close()
                        logger.debug("Closed Bluefin client HTTP session")

                    # Check for other common session attributes
                    if (
                        hasattr(self.client, "_session")
                        and self.client._session
                        and hasattr(self.client._session, "close")
                    ):
                        await self.client._session.close()
                        logger.debug("Closed Bluefin client private HTTP session")

                    # Check if client has a cleanup/close method
                    if hasattr(self.client, "close"):
                        await self.client.close()
                        logger.debug("Called Bluefin client close method")
                    elif hasattr(self.client, "cleanup"):
                        await self.client.cleanup()
                        logger.debug("Called Bluefin client cleanup method")

                except Exception as session_err:
                    logger.warning(
                        "Error closing Bluefin client sessions: %s", session_err
                    )

                # Clear client reference
                self.client = None
                logger.debug("Bluefin client reference cleared")

            self.initialized = False
            self._cleanup_complete = True
            logger.info("Bluefin SDK service cleanup completed")
        except Exception:
            logger.exception("Error during Bluefin SDK service cleanup")

    async def get_account_data(self) -> dict[str, Any]:
        """
        Get account data including balances with comprehensive error handling.

        Returns:
            Dictionary containing account data or error information
        """
        logger.debug(
            "Starting account data retrieval",
            extra={
                "service_id": self.service_id,
                "initialized": self.initialized,
                "has_client": bool(self.client),
            },
        )

        if not self.initialized:
            error_msg = "SDK not initialized - call initialize() first"
            logger.error(
                "Account data request failed - service not initialized",
                extra={
                    "service_id": self.service_id,
                    "error_type": "service_not_initialized",
                    "initialized": False,
                },
            )
            return {"error": error_msg}

        try:
            # Initialize default response
            account_data = {
                "balance": "0.0",
                "margin": {"available": "0.0", "used": "0.0"},
                "address": "",
                "error": None,
            }

            # Generate correlation ID for this balance operation
            correlation_id = str(uuid.uuid4())
            operation_start = time.time()

            logger.debug(
                "Initialized default account data structure",
                extra={
                    "service_id": self.service_id,
                    "correlation_id": correlation_id,
                    "operation": "get_account_data_init",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "default_balance": account_data["balance"],
                    "default_margin": account_data["margin"],
                },
            )

            # Get public address (this should always work)
            try:
                address_start = time.time()
                account_data["address"] = self.client.get_public_address()
                address_duration = time.time() - address_start

                logger.info(
                    "Successfully retrieved account address",
                    extra={
                        "service_id": self.service_id,
                        "correlation_id": correlation_id,
                        "operation": "get_public_address",
                        "duration_ms": round(address_duration * 1000, 2),
                        "timestamp": datetime.now(UTC).isoformat(),
                        "address_truncated": (
                            account_data["address"][:8]
                            + "..."
                            + account_data["address"][-4:]
                            if account_data["address"]
                            and len(account_data["address"])
                            > ADDRESS_DISPLAY_MIN_LENGTH
                            else account_data["address"]
                        ),
                        "address_length": (
                            len(account_data["address"])
                            if account_data["address"]
                            else 0
                        ),
                        "success": True,
                    },
                )

                # Record successful address retrieval
                self._record_balance_audit(
                    correlation_id,
                    "get_public_address",
                    "success",
                    duration_ms=round(address_duration * 1000, 2),
                    metadata={
                        "address_length": (
                            len(account_data["address"])
                            if account_data["address"]
                            else 0
                        )
                    },
                )

            except Exception as e:
                address_duration = time.time() - address_start
                error_context = self._get_error_context(
                    correlation_id,
                    "get_public_address",
                    e,
                    round(address_duration * 1000, 2),
                )

                logger.warning(
                    "Failed to retrieve public address - SDK client error",
                    extra=error_context,
                )

                # Record audit trail for address retrieval failure
                self._record_balance_audit(
                    correlation_id,
                    "get_public_address",
                    "failed",
                    error=str(e),
                    duration_ms=round(address_duration * 1000, 2),
                )

            # Try to get account balances with enhanced retry logic
            try:
                balance_start = time.time()

                # Check if we should use cached data
                if not self._is_balance_stale() and self.cached_balance_data:
                    logger.info(
                        "Using cached balance data",
                        extra={
                            "service_id": self.service_id,
                            "correlation_id": correlation_id,
                            "cache_age_seconds": time.time()
                            - self.balance_cache_timestamp,
                            "operation": "get_usdc_balance",
                        },
                    )
                    account_data["balance"] = str(
                        self.cached_balance_data.get("balance", "0.0")
                    )
                else:
                    logger.debug(
                        "Starting USDC balance retrieval with retry logic",
                        extra={
                            "service_id": self.service_id,
                            "correlation_id": correlation_id,
                            "operation": "get_usdc_balance",
                            "client_available": bool(self.client),
                            "retry_config": self.balance_retry_config,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )

                    # Define the balance fetching operation for retry logic
                    async def _fetch_usdc_balance():
                        """Inner function for balance fetching with SDK error handling."""
                        sdk_call_start = time.time()
                        try:
                            balance_response = await self.client.get_usdc_balance()
                            sdk_call_duration = time.time() - sdk_call_start

                            logger.debug(
                                "Received balance response from SDK",
                                extra={
                                    "service_id": self.service_id,
                                    "correlation_id": correlation_id,
                                    "response_type": type(balance_response).__name__,
                                    "response_value": str(balance_response)[:100],
                                    "operation": "get_usdc_balance",
                                    "sdk_call_duration_ms": round(
                                        sdk_call_duration * 1000, 2
                                    ),
                                    "timestamp": datetime.now(UTC).isoformat(),
                                },
                            )
                            return balance_response
                        except Exception as sdk_error:
                            sdk_call_duration = time.time() - sdk_call_start
                            error_str = str(sdk_error)
                            error_type = type(sdk_error).__name__

                            # Handle known Bluefin SDK issues
                            if (
                                "'data'" in error_str
                                or "KeyError" in error_type
                                or "Exception: 'data'" in error_str
                            ):
                                logger.warning(
                                    "Detected known Bluefin SDK data access error",
                                    extra={
                                        "service_id": self.service_id,
                                        "correlation_id": correlation_id,
                                        "error_type": "sdk_data_access_error",
                                        "error_message": error_str,
                                        "probable_cause": "account_not_initialized_or_no_balance",
                                        "operation": "get_usdc_balance",
                                        "sdk_call_duration_ms": round(
                                            sdk_call_duration * 1000, 2
                                        ),
                                    },
                                )
                                # Return a default balance structure for account not initialized
                                return {"balance": 0, "amount": 0}
                            # Log the error but re-raise for retry logic to handle
                            logger.debug(
                                "SDK balance call failed, will be retried",
                                extra={
                                    "service_id": self.service_id,
                                    "correlation_id": correlation_id,
                                    "sdk_error_type": error_type,
                                    "sdk_error_message": error_str,
                                    "operation": "get_usdc_balance",
                                    "sdk_call_duration_ms": round(
                                        sdk_call_duration * 1000, 2
                                    ),
                                },
                            )
                            # Re-raise the original exception for retry logic to handle
                            raise

                    # Execute with retry logic
                    balance_response = await self._retry_balance_operation(
                        _fetch_usdc_balance, "get_usdc_balance"
                    )

                # Handle different response formats
                if isinstance(balance_response, dict):
                    # If it's a dict, look for balance in common keys
                    balance = balance_response.get(
                        "balance",
                        balance_response.get("data", balance_response.get("amount", 0)),
                    )
                    logger.debug(
                        "Extracted balance from dictionary response",
                        extra={
                            "service_id": self.service_id,
                            "correlation_id": correlation_id,
                            "extracted_balance": balance,
                            "response_keys": list(balance_response.keys()),
                            "operation": "get_usdc_balance",
                            "response_format": "dictionary",
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )
                elif isinstance(balance_response, int | float | str):
                    # If it's a direct value
                    balance = balance_response
                    logger.debug(
                        "Using direct balance value",
                        extra={
                            "service_id": self.service_id,
                            "correlation_id": correlation_id,
                            "balance_value": balance,
                            "balance_type": type(balance).__name__,
                            "operation": "get_usdc_balance",
                            "response_format": "direct_value",
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )
                else:
                    logger.warning(
                        "Unexpected balance response format",
                        extra={
                            "service_id": self.service_id,
                            "response_type": type(balance_response).__name__,
                            "response_value": str(balance_response),
                            "operation": "get_usdc_balance",
                        },
                    )
                    balance = 0

                balance_duration = time.time() - balance_start
                account_data["balance"] = str(balance)

                logger.info(
                    "Successfully retrieved account balance",
                    extra={
                        "service_id": self.service_id,
                        "correlation_id": correlation_id,
                        "balance": account_data["balance"],
                        "currency": "USDC",
                        "operation": "get_usdc_balance",
                        "duration_ms": round(balance_duration * 1000, 2),
                        "timestamp": datetime.now(UTC).isoformat(),
                        "success": True,
                    },
                )

                # Record successful balance retrieval
                self._record_balance_audit(
                    correlation_id,
                    "get_usdc_balance",
                    "success",
                    balance_amount=account_data["balance"],
                    duration_ms=round(balance_duration * 1000, 2),
                    metadata={
                        "currency": "USDC",
                        "response_format": (
                            "dictionary"
                            if isinstance(balance_response, dict)
                            else "direct_value"
                        ),
                        "balance_numeric": float(balance) if balance else 0.0,
                    },
                )

                # Cache the successful balance data
                self.cached_balance_data = {"balance": balance}
                self.balance_cache_timestamp = time.time()

            except Exception as e:
                balance_duration = time.time() - balance_start
                error_type = self._classify_error(e)

                error_context = self._get_error_context(
                    correlation_id,
                    "get_usdc_balance_general",
                    e,
                    round(balance_duration * 1000, 2),
                    additional_context={
                        "traceback": traceback.format_exc(),
                        "error_stage": "general_processing",
                        "error_classification": error_type.value,
                        "has_cached_data": bool(self.cached_balance_data),
                    },
                )

                logger.exception(
                    "Balance retrieval failed after all retries",
                    extra=error_context,
                )

                # Record audit for general error
                self._record_balance_audit(
                    correlation_id,
                    "get_usdc_balance_general",
                    "failed",
                    error=str(e),
                    duration_ms=round(balance_duration * 1000, 2),
                    metadata={
                        "error_stage": "general_processing",
                        "error_classification": error_type.value,
                        "has_cached_data": bool(self.cached_balance_data),
                        "requires_investigation": True,
                    },
                )

                # Use cached data if available as fallback
                if self.cached_balance_data:
                    logger.warning(
                        "Using stale cached balance data as fallback",
                        extra={
                            "service_id": self.service_id,
                            "correlation_id": correlation_id,
                            "cache_age_seconds": time.time()
                            - self.balance_cache_timestamp,
                            "operation": "get_usdc_balance",
                        },
                    )
                    account_data["balance"] = str(
                        self.cached_balance_data.get("balance", "0.0")
                    )
                    account_data["error"] = f"Using cached data due to error: {e!s}"
                else:
                    account_data["error"] = f"Balance fetch failed: {e!s}"

            # Try to get margin info with retry logic
            try:
                logger.debug(
                    "Starting margin bank balance retrieval with retry logic",
                    extra={
                        "service_id": self.service_id,
                        "correlation_id": correlation_id,
                        "operation": "get_margin_bank_balance",
                        "client_available": bool(self.client),
                    },
                )

                async def _fetch_margin_balance():
                    """Inner function for margin balance fetching with SDK error handling."""
                    try:
                        margin_response = await self.client.get_margin_bank_balance()
                        logger.debug(
                            "Received margin response from SDK",
                            extra={
                                "service_id": self.service_id,
                                "correlation_id": correlation_id,
                                "response_type": type(margin_response).__name__,
                                "response_value": str(margin_response)[:100],
                                "operation": "get_margin_bank_balance",
                            },
                        )
                        return margin_response
                    except Exception as sdk_error:
                        error_str = str(sdk_error)
                        error_type = type(sdk_error).__name__

                        # Handle known Bluefin SDK issues
                        if (
                            "'data'" in error_str
                            or "KeyError" in error_type
                            or "Exception: 'data'" in error_str
                        ):
                            logger.warning(
                                "Detected known Bluefin SDK data access error in margin call",
                                extra={
                                    "service_id": self.service_id,
                                    "correlation_id": correlation_id,
                                    "error_type": "sdk_data_access_error",
                                    "error_message": error_str,
                                    "probable_cause": "account_not_initialized_or_no_margin",
                                    "operation": "get_margin_bank_balance",
                                },
                            )
                            # Return a default margin structure for account not initialized
                            return {"available": 0, "used": 0, "total": 0}
                        # Log the error but re-raise for retry logic to handle
                        logger.debug(
                            "SDK margin call failed, will be retried",
                            extra={
                                "service_id": self.service_id,
                                "correlation_id": correlation_id,
                                "sdk_error_type": error_type,
                                "sdk_error_message": error_str,
                                "operation": "get_margin_bank_balance",
                            },
                        )
                        # Re-raise the original exception for retry logic to handle
                        raise

                # Execute with retry logic
                margin_response = await self._retry_balance_operation(
                    _fetch_margin_balance, "get_margin_bank_balance"
                )

                # Handle different margin response formats
                if isinstance(margin_response, dict):
                    account_data["margin"] = {
                        "available": str(margin_response.get("available", 0)),
                        "used": str(margin_response.get("used", 0)),
                        "total": str(margin_response.get("total", 0)),
                    }
                else:
                    account_data["margin"] = {
                        "available": str(margin_response) if margin_response else "0.0",
                        "used": "0.0",
                        "total": str(margin_response) if margin_response else "0.0",
                    }

                logger.info("Account margin info: %s", account_data["margin"])

            except Exception as e:
                logger.exception("Failed to get margin info")
                if account_data["error"]:
                    account_data["error"] += f", Margin fetch failed: {e!s}"
                else:
                    account_data["error"] = f"Margin fetch failed: {e!s}"

            # Record final operation metrics
            total_duration = time.time() - operation_start
            logger.info(
                "Account data operation completed",
                extra={
                    "service_id": self.service_id,
                    "correlation_id": correlation_id,
                    "operation": "get_account_data_complete",
                    "total_duration_ms": round(total_duration * 1000, 2),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "has_balance": bool(account_data.get("balance", "0") != "0"),
                    "has_address": bool(account_data.get("address")),
                    "has_error": bool(account_data.get("error")),
                    "success": not bool(account_data.get("error")),
                },
            )

            # Record final audit entry
            final_status = (
                "success" if not account_data.get("error") else "partial_failure"
            )
            self._record_balance_audit(
                correlation_id,
                "get_account_data_complete",
                final_status,
                balance_amount=account_data.get("balance"),
                error=account_data.get("error"),
                duration_ms=round(total_duration * 1000, 2),
                metadata={
                    "has_address": bool(account_data.get("address")),
                    "has_margin": bool(account_data.get("margin")),
                    "operation_type": "complete_account_data_fetch",
                },
            )

            return account_data

        except Exception as e:
            logger.exception("Error getting account data")
            logger.exception("Exception type: %s", type(Exception).__name__)

            # Handle known Bluefin SDK data access issues
            error_msg = str(e)
            if "'data'" in error_msg and "KeyError" in str(type(e)):
                logger.warning(
                    "Detected Bluefin SDK data access error - likely missing 'data' field in API response"
                )
                return {
                    "error": "Bluefin API response missing expected data fields - account may not be initialized",
                    "balance": "0.0",
                    "margin": {"available": "0.0", "used": "0.0"},
                    "address": account_data.get("address", ""),
                }

            return {
                "error": str(e),
                "balance": "0.0",
                "margin": {"available": "0.0"},
                "address": "",
            }

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions."""
        if not self.initialized:
            return []

        try:
            # Get user position (returns a single position object)
            position = await self.client.get_user_position({})

            # Check if we got data
            if not position:
                return []

            # If it's a list of positions
            if isinstance(position, list):
                formatted_positions = []
                for pos in position:
                    if pos:  # Skip None/empty entries
                        # Log position symbol only for debugging
                        logger.debug(
                            "Processing position for symbol: %s",
                            pos.get("symbol", "Unknown"),
                        )

                        # Check if there's a direct side field
                        if "side" in pos:
                            side = pos["side"]
                            # Map Bluefin side to our convention
                            if side == "SELL":
                                side = "SHORT"
                            elif side == "BUY":
                                side = "LONG"
                        elif "positionSide" in pos:
                            side = pos["positionSide"]
                        else:
                            # For Bluefin, negative quantity = SHORT, positive = LONG
                            quantity = float(pos.get("quantity", 0))
                            side = "LONG" if quantity > 0 else "SHORT"

                        formatted_positions.append(
                            {
                                "symbol": pos.get("symbol", ""),
                                "quantity": pos.get("quantity", "0"),
                                "avgPrice": pos.get(
                                    "avgEntryPrice", pos.get("avgPrice", "0")
                                ),
                                "unrealizedPnl": pos.get("unrealizedPnl", "0"),
                                "realizedPnl": pos.get("realizedPnl", "0"),
                                "leverage": pos.get("leverage", 1),
                                "margin": pos.get("margin", "0"),
                                "side": side,
                            }
                        )
                return formatted_positions
            # If it's a single position dict
            if isinstance(position, dict):
                # Log position symbol only for debugging
                logger.debug(
                    "Processing position for symbol: %s",
                    position.get("symbol", "Unknown"),
                )

                # Check if there's a direct side field
                if "side" in position:
                    side = position["side"]
                    # Map Bluefin side to our convention
                    if side == "SELL":
                        side = "SHORT"
                    elif side == "BUY":
                        side = "LONG"
                elif "positionSide" in position:
                    side = position["positionSide"]
                else:
                    # For Bluefin, negative quantity = SHORT, positive = LONG
                    quantity = float(position.get("quantity", 0))
                    side = "LONG" if quantity > 0 else "SHORT"

                return [
                    {
                        "symbol": position.get("symbol", ""),
                        "quantity": position.get("quantity", "0"),
                        "avgPrice": position.get(
                            "avgEntryPrice", position.get("avgPrice", "0")
                        ),
                        "unrealizedPnl": position.get("unrealizedPnl", "0"),
                        "realizedPnl": position.get("realizedPnl", "0"),
                        "leverage": position.get("leverage", 1),
                        "margin": position.get("margin", "0"),
                        "side": side,
                    }
                ]
            logger.warning("Unexpected position type: %s", type(position))
            return []

        except Exception:
            logger.exception("Error getting positions")
            return []

    def _validate_and_normalize_symbol(self, symbol: str) -> str:
        """
        Validate and normalize a symbol before use in API calls.

        Args:
            symbol: Input symbol string

        Returns:
            Normalized symbol string suitable for Bluefin API

        Raises:
            BluefinDataError: If symbol is invalid or cannot be normalized
        """
        if not symbol or not isinstance(symbol, str):
            raise BluefinDataError(f"Invalid symbol input: {symbol}")

        # Check if symbol is supported on the current network and apply fallback if needed
        if self.network == "testnet" and not is_bluefin_symbol_supported(
            symbol, "testnet"
        ):
            fallback_symbol = get_testnet_symbol_fallback(symbol)
            logger.warning(
                "Symbol %s not available on testnet, using fallback: %s",
                symbol,
                fallback_symbol,
                extra={
                    "service_id": self.service_id,
                    "original_symbol": symbol,
                    "fallback_symbol": fallback_symbol,
                    "network": self.network,
                },
            )
            symbol = fallback_symbol

        # Validate using symbol utilities
        if not self._validate_symbol(symbol):
            # Try to normalize the symbol and validate again
            try:
                from bot.utils.symbol_utils import normalize_symbol

                normalized = normalize_symbol(symbol, "PERP")

                if self._validate_symbol(normalized):
                    logger.info(
                        "Normalized invalid symbol %s to %s", symbol, normalized
                    )
                    return normalized
                raise BluefinDataError(
                    f"Symbol {symbol} cannot be normalized to valid format"
                )
            except Exception as e:
                raise BluefinDataError(
                    f"Symbol validation failed for {symbol}: {e}"
                ) from e

        return symbol

    async def place_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Place an order."""
        if not self.initialized:
            return {"status": "error", "message": "SDK not initialized"}

        try:
            # Validate and normalize symbol first
            raw_symbol = order_data["symbol"]
            symbol = self._validate_and_normalize_symbol(raw_symbol)

            side = order_data["side"]
            quantity = float(order_data["quantity"])
            order_type = order_data.get("orderType", "MARKET")

            if order_type == "MARKET":
                # Place market order
                response = await self.client.place_market_order(
                    symbol=self._get_market_symbol_value(symbol),
                    side=side,
                    quantity=quantity,
                    reduce_only=order_data.get("reduceOnly", False),
                )
            else:
                # Place limit order
                price = float(order_data["price"])
                response = await self.client.place_limit_order(
                    symbol=self._get_market_symbol_value(symbol),
                    side=side,
                    price=price,
                    quantity=quantity,
                    reduce_only=order_data.get("reduceOnly", False),
                )

            return {"status": "success", "order": response}

        except Exception as e:
            logger.exception("Error placing order")
            return {"status": "error", "message": str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.initialized:
            return False

        try:
            await self.client.cancel_order(order_id)
        except Exception:
            logger.exception("Error canceling order")
            return False
        else:
            return True

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if not self.initialized:
            return False

        try:
            # Validate and normalize symbol first
            validated_symbol = self._validate_and_normalize_symbol(symbol)

            await self.client.adjust_leverage(
                symbol=self._get_market_symbol_value(validated_symbol),
                leverage=leverage,
            )
        except Exception:
            logger.exception("Error setting leverage for %s", symbol)
            return False
        else:
            return True

    async def get_candlestick_data(
        self, symbol: str, interval: str, limit: int, start_time: int, end_time: int
    ) -> list[list]:
        """Get historical candlestick data with robust fallback chain."""
        logger.info(
            "🔍 Fetching candlestick data for %s, interval: %s, limit: %s",
            symbol,
            interval,
            limit,
        )

        # Validate and normalize symbol first
        try:
            validated_symbol = self._validate_and_normalize_symbol(symbol)
            if validated_symbol != symbol:
                logger.info("Symbol normalized from %s to %s", symbol, validated_symbol)
                symbol = validated_symbol
        except Exception as e:
            logger.warning(
                "Symbol validation failed for %s: %s, proceeding with original",
                symbol,
                e,
            )

        fallback_sources = [
            ("REST_API", self._get_candles_via_rest_api_with_retry),
            ("SDK", self._get_candles_via_sdk_with_retry),
        ]

        for source_name, method in fallback_sources:
            try:
                logger.info("🔄 Attempting data fetch via %s", source_name)

                # All remaining methods use async retry logic
                candles = await method(symbol, interval, limit, start_time, end_time)

                if candles and len(candles) > 0:
                    logger.info(
                        "✅ Successfully retrieved %s candles from %s",
                        len(candles),
                        source_name,
                    )
                    return candles
                logger.warning(
                    "❌ %s returned no data, trying next fallback", source_name
                )

            except Exception:
                logger.exception(
                    "❌ %s failed with error, trying next fallback", source_name
                )

        # If we reach here, all methods failed
        logger.error("🚨 ALL DATA SOURCES FAILED - no real-time data available")
        raise BluefinDataFetchError(
            "Unable to fetch real-time market data from any source"
        )

    async def _get_candles_via_rest_api_with_retry(
        self, symbol: str, interval: str, limit: int, start_time: int, end_time: int
    ) -> list[list]:
        """Get candlestick data via REST API with retry logic and exponential backoff."""
        max_retries = 3
        base_delay = 1.0  # Start with 1 second

        for attempt in range(max_retries):
            try:
                logger.info(
                    "🌐 REST API attempt %s/%s for %s", attempt + 1, max_retries, symbol
                )
                candles = await self._get_candles_via_rest_api(
                    symbol, interval, limit, start_time, end_time
                )

                if candles and len(candles) > 0:
                    logger.info(
                        "✅ REST API attempt %s succeeded with %s candles",
                        attempt + 1,
                        len(candles),
                    )
                    return candles
                logger.warning("⚠️ REST API attempt %s returned empty data", attempt + 1)

            except Exception as e:
                wait_time = base_delay * (2**attempt) + (secrets.randbits(16) / (2**16))
                # Exponential backoff with jitter

                if attempt < max_retries - 1:
                    logger.warning(
                        "⚠️ REST API attempt %s failed: %s. Retrying in %.1fs",
                        attempt + 1,
                        e,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.exception(
                        "❌ REST API attempt %s failed (final)", attempt + 1
                    )
                    # Re-raise the last exception after exhausting all retries
                    raise

        # Should not reach here, but just in case
        return []

    async def _get_candles_via_sdk_with_retry(
        self, symbol: str, interval: str, limit: int, start_time: int, end_time: int
    ) -> list[list]:
        """Get candlestick data via SDK with retry logic."""
        if not self.initialized:
            logger.warning("⚠️ SDK not initialized, skipping SDK retry attempts")
            raise BluefinSDKNotInitializedError("SDK not initialized")

        max_retries = 2  # Fewer retries for SDK as it's often a dependency issue
        base_delay = 0.5

        for attempt in range(max_retries):
            try:
                logger.info(
                    "🔧 SDK attempt %s/%s for %s", attempt + 1, max_retries, symbol
                )

                # Convert symbol to market symbol enum object (for SDK compatibility)
                market_symbol_enum = self._get_market_symbol(symbol)
                logger.debug("Using market symbol enum: %s", market_symbol_enum)

                params = {
                    "symbol": market_symbol_enum,
                    "interval": interval,
                }

                # Add limit if specified and within Bluefin's limit (max 50 for SDK)
                if limit > 0:
                    params["limit"] = min(limit, 50)

                # Add time range if specified
                if start_time > 0:
                    params["startTime"] = start_time
                if end_time > 0:
                    params["endTime"] = end_time

                logger.debug("SDK call params: %s", params)
                candles = await self.client.get_market_candle_stick_data(params)

                if candles and len(candles) > 0:
                    logger.info(
                        "✅ SDK attempt %s got %s raw candles",
                        attempt + 1,
                        len(candles),
                    )

                    # Convert to expected format: [timestamp, open, high, low, close, volume]
                    formatted_candles = []
                    for candle in candles:
                        if isinstance(candle, list) and len(candle) >= 6:
                            # Bluefin format: [openTime, openPrice, highPrice, lowPrice, closePrice, volume, closeTime, ...]
                            # Prices and volume may be in 18-decimal format, use smart conversion
                            try:
                                from bot.utils.price_conversion import (
                                    convert_candle_data,
                                )

                                formatted_candle = convert_candle_data(candle, symbol)
                                formatted_candles.append(formatted_candle)
                            except (ValueError, IndexError, TypeError) as format_error:
                                logger.warning(
                                    "⚠️ Skipping malformed candle data: %s", format_error
                                )
                                continue

                    if formatted_candles:
                        logger.info(
                            "✅ SDK attempt %s succeeded with %s formatted candles",
                            attempt + 1,
                            len(formatted_candles),
                        )
                        return formatted_candles
                    logger.warning(
                        "⚠️ SDK attempt %s - no valid candles after formatting",
                        attempt + 1,
                    )
                else:
                    logger.warning("⚠️ SDK attempt %s returned empty data", attempt + 1)

            except Exception as e:
                wait_time = base_delay * (2**attempt)

                if attempt < max_retries - 1:
                    logger.warning(
                        "⚠️ SDK attempt %s failed: %s. Retrying in %.1fs",
                        attempt + 1,
                        e,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.exception("❌ SDK attempt %s failed (final)", attempt + 1)
                    # Re-raise the last exception after exhausting all retries
                    raise

        # Should not reach here
        return []

    def _generate_mock_candles(
        self, symbol: str, interval: str, limit: int, start_time: int, end_time: int
    ) -> list[list]:
        """Generate realistic mock candlestick data as fallback."""
        import math
        import time

        logger.info(
            "🎭 Generating realistic mock candles for %s from %s to %s",
            symbol,
            start_time,
            end_time,
        )

        # Updated base prices and volatility characteristics for different symbols
        market_data = {
            "SUI-PERP": {"price": 3.50, "volatility": 0.08, "volume_base": 50000},
            "BTC-PERP": {"price": 65000.0, "volatility": 0.04, "volume_base": 100000},
            "ETH-PERP": {"price": 3200.0, "volatility": 0.05, "volume_base": 80000},
            "SOL-PERP": {"price": 180.0, "volatility": 0.07, "volume_base": 60000},
        }

        # Extract base symbol for lookup
        base_symbol = symbol.split("-")[0] + "-PERP"
        market_info = market_data.get(base_symbol, market_data["SUI-PERP"])

        base_price = market_info["price"]
        base_volatility = market_info["volatility"]
        volume_base = market_info["volume_base"]

        # Convert interval to seconds
        interval_seconds = self._interval_to_seconds(interval)

        # Calculate number of candles to generate
        if start_time > 0 and end_time > 0:
            time_range = (end_time - start_time) / 1000  # Convert to seconds
            num_candles = min(int(time_range / interval_seconds), limit)
        else:
            # Fallback to limit if time range is invalid
            num_candles = min(limit, API_LIMIT_MAX)
            current_time = int(time.time() * 1000)
            end_time = current_time
            start_time = current_time - (num_candles * interval_seconds * 1000)

        if num_candles <= 0:
            logger.warning(
                "⚠️ Invalid number of candles calculated: %s, using default 100",
                num_candles,
            )
            num_candles = 100

        candles = []
        current_price = base_price

        # Add some market trend simulation
        trend_direction = secrets.choice([-1, 0, 1])  # Bear, sideways, bull
        trend_strength = (
            0.0001 + (secrets.randbits(16) / (2**16)) * 0.0004
        )  # Subtle trend

        for i in range(num_candles):
            # Calculate timestamp for this candle
            candle_time = start_time + (
                i * interval_seconds * 1000
            )  # Back to milliseconds

            # Apply subtle trend
            trend_effect = trend_direction * trend_strength * i

            # Generate realistic OHLCV data with variable volatility
            volatility = base_volatility * (0.5 + (secrets.randbits(16) / (2**16)))
            # Variable volatility

            # Use sine wave component for more realistic price movement
            sine_component = math.sin(i / 10) * 0.002

            # Calculate price change
            change = (
                (-volatility + (secrets.randbits(16) / (2**16)) * 2 * volatility)
                + trend_effect
                + sine_component
            ) / 100

            open_price = current_price
            close_price = open_price * (1 + change)

            # Ensure high >= max(open, close) and low <= min(open, close)
            high_base = max(open_price, close_price)
            low_base = min(open_price, close_price)

            # Add realistic wicks
            wick_volatility = volatility / 200  # Smaller wick movements
            high_price = high_base * (
                1 + (secrets.randbits(16) / (2**16)) * wick_volatility
            )
            low_price = low_base * (
                1 - (secrets.randbits(16) / (2**16)) * wick_volatility
            )

            # Generate volume with realistic patterns
            volume_multiplier = 0.3 + (secrets.randbits(16) / (2**16)) * 1.7

            # Higher volume on larger price moves
            price_change_impact = abs(change) * 5
            volume_from_volatility = 1 + price_change_impact

            volume = volume_base * volume_multiplier * volume_from_volatility

            # Ensure prices are positive and sensible
            if close_price <= 0 or high_price <= 0 or low_price <= 0:
                logger.warning("⚠️ Generated invalid prices, resetting to base")
                close_price = open_price
                high_price = open_price
                low_price = open_price

            # Create candle data: [timestamp, open, high, low, close, volume]
            candle = [
                int(candle_time),
                round(float(open_price), 6),
                round(float(high_price), 6),
                round(float(low_price), 6),
                round(float(close_price), 6),
                round(float(volume), 2),
            ]

            candles.append(candle)
            current_price = close_price

        # Validate generated data
        valid_candles = []
        for candle in candles:
            timestamp, open_p, high_p, low_p, close_p, volume = candle

            # Basic OHLC validation
            if (
                high_p >= max(open_p, close_p)
                and low_p <= min(open_p, close_p)
                and volume > 0
            ):
                valid_candles.append(candle)
            else:
                logger.debug("⚠️ Skipping invalid candle: %s", candle)

        logger.info(
            "✅ Generated %s realistic mock candles for %s (trend: %s)",
            len(valid_candles),
            symbol,
            ["bearish", "sideways", "bullish"][trend_direction + 1],
        )
        return valid_candles

    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds."""
        try:
            if interval.endswith("s"):
                return int(interval[:-1])
            if interval.endswith("m"):
                return int(interval[:-1]) * 60
            if interval.endswith("h"):
                return int(interval[:-1]) * 3600
            if interval.endswith("d"):
                return int(interval[:-1]) * 86400
        except Exception:
            return 60
        else:
            # Default to 60 seconds if format is unclear
            return 60

    def _normalize_interval(self, interval: str) -> str:
        """Normalize interval to Bluefin API supported values.

        Bluefin DEX officially supports: "1m", "3m", "5m", "15m", "30m", "1h",
        "2h", "4h", "6h", "8h", "12h", "1d", "1w", "1M"

        Note: Bluefin does NOT support sub-minute intervals like 15s or 30s.
        These will be mapped to 1m with a warning about data granularity loss.
        """
        if not interval:
            return "1m"

        # Bluefin API officially supported intervals (from API documentation)
        supported_intervals = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1d",
            "1w": "1w",
            "1M": "1M",
        }

        # Check if interval is directly supported
        if interval in supported_intervals:
            return supported_intervals[interval]

        # Handle unsupported sub-minute intervals with explicit warnings
        sub_minute_intervals = {"15s", "30s", "45s"}
        if interval in sub_minute_intervals:
            logger.warning(
                "⚠️ GRANULARITY LOSS: Bluefin does not support %s intervals. Converting to 1m - this will result in lower data granularity!",
                interval,
                extra={
                    "service_id": self.service_id,
                    "requested_interval": interval,
                    "converted_interval": "1m",
                    "granularity_loss": True,
                    "impact": "Data points will be aggregated from higher frequency to 1-minute candles",
                },
            )
            return "1m"

        # Handle other unsupported intervals
        logger.warning(
            "Unsupported interval '%s' - using default 1m. Supported intervals: %s",
            interval,
            list(supported_intervals.keys()),
            extra={
                "service_id": self.service_id,
                "requested_interval": interval,
                "converted_interval": "1m",
                "supported_intervals": list(supported_intervals.keys()),
            },
        )
        return "1m"

    async def _get_candles_via_rest_api(
        self, symbol: str, interval: str, limit: int, start_time: int, end_time: int
    ) -> list[list]:
        """Get candlestick data via direct REST API call to Bluefin."""
        # Validate parameters
        if not self._validate_symbol(symbol):
            logger.error("Invalid symbol: %s", symbol)
            return []

        if not self._validate_limit(limit):
            logger.error("Invalid limit: %s", limit)
            return []

        if start_time > 0 and not self._validate_timestamp(start_time):
            logger.error("Invalid start_time: %s", start_time)
            return []

        if end_time > 0 and not self._validate_timestamp(end_time):
            logger.error("Invalid end_time: %s", end_time)
            return []

        # Convert symbol to the string format that the REST API expects
        # Strip any MARKET_SYMBOLS prefix and ensure proper format
        symbol_str = str(symbol).replace("MARKET_SYMBOLS.", "").strip()

        # For real-time price data (small limits), try base symbol first, then PERP format
        if limit <= 10:
            # Try base symbol format first (SUI, BTC, ETH) as per API docs
            if symbol_str.endswith("-PERP"):
                base_symbol = symbol_str.replace("-PERP", "")
                logger.debug(
                    "Using base symbol format for real-time data: %s", base_symbol
                )
                symbol_str = base_symbol
            else:
                logger.debug(
                    "Using existing symbol format for real-time data: %s", symbol_str
                )
        elif not symbol_str.endswith("-PERP"):
            symbol_str = f"{symbol_str}-PERP"
            logger.debug("Using PERP format for historical data: %s", symbol_str)

        async def _make_request():
            # Use the correct Bluefin REST API endpoint from centralized config
            api_url = get_rest_api_url(self.network.lower())

            logger.debug(
                "Using Bluefin REST API endpoint for %s network: %s",
                self.network,
                api_url,
                extra={
                    "service_id": self.service_id,
                    "network": self.network,
                    "api_url": api_url,
                },
            )

            # Build request parameters according to Bluefin API docs
            params = {
                "symbol": symbol_str,
                "interval": self._normalize_interval(interval),
            }

            # Add limit if specified (Bluefin supports larger limits for REST API)
            if limit > 0:
                params["limit"] = min(
                    limit, 1000
                )  # Bluefin supports up to 1000 candles

            # Only add time parameters for historical data requests (not current price)
            # Skip time params if requesting recent data (limit <= 10) to get latest available
            if limit > 10 and start_time > 0 and end_time > 0:
                # Ensure we're not requesting future data
                current_time_ms = int(time.time() * 1000)
                if start_time < current_time_ms and end_time < current_time_ms:
                    params["startTime"] = start_time
                    params["endTime"] = end_time
                else:
                    logger.warning(
                        "Skipping future time params: start=%s, end=%s, current=%s",
                        start_time,
                        end_time,
                        current_time_ms,
                    )
            else:
                logger.info(
                    "Requesting latest %s candles without time constraints", limit
                )

            url = f"{api_url}/candlestickData"
            logger.info("Making REST API call to %s with params: %s", url, params)

            # Add headers for better API compatibility
            headers = {
                "Accept": "application/json",
                "User-Agent": "AI-Trading-Bot-Bluefin/1.0",
            }

            import aiohttp

            async with (
                aiohttp.ClientSession(timeout=self.heavy_timeout) as session,
                session.get(url, params=params, headers=headers) as response,
            ):
                # Validate response size
                content_length = response.headers.get("Content-Length")
                if (
                    content_length and int(content_length) > 10 * 1024 * 1024
                ):  # 10MB limit
                    raise ValueError(f"Response too large: {content_length} bytes")

                if response.status == HTTP_OK:
                    data = await response.json()

                    # Validate response structure
                    if not isinstance(data, list):
                        logger.warning(
                            "Expected list response, got %s: %s", type(data), data
                        )
                        return []

                    if len(data) == 0:
                        logger.info(
                            "API returned empty array - no data available for requested parameters"
                        )
                        return []

                    logger.info(
                        "Successfully got %s candles from Bluefin REST API",
                        len(data),
                    )

                    # Format candles: [timestamp, open, high, low, close, volume]
                    # Convert from Bluefin's 18-decimal format
                    formatted_candles = []
                    for i, candle in enumerate(data):
                        if not isinstance(candle, list) or len(candle) < 6:
                            logger.warning(
                                "Invalid candle format at index %s: %s", i, candle
                            )
                            continue

                        try:
                            # Import price conversion utility
                            from bot.utils.price_conversion import (
                                convert_candle_data,
                            )

                            # Use smart conversion that detects 18-decimal format
                            formatted_candle = convert_candle_data(candle, symbol)
                            formatted_candles.append(formatted_candle)
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                "Error formatting candle at index %s: %s", i, e
                            )
                            continue

                    if formatted_candles:
                        logger.info(
                            "Successfully formatted %s candles from REST API",
                            len(formatted_candles),
                        )
                        return formatted_candles
                    logger.warning("No valid candles after formatting")
                    return []
                if response.status == HTTP_TOO_MANY_REQUESTS:
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    raise ClientError(f"Rate limited, retry after {retry_after}s")
                if response.status >= 500:
                    error_text = await response.text()
                    raise ClientError(f"Server error {response.status}: {error_text}")
                error_text = await response.text()
                logger.error(
                    "REST API call failed with status %s: %s",
                    response.status,
                    error_text,
                )
                return []

        try:
            # For real-time data, try multiple symbol formats if first fails
            if limit <= 10:
                # First attempt with the symbol format we determined above
                result = await self._retry_request(_make_request)

                if not result:
                    # Try alternate symbol format if no data returned
                    logger.info(
                        "No data with symbol format '%s', trying alternate format",
                        symbol_str,
                    )

                    # Create alternate symbol format
                    original_symbol_str = (
                        str(symbol).replace("MARKET_SYMBOLS.", "").strip()
                    )
                    if original_symbol_str.endswith("-PERP"):
                        # We tried base symbol, now try PERP format
                        alternate_symbol = original_symbol_str
                    else:
                        # We tried PERP, now try base format
                        alternate_symbol = original_symbol_str.replace("-PERP", "")

                    if alternate_symbol != symbol_str:
                        # Create new request function with alternate symbol
                        async def _make_alternate_request():
                            return await self._make_symbol_request(
                                alternate_symbol, interval, limit, start_time, end_time
                            )

                        logger.info(
                            "Trying alternate symbol format: %s", alternate_symbol
                        )
                        result = await self._retry_request(_make_alternate_request)

                return result
            # For historical data, just use the original request
            return await self._retry_request(_make_request)
        except Exception:
            logger.exception("Error in REST API call after retries")
            return []

    async def _make_symbol_request(
        self, symbol_str: str, interval: str, limit: int, start_time: int, end_time: int
    ):
        """Make REST API request with specific symbol format."""
        # Use the correct Bluefin REST API endpoint from centralized config
        api_url = get_rest_api_url(self.network.lower())

        # Build request parameters
        params = {
            "symbol": symbol_str,
            "interval": self._normalize_interval(interval),
        }

        # Add limit if specified
        if limit > 0:
            params["limit"] = min(limit, API_LIMIT_MAX)

        # Add time parameters for historical data only
        if limit > 10 and start_time > 0 and end_time > 0:
            current_time_ms = int(time.time() * 1000)
            if start_time < current_time_ms and end_time < current_time_ms:
                params["startTime"] = start_time
                params["endTime"] = end_time

        url = f"{api_url}/candlestickData"
        logger.info("Making REST API call to %s with params: %s", url, params)

        headers = {
            "Accept": "application/json",
            "User-Agent": "AI-Trading-Bot-Bluefin/1.0",
        }

        import aiohttp

        async with (
            aiohttp.ClientSession(timeout=self.heavy_timeout) as session,
            session.get(url, params=params, headers=headers) as response,
        ):
            if response.status == HTTP_OK:
                data = await response.json()

                if not isinstance(data, list):
                    logger.warning(
                        "Expected list response, got %s: %s", type(data), data
                    )
                    return []

                if len(data) == 0:
                    logger.info(
                        "API returned empty array - no data available for requested parameters"
                    )
                    return []

                logger.info(
                    "Successfully got %s candles from Bluefin REST API", len(data)
                )

                # Format candles
                formatted_candles = []
                for i, candle in enumerate(data):
                    if not isinstance(candle, list) or len(candle) < 6:
                        logger.warning(
                            "Invalid candle format at index %s: %s", i, candle
                        )
                        continue

                    try:
                        from bot.utils.price_conversion import convert_candle_data

                        formatted_candle = convert_candle_data(candle, symbol_str)
                        formatted_candles.append(formatted_candle)
                    except (ValueError, TypeError) as e:
                        logger.warning("Error formatting candle at index %s: %s", i, e)
                        continue

                if formatted_candles:
                    logger.info(
                        "Successfully formatted %s candles from REST API",
                        len(formatted_candles),
                    )
                    return formatted_candles
                logger.warning("No valid candles after formatting")
                return []

            logger.error(
                "API request failed with status %s: %s",
                response.status,
                await response.text(),
            )
            return []


# Global service instance
service = BluefinSDKService()

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(
    os.getenv("BLUEFIN_SERVICE_RATE_LIMIT", "100")
)  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds
rate_limit_storage = defaultdict(lambda: {"count": 0, "window_start": time.time()})


# Authentication middleware
@web.middleware
async def auth_middleware(request, handler):
    """Validate API key authentication for private endpoints only."""
    try:
        from bot.exchange.bluefin_endpoints import is_public_endpoint
    except ImportError:
        # Fallback public endpoint checker if centralized config is not available
        def is_public_endpoint(path: str) -> bool:
            public_paths = {
                "/health",
                "/market/ticker",
                "/market/candles",
                "/debug/symbols",
                "/ping",
                "/time",
            }
            return any(path.startswith(pub_path) for pub_path in public_paths)

    # Skip auth for public endpoints
    if is_public_endpoint(request.path):
        logger.debug("Skipping auth for public endpoint: %s", request.path)
        return await handler(request)

    # Get API key from environment
    expected_api_key = os.getenv("BLUEFIN_SERVICE_API_KEY")
    if not expected_api_key:
        logger.error("BLUEFIN_SERVICE_API_KEY not configured")
        return web.json_response({"error": "Service misconfigured"}, status=500)

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.warning("Missing or invalid auth header from %s", request.remote)
        return web.json_response({"error": "Unauthorized"}, status=401)

    # Extract and validate token
    provided_api_key = auth_header[7:]  # Remove "Bearer " prefix
    if not secrets.compare_digest(provided_api_key, expected_api_key):
        logger.warning("Invalid API key attempt from %s", request.remote)
        return web.json_response({"error": "Unauthorized"}, status=401)

    # API key is valid, proceed to handler
    return await handler(request)


# Rate limiting middleware
@web.middleware
async def rate_limit_middleware(request, handler):
    """Apply rate limiting per client IP."""
    # Skip rate limiting for health check
    if request.path == "/health":
        return await handler(request)

    # Get client IP
    client_ip = request.headers.get("X-Forwarded-For", request.remote)
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()

    # Check rate limit
    current_time = time.time()
    client_data = rate_limit_storage[client_ip]

    # Reset window if expired
    if current_time - client_data["window_start"] > RATE_LIMIT_WINDOW:
        client_data["count"] = 0
        client_data["window_start"] = current_time

    # Check if limit exceeded
    if client_data["count"] >= RATE_LIMIT_REQUESTS:
        remaining_time = int(
            RATE_LIMIT_WINDOW - (current_time - client_data["window_start"])
        )
        logger.warning("Rate limit exceeded for %s", client_ip)
        return web.json_response(
            {"error": "Rate limit exceeded", "retry_after": remaining_time},
            status=429,
            headers={"Retry-After": str(remaining_time)},
        )

    # Increment counter
    client_data["count"] += 1

    # Add rate limit headers to response
    response = await handler(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(
        RATE_LIMIT_REQUESTS - client_data["count"]
    )
    response.headers["X-RateLimit-Reset"] = str(
        int(client_data["window_start"] + RATE_LIMIT_WINDOW)
    )

    return response


# Endpoint validation utilities
async def validate_endpoint_connectivity(
    endpoint_url: str, timeout: int = 10
) -> tuple[bool, str]:
    """
    Validate connectivity to a Bluefin API endpoint.

    Args:
        endpoint_url: Base URL of the endpoint to test
        timeout: Timeout in seconds for the connection test

    Returns:
        Tuple of (is_accessible, error_message)
    """
    try:
        import aiohttp

        logger.debug("Testing connectivity to endpoint: %s", endpoint_url)

        # Test with a simple ping endpoint
        test_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=test_timeout) as session:
            # Try health check or ping endpoint
            test_endpoints = ["/ping", "/time", "/exchangeInfo"]

            for test_path in test_endpoints:
                try:
                    async with session.get(f"{endpoint_url}{test_path}") as response:
                        if response.status == HTTP_OK:
                            logger.debug(
                                "Endpoint %s is accessible via %s",
                                endpoint_url,
                                test_path,
                            )
                            return True, ""
                        if response.status == HTTP_NOT_FOUND:
                            # Endpoint might not support this path, try next
                            continue
                        logger.warning(
                            "Endpoint %s%s returned status %s",
                            endpoint_url,
                            test_path,
                            response.status,
                        )
                        continue
                except Exception as e:
                    logger.debug("Test path %s failed: %s", test_path, e)
                    continue

            # If no test endpoint succeeded, consider it inaccessible
            return (
                False,
                f"No test endpoints responded successfully from {endpoint_url}",
            )

    except Exception as e:
        error_msg = f"Failed to test endpoint connectivity: {e!s}"
        logger.exception(error_msg)
        return False, error_msg


async def validate_network_endpoints(network: str) -> dict[str, tuple[bool, str]]:
    """
    Validate all endpoints for a specific network.

    Args:
        network: Network name ("mainnet" or "testnet")

    Returns:
        Dictionary with endpoint validation results
    """
    try:
        endpoints = BluefinEndpointConfig.get_endpoints(network)
        results = {}

        logger.info("Validating endpoints for %s network", network)

        # Test REST API endpoint
        rest_valid, rest_error = await validate_endpoint_connectivity(
            endpoints.rest_api
        )
        results["rest_api"] = (rest_valid, rest_error)

        # Note: We don't test WebSocket endpoints here as they require different validation
        # WebSocket validation would need a different approach with WebSocket client
        results["websocket_api"] = (True, "WebSocket validation not implemented")
        results["websocket_notifications"] = (
            True,
            "WebSocket validation not implemented",
        )

        return results

    except ImportError:
        logger.warning(
            "Centralized endpoint config not available, using fallback validation"
        )
        # Fallback validation using hardcoded URLs
        fallback_url = get_rest_api_url(network)
        rest_valid, rest_error = await validate_endpoint_connectivity(fallback_url)
        return {
            "rest_api": (rest_valid, rest_error),
            "websocket_api": (True, "Fallback - no validation"),
            "websocket_notifications": (True, "Fallback - no validation"),
        }
    except Exception as e:
        error_msg = f"Endpoint validation failed: {e!s}"
        logger.exception(error_msg)
        return {
            "rest_api": (False, error_msg),
            "websocket_api": (False, error_msg),
            "websocket_notifications": (False, error_msg),
        }


# REST API Routes
async def health_check(_request):
    """Health check endpoint."""
    try:
        # Basic health status
        health_data = {
            "status": "healthy" if service.initialized else "unhealthy",
            "initialized": service.initialized,
            "network": service.network,
        }

        # Add endpoint configuration if available
        try:
            endpoints = BluefinEndpointConfig.get_endpoints(service.network.lower())
            health_data["endpoints"] = {
                "rest_api": endpoints.rest_api,
                "websocket_api": endpoints.websocket_api,
                "websocket_notifications": endpoints.websocket_notifications,
                "configuration_source": "centralized",
            }
        except ImportError:
            health_data["endpoints"] = {
                "rest_api": get_rest_api_url(service.network.lower()),
                "websocket_api": "N/A (fallback mode)",
                "websocket_notifications": "N/A (fallback mode)",
                "configuration_source": "fallback",
            }

        # Add service metadata
        health_data["service_info"] = {
            "service_id": service.service_id,
            "version": "2.0.0",  # Update as needed
            "uptime_seconds": int(
                time.time() - getattr(service, "_start_time", time.time())
            ),
        }

        return web.json_response(health_data)

    except Exception as e:
        logger.exception("Health check failed")
        return web.json_response(
            {
                "status": "unhealthy",
                "error": str(e),
                "initialized": getattr(service, "initialized", False),
                "network": getattr(service, "network", "unknown"),
            },
            status=500,
        )


async def detailed_health_check(_request):
    """Comprehensive health check with detailed component status."""
    try:
        health_data = {
            "status": "healthy" if service.initialized else "unhealthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "service_info": {
                "service_id": service.service_id,
                "version": "2.0.0",
                "network": service.network,
                "initialized": service.initialized,
                "uptime_seconds": int(
                    time.time() - getattr(service, "_start_time", time.time())
                ),
            },
            "components": {},
            "performance": {},
            "dependencies": {},
        }

        # Check BlueFin client connection
        if hasattr(service, "client") and service.client:
            try:
                # Test basic client functionality
                test_start = time.time()
                # This is a lightweight test - we don't make actual API calls
                health_data["components"]["bluefin_client"] = {
                    "status": "healthy",
                    "initialized": True,
                    "response_time_ms": (time.time() - test_start) * 1000,
                }
            except Exception as e:
                health_data["components"]["bluefin_client"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "initialized": False,
                }
        else:
            health_data["components"]["bluefin_client"] = {
                "status": "unhealthy",
                "error": "Client not initialized",
                "initialized": False,
            }

        # Check endpoint configuration
        try:
            endpoints = BluefinEndpointConfig.get_endpoints(service.network.lower())
            health_data["components"]["endpoint_config"] = {
                "status": "healthy",
                "endpoints": {
                    "rest_api": endpoints.rest_api,
                    "websocket_api": endpoints.websocket_api,
                    "websocket_notifications": endpoints.websocket_notifications,
                },
                "configuration_source": "centralized",
            }
        except ImportError:
            health_data["components"]["endpoint_config"] = {
                "status": "degraded",
                "endpoints": {
                    "rest_api": get_rest_api_url(service.network.lower()),
                    "websocket_api": "N/A (fallback mode)",
                    "websocket_notifications": "N/A (fallback mode)",
                },
                "configuration_source": "fallback",
            }

        # Performance metrics
        if hasattr(service, "_request_count"):
            health_data["performance"]["total_requests"] = service._request_count
        if hasattr(service, "_success_count"):
            health_data["performance"]["successful_requests"] = service._success_count
        if hasattr(service, "_error_count"):
            health_data["performance"]["error_count"] = service._error_count

        # Memory and CPU info (if available)
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            health_data["performance"]["memory_usage_mb"] = (
                process.memory_info().rss / 1024 / 1024
            )
            health_data["performance"]["cpu_percent"] = process.cpu_percent()
        except ImportError:
            pass

        # Rate limiting status
        health_data["performance"]["rate_limiting"] = {
            "enabled": True,
            "limit_per_minute": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW,
        }

        # Check dependencies
        dependencies = ["aiohttp", "bluefin_v2_client"]
        for dep in dependencies:
            try:
                __import__(dep)
                health_data["dependencies"][dep] = {"status": "available"}
            except ImportError as e:
                health_data["dependencies"][dep] = {
                    "status": "missing",
                    "error": str(e),
                }

        # Overall health assessment
        component_statuses = [
            comp.get("status") for comp in health_data["components"].values()
        ]
        if all(status == "healthy" for status in component_statuses):
            health_data["status"] = "healthy"
        elif any(status == "unhealthy" for status in component_statuses):
            health_data["status"] = "degraded"
        else:
            health_data["status"] = "unhealthy"

        return web.json_response(health_data)

    except Exception as e:
        logger.exception("Detailed health check failed")
        return web.json_response(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            status=500,
        )


async def connectivity_health_check(_request):
    """Check external connectivity to BlueFin services."""
    start_time = time.time()

    try:
        connectivity_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": {},
            "overall_status": "unknown",
        }

        # Get endpoints to test
        try:
            endpoints = BluefinEndpointConfig.get_endpoints(service.network.lower())
            test_endpoints = {
                "rest_api": endpoints.rest_api,
                "websocket_api": endpoints.websocket_api,
            }
        except ImportError:
            test_endpoints = {
                "rest_api": get_rest_api_url(service.network.lower()),
            }

        # Test each endpoint
        import aiohttp
        from aiohttp import ClientTimeout

        timeout = ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for endpoint_name, endpoint_url in test_endpoints.items():
                check_start = time.time()
                try:
                    # For WebSocket URLs, just test HTTP connectivity to the host
                    if endpoint_url.startswith("ws"):
                        test_url = endpoint_url.replace("wss://", "https://").replace(
                            "ws://", "http://"
                        )
                        # Remove /ws path for basic connectivity test
                        test_url = test_url.removesuffix("/ws")
                    else:
                        test_url = endpoint_url

                    async with session.get(f"{test_url}/", timeout=5) as response:
                        response_time = (time.time() - check_start) * 1000

                        connectivity_data["checks"][endpoint_name] = {
                            "status": (
                                "reachable" if response.status < 500 else "degraded"
                            ),
                            "url": endpoint_url,
                            "response_time_ms": response_time,
                            "status_code": response.status,
                        }

                except TimeoutError:
                    connectivity_data["checks"][endpoint_name] = {
                        "status": "timeout",
                        "url": endpoint_url,
                        "response_time_ms": (time.time() - check_start) * 1000,
                        "error": "Request timeout",
                    }
                except Exception as e:
                    connectivity_data["checks"][endpoint_name] = {
                        "status": "unreachable",
                        "url": endpoint_url,
                        "response_time_ms": (time.time() - check_start) * 1000,
                        "error": str(e),
                    }

        # Determine overall connectivity status
        check_results = [
            check["status"] for check in connectivity_data["checks"].values()
        ]
        if all(status in ["reachable", "degraded"] for status in check_results):
            connectivity_data["overall_status"] = "healthy"
        elif any(status in ["reachable", "degraded"] for status in check_results):
            connectivity_data["overall_status"] = "degraded"
        else:
            connectivity_data["overall_status"] = "unhealthy"

        connectivity_data["total_check_time_ms"] = (time.time() - start_time) * 1000

        return web.json_response(connectivity_data)

    except Exception as e:
        logger.exception("Connectivity health check failed")
        return web.json_response(
            {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
                "total_check_time_ms": (time.time() - start_time) * 1000,
            },
            status=500,
        )


async def performance_metrics(_request):
    """Get performance metrics for monitoring."""
    try:
        metrics_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "service": {
                "uptime_seconds": int(
                    time.time() - getattr(service, "_start_time", time.time())
                ),
                "service_id": service.service_id,
                "network": service.network,
            },
            "requests": {},
            "performance": {},
            "resources": {},
        }

        # Request metrics
        if hasattr(service, "_request_count"):
            metrics_data["requests"]["total"] = service._request_count
        if hasattr(service, "_success_count"):
            metrics_data["requests"]["successful"] = service._success_count
        if hasattr(service, "_error_count"):
            metrics_data["requests"]["errors"] = service._error_count

        # Calculate derived metrics
        total_requests = metrics_data["requests"].get("total", 0)
        successful_requests = metrics_data["requests"].get("successful", 0)
        error_requests = metrics_data["requests"].get("errors", 0)

        if total_requests > 0:
            metrics_data["requests"]["success_rate"] = (
                successful_requests / total_requests
            ) * 100
            metrics_data["requests"]["error_rate"] = (
                error_requests / total_requests
            ) * 100
        else:
            metrics_data["requests"]["success_rate"] = 0
            metrics_data["requests"]["error_rate"] = 0

        # Performance metrics
        if hasattr(service, "_avg_response_time"):
            metrics_data["performance"][
                "avg_response_time_ms"
            ] = service._avg_response_time

        # Resource metrics (if available)
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            metrics_data["resources"] = {
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "connections": len(process.connections()),
            }

            # System metrics
            metrics_data["system"] = {
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            }
        except ImportError:
            metrics_data["resources"][
                "note"
            ] = "psutil not available for detailed metrics"

        # Rate limiting metrics
        current_time = time.time()
        active_limits = 0
        for client_data in rate_limit_storage.values():
            if current_time - client_data["window_start"] < RATE_LIMIT_WINDOW:
                active_limits += 1

        metrics_data["rate_limiting"] = {
            "limit_per_minute": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW,
            "active_clients": active_limits,
        }

        return web.json_response(metrics_data)

    except Exception as e:
        logger.exception("Performance metrics failed")
        return web.json_response({"error": str(e)}, status=500)


async def service_status(_request):
    """Get comprehensive service status."""
    try:
        status_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "service": {
                "name": "Bluefin SDK Service",
                "version": "2.0.0",
                "service_id": service.service_id,
                "status": "running" if service.initialized else "initializing",
                "network": service.network,
                "uptime_seconds": int(
                    time.time() - getattr(service, "_start_time", time.time())
                ),
            },
            "health": {
                "overall": "healthy" if service.initialized else "initializing",
                "components": {},
            },
            "configuration": {},
            "statistics": {},
        }

        # Component health
        components = ["bluefin_client", "endpoint_config", "rate_limiter"]
        for component in components:
            if component == "bluefin_client":
                status_data["health"]["components"][component] = {
                    "status": (
                        "healthy"
                        if hasattr(service, "client") and service.client
                        else "unhealthy"
                    )
                }
            elif component == "endpoint_config":
                try:
                    BluefinEndpointConfig.get_endpoints(service.network.lower())
                    status_data["health"]["components"][component] = {
                        "status": "healthy"
                    }
                except ImportError:
                    status_data["health"]["components"][component] = {
                        "status": "degraded"
                    }
            elif component == "rate_limiter":
                status_data["health"]["components"][component] = {"status": "healthy"}

        # Configuration
        status_data["configuration"] = {
            "network": service.network,
            "rate_limit_enabled": True,
            "rate_limit_per_minute": RATE_LIMIT_REQUESTS,
            "authentication_required": bool(os.getenv("BLUEFIN_SERVICE_API_KEY")),
        }

        # Statistics
        status_data["statistics"] = {
            "total_requests": getattr(service, "_request_count", 0),
            "successful_requests": getattr(service, "_success_count", 0),
            "error_requests": getattr(service, "_error_count", 0),
            "active_rate_limits": len(rate_limit_storage),
        }

        return web.json_response(status_data)

    except Exception as e:
        logger.exception("Service status failed")
        return web.json_response({"error": str(e)}, status=500)


async def service_diagnostics(_request):
    """Run diagnostic checks for troubleshooting."""
    try:
        diagnostics_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "diagnostics_id": f"diag_{int(time.time())}",
            "checks": {},
            "recommendations": [],
            "overall_status": "unknown",
        }

        # Environment diagnostics
        env_check = {"status": "healthy", "details": {}}

        required_vars = ["BLUEFIN_PRIVATE_KEY", "BLUEFIN_NETWORK"]
        optional_vars = ["BLUEFIN_SERVICE_API_KEY", "HOST", "PORT"]

        for var in required_vars:
            if os.getenv(var):
                env_check["details"][var] = "set"
            else:
                env_check["details"][var] = "missing"
                env_check["status"] = "degraded"
                diagnostics_data["recommendations"].append(
                    f"Set {var} environment variable"
                )

        for var in optional_vars:
            env_check["details"][var] = "set" if os.getenv(var) else "not_set"

        diagnostics_data["checks"]["environment"] = env_check

        # Service initialization check
        init_check = {
            "status": "healthy" if service.initialized else "unhealthy",
            "details": {
                "service_initialized": service.initialized,
                "client_available": hasattr(service, "client")
                and service.client is not None,
                "network_configured": bool(service.network),
            },
        }

        if not service.initialized:
            diagnostics_data["recommendations"].append(
                "Service not properly initialized - check logs for initialization errors"
            )

        diagnostics_data["checks"]["initialization"] = init_check

        # Dependency check
        dep_check = {"status": "healthy", "details": {}}

        dependencies = [
            "aiohttp",
            "bluefin_v2_client",
            "asyncio",
            "logging",
            "os",
            "time",
        ]

        for dep in dependencies:
            try:
                __import__(dep)
                dep_check["details"][dep] = "available"
            except ImportError:
                dep_check["details"][dep] = "missing"
                dep_check["status"] = "unhealthy"
                diagnostics_data["recommendations"].append(
                    f"Install missing dependency: {dep}"
                )

        diagnostics_data["checks"]["dependencies"] = dep_check

        # Network accessibility check
        network_check = {"status": "unknown", "details": {}}

        try:
            # Test basic network connectivity
            import socket

            hostname = "google.com"
            port = 80

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((hostname, port))
            sock.close()

            if result == 0:
                network_check["status"] = "healthy"
                network_check["details"]["internet_connectivity"] = "available"
            else:
                network_check["status"] = "degraded"
                network_check["details"]["internet_connectivity"] = "limited"
                diagnostics_data["recommendations"].append(
                    "Check internet connectivity"
                )

        except Exception as e:
            network_check["status"] = "error"
            network_check["details"]["error"] = str(e)
            diagnostics_data["recommendations"].append(
                "Network connectivity test failed"
            )

        diagnostics_data["checks"]["network"] = network_check

        # Performance check
        perf_check = {"status": "healthy", "details": {}}

        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_percent = process.memory_percent()
            cpu_percent = process.cpu_percent()

            perf_check["details"]["memory_usage_percent"] = memory_percent
            perf_check["details"]["cpu_usage_percent"] = cpu_percent

            if memory_percent > 80:
                perf_check["status"] = "degraded"
                diagnostics_data["recommendations"].append("High memory usage detected")

            if cpu_percent > 80:
                perf_check["status"] = "degraded"
                diagnostics_data["recommendations"].append("High CPU usage detected")

        except ImportError:
            perf_check["details"][
                "note"
            ] = "psutil not available for performance metrics"

        diagnostics_data["checks"]["performance"] = perf_check

        # Overall status assessment
        check_statuses = [
            check["status"] for check in diagnostics_data["checks"].values()
        ]

        if all(status == "healthy" for status in check_statuses):
            diagnostics_data["overall_status"] = "healthy"
        elif any(status == "unhealthy" for status in check_statuses):
            diagnostics_data["overall_status"] = "unhealthy"
        else:
            diagnostics_data["overall_status"] = "degraded"

        # Default recommendations if everything is healthy
        if not diagnostics_data["recommendations"]:
            diagnostics_data["recommendations"] = [
                "All diagnostic checks passed - service is operating normally"
            ]

        return web.json_response(diagnostics_data)

    except Exception as e:
        logger.exception("Service diagnostics failed")
        return web.json_response(
            {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            status=500,
        )


async def get_account(_request):
    """Get account data."""
    try:
        data = await service.get_account_data()
        return web.json_response(data)
    except Exception as e:
        logger.exception("Error in get_account")
        return web.json_response({"error": str(e)}, status=500)


async def get_positions(_request):
    """Get positions."""
    try:
        positions = await service.get_positions()
        return web.json_response({"positions": positions})
    except Exception as e:
        logger.exception("Error in get_positions")
        return web.json_response({"error": str(e)}, status=500)


async def place_order(request):
    """Place an order."""
    try:
        order_data = await request.json()
        result = await service.place_order(order_data)
        return web.json_response(result)
    except Exception as e:
        logger.exception("Error in place_order")
        return web.json_response({"error": str(e)}, status=500)


async def cancel_order(request):
    """Cancel an order."""
    try:
        order_id = request.match_info["order_id"]
        success = await service.cancel_order(order_id)
        return web.json_response({"success": success})
    except Exception as e:
        logger.exception("Error in cancel_order")
        return web.json_response({"error": str(e)}, status=500)


async def get_ticker(request):
    """Get market ticker."""
    try:
        raw_symbol = request.query.get("symbol", "SUI-PERP")

        # Validate and normalize symbol first
        try:
            symbol = service._validate_and_normalize_symbol(raw_symbol)
            if symbol != raw_symbol:
                logger.info("Symbol normalized from %s to %s", raw_symbol, symbol)
        except Exception as e:
            logger.warning(
                "Symbol validation failed for %s: %s, proceeding with original",
                raw_symbol,
                e,
            )
            symbol = raw_symbol

        # Convert symbol to market symbol enum value
        market_symbol_value = service._get_market_symbol_value(symbol)
        logger.info("Converted %s to %s", symbol, market_symbol_value)

        # Try to get market data from SDK first
        try:
            # Try to get recent candles to get current price
            current_time = int(time.time() * 1000)
            start_time = current_time - (60 * 1000)  # Last minute

            candles = await service.get_candlestick_data(
                symbol, "1m", 1, start_time, current_time
            )

            if candles and len(candles) > 0:
                latest_candle = candles[-1]
                if len(latest_candle) >= 5:
                    current_price = float(
                        latest_candle[4]
                    )  # close price (already converted from 18 decimals)
                    logger.info("Got current price from candles: %s", current_price)

                    return web.json_response(
                        {
                            "symbol": symbol,
                            "price": str(current_price),
                            "bestBid": str(current_price * 0.999),  # Approximate bid
                            "bestAsk": str(current_price * 1.001),  # Approximate ask
                        }
                    )
        except Exception as candle_error:
            logger.warning("Could not get price from candles: %s", candle_error)

        # Fallback: Try orderbook approach
        try:
            orderbook_request = GetOrderbookRequest(
                symbol=market_symbol_value, limit=10
            )
            orderbook = await service.client.get_orderbook(orderbook_request)
            logger.info("Got orderbook: %s", type(orderbook))

            best_bid = orderbook["bids"][0]["price"] if orderbook.get("bids") else 0
            best_ask = orderbook["asks"][0]["price"] if orderbook.get("asks") else 0

            return web.json_response(
                {
                    "symbol": symbol,
                    "price": (
                        str((float(best_bid) + float(best_ask)) / 2)
                        if best_bid and best_ask
                        else "0"
                    ),
                    "bestBid": str(best_bid),
                    "bestAsk": str(best_ask),
                }
            )
        except Exception:
            logger.exception("Orderbook error")

        # Final fallback: REST API for ticker
        try:
            # Use centralized endpoint configuration
            api_url = get_rest_api_url(service.network.lower())

            logger.debug(
                "Using ticker API endpoint for %s network: %s",
                service.network,
                api_url,
                extra={
                    "service_id": service.service_id,
                    "network": service.network,
                    "api_url": api_url,
                    "symbol": symbol,
                },
            )

            # Get ticker via REST API
            symbol_str = str(symbol).replace("MARKET_SYMBOLS.", "")

            import aiohttp

            logger.debug("Creating HTTP session for ticker API call")
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.debug("HTTP session created for ticker call")
                try:
                    async with session.get(
                        f"{api_url}/ticker24hr", params={"symbol": symbol_str}
                    ) as response:
                        if response.status == HTTP_OK:
                            data = await response.json()
                            if data and "price" in data:
                                # Import price conversion utility
                                from bot.utils.price_conversion import (
                                    convert_from_18_decimal,
                                    log_price_conversion_stats,
                                )

                                # Use smart conversion that detects 18-decimal format
                                raw_price = data["price"]
                                price = float(
                                    convert_from_18_decimal(raw_price, symbol, "price")
                                )

                                # Log conversion stats for debugging
                                log_price_conversion_stats(
                                    raw_price, price, symbol, "ticker_price"
                                )

                                return web.json_response(
                                    {
                                        "symbol": symbol,
                                        "price": str(price),
                                        "bestBid": str(price * 0.999),
                                        "bestAsk": str(price * 1.001),
                                    }
                                )
                finally:
                    logger.debug("HTTP session closed for ticker call")
        except Exception:
            logger.exception("REST API ticker error")

        # Return default values if all methods fail
        logger.warning("All ticker methods failed, returning default")
        return web.json_response(
            {
                "symbol": symbol,
                "price": "3.50",  # Default SUI price
                "bestBid": "3.49",
                "bestAsk": "3.51",
                "error": "Unable to fetch real-time data, using default",
            }
        )

    except Exception as e:
        logger.exception("Error in get_ticker")
        return web.json_response({"error": str(e)}, status=500)


async def set_leverage(request):
    """Set leverage."""
    try:
        data = await request.json()
        success = await service.set_leverage(data["symbol"], data["leverage"])
        return web.json_response({"success": success})
    except Exception as e:
        logger.exception("Error in set_leverage")
        return web.json_response({"error": str(e)}, status=500)


async def get_market_candles(request):
    """Get historical candlestick data."""
    try:
        # Extract query parameters
        symbol = request.query.get("symbol", "SUI-PERP")
        interval = request.query.get("interval", "1m")
        limit = int(request.query.get("limit", "100"))
        start_time = int(request.query.get("startTime", "0"))
        end_time = int(request.query.get("endTime", "0"))

        logger.info(
            "Market candles request: symbol=%s, interval=%s, limit=%s, startTime=%s, endTime=%s",
            symbol,
            interval,
            limit,
            start_time,
            end_time,
        )

        # If start_time or end_time are 0, calculate them
        import time

        current_time_ms = int(time.time() * 1000)

        if end_time == 0:
            end_time = current_time_ms

        if start_time == 0:
            # Default to 24 hours ago
            start_time = end_time - (24 * 60 * 60 * 1000)

        # Get candlestick data from service
        candles = await service.get_candlestick_data(
            symbol, interval, limit, start_time, end_time
        )

        return web.json_response({"candles": candles})

    except ValueError as e:
        logger.exception("Invalid parameter in get_market_candles")
        return web.json_response({"error": f"Invalid parameter: {e}"}, status=400)
    except Exception as e:
        logger.exception("Error in get_market_candles")
        return web.json_response({"error": str(e)}, status=500)


async def debug_symbols(_request):
    """Debug endpoint to inspect available symbols."""
    try:
        # Get all attributes of MARKET_SYMBOLS
        all_attrs = dir(MARKET_SYMBOLS)
        symbol_attrs = [attr for attr in all_attrs if not attr.startswith("_")]

        # Try to get some sample values
        sample_values = {}
        for attr in symbol_attrs[:10]:  # First 10 attributes
            try:
                value = getattr(MARKET_SYMBOLS, attr)
                sample_values[attr] = str(value)
            except Exception as e:
                sample_values[attr] = f"Error: {e}"

        # Test getting market symbol
        test_symbol = None
        try:
            test_symbol = service._get_market_symbol("SUI-PERP")
        except Exception as e:
            test_symbol = f"Error: {e}"

        # Test direct attribute access
        direct_sui = None
        try:
            direct_sui = MARKET_SYMBOLS.SUI
        except Exception as e:
            direct_sui = f"Error: {e}"

        return web.json_response(
            {
                "all_attributes": all_attrs,
                "symbol_attributes": symbol_attrs,
                "sample_values": sample_values,
                "type": str(type(MARKET_SYMBOLS)),
                "test_get_market_symbol": str(test_symbol),
                "direct_sui_access": str(direct_sui),
            }
        )
    except Exception as e:
        logger.exception("Error in debug_symbols")
        return web.json_response({"error": str(e)}, status=500)


async def validate_endpoints(_request):
    """Validate endpoint connectivity and configuration."""
    try:
        logger.info("Validating endpoint configuration and connectivity")

        # Get network from query parameter or use service default
        network = _request.query.get("network", service.network.lower())

        if network not in ["mainnet", "testnet"]:
            return web.json_response(
                {
                    "error": f"Invalid network: {network}. Must be 'mainnet' or 'testnet'"
                },
                status=400,
            )

        # Validate all endpoints for the specified network
        validation_results = await validate_network_endpoints(network)

        # Determine overall status (checking if any endpoints are valid)
        rest_api_valid = validation_results.get("rest_api", (False, "Not tested"))[0]

        # Prepare response
        response_data = {
            "network": network,
            "overall_status": "healthy" if rest_api_valid else "unhealthy",
            "endpoints": {},
        }

        # Add detailed endpoint results
        for endpoint_type, (is_valid, error_msg) in validation_results.items():
            response_data["endpoints"][endpoint_type] = {
                "status": "healthy" if is_valid else "unhealthy",
                "error": error_msg if error_msg else None,
                "validated": endpoint_type == "rest_api",
            }

        # Add current endpoint URLs for reference
        try:
            endpoints = BluefinEndpointConfig.get_endpoints(network)
            response_data["endpoint_urls"] = {
                "rest_api": endpoints.rest_api,
                "websocket_api": endpoints.websocket_api,
                "websocket_notifications": endpoints.websocket_notifications,
            }
        except ImportError:
            response_data["endpoint_urls"] = {
                "rest_api": get_rest_api_url(network),
                "websocket_api": "Not available (fallback mode)",
                "websocket_notifications": "Not available (fallback mode)",
            }

        # Set appropriate HTTP status
        status_code = 200 if rest_api_valid else 503

        logger.info(
            "Endpoint validation completed for %s",
            network,
            extra={
                "network": network,
                "overall_status": response_data["overall_status"],
                "rest_api_valid": rest_api_valid,
                "validation_results": validation_results,
            },
        )

        return web.json_response(response_data, status=status_code)

    except Exception as e:
        logger.exception("Error validating endpoints")
        return web.json_response(
            {
                "error": "Endpoint validation failed",
                "details": str(e),
                "overall_status": "unhealthy",
            },
            status=500,
        )


async def startup(_app):
    """Initialize service on startup."""
    logger.info("Starting Bluefin SDK service...")
    await service.initialize()
    logger.info("Bluefin SDK service startup completed")


async def cleanup(_app):
    """Cleanup service on shutdown."""
    logger.info("Shutting down Bluefin SDK service...")
    await service.cleanup()
    logger.info("Bluefin SDK service shutdown completed")


def create_app():
    """Create the aiohttp application."""
    # Create app with middleware
    app = web.Application(middlewares=[rate_limit_middleware, auth_middleware])

    # Add routes
    # Health monitoring endpoints
    app.router.add_get("/health", health_check)
    app.router.add_get("/health/detailed", detailed_health_check)
    app.router.add_get("/health/connectivity", connectivity_health_check)
    app.router.add_get("/metrics", performance_metrics)
    app.router.add_get("/status", service_status)
    app.router.add_get("/diagnostics", service_diagnostics)

    # Business endpoints
    app.router.add_get("/account", get_account)
    app.router.add_get("/positions", get_positions)
    app.router.add_post("/orders", place_order)
    app.router.add_delete("/orders/{order_id}", cancel_order)
    app.router.add_get("/market/ticker", get_ticker)
    app.router.add_get("/market/candles", get_market_candles)
    app.router.add_post("/leverage", set_leverage)

    # Debug endpoints
    app.router.add_get("/debug/symbols", debug_symbols)
    app.router.add_get("/debug/validate-endpoints", validate_endpoints)

    # Add startup and cleanup handlers
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)

    return app


if __name__ == "__main__":
    # Get host and port from environment
    # DOCKER FIX: Default to 0.0.0.0 for Docker container networking
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))

    # Check if API key is configured
    if not os.getenv("BLUEFIN_SERVICE_API_KEY"):
        logger.warning("BLUEFIN_SERVICE_API_KEY not set - generating a random key")
        api_key = secrets.token_urlsafe(32)
        os.environ["BLUEFIN_SERVICE_API_KEY"] = api_key
        logger.info("Generated API key: %s", api_key)
        logger.info(
            "Please set BLUEFIN_SERVICE_API_KEY in your environment for production use"
        )

    # Create and run app
    app = create_app()
    logger.info("🚀 Starting Bluefin SDK service on %s:%s", host, port)
    logger.info("🔧 Docker container networking: HOST=%s, PORT=%s", host, port)
    logger.info(
        "📊 Rate limit: %s requests per %s seconds",
        RATE_LIMIT_REQUESTS,
        RATE_LIMIT_WINDOW,
    )
    logger.info("🌐 Service will be accessible at: http://%s:%s/health", host, port)
    logger.info("🔗 Docker internal URL: http://bluefin-service:8080/health")

    try:
        web.run_app(app, host=host, port=port)
    except Exception as e:
        logger.exception("❌ Failed to start Bluefin SDK service: %s", e)
        logger.exception(
            "🔍 Check if port %s is already in use or host %s is accessible", port, host
        )
        raise
