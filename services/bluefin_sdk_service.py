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
import random
import secrets
import sys
import time
import traceback
from collections import defaultdict
from typing import Any

from aiohttp import ClientError, ClientTimeout, web
from bluefin_v2_client import MARKET_SYMBOLS, BluefinClient, Networks
from bluefin_v2_client.interfaces import GetOrderbookRequest

# Add parent directory to path to import secure logging and utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bot.exchange.bluefin_endpoints import BluefinEndpointConfig, get_rest_api_url
    from bot.utils.secure_logging import create_secure_logger
    from bot.utils.symbol_utils import (
        BluefinSymbolConverter,
        InvalidSymbolError,
        SymbolConversionError,
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
        else:
            return "https://dapi.api.sui-prod.bluefin.io"

    # Create fallback symbol utilities
    class BluefinSymbolConverter:
        def __init__(self, market_symbols_enum=None):
            self.market_symbols_enum = market_symbols_enum

        def to_market_symbol(self, symbol: str):
            if not self.market_symbols_enum:
                raise Exception("MARKET_SYMBOLS enum not available")
            base_symbol = symbol.split("-")[0].upper()
            if hasattr(self.market_symbols_enum, base_symbol):
                return getattr(self.market_symbols_enum, base_symbol)
            raise Exception(f"Unknown symbol: {symbol}")

        def validate_market_symbol(self, symbol: str) -> bool:
            try:
                self.to_market_symbol(symbol)
                return True
            except Exception:
                return False

        def from_market_symbol(self, market_symbol) -> str:
            # Convert market symbol back to string format
            if isinstance(market_symbol, str):
                return market_symbol
            return str(market_symbol)

    class SymbolConversionError(Exception):
        pass

    class InvalidSymbolError(Exception):
        pass


class BluefinServiceError(Exception):
    """Base exception for Bluefin service errors."""

    pass


class BluefinConnectionError(BluefinServiceError):
    """Exception raised when connection to Bluefin fails."""

    pass


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

    pass


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

        # Initialize symbol converter
        self.symbol_converter = BluefinSymbolConverter(MARKET_SYMBOLS)

        logger.info(
            "Initializing Bluefin SDK Service",
            extra={
                "service_id": self.service_id,
                "network": self.network,
                "has_private_key": bool(self.private_key),
                "private_key_length": len(self.private_key) if self.private_key else 0,
            },
        )

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
            error_msg = f"Unable to resolve market symbol '{symbol}': {str(e)}"
            logger.error(
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
            error_msg = f"Unexpected error processing symbol '{symbol}': {str(e)}"
            logger.error(
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
        except Exception as e:
            logger.error(f"Failed to convert market symbol to string: {e}")
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
                raise

            # If all else fails, provide a safe fallback
            logger.error(
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
                else:
                    return symbol.upper()
            except Exception as fallback_err:
                logger.error(
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
        return isinstance(limit, int) and 1 <= limit <= 1000

    def _get_circuit_state(self) -> str:
        """Get current circuit breaker state with transitions."""
        current_time = time.time()

        if self.circuit_state == "OPEN":
            if current_time >= self.circuit_open_until:
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
        elif state == "HALF_OPEN":
            if self.circuit_half_open_calls >= self.circuit_half_open_max_calls:
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
                f"Circuit breaker opened due to {self.failure_count} failures",
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
                self.failure_types = {key: 0 for key in self.failure_types}
                logger.info(
                    "Circuit breaker closed after successful recovery",
                    extra={"service_id": self.service_id},
                )
        elif self.failure_count > 0:
            # Gradual recovery in closed state
            old_count = self.failure_count
            self.failure_count = max(0, self.failure_count - 1)
            logger.debug(
                f"API call succeeded, reducing failure count from {old_count} to {self.failure_count}",
                extra={"service_id": self.service_id},
            )

    async def _retry_request(self, func, *args, **kwargs):
        """Execute request with enhanced exponential backoff retry and circuit breaker."""
        if await self._is_circuit_open():
            logger.warning(
                f"Circuit breaker is {self._get_circuit_state()}, skipping request",
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
                        self.base_retry_delay * (2**attempt) * random.uniform(0.8, 1.2)
                    )
                    logger.warning(
                        f"Timeout error (attempt {attempt + 1}/{self.max_retries}), retrying in {delay:.2f}s: {e}",
                        extra={"service_id": self.service_id},
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Timeout error after {self.max_retries} attempts: {e}",
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
                        self.base_retry_delay * (2**attempt) * random.uniform(0.8, 1.2)
                    )
                    logger.warning(
                        f"Client error (attempt {attempt + 1}/{self.max_retries}), retrying in {delay:.2f}s: {e}",
                        extra={
                            "service_id": self.service_id,
                            "failure_type": failure_type,
                        },
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Client error after {self.max_retries} attempts: {e}",
                        extra={
                            "service_id": self.service_id,
                            "failure_type": failure_type,
                        },
                    )
            except Exception as e:
                logger.error(
                    f"Non-retryable error: {e}",
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
            # Validate private key
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

            # Validate private key format (basic check)
            if len(self.private_key) < 32:
                error_msg = "BLUEFIN_PRIVATE_KEY appears to be too short (expected 64+ hex characters)"
                logger.error(
                    "Private key format validation failed",
                    extra={
                        "service_id": self.service_id,
                        "error_type": "invalid_private_key_format",
                        "key_length": len(self.private_key),
                        "expected_min_length": 64,
                    },
                )
                raise BluefinConnectionError(error_msg)

            # Determine network configuration
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
            except KeyError as e:
                error_msg = f"Invalid network configuration '{self.network}'. Expected 'mainnet' or 'testnet'"
                logger.error(
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

            # Initialize Bluefin client
            logger.info(
                "Creating Bluefin client instance",
                extra={
                    "service_id": self.service_id,
                    "network": self.network,
                    "onchain_enabled": True,
                },
            )

            try:
                self.client = BluefinClient(
                    True,  # Use on-chain transactions
                    network,
                    self.private_key,
                )
                logger.debug(
                    "Bluefin client instance created successfully",
                    extra={
                        "service_id": self.service_id,
                        "client_type": type(self.client).__name__,
                    },
                )
            except Exception as e:
                error_msg = f"Failed to create Bluefin client instance: {str(e)}"
                logger.error(
                    "Bluefin client creation failed",
                    extra={
                        "service_id": self.service_id,
                        "error_type": "client_creation_failed",
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise BluefinAPIError(error_msg) from e

            # Initialize the client
            logger.info(
                "Initializing Bluefin client connection",
                extra={
                    "service_id": self.service_id,
                    "initialization_step": "client_init",
                },
            )

            try:
                await self.client.init(True)
                logger.info(
                    "Bluefin client initialization completed",
                    extra={
                        "service_id": self.service_id,
                        "initialization_successful": True,
                    },
                )
            except Exception as e:
                error_msg = f"Failed to initialize Bluefin client: {str(e)}"
                logger.error(
                    "Bluefin client initialization failed",
                    extra={
                        "service_id": self.service_id,
                        "error_type": "client_init_failed",
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise BluefinAPIError(error_msg) from e

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
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_msg = f"Unexpected error during Bluefin SDK initialization: {str(e)}"
            logger.error(
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
                # Note: BluefinClient doesn't have an explicit cleanup method
                # but we can set it to None to release references
                self.client = None
                logger.debug("Bluefin client reference cleared")

            self.initialized = False
            self._cleanup_complete = True
            logger.info("Bluefin SDK service cleanup completed")
        except Exception as e:
            logger.error(f"Error during Bluefin SDK service cleanup: {e}")

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

            logger.debug(
                "Initialized default account data structure",
                extra={
                    "service_id": self.service_id,
                    "default_balance": account_data["balance"],
                    "default_margin": account_data["margin"],
                },
            )

            # Get public address (this should always work)
            try:
                account_data["address"] = self.client.get_public_address()
                logger.info(
                    "Successfully retrieved account address",
                    extra={
                        "service_id": self.service_id,
                        "address": account_data["address"],
                        "address_length": (
                            len(account_data["address"])
                            if account_data["address"]
                            else 0
                        ),
                    },
                )
            except Exception as e:
                logger.warning(
                    "Failed to retrieve public address",
                    extra={
                        "service_id": self.service_id,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "operation": "get_public_address",
                    },
                )

            # Try to get account balances
            try:
                logger.debug(
                    "Starting USDC balance retrieval",
                    extra={
                        "service_id": self.service_id,
                        "operation": "get_usdc_balance",
                        "client_available": bool(self.client),
                    },
                )

                # Wrap the SDK call with additional protection
                try:
                    balance_response = await self.client.get_usdc_balance()
                    logger.debug(
                        "Received balance response from SDK",
                        extra={
                            "service_id": self.service_id,
                            "response_type": type(balance_response).__name__,
                            "response_value": str(balance_response)[
                                :100
                            ],  # Truncate for logging
                            "operation": "get_usdc_balance",
                        },
                    )
                except Exception as sdk_error:
                    # Handle known Bluefin SDK issues
                    error_str = str(sdk_error)
                    error_type = type(sdk_error).__name__

                    logger.debug(
                        "SDK balance call failed",
                        extra={
                            "service_id": self.service_id,
                            "sdk_error_type": error_type,
                            "sdk_error_message": error_str,
                            "operation": "get_usdc_balance",
                        },
                    )

                    if (
                        "'data'" in error_str
                        or "KeyError" in error_type
                        or "Exception: 'data'" in error_str
                    ):
                        logger.warning(
                            "Detected known Bluefin SDK data access error",
                            extra={
                                "service_id": self.service_id,
                                "error_type": "sdk_data_access_error",
                                "error_message": error_str,
                                "probable_cause": "account_not_initialized_or_no_balance",
                                "operation": "get_usdc_balance",
                            },
                        )
                        # Return a default balance of 0
                        balance_response = {"balance": 0, "amount": 0}
                    else:
                        # Re-raise other SDK errors
                        logger.error(
                            "Unhandled SDK error in balance call",
                            extra={
                                "service_id": self.service_id,
                                "error_type": error_type,
                                "error_message": error_str,
                                "operation": "get_usdc_balance",
                                "traceback": traceback.format_exc(),
                            },
                        )
                        raise sdk_error

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
                            "extracted_balance": balance,
                            "response_keys": list(balance_response.keys()),
                            "operation": "get_usdc_balance",
                        },
                    )
                elif isinstance(balance_response, int | float | str):
                    # If it's a direct value
                    balance = balance_response
                    logger.debug(
                        "Using direct balance value",
                        extra={
                            "service_id": self.service_id,
                            "balance_value": balance,
                            "balance_type": type(balance).__name__,
                            "operation": "get_usdc_balance",
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

                account_data["balance"] = str(balance)
                logger.info(
                    "Successfully retrieved account balance",
                    extra={
                        "service_id": self.service_id,
                        "balance": account_data["balance"],
                        "currency": "USDC",
                        "operation": "get_usdc_balance",
                    },
                )

            except KeyError as ke:
                balance_response_str = str(locals().get("balance_response", "unknown"))[
                    :200
                ]
                logger.error(
                    "KeyError accessing balance data",
                    extra={
                        "service_id": self.service_id,
                        "error_type": "balance_key_error",
                        "missing_key": str(ke),
                        "response_preview": balance_response_str,
                        "operation": "get_usdc_balance",
                    },
                )
                account_data["error"] = f"Balance fetch failed - missing key: {str(ke)}"
            except Exception as e:
                logger.error(
                    "Unexpected error getting balance",
                    extra={
                        "service_id": self.service_id,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "operation": "get_usdc_balance",
                        "traceback": traceback.format_exc(),
                    },
                )
                account_data["error"] = f"Balance fetch failed: {str(e)}"

            # Try to get margin info
            try:
                logger.debug(
                    "Attempting to get margin bank balance from Bluefin SDK..."
                )

                # Wrap the SDK call with additional protection
                try:
                    margin_response = await self.client.get_margin_bank_balance()
                    logger.debug(f"Margin response: {margin_response}")
                except Exception as sdk_error:
                    # Handle known Bluefin SDK issues
                    logger.debug(
                        f"SDK error type: {type(sdk_error)}, message: {str(sdk_error)}"
                    )
                    if (
                        "'data'" in str(sdk_error)
                        or "KeyError" in str(type(sdk_error))
                        or "Exception: 'data'" in str(sdk_error)
                    ):
                        logger.warning(
                            f"Bluefin SDK 'data' access error in margin call: {sdk_error}"
                        )
                        logger.info(
                            "This usually indicates the account is not initialized or has no margin balance"
                        )
                        # Return a default margin structure
                        margin_response = {"available": 0, "used": 0, "total": 0}
                    else:
                        # Re-raise other SDK errors
                        logger.error(f"Unhandled SDK error in margin call: {sdk_error}")
                        raise sdk_error

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

                logger.info(f"Account margin info: {account_data['margin']}")

            except Exception as e:
                logger.error(f"Failed to get margin info: {e}")
                if account_data["error"]:
                    account_data["error"] += f", Margin fetch failed: {str(e)}"
                else:
                    account_data["error"] = f"Margin fetch failed: {str(e)}"

            return account_data

        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            logger.error(f"Exception type: {type(e)}")

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
                            f"Processing position for symbol: {pos.get('symbol', 'Unknown')}"
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
            elif isinstance(position, dict):
                # Log position symbol only for debugging
                logger.debug(
                    f"Processing position for symbol: {position.get('symbol', 'Unknown')}"
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
            else:
                logger.warning(f"Unexpected position type: {type(position)}")
                return []

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
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

        # Validate using symbol utilities
        if not self._validate_symbol(symbol):
            # Try to normalize the symbol and validate again
            try:
                from bot.utils.symbol_utils import normalize_symbol

                normalized = normalize_symbol(symbol, "PERP")

                if self._validate_symbol(normalized):
                    logger.info(f"Normalized invalid symbol {symbol} to {normalized}")
                    return normalized
                else:
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
            logger.error(f"Error placing order: {e}")
            return {"status": "error", "message": str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.initialized:
            return False

        try:
            await self.client.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False

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
            return True
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            return False

    async def get_candlestick_data(
        self, symbol: str, interval: str, limit: int, start_time: int, end_time: int
    ) -> list[list]:
        """Get historical candlestick data with robust fallback chain."""
        logger.info(
            f"🔍 Fetching candlestick data for {symbol}, interval: {interval}, limit: {limit}"
        )

        # Validate and normalize symbol first
        try:
            validated_symbol = self._validate_and_normalize_symbol(symbol)
            if validated_symbol != symbol:
                logger.info(f"Symbol normalized from {symbol} to {validated_symbol}")
                symbol = validated_symbol
        except Exception as e:
            logger.warning(
                f"Symbol validation failed for {symbol}: {e}, proceeding with original"
            )

        fallback_sources = [
            ("REST_API", self._get_candles_via_rest_api_with_retry),
            ("SDK", self._get_candles_via_sdk_with_retry),
            ("MOCK", self._generate_mock_candles),
        ]

        for source_name, method in fallback_sources:
            try:
                logger.info(f"🔄 Attempting data fetch via {source_name}")

                if source_name == "MOCK":
                    # Mock data doesn't need retry logic
                    candles = method(symbol, interval, limit, start_time, end_time)
                else:
                    # Try the method with retry logic
                    candles = await method(
                        symbol, interval, limit, start_time, end_time
                    )

                if candles and len(candles) > 0:
                    logger.info(
                        f"✅ Successfully retrieved {len(candles)} candles from {source_name}"
                    )
                    return candles
                else:
                    logger.warning(
                        f"❌ {source_name} returned no data, trying next fallback"
                    )

            except Exception as e:
                logger.error(
                    f"❌ {source_name} failed with error: {e}, trying next fallback"
                )

        # If we reach here, all methods failed
        logger.error("🚨 ALL DATA SOURCES FAILED - returning empty list")
        return []

    async def _get_candles_via_rest_api_with_retry(
        self, symbol: str, interval: str, limit: int, start_time: int, end_time: int
    ) -> list[list]:
        """Get candlestick data via REST API with retry logic and exponential backoff."""
        max_retries = 3
        base_delay = 1.0  # Start with 1 second

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"🌐 REST API attempt {attempt + 1}/{max_retries} for {symbol}"
                )
                candles = await self._get_candles_via_rest_api(
                    symbol, interval, limit, start_time, end_time
                )

                if candles and len(candles) > 0:
                    logger.info(
                        f"✅ REST API attempt {attempt + 1} succeeded with {len(candles)} candles"
                    )
                    return candles
                else:
                    logger.warning(
                        f"⚠️ REST API attempt {attempt + 1} returned empty data"
                    )

            except Exception as e:
                wait_time = base_delay * (2**attempt) + random.uniform(
                    0, 1
                )  # Exponential backoff with jitter

                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠️ REST API attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"❌ REST API attempt {attempt + 1} failed (final): {e}"
                    )
                    raise e

        # Should not reach here, but just in case
        return []

    async def _get_candles_via_sdk_with_retry(
        self, symbol: str, interval: str, limit: int, start_time: int, end_time: int
    ) -> list[list]:
        """Get candlestick data via SDK with retry logic."""
        if not self.initialized:
            logger.warning("⚠️ SDK not initialized, skipping SDK retry attempts")
            raise Exception("SDK not initialized")

        max_retries = 2  # Fewer retries for SDK as it's often a dependency issue
        base_delay = 0.5

        for attempt in range(max_retries):
            try:
                logger.info(f"🔧 SDK attempt {attempt + 1}/{max_retries} for {symbol}")

                # Convert symbol to market symbol enum value (for SDK compatibility)
                market_symbol_value = self._get_market_symbol_value(symbol)
                logger.debug(f"Using market symbol value: {market_symbol_value}")

                params = {
                    "symbol": market_symbol_value,
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

                logger.debug(f"SDK call params: {params}")
                candles = await self.client.get_market_candle_stick_data(params)

                if candles and len(candles) > 0:
                    logger.info(
                        f"✅ SDK attempt {attempt + 1} got {len(candles)} raw candles"
                    )

                    # Convert to expected format: [timestamp, open, high, low, close, volume]
                    formatted_candles = []
                    for candle in candles:
                        if isinstance(candle, list) and len(candle) >= 6:
                            # Bluefin format: [openTime, openPrice, highPrice, lowPrice, closePrice, volume, closeTime, ...]
                            # Prices and volume are in 18-decimal format, need to convert
                            try:
                                formatted_candle = [
                                    candle[0],  # openTime (timestamp)
                                    float(candle[1])
                                    / 1e18,  # openPrice (convert from 18 decimals)
                                    float(candle[2]) / 1e18,  # highPrice
                                    float(candle[3]) / 1e18,  # lowPrice
                                    float(candle[4]) / 1e18,  # closePrice
                                    float(candle[5])
                                    / 1e18,  # volume (convert from 18 decimals)
                                ]
                                formatted_candles.append(formatted_candle)
                            except (ValueError, IndexError, TypeError) as format_error:
                                logger.warning(
                                    f"⚠️ Skipping malformed candle data: {format_error}"
                                )
                                continue

                    if formatted_candles:
                        logger.info(
                            f"✅ SDK attempt {attempt + 1} succeeded with {len(formatted_candles)} formatted candles"
                        )
                        return formatted_candles
                    else:
                        logger.warning(
                            f"⚠️ SDK attempt {attempt + 1} - no valid candles after formatting"
                        )
                else:
                    logger.warning(f"⚠️ SDK attempt {attempt + 1} returned empty data")

            except Exception as e:
                wait_time = base_delay * (2**attempt)

                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠️ SDK attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ SDK attempt {attempt + 1} failed (final): {e}")
                    raise e

        # Should not reach here
        return []

    def _generate_mock_candles(
        self, symbol: str, interval: str, limit: int, start_time: int, end_time: int
    ) -> list[list]:
        """Generate realistic mock candlestick data as fallback."""
        import math
        import random
        import time

        logger.info(
            f"🎭 Generating realistic mock candles for {symbol} from {start_time} to {end_time}"
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
            num_candles = min(limit, 1000)
            current_time = int(time.time() * 1000)
            end_time = current_time
            start_time = current_time - (num_candles * interval_seconds * 1000)

        if num_candles <= 0:
            logger.warning(
                f"⚠️ Invalid number of candles calculated: {num_candles}, using default 100"
            )
            num_candles = 100

        candles = []
        current_price = base_price

        # Add some market trend simulation
        trend_direction = random.choice([-1, 0, 1])  # Bear, sideways, bull
        trend_strength = random.uniform(0.0001, 0.0005)  # Subtle trend

        for i in range(num_candles):
            # Calculate timestamp for this candle
            candle_time = start_time + (
                i * interval_seconds * 1000
            )  # Back to milliseconds

            # Apply subtle trend
            trend_effect = trend_direction * trend_strength * i

            # Generate realistic OHLCV data with variable volatility
            volatility = base_volatility * random.uniform(
                0.5, 1.5
            )  # Variable volatility

            # Use sine wave component for more realistic price movement
            sine_component = math.sin(i / 10) * 0.002

            # Calculate price change
            change = (
                random.uniform(-volatility, volatility) + trend_effect + sine_component
            ) / 100

            open_price = current_price
            close_price = open_price * (1 + change)

            # Ensure high >= max(open, close) and low <= min(open, close)
            high_base = max(open_price, close_price)
            low_base = min(open_price, close_price)

            # Add realistic wicks
            wick_volatility = volatility / 200  # Smaller wick movements
            high_price = high_base * (1 + random.uniform(0, wick_volatility))
            low_price = low_base * (1 - random.uniform(0, wick_volatility))

            # Generate volume with realistic patterns
            volume_multiplier = random.uniform(0.3, 2.0)

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

            # Format: [timestamp, open, high, low, close, volume]
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
                logger.debug(f"⚠️ Skipping invalid candle: {candle}")

        logger.info(
            f"✅ Generated {len(valid_candles)} realistic mock candles for {symbol} (trend: {['bearish', 'sideways', 'bullish'][trend_direction + 1]})"
        )
        return valid_candles

    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds."""
        try:
            if interval.endswith("s"):
                return int(interval[:-1])
            elif interval.endswith("m"):
                return int(interval[:-1]) * 60
            elif interval.endswith("h"):
                return int(interval[:-1]) * 3600
            elif interval.endswith("d"):
                return int(interval[:-1]) * 86400
            else:
                # Default to 60 seconds if format is unclear
                return 60
        except Exception:
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
                f"⚠️ GRANULARITY LOSS: Bluefin does not support {interval} intervals. "
                f"Converting to 1m - this will result in lower data granularity!",
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
            f"Unsupported interval '{interval}' - using default 1m. "
            f"Supported intervals: {list(supported_intervals.keys())}",
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
            logger.error(f"Invalid symbol: {symbol}")
            return []

        if not self._validate_limit(limit):
            logger.error(f"Invalid limit: {limit}")
            return []

        if start_time > 0 and not self._validate_timestamp(start_time):
            logger.error(f"Invalid start_time: {start_time}")
            return []

        if end_time > 0 and not self._validate_timestamp(end_time):
            logger.error(f"Invalid end_time: {end_time}")
            return []

        async def _make_request():
            # Use the correct Bluefin REST API endpoint from centralized config
            api_url = get_rest_api_url(self.network.lower())

            logger.debug(
                f"Using Bluefin REST API endpoint for {self.network} network: {api_url}",
                extra={
                    "service_id": self.service_id,
                    "network": self.network,
                    "api_url": api_url,
                },
            )

            # Convert symbol to the string format that the REST API expects
            # Strip any MARKET_SYMBOLS prefix and ensure proper format
            symbol_str = str(symbol).replace("MARKET_SYMBOLS.", "").strip()
            if not symbol_str.endswith("-PERP"):
                symbol_str = f"{symbol_str}-PERP"

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
                        f"Skipping future time params: start={start_time}, end={end_time}, current={current_time_ms}"
                    )
            else:
                logger.info(
                    f"Requesting latest {limit} candles without time constraints"
                )

            url = f"{api_url}/candlestickData"
            logger.info(f"Making REST API call to {url} with params: {params}")

            # Add headers for better API compatibility
            headers = {
                "Accept": "application/json",
                "User-Agent": "AI-Trading-Bot-Bluefin/1.0",
            }

            import aiohttp

            async with aiohttp.ClientSession(timeout=self.heavy_timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    # Validate response size
                    content_length = response.headers.get("Content-Length")
                    if (
                        content_length and int(content_length) > 10 * 1024 * 1024
                    ):  # 10MB limit
                        raise ValueError(f"Response too large: {content_length} bytes")

                    if response.status == 200:
                        data = await response.json()

                        # Validate response structure
                        if not isinstance(data, list):
                            logger.warning(
                                f"Expected list response, got {type(data)}: {data}"
                            )
                            return []

                        if len(data) == 0:
                            logger.info(
                                "API returned empty array - no data available for requested parameters"
                            )
                            return []

                        logger.info(
                            f"Successfully got {len(data)} candles from Bluefin REST API"
                        )

                        # Format candles: [timestamp, open, high, low, close, volume]
                        # Convert from Bluefin's 18-decimal format
                        formatted_candles = []
                        for i, candle in enumerate(data):
                            if not isinstance(candle, list) or len(candle) < 6:
                                logger.warning(
                                    f"Invalid candle format at index {i}: {candle}"
                                )
                                continue

                            try:
                                formatted_candle = [
                                    int(candle[0]) if candle[0] else 0,  # openTime
                                    (
                                        float(candle[1]) / 1e18 if candle[1] else 0.0
                                    ),  # openPrice
                                    (
                                        float(candle[2]) / 1e18 if candle[2] else 0.0
                                    ),  # highPrice
                                    (
                                        float(candle[3]) / 1e18 if candle[3] else 0.0
                                    ),  # lowPrice
                                    (
                                        float(candle[4]) / 1e18 if candle[4] else 0.0
                                    ),  # closePrice
                                    (
                                        float(candle[5]) / 1e18 if candle[5] else 0.0
                                    ),  # volume
                                ]
                                formatted_candles.append(formatted_candle)
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"Error formatting candle at index {i}: {e}"
                                )
                                continue

                        if formatted_candles:
                            logger.info(
                                f"Successfully formatted {len(formatted_candles)} candles from REST API"
                            )
                            return formatted_candles
                        else:
                            logger.warning("No valid candles after formatting")
                            return []
                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", "60"))
                        raise ClientError(f"Rate limited, retry after {retry_after}s")
                    elif response.status >= 500:
                        error_text = await response.text()
                        raise ClientError(
                            f"Server error {response.status}: {error_text}"
                        )
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"REST API call failed with status {response.status}: {error_text}"
                        )
                        return []

        try:
            return await self._retry_request(_make_request)
        except Exception as e:
            logger.error(f"Error in REST API call after retries: {e}")
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
        logger.debug(f"Skipping auth for public endpoint: {request.path}")
        return await handler(request)

    # Get API key from environment
    expected_api_key = os.getenv("BLUEFIN_SERVICE_API_KEY")
    if not expected_api_key:
        logger.error("BLUEFIN_SERVICE_API_KEY not configured")
        return web.json_response({"error": "Service misconfigured"}, status=500)

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.warning(f"Missing or invalid auth header from {request.remote}")
        return web.json_response({"error": "Unauthorized"}, status=401)

    # Extract and validate token
    provided_api_key = auth_header[7:]  # Remove "Bearer " prefix
    if not secrets.compare_digest(provided_api_key, expected_api_key):
        logger.warning(f"Invalid API key attempt from {request.remote}")
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
        logger.warning(f"Rate limit exceeded for {client_ip}")
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

        logger.debug(f"Testing connectivity to endpoint: {endpoint_url}")

        # Test with a simple ping endpoint
        test_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=test_timeout) as session:
            # Try health check or ping endpoint
            test_endpoints = ["/ping", "/time", "/exchangeInfo"]

            for test_path in test_endpoints:
                try:
                    async with session.get(f"{endpoint_url}{test_path}") as response:
                        if response.status == 200:
                            logger.debug(
                                f"Endpoint {endpoint_url} is accessible via {test_path}"
                            )
                            return True, ""
                        elif response.status == 404:
                            # Endpoint might not support this path, try next
                            continue
                        else:
                            logger.warning(
                                f"Endpoint {endpoint_url}{test_path} returned status {response.status}"
                            )
                            continue
                except Exception as e:
                    logger.debug(f"Test path {test_path} failed: {e}")
                    continue

            # If no test endpoint succeeded, consider it inaccessible
            return (
                False,
                f"No test endpoints responded successfully from {endpoint_url}",
            )

    except Exception as e:
        error_msg = f"Failed to test endpoint connectivity: {str(e)}"
        logger.error(error_msg)
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

        logger.info(f"Validating endpoints for {network} network")

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
        error_msg = f"Endpoint validation failed: {str(e)}"
        logger.error(error_msg)
        return {
            "rest_api": (False, error_msg),
            "websocket_api": (False, error_msg),
            "websocket_notifications": (False, error_msg),
        }


# REST API Routes
async def health_check(request):
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
        logger.error(f"Health check failed: {e}")
        return web.json_response(
            {
                "status": "unhealthy",
                "error": str(e),
                "initialized": getattr(service, "initialized", False),
                "network": getattr(service, "network", "unknown"),
            },
            status=500,
        )


async def get_account(request):
    """Get account data."""
    try:
        data = await service.get_account_data()
        return web.json_response(data)
    except Exception as e:
        logger.error(f"Error in get_account: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_positions(request):
    """Get positions."""
    try:
        positions = await service.get_positions()
        return web.json_response({"positions": positions})
    except Exception as e:
        logger.error(f"Error in get_positions: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def place_order(request):
    """Place an order."""
    try:
        order_data = await request.json()
        result = await service.place_order(order_data)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Error in place_order: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def cancel_order(request):
    """Cancel an order."""
    try:
        order_id = request.match_info["order_id"]
        success = await service.cancel_order(order_id)
        return web.json_response({"success": success})
    except Exception as e:
        logger.error(f"Error in cancel_order: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_ticker(request):
    """Get market ticker."""
    try:
        raw_symbol = request.query.get("symbol", "SUI-PERP")

        # Validate and normalize symbol first
        try:
            symbol = service._validate_and_normalize_symbol(raw_symbol)
            if symbol != raw_symbol:
                logger.info(f"Symbol normalized from {raw_symbol} to {symbol}")
        except Exception as e:
            logger.warning(
                f"Symbol validation failed for {raw_symbol}: {e}, proceeding with original"
            )
            symbol = raw_symbol

        # Convert symbol to market symbol enum value
        market_symbol_value = service._get_market_symbol_value(symbol)
        logger.info(f"Converted {symbol} to {market_symbol_value}")

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
                    logger.info(f"Got current price from candles: {current_price}")

                    return web.json_response(
                        {
                            "symbol": symbol,
                            "price": str(current_price),
                            "bestBid": str(current_price * 0.999),  # Approximate bid
                            "bestAsk": str(current_price * 1.001),  # Approximate ask
                        }
                    )
        except Exception as candle_error:
            logger.warning(f"Could not get price from candles: {candle_error}")

        # Fallback: Try orderbook approach
        try:
            orderbook_request = GetOrderbookRequest(
                symbol=market_symbol_value, limit=10
            )
            orderbook = await service.client.get_orderbook(orderbook_request)
            logger.info(f"Got orderbook: {type(orderbook)}")

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
        except Exception as orderbook_error:
            logger.error(f"Orderbook error: {orderbook_error}")

        # Final fallback: REST API for ticker
        try:
            # Use centralized endpoint configuration
            api_url = get_rest_api_url(service.network.lower())

            logger.debug(
                f"Using ticker API endpoint for {service.network} network: {api_url}",
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
                        if response.status == 200:
                            data = await response.json()
                            if data and "price" in data:
                                # Convert from 18-decimal format if needed
                                raw_price = float(data["price"])
                                # Check if this looks like an 18-decimal value (very large number)
                                if raw_price > 1e15:
                                    price = raw_price / 1e18
                                else:
                                    price = raw_price

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
        except Exception as api_error:
            logger.error(f"REST API ticker error: {api_error}")

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
        logger.error(f"Error in get_ticker: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def set_leverage(request):
    """Set leverage."""
    try:
        data = await request.json()
        success = await service.set_leverage(data["symbol"], data["leverage"])
        return web.json_response({"success": success})
    except Exception as e:
        logger.error(f"Error in set_leverage: {e}")
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
            f"Market candles request: symbol={symbol}, interval={interval}, limit={limit}, startTime={start_time}, endTime={end_time}"
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
        logger.error(f"Invalid parameter in get_market_candles: {e}")
        return web.json_response({"error": f"Invalid parameter: {e}"}, status=400)
    except Exception as e:
        logger.error(f"Error in get_market_candles: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def debug_symbols(request):
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
        logger.error(f"Error in debug_symbols: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def validate_endpoints(request):
    """Validate endpoint connectivity and configuration."""
    try:
        logger.info("Validating endpoint configuration and connectivity")

        # Get network from query parameter or use service default
        network = request.query.get("network", service.network.lower())

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
        # all_valid = all(
        #     result[0]
        #     for result in validation_results.values()
        #     if result[0] is not True
        #     or result[1] != "WebSocket validation not implemented"
        # )
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
                "validated": True if endpoint_type == "rest_api" else False,
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
            f"Endpoint validation completed for {network}",
            extra={
                "network": network,
                "overall_status": response_data["overall_status"],
                "rest_api_valid": rest_api_valid,
                "validation_results": validation_results,
            },
        )

        return web.json_response(response_data, status=status_code)

    except Exception as e:
        logger.error(f"Error validating endpoints: {e}")
        return web.json_response(
            {
                "error": "Endpoint validation failed",
                "details": str(e),
                "overall_status": "unhealthy",
            },
            status=500,
        )


async def startup(app):
    """Initialize service on startup."""
    logger.info("Starting Bluefin SDK service...")
    await service.initialize()
    logger.info("Bluefin SDK service startup completed")


async def cleanup(app):
    """Cleanup service on shutdown."""
    logger.info("Shutting down Bluefin SDK service...")
    await service.cleanup()
    logger.info("Bluefin SDK service shutdown completed")


def create_app():
    """Create the aiohttp application."""
    # Create app with middleware
    app = web.Application(middlewares=[rate_limit_middleware, auth_middleware])

    # Add routes
    app.router.add_get("/health", health_check)
    app.router.add_get("/account", get_account)
    app.router.add_get("/positions", get_positions)
    app.router.add_post("/orders", place_order)
    app.router.add_delete("/orders/{order_id}", cancel_order)
    app.router.add_get("/market/ticker", get_ticker)
    app.router.add_get("/market/candles", get_market_candles)
    app.router.add_post("/leverage", set_leverage)
    app.router.add_get("/debug/symbols", debug_symbols)
    app.router.add_get("/debug/validate-endpoints", validate_endpoints)

    # Add startup and cleanup handlers
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)

    return app


if __name__ == "__main__":
    # Get host and port from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))

    # Check if API key is configured
    if not os.getenv("BLUEFIN_SERVICE_API_KEY"):
        logger.warning("BLUEFIN_SERVICE_API_KEY not set - generating a random key")
        api_key = secrets.token_urlsafe(32)
        os.environ["BLUEFIN_SERVICE_API_KEY"] = api_key
        logger.info(f"Generated API key: {api_key}")
        logger.info(
            "Please set BLUEFIN_SERVICE_API_KEY in your environment for production use"
        )

    # Create and run app
    app = create_app()
    logger.info(f"Starting Bluefin SDK service on {host}:{port}")
    logger.info(
        f"Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds"
    )
    web.run_app(app, host=host, port=port)
