"""Bluefin WebSocket client for real-time market data streaming."""

import asyncio
import contextlib
import json
import logging
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from bot.error_handling import exception_handler
from bot.trading_types import MarketData
from bot.utils.price_conversion import convert_from_18_decimal, is_likely_18_decimal

# Import optimized precision utilities
try:
    from bot.utils.precision_manager import (
        batch_convert_market_data,
        convert_price_optimized,
        format_price_for_display,
        get_precision_manager,
    )

    PRECISION_MANAGER_AVAILABLE = True
except ImportError:
    PRECISION_MANAGER_AVAILABLE = False

if TYPE_CHECKING:
    from websockets.client import WebSocketClientProtocol

# Import centralized endpoint configuration with fallback
try:
    from bot.exchange.bluefin_endpoints import get_notifications_url, get_websocket_url
except ImportError:
    # Fallback endpoint resolver if centralized config is not available
    def get_websocket_url(
        network: Literal["mainnet", "testnet"] | None = "mainnet",
    ) -> str:
        """Fallback WebSocket URL resolver."""
        if network and network.lower() == "testnet":
            return "wss://dapi.api.sui-staging.bluefin.io"
        return "wss://dapi.api.sui-prod.bluefin.io"

    def get_notifications_url(
        network: Literal["mainnet", "testnet"] | None = "mainnet",
    ) -> str:
        """Fallback notifications URL resolver."""
        if network and network.lower() == "testnet":
            return "wss://notifications.api.sui-staging.bluefin.io"
        return "wss://notifications.api.sui-prod.bluefin.io"


# Import authentication components
try:
    from bot.exchange.bluefin_websocket_auth import (
        BluefinWebSocketAuthenticator,
        BluefinWebSocketAuthError,
        create_websocket_authenticator,
    )

    WEBSOCKET_AUTH_AVAILABLE = True
except ImportError:
    WEBSOCKET_AUTH_AVAILABLE = False
    logger.warning(
        "Bluefin WebSocket authentication not available - private channels will not work"
    )


logger = logging.getLogger(__name__)


class BluefinWebSocketError(Exception):
    """Base exception for Bluefin WebSocket errors."""


class BluefinWebSocketConnectionError(BluefinWebSocketError):
    """Exception raised when WebSocket connection fails."""


class BluefinWebSocketClient:
    """
    WebSocket client for Bluefin real-time market data.

    Connects to Bluefin's notification WebSocket service to receive:
    - Real-time tick/trade data
    - Order book updates
    - Market status updates

    Builds OHLCV candles from incoming tick data and maintains
    a rolling buffer for indicator calculations.
    """

    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        candle_limit: int = 500,
        on_candle_update: Callable[[MarketData], None | Awaitable[None]] | None = None,
        network: str | None = None,
        use_trade_aggregation: bool = True,
        auth_token: str | None = None,
        private_key_hex: str | None = None,
        enable_private_channels: bool = False,
        on_account_update: Callable[[dict], None | Awaitable[None]] | None = None,
        on_position_update: Callable[[dict], None | Awaitable[None]] | None = None,
        on_order_update: Callable[[dict], None | Awaitable[None]] | None = None,
    ):
        """
        Initialize the Bluefin WebSocket client.

        Args:
            symbol: Trading symbol (e.g., 'SUI-PERP')
            interval: Candle interval for aggregation
            candle_limit: Maximum number of candles to maintain in buffer
            on_candle_update: Callback function for candle updates
            network: Network type ('mainnet' or 'testnet'). If None, uses environment variable
            use_trade_aggregation: Enable trade-to-candle aggregation for building 1-second candles
            auth_token: Static authentication token (deprecated - use private_key_hex instead)
            private_key_hex: ED25519 private key for authentication (enables private channels)
            enable_private_channels: Enable private channel subscriptions (requires private_key_hex)
            on_account_update: Callback for account balance updates
            on_position_update: Callback for position updates
            on_order_update: Callback for order status updates
        """
        self.symbol = symbol
        self.interval = interval
        self.candle_limit = candle_limit
        self.on_candle_update = on_candle_update
        self.use_trade_aggregation = use_trade_aggregation

        # Authentication and private channel support
        self.private_key_hex = private_key_hex
        self.enable_private_channels = enable_private_channels
        self.on_account_update = on_account_update
        self.on_position_update = on_position_update
        self.on_order_update = on_order_update
        self._auth_token = auth_token  # Static token (deprecated)

        # Initialize authenticator if private key provided
        self._authenticator: BluefinWebSocketAuthenticator | None = None
        if private_key_hex and WEBSOCKET_AUTH_AVAILABLE:
            try:
                self._authenticator = create_websocket_authenticator(private_key_hex)
                logger.info("Initialized WebSocket authenticator for private channels")
            except Exception as e:
                logger.exception("Failed to initialize WebSocket authenticator: %s", e)
                self.enable_private_channels = False
        elif enable_private_channels:
            logger.warning(
                "Private channels requested but no private key provided or auth not available"
            )
            self.enable_private_channels = False

        # Determine network and set appropriate URLs using centralized configuration
        network_value = network or os.getenv("EXCHANGE__BLUEFIN_NETWORK", "mainnet")
        self.network = network_value.lower() if network_value else "mainnet"

        # Set WebSocket URLs based on network
        if self.network == "testnet":
            self.NOTIFICATION_WS_URL = "wss://notifications.api.sui-staging.bluefin.io"
            self.DAPI_WS_URL = "wss://dapi.api.sui-staging.bluefin.io"
        else:
            # Default to mainnet for any other value
            self.NOTIFICATION_WS_URL = "wss://notifications.api.sui-prod.bluefin.io"
            self.DAPI_WS_URL = "wss://dapi.api.sui-prod.bluefin.io"

        # WebSocket connection state
        self._ws: WebSocketClientProtocol | None = None
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5  # seconds

        # Candle building state
        self._candle_buffer: deque[MarketData] = deque(maxlen=candle_limit)
        self._current_candle: MarketData | None = None
        self._tick_buffer: deque[dict[str, Any]] = deque(
            maxlen=1000
        )  # Store recent ticks

        # Trade aggregation buffer for building 1-second candles
        self._trade_buffer: deque[dict[str, Any]] = deque(maxlen=1000)
        self._last_aggregation_time: datetime = datetime.now(UTC)

        # Subscription tracking
        self._subscribed_channels: set[str] = set()
        self._subscription_id = 1
        self._pending_subscriptions: dict[int, str] = (
            {}
        )  # Track pending subscription requests

        # Tasks
        self._connection_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._candle_aggregation_task: asyncio.Task | None = None
        self._trade_aggregation_task: asyncio.Task | None = None
        self._token_refresh_task: asyncio.Task | None = None

        # Performance metrics
        self._last_message_time: datetime | None = None
        self._message_count = 0
        self._error_count = 0

        # Rate limiting for astronomical price logging
        self._astronomical_log_counter: dict[str, int] = {}
        self._log_every_n_instances = 25

        # Price continuity checking
        self._last_valid_price: Decimal | None = None
        self._price_jump_threshold = Decimal("0.20")  # 20% price jump threshold
        self._price_continuity_violations = 0
        self._max_price_violations = 5

        # Message validation tracking
        self._invalid_message_counter: dict[str, int] = {}
        self._rate_limit_invalid_messages = 10

        logger.info(
            "Initialized BluefinWebSocketClient for %s with %s candles on %s network (auth: %s, private: %s)",
            symbol,
            interval,
            self.network,
            "enabled" if self._authenticator else "disabled",
            "enabled" if self.enable_private_channels else "disabled",
            extra={
                "symbol": symbol,
                "interval": interval,
                "network": self.network,
                "notification_ws_url": self.NOTIFICATION_WS_URL,
                "dapi_ws_url": self.DAPI_WS_URL,
                "authentication_enabled": self._authenticator is not None,
                "private_channels_enabled": self.enable_private_channels,
            },
        )

    def _validate_websocket_message(self, data: dict[str, Any] | list[Any]) -> bool:
        """
        Validate WebSocket message structure before processing.

        Args:
            data: Parsed message data (dict for JSON-RPC, list for Socket.IO)

        Returns:
            True if message is valid, False otherwise
        """
        try:
            # Handle Socket.IO array format
            if isinstance(data, list):
                if len(data) == 0:
                    self._log_invalid_message(
                        "empty_socketio_array", "Socket.IO message array is empty"
                    )
                    return False

                # Validate first element is a string (event name)
                if not isinstance(data[0], str):
                    self._log_invalid_message(
                        "invalid_socketio_event",
                        f"Socket.IO event name must be string: {data[0]}",
                    )
                    return False

                return True

            # Basic structure validation for JSON-RPC
            if not isinstance(data, dict):
                self._log_invalid_message(
                    "invalid_data_type",
                    f"Message must be dict or list, got: {type(data)}",
                )
                return False

            # Check for required fields based on message type
            if "eventName" in data:
                # Event-based message validation
                event_name = data.get("eventName")
                if not isinstance(event_name, str) or not event_name.strip():
                    self._log_invalid_message(
                        "invalid_event_name", f"Invalid eventName: {event_name}"
                    )
                    return False

                # Validate data field exists for data events
                if (
                    event_name in ["TickerUpdate", "MarketDataUpdate", "RecentTrades"]
                    and "data" not in data
                ):
                    self._log_invalid_message(
                        "missing_data_field",
                        f"Missing data field for event: {event_name}",
                    )
                    return False

            elif "id" in data:
                # JSON-RPC response validation
                if not isinstance(data.get("id"), int | str):
                    self._log_invalid_message(
                        "invalid_id_field", f"Invalid ID field: {data.get('id')}"
                    )
                    return False

            # Symbol validation for market data events
            if self._contains_market_data(data):
                if not self._validate_market_data_symbol(data):
                    return False

            return True

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "message_validation_error": True,
                    "data_keys": list(data.keys()) if isinstance(data, dict) else [],
                },
                component="BluefinWebSocketClient",
                operation="validate_websocket_message",
            )
            return False

    def _contains_market_data(self, data: dict[str, Any] | list[Any]) -> bool:
        """Check if message contains market data that should be validated."""
        # Handle Socket.IO array format
        if isinstance(data, list) and len(data) > 0:
            event_name = data[0] if isinstance(data[0], str) else ""
            market_events = ["globalUpdates", "ticker", "trades", "kline", "orderbook"]
            return event_name in market_events

        # Handle JSON-RPC format
        if isinstance(data, dict):
            event_name = data.get("eventName", "")
            channel = data.get("channel", data.get("ch", ""))

            market_events = ["TickerUpdate", "MarketDataUpdate", "RecentTrades"]
            market_channels = ["trade", "ticker", "kline"]

            return (
                event_name in market_events
                or any(market_ch in str(channel) for market_ch in market_channels)
                or f"{self.symbol}@kline@{self.interval}" in str(event_name)
            )

        return False

    def _validate_market_data_symbol(self, data: dict[str, Any]) -> bool:
        """Validate that market data is for the correct symbol."""
        try:
            # Check data array for symbol matching
            data_array = data.get("data", [])
            if isinstance(data_array, list):
                for item in data_array:
                    if isinstance(item, dict) and "symbol" in item:
                        if item["symbol"] != self.symbol:
                            # Wrong symbol - silently ignore (not an error)
                            return False
            elif isinstance(data_array, dict) and "symbol" in data_array:
                if data_array["symbol"] != self.symbol:
                    return False

            return True

        except Exception:
            # If validation fails, err on the side of processing
            return True

    def _log_invalid_message(self, error_type: str, message: str) -> None:
        """Log invalid message with rate limiting."""
        if error_type not in self._invalid_message_counter:
            self._invalid_message_counter[error_type] = 0

        self._invalid_message_counter[error_type] += 1

        # Rate limit logging
        if (
            self._invalid_message_counter[error_type]
            % self._rate_limit_invalid_messages
            == 1
        ):
            logger.warning(
                "🚨 INVALID WEBSOCKET MESSAGE: %s - %s [Instance #%d]",
                error_type,
                message,
                self._invalid_message_counter[error_type],
                extra={
                    "symbol": self.symbol,
                    "error_type": error_type,
                    "instance_count": self._invalid_message_counter[error_type],
                    "rate_limit_interval": self._rate_limit_invalid_messages,
                    "invalid_message_detected": True,
                },
            )

    def _validate_price_continuity(self, price: Decimal, source: str) -> bool:
        """
        Check price continuity to detect bad data or sudden jumps.

        Args:
            price: Current price to validate
            source: Source of the price data

        Returns:
            True if price passes continuity check, False otherwise
        """
        try:
            if price <= 0:
                return False

            # First price is always valid
            if self._last_valid_price is None:
                self._last_valid_price = price
                return True

            # Calculate price change percentage
            price_change = abs(price - self._last_valid_price) / self._last_valid_price

            # Check if price jump exceeds threshold
            if price_change > self._price_jump_threshold:
                self._price_continuity_violations += 1

                # Rate limited logging for price jumps
                if self._price_continuity_violations % 5 == 1:
                    logger.warning(
                        "🚨 PRICE CONTINUITY VIOLATION: %s price jumped %.2f%% from $%s to $%s (source: %s, symbol: %s) [Instance #%d]",
                        source,
                        float(price_change * 100),
                        self._last_valid_price,
                        price,
                        source,
                        self.symbol,
                        self._price_continuity_violations,
                        extra={
                            "symbol": self.symbol,
                            "source": source,
                            "old_price": float(self._last_valid_price),
                            "new_price": float(price),
                            "price_change_percent": float(price_change * 100),
                            "threshold_percent": float(
                                self._price_jump_threshold * 100
                            ),
                            "price_continuity_violation": True,
                            "violation_count": self._price_continuity_violations,
                        },
                    )

                # If too many violations, something is seriously wrong
                if self._price_continuity_violations >= self._max_price_violations:
                    logger.error(
                        "🚨 TOO MANY PRICE CONTINUITY VIOLATIONS (%d), treating as bad data source",
                        self._price_continuity_violations,
                        extra={
                            "symbol": self.symbol,
                            "violation_count": self._price_continuity_violations,
                            "max_violations": self._max_price_violations,
                            "critical_price_data_error": True,
                        },
                    )
                    return False

                # For moderate violations, reject this price but don't update last_valid_price
                return False

            # Price is within acceptable range, update last valid price
            self._last_valid_price = price
            return True

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "price": float(price),
                    "source": source,
                    "price_continuity_check_error": True,
                },
                component="BluefinWebSocketClient",
                operation="validate_price_continuity",
            )
            # On error, err on the side of accepting the price
            return True

    def _log_astronomical_price_detection(
        self, value: str | float | Decimal, field_name: str, source: str
    ) -> None:
        """
        Log when astronomical price values are detected for debugging with rate limiting.

        Args:
            value: The raw value received
            field_name: Name of the field containing the value
            source: Source of the data (e.g., 'ticker', 'trade', 'kline')
        """
        if is_likely_18_decimal(value):
            # Create unique key for this source+field combination
            log_key = f"{source}:{field_name}"

            # Initialize counter if not exists
            if log_key not in self._astronomical_log_counter:
                self._astronomical_log_counter[log_key] = 0

            # Increment counter
            self._astronomical_log_counter[log_key] += 1

            # Only log every Nth instance to prevent spam
            if (
                self._astronomical_log_counter[log_key] % self._log_every_n_instances
                == 1
            ):
                logger.warning(
                    "🚨 ASTRONOMICAL PRICE DETECTED: %s=%s from %s (symbol: %s) - Converting from 18-decimal format [Instance #%d]",
                    field_name,
                    value,
                    source,
                    self.symbol,
                    int(self._astronomical_log_counter[log_key]),
                    extra={
                        "symbol": self.symbol,
                        "field_name": field_name,
                        "raw_value": str(value),
                        "source": source,
                        "astronomical_price_detected": True,
                        "instance_count": self._astronomical_log_counter[log_key],
                        "log_interval": self._log_every_n_instances,
                    },
                )

    async def connect(self) -> None:
        """Establish WebSocket connection and start data streaming."""
        if self._connected:
            logger.warning("Already connected to Bluefin WebSocket")
            return

        logger.info(
            "Initializing Bluefin WebSocket client for %s on %s network",
            self.symbol,
            self.network,
        )

        # Start connection task
        self._connection_task = asyncio.create_task(self._connection_handler())

        # Start candle aggregation task
        self._candle_aggregation_task = asyncio.create_task(self._candle_aggregator())

        # Start trade aggregation task if enabled
        if self.use_trade_aggregation:
            self._trade_aggregation_task = asyncio.create_task(self._trade_aggregator())

        # Start token refresh task if authentication is enabled
        if self._authenticator:
            self._token_refresh_task = asyncio.create_task(
                self._token_refresh_handler()
            )

        # Wait for initial connection with a timeout
        try:
            connected = await self._wait_for_connection(timeout=30)
            if not connected:
                logger.warning(
                    "Bluefin WebSocket connection timeout after 30s. "
                    "Service will continue attempting to connect in background."
                )
        except Exception as e:
            logger.warning(
                "Error during initial Bluefin WebSocket connection: %s. "
                "Service will continue attempting to connect in background.",
                str(e),
            )

    async def disconnect(self) -> None:
        """Disconnect from WebSocket and cleanup resources."""
        logger.info("Disconnecting from Bluefin WebSocket")

        self._connected = False

        # Cancel tasks
        tasks = [
            self._connection_task,
            self._heartbeat_task,
            self._candle_aggregation_task,
            self._trade_aggregation_task,
            self._token_refresh_task,
        ]

        for task in tasks:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Close WebSocket connection
        if self._ws:
            await self._ws.close()
            self._ws = None

        # Clear subscription state
        self._subscribed_channels.clear()
        self._pending_subscriptions.clear()

        logger.info("Disconnected from Bluefin WebSocket")

    async def _connection_handler(self) -> None:
        """Handle WebSocket connection with enhanced reconnection logic."""
        consecutive_failures = 0
        max_consecutive_failures = 5
        connection_healthy = False

        while True:
            try:
                # Check if we should give up on reconnection
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Bluefin WebSocket: %d consecutive failures, disabling connection",
                        consecutive_failures,
                    )
                    self._connected = False
                    break

                await self._connect_and_subscribe()
                self._reconnect_attempts = 0
                consecutive_failures = 0  # Reset on successful connection
                connection_healthy = True

                # Handle incoming messages
                await self._message_handler()

            except ConnectionClosed as e:
                if connection_healthy:
                    logger.warning(
                        "Bluefin WebSocket connection closed after being healthy: %s",
                        str(e),
                    )
                else:
                    logger.debug("Bluefin WebSocket connection closed: %s", str(e))

                self._connected = False
                self._ws = None
                self._subscribed_channels.clear()
                self._pending_subscriptions.clear()
                consecutive_failures += 1
                connection_healthy = False

            except (WebSocketException, OSError, ConnectionError) as e:
                logger.warning(
                    "Bluefin WebSocket network error (attempt %d/%d): %s",
                    self._reconnect_attempts + 1,
                    self._max_reconnect_attempts,
                    str(e),
                )
                self._connected = False
                self._ws = None
                self._subscribed_channels.clear()
                self._pending_subscriptions.clear()
                self._error_count += 1
                consecutive_failures += 1
                connection_healthy = False

            except Exception:
                logger.exception(
                    "Unexpected error in Bluefin WebSocket (attempt %d/%d)",
                    self._reconnect_attempts + 1,
                    self._max_reconnect_attempts,
                )
                self._connected = False
                self._ws = None
                self._subscribed_channels.clear()
                self._pending_subscriptions.clear()
                self._error_count += 1
                consecutive_failures += 1
                connection_healthy = False

            # Check if we should reconnect
            if self._reconnect_attempts >= self._max_reconnect_attempts:
                logger.error(
                    "Bluefin WebSocket: Max reconnection attempts (%d) reached, stopping",
                    self._max_reconnect_attempts,
                )
                break

            self._reconnect_attempts += 1

            # Enhanced exponential backoff with jitter
            base_delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))
            jitter = base_delay * 0.1 * (0.5 - (time.time() % 1))
            delay = min(base_delay + jitter, 60)

            # Additional delay for consecutive failures
            if consecutive_failures > 3:
                delay = min(delay * 1.5, 120)
                logger.warning(
                    "Bluefin WebSocket: Multiple consecutive failures (%s), extending delay to %.1fs",
                    consecutive_failures,
                    delay,
                )

            logger.info(
                "Bluefin WebSocket: Reconnecting in %.1fs (attempt %s/%s)",
                delay,
                self._reconnect_attempts,
                self._max_reconnect_attempts,
            )

            await asyncio.sleep(delay)

    async def _connect_and_subscribe(self) -> None:
        """Establish WebSocket connection and subscribe to channels."""
        try:
            logger.info(
                "Connecting to Bluefin WebSocket at %s (network: %s)",
                self.NOTIFICATION_WS_URL,
                self.network,
            )

            # Enhanced connection parameters for better stability
            # Note: additional_headers removed due to compatibility issues with some asyncio implementations
            self._ws = await websockets.connect(
                self.NOTIFICATION_WS_URL,
                open_timeout=30.0,  # Updated for websockets >= 15.0
                ping_interval=15,  # More frequent pings
                ping_timeout=8,  # Shorter timeout to detect issues faster
                close_timeout=5,  # Quick close timeout
                max_size=2**20,  # 1MB max message size
                compression=None,  # Disable compression for performance
            )

            self._connected = True
            logger.info(
                "Bluefin WebSocket connection established to %s",
                self.NOTIFICATION_WS_URL,
            )

            # Start heartbeat task
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._heartbeat_task

            self._heartbeat_task = asyncio.create_task(self._heartbeat_handler())

            # Subscribe to market data channels
            await self._subscribe_to_market_data()

        except TimeoutError:
            logger.exception(
                "Timeout connecting to Bluefin WebSocket at %s after 30s",
                self.NOTIFICATION_WS_URL,
            )
            raise
        except Exception as e:
            logger.exception(
                "Failed to connect to Bluefin WebSocket at %s: %s",
                self.NOTIFICATION_WS_URL,
                str(e),
            )
            raise

    async def _subscribe_to_market_data(self) -> None:
        """Subscribe to relevant market data channels using JSON-RPC format."""
        # Use proper JSON-RPC format for Bluefin WebSocket API

        # Subscribe to global updates for this symbol
        await self._subscribe_to_global_updates()

        # Subscribe to ticker updates for price data
        await self._subscribe_to_ticker()

        # Subscribe to data streams based on interval and trade aggregation settings
        await self._subscribe_to_all_streams()

        # Subscribe to private channels if authentication is enabled
        if self.enable_private_channels and self._authenticator:
            await self._subscribe_to_private_channels()

        # Channels will be tracked after subscription confirmation

    async def _subscribe_to_global_updates(self) -> None:
        """Subscribe to global updates for the symbol using Socket.IO array format."""
        global_subscription = [
            "SUBSCRIBE",
            [
                {
                    "e": "globalUpdates",
                    "p": self.symbol,  # Symbol like "SUI-PERP"
                    "t": self.auth_token,
                }
            ],
        ]
        channel_key = f"globalUpdates:{self.symbol}"
        self._subscribed_channels.add(channel_key)
        await self._send_message(global_subscription)
        logger.info(
            "Subscribing to globalUpdates for %s using Socket.IO format", self.symbol
        )

    async def _subscribe_to_ticker(self) -> None:
        """Subscribe to ticker updates for price data using Socket.IO array format."""
        ticker_subscription = [
            "SUBSCRIBE",
            [{"e": "ticker", "p": self.symbol, "t": self.auth_token}],
        ]
        channel_key = f"ticker:{self.symbol}"
        self._subscribed_channels.add(channel_key)
        await self._send_message(ticker_subscription)
        logger.info("Subscribing to ticker for %s using Socket.IO format", self.symbol)

    async def _subscribe_to_all_streams(self) -> None:
        """Subscribe to appropriate data streams based on interval and aggregation settings."""
        try:
            # Define sub-minute intervals that benefit from trade aggregation
            sub_minute_intervals = ["1s", "5s", "15s", "30s"]

            # Conditional subscription logic based on trade aggregation and interval
            if self.use_trade_aggregation and self.interval in sub_minute_intervals:
                # For sub-minute intervals with trade aggregation: subscribe to trade stream
                logger.info(
                    "Using trade stream for sub-minute interval %s with aggregation enabled",
                    self.interval,
                )
                await self._subscribe_to_trades()
            else:
                # For minute+ intervals or when aggregation disabled: use kline subscription
                logger.info(
                    "Using kline stream for interval %s (aggregation: %s)",
                    self.interval,
                    self.use_trade_aggregation,
                )
                await self._subscribe_to_klines()
                # Still subscribe to trades for enhanced price updates
                await self._subscribe_to_trades()

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "use_trade_aggregation": self.use_trade_aggregation,
                    "subscription_error": True,
                },
                component="BluefinWebSocketClient",
                operation="subscribe_to_all_streams",
            )
            # Fallback to kline subscription on error
            logger.warning(
                "Failed to subscribe to preferred streams, falling back to kline subscription"
            )
            await self._subscribe_to_klines()

    async def _subscribe_to_klines(self) -> None:
        """Subscribe to kline/candlestick data for the configured interval using Socket.IO array format."""
        try:
            kline_subscription = [
                "SUBSCRIBE",
                [
                    {
                        "e": "kline",
                        "p": self.symbol,
                        "i": self.interval,  # Format: "1m", "5m", etc.
                        "t": self.auth_token,
                    }
                ],
            ]
            channel_key = f"kline:{self.symbol}@{self.interval}"
            self._subscribed_channels.add(channel_key)
            await self._send_message(kline_subscription)
            logger.info(
                "Subscribing to kline data for %s@%s using Socket.IO format",
                self.symbol,
                self.interval,
            )
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "kline_subscription_error": True,
                },
                component="BluefinWebSocketClient",
                operation="subscribe_to_klines",
            )
            raise

    async def _subscribe_to_trades(self) -> None:
        """Subscribe to trade data for real-time price updates and aggregation using Socket.IO array format."""
        try:
            trade_subscription = [
                "SUBSCRIBE",
                [{"e": "trades", "p": self.symbol, "t": self.auth_token}],
            ]
            channel_key = f"trade:{self.symbol}"
            self._subscribed_channels.add(channel_key)
            await self._send_message(trade_subscription)
            logger.info(
                "Subscribing to trade data for %s using Socket.IO format", self.symbol
            )
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "trade_subscription_error": True,
                },
                component="BluefinWebSocketClient",
                operation="subscribe_to_trades",
            )
            raise

    async def _subscribe_to_private_channels(self) -> None:
        """Subscribe to private channels using authentication tokens."""
        if not self._authenticator:
            logger.warning("Cannot subscribe to private channels: no authenticator")
            return

        try:
            logger.info("Subscribing to private channels for authenticated user")

            # Subscribe to userUpdates channel for account, position, and order updates
            await self._subscribe_to_user_updates()

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "private_channel_subscription_error": True,
                    "has_authenticator": self._authenticator is not None,
                },
                component="BluefinWebSocketClient",
                operation="subscribe_to_private_channels",
            )
            logger.warning("Failed to subscribe to private channels: %s", e)

    async def _subscribe_to_user_updates(self) -> None:
        """Subscribe to userUpdates channel for private user data."""
        if not self._authenticator:
            return

        try:
            # Get authenticated subscription message
            subscription_msg = (
                self._authenticator.get_user_updates_subscription_message()
            )

            # Use Socket.IO array format consistent with other subscriptions
            user_updates_subscription = ["SUBSCRIBE", [subscription_msg]]

            channel_key = "userUpdates"
            self._subscribed_channels.add(channel_key)
            await self._send_message(user_updates_subscription)

            logger.info(
                "Subscribed to userUpdates channel with authentication (public key: %s)",
                self._authenticator.get_public_key()[:16] + "...",
            )

        except BluefinWebSocketAuthError as e:
            logger.exception("Authentication error subscribing to userUpdates: %s", e)
            raise
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "user_updates_subscription_error": True,
                },
                component="BluefinWebSocketClient",
                operation="subscribe_to_user_updates",
            )
            raise

    async def _message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        if not self._ws:
            raise RuntimeError("WebSocket connection is not available")

        async for message in self._ws:
            try:
                data = json.loads(message)
                self._last_message_time = datetime.now(UTC)
                self._message_count += 1

                # Log message count for debugging (without sensitive data)
                if self._message_count <= 20:
                    if isinstance(data, dict):
                        msg_type = data.get("type", data.get("eventName", "unknown"))
                    elif isinstance(data, list) and len(data) > 0:
                        msg_type = f"socketio:{data[0]}"
                    else:
                        msg_type = "unknown"
                    logger.debug(
                        "WebSocket message #%s type: %s", self._message_count, msg_type
                    )

                await self._process_message(data)

            except json.JSONDecodeError as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "message_count": self._message_count,
                        "raw_message_length": (
                            len(message) if isinstance(message, str) else 0
                        ),
                    },
                    component="BluefinWebSocketClient",
                    operation="message_handler",
                )
            except (ValueError, TypeError, KeyError) as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "message_count": self._message_count,
                        "data_processing_error": True,
                    },
                    component="BluefinWebSocketClient",
                    operation="message_handler",
                )
                self._error_count += 1
            except Exception as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "message_count": self._message_count,
                        "unexpected_message_error": True,
                    },
                    component="BluefinWebSocketClient",
                    operation="message_handler",
                )
                self._error_count += 1

    async def _process_message(self, data: dict[str, Any] | list[Any]) -> None:
        """
        Process incoming WebSocket message supporting both Socket.IO array and JSON-RPC formats.

        Args:
            data: Parsed message data (dict for JSON-RPC, list for Socket.IO)
        """
        # Handle Socket.IO array format
        if isinstance(data, list):
            # Handle subscription responses first
            if await self._handle_subscription_response(data):
                return

            # Handle Socket.IO event messages
            await self._handle_socketio_messages(data)
            return

        # Handle legacy JSON-RPC format
        if isinstance(data, dict):
            # Pre-processing validation
            if not self._validate_websocket_message(data):
                return

            # Handle subscription responses first
            if await self._handle_subscription_response(data):
                return

            # Handle error messages
            if "error" in data:
                error_data = data["error"]
                logger.error("WebSocket error: %s", error_data)

                # Check for authentication errors
                if isinstance(error_data, dict):
                    await self._handle_authentication_error(error_data)
                elif isinstance(error_data, str) and any(
                    auth_term in error_data.lower()
                    for auth_term in ["auth", "token", "unauthorized", "forbidden"]
                ):
                    await self._handle_authentication_error(
                        {"message": error_data, "code": "AUTH_ERROR"}
                    )

                return

            # Handle event-based messages
            await self._handle_event_messages(data)

    async def _handle_subscription_response(
        self, data: dict[str, Any] | list[Any]
    ) -> bool:
        """Handle subscription confirmations and errors for both JSON-RPC and Socket.IO formats.

        Returns:
            True if message was a subscription response, False otherwise
        """
        # Handle Socket.IO array format responses
        if isinstance(data, list) and len(data) >= 2:
            if data[0] == "SUBSCRIPTION_CONFIRMED":
                if isinstance(data[1], dict):
                    channel_info = data[1]
                    channel = f"{channel_info.get('e', 'unknown')}:{channel_info.get('p', 'unknown')}"
                    logger.info(
                        "Socket.IO subscription confirmed: %s",
                        channel,
                    )
                    return True
            elif data[0] == "SUBSCRIPTION_ERROR":
                if isinstance(data[1], dict):
                    error_info = data[1]
                    logger.error(
                        "Socket.IO subscription failed: %s",
                        error_info.get("message", "Unknown error"),
                    )
                    return True
            return False

        # Handle legacy JSON-RPC format responses (fallback)
        if not isinstance(data, dict) or "id" not in data:
            return False

        sub_id = data["id"]

        if "result" in data:
            # Successful subscription
            if sub_id in self._pending_subscriptions:
                channel = self._pending_subscriptions.pop(sub_id)
                self._subscribed_channels.add(channel)
                logger.info(
                    "Legacy subscription confirmed: %s (ID: %s, result: %s)",
                    channel,
                    sub_id,
                    data["result"],
                )
            else:
                logger.debug(
                    "Legacy subscription %s confirmed: %s", sub_id, data["result"]
                )
            return True

        if "error" in data:
            # Subscription error
            if sub_id in self._pending_subscriptions:
                channel = self._pending_subscriptions.pop(sub_id)
                logger.error(
                    "Legacy subscription failed: %s (ID: %s, error: %s)",
                    channel,
                    sub_id,
                    data["error"],
                )
            else:
                logger.error("Legacy subscription %s failed: %s", sub_id, data["error"])
            return True

        return False

    async def _handle_event_messages(self, data: dict[str, Any]) -> None:
        """Handle event-based WebSocket messages."""
        event_name = data.get("eventName")

        # Handle Bluefin-specific events
        if await self._handle_bluefin_events(data, event_name):
            return

        # Handle standard channel messages
        await self._handle_standard_channels(data, event_name)

    async def _handle_bluefin_events(
        self, data: dict[str, Any], event_name: str | None
    ) -> bool:
        """Handle Bluefin-specific event names.

        Returns:
            True if event was handled, False otherwise
        """
        if event_name == "TickerUpdate":
            await self._handle_bluefin_ticker_update(data)
            return True
        if event_name == "MarketDataUpdate":
            await self._handle_bluefin_market_update(data)
            return True
        if event_name == "RecentTrades":
            await self._handle_bluefin_trades(data)
            return True
        if event_name and f"{self.symbol}@kline@{self.interval}" in str(event_name):
            # Handle kline/candlestick updates
            await self._handle_bluefin_kline_update(data)
            return True

        return False

    async def _handle_socketio_messages(self, data: list[Any]) -> None:
        """Handle Socket.IO array format messages."""
        try:
            if len(data) < 2:
                logger.debug("Received incomplete Socket.IO message: %s", data)
                return

            event_name = data[0]
            event_data = data[1] if len(data) > 1 else {}

            logger.debug("Processing Socket.IO event: %s", event_name)

            # Handle different Socket.IO event types
            if event_name == "globalUpdates":
                await self._handle_socketio_global_updates(event_data)
            elif event_name == "ticker":
                await self._handle_socketio_ticker(event_data)
            elif event_name == "trades":
                await self._handle_socketio_trades(event_data)
            elif event_name == "kline":
                await self._handle_socketio_kline(event_data)
            elif event_name == "orderbook":
                await self._handle_socketio_orderbook(event_data)
            elif event_name == "userUpdates":
                await self._handle_socketio_user_updates(event_data)
            elif event_name == "pong":
                logger.debug("Received Socket.IO pong response")
            else:
                # Enhanced handling for unrecognized Socket.IO messages
                if self._message_count <= 10 or self._message_count % 100 == 0:
                    logger.info(
                        "Unhandled Socket.IO message #%d - event: %s, data_type: %s",
                        self._message_count,
                        event_name,
                        type(event_data).__name__,
                    )

                # Try to extract price data from unstructured Socket.IO messages
                if isinstance(event_data, dict):
                    await self._handle_generic_price_update(event_data)

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "message_count": self._message_count,
                    "socketio_message_error": True,
                    "event_name": data[0] if len(data) > 0 else "unknown",
                },
                component="BluefinWebSocketClient",
                operation="handle_socketio_messages",
            )

    async def _handle_socketio_global_updates(
        self, data: dict[str, Any] | list[Any]
    ) -> None:
        """Handle Socket.IO global updates."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    await self._handle_bluefin_ticker_update({"data": [item]})
        elif isinstance(data, dict):
            await self._handle_bluefin_ticker_update({"data": [data]})

    async def _handle_socketio_ticker(self, data: dict[str, Any] | list[Any]) -> None:
        """Handle Socket.IO ticker updates."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    await self._handle_ticker_update({"data": item})
        elif isinstance(data, dict):
            await self._handle_ticker_update({"data": data})

    async def _handle_socketio_trades(self, data: dict[str, Any] | list[Any]) -> None:
        """Handle Socket.IO trade updates."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    await self._handle_trade_update({"data": [item]})
        elif isinstance(data, dict):
            await self._handle_trade_update({"data": [data]})

    async def _handle_socketio_kline(self, data: dict[str, Any] | list[Any]) -> None:
        """Handle Socket.IO kline updates."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    await self._handle_bluefin_kline_update({"data": item})
        elif isinstance(data, dict):
            await self._handle_bluefin_kline_update({"data": data})

    async def _handle_socketio_orderbook(
        self, data: dict[str, Any] | list[Any]
    ) -> None:
        """Handle Socket.IO orderbook updates."""
        logger.debug("Received Socket.IO orderbook update for %s", self.symbol)

    async def _handle_socketio_user_updates(
        self, data: dict[str, Any] | list[Any]
    ) -> None:
        """Handle Socket.IO userUpdates for private account data."""
        try:
            if not self.enable_private_channels:
                logger.debug("Received userUpdates but private channels disabled")
                return

            logger.debug("Processing userUpdates event data")

            # Handle different data formats
            events_data = []
            if isinstance(data, list):
                events_data = data
            elif isinstance(data, dict):
                events_data = [data]

            for event_data in events_data:
                if not isinstance(event_data, dict):
                    continue

                event_type = event_data.get("eventType", event_data.get("type", ""))

                # Route to specific handlers based on event type
                if event_type == "AccountDataUpdate":
                    await self._handle_account_update(event_data)
                elif event_type == "PositionUpdate":
                    await self._handle_position_update(event_data)
                elif event_type == "OrderUpdate":
                    await self._handle_order_update(event_data)
                elif event_type == "UserTrade":
                    await self._handle_user_trade_update(event_data)
                elif event_type == "OrderSettlementUpdate":
                    await self._handle_order_settlement_update(event_data)
                else:
                    logger.debug(
                        "Unhandled userUpdates event type: %s, keys: %s",
                        event_type,
                        list(event_data.keys())[:10],
                    )

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "user_updates_processing_error": True,
                    "data_type": type(data).__name__,
                },
                component="BluefinWebSocketClient",
                operation="handle_socketio_user_updates",
            )

    async def _handle_account_update(self, data: dict[str, Any]) -> None:
        """Handle account balance and data updates."""
        try:
            logger.debug("Account update received: %s", list(data.keys()))

            # Call user callback if provided
            if self.on_account_update:
                try:
                    if asyncio.iscoroutinefunction(self.on_account_update):
                        await self.on_account_update(data)
                    else:
                        self.on_account_update(data)
                except Exception as e:
                    logger.exception("Error in account update callback: %s", e)

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "account_update_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_account_update",
            )

    async def _handle_position_update(self, data: dict[str, Any]) -> None:
        """Handle position updates."""
        try:
            logger.debug(
                "Position update received for %s", data.get("symbol", "unknown")
            )

            # Call user callback if provided
            if self.on_position_update:
                try:
                    if asyncio.iscoroutinefunction(self.on_position_update):
                        await self.on_position_update(data)
                    else:
                        self.on_position_update(data)
                except Exception as e:
                    logger.exception("Error in position update callback: %s", e)

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "position_update_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_position_update",
            )

    async def _handle_order_update(self, data: dict[str, Any]) -> None:
        """Handle order status updates."""
        try:
            order_id = data.get("orderId", data.get("id", "unknown"))
            order_status = data.get("status", "unknown")
            logger.debug("Order update received: %s status %s", order_id, order_status)

            # Call user callback if provided
            if self.on_order_update:
                try:
                    if asyncio.iscoroutinefunction(self.on_order_update):
                        await self.on_order_update(data)
                    else:
                        self.on_order_update(data)
                except Exception as e:
                    logger.exception("Error in order update callback: %s", e)

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "order_update_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_order_update",
            )

    async def _handle_user_trade_update(self, data: dict[str, Any]) -> None:
        """Handle user trade updates (filled orders)."""
        try:
            trade_id = data.get("tradeId", data.get("id", "unknown"))
            logger.debug("User trade update received: %s", trade_id)

            # These are user's own trades, different from market trades
            # Can be used for trade confirmations and PnL tracking

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "user_trade_update_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_user_trade_update",
            )

    async def _handle_order_settlement_update(self, data: dict[str, Any]) -> None:
        """Handle order settlement updates."""
        try:
            settlement_id = data.get("settlementId", data.get("id", "unknown"))
            logger.debug("Order settlement update received: %s", settlement_id)

            # These updates relate to on-chain settlement of orders

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "order_settlement_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_order_settlement_update",
            )

    async def _handle_standard_channels(
        self, data: dict[str, Any], event_name: str | None
    ) -> None:
        """Handle standard channel messages."""
        channel = data.get("channel", data.get("ch", ""))

        if channel == "trade" or "trade" in str(channel):
            await self._handle_trade_update(data)
        elif channel == "ticker" or "ticker" in str(channel):
            await self._handle_ticker_update(data)
        elif channel == "orderbook" or "orderbook" in str(channel):
            await self._handle_orderbook_update(data)
        else:
            # Enhanced handling for unrecognized messages
            # Log first few unhandled messages in detail for debugging
            if self._message_count <= 10 or self._message_count % 100 == 0:
                logger.info(
                    "Unhandled message #%d - event: %s, channel: %s, keys: %s",
                    self._message_count,
                    event_name,
                    channel,
                    list(data.keys())[:10],
                )

            # Try to extract price data from unstructured messages
            await self._handle_generic_price_update(data)

    async def _handle_trade_update(self, data: dict[str, Any]) -> None:
        """
        Handle trade/tick updates.

        Args:
            data: Trade message data
        """
        try:
            # Extract trade data (adjust based on actual Bluefin format)
            trades = data.get("data", data.get("trades", []))
            if isinstance(trades, dict):
                trades = [trades]

            for trade in trades:
                # Parse trade fields with price conversion
                price_str = trade.get("price", "0")
                size_str = trade.get("size", trade.get("amount", "0"))

                # Log astronomical price detection
                self._log_astronomical_price_detection(price_str, "price", "trade")
                self._log_astronomical_price_detection(size_str, "size", "trade")

                # OPTIMIZATION: Use batch conversion for better performance
                if PRECISION_MANAGER_AVAILABLE:
                    trade_data_raw = {"price": price_str, "size": size_str}
                    converted_data = batch_convert_market_data(
                        trade_data_raw, self.symbol
                    )
                    price = converted_data.get("price", Decimal(0))
                    size = converted_data.get("size", Decimal(0))
                else:
                    # Legacy conversion for compatibility
                    price = convert_from_18_decimal(
                        price_str, self.symbol, "trade_price"
                    )
                    size = convert_from_18_decimal(size_str, self.symbol, "trade_size")
                side = trade.get("side", "")
                timestamp = self._parse_timestamp(
                    trade.get("timestamp", trade.get("ts"))
                )

                # Validate price continuity before processing
                if (
                    price > 0
                    and size > 0
                    and self._validate_price_continuity(price, "trade")
                ):
                    trade_data = {
                        "price": price,
                        "size": size,
                        "side": side,
                        "timestamp": timestamp,
                        "trade_id": trade.get("id", ""),
                    }

                    # Add to tick buffer
                    self._tick_buffer.append(trade_data)

                    # Add to trade aggregation buffer if enabled
                    if self.use_trade_aggregation:
                        self._trade_buffer.append(trade_data)

                    # Update current candle (only if not using trade aggregation)
                    if not self.use_trade_aggregation:
                        await self._update_candle_with_trade(trade_data)

                    logger.debug("Trade: %s %s %s @ %s", self.symbol, side, size, price)

        except (ValueError, KeyError, TypeError) as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "trade_data_keys": (
                        list(data.keys()) if isinstance(data, dict) else []
                    ),
                    "trade_processing_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_trade_update",
            )
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "unexpected_trade_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_trade_update",
            )

    async def _handle_ticker_update(self, data: dict[str, Any]) -> None:
        """
        Handle ticker price updates.

        Args:
            data: Ticker message data
        """
        try:
            # Extract ticker data
            ticker = data.get("data", data)
            if isinstance(ticker, list) and ticker:
                ticker = ticker[0]

            # Convert ticker price from 18-decimal format if needed
            last_price_str = ticker.get("last", ticker.get("lastPrice", "0"))

            # Log astronomical price detection
            self._log_astronomical_price_detection(
                last_price_str, "lastPrice", "ticker"
            )

            last_price = convert_from_18_decimal(
                last_price_str, self.symbol, "ticker_last_price"
            )
            self._parse_timestamp(ticker.get("timestamp", ticker.get("ts")))

            # Validate price continuity before processing
            if last_price > 0 and self._validate_price_continuity(last_price, "ticker"):
                # Update current candle close price
                if self._current_candle:
                    self._current_candle = MarketData(
                        symbol=self._current_candle.symbol,
                        timestamp=self._current_candle.timestamp,
                        open=self._current_candle.open,
                        high=max(self._current_candle.high, last_price),
                        low=min(self._current_candle.low, last_price),
                        close=last_price,
                        volume=self._current_candle.volume,
                    )

                logger.debug("Ticker update: %s = %s", self.symbol, last_price)

        except (ValueError, KeyError, TypeError) as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "ticker_data_keys": (
                        list(data.keys()) if isinstance(data, dict) else []
                    ),
                    "ticker_processing_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_ticker_update",
            )
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "unexpected_ticker_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_ticker_update",
            )

    async def _handle_orderbook_update(self, _data: dict[str, Any]) -> None:
        """
        Handle orderbook updates.

        Args:
            data: Orderbook message data
        """
        # For now, just log that we received orderbook data
        # Full orderbook handling can be implemented if needed
        logger.debug("Received orderbook update for %s", self.symbol)

    async def _handle_bluefin_ticker_update(self, data: dict[str, Any]) -> None:
        """
        Handle Bluefin-specific ticker updates.

        Args:
            data: Ticker update message
        """
        try:
            tickers = data.get("data", [])

            for ticker in tickers:
                if ticker.get("symbol") == self.symbol:
                    # Extract price data (values may be in 18 decimal format)
                    price_str = ticker.get("price", "0")
                    last_price_str = ticker.get("lastPrice", "0")

                    # Log astronomical price detection
                    self._log_astronomical_price_detection(
                        price_str, "price", "bluefin_ticker"
                    )
                    self._log_astronomical_price_detection(
                        last_price_str, "lastPrice", "bluefin_ticker"
                    )

                    # Use smart conversion that detects 18-decimal format
                    price = convert_from_18_decimal(
                        price_str, self.symbol, "ticker_price"
                    )
                    last_price = convert_from_18_decimal(
                        last_price_str, self.symbol, "ticker_last_price"
                    )

                    # Use the most recent price
                    current_price = price if price > 0 else last_price

                    # Validate price continuity before processing
                    if current_price > 0 and self._validate_price_continuity(
                        current_price, "bluefin_ticker"
                    ):
                        # Create a tick from ticker data
                        trade_data = {
                            "price": current_price,
                            "size": Decimal("0.1"),  # Placeholder size
                            "side": (
                                "buy" if ticker.get("priceDirection", 0) > 0 else "sell"
                            ),
                            "timestamp": datetime.now(UTC),
                            "trade_id": f"ticker_{self._message_count}",
                        }

                        # Add to tick buffer
                        self._tick_buffer.append(trade_data)

                        # Add to trade aggregation buffer if enabled
                        if self.use_trade_aggregation:
                            self._trade_buffer.append(trade_data)

                        # Update current candle (only if not using trade aggregation)
                        if not self.use_trade_aggregation:
                            await self._update_candle_with_trade(trade_data)

                        logger.debug(
                            "Ticker update: %s = $%s", self.symbol, current_price
                        )

                    break

        except (ValueError, KeyError, TypeError, ArithmeticError) as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "ticker_count": (
                        len(data.get("data", [])) if isinstance(data, dict) else 0
                    ),
                    "bluefin_ticker_processing_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_bluefin_ticker_update",
            )
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "unexpected_bluefin_ticker_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_bluefin_ticker_update",
            )

    async def _handle_bluefin_market_update(self, data: dict[str, Any]) -> None:
        """
        Handle Bluefin market data updates.

        Args:
            data: Market update message
        """
        # Similar to ticker but with more detailed market data
        await self._handle_bluefin_ticker_update(data)

    async def _handle_bluefin_trades(self, data: dict[str, Any]) -> None:
        """
        Handle Bluefin recent trades.

        Args:
            data: Trades message
        """
        try:
            trades = data.get("data", [])

            for trade in trades:
                if trade.get("symbol") == self.symbol:
                    # Extract trade data
                    price_str = trade.get("price", "0")
                    size_str = trade.get("quantity", trade.get("size", "0"))

                    # Log astronomical price detection
                    self._log_astronomical_price_detection(
                        price_str, "price", "bluefin_trades"
                    )
                    self._log_astronomical_price_detection(
                        size_str, "quantity", "bluefin_trades"
                    )

                    # Use smart conversion that detects 18-decimal format
                    price = convert_from_18_decimal(
                        price_str, self.symbol, "trade_price"
                    )
                    size = convert_from_18_decimal(size_str, self.symbol, "trade_size")

                    # Validate price continuity before processing
                    if (
                        price > 0
                        and size > 0
                        and self._validate_price_continuity(price, "bluefin_trades")
                    ):
                        trade_data = {
                            "price": price,
                            "size": size,
                            "side": trade.get("side", "").lower(),
                            "timestamp": self._parse_timestamp(trade.get("timestamp")),
                            "trade_id": trade.get("id", ""),
                        }

                        # Add to tick buffer
                        self._tick_buffer.append(trade_data)

                        # Add to trade aggregation buffer if enabled
                        if self.use_trade_aggregation:
                            self._trade_buffer.append(trade_data)

                        # Update current candle (only if not using trade aggregation)
                        if not self.use_trade_aggregation:
                            await self._update_candle_with_trade(trade_data)

                        logger.debug(
                            "Trade: %s %s %s @ $%s",
                            self.symbol,
                            trade_data["side"],
                            size,
                            price,
                        )

        except (ValueError, KeyError, TypeError, ArithmeticError) as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "trades_count": (
                        len(data.get("data", [])) if isinstance(data, dict) else 0
                    ),
                    "bluefin_trades_processing_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_bluefin_trades",
            )
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "unexpected_bluefin_trades_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_bluefin_trades",
            )

    async def _handle_bluefin_kline_update(self, data: dict[str, Any]) -> None:
        """
        Handle Bluefin kline/candlestick updates.

        Args:
            data: Kline update message
        """
        try:
            kline_data = data.get("data", {})

            if isinstance(kline_data, dict):
                # Extract kline/candlestick data
                # Format expected: openTime, openPrice, highPrice, lowPrice, closePrice, volume
                open_time = kline_data.get("openTime")
                open_price = kline_data.get("openPrice", "0")
                high_price = kline_data.get("highPrice", "0")
                low_price = kline_data.get("lowPrice", "0")
                close_price = kline_data.get("closePrice", "0")
                volume = kline_data.get("volume", "0")

                # Log astronomical price detection for all OHLCV values
                self._log_astronomical_price_detection(open_price, "openPrice", "kline")
                self._log_astronomical_price_detection(high_price, "highPrice", "kline")
                self._log_astronomical_price_detection(low_price, "lowPrice", "kline")
                self._log_astronomical_price_detection(
                    close_price, "closePrice", "kline"
                )
                self._log_astronomical_price_detection(volume, "volume", "kline")

                # OPTIMIZATION: Use batch conversion for better performance
                try:
                    if PRECISION_MANAGER_AVAILABLE:
                        ohlcv_data = {
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": volume,
                        }
                        converted_data = batch_convert_market_data(
                            ohlcv_data, self.symbol
                        )
                        open_val = converted_data.get("open", Decimal(0))
                        high_val = converted_data.get("high", Decimal(0))
                        low_val = converted_data.get("low", Decimal(0))
                        close_val = converted_data.get("close", Decimal(0))
                        volume_val = converted_data.get("volume", Decimal(0))
                    else:
                        # Legacy conversion for compatibility
                        open_val = convert_from_18_decimal(
                            open_price, self.symbol, "kline_open"
                        )
                        high_val = convert_from_18_decimal(
                            high_price, self.symbol, "kline_high"
                        )
                        low_val = convert_from_18_decimal(
                            low_price, self.symbol, "kline_low"
                        )
                        close_val = convert_from_18_decimal(
                            close_price, self.symbol, "kline_close"
                        )
                        volume_val = convert_from_18_decimal(
                            volume, self.symbol, "kline_volume"
                        )
                except (ValueError, TypeError) as e:
                    exception_handler.log_exception_with_context(
                        e,
                        {
                            "symbol": self.symbol,
                            "kline_data": str(kline_data),
                            "price_conversion_error": True,
                        },
                        component="BluefinWebSocketClient",
                        operation="handle_bluefin_kline_update",
                    )
                    return

                # Validate price continuity for kline data (use close price as representative)
                if (
                    open_val > 0
                    and close_val > 0
                    and self._validate_price_continuity(close_val, "kline")
                ):
                    # Create MarketData object directly from kline
                    timestamp = (
                        datetime.fromtimestamp(open_time / 1000, UTC)
                        if open_time
                        else datetime.now(UTC)
                    )

                    market_data = MarketData(
                        symbol=self.symbol,
                        timestamp=timestamp,
                        open=open_val,
                        high=high_val,
                        low=low_val,
                        close=close_val,
                        volume=volume_val,
                    )

                    # Update current candle directly
                    self._current_candle = market_data

                    # Add to buffer
                    self._candle_buffer.append(market_data)

                    # Notify callback
                    if self.on_candle_update:
                        try:
                            if asyncio.iscoroutinefunction(self.on_candle_update):
                                await self.on_candle_update(market_data)
                            else:
                                self.on_candle_update(market_data)
                        except Exception as e:
                            exception_handler.log_exception_with_context(
                                e,
                                {
                                    "symbol": self.symbol,
                                    "callback_error": True,
                                    "has_async_callback": asyncio.iscoroutinefunction(
                                        self.on_candle_update
                                    ),
                                },
                                component="BluefinWebSocketClient",
                                operation="handle_bluefin_kline_update",
                            )

                    logger.debug(
                        "Kline update: %s $%s volume: %s",
                        self.symbol,
                        close_val,
                        volume_val,
                    )

        except (ValueError, KeyError, TypeError) as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "kline_data_type": (
                        type(data.get("data")) if isinstance(data, dict) else None
                    ),
                    "kline_processing_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_bluefin_kline_update",
            )
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "unexpected_kline_error": True,
                },
                component="BluefinWebSocketClient",
                operation="handle_bluefin_kline_update",
            )

    async def _update_candle_with_trade(self, trade: dict[str, Any]) -> None:
        """
        Update current candle with trade data.

        Args:
            trade: Trade data dictionary
        """
        price = trade["price"]
        size = trade["size"]
        timestamp = trade["timestamp"]

        # Check if we need to create a new candle
        if self._current_candle is None or self._should_create_new_candle(timestamp):
            await self._create_new_candle(timestamp, price)

        # Update current candle
        if self._current_candle is not None:
            self._current_candle = MarketData(
                symbol=self._current_candle.symbol,
                timestamp=self._current_candle.timestamp,
                open=self._current_candle.open,
                high=max(self._current_candle.high, price),
                low=min(self._current_candle.low, price),
                close=price,
                volume=self._current_candle.volume + size,
            )

    async def _create_new_candle(self, timestamp: datetime, price: Decimal) -> None:
        """
        Create a new candle and add the previous one to buffer.

        Args:
            timestamp: Candle timestamp
            price: Opening price
        """
        # Add previous candle to buffer
        if self._current_candle:
            self._candle_buffer.append(self._current_candle)

            # Notify callback
            if self.on_candle_update:
                try:
                    if asyncio.iscoroutinefunction(self.on_candle_update):
                        await self.on_candle_update(self._current_candle)
                    else:
                        self.on_candle_update(self._current_candle)
                except Exception as e:
                    exception_handler.log_exception_with_context(
                        e,
                        {
                            "symbol": self.symbol,
                            "callback_error": True,
                            "has_async_callback": asyncio.iscoroutinefunction(
                                self.on_candle_update
                            ),
                        },
                        component="BluefinWebSocketClient",
                        operation="create_new_candle",
                    )

        # Create new candle
        candle_timestamp = self._get_candle_timestamp(timestamp)
        self._current_candle = MarketData(
            symbol=self.symbol,
            timestamp=candle_timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=Decimal(0),
        )

        logger.debug("Created new %s candle at %s", self.interval, candle_timestamp)

    async def _candle_aggregator(self) -> None:
        """
        Background task to ensure candles are created at regular intervals.
        """
        interval_seconds = self._interval_to_seconds(self.interval)

        while self._connected:
            try:
                await asyncio.sleep(interval_seconds)

                # Check if we need to force a new candle
                if self._current_candle:
                    now = datetime.now(UTC)
                    candle_age = (now - self._current_candle.timestamp).total_seconds()

                    if candle_age >= interval_seconds:
                        # Use last close price for new candle
                        last_price = self._current_candle.close
                        await self._create_new_candle(now, last_price)

            except asyncio.CancelledError:
                break
            except (ValueError, TypeError, ArithmeticError) as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "interval": self.interval,
                        "candle_aggregation_error": True,
                    },
                    component="BluefinWebSocketClient",
                    operation="candle_aggregator",
                )
            except Exception as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "unexpected_aggregator_error": True,
                    },
                    component="BluefinWebSocketClient",
                    operation="candle_aggregator",
                )

    async def _trade_aggregator(self) -> None:
        """
        Background task to aggregate trades into 1-second candles.

        This runs every second and builds OHLCV candles from individual trades
        that occurred within the last second window.
        """
        while self._connected:
            try:
                await asyncio.sleep(1.0)  # Run every second

                if not self._connected:
                    break

                # Get current time aligned to second boundary
                now = datetime.now(UTC)
                current_second = now.replace(microsecond=0)

                # Find trades from the last second
                one_second_ago = current_second - timedelta(seconds=1)

                # Collect trades from the last second
                trades_in_window = []
                for trade in list(self._trade_buffer):
                    trade_time = trade.get("timestamp", now)
                    if isinstance(trade_time, datetime):
                        # Align trade time to second boundary for comparison
                        trade_second = trade_time.replace(microsecond=0)
                        if one_second_ago <= trade_second < current_second:
                            trades_in_window.append(trade)

                # Build candle from trades if any exist
                if trades_in_window:
                    candle = await self._build_candle_from_trades(
                        trades_in_window, one_second_ago
                    )
                    if candle:
                        await self._process_aggregated_candle(candle)

                        logger.debug(
                            "Aggregated %d trades into 1s candle: %s @ $%s (vol: %s)",
                            len(trades_in_window),
                            self.symbol,
                            candle.close,
                            candle.volume,
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "trades_in_buffer": len(self._trade_buffer),
                        "unexpected_trade_aggregator_error": True,
                    },
                    component="BluefinWebSocketClient",
                    operation="trade_aggregator",
                )

    async def _token_refresh_handler(self) -> None:
        """
        Background task to refresh authentication tokens before expiration.

        Runs periodically to check token expiration and refresh when needed.
        """
        if not self._authenticator:
            logger.debug("Token refresh handler started but no authenticator available")
            return

        logger.info("Started token refresh handler for WebSocket authentication")

        # Refresh token every 30 minutes (tokens are valid for 1 hour)
        refresh_interval = 30 * 60  # 30 minutes in seconds

        while self._connected:
            try:
                await asyncio.sleep(refresh_interval)

                if not self._connected:
                    break

                # Check if token needs refresh
                auth_status = self._authenticator.get_status()

                if auth_status.get("authenticated", False):
                    expires_in = auth_status.get("token_expires_in_seconds", 0)

                    # Refresh if token expires within 10 minutes
                    if expires_in < 600:  # 10 minutes
                        logger.info(
                            "Refreshing WebSocket token (expires in %d seconds)",
                            expires_in,
                        )

                        refresh_success = await self.refresh_authentication()
                        if refresh_success:
                            logger.info(
                                "Successfully refreshed WebSocket authentication token"
                            )
                        else:
                            logger.error("Failed to refresh WebSocket token")
                    else:
                        logger.debug(
                            "WebSocket token still valid (expires in %d seconds)",
                            expires_in,
                        )
                else:
                    logger.warning("WebSocket authenticator reports not authenticated")

            except asyncio.CancelledError:
                logger.info("Token refresh handler cancelled")
                break
            except Exception as e:
                logger.exception("Error in token refresh handler: %s", e)
                # Continue running even if there's an error
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _build_candle_from_trades(
        self, trades: list[dict[str, Any]], timestamp: datetime
    ) -> MarketData | None:
        """
        Build a OHLCV candle from a list of trades.

        Args:
            trades: List of trade data dictionaries
            timestamp: Candle timestamp (aligned to second boundary)

        Returns:
            MarketData object or None if no valid trades
        """
        if not trades:
            return None

        try:
            # Sort trades by timestamp to ensure proper OHLC order
            sorted_trades = sorted(trades, key=lambda t: t.get("timestamp", timestamp))

            # Extract OHLCV data
            prices = []
            total_volume = Decimal(0)

            for trade in sorted_trades:
                price = trade.get("price")
                size = trade.get("size")

                if price and size:
                    # Note: price and size are already converted Decimal values from trade data
                    # They don't need additional conversion here as they come from converted trade data
                    prices.append(Decimal(str(price)))
                    total_volume += Decimal(str(size))

            if not prices:
                return None

            # Build OHLCV
            open_price = prices[0]
            close_price = prices[-1]
            high_price = max(prices)
            low_price = min(prices)

            return MarketData(
                symbol=self.symbol,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=total_volume,
            )

        except (ValueError, TypeError, KeyError, ArithmeticError) as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "trades_count": len(trades),
                    "candle_building_error": True,
                },
                component="BluefinWebSocketClient",
                operation="build_candle_from_trades",
            )
            return None

    async def _process_aggregated_candle(self, candle: MarketData) -> None:
        """
        Process an aggregated 1-second candle, potentially building it into larger intervals.

        Args:
            candle: 1-second candle from trade aggregation
        """
        try:
            # For now, treat 1-second candles as tick data for the main candle building
            # This allows compatibility with existing interval-based candle logic

            # Convert candle back to trade format for candle building
            trade_data = {
                "price": candle.close,
                "size": candle.volume,
                "side": "unknown",  # We don't need side for candle building
                "timestamp": candle.timestamp,
                "trade_id": f"aggregated_{int(candle.timestamp.timestamp())}",
            }

            # Update current candle with the aggregated data
            await self._update_candle_with_trade(trade_data)

        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "candle_timestamp": candle.timestamp,
                    "aggregated_candle_processing_error": True,
                },
                component="BluefinWebSocketClient",
                operation="process_aggregated_candle",
            )

    async def _handle_generic_price_update(self, data: dict[str, Any]) -> None:
        """
        Handle generic price updates from unstructured messages.

        This method attempts to extract price data from messages that don't
        match the standard event/channel patterns.

        Args:
            data: Message data that might contain price information
        """
        try:
            # Look for price-related fields in the data
            price_fields = [
                "price",
                "lastPrice",
                "last",
                "close",
                "currentPrice",
                "markPrice",
                "indexPrice",
                "bid",
                "ask",
                "mid",
            ]

            price_value = None
            price_source = None

            # Search for price data in various structures
            if isinstance(data.get("data"), dict):
                # Check nested data structure
                for field in price_fields:
                    if field in data["data"]:
                        price_value = data["data"][field]
                        price_source = f"data.{field}"
                        break

            if not price_value:
                # Check top-level fields
                for field in price_fields:
                    if field in data:
                        price_value = data[field]
                        price_source = field
                        break

            if not price_value:
                # Check for tick data structure
                if "tick" in data and isinstance(data["tick"], dict):
                    for field in price_fields:
                        if field in data["tick"]:
                            price_value = data["tick"][field]
                            price_source = f"tick.{field}"
                            break

            # If we found a price, process it
            if price_value is not None:
                try:
                    # Convert from 18-decimal format if needed
                    if is_likely_18_decimal(price_value):
                        price_decimal = convert_from_18_decimal(
                            price_value, self.symbol, f"generic_{price_source}"
                        )
                    else:
                        price_decimal = Decimal(str(price_value))

                    # Validate price
                    if price_decimal > 0 and self._validate_price_continuity(
                        price_decimal, f"generic_{price_source}"
                    ):
                        # Create a tick update for the aggregator
                        timestamp = datetime.now(UTC)

                        # If we have timestamp in the data, use it
                        if "timestamp" in data:
                            try:
                                ts = data["timestamp"]
                                if isinstance(ts, int | float):
                                    # Convert from milliseconds if needed
                                    if ts > 1e10:
                                        ts = ts / 1000
                                    timestamp = datetime.fromtimestamp(ts, UTC)
                            except:
                                pass

                        # Create simplified market data for price update
                        MarketData(
                            symbol=self.symbol,
                            timestamp=timestamp,
                            open=price_decimal,
                            high=price_decimal,
                            low=price_decimal,
                            close=price_decimal,
                            volume=Decimal(0),
                            last_trade_price=price_decimal,
                        )

                        # Process as tick for aggregation
                        await self._process_tick_update(
                            price_decimal,
                            Decimal(1),  # Default volume
                            timestamp,
                        )

                        logger.debug(
                            "Extracted price from generic message: $%s (source: %s)",
                            price_decimal,
                            price_source,
                        )

                except Exception as e:
                    logger.debug("Failed to process generic price update: %s", str(e))

        except Exception as e:
            # Don't log errors for every unhandled message
            if self._message_count <= 10:
                logger.debug("Error in generic price handler: %s", str(e))

    async def _heartbeat_handler(self) -> None:
        """Enhanced heartbeat handler with ping/pong monitoring."""
        ping_failures = 0

        while self._connected and self._ws:
            try:
                # Send ping every 20 seconds (more frequent)
                await asyncio.sleep(20)

                if not self._connected or not self._ws:
                    break

                # Use WebSocket built-in ping for better reliability
                try:
                    pong_waiter = await self._ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    ping_failures = 0
                    logger.debug("Bluefin WebSocket ping successful")

                except TimeoutError as e:
                    ping_failures += 1
                    exception_handler.log_exception_with_context(
                        e,
                        {
                            "symbol": self.symbol,
                            "ping_failures": ping_failures,
                            "max_failures": 3,
                            "ping_timeout_error": True,
                        },
                        component="BluefinWebSocketClient",
                        operation="heartbeat_handler",
                    )

                    if ping_failures >= 3:
                        logger.exception(
                            "Multiple ping failures, connection appears dead"
                        )
                        self._connected = False
                        break

                except (OSError, ConnectionError, WebSocketException) as e:
                    ping_failures += 1
                    exception_handler.log_exception_with_context(
                        e,
                        {
                            "symbol": self.symbol,
                            "ping_failures": ping_failures,
                            "max_failures": 3,
                            "ping_connection_error": True,
                        },
                        component="BluefinWebSocketClient",
                        operation="heartbeat_handler",
                    )

                    if ping_failures >= 3:
                        logger.exception(
                            "Multiple ping failures, marking connection as dead"
                        )
                        self._connected = False
                        break

                except Exception as e:
                    ping_failures += 1
                    exception_handler.log_exception_with_context(
                        e,
                        {
                            "symbol": self.symbol,
                            "ping_failures": ping_failures,
                            "unexpected_ping_error": True,
                        },
                        component="BluefinWebSocketClient",
                        operation="heartbeat_handler",
                    )

                    if ping_failures >= 3:
                        logger.exception(
                            "Multiple ping failures, marking connection as dead"
                        )
                        self._connected = False
                        break

                # Also send application-level ping using Socket.IO format
                ping_message = [
                    "ping",
                    [{"timestamp": time.time(), "t": self.auth_token}],
                ]

                await self._send_message(ping_message)
                logger.debug("Sent Socket.IO application heartbeat ping")

            except asyncio.CancelledError:
                break
            except Exception as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "unexpected_heartbeat_error": True,
                    },
                    component="BluefinWebSocketClient",
                    operation="heartbeat_handler",
                )
                # Don't break on general errors, just log and continue
                await asyncio.sleep(5)

    async def _send_message(self, message: dict[str, Any] | list[Any]) -> None:
        """
        Send message to WebSocket with error handling.

        Args:
            message: Message to send (dict or list)
        """
        if not self._ws or not self._connected:
            raise RuntimeError("Bluefin WebSocket not connected")

        try:
            json_data = json.dumps(message)
            await self._ws.send(json_data)

            # Handle different message types for logging
            if isinstance(message, dict):
                msg_type = message.get("method", message.get("type", "unknown"))
            elif isinstance(message, list) and len(message) > 0:
                msg_type = str(message[0])  # First element is usually the command
            else:
                msg_type = "unknown"

            logger.debug("Sent Bluefin message: %s", msg_type)
        except (ConnectionClosed, WebSocketException) as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "message_type": msg_type,
                    "connection_lost": True,
                },
                component="BluefinWebSocketClient",
                operation="send_message",
            )
            self._connected = False
            raise BluefinWebSocketConnectionError(
                "WebSocket connection lost during message send"
            ) from e
        except (OSError, ConnectionError) as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "message_type": msg_type,
                    "network_error": True,
                },
                component="BluefinWebSocketClient",
                operation="send_message",
            )
            raise BluefinWebSocketConnectionError(
                f"Network error sending message: {e}"
            ) from e
        except Exception as e:
            exception_handler.log_exception_with_context(
                e,
                {
                    "symbol": self.symbol,
                    "message_type": msg_type,
                    "unexpected_send_error": True,
                },
                component="BluefinWebSocketClient",
                operation="send_message",
            )
            raise BluefinWebSocketError(f"Unexpected error sending message: {e}") from e

    async def _wait_for_connection(self, timeout: int = 30) -> bool:
        """
        Wait for WebSocket connection to be established.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if connected, False if timeout
        """
        start_time = datetime.now(UTC)

        while not self._connected:
            if (datetime.now(UTC) - start_time).total_seconds() > timeout:
                logger.warning(
                    "Timeout waiting for WebSocket connection after %ss", timeout
                )
                return False

            await asyncio.sleep(0.1)

        return True

    def get_candles(self, limit: int | None = None) -> list[MarketData]:
        """
        Get historical candles from buffer.

        Args:
            limit: Maximum number of candles to return

        Returns:
            List of MarketData objects
        """
        candles = list(self._candle_buffer)

        # Include current candle if available
        if self._current_candle:
            candles.append(self._current_candle)

        if limit:
            return candles[-limit:]

        return candles

    def get_latest_price(self) -> Decimal | None:
        """
        Get the latest price.

        Returns:
            Latest price or None if not available
        """
        if self._current_candle:
            return self._current_candle.close

        if self._candle_buffer:
            return self._candle_buffer[-1].close

        return None

    def get_ticks(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get recent tick data.

        Args:
            limit: Maximum number of ticks to return

        Returns:
            List of tick data dictionaries
        """
        ticks = list(self._tick_buffer)

        if limit:
            return ticks[-limit:]

        return ticks

    async def refresh_authentication(self) -> bool:
        """
        Refresh authentication token and re-subscribe to private channels.

        Returns:
            True if refresh successful, False otherwise
        """
        if not self._authenticator:
            logger.warning("Cannot refresh authentication: no authenticator")
            return False

        try:
            logger.info("Refreshing WebSocket authentication token")

            # Force refresh the token
            self._authenticator.refresh_token()

            # Re-subscribe to private channels with new token
            if self.enable_private_channels and self._connected:
                await self._subscribe_to_private_channels()

            logger.info("Successfully refreshed WebSocket authentication")
            return True

        except Exception as e:
            logger.exception("Failed to refresh WebSocket authentication: %s", e)
            return False

    async def _handle_authentication_error(self, error_data: dict[str, Any]) -> None:
        """
        Handle authentication errors and attempt recovery.

        Args:
            error_data: Error message data from WebSocket
        """
        try:
            error_code = error_data.get("code", "unknown")
            error_message = error_data.get("message", "Authentication failed")

            logger.warning(
                "WebSocket authentication error: %s (%s)", error_message, error_code
            )

            # Attempt to refresh authentication
            if error_code in ["401", "403", "TOKEN_EXPIRED", "INVALID_TOKEN"]:
                logger.info("Attempting to refresh authentication due to auth error")

                refresh_success = await self.refresh_authentication()
                if not refresh_success:
                    logger.error(
                        "Authentication refresh failed - disabling private channels"
                    )
                    self.enable_private_channels = False
            else:
                logger.error("Unrecoverable authentication error: %s", error_message)
                self.enable_private_channels = False

        except Exception as e:
            logger.exception("Error handling authentication error: %s", e)
            self.enable_private_channels = False

    def is_authenticated(self) -> bool:
        """
        Check if client is properly authenticated for private channels.

        Returns:
            True if authenticated and ready for private channels
        """
        return (
            self._authenticator is not None
            and self._authenticator.is_authenticated()
            and self.enable_private_channels
        )

    def get_authentication_status(self) -> dict[str, Any]:
        """
        Get detailed authentication status information.

        Returns:
            Authentication status dictionary
        """
        if not self._authenticator:
            return {
                "enabled": False,
                "authenticated": False,
                "error": "No authenticator configured",
            }

        return {
            "enabled": True,
            "authenticated": self._authenticator.is_authenticated(),
            "private_channels_enabled": self.enable_private_channels,
            "public_key": self._authenticator.get_public_key(),
            "status": self._authenticator.get_status(),
        }

    def get_status(self) -> dict[str, Any]:
        """
        Get WebSocket client status.

        Returns:
            Status dictionary
        """
        return {
            "connected": self._connected,
            "symbol": self.symbol,
            "interval": self.interval,
            "candles_buffered": len(self._candle_buffer),
            "current_candle": self._current_candle is not None,
            "ticks_buffered": len(self._tick_buffer),
            "use_trade_aggregation": self.use_trade_aggregation,
            "trades_buffered": (
                len(self._trade_buffer) if self.use_trade_aggregation else 0
            ),
            "subscribed_channels": list(self._subscribed_channels),
            "pending_subscriptions": list(self._pending_subscriptions.values()),
            "subscription_id": self._subscription_id,
            "reconnect_attempts": self._reconnect_attempts,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "last_message_time": self._last_message_time,
            "latest_price": self.get_latest_price(),
            # Enhanced validation metrics
            "last_valid_price": (
                float(self._last_valid_price) if self._last_valid_price else None
            ),
            "price_continuity_violations": self._price_continuity_violations,
            "price_jump_threshold_percent": float(self._price_jump_threshold * 100),
            "invalid_message_counts": dict(self._invalid_message_counter),
            "astronomical_price_counts": dict(self._astronomical_log_counter),
            # Authentication and private channel status
            "authentication_enabled": self._authenticator is not None,
            "private_channels_enabled": self.enable_private_channels,
            "authentication_status": (
                self._authenticator.get_status() if self._authenticator else None
            ),
        }

    def _get_next_subscription_id(self) -> int:
        """Get next subscription ID."""
        sub_id = self._subscription_id
        self._subscription_id += 1
        return sub_id

    def _should_create_new_candle(self, timestamp: datetime) -> bool:
        """
        Check if a new candle should be created based on timestamp.

        Args:
            timestamp: Current timestamp

        Returns:
            True if new candle needed
        """
        if not self._current_candle:
            return True

        interval_seconds = self._interval_to_seconds(self.interval)
        candle_age = (timestamp - self._current_candle.timestamp).total_seconds()

        return candle_age >= interval_seconds

    def _get_candle_timestamp(self, timestamp: datetime) -> datetime:
        """
        Get normalized candle timestamp based on interval.

        Args:
            timestamp: Raw timestamp

        Returns:
            Normalized candle timestamp
        """
        interval_seconds = self._interval_to_seconds(self.interval)

        # Round down to nearest interval
        epoch = timestamp.timestamp()
        rounded = (epoch // interval_seconds) * interval_seconds

        return datetime.fromtimestamp(rounded, UTC)

    def _interval_to_seconds(self, interval: str) -> int:
        """
        Convert interval string to seconds.

        Args:
            interval: Interval string (e.g., '1s', '5s', '30s', '1m', '5m', '1h')

        Returns:
            Interval in seconds
        """
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}

        if interval[-1] in multipliers:
            return int(interval[:-1]) * multipliers[interval[-1]]

        # Default to 1 minute
        return 60

    def _parse_timestamp(self, ts: Any) -> datetime:
        """
        Parse timestamp from various formats.

        Args:
            ts: Timestamp in various formats

        Returns:
            Parsed datetime object
        """
        if isinstance(ts, datetime):
            return ts

        if isinstance(ts, int | float):
            # Assume milliseconds if large number
            if ts > 1e10:
                return datetime.fromtimestamp(ts / 1000, UTC)
            return datetime.fromtimestamp(ts, UTC)

        if isinstance(ts, str):
            try:
                # Try ISO format
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                pass

        # Default to current time
        return datetime.now(UTC)


# Integration with BluefinMarketDataProvider
async def integrate_websocket_with_provider(
    provider: Any,
    symbol: str,
    interval: str,
    use_trade_aggregation: bool = True,
    auth_token: str | None = None,
) -> BluefinWebSocketClient:
    """
    Create and integrate WebSocket client with BluefinMarketDataProvider.

    Args:
        provider: BluefinMarketDataProvider instance
        symbol: Trading symbol
        interval: Candle interval
        use_trade_aggregation: Enable trade-to-candle aggregation
        auth_token: Authentication token for Socket.IO subscriptions

    Returns:
        Connected BluefinWebSocketClient instance
    """

    # Create WebSocket client with callback to update provider
    async def on_candle_update(candle: MarketData):
        # Update provider's cache
        provider._ohlcv_cache.append(candle)
        if len(provider._ohlcv_cache) > provider.candle_limit:
            provider._ohlcv_cache = provider._ohlcv_cache[-provider.candle_limit :]

        provider._last_update = datetime.now(UTC)
        provider._cache_timestamps["ohlcv"] = provider._last_update

        # Update price cache
        provider._price_cache["price"] = candle.close
        provider._cache_timestamps["price"] = datetime.now(UTC)

        # Notify provider's subscribers
        await provider._notify_subscribers(candle)

    # Create and connect WebSocket client
    ws_client = BluefinWebSocketClient(
        symbol=symbol,
        interval=interval,
        candle_limit=provider.candle_limit,
        on_candle_update=on_candle_update,
        network=getattr(provider, "network", None),  # Pass network if available
        use_trade_aggregation=use_trade_aggregation,
        auth_token=auth_token,
    )

    await ws_client.connect()

    # Update provider's initial cache with WebSocket data
    candles = ws_client.get_candles()
    if candles:
        provider._ohlcv_cache = candles[-provider.candle_limit :]
        provider._last_update = datetime.now(UTC)

        # Update price from latest candle
        latest_price = ws_client.get_latest_price()
        if latest_price:
            provider._price_cache["price"] = latest_price
            provider._cache_timestamps["price"] = datetime.now(UTC)

    return ws_client
