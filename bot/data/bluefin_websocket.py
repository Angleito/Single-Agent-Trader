"""Bluefin WebSocket client for real-time market data streaming."""

import asyncio
import contextlib
import json
import logging
import os
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from bot.error_handling import exception_handler
from bot.trading_types import MarketData

if TYPE_CHECKING:
    from websockets.client import WebSocketClientProtocol

# Import centralized endpoint configuration with fallback
try:
    from bot.exchange.bluefin_endpoints import get_notifications_url, get_websocket_url
except ImportError:
    # Fallback endpoint resolver if centralized config is not available
    def get_websocket_url(network: str | None = "mainnet") -> str:
        """Fallback WebSocket URL resolver."""
        if network.lower() == "testnet":
            return "wss://dapi.api.sui-staging.bluefin.io"
        else:
            return "wss://dapi.api.sui-prod.bluefin.io"

    def get_notifications_url(network: str | None = "mainnet") -> str:
        """Fallback notifications URL resolver."""
        if network.lower() == "testnet":
            return "wss://notifications.api.sui-staging.bluefin.io"
        else:
            return "wss://notifications.api.sui-prod.bluefin.io"


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
    ):
        """
        Initialize the Bluefin WebSocket client.

        Args:
            symbol: Trading symbol (e.g., 'SUI-PERP')
            interval: Candle interval for aggregation
            candle_limit: Maximum number of candles to maintain in buffer
            on_candle_update: Callback function for candle updates
            network: Network type ('mainnet' or 'testnet'). If None, uses environment variable
        """
        self.symbol = symbol
        self.interval = interval
        self.candle_limit = candle_limit
        self.on_candle_update = on_candle_update

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

        # Performance metrics
        self._last_message_time: datetime | None = None
        self._message_count = 0
        self._error_count = 0

        logger.info(
            "Initialized BluefinWebSocketClient for %s with %s candles on %s network",
            symbol,
            interval,
            self.network,
            extra={
                "symbol": symbol,
                "interval": interval,
                "network": self.network,
                "notification_ws_url": self.NOTIFICATION_WS_URL,
                "dapi_ws_url": self.DAPI_WS_URL,
            },
        )

    async def connect(self) -> None:
        """Establish WebSocket connection and start data streaming."""
        if self._connected:
            logger.warning("Already connected to Bluefin WebSocket")
            return

        logger.info("Connecting to Bluefin WebSocket at %s", self.NOTIFICATION_WS_URL)

        # Start connection task
        self._connection_task = asyncio.create_task(self._connection_handler())

        # Start candle aggregation task
        self._candle_aggregation_task = asyncio.create_task(self._candle_aggregator())

        # Wait for initial connection
        await self._wait_for_connection(timeout=30)

    async def disconnect(self) -> None:
        """Disconnect from WebSocket and cleanup resources."""
        logger.info("Disconnecting from Bluefin WebSocket")

        self._connected = False

        # Cancel tasks
        tasks = [
            self._connection_task,
            self._heartbeat_task,
            self._candle_aggregation_task,
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

        while True:
            try:
                await self._connect_and_subscribe()
                self._reconnect_attempts = 0
                consecutive_failures = 0  # Reset on successful connection

                # Handle incoming messages
                await self._message_handler()

            except ConnectionClosed as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "reconnect_attempts": self._reconnect_attempts,
                        "consecutive_failures": consecutive_failures,
                        "message_count": self._message_count,
                    },
                    component="BluefinWebSocketClient",
                    operation="connection_handler",
                )
                self._connected = False
                self._ws = None
                self._subscribed_channels.clear()
                self._pending_subscriptions.clear()
                consecutive_failures += 1

            except (WebSocketException, OSError, ConnectionError) as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "reconnect_attempts": self._reconnect_attempts,
                        "consecutive_failures": consecutive_failures,
                        "error_count": self._error_count,
                    },
                    component="BluefinWebSocketClient",
                    operation="connection_handler",
                )
                self._connected = False
                self._ws = None
                self._subscribed_channels.clear()
                self._pending_subscriptions.clear()
                self._error_count += 1
                consecutive_failures += 1

            except Exception as e:
                exception_handler.log_exception_with_context(
                    e,
                    {
                        "symbol": self.symbol,
                        "unexpected_error": True,
                        "reconnect_attempts": self._reconnect_attempts,
                        "consecutive_failures": consecutive_failures,
                    },
                    component="BluefinWebSocketClient",
                    operation="connection_handler",
                )
                self._connected = False
                self._ws = None
                self._subscribed_channels.clear()
                self._pending_subscriptions.clear()
                self._error_count += 1
                consecutive_failures += 1

            # Check if we should reconnect
            if self._reconnect_attempts >= self._max_reconnect_attempts:
                logger.error("Max reconnection attempts reached, stopping")
                break

            self._reconnect_attempts += 1

            # Enhanced exponential backoff with jitter
            base_delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))
            jitter = base_delay * 0.1 * (0.5 - (asyncio.get_event_loop().time() % 1))
            delay = min(base_delay + jitter, 60)

            # Additional delay for consecutive failures
            if consecutive_failures > 3:
                delay = min(delay * 1.5, 120)
                logger.warning(
                    "Multiple consecutive failures (%s), extending delay",
                    consecutive_failures,
                )

            logger.info(
                "Reconnecting to Bluefin in %.1fs (attempt %s/%s)",
                delay,
                self._reconnect_attempts,
                self._max_reconnect_attempts,
            )

            await asyncio.sleep(delay)

    async def _connect_and_subscribe(self) -> None:
        """Establish WebSocket connection and subscribe to channels."""
        # Enhanced connection parameters for better stability
        self._ws = await websockets.connect(
            self.NOTIFICATION_WS_URL,
            ping_interval=15,  # More frequent pings
            ping_timeout=8,  # Shorter timeout to detect issues faster
            close_timeout=5,  # Quick close timeout
            max_size=2**20,  # 1MB max message size
            compression=None,  # Disable compression for performance
            extra_headers={
                "User-Agent": "AI-Trading-Bot-Bluefin-Client/1.0",
                "Accept": "*/*",
                "Connection": "Upgrade",
            },
        )

        self._connected = True
        logger.info("Bluefin WebSocket connection established")

        # Start heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_handler())

        # Subscribe to market data channels
        await self._subscribe_to_market_data()

    async def _subscribe_to_market_data(self) -> None:
        """Subscribe to relevant market data channels using JSON-RPC format."""
        # Use proper JSON-RPC format for Bluefin WebSocket API

        # Subscribe to global updates for this symbol
        global_id = self._get_next_subscription_id()
        global_subscription = {
            "id": global_id,
            "method": "SUBSCRIBE",
            "params": {
                "channel": "globalUpdates",
                "symbol": self.symbol,  # Symbol like "SUI-PERP"
            },
        }
        self._pending_subscriptions[global_id] = f"globalUpdates:{self.symbol}"
        await self._send_message(global_subscription)
        logger.info(
            "Subscribing to globalUpdates for %s (ID: %s)", self.symbol, global_id
        )

        # Subscribe to ticker updates for price data
        ticker_id = self._get_next_subscription_id()
        ticker_subscription = {
            "id": ticker_id,
            "method": "SUBSCRIBE",
            "params": {
                "channel": "ticker",
                "symbol": self.symbol,
            },
        }
        self._pending_subscriptions[ticker_id] = f"ticker:{self.symbol}"
        await self._send_message(ticker_subscription)
        logger.info("Subscribing to ticker for %s (ID: %s)", self.symbol, ticker_id)

        # Subscribe to kline data for candlestick updates
        kline_id = self._get_next_subscription_id()
        kline_subscription = {
            "id": kline_id,
            "method": "SUBSCRIBE",
            "params": {
                "channel": "kline",
                "symbol": self.symbol,
                "interval": self.interval,  # Format: "1m", "5m", etc.
            },
        }
        self._pending_subscriptions[kline_id] = f"kline:{self.symbol}@{self.interval}"
        await self._send_message(kline_subscription)
        logger.info(
            "Subscribing to kline data for %s@%s (ID: %s)",
            self.symbol,
            self.interval,
            kline_id,
        )

        # Subscribe to trade data for real-time price updates
        trade_id = self._get_next_subscription_id()
        trade_subscription = {
            "id": trade_id,
            "method": "SUBSCRIBE",
            "params": {
                "channel": "trade",
                "symbol": self.symbol,
            },
        }
        self._pending_subscriptions[trade_id] = f"trade:{self.symbol}"
        await self._send_message(trade_subscription)
        logger.info("Subscribing to trade data for %s (ID: %s)", self.symbol, trade_id)

        # Channels will be tracked after subscription confirmation

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
                    msg_type = (
                        data.get("type", "unknown")
                        if isinstance(data, dict)
                        else "unknown"
                    )
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

    async def _process_message(self, data: dict[str, Any]) -> None:
        """
        Process incoming WebSocket message.

        Args:
            data: Parsed message data
        """
        # Handle subscription confirmations and errors
        if "id" in data:
            sub_id = data["id"]

            if "result" in data:
                # Successful subscription
                if sub_id in self._pending_subscriptions:
                    channel = self._pending_subscriptions.pop(sub_id)
                    self._subscribed_channels.add(channel)
                    logger.info(
                        "Subscription confirmed: %s (ID: %s, result: %s)",
                        channel,
                        sub_id,
                        data["result"],
                    )
                else:
                    logger.debug(
                        "Subscription %s confirmed: %s", sub_id, data["result"]
                    )
                return

            elif "error" in data:
                # Subscription error
                if sub_id in self._pending_subscriptions:
                    channel = self._pending_subscriptions.pop(sub_id)
                    logger.error(
                        "Subscription failed: %s (ID: %s, error: %s)",
                        channel,
                        sub_id,
                        data["error"],
                    )
                else:
                    logger.error("Subscription %s failed: %s", sub_id, data["error"])
                return

        # Handle error messages
        if "error" in data:
            logger.error("WebSocket error: %s", data["error"])
            return

        # Handle Bluefin-specific event names from WebSocket API
        event_name = data.get("eventName")

        if event_name == "TickerUpdate":
            await self._handle_bluefin_ticker_update(data)
        elif event_name == "MarketDataUpdate":
            await self._handle_bluefin_market_update(data)
        elif event_name == "RecentTrades":
            await self._handle_bluefin_trades(data)
        elif f"{self.symbol}@kline@{self.interval}" in str(event_name):
            # Handle kline/candlestick updates
            await self._handle_bluefin_kline_update(data)
        else:
            # Handle standard channels
            channel = data.get("channel", data.get("ch", ""))

            if channel == "trade" or "trade" in str(channel):
                await self._handle_trade_update(data)
            elif channel == "ticker" or "ticker" in str(channel):
                await self._handle_ticker_update(data)
            elif channel == "orderbook" or "orderbook" in str(channel):
                await self._handle_orderbook_update(data)
            elif self._message_count % 100 == 0:
                logger.debug("Unhandled event/channel: %s", event_name or channel)

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
                # Parse trade fields
                price = Decimal(str(trade.get("price", 0)))
                size = Decimal(str(trade.get("size", trade.get("amount", 0))))
                side = trade.get("side", "")
                timestamp = self._parse_timestamp(
                    trade.get("timestamp", trade.get("ts"))
                )

                if price > 0 and size > 0:
                    trade_data = {
                        "price": price,
                        "size": size,
                        "side": side,
                        "timestamp": timestamp,
                        "trade_id": trade.get("id", ""),
                    }

                    # Add to tick buffer
                    self._tick_buffer.append(trade_data)

                    # Update current candle
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

            last_price = Decimal(str(ticker.get("last", ticker.get("lastPrice", 0))))
            self._parse_timestamp(ticker.get("timestamp", ticker.get("ts")))

            if last_price > 0:
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

                    # Import price conversion utility
                    from bot.utils.price_conversion import convert_from_18_decimal

                    # Use smart conversion that detects 18-decimal format
                    price = convert_from_18_decimal(
                        price_str, self.symbol, "ticker_price"
                    )
                    last_price = convert_from_18_decimal(
                        last_price_str, self.symbol, "ticker_last_price"
                    )

                    # Use the most recent price
                    current_price = price if price > 0 else last_price

                    if current_price > 0:
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

                        # Update current candle
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

                    # Import price conversion utility
                    from bot.utils.price_conversion import convert_from_18_decimal

                    # Use smart conversion that detects 18-decimal format
                    price = convert_from_18_decimal(
                        price_str, self.symbol, "trade_price"
                    )
                    size = convert_from_18_decimal(size_str, self.symbol, "trade_size")

                    if price > 0 and size > 0:
                        trade_data = {
                            "price": price,
                            "size": size,
                            "side": trade.get("side", "").lower(),
                            "timestamp": self._parse_timestamp(trade.get("timestamp")),
                            "trade_id": trade.get("id", ""),
                        }

                        # Add to tick buffer
                        self._tick_buffer.append(trade_data)

                        # Update current candle
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

                # Convert from 18-decimal format if needed using utility function
                try:
                    from bot.utils.price_conversion import convert_from_18_decimal

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

                if open_val > 0 and close_val > 0:
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
            volume=Decimal("0"),
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

    async def _heartbeat_handler(self) -> None:
        """Enhanced heartbeat handler with ping/pong monitoring."""
        # last_pong_time = asyncio.get_event_loop().time()  # Used for monitoring
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
                    # last_pong_time = asyncio.get_event_loop().time()  # Monitor pong timing
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

                # Also send application-level ping
                ping_message = {
                    "id": self._get_next_subscription_id(),
                    "method": "ping",
                    "timestamp": asyncio.get_event_loop().time(),
                }

                await self._send_message(ping_message)
                logger.debug("Sent application heartbeat ping")

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
            "subscribed_channels": list(self._subscribed_channels),
            "pending_subscriptions": list(self._pending_subscriptions.values()),
            "subscription_id": self._subscription_id,
            "reconnect_attempts": self._reconnect_attempts,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "last_message_time": self._last_message_time,
            "latest_price": self.get_latest_price(),
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
            interval: Interval string (e.g., '1m', '5m', '1h')

        Returns:
            Interval in seconds
        """
        multipliers = {"m": 60, "h": 3600, "d": 86400}

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
            else:
                return datetime.fromtimestamp(ts, UTC)

        if isinstance(ts, str):
            try:
                # Try ISO format
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Default to current time
        return datetime.now(UTC)


# Integration with BluefinMarketDataProvider
async def integrate_websocket_with_provider(
    provider: Any, symbol: str, interval: str
) -> BluefinWebSocketClient:
    """
    Create and integrate WebSocket client with BluefinMarketDataProvider.

    Args:
        provider: BluefinMarketDataProvider instance
        symbol: Trading symbol
        interval: Candle interval

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
