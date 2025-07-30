"""Bluefin-specific market data provider implementation."""

import asyncio
import json
import logging
import os
from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, cast

import websockets

from bot.config import settings
from bot.data.base_market_provider import AbstractMarketDataProvider
from bot.trading_types import MarketData
from bot.utils.price_conversion import (
    convert_from_18_decimal,
)

logger = logging.getLogger(__name__)


class BluefinDataError(Exception):
    """Exception raised when Bluefin data operations fail."""


class BluefinMarketDataProvider(AbstractMarketDataProvider):
    """
    Market data provider for Bluefin perpetual futures with trade aggregation support.

    Handles real-time and historical market data from Bluefin DEX,
    providing OHLCV data for perpetual futures contracts on Sui.

    Features:
    - Real-time market data via WebSocket
    - Historical data fetching via Bluefin service
    - Trade aggregation for sub-minute intervals (15s, 30s, etc.)
    - Seamless switching between kline and trade aggregation modes
    - Perpetual futures symbol mapping
    - Enhanced interval validation and processing

    Trade Aggregation Mode:
    When use_trade_aggregation=True, the provider:
    - Receives individual trade/tick data via WebSocket
    - Aggregates trades into 1-second candles
    - Builds target interval candles from aggregated data
    - Provides sub-minute granularity not available via standard kline API

    Kline Mode (default):
    When use_trade_aggregation=False, the provider:
    - Receives pre-built kline/candlestick data via WebSocket
    - Uses standard Bluefin API intervals (1m, 5m, 1h, etc.)
    - More efficient for standard timeframes
    """

    def __init__(self, symbol: str | None = None, interval: str | None = None) -> None:
        """
        Initialize the Bluefin market data provider.

        Args:
            symbol: Trading symbol (e.g., 'SUI-PERP', 'ETH-PERP')
            interval: Candle interval (e.g., '1s', '5s', '15s', '30s', '1m', '5m', '1h').
                     Sub-minute intervals (1s, 5s, 15s, 30s) use trade aggregation when enabled.
        """
        # Convert symbol to Bluefin format before calling parent init
        converted_symbol = self._convert_symbol(symbol or settings.trading.symbol)
        super().__init__(converted_symbol, interval)

        # Generate provider ID for logging context
        import time

        self.provider_id = f"bluefin-market-{int(time.time())}"

        # Read trade aggregation setting from exchange configuration
        self.use_trade_aggregation = settings.exchange.use_trade_aggregation

        # Validate interval for trade aggregation compatibility
        self._validate_interval_for_trade_aggregation()

        # Extended history tracking
        self._extended_history_mode = False
        self._extended_history_limit = (
            2000  # Max candles to keep in extended mode (e.g., 8 hours of 15s candles)
        )

        # Add type annotation for WebSocket client
        self._ws_client: Any = None

        # Tick data buffer for candle building
        self._tick_buffer: deque[dict[str, Any]] = deque(maxlen=10000)
        self._candle_builder_task: asyncio.Task | None = None
        self._last_candle_timestamp: datetime | None = None

        # Use centralized endpoint configuration for consistency
        try:
            from bot.exchange.bluefin_endpoints import (
                get_notifications_url,
                get_rest_api_url,
                get_websocket_url,
            )

            self.network = os.getenv("EXCHANGE__BLUEFIN_NETWORK", "mainnet").lower()
            network_type = cast(
                "Literal['mainnet', 'testnet']",
                self.network if self.network in ["mainnet", "testnet"] else "mainnet",
            )
            self._api_base_url = get_rest_api_url(network_type)
            self._ws_url = get_websocket_url(network_type)
            self._notification_ws_url = get_notifications_url(network_type)
        except ImportError:
            # Fallback to hardcoded URLs if centralized config is not available
            self.network = os.getenv("EXCHANGE__BLUEFIN_NETWORK", "mainnet").lower()
            if self.network == "testnet":
                self._api_base_url = "https://dapi.api.sui-staging.bluefin.io"
                self._ws_url = "wss://dapi.api.sui-staging.bluefin.io"
                self._notification_ws_url = (
                    "wss://notifications.api.sui-staging.bluefin.io"
                )
            else:
                self._api_base_url = "https://dapi.api.sui-prod.bluefin.io"
                self._ws_url = "wss://dapi.api.sui-prod.bluefin.io"
                self._notification_ws_url = (
                    "wss://notifications.api.sui-prod.bluefin.io"
                )

        logger.info(
            "Initialized BluefinMarketDataProvider",
            extra={
                "provider_id": self.provider_id,
                "symbol": self.symbol,
                "interval": self.interval,
                "candle_limit": self.candle_limit,
                "network": self.network,
                "data_source": "real_market_data",
                "api_base_url": self._api_base_url,
                "ws_url": self._ws_url,
                "use_trade_aggregation": self.use_trade_aggregation,
                "trade_aggregation_mode": (
                    "kline" if not self.use_trade_aggregation else "trade_aggregation"
                ),
            },
        )

        # Log mode selection for debugging
        if self.use_trade_aggregation:
            logger.info(
                "Using trade aggregation mode for enhanced granularity",
                extra={
                    "provider_id": self.provider_id,
                    "data_source": "trade_aggregation",
                    "aggregation_interval": "1s",
                    "target_interval": self.interval,
                },
            )
        else:
            logger.info(
                "Using kline mode for standard intervals",
                extra={
                    "provider_id": self.provider_id,
                    "data_source": "kline_websocket",
                    "interval": self.interval,
                },
            )

        logger.info(
            "Using real Bluefin market data via WebSocket",
            extra={
                "provider_id": self.provider_id,
                "data_source": "live_websocket",
            },
        )

    def _convert_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Bluefin perpetual format."""
        # Map common symbols to Bluefin perpetual contracts
        symbol_map = {
            "BTC-USD": "BTC-PERP",
            "ETH-USD": "ETH-PERP",
            "SOL-USD": "SOL-PERP",
            "SUI-USD": "SUI-PERP",
        }

        return symbol_map.get(symbol, symbol)

    def _validate_interval_for_trade_aggregation(self) -> None:
        """Validate interval compatibility with trade aggregation mode."""
        # Parse interval to check if it's sub-minute
        is_sub_minute = self._is_sub_minute_interval(self.interval)

        if is_sub_minute and not self.use_trade_aggregation:
            logger.warning(
                "⚠️ Sub-minute interval '%s' detected but trade aggregation is disabled. "
                "Historical data may be limited. Consider enabling use_trade_aggregation.",
                self.interval,
            )
        elif not is_sub_minute and self.use_trade_aggregation:
            logger.info(
                "Trade aggregation enabled for standard interval '%s'. "
                "This provides enhanced granularity for candle building.",
                self.interval,
            )

    def _is_sub_minute_interval(self, interval: str) -> bool:
        """Check if interval is sub-minute (e.g., 15s, 30s)."""
        if interval.endswith("s"):
            try:
                seconds = int(interval[:-1])
            except ValueError:
                return False
            else:
                return seconds < 60
        return False

    async def connect(self, fetch_historical: bool = True) -> None:
        """
        Establish connection to Bluefin data feeds with comprehensive error handling.

        Args:
            fetch_historical: Whether to fetch historical data during connection

        Raises:
            BluefinConnectionError: When connection setup fails
            BluefinDataError: When initial data fetch fails
        """
        logger.info(
            "Starting Bluefin market data connection",
            extra={
                "provider_id": self.provider_id,
                "symbol": self.symbol,
                "interval": self.interval,
                "fetch_historical": fetch_historical,
                "data_source": "real_market_data",
            },
        )
        try:
            # Initialize HTTP session
            await self._initialize_http_session()

            # Fetch historical data if requested
            if fetch_historical:
                try:
                    await self.fetch_historical_data()
                except Exception as e:
                    logger.warning(
                        "Failed to fetch historical data, continuing with WebSocket only: %s",
                        e,
                    )

            # Start WebSocket connection for real-time updates
            await self._start_websocket()

            # Try to get current price to ensure connectivity
            try:
                current_price = await self.fetch_latest_price()
                if current_price:
                    logger.info(
                        "Successfully fetched current price: $%s", current_price
                    )
            except Exception as e:
                logger.warning("Could not fetch current price: %s", e)

            self._is_connected = True
            logger.info("Successfully connected to Bluefin market data feeds")

        except Exception:
            logger.exception("Failed to connect to Bluefin market data feeds")
            self._is_connected = False
            raise

    async def fetch_historical_data(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        granularity: str | None = None,
    ) -> list[MarketData]:
        """
        Fetch historical OHLCV data from Bluefin API with enhanced interval support.

        Args:
            start_time: Start time for data
            end_time: End time for data
            granularity: Candle granularity (supports sub-minute intervals with trade aggregation)

        Returns:
            List of MarketData objects
        """
        granularity = granularity or self.interval
        end_time = end_time or datetime.now(UTC)

        # Calculate start time based on interval and limit
        interval_seconds = self._interval_to_seconds(granularity)
        default_start = end_time - timedelta(
            seconds=interval_seconds * self.candle_limit
        )
        start_time = start_time or default_start

        logger.info(
            "Fetching Bluefin historical data for %s from %s to %s (interval: %s)",
            self.symbol,
            start_time,
            end_time,
            granularity,
        )

        # Check if we need to use trade aggregation for sub-minute intervals
        is_sub_minute = self._is_sub_minute_interval(granularity)
        if is_sub_minute and self.use_trade_aggregation:
            logger.info(
                "Using trade aggregation for sub-minute interval %s", granularity
            )
            # For sub-minute intervals, fetch 1m candles and we'll aggregate from trades
            api_granularity = "60"  # 1 minute in seconds
        else:
            api_granularity = self._format_granularity_for_bluefin(granularity)

        try:
            url = f"{self._api_base_url}/marketData/candles"

            params = {
                "symbol": self.symbol,
                "interval": api_granularity,
                "startTime": int(
                    start_time.timestamp() * 1000
                ),  # Bluefin uses milliseconds
                "endTime": int(end_time.timestamp() * 1000),
                "limit": min(self.candle_limit, 1000),  # Bluefin max limit
            }

            if self._session is None:
                raise RuntimeError("HTTP session not initialized")

            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise BluefinDataError(
                        f"API request failed with status {response.status}: {error_text}"
                    )

                data = await response.json()

                # Bluefin returns data in different format
                candles = data if isinstance(data, list) else data.get("data", [])

                # Convert to MarketData objects
                historical_data = []
                for candle in candles:
                    try:
                        market_data = self._parse_bluefin_candle(candle)
                        if self._validate_market_data(market_data):
                            historical_data.append(market_data)
                    except Exception as e:
                        logger.warning("Failed to parse candle data: %s", e)
                        continue

                # Sort by timestamp
                historical_data.sort(key=lambda x: x.timestamp)

                # Update cache
                self._ohlcv_cache = historical_data[-self.candle_limit :]
                self._last_update = datetime.now(UTC)
                self._cache_timestamps["ohlcv"] = self._last_update

                logger.info("Loaded %s historical candles", len(self._ohlcv_cache))

                # Validate we have sufficient data for indicators
                self._validate_data_sufficiency(len(self._ohlcv_cache))

                return historical_data

        except Exception:
            logger.exception("Failed to fetch Bluefin historical data")
            # Fallback to cached data if available
            if self._ohlcv_cache:
                logger.info("Using cached historical data")
                return self._ohlcv_cache
            raise

    async def fetch_latest_price(self) -> Decimal | None:
        """
        Fetch the latest price for the symbol from Bluefin API.

        Returns:
            Latest price as Decimal or None if unavailable
        """
        # Check cache first
        if self._is_cache_valid("price"):
            return self._price_cache.get("price")

        try:
            url = f"{self._api_base_url}/marketData/ticker"
            params = {"symbol": self.symbol}

            if self._session is None:
                raise RuntimeError("HTTP session not initialized")

            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning("Failed to fetch latest price: %s", response.status)
                    return self.get_latest_price()  # Fall back to cached data

                data = await response.json()

                # Bluefin ticker format
                ticker_data = (
                    data
                    if not isinstance(data, dict) or "data" not in data
                    else data["data"]
                )

                if ticker_data:
                    # Convert from 18 decimal format
                    price_raw = ticker_data.get("lastPrice") or ticker_data.get("price")
                    if price_raw:
                        price = convert_from_18_decimal(price_raw)
                        # Update cache
                        self._price_cache["price"] = price
                        self._cache_timestamps["price"] = datetime.now(UTC)
                        return price

        except Exception:
            logger.exception("Error fetching latest price from Bluefin")

        # Fall back to cached OHLCV data
        return self.get_latest_price()

    async def _connect_websocket(self) -> None:
        """
        Establish WebSocket connection to Bluefin.
        """
        # Choose WebSocket endpoint based on data mode
        ws_url = (
            self._ws_url
            if not self.use_trade_aggregation
            else self._notification_ws_url
        )

        # Build subscription message
        if self.use_trade_aggregation:
            # Subscribe to trades for aggregation
            subscription = {
                "type": "subscribe",
                "channel": "trades",
                "symbol": self.symbol,
            }
        else:
            # Subscribe to klines
            subscription = {
                "type": "subscribe",
                "channel": "candles",
                "symbol": self.symbol,
                "interval": self._format_granularity_for_bluefin(self.interval),
            }

        logger.info(
            "Connecting to Bluefin WebSocket at %s with subscription: %s",
            ws_url,
            subscription,
        )

        async with websockets.connect(ws_url) as websocket:
            self._ws_connection = websocket  # type: ignore[assignment]

            # Send subscription
            await websocket.send(json.dumps(subscription))
            logger.info("Sent subscription to Bluefin WebSocket")

            # Reset reconnection counter on successful connection
            self._reconnect_attempts = 0

            # Handle incoming messages
            message_count = 0
            async for message in websocket:
                try:
                    message_count += 1
                    parsed_message = json.loads(message)

                    # Log first few messages for debugging
                    if message_count <= 5:
                        logger.debug(
                            "WebSocket message #%s: %s",
                            message_count,
                            parsed_message.get("type", "unknown"),
                        )

                    # Add to queue for non-blocking processing
                    try:
                        self._message_queue.put_nowait(parsed_message)
                    except asyncio.QueueFull:
                        logger.warning("Message queue full, dropping oldest message")
                        # Drop oldest message and add new one
                        try:
                            self._message_queue.get_nowait()
                            self._message_queue.put_nowait(parsed_message)
                        except asyncio.QueueEmpty:
                            pass

                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse WebSocket message: %s", e)
                except Exception:
                    logger.exception("Error handling WebSocket message")

    async def _handle_websocket_message(self, message: dict[str, Any]) -> None:
        """
        Handle incoming WebSocket messages from Bluefin.

        Args:
            message: Parsed WebSocket message
        """
        msg_type = message.get("type")
        channel = message.get("channel")

        if msg_type == "subscribed":
            logger.info(
                "Successfully subscribed to Bluefin WebSocket channel: %s", channel
            )
        elif msg_type == "error":
            logger.error(
                "Bluefin WebSocket error: %s", message.get("error", "Unknown error")
            )
        elif channel == "trades" and self.use_trade_aggregation:
            await self._handle_trade_message(message)
        elif channel == "candles" and not self.use_trade_aggregation:
            await self._handle_kline_message(message)
        else:
            logger.debug("Unhandled Bluefin WebSocket message type: %s", msg_type)

    async def _handle_trade_message(self, message: dict[str, Any]) -> None:
        """
        Handle trade messages for aggregation.

        Args:
            message: Trade message from WebSocket
        """
        try:
            data = message.get("data", {})

            # Convert price from 18 decimal format
            price = convert_from_18_decimal(data.get("price", "0"))
            size = convert_from_18_decimal(data.get("size", "0"))

            trade_data = {
                "price": price,
                "size": size,
                "side": data.get("side", "BUY"),
                "timestamp": datetime.fromtimestamp(
                    int(data.get("timestamp", 0)) / 1000, tz=UTC
                ),
                "trade_id": data.get("id", ""),
            }

            # Mark that we've received real WebSocket data
            if not self._websocket_data_received:
                self._websocket_data_received = True
                self._first_websocket_data_time = datetime.now(UTC)
                logger.info(
                    "First WebSocket market data received for %s via trade at $%s",
                    self.symbol,
                    price,
                )

            # Update last update time
            self._last_update = datetime.now(UTC)

            # Add to tick buffer for aggregation
            self._tick_buffer.append(trade_data)

            # Update price cache
            self._price_cache["price"] = price
            self._cache_timestamps["price"] = datetime.now(UTC)

            # Trigger candle building if needed
            if self._candle_builder_task is None or self._candle_builder_task.done():
                self._candle_builder_task = asyncio.create_task(
                    self._build_candles_from_trades()
                )

        except Exception:
            logger.exception("Error handling Bluefin trade message")

    async def _handle_kline_message(self, message: dict[str, Any]) -> None:
        """
        Handle kline/candle messages.

        Args:
            message: Kline message from WebSocket
        """
        try:
            data = message.get("data", {})

            # Parse candle data
            candle = self._parse_bluefin_candle(data)

            if self._validate_market_data(candle):
                # Update or add candle to cache
                if (
                    self._ohlcv_cache
                    and self._ohlcv_cache[-1].timestamp == candle.timestamp
                ):
                    # Update existing candle
                    self._ohlcv_cache[-1] = candle
                else:
                    # Add new candle
                    self._ohlcv_cache.append(candle)
                    # Maintain cache limit
                    if len(self._ohlcv_cache) > self.candle_limit:
                        self._ohlcv_cache = self._ohlcv_cache[-self.candle_limit :]

                # Update price cache
                self._price_cache["price"] = candle.close
                self._cache_timestamps["price"] = datetime.now(UTC)

                # Mark that we've received real WebSocket data
                if not self._websocket_data_received:
                    self._websocket_data_received = True
                    self._first_websocket_data_time = datetime.now(UTC)
                    logger.info(
                        "First WebSocket market data received for %s at $%s",
                        self.symbol,
                        candle.close,
                    )

                # Update last update time
                self._last_update = datetime.now(UTC)

                # Notify subscribers
                await self._notify_subscribers(candle)

        except Exception:
            logger.exception("Error handling Bluefin kline message")

    async def _build_candles_from_trades(self) -> None:
        """
        Build candles from trade data in the tick buffer.
        This is used for sub-minute intervals when trade aggregation is enabled.
        """
        try:
            if not self._tick_buffer:
                return

            # Get interval in seconds
            interval_seconds = self._interval_to_seconds(self.interval)

            # Group trades by interval
            current_time = datetime.now(UTC)
            interval_start = current_time.replace(microsecond=0)
            interval_start = interval_start - timedelta(
                seconds=interval_start.second % interval_seconds
            )

            # Process trades in the buffer
            trades_in_interval = []
            for trade in list(self._tick_buffer):
                if trade["timestamp"] >= interval_start:
                    trades_in_interval.append(trade)

            if trades_in_interval:
                # Build candle from trades
                prices = [t["price"] for t in trades_in_interval]
                volumes = [t["size"] for t in trades_in_interval]

                candle = MarketData(
                    symbol=self.symbol,
                    timestamp=interval_start,
                    open=prices[0],
                    high=max(prices),
                    low=min(prices),
                    close=prices[-1],
                    volume=sum(volumes),
                )

                if self._validate_market_data(candle):
                    # Update or add candle to cache
                    if (
                        self._ohlcv_cache
                        and self._ohlcv_cache[-1].timestamp == candle.timestamp
                    ):
                        # Update existing candle
                        self._ohlcv_cache[-1] = candle
                    else:
                        # Add new candle
                        self._ohlcv_cache.append(candle)
                        # Maintain cache limit
                        if len(self._ohlcv_cache) > self.candle_limit:
                            self._ohlcv_cache = self._ohlcv_cache[-self.candle_limit :]

                    # Notify subscribers
                    await self._notify_subscribers(candle)

        except Exception:
            logger.exception("Error building candles from trades")

    def _format_granularity_for_bluefin(self, interval: str) -> str:
        """
        Format interval for Bluefin API.

        Args:
            interval: Interval string (e.g., '1m', '5m', '1h')

        Returns:
            Interval in seconds as string
        """
        # Bluefin uses seconds for interval
        seconds = self._interval_to_seconds(interval)

        # Bluefin supported intervals (in seconds)
        supported = [
            60,
            300,
            900,
            1800,
            3600,
            14400,
            86400,
        ]  # 1m, 5m, 15m, 30m, 1h, 4h, 1d

        # Find closest supported interval
        if seconds not in supported:
            closest = min(supported, key=lambda x: abs(x - seconds))
            if closest != seconds:
                logger.warning(
                    "Interval %s (%ss) not supported by Bluefin, using %ss",
                    interval,
                    seconds,
                    closest,
                )
            return str(closest)

        return str(seconds)

    def _parse_bluefin_candle(self, candle_data: dict[str, Any]) -> MarketData:
        """
        Parse candle data from Bluefin format.

        Args:
            candle_data: Raw candle data from API

        Returns:
            MarketData object
        """
        # Bluefin uses different field names and 18 decimal format
        return MarketData(
            symbol=self.symbol,
            timestamp=datetime.fromtimestamp(
                int(candle_data.get("time", candle_data.get("timestamp", 0))) / 1000,
                tz=UTC,
            ),
            open=convert_from_18_decimal(candle_data.get("open", "0")),
            high=convert_from_18_decimal(candle_data.get("high", "0")),
            low=convert_from_18_decimal(candle_data.get("low", "0")),
            close=convert_from_18_decimal(candle_data.get("close", "0")),
            volume=convert_from_18_decimal(candle_data.get("volume", "0")),
        )
