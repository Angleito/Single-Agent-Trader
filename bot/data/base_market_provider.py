"""Abstract base class for unified market data providers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import aiohttp
import pandas as pd

from bot.config import settings
from bot.trading_types import MarketData

logger = logging.getLogger(__name__)


class AbstractMarketDataProvider(ABC):
    """
    Abstract base class for market data providers.

    Provides common functionality for all exchange-specific market data providers
    including caching, WebSocket management, and subscriber notifications.
    """

    def __init__(self, symbol: str | None = None, interval: str | None = None) -> None:
        """
        Initialize the market data provider.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD', 'ETH-PERP')
            interval: Candle interval (e.g., '1m', '5m', '1h')
        """
        self.symbol = symbol or settings.trading.symbol
        self.interval = interval or settings.trading.interval
        self.candle_limit = settings.data.candle_limit

        # Data cache with TTL
        self._ohlcv_cache: list[MarketData] = []
        self._price_cache: dict[str, Decimal] = {}
        self._orderbook_cache: dict[str, Any] = {}
        self._tick_cache: list[dict[str, Any]] = []
        self._cache_timestamps: dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=settings.data.data_cache_ttl_seconds)
        self._last_update: datetime | None = None

        # HTTP session for API calls
        self._session: aiohttp.ClientSession | None = None

        # WebSocket connection management
        self._ws_connection = None
        self._ws_task: asyncio.Task | None = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = settings.exchange.websocket_reconnect_attempts
        self._is_connected = False
        self._connection_lock = asyncio.Lock()

        # Non-blocking message processing
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._processing_task: asyncio.Task | None = None
        self._running = False

        # Subscribers for real-time updates
        self._subscribers: list[Callable[[MarketData], None | Any]] = []

        # Background tasks tracking
        self._background_tasks: set[asyncio.Task] = set()

        # Data validation settings
        self._max_price_deviation = 0.1  # 10% max price deviation
        self._min_volume = Decimal(0)

        # Track WebSocket data reception
        self._websocket_data_received = False
        self._first_websocket_data_time: datetime | None = None

        logger.info(
            "Initialized %s for %s at %s",
            self.__class__.__name__,
            self.symbol,
            self.interval,
        )

    @abstractmethod
    async def connect(self, fetch_historical: bool = True) -> None:
        """
        Establish connection to exchange data feeds.

        Args:
            fetch_historical: Whether to fetch historical data during connection
        """

    @abstractmethod
    async def fetch_historical_data(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        granularity: str | None = None,
    ) -> list[MarketData]:
        """
        Fetch historical OHLCV data from exchange REST API.

        Args:
            start_time: Start time for data
            end_time: End time for data
            granularity: Candle granularity

        Returns:
            List of MarketData objects
        """

    @abstractmethod
    async def fetch_latest_price(self) -> Decimal | None:
        """
        Fetch the latest price for the symbol from exchange REST API.

        Returns:
            Latest price as Decimal or None if unavailable
        """

    @abstractmethod
    async def _connect_websocket(self) -> None:
        """
        Establish WebSocket connection and handle messages.
        Must be implemented by exchange-specific providers.
        """

    @abstractmethod
    async def _handle_websocket_message(self, message: dict[str, Any]) -> None:
        """
        Handle incoming WebSocket messages.
        Must be implemented by exchange-specific providers.

        Args:
            message: Parsed WebSocket message
        """

    async def disconnect(self) -> None:
        """Disconnect from all data feeds and cleanup resources."""
        self._is_connected = False
        self._running = False

        # Stop message processing task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Stop WebSocket connection
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None

        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("Disconnected from market data feeds")

    async def _initialize_http_session(self) -> None:
        """Initialize HTTP session for API calls."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.exchange.api_timeout)
        )

    async def _start_websocket(self) -> None:
        """
        Start WebSocket connection for real-time data with auto-reconnect.
        """
        self._running = True

        # Start non-blocking message processor
        self._processing_task = asyncio.create_task(self._process_websocket_messages())

        # Start WebSocket handler
        self._ws_task = asyncio.create_task(self._websocket_handler())
        logger.info("WebSocket connection and message processor started")

        # Give WebSocket a moment to establish connection
        await asyncio.sleep(0.5)

    async def _websocket_handler(self) -> None:
        """
        Handle WebSocket connection with automatic reconnection.
        """
        while self._is_connected:
            try:
                await self._connect_websocket()
            except Exception:
                logger.exception("WebSocket connection error")
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    delay = min(
                        2**self._reconnect_attempts, 60
                    )  # Exponential backoff, max 60s
                    logger.info(
                        "Reconnecting in %ss (attempt %s/%s)",
                        delay,
                        self._reconnect_attempts,
                        self._max_reconnect_attempts,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        "Max reconnection attempts reached, stopping WebSocket"
                    )
                    break

    async def _process_websocket_messages(self) -> None:
        """
        Non-blocking WebSocket message processor using asyncio queue.

        Processes messages in background without blocking WebSocket reception.
        """
        while self._running:
            try:
                # Get message without blocking for too long
                message = await asyncio.wait_for(self._message_queue.get(), timeout=0.1)

                # Process message in background task to avoid blocking
                task = asyncio.create_task(
                    self._handle_websocket_message_async(message)
                )
                # Store reference to prevent garbage collection
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            except TimeoutError:
                # No message available, continue loop
                continue
            except Exception:
                logger.exception("Error in message processor")
                await asyncio.sleep(0.1)

    async def _handle_websocket_message_async(self, message: dict[str, Any]) -> None:
        """
        Handle WebSocket message asynchronously without blocking the queue processor.

        Args:
            message: Parsed WebSocket message
        """
        try:
            await self._handle_websocket_message(message)
        except Exception:
            logger.exception("Error in async message handler")

    def get_latest_ohlcv(self, limit: int | None = None) -> list[MarketData]:
        """
        Get the latest OHLCV data.

        Args:
            limit: Number of candles to return (default: all cached data)

        Returns:
            List of MarketData objects
        """
        if limit is None:
            return self._ohlcv_cache.copy()

        return self._ohlcv_cache[-limit:].copy()

    def get_latest_price(self) -> Decimal | None:
        """
        Get the most recent price from cache.

        Returns:
            Latest price or None if no data available
        """
        # Try price cache first
        if "price" in self._price_cache:
            return self._price_cache["price"]

        # Fall back to latest candle close price
        if self._ohlcv_cache:
            return self._ohlcv_cache[-1].close

        return None

    def to_dataframe(self, limit: int | None = None) -> pd.DataFrame:
        """
        Convert OHLCV data to pandas DataFrame for indicator calculations.

        Args:
            limit: Number of candles to include

        Returns:
            DataFrame with OHLCV columns
        """
        data = self.get_latest_ohlcv(limit)

        if not data:
            return pd.DataFrame()

        df_data = []
        for candle in data:
            df_data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                }
            )

        market_df = pd.DataFrame(df_data)
        return market_df.set_index("timestamp")

    def subscribe_to_updates(self, callback: Callable[[MarketData], None]) -> None:
        """
        Subscribe to real-time data updates.

        Args:
            callback: Function to call when new data arrives
        """
        self._subscribers.append(callback)
        logger.debug("Added subscriber: %s", callback.__name__)

    def unsubscribe_from_updates(self, callback: Callable[[MarketData], None]) -> None:
        """
        Unsubscribe from real-time data updates.

        Args:
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug("Removed subscriber: %s", callback.__name__)

    async def _notify_subscribers(self, data: MarketData) -> None:
        """
        Notify all subscribers of new data using non-blocking tasks.

        Args:
            data: New market data to broadcast
        """
        # Create tasks for all subscriber callbacks to run concurrently
        tasks = []
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Create task for async callback
                    task = asyncio.create_task(
                        self._safe_callback_async(callback, data)
                    )
                    tasks.append(task)
                else:
                    # Create task for sync callback
                    task = asyncio.create_task(self._safe_callback_sync(callback, data))
                    tasks.append(task)
            except Exception:
                logger.exception(
                    "Error creating subscriber task for %s", callback.__name__
                )

        # Don't wait for all tasks to complete - fire and forget for non-blocking behavior
        # Tasks will run in background without blocking the caller

    async def _safe_callback_async(
        self, callback: Callable[[MarketData], Any], data: MarketData
    ) -> None:
        """Safely execute async callback."""
        try:
            await callback(data)
        except Exception:
            logger.exception("Error in async subscriber callback %s", callback.__name__)

    async def _safe_callback_sync(
        self, callback: Callable[[MarketData], Any], data: MarketData
    ) -> None:
        """Safely execute sync callback in thread pool."""
        try:
            # Run sync callback in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, callback, data)
        except Exception:
            logger.exception("Error in sync subscriber callback %s", callback.__name__)

    def is_connected(self) -> bool:
        """
        Check if the data provider is connected and receiving data.

        Returns:
            True if connected and data is fresh
        """
        if not self._is_connected:
            return False

        # Check if we have any data at all (either historical or WebSocket)
        if not self._last_update:
            return False

        # Consider data stale if older than 2 minutes
        staleness_threshold = timedelta(minutes=2)
        return datetime.now(UTC) - self._last_update < staleness_threshold

    def has_websocket_data(self) -> bool:
        """
        Check if WebSocket is receiving real market data.

        Returns:
            True if WebSocket data has been received
        """
        return self._websocket_data_received

    async def wait_for_websocket_data(self, timeout: int = 30) -> bool:
        """
        Wait for WebSocket to start receiving real market data.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if data received, False if timeout
        """
        start_time = datetime.now(UTC)

        while not self._websocket_data_received:
            if (datetime.now(UTC) - start_time).total_seconds() > timeout:
                logger.warning("Timeout waiting for WebSocket data after %ss", timeout)
                return False

            # Check WebSocket connection status
            if self._ws_connection and hasattr(self._ws_connection, "state"):
                ws_state = self._ws_connection.state
                if ws_state != 1:  # Not OPEN
                    logger.warning("WebSocket not in OPEN state: %s", ws_state)

            await asyncio.sleep(0.1)

        return True

    def get_data_status(self) -> dict[str, Any]:
        """
        Get comprehensive status information about the data provider.

        Returns:
            Dictionary with status information
        """
        # Check WebSocket connection status more accurately
        ws_connected = False
        if self._ws_connection is not None:
            try:
                # Check if WebSocket is in OPEN state (1)
                ws_connected = (
                    hasattr(self._ws_connection, "state")
                    and self._ws_connection.state == 1
                )
            except Exception:
                ws_connected = False

        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "connected": self.is_connected(),
            "websocket_connected": ws_connected,
            "websocket_data_received": self._websocket_data_received,
            "first_websocket_data_time": self._first_websocket_data_time,
            "cached_candles": len(self._ohlcv_cache),
            "cached_ticks": len(self._tick_cache),
            "last_update": self._last_update,
            "latest_price": self.get_latest_price(),
            "subscribers": len(self._subscribers),
            "reconnect_attempts": self._reconnect_attempts,
            "cache_status": {
                key: self._is_cache_valid(key) for key in self._cache_timestamps
            },
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid based on TTL.

        Args:
            cache_key: Cache key to check

        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self._cache_timestamps:
            return False

        cache_age = datetime.now(UTC) - self._cache_timestamps[cache_key]
        return cache_age < self._cache_ttl

    def _interval_to_seconds(self, interval: str) -> int:
        """
        Convert interval string to seconds.

        Args:
            interval: Interval string (e.g., '1m', '5m', '1h', '30s')

        Returns:
            Interval in seconds
        """
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}

        if interval[-1] in multipliers:
            return int(interval[:-1]) * multipliers[interval[-1]]

        # Default to 1 minute
        return 60

    def _validate_market_data(self, data: MarketData) -> bool:
        """
        Validate market data for quality and integrity.

        Args:
            data: MarketData to validate

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Basic validation
            if data.open <= 0 or data.high <= 0 or data.low <= 0 or data.close <= 0:
                logger.warning("Invalid price data: negative or zero prices")
                return False

            if data.volume < self._min_volume:
                logger.warning("Invalid volume: %s < %s", data.volume, self._min_volume)
                return False

            # Price consistency checks
            if data.high < max(data.open, data.close) or data.low > min(
                data.open, data.close
            ):
                logger.warning(
                    "Inconsistent OHLC data: High=%s, Low=%s, Open=%s, Close=%s",
                    data.high,
                    data.low,
                    data.open,
                    data.close,
                )
                return False

            # Check for extreme price movements (compared to previous candle)
            if self._ohlcv_cache:
                last_price = self._ohlcv_cache[-1].close
                price_change = abs(data.close - last_price) / last_price
                if price_change > self._max_price_deviation:
                    logger.warning(
                        "Extreme price movement detected: %.2f%% change",
                        price_change * 100,
                    )
                    # Don't reject, but log for monitoring

            return True

        except Exception:
            logger.exception("Error validating market data")
            return False

    def _validate_data_sufficiency(self, candle_count: int) -> None:
        """
        Validate that we have sufficient data for reliable indicator calculations.

        Args:
            candle_count: Number of candles available
        """
        # VuManChu indicators need ~80 candles minimum for reliable calculation
        # RSI+MFI period=60, EMA ribbon max=34, plus buffer
        min_required = 100  # Conservative estimate with buffer
        optimal_required = 200  # Optimal for stable calculations

        if candle_count < min_required:
            logger.error(
                "Insufficient data for reliable indicator calculations! Have %s candles, need minimum %s. Indicators may produce unreliable signals or errors.",
                candle_count,
                min_required,
            )
        elif candle_count < optimal_required:
            logger.warning(
                "Suboptimal data for indicator calculations. Have %s candles, recommend %s for best accuracy. Early indicator values may be less reliable.",
                candle_count,
                optimal_required,
            )
        else:
            logger.info(
                "Sufficient data for reliable indicator calculations: %s candles",
                candle_count,
            )

    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        self._ohlcv_cache.clear()
        self._price_cache.clear()
        self._orderbook_cache.clear()
        self._tick_cache.clear()
        self._cache_timestamps.clear()
        self._websocket_data_received = False
        self._first_websocket_data_time = None
        logger.info("All cached data cleared")

    def get_tick_data(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get recent tick/trade data.

        Args:
            limit: Maximum number of ticks to return

        Returns:
            List of tick data dictionaries
        """
        if limit is None:
            return self._tick_cache.copy()

        return self._tick_cache[-limit:].copy()

    async def fetch_orderbook(self, level: int = 2) -> dict[str, Any] | None:
        """
        Fetch order book data from exchange REST API.
        Default implementation returns None. Override in exchange-specific providers.

        Args:
            level: Order book level (1, 2, or 3)

        Returns:
            Order book data or None if unavailable
        """
        return None
