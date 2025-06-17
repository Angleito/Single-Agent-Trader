"""Bluefin market data provider for perpetual futures on Sui network."""

import asyncio
import json
import logging
import os
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import websockets
from websockets.client import WebSocketClientProtocol

from ..config import settings
from ..types import MarketData

logger = logging.getLogger(__name__)


class BluefinMarketDataProvider:
    """
    Market data provider for Bluefin perpetual futures.
    
    Handles real-time and historical market data from Bluefin DEX,
    providing OHLCV data for perpetual futures contracts on Sui.
    
    Features:
    - Mock data generation for paper trading
    - Historical data fetching via Bluefin service
    - Real-time price simulation
    - Perpetual futures symbol mapping
    """
    
    def __init__(self, symbol: str = None, interval: str = None):
        """
        Initialize the Bluefin market data provider.
        
        Args:
            symbol: Trading symbol (e.g., 'SUI-PERP', 'ETH-PERP')
            interval: Candle interval (e.g., '1m', '5m', '1h')
        """
        self.symbol = symbol or self._convert_symbol(settings.trading.symbol)
        self.interval = interval or settings.trading.interval
        self.candle_limit = settings.data.candle_limit
        
        # Data cache
        self._ohlcv_cache: list[MarketData] = []
        self._price_cache: dict[str, Any] = {}
        self._orderbook_cache: dict[str, Any] = {}
        self._tick_cache: list[dict[str, Any]] = []
        self._cache_timestamps: dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=settings.data.data_cache_ttl_seconds)
        self._last_update: datetime | None = None
        
        # HTTP session for API calls
        self._session: aiohttp.ClientSession | None = None
        
        # Connection state
        self._is_connected = False
        self._connection_lock = asyncio.Lock()
        
        # Subscribers for real-time updates
        self._subscribers: list[Callable] = []
        
        # Data validation settings
        self._max_price_deviation = 0.1  # 10% max price deviation
        self._min_volume = Decimal("0")
        
        # Extended history tracking
        self._extended_history_mode = False
        self._extended_history_limit = 2000  # Max candles to keep in extended mode (e.g., 8 hours of 15s candles)
        
        # Mock data generation for paper trading
        # Check if we should use real data even in dry run mode
        use_real_data = os.getenv("BLUEFIN_USE_REAL_DATA", "false").lower() == "true"
        # Use real data by default unless explicitly in dry_run mode without real data
        self._use_mock_data = settings.system.dry_run and not use_real_data
        self._mock_price_update_task: asyncio.Task | None = None
        
        # WebSocket connection
        self._ws: WebSocketClientProtocol | None = None
        self._ws_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._ws_connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5  # seconds
        
        # Tick data buffer for candle building
        self._tick_buffer: deque[dict[str, Any]] = deque(maxlen=10000)
        self._candle_builder_task: asyncio.Task | None = None
        self._last_candle_timestamp: datetime | None = None
        
        # Bluefin API configuration
        self._api_base_url = "https://dapi.api.sui-prod.bluefin.io"
        self._ws_url = "wss://dapi.api.sui-prod.bluefin.io"
        self._notification_ws_url = "wss://notifications.api.sui-prod.bluefin.io/"
        
        logger.info(
            f"Initialized BluefinMarketDataProvider for {self.symbol} at {self.interval}"
        )
        if self._use_mock_data:
            logger.info("Using mock data for Bluefin market data")
        else:
            logger.info("Using real Bluefin market data via WebSocket")
    
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
    
    async def connect(self, fetch_historical: bool = True) -> None:
        """
        Establish connection to Bluefin data feeds.
        
        Args:
            fetch_historical: Whether to fetch historical data during connection
        """
        try:
            # Initialize HTTP session
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=settings.exchange.api_timeout)
            )
            
            if self._use_mock_data:
                logger.info("Starting mock data generation")
                # Start mock price update task
                self._mock_price_update_task = asyncio.create_task(
                    self._mock_price_updates()
                )
            else:
                logger.info("Connecting to Bluefin WebSocket for real-time data")
                # Start WebSocket connection
                await self._connect_websocket()
                
                # Start candle builder task
                self._candle_builder_task = asyncio.create_task(
                    self._build_candles_from_ticks()
                )
            
            # Fetch initial historical data
            if fetch_historical:
                try:
                    # Fetch 8 hours of historical data on startup
                    end_time = datetime.now(UTC)
                    start_time = end_time - timedelta(hours=8)
                    
                    # Calculate expected number of candles for logging
                    interval_seconds = self._interval_to_seconds(self.interval)
                    expected_candles = int((8 * 3600) / interval_seconds)
                    
                    logger.info(
                        f"Fetching 8 hours of historical data on startup "
                        f"(~{expected_candles} candles at {self.interval} interval)"
                    )
                    
                    await self.fetch_historical_data(start_time=start_time, end_time=end_time)
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch historical data: {e}"
                    )
            
            # Try to get current price
            try:
                current_price = await self.fetch_latest_price()
                if current_price:
                    logger.info(f"Successfully fetched current price: ${current_price}")
            except Exception as e:
                logger.warning(f"Could not fetch current price: {e}")
            
            self._is_connected = True
            logger.info("Successfully connected to Bluefin market data feeds")
            
        except Exception as e:
            logger.error(f"Failed to connect to Bluefin market data: {e}")
            self._is_connected = False
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from all data feeds and cleanup resources."""
        self._is_connected = False
        
        # Stop mock price update task
        if self._mock_price_update_task and not self._mock_price_update_task.done():
            self._mock_price_update_task.cancel()
            try:
                await self._mock_price_update_task
            except asyncio.CancelledError:
                pass
        
        # Stop WebSocket tasks
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        if self._candle_builder_task and not self._candle_builder_task.done():
            self._candle_builder_task.cancel()
            try:
                await self._candle_builder_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connection
        if hasattr(self, '_ws_client') and self._ws_client:
            await self._ws_client.disconnect()
            self._ws_client = None
        elif self._ws:
            await self._ws.close()
            self._ws = None
        
        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
        
        logger.info("Disconnected from Bluefin market data feeds")
    
    async def fetch_historical_data(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        granularity: str | None = None,
    ) -> list[MarketData]:
        """
        Fetch historical OHLCV data from Bluefin.
        
        Args:
            start_time: Start time for data
            end_time: End time for data
            granularity: Candle granularity
            
        Returns:
            List of MarketData objects
        """
        granularity = granularity or self.interval
        end_time = end_time or datetime.now(UTC)
        
        # Calculate start time based on interval and limit
        interval_seconds = self._interval_to_seconds(granularity)
        
        # If start_time is provided (e.g., 8 hours for startup), calculate required candles
        if start_time:
            time_range_seconds = (end_time - start_time).total_seconds()
            required_candles = int(time_range_seconds / interval_seconds)
            # Ensure we have at least the configured limit
            required_candles = max(required_candles, self.candle_limit, 200)
        else:
            # Default behavior: use candle_limit
            required_candles = max(self.candle_limit, 200)
            default_start = end_time - timedelta(seconds=interval_seconds * required_candles)
            start_time = default_start
        
        logger.info(
            f"Fetching historical data for {self.symbol} from {start_time} to {end_time}"
        )
        
        try:
            if self._use_mock_data:
                # Generate mock historical data
                historical_data = self._generate_mock_historical_data(
                    self.symbol, granularity, required_candles
                )
            else:
                # Fetch real data from Bluefin API
                historical_data = await self._fetch_bluefin_candles(
                    self.symbol, granularity, start_time, end_time, required_candles
                )
            
            # Update cache - keep all historical data if we fetched more than candle_limit
            if len(historical_data) > self.candle_limit:
                # We fetched extended historical data (e.g., 8 hours on startup)
                self._extended_history_mode = True
                # Limit to extended history limit to prevent memory issues
                if len(historical_data) > self._extended_history_limit:
                    self._ohlcv_cache = historical_data[-self._extended_history_limit:]
                    logger.info(f"Loaded {len(self._ohlcv_cache)} historical candles (limited to {self._extended_history_limit})")
                else:
                    self._ohlcv_cache = historical_data
                    logger.info(f"Loaded {len(self._ohlcv_cache)} historical candles (extended history)")
            else:
                # Normal case: limit to candle_limit
                self._extended_history_mode = False
                self._ohlcv_cache = historical_data[-self.candle_limit:]
                logger.info(f"Loaded {len(self._ohlcv_cache)} historical candles")
            
            self._last_update = datetime.now(UTC)
            self._cache_timestamps["ohlcv"] = self._last_update
            
            # Validate data sufficiency
            self._validate_data_sufficiency(len(self._ohlcv_cache))
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            if self._ohlcv_cache:
                logger.info("Using cached historical data")
                return self._ohlcv_cache
            raise
    
    async def fetch_latest_price(self) -> Decimal | None:
        """
        Fetch the latest price for the symbol.
        
        Returns:
            Latest price as Decimal or None if unavailable
        """
        # Check cache first
        if self._is_cache_valid("price"):
            return self._price_cache.get("price")
        
        try:
            if self._use_mock_data:
                # Use mock price based on symbol
                base_prices = {
                    "ETH-PERP": 2500.0,
                    "BTC-PERP": 45000.0,
                    "SUI-PERP": 3.50,
                    "SOL-PERP": 125.0,
                }
                
                base_price = base_prices.get(self.symbol, 2500.0)
                # Add some random variation
                import random
                variation = random.uniform(-0.01, 0.01)  # ±1%
                price = Decimal(str(base_price * (1 + variation)))
                
                # Update cache
                self._price_cache["price"] = price
                self._cache_timestamps["price"] = datetime.now(UTC)
                return price
            else:
                # Fetch real-time price from Bluefin API
                return await self._fetch_bluefin_ticker_price()
                
        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")
        
        # Fall back to cached OHLCV data
        return self.get_latest_price()
    
    async def fetch_orderbook(self, level: int = 2) -> dict[str, Any] | None:
        """
        Fetch order book data.
        
        Args:
            level: Order book level
            
        Returns:
            Order book data or None if unavailable
        """
        # Check cache first
        cache_key = f"orderbook_l{level}"
        if self._is_cache_valid(cache_key):
            return self._orderbook_cache.get(cache_key)
        
        try:
            if self._use_mock_data:
                # Generate mock orderbook
                current_price = await self.fetch_latest_price()
                if not current_price:
                    return None
                
                orderbook = self._generate_mock_orderbook(current_price, level)
                
                # Update cache
                self._orderbook_cache[cache_key] = orderbook
                self._cache_timestamps[cache_key] = datetime.now(UTC)
                
                return orderbook
            else:
                # In production, fetch from Bluefin service
                pass
                
        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return None
    
    def get_latest_ohlcv(self, limit: int | None = None) -> list[MarketData]:
        """
        Get the latest OHLCV data.
        
        Args:
            limit: Number of candles to return
            
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
        
        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def subscribe_to_updates(self, callback: Callable[[MarketData], None]) -> None:
        """
        Subscribe to real-time data updates.
        
        Args:
            callback: Function to call when new data arrives
        """
        self._subscribers.append(callback)
        logger.debug(f"Added subscriber: {callback.__name__}")
    
    def unsubscribe_from_updates(self, callback: Callable[[MarketData], None]) -> None:
        """
        Unsubscribe from real-time data updates.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug(f"Removed subscriber: {callback.__name__}")
    
    async def _notify_subscribers(self, data: MarketData) -> None:
        """
        Notify all subscribers of new data.
        
        Args:
            data: New market data to broadcast
        """
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in subscriber callback {callback.__name__}: {e}")
    
    def is_connected(self) -> bool:
        """
        Check if the data provider is connected and receiving data.
        
        Returns:
            True if connected and data is fresh
        """
        if not self._is_connected:
            return False
        
        # Check if we have any data at all
        if not self._last_update:
            return False
        
        # Consider data stale if older than 2 minutes
        staleness_threshold = timedelta(minutes=2)
        return datetime.now(UTC) - self._last_update < staleness_threshold
    
    def get_data_status(self) -> dict[str, Any]:
        """
        Get comprehensive status information about the data provider.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "symbol": self.symbol,
            "interval": self.interval,
            "connected": self.is_connected(),
            "mock_data": self._use_mock_data,
            "cached_candles": len(self._ohlcv_cache),
            "cached_ticks": len(self._tick_cache),
            "last_update": self._last_update,
            "latest_price": self.get_latest_price(),
            "subscribers": len(self._subscribers),
            "extended_history_mode": self._extended_history_mode,
            "cache_status": {
                key: self._is_cache_valid(key) for key in self._cache_timestamps.keys()
            },
        }
        
        # Add WebSocket client status if available
        if hasattr(self, '_ws_client') and self._ws_client:
            ws_status = self._ws_client.get_status()
            status["ws_client"] = {
                "connected": ws_status.get("connected", False),
                "candles": ws_status.get("candles_buffered", 0),
                "ticks": ws_status.get("ticks_buffered", 0),
                "messages": ws_status.get("message_count", 0),
            }
        
        return status
    
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
    
    def has_websocket_data(self) -> bool:
        """
        Check if we have WebSocket data available.
        
        Returns:
            True if WebSocket data is available
        """
        if hasattr(self, '_ws_client') and self._ws_client:
            return len(self._ws_client.get_candles()) > 0
        return self._ws_connected and not self._use_mock_data
    
    def get_connection_status(self) -> str:
        """
        Get current connection status.
        
        Returns:
            Connection status string
        """
        if self._is_connected:
            if self._use_mock_data:
                return "Connected (Mock Data)"
            else:
                return "Connected (Live Data)"
        return "Disconnected"
    
    async def _on_websocket_candle(self, candle: MarketData) -> None:
        """
        Callback for when a new candle is received from WebSocket.
        
        Args:
            candle: New MarketData candle
        """
        # Add candle to cache
        self._ohlcv_cache.append(candle)
        
        # Maintain cache size based on mode
        if self._extended_history_mode:
            # In extended history mode, keep more candles
            if len(self._ohlcv_cache) > self._extended_history_limit:
                self._ohlcv_cache = self._ohlcv_cache[-self._extended_history_limit:]
        else:
            # Normal mode: maintain candle_limit
            if len(self._ohlcv_cache) > self.candle_limit:
                self._ohlcv_cache = self._ohlcv_cache[-self.candle_limit:]
        
        # Update last update time
        self._last_update = datetime.now(UTC)
        self._cache_timestamps["ohlcv"] = self._last_update
        
        # Update price cache
        self._price_cache["price"] = candle.close
        self._cache_timestamps["price"] = datetime.now(UTC)
        
        # Notify subscribers
        await self._notify_subscribers(candle)
        
        logger.debug(f"Received new candle: {candle.symbol} @ {candle.close}")
    
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
    
    def _generate_mock_historical_data(
        self, symbol: str, interval: str, limit: int
    ) -> list[MarketData]:
        """Generate mock historical data for paper trading."""
        import random
        
        # Base price for different symbols
        base_prices = {
            "ETH-PERP": 2500.0,
            "BTC-PERP": 45000.0,
            "SUI-PERP": 3.50,
            "SOL-PERP": 125.0,
        }
        
        base_price = base_prices.get(symbol, 2500.0)
        
        # Convert interval to seconds
        interval_seconds = self._interval_to_seconds(interval)
        
        data = []
        current_time = datetime.now(UTC)
        current_price = base_price
        
        for i in range(limit):
            # Generate realistic OHLCV data
            volatility = 0.02  # 2% volatility
            change = random.uniform(-volatility, volatility)
            
            open_price = current_price
            close_price = open_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
            volume = random.uniform(100, 10000)
            
            timestamp = current_time - timedelta(seconds=interval_seconds * (limit - i - 1))
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high_price, 2))),
                low=Decimal(str(round(low_price, 2))),
                close=Decimal(str(round(close_price, 2))),
                volume=Decimal(str(round(volume, 2))),
            )
            
            data.append(market_data)
            current_price = close_price
        
        logger.debug(f"Generated {len(data)} mock candles for {symbol}")
        return data
    
    def _generate_mock_orderbook(
        self, current_price: Decimal, level: int
    ) -> dict[str, Any]:
        """Generate mock orderbook data."""
        import random
        
        bids = []
        asks = []
        
        # Generate bids (below current price)
        for i in range(level * 5):
            price_offset = Decimal(str(0.01 * (i + 1)))  # 0.01, 0.02, 0.03...
            bid_price = current_price - price_offset
            bid_size = Decimal(str(random.uniform(0.1, 10.0)))
            bids.append((bid_price, bid_size))
        
        # Generate asks (above current price)
        for i in range(level * 5):
            price_offset = Decimal(str(0.01 * (i + 1)))
            ask_price = current_price + price_offset
            ask_size = Decimal(str(random.uniform(0.1, 10.0)))
            asks.append((ask_price, ask_size))
        
        return {
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now(UTC),
        }
    
    async def _mock_price_updates(self) -> None:
        """Generate mock real-time price updates."""
        import random
        
        while self._is_connected:
            try:
                # Update price every 1-5 seconds
                await asyncio.sleep(random.uniform(1, 5))
                
                if self._ohlcv_cache:
                    last_candle = self._ohlcv_cache[-1]
                    
                    # Generate new price with small variation
                    variation = random.uniform(-0.001, 0.001)  # ±0.1%
                    new_price = last_candle.close * (1 + Decimal(str(variation)))
                    
                    # Update price cache
                    self._price_cache["price"] = new_price
                    self._cache_timestamps["price"] = datetime.now(UTC)
                    
                    # Check if we need to create a new candle
                    interval_seconds = self._interval_to_seconds(self.interval)
                    time_since_last = (datetime.now(UTC) - last_candle.timestamp).total_seconds()
                    
                    if time_since_last >= interval_seconds:
                        # Create new candle
                        new_candle = MarketData(
                            symbol=self.symbol,
                            timestamp=datetime.now(UTC),
                            open=new_price,
                            high=new_price,
                            low=new_price,
                            close=new_price,
                            volume=Decimal(str(random.uniform(100, 1000))),
                        )
                        
                        self._ohlcv_cache.append(new_candle)
                        # Keep cache size limited based on mode
                        if self._extended_history_mode:
                            if len(self._ohlcv_cache) > self._extended_history_limit:
                                self._ohlcv_cache = self._ohlcv_cache[-self._extended_history_limit:]
                        else:
                            if len(self._ohlcv_cache) > self.candle_limit:
                                self._ohlcv_cache = self._ohlcv_cache[-self.candle_limit:]
                        
                        # Notify subscribers
                        await self._notify_subscribers(new_candle)
                    else:
                        # Update existing candle
                        updated_candle = MarketData(
                            symbol=last_candle.symbol,
                            timestamp=last_candle.timestamp,
                            open=last_candle.open,
                            high=max(last_candle.high, new_price),
                            low=min(last_candle.low, new_price),
                            close=new_price,
                            volume=last_candle.volume + Decimal(str(random.uniform(1, 10))),
                        )
                        
                        self._ohlcv_cache[-1] = updated_candle
                        await self._notify_subscribers(updated_candle)
                    
                    self._last_update = datetime.now(UTC)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in mock price update: {e}")
    
    async def _fetch_bluefin_ticker_price(self) -> Decimal | None:
        """
        Fetch real-time ticker price from Bluefin API.
        
        Returns:
            Current price or None if unavailable
        """
        if not self._session:
            raise RuntimeError("HTTP session not initialized")
            
        try:
            # Make API request for ticker data
            url = f"{self._api_base_url}/ticker24hr"
            params = {"symbol": self.symbol}
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Bluefin ticker API error {response.status}: {error_text}")
                    return None
                    
                data = await response.json()
                
                # Extract last price from ticker data
                if isinstance(data, dict):
                    last_price = data.get("lastPrice", data.get("last", data.get("price")))
                    if last_price:
                        price = Decimal(str(last_price))
                        # Update cache
                        self._price_cache["price"] = price
                        self._cache_timestamps["price"] = datetime.now(UTC)
                        return price
                elif isinstance(data, list) and data:
                    # If multiple tickers returned, find our symbol
                    for ticker in data:
                        if ticker.get("symbol") == self.symbol:
                            last_price = ticker.get("lastPrice", ticker.get("last", ticker.get("price")))
                            if last_price:
                                price = Decimal(str(last_price))
                                # Update cache
                                self._price_cache["price"] = price
                                self._cache_timestamps["price"] = datetime.now(UTC)
                                return price
                
                logger.warning(f"No price data found in ticker response: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Bluefin ticker price: {e}")
            return None
    
    async def _fetch_bluefin_candles(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int
    ) -> list[MarketData]:
        """
        Fetch real candle data from Bluefin API.
        
        Args:
            symbol: Trading symbol (e.g., 'SUI-PERP')
            interval: Candle interval
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of candles
            
        Returns:
            List of MarketData objects
        """
        if not self._session:
            raise RuntimeError("HTTP session not initialized")
            
        try:
            # Convert interval to Bluefin format
            interval_map = {
                "1m": "1",
                "3m": "3", 
                "5m": "5",
                "15m": "15",
                "30m": "30",
                "1h": "60",
                "2h": "120",
                "4h": "240",
                "1d": "1D",
                "1w": "1W"
            }
            
            bluefin_interval = interval_map.get(interval, "5")
            
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "interval": bluefin_interval,
                "limit": str(limit),
                "startTime": str(int(start_time.timestamp() * 1000)),
                "endTime": str(int(end_time.timestamp() * 1000))
            }
            
            # Make API request to klines endpoint
            url = f"{self._api_base_url}/klines"
            logger.info(f"Fetching candles from Bluefin API: {url} with params: {params}")
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Bluefin API error {response.status}: {error_text}")
                    # Fall back to mock data on API error
                    return self._generate_mock_historical_data(symbol, interval, limit)
                    
                data = await response.json()
                
                # Parse Bluefin candle data
                candles = []
                for candle in data:
                    # Expected format: [timestamp, open, high, low, close, volume]
                    if isinstance(candle, list) and len(candle) >= 6:
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(candle[0] / 1000, UTC),
                            open=Decimal(str(candle[1])),
                            high=Decimal(str(candle[2])),
                            low=Decimal(str(candle[3])),
                            close=Decimal(str(candle[4])),
                            volume=Decimal(str(candle[5]))
                        )
                        candles.append(market_data)
                    elif isinstance(candle, dict):
                        # Alternative format with keys
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(
                                candle.get("timestamp", candle.get("time", 0)) / 1000, UTC
                            ),
                            open=Decimal(str(candle.get("open", 0))),
                            high=Decimal(str(candle.get("high", 0))),
                            low=Decimal(str(candle.get("low", 0))),
                            close=Decimal(str(candle.get("close", 0))),
                            volume=Decimal(str(candle.get("volume", 0)))
                        )
                        candles.append(market_data)
                
                logger.info(f"Successfully fetched {len(candles)} candles from Bluefin")
                return candles
                
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching Bluefin candles: {e}")
            # Fall back to mock data on network error
            return self._generate_mock_historical_data(symbol, interval, limit)
        except Exception as e:
            logger.error(f"Unexpected error fetching Bluefin candles: {e}")
            # Fall back to mock data on any error
            return self._generate_mock_historical_data(symbol, interval, limit)
    
    def _validate_data_sufficiency(self, candle_count: int) -> None:
        """
        Validate that we have sufficient data for reliable indicator calculations.
        
        Args:
            candle_count: Number of candles available
        """
        # VuManChu indicators need ~80 candles minimum
        min_required = 100
        optimal_required = 200
        
        if candle_count < min_required:
            logger.error(
                f"Insufficient data for reliable indicator calculations! "
                f"Have {candle_count} candles, need minimum {min_required}."
            )
        elif candle_count < optimal_required:
            logger.warning(
                f"Suboptimal data for indicator calculations. "
                f"Have {candle_count} candles, recommend {optimal_required}."
            )
        else:
            logger.info(
                f"Sufficient data for reliable indicator calculations: {candle_count} candles"
            )
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._ohlcv_cache.clear()
        self._price_cache.clear()
        self._orderbook_cache.clear()
        self._tick_cache.clear()
        self._cache_timestamps.clear()
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
    
    async def _connect_websocket(self) -> None:
        """Establish WebSocket connection to Bluefin."""
        try:
            # Import and use BluefinWebSocketClient for proper ticker handling
            from .bluefin_websocket import BluefinWebSocketClient
            
            # Create WebSocket client with candle update callback
            self._ws_client = BluefinWebSocketClient(
                symbol=self.symbol,
                interval=self.interval,
                candle_limit=self.candle_limit,
                on_candle_update=self._on_websocket_candle
            )
            
            # Connect to WebSocket
            await self._ws_client.connect()
            self._ws_connected = True
            self._reconnect_attempts = 0
            
            logger.info("Successfully connected to Bluefin WebSocket")
            
        except Exception as e:
            logger.error(f"Failed to connect to Bluefin WebSocket: {e}")
            self._ws_connected = False
            await self._schedule_reconnect()
    
    async def _subscribe_to_market_data(self) -> None:
        """Subscribe to market data updates for the current symbol."""
        if not self._ws:
            return
        
        # Subscribe to globalUpdates room for the symbol
        subscription_message = [
            "SUBSCRIBE",
            [{"e": "globalUpdates", "p": self.symbol}]
        ]
        
        await self._ws.send(json.dumps(subscription_message))
        logger.info(f"Subscribed to market data for {self.symbol}")
    
    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        if not self._ws:
            return
        
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in WebSocket message: {message}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._ws_connected = False
            await self._schedule_reconnect()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self._ws_connected = False
            await self._schedule_reconnect()
    
    async def _process_websocket_message(self, data: Any) -> None:
        """Process a WebSocket message."""
        if isinstance(data, list) and len(data) > 1:
            event_type = data[0]
            event_data = data[1]
            
            if event_type == "MarketDataUpdate":
                await self._process_market_data_update(event_data)
            elif event_type == "RecentTrades":
                await self._process_recent_trades(event_data)
            elif event_type == "OrderbookUpdate":
                await self._process_orderbook_update(event_data)
            elif event_type == "MarketHealth":
                await self._process_market_health(event_data)
            else:
                logger.debug(f"Unhandled WebSocket event type: {event_type}")
    
    async def _process_market_data_update(self, data: dict[str, Any]) -> None:
        """Process market data update event."""
        try:
            # Extract price and volume data
            if "lastPrice" in data:
                price = Decimal(str(data["lastPrice"]))
                self._price_cache["price"] = price
                self._cache_timestamps["price"] = datetime.now(UTC)
                
                # Add to tick buffer for candle building
                tick_data = {
                    "timestamp": datetime.now(UTC),
                    "price": price,
                    "volume": Decimal(str(data.get("volume", "0"))),
                    "symbol": self.symbol
                }
                self._tick_buffer.append(tick_data)
                
                logger.debug(f"Market data update: {self.symbol} @ {price}")
                
        except Exception as e:
            logger.error(f"Error processing market data update: {e}")
    
    async def _process_recent_trades(self, data: dict[str, Any]) -> None:
        """Process recent trades event."""
        try:
            trades = data.get("trades", [])
            for trade in trades:
                if isinstance(trade, dict):
                    tick_data = {
                        "timestamp": datetime.fromtimestamp(
                            trade.get("timestamp", datetime.now(UTC).timestamp())
                        ),
                        "price": Decimal(str(trade.get("price", "0"))),
                        "volume": Decimal(str(trade.get("size", "0"))),
                        "side": trade.get("side", ""),
                        "symbol": self.symbol
                    }
                    self._tick_buffer.append(tick_data)
                    
                    # Update latest price
                    self._price_cache["price"] = tick_data["price"]
                    self._cache_timestamps["price"] = datetime.now(UTC)
                    
        except Exception as e:
            logger.error(f"Error processing recent trades: {e}")
    
    async def _process_orderbook_update(self, data: dict[str, Any]) -> None:
        """Process orderbook update event."""
        try:
            # Cache orderbook data
            self._orderbook_cache["orderbook_l2"] = {
                "bids": data.get("bids", []),
                "asks": data.get("asks", []),
                "timestamp": datetime.now(UTC)
            }
            self._cache_timestamps["orderbook_l2"] = datetime.now(UTC)
            
        except Exception as e:
            logger.error(f"Error processing orderbook update: {e}")
    
    async def _process_market_health(self, data: dict[str, Any]) -> None:
        """Process market health event."""
        # Log market health information
        logger.debug(f"Market health update: {data}")
    
    async def _build_candles_from_ticks(self) -> None:
        """Build OHLCV candles from tick data."""
        interval_seconds = self._interval_to_seconds(self.interval)
        
        while self._is_connected:
            try:
                await asyncio.sleep(1)  # Check every second
                
                if not self._tick_buffer:
                    continue
                
                # Get current candle timestamp
                current_time = datetime.now(UTC)
                candle_timestamp = self._get_candle_timestamp(current_time, interval_seconds)
                
                # Check if we need to create a new candle
                if self._last_candle_timestamp != candle_timestamp:
                    # Build candle from ticks in the previous period
                    if self._last_candle_timestamp:
                        candle = self._build_candle_from_ticks(
                            self._last_candle_timestamp, 
                            candle_timestamp
                        )
                        if candle:
                            self._ohlcv_cache.append(candle)
                            # Keep cache size limited based on mode
                            if self._extended_history_mode:
                                if len(self._ohlcv_cache) > self._extended_history_limit:
                                    self._ohlcv_cache = self._ohlcv_cache[-self._extended_history_limit:]
                            else:
                                if len(self._ohlcv_cache) > self.candle_limit:
                                    self._ohlcv_cache = self._ohlcv_cache[-self.candle_limit:]
                            
                            # Notify subscribers
                            await self._notify_subscribers(candle)
                            self._last_update = datetime.now(UTC)
                    
                    self._last_candle_timestamp = candle_timestamp
                
                # Update current candle with latest tick
                if self._ohlcv_cache and self._tick_buffer:
                    self._update_current_candle()
                    
            except Exception as e:
                logger.error(f"Error building candles from ticks: {e}")
    
    def _get_candle_timestamp(self, time: datetime, interval_seconds: int) -> datetime:
        """Get the candle timestamp for a given time."""
        timestamp_seconds = int(time.timestamp())
        candle_seconds = (timestamp_seconds // interval_seconds) * interval_seconds
        return datetime.fromtimestamp(candle_seconds, UTC)
    
    def _build_candle_from_ticks(
        self, start_time: datetime, end_time: datetime
    ) -> MarketData | None:
        """Build a candle from ticks in the given time range."""
        # Filter ticks in the time range
        ticks_in_range = [
            tick for tick in self._tick_buffer
            if start_time <= tick["timestamp"] < end_time
        ]
        
        if not ticks_in_range:
            # No ticks in this period, use last known price if available
            if self._ohlcv_cache:
                last_candle = self._ohlcv_cache[-1]
                return MarketData(
                    symbol=self.symbol,
                    timestamp=start_time,
                    open=last_candle.close,
                    high=last_candle.close,
                    low=last_candle.close,
                    close=last_candle.close,
                    volume=Decimal("0")
                )
            return None
        
        # Sort ticks by timestamp
        ticks_in_range.sort(key=lambda x: x["timestamp"])
        
        # Build candle
        prices = [tick["price"] for tick in ticks_in_range]
        volumes = [tick["volume"] for tick in ticks_in_range]
        
        return MarketData(
            symbol=self.symbol,
            timestamp=start_time,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(volumes)
        )
    
    def _update_current_candle(self) -> None:
        """Update the current candle with latest tick data."""
        if not self._ohlcv_cache or not self._tick_buffer:
            return
        
        current_candle = self._ohlcv_cache[-1]
        latest_tick = self._tick_buffer[-1]
        
        # Update candle with latest tick
        updated_candle = MarketData(
            symbol=current_candle.symbol,
            timestamp=current_candle.timestamp,
            open=current_candle.open,
            high=max(current_candle.high, latest_tick["price"]),
            low=min(current_candle.low, latest_tick["price"]),
            close=latest_tick["price"],
            volume=current_candle.volume + latest_tick.get("volume", Decimal("0"))
        )
        
        self._ohlcv_cache[-1] = updated_candle
    
    async def _schedule_reconnect(self) -> None:
        """Schedule a WebSocket reconnection attempt."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached, giving up")
            return
        
        self._reconnect_attempts += 1
        delay = self._reconnect_delay * self._reconnect_attempts
        
        logger.info(f"Scheduling reconnection attempt {self._reconnect_attempts} in {delay}s")
        
        self._reconnect_task = asyncio.create_task(self._reconnect_after_delay(delay))
    
    async def _reconnect_after_delay(self, delay: float) -> None:
        """Reconnect after a delay."""
        await asyncio.sleep(delay)
        await self._connect_websocket()


class BluefinMarketDataClient:
    """
    High-level client for Bluefin market data operations.
    
    Provides a simplified interface to the BluefinMarketDataProvider
    with additional convenience methods and error handling.
    """
    
    def __init__(self, symbol: str = None, interval: str = None):
        """
        Initialize the Bluefin market data client.
        
        Args:
            symbol: Trading symbol (e.g., 'SUI-PERP')
            interval: Candle interval (e.g., '1m', '5m', '1h')
        """
        self.provider = BluefinMarketDataProvider(symbol, interval)
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Connect to market data feeds."""
        if not self._initialized:
            await self.provider.connect()
            self._initialized = True
            logger.info("BluefinMarketDataClient connected successfully")
    
    async def disconnect(self) -> None:
        """Disconnect from market data feeds."""
        if self._initialized:
            await self.provider.disconnect()
            self._initialized = False
            logger.info("BluefinMarketDataClient disconnected")
    
    async def get_historical_data(
        self, lookback_hours: int = 24, granularity: str | None = None
    ) -> pd.DataFrame:
        """
        Get historical data as a pandas DataFrame.
        
        Args:
            lookback_hours: Hours of historical data to fetch
            granularity: Candle granularity
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self._initialized:
            await self.connect()
        
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=lookback_hours)
        
        data = await self.provider.fetch_historical_data(
            start_time=start_time, end_time=end_time, granularity=granularity
        )
        
        return self._to_dataframe(data)
    
    async def get_current_price(self) -> Decimal | None:
        """
        Get the current market price.
        
        Returns:
            Current price or None if unavailable
        """
        if not self._initialized:
            await self.connect()
        
        return await self.provider.fetch_latest_price()
    
    async def get_orderbook_snapshot(self, level: int = 2) -> dict[str, Any] | None:
        """
        Get a snapshot of the current order book.
        
        Args:
            level: Order book depth level
            
        Returns:
            Order book data or None if unavailable
        """
        if not self._initialized:
            await self.connect()
        
        return await self.provider.fetch_orderbook(level)
    
    def get_latest_ohlcv_dataframe(self, limit: int | None = None) -> pd.DataFrame:
        """
        Get latest OHLCV data as DataFrame.
        
        Args:
            limit: Number of candles to include
            
        Returns:
            DataFrame with OHLCV data
        """
        return self.provider.to_dataframe(limit)
    
    def subscribe_to_price_updates(
        self, callback: Callable[[MarketData], None]
    ) -> None:
        """
        Subscribe to real-time price updates.
        
        Args:
            callback: Function to call on price updates
        """
        self.provider.subscribe_to_updates(callback)
    
    def unsubscribe_from_price_updates(
        self, callback: Callable[[MarketData], None]
    ) -> None:
        """
        Unsubscribe from real-time price updates.
        
        Args:
            callback: Function to remove from subscribers
        """
        self.provider.unsubscribe_from_updates(callback)
    
    def get_connection_status(self) -> dict[str, Any]:
        """
        Get detailed connection and data status.
        
        Returns:
            Status dictionary
        """
        return self.provider.get_data_status()
    
    def _to_dataframe(self, data: list[MarketData]) -> pd.DataFrame:
        """
        Convert MarketData list to DataFrame.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            DataFrame with OHLCV columns
        """
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
        
        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        return df


# Factory function for easy client creation
def create_bluefin_market_data_client(
    symbol: str = None, interval: str = None
) -> BluefinMarketDataClient:
    """
    Factory function to create a BluefinMarketDataClient instance.
    
    Args:
        symbol: Trading symbol (default: from settings)
        interval: Candle interval (default: from settings)
        
    Returns:
        BluefinMarketDataClient instance
    """
    return BluefinMarketDataClient(symbol, interval)