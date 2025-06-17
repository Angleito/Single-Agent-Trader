"""
Bluefin-native market data provider for enhanced historical data.

This provider fetches historical candlestick data directly from Bluefin DEX
to provide 500+ candles for better technical indicator calculations.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, List, Optional

import pandas as pd

from ..config import settings
from ..types import MarketData
from ..exchange.factory import ExchangeFactory

logger = logging.getLogger(__name__)


class BluefinMarketDataProvider:
    """
    Enhanced market data provider using Bluefin's native APIs.
    
    Provides historical OHLCV data directly from Bluefin DEX with support
    for fetching 500+ candles to improve technical indicator accuracy.
    """
    
    def __init__(self, symbol: str = None, interval: str = None):
        """
        Initialize the Bluefin market data provider.
        
        Args:
            symbol: Trading symbol (e.g., 'ETH-USD', 'BTC-USD')
            interval: Candle interval (e.g., '1m', '5m', '1h')
        """
        self.symbol = symbol or settings.trading.symbol
        self.interval = interval or settings.trading.interval
        self.candle_limit = max(500, settings.data.candle_limit)  # Minimum 500 for good indicators
        
        # Bluefin client for market data
        self._exchange = None
        self._connected = False
        
        # Data cache
        self._historical_cache: List[MarketData] = []
        self._last_update: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)  # Cache historical data for 5 minutes
        
        logger.info(
            f"Initialized BluefinMarketDataProvider for {self.symbol} "
            f"({self.interval} interval, {self.candle_limit} candles)"
        )
    
    async def connect(self) -> None:
        """Connect to Bluefin exchange for market data."""
        try:
            # Create Bluefin exchange client in read-only mode for market data
            self._exchange = ExchangeFactory.create_exchange(
                exchange_type="bluefin",
                dry_run=True  # Use read-only mode for market data
            )
            
            # Connect to the exchange
            await self._exchange.connect()
            self._connected = True
            
            logger.info("Successfully connected to Bluefin for market data")
            
        except Exception as e:
            logger.error(f"Failed to connect to Bluefin market data: {e}")
            self._connected = False
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Bluefin exchange."""
        if self._exchange:
            try:
                await self._exchange.disconnect()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
        
        self._exchange = None
        self._connected = False
        logger.info("Disconnected from Bluefin market data")
    
    async def fetch_historical_data(self, force_refresh: bool = False) -> List[MarketData]:
        """
        Fetch historical candlestick data from Bluefin.
        
        Args:
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            List of MarketData objects with OHLCV data
        """
        # Check cache validity
        if (not force_refresh and 
            self._historical_cache and 
            self._last_update and 
            datetime.utcnow() - self._last_update < self._cache_ttl):
            logger.debug(
                f"Using cached historical data ({len(self._historical_cache)} candles)"
            )
            return self._historical_cache
        
        if not self._connected or not self._exchange:
            raise RuntimeError("Not connected to Bluefin exchange")
        
        try:
            logger.info(
                f"Fetching {self.candle_limit} historical candles for {self.symbol} "
                f"from Bluefin ({self.interval} interval)"
            )
            
            # Fetch historical candles using the new method
            if hasattr(self._exchange, 'get_historical_candles'):
                raw_candles = await self._exchange.get_historical_candles(
                    symbol=self.symbol,
                    interval=self.interval,
                    limit=self.candle_limit
                )
            else:
                logger.warning("Exchange doesn't support get_historical_candles, using fallback")
                raw_candles = []
            
            # Convert to MarketData objects
            market_data = self._convert_to_market_data(raw_candles)
            
            # Update cache
            self._historical_cache = market_data
            self._last_update = datetime.utcnow()
            
            logger.info(
                f"Successfully fetched and cached {len(market_data)} candles from Bluefin"
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data from Bluefin: {e}")
            # Return cached data if available, even if stale
            if self._historical_cache:
                logger.warning("Returning stale cached data due to fetch failure")
                return self._historical_cache
            else:
                raise
    
    def _convert_to_market_data(self, raw_candles: List[dict]) -> List[MarketData]:
        """
        Convert raw Bluefin candle data to MarketData objects.
        
        Args:
            raw_candles: Raw candlestick data from Bluefin API
            
        Returns:
            List of MarketData objects
        """
        market_data = []
        
        for candle in raw_candles:
            try:
                # Handle different possible formats from Bluefin
                if isinstance(candle, dict):
                    # Extract OHLCV data - handle various field names
                    timestamp = candle.get('timestamp', candle.get('time', 0))
                    open_price = candle.get('open', candle.get('o', 0))
                    high_price = candle.get('high', candle.get('h', 0))
                    low_price = candle.get('low', candle.get('l', 0))
                    close_price = candle.get('close', candle.get('c', 0))
                    volume = candle.get('volume', candle.get('v', 0))
                    
                    # Convert timestamp (handle both seconds and milliseconds)
                    if timestamp > 1e12:  # Milliseconds
                        dt = datetime.fromtimestamp(timestamp / 1000, tz=UTC)
                    else:  # Seconds
                        dt = datetime.fromtimestamp(timestamp, tz=UTC)
                    
                    # Validate prices before creating MarketData object
                    if all(p > 0 for p in [open_price, high_price, low_price, close_price, volume]):
                        # Create MarketData object
                        data = MarketData(
                            symbol=self.symbol,
                            timestamp=dt,
                            open=Decimal(str(open_price)),
                            high=Decimal(str(high_price)),
                            low=Decimal(str(low_price)),
                            close=Decimal(str(close_price)),
                            volume=Decimal(str(volume))
                        )
                        market_data.append(data)
                    else:
                        logger.warning(f"Skipping invalid candle data: O={open_price}, H={high_price}, L={low_price}, C={close_price}, V={volume}")
                    
                elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                    # Handle array format [timestamp, open, high, low, close, volume]
                    timestamp, open_price, high_price, low_price, close_price, volume = candle[:6]
                    
                    if timestamp > 1e12:  # Milliseconds
                        dt = datetime.fromtimestamp(timestamp / 1000, tz=UTC)
                    else:  # Seconds
                        dt = datetime.fromtimestamp(timestamp, tz=UTC)
                    
                    # Validate prices before creating MarketData object
                    if all(p > 0 for p in [open_price, high_price, low_price, close_price, volume]):
                        data = MarketData(
                            symbol=self.symbol,
                            timestamp=dt,
                            open=Decimal(str(open_price)),
                            high=Decimal(str(high_price)),
                            low=Decimal(str(low_price)),
                            close=Decimal(str(close_price)),
                            volume=Decimal(str(volume))
                        )
                        market_data.append(data)
                    else:
                        logger.warning(f"Skipping invalid list candle data: O={open_price}, H={high_price}, L={low_price}, C={close_price}, V={volume}")
                
            except Exception as e:
                logger.warning(f"Failed to parse candle data: {candle}, error: {e}")
                continue
        
        # Sort by timestamp (oldest first)
        market_data.sort(key=lambda x: x.timestamp)
        
        logger.debug(f"Converted {len(market_data)} raw candles to MarketData objects")
        return market_data
    
    async def get_latest_price(self) -> Optional[Decimal]:
        """Get the latest price for the symbol."""
        if not self._connected or not self._exchange:
            return None
        
        try:
            # Get latest candle data
            candles = await self.fetch_historical_data()
            if candles:
                return candles[-1].close
        except Exception as e:
            logger.error(f"Failed to get latest price: {e}")
        
        return None
    
    def is_connected(self) -> bool:
        """Check if connected to Bluefin."""
        return self._connected and self._exchange is not None
    
    def get_cached_data_count(self) -> int:
        """Get number of cached candles."""
        return len(self._historical_cache)
    
    def get_cache_age(self) -> Optional[timedelta]:
        """Get age of cached data."""
        if self._last_update:
            return datetime.utcnow() - self._last_update
        return None
    
    def has_websocket_data(self) -> bool:
        """
        Check if websocket data is available.
        For Bluefin provider, we don't use websockets, so this always returns False.
        """
        return False
    
    def get_data_status(self) -> dict[str, Any]:
        """
        Get comprehensive status information about the data provider.
        
        Returns:
            Dictionary with status information
        """
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "connected": self.is_connected(),
            "websocket_connected": False,  # Bluefin provider doesn't use websockets
            "websocket_data_received": False,
            "first_websocket_data_time": None,
            "cached_candles": len(self._historical_cache),
            "cached_ticks": 0,
            "last_update": self._last_update,
            "latest_price": None,  # Could implement if needed
            "subscribers": 0,
            "reconnect_attempts": 0,
            "cache_status": {
                "historical_data": self._last_update is not None
            },
            "data_source": "Bluefin DEX",
            "cache_age_seconds": (
                (datetime.utcnow() - self._last_update).total_seconds()
                if self._last_update else None
            ),
        }
    
    def to_dataframe(self, limit: int = None) -> pd.DataFrame:
        """
        Convert OHLCV data to pandas DataFrame for indicator calculations.
        
        Args:
            limit: Number of candles to include
            
        Returns:
            DataFrame with OHLCV columns
        """
        # Get the cached data
        if limit and len(self._historical_cache) > limit:
            data = self._historical_cache[-limit:]
        else:
            data = self._historical_cache
        
        if not data:
            return pd.DataFrame()
        
        df_data = []
        for candle in data:
            df_data.append({
                "timestamp": candle.timestamp,
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume),
            })
        
        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)
        
        return df
    
    async def get_latest_ohlcv(self, limit: int = None) -> List[MarketData]:
        """
        Get latest OHLCV data from Bluefin.
        
        Args:
            limit: Number of candles to return (ignored, uses configured candle_limit)
        
        Returns:
            List of MarketData objects with latest OHLCV data
        """
        try:
            # Fetch latest historical data (will use cache if recent)
            data = await self.fetch_historical_data()
            
            # If limit is specified and less than our data, return only the last 'limit' candles
            if limit and len(data) > limit:
                return data[-limit:]
            
            return data
        except Exception as e:
            logger.error(f"Failed to get latest OHLCV data from Bluefin: {e}")
            return []
    
    def get_status(self) -> dict[str, Any]:
        """Get provider status information."""
        return {
            "connected": self._connected,
            "symbol": self.symbol,
            "interval": self.interval,
            "candle_limit": self.candle_limit,
            "cached_candles": len(self._historical_cache),
            "cache_age_seconds": (
                (datetime.utcnow() - self._last_update).total_seconds()
                if self._last_update else None
            ),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "data_source": "Bluefin DEX",
        }


class HybridMarketDataProvider:
    """
    Hybrid provider that uses Bluefin for historical data and Coinbase for real-time updates.
    
    This combines the best of both worlds:
    - Bluefin: Extended historical data (500+ candles)
    - Coinbase: Proven real-time WebSocket feeds
    """
    
    def __init__(self, symbol: str = None, interval: str = None):
        """Initialize hybrid market data provider."""
        self.symbol = symbol or settings.trading.symbol
        self.interval = interval or settings.trading.interval
        
        # Initialize both providers
        self.bluefin_provider = BluefinMarketDataProvider(symbol, interval)
        
        # Import Coinbase provider
        from .market import MarketDataProvider
        self.coinbase_provider = MarketDataProvider(symbol, interval)
        
        self._connected = False
        
        logger.info(
            f"Initialized HybridMarketDataProvider: "
            f"Bluefin (historical) + Coinbase (real-time) for {symbol}"
        )
    
    async def connect(self) -> None:
        """Connect to both data sources."""
        try:
            # Connect to Bluefin for historical data
            await self.bluefin_provider.connect()
            
            # Connect to Coinbase for real-time data (no historical fetch)
            await self.coinbase_provider.connect(fetch_historical=False)
            
            self._connected = True
            logger.info("Successfully connected hybrid market data provider")
            
        except Exception as e:
            logger.error(f"Failed to connect hybrid provider: {e}")
            await self.disconnect()
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from both data sources."""
        tasks = []
        if self.bluefin_provider:
            tasks.append(self.bluefin_provider.disconnect())
        if self.coinbase_provider:
            tasks.append(self.coinbase_provider.disconnect())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._connected = False
        logger.info("Disconnected hybrid market data provider")
    
    async def fetch_historical_data(self) -> List[MarketData]:
        """Fetch historical data from Bluefin (500+ candles)."""
        return await self.bluefin_provider.fetch_historical_data()
    
    async def get_latest_price(self) -> Optional[Decimal]:
        """Get latest price from Coinbase real-time feed."""
        return await self.coinbase_provider.fetch_latest_price()
    
    async def get_latest_ohlcv(self, limit: int = 200) -> List[MarketData]:
        """Get historical OHLCV data from Bluefin (enhanced with more candles)."""
        try:
            # Fetch from Bluefin provider
            data = await self.bluefin_provider.fetch_historical_data()
            # Limit to requested number of candles (most recent)
            if data and len(data) > limit:
                return data[-limit:]
            return data if data else []
        except Exception as e:
            logger.error(f"Failed to get OHLCV data from Bluefin: {e}")
            # Fallback to Coinbase provider if Bluefin fails
            try:
                return await self.coinbase_provider.get_latest_ohlcv(limit)
            except Exception as fallback_error:
                logger.error(f"Fallback to Coinbase also failed: {fallback_error}")
                return []
    
    def is_connected(self) -> bool:
        """Check if both providers are connected."""
        return (self._connected and 
                self.bluefin_provider.is_connected() and 
                self.coinbase_provider.is_connected())
    
    def has_websocket_data(self) -> bool:
        """Check if websocket data is available from Coinbase provider."""
        return self.coinbase_provider.has_websocket_data()
    
    def get_data_status(self) -> dict[str, Any]:
        """Get combined data status from both providers."""
        coinbase_status = self.coinbase_provider.get_data_status()
        bluefin_status = self.bluefin_provider.get_data_status()
        
        return {
            "hybrid_provider": True,
            "overall_connected": self.is_connected(),
            "bluefin_status": bluefin_status,
            "coinbase_status": coinbase_status,
            "symbol": self.symbol,
            "interval": self.interval,
            # Use Coinbase websocket status for real-time data
            "websocket_connected": coinbase_status.get("websocket_connected", False),
            "websocket_data_received": coinbase_status.get("websocket_data_received", False),
            # Use Bluefin for historical data count
            "cached_candles": bluefin_status.get("cached_candles", 0),
            "data_sources": "Bluefin (historical) + Coinbase (real-time)",
        }
    
    def to_dataframe(self, limit: int = 200) -> pd.DataFrame:
        """Convert OHLCV data to pandas DataFrame using Bluefin historical data."""
        return self.bluefin_provider.to_dataframe(limit)
    
    def get_status(self) -> dict[str, Any]:
        """Get combined status from both providers."""
        return {
            "hybrid_provider": True,
            "bluefin_status": self.bluefin_provider.get_status(),
            "coinbase_connected": self.coinbase_provider.is_connected(),
            "overall_connected": self.is_connected(),
        }