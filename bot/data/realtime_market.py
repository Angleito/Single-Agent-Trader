"""
Real-time market data provider for high-frequency trading.

This provider integrates WebSocket feeds from Bluefin service with tick aggregation
to generate real-time candles for scalping algorithms.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Deque
from dataclasses import dataclass, field

import pandas as pd

from ..config import settings
from ..types import MarketData
from ..exchange.bluefin_client import BluefinServiceClient

logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """Individual price tick/trade data"""
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Decimal
    side: str  # "buy" or "sell"
    trade_id: Optional[str] = None


@dataclass
class RealtimeCandle:
    """Real-time candle being built from ticks"""
    symbol: str
    interval_seconds: int
    start_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    tick_count: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def update_with_tick(self, tick: Tick) -> bool:
        """
        Update candle with new tick data.
        
        Returns:
            True if tick was within this candle's time window
        """
        # Check if tick belongs to this candle
        candle_end_time = self.start_time + timedelta(seconds=self.interval_seconds)
        if not (self.start_time <= tick.timestamp < candle_end_time):
            return False
        
        # Update OHLC
        if self.tick_count == 0:
            # First tick - initialize OHLC
            self.open = tick.price
            self.high = tick.price
            self.low = tick.price
            self.close = tick.price
            self.volume = tick.volume
        else:
            # Update existing candle
            self.high = max(self.high, tick.price)
            self.low = min(self.low, tick.price)
            self.close = tick.price
            self.volume += tick.volume
        
        self.tick_count += 1
        self.last_update = datetime.now(UTC)
        return True
    
    def to_market_data(self) -> MarketData:
        """Convert to MarketData object"""
        return MarketData(
            symbol=self.symbol,
            timestamp=self.start_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume
        )
    
    def is_complete(self) -> bool:
        """Check if candle period is complete"""
        current_time = datetime.now(UTC)
        candle_end_time = self.start_time + timedelta(seconds=self.interval_seconds)
        return current_time >= candle_end_time


class TickAggregator:
    """Aggregates ticks into real-time candles"""
    
    def __init__(self, intervals: List[int] = None):
        """
        Initialize tick aggregator.
        
        Args:
            intervals: List of candle intervals in seconds (default: [1, 5, 15, 60])
        """
        self.intervals = intervals or [1, 5, 15, 60]  # 1s, 5s, 15s, 1m
        
        # Current candles for each interval
        self.current_candles: Dict[str, Dict[int, RealtimeCandle]] = defaultdict(dict)
        
        # Historical candles (limited buffer)
        self.candle_history: Dict[str, Dict[int, Deque[MarketData]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # Subscribers for candle updates
        self.candle_subscribers: List[Callable] = []
        
        # Tick buffer for each symbol
        self.tick_buffer: Dict[str, Deque[Tick]] = defaultdict(lambda: deque(maxlen=10000))
        
        logger.info(f"Initialized TickAggregator with intervals: {self.intervals}s")
    
    def add_tick(self, tick: Tick):
        """Add a new tick and update candles"""
        # Add to tick buffer
        self.tick_buffer[tick.symbol].append(tick)
        
        # Update all candles for this symbol
        for interval_seconds in self.intervals:
            self._update_candle_for_interval(tick, interval_seconds)
    
    def _update_candle_for_interval(self, tick: Tick, interval_seconds: int):
        """Update candle for specific interval"""
        symbol = tick.symbol
        
        # Get or create current candle
        if interval_seconds not in self.current_candles[symbol]:
            # Create new candle
            candle_start = self._get_candle_start_time(tick.timestamp, interval_seconds)
            self.current_candles[symbol][interval_seconds] = RealtimeCandle(
                symbol=symbol,
                interval_seconds=interval_seconds,
                start_time=candle_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=Decimal("0")
            )
        
        current_candle = self.current_candles[symbol][interval_seconds]
        
        # Try to update current candle
        if current_candle.update_with_tick(tick):
            # Tick was added to current candle
            pass
        else:
            # Need to complete current candle and start new one
            self._complete_candle(current_candle)
            
            # Create new candle
            candle_start = self._get_candle_start_time(tick.timestamp, interval_seconds)
            new_candle = RealtimeCandle(
                symbol=symbol,
                interval_seconds=interval_seconds,
                start_time=candle_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=Decimal("0")
            )
            new_candle.update_with_tick(tick)
            self.current_candles[symbol][interval_seconds] = new_candle
    
    def _get_candle_start_time(self, timestamp: datetime, interval_seconds: int) -> datetime:
        """Calculate candle start time for given timestamp and interval"""
        # Round down to nearest interval boundary
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        total_seconds = (timestamp - epoch).total_seconds()
        interval_start_seconds = (total_seconds // interval_seconds) * interval_seconds
        return epoch + timedelta(seconds=interval_start_seconds)
    
    def _complete_candle(self, candle: RealtimeCandle):
        """Complete a candle and move to history"""
        if candle.tick_count > 0:
            # Convert to MarketData and add to history
            market_data = candle.to_market_data()
            self.candle_history[candle.symbol][candle.interval_seconds].append(market_data)
            
            # Notify subscribers
            self._notify_candle_subscribers(market_data)
            
            logger.debug(
                f"Completed {candle.interval_seconds}s candle for {candle.symbol}: "
                f"O={candle.open} H={candle.high} L={candle.low} C={candle.close} "
                f"V={candle.volume} ({candle.tick_count} ticks)"
            )
    
    def _notify_candle_subscribers(self, candle: MarketData):
        """Notify subscribers of new completed candle"""
        for callback in self.candle_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(candle))
                else:
                    callback(candle)
            except Exception as e:
                logger.error(f"Error in candle subscriber callback: {e}")
    
    def get_current_candles(self, symbol: str) -> Dict[int, RealtimeCandle]:
        """Get current candles for symbol"""
        return self.current_candles.get(symbol, {}).copy()
    
    def get_candle_history(self, symbol: str, interval_seconds: int, limit: int = 100) -> List[MarketData]:
        """Get historical candles for symbol and interval"""
        history = self.candle_history[symbol][interval_seconds]
        return list(history)[-limit:]
    
    def get_recent_ticks(self, symbol: str, limit: int = 100) -> List[Tick]:
        """Get recent ticks for symbol"""
        ticks = self.tick_buffer.get(symbol, deque())
        return list(ticks)[-limit:]
    
    def subscribe_to_candles(self, callback: Callable):
        """Subscribe to completed candle notifications"""
        if callback not in self.candle_subscribers:
            self.candle_subscribers.append(callback)
            logger.debug(f"Added candle subscriber: {callback.__name__}")
    
    def unsubscribe_from_candles(self, callback: Callable):
        """Unsubscribe from candle notifications"""
        if callback in self.candle_subscribers:
            self.candle_subscribers.remove(callback)
            logger.debug(f"Removed candle subscriber: {callback.__name__}")
    
    def force_complete_candles(self, symbol: str = None):
        """Force completion of current candles (useful for testing)"""
        symbols_to_process = [symbol] if symbol else list(self.current_candles.keys())
        
        for sym in symbols_to_process:
            for interval_seconds, candle in self.current_candles[sym].items():
                if candle.tick_count > 0:
                    self._complete_candle(candle)
                    # Reset current candle
                    self.current_candles[sym][interval_seconds] = RealtimeCandle(
                        symbol=sym,
                        interval_seconds=interval_seconds,
                        start_time=datetime.now(UTC),
                        open=candle.close,
                        high=candle.close,
                        low=candle.close,
                        close=candle.close,
                        volume=Decimal("0")
                    )


class RealtimeMarketDataProvider:
    """
    Real-time market data provider that integrates WebSocket feeds with tick aggregation.
    
    Features:
    - WebSocket connection to Bluefin service
    - Real-time tick processing and candle generation
    - Multiple timeframe support (1s, 5s, 15s, 1m, etc.)
    - Historical data integration
    - Subscriber pattern for real-time updates
    """
    
    def __init__(self, symbol: str = None, intervals: List[int] = None):
        """
        Initialize real-time market data provider.
        
        Args:
            symbol: Trading symbol (e.g., 'ETH-PERP', 'BTC-PERP')
            intervals: List of candle intervals in seconds
        """
        self.symbol = symbol or settings.trading.symbol
        self.intervals = intervals or [1, 5, 15, 60]  # 1s, 5s, 15s, 1m
        
        # Bluefin client for WebSocket connection
        self.client = BluefinServiceClient()
        
        # Tick aggregator
        self.tick_aggregator = TickAggregator(self.intervals)
        
        # Connection state
        self.connected = False
        self.websocket_connected = False
        
        # Performance tracking
        self.tick_count = 0
        self.start_time: Optional[datetime] = None
        self.last_tick_time: Optional[datetime] = None
        
        # Current price cache
        self.current_price: Optional[Decimal] = None
        self.price_update_time: Optional[datetime] = None
        
        logger.info(
            f"Initialized RealtimeMarketDataProvider for {self.symbol} "
            f"with intervals: {self.intervals}s"
        )
    
    async def connect(self) -> bool:
        """Connect to data sources"""
        try:
            # Connect to Bluefin service
            success = await self.client.connect()
            if not success:
                logger.error("Failed to connect to Bluefin service")
                return False
            
            self.connected = True
            
            # Set up WebSocket callbacks
            self.client.subscribe_to_price_updates(self._handle_price_update)
            self.client.subscribe_to_trades(self._handle_trade_update)
            
            # Connect WebSocket for real-time data
            ws_success = await self.client.connect_websocket([self.symbol])
            if ws_success:
                self.websocket_connected = True
                logger.info(f"Connected to real-time data feeds for {self.symbol}")
            else:
                logger.warning("WebSocket connection failed, using REST API only")
            
            self.start_time = datetime.now(UTC)
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to real-time market data: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from data sources"""
        try:
            await self.client.disconnect()
            self.connected = False
            self.websocket_connected = False
            logger.info("Disconnected from real-time market data")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def _handle_price_update(self, price_data: Dict):
        """Handle incoming price updates from WebSocket"""
        try:
            symbol = price_data.get("symbol")
            price = price_data.get("price")
            volume = price_data.get("volume", 0)
            timestamp = price_data.get("timestamp", time.time())
            
            if symbol == self.symbol and price is not None:
                # Update current price
                self.current_price = Decimal(str(price))
                self.price_update_time = datetime.fromtimestamp(timestamp, tz=UTC)
                
                # Create tick from price update (assumes buy side)
                tick = Tick(
                    symbol=symbol,
                    timestamp=self.price_update_time,
                    price=self.current_price,
                    volume=Decimal(str(volume)),
                    side="buy"  # Price updates don't have side info
                )
                
                # Add to aggregator
                self.tick_aggregator.add_tick(tick)
                self.tick_count += 1
                self.last_tick_time = self.price_update_time
                
                logger.debug(f"Price update: {symbol} = ${price} (volume: {volume})")
                
        except Exception as e:
            logger.error(f"Error handling price update: {e}")
    
    async def _handle_trade_update(self, trade_data: Dict):
        """Handle incoming trade updates from WebSocket"""
        try:
            symbol = trade_data.get("symbol")
            price = trade_data.get("price")
            size = trade_data.get("size")
            side = trade_data.get("side", "buy")
            trade_id = trade_data.get("trade_id")
            timestamp = trade_data.get("timestamp", time.time())
            
            if symbol == self.symbol and price is not None and size is not None:
                # Create tick from trade
                tick = Tick(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(timestamp, tz=UTC),
                    price=Decimal(str(price)),
                    volume=Decimal(str(size)),
                    side=side,
                    trade_id=trade_id
                )
                
                # Add to aggregator
                self.tick_aggregator.add_tick(tick)
                self.tick_count += 1
                self.last_tick_time = tick.timestamp
                
                # Update current price
                self.current_price = tick.price
                self.price_update_time = tick.timestamp
                
                logger.debug(
                    f"Trade: {symbol} {side} {size} @ ${price} "
                    f"(ID: {trade_id})"
                )
                
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
    
    def get_current_price(self) -> Optional[Decimal]:
        """Get current market price"""
        return self.current_price
    
    def get_current_candles(self) -> Dict[int, RealtimeCandle]:
        """Get current candles being built"""
        return self.tick_aggregator.get_current_candles(self.symbol)
    
    def get_candle_history(self, interval_seconds: int, limit: int = 100) -> List[MarketData]:
        """Get historical candles"""
        return self.tick_aggregator.get_candle_history(self.symbol, interval_seconds, limit)
    
    def get_recent_ticks(self, limit: int = 100) -> List[Tick]:
        """Get recent tick data"""
        return self.tick_aggregator.get_recent_ticks(self.symbol, limit)
    
    def to_dataframe(self, interval_seconds: int = 60, limit: int = 100) -> pd.DataFrame:
        """Convert candle history to DataFrame"""
        candles = self.get_candle_history(interval_seconds, limit)
        
        if not candles:
            return pd.DataFrame()
        
        df_data = []
        for candle in candles:
            df_data.append({
                "timestamp": candle.timestamp,
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume),
            })
        
        df = pd.DataFrame(df_data)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def subscribe_to_candles(self, callback: Callable):
        """Subscribe to new candle notifications"""
        self.tick_aggregator.subscribe_to_candles(callback)
    
    def unsubscribe_from_candles(self, callback: Callable):
        """Unsubscribe from candle notifications"""
        self.tick_aggregator.unsubscribe_from_candles(callback)
    
    def is_connected(self) -> bool:
        """Check if provider is connected"""
        return self.connected
    
    def is_websocket_connected(self) -> bool:
        """Check if WebSocket is connected and receiving data"""
        return self.websocket_connected and self.client.is_websocket_connected()
    
    def has_websocket_data(self) -> bool:
        """Check if WebSocket is receiving data (compatible with base market provider interface)"""
        return self.is_websocket_connected()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        uptime = None
        tick_rate = None
        
        if self.start_time:
            uptime = (datetime.now(UTC) - self.start_time).total_seconds()
            if uptime > 0:
                tick_rate = self.tick_count / uptime
        
        return {
            "symbol": self.symbol,
            "connected": self.connected,
            "websocket_connected": self.websocket_connected,
            "uptime_seconds": uptime,
            "total_ticks": self.tick_count,
            "tick_rate_per_second": tick_rate,
            "last_tick_time": self.last_tick_time,
            "current_price": float(self.current_price) if self.current_price else None,
            "price_update_time": self.price_update_time,
            "intervals": self.intervals,
            "websocket_status": self.client.get_websocket_status() if self.client else {}
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information"""
        current_candles = self.get_current_candles()
        
        candle_info = {}
        for interval, candle in current_candles.items():
            candle_info[f"{interval}s"] = {
                "tick_count": candle.tick_count,
                "volume": float(candle.volume),
                "price_range": f"{candle.low}-{candle.high}",
                "last_update": candle.last_update.isoformat() if candle.last_update else None
            }
        
        return {
            **self.get_performance_stats(),
            "current_candles": candle_info,
            "candle_history_lengths": {
                f"{interval}s": len(self.get_candle_history(interval, 1000))
                for interval in self.intervals
            }
        }


# Factory function for easy creation
def create_realtime_provider(symbol: str = None, intervals: List[int] = None) -> RealtimeMarketDataProvider:
    """
    Factory function to create a RealtimeMarketDataProvider.
    
    Args:
        symbol: Trading symbol (default: from settings)
        intervals: Candle intervals in seconds (default: [1, 5, 15, 60])
    
    Returns:
        RealtimeMarketDataProvider instance
    """
    return RealtimeMarketDataProvider(symbol, intervals)