"""
Enhanced Market Data Aggregation Effects

This module provides advanced market data aggregation capabilities with functional programming
patterns, offering real-time data processing and streaming optimizations.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, TypeVar

from ..types.market import Candle, MarketSnapshot, OHLCV, OrderBook, Trade
from ..types.result import Result, Ok, Err
from .io import IO, AsyncIO, IOEither, from_try

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass(frozen=True)
class AggregationConfig:
    """Configuration for market data aggregation"""
    interval: timedelta
    buffer_size: int = 1000
    enable_real_time_updates: bool = True
    enable_volume_weighting: bool = True
    enable_outlier_detection: bool = True
    outlier_threshold: float = 0.05  # 5% price deviation
    enable_compression: bool = False
    max_memory_usage: int = 100_000_000  # 100MB


@dataclass(frozen=True)
class AggregationMetrics:
    """Metrics for aggregation performance"""
    processed_trades: int
    generated_candles: int
    processing_rate: float
    memory_usage: int
    last_update: datetime
    outliers_detected: int
    compression_ratio: float


@dataclass
class CandleBuilder:
    """Mutable candle builder for real-time aggregation"""
    symbol: str
    interval_start: datetime
    open_price: Decimal | None = None
    high_price: Decimal | None = None
    low_price: Decimal | None = None
    close_price: Decimal | None = None
    volume: Decimal = field(default_factory=lambda: Decimal(0))
    trade_count: int = 0
    volume_weighted_price: Decimal = field(default_factory=lambda: Decimal(0))
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the candle builder"""
        if self.open_price is None:
            self.open_price = trade.price
        
        self.high_price = max(self.high_price or trade.price, trade.price)
        self.low_price = min(self.low_price or trade.price, trade.price)
        self.close_price = trade.price
        
        # Update volume-weighted average price
        total_value = self.volume_weighted_price * self.volume + trade.price * trade.size
        self.volume += trade.size
        self.volume_weighted_price = total_value / self.volume if self.volume > 0 else trade.price
        
        self.trade_count += 1
    
    def to_candle(self) -> Candle:
        """Convert builder to immutable candle"""
        if self.open_price is None:
            raise ValueError("Cannot create candle without trades")
        
        return Candle(
            timestamp=self.interval_start,
            open=self.open_price,
            high=self.high_price or self.open_price,
            low=self.low_price or self.open_price,
            close=self.close_price or self.open_price,
            volume=self.volume
        )


class RealTimeAggregator:
    """
    Real-time market data aggregator with functional effects.
    
    Provides high-performance aggregation of trade data into OHLCV candles
    with support for multiple timeframes and real-time streaming.
    """

    def __init__(self, config: AggregationConfig):
        self.config = config
        self._active_builders: dict[str, CandleBuilder] = {}
        self._completed_candles: deque[Candle] = deque(maxlen=config.buffer_size)
        self._trade_buffer: deque[Trade] = deque(maxlen=10000)
        self._subscribers: list[Callable[[Candle], None]] = []
        self._metrics = AggregationMetrics(
            processed_trades=0,
            generated_candles=0,
            processing_rate=0.0,
            memory_usage=0,
            last_update=datetime.utcnow(),
            outliers_detected=0,
            compression_ratio=1.0
        )
        self._is_running = False


    # Core Aggregation Functions

    def aggregate_trades_real_time(
        self, 
        trades: Sequence[Trade]
    ) -> IO[list[Candle]]:
        """Aggregate trades in real-time, returning completed candles"""
        
        def aggregate():
            completed_candles = []
            
            for trade in sorted(trades, key=lambda t: t.timestamp):
                # Detect outliers if enabled
                if self.config.enable_outlier_detection:
                    if self._is_outlier(trade):
                        self._metrics = self._metrics._replace(
                            outliers_detected=self._metrics.outliers_detected + 1
                        )
                        logger.warning(f"Outlier trade detected: {trade.price} at {trade.timestamp}")
                        continue
                
                # Add trade to buffer
                self._trade_buffer.append(trade)
                
                # Get or create candle builder for this interval
                interval_start = self._get_interval_start(trade.timestamp)
                builder_key = f"{trade.symbol}_{interval_start.isoformat()}"
                
                if builder_key not in self._active_builders:
                    self._active_builders[builder_key] = CandleBuilder(
                        symbol=trade.symbol,
                        interval_start=interval_start
                    )
                
                # Add trade to builder
                self._active_builders[builder_key].add_trade(trade)
                
                # Check if we need to complete any candles
                current_interval = self._get_interval_start(datetime.utcnow())
                builders_to_complete = [
                    key for key, builder in self._active_builders.items()
                    if builder.interval_start < current_interval
                ]
                
                # Complete and remove old builders
                for key in builders_to_complete:
                    builder = self._active_builders.pop(key)
                    candle = builder.to_candle()
                    completed_candles.append(candle)
                    self._completed_candles.append(candle)
                    
                    # Notify subscribers
                    self._notify_subscribers(candle)
            
            # Update metrics
            self._metrics = self._metrics._replace(
                processed_trades=self._metrics.processed_trades + len(trades),
                generated_candles=self._metrics.generated_candles + len(completed_candles),
                last_update=datetime.utcnow()
            )
            
            return completed_candles
        
        return IO(aggregate)


    def aggregate_trades_batch(
        self, 
        trades: Sequence[Trade],
        symbol: str
    ) -> IO[list[Candle]]:
        """Aggregate trades in batch mode for historical data"""
        
        def batch_aggregate():
            if not trades:
                return []
            
            candles = []
            current_builder = None
            
            for trade in sorted(trades, key=lambda t: t.timestamp):
                interval_start = self._get_interval_start(trade.timestamp)
                
                # Check if we need a new builder
                if (current_builder is None or 
                    current_builder.interval_start != interval_start):
                    
                    # Complete previous candle if exists
                    if current_builder is not None:
                        candle = current_builder.to_candle()
                        candles.append(candle)
                    
                    # Create new builder
                    current_builder = CandleBuilder(
                        symbol=symbol,
                        interval_start=interval_start
                    )
                
                current_builder.add_trade(trade)
            
            # Complete final candle
            if current_builder is not None:
                candle = current_builder.to_candle()
                candles.append(candle)
            
            return candles
        
        return IO(batch_aggregate)


    def resample_candles(
        self, 
        candles: Sequence[OHLCV], 
        target_interval: timedelta
    ) -> IO[list[OHLCV]]:
        """Resample candles to different timeframe with volume weighting"""
        
        def resample():
            if not candles:
                return []
            
            resampled = []
            current_group = []
            current_interval_start = None
            
            for candle in sorted(candles, key=lambda c: c.timestamp):
                interval_start = self._get_interval_start_for_timeframe(
                    candle.timestamp, 
                    target_interval
                )
                
                if current_interval_start != interval_start:
                    # Process previous group
                    if current_group:
                        resampled_candle = self._merge_candles_with_volume_weighting(
                            current_group, 
                            current_interval_start
                        )
                        resampled.append(resampled_candle)
                    
                    # Start new group
                    current_interval_start = interval_start
                    current_group = [candle]
                else:
                    current_group.append(candle)
            
            # Process final group
            if current_group:
                resampled_candle = self._merge_candles_with_volume_weighting(
                    current_group, 
                    current_interval_start
                )
                resampled.append(resampled_candle)
            
            return resampled
        
        return IO(resample)


    def stream_aggregated_data(
        self, 
        trade_stream: AsyncIterator[Trade]
    ) -> AsyncIO[AsyncIterator[Candle]]:
        """Stream aggregated candles from trade stream"""
        
        async def create_aggregated_stream():
            async def candle_stream():
                trade_batch = []
                last_process_time = datetime.utcnow()
                
                async for trade in trade_stream:
                    trade_batch.append(trade)
                    
                    # Process batch when interval elapsed or batch size reached
                    now = datetime.utcnow()
                    if (len(trade_batch) >= 100 or 
                        now - last_process_time >= timedelta(seconds=1)):
                        
                        # Aggregate trades
                        candles = self.aggregate_trades_real_time(trade_batch).run()
                        
                        # Yield completed candles
                        for candle in candles:
                            yield candle
                        
                        trade_batch.clear()
                        last_process_time = now
                
                # Process remaining trades
                if trade_batch:
                    candles = self.aggregate_trades_real_time(trade_batch).run()
                    for candle in candles:
                        yield candle
            
            return candle_stream()
        
        return AsyncIO(lambda: asyncio.create_task(create_aggregated_stream()))


    # Utility Functions

    def _get_interval_start(self, timestamp: datetime) -> datetime:
        """Get interval start time for given timestamp"""
        return self._get_interval_start_for_timeframe(timestamp, self.config.interval)


    def _get_interval_start_for_timeframe(
        self, 
        timestamp: datetime, 
        interval: timedelta
    ) -> datetime:
        """Get interval start time for specific timeframe"""
        interval_seconds = int(interval.total_seconds())
        
        # Round down to interval boundary
        epoch = datetime(1970, 1, 1, tzinfo=timestamp.tzinfo)
        seconds_since_epoch = int((timestamp - epoch).total_seconds())
        interval_start_seconds = (seconds_since_epoch // interval_seconds) * interval_seconds
        
        return epoch + timedelta(seconds=interval_start_seconds)


    def _merge_candles_with_volume_weighting(
        self, 
        candles: list[OHLCV], 
        interval_start: datetime
    ) -> OHLCV:
        """Merge candles using volume-weighted average price"""
        if not candles:
            raise ValueError("Cannot merge empty candles")
        
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        
        if self.config.enable_volume_weighting:
            # Calculate volume-weighted average close price
            total_volume = sum(c.volume for c in candles)
            if total_volume > 0:
                vwap = sum(c.close * c.volume for c in candles) / total_volume
            else:
                vwap = sorted_candles[-1].close
        else:
            vwap = sorted_candles[-1].close
        
        return OHLCV(
            open=sorted_candles[0].open,
            high=max(c.high for c in candles),
            low=min(c.low for c in candles),
            close=vwap,  # Use VWAP for close if volume weighting enabled
            volume=sum(c.volume for c in candles),
            timestamp=interval_start
        )


    def _is_outlier(self, trade: Trade) -> bool:
        """Detect if trade is an outlier based on recent price history"""
        if len(self._trade_buffer) < 10:
            return False
        
        # Get recent trades for same symbol
        recent_trades = [
            t for t in list(self._trade_buffer)[-50:] 
            if t.symbol == trade.symbol
        ]
        
        if len(recent_trades) < 5:
            return False
        
        # Calculate average price of recent trades
        avg_price = sum(t.price for t in recent_trades) / len(recent_trades)
        
        # Check if current trade price deviates significantly
        deviation = abs(trade.price - avg_price) / avg_price
        return deviation > self.config.outlier_threshold


    def _notify_subscribers(self, candle: Candle) -> None:
        """Notify subscribers of new candle"""
        for callback in self._subscribers:
            try:
                callback(candle)
            except Exception as e:
                logger.error(f"Error in candle subscriber callback: {e}")


    # Subscription Management

    def subscribe_to_candles(self, callback: Callable[[Candle], None]) -> IO[None]:
        """Subscribe to real-time candle updates"""
        
        def subscribe():
            self._subscribers.append(callback)
            logger.debug("Added candle subscriber")
        
        return IO(subscribe)


    def unsubscribe_from_candles(self, callback: Callable[[Candle], None]) -> IO[None]:
        """Unsubscribe from candle updates"""
        
        def unsubscribe():
            try:
                self._subscribers.remove(callback)
                logger.debug("Removed candle subscriber")
            except ValueError:
                logger.warning("Subscriber not found")
        
        return IO(unsubscribe)


    # Data Access

    def get_completed_candles(self, limit: int | None = None) -> IO[list[Candle]]:
        """Get completed candles from buffer"""
        
        def get_candles():
            candles = list(self._completed_candles)
            if limit is not None:
                return candles[-limit:]
            return candles
        
        return IO(get_candles)


    def get_active_candles(self) -> IO[list[Candle]]:
        """Get current active (incomplete) candles"""
        
        def get_active():
            active_candles = []
            for builder in self._active_builders.values():
                try:
                    candle = builder.to_candle()
                    active_candles.append(candle)
                except ValueError:
                    # Builder has no trades yet
                    continue
            return active_candles
        
        return IO(get_active)


    # Performance and Monitoring

    def get_metrics(self) -> IO[AggregationMetrics]:
        """Get aggregation performance metrics"""
        return IO(lambda: self._metrics)


    def optimize_memory_usage(self) -> IO[int]:
        """Optimize memory usage by cleaning old data"""
        
        def optimize():
            initial_size = len(self._trade_buffer) + len(self._completed_candles)
            
            # Clean old completed candles if over buffer size
            while len(self._completed_candles) > self.config.buffer_size:
                self._completed_candles.popleft()
            
            # Clean old trades
            if len(self._trade_buffer) > 5000:
                # Keep only recent trades
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                recent_trades = deque([
                    trade for trade in self._trade_buffer 
                    if trade.timestamp > cutoff_time
                ], maxlen=5000)
                self._trade_buffer = recent_trades
            
            final_size = len(self._trade_buffer) + len(self._completed_candles)
            freed = initial_size - final_size
            
            logger.info(f"Memory optimization freed {freed} items")
            return freed
        
        return IO(optimize)


    def calculate_processing_rate(self) -> IO[float]:
        """Calculate current processing rate"""
        
        def calculate():
            if self._metrics.processed_trades == 0:
                return 0.0
            
            time_elapsed = (datetime.utcnow() - self._metrics.last_update).total_seconds()
            if time_elapsed <= 0:
                return 0.0
            
            return self._metrics.processed_trades / time_elapsed
        
        return IO(calculate)


# Factory Functions

def create_real_time_aggregator(
    interval: timedelta,
    buffer_size: int = 1000,
    enable_volume_weighting: bool = True
) -> RealTimeAggregator:
    """Create real-time aggregator with specified configuration"""
    config = AggregationConfig(
        interval=interval,
        buffer_size=buffer_size,
        enable_real_time_updates=True,
        enable_volume_weighting=enable_volume_weighting,
        enable_outlier_detection=True
    )
    return RealTimeAggregator(config)


def create_high_frequency_aggregator() -> RealTimeAggregator:
    """Create aggregator optimized for high-frequency trading"""
    config = AggregationConfig(
        interval=timedelta(seconds=1),
        buffer_size=10000,
        enable_real_time_updates=True,
        enable_volume_weighting=True,
        enable_outlier_detection=True,
        outlier_threshold=0.02,  # Stricter outlier detection
        enable_compression=True
    )
    return RealTimeAggregator(config)


def create_batch_aggregator(interval: timedelta) -> RealTimeAggregator:
    """Create aggregator optimized for batch processing"""
    config = AggregationConfig(
        interval=interval,
        buffer_size=50000,
        enable_real_time_updates=False,
        enable_volume_weighting=True,
        enable_outlier_detection=False,  # Skip for batch processing
        enable_compression=True
    )
    return RealTimeAggregator(config)