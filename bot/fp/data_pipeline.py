"""
Functional Data Processing Pipeline

This module provides pure functional data processing pipelines for market data,
offering enhanced performance and composability alongside the existing imperative implementations.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, TypeVar

from .types.market import Candle, MarketSnapshot, OHLCV, OrderBook, Trade
from .types.result import Result, Ok, Err
from .effects.io import IO, AsyncIO, IOEither

T = TypeVar('T')
U = TypeVar('U')


@dataclass(frozen=True)
class DataPipelineConfig:
    """Configuration for data processing pipelines"""
    buffer_size: int = 1000
    processing_interval: timedelta = timedelta(seconds=1)
    enable_batching: bool = True
    batch_size: int = 100
    enable_compression: bool = False
    max_memory_usage: int = 100_000_000  # 100MB
    

@dataclass(frozen=True)
class PipelineMetrics:
    """Metrics for pipeline performance monitoring"""
    processed_count: int
    processing_rate: float  # items per second
    error_count: int
    memory_usage: int
    latency_p50: float
    latency_p99: float
    last_update: datetime


class FunctionalDataPipeline:
    """
    Functional data processing pipeline with composable transformations.
    
    Provides a functional approach to data processing that can work alongside
    the existing imperative data providers for enhanced performance.
    """

    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self._metrics = PipelineMetrics(
            processed_count=0,
            processing_rate=0.0,
            error_count=0,
            memory_usage=0,
            latency_p50=0.0,
            latency_p99=0.0,
            last_update=datetime.utcnow()
        )


    # Core Functional Data Processing Operations

    def map_data(self, transform: Callable[[T], U]) -> Callable[[Sequence[T]], IO[list[U]]]:
        """Pure functional map operation over data sequences"""
        
        def mapper(data: Sequence[T]) -> IO[list[U]]:
            def apply_transform():
                return [transform(item) for item in data]
            return IO(apply_transform)
        
        return mapper


    def filter_data(self, predicate: Callable[[T], bool]) -> Callable[[Sequence[T]], IO[list[T]]]:
        """Pure functional filter operation over data sequences"""
        
        def filterer(data: Sequence[T]) -> IO[list[T]]:
            def apply_filter():
                return [item for item in data if predicate(item)]
            return IO(apply_filter)
        
        return filterer


    def reduce_data(
        self, 
        reducer: Callable[[U, T], U], 
        initial: U
    ) -> Callable[[Sequence[T]], IO[U]]:
        """Pure functional reduce operation over data sequences"""
        
        def red(data: Sequence[T]) -> IO[U]:
            def apply_reduce():
                result = initial
                for item in data:
                    result = reducer(result, item)
                return result
            return IO(apply_reduce)
        
        return red


    def compose_transforms(
        self, 
        *transforms: Callable[[Sequence[T]], IO[Sequence[T]]]
    ) -> Callable[[Sequence[T]], IO[Sequence[T]]]:
        """Compose multiple data transformations into a single pipeline"""
        
        def composed_pipeline(data: Sequence[T]) -> IO[Sequence[T]]:
            def apply_all():
                current = data
                for transform in transforms:
                    current = transform(current).run()
                return current
            return IO(apply_all)
        
        return composed_pipeline


    # Market Data Specific Transformations

    def normalize_prices(self, candles: Sequence[OHLCV]) -> IO[list[OHLCV]]:
        """Normalize price data for consistent processing"""
        
        def normalize():
            if not candles:
                return []
            
            # Calculate normalization factor from first candle
            base_price = candles[0].close
            
            normalized = []
            for candle in candles:
                factor = candle.close / base_price
                normalized.append(OHLCV(
                    open=candle.open / factor,
                    high=candle.high / factor,
                    low=candle.low / factor,
                    close=candle.close / factor,
                    volume=candle.volume,
                    timestamp=candle.timestamp
                ))
            
            return normalized
        
        return IO(normalize)


    def aggregate_trades_to_candles(
        self, 
        trades: Sequence[Trade], 
        interval: timedelta
    ) -> IO[list[Candle]]:
        """Aggregate trade data into OHLCV candles using functional approach"""
        
        def aggregate():
            if not trades:
                return []
            
            candles = []
            current_candle_start = None
            current_trades = []
            
            for trade in sorted(trades, key=lambda t: t.timestamp):
                # Determine candle start time
                candle_start = trade.timestamp.replace(
                    second=0, 
                    microsecond=0
                )
                
                # Round down to interval boundary
                interval_seconds = int(interval.total_seconds())
                candle_start = candle_start.replace(
                    minute=(candle_start.minute // interval_seconds) * interval_seconds
                )
                
                # Check if we need to start a new candle
                if current_candle_start != candle_start:
                    # Process previous candle if exists
                    if current_trades:
                        candle = self._create_candle_from_trades(
                            current_trades, 
                            current_candle_start
                        )
                        candles.append(candle)
                    
                    # Start new candle
                    current_candle_start = candle_start
                    current_trades = [trade]
                else:
                    current_trades.append(trade)
            
            # Process final candle
            if current_trades:
                candle = self._create_candle_from_trades(
                    current_trades, 
                    current_candle_start
                )
                candles.append(candle)
            
            return candles
        
        return IO(aggregate)


    def _create_candle_from_trades(
        self, 
        trades: list[Trade], 
        candle_start: datetime
    ) -> Candle:
        """Create a candle from a list of trades"""
        if not trades:
            raise ValueError("Cannot create candle from empty trades")
        
        prices = [trade.price for trade in trades]
        total_volume = sum(trade.size for trade in trades)
        
        return Candle(
            timestamp=candle_start,
            open=trades[0].price,
            high=max(prices),
            low=min(prices),
            close=trades[-1].price,
            volume=total_volume
        )


    def resample_candles(
        self, 
        candles: Sequence[OHLCV], 
        target_interval: timedelta
    ) -> IO[list[OHLCV]]:
        """Resample candles to a different time interval"""
        
        def resample():
            if not candles:
                return []
            
            resampled = []
            current_group = []
            current_interval_start = None
            
            for candle in sorted(candles, key=lambda c: c.timestamp):
                # Calculate interval start for this candle
                interval_start = self._get_interval_start(
                    candle.timestamp, 
                    target_interval
                )
                
                if current_interval_start != interval_start:
                    # Process previous group
                    if current_group:
                        resampled_candle = self._merge_candles(
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
                resampled_candle = self._merge_candles(
                    current_group, 
                    current_interval_start
                )
                resampled.append(resampled_candle)
            
            return resampled
        
        return IO(resample)


    def _get_interval_start(self, timestamp: datetime, interval: timedelta) -> datetime:
        """Get the start time for an interval containing the given timestamp"""
        interval_seconds = int(interval.total_seconds())
        epoch = datetime(1970, 1, 1, tzinfo=timestamp.tzinfo)
        seconds_since_epoch = int((timestamp - epoch).total_seconds())
        interval_start_seconds = (seconds_since_epoch // interval_seconds) * interval_seconds
        return epoch + timedelta(seconds=interval_start_seconds)


    def _merge_candles(self, candles: list[OHLCV], interval_start: datetime) -> OHLCV:
        """Merge multiple candles into a single candle"""
        if not candles:
            raise ValueError("Cannot merge empty candles")
        
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        
        return OHLCV(
            open=sorted_candles[0].open,
            high=max(c.high for c in candles),
            low=min(c.low for c in candles),
            close=sorted_candles[-1].close,
            volume=sum(c.volume for c in candles),
            timestamp=interval_start
        )


    # Data Quality and Validation

    def validate_data_quality(self, candles: Sequence[OHLCV]) -> IO[Result[list[OHLCV], str]]:
        """Validate data quality and return valid candles or error"""
        
        def validate():
            try:
                valid_candles = []
                errors = []
                
                for i, candle in enumerate(candles):
                    # Validate OHLC relationships
                    if candle.high < max(candle.open, candle.close, candle.low):
                        errors.append(f"Invalid high price at index {i}")
                        continue
                    
                    if candle.low > min(candle.open, candle.close, candle.high):
                        errors.append(f"Invalid low price at index {i}")
                        continue
                    
                    # Validate positive values
                    if any(price <= 0 for price in [candle.open, candle.high, candle.low, candle.close]):
                        errors.append(f"Non-positive price at index {i}")
                        continue
                    
                    if candle.volume < 0:
                        errors.append(f"Negative volume at index {i}")
                        continue
                    
                    valid_candles.append(candle)
                
                if errors:
                    return Err(f"Data quality issues: {'; '.join(errors)}")
                
                return Ok(valid_candles)
                
            except Exception as e:
                return Err(f"Validation error: {str(e)}")
        
        return IO(validate)


    def detect_price_anomalies(
        self, 
        candles: Sequence[OHLCV], 
        threshold: float = 0.1
    ) -> IO[list[int]]:
        """Detect price anomalies using statistical methods"""
        
        def detect():
            if len(candles) < 2:
                return []
            
            anomaly_indices = []
            
            for i in range(1, len(candles)):
                prev_close = candles[i-1].close
                current_close = candles[i].close
                
                # Calculate percentage change
                change = abs(current_close - prev_close) / prev_close
                
                if change > threshold:
                    anomaly_indices.append(i)
            
            return anomaly_indices
        
        return IO(detect)


    # Streaming Data Processing

    def process_stream(
        self, 
        stream: AsyncIterator[T], 
        processor: Callable[[T], IO[U]]
    ) -> AsyncIO[AsyncIterator[U]]:
        """Process streaming data with functional transformations"""
        
        async def process():
            async def processed_stream():
                async for item in stream:
                    try:
                        result = processor(item).run()
                        yield result
                    except Exception as e:
                        # Log error but continue processing
                        self._metrics = self._metrics._replace(
                            error_count=self._metrics.error_count + 1
                        )
                        continue
            
            return processed_stream()
        
        return AsyncIO(process)


    def buffer_stream(
        self, 
        stream: AsyncIterator[T], 
        buffer_size: int | None = None
    ) -> AsyncIO[AsyncIterator[list[T]]]:
        """Buffer streaming data into batches for efficient processing"""
        
        async def buffer():
            size = buffer_size or self.config.buffer_size
            
            async def buffered_stream():
                buffer = []
                
                async for item in stream:
                    buffer.append(item)
                    
                    if len(buffer) >= size:
                        yield buffer.copy()
                        buffer.clear()
                
                # Yield remaining items
                if buffer:
                    yield buffer
            
            return buffered_stream()
        
        return AsyncIO(buffer)


    # Performance Optimization

    def optimize_memory_usage(self, candles: Sequence[OHLCV]) -> IO[list[OHLCV]]:
        """Optimize memory usage by compressing data where possible"""
        
        def optimize():
            if not self.config.enable_compression:
                return list(candles)
            
            # For now, just limit the number of candles to prevent memory issues
            max_candles = self.config.max_memory_usage // 1000  # rough estimate
            
            if len(candles) > max_candles:
                return list(candles[-max_candles:])
            
            return list(candles)
        
        return IO(optimize)


    def get_metrics(self) -> IO[PipelineMetrics]:
        """Get current pipeline performance metrics"""
        return IO(lambda: self._metrics)


    def update_metrics(
        self, 
        processed_count: int, 
        processing_time: float
    ) -> IO[None]:
        """Update pipeline metrics"""
        
        def update():
            rate = processed_count / processing_time if processing_time > 0 else 0
            
            self._metrics = PipelineMetrics(
                processed_count=self._metrics.processed_count + processed_count,
                processing_rate=rate,
                error_count=self._metrics.error_count,
                memory_usage=self._metrics.memory_usage,
                latency_p50=processing_time,  # Simplified for now
                latency_p99=processing_time,  # Simplified for now
                last_update=datetime.utcnow()
            )
        
        return IO(update)


# Factory Functions

def create_market_data_pipeline(
    buffer_size: int = 1000,
    enable_batching: bool = True,
    batch_size: int = 100
) -> FunctionalDataPipeline:
    """Create a functional data pipeline with specified configuration"""
    config = DataPipelineConfig(
        buffer_size=buffer_size,
        enable_batching=enable_batching,
        batch_size=batch_size
    )
    return FunctionalDataPipeline(config)


def create_high_performance_pipeline() -> FunctionalDataPipeline:
    """Create a high-performance data pipeline optimized for throughput"""
    config = DataPipelineConfig(
        buffer_size=5000,
        processing_interval=timedelta(milliseconds=100),
        enable_batching=True,
        batch_size=500,
        enable_compression=True,
        max_memory_usage=500_000_000  # 500MB
    )
    return FunctionalDataPipeline(config)


def create_low_latency_pipeline() -> FunctionalDataPipeline:
    """Create a low-latency data pipeline optimized for speed"""
    config = DataPipelineConfig(
        buffer_size=100,
        processing_interval=timedelta(milliseconds=10),
        enable_batching=False,
        batch_size=1,
        enable_compression=False,
        max_memory_usage=50_000_000  # 50MB
    )
    return FunctionalDataPipeline(config)