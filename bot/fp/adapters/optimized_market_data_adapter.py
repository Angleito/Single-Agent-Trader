"""
Optimized Market Data Adapter

Agent 8: High-performance market data adapter that integrates optimized functional types
with enhanced aggregation and real-time streaming capabilities.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from ...trading_types import MarketData as LegacyMarketData
from ..core.either import Either, Left, Right
from ..effects.enhanced_aggregation import (
    AggregationMetrics,
    create_enhanced_aggregator,
    create_high_performance_aggregator,
    create_low_latency_aggregator,
)
from ..effects.io import IO, AsyncIO
from ..types.optimized_market import (
    OptimizedDataFactory,
    OptimizedOHLCV,
    create_optimized_ohlcv,
    create_optimized_trade,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizedAdapterConfig:
    """Configuration for optimized market data adapter"""

    # Basic settings
    symbol: str
    interval: str
    exchange_type: str = "coinbase"

    # Performance settings
    performance_mode: str = "balanced"  # "low_latency", "high_throughput", "balanced"
    buffer_size: int = 10000
    enable_caching: bool = True
    cache_size: int = 1000

    # Real-time settings
    enable_real_time_streaming: bool = True
    stream_batch_size: int = 100
    update_frequency_ms: int = 100

    # Data quality settings
    enable_data_validation: bool = True
    enable_outlier_detection: bool = True
    outlier_threshold: float = 0.05

    # Memory management
    max_memory_mb: float = 100.0
    auto_cleanup_interval_minutes: int = 5


class OptimizedMarketDataAdapter:
    """
    High-performance market data adapter with functional programming optimizations.

    Features:
    - Memory-optimized data structures
    - Real-time aggregation with adaptive batching
    - Automatic performance tuning
    - Comprehensive data validation
    - Legacy compatibility layer
    """

    def __init__(self, config: OptimizedAdapterConfig):
        self.config = config
        self.data_factory = OptimizedDataFactory(
            enable_caching=config.enable_caching, cache_size=config.cache_size
        )

        # Create aggregator based on performance mode
        interval_timedelta = self._parse_interval(config.interval)
        if config.performance_mode == "low_latency":
            self.aggregator = create_low_latency_aggregator(interval_timedelta)
        elif config.performance_mode == "high_throughput":
            self.aggregator = create_high_performance_aggregator(interval_timedelta)
        else:  # balanced
            self.aggregator = create_enhanced_aggregator(
                interval_timedelta, buffer_size=config.buffer_size
            )

        # State tracking
        self._is_connected = False
        self._is_streaming = False
        self._legacy_provider: Any = None

        # Subscribers
        self._ohlcv_subscribers: list[Callable[[OptimizedOHLCV], None]] = []
        self._legacy_subscribers: list[Callable[[LegacyMarketData], None]] = []
        self._metrics_subscribers: list[Callable[[AggregationMetrics], None]] = []

        # Background tasks
        self._streaming_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        logger.info(
            f"Optimized adapter initialized for {config.symbol} @ {config.interval}"
        )

    async def connect(self, legacy_provider: Any = None) -> Either[str, bool]:
        """Connect to market data feeds with optional legacy provider integration"""
        try:
            # Start the aggregator
            await self.aggregator.start()

            # Connect legacy provider if provided
            if legacy_provider:
                self._legacy_provider = legacy_provider
                await self._setup_legacy_integration()

            self._is_connected = True
            logger.info("Optimized market data adapter connected")

            return Right(True)

        except Exception as e:
            logger.error(f"Failed to connect adapter: {e}")
            return Left(str(e))

    async def disconnect(self) -> Either[str, bool]:
        """Disconnect from market data feeds"""
        try:
            self._is_connected = False
            self._is_streaming = False

            # Stop background tasks
            if self._streaming_task and not self._streaming_task.done():
                self._streaming_task.cancel()
                try:
                    await self._streaming_task
                except asyncio.CancelledError:
                    pass

            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Stop aggregator
            await self.aggregator.stop()

            # Disconnect legacy provider
            if self._legacy_provider and hasattr(self._legacy_provider, "disconnect"):
                await self._legacy_provider.disconnect()

            logger.info("Optimized market data adapter disconnected")
            return Right(True)

        except Exception as e:
            logger.error(f"Failed to disconnect adapter: {e}")
            return Left(str(e))

    def fetch_historical_data(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> IO[Either[str, list[OptimizedOHLCV]]]:
        """Fetch historical OHLCV data with optimizations"""

        def fetch():
            try:
                if not self._legacy_provider:
                    return Left("No data provider available")

                # Use legacy provider for historical data
                if hasattr(self._legacy_provider, "fetch_historical_data"):
                    legacy_data = asyncio.run(
                        self._legacy_provider.fetch_historical_data(
                            start_time=start_time,
                            end_time=end_time,
                            granularity=self.config.interval,
                        )
                    )
                elif hasattr(self._legacy_provider, "get_latest_ohlcv"):
                    legacy_data = self._legacy_provider.get_latest_ohlcv(limit or 200)
                else:
                    return Left("Legacy provider does not support historical data")

                # Convert to optimized format
                optimized_data = []
                for item in legacy_data:
                    try:
                        ohlcv = create_optimized_ohlcv(
                            open=item.open,
                            high=item.high,
                            low=item.low,
                            close=item.close,
                            volume=item.volume,
                            timestamp=item.timestamp,
                        )
                        optimized_data.append(ohlcv)
                    except Exception as e:
                        logger.warning(f"Failed to convert legacy data item: {e}")
                        continue

                # Validate data if enabled
                if self.config.enable_data_validation:
                    validated_data, errors = self._batch_validate_ohlcv(optimized_data)
                    if errors:
                        logger.warning(f"Data validation found {len(errors)} issues")
                    optimized_data = validated_data

                logger.info(f"Fetched {len(optimized_data)} historical candles")
                return Right(optimized_data)

            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                return Left(str(e))

        return IO(fetch)

    def get_latest_ohlcv(
        self, limit: int | None = None
    ) -> IO[Either[str, list[OptimizedOHLCV]]]:
        """Get latest OHLCV data from aggregator"""

        def get_latest():
            try:
                candles = self.aggregator.get_completed_candles(limit).run()
                return Right(candles)
            except Exception as e:
                logger.error(f"Error getting latest OHLCV: {e}")
                return Left(str(e))

        return IO(get_latest)

    def get_active_candles(self) -> IO[Either[str, list[OptimizedOHLCV]]]:
        """Get current active (incomplete) candles"""

        def get_active():
            try:
                candles = self.aggregator.get_active_candles().run()
                return Right(candles)
            except Exception as e:
                logger.error(f"Error getting active candles: {e}")
                return Left(str(e))

        return IO(get_active)

    def get_latest_price(self) -> IO[Either[str, Decimal]]:
        """Get latest price from most recent data"""

        def get_price():
            try:
                candles = self.aggregator.get_completed_candles(1).run()
                if candles:
                    return Right(candles[-1].close)

                # Fallback to legacy provider
                if self._legacy_provider:
                    price = self._legacy_provider.get_latest_price()
                    if price:
                        return Right(price)

                return Left("No price data available")

            except Exception as e:
                logger.error(f"Error getting latest price: {e}")
                return Left(str(e))

        return IO(get_price)

    async def stream_market_data(self) -> AsyncIO[AsyncIterator[OptimizedOHLCV]]:
        """Stream real-time market data with optimizations"""

        async def create_stream():
            if not self._is_connected:
                raise ConnectionError("Adapter not connected")

            self._is_streaming = True

            # Subscribe to aggregator candle updates
            candle_queue: asyncio.Queue[OptimizedOHLCV] = asyncio.Queue(maxsize=1000)

            def on_new_candle(candle: OptimizedOHLCV) -> None:
                try:
                    candle_queue.put_nowait(candle)
                except asyncio.QueueFull:
                    logger.warning("Candle queue full, dropping candle")

            # Subscribe to aggregator
            self.aggregator.subscribe_to_candles(on_new_candle).run()

            # Stream candles from queue
            async def candle_stream():
                while self._is_streaming:
                    try:
                        candle = await asyncio.wait_for(candle_queue.get(), timeout=1.0)
                        yield candle
                    except TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error in candle stream: {e}")
                        break

            return candle_stream()

        return AsyncIO(lambda: asyncio.create_task(create_stream()))

    def subscribe_to_ohlcv(
        self, callback: Callable[[OptimizedOHLCV], None]
    ) -> IO[None]:
        """Subscribe to OHLCV updates"""

        def subscribe():
            self._ohlcv_subscribers.append(callback)
            # Also subscribe to aggregator
            self.aggregator.subscribe_to_candles(callback).run()
            logger.debug("Added OHLCV subscriber")

        return IO(subscribe)

    def subscribe_to_legacy_format(
        self, callback: Callable[[LegacyMarketData], None]
    ) -> IO[None]:
        """Subscribe to legacy format updates for backward compatibility"""

        def subscribe():
            self._legacy_subscribers.append(callback)

            # Create wrapper callback that converts to legacy format
            def legacy_wrapper(ohlcv: OptimizedOHLCV) -> None:
                try:
                    legacy_data = LegacyMarketData(
                        symbol=self.config.symbol,
                        timestamp=ohlcv.timestamp,
                        open=ohlcv.open,
                        high=ohlcv.high,
                        low=ohlcv.low,
                        close=ohlcv.close,
                        volume=ohlcv.volume,
                    )
                    callback(legacy_data)
                except Exception as e:
                    logger.error(f"Error in legacy callback: {e}")

            # Subscribe wrapper to aggregator
            self.aggregator.subscribe_to_candles(legacy_wrapper).run()
            logger.debug("Added legacy format subscriber")

        return IO(subscribe)

    def get_performance_metrics(self) -> IO[Either[str, AggregationMetrics]]:
        """Get current performance metrics"""

        def get_metrics():
            try:
                metrics = self.aggregator.get_metrics().run()
                return Right(metrics)
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                return Left(str(e))

        return IO(get_metrics)

    def optimize_performance(self) -> IO[Either[str, dict[str, Any]]]:
        """Run performance optimization"""

        def optimize():
            try:
                # Optimize aggregator memory
                aggregator_stats = self.aggregator.optimize_memory().run()

                # Clear data factory cache
                self.data_factory.clear_cache()

                # Get updated metrics
                metrics = self.aggregator.get_metrics().run()

                optimization_results = {
                    "aggregator_optimization": aggregator_stats,
                    "cache_cleared": True,
                    "current_metrics": {
                        "memory_usage_mb": metrics.memory_usage_mb,
                        "processing_rate": metrics.processing_rate,
                        "avg_latency_ms": metrics.avg_latency_ms,
                    },
                }

                logger.info("Performance optimization completed")
                return Right(optimization_results)

            except Exception as e:
                logger.error(f"Error optimizing performance: {e}")
                return Left(str(e))

        return IO(optimize)

    def get_connection_status(self) -> IO[dict[str, Any]]:
        """Get comprehensive connection status"""

        def get_status():
            try:
                metrics = self.aggregator.get_metrics().run()
                cache_stats = self.data_factory.get_cache_stats()

                return {
                    "connected": self._is_connected,
                    "streaming": self._is_streaming,
                    "legacy_provider_available": self._legacy_provider is not None,
                    "performance_mode": self.config.performance_mode,
                    "metrics": {
                        "trades_processed": metrics.trades_processed,
                        "candles_generated": metrics.candles_generated,
                        "processing_rate": metrics.processing_rate,
                        "memory_usage_mb": metrics.memory_usage_mb,
                        "avg_latency_ms": metrics.avg_latency_ms,
                    },
                    "cache_stats": cache_stats,
                    "subscribers": {
                        "ohlcv": len(self._ohlcv_subscribers),
                        "legacy": len(self._legacy_subscribers),
                        "metrics": len(self._metrics_subscribers),
                    },
                }
            except Exception as e:
                logger.error(f"Error getting connection status: {e}")
                return {"error": str(e)}

        return IO(get_status)

    # Internal methods

    async def _setup_legacy_integration(self) -> None:
        """Set up integration with legacy market data provider"""
        if not self._legacy_provider:
            return

        try:
            # Connect legacy provider if not already connected
            if (
                hasattr(self._legacy_provider, "connect")
                and not self._legacy_provider.is_connected()
            ):
                await self._legacy_provider.connect()

            # Subscribe to legacy updates and convert to optimized format
            def on_legacy_update(market_data: LegacyMarketData) -> None:
                try:
                    # Convert to optimized trade (simulated from OHLCV)
                    trade = create_optimized_trade(
                        id=f"sim_{int(market_data.timestamp.timestamp())}",
                        price=market_data.close,
                        size=market_data.volume / Decimal(100),  # Simulate trade size
                        side="BUY" if market_data.is_bullish else "SELL",
                        symbol=self.config.symbol,
                        timestamp=market_data.timestamp,
                    )

                    # Feed to aggregator
                    self.aggregator.aggregate_trades_batch([trade]).run()

                except Exception as e:
                    logger.error(f"Error processing legacy update: {e}")

            # Subscribe to legacy provider
            if hasattr(self._legacy_provider, "subscribe_to_updates"):
                self._legacy_provider.subscribe_to_updates(on_legacy_update)

            logger.info("Legacy integration set up successfully")

        except Exception as e:
            logger.error(f"Failed to set up legacy integration: {e}")

    def _parse_interval(self, interval: str) -> timedelta:
        """Parse interval string to timedelta"""
        if interval.endswith("s"):
            seconds = int(interval[:-1])
            return timedelta(seconds=seconds)
        if interval.endswith("m"):
            minutes = int(interval[:-1])
            return timedelta(minutes=minutes)
        if interval.endswith("h"):
            hours = int(interval[:-1])
            return timedelta(hours=hours)
        if interval.endswith("d"):
            days = int(interval[:-1])
            return timedelta(days=days)
        # Default to 1 minute
        return timedelta(minutes=1)

    def _batch_validate_ohlcv(
        self, ohlcv_list: list[OptimizedOHLCV]
    ) -> tuple[list[OptimizedOHLCV], list[str]]:
        """Batch validate OHLCV data for performance"""
        valid_data = []
        errors = []

        for i, ohlcv in enumerate(ohlcv_list):
            try:
                # Basic validation checks
                if (
                    ohlcv.high >= max(ohlcv.open, ohlcv.close, ohlcv.low)
                    and ohlcv.low <= min(ohlcv.open, ohlcv.close, ohlcv.high)
                    and all(
                        price > 0
                        for price in [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close]
                    )
                    and ohlcv.volume >= 0
                ):
                    # Outlier detection if enabled
                    if self.config.enable_outlier_detection and valid_data:
                        prev_close = valid_data[-1].close
                        price_change = abs(ohlcv.close - prev_close) / prev_close
                        if price_change > self.config.outlier_threshold:
                            errors.append(
                                f"Outlier detected at index {i}: {price_change:.3%} change"
                            )
                            continue

                    valid_data.append(ohlcv)
                else:
                    errors.append(f"Invalid OHLCV at index {i}")
            except Exception as e:
                errors.append(f"Error validating index {i}: {e}")

        return valid_data, errors


# Factory functions


def create_optimized_adapter(
    symbol: str,
    interval: str = "1m",
    exchange_type: str = "coinbase",
    performance_mode: str = "balanced",
) -> OptimizedMarketDataAdapter:
    """Create optimized market data adapter with specified configuration"""
    config = OptimizedAdapterConfig(
        symbol=symbol,
        interval=interval,
        exchange_type=exchange_type,
        performance_mode=performance_mode,
    )
    return OptimizedMarketDataAdapter(config)


def create_high_performance_adapter(
    symbol: str, interval: str = "1m", exchange_type: str = "coinbase"
) -> OptimizedMarketDataAdapter:
    """Create adapter optimized for high throughput"""
    config = OptimizedAdapterConfig(
        symbol=symbol,
        interval=interval,
        exchange_type=exchange_type,
        performance_mode="high_throughput",
        buffer_size=50000,
        cache_size=5000,
        stream_batch_size=500,
        max_memory_mb=500.0,
    )
    return OptimizedMarketDataAdapter(config)


def create_low_latency_adapter(
    symbol: str, interval: str = "1m", exchange_type: str = "coinbase"
) -> OptimizedMarketDataAdapter:
    """Create adapter optimized for low latency"""
    config = OptimizedAdapterConfig(
        symbol=symbol,
        interval=interval,
        exchange_type=exchange_type,
        performance_mode="low_latency",
        buffer_size=5000,
        enable_caching=False,
        stream_batch_size=50,
        update_frequency_ms=10,
        max_memory_mb=50.0,
    )
    return OptimizedMarketDataAdapter(config)
