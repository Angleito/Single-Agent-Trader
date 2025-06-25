"""
Enhanced Functional Market Data Adapter

This module provides an enhanced functional adapter that integrates the new functional
data processing capabilities with the existing imperative market data providers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot.config import settings
from bot.fp.data_pipeline import FunctionalDataPipeline, create_market_data_pipeline
from bot.fp.effects.io import IO, AsyncIO, IOEither, from_try
from bot.fp.effects.market_data_aggregation import (
    RealTimeAggregator,
    create_real_time_aggregator,
)
from bot.fp.effects.websocket_enhanced import (
    EnhancedWebSocketManager,
    create_enhanced_websocket_manager,
)
from bot.fp.types.market import OHLCV, Candle, MarketSnapshot, Trade

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMarketDataConfig:
    """Enhanced configuration for market data adapter"""

    symbol: str
    interval: str
    exchange_type: str = "coinbase"
    use_trade_aggregation: bool = False
    candle_limit: int = 200
    enable_functional_pipeline: bool = True
    enable_enhanced_websocket: bool = True
    enable_real_time_aggregation: bool = True
    websocket_url: str | None = None
    performance_mode: str = "balanced"  # "low_latency", "high_throughput", "balanced"


class EnhancedFunctionalMarketDataAdapter:
    """
    Enhanced functional adapter for market data with performance optimizations.

    Integrates:
    - Existing imperative market data providers
    - Functional data processing pipelines
    - Enhanced WebSocket management
    - Real-time data aggregation
    - Performance monitoring and optimization
    """

    def __init__(self, config: EnhancedMarketDataConfig):
        self.config = config
        self._imperative_provider: Any = None
        self._functional_pipeline: FunctionalDataPipeline | None = None
        self._websocket_manager: EnhancedWebSocketManager | None = None
        self._aggregator: RealTimeAggregator | None = None
        self._is_connected = False
        self._performance_metrics = {
            "latency_ms": 0.0,
            "throughput_per_sec": 0.0,
            "error_rate": 0.0,
            "memory_usage_mb": 0.0,
        }

    # Connection Management

    def connect(self) -> IOEither[Exception, bool]:
        """Connect with enhanced functional capabilities"""

        async def enhanced_connect():
            try:
                # Initialize functional components based on configuration
                if self.config.enable_functional_pipeline:
                    self._functional_pipeline = (
                        self._create_pipeline_for_performance_mode()
                    )

                if self.config.enable_real_time_aggregation:
                    self._aggregator = self._create_aggregator_for_performance_mode()

                if self.config.enable_enhanced_websocket and self.config.websocket_url:
                    self._websocket_manager = self._create_websocket_manager()
                    ws_result = self._websocket_manager.connect().run()
                    if ws_result.is_left():
                        logger.warning(
                            "Enhanced WebSocket connection failed, falling back to standard"
                        )

                # Connect imperative provider
                await self._connect_imperative_provider()

                # Set up data flow integration
                if self._imperative_provider and self._aggregator:
                    self._setup_data_flow_integration()

                self._is_connected = True
                logger.info(
                    f"Enhanced functional adapter connected for {self.config.symbol} "
                    f"in {self.config.performance_mode} mode"
                )
                return True

            except Exception as e:
                logger.exception(f"Enhanced connection failed: {e}")
                raise

        return from_try(lambda: asyncio.run(enhanced_connect()))

    async def _connect_imperative_provider(self) -> None:
        """Connect the underlying imperative provider"""
        if self.config.exchange_type == "bluefin":
            from bot.data.bluefin_market import BluefinMarketDataProvider

            self._imperative_provider = BluefinMarketDataProvider(
                symbol=self.config.symbol, interval=self.config.interval
            )
        else:
            from bot.data.market import MarketDataProvider

            self._imperative_provider = MarketDataProvider(
                symbol=self.config.symbol, interval=self.config.interval
            )

        await self._imperative_provider.connect()

    def _create_pipeline_for_performance_mode(self) -> FunctionalDataPipeline:
        """Create pipeline optimized for selected performance mode"""
        if self.config.performance_mode == "low_latency":
            return create_market_data_pipeline(
                buffer_size=100, enable_batching=False, batch_size=1
            )
        if self.config.performance_mode == "high_throughput":
            return create_market_data_pipeline(
                buffer_size=10000, enable_batching=True, batch_size=1000
            )
        # balanced
        return create_market_data_pipeline(
            buffer_size=1000, enable_batching=True, batch_size=100
        )

    def _create_aggregator_for_performance_mode(self) -> RealTimeAggregator:
        """Create aggregator optimized for selected performance mode"""
        interval = self._parse_interval_to_timedelta(self.config.interval)

        if self.config.performance_mode == "low_latency":
            from bot.fp.effects.market_data_aggregation import (
                AggregationConfig,
                RealTimeAggregator,
            )

            config = AggregationConfig(
                interval=interval,
                buffer_size=500,
                enable_real_time_updates=True,
                enable_volume_weighting=False,  # Disable for speed
                enable_outlier_detection=False,  # Disable for speed
            )
            return RealTimeAggregator(config)
        if self.config.performance_mode == "high_throughput":
            from bot.fp.effects.market_data_aggregation import (
                AggregationConfig,
                RealTimeAggregator,
            )

            config = AggregationConfig(
                interval=interval,
                buffer_size=50000,
                enable_real_time_updates=True,
                enable_volume_weighting=True,
                enable_outlier_detection=True,
                enable_compression=True,
            )
            return RealTimeAggregator(config)
        # balanced
        return create_real_time_aggregator(
            interval=interval, buffer_size=2000, enable_volume_weighting=True
        )

    def _create_websocket_manager(self) -> EnhancedWebSocketManager:
        """Create WebSocket manager optimized for performance mode"""
        if not self.config.websocket_url:
            raise ValueError("WebSocket URL required for enhanced WebSocket mode")

        if self.config.performance_mode == "low_latency":
            from bot.fp.effects.websocket_enhanced import (
                create_low_latency_websocket_manager,
            )

            return create_low_latency_websocket_manager(self.config.websocket_url)
        if self.config.performance_mode == "high_throughput":
            from bot.fp.effects.websocket_enhanced import (
                create_high_reliability_websocket_manager,
            )

            return create_high_reliability_websocket_manager(self.config.websocket_url)
        # balanced
        return create_enhanced_websocket_manager(
            url=self.config.websocket_url, heartbeat_interval=30, reconnect_attempts=10
        )

    def _setup_data_flow_integration(self) -> None:
        """Set up integration between functional and imperative components"""
        if not self._imperative_provider or not self._aggregator:
            return

        def on_market_update(market_data):
            """Callback to integrate imperative data with functional processing"""
            try:
                # Convert imperative MarketData to functional Trade
                trade = Trade(
                    id=f"{market_data.symbol}_{market_data.timestamp.isoformat()}",
                    timestamp=market_data.timestamp,
                    price=market_data.close,
                    size=market_data.volume,
                    side="BUY",  # Simplified for aggregation
                )

                # Process through functional aggregator
                if self._aggregator:
                    candles = self._aggregator.aggregate_trades_real_time([trade]).run()
                    for candle in candles:
                        logger.debug(f"Functional aggregator produced candle: {candle}")

            except Exception as e:
                logger.exception(f"Error in data flow integration: {e}")

        # Subscribe to imperative provider updates
        self._imperative_provider.subscribe_to_updates(on_market_update)

    def disconnect(self) -> IOEither[Exception, None]:
        """Enhanced disconnect with cleanup"""

        async def enhanced_disconnect():
            # Disconnect imperative provider
            if self._imperative_provider and self._is_connected:
                await self._imperative_provider.disconnect()

            # Disconnect enhanced WebSocket
            if self._websocket_manager:
                self._websocket_manager.disconnect().run()

            self._is_connected = False
            logger.info("Enhanced functional adapter disconnected")

        return from_try(lambda: asyncio.run(enhanced_disconnect()))

    # Enhanced Data Access

    def fetch_historical_data_enhanced(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        apply_functional_processing: bool = True,
    ) -> IOEither[Exception, list[OHLCV]]:
        """Fetch historical data with optional functional processing"""

        async def fetch_enhanced():
            if not self._imperative_provider:
                raise RuntimeError("Provider not connected")

            # Fetch data from imperative provider
            data = await self._imperative_provider.fetch_historical_data(
                start_time=start_time,
                end_time=end_time,
                granularity=self.config.interval,
            )

            # Convert to functional types
            candles_to_convert = data[-limit:] if limit else data
            ohlcv_data = [
                OHLCV(
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    timestamp=candle.timestamp,
                )
                for candle in candles_to_convert
            ]

            # Apply functional processing if enabled
            if apply_functional_processing and self._functional_pipeline:
                # Normalize prices for better processing
                normalized_result = self._functional_pipeline.normalize_prices(
                    ohlcv_data
                ).run()

                # Validate data quality
                quality_result = self._functional_pipeline.validate_data_quality(
                    normalized_result
                ).run()
                if quality_result.is_right():
                    ohlcv_data = quality_result.value
                else:
                    logger.warning(f"Data quality issues: {quality_result.error}")

            return ohlcv_data

        return from_try(lambda: asyncio.run(fetch_enhanced()))

    def stream_market_data_enhanced(self) -> AsyncIO[AsyncIterator[MarketSnapshot]]:
        """Stream market data with enhanced functional processing"""

        async def create_enhanced_stream():
            if not self._imperative_provider:
                raise RuntimeError("Provider not connected")

            # Set up data queue for enhanced processing
            data_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

            def on_enhanced_update(market_data):
                """Enhanced callback with functional processing"""
                try:
                    # Apply functional transformations if available
                    if self._functional_pipeline:
                        # Convert to OHLCV and apply pipeline processing
                        ohlcv = OHLCV(
                            open=market_data.open,
                            high=market_data.high,
                            low=market_data.low,
                            close=market_data.close,
                            volume=market_data.volume,
                            timestamp=market_data.timestamp,
                        )

                        # Apply data quality validation
                        quality_result = (
                            self._functional_pipeline.validate_data_quality(
                                [ohlcv]
                            ).run()
                        )
                        if quality_result.is_left():
                            logger.warning(
                                f"Rejecting low-quality data: {quality_result.error}"
                            )
                            return

                    # Convert to MarketSnapshot
                    snapshot = self._convert_to_market_snapshot(market_data)

                    try:
                        data_queue.put_nowait(snapshot)
                    except asyncio.QueueFull:
                        logger.warning("Enhanced stream queue full, dropping update")

                except Exception as e:
                    logger.exception(f"Error in enhanced stream processing: {e}")

            # Subscribe to updates
            self._imperative_provider.subscribe_to_updates(on_enhanced_update)

            async def enhanced_stream():
                while self._is_connected:
                    try:
                        snapshot = await asyncio.wait_for(
                            data_queue.get(), timeout=10.0
                        )
                        yield snapshot
                    except TimeoutError:
                        if not self._imperative_provider.is_connected():
                            logger.warning("Provider disconnected")
                            break
                        continue
                    except Exception as e:
                        logger.exception(f"Error in enhanced stream: {e}")
                        break

            return enhanced_stream()

        return AsyncIO(lambda: asyncio.create_task(create_enhanced_stream()))

    def get_aggregated_candles(
        self, interval: str | None = None, limit: int = 100
    ) -> IOEither[Exception, list[Candle]]:
        """Get aggregated candles from functional aggregator"""

        def get_candles():
            if not self._aggregator:
                raise RuntimeError("Real-time aggregator not available")

            # Get completed candles
            completed = self._aggregator.get_completed_candles(limit).run()

            # If different interval requested, resample
            if interval and interval != self.config.interval:
                target_interval = self._parse_interval_to_timedelta(interval)
                ohlcv_data = [
                    OHLCV(
                        open=c.open,
                        high=c.high,
                        low=c.low,
                        close=c.close,
                        volume=c.volume,
                        timestamp=c.timestamp,
                    )
                    for c in completed
                ]

                resampled = self._aggregator.resample_candles(
                    ohlcv_data, target_interval
                ).run()
                return [
                    Candle(
                        timestamp=ohlcv.timestamp,
                        open=ohlcv.open,
                        high=ohlcv.high,
                        low=ohlcv.low,
                        close=ohlcv.close,
                        volume=ohlcv.volume,
                    )
                    for ohlcv in resampled
                ]

            return completed

        return from_try(get_candles)

    # Performance Monitoring

    def get_enhanced_metrics(self) -> IO[dict[str, Any]]:
        """Get comprehensive performance metrics"""

        def get_metrics():
            metrics = {
                "adapter_config": {
                    "symbol": self.config.symbol,
                    "interval": self.config.interval,
                    "exchange": self.config.exchange_type,
                    "performance_mode": self.config.performance_mode,
                    "functional_pipeline_enabled": self.config.enable_functional_pipeline,
                    "enhanced_websocket_enabled": self.config.enable_enhanced_websocket,
                    "real_time_aggregation_enabled": self.config.enable_real_time_aggregation,
                },
                "connection_status": {
                    "connected": self._is_connected,
                    "imperative_provider_connected": (
                        self._imperative_provider.is_connected()
                        if self._imperative_provider
                        else False
                    ),
                    "websocket_manager_connected": (
                        self._websocket_manager.check_connection_health().run()
                        if self._websocket_manager
                        else False
                    ),
                },
                "performance": self._performance_metrics,
            }

            # Add functional component metrics
            if self._functional_pipeline:
                metrics["pipeline_metrics"] = (
                    self._functional_pipeline.get_metrics().run()
                )

            if self._aggregator:
                metrics["aggregation_metrics"] = self._aggregator.get_metrics().run()

            if self._websocket_manager:
                metrics["websocket_metrics"] = (
                    self._websocket_manager.get_connection_metrics().run()
                )

            return metrics

        return IO(get_metrics)

    def optimize_performance(self) -> IO[dict[str, Any]]:
        """Optimize adapter performance"""

        def optimize():
            optimizations = {}

            # Optimize memory usage in functional components
            if self._functional_pipeline:
                optimizations["pipeline_memory_freed"] = (
                    self._functional_pipeline.optimize_memory_usage([]).run()
                )

            if self._aggregator:
                optimizations["aggregator_memory_freed"] = (
                    self._aggregator.optimize_memory_usage().run()
                )

            # Calculate processing rates
            if self._aggregator:
                optimizations["processing_rate"] = (
                    self._aggregator.calculate_processing_rate().run()
                )

            logger.info(f"Performance optimization completed: {optimizations}")
            return optimizations

        return IO(optimize)

    # Utility Methods

    def _convert_to_market_snapshot(self, market_data) -> MarketSnapshot:
        """Convert imperative MarketData to functional MarketSnapshot"""
        price = market_data.close
        volume = market_data.volume

        # Create simple bid/ask spread
        spread = price * Decimal("0.001")
        bid = price - spread / 2
        ask = price + spread / 2

        return MarketSnapshot(
            timestamp=market_data.timestamp,
            symbol=self.config.symbol,
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
        )

    def _parse_interval_to_timedelta(self, interval: str) -> timedelta:
        """Parse interval string to timedelta"""
        if interval.endswith("s"):
            return timedelta(seconds=int(interval[:-1]))
        if interval.endswith("m"):
            return timedelta(minutes=int(interval[:-1]))
        if interval.endswith("h"):
            return timedelta(hours=int(interval[:-1]))
        if interval.endswith("d"):
            return timedelta(days=int(interval[:-1]))
        # Default to minutes
        return timedelta(minutes=int(interval[:-1]) if interval[:-1].isdigit() else 1)


# Factory Functions


def create_enhanced_coinbase_adapter(
    symbol: str | None = None,
    interval: str | None = None,
    performance_mode: str = "balanced",
) -> EnhancedFunctionalMarketDataAdapter:
    """Create enhanced functional adapter for Coinbase"""
    config = EnhancedMarketDataConfig(
        symbol=symbol or settings.trading.symbol,
        interval=interval or settings.trading.interval,
        exchange_type="coinbase",
        performance_mode=performance_mode,
        websocket_url="wss://advanced-trade-ws.coinbase.com",
    )
    return EnhancedFunctionalMarketDataAdapter(config)


def create_enhanced_bluefin_adapter(
    symbol: str | None = None,
    interval: str | None = None,
    performance_mode: str = "balanced",
) -> EnhancedFunctionalMarketDataAdapter:
    """Create enhanced functional adapter for Bluefin"""
    network = getattr(settings.exchange, "bluefin_network", "mainnet")
    ws_url = (
        "wss://dapi.api.sui-prod.bluefin.io"
        if network == "mainnet"
        else "wss://dapi.api.sui-staging.bluefin.io"
    )

    config = EnhancedMarketDataConfig(
        symbol=symbol or settings.trading.symbol,
        interval=interval or settings.trading.interval,
        exchange_type="bluefin",
        use_trade_aggregation=settings.exchange.use_trade_aggregation,
        performance_mode=performance_mode,
        websocket_url=ws_url,
    )
    return EnhancedFunctionalMarketDataAdapter(config)


def create_enhanced_market_data_adapter(
    exchange_type: str | None = None,
    symbol: str | None = None,
    interval: str | None = None,
    performance_mode: str = "balanced",
) -> EnhancedFunctionalMarketDataAdapter:
    """Create enhanced functional adapter based on exchange type"""
    exchange = exchange_type or settings.exchange.exchange_type

    if exchange == "bluefin":
        return create_enhanced_bluefin_adapter(symbol, interval, performance_mode)
    return create_enhanced_coinbase_adapter(symbol, interval, performance_mode)
