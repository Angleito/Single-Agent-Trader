"""
Enhanced Data Runtime Integration

This module provides a comprehensive runtime that demonstrates how to use the enhanced
functional data layer capabilities alongside the existing imperative system.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..adapters.enhanced_market_data_adapter import (
    EnhancedFunctionalMarketDataAdapter,
    create_enhanced_market_data_adapter
)
from ..data_pipeline import FunctionalDataPipeline, create_high_performance_pipeline
from ..effects.market_data_aggregation import RealTimeAggregator, create_high_frequency_aggregator
from ..types.market import Candle, MarketSnapshot
from ..types.result import Result
from ..effects.io import IO, AsyncIO
from ...config import settings

logger = logging.getLogger(__name__)


@dataclass
class DataRuntimeConfig:
    """Configuration for enhanced data runtime"""
    symbol: str
    interval: str
    exchange_type: str = "coinbase"
    performance_mode: str = "balanced"  # "low_latency", "high_throughput", "balanced"
    enable_fallback_to_imperative: bool = True
    enable_performance_monitoring: bool = True
    enable_automatic_optimization: bool = True
    optimization_interval: timedelta = timedelta(minutes=5)


class EnhancedDataRuntime:
    """
    Enhanced data runtime that integrates functional and imperative approaches.
    
    Provides:
    - Automatic fallback between functional and imperative data sources
    - Performance monitoring and optimization
    - Real-time data streaming with enhanced capabilities
    - Comprehensive error handling and recovery
    """

    def __init__(self, config: DataRuntimeConfig):
        self.config = config
        self._enhanced_adapter: EnhancedFunctionalMarketDataAdapter | None = None
        self._fallback_provider: Any = None
        self._is_running = False
        self._performance_metrics = {}
        self._optimization_task: asyncio.Task | None = None


    # Initialization and Connection

    async def initialize(self) -> Result[bool, str]:
        """Initialize the enhanced data runtime"""
        try:
            logger.info(f"Initializing enhanced data runtime for {self.config.symbol}")
            
            # Create enhanced functional adapter
            self._enhanced_adapter = create_enhanced_market_data_adapter(
                exchange_type=self.config.exchange_type,
                symbol=self.config.symbol,
                interval=self.config.interval,
                performance_mode=self.config.performance_mode
            )
            
            # Connect enhanced adapter
            connection_result = self._enhanced_adapter.connect().run()
            if connection_result.is_left():
                if self.config.enable_fallback_to_imperative:
                    logger.warning("Enhanced adapter failed, setting up fallback")
                    await self._setup_fallback_provider()
                else:
                    return Result.error(f"Enhanced adapter connection failed: {connection_result.error}")
            
            self._is_running = True
            
            # Start performance monitoring if enabled
            if self.config.enable_performance_monitoring:
                self._start_performance_monitoring()
            
            # Start automatic optimization if enabled
            if self.config.enable_automatic_optimization:
                self._start_automatic_optimization()
            
            logger.info("Enhanced data runtime initialized successfully")
            return Result.ok(True)
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced data runtime: {e}")
            return Result.error(str(e))


    async def _setup_fallback_provider(self) -> None:
        """Set up fallback imperative provider"""
        try:
            if self.config.exchange_type == "bluefin":
                from ...data.bluefin_market import BluefinMarketDataProvider
                self._fallback_provider = BluefinMarketDataProvider(
                    symbol=self.config.symbol,
                    interval=self.config.interval
                )
            else:
                from ...data.market import MarketDataProvider
                self._fallback_provider = MarketDataProvider(
                    symbol=self.config.symbol,
                    interval=self.config.interval
                )
            
            await self._fallback_provider.connect()
            logger.info("Fallback provider connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup fallback provider: {e}")


    async def shutdown(self) -> None:
        """Shutdown the enhanced data runtime"""
        logger.info("Shutting down enhanced data runtime")
        
        self._is_running = False
        
        # Cancel background tasks
        if self._optimization_task and not self._optimization_task.done():
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect adapters
        if self._enhanced_adapter:
            self._enhanced_adapter.disconnect().run()
        
        if self._fallback_provider:
            await self._fallback_provider.disconnect()
        
        logger.info("Enhanced data runtime shutdown complete")


    # Data Access Methods

    async def get_historical_data_enhanced(
        self,
        lookback_hours: int = 24,
        apply_functional_processing: bool = True
    ) -> Result[list, str]:
        """Get historical data with enhanced functional processing"""
        try:
            if self._enhanced_adapter:
                # Try enhanced adapter first
                start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
                result = self._enhanced_adapter.fetch_historical_data_enhanced(
                    start_time=start_time,
                    apply_functional_processing=apply_functional_processing
                ).run()
                
                if result.is_right():
                    logger.debug(f"Retrieved {len(result.value)} historical candles via enhanced adapter")
                    return Result.ok(result.value)
                else:
                    logger.warning(f"Enhanced adapter failed: {result.error}")
            
            # Fallback to imperative provider
            if self._fallback_provider:
                logger.info("Using fallback provider for historical data")
                start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
                data = await self._fallback_provider.fetch_historical_data(
                    start_time=start_time,
                    granularity=self.config.interval
                )
                return Result.ok(data)
            
            return Result.error("No data providers available")
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return Result.error(str(e))


    async def stream_market_data_enhanced(self) -> AsyncIterator[MarketSnapshot]:
        """Stream market data with enhanced functional processing"""
        if self._enhanced_adapter:
            try:
                # Use enhanced streaming
                stream_result = self._enhanced_adapter.stream_market_data_enhanced()
                async_stream = await stream_result.run()
                
                logger.info("Starting enhanced market data stream")
                async for snapshot in async_stream:
                    yield snapshot
                    
            except Exception as e:
                logger.error(f"Enhanced stream failed: {e}")
                
                # Fallback to imperative stream if available
                if self._fallback_provider:
                    logger.info("Falling back to imperative market data stream")
                    async for data in self._stream_from_fallback():
                        yield data
        
        elif self._fallback_provider:
            # Direct fallback streaming
            logger.info("Using fallback provider for streaming")
            async for data in self._stream_from_fallback():
                yield data


    async def _stream_from_fallback(self) -> AsyncIterator[MarketSnapshot]:
        """Stream from fallback imperative provider"""
        # Set up fallback streaming
        data_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        
        def on_fallback_update(market_data):
            try:
                # Convert to MarketSnapshot
                snapshot = MarketSnapshot(
                    timestamp=market_data.timestamp,
                    symbol=self.config.symbol,
                    price=market_data.close,
                    volume=market_data.volume,
                    bid=market_data.close * 0.999,  # Simple spread
                    ask=market_data.close * 1.001
                )
                
                try:
                    data_queue.put_nowait(snapshot)
                except asyncio.QueueFull:
                    logger.warning("Fallback stream queue full")
                    
            except Exception as e:
                logger.error(f"Error in fallback stream: {e}")
        
        self._fallback_provider.subscribe_to_updates(on_fallback_update)
        
        while self._is_running:
            try:
                snapshot = await asyncio.wait_for(data_queue.get(), timeout=10.0)
                yield snapshot
            except asyncio.TimeoutError:
                if not self._fallback_provider.is_connected():
                    break
                continue
            except Exception as e:
                logger.error(f"Error in fallback streaming: {e}")
                break


    def get_aggregated_candles_enhanced(
        self,
        interval: str | None = None,
        limit: int = 100
    ) -> Result[list[Candle], str]:
        """Get aggregated candles with enhanced functional processing"""
        try:
            if self._enhanced_adapter:
                result = self._enhanced_adapter.get_aggregated_candles(
                    interval=interval,
                    limit=limit
                ).run()
                
                if result.is_right():
                    return Result.ok(result.value)
                else:
                    logger.warning(f"Enhanced aggregation failed: {result.error}")
            
            # Fallback to basic OHLCV data
            if self._fallback_provider:
                data = self._fallback_provider.get_latest_ohlcv(limit)
                # Convert to Candle format
                candles = [
                    Candle(
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume
                    )
                    for candle in data
                ]
                return Result.ok(candles)
            
            return Result.error("No aggregation providers available")
            
        except Exception as e:
            logger.error(f"Error getting aggregated candles: {e}")
            return Result.error(str(e))


    # Performance Monitoring and Optimization

    def _start_performance_monitoring(self) -> None:
        """Start background performance monitoring"""
        async def monitor_performance():
            while self._is_running:
                try:
                    await self._collect_performance_metrics()
                    await asyncio.sleep(30)  # Monitor every 30 seconds
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        asyncio.create_task(monitor_performance())
        logger.info("Performance monitoring started")


    def _start_automatic_optimization(self) -> None:
        """Start automatic performance optimization"""
        async def optimize_performance():
            while self._is_running:
                try:
                    await asyncio.sleep(self.config.optimization_interval.total_seconds())
                    await self._run_automatic_optimization()
                except Exception as e:
                    logger.error(f"Automatic optimization error: {e}")
        
        self._optimization_task = asyncio.create_task(optimize_performance())
        logger.info("Automatic optimization started")


    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics from all components"""
        metrics = {
            "timestamp": datetime.utcnow(),
            "runtime_status": {
                "running": self._is_running,
                "enhanced_adapter_available": self._enhanced_adapter is not None,
                "fallback_provider_available": self._fallback_provider is not None
            }
        }
        
        # Get enhanced adapter metrics
        if self._enhanced_adapter:
            try:
                enhanced_metrics = self._enhanced_adapter.get_enhanced_metrics().run()
                metrics["enhanced_adapter"] = enhanced_metrics
            except Exception as e:
                logger.warning(f"Failed to get enhanced metrics: {e}")
        
        # Get fallback provider metrics
        if self._fallback_provider:
            try:
                fallback_status = self._fallback_provider.get_data_status()
                metrics["fallback_provider"] = fallback_status
            except Exception as e:
                logger.warning(f"Failed to get fallback metrics: {e}")
        
        self._performance_metrics = metrics
        
        # Log performance summary
        if self._enhanced_adapter:
            logger.debug("Performance metrics collected successfully")


    async def _run_automatic_optimization(self) -> None:
        """Run automatic performance optimization"""
        if not self._enhanced_adapter:
            return
        
        try:
            # Run optimization
            optimization_result = self._enhanced_adapter.optimize_performance().run()
            logger.info(f"Automatic optimization completed: {optimization_result}")
            
            # Check if we need to switch performance modes
            await self._check_performance_mode_optimization()
            
        except Exception as e:
            logger.error(f"Automatic optimization failed: {e}")


    async def _check_performance_mode_optimization(self) -> None:
        """Check if performance mode should be optimized based on metrics"""
        if not self._performance_metrics:
            return
        
        try:
            enhanced_metrics = self._performance_metrics.get("enhanced_adapter", {})
            performance = enhanced_metrics.get("performance", {})
            
            latency = performance.get("latency_ms", 0)
            throughput = performance.get("throughput_per_sec", 0)
            
            # Simple optimization logic
            if latency > 100 and self.config.performance_mode != "low_latency":
                logger.info("High latency detected, recommending low_latency mode")
            elif throughput > 1000 and self.config.performance_mode != "high_throughput":
                logger.info("High throughput detected, recommending high_throughput mode")
            
        except Exception as e:
            logger.error(f"Performance mode optimization check failed: {e}")


    # Status and Diagnostics

    def get_runtime_status(self) -> dict[str, Any]:
        """Get comprehensive runtime status"""
        return {
            "config": {
                "symbol": self.config.symbol,
                "interval": self.config.interval,
                "exchange_type": self.config.exchange_type,
                "performance_mode": self.config.performance_mode
            },
            "status": {
                "running": self._is_running,
                "enhanced_adapter_connected": (
                    self._enhanced_adapter is not None and 
                    self._enhanced_adapter._is_connected
                ),
                "fallback_provider_connected": (
                    self._fallback_provider is not None and
                    self._fallback_provider.is_connected()
                )
            },
            "performance_metrics": self._performance_metrics,
            "capabilities": {
                "enhanced_websocket": True,
                "functional_pipeline": True,
                "real_time_aggregation": True,
                "automatic_optimization": self.config.enable_automatic_optimization,
                "performance_monitoring": self.config.enable_performance_monitoring
            }
        }


    def run_diagnostics(self) -> dict[str, Any]:
        """Run comprehensive diagnostics"""
        diagnostics = {
            "timestamp": datetime.utcnow(),
            "runtime_health": "healthy",
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check enhanced adapter health
            if self._enhanced_adapter:
                enhanced_status = self._enhanced_adapter.get_enhanced_metrics().run()
                if not enhanced_status.get("connection_status", {}).get("connected", False):
                    diagnostics["issues"].append("Enhanced adapter not connected")
                    diagnostics["runtime_health"] = "degraded"
            else:
                diagnostics["issues"].append("Enhanced adapter not available")
                diagnostics["runtime_health"] = "degraded"
            
            # Check fallback provider health
            if self._fallback_provider:
                if not self._fallback_provider.is_connected():
                    diagnostics["issues"].append("Fallback provider not connected")
                    if diagnostics["runtime_health"] == "degraded":
                        diagnostics["runtime_health"] = "critical"
            else:
                if diagnostics["runtime_health"] == "degraded":
                    diagnostics["issues"].append("No fallback provider available")
                    diagnostics["runtime_health"] = "critical"
            
            # Add recommendations based on issues
            if diagnostics["issues"]:
                diagnostics["recommendations"].append("Check network connectivity")
                diagnostics["recommendations"].append("Verify exchange API credentials")
                diagnostics["recommendations"].append("Review configuration settings")
            
        except Exception as e:
            diagnostics["issues"].append(f"Diagnostics error: {str(e)}")
            diagnostics["runtime_health"] = "unknown"
        
        return diagnostics


# Factory Functions

async def create_and_initialize_runtime(
    symbol: str | None = None,
    interval: str | None = None,
    exchange_type: str | None = None,
    performance_mode: str = "balanced"
) -> Result[EnhancedDataRuntime, str]:
    """Create and initialize enhanced data runtime"""
    try:
        config = DataRuntimeConfig(
            symbol=symbol or settings.trading.symbol,
            interval=interval or settings.trading.interval,
            exchange_type=exchange_type or settings.exchange.exchange_type,
            performance_mode=performance_mode
        )
        
        runtime = EnhancedDataRuntime(config)
        initialization_result = await runtime.initialize()
        
        if initialization_result.is_error():
            return Result.error(initialization_result.error_value)
        
        return Result.ok(runtime)
        
    except Exception as e:
        return Result.error(f"Failed to create runtime: {str(e)}")


async def create_high_performance_runtime(
    symbol: str | None = None,
    interval: str | None = None
) -> Result[EnhancedDataRuntime, str]:
    """Create runtime optimized for high performance"""
    return await create_and_initialize_runtime(
        symbol=symbol,
        interval=interval,
        performance_mode="high_throughput"
    )


async def create_low_latency_runtime(
    symbol: str | None = None,
    interval: str | None = None
) -> Result[EnhancedDataRuntime, str]:
    """Create runtime optimized for low latency"""
    return await create_and_initialize_runtime(
        symbol=symbol,
        interval=interval,
        performance_mode="low_latency"
    )