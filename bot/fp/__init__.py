"""
Functional Programming Framework for Trading Bot

This module provides a comprehensive functional programming framework that enhances
the existing imperative trading bot with pure functional capabilities, improved
performance, and advanced data processing features.

Key Components:
- Enhanced market data processing with functional pipelines
- Real-time data aggregation and streaming
- Enhanced WebSocket connection management
- Performance optimization and monitoring
- Automatic fallback and reliability features

Usage:
    # Create enhanced market data adapter
    from bot.fp.adapters.enhanced_market_data_adapter import create_enhanced_coinbase_adapter

    adapter = create_enhanced_coinbase_adapter(
        symbol="BTC-USD",
        interval="1m",
        performance_mode="balanced"
    )

    # Initialize enhanced runtime
    from bot.fp.runtime.enhanced_data_runtime import create_and_initialize_runtime

    runtime = await create_and_initialize_runtime(
        symbol="BTC-USD",
        performance_mode="low_latency"
    )
"""

# Core functional types and effects
from .effects.io import IO, AsyncIO, IOEither
from .types.market import OHLCV, Candle, MarketSnapshot, OrderBook, Trade
from .types.result import Err, Ok, Result

# Re-export standardized types for consistent usage
try:
    from ..types import (
        ConnectionState,
        MarketDataStatus,
        OrderId,
        Price,
        Quantity,
        Symbol,
        ValidationResult,
        ensure_decimal,
        is_valid_price,
        is_valid_quantity,
    )
except ImportError:
    # Fallback if types module is not available
    from ..types.base_types import (
        OrderId,
        Price,
        Quantity,
        Symbol,
    )
    from ..types.market_data import (
        ConnectionState,
        MarketDataStatus,
    )


# Enhanced data processing components - using lazy loading to avoid circular imports
def get_functional_data_pipeline():
    """Get FunctionalDataPipeline class (lazy loaded)"""
    from .data_pipeline import FunctionalDataPipeline

    return FunctionalDataPipeline


def create_market_data_pipeline(*args, **kwargs):
    """Create market data pipeline (lazy loaded)"""
    from .data_pipeline import create_market_data_pipeline as _create

    return _create(*args, **kwargs)


def create_high_performance_pipeline(*args, **kwargs):
    """Create high performance pipeline (lazy loaded)"""
    from .data_pipeline import create_high_performance_pipeline as _create

    return _create(*args, **kwargs)


def create_low_latency_pipeline(*args, **kwargs):
    """Create low latency pipeline (lazy loaded)"""
    from .data_pipeline import create_low_latency_pipeline as _create

    return _create(*args, **kwargs)


# Real-time aggregation - lazy loading
def get_real_time_aggregator():
    """Get RealTimeAggregator class (lazy loaded)"""
    from .effects.market_data_aggregation import RealTimeAggregator

    return RealTimeAggregator


def create_real_time_aggregator(*args, **kwargs):
    """Create real-time aggregator (lazy loaded)"""
    from .effects.market_data_aggregation import create_real_time_aggregator as _create

    return _create(*args, **kwargs)


def create_high_frequency_aggregator(*args, **kwargs):
    """Create high frequency aggregator (lazy loaded)"""
    from .effects.market_data_aggregation import (
        create_high_frequency_aggregator as _create,
    )

    return _create(*args, **kwargs)


def create_batch_aggregator(*args, **kwargs):
    """Create batch aggregator (lazy loaded)"""
    from .effects.market_data_aggregation import create_batch_aggregator as _create

    return _create(*args, **kwargs)


# Enhanced WebSocket - lazy loading
def get_enhanced_websocket_manager():
    """Get EnhancedWebSocketManager class (lazy loaded)"""
    from .effects.websocket_enhanced import EnhancedWebSocketManager

    return EnhancedWebSocketManager


def create_enhanced_websocket_manager(*args, **kwargs):
    """Create enhanced WebSocket manager (lazy loaded)"""
    from .effects.websocket_enhanced import create_enhanced_websocket_manager as _create

    return _create(*args, **kwargs)


def create_high_reliability_websocket_manager(*args, **kwargs):
    """Create high reliability WebSocket manager (lazy loaded)"""
    from .effects.websocket_enhanced import (
        create_high_reliability_websocket_manager as _create,
    )

    return _create(*args, **kwargs)


def create_low_latency_websocket_manager(*args, **kwargs):
    """Create low latency WebSocket manager (lazy loaded)"""
    from .effects.websocket_enhanced import (
        create_low_latency_websocket_manager as _create,
    )

    return _create(*args, **kwargs)


# Enhanced adapters - lazy loading
def get_enhanced_functional_market_data_adapter():
    """Get EnhancedFunctionalMarketDataAdapter class (lazy loaded)"""
    from .adapters.enhanced_market_data_adapter import (
        EnhancedFunctionalMarketDataAdapter,
    )

    return EnhancedFunctionalMarketDataAdapter


def create_enhanced_coinbase_adapter(*args, **kwargs):
    """Create enhanced Coinbase adapter (lazy loaded)"""
    from .adapters.enhanced_market_data_adapter import (
        create_enhanced_coinbase_adapter as _create,
    )

    return _create(*args, **kwargs)


def create_enhanced_bluefin_adapter(*args, **kwargs):
    """Create enhanced Bluefin adapter (lazy loaded)"""
    from .adapters.enhanced_market_data_adapter import (
        create_enhanced_bluefin_adapter as _create,
    )

    return _create(*args, **kwargs)


def create_enhanced_market_data_adapter(*args, **kwargs):
    """Create enhanced market data adapter (lazy loaded)"""
    from .adapters.enhanced_market_data_adapter import (
        create_enhanced_market_data_adapter as _create,
    )

    return _create(*args, **kwargs)


# Enhanced runtime - lazy loading
def get_enhanced_data_runtime():
    """Get EnhancedDataRuntime class (lazy loaded)"""
    from .runtime.enhanced_data_runtime import EnhancedDataRuntime

    return EnhancedDataRuntime


def create_and_initialize_runtime(*args, **kwargs):
    """Create and initialize runtime (lazy loaded)"""
    from .runtime.enhanced_data_runtime import create_and_initialize_runtime as _create

    return _create(*args, **kwargs)


def create_high_performance_runtime(*args, **kwargs):
    """Create high performance runtime (lazy loaded)"""
    from .runtime.enhanced_data_runtime import (
        create_high_performance_runtime as _create,
    )

    return _create(*args, **kwargs)


def create_low_latency_runtime(*args, **kwargs):
    """Create low latency runtime (lazy loaded)"""
    from .runtime.enhanced_data_runtime import create_low_latency_runtime as _create

    return _create(*args, **kwargs)


# Backward compatibility - lazy loading
def get_functional_market_data_adapter():
    """Get FunctionalMarketDataAdapter class (lazy loaded)"""
    from .adapters.market_data_adapter import FunctionalMarketDataAdapter

    return FunctionalMarketDataAdapter


def create_coinbase_adapter(*args, **kwargs):
    """Create Coinbase adapter (lazy loaded)"""
    from .adapters.market_data_adapter import create_coinbase_adapter as _create

    return _create(*args, **kwargs)


def create_bluefin_adapter(*args, **kwargs):
    """Create Bluefin adapter (lazy loaded)"""
    from .adapters.market_data_adapter import create_bluefin_adapter as _create

    return _create(*args, **kwargs)


def create_market_data_adapter(*args, **kwargs):
    """Create market data adapter (lazy loaded)"""
    from .adapters.market_data_adapter import create_market_data_adapter as _create

    return _create(*args, **kwargs)


__version__ = "2.0.0"

__all__ = [
    # Core types
    "Result",
    "Ok",
    "Err",
    "MarketSnapshot",
    "OHLCV",
    "OrderBook",
    "Trade",
    "Candle",
    "IO",
    "AsyncIO",
    "IOEither",
    # Standardized types (re-exported for FP usage)
    "Price",
    "Quantity",
    "Symbol",
    "OrderId",
    "MarketDataStatus",
    "ConnectionState",
    "ValidationResult",
    "is_valid_price",
    "is_valid_quantity",
    "ensure_decimal",
    # Enhanced data processing (lazy loaded)
    "get_functional_data_pipeline",
    "create_market_data_pipeline",
    "create_high_performance_pipeline",
    "create_low_latency_pipeline",
    # Real-time aggregation (lazy loaded)
    "get_real_time_aggregator",
    "create_real_time_aggregator",
    "create_high_frequency_aggregator",
    "create_batch_aggregator",
    # Enhanced WebSocket (lazy loaded)
    "get_enhanced_websocket_manager",
    "create_enhanced_websocket_manager",
    "create_high_reliability_websocket_manager",
    "create_low_latency_websocket_manager",
    # Enhanced adapters (lazy loaded)
    "get_enhanced_functional_market_data_adapter",
    "create_enhanced_coinbase_adapter",
    "create_enhanced_bluefin_adapter",
    "create_enhanced_market_data_adapter",
    # Enhanced runtime (lazy loaded)
    "get_enhanced_data_runtime",
    "create_and_initialize_runtime",
    "create_high_performance_runtime",
    "create_low_latency_runtime",
    # Backward compatibility (lazy loaded)
    "get_functional_market_data_adapter",
    "create_coinbase_adapter",
    "create_bluefin_adapter",
    "create_market_data_adapter",
]


def get_enhanced_capabilities() -> dict[str, bool]:
    """Get availability of enhanced functional capabilities"""
    return {
        "functional_data_pipeline": True,
        "real_time_aggregation": True,
        "enhanced_websocket_management": True,
        "performance_optimization": True,
        "automatic_fallback": True,
        "multi_exchange_support": True,
        "streaming_data_processing": True,
        "functional_error_handling": True,
        "memory_optimization": True,
        "health_monitoring": True,
    }


def get_performance_modes() -> list[str]:
    """Get available performance optimization modes"""
    return ["low_latency", "high_throughput", "balanced"]


def get_supported_exchanges() -> list[str]:
    """Get list of exchanges supported by enhanced adapters"""
    return ["coinbase", "bluefin"]
