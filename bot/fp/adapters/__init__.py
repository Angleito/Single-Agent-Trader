"""
Functional Programming Adapters

This module provides adapters for integrating imperative components
with the functional programming system.
"""

# Exchange adapters
from .bluefin_adapter import BluefinExchangeAdapter
from .coinbase_adapter import CoinbaseExchangeAdapter
from .enhanced_market_data_adapter import EnhancedFunctionalMarketDataAdapter

# Data adapters
from .market_data_adapter import FunctionalMarketDataAdapter

# Monitoring adapters
from .performance_monitor_adapter import FunctionalPerformanceMonitor

# Strategy adapters
from .strategy_adapter import (
    FunctionalLLMStrategy,
    LLMAgentAdapter,
    MemoryEnhancedLLMAgentAdapter,
)
from .system_monitor_adapter import FunctionalSystemMonitor

# Type converters
try:
    from .type_converters import (
        convert_market_data,
        convert_position,
        convert_trade_action,
    )
except ImportError:
    # Type converters may not be available in all configurations
    pass

__all__ = [
    # Exchange adapters
    "BluefinExchangeAdapter",
    "CoinbaseExchangeAdapter",
    "EnhancedFunctionalMarketDataAdapter",
    # Strategy adapters
    "FunctionalLLMStrategy",
    # Data adapters
    "FunctionalMarketDataAdapter",
    # Monitoring adapters
    "FunctionalPerformanceMonitor",
    "FunctionalSystemMonitor",
    "LLMAgentAdapter",
    "MemoryEnhancedLLMAgentAdapter",
]

# Add type converters if available
try:
    from .type_converters import (
        convert_market_data,
        convert_position,
        convert_trade_action,
    )

    __all__.extend(
        [
            "convert_market_data",
            "convert_position",
            "convert_trade_action",
        ]
    )
except ImportError:
    pass
