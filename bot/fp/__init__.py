"""
Functional programming utilities for the trading bot.

This package provides pure functional programming utilities and
immutable data structures for building reliable trading systems.
"""

# Core modules will be imported individually due to dependency issues
# from bot.fp.core import validation

# Indicator functions
from bot.fp.indicators import (
    calculate_all_momentum_indicators,
    macd,
    rate_of_change,
    rsi,
    stochastic,
)

# Type modules
from bot.fp.types import (
    config,
    effects,
    events,
    indicators,
    market,
    portfolio,
    risk,
    trading,
)

__all__ = [
    # Type modules
    "config",
    "effects",
    "events",
    "indicators",
    "market",
    "portfolio",
    "risk",
    "trading",
    # Indicator functions
    "rsi",
    "macd",
    "stochastic",
    "rate_of_change",
    "calculate_all_momentum_indicators",
]
