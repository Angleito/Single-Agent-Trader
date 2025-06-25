"""
Pure functional technical indicators.

This package provides pure functional implementations of various
technical indicators used in trading systems.
"""

from bot.fp.indicators.momentum import (
    calculate_all_momentum_indicators,
    macd,
    rate_of_change,
    rsi,
    stochastic,
)
from bot.fp.indicators.oscillators import calculate_rsi as calculate_rsi_oscillators
from bot.fp.indicators.vumanchu_functional import (
    calculate_ema,
    calculate_hlc3,
    calculate_sma,
)

# For compatibility, alias missing functions
calculate_rsi = calculate_rsi_oscillators


def calculate_volume_sma(volume: list[float], period: int) -> list[float]:
    """Calculate SMA for volume using the existing SMA function."""
    import numpy as np

    return calculate_sma(np.array(volume), period).tolist()


__all__ = [
    "calculate_all_momentum_indicators",
    "calculate_ema",
    "calculate_hlc3",
    "calculate_rsi",
    "calculate_sma",
    "calculate_volume_sma",
    "macd",
    "rate_of_change",
    "rsi",
    "stochastic",
]
