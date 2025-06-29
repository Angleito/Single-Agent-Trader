"""Technical indicators and analysis modules."""

from typing import Any, Union

import numpy as np

from .ema_ribbon import EMAribbon
from .rsimfi import RSIMFIIndicator
from .schaff_trend_cycle import SchaffTrendCycle
from .stochastic_rsi import StochasticRSI
from .vumanchu import CipherA, CipherB, VuManChuIndicators, VuManchuState

__all__ = [
    "CipherA",
    "CipherB",
    "EMAribbon",
    "RSIMFIIndicator",
    "SchaffTrendCycle",
    "StochasticRSI",
    "VuManChuIndicators",
    "VuManchuState",
]


# Utility functions for strategy manager
def calculate_atr(
    highs: list[float] | np.ndarray,
    lows: list[float] | np.ndarray,
    closes: list[float] | np.ndarray,
    period: int = 14,
) -> float:
    """Calculate Average True Range for volatility measurement."""
    if len(highs) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close_prev = abs(highs[i] - closes[i - 1])
        low_close_prev = abs(lows[i] - closes[i - 1])

        true_range = max(high_low, high_close_prev, low_close_prev)
        true_ranges.append(true_range)

    if len(true_ranges) >= period:
        return np.mean(true_ranges[-period:])
    return np.mean(true_ranges) if true_ranges else 0.0


def calculate_support_resistance(
    highs: list[float] | np.ndarray, lows: list[float] | np.ndarray, window: int = 20
) -> dict[str, float]:
    """Calculate basic support and resistance levels."""
    if len(highs) < window or len(lows) < window:
        return {"support": 0.0, "resistance": 0.0}

    recent_highs = highs[-window:]
    recent_lows = lows[-window:]

    resistance = float(np.max(recent_highs))
    support = float(np.min(recent_lows))

    return {"support": support, "resistance": resistance}
