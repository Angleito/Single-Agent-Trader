"""
Functional implementations of volatility indicators.

Pure functions for calculating Bollinger Bands, ATR, and other volatility measures.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from bot.fp.indicators.moving_averages import sma
from bot.fp.types.indicators import BollingerBandsResult

if TYPE_CHECKING:
    from collections.abc import Sequence


def _calculate_sma_simple(prices: Sequence[float], period: int) -> float | None:
    """Simple SMA calculation for internal use."""
    if len(prices) < period or period <= 0:
        return None

    prices_array = np.array(prices, dtype=np.float64)
    result = sma(prices_array, period, datetime.now())
    return result.value if result else None


def calculate_bollinger_bands(
    prices: Sequence[float], period: int = 20, std_dev: float = 2.0
) -> tuple[float | None, float | None, float | None]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Sequence of price values
        period: SMA period
        std_dev: Number of standard deviations

    Returns:
        Tuple of (upper_band, middle_band, lower_band) or (None, None, None)
    """
    if len(prices) < period or period <= 0:
        return None, None, None

    # Calculate middle band (SMA)
    middle = _calculate_sma_simple(prices, period)
    if middle is None:
        return None, None, None

    # Calculate standard deviation
    recent_prices = prices[-period:]
    variance = sum((price - middle) ** 2 for price in recent_prices) / period
    std = variance**0.5

    # Calculate bands
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return upper, middle, lower


def calculate_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """
    Calculate Average True Range.

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        closes: Sequence of close prices
        period: ATR period

    Returns:
        ATR value or None
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None

    if period <= 0:
        return None

    # Calculate True Range values
    tr_values = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close_prev = abs(highs[i] - closes[i - 1])
        low_close_prev = abs(lows[i] - closes[i - 1])

        tr = max(high_low, high_close_prev, low_close_prev)
        tr_values.append(tr)

    # Calculate ATR
    if len(tr_values) < period:
        return None

    # Initial ATR is simple average
    atr = sum(tr_values[:period]) / period

    # Apply smoothing for remaining values
    for i in range(period, len(tr_values)):
        atr = (atr * (period - 1) + tr_values[i]) / period

    return atr


def calculate_keltner_channels(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[float | None, float | None, float | None]:
    """
    Calculate Keltner Channels.

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        closes: Sequence of close prices
        ema_period: EMA period for middle line
        atr_period: ATR period
        multiplier: ATR multiplier for bands

    Returns:
        Tuple of (upper, middle, lower) or (None, None, None)
    """
    from bot.fp.indicators.moving_averages import calculate_ema

    if len(closes) < ema_period:
        return None, None, None

    # Calculate middle line (EMA)
    middle = calculate_ema(closes, ema_period)
    if middle is None:
        return None, None, None

    # Calculate ATR
    atr = calculate_atr(highs, lows, closes, atr_period)
    if atr is None:
        return None, None, None

    # Calculate channels
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)

    return upper, middle, lower


def calculate_donchian_channels(
    highs: Sequence[float], lows: Sequence[float], period: int = 20
) -> tuple[float | None, float | None, float | None]:
    """
    Calculate Donchian Channels.

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        period: Lookback period

    Returns:
        Tuple of (upper, middle, lower) or (None, None, None)
    """
    if len(highs) < period or len(lows) < period:
        return None, None, None

    if period <= 0:
        return None, None, None

    # Calculate channels
    upper = max(highs[-period:])
    lower = min(lows[-period:])
    middle = (upper + lower) / 2

    return upper, middle, lower


def calculate_standard_deviation(
    prices: Sequence[float], period: int = 20
) -> float | None:
    """
    Calculate standard deviation of prices.

    Args:
        prices: Sequence of price values
        period: Period for calculation

    Returns:
        Standard deviation or None
    """
    if len(prices) < period or period <= 0:
        return None

    # Calculate mean
    recent_prices = prices[-period:]
    mean = sum(recent_prices) / period

    # Calculate standard deviation
    variance = sum((price - mean) ** 2 for price in recent_prices) / period
    return variance**0.5


def calculate_historical_volatility(
    prices: Sequence[float], period: int = 20, annualization_factor: int = 252
) -> float | None:
    """
    Calculate historical volatility (annualized).

    Args:
        prices: Sequence of price values
        period: Period for calculation
        annualization_factor: Trading days per year (252 for daily)

    Returns:
        Annualized volatility percentage or None
    """
    if len(prices) < period + 1 or period <= 0:
        return None

    # Calculate returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)

    if len(returns) < period:
        return None

    # Calculate standard deviation of returns
    recent_returns = returns[-period:]
    mean = sum(recent_returns) / period
    variance = sum((ret - mean) ** 2 for ret in recent_returns) / period
    std_dev = variance**0.5

    # Annualize
    return std_dev * (annualization_factor**0.5) * 100


def create_bollinger_bands_result(
    upper: float | None,
    middle: float | None,
    lower: float | None,
    timestamp: datetime | None = None,
) -> BollingerBandsResult | None:
    """
    Create a BollingerBandsResult from calculated values.

    Args:
        upper: Upper band value
        middle: Middle band value
        lower: Lower band value
        timestamp: Optional timestamp (defaults to now)

    Returns:
        BollingerBandsResult or None if any value is None
    """
    if upper is None or middle is None or lower is None:
        return None

    return BollingerBandsResult(
        timestamp=timestamp or datetime.now(), upper=upper, middle=middle, lower=lower
    )
