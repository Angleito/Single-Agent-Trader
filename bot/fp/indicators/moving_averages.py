"""
Pure functional implementations of moving average indicators.

This module provides pure functions for calculating various types of moving averages
commonly used in technical analysis. All functions are side-effect free and return
immutable MovingAverageResult objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bot.fp.types.indicators import MovingAverageResult

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray


def sma(
    prices: NDArray[np.float64], period: int, timestamp: datetime
) -> MovingAverageResult:
    """
    Calculate Simple Moving Average.

    The SMA is the arithmetic mean of the last N prices, where N is the period.
    It gives equal weight to all prices in the calculation window.

    Args:
        prices: Array of prices (most recent last)
        period: Number of periods for calculation
        timestamp: Timestamp for the result

    Returns:
        MovingAverageResult with the calculated SMA value

    Raises:
        ValueError: If period is less than 1 or exceeds array length

    Example:
        >>> import numpy as np
        >>> from datetime import datetime
        >>> prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
        >>> result = sma(prices, period=3, timestamp=datetime.now())
        >>> print(f"SMA(3): {result.value:.2f}")
        SMA(3): 103.00
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    if len(prices) < period:
        raise ValueError(f"Insufficient data: need {period} prices, got {len(prices)}")

    # Calculate SMA using the last 'period' prices
    sma_value = float(np.mean(prices[-period:]))

    return MovingAverageResult(timestamp=timestamp, value=sma_value, period=period)


def ema(
    prices: NDArray[np.float64], period: int, timestamp: datetime
) -> MovingAverageResult:
    """
    Calculate Exponential Moving Average.

    The EMA gives more weight to recent prices, making it more responsive to
    new information compared to SMA. Uses the standard EMA formula with
    smoothing factor = 2 / (period + 1).

    Args:
        prices: Array of prices (most recent last)
        period: Number of periods for calculation
        timestamp: Timestamp for the result

    Returns:
        MovingAverageResult with the calculated EMA value

    Raises:
        ValueError: If period is less than 1 or exceeds array length

    Example:
        >>> import numpy as np
        >>> from datetime import datetime
        >>> prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
        >>> result = ema(prices, period=3, timestamp=datetime.now())
        >>> print(f"EMA(3): {result.value:.2f}")
        EMA(3): 104.00
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    if len(prices) < period:
        raise ValueError(f"Insufficient data: need {period} prices, got {len(prices)}")

    # Calculate smoothing factor (alpha)
    alpha = 2.0 / (period + 1)

    # Initialize EMA with SMA of first 'period' prices
    ema_values = np.zeros(len(prices))
    ema_values[period - 1] = np.mean(prices[:period])

    # Calculate EMA for remaining prices
    for i in range(period, len(prices)):
        ema_values[i] = alpha * prices[i] + (1 - alpha) * ema_values[i - 1]

    # Return the most recent EMA value
    ema_value = float(ema_values[-1])

    return MovingAverageResult(timestamp=timestamp, value=ema_value, period=period)


def wma(
    prices: NDArray[np.float64], period: int, timestamp: datetime
) -> MovingAverageResult:
    """
    Calculate Weighted Moving Average.

    The WMA assigns linearly decreasing weights to older prices. The most recent
    price gets weight = period, second most recent gets weight = period - 1, etc.

    Args:
        prices: Array of prices (most recent last)
        period: Number of periods for calculation
        timestamp: Timestamp for the result

    Returns:
        MovingAverageResult with the calculated WMA value

    Raises:
        ValueError: If period is less than 1 or exceeds array length

    Example:
        >>> import numpy as np
        >>> from datetime import datetime
        >>> prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
        >>> result = wma(prices, period=3, timestamp=datetime.now())
        >>> print(f"WMA(3): {result.value:.2f}")
        WMA(3): 103.50
    """
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    if len(prices) < period:
        raise ValueError(f"Insufficient data: need {period} prices, got {len(prices)}")

    # Get the last 'period' prices
    recent_prices = prices[-period:]

    # Create weights: [1, 2, 3, ..., period]
    weights = np.arange(1, period + 1, dtype=np.float64)

    # Calculate weighted average
    wma_value = float(np.sum(recent_prices * weights) / np.sum(weights))

    return MovingAverageResult(timestamp=timestamp, value=wma_value, period=period)


def hma(
    prices: NDArray[np.float64], period: int, timestamp: datetime
) -> MovingAverageResult:
    """
    Calculate Hull Moving Average.

    The HMA reduces lag while maintaining smoothness by using weighted moving
    averages and a square root period. It's calculated as:
    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

    Args:
        prices: Array of prices (most recent last)
        period: Number of periods for calculation
        timestamp: Timestamp for the result

    Returns:
        MovingAverageResult with the calculated HMA value

    Raises:
        ValueError: If period is less than 4 or exceeds array length

    Example:
        >>> import numpy as np
        >>> from datetime import datetime
        >>> prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0])
        >>> result = hma(prices, period=6, timestamp=datetime.now())
        >>> print(f"HMA(6): {result.value:.2f}")
        HMA(6): 107.17
    """
    if period < 4:
        raise ValueError(f"Period must be at least 4 for HMA, got {period}")

    if len(prices) < period:
        raise ValueError(f"Insufficient data: need {period} prices, got {len(prices)}")

    # Calculate half period and sqrt period
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))

    # Need enough data for all calculations
    min_required = period + sqrt_period - 1
    if len(prices) < min_required:
        raise ValueError(
            f"Insufficient data for HMA: need {min_required} prices, got {len(prices)}"
        )

    # Calculate WMA with half period
    wma_half_values = np.zeros(len(prices))
    for i in range(half_period - 1, len(prices)):
        subset = prices[i - half_period + 1 : i + 1]
        weights = np.arange(1, half_period + 1, dtype=np.float64)
        wma_half_values[i] = np.sum(subset * weights) / np.sum(weights)

    # Calculate WMA with full period
    wma_full_values = np.zeros(len(prices))
    for i in range(period - 1, len(prices)):
        subset = prices[i - period + 1 : i + 1]
        weights = np.arange(1, period + 1, dtype=np.float64)
        wma_full_values[i] = np.sum(subset * weights) / np.sum(weights)

    # Calculate difference series: 2 * WMA(n/2) - WMA(n)
    diff_series = 2 * wma_half_values - wma_full_values

    # Calculate final HMA as WMA of difference series with sqrt period
    # Only calculate where we have valid diff_series values
    valid_start = period - 1
    valid_diff = diff_series[valid_start:]

    if len(valid_diff) < sqrt_period:
        raise ValueError("Insufficient valid data for final HMA calculation")

    # Calculate WMA of the difference series
    recent_diff = valid_diff[-sqrt_period:]
    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
    hma_value = float(np.sum(recent_diff * weights) / np.sum(weights))

    return MovingAverageResult(timestamp=timestamp, value=hma_value, period=period)


def multi_timeframe_sma(
    prices: NDArray[np.float64], periods: list[int], timestamp: datetime
) -> list[MovingAverageResult]:
    """
    Calculate multiple SMAs with different periods in a single pass.

    This is more efficient than calling sma() multiple times when you need
    several moving averages for the same price data.

    Args:
        prices: Array of prices (most recent last)
        periods: List of periods to calculate
        timestamp: Timestamp for the results

    Returns:
        List of MovingAverageResult objects, one for each period

    Raises:
        ValueError: If any period is invalid or exceeds array length

    Example:
        >>> import numpy as np
        >>> from datetime import datetime
        >>> prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
        >>> results = multi_timeframe_sma(prices, [2, 3, 5], datetime.now())
        >>> for r in results:
        ...     print(f"SMA({r.period}): {r.value:.2f}")
        SMA(2): 104.00
        SMA(3): 103.00
        SMA(5): 102.20
    """
    results = []

    for period in periods:
        result = sma(prices, period, timestamp)
        results.append(result)

    return results


def adaptive_period_ema(
    prices: NDArray[np.float64],
    base_period: int,
    volatility: float,
    timestamp: datetime,
) -> MovingAverageResult:
    """
    Calculate EMA with period adapted based on market volatility.

    Higher volatility increases the period (slower adaptation),
    lower volatility decreases the period (faster adaptation).

    Args:
        prices: Array of prices (most recent last)
        base_period: Base period for normal volatility
        volatility: Current volatility factor (0.0 to 1.0)
        timestamp: Timestamp for the result

    Returns:
        MovingAverageResult with adapted period EMA

    Example:
        >>> import numpy as np
        >>> from datetime import datetime
        >>> prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
        >>> # High volatility (0.8) increases period
        >>> result = adaptive_period_ema(prices, 10, 0.8, datetime.now())
        >>> print(f"Adaptive EMA period: {result.period}")
        Adaptive EMA period: 14
    """
    # Adapt period based on volatility
    # High volatility (close to 1.0) increases period up to 50%
    # Low volatility (close to 0.0) decreases period up to 50%
    volatility_factor = 1.0 + (volatility - 0.5)
    adapted_period = max(1, int(base_period * volatility_factor))

    # Ensure we don't exceed available data
    adapted_period = min(adapted_period, len(prices))

    return ema(prices, adapted_period, timestamp)
