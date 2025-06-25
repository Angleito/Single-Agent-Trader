"""
Functional implementations of oscillator indicators.

Pure functions for calculating RSI, Stochastic, and other oscillators.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from bot.fp.types.indicators import RSIResult

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculate_rsi(prices: Sequence[float], period: int = 14) -> float | None:
    """
    Calculate Relative Strength Index.

    Args:
        prices: Sequence of price values
        period: RSI period (default: 14)

    Returns:
        The RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1 or period <= 0:
        return None

    # Calculate price changes
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # Separate gains and losses
    gains = [max(change, 0) for change in changes]
    losses = [abs(min(change, 0)) for change in changes]

    # Calculate initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Apply smoothing for remaining data
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Calculate RSI
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[float | None, float | None]:
    """
    Calculate Stochastic Oscillator (%K and %D).

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        closes: Sequence of close prices
        period: Lookback period
        smooth_k: Smoothing for %K
        smooth_d: Smoothing for %D

    Returns:
        Tuple of (%K, %D) or (None, None) if insufficient data
    """
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return None, None

    if period <= 0:
        return None, None

    # Calculate %K values
    k_values = []
    for i in range(period - 1, len(closes)):
        high_period = max(highs[i - period + 1 : i + 1])
        low_period = min(lows[i - period + 1 : i + 1])

        if high_period != low_period:
            k = ((closes[i] - low_period) / (high_period - low_period)) * 100
        else:
            k = 50.0  # Default to middle when no range

        k_values.append(k)

    if len(k_values) < smooth_k:
        return None, None

    # Smooth %K
    k_smooth = sum(k_values[-smooth_k:]) / smooth_k

    # Calculate %D (moving average of %K)
    if len(k_values) < smooth_d:
        return k_smooth, None

    d_smooth = sum(k_values[-smooth_d:]) / smooth_d

    return k_smooth, d_smooth


def calculate_williams_r(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """
    Calculate Williams %R indicator.

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        closes: Sequence of close prices
        period: Lookback period

    Returns:
        Williams %R value (-100 to 0) or None
    """
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return None

    if period <= 0:
        return None

    # Get highest high and lowest low
    highest = max(highs[-period:])
    lowest = min(lows[-period:])

    # Calculate Williams %R
    if highest != lowest:
        williams_r = ((highest - closes[-1]) / (highest - lowest)) * -100
    else:
        williams_r = -50.0  # Default to middle

    return williams_r


def calculate_cci(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 20,
) -> float | None:
    """
    Calculate Commodity Channel Index.

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        closes: Sequence of close prices
        period: CCI period

    Returns:
        CCI value or None
    """
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return None

    if period <= 0:
        return None

    # Calculate typical prices
    typical_prices = [
        (h + l + c) / 3
        for h, l, c in zip(
            highs[-period:], lows[-period:], closes[-period:], strict=False
        )
    ]

    # Calculate moving average of typical price
    ma = sum(typical_prices) / period

    # Calculate mean deviation
    deviations = [abs(tp - ma) for tp in typical_prices]
    mean_deviation = sum(deviations) / period

    # Calculate CCI
    if mean_deviation != 0:
        cci = (typical_prices[-1] - ma) / (0.015 * mean_deviation)
    else:
        cci = 0.0

    return cci


def calculate_mfi(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    period: int = 14,
) -> float | None:
    """
    Calculate Money Flow Index.

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        closes: Sequence of close prices
        volumes: Sequence of volume values
        period: MFI period

    Returns:
        MFI value (0-100) or None
    """
    if (
        len(highs) < period + 1
        or len(lows) < period + 1
        or len(closes) < period + 1
        or len(volumes) < period + 1
    ):
        return None

    if period <= 0:
        return None

    # Calculate typical prices and money flow
    typical_prices = [
        (h + l + c) / 3 for h, l, c in zip(highs, lows, closes, strict=False)
    ]

    # Calculate raw money flow
    money_flows = [tp * v for tp, v in zip(typical_prices, volumes, strict=False)]

    # Separate positive and negative money flow
    positive_flow = 0.0
    negative_flow = 0.0

    for i in range(len(typical_prices) - period, len(typical_prices)):
        if i > 0:
            if typical_prices[i] > typical_prices[i - 1]:
                positive_flow += money_flows[i]
            elif typical_prices[i] < typical_prices[i - 1]:
                negative_flow += money_flows[i]

    # Calculate MFI
    if negative_flow == 0:
        return 100.0

    money_ratio = positive_flow / negative_flow
    return 100 - (100 / (1 + money_ratio))


def create_rsi_result(
    value: float | None,
    overbought: float = 70.0,
    oversold: float = 30.0,
    timestamp: datetime | None = None,
) -> RSIResult | None:
    """
    Create an RSIResult from calculated value.

    Args:
        value: The calculated RSI value
        overbought: Overbought threshold
        oversold: Oversold threshold
        timestamp: Optional timestamp (defaults to now)

    Returns:
        RSIResult or None if value is None
    """
    if value is None:
        return None

    return RSIResult(
        timestamp=timestamp or datetime.now(),
        value=value,
        overbought=overbought,
        oversold=oversold,
    )
