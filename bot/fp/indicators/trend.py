"""
Functional implementations of trend indicators.

Pure functions for calculating MACD, ADX, and other trend indicators.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from bot.fp.indicators.moving_averages import calculate_ema
from bot.fp.types.indicators import MACDResult


def calculate_macd(
    prices: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[float | None, float | None, float | None]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Sequence of price values
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period

    Returns:
        Tuple of (macd_line, signal_line, histogram) or (None, None, None)
    """
    if len(prices) < slow_period + signal_period:
        return None, None, None

    # Calculate MACD line
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)

    if fast_ema is None or slow_ema is None:
        return None, None, None

    macd_line = fast_ema - slow_ema

    # For signal line, we need historical MACD values
    # Calculate MACD for each point to build history
    macd_history = []
    for i in range(slow_period, len(prices) + 1):
        fast = calculate_ema(prices[:i], fast_period)
        slow = calculate_ema(prices[:i], slow_period)
        if fast is not None and slow is not None:
            macd_history.append(fast - slow)

    # Calculate signal line (EMA of MACD)
    if len(macd_history) < signal_period:
        return macd_line, None, None

    signal_line = calculate_ema(macd_history, signal_period)
    if signal_line is None:
        return macd_line, None, None

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """
    Calculate Average Directional Index (ADX).

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        closes: Sequence of close prices
        period: ADX period

    Returns:
        ADX value (0-100) or None
    """
    if len(highs) < period * 2 or len(lows) < period * 2 or len(closes) < period * 2:
        return None

    if period <= 0:
        return None

    # Calculate True Range and Directional Movement
    tr_values = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(highs)):
        # True Range
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        tr = max(high_low, high_close, low_close)
        tr_values.append(tr)

        # Directional Movement
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
            minus_dm.append(0)
        elif down_move > up_move and down_move > 0:
            plus_dm.append(0)
            minus_dm.append(down_move)
        else:
            plus_dm.append(0)
            minus_dm.append(0)

    if len(tr_values) < period:
        return None

    # Calculate smoothed averages
    atr = sum(tr_values[:period]) / period
    plus_di_sum = sum(plus_dm[:period])
    minus_di_sum = sum(minus_dm[:period])

    # Smooth the values
    dx_values = []
    for i in range(period, len(tr_values)):
        atr = (atr * (period - 1) + tr_values[i]) / period
        plus_di_sum = (plus_di_sum * (period - 1) + plus_dm[i]) / period
        minus_di_sum = (minus_di_sum * (period - 1) + minus_dm[i]) / period

        # Calculate DI values
        if atr > 0:
            plus_di = (plus_di_sum / atr) * 100
            minus_di = (minus_di_sum / atr) * 100

            # Calculate DX
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = abs(plus_di - minus_di) / di_sum * 100
                dx_values.append(dx)

    # Calculate ADX (average of DX)
    if len(dx_values) < period:
        return None

    adx = sum(dx_values[-period:]) / period
    return adx


def calculate_psar(
    highs: Sequence[float],
    lows: Sequence[float],
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
) -> float | None:
    """
    Calculate Parabolic SAR.

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        af_start: Starting acceleration factor
        af_increment: AF increment
        af_max: Maximum AF value

    Returns:
        Current PSAR value or None
    """
    if len(highs) < 2 or len(lows) < 2:
        return None

    # Initialize
    is_long = highs[1] > highs[0]
    af = af_start
    ep = highs[1] if is_long else lows[1]
    psar = lows[0] if is_long else highs[0]

    # Calculate PSAR for each period
    for i in range(2, len(highs)):
        # Update PSAR
        psar = psar + af * (ep - psar)

        # Check for reversal
        if is_long:
            if lows[i] <= psar:
                is_long = False
                psar = ep
                ep = lows[i]
                af = af_start
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_increment, af_max)
                # Make sure PSAR is below the low
                psar = min(psar, lows[i - 1], lows[i])
        elif highs[i] >= psar:
            is_long = True
            psar = ep
            ep = highs[i]
            af = af_start
        else:
            if lows[i] < ep:
                ep = lows[i]
                af = min(af + af_increment, af_max)
            # Make sure PSAR is above the high
            psar = max(psar, highs[i - 1], highs[i])

    return psar


def calculate_aroon(
    highs: Sequence[float], lows: Sequence[float], period: int = 25
) -> tuple[float | None, float | None]:
    """
    Calculate Aroon Indicator (Up and Down).

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        period: Aroon period

    Returns:
        Tuple of (aroon_up, aroon_down) or (None, None)
    """
    if len(highs) < period or len(lows) < period:
        return None, None

    if period <= 0:
        return None, None

    # Find periods since highest high and lowest low
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]

    highest_idx = recent_highs.index(max(recent_highs))
    lowest_idx = recent_lows.index(min(recent_lows))

    # Calculate Aroon values
    # Aroon Up = ((period - periods since highest) / period) * 100
    # Aroon Down = ((period - periods since lowest) / period) * 100
    periods_since_high = period - 1 - highest_idx
    periods_since_low = period - 1 - lowest_idx

    aroon_up = ((period - periods_since_high) / period) * 100
    aroon_down = ((period - periods_since_low) / period) * 100

    return aroon_up, aroon_down


def create_macd_result(
    macd: float | None,
    signal: float | None,
    histogram: float | None,
    timestamp: datetime | None = None,
) -> MACDResult | None:
    """
    Create a MACDResult from calculated values.

    Args:
        macd: MACD line value
        signal: Signal line value
        histogram: Histogram value
        timestamp: Optional timestamp (defaults to now)

    Returns:
        MACDResult or None if any value is None
    """
    if macd is None or signal is None or histogram is None:
        return None

    return MACDResult(
        timestamp=timestamp or datetime.now(),
        macd=macd,
        signal=signal,
        histogram=histogram,
    )
