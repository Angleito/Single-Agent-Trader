"""
Functional implementation of VuManchu Cipher indicators.

Pure functions for calculating VuManchu Cipher A and B.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from bot.fp.indicators.oscillators import calculate_rsi, calculate_stochastic
from bot.fp.types.indicators import VuManchuResult

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculate_vumanchu(
    prices: Sequence[float], period: int = 9, mult: float = 0.3
) -> tuple[float | None, float | None]:
    """
    Calculate VuManchu Cipher waves (simplified version).

    Args:
        prices: Sequence of close prices
        period: Base period for calculations
        mult: Multiplier for wave calculations

    Returns:
        Tuple of (wave_a, wave_b) or (None, None)
    """
    if len(prices) < period * 2:
        return None, None

    # Calculate RSI
    rsi = calculate_rsi(prices, period)
    if rsi is None:
        return None, None

    # Wave A: Fast oscillator based on RSI
    wave_a = (rsi - 50) * mult

    # Wave B: Slower oscillator (simplified - would use WaveTrend in full implementation)
    # For now, using a smoothed version of Wave A
    wave_b = wave_a * 0.7  # Simplified

    return wave_a, wave_b


def calculate_vumanchu_cipher_a(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    hlc3_period: int = 9,
    stoch_k_period: int = 116,
    _stoch_d_period: int = 3,
    _k_smooth: int = 3,
) -> tuple[float | None, float | None, float | None]:
    """
    Calculate VuManchu Cipher A components.

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        closes: Sequence of close prices
        hlc3_period: Period for HLC3 average
        stoch_k_period: Stochastic K period
        stoch_d_period: Stochastic D period
        k_smooth: K smoothing period

    Returns:
        Tuple of (momentum, wave_trend, mf_rsi) or (None, None, None)
    """
    if (
        len(highs) < stoch_k_period
        or len(lows) < stoch_k_period
        or len(closes) < stoch_k_period
    ):
        return None, None, None

    # Calculate HLC3 (High + Low + Close) / 3
    hlc3 = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes, strict=False)]

    # Calculate momentum oscillator (simplified)
    if len(hlc3) < hlc3_period:
        return None, None, None

    momentum = sum(hlc3[-hlc3_period:]) / hlc3_period

    # Calculate WaveTrend (simplified using RSI)
    wave_trend_rsi = calculate_rsi(hlc3, 10)
    if wave_trend_rsi is None:
        return None, None, None

    wave_trend = (wave_trend_rsi - 50) * 2

    # Calculate Money Flow RSI (simplified)
    # In full implementation, this would use volume data
    mf_rsi = calculate_rsi(closes, 60)
    if mf_rsi is None:
        return None, None, None

    return momentum, wave_trend, mf_rsi


def calculate_vumanchu_cipher_b(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    wt_period: int = 10,
    _wt_ma_period: int = 21,  # Used in wave trend calculation
    rsi_period: int = 14,
    stoch_period: int = 14,
) -> dict[str, float | None]:
    """
    Calculate VuManchu Cipher B components.

    Args:
        highs: Sequence of high prices
        lows: Sequence of low prices
        closes: Sequence of close prices
        wt_period: WaveTrend period
        wt_ma_period: WaveTrend MA period
        rsi_period: RSI period
        stoch_period: Stochastic period

    Returns:
        Dictionary with cipher B components
    """
    components = {
        "wavetrend_1": None,
        "wavetrend_2": None,
        "rsi": None,
        "stoch_rsi": None,
        "divergence": None,
    }

    # Calculate RSI
    rsi = calculate_rsi(closes, rsi_period)
    if rsi is not None:
        components["rsi"] = rsi

    # Calculate Stochastic
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes, stoch_period)
    if stoch_k is not None:
        # Stochastic RSI (simplified)
        components["stoch_rsi"] = (stoch_k + rsi) / 2 if rsi is not None else stoch_k

    # Calculate WaveTrend (simplified)
    if len(closes) >= wt_period:
        # HLC3
        hlc3 = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]

        # ESA (Exponentially Smoothed Average)
        esa = calculate_ema_simple(hlc3, wt_period)

        if esa is not None and len(hlc3) >= wt_period:
            # CI (Choppiness Index - simplified)
            ci_values = [abs(hlc3[i] - esa) for i in range(len(hlc3))]
            ci = calculate_ema_simple(ci_values, wt_period)

            if ci is not None and ci != 0:
                # TCI (True Choppiness Index)
                tci = (hlc3[-1] - esa) / (0.015 * ci)
                components["wavetrend_1"] = tci

                # WaveTrend 2 (smoothed)
                components["wavetrend_2"] = tci * 0.8  # Simplified

    # Calculate divergence indicator (simplified)
    if components["wavetrend_1"] is not None and components["rsi"] is not None:
        # Simple divergence: difference between normalized indicators
        wt_norm = components["wavetrend_1"] / 100
        rsi_norm = (components["rsi"] - 50) / 50
        components["divergence"] = abs(wt_norm - rsi_norm) * 100

    return components


def calculate_ema_simple(values: Sequence[float], period: int) -> float | None:
    """
    Simple EMA calculation for internal use.

    Args:
        values: Sequence of values
        period: EMA period

    Returns:
        EMA value or None
    """
    if len(values) < period or period <= 0:
        return None

    # Initial SMA
    sma = sum(values[:period]) / period
    multiplier = 2 / (period + 1)

    ema = sma
    for value in values[period:]:
        ema = (value - ema) * multiplier + ema

    return ema


def create_vumanchu_result(
    wave_a: float | None, wave_b: float | None, timestamp: datetime | None = None
) -> VuManchuResult | None:
    """
    Create a VuManchuResult from calculated values.

    Args:
        wave_a: Wave A value
        wave_b: Wave B value
        timestamp: Optional timestamp (defaults to now)

    Returns:
        VuManchuResult or None if any value is None
    """
    if wave_a is None or wave_b is None:
        return None

    return VuManchuResult(
        timestamp=timestamp or datetime.now(), wave_a=wave_a, wave_b=wave_b
    )


def interpret_vumanchu_signal(
    wave_a: float,
    wave_b: float,
    prev_wave_a: float | None = None,
    prev_wave_b: float | None = None,
) -> str:
    """
    Interpret VuManchu waves to generate trading signal.

    Args:
        wave_a: Current Wave A value
        wave_b: Current Wave B value
        prev_wave_a: Previous Wave A value
        prev_wave_b: Previous Wave B value

    Returns:
        Signal: "LONG", "SHORT", or "NEUTRAL"
    """
    # Check for crossovers if we have previous values
    if prev_wave_a is not None and prev_wave_b is not None:
        # Bullish crossover: Wave A crosses above Wave B in oversold zone
        if prev_wave_a <= prev_wave_b and wave_a > wave_b and wave_a < 0:
            return "LONG"

        # Bearish crossover: Wave A crosses below Wave B in overbought zone
        if prev_wave_a >= prev_wave_b and wave_a < wave_b and wave_a > 0:
            return "SHORT"

    # Check extreme conditions
    if wave_a < -30 and wave_b < -30:  # Extreme oversold
        return "LONG"
    if wave_a > 30 and wave_b > 30:  # Extreme overbought
        return "SHORT"

    return "NEUTRAL"
