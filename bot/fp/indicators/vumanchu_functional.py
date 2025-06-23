"""
Functional VuManchu Cipher indicator implementation.

This module provides pure functional implementations of the VuManchu Cipher indicators,
decomposed into smaller, composable functions without any class structures.
"""

from datetime import datetime

import numpy as np

from bot.fp.types.indicators import VuManchuResult


def calculate_hlc3(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Calculate HLC3 (High-Low-Close average) values.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices

    Returns:
        Array of HLC3 values
    """
    return (high + low + close) / 3.0


def calculate_ema(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.

    Args:
        values: Input values
        period: EMA period

    Returns:
        Array of EMA values
    """
    if len(values) < period:
        return np.full_like(values, np.nan)

    alpha = 2.0 / (period + 1)
    ema = np.full_like(values, np.nan, dtype=np.float64)

    # Initialize with SMA for the first period
    ema[period - 1] = np.mean(values[:period])

    # Calculate EMA for remaining values
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_sma(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Args:
        values: Input values
        period: SMA period

    Returns:
        Array of SMA values
    """
    if len(values) < period:
        return np.full_like(values, np.nan)

    sma = np.full_like(values, np.nan, dtype=np.float64)

    # Use numpy's convolve for efficient SMA calculation
    kernel = np.ones(period) / period
    sma[period - 1 :] = np.convolve(values, kernel, mode="valid")

    return sma


def calculate_wavetrend_oscillator(
    src: np.ndarray, channel_length: int, average_length: int, ma_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate WaveTrend oscillator (core VuManchu component).

    Pine Script formula:
    _esa = ema(_src, _chlen)
    _de = ema(abs(_src - _esa), _chlen)
    _ci = (_src - _esa) / (0.015 * _de)
    _tci = ema(_ci, _avg)
    _wt1 = _tci
    _wt2 = sma(_wt1, _malen)

    Args:
        src: Source prices (typically HLC3)
        channel_length: Channel length for ESA and DE calculations
        average_length: Average length for TCI calculation
        ma_length: Moving average length for wt2 calculation

    Returns:
        Tuple of (wt1, wt2) arrays
    """
    # Validate inputs
    if len(src) < max(channel_length, average_length, ma_length):
        return np.full_like(src, np.nan), np.full_like(src, np.nan)

    # Calculate ESA (Exponential Smoothing Average)
    esa = calculate_ema(src, channel_length)

    # Calculate DE (Deviation)
    deviation = np.abs(src - esa)
    de = calculate_ema(deviation, channel_length)

    # Prevent division by zero
    de = np.where(de == 0, 1e-6, de)

    # Calculate CI (Commodity Channel Index style)
    ci = (src - esa) / (0.015 * de)

    # Clip extreme values for stability
    ci = np.clip(ci, -100, 100)

    # Calculate TCI (True Commodity Index)
    tci = calculate_ema(ci, average_length)

    # WaveTrend 1 = TCI
    wt1 = tci

    # WaveTrend 2 = SMA of WaveTrend 1
    wt2 = calculate_sma(wt1, ma_length)

    return wt1, wt2


def detect_crossovers(
    wave_a: np.ndarray, wave_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect crossover points between two waves.

    Args:
        wave_a: First wave values
        wave_b: Second wave values

    Returns:
        Tuple of (bullish_crossovers, bearish_crossovers) boolean arrays
    """
    # Calculate differences
    diff = wave_a - wave_b

    # Detect sign changes
    sign_changes = np.diff(np.sign(diff))

    # Bullish crossover: wave_a crosses above wave_b (negative to positive)
    bullish = np.zeros(len(wave_a), dtype=bool)
    bullish[1:] = sign_changes > 0

    # Bearish crossover: wave_a crosses below wave_b (positive to negative)
    bearish = np.zeros(len(wave_a), dtype=bool)
    bearish[1:] = sign_changes < 0

    return bullish, bearish


def determine_signal(
    wt1: float, wt2: float, overbought: float = 45.0, oversold: float = -45.0
) -> str:
    """
    Determine trading signal from current wave values.

    Args:
        wt1: Current WaveTrend 1 value
        wt2: Current WaveTrend 2 value
        overbought: Overbought threshold
        oversold: Oversold threshold

    Returns:
        Signal string: "LONG", "SHORT", or "NEUTRAL"
    """
    # Check for crossovers in oversold/overbought zones
    if wt1 > wt2 and wt1 < oversold:
        return "LONG"
    if wt1 < wt2 and wt1 > overbought:
        return "SHORT"
    return "NEUTRAL"


def vumanchu_cipher(
    ohlcv: np.ndarray,
    period: int = 9,
    mult: float = 0.3,
    timestamp: datetime | None = None,
    channel_length: int | None = None,
    average_length: int | None = None,
    ma_length: int | None = None,
    overbought: float = 45.0,
    oversold: float = -45.0,
) -> VuManchuResult:
    """
    Calculate VuManchu Cipher waves using functional approach.

    Args:
        ohlcv: OHLCV data array with columns [open, high, low, close, volume]
        period: Period for calculations (affects default parameters)
        mult: Multiplier for calculations
        timestamp: Timestamp for the result
        channel_length: Override channel length (default: period)
        average_length: Override average length (default: period * 2)
        ma_length: Override MA length (default: 3)
        overbought: Overbought threshold
        oversold: Oversold threshold

    Returns:
        VuManchuResult with wave values and signal
    """
    # Set default parameters based on period if not provided
    ch_len = channel_length or period
    avg_len = average_length or (period * 2)
    ma_len = ma_length or 3

    # Use current time if timestamp not provided
    if timestamp is None:
        timestamp = datetime.now()

    # Validate input shape
    if ohlcv.ndim != 2 or ohlcv.shape[1] < 4:
        raise ValueError(
            "OHLCV array must have at least 4 columns: open, high, low, close"
        )

    # Extract price columns
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]

    # Calculate HLC3 as source
    hlc3 = calculate_hlc3(high, low, close)

    # Calculate WaveTrend oscillator
    wt1, wt2 = calculate_wavetrend_oscillator(hlc3, ch_len, avg_len, ma_len)

    # Get the latest values
    if len(wt1) > 0 and not np.isnan(wt1[-1]) and not np.isnan(wt2[-1]):
        wave_a = float(wt1[-1])
        wave_b = float(wt2[-1])
        signal = determine_signal(wave_a, wave_b, overbought, oversold)
    else:
        wave_a = 0.0
        wave_b = 0.0
        signal = "NEUTRAL"

    return VuManchuResult(
        timestamp=timestamp, wave_a=wave_a, wave_b=wave_b, signal=signal
    )


def vumanchu_cipher_series(
    ohlcv: np.ndarray,
    period: int = 9,
    mult: float = 0.3,
    timestamps: list[datetime] | None = None,
    channel_length: int | None = None,
    average_length: int | None = None,
    ma_length: int | None = None,
    overbought: float = 45.0,
    oversold: float = -45.0,
) -> list[VuManchuResult]:
    """
    Calculate VuManchu Cipher for entire time series.

    Args:
        ohlcv: OHLCV data array with columns [open, high, low, close, volume]
        period: Period for calculations
        mult: Multiplier for calculations
        timestamps: List of timestamps for each data point
        channel_length: Override channel length
        average_length: Override average length
        ma_length: Override MA length
        overbought: Overbought threshold
        oversold: Oversold threshold

    Returns:
        List of VuManchuResult for each data point
    """
    # Set default parameters
    ch_len = channel_length or period
    avg_len = average_length or (period * 2)
    ma_len = ma_length or 3

    # Generate timestamps if not provided
    if timestamps is None:
        timestamps = [datetime.now()] * len(ohlcv)

    # Extract price columns
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]

    # Calculate HLC3 as source
    hlc3 = calculate_hlc3(high, low, close)

    # Calculate WaveTrend oscillator for entire series
    wt1, wt2 = calculate_wavetrend_oscillator(hlc3, ch_len, avg_len, ma_len)

    # Build results for each point
    results = []
    for i in range(len(ohlcv)):
        if not np.isnan(wt1[i]) and not np.isnan(wt2[i]):
            wave_a = float(wt1[i])
            wave_b = float(wt2[i])
            signal = determine_signal(wave_a, wave_b, overbought, oversold)
        else:
            wave_a = 0.0
            wave_b = 0.0
            signal = "NEUTRAL"

        results.append(
            VuManchuResult(
                timestamp=timestamps[i], wave_a=wave_a, wave_b=wave_b, signal=signal
            )
        )

    return results


# Helper functions for common operations


def is_overbought(wt1: float, threshold: float = 45.0) -> bool:
    """Check if WaveTrend indicates overbought conditions."""
    return wt1 > threshold


def is_oversold(wt1: float, threshold: float = -45.0) -> bool:
    """Check if WaveTrend indicates oversold conditions."""
    return wt1 < threshold


def calculate_divergence(
    prices: np.ndarray, wt_values: np.ndarray, lookback: int = 10
) -> tuple[bool, bool]:
    """
    Detect bullish and bearish divergences.

    Args:
        prices: Price array
        wt_values: WaveTrend values
        lookback: Number of periods to look back

    Returns:
        Tuple of (bullish_divergence, bearish_divergence)
    """
    if len(prices) < lookback or len(wt_values) < lookback:
        return False, False

    # Get recent values
    recent_prices = prices[-lookback:]
    recent_wt = wt_values[-lookback:]

    # Find peaks and troughs
    price_trend = recent_prices[-1] > recent_prices[0]
    wt_trend = recent_wt[-1] > recent_wt[0]

    # Bullish divergence: price makes lower low, but indicator makes higher low
    bullish_div = not price_trend and wt_trend and recent_wt[-1] < -20

    # Bearish divergence: price makes higher high, but indicator makes lower high
    bearish_div = price_trend and not wt_trend and recent_wt[-1] > 20

    return bullish_div, bearish_div


def create_default_vumanchu_config() -> dict:
    """Create default VuManchu configuration dictionary."""
    return {
        "channel_length": 6,  # Optimized for scalping
        "average_length": 8,  # Optimized for scalping
        "ma_length": 3,
        "overbought": 45.0,
        "oversold": -45.0,
        "period": 9,
        "mult": 0.3,
    }
