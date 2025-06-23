"""Pure functional volume indicator implementations.

This module provides volume-based technical indicators using pure numpy operations.
All functions are stateless and return typed results.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class OBVResult(NamedTuple):
    """On-Balance Volume calculation result."""

    obv: NDArray[np.float64]
    signal: NDArray[np.float64]  # SMA of OBV
    divergence: NDArray[np.float64]  # Price vs OBV divergence


class VWAPResult(NamedTuple):
    """Volume Weighted Average Price result."""

    vwap: NDArray[np.float64]
    upper_band: NDArray[np.float64]  # VWAP + std dev
    lower_band: NDArray[np.float64]  # VWAP - std dev
    volume_ratio: NDArray[np.float64]  # Current volume / avg volume


class VolumeProfileResult(NamedTuple):
    """Volume Profile analysis result."""

    price_levels: NDArray[np.float64]
    volume_at_price: NDArray[np.float64]
    poc: float  # Point of Control (price with highest volume)
    value_area_high: float
    value_area_low: float
    volume_nodes: NDArray[np.float64]  # High volume price levels


class ADLineResult(NamedTuple):
    """Accumulation/Distribution Line result."""

    ad_line: NDArray[np.float64]
    signal: NDArray[np.float64]  # EMA of AD Line
    divergence: NDArray[np.float64]  # Price vs AD divergence
    money_flow_multiplier: NDArray[np.float64]


class MFIResult(NamedTuple):
    """Money Flow Index result."""

    mfi: NDArray[np.float64]
    money_flow_ratio: NDArray[np.float64]
    overbought: NDArray[np.bool_]  # MFI > 80
    oversold: NDArray[np.bool_]  # MFI < 20


def calculate_obv(
    close: NDArray[np.float64], volume: NDArray[np.float64], signal_period: int = 20
) -> OBVResult:
    """Calculate On-Balance Volume indicator.

    OBV is a cumulative indicator that adds volume on up days and subtracts
    volume on down days to measure buying and selling pressure.

    Args:
        close: Closing prices
        volume: Trading volumes
        signal_period: Period for signal line (SMA of OBV)

    Returns:
        OBVResult with OBV values, signal line, and divergence
    """
    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have same length")

    if len(close) < 2:
        raise ValueError("Need at least 2 data points")

    # Calculate price changes
    price_change = np.diff(close, prepend=close[0])

    # Calculate OBV: add volume on up days, subtract on down days
    obv = np.cumsum(
        np.where(price_change > 0, volume, np.where(price_change < 0, -volume, 0))
    )

    # Calculate signal line (SMA of OBV)
    signal = _sma(obv, signal_period)

    # Calculate divergence (normalized price change vs normalized OBV change)
    price_norm = (close - np.min(close)) / (np.max(close) - np.min(close) + 1e-10)
    obv_norm = (obv - np.min(obv)) / (np.max(obv) - np.min(obv) + 1e-10)
    divergence = price_norm - obv_norm

    return OBVResult(obv=obv, signal=signal, divergence=divergence)


def calculate_vwap(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    std_multiplier: float = 2.0,
    volume_period: int = 20,
) -> VWAPResult:
    """Calculate Volume Weighted Average Price with bands.

    VWAP is the average price weighted by volume, often used as a
    benchmark for intraday trading.

    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Trading volumes
        std_multiplier: Standard deviation multiplier for bands
        volume_period: Period for average volume calculation

    Returns:
        VWAPResult with VWAP, bands, and volume ratio
    """
    if not (len(high) == len(low) == len(close) == len(volume)):
        raise ValueError("All price/volume arrays must have same length")

    # Calculate typical price (HLC average)
    typical_price = (high + low + close) / 3

    # Calculate cumulative values
    cum_volume = np.cumsum(volume)
    cum_pv = np.cumsum(typical_price * volume)

    # VWAP = cumulative(price * volume) / cumulative(volume)
    vwap = np.divide(
        cum_pv, cum_volume, where=cum_volume != 0, out=np.zeros_like(cum_pv)
    )

    # Calculate standard deviation for bands
    # Using a rolling window approach
    deviations = typical_price - vwap
    squared_deviations = deviations**2
    weighted_squared_deviations = squared_deviations * volume

    # Cumulative weighted variance
    cum_weighted_var = np.cumsum(weighted_squared_deviations)
    variance = np.divide(
        cum_weighted_var,
        cum_volume,
        where=cum_volume != 0,
        out=np.zeros_like(cum_weighted_var),
    )
    std_dev = np.sqrt(variance)

    # Calculate bands
    upper_band = vwap + (std_multiplier * std_dev)
    lower_band = vwap - (std_multiplier * std_dev)

    # Calculate volume ratio (current volume / average volume)
    avg_volume = _sma(volume, volume_period)
    volume_ratio = np.divide(
        volume, avg_volume, where=avg_volume != 0, out=np.ones_like(volume)
    )

    return VWAPResult(
        vwap=vwap,
        upper_band=upper_band,
        lower_band=lower_band,
        volume_ratio=volume_ratio,
    )


def calculate_volume_profile(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    num_bins: int = 30,
    value_area_percent: float = 0.70,
) -> VolumeProfileResult:
    """Calculate Volume Profile for price levels.

    Volume Profile shows the amount of volume traded at different price levels,
    helping identify support/resistance areas.

    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Trading volumes
        num_bins: Number of price bins for profile
        value_area_percent: Percentage of volume for value area (typically 70%)

    Returns:
        VolumeProfileResult with price levels, volumes, and key levels
    """
    if not (len(high) == len(low) == len(close) == len(volume)):
        raise ValueError("All price/volume arrays must have same length")

    # Determine price range
    price_min = np.min(low)
    price_max = np.max(high)

    # Create price bins
    price_levels = np.linspace(price_min, price_max, num_bins)
    bin_width = (price_max - price_min) / (num_bins - 1)

    # Distribute volume across price range for each bar
    volume_at_price = np.zeros(num_bins)

    for i in range(len(close)):
        # Find which bins this bar's range covers
        low_bin = int((low[i] - price_min) / bin_width)
        high_bin = int((high[i] - price_min) / bin_width) + 1

        # Ensure bins are within range
        low_bin = max(0, low_bin)
        high_bin = min(num_bins, high_bin)

        # Distribute volume evenly across the bar's range
        if high_bin > low_bin:
            volume_per_bin = volume[i] / (high_bin - low_bin)
            volume_at_price[low_bin:high_bin] += volume_per_bin

    # Find Point of Control (price level with highest volume)
    poc_index = np.argmax(volume_at_price)
    poc = price_levels[poc_index]

    # Calculate Value Area (70% of volume around POC)
    total_volume = np.sum(volume_at_price)
    target_volume = total_volume * value_area_percent

    # Expand from POC until we reach target volume
    accumulated_volume = volume_at_price[poc_index]
    low_index = poc_index
    high_index = poc_index

    while accumulated_volume < target_volume:
        # Check which side to expand
        expand_low = low_index > 0
        expand_high = high_index < num_bins - 1

        if expand_low and expand_high:
            # Expand to side with more volume
            if volume_at_price[low_index - 1] > volume_at_price[high_index + 1]:
                low_index -= 1
                accumulated_volume += volume_at_price[low_index]
            else:
                high_index += 1
                accumulated_volume += volume_at_price[high_index]
        elif expand_low:
            low_index -= 1
            accumulated_volume += volume_at_price[low_index]
        elif expand_high:
            high_index += 1
            accumulated_volume += volume_at_price[high_index]
        else:
            break

    value_area_low = price_levels[low_index]
    value_area_high = price_levels[high_index]

    # Identify high volume nodes (above average)
    avg_volume = np.mean(volume_at_price)
    volume_nodes = price_levels[volume_at_price > avg_volume * 1.5]

    return VolumeProfileResult(
        price_levels=price_levels,
        volume_at_price=volume_at_price,
        poc=poc,
        value_area_high=value_area_high,
        value_area_low=value_area_low,
        volume_nodes=volume_nodes,
    )


def calculate_ad_line(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    signal_period: int = 13,
) -> ADLineResult:
    """Calculate Accumulation/Distribution Line.

    The A/D Line measures the cumulative flow of money into and out of
    a security based on price and volume.

    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Trading volumes
        signal_period: Period for signal line (EMA)

    Returns:
        ADLineResult with A/D line, signal, and divergence
    """
    if not (len(high) == len(low) == len(close) == len(volume)):
        raise ValueError("All price/volume arrays must have same length")

    # Calculate Money Flow Multiplier
    # MFM = [(Close - Low) - (High - Close)] / (High - Low)
    numerator = (close - low) - (high - close)
    denominator = high - low

    # Handle division by zero (when high == low)
    money_flow_multiplier = np.divide(
        numerator, denominator, where=denominator != 0, out=np.zeros_like(numerator)
    )

    # Calculate Money Flow Volume
    money_flow_volume = money_flow_multiplier * volume

    # Calculate A/D Line (cumulative sum of money flow volume)
    ad_line = np.cumsum(money_flow_volume)

    # Calculate signal line (EMA of A/D Line)
    signal = _ema(ad_line, signal_period)

    # Calculate divergence
    price_norm = (close - np.min(close)) / (np.max(close) - np.min(close) + 1e-10)
    ad_norm = (ad_line - np.min(ad_line)) / (np.max(ad_line) - np.min(ad_line) + 1e-10)
    divergence = price_norm - ad_norm

    return ADLineResult(
        ad_line=ad_line,
        signal=signal,
        divergence=divergence,
        money_flow_multiplier=money_flow_multiplier,
    )


def calculate_mfi(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    period: int = 14,
) -> MFIResult:
    """Calculate Money Flow Index.

    MFI is a momentum oscillator that uses price and volume to identify
    overbought or oversold conditions.

    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Trading volumes
        period: Calculation period (typically 14)

    Returns:
        MFIResult with MFI values and overbought/oversold signals
    """
    if not (len(high) == len(low) == len(close) == len(volume)):
        raise ValueError("All price/volume arrays must have same length")

    if len(close) < period + 1:
        raise ValueError(f"Need at least {period + 1} data points")

    # Calculate typical price
    typical_price = (high + low + close) / 3

    # Calculate raw money flow
    raw_money_flow = typical_price * volume

    # Determine positive and negative money flow
    price_change = np.diff(typical_price, prepend=typical_price[0])
    positive_flow = np.where(price_change > 0, raw_money_flow, 0)
    negative_flow = np.where(price_change < 0, raw_money_flow, 0)

    # Calculate rolling sums
    positive_mf = _rolling_sum(positive_flow, period)
    negative_mf = _rolling_sum(negative_flow, period)

    # Calculate money flow ratio
    money_flow_ratio = np.divide(
        positive_mf,
        negative_mf,
        where=negative_mf != 0,
        out=np.full_like(positive_mf, 100.0),
    )

    # Calculate MFI: 100 - (100 / (1 + money_flow_ratio))
    mfi = 100 - (100 / (1 + money_flow_ratio))

    # Identify overbought/oversold conditions
    overbought = mfi > 80
    oversold = mfi < 20

    return MFIResult(
        mfi=mfi,
        money_flow_ratio=money_flow_ratio,
        overbought=overbought,
        oversold=oversold,
    )


# Helper functions
def _sma(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Simple Moving Average."""
    if period <= 0:
        raise ValueError("Period must be positive")

    result = np.full_like(data, np.nan)

    # Use cumsum for efficient SMA calculation
    cumsum = np.cumsum(data)
    result[period - 1 :] = (
        cumsum[period - 1 :] - np.concatenate([[0], cumsum[:-period]])
    ) / period

    # Fill initial values with expanding mean
    for i in range(min(period - 1, len(data))):
        result[i] = np.mean(data[: i + 1])

    return result


def _ema(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Exponential Moving Average."""
    if period <= 0:
        raise ValueError("Period must be positive")

    alpha = 2.0 / (period + 1)
    result = np.empty_like(data)
    result[0] = data[0]

    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


def _rolling_sum(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Rolling sum over specified period."""
    if period <= 0:
        raise ValueError("Period must be positive")

    result = np.full_like(data, np.nan)

    # Use cumsum for efficient rolling sum
    cumsum = np.cumsum(data)
    result[period - 1 :] = cumsum[period - 1 :] - np.concatenate(
        [[0], cumsum[:-period]]
    )

    # Fill initial values with expanding sum
    for i in range(min(period - 1, len(data))):
        result[i] = np.sum(data[: i + 1])

    return result
