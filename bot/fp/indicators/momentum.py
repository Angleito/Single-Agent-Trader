"""
Pure functional momentum indicators.

This module provides pure functional implementations of momentum-based
technical indicators including RSI, MACD, Stochastic, and ROC.
All functions are stateless and return immutable result types.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from bot.fp.types.indicators import (
    MACDResult,
    ROCResult,
    RSIResult,
    StochasticResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# Building Block Functions
# -----------------------------------------------------------------------------


def calculate_change(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate price changes between consecutive periods.

    Args:
        prices: Array of prices

    Returns:
        Array of price changes (length = len(prices) - 1)
    """
    if len(prices) < 2:
        return np.array([])
    return np.diff(prices)


def separate_gains_losses(
    changes: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Separate price changes into gains and losses.

    Args:
        changes: Array of price changes

    Returns:
        Tuple of (gains, losses) where losses are positive values
    """
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    return gains, losses


def exponential_moving_average(
    values: NDArray[np.float64], period: int, initial_sma: float | None = None
) -> NDArray[np.float64]:
    """
    Calculate exponential moving average.

    Args:
        values: Array of values
        period: EMA period
        initial_sma: Initial SMA for first EMA calculation

    Returns:
        Array of EMA values
    """
    if len(values) == 0:
        return np.array([])

    if len(values) < period:
        return np.full(len(values), np.nan)

    alpha = 2.0 / (period + 1)
    ema = np.empty(len(values))
    ema[: period - 1] = np.nan

    # Use provided initial SMA or calculate it
    if initial_sma is not None:
        ema[period - 1] = initial_sma
    else:
        ema[period - 1] = np.mean(values[:period])

    # Calculate EMA for remaining values
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

    return ema


def simple_moving_average(
    values: NDArray[np.float64], period: int
) -> NDArray[np.float64]:
    """
    Calculate simple moving average.

    Args:
        values: Array of values
        period: SMA period

    Returns:
        Array of SMA values
    """
    if len(values) < period:
        return np.full(len(values), np.nan)

    # Use convolution for efficient SMA calculation
    weights = np.ones(period) / period
    sma = np.convolve(values, weights, mode="valid")

    # Pad with NaN for the initial periods
    return np.concatenate([np.full(period - 1, np.nan), sma])


def wilder_moving_average(
    values: NDArray[np.float64], period: int
) -> NDArray[np.float64]:
    """
    Calculate Wilder's smoothed moving average (used in RSI).

    Args:
        values: Array of values
        period: Smoothing period

    Returns:
        Array of smoothed values
    """
    if len(values) < period:
        return np.full(len(values), np.nan)

    # Wilder's smoothing uses a different alpha
    alpha = 1.0 / period
    smoothed = np.empty(len(values))
    smoothed[: period - 1] = np.nan

    # Initial value is SMA
    smoothed[period - 1] = np.mean(values[:period])

    # Apply Wilder's smoothing
    for i in range(period, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed


# -----------------------------------------------------------------------------
# Main Indicator Functions
# -----------------------------------------------------------------------------


def rsi(
    prices: NDArray[np.float64],
    period: int,
    timestamp: datetime,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> RSIResult | None:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures momentum by comparing the magnitude of recent gains
    to recent losses. Values range from 0 to 100.

    Args:
        prices: Array of prices (oldest to newest)
        period: RSI period (typically 14)
        timestamp: Timestamp for the result
        overbought: Overbought threshold (default 70)
        oversold: Oversold threshold (default 30)

    Returns:
        RSIResult or None if insufficient data
    """
    if len(prices) < period + 1:
        return None

    # Calculate price changes
    changes = calculate_change(prices)

    # Separate gains and losses
    gains, losses = separate_gains_losses(changes)

    # Calculate smoothed averages using Wilder's method
    avg_gains = wilder_moving_average(gains, period)
    avg_losses = wilder_moving_average(losses, period)

    # Get the latest non-NaN values
    latest_avg_gain = (
        avg_gains[~np.isnan(avg_gains)][-1] if not np.all(np.isnan(avg_gains)) else 0
    )
    latest_avg_loss = (
        avg_losses[~np.isnan(avg_losses)][-1] if not np.all(np.isnan(avg_losses)) else 0
    )

    # Calculate RS and RSI
    if latest_avg_loss == 0:
        rsi_value = 100.0
    else:
        rs = latest_avg_gain / latest_avg_loss
        rsi_value = 100.0 - (100.0 / (1.0 + rs))

    return RSIResult(
        timestamp=timestamp,
        value=float(rsi_value),
        overbought=overbought,
        oversold=oversold,
    )


def macd(
    prices: NDArray[np.float64],
    fast_period: int,
    slow_period: int,
    signal_period: int,
    timestamp: datetime,
) -> MACDResult | None:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD is a trend-following momentum indicator that shows the
    relationship between two moving averages of prices.

    Args:
        prices: Array of prices (oldest to newest)
        fast_period: Fast EMA period (typically 12)
        slow_period: Slow EMA period (typically 26)
        signal_period: Signal line EMA period (typically 9)
        timestamp: Timestamp for the result

    Returns:
        MACDResult or None if insufficient data
    """
    if len(prices) < slow_period + signal_period:
        return None

    # Calculate EMAs
    ema_fast = exponential_moving_average(prices, fast_period)
    ema_slow = exponential_moving_average(prices, slow_period)

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Remove NaN values for signal calculation
    valid_macd = macd_line[~np.isnan(macd_line)]

    if len(valid_macd) < signal_period:
        return None

    # Calculate signal line (EMA of MACD)
    signal_line = exponential_moving_average(valid_macd, signal_period)

    # Get latest values
    latest_macd = valid_macd[-1]
    latest_signal = signal_line[-1] if not np.isnan(signal_line[-1]) else 0
    latest_histogram = latest_macd - latest_signal

    return MACDResult(
        timestamp=timestamp,
        macd=float(latest_macd),
        signal=float(latest_signal),
        histogram=float(latest_histogram),
    )


def stochastic(
    high_prices: NDArray[np.float64],
    low_prices: NDArray[np.float64],
    close_prices: NDArray[np.float64],
    k_period: int,
    d_period: int,
    smooth_k: int,
    timestamp: datetime,
    overbought: float = 80.0,
    oversold: float = 20.0,
) -> StochasticResult | None:
    """
    Calculate Stochastic oscillator.

    The Stochastic oscillator is a momentum indicator that compares
    a closing price to its price range over a given time period.

    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of closing prices
        k_period: Period for %K calculation
        d_period: Period for %D calculation (SMA of %K)
        smooth_k: Smoothing period for %K
        timestamp: Timestamp for the result
        overbought: Overbought threshold (default 80)
        oversold: Oversold threshold (default 20)

    Returns:
        StochasticResult or None if insufficient data
    """
    required_length = k_period + smooth_k + d_period - 2
    if len(close_prices) < required_length:
        return None

    # Ensure all arrays have the same length
    min_length = min(len(high_prices), len(low_prices), len(close_prices))
    high_prices = high_prices[-min_length:]
    low_prices = low_prices[-min_length:]
    close_prices = close_prices[-min_length:]

    # Calculate raw %K
    raw_k = np.empty(len(close_prices))

    for i in range(len(close_prices)):
        if i < k_period - 1:
            raw_k[i] = np.nan
        else:
            period_high = np.max(high_prices[i - k_period + 1 : i + 1])
            period_low = np.min(low_prices[i - k_period + 1 : i + 1])

            if period_high == period_low:
                raw_k[i] = 50.0  # Middle value when no range
            else:
                raw_k[i] = (
                    100.0 * (close_prices[i] - period_low) / (period_high - period_low)
                )

    # Apply smoothing to %K values
    smooth_k_values = simple_moving_average(raw_k, smooth_k)

    # Calculate %D (SMA of smoothed %K)
    d_values = simple_moving_average(smooth_k_values, d_period)

    # Get latest values
    latest_k = (
        smooth_k_values[~np.isnan(smooth_k_values)][-1]
        if not np.all(np.isnan(smooth_k_values))
        else 50.0
    )
    latest_d = (
        d_values[~np.isnan(d_values)][-1] if not np.all(np.isnan(d_values)) else 50.0
    )

    return StochasticResult(
        timestamp=timestamp,
        k_percent=float(latest_k),
        d_percent=float(latest_d),
        overbought=overbought,
        oversold=oversold,
    )


def rate_of_change(
    prices: NDArray[np.float64], period: int, timestamp: datetime
) -> ROCResult | None:
    """
    Calculate Rate of Change (ROC).

    ROC measures the percentage change in price between the current
    price and the price n periods ago.

    Args:
        prices: Array of prices (oldest to newest)
        period: Look-back period
        timestamp: Timestamp for the result

    Returns:
        ROCResult or None if insufficient data
    """
    if len(prices) < period + 1:
        return None

    # Get current and past prices
    current_price = prices[-1]
    past_price = prices[-period - 1]

    # Calculate ROC
    if past_price == 0:
        roc_value = 0.0
    else:
        roc_value = ((current_price - past_price) / past_price) * 100.0

    return ROCResult(timestamp=timestamp, value=float(roc_value), period=period)


# -----------------------------------------------------------------------------
# Batch Calculation Functions
# -----------------------------------------------------------------------------


def calculate_all_momentum_indicators(
    prices: NDArray[np.float64],
    high_prices: NDArray[np.float64] | None = None,
    low_prices: NDArray[np.float64] | None = None,
    timestamp: datetime | None = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    stoch_k_period: int = 14,
    stoch_d_period: int = 3,
    stoch_smooth_k: int = 3,
    roc_period: int = 12,
) -> dict[str, RSIResult | MACDResult | StochasticResult | ROCResult | None]:
    """
    Calculate all momentum indicators at once.

    Args:
        prices: Array of closing prices
        high_prices: Array of high prices (for Stochastic)
        low_prices: Array of low prices (for Stochastic)
        timestamp: Timestamp for results (defaults to now)
        rsi_period: RSI period
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
        stoch_k_period: Stochastic %K period
        stoch_d_period: Stochastic %D period
        stoch_smooth_k: Stochastic %K smoothing
        roc_period: ROC period

    Returns:
        Dictionary of indicator results
    """
    if timestamp is None:
        timestamp = datetime.now(UTC)

    results: dict[str, RSIResult | MACDResult | StochasticResult | ROCResult | None] = (
        {}
    )

    # Calculate RSI
    results["rsi"] = rsi(prices, rsi_period, timestamp)

    # Calculate MACD
    results["macd"] = macd(prices, macd_fast, macd_slow, macd_signal, timestamp)

    # Calculate Stochastic (if high/low prices provided)
    if high_prices is not None and low_prices is not None:
        results["stochastic"] = stochastic(
            high_prices,
            low_prices,
            prices,
            stoch_k_period,
            stoch_d_period,
            stoch_smooth_k,
            timestamp,
        )
    else:
        results["stochastic"] = None

    # Calculate ROC
    results["roc"] = rate_of_change(prices, roc_period, timestamp)

    return results
