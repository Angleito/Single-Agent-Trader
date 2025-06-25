"""Mean reversion trading strategies.

This module implements various mean reversion strategies including:
- Bollinger Band reversal strategy
- RSI oversold/overbought strategy
- Statistical arbitrage
- Pair trading logic
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from bot.fp.indicators.technical import (
    bollinger_bands,
    moving_average,
    rsi,
    standard_deviation,
)
from bot.fp.strategies.signals import Signal, SignalType


class MeanReversionType(Enum):
    """Types of mean reversion strategies."""

    BOLLINGER_REVERSAL = "bollinger_reversal"
    RSI_EXTREMES = "rsi_extremes"
    STATISTICAL_ARB = "statistical_arbitrage"
    PAIR_TRADING = "pair_trading"


@dataclass
class MeanReversionSignal(Signal):
    """Mean reversion specific signal with additional metrics."""

    deviation_from_mean: float
    mean_price: float
    zscore: float
    reversion_target: float
    risk_adjusted_size: float


def calculate_zscore(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """Calculate z-score for price deviation from mean.

    Args:
        prices: Price series
        lookback: Lookback period for mean and std

    Returns:
        Z-score series
    """
    mean = prices.rolling(window=lookback).mean()
    std = prices.rolling(window=lookback).std()

    # Avoid division by zero
    std = std.replace(0, np.nan)

    return (prices - mean) / std


def bollinger_band_reversal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
    entry_threshold: float = 0.95,
    exit_threshold: float = 0.5,
) -> list[MeanReversionSignal]:
    """Bollinger Band reversal strategy.

    Generates signals when price touches or exceeds Bollinger Bands,
    expecting reversion to the mean.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: BB period
        num_std: Number of standard deviations
        entry_threshold: Band penetration threshold for entry (0-1)
        exit_threshold: Mean reversion threshold for exit (0-1)

    Returns:
        List of mean reversion signals
    """
    signals = []

    # Calculate Bollinger Bands
    upper, middle, lower = bollinger_bands(close, period, num_std)

    # Calculate z-score for deviation measurement
    zscore = calculate_zscore(close, period)

    # Calculate band width for volatility-based sizing
    band_width = (upper - lower) / middle

    for i in range(period, len(close)):
        # Skip if any values are NaN
        if pd.isna([upper.iloc[i], lower.iloc[i], middle.iloc[i]]).any():
            continue

        current_close = close.iloc[i]
        current_high = high.iloc[i]
        current_low = low.iloc[i]

        # Calculate position relative to bands
        upper.iloc[i] - current_close
        current_close - lower.iloc[i]
        band_range = upper.iloc[i] - lower.iloc[i]

        # Long signal: Price near/below lower band
        if current_low <= lower.iloc[i] + (band_range * (1 - entry_threshold)):
            deviation = (middle.iloc[i] - current_close) / middle.iloc[i]

            # Risk-adjusted position size based on volatility
            volatility_adj = 1.0 / (1.0 + band_width.iloc[i])
            deviation_adj = min(abs(zscore.iloc[i]) / 2.0, 2.0)  # Cap at 2x
            risk_adjusted_size = volatility_adj * deviation_adj

            signal = MeanReversionSignal(
                signal_type=SignalType.LONG,
                strength=min(abs(zscore.iloc[i]) / 3.0, 1.0),  # Normalize to 0-1
                timestamp=close.index[i],
                price=current_close,
                reason=f"BB reversal: Price at lower band, z-score: {zscore.iloc[i]:.2f}",
                deviation_from_mean=deviation,
                mean_price=middle.iloc[i],
                zscore=zscore.iloc[i],
                reversion_target=middle.iloc[i] * (1 - exit_threshold * deviation),
                risk_adjusted_size=risk_adjusted_size,
            )
            signals.append(signal)

        # Short signal: Price near/above upper band
        elif current_high >= upper.iloc[i] - (band_range * (1 - entry_threshold)):
            deviation = (current_close - middle.iloc[i]) / middle.iloc[i]

            # Risk-adjusted position size
            volatility_adj = 1.0 / (1.0 + band_width.iloc[i])
            deviation_adj = min(abs(zscore.iloc[i]) / 2.0, 2.0)
            risk_adjusted_size = volatility_adj * deviation_adj

            signal = MeanReversionSignal(
                signal_type=SignalType.SHORT,
                strength=min(abs(zscore.iloc[i]) / 3.0, 1.0),
                timestamp=close.index[i],
                price=current_close,
                reason=f"BB reversal: Price at upper band, z-score: {zscore.iloc[i]:.2f}",
                deviation_from_mean=deviation,
                mean_price=middle.iloc[i],
                zscore=zscore.iloc[i],
                reversion_target=middle.iloc[i] * (1 + exit_threshold * abs(deviation)),
                risk_adjusted_size=risk_adjusted_size,
            )
            signals.append(signal)

    return signals


def rsi_extremes_strategy(
    close: pd.Series,
    rsi_period: int = 14,
    oversold_threshold: float = 30.0,
    overbought_threshold: float = 70.0,
    extreme_oversold: float = 20.0,
    extreme_overbought: float = 80.0,
    mean_period: int = 20,
) -> list[MeanReversionSignal]:
    """RSI extremes mean reversion strategy.

    Generates signals at RSI extremes expecting mean reversion.

    Args:
        close: Close price series
        rsi_period: RSI calculation period
        oversold_threshold: RSI oversold level
        overbought_threshold: RSI overbought level
        extreme_oversold: Extreme oversold for stronger signals
        extreme_overbought: Extreme overbought for stronger signals
        mean_period: Period for calculating price mean

    Returns:
        List of mean reversion signals
    """
    signals = []

    # Calculate RSI
    rsi_values = rsi(close, rsi_period)

    # Calculate price mean and z-score
    price_mean = moving_average(close, mean_period)
    zscore = calculate_zscore(close, mean_period)

    # Calculate volatility for position sizing
    volatility = standard_deviation(close, mean_period)

    for i in range(max(rsi_period, mean_period), len(close)):
        if pd.isna([rsi_values.iloc[i], price_mean.iloc[i]]).any():
            continue

        current_rsi = rsi_values.iloc[i]
        current_close = close.iloc[i]
        current_mean = price_mean.iloc[i]
        current_vol = volatility.iloc[i]

        # Long signal: RSI oversold
        if current_rsi <= oversold_threshold:
            # Stronger signal if extremely oversold
            if current_rsi <= extreme_oversold:
                strength = 1.0
                size_multiplier = 1.5
            else:
                strength = (oversold_threshold - current_rsi) / (
                    oversold_threshold - extreme_oversold
                )
                size_multiplier = 1.0

            deviation = (current_mean - current_close) / current_mean

            # Risk-adjusted size based on volatility and RSI extreme
            vol_adjusted = current_mean * 0.02 / current_vol if current_vol > 0 else 1.0
            risk_adjusted_size = min(vol_adjusted * size_multiplier, 2.0)

            signal = MeanReversionSignal(
                signal_type=SignalType.LONG,
                strength=strength,
                timestamp=close.index[i],
                price=current_close,
                reason=f"RSI oversold: {current_rsi:.1f}, z-score: {zscore.iloc[i]:.2f}",
                deviation_from_mean=deviation,
                mean_price=current_mean,
                zscore=zscore.iloc[i],
                reversion_target=current_mean * 0.98,  # Target 98% of mean
                risk_adjusted_size=risk_adjusted_size,
            )
            signals.append(signal)

        # Short signal: RSI overbought
        elif current_rsi >= overbought_threshold:
            # Stronger signal if extremely overbought
            if current_rsi >= extreme_overbought:
                strength = 1.0
                size_multiplier = 1.5
            else:
                strength = (current_rsi - overbought_threshold) / (
                    extreme_overbought - overbought_threshold
                )
                size_multiplier = 1.0

            deviation = (current_close - current_mean) / current_mean

            # Risk-adjusted size
            vol_adjusted = current_mean * 0.02 / current_vol if current_vol > 0 else 1.0
            risk_adjusted_size = min(vol_adjusted * size_multiplier, 2.0)

            signal = MeanReversionSignal(
                signal_type=SignalType.SHORT,
                strength=strength,
                timestamp=close.index[i],
                price=current_close,
                reason=f"RSI overbought: {current_rsi:.1f}, z-score: {zscore.iloc[i]:.2f}",
                deviation_from_mean=deviation,
                mean_price=current_mean,
                zscore=zscore.iloc[i],
                reversion_target=current_mean * 1.02,  # Target 102% of mean
                risk_adjusted_size=risk_adjusted_size,
            )
            signals.append(signal)

    return signals


def statistical_arbitrage(
    close: pd.Series,
    lookback: int = 60,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    min_half_life: int = 5,
    max_half_life: int = 30,
) -> list[MeanReversionSignal]:
    """Statistical arbitrage based on mean reversion.

    Uses statistical properties to identify mean reversion opportunities.

    Args:
        close: Close price series
        lookback: Lookback period for statistics
        entry_zscore: Z-score threshold for entry
        exit_zscore: Z-score threshold for exit
        min_half_life: Minimum acceptable half-life
        max_half_life: Maximum acceptable half-life

    Returns:
        List of mean reversion signals
    """
    signals = []

    # Calculate rolling statistics
    rolling_mean = close.rolling(window=lookback).mean()
    rolling_std = close.rolling(window=lookback).std()
    zscore = calculate_zscore(close, lookback)

    # Calculate half-life of mean reversion using OLS
    # This is a simplified calculation
    for i in range(lookback * 2, len(close)):
        # Get recent price window
        price_window = close.iloc[i - lookback : i]

        # Calculate lag-1 differences
        price_diff = price_window.diff().dropna()
        price_lag = price_window.shift(1).dropna()

        # Simple regression coefficient (simplified)
        if len(price_diff) > 0 and price_lag.std() > 0:
            beta = -price_diff.cov(price_lag) / price_lag.var()
            half_life = -np.log(2) / np.log(1 + beta) if beta < 0 else np.inf
        else:
            half_life = np.inf

        # Check if half-life is in acceptable range
        if min_half_life <= half_life <= max_half_life:
            current_z = zscore.iloc[i]
            current_close = close.iloc[i]
            current_mean = rolling_mean.iloc[i]
            current_std = rolling_std.iloc[i]

            if pd.isna([current_z, current_mean, current_std]).any():
                continue

            # Long signal: Z-score below -entry_zscore
            if current_z <= -entry_zscore:
                deviation = (current_mean - current_close) / current_mean

                # Risk sizing based on z-score and half-life
                z_size = min(abs(current_z) / entry_zscore - 1, 1.0)
                hl_size = 1.0 - (half_life - min_half_life) / (
                    max_half_life - min_half_life
                )
                risk_adjusted_size = (z_size + hl_size) / 2

                signal = MeanReversionSignal(
                    signal_type=SignalType.LONG,
                    strength=min(abs(current_z) / 3.0, 1.0),
                    timestamp=close.index[i],
                    price=current_close,
                    reason=f"Stat arb: z-score {current_z:.2f}, half-life {half_life:.1f}",
                    deviation_from_mean=deviation,
                    mean_price=current_mean,
                    zscore=current_z,
                    reversion_target=current_mean - exit_zscore * current_std,
                    risk_adjusted_size=risk_adjusted_size,
                )
                signals.append(signal)

            # Short signal: Z-score above entry_zscore
            elif current_z >= entry_zscore:
                deviation = (current_close - current_mean) / current_mean

                # Risk sizing
                z_size = min(abs(current_z) / entry_zscore - 1, 1.0)
                hl_size = 1.0 - (half_life - min_half_life) / (
                    max_half_life - min_half_life
                )
                risk_adjusted_size = (z_size + hl_size) / 2

                signal = MeanReversionSignal(
                    signal_type=SignalType.SHORT,
                    strength=min(abs(current_z) / 3.0, 1.0),
                    timestamp=close.index[i],
                    price=current_close,
                    reason=f"Stat arb: z-score {current_z:.2f}, half-life {half_life:.1f}",
                    deviation_from_mean=deviation,
                    mean_price=current_mean,
                    zscore=current_z,
                    reversion_target=current_mean + exit_zscore * current_std,
                    risk_adjusted_size=risk_adjusted_size,
                )
                signals.append(signal)

    return signals


def pair_trading_signals(
    asset1_close: pd.Series,
    asset2_close: pd.Series,
    lookback: int = 60,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    min_correlation: float = 0.7,
) -> tuple[list[MeanReversionSignal], list[MeanReversionSignal]]:
    """Pair trading strategy for two correlated assets.

    Identifies divergences between correlated assets and trades the spread.

    Args:
        asset1_close: First asset close prices
        asset2_close: Second asset close prices
        lookback: Lookback period for statistics
        entry_zscore: Entry threshold
        exit_zscore: Exit threshold
        min_correlation: Minimum correlation required

    Returns:
        Tuple of (asset1_signals, asset2_signals)
    """
    asset1_signals = []
    asset2_signals = []

    # Ensure series are aligned
    aligned_asset1, aligned_asset2 = asset1_close.align(asset2_close, join="inner")

    # Calculate rolling correlation
    rolling_corr = aligned_asset1.rolling(window=lookback).corr(aligned_asset2)

    # Calculate spread (log ratio for better statistical properties)
    spread = np.log(aligned_asset1 / aligned_asset2)
    spread_mean = spread.rolling(window=lookback).mean()
    spread_std = spread.rolling(window=lookback).std()
    spread_zscore = (spread - spread_mean) / spread_std

    for i in range(lookback, len(spread)):
        if pd.isna([rolling_corr.iloc[i], spread_zscore.iloc[i]]).any():
            continue

        # Only trade if correlation is high enough
        if rolling_corr.iloc[i] >= min_correlation:
            current_z = spread_zscore.iloc[i]
            timestamp = spread.index[i]

            # When spread is too low: Long asset1, Short asset2
            if current_z <= -entry_zscore:
                # Asset1 is relatively cheap
                signal1 = MeanReversionSignal(
                    signal_type=SignalType.LONG,
                    strength=min(abs(current_z) / 3.0, 1.0),
                    timestamp=timestamp,
                    price=aligned_asset1.iloc[i],
                    reason=f"Pair trade: spread z-score {current_z:.2f}, corr {rolling_corr.iloc[i]:.2f}",
                    deviation_from_mean=(spread_mean.iloc[i] - spread.iloc[i])
                    / abs(spread_mean.iloc[i]),
                    mean_price=aligned_asset1.iloc[i]
                    * np.exp(spread_mean.iloc[i] - spread.iloc[i]),
                    zscore=current_z,
                    reversion_target=aligned_asset1.iloc[i]
                    * np.exp(-exit_zscore * spread_std.iloc[i]),
                    risk_adjusted_size=min(abs(current_z) / entry_zscore, 2.0)
                    * rolling_corr.iloc[i],
                )
                asset1_signals.append(signal1)

                # Asset2 is relatively expensive
                signal2 = MeanReversionSignal(
                    signal_type=SignalType.SHORT,
                    strength=min(abs(current_z) / 3.0, 1.0),
                    timestamp=timestamp,
                    price=aligned_asset2.iloc[i],
                    reason=f"Pair trade: spread z-score {current_z:.2f}, corr {rolling_corr.iloc[i]:.2f}",
                    deviation_from_mean=(spread.iloc[i] - spread_mean.iloc[i])
                    / abs(spread_mean.iloc[i]),
                    mean_price=aligned_asset2.iloc[i]
                    * np.exp(spread.iloc[i] - spread_mean.iloc[i]),
                    zscore=-current_z,
                    reversion_target=aligned_asset2.iloc[i]
                    * np.exp(exit_zscore * spread_std.iloc[i]),
                    risk_adjusted_size=min(abs(current_z) / entry_zscore, 2.0)
                    * rolling_corr.iloc[i],
                )
                asset2_signals.append(signal2)

            # When spread is too high: Short asset1, Long asset2
            elif current_z >= entry_zscore:
                # Asset1 is relatively expensive
                signal1 = MeanReversionSignal(
                    signal_type=SignalType.SHORT,
                    strength=min(abs(current_z) / 3.0, 1.0),
                    timestamp=timestamp,
                    price=aligned_asset1.iloc[i],
                    reason=f"Pair trade: spread z-score {current_z:.2f}, corr {rolling_corr.iloc[i]:.2f}",
                    deviation_from_mean=(spread.iloc[i] - spread_mean.iloc[i])
                    / abs(spread_mean.iloc[i]),
                    mean_price=aligned_asset1.iloc[i]
                    * np.exp(spread_mean.iloc[i] - spread.iloc[i]),
                    zscore=current_z,
                    reversion_target=aligned_asset1.iloc[i]
                    * np.exp(exit_zscore * spread_std.iloc[i]),
                    risk_adjusted_size=min(abs(current_z) / entry_zscore, 2.0)
                    * rolling_corr.iloc[i],
                )
                asset1_signals.append(signal1)

                # Asset2 is relatively cheap
                signal2 = MeanReversionSignal(
                    signal_type=SignalType.LONG,
                    strength=min(abs(current_z) / 3.0, 1.0),
                    timestamp=timestamp,
                    price=aligned_asset2.iloc[i],
                    reason=f"Pair trade: spread z-score {current_z:.2f}, corr {rolling_corr.iloc[i]:.2f}",
                    deviation_from_mean=(spread_mean.iloc[i] - spread.iloc[i])
                    / abs(spread_mean.iloc[i]),
                    mean_price=aligned_asset2.iloc[i]
                    * np.exp(spread.iloc[i] - spread_mean.iloc[i]),
                    zscore=-current_z,
                    reversion_target=aligned_asset2.iloc[i]
                    * np.exp(-exit_zscore * spread_std.iloc[i]),
                    risk_adjusted_size=min(abs(current_z) / entry_zscore, 2.0)
                    * rolling_corr.iloc[i],
                )
                asset2_signals.append(signal2)

    return asset1_signals, asset2_signals


def mean_reversion_exit_signals(
    close: pd.Series,
    entry_signals: list[MeanReversionSignal],
    exit_ratio: float = 0.5,
    stop_loss_std: float = 3.0,
) -> list[Signal]:
    """Generate exit signals for mean reversion positions.

    Args:
        close: Current close prices
        entry_signals: Previous entry signals
        exit_ratio: Ratio of reversion to target for exit (0.5 = 50% reversion)
        stop_loss_std: Stop loss in standard deviations from entry

    Returns:
        List of exit signals
    """
    exit_signals = []

    for entry in entry_signals:
        # Find prices after entry
        mask = close.index > entry.timestamp
        if not mask.any():
            continue

        future_prices = close[mask]

        for _i, (timestamp, price) in enumerate(future_prices.items()):
            # Calculate progress toward reversion target
            if entry.signal_type == SignalType.LONG:
                reversion_progress = (price - entry.price) / (
                    entry.reversion_target - entry.price
                )
                stop_loss = entry.price - stop_loss_std * (
                    entry.price * 0.02
                )  # Simplified stop

                # Exit on target reached or stop loss
                if reversion_progress >= exit_ratio or price <= stop_loss:
                    exit_signal = Signal(
                        signal_type=SignalType.EXIT_LONG,
                        strength=1.0 if price <= stop_loss else reversion_progress,
                        timestamp=timestamp,
                        price=price,
                        reason=f"Mean reversion {'stop loss' if price <= stop_loss else 'target'}",
                    )
                    exit_signals.append(exit_signal)
                    break

            else:  # SHORT
                reversion_progress = (entry.price - price) / (
                    entry.price - entry.reversion_target
                )
                stop_loss = entry.price + stop_loss_std * (entry.price * 0.02)

                if reversion_progress >= exit_ratio or price >= stop_loss:
                    exit_signal = Signal(
                        signal_type=SignalType.EXIT_SHORT,
                        strength=1.0 if price >= stop_loss else reversion_progress,
                        timestamp=timestamp,
                        price=price,
                        reason=f"Mean reversion {'stop loss' if price >= stop_loss else 'target'}",
                    )
                    exit_signals.append(exit_signal)
                    break

    return exit_signals


# Compatibility functions for test suites and other modules
# --------------------------------------------------------


def calculate_mean_reversion_signal(
    market_snapshot: "MarketSnapshot", config: dict[str, Any] | None = None
) -> MeanReversionSignal:
    """
    Calculate mean reversion signal from a market snapshot.

    This function provides compatibility with existing test suites and modules
    that expect a simple calculate_mean_reversion_signal function.

    Args:
        market_snapshot: Market data snapshot with prices and indicators
        config: Optional configuration dict with parameters

    Returns:
        MeanReversionSignal with trade decision and metadata
    """
    # Import here to avoid circular imports
    from bot.fp.types.market import MarketSnapshot

    if not isinstance(market_snapshot, MarketSnapshot):
        raise ValueError(f"Expected MarketSnapshot, got {type(market_snapshot)}")

    # Extract configuration parameters
    config = config or {}
    z_score_threshold = config.get("z_score_threshold", 2.0)
    entry_threshold = config.get("entry_threshold", 0.95)
    exit_threshold = config.get("exit_threshold", 0.5)

    current_price = float(market_snapshot.price)
    current_time = market_snapshot.timestamp

    # If we have SMA and high/low data, calculate z-score
    if (
        market_snapshot.sma_20 is not None
        and market_snapshot.high_20 is not None
        and market_snapshot.low_20 is not None
    ):
        sma_20 = float(market_snapshot.sma_20)
        high_20 = float(market_snapshot.high_20)
        low_20 = float(market_snapshot.low_20)

        # Calculate approximate standard deviation from range
        price_range = high_20 - low_20
        approx_std = price_range / 4.0  # Approximation: range ≈ 4 * std

        if approx_std > 0:
            # Calculate z-score
            z_score = (current_price - sma_20) / approx_std
            deviation_from_mean = (current_price - sma_20) / sma_20

            # Strong mean reversion signals
            if z_score > z_score_threshold:  # Overbought
                return MeanReversionSignal(
                    signal_type=SignalType.SHORT,
                    strength=min(abs(z_score) / 3.0, 1.0),
                    timestamp=current_time,
                    price=Decimal(str(current_price)),
                    reason=f"Overbought: z-score {z_score:.2f} above threshold {z_score_threshold}",
                    deviation_from_mean=deviation_from_mean,
                    mean_price=sma_20,
                    zscore=z_score,
                    reversion_target=sma_20
                    * (1 + exit_threshold * abs(deviation_from_mean)),
                    risk_adjusted_size=min(abs(z_score) / z_score_threshold, 2.0),
                )
            if z_score < -z_score_threshold:  # Oversold
                return MeanReversionSignal(
                    signal_type=SignalType.LONG,
                    strength=min(abs(z_score) / 3.0, 1.0),
                    timestamp=current_time,
                    price=Decimal(str(current_price)),
                    reason=f"Oversold: z-score {z_score:.2f} below threshold {-z_score_threshold}",
                    deviation_from_mean=deviation_from_mean,
                    mean_price=sma_20,
                    zscore=z_score,
                    reversion_target=sma_20
                    * (1 - exit_threshold * abs(deviation_from_mean)),
                    risk_adjusted_size=min(abs(z_score) / z_score_threshold, 2.0),
                )

    # Fallback: simple deviation from SMA
    elif market_snapshot.sma_20 is not None:
        sma_20 = float(market_snapshot.sma_20)
        deviation = (current_price - sma_20) / sma_20

        # Strong deviation signals
        if deviation > 0.1:  # 10% above mean
            return MeanReversionSignal(
                signal_type=SignalType.SHORT,
                strength=min(abs(deviation) * 5, 1.0),
                timestamp=current_time,
                price=Decimal(str(current_price)),
                reason=f"Price {deviation:.1%} above SMA - mean reversion expected",
                deviation_from_mean=deviation,
                mean_price=sma_20,
                zscore=deviation * 5,  # Approximate z-score
                reversion_target=sma_20 * 1.05,  # Target 5% above mean
                risk_adjusted_size=min(abs(deviation) * 5, 1.5),
            )
        if deviation < -0.1:  # 10% below mean
            return MeanReversionSignal(
                signal_type=SignalType.LONG,
                strength=min(abs(deviation) * 5, 1.0),
                timestamp=current_time,
                price=Decimal(str(current_price)),
                reason=f"Price {deviation:.1%} below SMA - mean reversion expected",
                deviation_from_mean=deviation,
                mean_price=sma_20,
                zscore=deviation * 5,  # Approximate z-score
                reversion_target=sma_20 * 0.95,  # Target 5% below mean
                risk_adjusted_size=min(abs(deviation) * 5, 1.5),
            )

    # Default to hold if no clear mean reversion opportunity
    return MeanReversionSignal(
        signal_type=SignalType.HOLD,
        strength=0.0,
        timestamp=current_time,
        price=Decimal(str(current_price)),
        reason="No clear mean reversion signal detected",
        deviation_from_mean=0.0,
        mean_price=market_snapshot.sma_20 or current_price,
        zscore=0.0,
        reversion_target=current_price,
        risk_adjusted_size=0.0,
    )
