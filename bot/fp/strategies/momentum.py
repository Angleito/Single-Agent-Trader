"""Pure functional momentum trading strategies.

This module implements momentum-based trading strategies using functional programming
principles. All functions are pure, deterministic, and side-effect free.
"""

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from bot.fp.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_sma,
    calculate_volume_sma,
)

# Type aliases
Signal = Literal["LONG", "SHORT", "HOLD"]
PriceArray = NDArray[np.float64]
VolumeArray = NDArray[np.float64]


class MomentumSignal(NamedTuple):
    """Momentum trading signal with metadata."""

    signal: Signal
    strength: float  # 0.0 to 1.0
    reason: str
    indicators: dict[str, float]


def momentum_strategy(
    lookback_short: int = 10,
    lookback_long: int = 20,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    volume_multiplier: float = 1.5,
    breakout_threshold: float = 0.02,
) -> Callable[[PriceArray, VolumeArray], MomentumSignal]:
    """Create a momentum trading strategy function.

    Args:
        lookback_short: Short-term MA period
        lookback_long: Long-term MA period
        rsi_period: RSI calculation period
        rsi_oversold: RSI oversold threshold
        rsi_overbought: RSI overbought threshold
        volume_multiplier: Volume confirmation multiplier
        breakout_threshold: Price breakout threshold (e.g., 0.02 = 2%)

    Returns:
        Strategy function that takes prices and volumes and returns a signal
    """

    def strategy(prices: PriceArray, volumes: VolumeArray) -> MomentumSignal:
        """Execute momentum strategy on price and volume data.

        Args:
            prices: Array of closing prices
            volumes: Array of trading volumes

        Returns:
            MomentumSignal with trade decision and metadata
        """
        if len(prices) < max(lookback_long, rsi_period):
            return MomentumSignal(
                signal="HOLD", strength=0.0, reason="Insufficient data", indicators={}
            )

        # Calculate indicators
        sma_short = calculate_sma(prices, lookback_short)
        sma_long = calculate_sma(prices, lookback_long)
        ema_short = calculate_ema(prices, lookback_short)
        ema_long = calculate_ema(prices, lookback_long)
        rsi = calculate_rsi(prices, rsi_period)
        volume_sma = calculate_volume_sma(volumes, lookback_short)

        # Current values
        current_price = prices[-1]
        current_volume = volumes[-1]
        current_rsi = rsi[-1]

        # Moving average signals
        sma_signal = _ma_crossover_signal(sma_short, sma_long)
        ema_signal = _ma_crossover_signal(ema_short, ema_long)

        # RSI momentum signal
        rsi_signal = _rsi_momentum_signal(current_rsi, rsi_oversold, rsi_overbought)

        # Volume confirmation
        volume_confirmed = _is_volume_confirmed(
            current_volume, volume_sma[-1], volume_multiplier
        )

        # Price breakout signal
        breakout_signal = _breakout_signal(prices, lookback_long, breakout_threshold)

        # Trend strength
        trend_strength = _calculate_trend_strength(
            current_price, sma_short[-1], sma_long[-1]
        )

        # Combine signals
        signal, strength, reason = _combine_momentum_signals(
            sma_signal=sma_signal,
            ema_signal=ema_signal,
            rsi_signal=rsi_signal,
            breakout_signal=breakout_signal,
            volume_confirmed=volume_confirmed,
            trend_strength=trend_strength,
        )

        indicators = {
            "sma_short": sma_short[-1],
            "sma_long": sma_long[-1],
            "ema_short": ema_short[-1],
            "ema_long": ema_long[-1],
            "rsi": current_rsi,
            "volume": current_volume,
            "volume_sma": volume_sma[-1],
            "trend_strength": trend_strength,
        }

        return MomentumSignal(
            signal=signal, strength=strength, reason=reason, indicators=indicators
        )

    return strategy


def _ma_crossover_signal(
    ma_short: NDArray[np.float64], ma_long: NDArray[np.float64]
) -> Signal:
    """Determine signal from moving average crossover.

    Args:
        ma_short: Short-term moving average
        ma_long: Long-term moving average

    Returns:
        Trading signal based on crossover
    """
    if len(ma_short) < 2 or len(ma_long) < 2:
        return "HOLD"

    # Current positions
    short_above = ma_short[-1] > ma_long[-1]
    short_was_above = ma_short[-2] > ma_long[-2]

    # Golden cross (bullish)
    if not short_was_above and short_above:
        return "LONG"

    # Death cross (bearish)
    if short_was_above and not short_above:
        return "SHORT"

    return "HOLD"


def _rsi_momentum_signal(rsi: float, oversold: float, overbought: float) -> Signal:
    """Generate signal based on RSI momentum.

    Args:
        rsi: Current RSI value
        oversold: Oversold threshold
        overbought: Overbought threshold

    Returns:
        Trading signal based on RSI
    """
    if rsi < oversold:
        return "LONG"
    if rsi > overbought:
        return "SHORT"
    return "HOLD"


def _is_volume_confirmed(
    current_volume: float, average_volume: float, multiplier: float
) -> bool:
    """Check if current volume confirms the signal.

    Args:
        current_volume: Current trading volume
        average_volume: Average trading volume
        multiplier: Required volume multiplier

    Returns:
        True if volume is above average by multiplier
    """
    return current_volume > average_volume * multiplier


def _breakout_signal(
    prices: NDArray[np.float64], lookback: int, threshold: float
) -> Signal:
    """Detect price breakouts.

    Args:
        prices: Array of prices
        lookback: Lookback period for range
        threshold: Breakout threshold percentage

    Returns:
        Breakout signal
    """
    if len(prices) < lookback:
        return "HOLD"

    recent_prices = prices[-lookback:]
    current_price = prices[-1]

    high = np.max(recent_prices[:-1])  # Exclude current
    low = np.min(recent_prices[:-1])

    # Upward breakout
    if current_price > high * (1 + threshold):
        return "LONG"

    # Downward breakout
    if current_price < low * (1 - threshold):
        return "SHORT"

    return "HOLD"


def _calculate_trend_strength(price: float, sma_short: float, sma_long: float) -> float:
    """Calculate trend strength based on price and MAs.

    Args:
        price: Current price
        sma_short: Short-term SMA
        sma_long: Long-term SMA

    Returns:
        Trend strength from -1 (strong down) to 1 (strong up)
    """
    # Price relative to short MA
    short_diff = (price - sma_short) / sma_short

    # Short MA relative to long MA
    ma_diff = (sma_short - sma_long) / sma_long

    # Combine with weights
    strength = 0.6 * short_diff + 0.4 * ma_diff

    # Normalize to [-1, 1]
    return np.clip(strength, -1.0, 1.0)


def _combine_momentum_signals(
    sma_signal: Signal,
    ema_signal: Signal,
    rsi_signal: Signal,
    breakout_signal: Signal,
    volume_confirmed: bool,
    trend_strength: float,
) -> tuple[Signal, float, str]:
    """Combine multiple momentum signals into final decision.

    Args:
        sma_signal: SMA crossover signal
        ema_signal: EMA crossover signal
        rsi_signal: RSI momentum signal
        breakout_signal: Price breakout signal
        volume_confirmed: Volume confirmation flag
        trend_strength: Trend strength indicator

    Returns:
        Tuple of (signal, strength, reason)
    """
    # Count bullish and bearish signals
    signals = [sma_signal, ema_signal, rsi_signal, breakout_signal]
    long_count = sum(1 for s in signals if s == "LONG")
    short_count = sum(1 for s in signals if s == "SHORT")

    # Strong consensus with volume
    if long_count >= 3 and volume_confirmed and trend_strength > 0.3:
        return "LONG", 0.9, "Strong bullish momentum with volume"

    if short_count >= 3 and volume_confirmed and trend_strength < -0.3:
        return "SHORT", 0.9, "Strong bearish momentum with volume"

    # Moderate consensus
    if long_count >= 2 and short_count == 0:
        strength = 0.7 if volume_confirmed else 0.5
        reason = "Bullish momentum" + (" with volume" if volume_confirmed else "")
        return "LONG", strength, reason

    if short_count >= 2 and long_count == 0:
        strength = 0.7 if volume_confirmed else 0.5
        reason = "Bearish momentum" + (" with volume" if volume_confirmed else "")
        return "SHORT", strength, reason

    # Breakout with confirmation
    if breakout_signal != "HOLD" and volume_confirmed:
        direction = "Bullish" if breakout_signal == "LONG" else "Bearish"
        return breakout_signal, 0.6, f"{direction} breakout with volume"

    # MA crossover with trend alignment
    if sma_signal == ema_signal and sma_signal != "HOLD":
        if (sma_signal == "LONG" and trend_strength > 0.2) or (
            sma_signal == "SHORT" and trend_strength < -0.2
        ):
            direction = "Bullish" if sma_signal == "LONG" else "Bearish"
            return sma_signal, 0.5, f"{direction} MA crossover"

    # Default to hold
    return "HOLD", 0.0, "No clear momentum signal"


# Specialized momentum strategies


def trend_following_strategy(
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    atr_multiplier: float = 2.0,
) -> Callable[[PriceArray, VolumeArray], MomentumSignal]:
    """Create a trend following strategy using MACD-like signals.

    Args:
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        atr_multiplier: ATR multiplier for volatility adjustment

    Returns:
        Strategy function
    """

    def strategy(prices: PriceArray, volumes: VolumeArray) -> MomentumSignal:
        if len(prices) < slow_period + signal_period:
            return MomentumSignal(
                signal="HOLD", strength=0.0, reason="Insufficient data", indicators={}
            )

        # Calculate MACD components
        ema_fast = calculate_ema(prices, fast_period)
        ema_slow = calculate_ema(prices, slow_period)

        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        # Current values
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        prev_histogram = histogram[-2] if len(histogram) > 1 else 0

        # Trend direction
        trend_up = current_macd > current_signal
        trend_down = current_macd < current_signal

        # Momentum increasing
        momentum_increasing = current_histogram > prev_histogram

        # Signal generation
        if trend_up and momentum_increasing and current_histogram > 0:
            signal = "LONG"
            strength = min(abs(current_histogram) / (abs(current_macd) + 1e-10), 1.0)
            reason = "Uptrend with increasing momentum"
        elif trend_down and not momentum_increasing and current_histogram < 0:
            signal = "SHORT"
            strength = min(abs(current_histogram) / (abs(current_macd) + 1e-10), 1.0)
            reason = "Downtrend with increasing momentum"
        else:
            signal = "HOLD"
            strength = 0.0
            reason = "No clear trend signal"

        indicators = {
            "macd": current_macd,
            "signal": current_signal,
            "histogram": current_histogram,
            "ema_fast": ema_fast[-1],
            "ema_slow": ema_slow[-1],
        }

        return MomentumSignal(
            signal=signal, strength=strength, reason=reason, indicators=indicators
        )

    return strategy


def breakout_momentum_strategy(
    lookback: int = 20,
    breakout_pct: float = 0.015,
    volume_surge: float = 2.0,
    confirm_bars: int = 2,
) -> Callable[[PriceArray, VolumeArray], MomentumSignal]:
    """Create a breakout momentum strategy.

    Args:
        lookback: Period for detecting range
        breakout_pct: Breakout threshold percentage
        volume_surge: Required volume increase
        confirm_bars: Bars to confirm breakout

    Returns:
        Strategy function
    """

    def strategy(prices: PriceArray, volumes: VolumeArray) -> MomentumSignal:
        if len(prices) < lookback + confirm_bars:
            return MomentumSignal(
                signal="HOLD", strength=0.0, reason="Insufficient data", indicators={}
            )

        # Calculate range
        recent_high = np.max(prices[-lookback:-confirm_bars])
        recent_low = np.min(prices[-lookback:-confirm_bars])
        range_size = recent_high - recent_low

        # Volume analysis
        avg_volume = np.mean(volumes[-lookback:])
        recent_volumes = volumes[-confirm_bars:]

        # Check for breakout
        current_price = prices[-1]
        breakout_up = current_price > recent_high * (1 + breakout_pct)
        breakout_down = current_price < recent_low * (1 - breakout_pct)

        # Volume confirmation
        volume_surge_confirmed = np.all(recent_volumes > avg_volume * volume_surge)

        # Price confirmation (consecutive closes above/below)
        if breakout_up:
            price_confirmed = np.all(prices[-confirm_bars:] > recent_high)
        elif breakout_down:
            price_confirmed = np.all(prices[-confirm_bars:] < recent_low)
        else:
            price_confirmed = False

        # Generate signal
        if breakout_up and volume_surge_confirmed and price_confirmed:
            signal = "LONG"
            strength = min((current_price - recent_high) / range_size, 1.0)
            reason = f"Upward breakout confirmed over {confirm_bars} bars"
        elif breakout_down and volume_surge_confirmed and price_confirmed:
            signal = "SHORT"
            strength = min((recent_low - current_price) / range_size, 1.0)
            reason = f"Downward breakout confirmed over {confirm_bars} bars"
        else:
            signal = "HOLD"
            strength = 0.0
            reason = "No confirmed breakout"

        indicators = {
            "range_high": recent_high,
            "range_low": recent_low,
            "range_size": range_size,
            "avg_volume": avg_volume,
            "current_volume": volumes[-1],
            "breakout_distance": (
                abs(current_price - recent_high)
                if breakout_up
                else abs(current_price - recent_low)
                if breakout_down
                else 0.0
            ),
        }

        return MomentumSignal(
            signal=signal, strength=strength, reason=reason, indicators=indicators
        )

    return strategy


def rsi_reversal_strategy(
    rsi_period: int = 14,
    oversold: float = 25.0,
    overbought: float = 75.0,
    divergence_lookback: int = 10,
) -> Callable[[PriceArray, VolumeArray], MomentumSignal]:
    """Create an RSI reversal strategy with divergence detection.

    Args:
        rsi_period: RSI calculation period
        oversold: Oversold threshold
        overbought: Overbought threshold
        divergence_lookback: Period to check for divergences

    Returns:
        Strategy function
    """

    def strategy(prices: PriceArray, volumes: VolumeArray) -> MomentumSignal:
        if len(prices) < rsi_period + divergence_lookback:
            return MomentumSignal(
                signal="HOLD", strength=0.0, reason="Insufficient data", indicators={}
            )

        # Calculate RSI
        rsi = calculate_rsi(prices, rsi_period)
        current_rsi = rsi[-1]

        # Check for divergences
        recent_prices = prices[-divergence_lookback:]
        recent_rsi = rsi[-divergence_lookback:]

        # Price highs/lows
        price_high_idx = np.argmax(recent_prices)
        price_low_idx = np.argmin(recent_prices)

        # RSI highs/lows
        rsi_high_idx = np.argmax(recent_rsi)
        rsi_low_idx = np.argmin(recent_rsi)

        # Detect divergences
        bearish_divergence = (
            price_high_idx == len(recent_prices) - 1  # Price at new high
            and recent_rsi[price_high_idx] < recent_rsi[rsi_high_idx]  # RSI lower
            and current_rsi > overbought
        )

        bullish_divergence = (
            price_low_idx == len(recent_prices) - 1  # Price at new low
            and recent_rsi[price_low_idx] > recent_rsi[rsi_low_idx]  # RSI higher
            and current_rsi < oversold
        )

        # Generate signals
        if bullish_divergence:
            signal = "LONG"
            strength = (oversold - current_rsi) / oversold
            reason = "Bullish RSI divergence in oversold zone"
        elif bearish_divergence:
            signal = "SHORT"
            strength = (current_rsi - overbought) / (100 - overbought)
            reason = "Bearish RSI divergence in overbought zone"
        elif current_rsi < oversold:
            signal = "LONG"
            strength = (oversold - current_rsi) / oversold * 0.7
            reason = "RSI oversold reversal"
        elif current_rsi > overbought:
            signal = "SHORT"
            strength = (current_rsi - overbought) / (100 - overbought) * 0.7
            reason = "RSI overbought reversal"
        else:
            signal = "HOLD"
            strength = 0.0
            reason = "No RSI reversal signal"

        indicators = {
            "rsi": current_rsi,
            "rsi_high": np.max(recent_rsi),
            "rsi_low": np.min(recent_rsi),
            "price_high": np.max(recent_prices),
            "price_low": np.min(recent_prices),
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence,
        }

        return MomentumSignal(
            signal=signal,
            strength=np.clip(strength, 0.0, 1.0),
            reason=reason,
            indicators=indicators,
        )

    return strategy


# Compatibility functions for test suites and other modules
# --------------------------------------------------------


def calculate_momentum_signal(
    market_snapshot: "MarketSnapshot", config: dict[str, Any] | None = None
) -> MomentumSignal:
    """
    Calculate momentum signal from a market snapshot.

    This function provides compatibility with existing test suites and modules
    that expect a simple calculate_momentum_signal function.

    Args:
        market_snapshot: Market data snapshot with prices and indicators
        config: Optional configuration dict with parameters

    Returns:
        MomentumSignal with trade decision and metadata
    """
    # Import here to avoid circular imports
    from bot.fp.types.market import MarketSnapshot

    if not isinstance(market_snapshot, MarketSnapshot):
        raise ValueError(f"Expected MarketSnapshot, got {type(market_snapshot)}")

    # Extract configuration parameters
    config = config or {}
    lookback_short = config.get("lookback_short", 10)
    lookback_long = config.get("lookback_long", 20)
    rsi_period = config.get("rsi_period", 14)
    rsi_oversold = config.get("rsi_oversold", 30.0)
    rsi_overbought = config.get("rsi_overbought", 70.0)
    volume_multiplier = config.get("volume_multiplier", 1.5)
    breakout_threshold = config.get("breakout_threshold", 0.02)

    # For simplified calculation with snapshot data, use basic momentum
    current_price = float(market_snapshot.price)

    # If we have technical indicators, use them
    if market_snapshot.high_20 is not None and market_snapshot.low_20 is not None:
        high_20 = float(market_snapshot.high_20)
        low_20 = float(market_snapshot.low_20)

        # Calculate price position in range
        price_range = high_20 - low_20
        if price_range > 0:
            price_position = (current_price - low_20) / price_range

            # Strong momentum signals
            if price_position > 0.8:  # Near high
                return MomentumSignal(
                    signal="LONG",
                    strength=price_position,
                    reason=f"Strong upward momentum (price position: {price_position:.2f})",
                    indicators={
                        "price_position": price_position,
                        "high_20": high_20,
                        "low_20": low_20,
                        "current_price": current_price,
                    },
                )
            if price_position < 0.2:  # Near low
                return MomentumSignal(
                    signal="SHORT",
                    strength=1 - price_position,
                    reason=f"Strong downward momentum (price position: {price_position:.2f})",
                    indicators={
                        "price_position": price_position,
                        "high_20": high_20,
                        "low_20": low_20,
                        "current_price": current_price,
                    },
                )

    # If we have SMA, use it for trend determination
    if market_snapshot.sma_20 is not None:
        sma_20 = float(market_snapshot.sma_20)
        price_vs_sma = (current_price - sma_20) / sma_20

        # Trend momentum
        if price_vs_sma > 0.05:  # 5% above SMA
            return MomentumSignal(
                signal="LONG",
                strength=min(abs(price_vs_sma) * 10, 1.0),
                reason=f"Price above SMA trend (deviation: {price_vs_sma:.1%})",
                indicators={
                    "price_vs_sma": price_vs_sma,
                    "sma_20": sma_20,
                    "current_price": current_price,
                },
            )
        if price_vs_sma < -0.05:  # 5% below SMA
            return MomentumSignal(
                signal="SHORT",
                strength=min(abs(price_vs_sma) * 10, 1.0),
                reason=f"Price below SMA trend (deviation: {price_vs_sma:.1%})",
                indicators={
                    "price_vs_sma": price_vs_sma,
                    "sma_20": sma_20,
                    "current_price": current_price,
                },
            )

    # Default to hold if no clear momentum
    return MomentumSignal(
        signal="HOLD",
        strength=0.0,
        reason="No clear momentum signal detected",
        indicators={
            "current_price": current_price,
            "has_technical_data": any(
                [
                    market_snapshot.high_20 is not None,
                    market_snapshot.low_20 is not None,
                    market_snapshot.sma_20 is not None,
                ]
            ),
        },
    )
