"""
Functional programming types for technical indicators.

This module provides immutable data structures for representing technical
indicator results and configurations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class IndicatorResult:
    """Base type for all indicator results."""

    timestamp: datetime

    def is_recent(self, max_age_seconds: int = 60) -> bool:
        """Check if the indicator result is recent."""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age <= max_age_seconds


@dataclass(frozen=True)
class MovingAverageResult(IndicatorResult):
    """Result from a moving average calculation."""

    value: float
    period: int

    def is_above(self, price: float) -> bool:
        """Check if price is above the moving average."""
        return price > self.value

    def is_below(self, price: float) -> bool:
        """Check if price is below the moving average."""
        return price < self.value

    def distance_from(self, price: float) -> float:
        """Calculate percentage distance from price to MA."""
        return ((price - self.value) / self.value) * 100


@dataclass(frozen=True)
class RSIResult(IndicatorResult):
    """Result from RSI (Relative Strength Index) calculation."""

    value: float
    overbought: float = 70.0
    oversold: float = 30.0

    def is_overbought(self) -> bool:
        """Check if RSI indicates overbought conditions."""
        return self.value >= self.overbought

    def is_oversold(self) -> bool:
        """Check if RSI indicates oversold conditions."""
        return self.value <= self.oversold

    def is_neutral(self) -> bool:
        """Check if RSI is in neutral zone."""
        return self.oversold < self.value < self.overbought

    def strength_level(self) -> str:
        """Get descriptive strength level."""
        if self.is_overbought():
            return "overbought"
        if self.is_oversold():
            return "oversold"
        if self.value >= 60:
            return "strong"
        if self.value <= 40:
            return "weak"
        return "neutral"


@dataclass(frozen=True)
class MACDResult(IndicatorResult):
    """Result from MACD (Moving Average Convergence Divergence) calculation."""

    macd: float
    signal: float
    histogram: float

    def is_bullish_crossover(self) -> bool:
        """Check if MACD crosses above signal (bullish)."""
        return self.macd > self.signal and self.histogram > 0

    def is_bearish_crossover(self) -> bool:
        """Check if MACD crosses below signal (bearish)."""
        return self.macd < self.signal and self.histogram < 0

    def is_diverging(self) -> bool:
        """Check if MACD is diverging from signal."""
        return abs(self.histogram) > abs(self.histogram * 0.1)

    def momentum_direction(self) -> str:
        """Get momentum direction."""
        if self.histogram > 0:
            return "bullish"
        if self.histogram < 0:
            return "bearish"
        return "neutral"


@dataclass(frozen=True)
class BollingerBandsResult(IndicatorResult):
    """Result from Bollinger Bands calculation."""

    upper: float
    middle: float
    lower: float
    width: float | None = None

    def __post_init__(self) -> None:
        """Calculate band width if not provided."""
        if self.width is None:
            object.__setattr__(self, "width", self.upper - self.lower)

    def is_price_above_upper(self, price: float) -> bool:
        """Check if price is above upper band."""
        return price > self.upper

    def is_price_below_lower(self, price: float) -> bool:
        """Check if price is below lower band."""
        return price < self.lower

    def is_price_in_bands(self, price: float) -> bool:
        """Check if price is within bands."""
        return self.lower <= price <= self.upper

    def price_position(self, price: float) -> float:
        """Get price position within bands (0-1, 0.5 is middle)."""
        if self.width and self.width > 0:
            return (price - self.lower) / self.width
        return 0.5

    def is_squeeze(self, threshold: float = 0.02) -> bool:
        """Check if bands are in a squeeze (low volatility)."""
        if self.width and self.middle > 0:
            return (self.width / self.middle) < threshold
        return False


@dataclass(frozen=True)
class VuManchuResult(IndicatorResult):
    """Result from VuManchu Cipher calculation."""

    wave_a: float
    wave_b: float
    signal: str | None = None

    def __post_init__(self) -> None:
        """Determine signal if not provided."""
        if self.signal is None:
            signal_value = self._determine_signal()
            object.__setattr__(self, "signal", signal_value)

    def _determine_signal(self) -> str:
        """Determine trading signal from wave values."""
        if self.is_bullish_crossover():
            return "LONG"
        if self.is_bearish_crossover():
            return "SHORT"
        return "NEUTRAL"

    def is_bullish_crossover(self) -> bool:
        """Check if Wave A crosses above Wave B (bullish)."""
        return self.wave_a > self.wave_b and self.wave_a < 0

    def is_bearish_crossover(self) -> bool:
        """Check if Wave A crosses below Wave B (bearish)."""
        return self.wave_a < self.wave_b and self.wave_a > 0

    def momentum_strength(self) -> float:
        """Calculate momentum strength (0-100)."""
        return min(abs(self.wave_a - self.wave_b) * 10, 100)

    def is_divergence(self, price_trend: str) -> bool:
        """Check for divergence between price and indicator."""
        indicator_trend = "up" if self.wave_a > 0 else "down"
        return price_trend != indicator_trend


@dataclass(frozen=True)
class StochasticResult(IndicatorResult):
    """Result from Stochastic oscillator calculation."""

    k_percent: float
    d_percent: float
    overbought: float = 80.0
    oversold: float = 20.0

    def is_overbought(self) -> bool:
        """Check if stochastic indicates overbought conditions."""
        return self.k_percent >= self.overbought

    def is_oversold(self) -> bool:
        """Check if stochastic indicates oversold conditions."""
        return self.k_percent <= self.oversold

    def is_bullish_crossover(self) -> bool:
        """Check if %K crosses above %D (bullish)."""
        return self.k_percent > self.d_percent

    def is_bearish_crossover(self) -> bool:
        """Check if %K crosses below %D (bearish)."""
        return self.k_percent < self.d_percent

    def momentum_zone(self) -> str:
        """Get current momentum zone."""
        if self.is_overbought():
            return "overbought"
        if self.is_oversold():
            return "oversold"
        if self.k_percent >= 50:
            return "bullish"
        return "bearish"


@dataclass(frozen=True)
class ROCResult(IndicatorResult):
    """Result from Rate of Change (ROC) calculation."""

    value: float
    period: int

    def is_positive(self) -> bool:
        """Check if ROC is positive (price rising)."""
        return self.value > 0

    def is_negative(self) -> bool:
        """Check if ROC is negative (price falling)."""
        return self.value < 0

    def momentum_strength(self) -> str:
        """Categorize momentum strength."""
        abs_value = abs(self.value)
        if abs_value >= 10:
            return "strong"
        if abs_value >= 5:
            return "moderate"
        if abs_value >= 2:
            return "weak"
        return "neutral"

    def is_accelerating(self, previous_roc: float) -> bool:
        """Check if momentum is accelerating."""
        return abs(self.value) > abs(previous_roc)


@dataclass(frozen=True)
class TimeSeries(Generic[T]):
    """Immutable time series container for historical indicator data."""

    data: Sequence[T]
    symbol: str
    interval: str

    def latest(self) -> T | None:
        """Get the most recent data point."""
        return self.data[-1] if self.data else None

    def get_last_n(self, n: int) -> Sequence[T]:
        """Get the last n data points."""
        return self.data[-n:] if n > 0 else []

    def filter_by_time(self, start: datetime, end: datetime) -> TimeSeries[T]:
        """Filter data points by time range."""
        filtered = [
            d
            for d in self.data
            if hasattr(d, "timestamp") and start <= d.timestamp <= end
        ]
        return TimeSeries(data=filtered, symbol=self.symbol, interval=self.interval)

    def is_empty(self) -> bool:
        """Check if time series has no data."""
        return len(self.data) == 0

    def size(self) -> int:
        """Get number of data points."""
        return len(self.data)


@dataclass(frozen=True)
class IndicatorConfig:
    """Configuration for indicator parameters."""

    # Moving Average parameters
    ma_period: int = 20
    ma_type: str = "SMA"  # SMA, EMA, WMA

    # RSI parameters
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # VuManchu parameters
    vumanchu_period: int = 9
    vumanchu_mult: float = 0.3

    # Stochastic parameters
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth_k: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0

    # ROC parameters
    roc_period: int = 12

    def with_ma_period(self, period: int) -> IndicatorConfig:
        """Create new config with updated MA period."""
        return IndicatorConfig(
            ma_period=period,
            ma_type=self.ma_type,
            rsi_period=self.rsi_period,
            rsi_overbought=self.rsi_overbought,
            rsi_oversold=self.rsi_oversold,
            macd_fast=self.macd_fast,
            macd_slow=self.macd_slow,
            macd_signal=self.macd_signal,
            bb_period=self.bb_period,
            bb_std_dev=self.bb_std_dev,
            vumanchu_period=self.vumanchu_period,
            vumanchu_mult=self.vumanchu_mult,
            stoch_k_period=self.stoch_k_period,
            stoch_d_period=self.stoch_d_period,
            stoch_smooth_k=self.stoch_smooth_k,
            stoch_overbought=self.stoch_overbought,
            stoch_oversold=self.stoch_oversold,
            roc_period=self.roc_period,
        )

    def with_rsi_levels(self, overbought: float, oversold: float) -> IndicatorConfig:
        """Create new config with updated RSI levels."""
        return IndicatorConfig(
            ma_period=self.ma_period,
            ma_type=self.ma_type,
            rsi_period=self.rsi_period,
            rsi_overbought=overbought,
            rsi_oversold=oversold,
            macd_fast=self.macd_fast,
            macd_slow=self.macd_slow,
            macd_signal=self.macd_signal,
            bb_period=self.bb_period,
            bb_std_dev=self.bb_std_dev,
            vumanchu_period=self.vumanchu_period,
            vumanchu_mult=self.vumanchu_mult,
            stoch_k_period=self.stoch_k_period,
            stoch_d_period=self.stoch_d_period,
            stoch_smooth_k=self.stoch_smooth_k,
            stoch_overbought=self.stoch_overbought,
            stoch_oversold=self.stoch_oversold,
            roc_period=self.roc_period,
        )


# Type aliases for common indicator combinations
IndicatorSet = tuple[
    MovingAverageResult | None,
    RSIResult | None,
    MACDResult | None,
    BollingerBandsResult | None,
    VuManchuResult | None,
]

IndicatorHistory = dict[str, TimeSeries[IndicatorResult]]
