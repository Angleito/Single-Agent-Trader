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
class DiamondPattern(IndicatorResult):
    """Diamond pattern signal from VuManchu Cipher A."""

    pattern_type: str  # "red_diamond", "green_diamond", "dump_diamond", "moon_diamond"
    wt1_cross_condition: bool  # WaveTrend 1 cross condition
    wt2_cross_condition: bool  # WaveTrend 2 cross condition
    strength: float  # Pattern strength (0.0-1.0)
    overbought_level: float
    oversold_level: float

    def is_bullish(self) -> bool:
        """Check if diamond pattern is bullish."""
        return self.pattern_type in ["green_diamond", "moon_diamond"]

    def is_bearish(self) -> bool:
        """Check if diamond pattern is bearish."""
        return self.pattern_type in ["red_diamond", "dump_diamond"]

    def is_extreme(self) -> bool:
        """Check if pattern is extreme (dump/moon)."""
        return self.pattern_type in ["dump_diamond", "moon_diamond"]

    def confluence_score(self) -> float:
        """Calculate confluence score based on conditions."""
        score = self.strength
        if self.wt1_cross_condition:
            score += 0.3
        if self.wt2_cross_condition:
            score += 0.3
        return min(score, 1.0)


@dataclass(frozen=True)
class YellowCrossSignal(IndicatorResult):
    """Yellow cross signal from VuManchu Cipher A."""

    direction: str  # "up" or "down"
    has_diamond: bool  # Required diamond pattern present
    wt2_in_range: bool  # WT2 within required range
    rsi_in_range: bool  # RSI within required range
    rsimfi_condition: bool  # Optional RSI+MFI condition
    confidence: float  # Signal confidence (0.0-1.0)
    wt2_value: float
    rsi_value: float
    rsimfi_value: float | None = None

    def is_bullish(self) -> bool:
        """Check if yellow cross is bullish."""
        return self.direction == "up"

    def is_bearish(self) -> bool:
        """Check if yellow cross is bearish."""
        return self.direction == "down"

    def is_high_confidence(self) -> bool:
        """Check if signal has high confidence."""
        return self.confidence >= 0.8

    def all_conditions_met(self) -> bool:
        """Check if all required conditions are met."""
        return (
            self.has_diamond
            and self.wt2_in_range
            and self.rsi_in_range
        )


@dataclass(frozen=True)
class CandlePattern(IndicatorResult):
    """Candle pattern signal from VuManchu Cipher A."""

    pattern_type: str  # "bull_candle" or "bear_candle"
    ema_conditions_met: bool  # EMA position requirements
    candle_conditions_met: bool  # Candle shape requirements
    diamond_filter_passed: bool  # Diamond pattern filter
    strength: float  # Pattern strength (0.0-1.0)
    ema2_value: float
    ema8_value: float
    open_price: float
    close_price: float

    def is_bullish(self) -> bool:
        """Check if candle pattern is bullish."""
        return self.pattern_type == "bull_candle"

    def is_bearish(self) -> bool:
        """Check if candle pattern is bearish."""
        return self.pattern_type == "bear_candle"

    def is_valid(self) -> bool:
        """Check if pattern meets all requirements."""
        return (
            self.ema_conditions_met
            and self.candle_conditions_met
            and self.diamond_filter_passed
        )

    def trend_alignment(self) -> bool:
        """Check if pattern aligns with EMA trend."""
        if self.is_bullish():
            return self.open_price > self.ema2_value and self.open_price > self.ema8_value
        else:
            return self.open_price < self.ema2_value and self.open_price < self.ema8_value


@dataclass(frozen=True)
class DivergencePattern(IndicatorResult):
    """Divergence pattern between price and indicators."""

    divergence_type: str  # "bullish", "bearish", "hidden_bullish", "hidden_bearish"
    price_trend: str  # "higher_high", "lower_low", etc.
    indicator_trend: str  # Trend in the indicator
    strength: float  # Divergence strength (0.0-1.0)
    lookback_periods: int
    price_slope: float
    indicator_slope: float

    def is_bullish(self) -> bool:
        """Check if divergence is bullish."""
        return "bullish" in self.divergence_type

    def is_bearish(self) -> bool:
        """Check if divergence is bearish."""
        return "bearish" in self.divergence_type

    def is_hidden(self) -> bool:
        """Check if divergence is hidden (continuation pattern)."""
        return "hidden" in self.divergence_type

    def is_regular(self) -> bool:
        """Check if divergence is regular (reversal pattern)."""
        return "hidden" not in self.divergence_type

    def slope_divergence_magnitude(self) -> float:
        """Calculate magnitude of slope divergence."""
        return abs(self.price_slope - self.indicator_slope)


@dataclass(frozen=True)
class CompositeSignal(IndicatorResult):
    """Composite signal combining multiple indicators."""

    signal_direction: int  # -1 (bearish), 0 (neutral), 1 (bullish)
    components: dict[str, IndicatorResult]  # Individual component signals
    confidence: float  # Overall confidence (0.0-1.0)
    strength: float  # Signal strength (0.0-1.0)
    agreement_score: float  # How well components agree (0.0-1.0)
    dominant_component: str  # Which component drives the signal

    def is_bullish(self) -> bool:
        """Check if composite signal is bullish."""
        return self.signal_direction > 0

    def is_bearish(self) -> bool:
        """Check if composite signal is bearish."""
        return self.signal_direction < 0

    def is_neutral(self) -> bool:
        """Check if composite signal is neutral."""
        return self.signal_direction == 0

    def is_high_quality(self) -> bool:
        """Check if signal meets high quality criteria."""
        return (
            self.confidence >= 0.7
            and self.strength >= 0.6
            and self.agreement_score >= 0.8
        )

    def component_count(self) -> int:
        """Get number of contributing components."""
        return len(self.components)

    def bullish_components(self) -> list[str]:
        """Get list of bullish component names."""
        bullish = []
        for name, signal in self.components.items():
            if hasattr(signal, 'is_bullish') and signal.is_bullish():
                bullish.append(name)
        return bullish

    def bearish_components(self) -> list[str]:
        """Get list of bearish component names."""
        bearish = []
        for name, signal in self.components.items():
            if hasattr(signal, 'is_bearish') and signal.is_bearish():
                bearish.append(name)
        return bearish


@dataclass(frozen=True)
class VolumeProfile(IndicatorResult):
    """Volume profile analysis result."""

    poc_price: float  # Point of Control (highest volume price)
    vah_price: float  # Value Area High
    val_price: float  # Value Area Low
    volume_weighted_price: float  # VWAP-style calculation
    volume_distribution: dict[float, float]  # Price -> Volume mapping
    value_area_percentage: float  # Percentage of volume in value area

    def is_price_in_value_area(self, price: float) -> bool:
        """Check if price is within value area."""
        return self.val_price <= price <= self.vah_price

    def is_price_above_poc(self, price: float) -> bool:
        """Check if price is above Point of Control."""
        return price > self.poc_price

    def is_price_below_poc(self, price: float) -> bool:
        """Check if price is below Point of Control."""
        return price < self.poc_price

    def distance_from_poc(self, price: float) -> float:
        """Calculate percentage distance from POC."""
        return ((price - self.poc_price) / self.poc_price) * 100


@dataclass(frozen=True)
class MarketStructure(IndicatorResult):
    """Market structure analysis result."""

    trend_direction: str  # "uptrend", "downtrend", "sideways"
    swing_highs: list[float]  # Recent swing high prices
    swing_lows: list[float]  # Recent swing low prices
    support_levels: list[float]  # Identified support levels
    resistance_levels: list[float]  # Identified resistance levels
    trend_strength: float  # Trend strength (0.0-1.0)
    volatility_regime: str  # "low", "medium", "high"

    def is_bullish_structure(self) -> bool:
        """Check if market structure is bullish."""
        return self.trend_direction == "uptrend"

    def is_bearish_structure(self) -> bool:
        """Check if market structure is bearish."""
        return self.trend_direction == "downtrend"

    def is_sideways(self) -> bool:
        """Check if market is in sideways structure."""
        return self.trend_direction == "sideways"

    def is_strong_trend(self) -> bool:
        """Check if trend is strong."""
        return self.trend_strength >= 0.7

    def nearest_support(self, price: float) -> float | None:
        """Find nearest support level below price."""
        supports_below = [s for s in self.support_levels if s < price]
        return max(supports_below) if supports_below else None

    def nearest_resistance(self, price: float) -> float | None:
        """Find nearest resistance level above price."""
        resistances_above = [r for r in self.resistance_levels if r > price]
        return min(resistances_above) if resistances_above else None


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


# Enhanced VuManchu Signal Set
@dataclass(frozen=True)
class VuManchuSignalSet(IndicatorResult):
    """Complete VuManchu signal set with all components."""

    # Core VuManchu components
    vumanchu_result: VuManchuResult
    diamond_patterns: list[DiamondPattern]
    yellow_cross_signals: list[YellowCrossSignal]
    candle_patterns: list[CandlePattern]
    divergence_patterns: list[DivergencePattern]
    
    # Technical indicators
    rsi_result: RSIResult | None = None
    macd_result: MACDResult | None = None
    bollinger_result: BollingerBandsResult | None = None
    
    # Market context
    volume_profile: VolumeProfile | None = None
    market_structure: MarketStructure | None = None
    
    # Composite analysis
    composite_signal: CompositeSignal | None = None

    def get_active_patterns(self) -> list[str]:
        """Get list of currently active pattern types."""
        active = []
        
        if self.diamond_patterns:
            active.extend([p.pattern_type for p in self.diamond_patterns])
        
        if self.yellow_cross_signals:
            active.extend([f"yellow_cross_{s.direction}" for s in self.yellow_cross_signals])
        
        if self.candle_patterns:
            active.extend([p.pattern_type for p in self.candle_patterns])
        
        if self.divergence_patterns:
            active.extend([p.divergence_type for p in self.divergence_patterns])
        
        return active

    def get_bullish_signals(self) -> list[IndicatorResult]:
        """Get all bullish signals."""
        bullish = []
        
        if self.vumanchu_result.signal == "LONG":
            bullish.append(self.vumanchu_result)
        
        bullish.extend([p for p in self.diamond_patterns if p.is_bullish()])
        bullish.extend([s for s in self.yellow_cross_signals if s.is_bullish()])
        bullish.extend([p for p in self.candle_patterns if p.is_bullish()])
        bullish.extend([d for d in self.divergence_patterns if d.is_bullish()])
        
        return bullish

    def get_bearish_signals(self) -> list[IndicatorResult]:
        """Get all bearish signals."""
        bearish = []
        
        if self.vumanchu_result.signal == "SHORT":
            bearish.append(self.vumanchu_result)
        
        bearish.extend([p for p in self.diamond_patterns if p.is_bearish()])
        bearish.extend([s for s in self.yellow_cross_signals if s.is_bearish()])
        bearish.extend([p for p in self.candle_patterns if p.is_bearish()])
        bearish.extend([d for d in self.divergence_patterns if d.is_bearish()])
        
        return bearish

    def signal_confluence_score(self) -> float:
        """Calculate signal confluence score."""
        bullish_count = len(self.get_bullish_signals())
        bearish_count = len(self.get_bearish_signals())
        total_signals = bullish_count + bearish_count
        
        if total_signals == 0:
            return 0.0
        
        # Higher score when signals agree
        max_direction = max(bullish_count, bearish_count)
        return max_direction / total_signals

    def overall_direction(self) -> str:
        """Get overall signal direction."""
        bullish_count = len(self.get_bullish_signals())
        bearish_count = len(self.get_bearish_signals())
        
        if bullish_count > bearish_count:
            return "BULLISH"
        elif bearish_count > bullish_count:
            return "BEARISH"
        else:
            return "NEUTRAL"


# Type aliases for common indicator combinations
IndicatorSet = tuple[
    MovingAverageResult | None,
    RSIResult | None,
    MACDResult | None,
    BollingerBandsResult | None,
    VuManchuResult | None,
]

# VuManChu State alias for backward compatibility
VuManchuState = VuManchuSignalSet

# Enhanced type aliases
VuManchuSet = tuple[
    VuManchuResult,
    list[DiamondPattern],
    list[YellowCrossSignal],
    list[CandlePattern],
    list[DivergencePattern],
]

SignalSet = tuple[
    VuManchuSignalSet,
    CompositeSignal | None,
    MarketStructure | None,
]

IndicatorHistory = dict[str, TimeSeries[IndicatorResult]]
SignalHistory = dict[str, TimeSeries[VuManchuSignalSet]]
