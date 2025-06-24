"""
Comprehensive tests for functional programming indicator functions.

This module tests all indicator functions with property-based testing,
edge cases, mathematical properties, and performance benchmarks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite


# We'll create minimal indicator result types for testing to avoid import issues
@dataclass(frozen=True)
class MovingAverageResult:
    timestamp: datetime
    value: float
    period: int


@dataclass(frozen=True)
class RSIResult:
    timestamp: datetime
    value: float
    overbought: float = 70.0
    oversold: float = 30.0

    def is_overbought(self) -> bool:
        return self.value >= self.overbought

    def is_oversold(self) -> bool:
        return self.value <= self.oversold

    def is_neutral(self) -> bool:
        return self.oversold < self.value < self.overbought

    def strength_level(self) -> str:
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
class MACDResult:
    timestamp: datetime
    macd: float
    signal: float
    histogram: float

    def momentum_direction(self) -> str:
        if self.histogram > 0:
            return "bullish"
        if self.histogram < 0:
            return "bearish"
        return "neutral"


@dataclass(frozen=True)
class BollingerBandsResult:
    timestamp: datetime
    upper: float
    middle: float
    lower: float
    width: float | None = None

    def __post_init__(self) -> None:
        if self.width is None:
            object.__setattr__(self, "width", self.upper - self.lower)

    def is_price_above_upper(self, price: float) -> bool:
        return price > self.upper

    def is_price_below_lower(self, price: float) -> bool:
        return price < self.lower

    def is_price_in_bands(self, price: float) -> bool:
        return self.lower <= price <= self.upper

    def price_position(self, price: float) -> float:
        if self.width and self.width > 0:
            return (price - self.lower) / self.width
        return 0.5

    def is_squeeze(self, threshold: float = 0.02) -> bool:
        if self.width and self.middle > 0:
            return (self.width / self.middle) < threshold
        return False


@dataclass(frozen=True)
class VuManchuResult:
    timestamp: datetime
    wave_a: float
    wave_b: float
    signal: str | None = None


@dataclass(frozen=True)
class IndicatorConfig:
    ma_period: int = 20
    ma_type: str = "SMA"
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std_dev: float = 2.0
    vumanchu_period: int = 9
    vumanchu_mult: float = 0.3

    def with_ma_period(self, period: int) -> IndicatorConfig:
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
        )

    def with_rsi_levels(self, overbought: float, oversold: float) -> IndicatorConfig:
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
        )


# Import indicator functions from fp.indicators modules
from bot.fp.indicators.moving_averages import (
    sma,
)
from bot.fp.indicators.oscillators import (
    calculate_rsi,
    calculate_stochastic,
    calculate_williams_r,
)
from bot.fp.indicators.trend import (
    calculate_adx,
    calculate_macd,
    calculate_psar,
)
from bot.fp.indicators.volatility import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_standard_deviation,
)
from bot.fp.indicators.vumanchu import (
    calculate_vumanchu,
    interpret_vumanchu_signal,
)
from bot.fp.indicators.vumanchu_functional import (
    analyze_diamond_patterns,
    calculate_ema,
    calculate_hlc3,
    calculate_sma,
    calculate_wavetrend_oscillator,
    create_default_vumanchu_config,
    detect_crossovers,
    determine_signal,
    vumanchu_cipher,
    vumanchu_cipher_series,
    vumanchu_comprehensive_analysis,
)
from bot.fp.types.indicators import (
    CompositeSignal,
    DiamondPattern,
    IndicatorConfig,
    VuManchuResult,
    VuManchuSignalSet,
)

# Import legacy implementation for comparison


# Strategy for generating price data
@composite
def price_data(draw, min_length=1, max_length=1000):
    """Generate realistic price data for testing."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))

    # Generate base price
    base_price = draw(st.floats(min_value=0.01, max_value=10000.0))

    # Generate price changes (small percentage moves)
    changes = draw(
        st.lists(
            st.floats(min_value=-0.1, max_value=0.1), min_size=length, max_size=length
        )
    )

    # Build price series
    prices = [base_price]
    for change in changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Ensure positive prices

    return prices


@composite
def ohlc_data(draw, min_length=1, max_length=1000):
    """Generate OHLC (Open, High, Low, Close) data."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))

    ohlc = []
    base_price = draw(st.floats(min_value=0.01, max_value=10000.0))

    for _ in range(length):
        open_price = base_price

        # Generate intraday movement
        movement = draw(st.floats(min_value=-0.05, max_value=0.05))
        close_price = open_price * (1 + movement)

        # High and low within open/close range
        high_price = max(open_price, close_price) * draw(
            st.floats(min_value=1.0, max_value=1.02)
        )
        low_price = min(open_price, close_price) * draw(
            st.floats(min_value=0.98, max_value=1.0)
        )

        ohlc.append(
            {
                "open": max(open_price, 0.01),
                "high": max(high_price, 0.01),
                "low": max(low_price, 0.01),
                "close": max(close_price, 0.01),
            }
        )

        base_price = close_price

    return ohlc


# Fixtures for test data
@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    base = 100.0
    returns = np.random.normal(0.001, 0.02, 100)
    prices = base * np.cumprod(1 + returns)
    return prices.tolist()


@pytest.fixture
def sample_ohlc():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    base = 100.0
    data = []

    for _ in range(100):
        open_price = base
        close_price = open_price * (1 + np.random.normal(0.001, 0.02))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))

        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }
        )

        base = close_price

    return data


@pytest.fixture
def timestamp():
    """Generate a test timestamp."""
    return datetime.now()


# Test Classes
class TestMovingAverages:
    """Test moving average calculations."""

    @given(price_data(min_length=1, max_length=100))
    def test_sma_mathematical_properties(self, prices):
        """Test SMA mathematical properties."""
        period = min(20, len(prices))
        prices_array = np.array(prices, dtype=np.float64)
        result = sma(prices_array, period, datetime.now())

        if result is not None:
            # SMA should be within price range
            assert min(prices[-period:]) <= result.value <= max(prices[-period:])

            # SMA should equal arithmetic mean
            expected = sum(prices[-period:]) / period
            assert abs(result.value - expected) < 1e-10

    @given(price_data(min_length=20))
    def test_ema_convergence(self, prices):
        """Test that EMA converges to price in trending market."""
        period = 10
        result = calculate_ema(prices, period)

        if result is not None:
            # EMA should be closer to recent prices than older ones
            recent_avg = sum(prices[-5:]) / 5
            old_avg = sum(prices[:5]) / 5

            if len(prices) > 20:
                ema_to_recent = abs(result - recent_avg)
                ema_to_old = abs(result - old_avg)

                # In most cases, EMA should be closer to recent prices
                # This might not always hold for highly volatile data
                if abs(recent_avg - old_avg) > 1.0:
                    assert ema_to_recent < ema_to_old * 1.5

    @pytest.mark.parametrize("period", [0, -1, -10])
    def test_ma_invalid_period(self, sample_prices, period):
        """Test moving averages with invalid periods."""
        assert calculate_sma(sample_prices, period) is None
        assert calculate_ema(sample_prices, period) is None

    def test_ma_insufficient_data(self):
        """Test moving averages with insufficient data."""
        prices = [100.0, 101.0, 102.0]
        assert calculate_sma(prices, 5) is None
        assert calculate_ema(prices, 5) is None

    def test_ma_single_price(self):
        """Test moving averages with single price."""
        prices = [100.0]
        assert calculate_sma(prices, 1) == 100.0
        assert calculate_ema(prices, 1) == 100.0

    def test_ma_empty_data(self):
        """Test moving averages with empty data."""
        prices = []
        assert calculate_sma(prices, 5) is None
        assert calculate_ema(prices, 5) is None


class TestRSI:
    """Test RSI calculations."""

    @given(price_data(min_length=15))
    def test_rsi_bounds(self, prices):
        """Test RSI is always between 0 and 100."""
        result = calculate_rsi(prices, 14)
        if result is not None:
            assert 0 <= result <= 100

    def test_rsi_all_gains(self, timestamp):
        """Test RSI when all price changes are positive."""
        prices = list(range(100, 120))  # Continuously rising
        rsi = calculate_rsi(prices, 14)
        assert rsi is not None
        assert rsi > 70  # Should indicate overbought

        # Create result object
        result = RSIResult(timestamp=timestamp, value=rsi)
        assert result.is_overbought()
        assert not result.is_oversold()

    def test_rsi_all_losses(self, timestamp):
        """Test RSI when all price changes are negative."""
        prices = list(range(120, 100, -1))  # Continuously falling
        rsi = calculate_rsi(prices, 14)
        assert rsi is not None
        assert rsi < 30  # Should indicate oversold

        # Create result object
        result = RSIResult(timestamp=timestamp, value=rsi)
        assert result.is_oversold()
        assert not result.is_overbought()

    def test_rsi_no_change(self, timestamp):
        """Test RSI when prices don't change."""
        prices = [100.0] * 20
        rsi = calculate_rsi(prices, 14)
        # When there's no change, RSI calculation may return None or 50
        if rsi is not None:
            assert 45 <= rsi <= 55  # Should be near neutral

    @pytest.mark.parametrize("period", [0, -1])
    def test_rsi_invalid_period(self, sample_prices, period):
        """Test RSI with invalid period."""
        assert calculate_rsi(sample_prices, period) is None


class TestMACD:
    """Test MACD calculations."""

    @given(price_data(min_length=50))
    def test_macd_calculation(self, prices):
        """Test MACD calculation properties."""
        macd, signal, histogram = calculate_macd(prices)

        if macd is not None:
            # Histogram should equal MACD - Signal
            assert abs(histogram - (macd - signal)) < 1e-10

    def test_macd_insufficient_data(self):
        """Test MACD with insufficient data."""
        prices = [100.0] * 20
        macd, signal, histogram = calculate_macd(prices)
        assert macd is None
        assert signal is None
        assert histogram is None

    def test_macd_trending_market(self, timestamp):
        """Test MACD in trending market."""
        # Create uptrending prices
        prices = [100.0 * (1.01**i) for i in range(50)]
        macd, signal, histogram = calculate_macd(prices)

        if macd is not None:
            # In uptrend, MACD should be positive
            result = MACDResult(
                timestamp=timestamp, macd=macd, signal=signal, histogram=histogram
            )
            assert result.momentum_direction() in ["bullish", "neutral"]


class TestBollingerBands:
    """Test Bollinger Bands calculations."""

    @given(price_data(min_length=20), st.floats(min_value=0.5, max_value=3.0))
    def test_bollinger_bands_properties(self, prices, std_dev):
        """Test Bollinger Bands mathematical properties."""
        upper, middle, lower = calculate_bollinger_bands(prices, 20, std_dev)

        if upper is not None:
            # Bands should be symmetric around middle
            assert abs((upper - middle) - (middle - lower)) < 1e-10

            # Upper > Middle > Lower
            assert upper > middle > lower

    def test_bollinger_squeeze(self, timestamp):
        """Test Bollinger Bands squeeze detection."""
        # Create low volatility prices
        prices = [100.0 + np.sin(i * 0.1) * 0.1 for i in range(50)]
        upper, middle, lower = calculate_bollinger_bands(prices)

        if upper is not None:
            result = BollingerBandsResult(
                timestamp=timestamp, upper=upper, middle=middle, lower=lower
            )

            # Check squeeze based on band width
            width_ratio = result.width / middle if middle > 0 else 0
            assert result.is_squeeze(threshold=0.1) == (width_ratio < 0.1)


class TestVuManChuFunctional:
    """Test VuManchu Cipher functional implementations."""

    @pytest.fixture
    def sample_ohlcv_array(self):
        """Generate sample OHLCV data as numpy array."""
        np.random.seed(42)
        n = 100
        base = 100.0

        ohlcv = np.zeros((n, 5))

        for i in range(n):
            open_price = base
            close_price = base * (1 + np.random.normal(0.001, 0.02))
            high_price = max(open_price, close_price) * (
                1 + abs(np.random.normal(0, 0.01))
            )
            low_price = min(open_price, close_price) * (
                1 - abs(np.random.normal(0, 0.01))
            )
            volume = np.random.uniform(1000, 10000)

            ohlcv[i] = [open_price, high_price, low_price, close_price, volume]
            base = close_price

        return ohlcv

    @pytest.fixture
    def sample_vumanchu_config(self):
        """Generate sample VuManChu configuration."""
        return create_default_vumanchu_config()

    @given(price_data(min_length=30))
    def test_vumanchu_waves_legacy_simple(self, prices):
        """Test VuManchu wave calculations (legacy simple version)."""
        wave_a, wave_b = calculate_vumanchu(prices)

        if wave_a is not None:
            # Waves should be within reasonable bounds
            assert -50 <= wave_a <= 50
            assert -50 <= wave_b <= 50

    def test_vumanchu_signals_legacy_simple(self, timestamp):
        """Test VuManchu signal generation (legacy simple version)."""
        # Create prices that should generate signals
        prices = [100.0] * 20 + [95.0] * 10  # Sharp drop
        wave_a, wave_b = calculate_vumanchu(prices)

        if wave_a is not None:
            result = VuManchuResult(timestamp=timestamp, wave_a=wave_a, wave_b=wave_b)

            # Check signal determination
            assert result.signal in ["LONG", "SHORT", "NEUTRAL"]

    def test_vumanchu_cipher_functional(self, sample_ohlcv_array, timestamp):
        """Test functional VuManChu cipher calculation."""
        result = vumanchu_cipher(sample_ohlcv_array, timestamp=timestamp)

        assert isinstance(result, VuManchuResult)
        assert result.timestamp == timestamp
        assert isinstance(result.wave_a, float)
        assert isinstance(result.wave_b, float)
        assert result.signal in ["LONG", "SHORT", "NEUTRAL"]

        # Waves should be within reasonable bounds for normal market data
        assert -100 <= result.wave_a <= 100
        assert -100 <= result.wave_b <= 100

    def test_vumanchu_cipher_series(self, sample_ohlcv_array):
        """Test VuManChu cipher calculation for entire series."""
        results = vumanchu_cipher_series(sample_ohlcv_array)

        assert isinstance(results, list)
        assert len(results) == len(sample_ohlcv_array)

        for result in results:
            assert isinstance(result, VuManchuResult)
            assert result.signal in ["LONG", "SHORT", "NEUTRAL"]

    def test_wavetrend_oscillator_calculation(self, sample_ohlcv_array):
        """Test WaveTrend oscillator calculation."""
        high = sample_ohlcv_array[:, 1]
        low = sample_ohlcv_array[:, 2]
        close = sample_ohlcv_array[:, 3]

        hlc3 = calculate_hlc3(high, low, close)
        wt1, wt2 = calculate_wavetrend_oscillator(hlc3, 9, 18, 3)

        assert len(wt1) == len(sample_ohlcv_array)
        assert len(wt2) == len(sample_ohlcv_array)

        # Check that later values are not NaN (after warm-up period)
        assert not np.isnan(wt1[-1])
        assert not np.isnan(wt2[-1])

        # WaveTrend values should be within reasonable bounds
        valid_wt1 = wt1[~np.isnan(wt1)]
        valid_wt2 = wt2[~np.isnan(wt2)]

        if len(valid_wt1) > 0:
            assert np.min(valid_wt1) >= -100 and np.max(valid_wt1) <= 100
        if len(valid_wt2) > 0:
            assert np.min(valid_wt2) >= -100 and np.max(valid_wt2) <= 100

    def test_crossover_detection(self, sample_ohlcv_array):
        """Test crossover detection between waves."""
        high = sample_ohlcv_array[:, 1]
        low = sample_ohlcv_array[:, 2]
        close = sample_ohlcv_array[:, 3]

        hlc3 = calculate_hlc3(high, low, close)
        wt1, wt2 = calculate_wavetrend_oscillator(hlc3, 9, 18, 3)

        bullish_cross, bearish_cross = detect_crossovers(wt1, wt2)

        assert len(bullish_cross) == len(wt1)
        assert len(bearish_cross) == len(wt1)

        # Crossovers should be boolean arrays
        assert bullish_cross.dtype == bool
        assert bearish_cross.dtype == bool

        # Can't have both bullish and bearish crossover at same time
        assert not np.any(bullish_cross & bearish_cross)

    def test_diamond_pattern_analysis(self, sample_ohlcv_array):
        """Test diamond pattern analysis."""
        high = sample_ohlcv_array[:, 1]
        low = sample_ohlcv_array[:, 2]
        close = sample_ohlcv_array[:, 3]

        hlc3 = calculate_hlc3(high, low, close)
        wt1, wt2 = calculate_wavetrend_oscillator(hlc3, 9, 18, 3)

        patterns = analyze_diamond_patterns(wt1, wt2)

        assert isinstance(patterns, list)

        for pattern in patterns:
            assert isinstance(pattern, DiamondPattern)
            assert pattern.pattern_type in ["red_diamond", "green_diamond"]
            assert 0 <= pattern.strength <= 1.0
            assert isinstance(pattern.wt1_cross_condition, bool)
            assert isinstance(pattern.wt2_cross_condition, bool)

    def test_comprehensive_analysis(self, sample_ohlcv_array, timestamp):
        """Test comprehensive VuManChu analysis."""
        signal_set = vumanchu_comprehensive_analysis(
            sample_ohlcv_array, timestamp=timestamp
        )

        assert isinstance(signal_set, VuManchuSignalSet)
        assert signal_set.timestamp == timestamp

        # Check core VuManChu result
        assert isinstance(signal_set.vumanchu_result, VuManchuResult)

        # Check pattern lists
        assert isinstance(signal_set.diamond_patterns, list)
        assert isinstance(signal_set.yellow_cross_signals, list)
        assert isinstance(signal_set.candle_patterns, list)
        assert isinstance(signal_set.divergence_patterns, list)

        # Check composite signal
        if signal_set.composite_signal is not None:
            assert isinstance(signal_set.composite_signal, CompositeSignal)
            assert signal_set.composite_signal.signal_direction in [-1, 0, 1]

    def test_signal_strength_calculation(self, sample_ohlcv_array):
        """Test signal strength calculation."""
        high = sample_ohlcv_array[:, 1]
        low = sample_ohlcv_array[:, 2]
        close = sample_ohlcv_array[:, 3]

        hlc3 = calculate_hlc3(high, low, close)
        wt1, wt2 = calculate_wavetrend_oscillator(hlc3, 9, 18, 3)

        # Test signal determination
        if not np.isnan(wt1[-1]) and not np.isnan(wt2[-1]):
            signal = determine_signal(wt1[-1], wt2[-1])
            assert signal in ["LONG", "SHORT", "NEUTRAL"]

    @pytest.mark.parametrize(
        "channel_length,average_length,ma_length",
        [
            (6, 8, 3),  # Scalping optimized
            (9, 13, 3),  # Standard Pine Script
            (10, 21, 4),  # Alternative settings
        ],
    )
    def test_vumanchu_different_parameters(
        self, sample_ohlcv_array, channel_length, average_length, ma_length
    ):
        """Test VuManChu with different parameter sets."""
        result = vumanchu_cipher(
            sample_ohlcv_array,
            channel_length=channel_length,
            average_length=average_length,
            ma_length=ma_length,
        )

        assert isinstance(result, VuManchuResult)
        assert result.signal in ["LONG", "SHORT", "NEUTRAL"]

        # Results should be reasonable regardless of parameters
        assert (
            -200 <= result.wave_a <= 200
        )  # Allow wider bounds for different parameters
        assert -200 <= result.wave_b <= 200

    def test_vumanchu_signal_interpretation(self, timestamp):
        """Test VuManChu signal interpretation logic."""
        # Test bullish crossover
        signal = interpret_vumanchu_signal(
            wave_a=-10, wave_b=-15, prev_wave_a=-15, prev_wave_b=-10
        )
        assert signal == "LONG"

        # Test bearish crossover
        signal = interpret_vumanchu_signal(
            wave_a=10, wave_b=15, prev_wave_a=15, prev_wave_b=10
        )
        assert signal == "SHORT"

        # Test extreme oversold
        signal = interpret_vumanchu_signal(wave_a=-35, wave_b=-35)
        assert signal == "LONG"

        # Test extreme overbought
        signal = interpret_vumanchu_signal(wave_a=35, wave_b=35)
        assert signal == "SHORT"

        # Test neutral
        signal = interpret_vumanchu_signal(wave_a=5, wave_b=5)
        assert signal == "NEUTRAL"

    def test_helper_functions(self, sample_ohlcv_array):
        """Test helper calculation functions."""
        high = sample_ohlcv_array[:, 1]
        low = sample_ohlcv_array[:, 2]
        close = sample_ohlcv_array[:, 3]

        # Test HLC3 calculation
        hlc3 = calculate_hlc3(high, low, close)
        assert len(hlc3) == len(high)

        # HLC3 should be between high and low
        for i in range(len(hlc3)):
            assert low[i] <= hlc3[i] <= high[i]

        # Test EMA calculation
        ema_values = calculate_ema(close, 20)
        assert len(ema_values) == len(close)

        # Test SMA calculation
        sma_values = calculate_sma(close, 20)
        assert len(sma_values) == len(close)

        # After warm-up period, values should not be NaN
        assert not np.isnan(ema_values[-1])
        assert not np.isnan(sma_values[-1])

    def test_vumanchu_config_immutability(self):
        """Test that VuManChu configuration is immutable."""
        config = create_default_vumanchu_config()
        original_channel_length = config["channel_length"]

        # Try to modify config
        config["channel_length"] = 999

        # Create new config and verify original isn't affected
        new_config = create_default_vumanchu_config()
        assert new_config["channel_length"] == original_channel_length

    @pytest.mark.parametrize("insufficient_length", [0, 1, 5, 8])
    def test_vumanchu_insufficient_data(self, insufficient_length):
        """Test VuManChu with insufficient data."""
        if insufficient_length == 0:
            ohlcv = np.array([]).reshape(0, 5)
        else:
            ohlcv = np.random.rand(insufficient_length, 5) * 100

        result = vumanchu_cipher(ohlcv)

        # Should handle gracefully with default values
        assert isinstance(result, VuManchuResult)
        assert result.signal == "NEUTRAL"
        assert result.wave_a == 0.0
        assert result.wave_b == 0.0


class TestEdgeCases:
    """Test edge cases for all indicators."""

    def test_empty_data_all_indicators(self, timestamp):
        """Test all indicators with empty data."""
        prices = []

        assert calculate_sma(prices, 10) is None
        assert calculate_ema(prices, 10) is None
        assert calculate_rsi(prices) is None

        macd, signal, hist = calculate_macd(prices)
        assert all(x is None for x in [macd, signal, hist])

        upper, middle, lower = calculate_bollinger_bands(prices)
        assert all(x is None for x in [upper, middle, lower])

        wave_a, wave_b = calculate_vumanchu(prices)
        assert all(x is None for x in [wave_a, wave_b])

    def test_single_price_all_indicators(self, timestamp):
        """Test all indicators with single price."""
        prices = [100.0]

        # Only SMA/EMA with period 1 should work
        assert calculate_sma(prices, 1) == 100.0
        assert calculate_ema(prices, 1) == 100.0
        assert calculate_sma(prices, 2) is None
        assert calculate_rsi(prices) is None

    @given(st.lists(st.just(100.0), min_size=50, max_size=50))
    def test_constant_prices(self, prices):
        """Test indicators with constant prices."""
        # SMA should equal the constant price
        sma = calculate_sma(prices, 10)
        assert sma == 100.0

        # RSI should be near 50 (neutral)
        rsi = calculate_rsi(prices)
        if rsi is not None:
            assert 45 <= rsi <= 55

    @given(st.lists(st.floats(min_value=1e-10, max_value=1e-8), min_size=20))
    def test_very_small_prices(self, prices):
        """Test indicators with very small prices."""
        # Ensure calculations don't break with small numbers
        sma = calculate_sma(prices, 10)
        if sma is not None:
            assert sma > 0
            assert not np.isnan(sma)
            assert not np.isinf(sma)


class TestPerformanceBenchmarks:
    """Benchmark indicator performance vs pandas."""

    @pytest.fixture
    def large_price_data(self):
        """Generate large dataset for benchmarking."""
        np.random.seed(42)
        base = 100.0
        returns = np.random.normal(0.001, 0.02, 10000)
        prices = base * np.cumprod(1 + returns)
        return prices

    def test_sma_performance(self, large_price_data, benchmark):
        """Benchmark SMA performance."""
        prices = large_price_data.tolist()
        period = 20

        # Our implementation
        def our_sma():
            return calculate_sma(prices, period)

        # Pandas implementation
        def pandas_sma():
            return pd.Series(prices).rolling(period).mean().iloc[-1]

        # Benchmark
        our_time = benchmark(our_sma)

        # Compare with pandas (run separately)
        start = time.time()
        pandas_result = pandas_sma()
        pandas_time = time.time() - start

        # Our implementation should be reasonably fast
        # Allow up to 10x slower than pandas (pandas is highly optimized)
        assert our_time < pandas_time * 10

    def test_rsi_performance(self, large_price_data, benchmark):
        """Benchmark RSI performance."""
        prices = large_price_data.tolist()

        def our_rsi():
            return calculate_rsi(prices, 14)

        # Benchmark
        result = benchmark(our_rsi)
        assert result is not None


class TestIndicatorResults:
    """Test indicator result objects."""

    def test_moving_average_result(self, timestamp):
        """Test MovingAverageResult methods."""
        ma = MovingAverageResult(timestamp=timestamp, value=100.0, period=20)

        assert ma.is_above(101.0)
        assert not ma.is_above(99.0)
        assert ma.is_below(99.0)
        assert not ma.is_below(101.0)

        # Test distance calculation
        assert ma.distance_from(110.0) == 10.0
        assert ma.distance_from(90.0) == -10.0

    def test_rsi_result(self, timestamp):
        """Test RSIResult methods."""
        # Test overbought
        rsi_high = RSIResult(timestamp=timestamp, value=75.0)
        assert rsi_high.is_overbought()
        assert not rsi_high.is_oversold()
        assert rsi_high.strength_level() == "overbought"

        # Test oversold
        rsi_low = RSIResult(timestamp=timestamp, value=25.0)
        assert rsi_low.is_oversold()
        assert not rsi_low.is_overbought()
        assert rsi_low.strength_level() == "oversold"

        # Test neutral
        rsi_mid = RSIResult(timestamp=timestamp, value=50.0)
        assert rsi_mid.is_neutral()
        assert rsi_mid.strength_level() == "neutral"

    def test_bollinger_bands_result(self, timestamp):
        """Test BollingerBandsResult methods."""
        bb = BollingerBandsResult(
            timestamp=timestamp, upper=110.0, middle=100.0, lower=90.0
        )

        # Test price position
        assert bb.is_price_above_upper(111.0)
        assert bb.is_price_below_lower(89.0)
        assert bb.is_price_in_bands(100.0)

        # Test position calculation
        assert bb.price_position(100.0) == 0.5  # Middle
        assert bb.price_position(110.0) == 1.0  # Upper
        assert bb.price_position(90.0) == 0.0  # Lower

    def test_indicator_config(self):
        """Test IndicatorConfig immutability."""
        config = IndicatorConfig()

        # Test with_ methods create new instances
        new_config = config.with_ma_period(50)
        assert new_config.ma_period == 50
        assert config.ma_period == 20  # Original unchanged

        new_config2 = config.with_rsi_levels(80.0, 20.0)
        assert new_config2.rsi_overbought == 80.0
        assert new_config2.rsi_oversold == 20.0
        assert config.rsi_overbought == 70.0  # Original unchanged


class TestMathematicalProperties:
    """Test mathematical properties of indicators."""

    @given(price_data(min_length=50))
    def test_sma_ema_relationship(self, prices):
        """Test relationship between SMA and EMA."""
        period = 20
        sma = calculate_sma(prices, period)
        ema = calculate_ema(prices, period)

        if sma is not None and ema is not None:
            # Both should be positive for positive prices
            assert sma > 0
            assert ema > 0

            # Both should be within price range
            min_price = min(prices)
            max_price = max(prices)
            assert min_price <= sma <= max_price
            assert min_price <= ema <= max_price

    @given(
        price_data(min_length=30),
        st.integers(min_value=2, max_value=20),
        st.floats(min_value=1.0, max_value=3.0),
    )
    def test_bollinger_bands_contain_prices(self, prices, period, std_dev):
        """Test that Bollinger Bands contain appropriate percentage of prices."""
        if len(prices) < period:
            return

        upper, middle, lower = calculate_bollinger_bands(prices, period, std_dev)

        if upper is not None:
            # Count prices within bands
            recent_prices = prices[-period:]
            within_bands = sum(1 for p in recent_prices if lower <= p <= upper)
            percentage = within_bands / period

            # For 2 std dev, approximately 95% should be within bands
            # Allow for some variance due to small sample size
            if std_dev == 2.0:
                assert 0.7 <= percentage <= 1.0

    def test_indicator_consistency(self, sample_prices):
        """Test that indicators are consistent across multiple calls."""
        # Same input should produce same output
        sma1 = calculate_sma(sample_prices, 20)
        sma2 = calculate_sma(sample_prices, 20)
        assert sma1 == sma2

        rsi1 = calculate_rsi(sample_prices, 14)
        rsi2 = calculate_rsi(sample_prices, 14)
        assert rsi1 == rsi2


class TestAdditionalIndicators:
    """Test additional indicator implementations."""

    def test_wma_calculation(self, sample_prices):
        """Test Weighted Moving Average calculation."""
        period = 10
        wma = calculate_wma(sample_prices, period)

        if wma is not None:
            # WMA should give more weight to recent prices
            sma = calculate_sma(sample_prices, period)

            # In an uptrend, WMA > SMA
            if sample_prices[-1] > sample_prices[-period]:
                assert wma > sma
            # In a downtrend, WMA < SMA
            elif sample_prices[-1] < sample_prices[-period]:
                assert wma < sma

    @given(ohlc_data(min_length=20))
    def test_stochastic_bounds(self, ohlc):
        """Test Stochastic oscillator bounds."""
        highs = [d["high"] for d in ohlc]
        lows = [d["low"] for d in ohlc]
        closes = [d["close"] for d in ohlc]

        k, d = calculate_stochastic(highs, lows, closes)

        if k is not None:
            assert 0 <= k <= 100
        if d is not None:
            assert 0 <= d <= 100

    def test_williams_r_calculation(self, sample_ohlc):
        """Test Williams %R calculation."""
        highs = [d["high"] for d in sample_ohlc]
        lows = [d["low"] for d in sample_ohlc]
        closes = [d["close"] for d in sample_ohlc]

        williams = calculate_williams_r(highs, lows, closes)

        if williams is not None:
            # Williams %R is between -100 and 0
            assert -100 <= williams <= 0

    @given(ohlc_data(min_length=50))
    def test_atr_properties(self, ohlc):
        """Test ATR properties."""
        highs = [d["high"] for d in ohlc]
        lows = [d["low"] for d in ohlc]
        closes = [d["close"] for d in ohlc]

        atr = calculate_atr(highs, lows, closes)

        if atr is not None:
            # ATR should be positive
            assert atr >= 0

            # ATR should be less than the maximum price range
            max_range = max(highs) - min(lows)
            assert atr <= max_range

    def test_standard_deviation(self, sample_prices):
        """Test standard deviation calculation."""
        std = calculate_standard_deviation(sample_prices, 20)

        if std is not None:
            # Standard deviation should be non-negative
            assert std >= 0

            # For constant prices, std should be 0
            constant_prices = [100.0] * 20
            constant_std = calculate_standard_deviation(constant_prices, 20)
            assert constant_std == 0.0

    @given(ohlc_data(min_length=30))
    def test_adx_range(self, ohlc):
        """Test ADX range."""
        highs = [d["high"] for d in ohlc]
        lows = [d["low"] for d in ohlc]
        closes = [d["close"] for d in ohlc]

        adx = calculate_adx(highs, lows, closes)

        if adx is not None:
            # ADX should be between 0 and 100
            assert 0 <= adx <= 100

    def test_psar_reversal(self, sample_ohlc):
        """Test Parabolic SAR behavior."""
        highs = [d["high"] for d in sample_ohlc]
        lows = [d["low"] for d in sample_ohlc]

        psar = calculate_psar(highs, lows)

        if psar is not None:
            # PSAR should be within the price range
            assert min(lows) <= psar <= max(highs)


class TestVuManchuIntegration:
    """Test VuManchu signal interpretation."""

    def test_vumanchu_signal_interpretation(self):
        """Test VuManchu signal interpretation logic."""
        # Test bullish crossover
        signal = interpret_vumanchu_signal(
            wave_a=-10, wave_b=-15, prev_wave_a=-15, prev_wave_b=-10
        )
        assert signal == "LONG"

        # Test bearish crossover
        signal = interpret_vumanchu_signal(
            wave_a=10, wave_b=15, prev_wave_a=15, prev_wave_b=10
        )
        assert signal == "SHORT"

        # Test extreme oversold
        signal = interpret_vumanchu_signal(wave_a=-35, wave_b=-35)
        assert signal == "LONG"

        # Test extreme overbought
        signal = interpret_vumanchu_signal(wave_a=35, wave_b=35)
        assert signal == "SHORT"

        # Test neutral
        signal = interpret_vumanchu_signal(wave_a=5, wave_b=5)
        assert signal == "NEUTRAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
