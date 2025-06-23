"""
Standalone comprehensive tests for functional programming indicator functions.

This module tests indicator functions without external dependencies,
focusing on property-based testing, edge cases, and mathematical properties.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite

# ============================================================================
# Test Data Strategies
# ============================================================================


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


# ============================================================================
# Pure Indicator Implementations (for testing)
# ============================================================================


def calculate_sma(prices: Sequence[float], period: int) -> float | None:
    """Calculate Simple Moving Average."""
    if len(prices) < period or period <= 0:
        return None
    return sum(prices[-period:]) / period


def calculate_ema(prices: Sequence[float], period: int) -> float | None:
    """Calculate Exponential Moving Average."""
    if len(prices) < period or period <= 0:
        return None

    # Calculate initial SMA
    sma = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)

    ema = sma
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def calculate_rsi(prices: Sequence[float], period: int = 14) -> float | None:
    """Calculate Relative Strength Index."""
    if len(prices) < period + 1 or period <= 0:
        return None

    # Calculate price changes
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # Separate gains and losses
    gains = [max(change, 0) for change in changes]
    losses = [abs(min(change, 0)) for change in changes]

    # Calculate initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Apply smoothing for remaining data
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Calculate RSI
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_bollinger_bands(
    prices: Sequence[float], period: int = 20, std_dev: float = 2.0
) -> tuple[float | None, float | None, float | None]:
    """Calculate Bollinger Bands."""
    if len(prices) < period or period <= 0:
        return None, None, None

    sma = calculate_sma(prices, period)
    if sma is None:
        return None, None, None

    # Calculate standard deviation
    recent_prices = prices[-period:]
    variance = sum((p - sma) ** 2 for p in recent_prices) / period
    std = variance**0.5

    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)

    return upper, sma, lower


def calculate_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Calculate Average True Range."""
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None

    if period <= 0:
        return None

    # Calculate True Range values
    tr_values = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close_prev = abs(highs[i] - closes[i - 1])
        low_close_prev = abs(lows[i] - closes[i - 1])

        tr = max(high_low, high_close_prev, low_close_prev)
        tr_values.append(tr)

    # Calculate ATR
    if len(tr_values) < period:
        return None

    # Initial ATR is simple average
    atr = sum(tr_values[:period]) / period

    # Apply smoothing for remaining values
    for i in range(period, len(tr_values)):
        atr = (atr * (period - 1) + tr_values[i]) / period

    return atr


# ============================================================================
# Test Fixtures
# ============================================================================


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


# ============================================================================
# Test Classes
# ============================================================================


class TestMovingAverages:
    """Test moving average calculations."""

    @given(price_data(min_length=1, max_length=100))
    def test_sma_mathematical_properties(self, prices):
        """Test SMA mathematical properties."""
        period = min(20, len(prices))
        result = calculate_sma(prices, period)

        if result is not None:
            # SMA should be within price range
            assert min(prices[-period:]) <= result <= max(prices[-period:])

            # SMA should equal arithmetic mean
            expected = sum(prices[-period:]) / period
            assert abs(result - expected) < 1e-10

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

    def test_rsi_all_gains(self):
        """Test RSI when all price changes are positive."""
        prices = list(range(100, 120))  # Continuously rising
        rsi = calculate_rsi(prices, 14)
        assert rsi is not None
        assert rsi > 70  # Should indicate overbought

    def test_rsi_all_losses(self):
        """Test RSI when all price changes are negative."""
        prices = list(range(120, 100, -1))  # Continuously falling
        rsi = calculate_rsi(prices, 14)
        assert rsi is not None
        assert rsi < 30  # Should indicate oversold

    def test_rsi_no_change(self):
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

    def test_bollinger_squeeze(self):
        """Test Bollinger Bands squeeze detection."""
        # Create low volatility prices
        prices = [100.0 + np.sin(i * 0.1) * 0.1 for i in range(50)]
        upper, middle, lower = calculate_bollinger_bands(prices)

        if upper is not None:
            width = upper - lower
            width_ratio = width / middle if middle > 0 else 0

            # Low volatility should result in narrow bands
            assert width_ratio < 0.1


class TestATR:
    """Test Average True Range calculations."""

    @given(ohlc_data(min_length=20))
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

    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        highs = [100.0, 101.0]
        lows = [99.0, 100.0]
        closes = [100.0, 100.5]

        assert calculate_atr(highs, lows, closes, 14) is None


class TestEdgeCases:
    """Test edge cases for all indicators."""

    def test_empty_data_all_indicators(self):
        """Test all indicators with empty data."""
        prices = []

        assert calculate_sma(prices, 10) is None
        assert calculate_ema(prices, 10) is None
        assert calculate_rsi(prices) is None

        upper, middle, lower = calculate_bollinger_bands(prices)
        assert all(x is None for x in [upper, middle, lower])

    def test_single_price_all_indicators(self):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
