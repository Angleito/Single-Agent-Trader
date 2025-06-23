"""
Comprehensive tests for existing functional programming indicator functions.

This module tests the actual indicator functions from bot.fp.indicators
with property-based testing, edge cases, and mathematical properties.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite

# Import the actual indicator functions
from bot.fp.indicators import (
    calculate_all_momentum_indicators,
    macd,
    rate_of_change,
    rsi,
    stochastic,
)

# ============================================================================
# Test Data Strategies
# ============================================================================


@composite
def price_array(draw, min_length=1, max_length=1000):
    """Generate realistic price data as numpy array."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))

    # Generate base price
    base_price = draw(st.floats(min_value=0.01, max_value=10000.0))

    # Generate price changes (small percentage moves)
    changes = draw(
        st.lists(
            st.floats(
                min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False
            ),
            min_size=length,
            max_size=length,
        )
    )

    # Build price series
    prices = [base_price]
    for change in changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Ensure positive prices

    return np.array(prices, dtype=np.float64)


@composite
def ohlc_arrays(draw, min_length=1, max_length=1000):
    """Generate OHLC data as numpy arrays."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))

    opens = []
    highs = []
    lows = []
    closes = []

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

        opens.append(max(open_price, 0.01))
        highs.append(max(high_price, 0.01))
        lows.append(max(low_price, 0.01))
        closes.append(max(close_price, 0.01))

        base_price = close_price

    return (
        np.array(opens, dtype=np.float64),
        np.array(highs, dtype=np.float64),
        np.array(lows, dtype=np.float64),
        np.array(closes, dtype=np.float64),
    )


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
    return prices


@pytest.fixture
def sample_ohlc():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    base = 100.0
    n = 100

    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)

    for i in range(n):
        opens[i] = base
        close_price = base * (1 + np.random.normal(0.001, 0.02))
        closes[i] = close_price
        highs[i] = max(opens[i], closes[i]) * (1 + abs(np.random.normal(0, 0.01)))
        lows[i] = min(opens[i], closes[i]) * (1 - abs(np.random.normal(0, 0.01)))
        base = close_price

    return opens, highs, lows, closes


@pytest.fixture
def test_timestamp():
    """Generate a test timestamp."""
    return datetime.now()


# ============================================================================
# Test Classes
# ============================================================================


class TestRSI:
    """Test RSI calculations."""

    @given(price_array(min_length=15, max_length=1000))
    def test_rsi_bounds(self, prices):
        """Test RSI is always between 0 and 100."""
        result = rsi(prices)

        if result is not None:
            assert 0 <= result.value <= 100
            assert result.timestamp is not None

    def test_rsi_all_gains(self, test_timestamp):
        """Test RSI when all price changes are positive."""
        prices = np.array(
            list(range(100, 120)), dtype=np.float64
        )  # Continuously rising
        result = rsi(prices)

        assert result is not None
        assert result.value > 70  # Should indicate overbought
        assert result.is_overbought()
        assert not result.is_oversold()

    def test_rsi_all_losses(self, test_timestamp):
        """Test RSI when all price changes are negative."""
        prices = np.array(
            list(range(120, 100, -1)), dtype=np.float64
        )  # Continuously falling
        result = rsi(prices)

        assert result is not None
        assert result.value < 30  # Should indicate oversold
        assert result.is_oversold()
        assert not result.is_overbought()

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = np.array([100.0, 101.0, 102.0])
        result = rsi(prices, period=14)
        assert result is None

    @pytest.mark.parametrize("period", [0, -1])
    def test_rsi_invalid_period(self, sample_prices, period):
        """Test RSI with invalid period."""
        result = rsi(sample_prices, period=period)
        assert result is None


class TestMACD:
    """Test MACD calculations."""

    @given(price_array(min_length=50, max_length=1000))
    def test_macd_calculation(self, prices):
        """Test MACD calculation properties."""
        result = macd(prices)

        if result is not None:
            # Histogram should equal MACD - Signal
            assert abs(result.histogram - (result.macd - result.signal)) < 1e-10
            assert result.timestamp is not None

    def test_macd_insufficient_data(self):
        """Test MACD with insufficient data."""
        prices = np.array([100.0] * 20)
        result = macd(prices)
        assert result is None

    def test_macd_trending_market(self, test_timestamp):
        """Test MACD in trending market."""
        # Create uptrending prices
        prices = np.array([100.0 * (1.01**i) for i in range(50)])
        result = macd(prices)

        if result is not None:
            # In uptrend, MACD should be positive or neutral
            assert result.momentum_direction() in ["bullish", "neutral"]


class TestStochastic:
    """Test Stochastic oscillator."""

    @given(ohlc_arrays(min_length=20, max_length=1000))
    def test_stochastic_bounds(self, ohlc):
        """Test Stochastic oscillator bounds."""
        opens, highs, lows, closes = ohlc
        result = stochastic(highs, lows, closes)

        if result is not None:
            assert 0 <= result.k <= 100
            assert 0 <= result.d <= 100
            assert result.timestamp is not None

    def test_stochastic_insufficient_data(self):
        """Test Stochastic with insufficient data."""
        highs = np.array([100.0, 101.0])
        lows = np.array([99.0, 100.0])
        closes = np.array([100.0, 100.5])

        result = stochastic(highs, lows, closes, period=14)
        assert result is None

    def test_stochastic_extreme_values(self, sample_ohlc):
        """Test Stochastic with extreme market conditions."""
        opens, highs, lows, closes = sample_ohlc

        # Test when close is at high (should be near 100)
        closes_at_high = highs.copy()
        result = stochastic(highs, lows, closes_at_high)
        if result is not None:
            assert result.k > 80  # Should be high

        # Test when close is at low (should be near 0)
        closes_at_low = lows.copy()
        result = stochastic(highs, lows, closes_at_low)
        if result is not None:
            assert result.k < 20  # Should be low


class TestRateOfChange:
    """Test Rate of Change indicator."""

    @given(price_array(min_length=15, max_length=1000))
    def test_roc_calculation(self, prices):
        """Test ROC calculation."""
        result = rate_of_change(prices)

        if result is not None:
            # ROC should be reasonable percentage
            assert -100 < result.value < 1000  # Allow for large gains
            assert result.timestamp is not None

    def test_roc_zero_change(self):
        """Test ROC with no price change."""
        prices = np.array([100.0] * 20)
        result = rate_of_change(prices, period=10)

        if result is not None:
            assert abs(result.value) < 0.001  # Should be near 0

    def test_roc_positive_trend(self):
        """Test ROC in positive trend."""
        prices = np.array([100.0 * (1.01**i) for i in range(30)])
        result = rate_of_change(prices, period=10)

        if result is not None:
            assert result.value > 0  # Should be positive
            assert result.is_positive()


class TestCombinedIndicators:
    """Test calculate_all_momentum_indicators function."""

    @given(ohlc_arrays(min_length=50, max_length=200))
    def test_all_indicators(self, ohlc):
        """Test combined indicator calculation."""
        opens, highs, lows, closes = ohlc
        results = calculate_all_momentum_indicators(highs, lows, closes)

        # Check that we get a dictionary with expected keys
        assert isinstance(results, dict)

        # Check individual results if present
        if "rsi" in results and results["rsi"] is not None:
            assert 0 <= results["rsi"].value <= 100

        if "macd" in results and results["macd"] is not None:
            macd_result = results["macd"]
            assert hasattr(macd_result, "macd")
            assert hasattr(macd_result, "signal")
            assert hasattr(macd_result, "histogram")

        if "stochastic" in results and results["stochastic"] is not None:
            stoch_result = results["stochastic"]
            assert 0 <= stoch_result.k <= 100
            assert 0 <= stoch_result.d <= 100

        if "roc" in results and results["roc"] is not None:
            assert hasattr(results["roc"], "value")

    def test_all_indicators_insufficient_data(self):
        """Test all indicators with insufficient data."""
        highs = np.array([100.0, 101.0])
        lows = np.array([99.0, 100.0])
        closes = np.array([100.0, 100.5])

        results = calculate_all_momentum_indicators(highs, lows, closes)

        # Most indicators should return None with insufficient data
        assert results["rsi"] is None
        assert results["macd"] is None


class TestEdgeCases:
    """Test edge cases for all indicators."""

    def test_empty_data_all_indicators(self):
        """Test all indicators with empty data."""
        prices = np.array([])
        highs = np.array([])
        lows = np.array([])
        closes = np.array([])

        assert rsi(prices) is None
        assert macd(prices) is None
        assert stochastic(highs, lows, closes) is None
        assert rate_of_change(prices) is None

    def test_single_price_all_indicators(self):
        """Test all indicators with single price."""
        prices = np.array([100.0])
        highs = np.array([101.0])
        lows = np.array([99.0])
        closes = np.array([100.0])

        assert rsi(prices) is None
        assert macd(prices) is None
        assert stochastic(highs, lows, closes) is None
        assert rate_of_change(prices) is None

    @given(st.lists(st.just(100.0), min_size=50, max_size=50))
    def test_constant_prices(self, price_list):
        """Test indicators with constant prices."""
        prices = np.array(price_list)

        # RSI should handle constant prices
        rsi_result = rsi(prices)
        if rsi_result is not None:
            # With no price movement, RSI might be 50 or handle it specially
            assert 0 <= rsi_result.value <= 100

        # ROC should be 0 for constant prices
        roc_result = rate_of_change(prices)
        if roc_result is not None:
            assert abs(roc_result.value) < 0.001

    @given(
        st.lists(
            st.floats(
                min_value=1e-10, max_value=1e-8, allow_nan=False, allow_infinity=False
            ),
            min_size=20,
        )
    )
    def test_very_small_prices(self, price_list):
        """Test indicators with very small prices."""
        prices = np.array(price_list)

        # Ensure calculations don't break with small numbers
        rsi_result = rsi(prices)
        if rsi_result is not None:
            assert not np.isnan(rsi_result.value)
            assert not np.isinf(rsi_result.value)
            assert 0 <= rsi_result.value <= 100


class TestMathematicalProperties:
    """Test mathematical properties of indicators."""

    def test_indicator_consistency(self, sample_prices):
        """Test that indicators are consistent across multiple calls."""
        # Same input should produce same output
        rsi1 = rsi(sample_prices)
        rsi2 = rsi(sample_prices)

        if rsi1 is not None and rsi2 is not None:
            assert rsi1.value == rsi2.value

        macd1 = macd(sample_prices)
        macd2 = macd(sample_prices)

        if macd1 is not None and macd2 is not None:
            assert macd1.macd == macd2.macd
            assert macd1.signal == macd2.signal
            assert macd1.histogram == macd2.histogram

    @given(
        price_array(min_length=50, max_length=200),
        st.integers(min_value=5, max_value=20),
    )
    def test_rsi_period_relationship(self, prices, period):
        """Test RSI behavior with different periods."""
        result = rsi(prices, period=period)

        if result is not None:
            # Shorter periods should be more volatile
            # This is a general property, not always true
            assert 0 <= result.value <= 100

    def test_macd_signal_relationship(self, sample_prices):
        """Test MACD and signal line relationship."""
        result = macd(sample_prices)

        if result is not None:
            # Histogram is the difference
            calculated_hist = result.macd - result.signal
            assert abs(result.histogram - calculated_hist) < 1e-10

            # Check crossover methods
            if result.histogram > 0:
                assert (
                    result.is_bullish_crossover() or not result.is_bearish_crossover()
                )
            elif result.histogram < 0:
                assert (
                    result.is_bearish_crossover() or not result.is_bullish_crossover()
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
