"""Comprehensive tests for functional momentum trading strategies.

This module tests the momentum strategy implementations including:
- Basic momentum strategy
- Trend following strategy
- Breakout momentum strategy
- RSI reversal strategy
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite

from bot.fp.strategies.momentum import (
    MomentumSignal,
    _breakout_signal,
    _calculate_trend_strength,
    _combine_momentum_signals,
    _is_volume_confirmed,
    _ma_crossover_signal,
    _rsi_momentum_signal,
    breakout_momentum_strategy,
    momentum_strategy,
    rsi_reversal_strategy,
    trend_following_strategy,
)


class TestMomentumHelperFunctions:
    """Test internal helper functions."""

    def test_ma_crossover_signal_golden_cross(self):
        """Test golden cross detection (bullish)."""
        ma_short = np.array([95, 98, 102, 105])
        ma_long = np.array([100, 100, 100, 100])

        signal = _ma_crossover_signal(ma_short, ma_long)
        assert signal == "LONG"

    def test_ma_crossover_signal_death_cross(self):
        """Test death cross detection (bearish)."""
        ma_short = np.array([105, 102, 98, 95])
        ma_long = np.array([100, 100, 100, 100])

        signal = _ma_crossover_signal(ma_short, ma_long)
        assert signal == "SHORT"

    def test_ma_crossover_signal_no_cross(self):
        """Test no crossover."""
        ma_short = np.array([105, 106, 107, 108])
        ma_long = np.array([100, 100, 100, 100])

        signal = _ma_crossover_signal(ma_short, ma_long)
        assert signal == "HOLD"

    def test_ma_crossover_insufficient_data(self):
        """Test with insufficient data."""
        ma_short = np.array([100])
        ma_long = np.array([95])

        signal = _ma_crossover_signal(ma_short, ma_long)
        assert signal == "HOLD"

    def test_rsi_momentum_signal_oversold(self):
        """Test RSI oversold signal."""
        signal = _rsi_momentum_signal(25.0, 30.0, 70.0)
        assert signal == "LONG"

    def test_rsi_momentum_signal_overbought(self):
        """Test RSI overbought signal."""
        signal = _rsi_momentum_signal(75.0, 30.0, 70.0)
        assert signal == "SHORT"

    def test_rsi_momentum_signal_neutral(self):
        """Test RSI neutral zone."""
        signal = _rsi_momentum_signal(50.0, 30.0, 70.0)
        assert signal == "HOLD"

    def test_volume_confirmation(self):
        """Test volume confirmation logic."""
        assert _is_volume_confirmed(1500, 1000, 1.5) == True
        assert _is_volume_confirmed(1499, 1000, 1.5) == False
        assert _is_volume_confirmed(2000, 1000, 2.0) == True

    def test_breakout_signal_upward(self):
        """Test upward breakout detection."""
        prices = np.array([100, 102, 101, 103, 104, 105, 106, 112])
        signal = _breakout_signal(prices, lookback=5, threshold=0.05)
        assert signal == "LONG"

    def test_breakout_signal_downward(self):
        """Test downward breakout detection."""
        prices = np.array([100, 98, 99, 97, 96, 95, 94, 88])
        signal = _breakout_signal(prices, lookback=5, threshold=0.05)
        assert signal == "SHORT"

    def test_breakout_signal_no_breakout(self):
        """Test no breakout."""
        prices = np.array([100, 101, 99, 100, 101, 99, 100])
        signal = _breakout_signal(prices, lookback=5, threshold=0.05)
        assert signal == "HOLD"

    def test_calculate_trend_strength(self):
        """Test trend strength calculation."""
        # Strong uptrend
        strength_up = _calculate_trend_strength(110, 105, 100)
        assert 0 < strength_up <= 1

        # Strong downtrend
        strength_down = _calculate_trend_strength(90, 95, 100)
        assert -1 <= strength_down < 0

        # Neutral
        strength_neutral = _calculate_trend_strength(100, 100, 100)
        assert abs(strength_neutral) < 0.1

    def test_combine_momentum_signals_strong_consensus(self):
        """Test signal combination with strong consensus."""
        signal, strength, reason = _combine_momentum_signals(
            sma_signal="LONG",
            ema_signal="LONG",
            rsi_signal="LONG",
            breakout_signal="LONG",
            volume_confirmed=True,
            trend_strength=0.5,
        )
        assert signal == "LONG"
        assert strength == 0.9
        assert "Strong bullish momentum" in reason

    def test_combine_momentum_signals_moderate_consensus(self):
        """Test signal combination with moderate consensus."""
        signal, strength, reason = _combine_momentum_signals(
            sma_signal="LONG",
            ema_signal="LONG",
            rsi_signal="HOLD",
            breakout_signal="HOLD",
            volume_confirmed=False,
            trend_strength=0.3,
        )
        assert signal == "LONG"
        assert strength == 0.5
        assert "Bullish momentum" in reason

    def test_combine_momentum_signals_no_consensus(self):
        """Test signal combination with no consensus."""
        signal, strength, reason = _combine_momentum_signals(
            sma_signal="LONG",
            ema_signal="SHORT",
            rsi_signal="HOLD",
            breakout_signal="HOLD",
            volume_confirmed=False,
            trend_strength=0.0,
        )
        assert signal == "HOLD"
        assert strength == 0.0


class TestMomentumStrategy:
    """Test the main momentum strategy."""

    def test_momentum_strategy_creation(self):
        """Test creating momentum strategy with custom parameters."""
        strategy = momentum_strategy(
            lookback_short=10,
            lookback_long=20,
            rsi_period=14,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            volume_multiplier=1.5,
            breakout_threshold=0.02,
        )
        assert callable(strategy)

    def test_momentum_strategy_insufficient_data(self):
        """Test momentum strategy with insufficient data."""
        strategy = momentum_strategy()
        prices = np.array([100, 101, 102])
        volumes = np.array([1000, 1100, 1200])

        signal = strategy(prices, volumes)
        assert signal.signal == "HOLD"
        assert signal.strength == 0.0
        assert "Insufficient data" in signal.reason

    def test_momentum_strategy_bullish_signal(self):
        """Test momentum strategy generating bullish signal."""
        strategy = momentum_strategy(lookback_short=5, lookback_long=10, rsi_period=5)

        # Create uptrending data
        prices = np.array([100 + i for i in range(15)])
        volumes = np.array([1000] * 15)
        volumes[-1] = 2000  # High volume on last bar

        signal = strategy(prices, volumes)
        assert isinstance(signal, MomentumSignal)
        assert signal.signal in ["LONG", "HOLD"]  # Should lean bullish

    def test_momentum_strategy_indicators(self):
        """Test that momentum strategy returns correct indicators."""
        strategy = momentum_strategy(lookback_short=5, lookback_long=10, rsi_period=5)

        prices = np.array([100 + i * 0.5 for i in range(15)])
        volumes = np.array([1000] * 15)

        signal = strategy(prices, volumes)

        # Check all required indicators are present
        assert "sma_short" in signal.indicators
        assert "sma_long" in signal.indicators
        assert "ema_short" in signal.indicators
        assert "ema_long" in signal.indicators
        assert "rsi" in signal.indicators
        assert "volume" in signal.indicators
        assert "volume_sma" in signal.indicators
        assert "trend_strength" in signal.indicators

        # Validate indicator values
        assert signal.indicators["sma_short"] > 0
        assert signal.indicators["sma_long"] > 0
        assert 0 <= signal.indicators["rsi"] <= 100
        assert -1 <= signal.indicators["trend_strength"] <= 1


class TestTrendFollowingStrategy:
    """Test trend following strategy."""

    def test_trend_following_creation(self):
        """Test creating trend following strategy."""
        strategy = trend_following_strategy(
            fast_period=12, slow_period=26, signal_period=9, atr_multiplier=2.0
        )
        assert callable(strategy)

    def test_trend_following_insufficient_data(self):
        """Test with insufficient data."""
        strategy = trend_following_strategy()
        prices = np.array([100, 101, 102])
        volumes = np.array([1000, 1100, 1200])

        signal = strategy(prices, volumes)
        assert signal.signal == "HOLD"
        assert "Insufficient data" in signal.reason

    def test_trend_following_uptrend(self):
        """Test trend following in uptrend."""
        strategy = trend_following_strategy(
            fast_period=5, slow_period=10, signal_period=3
        )

        # Create strong uptrend
        prices = np.array([100 + i * 2 for i in range(20)])
        volumes = np.array([1000] * 20)

        signal = strategy(prices, volumes)
        assert signal.signal == "LONG"
        assert signal.strength > 0
        assert "Uptrend" in signal.reason

    def test_trend_following_downtrend(self):
        """Test trend following in downtrend."""
        strategy = trend_following_strategy(
            fast_period=5, slow_period=10, signal_period=3
        )

        # Create strong downtrend
        prices = np.array([120 - i * 2 for i in range(20)])
        volumes = np.array([1000] * 20)

        signal = strategy(prices, volumes)
        assert signal.signal == "SHORT"
        assert signal.strength > 0
        assert "Downtrend" in signal.reason

    def test_trend_following_indicators(self):
        """Test MACD indicators returned."""
        strategy = trend_following_strategy()

        prices = np.array([100 + np.sin(i * 0.1) * 5 for i in range(40)])
        volumes = np.array([1000] * 40)

        signal = strategy(prices, volumes)

        assert "macd" in signal.indicators
        assert "signal" in signal.indicators
        assert "histogram" in signal.indicators
        assert "ema_fast" in signal.indicators
        assert "ema_slow" in signal.indicators


class TestBreakoutMomentumStrategy:
    """Test breakout momentum strategy."""

    def test_breakout_strategy_creation(self):
        """Test creating breakout strategy."""
        strategy = breakout_momentum_strategy(
            lookback=20, breakout_pct=0.015, volume_surge=2.0, confirm_bars=2
        )
        assert callable(strategy)

    def test_breakout_upward_confirmed(self):
        """Test confirmed upward breakout."""
        strategy = breakout_momentum_strategy(
            lookback=10, breakout_pct=0.02, volume_surge=1.5, confirm_bars=2
        )

        # Create range-bound followed by breakout
        prices = np.array(
            [
                100,
                101,
                99,
                100,
                101,
                99,
                100,
                101,
                99,
                100,  # Range
                103,
                104,
                105,
            ]  # Breakout
        )
        volumes = np.array(
            [1000] * 10  # Normal volume in range
            + [1600, 1700, 1800]  # High volume on breakout
        )

        signal = strategy(prices, volumes)
        assert signal.signal == "LONG"
        assert "Upward breakout confirmed" in signal.reason

    def test_breakout_downward_confirmed(self):
        """Test confirmed downward breakout."""
        strategy = breakout_momentum_strategy(
            lookback=10, breakout_pct=0.02, volume_surge=1.5, confirm_bars=2
        )

        # Create range-bound followed by breakdown
        prices = np.array(
            [
                100,
                101,
                99,
                100,
                101,
                99,
                100,
                101,
                99,
                100,  # Range
                97,
                96,
                95,
            ]  # Breakdown
        )
        volumes = np.array(
            [1000] * 10  # Normal volume in range
            + [1600, 1700, 1800]  # High volume on breakdown
        )

        signal = strategy(prices, volumes)
        assert signal.signal == "SHORT"
        assert "Downward breakout confirmed" in signal.reason

    def test_breakout_no_volume_confirmation(self):
        """Test breakout without volume confirmation."""
        strategy = breakout_momentum_strategy(
            lookback=10, breakout_pct=0.02, volume_surge=2.0, confirm_bars=2
        )

        # Breakout without volume
        prices = np.array(
            [100, 101, 99, 100, 101, 99, 100, 101, 99, 100, 103, 104, 105]
        )
        volumes = np.array([1000] * 13)  # No volume surge

        signal = strategy(prices, volumes)
        assert signal.signal == "HOLD"
        assert "No confirmed breakout" in signal.reason

    def test_breakout_indicators(self):
        """Test breakout strategy indicators."""
        strategy = breakout_momentum_strategy()

        prices = np.array([100 + i * 0.1 for i in range(25)])
        volumes = np.array([1000] * 25)

        signal = strategy(prices, volumes)

        assert "range_high" in signal.indicators
        assert "range_low" in signal.indicators
        assert "range_size" in signal.indicators
        assert "avg_volume" in signal.indicators
        assert "current_volume" in signal.indicators
        assert "breakout_distance" in signal.indicators


class TestRSIReversalStrategy:
    """Test RSI reversal strategy."""

    def test_rsi_reversal_creation(self):
        """Test creating RSI reversal strategy."""
        strategy = rsi_reversal_strategy(
            rsi_period=14, oversold=25.0, overbought=75.0, divergence_lookback=10
        )
        assert callable(strategy)

    def test_rsi_reversal_oversold(self):
        """Test RSI oversold reversal signal."""
        strategy = rsi_reversal_strategy(
            rsi_period=5, oversold=30.0, overbought=70.0, divergence_lookback=5
        )

        # Create oversold condition
        prices = np.array([100 - i * 2 for i in range(10)] + [82, 83, 84])
        volumes = np.array([1000] * 13)

        signal = strategy(prices, volumes)
        assert signal.signal == "LONG"
        assert "oversold" in signal.reason.lower()

    def test_rsi_reversal_overbought(self):
        """Test RSI overbought reversal signal."""
        strategy = rsi_reversal_strategy(
            rsi_period=5, oversold=30.0, overbought=70.0, divergence_lookback=5
        )

        # Create overbought condition
        prices = np.array([100 + i * 2 for i in range(10)] + [118, 117, 116])
        volumes = np.array([1000] * 13)

        signal = strategy(prices, volumes)
        assert signal.signal == "SHORT"
        assert "overbought" in signal.reason.lower()

    def test_rsi_bullish_divergence(self):
        """Test RSI bullish divergence detection."""
        strategy = rsi_reversal_strategy(
            rsi_period=5, oversold=35.0, overbought=65.0, divergence_lookback=8
        )

        # Create bullish divergence: price makes lower low, RSI makes higher low
        prices = np.array(
            [
                100,
                98,
                96,
                94,
                92,  # First decline
                93,
                94,
                95,  # Small bounce
                93,
                92,
                91,
                90,
                89,  # Lower low in price
            ]
        )
        volumes = np.array([1000] * len(prices))

        signal = strategy(prices, volumes)
        # Should detect bullish divergence if RSI doesn't make lower low
        assert signal.signal in ["LONG", "HOLD"]
        if signal.signal == "LONG":
            assert "divergence" in signal.reason.lower()

    def test_rsi_reversal_indicators(self):
        """Test RSI reversal strategy indicators."""
        strategy = rsi_reversal_strategy()

        prices = np.array([100 + np.sin(i * 0.2) * 10 for i in range(20)])
        volumes = np.array([1000] * 20)

        signal = strategy(prices, volumes)

        assert "rsi" in signal.indicators
        assert "rsi_high" in signal.indicators
        assert "rsi_low" in signal.indicators
        assert "price_high" in signal.indicators
        assert "price_low" in signal.indicators
        assert "bullish_divergence" in signal.indicators
        assert "bearish_divergence" in signal.indicators

        assert 0 <= signal.indicators["rsi"] <= 100
        assert isinstance(signal.indicators["bullish_divergence"], bool)
        assert isinstance(signal.indicators["bearish_divergence"], bool)


# Property-based tests
@composite
def price_volume_data(draw):
    """Generate valid price and volume data for testing."""
    length = draw(st.integers(min_value=20, max_value=100))

    # Generate prices with realistic constraints
    initial_price = draw(st.floats(min_value=10.0, max_value=10000.0))
    price_changes = draw(
        st.lists(
            st.floats(min_value=-0.05, max_value=0.05),
            min_size=length - 1,
            max_size=length - 1,
        )
    )

    prices = [initial_price]
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(max(1.0, new_price))  # Ensure positive

    # Generate volumes
    volumes = draw(
        st.lists(
            st.floats(min_value=100.0, max_value=100000.0),
            min_size=length,
            max_size=length,
        )
    )

    return np.array(prices), np.array(volumes)


class TestPropertyBased:
    """Property-based tests for momentum strategies."""

    @given(price_volume_data())
    def test_momentum_strategy_always_valid_output(self, data):
        """Test momentum strategy always produces valid output."""
        prices, volumes = data
        strategy = momentum_strategy(lookback_short=5, lookback_long=10, rsi_period=5)

        signal = strategy(prices, volumes)

        assert isinstance(signal, MomentumSignal)
        assert signal.signal in ["LONG", "SHORT", "HOLD"]
        assert 0.0 <= signal.strength <= 1.0
        assert isinstance(signal.reason, str)
        assert isinstance(signal.indicators, dict)

    @given(price_volume_data())
    def test_trend_following_deterministic(self, data):
        """Test trend following strategy is deterministic."""
        prices, volumes = data
        strategy = trend_following_strategy(
            fast_period=5, slow_period=10, signal_period=3
        )

        # Run twice with same data
        signal1 = strategy(prices, volumes)
        signal2 = strategy(prices, volumes)

        assert signal1.signal == signal2.signal
        assert signal1.strength == signal2.strength
        assert signal1.reason == signal2.reason

    @given(
        st.floats(min_value=5, max_value=50),
        st.floats(min_value=0.001, max_value=0.1),
        st.floats(min_value=1.1, max_value=5.0),
    )
    def test_breakout_strategy_parameter_sensitivity(
        self, lookback, breakout_pct, volume_surge
    ):
        """Test breakout strategy with various parameters."""
        strategy = breakout_momentum_strategy(
            lookback=int(lookback),
            breakout_pct=breakout_pct,
            volume_surge=volume_surge,
            confirm_bars=2,
        )

        # Generate simple test data
        prices = np.array([100] * 50 + [100 * (1 + breakout_pct * 1.5)] * 3)
        volumes = np.array([1000] * 50 + [1000 * volume_surge * 1.1] * 3)

        signal = strategy(prices, volumes)

        # Should detect breakout
        assert signal.signal in ["LONG", "HOLD"]
        if signal.signal == "LONG":
            assert signal.strength > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_all_same_prices(self):
        """Test strategies with constant prices."""
        prices = np.array([100.0] * 50)
        volumes = np.array([1000.0] * 50)

        strategies = [
            momentum_strategy(),
            trend_following_strategy(),
            breakout_momentum_strategy(),
            rsi_reversal_strategy(),
        ]

        for strategy in strategies:
            signal = strategy(prices, volumes)
            assert signal.signal == "HOLD"
            assert signal.strength == 0.0

    def test_zero_volume(self):
        """Test strategies with zero volume."""
        prices = np.array([100 + i for i in range(30)])
        volumes = np.array([0.0] * 30)

        strategy = momentum_strategy()
        signal = strategy(prices, volumes)

        # Should still work but volume confirmation will fail
        assert isinstance(signal, MomentumSignal)

    def test_extreme_price_movements(self):
        """Test strategies with extreme price movements."""
        # 50% daily moves
        prices = np.array([100, 150, 75, 112, 56, 84, 126, 63])
        volumes = np.array([10000] * 8)

        strategy = momentum_strategy(lookback_short=2, lookback_long=4, rsi_period=3)

        signal = strategy(prices, volumes)
        assert isinstance(signal, MomentumSignal)
        assert signal.signal in ["LONG", "SHORT", "HOLD"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
