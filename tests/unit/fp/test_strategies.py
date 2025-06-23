"""Comprehensive tests for functional programming strategy functions.

This module tests:
1. Each strategy type
2. Strategy combinators
3. Property-based testing for signal generation
4. Backtest simulation tests
5. Risk management validation
6. Performance metrics calculation
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite


# Mock imports - we'll define simplified versions for testing
class SignalType(Enum):
    """Trading signal types."""

    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass(frozen=True)
class TradeSignal:
    """Trade signal with metadata."""

    signal: SignalType
    strength: float
    reason: str
    metadata: dict

    def __post_init__(self):
        """Validate signal strength."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Signal strength must be between 0 and 1, got {self.strength}"
            )


@dataclass(frozen=True)
class MarketSnapshot:
    """Simplified market snapshot for testing."""

    timestamp: datetime
    symbol: str
    current_price: float
    volume: float
    sma_20: float | None = None
    ema_20: float | None = None
    high_20: float | None = None
    low_20: float | None = None
    rsi: float | None = None

    def __post_init__(self):
        """Validate market data."""
        if self.current_price <= 0:
            raise ValueError("Price must be positive")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")


# Import the strategies module (adjust path as needed)
from bot.fp.strategies.base import (
    StrategyMetadata,
    StrategyResult,
    backtest_strategy,
    chain_strategies,
    combine_strategies,
    create_mean_reversion_strategy,
    create_momentum_strategy,
    evaluate_multiple_strategies,
    evaluate_strategy,
    filter_strategy,
    map_strategy,
    threshold_strategy,
)


# Test Strategies
def always_long_strategy(snapshot: MarketSnapshot) -> TradeSignal:
    """Strategy that always returns LONG signal."""
    return TradeSignal(
        signal=SignalType.LONG,
        strength=0.8,
        reason="Always bullish",
        metadata={"strategy": "always_long"},
    )


def always_short_strategy(snapshot: MarketSnapshot) -> TradeSignal:
    """Strategy that always returns SHORT signal."""
    return TradeSignal(
        signal=SignalType.SHORT,
        strength=0.8,
        reason="Always bearish",
        metadata={"strategy": "always_short"},
    )


def always_hold_strategy(snapshot: MarketSnapshot) -> TradeSignal:
    """Strategy that always returns HOLD signal."""
    return TradeSignal(
        signal=SignalType.HOLD,
        strength=0.0,
        reason="Always neutral",
        metadata={"strategy": "always_hold"},
    )


def price_based_strategy(snapshot: MarketSnapshot) -> TradeSignal:
    """Strategy based on price level."""
    if snapshot.current_price > 50000:
        return TradeSignal(
            signal=SignalType.SHORT,
            strength=0.7,
            reason="Price too high",
            metadata={"price": snapshot.current_price},
        )
    if snapshot.current_price < 30000:
        return TradeSignal(
            signal=SignalType.LONG,
            strength=0.7,
            reason="Price too low",
            metadata={"price": snapshot.current_price},
        )
    return TradeSignal(
        signal=SignalType.HOLD,
        strength=0.0,
        reason="Price in range",
        metadata={"price": snapshot.current_price},
    )


class TestBasicStrategies:
    """Test individual strategy functions."""

    def test_momentum_strategy_creation(self):
        """Test creating a momentum strategy."""
        strategy = create_momentum_strategy(lookback=20, threshold=0.02)
        assert callable(strategy)

    def test_momentum_strategy_insufficient_data(self):
        """Test momentum strategy with insufficient data."""
        strategy = create_momentum_strategy()
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )
        signal = strategy(snapshot)
        assert signal.signal == SignalType.HOLD
        assert "Insufficient data" in signal.reason

    def test_momentum_strategy_long_signal(self):
        """Test momentum strategy generates long signal."""
        strategy = create_momentum_strategy(lookback=20, threshold=0.02)
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=50000.0,
            volume=1000.0,
            high_20=51000.0,
            low_20=40000.0,
        )
        signal = strategy(snapshot)
        # Should be LONG as price is near high
        assert signal.signal == SignalType.LONG
        assert signal.strength > 0.5

    def test_mean_reversion_strategy_creation(self):
        """Test creating a mean reversion strategy."""
        strategy = create_mean_reversion_strategy(lookback=20, z_score_threshold=2.0)
        assert callable(strategy)

    def test_mean_reversion_strategy_short_signal(self):
        """Test mean reversion strategy generates short signal."""
        strategy = create_mean_reversion_strategy(lookback=20, z_score_threshold=1.5)
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=50000.0,
            volume=1000.0,
            sma_20=45000.0,
            high_20=48000.0,
            low_20=42000.0,
        )
        signal = strategy(snapshot)
        # Should be SHORT as price is well above SMA
        assert signal.signal == SignalType.SHORT
        assert signal.strength > 0


class TestStrategyCombinators:
    """Test strategy combinator functions."""

    def test_combine_strategies_weighted_average(self):
        """Test combining strategies with weighted average."""
        strategies = [
            ("long", always_long_strategy, 0.6),
            ("hold", always_hold_strategy, 0.4),
        ]
        combined = combine_strategies(strategies, aggregation="weighted_average")

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )
        signal = combined(snapshot)

        # Weighted average should lean toward LONG
        assert signal.signal == SignalType.LONG
        assert 0.4 < signal.strength < 0.6

    def test_combine_strategies_majority_vote(self):
        """Test combining strategies with majority vote."""
        strategies = [
            ("long1", always_long_strategy, 1.0),
            ("long2", always_long_strategy, 1.0),
            ("short", always_short_strategy, 1.0),
        ]
        combined = combine_strategies(strategies, aggregation="majority")

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )
        signal = combined(snapshot)

        # Majority should be LONG (2 vs 1)
        assert signal.signal == SignalType.LONG

    def test_combine_strategies_unanimous(self):
        """Test combining strategies requiring unanimity."""
        strategies_agree = [
            ("long1", always_long_strategy, 1.0),
            ("long2", always_long_strategy, 1.0),
        ]
        strategies_disagree = [
            ("long", always_long_strategy, 1.0),
            ("short", always_short_strategy, 1.0),
        ]

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )

        # Test unanimous agreement
        combined_agree = combine_strategies(strategies_agree, aggregation="unanimous")
        signal_agree = combined_agree(snapshot)
        assert signal_agree.signal == SignalType.LONG

        # Test disagreement
        combined_disagree = combine_strategies(
            strategies_disagree, aggregation="unanimous"
        )
        signal_disagree = combined_disagree(snapshot)
        assert signal_disagree.signal == SignalType.HOLD

    def test_filter_strategy(self):
        """Test filtering strategy based on condition."""

        # Only trade when volume is high
        def high_volume_condition(snapshot: MarketSnapshot) -> bool:
            return snapshot.volume > 5000

        filtered = filter_strategy(always_long_strategy, high_volume_condition)

        # Low volume - should HOLD
        low_vol_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )
        signal_low = filtered(low_vol_snapshot)
        assert signal_low.signal == SignalType.HOLD

        # High volume - should pass through
        high_vol_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=10000.0,
        )
        signal_high = filtered(high_vol_snapshot)
        assert signal_high.signal == SignalType.LONG

    def test_map_strategy(self):
        """Test transforming strategy output."""

        # Invert signals
        def invert_signal(signal: TradeSignal) -> TradeSignal:
            if signal.signal == SignalType.LONG:
                new_signal = SignalType.SHORT
            elif signal.signal == SignalType.SHORT:
                new_signal = SignalType.LONG
            else:
                new_signal = SignalType.HOLD

            return TradeSignal(
                signal=new_signal,
                strength=signal.strength,
                reason=f"Inverted: {signal.reason}",
                metadata=signal.metadata,
            )

        inverted = map_strategy(always_long_strategy, invert_signal)

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )
        signal = inverted(snapshot)

        assert signal.signal == SignalType.SHORT
        assert "Inverted" in signal.reason

    def test_chain_strategies(self):
        """Test chaining strategies with priority."""
        strategies = [always_hold_strategy, price_based_strategy, always_long_strategy]
        chained = chain_strategies(strategies)

        # Price in middle range - price_based should return HOLD,
        # so we fall through to always_long
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )
        signal = chained(snapshot)
        assert signal.signal == SignalType.LONG

    def test_threshold_strategy(self):
        """Test threshold filtering on signal strength."""

        # Create a strategy with variable strength
        def variable_strength_strategy(snapshot: MarketSnapshot) -> TradeSignal:
            strength = snapshot.current_price / 100000  # 0.4 for 40k price
            return TradeSignal(
                signal=SignalType.LONG,
                strength=strength,
                reason=f"Strength {strength:.2f}",
                metadata={},
            )

        thresholded = threshold_strategy(variable_strength_strategy, min_strength=0.7)

        # Low strength - should HOLD
        low_price_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )
        signal_low = thresholded(low_price_snapshot)
        assert signal_low.signal == SignalType.HOLD

        # High strength - should pass through
        high_price_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=80000.0,
            volume=1000.0,
        )
        signal_high = thresholded(high_price_snapshot)
        assert signal_high.signal == SignalType.LONG


class TestStrategyEvaluation:
    """Test strategy evaluation functions."""

    def test_evaluate_strategy(self):
        """Test evaluating a single strategy."""
        metadata = StrategyMetadata(
            name="Test Strategy",
            version="1.0.0",
            description="Test strategy for unit tests",
            parameters={"threshold": 0.5},
            risk_level="low",
            expected_frequency="intraday",
            created_at=datetime.now(),
            tags=["test", "momentum"],
        )

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )

        result = evaluate_strategy(always_long_strategy, snapshot, metadata)

        assert isinstance(result, StrategyResult)
        assert result.signal.signal == SignalType.LONG
        assert result.metadata == metadata
        assert result.computation_time_ms >= 0

    def test_evaluate_multiple_strategies(self):
        """Test evaluating multiple strategies."""
        strategies_with_metadata = [
            (
                always_long_strategy,
                StrategyMetadata(
                    name="Always Long",
                    version="1.0.0",
                    description="Always bullish",
                    parameters={},
                    risk_level="high",
                    expected_frequency="scalping",
                    created_at=datetime.now(),
                    tags=["aggressive"],
                ),
            ),
            (
                always_short_strategy,
                StrategyMetadata(
                    name="Always Short",
                    version="1.0.0",
                    description="Always bearish",
                    parameters={},
                    risk_level="high",
                    expected_frequency="scalping",
                    created_at=datetime.now(),
                    tags=["aggressive"],
                ),
            ),
        ]

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )

        results = evaluate_multiple_strategies(strategies_with_metadata, snapshot)

        assert len(results) == 2
        assert results[0].signal.signal == SignalType.LONG
        assert results[1].signal.signal == SignalType.SHORT
        assert all(r.computation_time_ms >= 0 for r in results)


class TestBacktesting:
    """Test backtesting functionality."""

    def test_backtest_no_trades(self):
        """Test backtesting a strategy that never trades."""
        snapshots = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                current_price=40000.0 + i * 100,
                volume=1000.0,
            )
            for i in range(10)
        ]

        result = backtest_strategy(
            always_hold_strategy, snapshots, initial_capital=10000.0
        )

        assert result.total_trades == 0
        assert result.winning_trades == 0
        assert result.losing_trades == 0
        assert result.total_return == 0.0

    def test_backtest_with_trades(self):
        """Test backtesting with actual trades."""
        # Create price series with trend
        prices = [40000.0, 41000.0, 42000.0, 41500.0, 43000.0]
        snapshots = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                current_price=price,
                volume=1000.0,
            )
            for price in prices
        ]

        result = backtest_strategy(
            always_long_strategy,
            snapshots,
            initial_capital=10000.0,
            position_size=0.1,
            fees=0.001,
        )

        assert result.total_trades > 0
        assert result.total_return != 0.0
        assert 0 <= result.win_rate <= 1.0

    def test_backtest_metrics_calculation(self):
        """Test that backtest metrics are calculated correctly."""

        # Create alternating profitable/losing trades
        def alternating_strategy(snapshot: MarketSnapshot) -> TradeSignal:
            # Trade based on price ending (odd/even)
            if int(snapshot.current_price) % 2 == 0:
                return TradeSignal(
                    signal=SignalType.LONG,
                    strength=0.8,
                    reason="Even price",
                    metadata={},
                )
            return TradeSignal(
                signal=SignalType.SHORT, strength=0.8, reason="Odd price", metadata={}
            )

        # Create price series with known outcomes
        prices = [40000.0, 40100.0, 39900.0, 40200.0, 39800.0]
        snapshots = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                current_price=price,
                volume=1000.0,
            )
            for price in prices
        ]

        result = backtest_strategy(
            alternating_strategy, snapshots, initial_capital=10000.0
        )

        assert result.total_trades >= len(snapshots) - 1
        assert result.profit_factor > 0 or result.losing_trades == 0


# Property-based tests
@composite
def market_snapshot_strategy(draw):
    """Generate valid market snapshots for property testing."""
    price = draw(st.floats(min_value=1.0, max_value=100000.0))
    volume = draw(st.floats(min_value=0.0, max_value=1000000.0))

    # Optional fields
    sma_20 = draw(
        st.one_of(st.none(), st.floats(min_value=price * 0.8, max_value=price * 1.2))
    )

    high_20 = draw(
        st.one_of(st.none(), st.floats(min_value=price, max_value=price * 1.5))
    )

    low_20 = draw(
        st.one_of(st.none(), st.floats(min_value=price * 0.5, max_value=price))
    )

    return MarketSnapshot(
        timestamp=datetime.now(),
        symbol="BTC-USD",
        current_price=price,
        volume=volume,
        sma_20=sma_20,
        high_20=high_20,
        low_20=low_20,
    )


class TestPropertyBased:
    """Property-based tests for strategies."""

    @given(market_snapshot_strategy())
    def test_strategy_always_returns_valid_signal(self, snapshot):
        """Test that strategies always return valid signals."""
        strategies = [
            always_long_strategy,
            always_short_strategy,
            always_hold_strategy,
            price_based_strategy,
            create_momentum_strategy(),
            create_mean_reversion_strategy(),
        ]

        for strategy in strategies:
            signal = strategy(snapshot)
            assert isinstance(signal, TradeSignal)
            assert isinstance(signal.signal, SignalType)
            assert 0.0 <= signal.strength <= 1.0
            assert isinstance(signal.reason, str)
            assert isinstance(signal.metadata, dict)

    @given(market_snapshot_strategy())
    def test_combinator_preserves_signal_validity(self, snapshot):
        """Test that combinators preserve signal validity."""
        base_strategy = create_momentum_strategy()

        # Test all combinators
        filtered = filter_strategy(base_strategy, lambda _: True)
        mapped = map_strategy(base_strategy, lambda s: s)
        thresholded = threshold_strategy(base_strategy, 0.5)

        for strategy in [filtered, mapped, thresholded]:
            signal = strategy(snapshot)
            assert isinstance(signal, TradeSignal)
            assert 0.0 <= signal.strength <= 1.0

    @given(st.lists(market_snapshot_strategy(), min_size=2, max_size=100))
    def test_backtest_invariants(self, snapshots):
        """Test backtest invariants hold for any data."""
        result = backtest_strategy(
            price_based_strategy, snapshots, initial_capital=10000.0
        )

        # Invariants
        assert result.total_trades == result.winning_trades + result.losing_trades
        assert 0 <= result.win_rate <= 1.0
        assert result.max_drawdown >= 0.0
        assert result.winning_trades >= 0
        assert result.losing_trades >= 0

        if result.total_trades > 0:
            assert result.win_rate == result.winning_trades / result.total_trades


class TestRiskManagement:
    """Test risk management aspects of strategies."""

    def test_position_sizing_in_strategies(self):
        """Test that strategies respect position sizing."""

        # Create a strategy with variable position sizing
        def risk_aware_strategy(snapshot: MarketSnapshot) -> TradeSignal:
            # Reduce position size in high volatility
            if snapshot.high_20 and snapshot.low_20:
                volatility = (
                    snapshot.high_20 - snapshot.low_20
                ) / snapshot.current_price
                size = max(0.1, 1.0 - volatility * 2)  # Reduce size with volatility
            else:
                size = 0.5

            return TradeSignal(
                signal=SignalType.LONG,
                strength=0.8,
                reason="Risk-adjusted position",
                metadata={"position_size": size},
            )

        # Low volatility
        low_vol_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
            high_20=41000.0,
            low_20=39000.0,
        )

        # High volatility
        high_vol_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
            high_20=50000.0,
            low_20=30000.0,
        )

        low_vol_signal = risk_aware_strategy(low_vol_snapshot)
        high_vol_signal = risk_aware_strategy(high_vol_snapshot)

        assert (
            low_vol_signal.metadata["position_size"]
            > high_vol_signal.metadata["position_size"]
        )

    def test_stop_loss_in_signal_metadata(self):
        """Test strategies can include risk parameters in metadata."""

        def risk_managed_strategy(snapshot: MarketSnapshot) -> TradeSignal:
            stop_loss = snapshot.current_price * 0.98  # 2% stop loss
            take_profit = snapshot.current_price * 1.03  # 3% take profit

            return TradeSignal(
                signal=SignalType.LONG,
                strength=0.7,
                reason="Risk managed entry",
                metadata={
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": 1.5,
                },
            )

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=40000.0,
            volume=1000.0,
        )

        signal = risk_managed_strategy(snapshot)
        assert "stop_loss" in signal.metadata
        assert "take_profit" in signal.metadata
        assert signal.metadata["stop_loss"] < snapshot.current_price
        assert signal.metadata["take_profit"] > snapshot.current_price


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio is calculated correctly."""
        # Create a strategy with known returns
        snapshots = []
        prices = [40000]

        # Generate steady upward trend (positive Sharpe)
        for i in range(1, 21):
            prices.append(prices[-1] * 1.01)  # 1% gain each period
            snapshots.append(
                MarketSnapshot(
                    timestamp=datetime.now(),
                    symbol="BTC-USD",
                    current_price=prices[-2],
                    volume=1000.0,
                )
            )

        result = backtest_strategy(
            always_long_strategy, snapshots, initial_capital=10000.0
        )

        # With steady positive returns, Sharpe should be positive
        assert result.sharpe_ratio > 0

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown is calculated correctly."""
        # Create price series with known drawdown
        prices = [40000, 45000, 50000, 35000, 40000]  # 30% drawdown
        snapshots = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                current_price=price,
                volume=1000.0,
            )
            for price in prices
        ]

        result = backtest_strategy(
            always_long_strategy, snapshots, initial_capital=10000.0
        )

        # Should have significant drawdown
        assert result.max_drawdown > 0.2  # At least 20%

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""

        # Create alternating wins and losses
        def win_loss_strategy(snapshot: MarketSnapshot) -> TradeSignal:
            # Alternate between long and short based on integer price
            if int(snapshot.current_price / 1000) % 2 == 0:
                return TradeSignal(
                    signal=SignalType.LONG, strength=0.8, reason="Even", metadata={}
                )
            return TradeSignal(
                signal=SignalType.SHORT, strength=0.8, reason="Odd", metadata={}
            )

        # Prices that create 2 wins, 1 loss pattern
        prices = [40000, 41000, 42000, 41000, 43000, 44000]
        snapshots = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                current_price=price,
                volume=1000.0,
            )
            for price in prices
        ]

        result = backtest_strategy(
            win_loss_strategy, snapshots, initial_capital=10000.0
        )

        if result.losing_trades > 0:
            assert result.profit_factor > 0
            # Profit factor should be total wins / total losses
            assert result.profit_factor == abs(
                (result.avg_win * result.winning_trades)
                / (result.avg_loss * result.losing_trades)
            )


# Integration tests
class TestStrategyIntegration:
    """Integration tests combining multiple components."""

    def test_complete_strategy_pipeline(self):
        """Test complete strategy evaluation pipeline."""
        # Create ensemble strategy
        strategies = [
            ("momentum", create_momentum_strategy(), 0.4),
            ("mean_rev", create_mean_reversion_strategy(), 0.3),
            ("price", price_based_strategy, 0.3),
        ]

        # Combine with risk filters
        ensemble = combine_strategies(strategies, "weighted_average")
        filtered = filter_strategy(ensemble, lambda s: s.volume > 500)
        final_strategy = threshold_strategy(filtered, min_strength=0.5)

        # Create test data
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_price=45000.0,
            volume=2000.0,
            sma_20=44000.0,
            high_20=46000.0,
            low_20=43000.0,
        )

        # Evaluate
        metadata = StrategyMetadata(
            name="Ensemble Strategy",
            version="1.0.0",
            description="Combined strategy with filters",
            parameters={"strategies": 3, "threshold": 0.5},
            risk_level="medium",
            expected_frequency="intraday",
            created_at=datetime.now(),
            tags=["ensemble", "filtered"],
        )

        result = evaluate_strategy(final_strategy, snapshot, metadata)

        assert isinstance(result.signal, TradeSignal)
        assert result.computation_time_ms > 0

    def test_multi_strategy_backtest_comparison(self):
        """Test comparing multiple strategies via backtest."""
        strategies = [
            ("momentum", create_momentum_strategy()),
            ("mean_reversion", create_mean_reversion_strategy()),
            ("always_long", always_long_strategy),
            ("price_based", price_based_strategy),
        ]

        # Generate test data with trend and reversals
        prices = []
        current = 40000.0
        for i in range(50):
            if i < 20:
                current *= 1.01  # Uptrend
            elif i < 30:
                current *= 0.99  # Downtrend
            else:
                current *= 1.005  # Slow recovery
            prices.append(current)

        snapshots = []
        for i, price in enumerate(prices):
            # Calculate simple indicators
            if i >= 20:
                sma_20 = sum(prices[i - 20 : i]) / 20
                high_20 = max(prices[i - 20 : i])
                low_20 = min(prices[i - 20 : i])
            else:
                sma_20 = high_20 = low_20 = None

            snapshots.append(
                MarketSnapshot(
                    timestamp=datetime.now(),
                    symbol="BTC-USD",
                    current_price=price,
                    volume=1000.0 + i * 10,
                    sma_20=sma_20,
                    high_20=high_20,
                    low_20=low_20,
                )
            )

        # Backtest all strategies
        results = {}
        for name, strategy in strategies:
            result = backtest_strategy(
                strategy,
                snapshots,
                initial_capital=10000.0,
                position_size=0.1,
                fees=0.001,
            )
            results[name] = result

        # Verify we got results for all strategies
        assert len(results) == len(strategies)

        # Different strategies should have different performance
        returns = [r.total_return for r in results.values()]
        assert len(set(returns)) > 1  # Not all the same


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
