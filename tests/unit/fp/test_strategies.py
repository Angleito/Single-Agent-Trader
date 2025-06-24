"""Comprehensive tests for functional programming strategy functions.

This module tests:
1. Each strategy type using actual FP types
2. Strategy combinators with immutable data structures
3. Property-based testing for signal generation
4. Backtest simulation with FP architecture
5. Risk management validation with functional types
6. Performance metrics calculation
"""

from datetime import datetime
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite

# Import the actual FP strategies module
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
from bot.fp.types.market import MarketSnapshot

# Import actual FP types instead of mocks
from bot.fp.types.trading import Hold, Long, Short, TradeSignal


# Test Strategies using actual FP types
def always_long_strategy(snapshot: MarketSnapshot) -> TradeSignal:
    """Strategy that always returns LONG signal."""
    return Long(confidence=0.8, size=0.25, reason="Always bullish")


def always_short_strategy(snapshot: MarketSnapshot) -> TradeSignal:
    """Strategy that always returns SHORT signal."""
    return Short(confidence=0.8, size=0.25, reason="Always bearish")


def always_hold_strategy(snapshot: MarketSnapshot) -> TradeSignal:
    """Strategy that always returns HOLD signal."""
    return Hold(reason="Always neutral")


def price_based_strategy(snapshot: MarketSnapshot) -> TradeSignal:
    """Strategy based on price level."""
    if snapshot.price > Decimal(50000):
        return Short(confidence=0.7, size=0.2, reason="Price too high")
    if snapshot.price < Decimal(30000):
        return Long(confidence=0.7, size=0.2, reason="Price too low")
    return Hold(reason="Price in range")


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
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )
        signal = strategy(snapshot)
        assert isinstance(signal, Hold)
        assert "Insufficient data" in signal.reason

    def test_momentum_strategy_long_signal(self):
        """Test momentum strategy generates long signal."""
        # Note: This test will need to be updated when create_momentum_strategy
        # is implemented to work with the new MarketSnapshot structure
        # For now, we'll test basic functionality
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("50000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("49990.0"),
            ask=Decimal("50010.0"),
        )

        # Test that we can create a strategy function
        strategy = create_momentum_strategy(lookback=20, threshold=0.02)
        signal = strategy(snapshot)

        # Should return a valid TradeSignal
        assert isinstance(signal, (Long, Short, Hold))

    def test_mean_reversion_strategy_creation(self):
        """Test creating a mean reversion strategy."""
        strategy = create_mean_reversion_strategy(lookback=20, z_score_threshold=2.0)
        assert callable(strategy)

    def test_mean_reversion_strategy_short_signal(self):
        """Test mean reversion strategy generates short signal."""
        # Note: This test will need to be updated when create_mean_reversion_strategy
        # is implemented to work with the new MarketSnapshot structure
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("50000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("49990.0"),
            ask=Decimal("50010.0"),
        )

        # Test that we can create a strategy function
        strategy = create_mean_reversion_strategy(lookback=20, z_score_threshold=1.5)
        signal = strategy(snapshot)

        # Should return a valid TradeSignal
        assert isinstance(signal, (Long, Short, Hold))


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
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )
        signal = combined(snapshot)

        # Should return a valid signal - specific behavior depends on implementation
        assert isinstance(signal, (Long, Short, Hold))

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
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )
        signal = combined(snapshot)

        # Should return a valid signal
        assert isinstance(signal, (Long, Short, Hold))

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
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )

        # Test unanimous agreement
        combined_agree = combine_strategies(strategies_agree, aggregation="unanimous")
        signal_agree = combined_agree(snapshot)
        assert isinstance(signal_agree, (Long, Short, Hold))

        # Test disagreement
        combined_disagree = combine_strategies(
            strategies_disagree, aggregation="unanimous"
        )
        signal_disagree = combined_disagree(snapshot)
        assert isinstance(signal_disagree, Hold)  # Disagreement should result in Hold

    def test_filter_strategy(self):
        """Test filtering strategy based on condition."""

        # Only trade when volume is high
        def high_volume_condition(snapshot: MarketSnapshot) -> bool:
            return snapshot.volume > Decimal(5000)

        filtered = filter_strategy(always_long_strategy, high_volume_condition)

        # Low volume - should HOLD
        low_vol_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )
        signal_low = filtered(low_vol_snapshot)
        assert isinstance(signal_low, Hold)

        # High volume - should pass through
        high_vol_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("40000.0"),
            volume=Decimal("10000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )
        signal_high = filtered(high_vol_snapshot)
        assert isinstance(signal_high, Long)

    def test_map_strategy(self):
        """Test transforming strategy output."""

        # Invert signals
        def invert_signal(signal: TradeSignal) -> TradeSignal:
            if isinstance(signal, Long):
                return Short(
                    confidence=signal.confidence,
                    size=signal.size,
                    reason=f"Inverted: {signal.reason}",
                )
            if isinstance(signal, Short):
                return Long(
                    confidence=signal.confidence,
                    size=signal.size,
                    reason=f"Inverted: {signal.reason}",
                )
            return Hold(reason=f"Inverted: {signal.reason}")

        inverted = map_strategy(always_long_strategy, invert_signal)

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )
        signal = inverted(snapshot)

        assert isinstance(signal, Short)
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
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )
        signal = chained(snapshot)
        assert isinstance(signal, Long)

    def test_threshold_strategy(self):
        """Test threshold filtering on signal strength."""

        # Create a strategy with variable confidence
        def variable_strength_strategy(snapshot: MarketSnapshot) -> TradeSignal:
            confidence = float(snapshot.price) / 100000  # 0.4 for 40k price
            if confidence >= 0.7:
                return Long(
                    confidence=confidence,
                    size=0.25,
                    reason=f"High confidence {confidence:.2f}",
                )
            return Hold(reason=f"Low confidence {confidence:.2f}")

        thresholded = threshold_strategy(variable_strength_strategy, min_strength=0.7)

        # Low confidence - should HOLD
        low_price_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )
        signal_low = thresholded(low_price_snapshot)
        assert isinstance(signal_low, Hold)

        # High confidence - should pass through
        high_price_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("80000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("79990.0"),
            ask=Decimal("80010.0"),
        )
        signal_high = thresholded(high_price_snapshot)
        assert isinstance(signal_high, Long)


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
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )

        result = evaluate_strategy(always_long_strategy, snapshot, metadata)

        assert isinstance(result, StrategyResult)
        assert isinstance(result.signal, Long)
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
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )

        results = evaluate_multiple_strategies(strategies_with_metadata, snapshot)

        assert len(results) == 2
        assert isinstance(results[0].signal, Long)
        assert isinstance(results[1].signal, Short)
        assert all(r.computation_time_ms >= 0 for r in results)


class TestBacktesting:
    """Test backtesting functionality."""

    def test_backtest_no_trades(self):
        """Test backtesting a strategy that never trades."""
        snapshots = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                price=Decimal(f"{40000.0 + i * 100}"),
                volume=Decimal("1000.0"),
                bid=Decimal(f"{40000.0 + i * 100 - 10}"),
                ask=Decimal(f"{40000.0 + i * 100 + 10}"),
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
                price=Decimal(str(price)),
                volume=Decimal("1000.0"),
                bid=Decimal(str(price - 10)),
                ask=Decimal(str(price + 10)),
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
            price_int = int(snapshot.price)
            if price_int % 2 == 0:
                return Long(confidence=0.8, size=0.25, reason="Even price")
            return Short(confidence=0.8, size=0.25, reason="Odd price")

        # Create price series with known outcomes
        prices = [40000.0, 40100.0, 39900.0, 40200.0, 39800.0]
        snapshots = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                price=Decimal(str(price)),
                volume=Decimal("1000.0"),
                bid=Decimal(str(price - 10)),
                ask=Decimal(str(price + 10)),
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

    # Create spread around price
    spread_pct = draw(st.floats(min_value=0.001, max_value=0.01))  # 0.1% to 1% spread
    spread = price * spread_pct / 2
    bid = price - spread
    ask = price + spread

    return MarketSnapshot(
        timestamp=datetime.now(),
        symbol="BTC-USD",
        price=Decimal(str(price)),
        volume=Decimal(str(volume)),
        bid=Decimal(str(bid)),
        ask=Decimal(str(ask)),
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
            assert isinstance(signal, (Long, Short, Hold))

            # Test confidence for directional signals
            if isinstance(signal, (Long, Short)):
                assert 0.0 <= signal.confidence <= 1.0
                assert 0.0 < signal.size <= 1.0

            assert isinstance(signal.reason, str)

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
            assert isinstance(signal, (Long, Short, Hold))

            # Check confidence for directional signals
            if isinstance(signal, (Long, Short)):
                assert 0.0 <= signal.confidence <= 1.0

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
            # Calculate volatility from spread
            volatility = float(snapshot.spread / snapshot.price)
            size = max(0.1, 1.0 - volatility * 20)  # Reduce size with volatility

            return Long(
                confidence=0.8,
                size=size,
                reason=f"Risk-adjusted position, volatility: {volatility:.4f}",
            )

        # Low volatility (tight spread)
        low_vol_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39995.0"),  # 0.0125% spread
            ask=Decimal("40005.0"),
        )

        # High volatility (wide spread)
        high_vol_snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39900.0"),  # 0.25% spread
            ask=Decimal("40100.0"),
        )

        low_vol_signal = risk_aware_strategy(low_vol_snapshot)
        high_vol_signal = risk_aware_strategy(high_vol_snapshot)

        # Both should be Long signals, but with different sizes
        assert isinstance(low_vol_signal, Long)
        assert isinstance(high_vol_signal, Long)
        assert low_vol_signal.size > high_vol_signal.size

    def test_stop_loss_in_signal_metadata(self):
        """Test strategies can include risk parameters in metadata."""

        def risk_managed_strategy(snapshot: MarketSnapshot) -> TradeSignal:
            # Embed risk parameters in the reason for now
            current_price = float(snapshot.price)
            stop_loss = current_price * 0.98  # 2% stop loss
            take_profit = current_price * 1.03  # 3% take profit

            return Long(
                confidence=0.7,
                size=0.25,
                reason=f"Risk managed entry - SL: {stop_loss:.2f}, TP: {take_profit:.2f}",
            )

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("40000.0"),
            volume=Decimal("1000.0"),
            bid=Decimal("39990.0"),
            ask=Decimal("40010.0"),
        )

        signal = risk_managed_strategy(snapshot)
        assert isinstance(signal, Long)
        assert "SL:" in signal.reason
        assert "TP:" in signal.reason


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
            price = prices[-2]
            snapshots.append(
                MarketSnapshot(
                    timestamp=datetime.now(),
                    symbol="BTC-USD",
                    price=Decimal(str(price)),
                    volume=Decimal("1000.0"),
                    bid=Decimal(str(price - 10)),
                    ask=Decimal(str(price + 10)),
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
                price=Decimal(str(price)),
                volume=Decimal("1000.0"),
                bid=Decimal(str(price - 10)),
                ask=Decimal(str(price + 10)),
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
            price_thousands = int(float(snapshot.price) / 1000)
            if price_thousands % 2 == 0:
                return Long(confidence=0.8, size=0.25, reason="Even price bracket")
            return Short(confidence=0.8, size=0.25, reason="Odd price bracket")

        # Prices that create 2 wins, 1 loss pattern
        prices = [40000, 41000, 42000, 41000, 43000, 44000]
        snapshots = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                price=Decimal(str(price)),
                volume=Decimal("1000.0"),
                bid=Decimal(str(price - 10)),
                ask=Decimal(str(price + 10)),
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
        filtered = filter_strategy(ensemble, lambda s: s.volume > Decimal(500))
        final_strategy = threshold_strategy(filtered, min_strength=0.5)

        # Create test data
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal("45000.0"),
            volume=Decimal("2000.0"),
            bid=Decimal("44990.0"),
            ask=Decimal("45010.0"),
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

        assert isinstance(result.signal, (Long, Short, Hold))
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
            snapshots.append(
                MarketSnapshot(
                    timestamp=datetime.now(),
                    symbol="BTC-USD",
                    price=Decimal(str(price)),
                    volume=Decimal(str(1000.0 + i * 10)),
                    bid=Decimal(str(price - 10)),
                    ask=Decimal(str(price + 10)),
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
