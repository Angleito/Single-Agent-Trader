"""
Functional programming strategy types and combinators.

This module provides:
- Strategy type definition
- Strategy combinators
- Evaluation functions
- Strategy metadata
- Backtesting helpers
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bot.fp.types.market import MarketSnapshot
from bot.fp.types.trading import TradeSignal, Long, Short, Hold
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"

# Strategy type: A function that takes market data and returns a signal
Strategy = Callable[[MarketSnapshot], TradeSignal]


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for trading strategies."""
    
    name: str
    risk_level: str = "medium"
    max_position_size: float = 0.1
    stop_loss_percentage: float = 5.0
    take_profit_percentage: float = 15.0
    enabled: bool = True
    parameters: dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.parameters is None:
            object.__setattr__(self, "parameters", {})
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError(f"Position size must be between 0 and 1: {self.max_position_size}")
        if self.stop_loss_percentage <= 0:
            raise ValueError(f"Stop loss must be positive: {self.stop_loss_percentage}")
        if self.take_profit_percentage <= 0:
            raise ValueError(f"Take profit must be positive: {self.take_profit_percentage}")


@dataclass(frozen=True)
class BaseStrategy:
    """Base strategy class."""
    
    config: StrategyConfig
    strategy_func: Strategy
    metadata: 'StrategyMetadata'
    
    def evaluate(self, snapshot: MarketSnapshot) -> TradeSignal:
        """Evaluate strategy with market data."""
        return self.strategy_func(snapshot)


@dataclass(frozen=True)
class StrategyComposition:
    """Composition of multiple strategies."""
    
    strategies: list[BaseStrategy]
    aggregation_method: str = "weighted_average"
    weights: list[float] = None
    
    def __post_init__(self) -> None:
        """Validate composition."""
        if self.weights is None:
            # Equal weights
            equal_weight = 1.0 / len(self.strategies) if self.strategies else 0.0
            object.__setattr__(self, "weights", [equal_weight] * len(self.strategies))
        elif len(self.weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
        elif abs(sum(self.weights) - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")


@dataclass(frozen=True)
class StrategyMetadata:
    """Metadata for a trading strategy."""

    name: str
    version: str
    description: str
    parameters: dict[str, Any]
    risk_level: str  # "low", "medium", "high"
    expected_frequency: str  # "scalping", "intraday", "swing"
    created_at: datetime
    tags: list[str]


@dataclass(frozen=True)
class StrategyResult:
    """Result of strategy evaluation with metadata."""

    signal: TradeSignal
    metadata: StrategyMetadata
    computation_time_ms: float
    sub_signals: list[tuple[str, TradeSignal]] | None = None  # For combined strategies


# Strategy Combinators


def combine_strategies(
    strategies: list[tuple[str, Strategy, float]], aggregation: str = "weighted_average"
) -> Strategy:
    """
    Combine multiple strategies into one.

    Args:
        strategies: List of (name, strategy, weight) tuples
        aggregation: How to combine signals - "weighted_average", "majority", "unanimous"

    Returns:
        Combined strategy function
    """

    def combined_strategy(snapshot: MarketSnapshot) -> TradeSignal:
        signals_with_weights = [
            (name, strategy(snapshot), weight) for name, strategy, weight in strategies
        ]

        if aggregation == "weighted_average":
            return _weighted_average_signals(signals_with_weights)
        if aggregation == "majority":
            return _majority_vote_signals(signals_with_weights)
        if aggregation == "unanimous":
            return _unanimous_signals(signals_with_weights)
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return combined_strategy


def filter_strategy(
    strategy: Strategy, condition: Callable[[MarketSnapshot], bool]
) -> Strategy:
    """
    Apply a filter condition to a strategy.

    Args:
        strategy: Base strategy
        condition: Filter function

    Returns:
        Filtered strategy that returns HOLD when condition is False
    """

    def filtered_strategy(snapshot: MarketSnapshot) -> TradeSignal:
        if condition(snapshot):
            return strategy(snapshot)
        return Hold(reason="Filter condition not met")

    return filtered_strategy


def map_strategy(
    strategy: Strategy, transform: Callable[[TradeSignal], TradeSignal]
) -> Strategy:
    """
    Transform the output of a strategy.

    Args:
        strategy: Base strategy
        transform: Signal transformation function

    Returns:
        Strategy with transformed signals
    """

    def mapped_strategy(snapshot: MarketSnapshot) -> TradeSignal:
        signal = strategy(snapshot)
        return transform(signal)

    return mapped_strategy


def chain_strategies(strategies: list[Strategy]) -> Strategy:
    """
    Chain strategies - each strategy can override previous signals.

    Args:
        strategies: List of strategies in order of priority

    Returns:
        Chained strategy
    """

    def chained_strategy(snapshot: MarketSnapshot) -> TradeSignal:
        current_signal: TradeSignal = Hold(reason="No strategy triggered")

        for strategy in strategies:
            signal = strategy(snapshot)
            if not isinstance(signal, Hold):
                current_signal = signal

        return current_signal

    return chained_strategy


def threshold_strategy(strategy: Strategy, min_strength: float = 0.7) -> Strategy:
    """
    Only trigger signals above a strength threshold.

    Args:
        strategy: Base strategy
        min_strength: Minimum signal strength

    Returns:
        Thresholded strategy
    """

    def thresholded_strategy(snapshot: MarketSnapshot) -> TradeSignal:
        signal = strategy(snapshot)
        # For directional signals, check their strength/confidence
        if isinstance(signal, (Long, Short)):
            if signal.confidence >= min_strength:
                return signal
            return Hold(reason=f"Signal confidence {signal.confidence:.2f} below threshold {min_strength}")
        return signal

    return thresholded_strategy


# Evaluation Functions


def evaluate_strategy(
    strategy: Strategy, snapshot: MarketSnapshot, metadata: StrategyMetadata
) -> StrategyResult:
    """
    Evaluate a strategy and return result with metadata.

    Args:
        strategy: Strategy to evaluate
        snapshot: Market data
        metadata: Strategy metadata

    Returns:
        Strategy result with timing and metadata
    """
    import time

    start_time = time.time()
    signal = strategy(snapshot)
    end_time = time.time()

    return StrategyResult(
        signal=signal,
        metadata=metadata,
        computation_time_ms=(end_time - start_time) * 1000,
    )


def evaluate_multiple_strategies(
    strategies: list[tuple[Strategy, StrategyMetadata]], snapshot: MarketSnapshot
) -> list[StrategyResult]:
    """
    Evaluate multiple strategies in parallel.

    Args:
        strategies: List of (strategy, metadata) tuples
        snapshot: Market data

    Returns:
        List of strategy results
    """
    return [
        evaluate_strategy(strategy, snapshot, metadata)
        for strategy, metadata in strategies
    ]


# Backtesting Helpers


@dataclass(frozen=True)
class BacktestResult:
    """Result of strategy backtest."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float


def backtest_strategy(
    strategy: Strategy,
    historical_snapshots: list[MarketSnapshot],
    initial_capital: float = 10000.0,
    position_size: float = 0.1,
    fees: float = 0.001,
) -> BacktestResult:
    """
    Simple backtest of a strategy.

    Args:
        strategy: Strategy to test
        historical_snapshots: Historical market data
        initial_capital: Starting capital
        position_size: Fraction of capital per trade
        fees: Trading fees as fraction

    Returns:
        Backtest results
    """
    # This is a simplified backtest - full implementation would track positions
    trades = []
    capital = initial_capital

    for i, snapshot in enumerate(historical_snapshots):
        signal = strategy(snapshot)

        if signal.signal != SignalType.HOLD:
            # Simplified trade simulation
            trade_size = capital * position_size
            trade_fees = trade_size * fees

            # Simulate trade outcome (simplified)
            if i < len(historical_snapshots) - 1:
                next_price = historical_snapshots[i + 1].current_price
                price_change = (
                    next_price - snapshot.current_price
                ) / snapshot.current_price

                if signal.signal == SignalType.SHORT:
                    price_change = -price_change

                profit = trade_size * price_change - trade_fees
                capital += profit
                trades.append(profit)

    if not trades:
        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
        )

    winning_trades = [t for t in trades if t > 0]
    losing_trades = [t for t in trades if t < 0]

    return BacktestResult(
        total_trades=len(trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        total_return=(capital - initial_capital) / initial_capital,
        sharpe_ratio=_calculate_sharpe_ratio(trades),
        max_drawdown=_calculate_max_drawdown(trades, initial_capital),
        win_rate=len(winning_trades) / len(trades) if trades else 0,
        avg_win=sum(winning_trades) / len(winning_trades) if winning_trades else 0,
        avg_loss=sum(losing_trades) / len(losing_trades) if losing_trades else 0,
        profit_factor=(
            abs(sum(winning_trades) / sum(losing_trades))
            if losing_trades
            else float("inf")
        ),
    )


# Helper functions


def _weighted_average_signals(
    signals_with_weights: list[tuple[str, TradeSignal, float]],
) -> TradeSignal:
    """Combine signals using weighted average."""
    total_weight = sum(weight for _, _, weight in signals_with_weights)

    # Calculate weighted strength
    weighted_strength = sum(
        signal.strength * weight / total_weight
        for _, signal, weight in signals_with_weights
    )

    # Determine signal type based on weighted strength
    if weighted_strength > 0.1:
        signal_type = SignalType.LONG
    elif weighted_strength < -0.1:
        signal_type = SignalType.SHORT
    else:
        signal_type = SignalType.HOLD

    reasons = [
        f"{name}: {signal.signal.value} ({signal.strength:.2f})"
        for name, signal, _ in signals_with_weights
    ]

    return TradeSignal(
        signal=signal_type,
        strength=abs(weighted_strength),
        reason=f"Weighted average: {', '.join(reasons)}",
        metadata={
            "aggregation": "weighted_average",
            "sub_signals": [
                (name, signal.signal.value, signal.strength)
                for name, signal, _ in signals_with_weights
            ],
        },
    )


def _majority_vote_signals(
    signals_with_weights: list[tuple[str, TradeSignal, float]],
) -> TradeSignal:
    """Combine signals using majority vote."""
    votes = {"LONG": 0, "SHORT": 0, "HOLD": 0}

    for _, signal, weight in signals_with_weights:
        votes[signal.signal.value] += weight

    # Find winning signal
    winning_signal = max(votes.items(), key=lambda x: x[1])
    signal_type = SignalType(winning_signal[0])

    # Calculate strength as vote proportion
    total_votes = sum(votes.values())
    strength = winning_signal[1] / total_votes if total_votes > 0 else 0

    return TradeSignal(
        signal=signal_type,
        strength=strength,
        reason=f"Majority vote: {winning_signal[0]} ({winning_signal[1]:.1f}/{total_votes:.1f})",
        metadata={"aggregation": "majority", "votes": votes},
    )


def _unanimous_signals(
    signals_with_weights: list[tuple[str, TradeSignal, float]],
) -> TradeSignal:
    """Only trigger if all strategies agree."""
    signals = [signal.signal for _, signal, _ in signals_with_weights]

    if all(s == signals[0] for s in signals) and signals[0] != SignalType.HOLD:
        # All agree on non-HOLD signal
        avg_strength = sum(
            signal.strength for _, signal, _ in signals_with_weights
        ) / len(signals_with_weights)
        return TradeSignal(
            signal=signals[0],
            strength=avg_strength,
            reason=f"Unanimous {signals[0].value} signal",
            metadata={"aggregation": "unanimous"},
        )
    # Not unanimous
    return TradeSignal(
        signal=SignalType.HOLD,
        strength=0.0,
        reason="Strategies not unanimous",
        metadata={
            "aggregation": "unanimous",
            "signals": [signal.signal.value for _, signal, _ in signals_with_weights],
        },
    )


def _calculate_sharpe_ratio(
    returns: list[float], risk_free_rate: float = 0.02
) -> float:
    """Calculate Sharpe ratio from returns."""
    if not returns:
        return 0.0

    import numpy as np

    returns_array = np.array(returns)
    avg_return = np.mean(returns_array)
    std_return = np.std(returns_array)

    if std_return == 0:
        return 0.0

    return (avg_return - risk_free_rate) / std_return


def _calculate_max_drawdown(returns: list[float], initial_capital: float) -> float:
    """Calculate maximum drawdown from returns."""
    if not returns:
        return 0.0

    capital = initial_capital
    peak = capital
    max_dd = 0.0

    for ret in returns:
        capital += ret
        peak = max(peak, capital)
        drawdown = (peak - capital) / peak
        max_dd = max(max_dd, drawdown)

    return max_dd


# Strategy factories for common patterns


def create_momentum_strategy(lookback: int = 20, threshold: float = 0.02) -> Strategy:
    """Create a simple momentum strategy."""

    def momentum_strategy(snapshot: MarketSnapshot) -> TradeSignal:
        if snapshot.high_20 is None or snapshot.low_20 is None:
            return TradeSignal(
                signal=SignalType.HOLD,
                strength=0.0,
                reason="Insufficient data for momentum calculation",
                metadata={},
            )

        # Simple momentum: compare current price to range
        price_position = (snapshot.current_price - snapshot.low_20) / (
            snapshot.high_20 - snapshot.low_20
        )

        if price_position > (1 - threshold):
            return TradeSignal(
                signal=SignalType.LONG,
                strength=price_position,
                reason=f"Strong upward momentum (position: {price_position:.2f})",
                metadata={"strategy": "momentum", "lookback": lookback},
            )
        if price_position < threshold:
            return TradeSignal(
                signal=SignalType.SHORT,
                strength=1 - price_position,
                reason=f"Strong downward momentum (position: {price_position:.2f})",
                metadata={"strategy": "momentum", "lookback": lookback},
            )
        return TradeSignal(
            signal=SignalType.HOLD,
            strength=0.0,
            reason="No clear momentum signal",
            metadata={"strategy": "momentum", "price_position": price_position},
        )

    return momentum_strategy


def create_mean_reversion_strategy(
    lookback: int = 20, z_score_threshold: float = 2.0
) -> Strategy:
    """Create a mean reversion strategy."""

    def mean_reversion_strategy(snapshot: MarketSnapshot) -> TradeSignal:
        if snapshot.sma_20 is None:
            return TradeSignal(
                signal=SignalType.HOLD,
                strength=0.0,
                reason="Insufficient data for mean reversion",
                metadata={},
            )

        # Calculate z-score
        price_range = (
            snapshot.high_20 - snapshot.low_20
            if snapshot.high_20 and snapshot.low_20
            else 0
        )
        if price_range == 0:
            return TradeSignal(
                signal=SignalType.HOLD,
                strength=0.0,
                reason="No price variation",
                metadata={},
            )

        z_score = (snapshot.current_price - snapshot.sma_20) / (
            price_range / 4
        )  # Approximate std

        if z_score > z_score_threshold:
            return TradeSignal(
                signal=SignalType.SHORT,
                strength=min(abs(z_score) / 3, 1.0),
                reason=f"Overbought (z-score: {z_score:.2f})",
                metadata={"strategy": "mean_reversion", "z_score": z_score},
            )
        if z_score < -z_score_threshold:
            return TradeSignal(
                signal=SignalType.LONG,
                strength=min(abs(z_score) / 3, 1.0),
                reason=f"Oversold (z-score: {z_score:.2f})",
                metadata={"strategy": "mean_reversion", "z_score": z_score},
            )
        return TradeSignal(
            signal=SignalType.HOLD,
            strength=0.0,
            reason="Within normal range",
            metadata={"strategy": "mean_reversion", "z_score": z_score},
        )

    return mean_reversion_strategy
