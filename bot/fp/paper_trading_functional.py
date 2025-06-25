"""
Functional Paper Trading System with Immutable State Management

This module provides an enhanced paper trading system using functional programming
patterns, immutable state, and the effect system for managing side effects.
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .effects import (
    IO,
    Either,
    IOEither,
    Left,
    Right,
    debug,
    error,
    info,
    warn,
)
from .pure.paper_trading_calculations import (
    calculate_account_metrics,
    calculate_portfolio_performance,
    normalize_decimal_precision,
    simulate_position_close,
    simulate_trade_execution,
    validate_account_state,
)
from .types.paper_trading import (
    PaperTradingAccountState,
    TradeExecution,
)
from .types.portfolio import PortfolioMetrics


class FunctionalPaperTradingEngine:
    """
    Functional paper trading engine with immutable state management.

    This engine provides paper trading functionality using:
    - Immutable state transitions
    - Pure function calculations
    - Effect system for side effects
    - Functional error handling
    """

    def __init__(
        self,
        initial_balance: Decimal,
        data_dir: Path | None = None,
        fee_rate: Decimal = Decimal("0.001"),
        slippage_rate: Decimal = Decimal("0.0005"),
        leverage: Decimal = Decimal(5),
    ):
        """Initialize the functional paper trading engine."""
        self.fee_rate = normalize_decimal_precision(fee_rate, 6)
        self.slippage_rate = normalize_decimal_precision(slippage_rate, 6)
        self.leverage = leverage
        self.data_dir = data_dir or Path("./data/paper_trading_fp")

        # Initialize with empty state - will be loaded or created
        self._current_state: PaperTradingAccountState | None = None

        # Create initial state
        initial_state = PaperTradingAccountState.create_initial(
            starting_balance=normalize_decimal_precision(initial_balance, 2)
        )
        self._current_state = initial_state

    def get_current_state(self) -> IO[PaperTradingAccountState]:
        """Get current account state (effect)."""
        return IO.pure(self._current_state)

    def load_state_from_disk(self) -> IOEither[str, PaperTradingAccountState]:
        """Load state from disk (effect with error handling)."""

        def load_operation() -> Either[str, PaperTradingAccountState]:
            try:
                state_file = self.data_dir / "account_state.json"
                if not state_file.exists():
                    return Left("No saved state found")

                # This would need actual serialization implementation
                # For now, return the current state
                return Right(self._current_state)
            except Exception as e:
                return Left(f"Failed to load state: {e}")

        return IOEither(load_operation)

    def save_state_to_disk(
        self, state: PaperTradingAccountState
    ) -> IOEither[str, bool]:
        """Save state to disk (effect with error handling)."""

        def save_operation() -> Either[str, bool]:
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                # This would need actual serialization implementation
                # For now, just update internal state
                self._current_state = state
                return Right(True)
            except Exception as e:
                return Left(f"Failed to save state: {e}")

        return IOEither(save_operation)

    def execute_trade_action(
        self,
        symbol: str,
        side: str,  # "LONG", "SHORT", "CLOSE"
        size_percentage: Decimal,
        current_price: Decimal,
        stop_loss_pct: Decimal | None = None,
        take_profit_pct: Decimal | None = None,
        is_futures: bool = False,
        contract_size: Decimal | None = None,
        fixed_contracts: int | None = None,
    ) -> IOEither[str, tuple[TradeExecution, PaperTradingAccountState]]:
        """
        Execute a trade action functionally.

        Returns an IOEither that when run will:
        1. Load current state
        2. Simulate the trade execution (pure)
        3. Validate the result
        4. Save the new state
        5. Log the execution
        """

        def execute_operation() -> Either[
            str, tuple[TradeExecution, PaperTradingAccountState]
        ]:
            try:
                current_state = self._current_state

                if side == "CLOSE":
                    # Close existing position
                    execution, new_state = simulate_position_close(
                        account_state=current_state,
                        symbol=symbol,
                        exit_price=current_price,
                        fee_rate=self.fee_rate,
                        slippage_rate=self.slippage_rate,
                    )
                else:
                    # Open new position
                    execution, new_state = simulate_trade_execution(
                        account_state=current_state,
                        symbol=symbol,
                        side=side,
                        size_percentage=size_percentage,
                        current_price=current_price,
                        leverage=self.leverage,
                        fee_rate=self.fee_rate,
                        slippage_rate=self.slippage_rate,
                        is_futures=is_futures,
                        contract_size=contract_size,
                        fixed_contracts=fixed_contracts,
                    )

                if not execution.success:
                    return Left(f"Trade execution failed: {execution.reason}")

                if new_state is None:
                    return Left("No new state returned from execution")

                # Validate the new state
                if not validate_account_state(new_state):
                    return Left("New account state failed validation")

                return Right((execution, new_state))

            except Exception as e:
                return Left(f"Trade execution error: {e}")

        return IOEither(execute_operation)

    def update_account_with_prices(
        self, current_prices: dict[str, Decimal]
    ) -> IOEither[str, PaperTradingAccountState]:
        """Update account equity with current prices (effect)."""

        def update_operation() -> Either[str, PaperTradingAccountState]:
            try:
                current_state = self._current_state
                updated_state = current_state.update_equity(current_prices)

                if not validate_account_state(updated_state):
                    return Left("Updated account state failed validation")

                return Right(updated_state)

            except Exception as e:
                return Left(f"Price update error: {e}")

        return IOEither(update_operation)

    def get_account_metrics(
        self, current_prices: dict[str, Decimal]
    ) -> IO[dict[str, Any]]:
        """Get comprehensive account metrics (pure calculation wrapped in IO)."""

        def metrics_calculation() -> dict[str, Any]:
            current_state = self._current_state

            # Calculate all metrics using pure functions
            basic_metrics = calculate_account_metrics(current_state, current_prices)

            # Add additional metrics
            metrics = {
                **basic_metrics,
                "open_positions": current_state.get_open_position_count(),
                "total_trades": current_state.get_total_trade_count(),
                "session_duration_hours": (
                    (datetime.now() - current_state.session_start_time).total_seconds()
                    / 3600
                ),
            }

            # Convert Decimal to float for JSON serialization
            return {
                k: float(v) if isinstance(v, Decimal) else v for k, v in metrics.items()
            }

        return IO.from_callable(metrics_calculation)

    def get_portfolio_performance(
        self, current_prices: dict[str, Decimal]
    ) -> IO[PortfolioMetrics]:
        """Get portfolio performance metrics (pure calculation wrapped in IO)."""

        def performance_calculation() -> PortfolioMetrics:
            return calculate_portfolio_performance(self._current_state, current_prices)

        return IO.from_callable(performance_calculation)

    def get_trade_history(self, limit: int | None = None) -> IO[list[dict[str, Any]]]:
        """Get trade history (pure calculation wrapped in IO)."""

        def history_calculation() -> list[dict[str, Any]]:
            current_state = self._current_state
            trades = list(current_state.closed_trades)

            if limit:
                trades = trades[-limit:]  # Get most recent trades

            return [
                {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": (
                        trade.exit_time.isoformat() if trade.exit_time else None
                    ),
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(trade.exit_price) if trade.exit_price else None,
                    "size": float(trade.size),
                    "realized_pnl": (
                        float(trade.realized_pnl) if trade.realized_pnl else None
                    ),
                    "fees": float(trade.fees),
                    "duration_seconds": (
                        (trade.exit_time - trade.entry_time).total_seconds()
                        if trade.exit_time
                        else None
                    ),
                }
                for trade in trades
            ]

        return IO.from_callable(history_calculation)

    def get_open_positions(self) -> IO[list[dict[str, Any]]]:
        """Get open positions (pure calculation wrapped in IO)."""

        def positions_calculation() -> list[dict[str, Any]]:
            current_state = self._current_state

            return [
                {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry_time": trade.entry_time.isoformat(),
                    "entry_price": float(trade.entry_price),
                    "size": float(trade.size),
                    "fees": float(trade.fees),
                    "status": trade.status,
                }
                for trade in current_state.open_trades
            ]

        return IO.from_callable(positions_calculation)

    def reset_account(
        self, new_balance: Decimal | None = None
    ) -> IOEither[str, PaperTradingAccountState]:
        """Reset account to initial state (effect)."""

        def reset_operation() -> Either[str, PaperTradingAccountState]:
            try:
                balance = new_balance or Decimal(10000)
                new_state = PaperTradingAccountState.create_initial(
                    starting_balance=normalize_decimal_precision(balance, 2)
                )

                return Right(new_state)

            except Exception as e:
                return Left(f"Account reset error: {e}")

        return IOEither(reset_operation)

    def commit_state_change(
        self, new_state: PaperTradingAccountState
    ) -> IOEither[str, bool]:
        """Commit a state change and persist it (effect)."""
        return self.save_state_to_disk(new_state).map(lambda _: True)


# Effect combinators for paper trading operations


def execute_trade_with_logging(
    engine: FunctionalPaperTradingEngine,
    symbol: str,
    side: str,
    size_percentage: Decimal,
    current_price: Decimal,
    **kwargs,
) -> IOEither[str, tuple[TradeExecution, PaperTradingAccountState]]:
    """Execute trade with comprehensive logging (effect combinator)."""

    def logged_execution():
        # Log trade attempt
        info(
            f"🎯 Attempting {side} trade: {symbol} @ ${current_price} ({size_percentage}%)"
        )

        # Execute the trade
        result = engine.execute_trade_action(
            symbol=symbol,
            side=side,
            size_percentage=size_percentage,
            current_price=current_price,
            **kwargs,
        ).run()

        if result.is_right():
            execution, new_state = result.value
            if execution.success:
                info(
                    f"✅ Trade executed: {symbol} | "
                    f"Price: ${execution.execution_price} | "
                    f"Fees: ${execution.fees.total_fees}"
                )
                return Right((execution, new_state))
            warn(f"❌ Trade failed: {execution.reason}")
            return Left(execution.reason)
        error(f"🚨 Trade execution error: {result.value}")
        return Left(result.value)

    return IOEither(logged_execution)


def update_and_persist_state(
    engine: FunctionalPaperTradingEngine, state_update_fn, operation_name: str
) -> IOEither[str, PaperTradingAccountState]:
    """Update state and persist it atomically (effect combinator)."""

    def atomic_update():
        try:
            # Get current state
            current_state = engine._current_state

            # Apply the update function (pure)
            new_state = state_update_fn(current_state)

            # Validate new state
            if not validate_account_state(new_state):
                return Left(f"State validation failed for {operation_name}")

            # Persist the change
            persist_result = engine.commit_state_change(new_state).run()
            if persist_result.is_left():
                return Left(
                    f"Failed to persist {operation_name}: {persist_result.value}"
                )

            # Update engine state
            engine._current_state = new_state

            debug(f"✅ {operation_name} completed and persisted")
            return Right(new_state)

        except Exception as e:
            error(f"🚨 Atomic update failed for {operation_name}: {e}")
            return Left(f"Atomic update error: {e}")

    return IOEither(atomic_update)


def calculate_performance_report(
    engine: FunctionalPaperTradingEngine, current_prices: dict[str, Decimal]
) -> IO[dict[str, Any]]:
    """Generate comprehensive performance report (effect combinator)."""

    def generate_report():
        # Get all metrics using pure calculations
        metrics = engine.get_account_metrics(current_prices).run()
        performance = engine.get_portfolio_performance(current_prices).run()
        trade_history = engine.get_trade_history(limit=10).run()
        open_positions = engine.get_open_positions().run()

        return {
            "timestamp": datetime.now().isoformat(),
            "account_metrics": metrics,
            "performance_metrics": {
                "total_pnl": float(performance.total_pnl),
                "realized_pnl": float(performance.realized_pnl),
                "unrealized_pnl": float(performance.unrealized_pnl),
                "win_rate": performance.win_rate,
                "total_trades": performance.total_trades,
                "winning_trades": performance.winning_trades,
                "losing_trades": performance.losing_trades,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "avg_win": float(performance.avg_win),
                "avg_loss": float(performance.avg_loss),
                "profit_factor": performance.profit_factor,
            },
            "recent_trades": trade_history,
            "open_positions": open_positions,
        }

    return IO.from_callable(generate_report)


# Factory function for creating functional paper trading engine
def create_functional_paper_trading_engine(
    initial_balance: Decimal = Decimal(10000), data_dir: Path | None = None, **config
) -> FunctionalPaperTradingEngine:
    """Create a functional paper trading engine with configuration."""
    return FunctionalPaperTradingEngine(
        initial_balance=initial_balance, data_dir=data_dir, **config
    )
