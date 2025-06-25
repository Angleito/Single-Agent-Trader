"""
Backtesting Runtime for Functional Trading Bot

This module provides functional backtesting capabilities with
historical data replay and performance analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from bot.fp.effects.io import IO
from bot.fp.effects.logging import info, warn
from bot.fp.effects.persistence import Event, save_event


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    start_date: datetime
    end_date: datetime
    symbol: str = "BTC-USD"
    initial_balance: Decimal = Decimal(10000)
    commission_rate: Decimal = Decimal("0.001")
    slippage_rate: Decimal = Decimal("0.0001")


@dataclass
class BacktestResult:
    """Backtesting results"""

    total_return: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: Decimal
    sharpe_ratio: float
    profit_factor: float
    win_rate: float


@dataclass
class Trade:
    """A completed trade"""

    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    side: str  # "long" or "short"
    pnl: Decimal
    commission: Decimal


class BacktestEngine:
    """Functional backtesting engine"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.current_time = config.start_date
        self.balance = config.initial_balance
        self.positions: list[dict[str, Any]] = []
        self.completed_trades: list[Trade] = []
        self.portfolio_values: list[Tuple[datetime, Decimal]] = []

    def load_historical_data(self) -> IO[list[dict[str, Any]]]:
        """Load historical market data for backtesting"""

        def load():
            # Simulate loading historical data
            data = []
            current = self.config.start_date

            while current <= self.config.end_date:
                # Generate synthetic price data
                base_price = Decimal(50000)
                variance = Decimal(1000)

                price = base_price + (
                    Decimal(str(hash(current) % 1000)) - Decimal(500)
                ) * variance / Decimal(500)

                data.append(
                    {
                        "timestamp": current,
                        "open": price,
                        "high": price * Decimal("1.01"),
                        "low": price * Decimal("0.99"),
                        "close": price,
                        "volume": Decimal(100),
                    }
                )

                current += timedelta(minutes=1)

            info(
                f"Loaded {len(data)} historical data points",
                {
                    "symbol": self.config.symbol,
                    "start": self.config.start_date.isoformat(),
                    "end": self.config.end_date.isoformat(),
                },
            ).run()

            return data

        return IO(load)

    def simulate_strategy(self, data: list[dict[str, Any]]) -> IO[None]:
        """Simulate trading strategy execution"""

        def simulate():
            info("Starting backtest simulation").run()

            for i, candle in enumerate(data):
                self.current_time = candle["timestamp"]
                current_price = candle["close"]

                # Record portfolio value
                portfolio_value = self.calculate_portfolio_value(current_price)
                self.portfolio_values.append((self.current_time, portfolio_value))

                # Simple strategy: Buy when price is low, sell when high
                if i > 50:  # Need some history
                    recent_prices = [d["close"] for d in data[i - 50 : i]]
                    avg_price = sum(recent_prices) / len(recent_prices)

                    if (
                        current_price < avg_price * Decimal("0.95")
                        and not self.positions
                    ):
                        # Buy signal
                        self.enter_position(
                            "long", current_price, self.balance * Decimal("0.1")
                        )

                    elif current_price > avg_price * Decimal("1.05") and self.positions:
                        # Sell signal
                        for position in self.positions.copy():
                            self.exit_position(position, current_price)

            # Close any remaining positions
            if data:
                final_price = data[-1]["close"]
                for position in self.positions.copy():
                    self.exit_position(position, final_price)

            info(
                "Backtest simulation completed",
                {
                    "total_trades": len(self.completed_trades),
                    "final_balance": str(self.balance),
                },
            ).run()

        return IO(simulate)

    def enter_position(self, side: str, price: Decimal, size: Decimal) -> None:
        """Enter a trading position"""
        commission = size * self.config.commission_rate

        if commission > self.balance:
            warn("Insufficient balance for trade").run()
            return

        position = {
            "side": side,
            "entry_price": price,
            "entry_time": self.current_time,
            "size": size,
            "commission": commission,
        }

        self.positions.append(position)
        self.balance -= commission

        # Save trade event
        event = Event(
            id=f"trade_{self.current_time.timestamp()}",
            type="position_entered",
            data=position,
            timestamp=self.current_time,
        )
        save_event(event).run()

    def exit_position(self, position: dict[str, Any], exit_price: Decimal) -> None:
        """Exit a trading position"""
        commission = position["size"] * self.config.commission_rate

        if position["side"] == "long":
            pnl = (
                (exit_price - position["entry_price"])
                * position["size"]
                / position["entry_price"]
            )
        else:
            pnl = (
                (position["entry_price"] - exit_price)
                * position["size"]
                / position["entry_price"]
            )

        # Apply slippage
        slippage = position["size"] * self.config.slippage_rate
        pnl -= slippage

        total_commission = position["commission"] + commission
        net_pnl = pnl - total_commission

        trade = Trade(
            entry_time=position["entry_time"],
            exit_time=self.current_time,
            entry_price=position["entry_price"],
            exit_price=exit_price,
            size=position["size"],
            side=position["side"],
            pnl=net_pnl,
            commission=total_commission,
        )

        self.completed_trades.append(trade)
        self.balance += net_pnl
        self.positions.remove(position)

        # Save trade event
        event = Event(
            id=f"trade_exit_{self.current_time.timestamp()}",
            type="position_exited",
            data={
                "trade": {
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "pnl": str(trade.pnl),
                    "side": trade.side,
                }
            },
            timestamp=self.current_time,
        )
        save_event(event).run()

    def calculate_portfolio_value(self, current_price: Decimal) -> Decimal:
        """Calculate current portfolio value"""
        value = self.balance

        for position in self.positions:
            if position["side"] == "long":
                unrealized_pnl = (
                    (current_price - position["entry_price"])
                    * position["size"]
                    / position["entry_price"]
                )
            else:
                unrealized_pnl = (
                    (position["entry_price"] - current_price)
                    * position["size"]
                    / position["entry_price"]
                )

            value += unrealized_pnl

        return value

    def analyze_results(self) -> IO[BacktestResult]:
        """Analyze backtest results"""

        def analyze():
            if not self.completed_trades:
                return BacktestResult(
                    total_return=Decimal(0),
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    max_drawdown=Decimal(0),
                    sharpe_ratio=0.0,
                    profit_factor=0.0,
                    win_rate=0.0,
                )

            # Calculate metrics
            total_pnl = sum(trade.pnl for trade in self.completed_trades)
            total_return = total_pnl / self.config.initial_balance * Decimal(100)

            winning_trades = sum(1 for trade in self.completed_trades if trade.pnl > 0)
            losing_trades = len(self.completed_trades) - winning_trades
            win_rate = (
                winning_trades / len(self.completed_trades)
                if self.completed_trades
                else 0.0
            )

            # Calculate max drawdown
            max_value = self.config.initial_balance
            max_drawdown = Decimal(0)

            for _timestamp, value in self.portfolio_values:
                max_value = max(max_value, value)

                drawdown = (
                    (max_value - value) / max_value if max_value > 0 else Decimal(0)
                )
                max_drawdown = max(max_drawdown, drawdown)

            # Simple profit factor calculation
            gross_profit = sum(
                trade.pnl for trade in self.completed_trades if trade.pnl > 0
            )
            gross_loss = abs(
                sum(trade.pnl for trade in self.completed_trades if trade.pnl <= 0)
            )
            profit_factor = (
                float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
            )

            result = BacktestResult(
                total_return=total_return,
                total_trades=len(self.completed_trades),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                max_drawdown=max_drawdown,
                sharpe_ratio=0.0,  # Simplified
                profit_factor=profit_factor,
                win_rate=win_rate,
            )

            info(
                "Backtest analysis completed",
                {
                    "total_return": f"{result.total_return:.2f}%",
                    "total_trades": result.total_trades,
                    "win_rate": f"{result.win_rate:.2%}",
                    "max_drawdown": f"{result.max_drawdown:.2%}",
                },
            ).run()

            return result

        return IO(analyze)

    def run_backtest(self) -> IO[BacktestResult]:
        """Run complete backtest"""

        def run():
            info(
                "Starting backtest",
                {
                    "symbol": self.config.symbol,
                    "start_date": self.config.start_date.isoformat(),
                    "end_date": self.config.end_date.isoformat(),
                    "initial_balance": str(self.config.initial_balance),
                },
            ).run()

            # Load data
            data = self.load_historical_data().run()

            # Simulate strategy
            self.simulate_strategy(data).run()

            # Analyze results
            results = self.analyze_results().run()

            info("Backtest completed successfully").run()

            return results

        return IO(run)


def run_backtest(config: BacktestConfig) -> IO[BacktestResult]:
    """Run a backtest with the given configuration"""
    engine = BacktestEngine(config)
    return engine.run_backtest()
