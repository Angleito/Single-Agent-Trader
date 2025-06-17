"""
Backtesting engine for strategy evaluation.

This module provides comprehensive backtesting capabilities
to evaluate trading strategies against historical data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import pandas as pd

from ..indicators.vumanchu import VuManChuIndicators
from ..risk import RiskManager
from ..strategy.core import CoreStrategy
from ..strategy.llm_agent import LLMAgent
from ..types import IndicatorData, MarketData, MarketState, Position, TradeAction
from ..validator import TradeValidator

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Individual trade record for backtesting."""

    entry_time: datetime
    exit_time: datetime | None = None
    symbol: str = ""
    side: str = ""  # LONG/SHORT
    entry_price: Decimal = Decimal("0")
    exit_price: Decimal | None = None
    size: Decimal = Decimal("0")
    pnl: Decimal = Decimal("0")
    pnl_pct: float = 0.0
    duration_minutes: int = 0
    exit_reason: str = ""


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""

    start_date: datetime
    end_date: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: Decimal = Decimal("0")
    total_return_pct: float = 0.0
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: int = 0
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    trades: list[BacktestTrade] = field(default_factory=list)


class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies.

    Simulates trading strategies against historical data with
    realistic execution modeling and comprehensive performance metrics.
    """

    def __init__(self, initial_balance: Decimal = Decimal("10000")):
        """
        Initialize the backtest engine.

        Args:
            initial_balance: Starting account balance
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance

        # Strategy components
        self.indicator_calc = VuManChuIndicators()
        self.llm_agent = LLMAgent()
        self.core_strategy = CoreStrategy()
        self.validator = TradeValidator()
        self.risk_manager = RiskManager()

        # Backtest state
        self.trades: list[BacktestTrade] = []
        self.current_position: BacktestTrade | None = None
        self.equity_curve: list[tuple[datetime, Decimal]] = []

        logger.info(
            f"Initialized BacktestEngine with ${initial_balance} starting balance"
        )

    async def run_backtest(
        self,
        historical_data: pd.DataFrame,
        start_date: datetime = None,
        end_date: datetime = None,
        strategy_type: str = "llm",
    ) -> BacktestResults:
        """
        Run a complete backtest on historical data.

        Args:
            historical_data: DataFrame with OHLCV data
            start_date: Backtest start date
            end_date: Backtest end date
            strategy_type: Strategy to use ("llm" or "core")

        Returns:
            BacktestResults with comprehensive metrics
        """
        logger.info(f"Starting backtest with {strategy_type} strategy")

        # Prepare data
        data = self._prepare_data(historical_data, start_date, end_date)

        if data.empty:
            logger.error("No data available for backtesting")
            return BacktestResults(
                start_date=start_date or datetime.now(),
                end_date=end_date or datetime.now(),
            )

        # Calculate indicators
        data_with_indicators = self.indicator_calc.calculate_all(data)

        # Reset state
        self._reset_state()

        # Run simulation
        await self._simulate_trading(data_with_indicators, strategy_type)

        # Calculate results
        results = self._calculate_results(data.index[0], data.index[-1])

        logger.info(
            f"Backtest completed: {results.total_trades} trades, "
            f"{results.win_rate:.1f}% win rate, "
            f"{results.total_return_pct:.2f}% return"
        )

        return results

    def _prepare_data(
        self, data: pd.DataFrame, start_date: datetime = None, end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Prepare and filter historical data for backtesting.

        Args:
            data: Raw historical data
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            Prepared DataFrame
        """
        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Filter by date range if specified
        filtered_data = data.copy()

        if start_date:
            filtered_data = filtered_data[filtered_data.index >= start_date]

        if end_date:
            filtered_data = filtered_data[filtered_data.index <= end_date]

        # Sort by timestamp
        filtered_data = filtered_data.sort_index()

        logger.info(f"Prepared {len(filtered_data)} candles for backtesting")
        return filtered_data

    def _reset_state(self) -> None:
        """Reset backtest state for new run."""
        self.current_balance = self.initial_balance
        self.trades = []
        self.current_position = None
        self.equity_curve = []
        self.risk_manager = RiskManager()  # Reset risk manager

    async def _simulate_trading(self, data: pd.DataFrame, strategy_type: str) -> None:
        """
        Simulate trading on historical data.

        Args:
            data: Historical data with indicators
            strategy_type: Strategy to use
        """
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Skip if not enough data for indicators
            if i < 50:  # Warmup period
                continue

            # Create market state
            market_state = self._create_market_state(data, i, timestamp)

            # Get trading decision
            if strategy_type == "llm":
                trade_action = await self.llm_agent.analyze_market(market_state)
            else:
                trade_action = self.core_strategy.analyze_market(market_state)

            # Validate action
            validated_action = self.validator.validate(trade_action)

            # Apply risk management
            approved, final_action, reason = self.risk_manager.evaluate_risk(
                validated_action, market_state.current_position, row["close"]
            )

            if not approved:
                continue

            # Execute trade
            await self._execute_backtest_trade(final_action, timestamp, row)

            # Update equity curve
            self._update_equity_curve(timestamp)

            # Check for position exit conditions
            await self._check_exit_conditions(timestamp, row)

    def _create_market_state(
        self, data: pd.DataFrame, index: int, timestamp: datetime
    ) -> MarketState:
        """
        Create market state for a specific point in time.

        Args:
            data: Full historical data
            index: Current index position
            timestamp: Current timestamp

        Returns:
            MarketState object
        """
        # Get recent OHLCV data
        lookback = min(50, index)
        recent_data = data.iloc[index - lookback : index + 1]

        ohlcv_list = []
        for ts, row in recent_data.iterrows():
            ohlcv_list.append(
                MarketData(
                    symbol="BTC-USD",  # Default symbol
                    timestamp=ts,
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=Decimal(str(row["volume"])),
                )
            )

        # Get current indicators
        current_row = data.iloc[index]
        indicators = IndicatorData(
            timestamp=timestamp,
            cipher_a_dot=current_row.get("trend_dot"),
            cipher_b_wave=current_row.get("wave"),
            cipher_b_money_flow=current_row.get("money_flow"),
            rsi=current_row.get("rsi"),
            ema_fast=current_row.get("ema_fast"),
            ema_slow=current_row.get("ema_slow"),
            vwap=current_row.get("vwap"),
        )

        # Current position
        current_position = Position(
            symbol="BTC-USD",
            side="FLAT" if not self.current_position else self.current_position.side,
            size=(
                Decimal("0")
                if not self.current_position
                else self.current_position.size
            ),
            entry_price=(
                None if not self.current_position else self.current_position.entry_price
            ),
            timestamp=timestamp,
        )

        return MarketState(
            symbol="BTC-USD",
            interval="1m",
            timestamp=timestamp,
            current_price=Decimal(str(current_row["close"])),
            ohlcv_data=ohlcv_list,
            indicators=indicators,
            current_position=current_position,
        )

    async def _execute_backtest_trade(
        self, action: TradeAction, timestamp: datetime, market_data: pd.Series
    ) -> None:
        """
        Execute a trade in the backtest simulation.

        Args:
            action: Trade action to execute
            timestamp: Execution timestamp
            market_data: Current market data
        """
        current_price = Decimal(str(market_data["close"]))

        if action.action == "HOLD":
            return

        elif action.action == "CLOSE" and self.current_position:
            await self._close_backtest_position(
                timestamp, current_price, "Manual close"
            )

        elif action.action in ["LONG", "SHORT"] and not self.current_position:
            await self._open_backtest_position(action, timestamp, current_price)

    async def _open_backtest_position(
        self, action: TradeAction, timestamp: datetime, price: Decimal
    ) -> None:
        """
        Open a new position in backtest.

        Args:
            action: Trade action
            timestamp: Entry timestamp
            price: Entry price
        """
        # Calculate position size
        position_value = (
            self.current_balance * Decimal(action.size_pct) / Decimal("100")
        )
        leverage = Decimal("5")  # Default leverage
        position_size = position_value * leverage / price

        self.current_position = BacktestTrade(
            entry_time=timestamp,
            symbol="BTC-USD",
            side=action.action,
            entry_price=price,
            size=position_size,
        )

        logger.debug(f"Opened {action.action} position: {position_size} @ {price}")

    async def _close_backtest_position(
        self, timestamp: datetime, price: Decimal, reason: str
    ) -> None:
        """
        Close current position in backtest.

        Args:
            timestamp: Exit timestamp
            price: Exit price
            reason: Exit reason
        """
        if not self.current_position:
            return

        # Calculate P&L
        if self.current_position.side == "LONG":
            pnl = (
                price - self.current_position.entry_price
            ) * self.current_position.size
        else:  # SHORT
            pnl = (
                self.current_position.entry_price - price
            ) * self.current_position.size

        # Update trade record
        self.current_position.exit_time = timestamp
        self.current_position.exit_price = price
        self.current_position.pnl = pnl
        self.current_position.exit_reason = reason
        self.current_position.duration_minutes = int(
            (timestamp - self.current_position.entry_time).total_seconds() / 60
        )
        self.current_position.pnl_pct = float(
            pnl / (self.current_position.entry_price * self.current_position.size) * 100
        )

        # Update balance
        self.current_balance += pnl

        # Save completed trade
        self.trades.append(self.current_position)
        self.current_position = None

        logger.debug(f"Closed position: {pnl:.2f} PnL ({reason})")

    async def _check_exit_conditions(
        self, timestamp: datetime, market_data: pd.Series
    ) -> None:
        """
        Check for automatic exit conditions (TP/SL).

        Args:
            timestamp: Current timestamp
            market_data: Current market data
        """
        if not self.current_position:
            return

        current_price = Decimal(str(market_data["close"]))
        entry_price = self.current_position.entry_price

        # Calculate price levels (simplified - would use actual TP/SL from trade)
        if self.current_position.side == "LONG":
            # Take profit: 2% above entry
            tp_price = entry_price * Decimal("1.02")
            # Stop loss: 1.5% below entry
            sl_price = entry_price * Decimal("0.985")

            if current_price >= tp_price:
                await self._close_backtest_position(
                    timestamp, current_price, "Take Profit"
                )
            elif current_price <= sl_price:
                await self._close_backtest_position(
                    timestamp, current_price, "Stop Loss"
                )

        else:  # SHORT
            # Take profit: 2% below entry
            tp_price = entry_price * Decimal("0.98")
            # Stop loss: 1.5% above entry
            sl_price = entry_price * Decimal("1.015")

            if current_price <= tp_price:
                await self._close_backtest_position(
                    timestamp, current_price, "Take Profit"
                )
            elif current_price >= sl_price:
                await self._close_backtest_position(
                    timestamp, current_price, "Stop Loss"
                )

    def _update_equity_curve(self, timestamp: datetime) -> None:
        """Update equity curve with current balance."""
        total_equity = self.current_balance

        # Add unrealized P&L if position exists
        if self.current_position:
            # Simplified unrealized P&L calculation
            pass

        self.equity_curve.append((timestamp, total_equity))

    def _calculate_results(
        self, start_date: datetime, end_date: datetime
    ) -> BacktestResults:
        """
        Calculate comprehensive backtest results.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResults object
        """
        if not self.trades:
            return BacktestResults(start_date=start_date, end_date=end_date)

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_return_pct = float(total_pnl / self.initial_balance * 100)

        # Drawdown calculation
        max_balance = self.initial_balance
        max_drawdown = Decimal("0")

        for _, equity in self.equity_curve:
            if equity > max_balance:
                max_balance = equity

            drawdown = max_balance - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        max_drawdown_pct = (
            float(max_drawdown / max_balance * 100) if max_balance > 0 else 0
        )

        # Additional metrics
        winning_pnls = [trade.pnl for trade in self.trades if trade.pnl > 0]
        losing_pnls = [abs(trade.pnl) for trade in self.trades if trade.pnl < 0]

        profit_factor = (
            sum(winning_pnls) / sum(losing_pnls) if losing_pnls else float("inf")
        )

        avg_trade_duration = (
            sum(trade.duration_minutes for trade in self.trades) // total_trades
            if total_trades > 0
            else 0
        )

        largest_win = max((trade.pnl for trade in self.trades), default=Decimal("0"))
        largest_loss = min((trade.pnl for trade in self.trades), default=Decimal("0"))

        return BacktestResults(
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=0.0,  # TODO: Calculate actual Sharpe ratio
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            largest_win=largest_win,
            largest_loss=largest_loss,
            trades=self.trades.copy(),
        )
