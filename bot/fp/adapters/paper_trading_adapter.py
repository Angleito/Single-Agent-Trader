"""
Functional adapter for paper trading account to enable functional portfolio management.

This adapter bridges the gap between the existing paper trading system and the new
functional portfolio types, providing enhanced portfolio analytics and management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from bot.fp.types.portfolio import (
    AccountSnapshot,
    PerformanceSnapshot,
    RiskMetrics,
    create_spot_account,
)
from bot.fp.types.positions import (
    FunctionalPosition,
)
from bot.fp.types.positions import TradeResult as FunctionalTradeResult
from bot.fp.types.result import Failure, Result, Success
from bot.paper_trading import PaperTradingAccount

if TYPE_CHECKING:
    from bot.trading_types import TradeAction

logger = logging.getLogger(__name__)


class FunctionalPaperTradingAdapter:
    """
    Functional adapter for paper trading that provides enhanced portfolio analytics.

    This adapter converts paper trading data to functional types and provides
    advanced portfolio management capabilities.
    """

    def __init__(self, paper_account: PaperTradingAccount) -> None:
        """
        Initialize the adapter with a paper trading account.

        Args:
            paper_account: The paper trading account to adapt
        """
        self.paper_account = paper_account
        self._last_performance_snapshot: PerformanceSnapshot | None = None
        self._performance_history: list[PerformanceSnapshot] = []

    def get_functional_account_snapshot(
        self, current_prices: dict[str, Decimal], base_currency: str = "USD"
    ) -> Result[str, AccountSnapshot]:
        """
        Get account snapshot in functional form with current valuations.

        Args:
            current_prices: Current market prices for asset valuation
            base_currency: Base currency for account calculations

        Returns:
            Result containing AccountSnapshot or error
        """
        try:
            # Get account status from paper trading
            account_status = self.paper_account.get_account_status()

            # Calculate total equity including unrealized P&L
            equity = account_status["equity"]

            # Create asset balances (simplified for paper trading)
            balances = {base_currency: max(Decimal(0), Decimal(str(equity)))}

            # Create account snapshot
            account_snapshot = create_spot_account(balances, base_currency)

            return Success(account_snapshot)

        except Exception as e:
            return Failure(f"Failed to create functional account snapshot: {e!s}")

    def get_functional_trades(self, days: int = 30) -> list[FunctionalTradeResult]:
        """
        Get trade history as functional trade results.

        Args:
            days: Number of days of history to retrieve

        Returns:
            List of functional trade results
        """
        try:
            # Get trade history from paper account
            trade_history = self.paper_account.get_trade_history(days=days)

            functional_trades = []
            for trade_data in trade_history:
                # Convert to functional trade result
                if isinstance(trade_data, dict):
                    functional_trade = self._convert_trade_to_functional(trade_data)
                    if functional_trade is not None:
                        functional_trades.append(functional_trade)

            return functional_trades

        except Exception as e:
            logger.exception(f"Failed to get functional trades: {e}")
            return []

    def calculate_functional_performance(
        self, days: int = 30, benchmark_returns: list[Decimal] | None = None
    ) -> Result[str, PerformanceSnapshot]:
        """
        Calculate comprehensive performance metrics using functional types.

        Args:
            days: Number of days to analyze
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Result containing PerformanceSnapshot
        """
        try:
            # Get performance summary from paper account
            performance = self.paper_account.get_performance_summary(days=days)

            # Get account status
            account_status = self.paper_account.get_account_status()

            # Calculate additional metrics
            total_value = Decimal(str(account_status["equity"]))
            realized_pnl = Decimal(str(performance.get("total_realized_pnl", 0)))
            unrealized_pnl = Decimal(str(performance.get("total_unrealized_pnl", 0)))

            # Calculate daily return
            daily_return = None
            if "daily_pnl" in performance:
                daily_pnl = Decimal(str(performance["daily_pnl"]))
                if total_value > 0:
                    daily_return = (daily_pnl / total_value) * 100

            # Calculate benchmark return if provided
            benchmark_return = None
            if benchmark_returns and len(benchmark_returns) > 0:
                benchmark_return = benchmark_returns[-1]  # Latest benchmark return

            # Calculate drawdown
            drawdown = Decimal(str(performance.get("max_drawdown", 0)))

            # Create performance snapshot
            performance_snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                total_value=total_value,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                daily_return=daily_return,
                benchmark_return=benchmark_return,
                drawdown=drawdown,
            )

            # Store in history
            self._last_performance_snapshot = performance_snapshot
            self._performance_history.append(performance_snapshot)

            # Keep only recent history
            cutoff_date = datetime.now() - timedelta(days=90)
            self._performance_history = [
                snap
                for snap in self._performance_history
                if snap.timestamp >= cutoff_date
            ]

            return Success(performance_snapshot)

        except Exception as e:
            return Failure(f"Failed to calculate functional performance: {e!s}")

    def calculate_risk_metrics(
        self, days: int = 30, confidence_level: float = 0.95
    ) -> Result[str, RiskMetrics]:
        """
        Calculate comprehensive risk metrics for the portfolio.

        Args:
            days: Number of days to analyze
            confidence_level: Confidence level for VaR calculation

        Returns:
            Result containing RiskMetrics
        """
        try:
            # Get trade history for calculations
            functional_trades = self.get_functional_trades(days)

            if not functional_trades:
                # Return default risk metrics if no trades
                return Success(
                    RiskMetrics(
                        var_95=Decimal(0),
                        max_drawdown=Decimal(0),
                        volatility=Decimal(0),
                        sharpe_ratio=Decimal(0),
                        sortino_ratio=Decimal(0),
                        beta=None,
                        correlation_to_benchmark=None,
                        concentration_risk=Decimal(0),
                        timestamp=datetime.now(),
                    )
                )

            # Calculate returns from trades
            returns = [float(trade.return_pct) for trade in functional_trades]

            # Calculate basic risk metrics
            if len(returns) > 1:
                # Volatility (standard deviation of returns)
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / (
                    len(returns) - 1
                )
                volatility = Decimal(str(variance**0.5))

                # Sharpe ratio (simplified)
                risk_free_rate = 0.02  # 2% annual risk-free rate
                excess_return = (
                    mean_return - risk_free_rate / 252
                )  # Daily risk-free rate
                sharpe_ratio = (
                    Decimal(str(excess_return / (variance**0.5)))
                    if variance > 0
                    else Decimal(0)
                )

                # Sortino ratio (downside deviation)
                negative_returns = [r for r in returns if r < 0]
                if negative_returns:
                    downside_variance = sum(r**2 for r in negative_returns) / len(
                        negative_returns
                    )
                    sortino_ratio = (
                        Decimal(str(excess_return / (downside_variance**0.5)))
                        if downside_variance > 0
                        else Decimal(0)
                    )
                else:
                    sortino_ratio = sharpe_ratio  # No negative returns

                # Value at Risk (VaR) - simplified historical method
                sorted_returns = sorted(returns)
                var_index = int((1 - confidence_level) * len(sorted_returns))
                var_95 = (
                    Decimal(str(abs(sorted_returns[var_index])))
                    if var_index < len(sorted_returns)
                    else Decimal(0)
                )

                # Max drawdown calculation
                cumulative_returns = []
                cumulative = 0
                for ret in returns:
                    cumulative += ret
                    cumulative_returns.append(cumulative)

                peak = cumulative_returns[0]
                max_dd = 0
                for cum_ret in cumulative_returns:
                    peak = max(peak, cum_ret)
                    drawdown = (peak - cum_ret) / peak if peak > 0 else 0
                    max_dd = max(max_dd, drawdown)

                max_drawdown = Decimal(str(max_dd * 100))  # Convert to percentage

            else:
                volatility = Decimal(0)
                sharpe_ratio = Decimal(0)
                sortino_ratio = Decimal(0)
                var_95 = Decimal(0)
                max_drawdown = Decimal(0)

            # Concentration risk (simplified)
            # In a real implementation, this would analyze position concentrations
            concentration_risk = Decimal("0.2")  # Placeholder

            risk_metrics = RiskMetrics(
                var_95=var_95,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=None,  # Would need benchmark data
                correlation_to_benchmark=None,  # Would need benchmark data
                concentration_risk=concentration_risk,
                timestamp=datetime.now(),
            )

            return Success(risk_metrics)

        except Exception as e:
            return Failure(f"Failed to calculate risk metrics: {e!s}")

    def simulate_trade_impact(
        self,
        trade_action: TradeAction,
        current_price: Decimal,
        current_positions: list[FunctionalPosition],
    ) -> Result[str, dict[str, Decimal]]:
        """
        Simulate the impact of a potential trade on portfolio metrics.

        Args:
            trade_action: Proposed trade action
            current_price: Current market price
            current_positions: Current portfolio positions

        Returns:
            Result containing impact analysis
        """
        try:
            # Get current account status
            account_status = self.paper_account.get_account_status()
            current_equity = Decimal(str(account_status["equity"]))

            # Calculate trade size
            trade_size = current_equity * (Decimal(str(trade_action.size_pct)) / 100)
            quantity = trade_size / current_price

            # Calculate potential P&L scenarios
            take_profit_price = current_price * (
                1 + Decimal(str(trade_action.take_profit_pct)) / 100
            )
            stop_loss_price = current_price * (
                1 - Decimal(str(trade_action.stop_loss_pct)) / 100
            )

            if trade_action.action == "LONG":
                profit_scenario = (take_profit_price - current_price) * quantity
                loss_scenario = (current_price - stop_loss_price) * quantity
            elif trade_action.action == "SHORT":
                profit_scenario = (current_price - take_profit_price) * quantity
                loss_scenario = (stop_loss_price - current_price) * quantity
            else:
                profit_scenario = Decimal(0)
                loss_scenario = Decimal(0)

            # Calculate impact on portfolio
            profit_impact_pct = (
                (profit_scenario / current_equity) * 100
                if current_equity > 0
                else Decimal(0)
            )
            loss_impact_pct = (
                (loss_scenario / current_equity) * 100
                if current_equity > 0
                else Decimal(0)
            )

            # Calculate position concentration after trade
            new_portfolio_value = current_equity + trade_size
            concentration_after_trade = (
                (trade_size / new_portfolio_value) * 100
                if new_portfolio_value > 0
                else Decimal(0)
            )

            impact_analysis = {
                "trade_size": trade_size,
                "quantity": quantity,
                "profit_scenario": profit_scenario,
                "loss_scenario": loss_scenario,
                "profit_impact_pct": profit_impact_pct,
                "loss_impact_pct": loss_impact_pct,
                "concentration_pct": concentration_after_trade,
                "risk_reward_ratio": (
                    abs(profit_scenario / loss_scenario)
                    if loss_scenario != 0
                    else Decimal(0)
                ),
            }

            return Success(impact_analysis)

        except Exception as e:
            return Failure(f"Failed to simulate trade impact: {e!s}")

    def get_performance_attribution(
        self, days: int = 30
    ) -> Result[str, dict[str, Decimal]]:
        """
        Calculate performance attribution by symbol/strategy.

        Args:
            days: Number of days to analyze

        Returns:
            Result containing performance attribution
        """
        try:
            functional_trades = self.get_functional_trades(days)

            if not functional_trades:
                return Success({})

            # Group trades by symbol
            symbol_performance = {}
            for trade in functional_trades:
                symbol = trade.symbol
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {
                        "total_pnl": Decimal(0),
                        "trade_count": 0,
                        "win_count": 0,
                        "total_return_pct": Decimal(0),
                    }

                symbol_performance[symbol]["total_pnl"] += trade.pnl
                symbol_performance[symbol]["trade_count"] += 1
                symbol_performance[symbol]["total_return_pct"] += trade.return_pct

                if trade.pnl > 0:
                    symbol_performance[symbol]["win_count"] += 1

            # Calculate win rates and average returns
            attribution = {}
            for symbol, perf in symbol_performance.items():
                win_rate = (
                    (perf["win_count"] / perf["trade_count"]) * 100
                    if perf["trade_count"] > 0
                    else Decimal(0)
                )
                avg_return = (
                    perf["total_return_pct"] / perf["trade_count"]
                    if perf["trade_count"] > 0
                    else Decimal(0)
                )

                attribution[f"{symbol}_pnl"] = perf["total_pnl"]
                attribution[f"{symbol}_win_rate"] = win_rate
                attribution[f"{symbol}_avg_return"] = avg_return
                attribution[f"{symbol}_trade_count"] = Decimal(str(perf["trade_count"]))

            return Success(attribution)

        except Exception as e:
            return Failure(f"Failed to calculate performance attribution: {e!s}")

    def generate_functional_report(self, days: int = 7) -> Result[str, dict[str, any]]:
        """
        Generate comprehensive portfolio report using functional types.

        Args:
            days: Number of days to analyze

        Returns:
            Result containing comprehensive report
        """
        try:
            # Get performance snapshot
            performance_result = self.calculate_functional_performance(days)
            if performance_result.is_failure():
                return Failure(performance_result.failure())

            performance = performance_result.success()

            # Get risk metrics
            risk_result = self.calculate_risk_metrics(days)
            if risk_result.is_failure():
                return Failure(risk_result.failure())

            risk_metrics = risk_result.success()

            # Get performance attribution
            attribution_result = self.get_performance_attribution(days)
            if attribution_result.is_failure():
                return Failure(attribution_result.failure())

            attribution = attribution_result.success()

            # Get trade statistics
            functional_trades = self.get_functional_trades(days)

            report = {
                "report_date": datetime.now().isoformat(),
                "analysis_period_days": days,
                "performance": {
                    "total_value": float(performance.total_value),
                    "total_pnl": float(performance.total_pnl),
                    "realized_pnl": float(performance.realized_pnl),
                    "unrealized_pnl": float(performance.unrealized_pnl),
                    "total_return_pct": float(performance.total_return_pct),
                    "daily_return_pct": (
                        float(performance.daily_return)
                        if performance.daily_return
                        else None
                    ),
                    "drawdown_pct": float(performance.drawdown),
                },
                "risk_metrics": {
                    "var_95_pct": float(risk_metrics.var_95),
                    "max_drawdown_pct": float(risk_metrics.max_drawdown),
                    "volatility": float(risk_metrics.volatility),
                    "sharpe_ratio": float(risk_metrics.sharpe_ratio),
                    "sortino_ratio": float(risk_metrics.sortino_ratio),
                    "risk_score": risk_metrics.risk_score,
                    "concentration_risk": float(risk_metrics.concentration_risk),
                },
                "trading_statistics": {
                    "total_trades": len(functional_trades),
                    "winning_trades": len([t for t in functional_trades if t.pnl > 0]),
                    "losing_trades": len([t for t in functional_trades if t.pnl < 0]),
                    "win_rate_pct": (
                        (
                            len([t for t in functional_trades if t.pnl > 0])
                            / len(functional_trades)
                            * 100
                        )
                        if functional_trades
                        else 0
                    ),
                    "avg_trade_duration_hours": (
                        sum(t.duration for t in functional_trades)
                        / len(functional_trades)
                        / 3600
                        if functional_trades
                        else 0
                    ),
                },
                "performance_attribution": {
                    k: float(v) for k, v in attribution.items()
                },
                "functional_types_enabled": True,
                "adapter_version": "1.0.0",
            }

            return Success(report)

        except Exception as e:
            return Failure(f"Failed to generate functional report: {e!s}")

    def _convert_trade_to_functional(
        self, trade_data: dict
    ) -> FunctionalTradeResult | None:
        """
        Convert paper trade data to functional trade result.

        Args:
            trade_data: Paper trade data dictionary

        Returns:
            FunctionalTradeResult or None if conversion fails
        """
        try:
            # Extract trade information
            symbol = trade_data.get("symbol", "UNKNOWN")
            side = trade_data.get("side", "LONG")
            entry_price = Decimal(str(trade_data.get("entry_price", 0)))
            exit_price = Decimal(str(trade_data.get("exit_price", entry_price)))
            size = Decimal(str(trade_data.get("size", 0)))

            # Parse timestamps
            entry_time_str = trade_data.get("entry_time")
            exit_time_str = trade_data.get("exit_time")

            if isinstance(entry_time_str, str):
                entry_time = datetime.fromisoformat(entry_time_str)
            else:
                entry_time = entry_time_str or datetime.now()

            if isinstance(exit_time_str, str):
                exit_time = datetime.fromisoformat(exit_time_str)
            else:
                exit_time = exit_time_str or entry_time

            # Generate trade ID
            trade_id = f"{symbol}_{entry_time.isoformat()}"

            return FunctionalTradeResult(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                entry_time=entry_time,
                exit_time=exit_time,
            )

        except Exception as e:
            logger.exception(f"Failed to convert trade to functional: {e}")
            return None


# Utility functions for paper trading functional integration


def create_functional_paper_account(
    initial_balance: Decimal = Decimal(10000), base_currency: str = "USD"
) -> tuple[PaperTradingAccount, FunctionalPaperTradingAdapter]:
    """
    Create a new paper trading account with functional adapter.

    Args:
        initial_balance: Initial account balance
        base_currency: Base currency for the account

    Returns:
        Tuple of (PaperTradingAccount, FunctionalPaperTradingAdapter)
    """
    # Create paper trading account
    paper_account = PaperTradingAccount()

    # Create functional adapter
    functional_adapter = FunctionalPaperTradingAdapter(paper_account)

    return paper_account, functional_adapter


def validate_paper_trading_migration(
    adapter: FunctionalPaperTradingAdapter, current_prices: dict[str, Decimal]
) -> bool:
    """
    Validate that paper trading functional migration is working correctly.

    Args:
        adapter: The functional paper trading adapter
        current_prices: Current market prices

    Returns:
        True if migration is valid, False otherwise
    """
    try:
        # Test account snapshot creation
        account_result = adapter.get_functional_account_snapshot(current_prices)
        if account_result.is_failure():
            return False

        # Test performance calculation
        performance_result = adapter.calculate_functional_performance()
        if performance_result.is_failure():
            return False

        # Test risk metrics calculation
        risk_result = adapter.calculate_risk_metrics()
        if risk_result.is_failure():
            return False

        # Test report generation
        report_result = adapter.generate_functional_report()
        return not report_result.is_failure()

    except Exception as e:
        logger.exception(f"Paper trading migration validation failed: {e}")
        return False
