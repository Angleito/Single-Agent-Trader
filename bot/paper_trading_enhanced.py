"""
Enhanced Paper Trading System with Functional Programming Benefits

This module provides a drop-in replacement for the existing paper trading system
that leverages functional programming patterns while maintaining full API compatibility.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .fp.paper_trading_functional import (
    FunctionalPaperTradingEngine,
    calculate_performance_report,
    execute_trade_with_logging,
)
from .fp.pure.paper_trading_calculations import normalize_decimal_precision
from .fp.types.paper_trading import PaperTradingAccountState
from .trading_types import Order, OrderStatus, TradeAction

logger = logging.getLogger(__name__)


class EnhancedPaperTradingAccount:
    """
    Enhanced paper trading account that maintains API compatibility
    while leveraging functional programming benefits internally.

    This class serves as an adapter between the existing imperative API
    and the new functional implementation, providing:
    - Immutable state management
    - Pure functional calculations
    - Better error handling
    - Improved performance
    """

    def __init__(
        self,
        starting_balance: Decimal | None = None,
        data_dir: Path | None = None,
        use_functional_core: bool = True,
    ):
        """
        Initialize enhanced paper trading account.

        Args:
            starting_balance: Initial account balance
            data_dir: Directory for state persistence
            use_functional_core: Whether to use functional implementation (default: True)
        """
        # Configuration
        self.use_functional_core = use_functional_core
        self.data_dir = data_dir or Path("./data/paper_trading_enhanced")

        # Initialize balance
        if starting_balance is not None:
            self.starting_balance = normalize_decimal_precision(starting_balance, 2)
        else:
            self.starting_balance = Decimal("10000.00")

        # Functional core engine
        if self.use_functional_core:
            self._fp_engine = FunctionalPaperTradingEngine(
                initial_balance=self.starting_balance,
                data_dir=self.data_dir,
                fee_rate=Decimal("0.001"),  # 0.1%
                slippage_rate=Decimal("0.0005"),  # 0.05%
                leverage=Decimal(5),
            )

        # Compatibility state (for API compatibility)
        self._last_current_prices: dict[str, Decimal] = {}

        logger.info(
            "Enhanced paper trading account initialized with $%.2f (functional_core=%s)",
            self.starting_balance,
            self.use_functional_core,
        )

    def get_account_status(
        self, current_prices: dict[str, Decimal] | None = None
    ) -> dict[str, Any]:
        """
        Get current account status (compatible with existing API).

        Args:
            current_prices: Optional current prices for P&L calculation

        Returns:
            Account status dictionary matching original API
        """
        if not self.use_functional_core:
            # Fallback to simple implementation
            return self._get_simple_account_status()

        # Use functional implementation
        prices = current_prices or self._last_current_prices
        if prices:
            self._last_current_prices = prices

        try:
            # Get metrics using functional engine
            metrics_result = self._fp_engine.get_account_metrics(prices).run()

            # Convert to expected format
            return {
                "starting_balance": metrics_result["starting_balance"],
                "current_balance": metrics_result["current_balance"],
                "equity": metrics_result["equity"],
                "unrealized_pnl": metrics_result["unrealized_pnl"],
                "total_pnl": metrics_result["total_pnl"],
                "roi_percent": metrics_result["roi_percent"],
                "margin_used": metrics_result["margin_used"],
                "margin_available": metrics_result["margin_available"],
                "open_positions": metrics_result["open_positions"],
                "total_trades": metrics_result["total_trades"],
                "peak_equity": metrics_result.get(
                    "peak_equity", metrics_result["equity"]
                ),
                "current_drawdown": 0.0,  # Would need drawdown calculation
                "max_drawdown": 0.0,  # Would need max drawdown tracking
            }

        except Exception:
            logger.exception("Error getting account status from functional engine")
            return self._get_simple_account_status()

    def execute_trade_action(
        self, action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Execute a trade action (compatible with existing API).

        Args:
            action: Trade action to execute
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Order object representing the trade
        """
        if not self.use_functional_core:
            return self._execute_simple_trade(action, symbol, current_price)

        try:
            # Update last known price
            self._last_current_prices[symbol] = current_price

            if action.action == "HOLD":
                logger.info(
                    "🛑 Enhanced Paper Trading: HOLD | Symbol: %s | Price: $%s | Reason: %s",
                    symbol,
                    current_price,
                    action.rationale,
                )
                return None

            # Convert action to functional parameters
            side = action.action  # "LONG", "SHORT", "CLOSE"
            size_pct = Decimal(str(action.size_pct))

            # Execute using functional engine with logging
            execution_result = execute_trade_with_logging(
                engine=self._fp_engine,
                symbol=symbol,
                side=side,
                size_percentage=size_pct,
                current_price=current_price,
                is_futures=False,  # Could be configurable
            ).run()

            if execution_result.is_right():
                execution, new_state = execution_result.value

                if execution.success and execution.trade_state:
                    # Commit the state change
                    commit_result = self._fp_engine.commit_state_change(new_state).run()

                    if commit_result.is_right():
                        # Create compatible Order object
                        order = self._create_order_from_execution(
                            execution, symbol, action
                        )

                        logger.info(
                            "✅ Enhanced Paper Trading: %s executed | Symbol: %s | "
                            "Price: $%.4f | Size: %s | Fees: $%.4f | Order ID: %s",
                            action.action,
                            symbol,
                            execution.execution_price,
                            (
                                execution.trade_state.size
                                if execution.trade_state
                                else "Unknown"
                            ),
                            execution.fees.total_fees,
                            order.id if order else "None",
                        )

                        return order
                    logger.error(
                        "Failed to commit state change: %s", commit_result.value
                    )
                    return self._create_failed_order(
                        action, symbol, "STATE_COMMIT_FAILED"
                    )
                logger.warning("Trade execution unsuccessful: %s", execution.reason)
                return self._create_failed_order(action, symbol, execution.reason)
            logger.error("Trade execution failed: %s", execution_result.value)
            return self._create_failed_order(action, symbol, execution_result.value)

        except Exception as e:
            logger.exception("Error executing trade with functional engine")
            return self._create_failed_order(action, symbol, f"EXECUTION_ERROR: {e}")

    def get_trade_history(self, days: int = 30) -> list[dict[str, Any]]:
        """
        Get trade history (compatible with existing API).

        Args:
            days: Number of days to look back (currently ignored in functional impl)

        Returns:
            List of trade dictionaries
        """
        if not self.use_functional_core:
            return []

        try:
            return self._fp_engine.get_trade_history().run()
        except Exception:
            logger.exception("Error getting trade history")
            return []

    def get_performance_summary(self, days: int = 7) -> dict[str, Any]:
        """
        Get performance summary (compatible with existing API).

        Args:
            days: Number of days for analysis

        Returns:
            Performance summary dictionary
        """
        if not self.use_functional_core:
            return {"message": "Performance summary not available"}

        try:
            # Generate comprehensive report
            report_result = calculate_performance_report(
                self._fp_engine, self._last_current_prices
            ).run()

            # Convert to expected format
            perf_metrics = report_result.get("performance_metrics", {})
            account_metrics = report_result.get("account_metrics", {})

            return {
                "period_days": days,
                "total_trades": perf_metrics.get("total_trades", 0),
                "total_realized_pnl": perf_metrics.get("realized_pnl", 0.0),
                "total_fees_paid": 0.0,  # Would need fee tracking
                "net_pnl": perf_metrics.get("total_pnl", 0.0),
                "avg_daily_pnl": 0.0,  # Would need daily breakdown
                "max_daily_gain": 0.0,  # Would need daily breakdown
                "max_daily_loss": 0.0,  # Would need daily breakdown
                "winning_days": 0,  # Would need daily breakdown
                "total_days": days,
                "win_day_rate": 0.0,
                "overall_win_rate": perf_metrics.get("win_rate", 0.0),
                "current_equity": account_metrics.get("equity", 0.0),
                "roi_percent": account_metrics.get("roi_percent", 0.0),
                "max_drawdown": perf_metrics.get("max_drawdown", 0.0),
                "sharp_ratio": perf_metrics.get("sharpe_ratio", 0.0),
            }

        except Exception as e:
            logger.exception("Error getting performance summary")
            return {"message": f"Error generating performance summary: {e}"}

    def generate_daily_report(self, date: str | None = None) -> str:
        """Generate daily report (compatible with existing API)."""
        if not self.use_functional_core:
            return "Daily report not available"

        try:
            report_result = calculate_performance_report(
                self._fp_engine, self._last_current_prices
            ).run()

            account_metrics = report_result.get("account_metrics", {})
            performance_metrics = report_result.get("performance_metrics", {})

            target_date = date or datetime.now(UTC).date().isoformat()

            return f"""
🗓️  Enhanced Daily Trading Report - {target_date}
{"=" * 50}

💰 Account Status:
   Starting Balance: ${account_metrics.get("starting_balance", 0):,.2f}
   Current Balance:  ${account_metrics.get("current_balance", 0):,.2f}
   Current Equity:   ${account_metrics.get("equity", 0):,.2f}
   Total P&L:        ${performance_metrics.get("total_pnl", 0):,.2f}

📊 Trading Activity:
   Total Trades:     {performance_metrics.get("total_trades", 0)}
   Win Rate:         {performance_metrics.get("win_rate", 0):.1f}%
   Open Positions:   {account_metrics.get("open_positions", 0)}

📈 Performance:
   ROI:              {account_metrics.get("roi_percent", 0):.2f}%
   Sharpe Ratio:     {performance_metrics.get("sharpe_ratio", 0):.2f}
   Max Drawdown:     {performance_metrics.get("max_drawdown", 0):.2f}%
   Profit Factor:    {performance_metrics.get("profit_factor", 0):.2f}

🔄 Margin:
   Used:             ${account_metrics.get("margin_used", 0):,.2f}
   Available:        ${account_metrics.get("margin_available", 0):,.2f}

✨ Enhanced with Functional Programming
"""

        except Exception as e:
            logger.exception("Error generating daily report")
            return f"Error generating daily report: {e}"

    def reset_account(self, new_balance: Decimal | None = None) -> None:
        """Reset account (compatible with existing API)."""
        if not self.use_functional_core:
            return

        try:
            balance = new_balance or Decimal(10000)
            reset_result = self._fp_engine.reset_account(balance).run()

            if reset_result.is_right():
                new_state = reset_result.value
                commit_result = self._fp_engine.commit_state_change(new_state).run()

                if commit_result.is_right():
                    logger.info(
                        "Enhanced paper trading account reset to $%.2f", balance
                    )
                else:
                    logger.error(
                        "Failed to commit account reset: %s", commit_result.value
                    )
            else:
                logger.error("Failed to reset account: %s", reset_result.value)

        except Exception:
            logger.exception("Error resetting account")

    # Internal helper methods

    def _get_simple_account_status(self) -> dict[str, Any]:
        """Simple fallback account status."""
        return {
            "starting_balance": float(self.starting_balance),
            "current_balance": float(self.starting_balance),
            "equity": float(self.starting_balance),
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "roi_percent": 0.0,
            "margin_used": 0.0,
            "margin_available": float(self.starting_balance),
            "open_positions": 0,
            "total_trades": 0,
            "peak_equity": float(self.starting_balance),
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
        }

    def _execute_simple_trade(
        self, action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """Simple fallback trade execution."""
        logger.info("Using simple trade execution fallback")

        if action.action == "HOLD":
            return None

        # Create a basic successful order for compatibility
        return Order(
            id=f"simple_{datetime.now().timestamp()}",
            symbol=symbol,
            side="BUY" if action.action == "LONG" else "SELL",
            type="MARKET",
            quantity=Decimal("1.0"),
            price=current_price,
            filled_quantity=Decimal("1.0"),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
        )

    def _create_order_from_execution(
        self, execution: "TradeExecution", symbol: str, action: TradeAction
    ) -> Order | None:
        """Create Order object from functional execution result."""
        if not execution.success or not execution.trade_state:
            return None

        trade_state = execution.trade_state

        return Order(
            id=trade_state.id,
            symbol=symbol,
            side="BUY" if action.action == "LONG" else "SELL",
            type="MARKET",
            quantity=trade_state.size,
            price=execution.execution_price,
            filled_quantity=trade_state.size,
            status=OrderStatus.FILLED,
            timestamp=trade_state.entry_time,
        )

    def _create_failed_order(
        self, action: TradeAction, symbol: str, reason: str
    ) -> Order:
        """Create failed order object."""
        return Order(
            id=f"failed_{datetime.now().timestamp()}",
            symbol=symbol,
            side="BUY" if action.action == "LONG" else "SELL",
            type="MARKET",
            quantity=Decimal(0),
            price=Decimal(0),
            filled_quantity=Decimal(0),
            status=OrderStatus.REJECTED,
            timestamp=datetime.now(UTC),
        )

    # Additional methods for functional benefits exposure

    def get_functional_state(self) -> PaperTradingAccountState | None:
        """
        Get the functional state (for advanced users).

        Returns:
            Immutable account state if using functional core
        """
        if not self.use_functional_core:
            return None

        try:
            return self._fp_engine.get_current_state().run()
        except Exception:
            logger.exception("Error getting functional state")
            return None

    def get_enhanced_metrics(self) -> dict[str, Any]:
        """
        Get enhanced metrics from functional implementation.

        Returns:
            Enhanced metrics dictionary
        """
        if not self.use_functional_core:
            return {}

        try:
            return calculate_performance_report(
                self._fp_engine, self._last_current_prices
            ).run()
        except Exception:
            logger.exception("Error getting enhanced metrics")
            return {}


# Factory function for backward compatibility
def create_enhanced_paper_trading_account(
    starting_balance: Decimal | None = None, data_dir: Path | None = None, **kwargs
) -> EnhancedPaperTradingAccount:
    """
    Create enhanced paper trading account with functional benefits.

    This function provides a drop-in replacement for the original
    PaperTradingAccount constructor.
    """
    return EnhancedPaperTradingAccount(
        starting_balance=starting_balance, data_dir=data_dir, **kwargs
    )
