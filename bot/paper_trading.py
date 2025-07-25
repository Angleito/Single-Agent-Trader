"""
Enhanced paper trading system for realistic trading simulation.

This module provides a comprehensive paper trading system that simulates
real trading conditions including:
- Realistic account balance tracking
- Trade execution with fees and slippage
- Position management and P&L tracking
- Performance analytics and reporting
- State persistence between sessions
"""

import json
import logging
import os
import statistics
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, getcontext
from pathlib import Path
from typing import Any

from .config import settings
from .fee_calculator import fee_calculator
from .trading_types import Order, OrderStatus, TradeAction
from .validation import BalanceValidator

# Import monitoring components
try:
    from .monitoring.balance_alerts import get_balance_alert_manager
    from .monitoring.balance_metrics import (
        get_balance_metrics_collector,
        record_operation_complete,
        record_operation_start,
    )

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Set global decimal precision for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN


@dataclass
class PaperTrade:
    """Represents a paper trade with entry and exit information."""

    id: str
    symbol: str
    side: str  # LONG or SHORT
    entry_time: datetime
    entry_price: Decimal
    size: Decimal
    exit_time: datetime | None = None
    exit_price: Decimal | None = None
    realized_pnl: Decimal | None = None
    fees: Decimal = Decimal(0)
    slippage: Decimal = Decimal(0)
    status: str = "OPEN"  # OPEN, CLOSED, PARTIAL

    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L based on current price."""
        if self.status == "CLOSED":
            return Decimal(0)

        price_diff = current_price - self.entry_price
        if self.side == "LONG":
            return self.size * price_diff - self.fees
        # SHORT
        return self.size * (-price_diff) - self.fees

    def close_trade(
        self,
        exit_price: Decimal,
        exit_time: datetime,
        additional_fees: Decimal = Decimal(0),
    ) -> Decimal:
        """Close the trade and calculate realized P&L."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = "CLOSED"

        price_diff = exit_price - self.entry_price
        if self.side == "LONG":
            gross_pnl = self.size * price_diff
        else:  # SHORT
            gross_pnl = self.size * (-price_diff)

        total_fees = self.fees + additional_fees
        self.realized_pnl = gross_pnl - total_fees
        return self.realized_pnl


@dataclass
class DailyPerformance:
    """Daily performance metrics."""

    date: str
    starting_balance: Decimal
    ending_balance: Decimal
    trades_opened: int
    trades_closed: int
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    fees_paid: Decimal
    win_rate: float
    largest_win: Decimal
    largest_loss: Decimal
    drawdown: Decimal


class PaperTradingEngine:
    """
    Paper Trading Engine that bridges imperative and functional programming patterns.

    This class provides a simplified interface for paper trading operations that can
    be used by existing system components while internally leveraging the functional
    paper trading system for better reliability and testability.
    """

    def __init__(
        self, initial_balance: Decimal | None = None, data_dir: Path | None = None
    ):
        """Initialize the paper trading engine."""
        self.account = PaperTradingAccount(initial_balance, data_dir)

    def get_balance(self) -> float:
        """Get current account balance."""
        return float(self.account.current_balance)

    def get_equity(self) -> float:
        """Get current account equity."""
        return float(self.account.equity)

    def get_account_status(
        self, current_prices: dict[str, Decimal] | None = None
    ) -> dict[str, Any]:
        """Get comprehensive account status."""
        return self.account.get_account_status(current_prices)

    def execute_trade(
        self, action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """Execute a trade action."""
        return self.account.execute_trade_action(action, symbol, current_price)

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Get all open positions."""
        with self.account._lock:
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
                for trade in self.account.open_trades.values()
            ]

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics formatted for monitoring systems."""
        return {
            "total_pnl": float(self.account.equity - self.account.starting_balance),
            "roi_percent": (
                float(
                    (self.account.equity - self.account.starting_balance)
                    / self.account.starting_balance
                    * 100
                )
                if self.account.starting_balance > 0
                else 0.0
            ),
            "open_positions": len(self.account.open_trades),
            "total_trades": len(self.account.closed_trades),
            "current_balance": float(self.account.current_balance),
            "equity": float(self.account.equity),
            "margin_used": float(self.account.margin_used),
            "max_drawdown": float(self.account.max_drawdown),
        }

    def reset(self, new_balance: Decimal | None = None) -> None:
        """Reset the paper trading account."""
        self.account.reset_account(new_balance)

    def close(self) -> None:
        """Close and cleanup the paper trading engine."""
        # Save final state
        self.account._save_state()


class PaperTradingAccount:
    """
    Paper trading account simulation with realistic features.

    Simulates a trading account with:
    - Starting balance and equity tracking
    - Position management with leverage
    - Fee calculation and slippage simulation
    - Performance tracking and analytics
    - State persistence between sessions
    """

    def __init__(
        self, starting_balance: Decimal | None = None, data_dir: Path | None = None
    ):
        """
        Initialize paper trading account.

        Args:
            starting_balance: Initial account balance (default from config)
            data_dir: Directory for state persistence
        """
        # Use fallback directory if available from Docker entrypoint
        fallback_data_dir = os.getenv("FALLBACK_DATA_DIR")
        if data_dir:
            self.data_dir = data_dir
        elif fallback_data_dir:
            self.data_dir = Path(fallback_data_dir) / "paper_trading"
        else:
            from .utils.path_utils import get_data_directory

            try:
                self.data_dir = get_data_directory() / "paper_trading"
            except OSError:
                # Fallback to temporary directory

                self.data_dir = Path(tempfile.mkdtemp(prefix="paper_trading_"))

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            if fallback_data_dir:
                # Already using fallback, create secure temporary directory
                temp_dir = tempfile.mkdtemp(prefix="paper_trading_")
                self.data_dir = Path(temp_dir)
                logging.warning(
                    "Created paper trading directory in secure temporary space: %s",
                    self.data_dir,
                )
            else:
                logging.exception("Failed to create paper trading directory")
                raise

        # Account state with normalized precision
        if starting_balance is not None:
            self.starting_balance = self._normalize_balance(starting_balance)
        else:
            self.starting_balance = self._normalize_balance(
                Decimal(str(getattr(settings.paper_trading, "starting_balance", 10000)))
            )
        self.current_balance = self.starting_balance
        self.equity = self.starting_balance
        self.margin_used = Decimal(0)

        # Trading state
        self.open_trades: dict[str, PaperTrade] = {}
        self.closed_trades: list[PaperTrade] = []
        self.trade_counter = 0

        # Performance tracking
        self.session_start_time = datetime.now(UTC)
        self.daily_metrics: dict[str, DailyPerformance] = {}
        self.peak_equity = self.starting_balance
        self.max_drawdown = Decimal(0)

        # Configuration
        self.fee_rate = Decimal(
            str(getattr(settings.paper_trading, "fee_rate", 0.001))
        )  # 0.1%
        self.slippage_rate = Decimal(
            str(getattr(settings.paper_trading, "slippage_rate", 0.0005))
        )  # 0.05%

        # Thread safety
        self._lock = threading.RLock()

        # Balance validation system
        self.balance_validator = BalanceValidator()

        # Monitoring system integration
        self.monitoring_enabled = MONITORING_AVAILABLE
        if self.monitoring_enabled:
            try:
                self.metrics_collector = get_balance_metrics_collector()
                self.alert_manager = get_balance_alert_manager()
                logger.info("✅ Paper trading monitoring enabled")
            except Exception as e:
                logger.warning("Failed to initialize monitoring: %s", e)
                self.monitoring_enabled = False

        # State files
        self.account_file = self.data_dir / "account.json"
        self.trades_file = self.data_dir / "trades.json"
        self.performance_file = self.data_dir / "performance.json"

        # Load persisted state
        self._load_state()

        # Record initialization in monitoring system
        if self.monitoring_enabled:
            self._record_account_initialization()

        logger.info(
            "Initialized paper trading account with $%.2f and balance validation",
            self.current_balance,
        )

    def _record_account_initialization(self) -> None:
        """Record account initialization in monitoring system."""
        try:
            if self.monitoring_enabled:
                start_time = record_operation_start(
                    operation="account_initialization",
                    component="paper_trading",
                    correlation_id=f"init_{int(time.time())}",
                    metadata={
                        "starting_balance": float(self.starting_balance),
                        "data_dir": str(self.data_dir),
                    },
                )

                record_operation_complete(
                    operation="account_initialization",
                    component="paper_trading",
                    start_time=start_time,
                    success=True,
                    balance_before=None,
                    balance_after=float(self.starting_balance),
                    correlation_id=f"init_{int(time.time())}",
                )
        except Exception as e:
            logger.warning("Failed to record account initialization: %s", e)

    def _record_balance_operation(
        self,
        operation: str,
        balance_before: float | None = None,
        balance_after: float | None = None,
        success: bool = True,
        error_type: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record a balance operation in the monitoring system."""
        try:
            if self.monitoring_enabled:
                correlation_id = f"paper_{operation}_{int(time.time() * 1000)}"

                start_time = record_operation_start(
                    operation=operation,
                    component="paper_trading",
                    correlation_id=correlation_id,
                    metadata=metadata or {},
                )

                record_operation_complete(
                    operation=operation,
                    component="paper_trading",
                    start_time=start_time,
                    success=success,
                    balance_before=balance_before,
                    balance_after=balance_after,
                    error_type=error_type,
                    correlation_id=correlation_id,
                    metadata=metadata,
                )
        except Exception as e:
            logger.debug("Failed to record balance operation %s: %s", operation, e)

    def _normalize_balance(self, amount: Decimal) -> Decimal:
        """
        Normalize balance to 2 decimal places for USD currency.
        This prevents floating-point precision errors and excessive decimal places.

        Args:
            amount: USD amount to normalize

        Returns:
            Normalized amount with 2 decimal places

        Raises:
            ValueError: If amount is invalid (NaN, infinite, or extremely large)
        """
        if amount is None:
            return Decimal("0.00")

        # Validate that the amount is a proper decimal
        if not isinstance(amount, Decimal):
            try:
                amount = Decimal(str(amount))
            except (ValueError, TypeError) as e:
                logger.exception(
                    "Invalid balance amount (type: %s)", type(amount).__name__
                )
                raise ValueError(f"Cannot convert to Decimal: {amount}") from e

        # Check for invalid values
        if amount.is_nan():
            logger.error("Balance amount is NaN")
            raise ValueError("Balance amount cannot be NaN")

        if amount.is_infinite():
            logger.error("Balance amount is infinite")
            raise ValueError("Balance amount cannot be infinite")

        # Check for extremely large values (> $1 billion)
        if abs(amount) > Decimal(1000000000):
            logger.warning("Extremely large balance amount: %s", amount)

        # Quantize to 2 decimal places using banker's rounding
        return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)

    def _normalize_crypto_amount(self, amount: Decimal) -> Decimal:
        """
        Normalize crypto amounts to 8 decimal places for precision.

        Args:
            amount: Crypto amount to normalize

        Returns:
            Normalized amount with 8 decimal places

        Raises:
            ValueError: If amount is invalid (NaN, infinite, or negative)
        """
        if amount is None:
            return Decimal("0.00000000")

        # Validate that the amount is a proper decimal
        if not isinstance(amount, Decimal):
            try:
                amount = Decimal(str(amount))
            except (ValueError, TypeError) as e:
                logger.exception(
                    "Invalid crypto amount (type: %s)", type(amount).__name__
                )
                raise ValueError(f"Cannot convert to Decimal: {amount}") from e

        # Check for invalid values
        if amount.is_nan():
            logger.error("Crypto amount is NaN")
            raise ValueError("Crypto amount cannot be NaN")

        if amount.is_infinite():
            logger.error("Crypto amount is infinite")
            raise ValueError("Crypto amount cannot be infinite")

        # Check for negative values (most crypto amounts should be positive)
        if amount < Decimal(0):
            logger.warning("Negative crypto amount: %s", amount)

        # Check for extremely small values (dust)
        if Decimal(0) < amount < Decimal("0.00000001"):
            logger.warning("Dust amount (< 1 satoshi): %s", amount)

        return amount.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN)

    def _validate_balance_update(
        self,
        new_balance: Decimal,
        operation: str = "unknown",
        _metadata: dict | None = None,
    ) -> bool:
        """
        Validate a balance update before applying it.

        Args:
            new_balance: The proposed new balance
            operation: Description of the operation for logging

        Returns:
            True if the balance update is valid

        Raises:
            ValueError: If the balance update would result in invalid state
        """

        def _raise_balance_rejection_error(balance: Decimal) -> None:
            raise ValueError(f"Balance update rejected: would result in ${balance}")

        try:
            # Normalize to check for validation issues
            normalized_balance = self._normalize_balance(new_balance)

            # Check for critical issues
            if normalized_balance < Decimal(-1000):
                logger.error(
                    "Critical: Balance would become severely negative: $%s from %s",
                    normalized_balance,
                    operation,
                )
                _raise_balance_rejection_error(normalized_balance)

            # Check for suspicious large changes
            if self.current_balance > Decimal(0):
                change_ratio = (
                    abs(normalized_balance - self.current_balance)
                    / self.current_balance
                )
                if change_ratio > Decimal(10):  # 1000% change
                    logger.warning(
                        "Large balance change detected: $%s -> $%s from %s",
                        self.current_balance,
                        normalized_balance,
                        operation,
                    )

            return True

        except Exception as e:
            logger.exception("Balance validation failed for %s", operation)
            raise ValueError(f"Invalid balance update: {e}") from e

    def get_account_status(
        self, current_prices: dict[str, Decimal] | None = None
    ) -> dict[str, Any]:
        """Get current account status."""
        with self._lock:
            # Calculate unrealized P&L for open trades
            unrealized_pnl = Decimal(0)
            for trade in self.open_trades.values():
                if current_prices and trade.symbol in current_prices:
                    price = current_prices[trade.symbol]
                else:
                    price = self._get_current_price(trade.symbol)
                unrealized_pnl += trade.calculate_unrealized_pnl(price)

            # Normalize equity calculation
            self.equity = self._normalize_balance(self.current_balance + unrealized_pnl)

            # Calculate performance metrics
            total_pnl = self.equity - self.starting_balance
            roi = (
                (total_pnl / self.starting_balance * 100)
                if self.starting_balance > 0
                else Decimal(0)
            )

            # Update drawdown
            self.peak_equity = max(self.peak_equity, self.equity)
            current_drawdown = (
                (self.peak_equity - self.equity) / self.peak_equity * 100
                if self.peak_equity > 0
                else Decimal(0)
            )
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

            return {
                "starting_balance": float(
                    self._normalize_balance(self.starting_balance)
                ),
                "current_balance": float(self._normalize_balance(self.current_balance)),
                "equity": float(self._normalize_balance(self.equity)),
                "unrealized_pnl": float(self._normalize_balance(unrealized_pnl)),
                "total_pnl": float(self._normalize_balance(total_pnl)),
                "roi_percent": float(roi),
                "margin_used": float(self._normalize_balance(self.margin_used)),
                "margin_available": float(
                    self._normalize_balance(self.equity - self.margin_used)
                ),
                "open_positions": len(self.open_trades),
                "total_trades": len(self.closed_trades),
                "peak_equity": float(self._normalize_balance(self.peak_equity)),
                "current_drawdown": float(current_drawdown),
                "max_drawdown": float(self.max_drawdown),
            }

    def execute_trade_action(
        self, action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Execute a trade action in paper trading mode using real market data.

        This method simulates what would happen if the trade were placed on the real exchange,
        using actual market prices and comprehensive logging to show the trading decisions.

        Args:
            action: Trade action to execute
            symbol: Trading symbol
            current_price: Current market price (from real market data)

        Returns:
            Order object representing the simulated trade
        """
        # Record trade action start in monitoring
        balance_before = float(self.current_balance)

        with self._lock:
            try:
                if action.action == "HOLD":
                    logger.info(
                        "🛑 PAPER TRADING DECISION: HOLD | Symbol: %s | "
                        "Current Price: $%s | Reason: %s",
                        symbol,
                        current_price,
                        action.rationale,
                    )

                    # Record HOLD decision in monitoring
                    self._record_balance_operation(
                        operation="trade_decision_hold",
                        balance_before=balance_before,
                        balance_after=balance_before,
                        success=True,
                        metadata={
                            "action": action.action,
                            "symbol": symbol,
                            "price": float(current_price),
                            "rationale": action.rationale,
                        },
                    )
                    return None

                # Log the trading decision with full context
                logger.info(
                    "🎯 PAPER TRADING DECISION: %s | Symbol: %s | "
                    "Current Price: $%s | Size: %s%% | "
                    "Reason: %s",
                    action.action,
                    symbol,
                    current_price,
                    action.size_pct,
                    action.rationale,
                )

                # Calculate trade size based on action
                trade_size = self._calculate_trade_size(action, current_price, symbol)
                if trade_size <= 0:
                    logger.warning("❌ Invalid trade size calculated: %s", trade_size)
                    return None

                # Log what would happen with real market execution
                trade_value = trade_size * current_price
                required_margin = trade_value / Decimal(str(settings.trading.leverage))

                logger.info(
                    "📊 TRADE SIMULATION DETAILS:"
                    "\n  • Symbol: %s"
                    "\n  • Action: %s"
                    "\n  • Current Real Price: $%s"
                    "\n  • Position Size: %s %s"
                    "\n  • Position Value: $%s"
                    "\n  • Leverage: %sx"
                    "\n  • Required Margin: $%s"
                    "\n  • Available Balance: $%s"
                    "\n  • Stop Loss: %s%%"
                    "\n  • Take Profit: %s%%",
                    symbol,
                    action.action,
                    current_price,
                    trade_size,
                    symbol.split("-")[0],
                    trade_value,
                    settings.trading.leverage,
                    required_margin,
                    self.equity - self.margin_used,
                    action.stop_loss_pct,
                    action.take_profit_pct,
                )

                # Apply realistic slippage based on real market conditions
                execution_price = self._apply_slippage(current_price, action.action)
                slippage_amount = abs(execution_price - current_price)
                slippage_pct = (slippage_amount / current_price) * 100

                logger.info(
                    "📈 EXECUTION SIMULATION:"
                    "\n  • Market Price: $%s"
                    "\n  • Execution Price: $%s (slippage: %s%%)"
                    "\n  • Slippage Cost: $%s",
                    current_price,
                    execution_price,
                    f"{slippage_pct:.3f}",
                    f"{slippage_amount * trade_size:.2f}",
                )

                # Calculate realistic trading fees using the fee calculator
                trade_fees = fee_calculator.calculate_trade_fees(
                    action, trade_value, execution_price, is_market_order=True
                )

                # For paper trading, we only charge entry fee for now
                fees = trade_fees.entry_fee

                logger.info(
                    "💰 FEE CALCULATION:"
                    "\n  • Fee Rate: %s"
                    "\n  • Entry Fee: $%s"
                    "\n  • Trade Value: $%s",
                    f"{trade_fees.fee_rate:.4%}",
                    f"{fees:.4f}",
                    f"{trade_value:.2f}",
                )

                # Check available balance
                if action.action in ["LONG", "SHORT"] and required_margin + fees > (
                    self.equity - self.margin_used
                ):
                    # Opening new position
                    logger.warning(
                        "❌ INSUFFICIENT FUNDS SIMULATION:"
                        "\n  • Required: $%.2f"
                        "\n  • Available: $%.2f"
                        "\n  • Shortfall: $%.2f",
                        required_margin + fees,
                        self.equity - self.margin_used,
                        (required_margin + fees) - (self.equity - self.margin_used),
                    )
                    return self._create_failed_order(
                        action, symbol, "INSUFFICIENT_FUNDS"
                    )

                # Execute the simulated trade
                order = self._execute_paper_trade(
                    action, symbol, execution_price, trade_size, fees
                )

                if order and order.status == OrderStatus.FILLED:
                    # Calculate potential profit/loss scenarios
                    stop_loss_price = (
                        execution_price * (1 - Decimal(str(action.stop_loss_pct)) / 100)
                        if action.action == "LONG"
                        else execution_price
                        * (1 + Decimal(str(action.stop_loss_pct)) / 100)
                    )
                    take_profit_price = (
                        execution_price
                        * (1 + Decimal(str(action.take_profit_pct)) / 100)
                        if action.action == "LONG"
                        else execution_price
                        * (1 - Decimal(str(action.take_profit_pct)) / 100)
                    )

                    stop_loss_pnl = (
                        (stop_loss_price - execution_price) * trade_size
                        if action.action == "LONG"
                        else (execution_price - stop_loss_price) * trade_size
                    )
                    take_profit_pnl = (
                        (take_profit_price - execution_price) * trade_size
                        if action.action == "LONG"
                        else (execution_price - take_profit_price) * trade_size
                    )

                    # Add comprehensive execution logging
                    if settings.trading.enable_futures and symbol == "ETH-USD":
                        num_contracts = int(trade_size / Decimal("0.1"))
                        logger.info(
                            "✅ PAPER TRADING FUTURES EXECUTION COMPLETE:"
                            "\n  🎯 Action: %s"
                            "\n  📊 Contracts: %s contracts (%s ETH)"
                            "\n  💵 Price: $%s"
                            "\n  💸 Value: $%.2f"
                            "\n  🏷️ Fees: $%.4f @ %.4f%%"
                            "\n  🔴 Stop Loss: $%.2f ($%+.2f)"
                            "\n  🟢 Take Profit: $%.2f ($%+.2f)"
                            "\n  🏦 New Balance: $%.2f",
                            action.action,
                            num_contracts,
                            trade_size,
                            execution_price,
                            trade_size * execution_price,
                            fees,
                            trade_fees.fee_rate * 100,
                            stop_loss_price,
                            stop_loss_pnl,
                            take_profit_price,
                            take_profit_pnl,
                            self.current_balance,
                        )
                    else:
                        logger.info(
                            "✅ PAPER TRADING EXECUTION COMPLETE:"
                            "\n  🎯 Action: %s"
                            "\n  📊 Size: %s %s"
                            "\n  💵 Price: $%.2f"
                            "\n  💸 Value: $%.2f"
                            "\n  🏷️ Fees: $%.4f @ %.4f%%"
                            "\n  🔴 Stop Loss: $%.2f ($%+.2f)"
                            "\n  🟢 Take Profit: $%.2f ($%+.2f)"
                            "\n  🏦 New Balance: $%.2f"
                            "\n  📈 New Equity: $%.2f"
                            "\n  🔒 Margin Used: $%.2f"
                            "\n  💳 Available: $%.2f"
                            "\n  📋 Order ID: %s",
                            action.action,
                            trade_size,
                            symbol,
                            execution_price,
                            trade_size * execution_price,
                            fees,
                            trade_fees.fee_rate * 100,
                            stop_loss_price,
                            stop_loss_pnl,
                            take_profit_price,
                            take_profit_pnl,
                            self.current_balance,
                            self.equity,
                            self.margin_used,
                            self.equity - self.margin_used,
                            order.id,
                        )

                    # Also log a summary for easy scanning
                    logger.info(
                        "🎬 TRADE SUMMARY: %s %s %s @ $%.2f | "
                        "Value: $%.2f | Balance: $%.2f → $%.2f",
                        action.action,
                        trade_size,
                        symbol,
                        execution_price,
                        trade_value,
                        self.current_balance,
                        self.equity,
                    )

                    # Record successful trade execution in monitoring
                    balance_after = float(self.current_balance)
                    self._record_balance_operation(
                        operation="trade_execution",
                        balance_before=balance_before,
                        balance_after=balance_after,
                        success=True,
                        metadata={
                            "action": action.action,
                            "symbol": symbol,
                            "execution_price": float(execution_price),
                            "trade_size": float(trade_size),
                            "trade_value": float(trade_value),
                            "fees": float(fees),
                            "order_id": order.id if order else None,
                            "equity": float(self.equity),
                            "margin_used": float(self.margin_used),
                        },
                    )

                return order

            except Exception as e:
                logger.exception("❌ Error simulating paper trade")

                # Record failed trade execution in monitoring
                self._record_balance_operation(
                    operation="trade_execution",
                    balance_before=balance_before,
                    balance_after=balance_before,
                    success=False,
                    error_type="trade_execution_error",
                    metadata={
                        "action": action.action,
                        "symbol": symbol,
                        "price": float(current_price),
                        "error": str(e),
                    },
                )

                return self._create_failed_order(action, symbol, f"ERROR: {e!s}")

    def _calculate_trade_size(
        self, action: TradeAction, current_price: Decimal, symbol: str
    ) -> Decimal:
        """Calculate trade size based on action parameters."""
        if action.action == "CLOSE":
            # Close existing position
            existing_position = self._find_open_position(symbol)
            return existing_position.size if existing_position else Decimal(0)

        # Calculate position size based on percentage of equity
        position_value = self.equity * (Decimal(str(action.size_pct)) / 100)
        leveraged_value = position_value * Decimal(str(settings.trading.leverage))

        # Check if this is a futures trade for ETH
        trading_symbol = (
            action.symbol if hasattr(action, "symbol") else settings.trading.symbol
        )
        if settings.trading.enable_futures and trading_symbol == "ETH-USD":
            # Apply futures contract size logic for ETH
            contract_size = Decimal("0.1")  # 0.1 ETH per contract

            # Check if we're using fixed contract size from config
            if settings.trading.fixed_contract_size:
                # Use fixed number of contracts
                num_contracts = settings.trading.fixed_contract_size
                trade_size = contract_size * num_contracts
                logger.debug(
                    "Using fixed contract size: %s contracts = %s ETH",
                    num_contracts,
                    trade_size,
                )
            else:
                # Calculate quantity in ETH based on position value
                quantity_in_eth = leveraged_value / current_price

                # Convert to number of contracts and round down
                num_contracts = int(quantity_in_eth / contract_size)
                num_contracts = max(1, num_contracts)  # Minimum 1 contract

                # Return the actual quantity in ETH (multiples of 0.1)
                trade_size = contract_size * num_contracts

                logger.debug(
                    "Futures contract calculation: %.6f ETH -> %s contracts = %s ETH",
                    quantity_in_eth,
                    num_contracts,
                    trade_size,
                )
        else:
            # For spot trading or non-ETH futures, use the original calculation
            trade_size = leveraged_value / current_price
            # Normalize to 8 decimal places for crypto precision
            trade_size = self._normalize_crypto_amount(trade_size)

        return trade_size

    def _apply_slippage(self, price: Decimal, action: str) -> Decimal:
        """Apply realistic slippage to trade execution."""
        slippage_amount = price * self.slippage_rate

        # Market orders typically get worse fill prices
        if action in ["LONG", "BUY"]:
            return price + slippage_amount  # Pay slightly more when buying
        # SHORT, SELL
        return price - slippage_amount  # Receive slightly less when selling

    def _execute_paper_trade(
        self,
        action: TradeAction,
        symbol: str,
        price: Decimal,
        size: Decimal,
        fees: Decimal,
    ) -> Order:
        """Execute the actual paper trade."""
        trade_id = f"paper_{self.trade_counter:06d}"
        self.trade_counter += 1
        current_time = datetime.now(UTC)

        if action.action == "CLOSE":
            # Close existing position
            return self._close_position(symbol, price, current_time, fees)

        if action.action in ["LONG", "SHORT"]:
            # Check if we already have an open position for this symbol
            existing_position = self._find_open_position(symbol)
            if existing_position:
                # Handle different scenarios for existing positions
                if existing_position.side == action.action:
                    # Same direction - increase position size
                    logger.info(
                        "Adding to existing %s position for %s", action.action, symbol
                    )
                    return self._increase_position(
                        existing_position, price, size, fees, current_time
                    )
                # Opposite direction - close existing position and open new one
                logger.info(
                    "Closing existing %s position and opening new %s position for %s",
                    existing_position.side,
                    action.action,
                    symbol,
                )
                # First close the existing position
                close_order = self._close_position(symbol, price, current_time, fees)
                if close_order.status != OrderStatus.FILLED:
                    logger.error(
                        "Failed to close existing position: %s", close_order.status
                    )
                    return self._create_failed_order(action, symbol, "CLOSE_FAILED")

                    # Now continue to open the new position (fall through to position opening logic)

            # Open new position
            trade = PaperTrade(
                id=trade_id,
                symbol=symbol,
                side=action.action,
                entry_time=current_time,
                entry_price=self._normalize_balance(price),
                size=self._normalize_crypto_amount(size),
                fees=self._normalize_balance(fees),
                slippage=self._normalize_balance(
                    abs(price - (price / (1 + self.slippage_rate)))
                ),
            )

            self.open_trades[trade_id] = trade

            # Calculate slippage cost based on the difference between execution price and market price
            # Since price is the execution price (with slippage), calculate the market price
            slippage_amount = (
                price * self.slippage_rate
            )  # This matches the slippage calculation
            slippage_cost = self._normalize_balance(slippage_amount * size)

            # Update account balance and margin with proper normalization
            trade_value = self._normalize_balance(size * price)
            required_margin = self._normalize_balance(
                trade_value / Decimal(str(settings.trading.leverage))
            )
            new_margin_used = self._normalize_balance(
                self.margin_used + required_margin
            )

            # Deduct both fees and slippage cost from balance
            total_costs = self._normalize_balance(fees + slippage_cost)
            new_balance = self._normalize_balance(self.current_balance - total_costs)

            # Validate balance update before applying
            try:
                self._validate_balance_update(
                    new_balance, f"trade_execution_{action.action}"
                )
                self.margin_used = new_margin_used
                self.current_balance = new_balance
                logger.debug("✅ Balance update validated: $%s", self.current_balance)
            except ValueError as e:
                logger.exception(
                    "❌ Balance validation failed during trade execution - trade rejected"
                )
                return self._create_failed_order(
                    action, symbol, f"BALANCE_VALIDATION_FAILED: {e}"
                )

            # Create order object
            order = Order(
                id=trade_id,
                symbol=symbol,
                side="BUY" if action.action == "LONG" else "SELL",
                type="MARKET",
                quantity=size,
                price=price,
                filled_quantity=size,
                status=OrderStatus.FILLED,
                timestamp=current_time,
            )

            # Log with contract information for futures
            if settings.trading.enable_futures and symbol == "ETH-USD":
                num_contracts = int(size / Decimal("0.1"))
                logger.info(
                    "📈 Paper Trading FUTURES: Opened %s position | "
                    "%s contracts (%s ETH) @ $%.2f | Trade ID: %s",
                    action.action,
                    num_contracts,
                    size,
                    price,
                    trade_id,
                )
            else:
                logger.info(
                    "📈 Paper Trading: Opened %s position | %s %s @ $%.2f | Trade ID: %s",
                    action.action,
                    size,
                    symbol,
                    price,
                    trade_id,
                )

            # Save state immediately after opening position
            self._save_state()
            return order

        return self._create_failed_order(action, symbol, "INVALID_ACTION")

    def _close_position(
        self, symbol: str, price: Decimal, close_time: datetime, fees: Decimal
    ) -> Order:
        """Close an existing position."""
        # Find open position for symbol
        trade_to_close = None
        for trade in self.open_trades.values():
            if trade.symbol == symbol:
                trade_to_close = trade
                break

        if not trade_to_close:
            logger.warning("No open position found for %s", symbol)
            return self._create_failed_order(
                TradeAction(
                    action="CLOSE",
                    size_pct=0,
                    take_profit_pct=0.0,
                    stop_loss_pct=0.0,
                    rationale="No position",
                ),
                symbol,
                "NO_POSITION",
            )

        # Calculate slippage cost for closing
        slippage_amount = price * self.slippage_rate
        slippage_cost = self._normalize_balance(slippage_amount * trade_to_close.size)

        # Calculate realized P&L
        realized_pnl = trade_to_close.close_trade(price, close_time, fees)

        # Update account with proper normalization - deduct additional slippage cost
        balance_change = self._normalize_balance(realized_pnl - fees - slippage_cost)
        new_balance = self._normalize_balance(self.current_balance + balance_change)
        trade_value = self._normalize_balance(
            trade_to_close.size * trade_to_close.entry_price
        )
        released_margin = self._normalize_balance(
            trade_value / Decimal(str(settings.trading.leverage))
        )
        new_margin_used = self._normalize_balance(self.margin_used - released_margin)

        # Validate balance update
        if self._validate_balance_update(
            new_balance,
            "position_close",
            {
                "trade_id": trade_to_close.id,
                "symbol": symbol,
                "realized_pnl": float(realized_pnl),
                "fees": float(fees),
                "slippage_cost": float(slippage_cost),
            },
        ):
            self.current_balance = new_balance
            self.margin_used = new_margin_used
        else:
            logger.warning(
                "⚠️ Balance validation failed during position close - continuing with original values"
            )
            # For closing positions, we proceed anyway but log the issue
            self.current_balance = new_balance
            self.margin_used = new_margin_used

        # Move to closed trades
        self.closed_trades.append(trade_to_close)
        del self.open_trades[trade_to_close.id]

        # Create order
        order = Order(
            id=f"close_{trade_to_close.id}",
            symbol=symbol,
            side="SELL" if trade_to_close.side == "LONG" else "BUY",
            type="MARKET",
            quantity=trade_to_close.size,
            price=price,
            filled_quantity=trade_to_close.size,
            status=OrderStatus.FILLED,
            timestamp=close_time,
        )

        logger.info(
            "📊 Paper Trading: Closed %s position | %s %s @ $%.2f | "
            "P&L: $%.2f (%s%.2f%%) | %s",
            trade_to_close.side,
            trade_to_close.size,
            symbol,
            price,
            realized_pnl,
            "+" if realized_pnl > 0 else "",
            realized_pnl / trade_to_close.entry_price * 100,
            "✅ WIN" if realized_pnl > 0 else "❌ LOSS",
        )

        # Save state immediately after closing position
        self._save_state()
        return order

    def _find_open_position(self, symbol: str) -> PaperTrade | None:
        """Find open position for a symbol."""
        for trade in self.open_trades.values():
            if trade.symbol == symbol:
                return trade
        return None

    def _increase_position(
        self,
        existing_position: PaperTrade,
        price: Decimal,
        size: Decimal,
        fees: Decimal,
        current_time: datetime,
    ) -> Order:
        """Increase the size of an existing position."""
        trade_id = f"paper_{self.trade_counter:06d}"
        self.trade_counter += 1

        # Calculate average entry price (weighted by position sizes)
        current_value = existing_position.size * existing_position.entry_price
        new_value = size * price
        total_size = self._normalize_crypto_amount(existing_position.size + size)
        average_price = self._normalize_balance(
            (current_value + new_value) / total_size
        )

        # Update existing position
        existing_position.size = total_size
        existing_position.entry_price = average_price
        existing_position.fees += fees

        # Calculate slippage cost for the increase
        slippage_amount = price * self.slippage_rate
        slippage_cost = self._normalize_balance(slippage_amount * size)

        # Update account balance and margin with proper normalization
        trade_value = self._normalize_balance(size * price)
        required_margin = self._normalize_balance(
            trade_value / Decimal(str(settings.trading.leverage))
        )
        new_margin_used = self._normalize_balance(self.margin_used + required_margin)

        # Deduct both fees and slippage cost from balance
        total_costs = self._normalize_balance(fees + slippage_cost)
        new_balance = self._normalize_balance(self.current_balance - total_costs)

        # Validate balance update
        if self._validate_balance_update(
            new_balance,
            "position_increase",
            {
                "trade_id": trade_id,
                "symbol": existing_position.symbol,
                "size_increase": float(size),
                "fees": float(fees),
                "slippage_cost": float(slippage_cost),
            },
        ):
            self.margin_used = new_margin_used
            self.current_balance = new_balance
        else:
            logger.error(
                "❌ Balance validation failed during position increase - operation rejected"
            )
            return self._create_failed_order(
                TradeAction(
                    action="LONG" if existing_position.side == "LONG" else "SHORT",
                    size_pct=0,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.5,
                    rationale="Balance validation failed",
                ),
                existing_position.symbol,
                "BALANCE_VALIDATION_FAILED",
            )

        # Create order object for the increase
        order = Order(
            id=trade_id,
            symbol=existing_position.symbol,
            side="BUY" if existing_position.side == "LONG" else "SELL",
            type="MARKET",
            quantity=size,
            price=price,
            filled_quantity=size,
            status=OrderStatus.FILLED,
            timestamp=current_time,
        )

        logger.info(
            "📈 Paper Trading: Increased %s position | "
            "+%s %s @ $%.2f | "
            "Total size: %s @ avg $%.4f | Trade ID: %s",
            existing_position.side,
            size,
            existing_position.symbol,
            price,
            total_size,
            average_price,
            trade_id,
        )

        # Save state immediately after increasing position
        self._save_state()
        return order

    def _create_failed_order(
        self, action: TradeAction, symbol: str, _reason: str
    ) -> Order:
        """Create a failed order object."""
        return Order(
            id=f"failed_{self.trade_counter}",
            symbol=symbol,
            side="BUY" if action.action == "LONG" else "SELL",
            type="MARKET",
            quantity=Decimal(0),
            price=Decimal(0),
            filled_quantity=Decimal(0),
            status=OrderStatus.REJECTED,
            timestamp=datetime.now(UTC),
        )

    def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol from real market data."""
        try:
            # Try to get real market data from the price conversion utility
            from .utils.price_conversion import get_current_real_price

            real_price = get_current_real_price(symbol)
            if real_price:
                logger.debug(
                    "📊 Using real market price for %s: $%s", symbol, real_price
                )
                return Decimal(str(real_price))
        except (ImportError, Exception) as e:
            logger.debug("Could not fetch real price for %s: %s", symbol, e)

        # Fallback to realistic prices based on corrected price data
        if "SUI" in symbol:
            price = Decimal("3.50")  # Realistic SUI price post-fix
        elif "ETH" in symbol:
            price = Decimal(2500)  # Realistic ETH price
        elif "BTC" in symbol:
            price = Decimal(50000)  # Realistic BTC price
        else:
            price = Decimal(100)  # Generic reasonable price

        logger.debug("📊 Using fallback price for %s: $%s", symbol, price)
        return price

    def update_daily_performance(self) -> None:
        """Update daily performance metrics."""
        with self._lock:
            today = datetime.now(UTC).date().isoformat()

            if today in self.daily_metrics:
                return  # Already updated today

            # Calculate daily metrics
            account_status = self.get_account_status()

            # Count trades for today
            today_trades_opened = len(
                [
                    t
                    for t in self.open_trades.values()
                    if t.entry_time.date().isoformat() == today
                ]
            )
            today_trades_closed = len(
                [
                    t
                    for t in self.closed_trades
                    if t.exit_time and t.exit_time.date().isoformat() == today
                ]
            )

            # Calculate P&L for today
            today_realized = sum(
                t.realized_pnl
                for t in self.closed_trades
                if t.exit_time
                and t.exit_time.date().isoformat() == today
                and t.realized_pnl is not None
            ) or Decimal(0)
            today_fees = sum(
                t.fees
                for t in self.closed_trades
                if t.exit_time
                and t.exit_time.date().isoformat() == today
                and t.fees is not None
            ) or Decimal(0)

            # Win rate calculation
            today_closed_trades = [
                t
                for t in self.closed_trades
                if t.exit_time and t.exit_time.date().isoformat() == today
            ]
            winning_trades = len(
                [
                    t
                    for t in today_closed_trades
                    if t.realized_pnl is not None and t.realized_pnl > 0
                ]
            )
            win_rate = (
                (winning_trades / len(today_closed_trades) * 100)
                if today_closed_trades
                else 0
            )

            # Largest win/loss
            if today_closed_trades:
                pnl_values = [
                    t.realized_pnl
                    for t in today_closed_trades
                    if t.realized_pnl is not None
                ]
                if pnl_values:
                    largest_win = max(pnl_values)
                    largest_loss = min(pnl_values)
                else:
                    largest_win = Decimal(0)
                    largest_loss = Decimal(0)
            else:
                largest_win = Decimal(0)
                largest_loss = Decimal(0)

            # Get previous day's balance
            yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
            yesterday_metrics = self.daily_metrics.get(yesterday)
            if yesterday_metrics:
                prev_balance = yesterday_metrics.ending_balance
            else:
                prev_balance = self.starting_balance

            daily_perf = DailyPerformance(
                date=today,
                starting_balance=Decimal(str(prev_balance)),
                ending_balance=Decimal(str(account_status["equity"])),
                trades_opened=today_trades_opened,
                trades_closed=today_trades_closed,
                realized_pnl=Decimal(str(today_realized)),
                unrealized_pnl=Decimal(str(account_status["unrealized_pnl"])),
                total_pnl=today_realized
                + Decimal(str(account_status["unrealized_pnl"])),
                fees_paid=today_fees,
                win_rate=win_rate,
                largest_win=Decimal(str(largest_win)),
                largest_loss=Decimal(str(largest_loss)),
                drawdown=Decimal(str(account_status["current_drawdown"])),
            )

            self.daily_metrics[today] = daily_perf
            self._save_state()

    def get_trade_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get trade history for the last N days."""
        with self._lock:
            cutoff_date = datetime.now(UTC) - timedelta(days=days)

            recent_trades = [
                trade
                for trade in self.closed_trades
                if trade.exit_time and trade.exit_time >= cutoff_date
            ]

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
                    "duration_hours": (
                        (trade.exit_time - trade.entry_time).total_seconds() / 3600
                        if trade.exit_time
                        else None
                    ),
                }
                for trade in recent_trades
            ]

    def get_performance_summary(self, days: int = 7) -> dict[str, Any]:
        """Get performance summary for the last N days."""
        with self._lock:
            cutoff_date = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()

            # Get relevant daily metrics
            relevant_metrics = {
                date: metrics
                for date, metrics in self.daily_metrics.items()
                if date >= cutoff_date
            }

            if not relevant_metrics:
                return {
                    "message": "No performance data available for the specified period"
                }

            # Calculate summary statistics
            total_trades = sum(m.trades_closed for m in relevant_metrics.values())
            total_realized_pnl = sum(m.realized_pnl for m in relevant_metrics.values())
            total_fees = sum(m.fees_paid for m in relevant_metrics.values())

            winning_days = len(
                [m for m in relevant_metrics.values() if m.total_pnl > 0]
            )
            total_days = len(relevant_metrics)

            avg_daily_pnl = (
                total_realized_pnl / total_days if total_days > 0 else Decimal(0)
            )
            max_daily_gain = (
                max(m.total_pnl for m in relevant_metrics.values())
                if relevant_metrics
                else Decimal(0)
            )
            max_daily_loss = (
                min(m.total_pnl for m in relevant_metrics.values())
                if relevant_metrics
                else Decimal(0)
            )

            # Win rate across all trades
            closed_trades_period = [
                t
                for t in self.closed_trades
                if t.exit_time and t.exit_time.date().isoformat() >= cutoff_date
            ]
            winning_trades = len(
                [
                    t
                    for t in closed_trades_period
                    if t.realized_pnl is not None and t.realized_pnl > 0
                ]
            )
            overall_win_rate = (
                (winning_trades / len(closed_trades_period) * 100)
                if closed_trades_period
                else 0
            )

            account_status = self.get_account_status()

            return {
                "period_days": days,
                "total_trades": total_trades,
                "total_realized_pnl": float(total_realized_pnl),
                "total_fees_paid": float(total_fees),
                "net_pnl": float(total_realized_pnl - total_fees),
                "avg_daily_pnl": float(avg_daily_pnl),
                "max_daily_gain": float(max_daily_gain),
                "max_daily_loss": float(max_daily_loss),
                "winning_days": winning_days,
                "total_days": total_days,
                "win_day_rate": (
                    (winning_days / total_days * 100) if total_days > 0 else 0
                ),
                "overall_win_rate": overall_win_rate,
                "current_equity": float(account_status["equity"]),
                "roi_percent": float(account_status["roi_percent"]),
                "max_drawdown": float(account_status["max_drawdown"]),
                "sharp_ratio": self._calculate_sharpe_ratio(relevant_metrics),
            }

    def _calculate_sharpe_ratio(
        self, daily_metrics: dict[str, DailyPerformance]
    ) -> float:
        """Calculate Sharpe ratio for the period."""
        if len(daily_metrics) < 2:
            return 0.0

        daily_returns = [
            float(m.total_pnl / m.starting_balance) for m in daily_metrics.values()
        ]

        mean_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0

        # Annualized Sharpe ratio (assuming 252 trading days)
        return (mean_return * 252 / std_return) if std_return > 0 else 0

    def get_performance_metrics_for_monitor(self) -> list[dict[str, Any]]:
        """
        Get performance metrics formatted for the PerformanceMonitor system.

        Returns:
            List of metric dictionaries with name, value, timestamp, unit, and tags
        """
        with self._lock:
            metrics = []
            timestamp = datetime.now(UTC)
            account_status = self.get_account_status()

            # Core financial metrics
            metrics.extend(
                [
                    {
                        "name": "paper_trading.equity",
                        "value": account_status["equity"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.balance",
                        "value": account_status["current_balance"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.total_pnl",
                        "value": account_status["total_pnl"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.roi_percent",
                        "value": account_status["roi_percent"],
                        "timestamp": timestamp,
                        "unit": "percent",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.drawdown_percent",
                        "value": account_status["current_drawdown"],
                        "timestamp": timestamp,
                        "unit": "percent",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.max_drawdown_percent",
                        "value": account_status["max_drawdown"],
                        "timestamp": timestamp,
                        "unit": "percent",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                ]
            )

            # Trading activity metrics
            metrics.extend(
                [
                    {
                        "name": "paper_trading.open_positions",
                        "value": account_status["open_positions"],
                        "timestamp": timestamp,
                        "unit": "count",
                        "tags": {"mode": "paper", "type": "positions"},
                    },
                    {
                        "name": "paper_trading.total_trades",
                        "value": account_status["total_trades"],
                        "timestamp": timestamp,
                        "unit": "count",
                        "tags": {"mode": "paper", "type": "trades"},
                    },
                    {
                        "name": "paper_trading.margin_used",
                        "value": account_status["margin_used"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "type": "margin"},
                    },
                    {
                        "name": "paper_trading.margin_available",
                        "value": account_status["margin_available"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "type": "margin"},
                    },
                ]
            )

            # Calculate win rate if we have closed trades
            if self.closed_trades:
                winning_trades = len(
                    [
                        t
                        for t in self.closed_trades
                        if t.realized_pnl is not None and t.realized_pnl > 0
                    ]
                )
                win_rate = (winning_trades / len(self.closed_trades)) * 100

                metrics.append(
                    {
                        "name": "paper_trading.win_rate_percent",
                        "value": win_rate,
                        "timestamp": timestamp,
                        "unit": "percent",
                        "tags": {"mode": "paper", "type": "performance"},
                    }
                )

                # Average trade duration in hours
                durations = []
                for trade in self.closed_trades:
                    if trade.exit_time and trade.entry_time:
                        duration = (
                            trade.exit_time - trade.entry_time
                        ).total_seconds() / 3600
                        durations.append(duration)

                if durations:
                    avg_duration = sum(durations) / len(durations)
                    metrics.append(
                        {
                            "name": "paper_trading.avg_trade_duration_hours",
                            "value": avg_duration,
                            "timestamp": timestamp,
                            "unit": "hours",
                            "tags": {"mode": "paper", "type": "performance"},
                        }
                    )

            return metrics

    def generate_daily_report(self, date: str | None = None) -> str:
        """Generate a daily performance report."""
        target_date = date or datetime.now(UTC).date().isoformat()

        if target_date not in self.daily_metrics:
            return f"No trading data available for {target_date}"

        metrics = self.daily_metrics[target_date]
        account_status = self.get_account_status()

        return f"""
🗓️  Daily Trading Report - {target_date}
{"=" * 50}

💰 Account Status:
   Starting Balance: ${metrics.starting_balance:,.2f}
   Ending Balance:   ${metrics.ending_balance:,.2f}
   Current Equity:   ${account_status["equity"]:,.2f}
   Daily P&L:        ${metrics.total_pnl:,.2f}

📊 Trading Activity:
   Trades Opened:    {metrics.trades_opened}
   Trades Closed:    {metrics.trades_closed}
   Win Rate:         {metrics.win_rate:.1f}%
   Fees Paid:        ${metrics.fees_paid:.2f}

📈 Performance:
   Largest Win:      ${metrics.largest_win:.2f}
   Largest Loss:     ${metrics.largest_loss:.2f}
   Drawdown:         {metrics.drawdown:.2f}%
   ROI (Total):      {account_status["roi_percent"]:.2f}%

🔄 Open Positions:   {account_status["open_positions"]}
💸 Margin Used:      ${account_status["margin_used"]:,.2f}
💳 Available:        ${account_status["margin_available"]:,.2f}
"""

    def _save_state(self) -> None:
        """Save current state to files with robust error handling and logging."""
        save_start_time = time.perf_counter()

        # Use file-level locking to prevent race conditions
        with self._lock:
            try:
                logger.info("📝 Starting paper trading state save...")

                # Ensure data directory exists with proper permissions
                try:
                    self.data_dir.mkdir(parents=True, exist_ok=True)
                    # Check write permissions
                    test_file = self.data_dir / ".write_test"
                    test_file.write_text("test")
                    test_file.unlink()
                except PermissionError as e:
                    raise OSError(
                        f"Permission denied writing to {self.data_dir}: {e}"
                    ) from e
                except Exception as e:
                    raise OSError(
                        f"Cannot access data directory {self.data_dir}: {e}"
                    ) from e

                # Save account state with detailed error context
                try:
                    account_data = {
                        "starting_balance": str(
                            self._normalize_balance(self.starting_balance)
                        ),
                        "current_balance": str(
                            self._normalize_balance(self.current_balance)
                        ),
                        "equity": str(self._normalize_balance(self.equity)),
                        "margin_used": str(self._normalize_balance(self.margin_used)),
                        "trade_counter": self.trade_counter,
                        "session_start_time": self.session_start_time.isoformat(),
                        "peak_equity": str(self._normalize_balance(self.peak_equity)),
                        "max_drawdown": str(self.max_drawdown),
                        "last_save_time": datetime.now(UTC).isoformat(),
                    }

                    # Write to temporary file first, then rename for atomic operation
                    temp_account_file = self.account_file.with_suffix(".tmp")
                    with temp_account_file.open("w") as f:
                        json.dump(account_data, f, indent=2)
                    temp_account_file.rename(self.account_file)
                    logger.info(
                        "✅ Account state saved: balance=$%s, trades=%s",
                        self._normalize_balance(self.current_balance),
                        self.trade_counter,
                    )

                except Exception as e:
                    logger.exception(
                        "❌ Failed to save account state to %s: %s",
                        self.account_file,
                        type(e).__name__,
                    )
                    raise OSError(f"Account save failed: {e}") from e

                # Save trades with enhanced serialization error handling
                try:
                    trades_data = {
                        "open_trades": {
                            trade_id: asdict(trade)
                            for trade_id, trade in self.open_trades.items()
                        },
                        "closed_trades": [
                            asdict(trade) for trade in self.closed_trades
                        ],
                        "save_metadata": {
                            "save_time": datetime.now(UTC).isoformat(),
                            "open_count": len(self.open_trades),
                            "closed_count": len(self.closed_trades),
                        },
                    }

                    # Enhanced serialization with better error handling
                    def serialize_value(obj):
                        try:
                            if isinstance(obj, datetime):
                                return obj.isoformat()
                            if isinstance(obj, Decimal):
                                return str(obj)
                            if obj is None:
                                return None
                            # Test if object is JSON serializable
                            json.dumps(obj)
                            return obj
                        except (TypeError, ValueError):
                            logger.warning(
                                "⚠️ Serialization issue with %s '%s'",
                                type(obj).__name__,
                                obj,
                            )
                            return str(obj)  # Fallback to string representation

                    def serialize_dict(d):
                        if isinstance(d, dict):
                            return {k: serialize_dict(v) for k, v in d.items()}
                        if isinstance(d, list):
                            return [serialize_dict(item) for item in d]
                        return serialize_value(d)

                    trades_data = serialize_dict(trades_data)

                    # Atomic write for trades file
                    temp_trades_file = self.trades_file.with_suffix(".tmp")
                    with temp_trades_file.open("w") as f:
                        json.dump(trades_data, f, indent=2)
                    temp_trades_file.rename(self.trades_file)
                    logger.info(
                        "✅ Trades saved: %s open, %s closed",
                        len(self.open_trades),
                        len(self.closed_trades),
                    )

                except Exception as e:
                    logger.exception(
                        "❌ Failed to save trades to %s: %s",
                        self.trades_file,
                        type(e).__name__,
                    )
                    raise OSError(f"Trades save failed: {e}") from e

                # Save performance data with error handling
                try:
                    performance_data = {
                        date: asdict(metrics)
                        for date, metrics in self.daily_metrics.items()
                    }
                    performance_data = serialize_dict(performance_data)
                    performance_data["save_metadata"] = {
                        "save_time": datetime.now(UTC).isoformat(),
                        "metrics_count": len(self.daily_metrics),
                    }

                    # Atomic write for performance file
                    temp_perf_file = self.performance_file.with_suffix(".tmp")
                    with temp_perf_file.open("w") as f:
                        json.dump(performance_data, f, indent=2)
                    temp_perf_file.rename(self.performance_file)
                    logger.info(
                        "✅ Performance data saved: %s daily metrics",
                        len(self.daily_metrics),
                    )

                except Exception as e:
                    logger.exception(
                        "❌ Failed to save performance data to %s: %s",
                        self.performance_file,
                        type(e).__name__,
                    )
                    raise OSError(f"Performance save failed: {e}") from e

                # Save session trades for dashboard access
                try:
                    self.save_session_trades()
                except Exception as e:
                    logger.warning(
                        "⚠️ Failed to save session trades (non-critical): %s", e
                    )
                    # Don't fail the entire save operation for this

                # Calculate and log save performance
                save_duration = (time.perf_counter() - save_start_time) * 1000
                logger.info(
                    "💾 Paper trading state saved successfully in %.1fms", save_duration
                )

                # Clean up any temporary files that might have been left behind
                for temp_file in self.data_dir.glob("*.tmp"):
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        # Log cleanup errors but don't fail the save operation
                        logger.debug(
                            "Failed to clean up temporary file %s: %s", temp_file, e
                        )

            except OSError:
                # IO-specific errors (permissions, disk space, etc.)
                logger.exception(
                    "🚨 CRITICAL: Paper trading state save failed due to IO error"
                )
                logger.info("🚨 Data directory: %s", self.data_dir)
                logger.info(
                    "🚨 Files: account=%s, trades=%s, perf=%s",
                    self.account_file.exists(),
                    self.trades_file.exists(),
                    self.performance_file.exists(),
                )
                raise  # Re-raise IO errors as they're critical

            except Exception as e:
                # Catch-all for unexpected errors
                logger.exception(
                    "🚨 CRITICAL: Unexpected error during paper trading state save: %s",
                    type(e).__name__,
                )
                logger.info(
                    "🚨 Account state: balance=%s, trades=%s",
                    self.current_balance,
                    self.trade_counter,
                )
                logger.info(
                    "🚨 Data state: open_trades=%s, closed_trades=%s",
                    len(self.open_trades),
                    len(self.closed_trades),
                )

                # Log stack trace for debugging

                logger.info("🚨 Stack trace:\n%s", traceback.format_exc())

                raise  # Re-raise unexpected errors

    def save_session_trades(self) -> None:
        """Save session trades to session_trades.json file for dashboard access."""
        try:
            session_trades_file = self.data_dir / "session_trades.json"

            # Get all trades (both open and closed) for this session
            session_trades = []

            # Add open trades with error handling
            for trade in self.open_trades.values():
                try:
                    session_trades.append(
                        {
                            "id": trade.id,
                            "symbol": trade.symbol,
                            "side": trade.side,
                            "entry_time": trade.entry_time.isoformat(),
                            "exit_time": None,
                            "entry_price": float(trade.entry_price),
                            "exit_price": None,
                            "size": float(trade.size),
                            "realized_pnl": None,
                            "unrealized_pnl": float(
                                trade.calculate_unrealized_pnl(trade.entry_price)
                            ),  # Placeholder
                            "fees": float(trade.fees),
                            "status": trade.status,
                            "duration_hours": None,
                        }
                    )
                except Exception as e:
                    logger.warning("⚠️ Error serializing open trade %s: %s", trade.id, e)
                    continue

            # Add closed trades with error handling
            for trade in self.closed_trades:
                try:
                    duration_hours = None
                    if trade.exit_time:
                        duration_hours = (
                            trade.exit_time - trade.entry_time
                        ).total_seconds() / 3600

                    session_trades.append(
                        {
                            "id": trade.id,
                            "symbol": trade.symbol,
                            "side": trade.side,
                            "entry_time": trade.entry_time.isoformat(),
                            "exit_time": (
                                trade.exit_time.isoformat() if trade.exit_time else None
                            ),
                            "entry_price": float(trade.entry_price),
                            "exit_price": (
                                float(trade.exit_price) if trade.exit_price else None
                            ),
                            "size": float(trade.size),
                            "realized_pnl": (
                                float(trade.realized_pnl)
                                if trade.realized_pnl
                                else None
                            ),
                            "unrealized_pnl": 0.0,  # Closed trades have no unrealized P&L
                            "fees": float(trade.fees),
                            "status": trade.status,
                            "duration_hours": duration_hours,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        "⚠️ Error serializing closed trade %s: %s", trade.id, e
                    )
                    continue

            # Add metadata for debugging
            session_data = {
                "trades": session_trades,
                "metadata": {
                    "save_time": datetime.now(UTC).isoformat(),
                    "total_trades": len(session_trades),
                    "open_count": len(self.open_trades),
                    "closed_count": len(self.closed_trades),
                },
            }

            # Atomic write for session trades
            temp_session_file = session_trades_file.with_suffix(".tmp")
            with temp_session_file.open("w") as f:
                json.dump(session_data, f, indent=2)
            temp_session_file.rename(session_trades_file)

            logger.info(
                "📊 Session trades saved: %s trades to %s",
                len(session_trades),
                session_trades_file,
            )

        except Exception as e:
            logger.exception(
                "❌ Failed to save session trades to %s: %s",
                session_trades_file,
                type(e).__name__,
            )

    def _load_state(self) -> None:
        """Load state from files."""
        try:
            # Load account state
            if self.account_file.exists():
                with self.account_file.open() as f:
                    account_data = json.load(f)

                self.starting_balance = self._normalize_balance(
                    Decimal(
                        account_data.get("starting_balance", str(self.starting_balance))
                    )
                )
                self.current_balance = self._normalize_balance(
                    Decimal(
                        account_data.get("current_balance", str(self.current_balance))
                    )
                )
                self.equity = self._normalize_balance(
                    Decimal(account_data.get("equity", str(self.equity)))
                )
                self.margin_used = self._normalize_balance(
                    Decimal(account_data.get("margin_used", "0"))
                )
                self.trade_counter = account_data.get("trade_counter", 0)
                self.peak_equity = self._normalize_balance(
                    Decimal(account_data.get("peak_equity", str(self.peak_equity)))
                )
                self.max_drawdown = Decimal(account_data.get("max_drawdown", "0"))

                if "session_start_time" in account_data:
                    self.session_start_time = datetime.fromisoformat(
                        account_data["session_start_time"]
                    )

                logger.info(
                    "Loaded paper trading account: $%.2f balance",
                    float(self.current_balance),
                )

            # Load trades
            if self.trades_file.exists():
                with self.trades_file.open() as f:
                    trades_data = json.load(f)

                # Load open trades
                for trade_id, trade_data in trades_data.get("open_trades", {}).items():
                    trade = PaperTrade(
                        id=trade_data["id"],
                        symbol=trade_data["symbol"],
                        side=trade_data["side"],
                        entry_time=datetime.fromisoformat(trade_data["entry_time"]),
                        entry_price=Decimal(trade_data["entry_price"]),
                        size=Decimal(trade_data["size"]),
                        exit_time=(
                            datetime.fromisoformat(trade_data["exit_time"])
                            if trade_data.get("exit_time")
                            else None
                        ),
                        exit_price=(
                            Decimal(trade_data["exit_price"])
                            if trade_data.get("exit_price")
                            else None
                        ),
                        realized_pnl=(
                            Decimal(trade_data["realized_pnl"])
                            if trade_data.get("realized_pnl")
                            else None
                        ),
                        fees=Decimal(trade_data.get("fees", "0")),
                        slippage=Decimal(trade_data.get("slippage", "0")),
                        status=trade_data.get("status", "OPEN"),
                    )
                    self.open_trades[trade_id] = trade

                # Load closed trades
                for trade_data in trades_data.get("closed_trades", []):
                    trade = PaperTrade(
                        id=trade_data["id"],
                        symbol=trade_data["symbol"],
                        side=trade_data["side"],
                        entry_time=datetime.fromisoformat(trade_data["entry_time"]),
                        entry_price=Decimal(trade_data["entry_price"]),
                        size=Decimal(trade_data["size"]),
                        exit_time=(
                            datetime.fromisoformat(trade_data["exit_time"])
                            if trade_data.get("exit_time")
                            else None
                        ),
                        exit_price=(
                            Decimal(trade_data["exit_price"])
                            if trade_data.get("exit_price")
                            else None
                        ),
                        realized_pnl=(
                            Decimal(trade_data["realized_pnl"])
                            if trade_data.get("realized_pnl")
                            else None
                        ),
                        fees=Decimal(trade_data.get("fees", "0")),
                        slippage=Decimal(trade_data.get("slippage", "0")),
                        status=trade_data.get("status", "CLOSED"),
                    )
                    self.closed_trades.append(trade)

                logger.info(
                    "Loaded %s open and %s closed trades",
                    len(self.open_trades),
                    len(self.closed_trades),
                )

            # Load performance data
            if self.performance_file.exists():
                with self.performance_file.open() as f:
                    performance_data = json.load(f)

                for date, metrics_data in performance_data.items():
                    # Skip metadata entries that don't contain daily performance data
                    if (
                        date == "save_metadata"
                        or not isinstance(metrics_data, dict)
                        or "date" not in metrics_data
                    ):
                        continue

                    metrics = DailyPerformance(
                        date=metrics_data["date"],
                        starting_balance=Decimal(metrics_data["starting_balance"]),
                        ending_balance=Decimal(metrics_data["ending_balance"]),
                        trades_opened=metrics_data["trades_opened"],
                        trades_closed=metrics_data["trades_closed"],
                        realized_pnl=Decimal(metrics_data["realized_pnl"]),
                        unrealized_pnl=Decimal(metrics_data["unrealized_pnl"]),
                        total_pnl=Decimal(metrics_data["total_pnl"]),
                        fees_paid=Decimal(metrics_data["fees_paid"]),
                        win_rate=metrics_data["win_rate"],
                        largest_win=Decimal(metrics_data["largest_win"]),
                        largest_loss=Decimal(metrics_data["largest_loss"]),
                        drawdown=Decimal(metrics_data["drawdown"]),
                    )
                    self.daily_metrics[date] = metrics

                logger.info(
                    "Loaded performance data for %s days", len(self.daily_metrics)
                )

        except Exception:
            logger.exception("Failed to load paper trading state")
            # Continue with default state

    def reset_account(self, new_balance: Decimal | None = None) -> None:
        """Reset the paper trading account."""
        with self._lock:
            self.starting_balance = new_balance or Decimal(10000)
            self.current_balance = self.starting_balance
            self.equity = self.starting_balance
            self.margin_used = Decimal(0)
            self.open_trades.clear()
            self.closed_trades.clear()
            self.daily_metrics.clear()
            self.trade_counter = 0
            self.session_start_time = datetime.now(UTC)
            self.peak_equity = self.starting_balance
            self.max_drawdown = Decimal(0)

            self._save_state()
            logger.info(
                "Paper trading account reset to $%.2f", float(self.starting_balance)
            )

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get comprehensive monitoring summary for paper trading."""
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "monitoring_enabled": self.monitoring_enabled,
            "account_status": self.get_account_status(),
            "performance_metrics": self.get_performance_metrics_for_monitor(),
        }

        if self.monitoring_enabled:
            try:
                # Get enhanced metrics
                enhanced_metrics = self.metrics_collector.get_metrics_summary(
                    component="paper_trading"
                )
                summary["enhanced_metrics"] = enhanced_metrics

                # Get alert summary
                alert_summary = self.alert_manager.get_alert_summary()
                summary["alert_summary"] = alert_summary

                # Get active alerts for paper trading
                active_alerts = self.alert_manager.get_active_alerts()
                paper_trading_alerts = [
                    {
                        "id": alert.id,
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "created_at": alert.created_at.isoformat(),
                        "duration_minutes": alert.duration_minutes,
                    }
                    for alert in active_alerts
                    if alert.component == "paper_trading"
                ]
                summary["active_alerts"] = paper_trading_alerts

            except Exception as e:
                logger.warning("Failed to get enhanced monitoring summary: %s", e)
                summary["monitoring_error"] = str(e)

        return summary

    def enable_monitoring(self) -> bool:
        """Enable monitoring for this paper trading account."""
        if not MONITORING_AVAILABLE:
            logger.warning("Monitoring components not available")
            return False

        try:
            self.metrics_collector = get_balance_metrics_collector()
            self.alert_manager = get_balance_alert_manager()
            self.monitoring_enabled = True

            # Record monitoring enablement
            self._record_balance_operation(
                operation="monitoring_enabled",
                balance_before=float(self.current_balance),
                balance_after=float(self.current_balance),
                success=True,
                metadata={"enabled_at": datetime.now(UTC).isoformat()},
            )

            logger.info("✅ Paper trading monitoring enabled")
            return True

        except Exception:
            logger.exception("Failed to enable monitoring")
            return False

    def disable_monitoring(self) -> None:
        """Disable monitoring for this paper trading account."""
        if self.monitoring_enabled:
            try:
                # Record monitoring disablement before disabling
                self._record_balance_operation(
                    operation="monitoring_disabled",
                    balance_before=float(self.current_balance),
                    balance_after=float(self.current_balance),
                    success=True,
                    metadata={"disabled_at": datetime.now(UTC).isoformat()},
                )
            except Exception as e:
                logger.warning("Failed to record monitoring disablement: %s", e)

        self.monitoring_enabled = False
        logger.info("🔇 Paper trading monitoring disabled")

    def get_prometheus_metrics(self) -> list[str]:
        """Get Prometheus format metrics for paper trading."""
        metrics = []

        if self.monitoring_enabled:
            try:
                # Get enhanced prometheus metrics
                enhanced_metrics = self.metrics_collector.get_prometheus_metrics()
                metrics.extend(enhanced_metrics)
            except Exception as e:
                logger.warning("Failed to get enhanced Prometheus metrics: %s", e)

        # Add paper trading specific metrics
        account_status = self.get_account_status()
        timestamp = int(datetime.now(UTC).timestamp() * 1000)

        base_metrics = [
            f"paper_trading_balance_usd {account_status['current_balance']} {timestamp}",
            f"paper_trading_equity_usd {account_status['equity']} {timestamp}",
            f"paper_trading_pnl_usd {account_status['total_pnl']} {timestamp}",
            f"paper_trading_roi_percent {account_status['roi_percent']} {timestamp}",
            f"paper_trading_drawdown_percent {account_status['current_drawdown']} {timestamp}",
            f"paper_trading_open_positions {account_status['open_positions']} {timestamp}",
            f"paper_trading_total_trades {account_status['total_trades']} {timestamp}",
            f"paper_trading_margin_used_usd {account_status['margin_used']} {timestamp}",
        ]

        metrics.extend(base_metrics)
        return metrics
