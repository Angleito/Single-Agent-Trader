"""
Risk management and position sizing module.

This module handles risk management, position sizing, leverage control,
and daily loss limits to protect the trading account.
"""

import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from .config import settings
from .position_manager import PositionManager
from .types import Position, RiskMetrics, TradeAction

logger = logging.getLogger(__name__)


@dataclass
class DailyPnL:
    """Daily P&L tracking."""

    date: date
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    trades_count: int = 0
    max_drawdown: Decimal = Decimal("0")


class RiskManager:
    """
    Risk management system for position sizing and loss control.

    Enforces position size limits, leverage constraints, daily loss limits,
    and maximum concurrent positions to protect the trading account.
    """

    def __init__(self, position_manager: PositionManager | None = None):
        """Initialize the risk manager.

        Args:
            position_manager: Position manager instance for position data
        """
        self.max_size_pct = settings.trading.max_size_pct
        self.leverage = settings.trading.leverage
        self.max_daily_loss_pct = settings.risk.max_daily_loss_pct
        self.max_concurrent_trades = settings.risk.max_concurrent_trades

        # Position manager integration
        self.position_manager = position_manager

        # Risk tracking
        self._daily_pnl: dict[date, DailyPnL] = {}
        self._account_balance = Decimal("10000")  # Default starting balance

        logger.info(
            f"Initialized RiskManager with max size {self.max_size_pct}%, "
            f"leverage {self.leverage}x, max daily loss {self.max_daily_loss_pct}%"
        )

    def evaluate_risk(
        self,
        trade_action: TradeAction,
        current_position: Position,
        current_price: Decimal,
    ) -> tuple[bool, TradeAction, str]:
        """
        Evaluate risk for a proposed trade action.

        Args:
            trade_action: Proposed trade action
            current_position: Current position
            current_price: Current market price

        Returns:
            Tuple of (approved, modified_action, reason)
        """
        try:
            # Check daily loss limit
            if self._is_daily_loss_limit_reached():
                return (
                    False,
                    self._get_hold_action("Daily loss limit reached"),
                    "Daily loss limit",
                )

            # Check maximum concurrent positions
            if not self._can_open_new_position(trade_action, current_position):
                return (
                    False,
                    self._get_hold_action("Max concurrent positions reached"),
                    "Position limit",
                )

            # Validate position size
            modified_action = self._validate_position_size(trade_action)

            # Calculate risk metrics
            risk_metrics = self._calculate_position_risk(modified_action, current_price)

            # Check if risk is acceptable
            if risk_metrics["max_loss_usd"] > self._get_max_acceptable_loss():
                # Reduce position size
                modified_action = self._reduce_position_size(
                    modified_action, risk_metrics
                )
                logger.warning("Position size reduced due to excessive risk")

            # Final validation
            if modified_action.size_pct == 0 and trade_action.action in [
                "LONG",
                "SHORT",
            ]:
                return (
                    False,
                    self._get_hold_action("Risk too high for any position"),
                    "Risk too high",
                )

            return True, modified_action, "Risk approved"

        except Exception as e:
            logger.error(f"Risk evaluation error: {e}")
            return False, self._get_hold_action("Risk evaluation error"), "Error"

    def _is_daily_loss_limit_reached(self) -> bool:
        """
        Check if daily loss limit has been reached.

        Returns:
            True if daily loss limit exceeded
        """
        today = date.today()

        if today not in self._daily_pnl:
            return False

        daily_data = self._daily_pnl[today]
        total_pnl = daily_data.realized_pnl + daily_data.unrealized_pnl

        max_loss_usd = self._account_balance * (Decimal(str(self.max_daily_loss_pct)) / Decimal("100"))

        if total_pnl <= -max_loss_usd:
            logger.warning(f"Daily loss limit reached: {total_pnl} <= -{max_loss_usd}")
            return True

        return False

    def _can_open_new_position(
        self, trade_action: TradeAction, current_position: Position
    ) -> bool:
        """
        Check if a new position can be opened.

        Args:
            trade_action: Proposed action
            current_position: Current position

        Returns:
            True if new position can be opened
        """
        # If closing or holding, always allow
        if trade_action.action in ["CLOSE", "HOLD"]:
            return True

        # Get current position count from position manager if available
        if self.position_manager:
            active_positions = len(
                [
                    p
                    for p in self.position_manager.get_all_positions()
                    if p.side != "FLAT"
                ]
            )
        else:
            # Fallback to internal tracking
            active_positions = len([p for p in [current_position] if p.side != "FLAT"])

        # If already have a position and not closing, check if it's a new position
        if current_position.side != "FLAT":
            # Already have a position - can only close or hold unless we allow multiple positions
            if trade_action.action not in ["CLOSE", "HOLD"]:
                return active_positions < self.max_concurrent_trades

        return active_positions < self.max_concurrent_trades

    def _validate_position_size(self, trade_action: TradeAction) -> TradeAction:
        """
        Validate and adjust position size based on risk limits.

        Args:
            trade_action: Original trade action

        Returns:
            Modified trade action with validated size
        """
        modified = trade_action.copy()

        # Ensure size doesn't exceed maximum
        if modified.size_pct > self.max_size_pct:
            modified.size_pct = self.max_size_pct
            logger.warning(f"Position size capped at {self.max_size_pct}%")

        # Account for available margin
        available_margin = self._get_available_margin()
        max_allowed_size = self._calculate_max_size_for_margin(available_margin)

        if modified.size_pct > max_allowed_size:
            modified.size_pct = max_allowed_size
            logger.warning(
                f"Position size reduced to {max_allowed_size}% due to margin constraints"
            )

        return modified

    def _calculate_position_risk(
        self, trade_action: TradeAction, current_price: Decimal
    ) -> dict[str, Any]:
        """
        Calculate risk metrics for a position.

        Args:
            trade_action: Trade action to evaluate
            current_price: Current market price

        Returns:
            Dictionary with risk metrics
        """
        if trade_action.size_pct == 0:
            return {"max_loss_usd": Decimal("0"), "max_gain_usd": Decimal("0")}

        position_value = self._account_balance * (Decimal(str(trade_action.size_pct)) / Decimal("100"))
        leveraged_exposure = position_value * Decimal(str(self.leverage))

        # Calculate potential loss (stop loss)
        max_loss_pct = Decimal(str(trade_action.stop_loss_pct)) / Decimal("100")
        max_loss_usd = leveraged_exposure * max_loss_pct

        # Calculate potential gain (take profit)
        max_gain_pct = Decimal(str(trade_action.take_profit_pct)) / Decimal("100")
        max_gain_usd = leveraged_exposure * max_gain_pct

        return {
            "position_value": position_value,
            "leveraged_exposure": leveraged_exposure,
            "max_loss_usd": max_loss_usd,
            "max_gain_usd": max_gain_usd,
            "risk_reward_ratio": trade_action.take_profit_pct
            / trade_action.stop_loss_pct,
        }

    def _get_max_acceptable_loss(self) -> Decimal:
        """
        Get maximum acceptable loss per trade.

        Returns:
            Maximum loss in USD
        """
        # Risk maximum 5% of account per trade (increased for testing)
        max_loss_pct = Decimal("0.05")
        return self._account_balance * max_loss_pct

    def _reduce_position_size(
        self, trade_action: TradeAction, risk_metrics: dict[str, Any]
    ) -> TradeAction:
        """
        Reduce position size to acceptable risk level.

        Args:
            trade_action: Original trade action
            risk_metrics: Risk metrics for current size

        Returns:
            Trade action with reduced size
        """
        modified = trade_action.copy()
        max_acceptable_loss = self._get_max_acceptable_loss()

        if risk_metrics["max_loss_usd"] > 0:
            # Calculate size reduction factor
            reduction_factor = max_acceptable_loss / risk_metrics["max_loss_usd"]
            new_size_pct = int(Decimal(str(modified.size_pct)) * reduction_factor)

            # Ensure minimum viable size or zero (allow smaller positions for testing)
            if new_size_pct < 1:
                new_size_pct = 0

            modified.size_pct = new_size_pct
            logger.info(
                f"Position size reduced from {trade_action.size_pct}% to {new_size_pct}%"
            )

        return modified

    def _get_available_margin(self) -> Decimal:
        """
        Calculate available margin for new positions.

        Returns:
            Available margin as percentage of account
        """
        if self.position_manager:
            # Use position manager to get accurate margin usage
            positions = self.position_manager.get_all_positions()
            used_margin_pct = 0.0

            for pos in positions:
                if pos.side != "FLAT" and pos.entry_price is not None:
                    position_value = float(pos.size * pos.entry_price)
                    margin_used = position_value / float(self._account_balance) * 100
                    used_margin_pct += margin_used
        else:
            # Fallback to simplified calculation
            used_margin_pct = 0.0

        available_pct = 100.0 - used_margin_pct
        return Decimal(str(max(0, available_pct)))

    def _calculate_max_size_for_margin(self, available_margin: Decimal) -> int:
        """
        Calculate maximum position size based on available margin.

        Args:
            available_margin: Available margin percentage

        Returns:
            Maximum position size percentage
        """
        # For futures trading with leverage, we can use more of our available margin
        # The leverage multiplier allows larger positions with the same margin requirement
        # Available margin represents what we can actually use for position size
        max_size = available_margin
        return int(min(float(max_size), self.max_size_pct))

    def _get_hold_action(self, reason: str) -> TradeAction:
        """
        Get a safe HOLD action with reason.

        Args:
            reason: Reason for holding

        Returns:
            TradeAction with HOLD
        """
        return TradeAction(
            action="HOLD",
            size_pct=0,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale=f"Risk manager: {reason}",
        )

    def update_daily_pnl(
        self, realized_pnl: Decimal, unrealized_pnl: Decimal = Decimal("0")
    ) -> None:
        """
        Update daily P&L tracking.

        Args:
            realized_pnl: Realized profit/loss
            unrealized_pnl: Unrealized profit/loss
        """
        today = date.today()

        if today not in self._daily_pnl:
            self._daily_pnl[today] = DailyPnL(date=today)

        daily_data = self._daily_pnl[today]
        daily_data.realized_pnl += realized_pnl
        daily_data.unrealized_pnl = unrealized_pnl

        if realized_pnl != 0:
            daily_data.trades_count += 1

        # Update max drawdown
        total_pnl = daily_data.realized_pnl + daily_data.unrealized_pnl
        if total_pnl < daily_data.max_drawdown:
            daily_data.max_drawdown = total_pnl

    def update_account_balance(self, new_balance: Decimal) -> None:
        """
        Update account balance.

        Args:
            new_balance: New account balance
        """
        old_balance = self._account_balance
        self._account_balance = new_balance

        logger.info(f"Account balance updated: {old_balance} -> {new_balance}")

    def get_risk_metrics(self) -> RiskMetrics:
        """
        Get current risk metrics.

        Returns:
            RiskMetrics object with current status
        """
        today = date.today()
        daily_data = self._daily_pnl.get(today, DailyPnL(date=today))

        # Get position count from position manager if available
        if self.position_manager:
            current_positions_count = len(
                [
                    p
                    for p in self.position_manager.get_all_positions()
                    if p.side != "FLAT"
                ]
            )
            # Update daily P&L with position manager data
            realized_pnl, unrealized_pnl = self.position_manager.calculate_total_pnl()
            daily_data.unrealized_pnl = unrealized_pnl
        else:
            current_positions_count = 0
            realized_pnl = daily_data.realized_pnl
            unrealized_pnl = daily_data.unrealized_pnl

        return RiskMetrics(
            account_balance=self._account_balance,
            available_margin=self._get_available_margin(),
            used_margin=Decimal("100") - self._get_available_margin(),
            daily_pnl=realized_pnl + unrealized_pnl,
            max_position_size=Decimal(str(self.max_size_pct)),
            current_positions=current_positions_count,
            max_daily_loss_reached=self._is_daily_loss_limit_reached(),
        )

    def get_daily_summary(self, target_date: date = None) -> dict[str, Any]:
        """
        Get daily trading summary.

        Args:
            target_date: Date to get summary for (default: today)

        Returns:
            Dictionary with daily summary
        """
        if target_date is None:
            target_date = date.today()

        daily_data = self._daily_pnl.get(target_date, DailyPnL(date=target_date))

        return {
            "date": target_date,
            "realized_pnl": float(daily_data.realized_pnl),
            "unrealized_pnl": float(daily_data.unrealized_pnl),
            "total_pnl": float(daily_data.realized_pnl + daily_data.unrealized_pnl),
            "trades_count": daily_data.trades_count,
            "max_drawdown": float(daily_data.max_drawdown),
            "daily_loss_limit_reached": self._is_daily_loss_limit_reached(),
            "account_balance": float(self._account_balance),
        }
