"""
Functional Risk Management System

This module migrates from imperative to functional risk management while maintaining
exact API compatibility. It leverages pure functions from bot.fp.strategies.risk_management
and immutable types from bot.fp.types.risk while preserving all safety mechanisms.

The main RiskManager class acts as a thin adapter that:
1. Maintains the exact same interface as the original imperative system
2. Uses functional implementations internally for all calculations
3. Preserves circuit breakers, emergency stops, and validation systems
4. Ensures identical risk calculations and position sizing behavior
"""

import logging
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot.config import settings
from bot.fee_calculator import fee_calculator

# Import functional risk management components
from bot.fp.strategies.risk_management import (
    calculate_atr_stop_loss,
    calculate_correlation_adjustment,
    calculate_drawdown_adjusted_size,
    calculate_fixed_fractional_size,
    calculate_kelly_criterion,
    calculate_optimal_leverage,
    calculate_percentage_stop_loss,
    calculate_portfolio_heat,
    calculate_position_size_with_stop,
    calculate_risk_reward_take_profit,
    calculate_trailing_stop,
    calculate_volatility_based_size,
    enforce_risk_limits,
)
from bot.fp.types.risk import (
    DailyLossLimit,
    MarginCall,
    MarginInfo,
    PositionLimitExceeded,
    RiskAlert,
    RiskLimits,
    RiskParameters,
    calculate_free_margin,
    calculate_margin_ratio,
    calculate_max_position_value,
    calculate_position_risk,
    calculate_position_size,
    calculate_required_margin,
    calculate_stop_loss_price,
    calculate_take_profit_price,
    check_risk_alerts,
    is_within_risk_limits,
)
from bot.trading_types import Position, RiskMetrics, TradeAction
from bot.types.exceptions import BalanceValidationError
from bot.validation.balance_validator import BalanceValidator

# Import legacy risk components that still need imperative behavior
from .api_protection import APIFailureProtection
from .circuit_breaker import TradingCircuitBreaker
from .emergency_stop import EmergencyStopManager
from .types import DailyPnL, FailureRecord

if TYPE_CHECKING:
    from bot.position_manager import PositionManager

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Functional Risk Management Adapter

    This class maintains the exact same interface as the original imperative RiskManager
    while using functional implementations internally. It acts as an adapter that:

    - Preserves all existing API methods and signatures
    - Uses pure functional calculations where possible
    - Maintains stateful components only where absolutely necessary (circuit breakers, etc.)
    - Ensures identical risk calculations and safety behavior
    """

    def __init__(self, position_manager: "PositionManager | None" = None) -> None:
        """Initialize the functional risk manager with exact same interface."""

        # Basic trading parameters
        self.max_size_pct = settings.trading.max_size_pct
        self.leverage = settings.trading.leverage
        self.max_daily_loss_pct = settings.risk.max_daily_loss_pct
        self.max_concurrent_trades = settings.risk.max_concurrent_trades
        self.max_position_size = Decimal(100000)  # Maximum absolute position size
        self.stop_loss_percentage: float = 2.0  # Default stop loss percentage
        self.max_daily_loss: Decimal = Decimal(500)  # Maximum daily loss in USD

        # Position manager integration
        self.position_manager = position_manager

        # Risk tracking state (minimal imperative state)
        self._daily_pnl: dict[date, DailyPnL] = {}
        self._account_balance = Decimal(10000)  # Default starting balance

        # Balance validation system
        self.balance_validator = BalanceValidator()

        # Advanced risk management components (still need imperative behavior)
        self.circuit_breaker = TradingCircuitBreaker(
            failure_threshold=5, timeout_seconds=300
        )
        self.api_protection = APIFailureProtection(max_retries=3, base_delay=1.0)
        self.emergency_stop = EmergencyStopManager()

        # Risk monitoring state
        self._position_errors_count = 0
        self._consecutive_losses = 0
        self._risk_metrics_history: list[dict[str, Any]] = []

        # Create functional risk parameters from settings
        self._risk_params = RiskParameters(
            max_position_size=Decimal(str(self.max_size_pct)),
            max_leverage=Decimal(str(self.leverage)),
            stop_loss_pct=Decimal(str(self.stop_loss_percentage)),
            take_profit_pct=Decimal("4.0"),  # Default 2:1 risk/reward
        )

        self._risk_limits = RiskLimits(
            daily_loss_limit=self.max_daily_loss,
            position_limit=self.max_concurrent_trades,
            margin_requirement=Decimal("5.0"),  # 5% minimum margin
        )

        logger.info(
            "Initialized Functional RiskManager:\n  • Max size: %s%%\n  • Leverage: %sx\n  • Max daily loss: %s%%\n  • Circuit breaker: %s failures\n  • API protection: %s retries\n  • Emergency stop: enabled\n  • Balance validation: enabled\n  • Functional calculations: enabled",
            self.max_size_pct,
            self.leverage,
            self.max_daily_loss_pct,
            self.circuit_breaker.failure_threshold,
            self.api_protection.max_retries,
        )

    def evaluate_risk(
        self,
        trade_action: TradeAction,
        current_position: Position,
        current_price: Decimal,
    ) -> tuple[bool, TradeAction, str]:
        """
        Comprehensive risk evaluation using functional risk management.

        Maintains exact same signature and behavior as imperative version
        but uses functional calculations internally.
        """
        try:
            # Perform critical safety checks first (still imperative for circuit breakers)
            safety_result = self._evaluate_critical_safety_checks(
                current_position, current_price
            )
            if safety_result is not None:
                return safety_result

            # Perform functional risk validations
            functional_result = self._evaluate_functional_risk_validations(
                trade_action, current_position, current_price
            )
            if functional_result is not None:
                return functional_result

            # All checks passed - approve the trade using functional calculations
            return self._approve_trade_action_functionally(trade_action, current_price)

        except Exception:
            logger.exception("Risk evaluation error")
            self.circuit_breaker.record_failure(
                "risk_evaluation_error", "exception_occurred", "high"
            )
            return False, self._get_hold_action("Risk evaluation error"), "Error"

    def _evaluate_critical_safety_checks(
        self, current_position: Position, current_price: Decimal
    ) -> tuple[bool, TradeAction, str] | None:
        """Evaluate critical safety checks (emergency stop, circuit breaker, etc.)."""

        # 🚨 EMERGENCY STOP CHECK - Highest priority
        if self.emergency_stop.is_stopped:
            self.circuit_breaker.record_failure(
                "emergency_stop",
                f"Emergency stop active: {self.emergency_stop.stop_reason}",
                "critical",
            )
            return (
                False,
                self._get_hold_action(
                    f"EMERGENCY STOP: {self.emergency_stop.stop_reason}"
                ),
                "Emergency stop active",
            )

        # 🔄 CIRCUIT BREAKER CHECK - Second priority
        if not self.circuit_breaker.can_execute_trade():
            return (
                False,
                self._get_hold_action(
                    f"Circuit breaker OPEN - trading halted for "
                    f"{self.circuit_breaker.get_status()['timeout_remaining']:.0f}s"
                ),
                "Circuit breaker",
            )

        # 📊 MONITOR RISK METRICS for emergency conditions
        current_metrics = self._monitor_risk_metrics()
        if self.emergency_stop.check_emergency_conditions(current_metrics):
            return (
                False,
                self._get_hold_action("Emergency conditions detected"),
                "Emergency triggered",
            )

        # ✅ FUNCTIONAL POSITION VALIDATION
        position_validation_result = self._validate_position_consistency_functionally(
            current_position, current_price
        )
        if not position_validation_result["valid"]:
            self._position_errors_count += 1
            self.circuit_breaker.record_failure(
                "position_validation",
                position_validation_result["reason"],
                "medium",
            )
            return (
                False,
                self._get_hold_action(
                    f"Position validation failed: {position_validation_result['reason']}"
                ),
                "Position validation",
            )

        return None

    def _evaluate_functional_risk_validations(
        self,
        trade_action: TradeAction,
        current_position: Position,
        current_price: Decimal,
    ) -> tuple[bool, TradeAction, str] | None:
        """Evaluate risk validations using functional calculations."""

        # 🛡️ FUNCTIONAL: Validate stop loss using pure functions
        if not self._validate_mandatory_stop_loss_functionally(trade_action):
            self.circuit_breaker.record_failure(
                "missing_stop_loss", "Stop loss validation failed", "high"
            )
            return (
                False,
                self._get_hold_action("Stop loss is mandatory for all trades"),
                "Missing stop loss",
            )

        # 📉 FUNCTIONAL: Check daily loss limit using pure calculations
        if self._is_daily_loss_limit_reached_functionally():
            return (
                False,
                self._get_hold_action("Daily loss limit reached"),
                "Daily loss limit",
            )

        # 🔢 FUNCTIONAL: Check maximum concurrent positions
        if not self._can_open_new_position_functionally(trade_action, current_position):
            return (
                False,
                self._get_hold_action("Max concurrent positions reached"),
                "Position limit",
            )

        # 💰 FUNCTIONAL: Validate position size using functional risk limits
        margin_info = self._get_current_margin_info_functionally()
        current_positions = self._get_current_positions_count()
        daily_pnl = self._get_daily_pnl_functionally()

        risk_alerts = check_risk_alerts(
            margin_info, self._risk_limits, current_positions, daily_pnl
        )

        if risk_alerts:
            alert_reasons = [self._format_risk_alert(alert) for alert in risk_alerts]
            return (
                False,
                self._get_hold_action(f"Risk alerts: {', '.join(alert_reasons)}"),
                "Risk alerts triggered",
            )

        return None

    def _approve_trade_action_functionally(
        self, trade_action: TradeAction, current_price: Decimal
    ) -> tuple[bool, TradeAction, str]:
        """Approve trade action using functional risk management calculations."""

        # 📏 FUNCTIONAL: Validate and adjust position size using pure functions
        modified_action = self._validate_position_size_functionally(trade_action)

        # 💰 Adjust position size for trading fees (still uses fee_calculator)
        try:
            modified_action, trade_fees = fee_calculator.adjust_position_size_for_fees(
                modified_action, self._account_balance, current_price
            )
        except Exception as e:
            self.circuit_breaker.record_failure(
                "fee_calculation", f"Fee calculation failed: {e}", "medium"
            )
            return (
                False,
                self._get_hold_action("Fee calculation error"),
                "Fee calculation error",
            )

        # 🏦 FUNCTIONAL: Balance validation using functional calculations
        balance_valid, balance_reason = self._validate_balance_for_trade_functionally(
            modified_action,
            current_price,
            trade_fees.total_fee if trade_fees else Decimal(0),
        )

        if not balance_valid:
            self.circuit_breaker.record_failure(
                "balance_validation",
                f"Balance validation failed: {balance_reason}",
                "medium",
            )
            return (
                False,
                self._get_hold_action(f"Balance validation failed: {balance_reason}"),
                "Balance validation",
            )

        if modified_action.size_pct == 0 and trade_action.action in ["LONG", "SHORT"]:
            return (
                False,
                self._get_hold_action("Position too small after fee adjustment"),
                "Insufficient funds for fees",
            )

        # 💹 FUNCTIONAL: Validate trade profitability using functional calculations
        if modified_action.action not in ["HOLD", "CLOSE"]:
            is_profitable, profit_reason = (
                self._validate_trade_profitability_functionally(
                    modified_action, current_price, trade_fees
                )
            )

            if not is_profitable:
                return (
                    False,
                    self._get_hold_action(f"Trade not profitable: {profit_reason}"),
                    "Fee profitability",
                )

        # 📊 FUNCTIONAL: Calculate comprehensive risk metrics using pure functions
        risk_metrics = self._calculate_position_risk_functionally(
            modified_action, current_price, trade_fees
        )

        # 🎯 FUNCTIONAL: Check if risk is acceptable using functional risk enforcement
        max_acceptable_loss = self._get_max_acceptable_loss()
        if risk_metrics["max_loss_usd"] > max_acceptable_loss:
            modified_action = self._reduce_position_size_functionally(
                modified_action, risk_metrics
            )
            logger.warning("Position size reduced due to excessive risk")

        # 🔍 Final validation
        if modified_action.size_pct == 0 and trade_action.action in ["LONG", "SHORT"]:
            return (
                False,
                self._get_hold_action("Risk too high for any position"),
                "Risk too high",
            )

        # ✅ Success - record for circuit breaker
        self.circuit_breaker.record_success()

        # 📝 Log comprehensive information
        if trade_fees and trade_fees.total_fee > 0:
            logger.info(
                "✅ Functional risk approved - Position: %s%% -> %s%%\n   • Fees: $%.2f\n   • Max Loss: $%.2f\n   • R/R Ratio: %.2f\n   • Circuit Breaker: %s",
                trade_action.size_pct,
                modified_action.size_pct,
                trade_fees.total_fee,
                risk_metrics["max_loss_usd"],
                risk_metrics["risk_reward_ratio"],
                self.circuit_breaker.state,
            )

        return True, modified_action, "Functional risk checks passed"

    def _validate_mandatory_stop_loss_functionally(
        self, trade_action: TradeAction
    ) -> bool:
        """Validate mandatory stop loss using functional approach."""
        # Allow HOLD and CLOSE actions without stop loss validation
        if trade_action.action in ["HOLD", "CLOSE"]:
            return True

        # For LONG/SHORT actions, stop loss MUST be > 0
        if trade_action.action in ["LONG", "SHORT"]:
            if trade_action.stop_loss_pct <= 0:
                logger.error(
                    "❌ MANDATORY STOP LOSS MISSING: %s action requires stop_loss_pct > 0, got %s",
                    trade_action.action,
                    trade_action.stop_loss_pct,
                )
                return False

            # Ensure stop loss is reasonable (between 0.1% and 10%)
            if trade_action.stop_loss_pct < 0.1 or trade_action.stop_loss_pct > 10.0:
                logger.error(
                    "❌ INVALID STOP LOSS: %s%% is outside acceptable range (0.1%% - 10.0%%)",
                    trade_action.stop_loss_pct,
                )
                return False

            logger.info(
                "✅ Functional stop loss validation passed: %s%",
                trade_action.stop_loss_pct,
            )
            return True

        return True

    def _is_daily_loss_limit_reached_functionally(self) -> bool:
        """Check daily loss limit using functional calculations."""
        today = datetime.now(UTC).date()

        if today not in self._daily_pnl:
            return False

        daily_data = self._daily_pnl[today]
        total_pnl = daily_data.realized_pnl + daily_data.unrealized_pnl

        # Use functional calculation for max loss threshold
        max_loss_usd = self._account_balance * (
            Decimal(str(self.max_daily_loss_pct)) / Decimal(100)
        )

        if total_pnl <= -max_loss_usd:
            logger.warning(
                "Functional daily loss limit reached: %s <= -%s",
                total_pnl,
                max_loss_usd,
            )
            return True

        return False

    def _can_open_new_position_functionally(
        self, trade_action: TradeAction, current_position: Position
    ) -> bool:
        """Check if new position can be opened using functional approach."""
        # If closing or holding, always allow
        if trade_action.action in ["CLOSE", "HOLD"]:
            return True

        # SINGLE POSITION ENFORCEMENT using functional logic
        # Check if we already have an active position
        if current_position.side != "FLAT":
            logger.warning(
                "Cannot open new %s position - existing %s position for %s",
                trade_action.action,
                current_position.side,
                current_position.symbol,
            )
            return False

        # Get current position count functionally
        active_positions_count = self._get_current_positions_count()

        if active_positions_count > 0:
            logger.warning(
                "Cannot open new position - %s position(s) already exist",
                active_positions_count,
            )
            return False

        # No positions exist - can open one
        return True

    def _validate_position_size_functionally(
        self, trade_action: TradeAction
    ) -> TradeAction:
        """Validate and adjust position size using functional risk management."""
        modified = trade_action.copy()

        # Functional position size enforcement
        current_positions = self._get_current_positions_for_enforcement()

        # Use functional risk limit enforcement
        adjusted_size_usd = self._account_balance * (
            Decimal(str(modified.size_pct)) / 100
        )

        final_size_usd, reason = enforce_risk_limits(
            proposed_size=float(adjusted_size_usd),
            current_positions=current_positions,
            account_balance=float(self._account_balance),
            max_position_size_pct=self.max_size_pct,
            max_portfolio_heat_pct=6.0,  # 6% max portfolio heat
        )

        # Convert back to percentage
        final_size_pct = int(
            (Decimal(str(final_size_usd)) / self._account_balance) * 100
        )

        if final_size_pct != modified.size_pct:
            modified.size_pct = final_size_pct
            logger.warning("Functional position size adjustment: %s", reason)

        return modified

    def _validate_balance_for_trade_functionally(
        self,
        trade_action: TradeAction,
        current_price: Decimal,
        estimated_fees: Decimal = Decimal(0),
    ) -> tuple[bool, str]:
        """Validate balance using functional risk calculations."""
        try:
            # Early return for actions that don't need validation
            if trade_action.action in ["HOLD", "CLOSE"]:
                return True, "No balance validation required for HOLD/CLOSE"

            # Create functional margin info
            margin_info = self._get_current_margin_info_functionally()

            # Calculate required margin functionally
            position_size = self._account_balance * (
                Decimal(str(trade_action.size_pct)) / 100
            )

            required_margin = calculate_required_margin(
                position_size=position_size,
                entry_price=current_price,
                leverage=Decimal(str(self.leverage)),
            )

            # Check if we have sufficient free margin
            if margin_info.free_margin < required_margin + estimated_fees:
                return (
                    False,
                    f"Insufficient margin: required {required_margin + estimated_fees}, available {margin_info.free_margin}",
                )

            logger.debug(
                "✅ Functional balance validation passed for %s trade",
                trade_action.action,
            )
            return True, "Functional balance validation successful"

        except Exception:
            logger.exception("Error in functional balance validation for trade")
            return False, "Functional balance validation error occurred"

    def _validate_trade_profitability_functionally(
        self, trade_action: TradeAction, current_price: Decimal, trade_fees: Any
    ) -> tuple[bool, str]:
        """Validate trade profitability using functional calculations."""
        position_value = self._account_balance * Decimal(
            str(trade_action.size_pct / 100)
        )

        # Use fee calculator for profitability check (maintains existing behavior)
        is_profitable, profit_reason = fee_calculator.validate_trade_profitability(
            trade_action, position_value, current_price
        )

        return is_profitable, profit_reason

    def _calculate_position_risk_functionally(
        self, trade_action: TradeAction, current_price: Decimal, trade_fees: Any = None
    ) -> dict[str, Any]:
        """Calculate position risk using functional risk management."""
        if trade_action.size_pct == 0:
            return {"max_loss_usd": Decimal(0), "max_gain_usd": Decimal(0)}

        position_value = self._account_balance * (
            Decimal(str(trade_action.size_pct)) / Decimal(100)
        )
        leveraged_exposure = position_value * Decimal(str(self.leverage))

        # Use functional stop loss calculation
        is_long = trade_action.action == "LONG"
        stop_loss_price = calculate_stop_loss_price(
            entry_price=current_price,
            stop_loss_pct=Decimal(str(trade_action.stop_loss_pct)),
            is_long=is_long,
        )

        take_profit_price = calculate_take_profit_price(
            entry_price=current_price,
            take_profit_pct=Decimal(str(trade_action.take_profit_pct)),
            is_long=is_long,
        )

        # Calculate functional position risk
        position_size = leveraged_exposure / current_price
        max_loss_usd = calculate_position_risk(
            position_size=position_size,
            entry_price=current_price,
            stop_loss_price=stop_loss_price,
        )

        # Calculate max gain
        price_diff_gain = abs(take_profit_price - current_price)
        max_gain_usd = position_size * price_diff_gain

        # Add/subtract trading fees
        if trade_fees and trade_fees.total_fee > 0:
            max_loss_usd += trade_fees.total_fee
            max_gain_usd = max(Decimal(0), max_gain_usd - trade_fees.total_fee)

        # Calculate risk/reward ratio
        risk_reward_ratio = (
            float(max_gain_usd / max_loss_usd)
            if max_loss_usd > 0
            else trade_action.take_profit_pct / trade_action.stop_loss_pct
        )

        return {
            "position_value": position_value,
            "leveraged_exposure": leveraged_exposure,
            "max_loss_usd": max_loss_usd,
            "max_gain_usd": max_gain_usd,
            "risk_reward_ratio": risk_reward_ratio,
            "fees_included": trade_fees is not None,
            "total_fees": trade_fees.total_fee if trade_fees else Decimal(0),
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
        }

    def _reduce_position_size_functionally(
        self, trade_action: TradeAction, risk_metrics: dict[str, Any]
    ) -> TradeAction:
        """Reduce position size using functional risk management."""
        modified = trade_action.copy()
        max_acceptable_loss = self._get_max_acceptable_loss()

        if risk_metrics["max_loss_usd"] > 0:
            # Calculate size reduction factor
            reduction_factor = max_acceptable_loss / risk_metrics["max_loss_usd"]
            new_size_pct = int(Decimal(str(modified.size_pct)) * reduction_factor)

            # Ensure minimum viable size or zero
            if new_size_pct < 1:
                new_size_pct = 0

            modified.size_pct = new_size_pct
            logger.info(
                "Functional position size reduced from %s% to %s%",
                trade_action.size_pct,
                new_size_pct,
            )

        return modified

    def _validate_position_consistency_functionally(
        self, position: Position, market_price: Decimal
    ) -> dict[str, Any]:
        """Enhanced position consistency validation using functional approach."""
        try:
            # Check position size limits using functional validation
            if position.size > self.max_position_size:
                return {
                    "valid": False,
                    "reason": f"Position size {position.size} exceeds maximum {self.max_position_size}",
                    "severity": "high",
                }

            # Check for valid position side
            if position.side not in ["LONG", "SHORT", "FLAT"]:
                return {
                    "valid": False,
                    "reason": f"Invalid position side: {position.side}",
                    "severity": "critical",
                }

            # Functional entry price consistency check
            if position.side != "FLAT":
                if position.entry_price is None or position.entry_price <= 0:
                    return {
                        "valid": False,
                        "reason": f"Invalid entry price for {position.side} position: {position.entry_price}",
                        "severity": "high",
                    }

                # Check if entry price is reasonable using functional calculation
                price_deviation = (
                    abs(market_price - position.entry_price) / market_price
                )
                if price_deviation > Decimal("0.5"):  # 50% deviation threshold
                    return {
                        "valid": False,
                        "reason": f"Entry price {position.entry_price} deviates {price_deviation * 100:.1f}% from market {market_price}",
                        "severity": "medium",
                    }

                # Functional P&L consistency check
                if position.entry_price and position.side != "FLAT":
                    expected_pnl = self._calculate_expected_pnl_functionally(
                        position, market_price
                    )
                    pnl_diff = abs(position.unrealized_pnl - expected_pnl)
                    tolerance = abs(expected_pnl * Decimal("0.1"))  # 10% tolerance

                    if pnl_diff > tolerance > 0:
                        return {
                            "valid": False,
                            "reason": f"P&L inconsistency: expected {expected_pnl}, got {position.unrealized_pnl}",
                            "severity": "medium",
                        }

            # Check position age
            if position.timestamp:
                age_hours = (
                    datetime.now(UTC) - position.timestamp.replace(tzinfo=None)
                ).total_seconds() / 3600
                if age_hours > 168:  # 1 week
                    return {
                        "valid": False,
                        "reason": f"Position timestamp too old: {age_hours:.1f} hours",
                        "severity": "low",
                    }

            return {
                "valid": True,
                "reason": "Functional position validation passed",
                "severity": "none",
            }
        except Exception:
            logger.exception("Functional position validation error")
            return {
                "valid": False,
                "reason": "Functional validation error occurred",
                "severity": "high",
            }

    def _calculate_expected_pnl_functionally(
        self, position: Position, current_price: Decimal
    ) -> Decimal:
        """Calculate expected unrealized P&L using functional approach."""
        if position.side == "FLAT" or position.entry_price is None:
            return Decimal(0)

        price_diff = current_price - position.entry_price

        if position.side == "LONG":
            return position.size * price_diff
        # SHORT
        return position.size * (-price_diff)

    def _get_current_margin_info_functionally(self) -> MarginInfo:
        """Get current margin information using functional calculations."""
        used_margin = self._calculate_current_margin_usage()
        free_margin = calculate_free_margin(self._account_balance, used_margin)
        margin_ratio = calculate_margin_ratio(used_margin, self._account_balance)

        return MarginInfo(
            total_balance=self._account_balance,
            used_margin=used_margin,
            free_margin=free_margin,
            margin_ratio=margin_ratio,
        )

    def _get_current_positions_count(self) -> int:
        """Get current positions count functionally."""
        if self.position_manager:
            return len(
                [
                    p
                    for p in self.position_manager.get_all_positions()
                    if p.side != "FLAT"
                ]
            )
        return 0

    def _get_current_positions_for_enforcement(self) -> list[dict[str, float]]:
        """Get current positions in format needed for functional risk enforcement."""
        positions = []

        if self.position_manager:
            for pos in self.position_manager.get_all_positions():
                if pos.side != "FLAT" and pos.entry_price:
                    positions.append(
                        {
                            "size": float(pos.size * pos.entry_price),
                            "entry_price": float(pos.entry_price),
                            "stop_loss": 0.0,  # Will be calculated from stop_loss_pct
                            "is_long": pos.side == "LONG",
                        }
                    )

        return positions

    def _get_daily_pnl_functionally(self) -> Decimal:
        """Get daily P&L using functional calculation."""
        today = datetime.now(UTC).date()

        if today not in self._daily_pnl:
            return Decimal(0)

        daily_data = self._daily_pnl[today]
        return daily_data.realized_pnl + daily_data.unrealized_pnl

    def _format_risk_alert(self, alert: RiskAlert) -> str:
        """Format risk alert for display."""
        if isinstance(alert, PositionLimitExceeded):
            return f"Position limit exceeded: {alert.current_positions}/{alert.limit}"
        if isinstance(alert, MarginCall):
            return f"Margin call: {alert.margin_ratio:.2%} > {alert.threshold:.2%}"
        if isinstance(alert, DailyLossLimit):
            return f"Daily loss limit: ${alert.current_loss} >= ${alert.limit}"
        return str(alert)

    # Legacy methods that maintain exact same interface and behavior

    def validate_balance_for_trade(
        self,
        trade_action: TradeAction,
        current_price: Decimal,
        estimated_fees: Decimal = Decimal(0),
    ) -> tuple[bool, str]:
        """
        Validate account balance can support the proposed trade.

        Maintains exact same interface while using functional calculations internally.
        """
        return self._validate_balance_for_trade_functionally(
            trade_action, current_price, estimated_fees
        )

    def _calculate_current_margin_usage(self) -> Decimal:
        """Calculate current margin usage across all positions."""
        if not self.position_manager:
            return Decimal(0)

        total_margin = Decimal(0)
        positions = self.position_manager.get_all_positions()

        for position in positions:
            if position.side != "FLAT" and position.entry_price:
                # Use functional margin calculation
                margin_required = calculate_required_margin(
                    position_size=position.size,
                    entry_price=position.entry_price,
                    leverage=Decimal(str(self.leverage)),
                )
                total_margin += margin_required

        return total_margin

    def _monitor_risk_metrics(self) -> dict[str, Any]:
        """Monitor comprehensive risk metrics using functional approach where possible."""
        try:
            metrics: dict[str, int | float | datetime] = {
                "consecutive_losses": self._count_consecutive_losses(),
                "rapid_loss": self._calculate_rapid_loss_percentage(),
                "api_failures": self.api_protection.consecutive_failures,
                "position_errors": self._position_errors_count,
                "margin_critical": self._calculate_margin_usage_functionally(),
                "circuit_breaker_failures": self.circuit_breaker.failure_count,
                "daily_loss_pct": self._calculate_daily_loss_percentage_functionally(),
            }

            # Log risk metrics if any are concerning
            concerning_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, int | float) and (
                    (k == "rapid_loss" and v >= 0.03)  # 3% rapid loss
                    or (k == "api_failures" and v >= 5)
                    or (k == "position_errors" and v >= 2)
                    or (k == "margin_critical" and v >= 0.8)  # 80% margin usage
                    or (k == "consecutive_losses" and v >= 3)
                ):
                    concerning_metrics[k] = v

            if concerning_metrics:
                logger.warning(
                    "⚠️  Concerning risk metrics detected: %s", concerning_metrics
                )

            # Store metrics history
            metrics["timestamp"] = datetime.now(UTC)
            self._risk_metrics_history.append(metrics)

            # Keep only recent history (last 24 hours of samples)
            cutoff_time = datetime.now(UTC) - timedelta(hours=24)
            self._risk_metrics_history = [
                m
                for m in self._risk_metrics_history
                if isinstance(m["timestamp"], datetime)
                and m["timestamp"] >= cutoff_time
            ]

            return metrics

        except Exception:
            logger.exception("Risk metrics monitoring error")
            return {}

    def _calculate_margin_usage_functionally(self) -> float:
        """Calculate current margin usage percentage using functional approach."""
        margin_info = self._get_current_margin_info_functionally()
        return float(margin_info.margin_ratio)

    def _calculate_daily_loss_percentage_functionally(self) -> float:
        """Calculate daily loss as percentage of account using functional approach."""
        daily_pnl = self._get_daily_pnl_functionally()

        if self._account_balance <= 0:
            return 0.0

        return max(0, float(-daily_pnl / self._account_balance))

    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing trades."""
        return self._consecutive_losses

    def _calculate_rapid_loss_percentage(self) -> float:
        """Calculate rapid loss percentage over last 5 minutes."""
        if not self._risk_metrics_history:
            return 0.0

        five_minutes_ago = datetime.now(UTC) - timedelta(minutes=5)
        recent_metrics = [
            m
            for m in self._risk_metrics_history[-10:]  # Last 10 samples
            if m["timestamp"] >= five_minutes_ago
        ]

        if len(recent_metrics) < 2:
            return 0.0

        # Calculate loss over the period
        initial_balance = recent_metrics[0].get(
            "account_balance", float(self._account_balance)
        )
        current_balance = float(self._account_balance)

        if initial_balance <= 0:
            return 0.0

        return max(0, (initial_balance - current_balance) / initial_balance)

    def _get_max_acceptable_loss(self) -> Decimal:
        """Get maximum acceptable loss per trade."""
        # Risk maximum 8% of account per trade (increased from 5% for more trading opportunities)
        max_loss_pct = Decimal("0.08")
        return self._account_balance * max_loss_pct

    def _get_hold_action(self, reason: str) -> TradeAction:
        """Get a safe HOLD action with reason."""
        return TradeAction(
            action="HOLD",
            size_pct=0,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale=f"Functional risk manager: {reason}",
        )

    def record_trade_outcome(self, profit_loss: Decimal):
        """Record trade outcome for consecutive loss tracking."""
        if profit_loss < 0:
            self._consecutive_losses += 1
            logger.info(
                "Trade loss recorded - consecutive losses: %s", self._consecutive_losses
            )
        else:
            if self._consecutive_losses > 0:
                logger.info(
                    "Consecutive loss streak broken at %s", self._consecutive_losses
                )
            self._consecutive_losses = 0

    def get_advanced_risk_status(self) -> dict[str, Any]:
        """Get comprehensive risk management status."""
        current_metrics = self._monitor_risk_metrics()

        return {
            "risk_manager": {
                "type": "functional",
                "max_size_pct": self.max_size_pct,
                "account_balance": float(self._account_balance),
                "consecutive_losses": self._consecutive_losses,
                "position_errors": self._position_errors_count,
            },
            "circuit_breaker": self.circuit_breaker.get_status(),
            "api_protection": self.api_protection.get_health_status(),
            "emergency_stop": self.emergency_stop.get_status(),
            "balance_validation": self.balance_validator.get_validation_statistics(),
            "current_metrics": current_metrics,
            "risk_assessment": self._assess_overall_risk_level(current_metrics),
            "functional_components": {
                "risk_parameters": {
                    "max_position_size": float(self._risk_params.max_position_size),
                    "max_leverage": float(self._risk_params.max_leverage),
                    "stop_loss_pct": float(self._risk_params.stop_loss_pct),
                    "take_profit_pct": float(self._risk_params.take_profit_pct),
                },
                "risk_limits": {
                    "daily_loss_limit": float(self._risk_limits.daily_loss_limit),
                    "position_limit": self._risk_limits.position_limit,
                    "margin_requirement": float(self._risk_limits.margin_requirement),
                },
            },
        }

    def _assess_overall_risk_level(self, metrics: dict[str, Any]) -> str:
        """Assess overall system risk level."""
        if self.emergency_stop.is_stopped:
            return "CRITICAL - Emergency stop active"

        if self.circuit_breaker.state == "OPEN":
            return "HIGH - Circuit breaker open"

        high_risk_conditions = [
            metrics.get("rapid_loss", 0) >= 0.03,  # 3% rapid loss
            metrics.get("consecutive_losses", 0) >= 5,
            metrics.get("margin_critical", 0) >= 0.9,  # 90% margin usage
            metrics.get("api_failures", 0) >= 8,
        ]

        medium_risk_conditions = [
            metrics.get("rapid_loss", 0) >= 0.02,  # 2% rapid loss
            metrics.get("consecutive_losses", 0) >= 3,
            metrics.get("margin_critical", 0) >= 0.7,  # 70% margin usage
            metrics.get("api_failures", 0) >= 5,
            metrics.get("position_errors", 0) >= 2,
        ]

        if any(high_risk_conditions):
            return "HIGH"
        if any(medium_risk_conditions):
            return "MEDIUM"
        return "LOW"

    def update_daily_pnl(
        self, realized_pnl: Decimal, unrealized_pnl: Decimal = Decimal(0)
    ) -> None:
        """Update daily P&L tracking."""
        today = datetime.now(UTC).date()

        if today not in self._daily_pnl:
            self._daily_pnl[today] = DailyPnL(date=today)

        daily_data = self._daily_pnl[today]
        daily_data.realized_pnl += realized_pnl
        daily_data.unrealized_pnl = unrealized_pnl

        if realized_pnl != 0:
            daily_data.trades_count += 1

        # Update max drawdown
        total_pnl = daily_data.realized_pnl + daily_data.unrealized_pnl
        daily_data.max_drawdown = min(daily_data.max_drawdown, total_pnl)

    def update_account_balance(self, new_balance: Decimal) -> None:
        """Update account balance with validation."""
        old_balance = self._account_balance

        # Validate the balance update
        try:
            validation_result = self.balance_validator.comprehensive_balance_validation(
                balance=new_balance,
                previous_balance=old_balance,
                operation_type="account_balance_update",
            )

            if validation_result["valid"]:
                self._account_balance = validation_result["balance"]
                logger.info(
                    "Functional account balance updated and validated: %s -> %s",
                    old_balance,
                    self._account_balance,
                )
            else:
                logger.error(
                    "❌ Functional balance update validation failed: %s",
                    validation_result.get("error", {}).get("message", "Unknown error"),
                )
                # Still update the balance but log the issue
                self._account_balance = new_balance
                logger.warning(
                    "⚠️ Balance updated despite validation failure: %s -> %s",
                    old_balance,
                    new_balance,
                )

        except BalanceValidationError:
            logger.exception("❌ Balance validation error during functional update")
            # Still update the balance but log the issue
            self._account_balance = new_balance
            logger.warning(
                "⚠️ Balance updated despite validation error: %s -> %s",
                old_balance,
                new_balance,
            )
        except Exception:
            logger.exception("❌ Unexpected error during functional balance validation")
            # Still update the balance but log the issue
            self._account_balance = new_balance
            logger.warning(
                "⚠️ Balance updated despite unexpected error: %s -> %s",
                old_balance,
                new_balance,
            )

    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        today = datetime.now(UTC).date()
        daily_data = self._daily_pnl.get(today, DailyPnL(date=today))

        # Get position count functionally
        current_positions_count = self._get_current_positions_count()

        # Update daily P&L with position manager data if available
        if self.position_manager:
            realized_pnl, unrealized_pnl = self.position_manager.calculate_total_pnl()
            daily_data.unrealized_pnl = unrealized_pnl
        else:
            realized_pnl = daily_data.realized_pnl
            unrealized_pnl = daily_data.unrealized_pnl

        # Calculate available margin functionally
        margin_info = self._get_current_margin_info_functionally()

        return RiskMetrics(
            account_balance=self._account_balance,
            available_margin=margin_info.free_margin,
            used_margin=margin_info.used_margin,
            daily_pnl=realized_pnl + unrealized_pnl,
            max_position_size=Decimal(str(self.max_size_pct)),
            current_positions=current_positions_count,
            max_daily_loss_reached=self._is_daily_loss_limit_reached_functionally(),
        )

    def get_daily_summary(self, target_date: date | None = None) -> dict[str, Any]:
        """Get daily trading summary."""
        if target_date is None:
            target_date = datetime.now(UTC).date()

        daily_data = self._daily_pnl.get(target_date, DailyPnL(date=target_date))

        return {
            "date": target_date,
            "realized_pnl": float(daily_data.realized_pnl),
            "unrealized_pnl": float(daily_data.unrealized_pnl),
            "total_pnl": float(daily_data.realized_pnl + daily_data.unrealized_pnl),
            "trades_count": daily_data.trades_count,
            "max_drawdown": float(daily_data.max_drawdown),
            "daily_loss_limit_reached": self._is_daily_loss_limit_reached_functionally(),
            "account_balance": float(self._account_balance),
            "risk_manager_type": "functional",
        }


# Export functional risk management functions for direct use
__all__ = [
    # Main risk manager (functional implementation)
    "RiskManager",
    # Legacy risk components (still imperative where needed)
    "APIFailureProtection",
    "DailyPnL",
    "EmergencyStopManager",
    "FailureRecord",
    "TradingCircuitBreaker",
    # Functional risk management functions
    "calculate_kelly_criterion",
    "calculate_fixed_fractional_size",
    "calculate_volatility_based_size",
    "calculate_atr_stop_loss",
    "calculate_percentage_stop_loss",
    "calculate_risk_reward_take_profit",
    "calculate_trailing_stop",
    "calculate_portfolio_heat",
    "enforce_risk_limits",
    "calculate_position_size_with_stop",
    "calculate_correlation_adjustment",
    "calculate_optimal_leverage",
    "calculate_drawdown_adjusted_size",
    # Functional risk types
    "RiskParameters",
    "RiskLimits",
    "MarginInfo",
    "RiskAlert",
    "PositionLimitExceeded",
    "MarginCall",
    "DailyLossLimit",
    # Functional risk calculations
    "calculate_position_size",
    "calculate_margin_ratio",
    "calculate_free_margin",
    "calculate_max_position_value",
    "calculate_required_margin",
    "calculate_stop_loss_price",
    "calculate_take_profit_price",
    "check_risk_alerts",
    "calculate_position_risk",
    "is_within_risk_limits",
]
