"""Risk Manager - Advanced risk management system for position sizing and loss control."""

import logging
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot.config import settings
from bot.fee_calculator import fee_calculator
from bot.trading_types import Position, RiskMetrics, TradeAction
from bot.types.exceptions import BalanceValidationError
from bot.validation.balance_validator import BalanceValidator

from .api_protection import APIFailureProtection
from .circuit_breaker import TradingCircuitBreaker
from .emergency_stop import EmergencyStopManager
from .types import DailyPnL

if TYPE_CHECKING:
    from bot.position_manager import PositionManager

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Advanced risk management system for position sizing and loss control.

    Enforces position size limits, leverage constraints, daily loss limits,
    circuit breaker protection, emergency stops, and advanced position validation
    to protect the trading account from all failure modes.
    """

    def __init__(self, position_manager: "PositionManager | None" = None) -> None:
        """Initialize the risk manager.

        Args:
            position_manager: Position manager instance for position data
        """
        self.max_size_pct = settings.trading.max_size_pct
        self.leverage = settings.trading.leverage
        self.max_daily_loss_pct = settings.risk.max_daily_loss_pct
        self.max_concurrent_trades = settings.risk.max_concurrent_trades
        self.max_position_size = Decimal(100000)  # Maximum absolute position size
        self.stop_loss_percentage: float = 2.0  # Default stop loss percentage
        self.max_daily_loss: Decimal = Decimal(500)  # Maximum daily loss in USD

        # Position manager integration
        self.position_manager = position_manager

        # Risk tracking
        self._daily_pnl: dict[date, DailyPnL] = {}
        self._account_balance = Decimal(10000)  # Default starting balance

        # Balance validation system
        self.balance_validator = BalanceValidator()

        # Advanced risk management components
        self.circuit_breaker = TradingCircuitBreaker(
            failure_threshold=5, timeout_seconds=300
        )
        self.api_protection = APIFailureProtection(max_retries=3, base_delay=1.0)
        self.emergency_stop = EmergencyStopManager()

        # Risk monitoring
        self._position_errors_count = 0
        self._consecutive_losses = 0
        self._risk_metrics_history: list[dict[str, Any]] = []

        logger.info(
            "Initialized Advanced RiskManager:\n  • Max size: %s%%\n  • Leverage: %sx\n  • Max daily loss: %s%%\n  • Circuit breaker: %s failures\n  • API protection: %s retries\n  • Emergency stop: enabled\n  • Balance validation: enabled",
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
        Comprehensive risk evaluation with advanced safety measures.

        Args:
            trade_action: Proposed trade action
            current_position: Current position
            current_price: Current market price

        Returns:
            Tuple of (approved, modified_action, reason)
        """
        try:
            # Perform critical safety checks first
            safety_result = self._evaluate_critical_safety_checks(
                current_position, current_price
            )
            if safety_result is not None:
                return safety_result

            # Perform risk validations
            risk_result = self._evaluate_risk_validations(
                trade_action, current_position, current_price
            )
            if risk_result is not None:
                return risk_result

            # All checks passed - approve the trade
            return self._approve_trade_action(trade_action, current_price)

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

        # ✅ ENHANCED POSITION VALIDATION
        position_validation_result = self._validate_position_consistency(
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

    def _evaluate_risk_validations(
        self,
        trade_action: TradeAction,
        current_position: Position,
        _current_price: Decimal,
    ) -> tuple[bool, TradeAction, str] | None:
        """Evaluate risk validations (stop loss, limits, etc.)."""
        # 🛡️ MANDATORY: Validate stop loss for LONG/SHORT actions
        if not self._validate_mandatory_stop_loss(trade_action):
            self.circuit_breaker.record_failure(
                "missing_stop_loss", "Stop loss validation failed", "high"
            )
            return (
                False,
                self._get_hold_action("Stop loss is mandatory for all trades"),
                "Missing stop loss",
            )

        # 📉 Check daily loss limit
        if self._is_daily_loss_limit_reached():
            return (
                False,
                self._get_hold_action("Daily loss limit reached"),
                "Daily loss limit",
            )

        # 🔢 Check maximum concurrent positions
        if not self._can_open_new_position(trade_action, current_position):
            return (
                False,
                self._get_hold_action("Max concurrent positions reached"),
                "Position limit",
            )

        return None

    def _approve_trade_action(
        self, trade_action: TradeAction, current_price: Decimal
    ) -> tuple[bool, TradeAction, str]:
        """Approve trade action after all validations pass."""
        # 📏 Validate and adjust position size
        modified_action = self._validate_position_size(trade_action)

        # 💰 Adjust position size for trading fees
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

        # 🏦 BALANCE VALIDATION - Comprehensive balance checks
        balance_valid, balance_reason = self.validate_balance_for_trade(
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

        # 💹 Validate trade profitability after fees (skip for HOLD/CLOSE actions)
        if modified_action.action not in ["HOLD", "CLOSE"]:
            position_value = self._account_balance * Decimal(
                str(modified_action.size_pct / 100)
            )
            is_profitable, profit_reason = fee_calculator.validate_trade_profitability(
                modified_action, position_value, current_price
            )

            if not is_profitable:
                return (
                    False,
                    self._get_hold_action(f"Trade not profitable: {profit_reason}"),
                    "Fee profitability",
                )

        # 📊 Calculate comprehensive risk metrics
        risk_metrics = self._calculate_position_risk(
            modified_action, current_price, trade_fees
        )

        # 🎯 Check if risk is acceptable
        if risk_metrics["max_loss_usd"] > self._get_max_acceptable_loss():
            modified_action = self._reduce_position_size(modified_action, risk_metrics)
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
        if trade_fees.total_fee > 0:
            logger.info(
                "✅ Risk approved - Position: %s%% -> %s%%\n   • Fees: $%.2f\n   • Max Loss: $%.2f\n   • R/R Ratio: %.2f\n   • Circuit Breaker: %s",
                trade_action.size_pct,
                modified_action.size_pct,
                trade_fees.total_fee,
                risk_metrics["max_loss_usd"],
                risk_metrics["risk_reward_ratio"],
                self.circuit_breaker.state,
            )

        return True, modified_action, "Advanced risk checks passed"

    def validate_balance_for_trade(
        self,
        trade_action: TradeAction,
        _current_price: Decimal,
        estimated_fees: Decimal = Decimal(0),
    ) -> tuple[bool, str]:
        """
        Validate account balance can support the proposed trade.

        Args:
            trade_action: Proposed trade action
            current_price: Current market price
            estimated_fees: Estimated trading fees

        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Early return for actions that don't need validation
            if trade_action.action in ["HOLD", "CLOSE"]:
                return True, "No balance validation required for HOLD/CLOSE"

            # Perform balance validation checks
            return self._perform_balance_validation_checks(trade_action, estimated_fees)

        except Exception:
            logger.exception("Error in balance validation for trade")
            return False, "Balance validation error occurred"

    def _perform_balance_validation_checks(
        self, trade_action: TradeAction, estimated_fees: Decimal
    ) -> tuple[bool, str]:
        """Perform the balance validation checks."""
        # Calculate required balance for the trade
        position_value = self._account_balance * (
            Decimal(str(trade_action.size_pct)) / 100
        )
        leveraged_value = position_value * Decimal(str(self.leverage))
        required_margin = leveraged_value / Decimal(str(self.leverage))

        # Validate current balance
        validation_result = self.balance_validator.validate_balance_range(
            self._account_balance, f"trade_preparation_{trade_action.action}"
        )

        if not validation_result["valid"]:
            return (
                False,
                f"Current balance validation failed: {validation_result.get('message', 'Unknown error')}",
            )

        # Check post-trade balance
        post_trade_balance = self._account_balance - estimated_fees
        if post_trade_balance < Decimal(0):
            return (
                False,
                f"Trade would result in negative balance: ${post_trade_balance}",
            )

        # Validate post-trade balance
        post_trade_result = self._validate_post_trade_balance(
            trade_action, post_trade_balance
        )
        if post_trade_result is not None:
            return post_trade_result

        # Validate margin requirements
        margin_result = self._validate_margin_requirements(
            required_margin, leveraged_value
        )
        if margin_result is not None:
            return margin_result

        logger.debug("✅ Balance validation passed for %s trade", trade_action.action)
        return True, "Balance validation successful"

    def _validate_post_trade_balance(
        self, trade_action: TradeAction, post_trade_balance: Decimal
    ) -> tuple[bool, str] | None:
        """Validate post-trade balance."""
        try:
            post_trade_validation = self.balance_validator.validate_balance_range(
                post_trade_balance, f"post_trade_{trade_action.action}"
            )

            if not post_trade_validation["valid"]:
                return (
                    False,
                    f"Post-trade balance would be invalid: {post_trade_validation.get('message', 'Unknown error')}",
                )
        except BalanceValidationError as e:
            return False, f"Post-trade balance validation failed: {e}"

        return None

    def _validate_margin_requirements(
        self, required_margin: Decimal, leveraged_value: Decimal
    ) -> tuple[bool, str] | None:
        """Validate margin requirements."""
        try:
            current_margin_used = self._calculate_current_margin_usage()
            new_margin_used = current_margin_used + required_margin

            margin_validation = self.balance_validator.validate_margin_calculation(
                balance=self._account_balance,
                used_margin=new_margin_used,
                position_value=leveraged_value,
                leverage=self.leverage,
            )

            if not margin_validation["valid"]:
                return (
                    False,
                    f"Margin validation failed: {margin_validation.get('message', 'Unknown error')}",
                )
        except BalanceValidationError as e:
            return False, f"Margin validation error: {e}"

        return None

    def _calculate_current_margin_usage(self) -> Decimal:
        """Calculate current margin usage across all positions."""
        if not self.position_manager:
            return Decimal(0)

        total_margin = Decimal(0)
        positions = self.position_manager.get_all_positions()

        for position in positions:
            if position.side != "FLAT" and position.entry_price:
                position_value = position.size * position.entry_price
                margin_required = position_value / Decimal(str(self.leverage))
                total_margin += margin_required

        return total_margin

    def _is_daily_loss_limit_reached(self) -> bool:
        """
        Check if daily loss limit has been reached.

        Returns:
            True if daily loss limit exceeded
        """
        today = datetime.now(UTC).date()

        if today not in self._daily_pnl:
            return False

        daily_data = self._daily_pnl[today]
        total_pnl = daily_data.realized_pnl + daily_data.unrealized_pnl

        max_loss_usd = self._account_balance * (
            Decimal(str(self.max_daily_loss_pct)) / Decimal(100)
        )

        if total_pnl <= -max_loss_usd:
            logger.warning(
                "Daily loss limit reached: %s <= -%s", total_pnl, max_loss_usd
            )
            return True

        return False

    def _can_open_new_position(
        self, trade_action: TradeAction, current_position: Position
    ) -> bool:
        """
        Check if a new position can be opened.
        ENFORCES SINGLE POSITION RULE: Only one position allowed at a time.

        Args:
            trade_action: Proposed action
            current_position: Current position

        Returns:
            True if new position can be opened
        """
        # If closing or holding, always allow
        if trade_action.action in ["CLOSE", "HOLD"]:
            return True

        # SINGLE POSITION ENFORCEMENT
        # Check if we already have an active position
        if current_position.side != "FLAT":
            # We have a position - cannot open a new one
            logger.warning(
                "Cannot open new %s position - existing %s position for %s",
                trade_action.action,
                current_position.side,
                current_position.symbol,
            )
            return False

        # Get current position count from position manager if available
        if self.position_manager:
            active_positions = len(
                [
                    p
                    for p in self.position_manager.get_all_positions()
                    if p.side != "FLAT"
                ]
            )

            # Double-check: ensure no other positions exist
            if active_positions > 0:
                logger.warning(
                    "Cannot open new position - %s position(s) already exist",
                    active_positions,
                )
                return False
        else:
            # Fallback to internal tracking
            active_positions = len([p for p in [current_position] if p.side != "FLAT"])
            if active_positions > 0:
                return False

        # No positions exist - can open one
        return True

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
            logger.warning("Position size capped at %s%", self.max_size_pct)

        # Account for available margin
        available_margin = self._get_available_margin()
        max_allowed_size = self._calculate_max_size_for_margin(available_margin)

        if modified.size_pct > max_allowed_size:
            modified.size_pct = int(max_allowed_size)
            logger.warning(
                "Position size reduced to %s%% due to margin constraints",
                max_allowed_size,
            )

        return modified

    def _calculate_position_risk(
        self, trade_action: TradeAction, _current_price: Decimal, trade_fees: Any = None
    ) -> dict[str, Any]:
        """
        Calculate risk metrics for a position.

        Args:
            trade_action: Trade action to evaluate
            current_price: Current market price
            trade_fees: Optional TradeFees object with fee information

        Returns:
            Dictionary with risk metrics
        """
        if trade_action.size_pct == 0:
            return {"max_loss_usd": Decimal(0), "max_gain_usd": Decimal(0)}

        position_value = self._account_balance * (
            Decimal(str(trade_action.size_pct)) / Decimal(100)
        )
        leveraged_exposure = position_value * Decimal(str(self.leverage))

        # Calculate potential loss (stop loss) + fees
        max_loss_pct = Decimal(str(trade_action.stop_loss_pct)) / Decimal(100)
        max_loss_usd = leveraged_exposure * max_loss_pct

        # Add trading fees to the max loss
        if trade_fees and trade_fees.total_fee > 0:
            max_loss_usd += trade_fees.total_fee

        # Calculate potential gain (take profit) - fees
        max_gain_pct = Decimal(str(trade_action.take_profit_pct)) / Decimal(100)
        max_gain_usd = leveraged_exposure * max_gain_pct

        # Subtract trading fees from the max gain
        if trade_fees and trade_fees.total_fee > 0:
            max_gain_usd -= trade_fees.total_fee
            max_gain_usd = max(Decimal(0), max_gain_usd)  # Ensure non-negative

        # Recalculate risk/reward ratio with fees
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
        }

    def _get_max_acceptable_loss(self) -> Decimal:
        """
        Get maximum acceptable loss per trade.

        Returns:
            Maximum loss in USD
        """
        # Risk maximum 8% of account per trade (increased from 5% for more trading opportunities)
        max_loss_pct = Decimal("0.08")
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
                "Position size reduced from %s% to %s%",
                trade_action.size_pct,
                new_size_pct,
            )

        return modified

    def _validate_position_consistency(
        self, position: Position, market_price: Decimal
    ) -> dict[str, Any]:
        """
        Enhanced position consistency validation with comprehensive checks.

        Args:
            position: Position to validate
            market_price: Current market price

        Returns:
            Dictionary with validation result and details
        """
        try:
            # Check position size limits
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

            # Check entry price consistency for non-flat positions
            if position.side != "FLAT":
                if position.entry_price is None or position.entry_price <= 0:
                    return {
                        "valid": False,
                        "reason": f"Invalid entry price for {position.side} position: {position.entry_price}",
                        "severity": "high",
                    }

                # Check if entry price is reasonable (not too far from current market)
                price_deviation = (
                    abs(market_price - position.entry_price) / market_price
                )
                if price_deviation > Decimal("0.5"):  # 50% deviation threshold
                    return {
                        "valid": False,
                        "reason": f"Entry price {position.entry_price} deviates {price_deviation * 100:.1f}% from market {market_price}",
                        "severity": "medium",
                    }

                # Check unrealized P&L sanity
                if position.entry_price and position.side != "FLAT":
                    expected_pnl = self._calculate_expected_pnl(position, market_price)
                    pnl_diff = abs(position.unrealized_pnl - expected_pnl)
                    tolerance = abs(expected_pnl * Decimal("0.1"))  # 10% tolerance

                    if pnl_diff > tolerance > 0:
                        return {
                            "valid": False,
                            "reason": f"P&L inconsistency: expected {expected_pnl}, got {position.unrealized_pnl}",
                            "severity": "medium",
                        }

            # Check position timestamp is not too old
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
                "reason": "Position validation passed",
                "severity": "none",
            }
        except Exception:
            logger.exception("Position validation error")
            return {
                "valid": False,
                "reason": "Validation error occurred",
                "severity": "high",
            }

    def _calculate_expected_pnl(
        self, position: Position, current_price: Decimal
    ) -> Decimal:
        """Calculate expected unrealized P&L for position validation."""
        if position.side == "FLAT" or position.entry_price is None:
            return Decimal(0)

        price_diff = current_price - position.entry_price

        if position.side == "LONG":
            return position.size * price_diff
        # SHORT
        return position.size * (-price_diff)

    def _monitor_risk_metrics(self) -> dict[str, Any]:
        """
        Monitor comprehensive risk metrics for emergency conditions.

        Returns:
            Dictionary with current risk metrics
        """
        try:
            metrics: dict[str, int | float | datetime] = {
                "consecutive_losses": self._count_consecutive_losses(),
                "rapid_loss": self._calculate_rapid_loss_percentage(),
                "api_failures": self.api_protection.consecutive_failures,
                "position_errors": self._position_errors_count,
                "margin_critical": self._calculate_margin_usage(),
                "circuit_breaker_failures": self.circuit_breaker.failure_count,
                "daily_loss_pct": self._calculate_daily_loss_percentage(),
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

    def _calculate_margin_usage(self) -> float:
        """Calculate current margin usage percentage."""
        available_margin = float(self._get_available_margin())
        if available_margin >= 100:
            return 0.0
        return (100 - available_margin) / 100.0

    def _calculate_daily_loss_percentage(self) -> float:
        """Calculate daily loss as percentage of account."""
        today = datetime.now(UTC).date()
        if today not in self._daily_pnl:
            return 0.0

        daily_data = self._daily_pnl[today]
        total_pnl = daily_data.realized_pnl + daily_data.unrealized_pnl

        if self._account_balance <= 0:
            return 0.0

        return max(0, float(-total_pnl / self._account_balance))

    def record_trade_outcome(self, profit_loss: Decimal):
        """
        Record trade outcome for consecutive loss tracking.

        Args:
            profit_loss: Trade P&L (positive for profit, negative for loss)
        """
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
        """
        Get comprehensive risk management status.

        Returns:
            Dictionary with all risk management component statuses
        """
        current_metrics = self._monitor_risk_metrics()

        return {
            "risk_manager": {
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

    def _validate_mandatory_stop_loss(self, trade_action: TradeAction) -> bool:
        """
        Validate that stop loss is mandatory for LONG/SHORT actions.

        Args:
            trade_action: Trade action to validate

        Returns:
            True if stop loss requirements are met
        """
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
                "✅ Stop loss validation passed: %s%", trade_action.stop_loss_pct
            )
            return True

        return True

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
        self, realized_pnl: Decimal, unrealized_pnl: Decimal = Decimal(0)
    ) -> None:
        """
        Update daily P&L tracking.

        Args:
            realized_pnl: Realized profit/loss
            unrealized_pnl: Unrealized profit/loss
        """
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
        """
        Update account balance with validation.

        Args:
            new_balance: New account balance
        """
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
                    "Account balance updated and validated: %s -> %s",
                    old_balance,
                    self._account_balance,
                )
            else:
                logger.error(
                    "❌ Balance update validation failed: %s",
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
            logger.exception("❌ Balance validation error during update")
            # Still update the balance but log the issue
            self._account_balance = new_balance
            logger.warning(
                "⚠️ Balance updated despite validation error: %s -> %s",
                old_balance,
                new_balance,
            )
        except Exception:
            logger.exception("❌ Unexpected error during balance validation")
            # Still update the balance but log the issue
            self._account_balance = new_balance
            logger.warning(
                "⚠️ Balance updated despite unexpected error: %s -> %s",
                old_balance,
                new_balance,
            )

    def get_risk_metrics(self) -> RiskMetrics:
        """
        Get current risk metrics.

        Returns:
            RiskMetrics object with current status
        """
        today = datetime.now(UTC).date()
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
            used_margin=Decimal(100) - self._get_available_margin(),
            daily_pnl=realized_pnl + unrealized_pnl,
            max_position_size=Decimal(str(self.max_size_pct)),
            current_positions=current_positions_count,
            max_daily_loss_reached=self._is_daily_loss_limit_reached(),
        )

    def get_daily_summary(self, target_date: date | None = None) -> dict[str, Any]:
        """
        Get daily trading summary.

        Args:
            target_date: Date to get summary for (default: today)

        Returns:
            Dictionary with daily summary
        """
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
            "daily_loss_limit_reached": self._is_daily_loss_limit_reached(),
            "account_balance": float(self._account_balance),
        }
