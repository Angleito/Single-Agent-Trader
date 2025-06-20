"""
Risk management and position sizing module.

This module handles risk management, position sizing, leverage control,
daily loss limits, circuit breakers, and advanced position validation
to protect the trading account.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from .config import settings
from .fee_calculator import fee_calculator
from .position_manager import PositionManager
from .trading_types import Position, RiskMetrics, TradeAction
from .validation import BalanceValidationError, BalanceValidator

logger = logging.getLogger(__name__)


@dataclass
class DailyPnL:
    """Daily P&L tracking."""

    date: date
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    trades_count: int = 0
    max_drawdown: Decimal = Decimal("0")


@dataclass
class FailureRecord:
    """Record of trading failures for circuit breaker."""

    timestamp: datetime
    failure_type: str
    error_message: str
    severity: str = "medium"  # low, medium, high, critical


class TradingCircuitBreaker:
    """
    Circuit breaker pattern for trading operations.

    Automatically halts trading when consecutive failures exceed threshold,
    preventing cascade failures and protecting the account.
    """

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 300):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_history: list[FailureRecord] = []

        logger.info(
            "Circuit breaker initialized: threshold=%s, timeout=%ss",
            failure_threshold,
            timeout_seconds,
        )

    def can_execute_trade(self) -> bool:
        """
        Check if trading is allowed based on circuit breaker state.

        Returns:
            True if trading is allowed, False if circuit is open
        """
        if self.state == "OPEN":
            if self.last_failure_time and datetime.now(
                UTC
            ) - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
                self.failure_count = 0
                logger.info("🔄 Circuit breaker moving to HALF_OPEN state")
                return True
            return False
        return True

    def record_success(self):
        """Record successful trading operation."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            logger.info("✅ Circuit breaker reset to CLOSED state")
        elif self.state == "CLOSED" and self.failure_count > 0:
            # Gradually reduce failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(
        self, failure_type: str, error_message: str, severity: str = "medium"
    ):
        """
        Record trading failure and potentially open circuit.

        Args:
            failure_type: Type of failure (order_failure, api_error, etc.)
            error_message: Description of the failure
            severity: Severity level (low, medium, high, critical)
        """
        failure_record = FailureRecord(
            timestamp=datetime.now(UTC),
            failure_type=failure_type,
            error_message=error_message,
            severity=severity,
        )

        self.failure_history.append(failure_record)
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)

        # Critical failures immediately open circuit
        if severity == "critical":
            self.failure_count = self.failure_threshold

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.critical(
                "🚨 CIRCUIT BREAKER TRIGGERED - Trading halted for %ss\nFailure count: %s, Last error: %s",
                self.timeout,
                self.failure_count,
                error_message,
            )
        else:
            logger.warning(
                "⚠️ Trading failure recorded (%s/%s): %s - %s",
                self.failure_count,
                self.failure_threshold,
                failure_type,
                error_message,
            )

        # Keep only recent failure history
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)
        self.failure_history = [
            f for f in self.failure_history if f.timestamp >= cutoff_time
        ]

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status information."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "timeout_remaining": (
                max(
                    0,
                    self.timeout
                    - (datetime.now(UTC) - self.last_failure_time).total_seconds(),
                )
                if self.last_failure_time and self.state == "OPEN"
                else 0
            ),
            "recent_failures": len(
                [
                    f
                    for f in self.failure_history
                    if f.timestamp >= datetime.now(UTC) - timedelta(hours=1)
                ]
            ),
        }


class APIFailureProtection:
    """
    API failure protection with exponential backoff.

    Provides resilient API call execution with automatic retries
    and exponential backoff to handle temporary API issues.
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize API failure protection.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.consecutive_failures = 0
        self.total_failures = 0
        self.last_success_time: datetime | None = None

    async def execute_with_protection(self, func, *args, **kwargs):
        """
        Execute function with failure protection and exponential backoff.

        Args:
            func: Function to execute (can be async or sync)
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Execute function (handle both sync and async)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success - reset failure counters
                self.consecutive_failures = 0
                self.last_success_time = datetime.now(UTC)
                return result

            except Exception as e:
                last_exception = e
                self.consecutive_failures += 1
                self.total_failures += 1

                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = self.base_delay * (2**attempt)
                    logger.warning(
                        "API call failed (attempt %s/%s): %s\nRetrying in %.1fs...",
                        attempt + 1,
                        self.max_retries,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        "API call failed after %s attempts: %s", self.max_retries, e
                    )

        # All attempts failed
        raise last_exception or Exception("All retry attempts exhausted")

    def get_health_status(self) -> dict[str, Any]:
        """Get API health status information."""
        return {
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "last_success": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
            "health_score": max(0, 100 - min(100, self.consecutive_failures * 20)),
        }


class EmergencyStopManager:
    """
    Emergency stop system for critical trading conditions.

    Monitors multiple risk factors and triggers automatic trading halts
    when dangerous conditions are detected.
    """

    def __init__(self):
        """Initialize emergency stop manager."""
        self.stop_triggers = {
            "rapid_loss": {"threshold": 0.05, "timeframe": 300},  # 5% in 5 min
            "api_failures": {
                "threshold": 10,
                "timeframe": 600,
            },  # 10 failures in 10 min
            "position_errors": {"threshold": 3, "timeframe": 60},  # 3 errors in 1 min
            "margin_critical": {"threshold": 0.95, "timeframe": 0},  # 95% margin usage
            "consecutive_losses": {
                "threshold": 5,
                "timeframe": 3600,
            },  # 5 losses in 1 hour
        }
        self.is_stopped = False
        self.stop_reason: str | None = None
        self.stop_timestamp: datetime | None = None
        self.trigger_history: list[dict[str, Any]] = []

        logger.info("Emergency stop manager initialized")

    def check_emergency_conditions(self, metrics: dict[str, Any]) -> bool:
        """
        Check if emergency stop conditions are met.

        Args:
            metrics: Current system metrics

        Returns:
            True if emergency stop should be triggered
        """
        for trigger_name, config in self.stop_triggers.items():
            if self._evaluate_trigger(trigger_name, metrics, config):
                self.trigger_emergency_stop(
                    trigger_name, metrics.get(trigger_name, "unknown")
                )
                return True
        return False

    def _evaluate_trigger(
        self, trigger_name: str, metrics: dict[str, Any], config: dict[str, Any]
    ) -> bool:
        """Evaluate if a specific trigger condition is met."""
        current_value = metrics.get(trigger_name, 0)
        threshold = config["threshold"]
        config["timeframe"]

        if trigger_name == "rapid_loss":
            # Check if loss percentage exceeds threshold in timeframe
            recent_loss = current_value
            return recent_loss >= threshold

        elif trigger_name == "api_failures":
            # Check API failure rate
            failure_count = current_value
            return failure_count >= threshold

        elif trigger_name == "position_errors":
            # Check position validation errors
            error_count = current_value
            return error_count >= threshold

        elif trigger_name == "margin_critical":
            # Check margin usage
            margin_usage = current_value
            return margin_usage >= threshold

        elif trigger_name == "consecutive_losses":
            # Check consecutive losing trades
            consecutive_losses = current_value
            return consecutive_losses >= threshold

        return False

    def trigger_emergency_stop(self, reason: str, trigger_value: Any = None):
        """
        Trigger emergency stop with specified reason.

        Args:
            reason: Reason for emergency stop
            trigger_value: Value that triggered the stop
        """
        self.is_stopped = True
        self.stop_reason = reason
        self.stop_timestamp = datetime.now(UTC)

        trigger_record = {
            "timestamp": self.stop_timestamp,
            "reason": reason,
            "trigger_value": str(trigger_value) if trigger_value is not None else "N/A",
        }
        self.trigger_history.append(trigger_record)

        logger.critical(
            "🚨 EMERGENCY STOP TRIGGERED: %s\nTrigger value: %s\nAll trading operations halted immediately!",
            reason,
            trigger_value,
        )

        # Keep only recent trigger history
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)
        self.trigger_history = [
            t for t in self.trigger_history if t["timestamp"] >= cutoff_time
        ]

    def reset_emergency_stop(self, manual_reset: bool = False):
        """
        Reset emergency stop (should only be done manually after investigation).

        Args:
            manual_reset: Whether this is a manual reset (requires admin action)
        """
        if manual_reset:
            self.is_stopped = False
            self.stop_reason = None
            self.stop_timestamp = None
            logger.warning("⚠️  Emergency stop manually reset - trading resumed")
        else:
            logger.error("❌ Emergency stop reset requires manual intervention")

    def get_status(self) -> dict[str, Any]:
        """Get emergency stop status information."""
        return {
            "is_stopped": self.is_stopped,
            "stop_reason": self.stop_reason,
            "stop_timestamp": (
                self.stop_timestamp.isoformat() if self.stop_timestamp else None
            ),
            "trigger_count_24h": len(self.trigger_history),
            "last_trigger": (
                self.trigger_history[-1]["timestamp"].isoformat()
                if self.trigger_history
                else None
            ),
        }


class RiskManager:
    """
    Advanced risk management system for position sizing and loss control.

    Enforces position size limits, leverage constraints, daily loss limits,
    circuit breaker protection, emergency stops, and advanced position validation
    to protect the trading account from all failure modes.
    """

    def __init__(self, position_manager: PositionManager | None = None) -> None:
        """Initialize the risk manager.

        Args:
            position_manager: Position manager instance for position data
        """
        self.max_size_pct = settings.trading.max_size_pct
        self.leverage = settings.trading.leverage
        self.max_daily_loss_pct = settings.risk.max_daily_loss_pct
        self.max_concurrent_trades = settings.risk.max_concurrent_trades
        self.max_position_size = Decimal("100000")  # Maximum absolute position size
        self.stop_loss_percentage: float = 2.0  # Default stop loss percentage
        self.max_daily_loss: Decimal = Decimal("500")  # Maximum daily loss in USD

        # Position manager integration
        self.position_manager = position_manager

        # Risk tracking
        self._daily_pnl: dict[date, DailyPnL] = {}
        self._account_balance = Decimal("10000")  # Default starting balance

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

            # 📏 Validate and adjust position size
            modified_action = self._validate_position_size(trade_action)

            # 💰 Adjust position size for trading fees
            try:
                modified_action, trade_fees = (
                    fee_calculator.adjust_position_size_for_fees(
                        modified_action, self._account_balance, current_price
                    )
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
                trade_fees.total_fee if trade_fees else Decimal("0"),
            )

            if not balance_valid:
                self.circuit_breaker.record_failure(
                    "balance_validation",
                    f"Balance validation failed: {balance_reason}",
                    "medium",
                )
                return (
                    False,
                    self._get_hold_action(
                        f"Balance validation failed: {balance_reason}"
                    ),
                    "Balance validation",
                )

            if modified_action.size_pct == 0 and trade_action.action in [
                "LONG",
                "SHORT",
            ]:
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
                is_profitable, profit_reason = (
                    fee_calculator.validate_trade_profitability(
                        modified_action, position_value, current_price
                    )
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
                modified_action = self._reduce_position_size(
                    modified_action, risk_metrics
                )
                logger.warning("Position size reduced due to excessive risk")

            # 🔍 Final validation
            if modified_action.size_pct == 0 and trade_action.action in [
                "LONG",
                "SHORT",
            ]:
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

        except Exception as e:
            logger.exception("Risk evaluation error")
            self.circuit_breaker.record_failure("risk_evaluation_error", str(e), "high")
            return False, self._get_hold_action("Risk evaluation error"), "Error"

    def validate_balance_for_trade(
        self,
        trade_action: TradeAction,
        _current_price: Decimal,
        estimated_fees: Decimal = Decimal("0"),
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
            if trade_action.action in ["HOLD", "CLOSE"]:
                return True, "No balance validation required for HOLD/CLOSE"

            # Calculate required balance for the trade
            position_value = self._account_balance * (
                Decimal(str(trade_action.size_pct)) / 100
            )
            leveraged_value = position_value * Decimal(str(self.leverage))
            required_margin = leveraged_value / Decimal(str(self.leverage))
            required_margin + estimated_fees

            # Validate current balance can support this trade
            validation_result = self.balance_validator.validate_balance_range(
                self._account_balance, f"trade_preparation_{trade_action.action}"
            )

            if not validation_result["valid"]:
                return (
                    False,
                    f"Current balance validation failed: {validation_result.get('message', 'Unknown error')}",
                )

            # Check if balance after trade would be valid
            post_trade_balance = self._account_balance - estimated_fees
            if post_trade_balance < Decimal("0"):
                return (
                    False,
                    f"Trade would result in negative balance: ${post_trade_balance}",
                )

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

            # Validate margin requirements
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

            logger.debug(
                "✅ Balance validation passed for %s trade", trade_action.action
            )
            return True, "Balance validation successful"

        except Exception as e:
            logger.exception("Error in balance validation for trade")
            return False, f"Balance validation error: {e}"

    def _calculate_current_margin_usage(self) -> Decimal:
        """Calculate current margin usage across all positions."""
        if not self.position_manager:
            return Decimal("0")

        total_margin = Decimal("0")
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
            Decimal(str(self.max_daily_loss_pct)) / Decimal("100")
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
                "Position size reduced to %s% due to margin constraints",
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
            return {"max_loss_usd": Decimal("0"), "max_gain_usd": Decimal("0")}

        position_value = self._account_balance * (
            Decimal(str(trade_action.size_pct)) / Decimal("100")
        )
        leveraged_exposure = position_value * Decimal(str(self.leverage))

        # Calculate potential loss (stop loss) + fees
        max_loss_pct = Decimal(str(trade_action.stop_loss_pct)) / Decimal("100")
        max_loss_usd = leveraged_exposure * max_loss_pct

        # Add trading fees to the max loss
        if trade_fees and trade_fees.total_fee > 0:
            max_loss_usd += trade_fees.total_fee

        # Calculate potential gain (take profit) - fees
        max_gain_pct = Decimal(str(trade_action.take_profit_pct)) / Decimal("100")
        max_gain_usd = leveraged_exposure * max_gain_pct

        # Subtract trading fees from the max gain
        if trade_fees and trade_fees.total_fee > 0:
            max_gain_usd -= trade_fees.total_fee
            max_gain_usd = max(Decimal("0"), max_gain_usd)  # Ensure non-negative

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
            "total_fees": trade_fees.total_fee if trade_fees else Decimal("0"),
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
                        "reason": f"Entry price {position.entry_price} deviates {price_deviation*100:.1f}% from market {market_price}",
                        "severity": "medium",
                    }

                # Check unrealized P&L sanity
                if position.entry_price and position.side != "FLAT":
                    expected_pnl = self._calculate_expected_pnl(position, market_price)
                    pnl_diff = abs(position.unrealized_pnl - expected_pnl)
                    tolerance = abs(expected_pnl * Decimal("0.1"))  # 10% tolerance

                    if pnl_diff > tolerance and tolerance > 0:
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

        except Exception as e:
            logger.exception("Position validation error")
            return {
                "valid": False,
                "reason": f"Validation error: {e}",
                "severity": "high",
            }

    def _calculate_expected_pnl(
        self, position: Position, current_price: Decimal
    ) -> Decimal:
        """Calculate expected unrealized P&L for position validation."""
        if position.side == "FLAT" or position.entry_price is None:
            return Decimal("0")

        price_diff = current_price - position.entry_price

        if position.side == "LONG":
            return position.size * price_diff
        else:  # SHORT
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
        elif any(medium_risk_conditions):
            return "MEDIUM"
        else:
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
        self, realized_pnl: Decimal, unrealized_pnl: Decimal = Decimal("0")
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
        if total_pnl < daily_data.max_drawdown:
            daily_data.max_drawdown = total_pnl

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
            used_margin=Decimal("100") - self._get_available_margin(),
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
