"""
Comprehensive balance validation system for trading operations.

This module provides robust validation for account balances, margin calculations,
range checks, anomaly detection, and sanity testing to prevent balance corruption
and ensure production reliability.
"""

import logging
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, getcontext
from typing import Any, NamedTuple

# Set decimal precision for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

logger = logging.getLogger(__name__)


class BalanceValidationError(Exception):
    """Custom exception for balance validation failures."""

    def __init__(
        self, message: str, error_code: str | None = None, validation_type: str | None = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.validation_type = validation_type
        self.timestamp = datetime.now(UTC)


class ValidationThresholds(NamedTuple):
    """Configuration thresholds for balance validation."""

    # Range validation
    min_balance: Decimal = Decimal("0.00")
    max_balance: Decimal = Decimal("10000000.00")  # $10M max reasonable balance

    # Change detection
    max_balance_change_pct: float = 25.0  # 25% max change per operation
    max_absolute_change: Decimal = Decimal("50000.00")  # $50k max absolute change

    # Precision validation
    decimal_places: int = 2  # USD currency precision
    crypto_decimal_places: int = 8  # Crypto precision

    # Anomaly detection
    anomaly_threshold_pct: float = 10.0  # 10% threshold for anomaly detection
    rapid_change_window_seconds: int = 300  # 5 minutes for rapid change detection

    # Margin validation
    max_margin_ratio: float = 0.95  # 95% max margin usage
    min_margin_buffer: Decimal = Decimal("100.00")  # $100 minimum margin buffer


class BalanceChangeRecord(NamedTuple):
    """Record of balance change for history tracking."""

    timestamp: datetime
    old_balance: Decimal
    new_balance: Decimal
    change_amount: Decimal
    change_pct: float
    operation_type: str
    metadata: dict[str, Any] = {}


class BalanceValidator:
    """
    Comprehensive balance validation system.

    Provides validation for:
    - Balance range checks (min/max limits)
    - Balance change validation (sudden large changes)
    - Precision validation (correct decimal places)
    - Anomaly detection (unusual patterns)
    - Margin calculation validation
    - Balance reconciliation checks
    """

    def __init__(self, thresholds: ValidationThresholds = None):
        """
        Initialize balance validator.

        Args:
            thresholds: Custom validation thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or ValidationThresholds()
        self.balance_history: list[BalanceChangeRecord] = []
        self.last_validation_time = datetime.now(UTC)
        self.validation_count = 0
        self.error_count = 0

        logger.info(
            f"BalanceValidator initialized with thresholds:\n"
            f"  • Balance range: ${self.thresholds.min_balance} - ${self.thresholds.max_balance}\n"
            f"  • Max change: {self.thresholds.max_balance_change_pct}% or ${self.thresholds.max_absolute_change}\n"
            f"  • Precision: {self.thresholds.decimal_places} decimal places\n"
            f"  • Anomaly threshold: {self.thresholds.anomaly_threshold_pct}%\n"
            f"  • Max margin ratio: {self.thresholds.max_margin_ratio:.1%}"
        )

    def validate_balance_range(
        self, balance: Decimal, context: str = "balance_check"
    ) -> dict[str, Any]:
        """
        Validate balance is within acceptable range.

        Args:
            balance: Balance to validate
            context: Context of the validation for logging

        Returns:
            Validation result dictionary

        Raises:
            BalanceValidationError: If balance is outside acceptable range
        """
        self.validation_count += 1

        try:
            # Normalize balance
            normalized_balance = self._normalize_balance(balance)

            # Check minimum threshold
            if normalized_balance < self.thresholds.min_balance:
                error_msg = f"Balance below minimum threshold: ${normalized_balance} < ${self.thresholds.min_balance}"
                self._log_validation_error(error_msg, context, "RANGE_MIN")
                raise BalanceValidationError(
                    error_msg, error_code="BALANCE_BELOW_MIN", validation_type="range"
                )

            # Check maximum threshold
            if normalized_balance > self.thresholds.max_balance:
                error_msg = f"Balance exceeds maximum threshold: ${normalized_balance} > ${self.thresholds.max_balance}"
                self._log_validation_error(error_msg, context, "RANGE_MAX")
                raise BalanceValidationError(
                    error_msg, error_code="BALANCE_ABOVE_MAX", validation_type="range"
                )

            # Check for negative balance (additional safety check)
            if normalized_balance < Decimal("0"):
                error_msg = f"Negative balance detected: ${normalized_balance}"
                self._log_validation_error(error_msg, context, "NEGATIVE")
                raise BalanceValidationError(
                    error_msg, error_code="NEGATIVE_BALANCE", validation_type="range"
                )

            logger.debug(
                f"✅ Balance range validation passed: ${normalized_balance} ({context})"
            )

            return {
                "valid": True,
                "normalized_balance": normalized_balance,
                "message": "Balance within acceptable range",
                "context": context,
                "validation_time": datetime.now(UTC),
            }

        except BalanceValidationError:
            self.error_count += 1
            raise
        except Exception as e:
            self.error_count += 1
            error_msg = f"Balance range validation error: {e}"
            self._log_validation_error(error_msg, context, "VALIDATION_ERROR")
            raise BalanceValidationError(
                error_msg, error_code="VALIDATION_ERROR", validation_type="range"
            ) from e

    def validate_balance_change(
        self,
        old_balance: Decimal,
        new_balance: Decimal,
        operation_type: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Validate balance change is reasonable and not anomalous.

        Args:
            old_balance: Previous balance
            new_balance: New balance
            operation_type: Type of operation causing the change
            metadata: Additional context information

        Returns:
            Validation result dictionary

        Raises:
            BalanceValidationError: If balance change is invalid
        """
        self.validation_count += 1
        metadata = metadata or {}

        try:
            # Normalize balances
            old_normalized = self._normalize_balance(old_balance)
            new_normalized = self._normalize_balance(new_balance)

            # Calculate change metrics
            change_amount = new_normalized - old_normalized
            change_pct = (
                float(abs(change_amount) / old_normalized * 100)
                if old_normalized > 0
                else 0.0
            )

            # Create change record
            change_record = BalanceChangeRecord(
                timestamp=datetime.now(UTC),
                old_balance=old_normalized,
                new_balance=new_normalized,
                change_amount=change_amount,
                change_pct=change_pct,
                operation_type=operation_type,
                metadata=metadata,
            )

            # Validate percentage change
            if change_pct > self.thresholds.max_balance_change_pct:
                error_msg = (
                    f"Balance change too large: {change_pct:.2f}% > {self.thresholds.max_balance_change_pct}% "
                    f"(${old_normalized} -> ${new_normalized})"
                )
                self._log_validation_error(error_msg, operation_type, "CHANGE_PCT")
                raise BalanceValidationError(
                    error_msg,
                    error_code="EXCESSIVE_CHANGE_PCT",
                    validation_type="change",
                )

            # Validate absolute change
            if abs(change_amount) > self.thresholds.max_absolute_change:
                error_msg = (
                    f"Absolute balance change too large: ${abs(change_amount)} > ${self.thresholds.max_absolute_change} "
                    f"(${old_normalized} -> ${new_normalized})"
                )
                self._log_validation_error(error_msg, operation_type, "CHANGE_ABS")
                raise BalanceValidationError(
                    error_msg,
                    error_code="EXCESSIVE_CHANGE_ABS",
                    validation_type="change",
                )

            # Check for impossible changes (e.g., balance going from 0 to very high instantly)
            if old_normalized == Decimal("0") and new_normalized > Decimal("1000"):
                error_msg = (
                    f"Impossible balance change: ${old_normalized} -> ${new_normalized} "
                    f"(zero to significant amount instantly)"
                )
                self._log_validation_error(error_msg, operation_type, "IMPOSSIBLE")
                raise BalanceValidationError(
                    error_msg, error_code="IMPOSSIBLE_CHANGE", validation_type="change"
                )

            # Record the change
            self.balance_history.append(change_record)
            self._cleanup_old_history()

            logger.debug(
                f"✅ Balance change validation passed: ${old_normalized} -> ${new_normalized} "
                f"({change_pct:.2f}%, {operation_type})"
            )

            return {
                "valid": True,
                "old_balance": old_normalized,
                "new_balance": new_normalized,
                "change_amount": change_amount,
                "change_pct": change_pct,
                "operation_type": operation_type,
                "message": "Balance change within acceptable limits",
                "validation_time": datetime.now(UTC),
            }

        except BalanceValidationError:
            self.error_count += 1
            raise
        except Exception as e:
            self.error_count += 1
            error_msg = f"Balance change validation error: {e}"
            self._log_validation_error(error_msg, operation_type, "VALIDATION_ERROR")
            raise BalanceValidationError(
                error_msg, error_code="VALIDATION_ERROR", validation_type="change"
            ) from e

    def validate_balance_precision(
        self, balance: Decimal, is_crypto: bool = False
    ) -> dict[str, Any]:
        """
        Validate balance has correct decimal precision.

        Args:
            balance: Balance to validate
            is_crypto: Whether this is a crypto balance (higher precision)

        Returns:
            Validation result dictionary

        Raises:
            BalanceValidationError: If precision is incorrect
        """
        self.validation_count += 1

        try:
            # Determine expected decimal places
            expected_places = (
                self.thresholds.crypto_decimal_places
                if is_crypto
                else self.thresholds.decimal_places
            )

            # Check decimal places
            balance_str = str(balance)
            if "." in balance_str:
                actual_places = len(balance_str.split(".")[1])
                if actual_places > expected_places:
                    error_msg = (
                        f"Balance precision too high: {actual_places} > {expected_places} decimal places "
                        f"(balance: {balance})"
                    )
                    self._log_validation_error(
                        error_msg, "precision_check", "PRECISION"
                    )
                    raise BalanceValidationError(
                        error_msg,
                        error_code="EXCESSIVE_PRECISION",
                        validation_type="precision",
                    )

            # Normalize to correct precision
            normalized_balance = self._normalize_balance(balance, is_crypto)

            logger.debug(
                f"✅ Balance precision validation passed: {normalized_balance}"
            )

            return {
                "valid": True,
                "normalized_balance": normalized_balance,
                "expected_precision": expected_places,
                "message": "Balance precision is correct",
                "validation_time": datetime.now(UTC),
            }

        except BalanceValidationError:
            self.error_count += 1
            raise
        except Exception as e:
            self.error_count += 1
            error_msg = f"Balance precision validation error: {e}"
            self._log_validation_error(error_msg, "precision_check", "VALIDATION_ERROR")
            raise BalanceValidationError(
                error_msg, error_code="VALIDATION_ERROR", validation_type="precision"
            ) from e

    def detect_balance_anomalies(self, current_balance: Decimal) -> dict[str, Any]:
        """
        Detect anomalous balance patterns.

        Args:
            current_balance: Current balance to analyze

        Returns:
            Anomaly detection result dictionary

        Raises:
            BalanceValidationError: If anomalies are detected
        """
        self.validation_count += 1

        try:
            normalized_balance = self._normalize_balance(current_balance)
            anomalies_detected = []

            # Check rapid successive changes
            rapid_changes = self._detect_rapid_changes()
            if rapid_changes:
                anomalies_detected.extend(rapid_changes)

            # Check unusual patterns
            patterns = self._detect_unusual_patterns(normalized_balance)
            if patterns:
                anomalies_detected.extend(patterns)

            # Check for oscillating balance (multiple back-and-forth changes)
            oscillation = self._detect_balance_oscillation()
            if oscillation:
                anomalies_detected.append(oscillation)

            # If any anomalies detected, raise error
            if anomalies_detected:
                error_msg = (
                    f"Balance anomalies detected: {', '.join(anomalies_detected)}"
                )
                self._log_validation_error(error_msg, "anomaly_detection", "ANOMALY")
                raise BalanceValidationError(
                    error_msg, error_code="BALANCE_ANOMALY", validation_type="anomaly"
                )

            logger.debug(f"✅ No balance anomalies detected for: ${normalized_balance}")

            return {
                "valid": True,
                "balance": normalized_balance,
                "anomalies": [],
                "message": "No anomalies detected",
                "validation_time": datetime.now(UTC),
            }

        except BalanceValidationError:
            self.error_count += 1
            raise
        except Exception as e:
            self.error_count += 1
            error_msg = f"Anomaly detection error: {e}"
            self._log_validation_error(
                error_msg, "anomaly_detection", "VALIDATION_ERROR"
            )
            raise BalanceValidationError(
                error_msg, error_code="VALIDATION_ERROR", validation_type="anomaly"
            ) from e

    def validate_margin_calculation(
        self,
        balance: Decimal,
        used_margin: Decimal,
        position_value: Decimal | None = None,
        leverage: int | None = None,
    ) -> dict[str, Any]:
        """
        Validate margin calculations for sanity.

        Args:
            balance: Account balance
            used_margin: Currently used margin
            position_value: Total position value (optional)
            leverage: Leverage ratio (optional)

        Returns:
            Margin validation result dictionary

        Raises:
            BalanceValidationError: If margin calculations are invalid
        """
        self.validation_count += 1

        try:
            normalized_balance = self._normalize_balance(balance)
            normalized_used_margin = self._normalize_balance(used_margin)

            # Check margin doesn't exceed balance
            if normalized_used_margin > normalized_balance:
                error_msg = f"Used margin exceeds balance: ${normalized_used_margin} > ${normalized_balance}"
                self._log_validation_error(
                    error_msg, "margin_validation", "MARGIN_EXCESS"
                )
                raise BalanceValidationError(
                    error_msg,
                    error_code="MARGIN_EXCEEDS_BALANCE",
                    validation_type="margin",
                )

            # Check margin ratio
            margin_ratio = (
                float(normalized_used_margin / normalized_balance)
                if normalized_balance > 0
                else 0
            )
            if margin_ratio > self.thresholds.max_margin_ratio:
                error_msg = f"Margin ratio too high: {margin_ratio:.1%} > {self.thresholds.max_margin_ratio:.1%}"
                self._log_validation_error(
                    error_msg, "margin_validation", "MARGIN_RATIO"
                )
                raise BalanceValidationError(
                    error_msg,
                    error_code="EXCESSIVE_MARGIN_RATIO",
                    validation_type="margin",
                )

            # Check minimum margin buffer
            available_margin = normalized_balance - normalized_used_margin
            if available_margin < self.thresholds.min_margin_buffer:
                error_msg = f"Insufficient margin buffer: ${available_margin} < ${self.thresholds.min_margin_buffer}"
                self._log_validation_error(
                    error_msg, "margin_validation", "MARGIN_BUFFER"
                )
                raise BalanceValidationError(
                    error_msg,
                    error_code="INSUFFICIENT_MARGIN_BUFFER",
                    validation_type="margin",
                )

            # Validate position-margin relationship if position value provided
            if position_value is not None and leverage is not None:
                normalized_position_value = self._normalize_balance(position_value)
                expected_margin = normalized_position_value / Decimal(str(leverage))
                margin_tolerance = expected_margin * Decimal("0.1")  # 10% tolerance

                if abs(normalized_used_margin - expected_margin) > margin_tolerance:
                    error_msg = (
                        f"Margin calculation mismatch: expected ${expected_margin}, got ${normalized_used_margin} "
                        f"(position: ${normalized_position_value}, leverage: {leverage}x)"
                    )
                    self._log_validation_error(
                        error_msg, "margin_validation", "MARGIN_CALC"
                    )
                    raise BalanceValidationError(
                        error_msg,
                        error_code="MARGIN_CALCULATION_MISMATCH",
                        validation_type="margin",
                    )

            logger.debug(
                f"✅ Margin validation passed: balance=${normalized_balance}, "
                f"used=${normalized_used_margin}, ratio={margin_ratio:.1%}"
            )

            return {
                "valid": True,
                "balance": normalized_balance,
                "used_margin": normalized_used_margin,
                "available_margin": available_margin,
                "margin_ratio": margin_ratio,
                "message": "Margin calculations are valid",
                "validation_time": datetime.now(UTC),
            }

        except BalanceValidationError:
            self.error_count += 1
            raise
        except Exception as e:
            self.error_count += 1
            error_msg = f"Margin validation error: {e}"
            self._log_validation_error(
                error_msg, "margin_validation", "VALIDATION_ERROR"
            )
            raise BalanceValidationError(
                error_msg, error_code="VALIDATION_ERROR", validation_type="margin"
            ) from e

    def validate_balance_reconciliation(
        self,
        calculated_balance: Decimal,
        reported_balance: Decimal,
        tolerance_pct: float = 0.1,
    ) -> dict[str, Any]:
        """
        Validate balance reconciliation between calculated and reported values.

        Args:
            calculated_balance: Balance calculated from transactions
            reported_balance: Balance reported by exchange/system
            tolerance_pct: Acceptable difference percentage

        Returns:
            Reconciliation validation result dictionary

        Raises:
            BalanceValidationError: If reconciliation fails
        """
        self.validation_count += 1

        try:
            calc_normalized = self._normalize_balance(calculated_balance)
            reported_normalized = self._normalize_balance(reported_balance)

            # Calculate difference
            difference = abs(calc_normalized - reported_normalized)
            diff_pct = (
                float(difference / calc_normalized * 100)
                if calc_normalized > 0
                else 0.0
            )

            # Check if difference is within tolerance
            if diff_pct > tolerance_pct:
                error_msg = (
                    f"Balance reconciliation failed: calculated=${calc_normalized}, "
                    f"reported=${reported_normalized}, difference={diff_pct:.4f}% > {tolerance_pct}%"
                )
                self._log_validation_error(
                    error_msg, "reconciliation", "RECONCILIATION"
                )
                raise BalanceValidationError(
                    error_msg,
                    error_code="RECONCILIATION_FAILED",
                    validation_type="reconciliation",
                )

            logger.debug(
                f"✅ Balance reconciliation passed: calculated=${calc_normalized}, "
                f"reported=${reported_normalized}, difference={diff_pct:.4f}%"
            )

            return {
                "valid": True,
                "calculated_balance": calc_normalized,
                "reported_balance": reported_normalized,
                "difference": difference,
                "difference_pct": diff_pct,
                "tolerance_pct": tolerance_pct,
                "message": "Balance reconciliation successful",
                "validation_time": datetime.now(UTC),
            }

        except BalanceValidationError:
            self.error_count += 1
            raise
        except Exception as e:
            self.error_count += 1
            error_msg = f"Balance reconciliation error: {e}"
            self._log_validation_error(error_msg, "reconciliation", "VALIDATION_ERROR")
            raise BalanceValidationError(
                error_msg,
                error_code="VALIDATION_ERROR",
                validation_type="reconciliation",
            ) from e

    def comprehensive_balance_validation(
        self,
        balance: Decimal,
        previous_balance: Decimal | None = None,
        operation_type: str = "update",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run comprehensive validation on a balance value.

        Args:
            balance: Balance to validate
            previous_balance: Previous balance for change validation
            operation_type: Type of operation
            metadata: Additional context

        Returns:
            Comprehensive validation result

        Raises:
            BalanceValidationError: If any validation fails
        """
        validation_results = {}

        try:
            # Range validation
            validation_results["range"] = self.validate_balance_range(
                balance, operation_type
            )

            # Precision validation
            validation_results["precision"] = self.validate_balance_precision(balance)

            # Change validation (if previous balance provided)
            if previous_balance is not None:
                validation_results["change"] = self.validate_balance_change(
                    previous_balance, balance, operation_type, metadata
                )

            # Anomaly detection
            validation_results["anomaly"] = self.detect_balance_anomalies(balance)

            logger.info(f"✅ Comprehensive balance validation passed for ${balance}")

            return {
                "valid": True,
                "balance": self._normalize_balance(balance),
                "validations": validation_results,
                "message": "All balance validations passed",
                "validation_time": datetime.now(UTC),
            }

        except BalanceValidationError as e:
            logger.exception(f"❌ Comprehensive balance validation failed: {e}")
            validation_results["error"] = {
                "error_code": e.error_code,
                "validation_type": e.validation_type,
                "message": str(e),
                "timestamp": e.timestamp,
            }

            return {
                "valid": False,
                "balance": balance,
                "validations": validation_results,
                "error": validation_results["error"],
                "validation_time": datetime.now(UTC),
            }

    def get_validation_statistics(self) -> dict[str, Any]:
        """
        Get validation statistics and health metrics.

        Returns:
            Dictionary with validation statistics
        """
        success_rate = (
            (self.validation_count - self.error_count) / self.validation_count * 100
            if self.validation_count > 0
            else 0.0
        )

        recent_changes = [
            record
            for record in self.balance_history
            if record.timestamp >= datetime.now(UTC) - timedelta(hours=1)
        ]

        return {
            "total_validations": self.validation_count,
            "total_errors": self.error_count,
            "success_rate": success_rate,
            "last_validation": self.last_validation_time.isoformat(),
            "balance_changes_last_hour": len(recent_changes),
            "balance_history_size": len(self.balance_history),
            "thresholds": {
                "min_balance": float(self.thresholds.min_balance),
                "max_balance": float(self.thresholds.max_balance),
                "max_change_pct": self.thresholds.max_balance_change_pct,
                "max_absolute_change": float(self.thresholds.max_absolute_change),
                "anomaly_threshold": self.thresholds.anomaly_threshold_pct,
            },
        }

    def _normalize_balance(self, balance: Decimal, is_crypto: bool = False) -> Decimal:
        """
        Normalize balance to correct precision.

        Args:
            balance: Balance to normalize
            is_crypto: Whether this is a crypto balance

        Returns:
            Normalized balance
        """
        if balance is None:
            return Decimal("0.00") if not is_crypto else Decimal("0.00000000")

        precision = (
            f"0.{'0' * self.thresholds.crypto_decimal_places}"
            if is_crypto
            else f"0.{'0' * self.thresholds.decimal_places}"
        )

        return balance.quantize(Decimal(precision), rounding=ROUND_HALF_EVEN)

    def _detect_rapid_changes(self) -> list[str]:
        """Detect rapid successive balance changes."""
        anomalies: list[str] = []

        if len(self.balance_history) < 3:
            return anomalies

        # Check last few changes
        recent_window = datetime.now(UTC) - timedelta(
            seconds=self.thresholds.rapid_change_window_seconds
        )
        recent_changes = [
            record
            for record in self.balance_history[-5:]
            if record.timestamp >= recent_window
        ]

        if len(recent_changes) >= 3:
            # Check if all changes are significant
            significant_changes = [
                change
                for change in recent_changes
                if change.change_pct > self.thresholds.anomaly_threshold_pct
            ]

            if len(significant_changes) >= 3:
                anomalies.append(
                    f"Rapid successive changes: {len(significant_changes)} significant changes in {self.thresholds.rapid_change_window_seconds}s"
                )

        return anomalies

    def _detect_unusual_patterns(self, current_balance: Decimal) -> list[str]:
        """Detect unusual balance patterns."""
        anomalies: list[str] = []

        if len(self.balance_history) < 5:
            return anomalies

        # Check for repeated exact amounts (possible system error)
        recent_balances = [record.new_balance for record in self.balance_history[-5:]]
        if len(set(recent_balances)) == 1 and len(recent_balances) > 2:
            anomalies.append("Repeated exact balance values")

        # Check for impossible precision (e.g., trading fees resulting in round numbers)
        if current_balance > Decimal("100") and str(current_balance).endswith(".00"):
            # This might be suspicious for an actively trading account
            recent_trades = [
                record
                for record in self.balance_history[-3:]
                if record.operation_type in ["trade", "fee", "settlement"]
            ]
            if recent_trades:
                anomalies.append("Suspiciously round balance after trading operations")

        return anomalies

    def _detect_balance_oscillation(self) -> str | None:
        """Detect balance oscillation patterns."""
        if len(self.balance_history) < 4:
            return None

        # Check last 4 changes for oscillation pattern
        recent_changes = self.balance_history[-4:]
        directions = []

        for change in recent_changes:
            if change.change_amount > 0:
                directions.append("up")
            elif change.change_amount < 0:
                directions.append("down")
            else:
                directions.append("none")

        # Check for alternating pattern
        if len(directions) >= 4 and (
            directions[0] != directions[1]
            and directions[1] != directions[2]
            and directions[2] != directions[3]
            and directions[0] == directions[2]
        ):
            return "Balance oscillation detected (alternating up/down pattern)"

        return None

    def _cleanup_old_history(self):
        """Clean up old balance history to prevent memory issues."""
        # Keep only last 24 hours of history
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)
        self.balance_history = [
            record for record in self.balance_history if record.timestamp >= cutoff_time
        ]

    def _log_validation_error(self, message: str, context: str, error_type: str):
        """Log validation error with context."""
        logger.error(
            f"❌ Balance Validation Error [{error_type}] in {context}: {message}"
        )
        self.last_validation_time = datetime.now(UTC)
