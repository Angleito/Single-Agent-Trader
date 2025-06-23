"""
Circuit breaker pattern for trading operations.

This module implements a circuit breaker that automatically halts trading
when consecutive failures exceed threshold, preventing cascade failures
and protecting the account.
"""

import logging
from datetime import UTC, datetime, timedelta

from bot.types.base_types import CircuitBreakerStatus

from .types import FailureRecord

logger = logging.getLogger(__name__)


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

    def get_status(self) -> CircuitBreakerStatus:
        """Get circuit breaker status information."""
        return CircuitBreakerStatus(
            state=self.state,  # type: ignore
            failure_count=self.failure_count,
            failure_threshold=self.failure_threshold,
            last_failure=(
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            timeout_remaining=(
                max(
                    0,
                    self.timeout
                    - (datetime.now(UTC) - self.last_failure_time).total_seconds(),
                )
                if self.last_failure_time and self.state == "OPEN"
                else 0
            ),
            recent_failures=len(
                [
                    f
                    for f in self.failure_history
                    if f.timestamp >= datetime.now(UTC) - timedelta(hours=1)
                ]
            ),
        )
