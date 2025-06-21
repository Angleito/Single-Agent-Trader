"""
Unit tests for the risk management system.

Tests the critical risk management, position sizing, leverage control,
daily loss limits, and circuit breaker functionality.
"""

import unittest
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from bot.risk import DailyPnL, FailureRecord, RiskManager, TradingCircuitBreaker
from bot.trading_types import Position, TradeAction


class TestDailyPnL(unittest.TestCase):
    """Test daily P&L tracking."""

    def test_daily_pnl_creation(self):
        """Test creating a daily P&L record."""
        today = date.today()
        pnl = DailyPnL(date=today)

        assert pnl.date == today
        assert pnl.realized_pnl == Decimal(0)
        assert pnl.unrealized_pnl == Decimal(0)
        assert pnl.trades_count == 0
        assert pnl.max_drawdown == Decimal(0)

    def test_daily_pnl_with_values(self):
        """Test daily P&L with specific values."""
        today = date.today()
        pnl = DailyPnL(
            date=today,
            realized_pnl=Decimal("150.50"),
            unrealized_pnl=Decimal("-25.30"),
            trades_count=5,
            max_drawdown=Decimal("-45.75"),
        )

        assert pnl.realized_pnl == Decimal("150.50")
        assert pnl.unrealized_pnl == Decimal("-25.30")
        assert pnl.trades_count == 5
        assert pnl.max_drawdown == Decimal("-45.75")


class TestFailureRecord(unittest.TestCase):
    """Test failure record for circuit breaker."""

    def test_failure_record_creation(self):
        """Test creating a failure record."""
        timestamp = datetime.now(UTC)
        record = FailureRecord(
            timestamp=timestamp,
            failure_type="order_rejection",
            error_message="Insufficient balance",
        )

        assert record.timestamp == timestamp
        assert record.failure_type == "order_rejection"
        assert record.error_message == "Insufficient balance"
        assert record.severity == "medium"

    def test_failure_record_with_severity(self):
        """Test failure record with custom severity."""
        timestamp = datetime.now(UTC)
        record = FailureRecord(
            timestamp=timestamp,
            failure_type="exchange_down",
            error_message="Exchange unavailable",
            severity="critical",
        )

        assert record.severity == "critical"


class TestTradingCircuitBreaker(unittest.TestCase):
    """Test trading circuit breaker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.circuit_breaker = TradingCircuitBreaker(
            failure_threshold=3, time_window_minutes=30, cooldown_minutes=60
        )

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        assert self.circuit_breaker.failure_threshold == 3
        assert self.circuit_breaker.time_window_minutes == 30
        assert self.circuit_breaker.cooldown_minutes == 60
        assert not self.circuit_breaker.is_tripped
        assert len(self.circuit_breaker.failure_history) == 0

    def test_circuit_breaker_single_failure(self):
        """Test circuit breaker with single failure."""
        self.circuit_breaker.record_failure(
            failure_type="order_rejection", error_message="Test error"
        )

        assert not self.circuit_breaker.is_tripped
        assert len(self.circuit_breaker.failure_history) == 1

    def test_circuit_breaker_threshold_exceeded(self):
        """Test circuit breaker when threshold is exceeded."""
        # Record failures up to threshold
        for i in range(3):
            self.circuit_breaker.record_failure(
                failure_type="order_rejection", error_message=f"Test error {i}"
            )

        assert self.circuit_breaker.is_tripped
        assert len(self.circuit_breaker.failure_history) == 3

    def test_circuit_breaker_old_failures_ignored(self):
        """Test that old failures outside time window are ignored."""
        circuit_breaker = TradingCircuitBreaker(
            failure_threshold=2, time_window_minutes=30, cooldown_minutes=60
        )

        # Record old failure (outside time window)
        old_timestamp = datetime.now(UTC) - timedelta(minutes=45)
        circuit_breaker.failure_history.append(
            FailureRecord(
                timestamp=old_timestamp,
                failure_type="old_error",
                error_message="Old error",
            )
        )

        # Record new failure
        circuit_breaker.record_failure(
            failure_type="new_error", error_message="New error"
        )

        # Should not be tripped (only 1 recent failure)
        assert not circuit_breaker.is_tripped

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset functionality."""
        # Trip the circuit breaker
        for i in range(3):
            self.circuit_breaker.record_failure(
                failure_type="order_rejection", error_message=f"Test error {i}"
            )

        assert self.circuit_breaker.is_tripped

        # Reset
        self.circuit_breaker.reset()

        assert not self.circuit_breaker.is_tripped
        assert len(self.circuit_breaker.failure_history) == 0


class TestRiskManager(unittest.TestCase):
    """Test the main risk manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_position_manager = Mock()
        self.mock_balance_validator = Mock()

        # Create risk manager with mocked dependencies
        with (
            patch("bot.risk.PositionManager", return_value=self.mock_position_manager),
            patch(
                "bot.risk.BalanceValidator", return_value=self.mock_balance_validator
            ),
        ):
            self.risk_manager = RiskManager()

    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        assert hasattr(self.risk_manager, "max_size_pct")
        assert hasattr(self.risk_manager, "leverage")
        assert hasattr(self.risk_manager, "max_daily_loss_pct")
        assert hasattr(self.risk_manager, "max_concurrent_trades")
        assert self.risk_manager.max_position_size == Decimal(100000)
        assert self.risk_manager.max_daily_loss == Decimal(500)
        assert hasattr(self.risk_manager, "circuit_breaker")
        assert hasattr(self.risk_manager, "emergency_stop")
        assert hasattr(self.risk_manager, "balance_validator")

    def test_evaluate_risk_basic(self):
        """Test basic risk evaluation."""
        # Create test data
        trade_action = TradeAction(
            action="LONG",
            size_pct=5.0,  # Small position
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Test trade",
        )

        current_position = Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal(0),
            entry_price=None,
            unrealized_pnl=Decimal(0),
            realized_pnl=Decimal(0),
            timestamp=datetime.now(UTC),
        )

        # Test risk evaluation
        is_approved, modified_action, reason = self.risk_manager.evaluate_risk(
            trade_action=trade_action,
            current_position=current_position,
            current_price=Decimal(50000),
        )

        # Basic test - should return some result
        assert isinstance(is_approved, bool)
        assert isinstance(modified_action, TradeAction)
        assert isinstance(reason, str)

    def test_circuit_breaker_component(self):
        """Test circuit breaker component exists and functions."""
        # Test circuit breaker exists
        assert hasattr(self.risk_manager, "circuit_breaker")
        assert hasattr(self.risk_manager.circuit_breaker, "record_failure")

        # Test recording failure
        self.risk_manager.circuit_breaker.record_failure(
            "test_failure", "test_error", "medium"
        )

        # Should have recorded the failure
        assert len(self.risk_manager.circuit_breaker.failure_history) > 0

    def test_emergency_stop_component(self):
        """Test emergency stop component exists."""
        assert hasattr(self.risk_manager, "emergency_stop")
        assert hasattr(self.risk_manager.emergency_stop, "is_stopped")

        # Initially not stopped
        assert not self.risk_manager.emergency_stop.is_stopped

    def test_balance_validator_component(self):
        """Test balance validator component exists."""
        assert hasattr(self.risk_manager, "balance_validator")
        assert self.risk_manager.balance_validator is not None

    def test_daily_pnl_tracking(self):
        """Test daily P&L tracking structure."""
        assert hasattr(self.risk_manager, "_daily_pnl")
        assert isinstance(self.risk_manager._daily_pnl, dict)

        # Test adding daily P&L
        today = date.today()
        test_pnl = DailyPnL(date=today, realized_pnl=Decimal("100.50"), trades_count=3)

        self.risk_manager._daily_pnl[today] = test_pnl
        assert today in self.risk_manager._daily_pnl
        assert self.risk_manager._daily_pnl[today].realized_pnl == Decimal("100.50")

    def test_risk_attributes(self):
        """Test risk manager has expected attributes."""
        # Check all the critical risk management attributes
        expected_attributes = [
            "max_size_pct",
            "leverage",
            "max_daily_loss_pct",
            "max_concurrent_trades",
            "max_position_size",
            "max_daily_loss",
            "position_manager",
            "balance_validator",
            "circuit_breaker",
            "emergency_stop",
        ]

        for attr in expected_attributes:
            assert hasattr(self.risk_manager, attr), f"Missing attribute: {attr}"


if __name__ == "__main__":
    unittest.main()
