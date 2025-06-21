"""
Unit tests for the position management system.

Tests position tracking, P&L calculations, risk metrics,
and state persistence functionality.
"""

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bot.position_manager import (
    PositionManager,
    PositionManagerError,
    PositionStateError,
    PositionValidationError,
)
from bot.trading_types import Position


class TestPositionManagerExceptions(unittest.TestCase):
    """Test position manager exception hierarchy."""

    def test_position_manager_error(self):
        """Test base position manager error."""
        error = PositionManagerError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_position_validation_error(self):
        """Test position validation error."""
        error = PositionValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, PositionManagerError)

    def test_position_state_error(self):
        """Test position state error."""
        error = PositionStateError("State operation failed")
        assert str(error) == "State operation failed"
        assert isinstance(error, PositionManagerError)


class TestPositionManager(unittest.TestCase):
    """Test position manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())

        # Mock dependencies
        self.mock_paper_trading = Mock()
        # Configure mock to return proper account status dictionary
        self.mock_paper_trading.get_account_status.return_value = {
            "equity": 10000.0,
            "balance": 10000.0,
            "open_positions": 0,
            "current_balance": 10000.0,
            "margin_used": 0.0,
            "margin_available": 10000.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
        }

        with patch(
            "bot.position_manager.PaperTradingAccount",
            return_value=self.mock_paper_trading,
        ):
            self.position_manager = PositionManager(data_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_position_manager_initialization(self):
        """Test position manager initialization."""
        assert isinstance(self.position_manager, PositionManager)
        assert self.position_manager._positions == {}
        assert self.position_manager._position_history == []

    def test_create_position(self):
        """Test creating a new position."""
        position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        assert isinstance(position, Position)
        assert position.symbol == "BTC-USD"
        assert position.side == "LONG"
        assert position.size == Decimal("0.5")
        assert position.entry_price == Decimal("50000.00")
        assert position.unrealized_pnl == Decimal(0)
        assert position.realized_pnl == Decimal(0)

    def test_get_position(self):
        """Test retrieving a position."""
        position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        retrieved_position = self.position_manager.get_position("BTC-USD")
        assert retrieved_position == position

    def test_get_nonexistent_position(self):
        """Test retrieving a non-existent position."""
        position = self.position_manager.get_position("ETH-USD")
        assert position is None

    def test_update_position_size(self):
        """Test updating position size."""
        position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        # Increase position size
        updated_position = self.position_manager.update_position_size(
            symbol="BTC-USD", size_change=Decimal("0.3"), price=Decimal("51000.00")
        )

        assert updated_position.size == Decimal("0.8")
        # Entry price should be weighted average
        # (0.5 * 50000 + 0.3 * 51000) / 0.8 = 50375
        expected_entry = (
            Decimal("0.5") * Decimal(50000) + Decimal("0.3") * Decimal(51000)
        ) / Decimal("0.8")
        assert updated_position.entry_price == expected_entry

    def test_update_position_size_reduction(self):
        """Test reducing position size."""
        position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
        )

        # Reduce position size by selling
        updated_position = self.position_manager.update_position_size(
            symbol="BTC-USD", size_change=Decimal("-0.4"), price=Decimal("52000.00")
        )

        assert updated_position.size == Decimal("0.6")
        # Entry price should remain the same for reductions
        assert updated_position.entry_price == Decimal("50000.00")
        # Should have realized P&L: 0.4 * (52000 - 50000) = 800
        assert updated_position.realized_pnl == Decimal("800.00")

    def test_close_position(self):
        """Test closing a position."""
        position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        # Close position at profit
        closed_position = self.position_manager.close_position(
            symbol="BTC-USD", exit_price=Decimal("55000.00")
        )

        assert closed_position.side == "FLAT"
        assert closed_position.size == Decimal(0)
        # Realized P&L: 0.5 * (55000 - 50000) = 2500
        assert closed_position.realized_pnl == Decimal("2500.00")

        # Position should be removed from active positions
        assert self.position_manager.get_position("BTC-USD") is None

    def test_close_nonexistent_position(self):
        """Test closing a non-existent position."""
        closed_position = self.position_manager.close_position(
            symbol="ETH-USD", exit_price=Decimal("3000.00")
        )
        assert closed_position is None

    def test_calculate_unrealized_pnl_long(self):
        """Test unrealized P&L calculation for long position."""
        position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        # Calculate P&L at higher price
        pnl = self.position_manager.calculate_unrealized_pnl(
            symbol="BTC-USD", current_price=Decimal("52000.00")
        )

        # P&L: 0.5 * (52000 - 50000) = 1000
        assert pnl == Decimal("1000.00")

    def test_calculate_unrealized_pnl_short(self):
        """Test unrealized P&L calculation for short position."""
        position = self.position_manager.create_position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
        )

        # Calculate P&L at lower price (profit for short)
        pnl = self.position_manager.calculate_unrealized_pnl(
            symbol="ETH-USD", current_price=Decimal("2800.00")
        )

        # P&L: 2.0 * (3000 - 2800) = 400
        assert pnl == Decimal("400.00")

    def test_calculate_unrealized_pnl_nonexistent(self):
        """Test unrealized P&L calculation for non-existent position."""
        pnl = self.position_manager.calculate_unrealized_pnl(
            symbol="DOGE-USD", current_price=Decimal("0.50")
        )
        assert pnl == Decimal(0)

    def test_update_unrealized_pnl(self):
        """Test updating unrealized P&L for positions."""
        # Create multiple positions
        btc_position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        eth_position = self.position_manager.create_position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
        )

        # Update with current prices
        current_prices = {"BTC-USD": Decimal("52000.00"), "ETH-USD": Decimal("2900.00")}

        self.position_manager.update_unrealized_pnl(current_prices)

        # Check updated P&L
        updated_btc = self.position_manager.get_position("BTC-USD")
        updated_eth = self.position_manager.get_position("ETH-USD")

        assert updated_btc.unrealized_pnl == Decimal("1000.00")  # 0.5 * (52000 - 50000)
        assert updated_eth.unrealized_pnl == Decimal("200.00")  # 2.0 * (3000 - 2900)

    def test_get_all_positions(self):
        """Test retrieving all positions."""
        # Create multiple positions
        btc_position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        eth_position = self.position_manager.create_position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
        )

        all_positions = self.position_manager.get_all_positions()
        assert len(all_positions) == 2
        assert btc_position in all_positions
        assert eth_position in all_positions

    def test_get_positions_by_side(self):
        """Test retrieving positions by side."""
        # Create positions with different sides
        btc_long = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        eth_short = self.position_manager.create_position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
        )

        doge_long = self.position_manager.create_position(
            symbol="DOGE-USD",
            side="LONG",
            size=Decimal("1000.0"),
            entry_price=Decimal("0.25"),
        )

        long_positions = self.position_manager.get_positions_by_side("LONG")
        short_positions = self.position_manager.get_positions_by_side("SHORT")

        assert len(long_positions) == 2
        assert len(short_positions) == 1
        assert btc_long in long_positions
        assert doge_long in long_positions
        assert eth_short in short_positions

    def test_get_total_exposure(self):
        """Test calculating total exposure."""
        # Create positions
        self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        self.position_manager.create_position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
        )

        total_exposure = self.position_manager.get_total_exposure()

        # Total exposure: (0.5 * 50000) + (2.0 * 3000) = 25000 + 6000 = 31000
        assert total_exposure == Decimal("31000.00")

    def test_get_net_position_value(self):
        """Test calculating net position value."""
        # Create long and short positions
        self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        self.position_manager.create_position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
        )

        current_prices = {"BTC-USD": Decimal("52000.00"), "ETH-USD": Decimal("2900.00")}

        net_value = self.position_manager.get_net_position_value(current_prices)

        # BTC: 0.5 * 52000 = 26000 (long)
        # ETH: -(2.0 * 2900) = -5800 (short, negative)
        # Net: 26000 - 5800 = 20200
        assert net_value == Decimal("20200.00")

    def test_get_position_count(self):
        """Test getting position count."""
        assert self.position_manager.get_position_count() == 0

        # Create positions
        self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        self.position_manager.create_position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
        )

        assert self.position_manager.get_position_count() == 2

    def test_position_history_tracking(self):
        """Test position history tracking."""
        # Create and close position
        position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        closed_position = self.position_manager.close_position(
            symbol="BTC-USD", exit_price=Decimal("55000.00")
        )

        # Check history
        history = self.position_manager.get_position_history()
        assert len(history) >= 1
        assert any(pos.symbol == "BTC-USD" for pos in history)

    def test_position_validation(self):
        """Test position validation."""
        # Test invalid size
        with pytest.raises(PositionValidationError):
            self.position_manager.create_position(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal(0),  # Invalid size
                entry_price=Decimal("50000.00"),
            )

        # Test invalid price
        with pytest.raises(PositionValidationError):
            self.position_manager.create_position(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("0.5"),
                entry_price=Decimal(0),  # Invalid price
            )

    def test_position_risk_metrics(self):
        """Test position risk metrics calculation."""
        # Create positions
        self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        self.position_manager.create_position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
        )

        # Update with current prices
        current_prices = {"BTC-USD": Decimal("52000.00"), "ETH-USD": Decimal("2900.00")}

        self.position_manager.update_unrealized_pnl(current_prices)

        risk_metrics = self.position_manager.calculate_risk_metrics(
            total_balance=Decimal("100000.00")
        )

        assert "total_exposure" in risk_metrics
        assert "portfolio_risk_pct" in risk_metrics
        assert "largest_position_pct" in risk_metrics
        assert "long_exposure" in risk_metrics
        assert "short_exposure" in risk_metrics
        assert "net_exposure" in risk_metrics

    def test_position_persistence(self):
        """Test position state persistence."""
        # Create position
        position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
        )

        # Save state
        self.position_manager.save_state()

        # Create new position manager
        # Create a new mock for the new position manager instance
        new_mock_paper_trading = Mock()
        new_mock_paper_trading.get_account_status.return_value = {
            "equity": 10000.0,
            "balance": 10000.0,
            "open_positions": 0,
            "current_balance": 10000.0,
            "margin_used": 0.0,
            "margin_available": 10000.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
        }

        with patch(
            "bot.position_manager.PaperTradingAccount",
            return_value=new_mock_paper_trading,
        ):
            new_position_manager = PositionManager(data_dir=self.temp_dir)
            new_position_manager.load_state()

        # Position should be restored
        restored_position = new_position_manager.get_position("BTC-USD")
        assert restored_position is not None
        assert restored_position.symbol == position.symbol
        assert restored_position.size == position.size
        assert restored_position.entry_price == position.entry_price

    def test_opposite_side_position_netting(self):
        """Test handling opposite side positions (netting)."""
        # Create long position
        long_position = self.position_manager.create_position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
        )

        # Create opposing short (should net out)
        self.position_manager.update_position_size(
            symbol="BTC-USD",
            size_change=Decimal("-0.6"),  # Reduce by 0.6
            price=Decimal("52000.00"),
        )

        updated_position = self.position_manager.get_position("BTC-USD")
        assert updated_position.size == Decimal("0.4")  # 1.0 - 0.6
        assert updated_position.side == "LONG"  # Still long, just smaller


if __name__ == "__main__":
    unittest.main()
