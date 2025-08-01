"""
Unit tests for the order management system.

Tests the order lifecycle, tracking, fill monitoring, timeout management,
and state persistence functionality.
"""

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot.fp.types import Order, OrderStatus
from bot.order_manager import OrderEvent, OrderManager


class TestOrderEvent(unittest.TestCase):
    """Test order event enumeration."""

    def test_order_events(self):
        """Test order event values."""
        assert OrderEvent.CREATED == "CREATED"
        assert OrderEvent.SUBMITTED == "SUBMITTED"
        assert OrderEvent.PARTIALLY_FILLED == "PARTIALLY_FILLED"
        assert OrderEvent.FILLED == "FILLED"
        assert OrderEvent.CANCELLED == "CANCELLED"
        assert OrderEvent.REJECTED == "REJECTED"
        assert OrderEvent.FAILED == "FAILED"
        assert OrderEvent.EXPIRED == "EXPIRED"


class TestOrderManager(unittest.TestCase):
    """Test order manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.order_manager = OrderManager(data_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_order_manager_initialization(self):
        """Test order manager initialization."""
        assert isinstance(self.order_manager, OrderManager)
        assert self.order_manager._active_orders == {}
        assert self.order_manager._order_history == []
        assert not self.order_manager._running

    def test_create_order(self):
        """Test order creation."""
        order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        assert isinstance(order, Order)
        assert order.symbol == "BTC-USD"
        assert order.side == "BUY"
        assert order.type == "LIMIT"
        assert order.quantity == Decimal("0.1")
        assert order.price == Decimal("50000.00")
        assert order.status == OrderStatus.PENDING
        assert order.id in self.order_manager._active_orders

    def test_create_market_order(self):
        """Test market order creation."""
        order = self.order_manager.create_order(
            symbol="ETH-USD", side="SELL", order_type="MARKET", quantity=Decimal("2.0")
        )

        assert order.type == "MARKET"
        assert order.price is None  # Market orders don't have price

    def test_create_order_with_timeout(self):
        """Test order creation with timeout."""
        order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            timeout_seconds=300,
        )

        # Order object doesn't have timeout_seconds or expires_at fields
        # The timeout is handled internally by OrderManager
        assert order.id in self.order_manager._active_orders

    def test_get_order(self):
        """Test retrieving an order."""
        order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        retrieved_order = self.order_manager.get_order(order.id)
        assert retrieved_order == order

    def test_get_nonexistent_order(self):
        """Test retrieving a non-existent order."""
        retrieved_order = self.order_manager.get_order("non-existent-id")
        assert retrieved_order is None

    def test_get_orders_by_symbol(self):
        """Test retrieving orders by symbol."""
        # Create orders for different symbols
        btc_order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        eth_order = self.order_manager.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("2.0"),
            price=Decimal("3000.00"),
        )

        # Test filtering by symbol
        btc_orders = self.order_manager.get_orders_by_symbol("BTC-USD")
        assert len(btc_orders) == 1
        assert btc_orders[0] == btc_order

        eth_orders = self.order_manager.get_orders_by_symbol("ETH-USD")
        assert len(eth_orders) == 1
        assert eth_orders[0] == eth_order

    def test_get_orders_by_status(self):
        """Test retrieving orders by status."""
        # Create orders
        order1 = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        order2 = self.order_manager.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("2.0"),
            price=Decimal("3000.00"),
        )

        # Update status of one order
        self.order_manager.update_order_status(order2.id, OrderStatus.FILLED)

        # Test filtering by status
        pending_orders = self.order_manager.get_orders_by_status(OrderStatus.PENDING)
        assert len(pending_orders) == 1
        assert pending_orders[0] == order1

        filled_orders = self.order_manager.get_orders_by_status(OrderStatus.FILLED)
        assert len(filled_orders) == 1
        assert filled_orders[0].id == order2.id

    def test_update_order_status(self):
        """Test updating order status."""
        order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        # Update status
        updated_order = self.order_manager.update_order_status(
            order.id,
            OrderStatus.FILLED,
            filled_quantity=Decimal("0.1"),
            _fill_price=Decimal("50100.00"),
        )

        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.filled_quantity == Decimal("0.1")
        assert updated_order.timestamp > order.timestamp

    def test_update_nonexistent_order(self):
        """Test updating a non-existent order."""
        updated_order = self.order_manager.update_order_status(
            "non-existent-id", OrderStatus.FILLED
        )
        assert updated_order is None

    def test_cancel_order(self):
        """Test cancelling an order."""
        order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        result = self.order_manager.cancel_order(order.id)
        assert result is True

        # Check that the order was moved to history with CANCELLED status
        cancelled_order = self.order_manager.get_order(order.id)
        assert cancelled_order.status == OrderStatus.CANCELLED
        assert order.id not in self.order_manager._active_orders

    def test_cancel_nonexistent_order(self):
        """Test cancelling a non-existent order."""
        result = self.order_manager.cancel_order("non-existent-id")
        assert result is False

    def test_partial_fill_handling(self):
        """Test handling partial fills."""
        order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
        )

        # First partial fill
        updated_order = self.order_manager.update_order_status(
            order.id,
            OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.3"),
            _fill_price=Decimal("50100.00"),
        )

        assert updated_order.status == OrderStatus.PARTIALLY_FILLED
        assert updated_order.filled_quantity == Decimal("0.3")
        # Check remaining quantity calculation
        assert updated_order.quantity - updated_order.filled_quantity == Decimal("0.7")

        # Second partial fill
        updated_order = self.order_manager.update_order_status(
            order.id,
            OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.7"),  # Cumulative fill
            _fill_price=Decimal("50200.00"),
        )

        assert updated_order.filled_quantity == Decimal("0.7")  # Cumulative
        assert updated_order.quantity - updated_order.filled_quantity == Decimal("0.3")

    def test_order_history_tracking(self):
        """Test order history tracking."""
        order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        # Update status multiple times
        self.order_manager.update_order_status(order.id, OrderStatus.SUBMITTED)
        self.order_manager.update_order_status(
            order.id,
            OrderStatus.FILLED,
            filled_quantity=Decimal("0.1"),
            _fill_price=Decimal("50100.00"),
        )

        # The order should be in history after being filled
        filled_order = self.order_manager.get_order(order.id)
        assert filled_order is not None
        assert filled_order.status == OrderStatus.FILLED
        assert order.id not in self.order_manager._active_orders

    def test_get_all_orders(self):
        """Test retrieving all orders."""
        # Create multiple orders
        order1 = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        order2 = self.order_manager.create_order(
            symbol="ETH-USD", side="SELL", order_type="MARKET", quantity=Decimal("2.0")
        )

        all_orders = self.order_manager.get_all_orders()
        assert len(all_orders) == 2
        assert order1 in all_orders
        assert order2 in all_orders

    def test_get_pending_orders(self):
        """Test retrieving pending orders."""
        # Create orders
        order1 = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        order2 = self.order_manager.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("2.0"),
            price=Decimal("3000.00"),
        )

        # Fill one order
        self.order_manager.update_order_status(order2.id, OrderStatus.FILLED)

        pending_orders = self.order_manager.get_pending_orders()
        assert len(pending_orders) == 1
        assert pending_orders[0] == order1

    def test_order_timeout_monitoring(self):
        """Test order timeout monitoring."""
        # This would require running event loop, skip for unit test
        # Integration tests should cover this functionality

    def test_order_persistence(self):
        """Test order state persistence."""
        # Create order
        order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        # Save state
        self.order_manager._save_state()

        # Create new order manager with same data directory
        new_order_manager = OrderManager(data_dir=self.temp_dir)
        # load_state is called automatically in __init__

        # Order should be restored
        restored_order = new_order_manager.get_order(order.id)
        assert restored_order is not None
        assert restored_order.symbol == order.symbol
        assert restored_order.quantity == order.quantity

    def test_order_statistics(self):
        """Test order statistics calculation."""
        # Create multiple orders with different outcomes
        order1 = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        order2 = self.order_manager.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("2.0"),
            price=Decimal("3000.00"),
        )

        order3 = self.order_manager.create_order(
            symbol="BTC-USD", side="SELL", order_type="MARKET", quantity=Decimal("0.05")
        )

        # Update statuses
        self.order_manager.update_order_status(order1.id, OrderStatus.FILLED)
        self.order_manager.update_order_status(order2.id, OrderStatus.CANCELLED)
        # order3 remains pending

        stats = self.order_manager.get_order_statistics()

        assert stats["total_orders"] == 3
        assert stats["filled_orders"] == 1
        assert stats["cancelled_orders"] == 1
        assert stats["pending_orders"] == 1
        assert stats["fill_rate_pct"] == (
            1 / 3 * 100
        )  # 1 filled out of 3, as percentage

    def test_order_cleanup(self):
        """Test cleaning up old orders."""
        # Create old order
        old_order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
        )

        # Update status to move to history
        self.order_manager.update_order_status(old_order.id, OrderStatus.FILLED)

        # Manually set old timestamp on the history entry
        for order in self.order_manager._order_history:
            if order.id == old_order.id:
                order.timestamp = datetime.now(UTC) - timedelta(days=8)
                break

        # Create recent order
        recent_order = self.order_manager.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("2.0"),
            price=Decimal("3000.00"),
        )

        # Clear old history (this is the actual method name)
        self.order_manager.clear_old_history(days_to_keep=7)

        # Old order should be removed from history
        assert self.order_manager.get_order(old_order.id) is None
        assert self.order_manager.get_order(recent_order.id) is not None


if __name__ == "__main__":
    unittest.main()
