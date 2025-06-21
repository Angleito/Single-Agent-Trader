"""
Unit tests for the order management system.

Tests the order lifecycle, tracking, fill monitoring, timeout management,
and state persistence functionality.
"""

import asyncio
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from bot.order_manager import OrderEvent, OrderManager
from bot.trading_types import Order, OrderStatus


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
        assert self.order_manager._orders == {}
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
        assert order.id in self.order_manager._orders

    def test_create_market_order(self):
        """Test market order creation."""
        order = self.order_manager.create_order(
            symbol="ETH-USD", side="SELL", order_type="MARKET", quantity=Decimal("2.0")
        )

        assert order.order_type == "MARKET"
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

        assert order.timeout_seconds == 300
        assert order.expires_at is not None

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
            fill_price=Decimal("50100.00"),
            fill_quantity=Decimal("0.1"),
        )

        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.fill_price == Decimal("50100.00")
        assert updated_order.filled_quantity == Decimal("0.1")
        assert updated_order.updated_at > order.created_at

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

        cancelled_order = self.order_manager.cancel_order(order.id)

        assert cancelled_order.status == OrderStatus.CANCELLED
        assert cancelled_order.id not in self.order_manager._orders

    def test_cancel_nonexistent_order(self):
        """Test cancelling a non-existent order."""
        cancelled_order = self.order_manager.cancel_order("non-existent-id")
        assert cancelled_order is None

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
            fill_price=Decimal("50100.00"),
            fill_quantity=Decimal("0.3"),
        )

        assert updated_order.status == OrderStatus.PARTIALLY_FILLED
        assert updated_order.filled_quantity == Decimal("0.3")
        assert updated_order.remaining_quantity == Decimal("0.7")

        # Second partial fill
        updated_order = self.order_manager.update_order_status(
            order.id,
            OrderStatus.PARTIALLY_FILLED,
            fill_price=Decimal("50200.00"),
            fill_quantity=Decimal("0.4"),  # Additional fill
        )

        assert updated_order.filled_quantity == Decimal("0.7")  # Cumulative
        assert updated_order.remaining_quantity == Decimal("0.3")

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
            fill_price=Decimal("50100.00"),
            fill_quantity=Decimal("0.1"),
        )

        # Check history
        history = self.order_manager.get_order_history(order.id)
        assert len(history) >= 3  # Created, Submitted, Filled

        # Check events are in chronological order
        timestamps = [event["timestamp"] for event in history]
        assert timestamps == sorted(timestamps)

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

    @pytest.mark.asyncio
    async def test_order_timeout_monitoring(self):
        """Test order timeout monitoring."""
        # Create order with short timeout
        order = self.order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            timeout_seconds=0.1,  # Very short timeout
        )

        # Start monitoring
        await self.order_manager.start()

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Order should be expired
        expired_order = self.order_manager.get_order(order.id)
        assert expired_order is None or expired_order.status == OrderStatus.EXPIRED

        # Stop monitoring
        await self.order_manager.stop()

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
        self.order_manager.save_state()

        # Create new order manager with same data directory
        new_order_manager = OrderManager(data_dir=self.temp_dir)
        new_order_manager.load_state()

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
        assert stats["fill_rate"] == 1 / 3  # 1 filled out of 3

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

        # Manually set old timestamp
        old_order.created_at = datetime.now(UTC) - timedelta(days=8)
        self.order_manager.update_order_status(old_order.id, OrderStatus.FILLED)

        # Create recent order
        recent_order = self.order_manager.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("2.0"),
            price=Decimal("3000.00"),
        )

        # Clean up orders older than 7 days
        cleaned_count = self.order_manager.cleanup_old_orders(days=7)

        assert cleaned_count == 1
        assert self.order_manager.get_order(old_order.id) is None
        assert self.order_manager.get_order(recent_order.id) is not None


if __name__ == "__main__":
    unittest.main()
