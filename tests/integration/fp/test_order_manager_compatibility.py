"""
Integration tests for order manager compatibility.

Tests that the functional order management system can be used as a drop-in
replacement for the original imperative order manager through the compatibility adapter.
"""

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from bot.fp.order_manager_adapter import CompatibilityOrderManager
from bot.trading_types import OrderStatus


class TestOrderManagerCompatibility:
    """Test that the functional order manager maintains full API compatibility."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def order_manager(self, temp_dir):
        """Create compatibility order manager for testing."""
        return CompatibilityOrderManager(temp_dir / "orders")

    @pytest.fixture
    def settings_mock(self):
        """Mock settings for testing."""
        with patch("bot.fp.order_manager_adapter.settings") as mock_settings:
            mock_settings.trading.order_timeout_seconds = 300
            yield mock_settings

    def test_create_order_api_compatibility(self, order_manager, settings_mock):
        """Test that create_order API is fully compatible."""
        # Create order using legacy API
        order = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        # Verify legacy order format
        assert hasattr(order, "id")
        assert hasattr(order, "symbol")
        assert hasattr(order, "side")
        assert hasattr(order, "type")
        assert hasattr(order, "quantity")
        assert hasattr(order, "price")
        assert hasattr(order, "status")
        assert hasattr(order, "timestamp")
        assert hasattr(order, "filled_quantity")

        # Verify order properties
        assert order.symbol == "BTC-USD"
        assert order.side == "BUY"
        assert order.type == "LIMIT"
        assert order.quantity == Decimal("1.0")
        assert order.price == Decimal("50000.0")
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == Decimal(0)

    def test_get_order_api_compatibility(self, order_manager, settings_mock):
        """Test that get_order API is fully compatible."""
        # Create order
        order = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        # Retrieve order using legacy API
        retrieved_order = order_manager.get_order(order.id)

        assert retrieved_order is not None
        assert retrieved_order.id == order.id
        assert retrieved_order.symbol == order.symbol
        assert retrieved_order.side == order.side

    def test_update_order_status_api_compatibility(self, order_manager, settings_mock):
        """Test that update_order_status API is fully compatible."""
        # Create order
        order = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        # Update status using legacy API
        updated_order = order_manager.update_order_status(
            order.id,
            OrderStatus.OPEN,
        )

        assert updated_order is not None
        assert updated_order.status == OrderStatus.OPEN
        assert updated_order.id == order.id

    def test_process_fill_api_compatibility(self, order_manager, settings_mock):
        """Test that fill processing through update_order_status is compatible."""
        # Create and open order
        order = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        order_manager.update_order_status(order.id, OrderStatus.OPEN)

        # Process partial fill using legacy API
        partially_filled = order_manager.update_order_status(
            order.id,
            OrderStatus.OPEN,  # Status stays open for partial fill
            filled_quantity=Decimal("0.5"),
            _fill_price=Decimal("50100.0"),
        )

        assert partially_filled.filled_quantity == Decimal("0.5")

        # Complete the fill
        fully_filled = order_manager.update_order_status(
            order.id,
            OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            _fill_price=Decimal("50200.0"),
        )

        assert fully_filled.status == OrderStatus.FILLED
        assert fully_filled.filled_quantity == Decimal("1.0")

    def test_cancel_order_api_compatibility(self, order_manager, settings_mock):
        """Test that cancel_order API is fully compatible."""
        # Create order
        order = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        # Cancel using legacy API
        success = order_manager.cancel_order(order.id)

        assert success is True

        # Verify order is cancelled
        cancelled_order = order_manager.get_order(order.id)
        assert cancelled_order.status == OrderStatus.CANCELLED

    def test_get_active_orders_api_compatibility(self, order_manager, settings_mock):
        """Test that get_active_orders API is fully compatible."""
        # Create multiple orders
        order1 = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        order2 = order_manager.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("2.0"),
            price=Decimal("3000.0"),
        )

        # Get all active orders
        active_orders = order_manager.get_active_orders()
        assert len(active_orders) == 2

        # Get active orders by symbol
        btc_orders = order_manager.get_active_orders("BTC-USD")
        assert len(btc_orders) == 1
        assert btc_orders[0].symbol == "BTC-USD"

    def test_get_orders_by_status_api_compatibility(self, order_manager, settings_mock):
        """Test that get_orders_by_status API is fully compatible."""
        # Create orders with different statuses
        pending_order = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        open_order = order_manager.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("2.0"),
            price=Decimal("3000.0"),
        )
        order_manager.update_order_status(open_order.id, OrderStatus.OPEN)

        # Get orders by status using legacy API
        pending_orders = order_manager.get_orders_by_status(OrderStatus.PENDING)
        assert len(pending_orders) == 1
        assert pending_orders[0].id == pending_order.id

        open_orders = order_manager.get_orders_by_status(OrderStatus.OPEN)
        assert len(open_orders) == 1
        assert open_orders[0].id == open_order.id

    def test_cancel_all_orders_api_compatibility(self, order_manager, settings_mock):
        """Test that cancel_all_orders API is fully compatible."""
        # Create multiple orders
        order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        order_manager.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("2.0"),
            price=Decimal("3000.0"),
        )

        # Cancel all orders using legacy API
        cancelled_count = order_manager.cancel_all_orders()

        assert cancelled_count == 2

        # Verify no active orders remain
        active_orders = order_manager.get_active_orders()
        assert len(active_orders) == 0

    def test_order_statistics_api_compatibility(self, order_manager, settings_mock):
        """Test that get_order_statistics API is fully compatible."""
        # Create and fill an order
        order = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        order_manager.update_order_status(order.id, OrderStatus.OPEN)
        order_manager.update_order_status(
            order.id,
            OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
        )

        # Get statistics using legacy API
        stats = order_manager.get_order_statistics()

        # Verify expected legacy statistics format
        assert "total_orders" in stats
        assert "filled_orders" in stats
        assert "cancelled_orders" in stats
        assert "rejected_orders" in stats
        assert "failed_orders" in stats
        assert "pending_orders" in stats
        assert "fill_rate_pct" in stats
        assert "avg_fill_time_seconds" in stats
        assert "time_window_hours" in stats
        assert "symbol" in stats

        assert stats["total_orders"] >= 1
        assert stats["filled_orders"] >= 1

    def test_callback_registration_api_compatibility(
        self, order_manager, settings_mock
    ):
        """Test that callback registration API is fully compatible."""
        callback_called = False
        callback_order_id = None
        callback_event = None

        def test_callback(order_id: str, event: str):
            nonlocal callback_called, callback_order_id, callback_event
            callback_called = True
            callback_order_id = order_id
            callback_event = event

        # Create order
        order = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        # Register callback using legacy API
        order_manager.register_callback(order.id, test_callback)

        # Update order status to trigger callback
        order_manager.update_order_status(order.id, OrderStatus.OPEN)

        assert callback_called is True
        assert callback_order_id == order.id
        assert callback_event == "SUBMITTED"

    @pytest.mark.asyncio
    async def test_async_lifecycle_api_compatibility(
        self, order_manager, settings_mock
    ):
        """Test that async start/stop API is fully compatible."""
        # Test async start/stop using legacy API
        await order_manager.start()
        assert order_manager._running is True

        await order_manager.stop()
        assert order_manager._running is False

    def test_state_persistence_api_compatibility(
        self, order_manager, temp_dir, settings_mock
    ):
        """Test that state persistence is compatible with legacy format."""
        # Create order
        order = order_manager.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        # Verify state files are created in expected locations
        assert order_manager.orders_file.exists()

        # Create new order manager with same directory
        new_manager = CompatibilityOrderManager(temp_dir / "orders")

        # Should load existing state
        loaded_order = new_manager.get_order(order.id)
        assert loaded_order is not None
        assert loaded_order.id == order.id
        assert loaded_order.symbol == order.symbol

    def test_enhanced_functionality_access(self, order_manager, settings_mock):
        """Test that enhanced functional features are accessible when needed."""
        # Access the underlying functional manager for advanced features
        functional_manager = order_manager._functional_manager

        # Verify it has functional capabilities
        assert hasattr(functional_manager, "state")
        assert hasattr(functional_manager.state, "get_statistics")

        # Test advanced analytics through functional manager
        stats = functional_manager.state.get_statistics()
        assert hasattr(stats, "total_orders")
        assert hasattr(stats, "fill_rate_pct")

    def test_error_handling_compatibility(self, order_manager, settings_mock):
        """Test that error handling behaves like legacy system."""
        # Test invalid order creation
        with pytest.raises(ValueError):
            order_manager.create_order(
                symbol="INVALID-SYMBOL",
                side="BUY",
                order_type="LIMIT",
                quantity=Decimal("1.0"),
                price=Decimal("50000.0"),
            )

        # Test cancelling non-existent order
        success = order_manager.cancel_order("non-existent-id")
        assert success is False

        # Test getting non-existent order
        order = order_manager.get_order("non-existent-id")
        assert order is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
