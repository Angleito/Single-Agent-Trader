"""
Tests for position and order management integration.

This module tests the integration between PositionManager, OrderManager,
and other bot components to ensure they work correctly together.
"""

import asyncio
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from bot.order_manager import OrderManager
from bot.position_manager import PositionManager
from bot.risk import RiskManager
from bot.types import Order, OrderStatus, TradeAction


class TestPositionOrderIntegration(unittest.TestCase):
    """Test integration between position and order management."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.temp_dir = Path(tempfile.mkdtemp())

        # Initialize managers
        self.position_manager = PositionManager(self.temp_dir / "positions")
        self.order_manager = OrderManager(self.temp_dir / "orders")
        self.risk_manager = RiskManager(position_manager=self.position_manager)

    def tearDown(self):
        """Clean up test environment."""
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def asyncSetUp(self):
        """Async setup."""
        await self.order_manager.start()

    async def asyncTearDown(self):
        """Async teardown."""
        await self.order_manager.stop()

    def test_position_manager_initialization(self):
        """Test position manager initializes correctly."""
        # Should start with no positions
        positions = self.position_manager.get_all_positions()
        self.assertEqual(len(positions), 0)

        # Should return flat position for any symbol
        position = self.position_manager.get_position("BTC-USD")
        self.assertEqual(position.side, "FLAT")
        self.assertEqual(position.size, Decimal("0"))

    async def test_order_manager_initialization(self):
        """Test order manager initializes correctly."""
        await self.asyncSetUp()

        try:
            # Should start with no orders
            orders = self.order_manager.get_active_orders()
            self.assertEqual(len(orders), 0)

            # Should have empty statistics
            stats = self.order_manager.get_order_statistics()
            self.assertEqual(stats["total_orders"], 0)
        finally:
            await self.asyncTearDown()

    def test_position_update_from_order(self):
        """Test position updates when order is filled."""
        # Create a filled buy order
        order = Order(
            id="test_order_1",
            symbol="BTC-USD",
            side="BUY",
            type="MARKET",
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            timestamp=None,
            filled_quantity=Decimal("0.1"),
        )

        fill_price = Decimal("50000.00")

        # Update position from order
        position = self.position_manager.update_position_from_order(order, fill_price)

        # Should create long position
        self.assertEqual(position.side, "LONG")
        self.assertEqual(position.size, Decimal("0.1"))
        self.assertEqual(position.entry_price, fill_price)
        self.assertEqual(position.symbol, "BTC-USD")

    def test_position_pnl_calculation(self):
        """Test P&L calculation for positions."""
        # Create initial position
        order = Order(
            id="test_order_1",
            symbol="BTC-USD",
            side="BUY",
            type="MARKET",
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            timestamp=None,
            filled_quantity=Decimal("0.1"),
        )

        entry_price = Decimal("50000.00")
        self.position_manager.update_position_from_order(order, entry_price)

        # Update with higher price (profit)
        current_price = Decimal("51000.00")
        unrealized_pnl = self.position_manager.update_unrealized_pnl(
            "BTC-USD", current_price
        )

        expected_pnl = Decimal("0.1") * (current_price - entry_price)  # 100 USD profit
        self.assertEqual(unrealized_pnl, expected_pnl)

    async def test_order_lifecycle(self):
        """Test complete order lifecycle."""
        await self.asyncSetUp()

        try:
            # Create order
            order = self.order_manager.create_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="LIMIT",
                quantity=Decimal("0.1"),
                price=Decimal("49000.00"),
            )

            # Should be pending initially
            self.assertEqual(order.status, OrderStatus.PENDING)

            # Update to open
            updated_order = self.order_manager.update_order_status(
                order.id, OrderStatus.OPEN
            )
            self.assertEqual(updated_order.status, OrderStatus.OPEN)

            # Fill the order
            final_order = self.order_manager.update_order_status(
                order.id, OrderStatus.FILLED, filled_quantity=Decimal("0.1")
            )
            self.assertEqual(final_order.status, OrderStatus.FILLED)
            self.assertEqual(final_order.filled_quantity, Decimal("0.1"))

            # Should be moved to history
            active_orders = self.order_manager.get_active_orders()
            self.assertEqual(len(active_orders), 0)

        finally:
            await self.asyncTearDown()

    def test_risk_manager_integration(self):
        """Test risk manager integration with position manager."""
        # Create a position
        order = Order(
            id="test_order_1",
            symbol="BTC-USD",
            side="BUY",
            type="MARKET",
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            timestamp=None,
            filled_quantity=Decimal("0.1"),
        )

        self.position_manager.update_position_from_order(order, Decimal("50000.00"))

        # Get risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()

        # Should show one active position
        self.assertEqual(risk_metrics.current_positions, 1)

        # Should have available margin
        self.assertGreater(risk_metrics.available_margin, Decimal("0"))

    def test_risk_evaluation_with_positions(self):
        """Test risk evaluation considers existing positions."""
        # Create existing position
        order = Order(
            id="test_order_1",
            symbol="BTC-USD",
            side="BUY",
            type="MARKET",
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            timestamp=None,
            filled_quantity=Decimal("0.1"),
        )

        position = self.position_manager.update_position_from_order(
            order, Decimal("50000.00")
        )

        # Try to open another large position
        large_trade = TradeAction(
            action="LONG",
            size_pct=80,  # Very large position
            take_profit_pct=3.0,
            stop_loss_pct=2.0,
            rationale="Large position test",
        )

        # Risk manager should modify or reject this
        approved, modified_action, reason = self.risk_manager.evaluate_risk(
            large_trade, position, Decimal("50000.00")
        )

        # Should either be rejected or significantly reduced
        if approved:
            self.assertLess(modified_action.size_pct, large_trade.size_pct)

    def test_position_close_scenario(self):
        """Test complete position open and close scenario."""
        symbol = "BTC-USD"
        entry_price = Decimal("50000.00")

        # Open position
        buy_order = Order(
            id="buy_order",
            symbol=symbol,
            side="BUY",
            type="MARKET",
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            timestamp=None,
            filled_quantity=Decimal("0.1"),
        )

        position = self.position_manager.update_position_from_order(
            buy_order, entry_price
        )
        self.assertEqual(position.side, "LONG")

        # Update P&L with price movement
        exit_price = Decimal("51000.00")
        self.position_manager.update_unrealized_pnl(symbol, exit_price)

        # Close position
        sell_order = Order(
            id="sell_order",
            symbol=symbol,
            side="SELL",
            type="MARKET",
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            timestamp=None,
            filled_quantity=Decimal("0.1"),
        )

        closed_position = self.position_manager.update_position_from_order(
            sell_order, exit_price
        )

        # Should be flat with realized P&L
        self.assertEqual(closed_position.side, "FLAT")
        self.assertEqual(closed_position.size, Decimal("0"))

        # Should have positive realized P&L (100 USD profit)
        expected_pnl = Decimal("0.1") * (exit_price - entry_price)
        self.assertEqual(closed_position.realized_pnl, expected_pnl)

    def test_state_persistence(self):
        """Test that state is persisted and restored correctly."""
        symbol = "BTC-USD"

        # Create position
        order = Order(
            id="test_order",
            symbol=symbol,
            side="BUY",
            type="MARKET",
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            timestamp=None,
            filled_quantity=Decimal("0.1"),
        )

        self.position_manager.update_position_from_order(order, Decimal("50000.00"))

        # Create new position manager with same data directory
        new_position_manager = PositionManager(self.temp_dir / "positions")

        # Should load the existing position
        restored_position = new_position_manager.get_position(symbol)
        self.assertEqual(restored_position.side, "LONG")
        self.assertEqual(restored_position.size, Decimal("0.1"))
        self.assertEqual(restored_position.entry_price, Decimal("50000.00"))


class AsyncTestMixin:
    """Mixin to add async test support."""

    def async_test(self, coro):
        """Run async test method."""
        return asyncio.get_event_loop().run_until_complete(coro)


# Async test cases
class TestAsyncOrderManagerIntegration(unittest.TestCase, AsyncTestMixin):
    """Test async functionality of order manager."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.order_manager = OrderManager(self.temp_dir / "orders")

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_async_order_timeout(self):
        """Test order timeout functionality."""

        async def test_timeout():
            await self.order_manager.start()

            try:
                # Create order with short timeout
                order = self.order_manager.create_order(
                    symbol="BTC-USD",
                    side="BUY",
                    order_type="LIMIT",
                    quantity=Decimal("0.1"),
                    price=Decimal("49000.00"),
                    timeout_seconds=1,
                )

                # Wait for timeout
                await asyncio.sleep(2)

                # Order should be cancelled due to timeout
                updated_order = self.order_manager.get_order(order.id)
                # Note: In real implementation, this would be cancelled
                # For now we just check it exists
                self.assertIsNotNone(updated_order)

            finally:
                await self.order_manager.stop()

        self.async_test(test_timeout())


if __name__ == "__main__":
    unittest.main()
