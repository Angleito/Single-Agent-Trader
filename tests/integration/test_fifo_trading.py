"""Integration test for FIFO trading functionality."""

import json
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.position_manager import PositionManager
from bot.trading_types import Order, OrderStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/fifo_test.log"),
    ],
)

logger = logging.getLogger(__name__)


class FIFOTradingTest:
    """Test FIFO trading functionality."""

    def __init__(self):
        """Initialize test environment."""
        self.test_dir = Path("/tmp/fifo_test_data")
        self.test_dir.mkdir(exist_ok=True)

        # Initialize position manager with FIFO enabled
        self.position_manager = PositionManager(data_dir=self.test_dir, use_fifo=True)

        logger.info("Initialized FIFO trading test environment")

    def create_test_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal, order_id: str
    ) -> Order:
        """Create a test order."""
        return Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity,
            price=price,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc),
            filled_quantity=quantity,
        )

    def test_basic_fifo_flow(self):
        """Test basic FIFO buy and sell flow."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 1: Basic FIFO Flow")
        logger.info("=" * 80)

        symbol = "BTC-USD"

        # Buy 1 BTC at $40,000
        order1 = self.create_test_order(
            symbol=symbol,
            side="BUY",
            quantity=Decimal("1"),
            price=Decimal("40000"),
            order_id="order_1",
        )
        position = self.position_manager.update_position_from_order(
            order1, order1.price
        )
        logger.info(
            "After Buy 1: Position = %s %s @ %s",
            position.side,
            position.size,
            position.entry_price,
        )

        # Buy 0.5 BTC at $42,000
        order2 = self.create_test_order(
            symbol=symbol,
            side="BUY",
            quantity=Decimal("0.5"),
            price=Decimal("42000"),
            order_id="order_2",
        )
        position = self.position_manager.update_position_from_order(
            order2, order2.price
        )
        logger.info(
            "After Buy 2: Position = %s %s @ %s",
            position.side,
            position.size,
            position.entry_price,
        )

        # Get tax lots report
        lots_report = self.position_manager.get_tax_lots_report(symbol)
        logger.info("\nTax Lots Report after buys:")
        logger.info(json.dumps(lots_report, indent=2))

        # Sell 0.75 BTC at $45,000 (should sell all of lot 1 first due to FIFO)
        order3 = self.create_test_order(
            symbol=symbol,
            side="SELL",
            quantity=Decimal("0.75"),
            price=Decimal("45000"),
            order_id="order_3",
        )
        position = self.position_manager.update_position_from_order(
            order3, order3.price
        )
        logger.info(
            "\nAfter Sell 1: Position = %s %s @ %s",
            position.side,
            position.size,
            position.entry_price,
        )
        logger.info("Realized P&L: $%s", position.realized_pnl)

        # Get updated tax lots report
        lots_report = self.position_manager.get_tax_lots_report(symbol)
        logger.info("\nTax Lots Report after first sell:")
        logger.info(json.dumps(lots_report, indent=2))

        # Sell remaining 0.75 BTC at $44,000
        order4 = self.create_test_order(
            symbol=symbol,
            side="SELL",
            quantity=Decimal("0.75"),
            price=Decimal("44000"),
            order_id="order_4",
        )
        position = self.position_manager.update_position_from_order(
            order4, order4.price
        )
        logger.info("\nAfter Sell 2: Position = %s %s", position.side, position.size)
        logger.info("Total Realized P&L: $%s", position.realized_pnl)

        # Final tax lots report
        lots_report = self.position_manager.get_tax_lots_report(symbol)
        logger.info("\nFinal Tax Lots Report:")
        logger.info(json.dumps(lots_report, indent=2))

        # Verify expected P&L
        # Lot 1: Bought 1 BTC @ $40,000, sold 0.75 @ $45,000 = $3,750 profit
        # Lot 2: Bought 0.5 BTC @ $42,000, sold 0.5 @ $44,000 = $1,000 profit
        # Total expected P&L: $4,750
        expected_pnl = Decimal("3750") + Decimal("1000")
        actual_pnl = self.position_manager.fifo_manager.get_realized_pnl(symbol)

        logger.info("\nExpected P&L: $%s", expected_pnl)
        logger.info("Actual P&L: $%s", actual_pnl)

        assert abs(actual_pnl - expected_pnl) < Decimal(
            "0.01"
        ), f"P&L mismatch: expected {expected_pnl}, got {actual_pnl}"
        logger.info("✅ Test 1 PASSED: Basic FIFO flow working correctly")

    def test_multiple_buys_partial_sells(self):
        """Test multiple buys with partial sells."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: Multiple Buys with Partial Sells")
        logger.info("=" * 80)

        symbol = "ETH-USD"

        # Buy orders at different prices
        buys = [
            (Decimal("2"), Decimal("2000"), "buy_1"),  # 2 ETH @ $2,000
            (Decimal("1.5"), Decimal("2100"), "buy_2"),  # 1.5 ETH @ $2,100
            (Decimal("3"), Decimal("2200"), "buy_3"),  # 3 ETH @ $2,200
        ]

        for quantity, price, order_id in buys:
            order = self.create_test_order(symbol, "BUY", quantity, price, order_id)
            position = self.position_manager.update_position_from_order(order, price)
            logger.info(
                "Buy %s ETH @ $%s: Total position = %s ETH",
                quantity,
                price,
                position.size,
            )

        # Show all lots
        lots_report = self.position_manager.get_tax_lots_report(symbol)
        logger.info("\nAll lots after buys:")
        logger.info(json.dumps(lots_report, indent=2))

        # Partial sell - should use FIFO
        sell_order = self.create_test_order(
            symbol=symbol,
            side="SELL",
            quantity=Decimal("4"),  # Sell 4 ETH
            price=Decimal("2500"),
            order_id="sell_1",
        )
        position = self.position_manager.update_position_from_order(
            sell_order, sell_order.price
        )

        logger.info("\nAfter selling 4 ETH @ $2,500:")
        logger.info("Remaining position: %s ETH", position.size)
        logger.info("Realized P&L: $%s", position.realized_pnl)

        # Show remaining lots
        lots_report = self.position_manager.get_tax_lots_report(symbol)
        logger.info("\nRemaining lots:")
        logger.info(json.dumps(lots_report, indent=2))

        # Verify FIFO logic
        # Should have sold: 2 ETH from lot 1, 1.5 ETH from lot 2, 0.5 ETH from lot 3
        # Remaining: 2.5 ETH from lot 3
        fifo_pos = self.position_manager.fifo_manager.get_fifo_position(symbol)
        assert (
            len(fifo_pos.lots) == 1
        ), f"Expected 1 lot remaining, got {len(fifo_pos.lots)}"
        assert fifo_pos.lots[0].remaining_quantity == Decimal(
            "2.5"
        ), "Incorrect remaining quantity"

        logger.info(
            "✅ Test 2 PASSED: Multiple buys with partial sells working correctly"
        )

    def test_position_closure_and_reopening(self):
        """Test closing a position completely and reopening."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: Position Closure and Reopening")
        logger.info("=" * 80)

        symbol = "SOL-USD"

        # Open position
        buy_order = self.create_test_order(
            symbol=symbol,
            side="BUY",
            quantity=Decimal("100"),
            price=Decimal("50"),
            order_id="sol_buy_1",
        )
        position = self.position_manager.update_position_from_order(
            buy_order, buy_order.price
        )
        logger.info(
            "Opened position: %s SOL @ $%s", position.size, position.entry_price
        )

        # Close position
        sell_order = self.create_test_order(
            symbol=symbol,
            side="SELL",
            quantity=Decimal("100"),
            price=Decimal("60"),
            order_id="sol_sell_1",
        )
        position = self.position_manager.update_position_from_order(
            sell_order, sell_order.price
        )
        logger.info("Closed position: P&L = $%s", position.realized_pnl)
        assert position.side == "FLAT", "Position should be FLAT"

        # Reopen position
        buy_order2 = self.create_test_order(
            symbol=symbol,
            side="BUY",
            quantity=Decimal("50"),
            price=Decimal("55"),
            order_id="sol_buy_2",
        )
        position = self.position_manager.update_position_from_order(
            buy_order2, buy_order2.price
        )
        logger.info(
            "Reopened position: %s SOL @ $%s", position.size, position.entry_price
        )

        # Check total realized P&L includes previous trade
        total_pnl = self.position_manager.fifo_manager.get_realized_pnl(symbol)
        expected_pnl = Decimal("1000")  # 100 * (60 - 50)
        assert (
            total_pnl == expected_pnl
        ), f"Expected total P&L ${expected_pnl}, got ${total_pnl}"

        logger.info(
            "✅ Test 3 PASSED: Position closure and reopening working correctly"
        )

    def test_fifo_vs_average_cost(self):
        """Compare FIFO accounting with average cost method."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 4: FIFO vs Average Cost Comparison")
        logger.info("=" * 80)

        # Create two position managers - one with FIFO, one without
        fifo_manager = PositionManager(data_dir=self.test_dir / "fifo", use_fifo=True)
        avg_manager = PositionManager(data_dir=self.test_dir / "avg", use_fifo=False)

        symbol = "MATIC-USD"

        # Make identical trades on both
        trades = [
            ("BUY", Decimal("1000"), Decimal("1.00")),
            ("BUY", Decimal("1000"), Decimal("1.20")),
            ("SELL", Decimal("1000"), Decimal("1.50")),
        ]

        for i, (side, quantity, price) in enumerate(trades):
            order = self.create_test_order(
                symbol, side, quantity, price, f"compare_{i}"
            )

            fifo_pos = fifo_manager.update_position_from_order(order, price)
            avg_pos = avg_manager.update_position_from_order(order, price)

            logger.info("\nAfter %s %s @ $%s:", side, quantity, price)
            logger.info(
                "  FIFO: Size=%s, Entry=$%s, P&L=$%s",
                fifo_pos.size,
                fifo_pos.entry_price,
                fifo_pos.realized_pnl,
            )
            logger.info(
                "  AVG:  Size=%s, Entry=$%s, P&L=$%s",
                avg_pos.size,
                avg_pos.entry_price,
                avg_pos.realized_pnl,
            )

        # FIFO should show $500 profit (sold first lot bought at $1.00)
        # Average cost would show $400 profit (sold at avg cost of $1.10)
        fifo_pnl = fifo_manager.fifo_manager.get_realized_pnl(symbol)
        logger.info("\nFIFO Realized P&L: $%s", fifo_pnl)

        assert fifo_pnl == Decimal("500"), f"Expected FIFO P&L $500, got ${fifo_pnl}"
        logger.info("✅ Test 4 PASSED: FIFO accounting working as expected")

    def run_all_tests(self):
        """Run all FIFO tests."""
        logger.info("\n" + "=" * 80)
        logger.info("Starting FIFO Trading Tests")
        logger.info("=" * 80)

        try:
            self.test_basic_fifo_flow()
            self.test_multiple_buys_partial_sells()
            self.test_position_closure_and_reopening()
            self.test_fifo_vs_average_cost()

            logger.info("\n" + "=" * 80)
            logger.info("✅ ALL TESTS PASSED!")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
            return False


def main():
    """Run FIFO trading tests."""
    test = FIFOTradingTest()
    success = test.run_all_tests()

    # Print log file location
    logger.info("\nTest logs saved to: /tmp/fifo_test.log")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
