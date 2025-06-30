"""Test single position enforcement with FIFO tracking."""

import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.position_manager import PositionManager
from bot.risk import RiskManager
from bot.trading_types import Order, OrderStatus, TradeAction
from bot.validator import TradeValidator

# Configure logging
temp_log_file = Path(tempfile.gettempdir()) / "single_position_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(temp_log_file),
    ],
)

logger = logging.getLogger(__name__)


class SinglePositionFIFOTest:
    """Test single position enforcement with FIFO tracking."""

    def __init__(self):
        """Initialize test environment."""
        self.test_dir = Path(tempfile.gettempdir()) / "single_position_test"
        self.test_dir.mkdir(exist_ok=True)

        # Initialize components
        self.position_manager = PositionManager(data_dir=self.test_dir, use_fifo=True)
        self.risk_manager = RiskManager(position_manager=self.position_manager)
        self.validator = TradeValidator()

        logger.info("Initialized single position FIFO test environment")

    def create_trade_action(self, action: str, size_pct: int = 10) -> TradeAction:
        """Create a test trade action."""
        return TradeAction(
            action=action,
            size_pct=size_pct if action not in ["HOLD", "CLOSE"] else 0,
            take_profit_pct=2.0 if action in ["LONG", "SHORT"] else 0,
            stop_loss_pct=1.5 if action in ["LONG", "SHORT"] else 0,
            rationale=f"Test {action} action",
        )

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
            timestamp=datetime.now(UTC),
            filled_quantity=quantity,
        )

    def test_single_position_enforcement(self):
        """Test that only one position can be open at a time."""
        logger.info("\n%s", "=" * 80)
        logger.info("TEST 1: Single Position Enforcement")
        logger.info("=" * 80)

        symbol = "BTC-USD"

        # Start with no position
        current_position = self.position_manager.get_position(symbol)
        logger.info("Initial position: %s", current_position.side)

        # Test 1: Open LONG position
        long_action = self.create_trade_action("LONG", 10)
        validated = self.validator.validate(long_action, current_position)
        approved, modified, reason = self.risk_manager.evaluate_risk(
            validated, current_position, Decimal(40000)
        )

        logger.info("\nAttempting LONG when flat:")
        logger.info("  Validated action: %s", validated.action)
        logger.info("  Risk approved: %s", approved)
        logger.info("  Reason: %s", reason)

        assert approved, "Should be able to open LONG when flat"
        assert modified.action == "LONG", "Action should remain LONG"

        # Execute the LONG trade
        long_order = self.create_test_order(
            symbol, "BUY", Decimal("0.1"), Decimal(40000), "long_1"
        )
        current_position = self.position_manager.update_position_from_order(
            long_order, long_order.price
        )
        logger.info(
            "Position after LONG: %s %s @ %s",
            current_position.side,
            current_position.size,
            current_position.entry_price,
        )

        # Test 2: Try to open another LONG while one exists
        long_action2 = self.create_trade_action("LONG", 10)
        validated2 = self.validator.validate(long_action2, current_position)
        approved2, modified2, reason2 = self.risk_manager.evaluate_risk(
            validated2, current_position, Decimal(41000)
        )

        logger.info("\nAttempting LONG when already LONG:")
        logger.info("  Validated action: %s", validated2.action)
        logger.info("  Risk approved: %s", approved2)
        logger.info("  Reason: %s", reason2)

        assert not approved2, "Should NOT be able to open another LONG"
        assert modified2.action == "HOLD", "Action should be changed to HOLD"
        assert (
            "one position" in reason2.lower()
        ), "Reason should mention single position rule"

        # Test 3: Try to open SHORT while LONG exists
        short_action = self.create_trade_action("SHORT", 10)
        validated3 = self.validator.validate(short_action, current_position)
        approved3, modified3, reason3 = self.risk_manager.evaluate_risk(
            validated3, current_position, Decimal(39000)
        )

        logger.info("\nAttempting SHORT when LONG:")
        logger.info("  Validated action: %s", validated3.action)
        logger.info("  Risk approved: %s", approved3)
        logger.info("  Reason: %s", reason3)

        assert not approved3, "Should NOT be able to open SHORT while LONG"
        assert modified3.action == "HOLD", "Action should be changed to HOLD"

        # Test 4: CLOSE position should be allowed
        close_action = self.create_trade_action("CLOSE")
        validated4 = self.validator.validate(close_action, current_position)
        approved4, modified4, reason4 = self.risk_manager.evaluate_risk(
            validated4, current_position, Decimal(42000)
        )

        logger.info("\nAttempting CLOSE when LONG:")
        logger.info("  Validated action: %s", validated4.action)
        logger.info("  Risk approved: %s", approved4)
        logger.info("  Reason: %s", reason4)

        assert approved4, "Should be able to CLOSE existing position"
        assert modified4.action == "CLOSE", "Action should remain CLOSE"

        # Execute the CLOSE
        close_order = self.create_test_order(
            symbol, "SELL", Decimal("0.1"), Decimal(42000), "close_1"
        )
        current_position = self.position_manager.update_position_from_order(
            close_order, close_order.price
        )
        logger.info(
            "Position after CLOSE: %s, P&L: $%s",
            current_position.side,
            current_position.realized_pnl,
        )

        # Test 5: Can open new position after closing
        short_action2 = self.create_trade_action("SHORT", 10)
        validated5 = self.validator.validate(short_action2, current_position)
        approved5, modified5, reason5 = self.risk_manager.evaluate_risk(
            validated5, current_position, Decimal(41500)
        )

        logger.info("\nAttempting SHORT when flat (after close):")
        logger.info("  Validated action: %s", validated5.action)
        logger.info("  Risk approved: %s", approved5)
        logger.info("  Reason: %s", reason5)

        assert approved5, "Should be able to open SHORT after closing previous position"
        assert modified5.action == "SHORT", "Action should remain SHORT"

        logger.info("\n✅ Test 1 PASSED: Single position enforcement working correctly")

    def test_fifo_with_single_position(self):
        """Test FIFO tracking with single position rule."""
        logger.info("\n%s", "=" * 80)
        logger.info("TEST 2: FIFO Tracking with Single Position")
        logger.info("=" * 80)

        symbol = "ETH-USD"

        # Open position
        buy_order = self.create_test_order(
            symbol, "BUY", Decimal(1), Decimal(2000), "eth_buy_1"
        )
        position = self.position_manager.update_position_from_order(
            buy_order, buy_order.price
        )
        logger.info(
            "Opened position: %s %s @ %s",
            position.side,
            position.size,
            position.entry_price,
        )

        # Get FIFO report
        fifo_report = self.position_manager.get_tax_lots_report(symbol)
        logger.info("\nFIFO Report after opening:")
        logger.info(json.dumps(fifo_report, indent=2))

        assert len(fifo_report["active_lots"]) == 1, "Should have exactly 1 lot"
        assert fifo_report["total_quantity"] == "1", "Total quantity should be 1"

        # Try to add to position (should fail due to single position rule)
        current_position = self.position_manager.get_position(symbol)
        add_action = self.create_trade_action("LONG", 10)
        validated = self.validator.validate(add_action, current_position)

        logger.info("\nValidator result when trying to add to position:")
        logger.info("  Input action: %s", add_action.action)
        logger.info("  Validated action: %s", validated.action)
        logger.info("  Rationale: %s", validated.rationale)

        assert (
            validated.action == "HOLD"
        ), "Validator should change LONG to HOLD when position exists"

        # Close position
        sell_order = self.create_test_order(
            symbol, "SELL", Decimal(1), Decimal(2200), "eth_sell_1"
        )
        position = self.position_manager.update_position_from_order(
            sell_order, sell_order.price
        )

        logger.info("\nClosed position: P&L = $%s", position.realized_pnl)

        # Final FIFO report
        final_report = self.position_manager.get_tax_lots_report(symbol)
        logger.info("\nFinal FIFO Report:")
        logger.info(json.dumps(final_report, indent=2))

        assert final_report["side"] == "FLAT", "Position should be flat"
        assert len(final_report["active_lots"]) == 0, "Should have no active lots"
        assert float(final_report["total_realized_pnl"]) == 200.0, "P&L should be $200"

        logger.info(
            "\n✅ Test 2 PASSED: FIFO tracking with single position working correctly"
        )

    def test_position_reversal_workflow(self):
        """Test the workflow for reversing a position."""
        logger.info("\n%s", "=" * 80)
        logger.info("TEST 3: Position Reversal Workflow")
        logger.info("=" * 80)

        symbol = "BTC-USD"

        # Step 1: Open LONG position
        buy_order = self.create_test_order(
            symbol, "BUY", Decimal("0.5"), Decimal(45000), "btc_long"
        )
        position = self.position_manager.update_position_from_order(
            buy_order, buy_order.price
        )
        logger.info(
            "Step 1 - Opened LONG: %s @ %s", position.size, position.entry_price
        )

        # Step 2: Try to go SHORT directly (should fail)
        current_position = self.position_manager.get_position(symbol)
        short_action = self.create_trade_action("SHORT", 10)
        validated = self.validator.validate(short_action, current_position)
        approved, modified, reason = self.risk_manager.evaluate_risk(
            validated, current_position, Decimal(44000)
        )

        logger.info("\nStep 2 - Trying to SHORT while LONG:")
        logger.info("  Approved: %s", approved)
        logger.info("  Modified action: %s", modified.action)
        logger.info("  Reason: %s", reason)

        assert not approved, "Direct reversal should not be allowed"

        # Step 3: Close LONG position first
        close_order = self.create_test_order(
            symbol, "SELL", Decimal("0.5"), Decimal(44000), "btc_close"
        )
        position = self.position_manager.update_position_from_order(
            close_order, close_order.price
        )
        logger.info("\nStep 3 - Closed LONG: P&L = $%s", position.realized_pnl)

        # Step 4: Now can open SHORT position
        current_position = self.position_manager.get_position(symbol)
        short_action2 = self.create_trade_action("SHORT", 10)
        validated2 = self.validator.validate(short_action2, current_position)
        approved2, modified2, reason2 = self.risk_manager.evaluate_risk(
            validated2, current_position, Decimal(44000)
        )

        logger.info("\nStep 4 - Trying to SHORT after closing:")
        logger.info("  Approved: %s", approved2)
        logger.info("  Action: %s", modified2.action)

        assert approved2, "Should be able to SHORT after closing LONG"
        assert modified2.action == "SHORT", "Action should remain SHORT"

        logger.info("\n✅ Test 3 PASSED: Position reversal workflow enforced correctly")

    def run_all_tests(self):
        """Run all single position FIFO tests."""
        logger.info("\n%s", "=" * 80)
        logger.info("Starting Single Position FIFO Tests")
        logger.info("=" * 80)

        try:
            self.test_single_position_enforcement()
            self.test_fifo_with_single_position()
            self.test_position_reversal_workflow()

            logger.info("\n%s", "=" * 80)
            logger.info("✅ ALL TESTS PASSED!")
            logger.info("=" * 80)

            return True

        except Exception:
            logger.exception("\n❌ TEST FAILED")
            return False


def main():
    """Run single position FIFO tests."""
    test = SinglePositionFIFOTest()
    success = test.run_all_tests()

    # Print log file location
    logger.info("\nTest logs saved to: /tmp/single_position_test.log")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
