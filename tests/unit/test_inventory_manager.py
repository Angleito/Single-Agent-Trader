"""
Unit tests for the InventoryManager class.

Tests comprehensive inventory management functionality including:
- Position tracking and imbalance calculation
- VuManChu signal integration for rebalancing decisions
- Risk assessment and emergency conditions
- State persistence and recovery
"""

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot.strategy.inventory_manager import (
    InventoryManager,
    InventoryMetrics,
    RebalancingAction,
    VuManChuBias,
)
from bot.trading_types import Order, OrderStatus, Position


class TestInventoryManager(unittest.TestCase):
    """Test cases for InventoryManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.symbol = "BTC-USD"

        # Create inventory manager with test parameters
        self.inventory_manager = InventoryManager(
            symbol=self.symbol,
            max_position_pct=10.0,
            rebalancing_threshold=5.0,
            emergency_threshold=15.0,
            inventory_timeout_hours=2.0,
            data_dir=self.temp_dir,
        )

        # Set test account equity
        self.inventory_manager.update_account_equity(Decimal(10000))

        # Ensure clean state for each test
        self.inventory_manager.reset_inventory()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test InventoryManager initialization."""
        assert self.inventory_manager.symbol == self.symbol
        assert self.inventory_manager.max_position_pct == 10.0
        assert self.inventory_manager.rebalancing_threshold == 5.0
        assert self.inventory_manager.emergency_threshold == 15.0
        assert self.inventory_manager.inventory_timeout_hours == 2.0

        # Check initial state
        assert self.inventory_manager._current_position == Decimal(0)
        assert self.inventory_manager._average_entry_price is None
        assert self.inventory_manager._position_start_time is None

    def test_track_position_changes_with_fills(self):
        """Test tracking position changes from order fills."""
        # Create test order fills
        buy_order = Order(
            id="order_1",
            symbol=self.symbol,
            side="BUY",
            type="MARKET",
            quantity=Decimal("1.0"),
            price=Decimal(50000),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
            filled_quantity=Decimal("1.0"),
        )

        sell_order = Order(
            id="order_2",
            symbol=self.symbol,
            side="SELL",
            type="MARKET",
            quantity=Decimal("0.5"),
            price=Decimal(51000),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
            filled_quantity=Decimal("0.5"),
        )

        # Create current position
        current_position = Position(
            symbol=self.symbol,
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal(50000),
            timestamp=datetime.now(UTC),
        )

        # Track position changes
        metrics = self.inventory_manager.track_position_changes(
            [buy_order, sell_order], current_position
        )

        # Verify position tracking
        assert self.inventory_manager._current_position == Decimal("0.5")
        assert metrics is not None
        assert isinstance(metrics, InventoryMetrics)

        # Verify metrics calculation
        assert metrics.symbol == self.symbol
        assert metrics.net_position == Decimal("0.5")
        assert metrics.position_value > 0

    def test_calculate_inventory_imbalance(self):
        """Test inventory imbalance calculation."""
        # Test with no position
        imbalance = self.inventory_manager.calculate_inventory_imbalance()
        assert imbalance == 0.0

        # Set a long position
        self.inventory_manager._current_position = Decimal("1.0")
        self.inventory_manager._average_entry_price = Decimal(50000)

        # Calculate imbalance
        imbalance = self.inventory_manager.calculate_inventory_imbalance()

        # Should be positive for long position
        assert imbalance > 0

        # Test with short position
        self.inventory_manager._current_position = Decimal("-1.0")
        imbalance = self.inventory_manager.calculate_inventory_imbalance()

        # Should be negative for short position
        assert imbalance < 0

    def test_vumanchu_bias_integration(self):
        """Test VuManChu bias integration in rebalancing decisions."""
        # Create VuManChu bias
        bullish_bias = VuManChuBias(
            overall_bias="BULLISH",
            cipher_a_signal="GREEN_DIAMOND",
            cipher_b_signal="BUY_CIRCLE",
            wave_trend_direction="UP",
            signal_strength=0.8,
            confidence=0.9,
        )

        bearish_bias = VuManChuBias(
            overall_bias="BEARISH",
            cipher_a_signal="RED_DIAMOND",
            cipher_b_signal="SELL_CIRCLE",
            wave_trend_direction="DOWN",
            signal_strength=0.7,
            confidence=0.8,
        )

        # Test with long position and bullish bias (supporting)
        self.inventory_manager._current_position = Decimal("1.0")
        self.inventory_manager._average_entry_price = Decimal(50000)

        imbalance = 6.0  # Above rebalancing threshold
        market_price = Decimal(51000)

        action = self.inventory_manager.suggest_rebalancing_action(
            imbalance, bullish_bias, market_price
        )

        # Should be conservative or hold due to supporting bias
        assert action.action_type in ["HOLD", "SELL"]
        if action.action_type == "SELL":
            assert action.urgency == "LOW"

        # Test with long position and bearish bias (conflicting)
        action = self.inventory_manager.suggest_rebalancing_action(
            imbalance, bearish_bias, market_price
        )

        # Should be more aggressive due to conflicting bias
        assert action.action_type == "SELL"
        assert action.urgency in ["MEDIUM", "HIGH"]

    def test_emergency_conditions(self):
        """Test emergency condition detection and response."""
        # Set up position that exceeds emergency threshold
        self.inventory_manager._current_position = Decimal("2.0")
        self.inventory_manager._average_entry_price = Decimal(50000)

        # Calculate imbalance that should trigger emergency
        imbalance = 20.0  # Above emergency threshold of 15%

        neutral_bias = VuManChuBias(
            overall_bias="NEUTRAL",
            signal_strength=0.5,
            confidence=0.5,
        )

        market_price = Decimal(51000)

        action = self.inventory_manager.suggest_rebalancing_action(
            imbalance, neutral_bias, market_price
        )

        # Should trigger emergency action
        assert action.urgency == "EMERGENCY"
        assert action.action_type == "SELL"  # Flatten long position
        assert action.quantity == Decimal("2.0")

    def test_timeout_conditions(self):
        """Test position timeout detection and handling."""
        # Set up position that has been held too long
        self.inventory_manager._current_position = Decimal("1.0")
        self.inventory_manager._average_entry_price = Decimal(50000)
        self.inventory_manager._position_start_time = datetime.now(UTC) - timedelta(
            hours=3
        )

        # Imbalance below emergency but position is old
        imbalance = 8.0  # Below emergency threshold

        neutral_bias = VuManChuBias(
            overall_bias="NEUTRAL",
            signal_strength=0.5,
            confidence=0.5,
        )

        market_price = Decimal(51000)

        action = self.inventory_manager.suggest_rebalancing_action(
            imbalance, neutral_bias, market_price
        )

        # Should trigger timeout action
        assert action.urgency == "HIGH"
        assert action.action_type == "SELL"
        assert "timeout" in action.reason.lower()

    def test_rebalancing_execution(self):
        """Test rebalancing trade execution."""
        # Create rebalancing action
        action = RebalancingAction(
            action_type="SELL",
            quantity=Decimal("0.5"),
            urgency="MEDIUM",
            reason="Test rebalancing",
        )

        # Set initial position
        self.inventory_manager._current_position = Decimal("1.0")
        initial_position = self.inventory_manager._current_position

        market_price = Decimal(51000)

        # Execute rebalancing
        success = self.inventory_manager.execute_rebalancing_trade(action, market_price)

        # Verify execution
        assert success

        # Verify position was updated
        expected_position = initial_position - action.quantity
        assert self.inventory_manager._current_position == expected_position

        # Verify rebalancing history was recorded
        assert len(self.inventory_manager._rebalancing_history) > 0

        latest_record = self.inventory_manager._rebalancing_history[-1]
        assert latest_record["executed"]

    def test_inventory_metrics_calculation(self):
        """Test comprehensive inventory metrics calculation."""
        # Set up position
        self.inventory_manager._current_position = Decimal("1.5")
        self.inventory_manager._average_entry_price = Decimal(50000)
        self.inventory_manager._position_start_time = datetime.now(UTC) - timedelta(
            hours=1
        )

        # Get metrics
        metrics = self.inventory_manager.get_inventory_metrics()

        # Verify metrics
        assert isinstance(metrics, InventoryMetrics)
        assert metrics.symbol == self.symbol
        assert metrics.net_position == Decimal("1.5")
        assert metrics.position_value > 0
        assert metrics.imbalance_percentage > 0
        assert metrics.risk_score > 0
        self.assertAlmostEqual(metrics.inventory_duration_hours, 1.0, places=1)

    def test_position_summary(self):
        """Test comprehensive position summary generation."""
        # Set up some position and history
        self.inventory_manager._current_position = Decimal("1.0")
        self.inventory_manager._average_entry_price = Decimal(50000)
        self.inventory_manager._rebalancing_success_count = 5
        self.inventory_manager._rebalancing_failure_count = 1

        # Get summary
        summary = self.inventory_manager.get_position_summary()

        # Verify summary structure
        assert "symbol" in summary
        assert "current_position" in summary
        assert "position_value" in summary
        assert "imbalance_percentage" in summary
        assert "risk_score" in summary
        assert "rebalancing_stats" in summary

        # Verify rebalancing stats
        rebal_stats = summary["rebalancing_stats"]
        assert rebal_stats["total_success"] == 5
        assert rebal_stats["total_failure"] == 1

    def test_state_persistence(self):
        """Test state saving and loading."""
        # Set up some state
        self.inventory_manager._current_position = Decimal("1.5")
        self.inventory_manager._average_entry_price = Decimal(50000)
        self.inventory_manager._total_realized_pnl = Decimal(100)
        self.inventory_manager._rebalancing_success_count = 3

        # Save state
        self.inventory_manager._save_state()

        # Verify state file exists
        assert self.inventory_manager.state_file.exists()

        # Create new manager and load state
        new_manager = InventoryManager(
            symbol=self.symbol,
            data_dir=self.temp_dir,
        )

        # Verify state was loaded
        assert new_manager._current_position == Decimal("1.5")
        assert new_manager._average_entry_price == Decimal(50000)
        assert new_manager._total_realized_pnl == Decimal(100)
        assert new_manager._rebalancing_success_count == 3

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with invalid fills
        invalid_order = Order(
            id="invalid",
            symbol="WRONG-SYMBOL",  # Wrong symbol
            side="BUY",
            type="MARKET",
            quantity=Decimal("1.0"),
            price=None,  # No price
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
            filled_quantity=Decimal("1.0"),
        )

        current_position = Position(
            symbol=self.symbol,
            side="FLAT",
            size=Decimal(0),
            timestamp=datetime.now(UTC),
        )

        # Should handle gracefully
        metrics = self.inventory_manager.track_position_changes(
            [invalid_order], current_position
        )

        # Should return valid metrics despite invalid order
        assert isinstance(metrics, InventoryMetrics)

    def test_risk_score_calculation(self):
        """Test risk score calculation logic."""
        # Test with moderate imbalance
        imbalance_pct = 8.0
        risk_score = self.inventory_manager._calculate_risk_score(imbalance_pct)

        # Should return reasonable risk score
        assert risk_score >= 0
        assert risk_score <= 100

        # Test with high imbalance
        high_imbalance = 20.0
        high_risk_score = self.inventory_manager._calculate_risk_score(high_imbalance)

        # Higher imbalance should result in higher risk score
        assert high_risk_score > risk_score

    def test_multiple_position_changes(self):
        """Test handling multiple position changes over time."""
        orders = []
        positions = []

        # Create sequence of orders
        for i in range(5):
            order = Order(
                id=f"order_{i}",
                symbol=self.symbol,
                side="BUY" if i % 2 == 0 else "SELL",
                type="MARKET",
                quantity=Decimal("0.5"),
                price=Decimal(str(50000 + i * 100)),
                status=OrderStatus.FILLED,
                timestamp=datetime.now(UTC),
                filled_quantity=Decimal("0.5"),
            )
            orders.append(order)

            # Create corresponding position
            net_size = sum(
                o.filled_quantity if o.side == "BUY" else -o.filled_quantity
                for o in orders
            )

            if net_size == 0:
                side = "FLAT"
                size = Decimal(0)
            elif net_size > 0:
                side = "LONG"
                size = net_size
            else:
                side = "SHORT"
                size = abs(net_size)

            position = Position(
                symbol=self.symbol,
                side=side,
                size=size,
                entry_price=Decimal(str(50000 + i * 100)) if size > 0 else None,
                timestamp=datetime.now(UTC),
            )
            positions.append(position)

        # Process all changes
        for i, (order, position) in enumerate(zip(orders, positions, strict=False)):
            metrics = self.inventory_manager.track_position_changes([order], position)

            # Verify metrics are valid
            assert isinstance(metrics, InventoryMetrics)
            assert metrics.symbol == self.symbol

        # Verify final state
        final_position = sum(
            o.filled_quantity if o.side == "BUY" else -o.filled_quantity for o in orders
        )
        assert self.inventory_manager._current_position == final_position

    def test_account_equity_update(self):
        """Test account equity updates and impact on position limits."""
        # Initial equity
        initial_equity = Decimal(10000)
        self.inventory_manager.update_account_equity(initial_equity)

        # Get initial metrics
        initial_metrics = self.inventory_manager.get_inventory_metrics()
        initial_limit = initial_metrics.max_position_limit

        # Update equity
        new_equity = Decimal(20000)
        self.inventory_manager.update_account_equity(new_equity)

        # Get updated metrics
        updated_metrics = self.inventory_manager.get_inventory_metrics()
        updated_limit = updated_metrics.max_position_limit

        # Position limit should have doubled
        assert updated_limit == initial_limit * 2

    def test_reset_inventory(self):
        """Test inventory reset functionality."""
        # Set up some state
        self.inventory_manager._current_position = Decimal("1.0")
        self.inventory_manager._average_entry_price = Decimal(50000)
        self.inventory_manager._total_realized_pnl = Decimal(100)
        self.inventory_manager._position_history.append({"test": "data"})

        # Reset inventory
        self.inventory_manager.reset_inventory()

        # Verify everything is cleared
        assert self.inventory_manager._current_position == Decimal(0)
        assert self.inventory_manager._average_entry_price is None
        assert self.inventory_manager._total_realized_pnl == Decimal(0)
        assert len(self.inventory_manager._position_history) == 0


class TestVuManChuBias(unittest.TestCase):
    """Test cases for VuManChuBias functionality."""

    def test_vumanchu_bias_creation(self):
        """Test VuManChu bias object creation and serialization."""
        bias = VuManChuBias(
            overall_bias="BULLISH",
            cipher_a_signal="GREEN_DIAMOND",
            cipher_b_signal="BUY_CIRCLE",
            wave_trend_direction="UP",
            signal_strength=0.8,
            confidence=0.9,
        )

        # Verify properties
        assert bias.overall_bias == "BULLISH"
        assert bias.cipher_a_signal == "GREEN_DIAMOND"
        assert bias.cipher_b_signal == "BUY_CIRCLE"
        assert bias.wave_trend_direction == "UP"
        assert bias.signal_strength == 0.8
        assert bias.confidence == 0.9

        # Test serialization
        bias_dict = bias.to_dict()
        assert "overall_bias" in bias_dict
        assert "cipher_a_signal" in bias_dict
        assert "timestamp" in bias_dict


class TestRebalancingAction(unittest.TestCase):
    """Test cases for RebalancingAction functionality."""

    def test_rebalancing_action_creation(self):
        """Test rebalancing action creation and serialization."""
        action = RebalancingAction(
            action_type="SELL",
            quantity=Decimal("1.5"),
            urgency="HIGH",
            reason="Emergency rebalancing",
            target_price=Decimal(51000),
            vumanchu_bias="BEARISH",
            confidence=0.8,
        )

        # Verify properties
        assert action.action_type == "SELL"
        assert action.quantity == Decimal("1.5")
        assert action.urgency == "HIGH"
        assert action.reason == "Emergency rebalancing"
        assert action.target_price == Decimal(51000)
        assert action.vumanchu_bias == "BEARISH"
        assert action.confidence == 0.8

        # Test serialization
        action_dict = action.to_dict()
        assert "action_type" in action_dict
        assert "quantity" in action_dict
        assert "timestamp" in action_dict


class TestInventoryMetrics(unittest.TestCase):
    """Test cases for InventoryMetrics functionality."""

    def test_inventory_metrics_creation(self):
        """Test inventory metrics creation and serialization."""
        metrics = InventoryMetrics(
            symbol="BTC-USD",
            net_position=Decimal("1.5"),
            position_value=Decimal(75000),
            imbalance_percentage=7.5,
            risk_score=25.0,
            max_position_limit=Decimal(100000),
            rebalancing_threshold=5.0,
            time_weighted_exposure=Decimal("10.5"),
            inventory_duration_hours=2.5,
        )

        # Verify properties
        assert metrics.symbol == "BTC-USD"
        assert metrics.net_position == Decimal("1.5")
        assert metrics.position_value == Decimal(75000)
        assert metrics.imbalance_percentage == 7.5
        assert metrics.risk_score == 25.0

        # Test serialization
        metrics_dict = metrics.to_dict()
        assert "symbol" in metrics_dict
        assert "net_position" in metrics_dict
        assert "timestamp" in metrics_dict


if __name__ == "__main__":
    unittest.main()
