"""
Unit tests for the Market Making Performance Monitor.

Tests the comprehensive performance monitoring system for market making operations,
including P&L tracking, spread analysis, fill rate monitoring, and alert generation.
"""

import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

from bot.exchange.bluefin_fee_calculator import BluefinFeeCalculator
from bot.strategy.market_making_order_manager import ManagedOrder, OrderState
from bot.strategy.market_making_performance_monitor import (
    MarketMakingPerformanceMonitor,
    MarketMakingThresholds,
    TradePnL,
)
from bot.strategy.market_making_strategy import DirectionalBias, SpreadCalculation
from bot.trading_types import IndicatorData, Order, OrderStatus


class TestMarketMakingPerformanceMonitor(unittest.TestCase):
    """Test cases for MarketMakingPerformanceMonitor."""

    def setUp(self):
        """Set up test fixtures."""
        self.fee_calculator = BluefinFeeCalculator()
        self.symbol = "BTC-PERP"
        self.thresholds = MarketMakingThresholds()
        self.monitor = MarketMakingPerformanceMonitor(
            fee_calculator=self.fee_calculator,
            symbol=self.symbol,
            thresholds=self.thresholds,
            max_history_size=1000,
        )

    def test_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.symbol, self.symbol)
        self.assertEqual(self.monitor.fee_calculator, self.fee_calculator)
        self.assertEqual(self.monitor.thresholds, self.thresholds)
        self.assertEqual(len(self.monitor.completed_trades), 0)
        self.assertEqual(len(self.monitor.fill_events), 0)
        self.assertIsNotNone(self.monitor.session_start)

    def test_thresholds_configuration(self):
        """Test market making specific thresholds."""
        self.assertEqual(self.thresholds.min_fill_rate, 0.3)
        self.assertEqual(self.thresholds.min_spread_capture_rate, 0.6)
        self.assertEqual(self.thresholds.max_fee_ratio, 0.5)
        self.assertEqual(self.thresholds.max_inventory_imbalance, 0.2)
        self.assertEqual(self.thresholds.min_turnover_rate, 2.0)
        self.assertEqual(self.thresholds.max_negative_pnl_streak, 5)
        self.assertEqual(self.thresholds.min_signal_effectiveness, 0.4)
        self.assertEqual(self.thresholds.max_slippage_bps, 5)

    def test_record_order_fill(self):
        """Test recording order fill events."""
        # Create a mock managed order
        order = Order(
            id="test_order_1",
            symbol=self.symbol,
            side="BUY",
            type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now(UTC),
        )

        managed_order = ManagedOrder(
            order=order, level=0, target_price=Decimal("50000.0")
        )
        managed_order.state = OrderState.OPEN

        # Record a fill
        fill_timestamp = datetime.now(UTC)
        self.monitor.record_order_fill(
            managed_order=managed_order,
            fill_quantity=Decimal("1.0"),
            fill_price=Decimal("50000.0"),
            fill_timestamp=fill_timestamp,
            fees_paid=Decimal("5.0"),
        )

        # Verify fill was recorded
        self.assertEqual(len(self.monitor.fill_events), 1)
        fill_event = self.monitor.fill_events[0]
        self.assertEqual(fill_event["order_id"], "test_order_1")
        self.assertEqual(fill_event["side"], "BUY")
        self.assertEqual(fill_event["quantity"], Decimal("1.0"))
        self.assertEqual(fill_event["price"], Decimal("50000.0"))
        self.assertEqual(fill_event["fees"], Decimal("5.0"))

        # Verify active position was created
        position_key = f"{self.symbol}_0"
        self.assertIn(position_key, self.monitor.active_positions)
        position = self.monitor.active_positions[position_key]
        self.assertEqual(position["side"], "BUY")
        self.assertEqual(position["entry_price"], Decimal("50000.0"))
        self.assertEqual(position["quantity"], Decimal("1.0"))
        self.assertEqual(position["fees_paid"], Decimal("5.0"))

    def test_record_spread_target(self):
        """Test recording spread targets."""
        spread_calc = SpreadCalculation(
            base_spread=Decimal("50.0"),
            adjusted_spread=Decimal("60.0"),
            bid_adjustment=Decimal("-5.0"),
            ask_adjustment=Decimal("5.0"),
            min_profitable_spread=Decimal("20.0"),
        )

        current_price = Decimal("50000.0")
        timestamp = datetime.now(UTC)

        bias = DirectionalBias(
            direction="bullish",
            strength=0.7,
            confidence=0.8,
            signals={"cipher_a_dot": 0.5},
        )

        self.monitor.record_spread_target(
            spread_calc=spread_calc,
            current_price=current_price,
            timestamp=timestamp,
            vumanchu_bias=bias,
        )

        # Verify spread target was recorded
        self.assertEqual(len(self.monitor.spread_targets), 1)
        recorded_spread = self.monitor.spread_targets[0]
        self.assertEqual(recorded_spread.adjusted_spread, Decimal("60.0"))

    def test_record_vumanchu_signal(self):
        """Test recording VuManChu signals."""
        indicators = IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=0.5,
            cipher_b_wave=25.0,
            cipher_b_money_flow=60.0,
            rsi=55.0,
            ema_fast=50000.0,
            ema_slow=49500.0,
        )

        bias = DirectionalBias(
            direction="bullish",
            strength=0.7,
            confidence=0.8,
            signals={"cipher_a_dot": 0.5, "cipher_b_wave": 25.0},
        )

        timestamp = datetime.now(UTC)

        signal_id = self.monitor.record_vumanchu_signal(
            indicators=indicators,
            bias=bias,
            timestamp=timestamp,
        )

        # Verify signal was recorded
        self.assertTrue(signal_id.startswith("signal_"))
        self.assertEqual(len(self.monitor.signal_events), 1)

        signal_event = self.monitor.signal_events[0]
        self.assertEqual(signal_event["signal_id"], signal_id)
        self.assertEqual(signal_event["bias_direction"], "bullish")
        self.assertEqual(signal_event["bias_strength"], 0.7)
        self.assertEqual(signal_event["bias_confidence"], 0.8)
        self.assertEqual(signal_event["cipher_a_dot"], 0.5)
        self.assertFalse(signal_event["outcome_recorded"])

    def test_record_signal_outcome(self):
        """Test recording signal outcomes."""
        # First record a signal
        indicators = IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=0.5,
            rsi=55.0,
        )

        bias = DirectionalBias(
            direction="bullish",
            strength=0.7,
            confidence=0.8,
            signals={"cipher_a_dot": 0.5},
        )

        signal_id = self.monitor.record_vumanchu_signal(
            indicators=indicators,
            bias=bias,
            timestamp=datetime.now(UTC),
        )

        # Record outcome
        pnl_result = Decimal("100.0")
        success = True
        outcome_timestamp = datetime.now(UTC)

        self.monitor.record_signal_outcome(
            signal_id=signal_id,
            pnl_result=pnl_result,
            success=success,
            outcome_timestamp=outcome_timestamp,
        )

        # Verify outcome was recorded
        self.assertEqual(len(self.monitor.signal_outcomes), 1)
        outcome = self.monitor.signal_outcomes[0]
        self.assertEqual(outcome["signal_id"], signal_id)
        self.assertEqual(outcome["pnl_result"], pnl_result)
        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["bias_direction"], "bullish")

        # Verify signal event was marked as outcome recorded
        signal_event = self.monitor.signal_events[0]
        self.assertTrue(signal_event["outcome_recorded"])

    def test_get_real_time_pnl(self):
        """Test real-time P&L calculation."""
        # Add some completed trades
        trade1 = TradePnL(
            trade_id="trade_1",
            timestamp=datetime.now(UTC),
            side="BUY",
            entry_price=Decimal("50000.0"),
            exit_price=Decimal("50100.0"),
            quantity=Decimal("1.0"),
            gross_pnl=Decimal("100.0"),
            fees_paid=Decimal("10.0"),
            net_pnl=Decimal("90.0"),
            holding_time_seconds=300.0,
            spread_captured=Decimal("50.0"),
        )

        trade2 = TradePnL(
            trade_id="trade_2",
            timestamp=datetime.now(UTC),
            side="SELL",
            entry_price=Decimal("50200.0"),
            exit_price=Decimal("50050.0"),
            quantity=Decimal("0.5"),
            gross_pnl=Decimal("75.0"),
            fees_paid=Decimal("5.0"),
            net_pnl=Decimal("70.0"),
            holding_time_seconds=600.0,
            spread_captured=Decimal("25.0"),
        )

        self.monitor.completed_trades.extend([trade1, trade2])

        # Add an active position
        self.monitor.active_positions["BTC-PERP_0"] = {
            "side": "BUY",
            "entry_price": Decimal("50000.0"),
            "quantity": Decimal("0.5"),
            "fees_paid": Decimal("2.5"),
            "entry_time": datetime.now(UTC),
        }

        # Calculate P&L with current prices
        current_prices = {self.symbol: Decimal("50150.0")}
        pnl_breakdown = self.monitor.get_real_time_pnl(current_prices)

        # Verify calculations
        self.assertEqual(pnl_breakdown["realized_pnl"], Decimal("160.0"))  # 90 + 70
        self.assertEqual(
            pnl_breakdown["unrealized_pnl"], Decimal("75.0")
        )  # (50150 - 50000) * 0.5
        self.assertEqual(pnl_breakdown["gross_pnl"], Decimal("235.0"))  # 160 + 75
        self.assertEqual(pnl_breakdown["total_fees"], Decimal("17.5"))  # 10 + 5 + 2.5
        self.assertEqual(pnl_breakdown["net_pnl"], Decimal("217.5"))  # 235 - 17.5

    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        # Add some test data
        current_time = datetime.now(UTC)

        # Add completed trades
        for i in range(5):
            trade = TradePnL(
                trade_id=f"trade_{i}",
                timestamp=current_time - timedelta(minutes=i * 10),
                side="BUY" if i % 2 == 0 else "SELL",
                entry_price=Decimal("50000.0"),
                exit_price=Decimal("50100.0") if i < 3 else Decimal("49950.0"),
                quantity=Decimal("1.0"),
                gross_pnl=Decimal("100.0") if i < 3 else Decimal("-50.0"),
                fees_paid=Decimal("10.0"),
                net_pnl=Decimal("90.0") if i < 3 else Decimal("-60.0"),
                holding_time_seconds=300.0,
                spread_captured=Decimal("50.0"),
            )
            self.monitor.completed_trades.append(trade)

        # Add fill events
        for i in range(8):
            fill_event = {
                "order_id": f"order_{i}",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "quantity": Decimal("1.0"),
                "price": Decimal("50000.0"),
                "fees": Decimal("5.0"),
                "timestamp": current_time - timedelta(minutes=i * 5),
                "fill_delay_seconds": 5.0 + i,
            }
            self.monitor.fill_events.append(fill_event)

        # Get metrics
        metrics = self.monitor.get_performance_metrics(timedelta(hours=1))

        # Verify basic metrics
        self.assertEqual(metrics["total_trades"], 5)
        self.assertEqual(metrics["total_fills"], 8)
        self.assertEqual(metrics["winning_trades"], 3)
        self.assertEqual(metrics["losing_trades"], 2)
        self.assertEqual(metrics["win_rate"], 0.6)  # 3/5
        self.assertAlmostEqual(metrics["gross_pnl"], 200.0)  # 3*100 + 2*(-50)
        self.assertAlmostEqual(metrics["net_pnl"], 150.0)  # 3*90 + 2*(-60)
        self.assertAlmostEqual(metrics["total_fees"], 50.0)  # 5*10
        self.assertEqual(metrics["buy_fills"], 4)
        self.assertEqual(metrics["sell_fills"], 4)

    def test_get_dashboard_data(self):
        """Test dashboard data generation."""
        dashboard_data = self.monitor.get_dashboard_data()

        # Verify structure
        self.assertIn("timestamp", dashboard_data)
        self.assertIn("symbol", dashboard_data)
        self.assertIn("session_duration_hours", dashboard_data)
        self.assertIn("current_performance", dashboard_data)
        self.assertIn("active_positions", dashboard_data)
        self.assertIn("inventory_imbalance", dashboard_data)
        self.assertIn("recent_alerts", dashboard_data)
        self.assertIn("health_score", dashboard_data)

        # Verify performance windows
        performance = dashboard_data["current_performance"]
        self.assertIn("1h", performance)
        self.assertIn("24h", performance)
        self.assertIn("7d", performance)

        # Verify 1h metrics structure
        h1_metrics = performance["1h"]
        self.assertIn("net_pnl", h1_metrics)
        self.assertIn("trades", h1_metrics)
        self.assertIn("win_rate", h1_metrics)
        self.assertIn("fill_efficiency", h1_metrics)
        self.assertIn("spread_capture", h1_metrics)

    def test_check_performance_alerts(self):
        """Test performance alert generation."""
        # Set metrics that will trigger alerts
        with patch.object(self.monitor, "get_performance_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "fill_efficiency": 0.2,  # Below threshold of 0.3
                "spread_capture_rate": 0.4,  # Below threshold of 0.6
                "fee_efficiency_ratio": 0.6,  # Above threshold of 0.5
                "current_inventory_imbalance": 25.0,  # Above threshold of 20%
                "signal_success_rate": 0.3,  # Below threshold of 0.4
                "net_pnl": -100.0,  # Negative
            }

            alerts = self.monitor.check_performance_alerts()

            # Should generate multiple alerts
            self.assertGreater(len(alerts), 0)

            # Check specific alert types
            alert_types = [alert.metric_name for alert in alerts]
            self.assertIn("fill_efficiency", alert_types)
            self.assertIn("spread_capture_rate", alert_types)
            self.assertIn("fee_efficiency_ratio", alert_types)
            self.assertIn("inventory_imbalance", alert_types)

    def test_export_performance_data(self):
        """Test performance data export."""
        # Add some test data
        trade = TradePnL(
            trade_id="test_trade",
            timestamp=datetime.now(UTC),
            side="BUY",
            entry_price=Decimal("50000.0"),
            exit_price=Decimal("50100.0"),
            quantity=Decimal("1.0"),
            gross_pnl=Decimal("100.0"),
            fees_paid=Decimal("10.0"),
            net_pnl=Decimal("90.0"),
            holding_time_seconds=300.0,
            spread_captured=Decimal("50.0"),
        )
        self.monitor.completed_trades.append(trade)

        # Export without raw data
        export_data = self.monitor.export_performance_data(
            time_window=timedelta(hours=1),
            include_raw_data=False,
        )

        # Verify structure
        self.assertIn("export_timestamp", export_data)
        self.assertIn("symbol", export_data)
        self.assertIn("performance_metrics", export_data)
        self.assertIn("dashboard_data", export_data)
        self.assertIn("alert_summary", export_data)
        self.assertIn("thresholds", export_data)
        self.assertNotIn("raw_data", export_data)

        # Export with raw data
        export_data_raw = self.monitor.export_performance_data(
            time_window=timedelta(hours=1),
            include_raw_data=True,
        )

        # Verify raw data included
        self.assertIn("raw_data", export_data_raw)
        raw_data = export_data_raw["raw_data"]
        self.assertIn("completed_trades", raw_data)
        self.assertIn("fill_events", raw_data)
        self.assertIn("signal_outcomes", raw_data)

        # Verify trade data
        self.assertEqual(len(raw_data["completed_trades"]), 1)
        trade_data = raw_data["completed_trades"][0]
        self.assertEqual(trade_data["trade_id"], "test_trade")
        self.assertEqual(trade_data["side"], "BUY")

    def test_thread_safety(self):
        """Test thread safety of concurrent operations."""
        import threading
        import time

        def record_fills():
            for i in range(50):
                order = Order(
                    id=f"order_{i}",
                    symbol=self.symbol,
                    side="BUY" if i % 2 == 0 else "SELL",
                    type="LIMIT",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000.0"),
                    status=OrderStatus.OPEN,
                    timestamp=datetime.now(UTC),
                )

                managed_order = ManagedOrder(order, i % 3, Decimal("50000.0"))

                self.monitor.record_order_fill(
                    managed_order=managed_order,
                    fill_quantity=Decimal("1.0"),
                    fill_price=Decimal("50000.0"),
                    fill_timestamp=datetime.now(UTC),
                    fees_paid=Decimal("5.0"),
                )
                time.sleep(0.001)  # Small delay

        def get_metrics():
            for _ in range(20):
                self.monitor.get_performance_metrics()
                time.sleep(0.005)

        # Run concurrent operations
        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=record_fills)
            t2 = threading.Thread(target=get_metrics)
            threads.extend([t1, t2])

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no data corruption
        self.assertGreater(len(self.monitor.fill_events), 0)
        self.assertLessEqual(
            len(self.monitor.fill_events), 150
        )  # 3 threads * 50 fills each

    def test_completed_trade_detection(self):
        """Test automatic detection of completed round-trip trades."""
        # Create buy order and fill
        buy_order = Order(
            id="buy_order_1",
            symbol=self.symbol,
            side="BUY",
            type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now(UTC),
        )

        buy_managed_order = ManagedOrder(buy_order, 0, Decimal("50000.0"))

        # Record buy fill
        self.monitor.record_order_fill(
            managed_order=buy_managed_order,
            fill_quantity=Decimal("1.0"),
            fill_price=Decimal("50000.0"),
            fill_timestamp=datetime.now(UTC),
            fees_paid=Decimal("5.0"),
        )

        # Verify position created, no completed trades yet
        self.assertEqual(len(self.monitor.active_positions), 1)
        self.assertEqual(len(self.monitor.completed_trades), 0)

        # Create sell order and fill (opposite side)
        sell_order = Order(
            id="sell_order_1",
            symbol=self.symbol,
            side="SELL",
            type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50100.0"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now(UTC),
        )

        sell_managed_order = ManagedOrder(sell_order, 1, Decimal("50100.0"))

        # Record sell fill
        self.monitor.record_order_fill(
            managed_order=sell_managed_order,
            fill_quantity=Decimal("1.0"),
            fill_price=Decimal("50100.0"),
            fill_timestamp=datetime.now(UTC) + timedelta(seconds=60),
            fees_paid=Decimal("5.0"),
        )

        # Should now have a completed trade
        self.assertEqual(len(self.monitor.completed_trades), 1)

        completed_trade = self.monitor.completed_trades[0]
        self.assertEqual(completed_trade.side, "BUY")  # Original position side
        self.assertEqual(completed_trade.entry_price, Decimal("50000.0"))
        self.assertEqual(completed_trade.exit_price, Decimal("50100.0"))
        self.assertEqual(
            completed_trade.gross_pnl, Decimal("100.0")
        )  # (50100 - 50000) * 1
        self.assertEqual(completed_trade.fees_paid, Decimal("10.0"))  # 5 + 5
        self.assertEqual(completed_trade.net_pnl, Decimal("90.0"))  # 100 - 10
        self.assertEqual(completed_trade.holding_time_seconds, 60.0)


if __name__ == "__main__":
    unittest.main()
