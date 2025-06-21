"""
Market Making Performance Monitor Integration Example.

This example demonstrates how to integrate the MarketMakingPerformanceMonitor
with a market making trading system to track performance metrics in real-time.

Key demonstration areas:
- Setting up performance monitoring
- Recording trading events and fills
- Tracking VuManChu signal effectiveness
- Monitoring inventory and spread capture
- Generating performance alerts
- Exporting data for analysis
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from bot.exchange.bluefin_fee_calculator import BluefinFeeCalculator
from bot.performance_monitor import PerformanceAlert
from bot.strategy.inventory_manager import InventoryMetrics
from bot.strategy.market_making_order_manager import ManagedOrder, OrderState
from bot.strategy.market_making_performance_monitor import (
    MarketMakingPerformanceMonitor,
    MarketMakingThresholds,
)
from bot.strategy.market_making_strategy import DirectionalBias, SpreadCalculation
from bot.trading_types import IndicatorData, Order, OrderStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MarketMakingPerformanceExample:
    """
    Example market making system with integrated performance monitoring.

    Demonstrates real-world usage patterns for the performance monitor
    including event recording, metric tracking, and alert handling.
    """

    def __init__(self, symbol: str = "BTC-PERP"):
        """Initialize the example system."""
        self.symbol = symbol

        # Initialize components
        self.fee_calculator = BluefinFeeCalculator()

        # Configure custom thresholds for demonstration
        self.thresholds = MarketMakingThresholds()
        self.thresholds.min_fill_rate = 0.4  # 40% minimum fill rate
        self.thresholds.min_spread_capture_rate = 0.7  # 70% minimum spread capture
        self.thresholds.max_fee_ratio = 0.3  # Max 30% of profit consumed by fees

        # Initialize performance monitor
        self.performance_monitor = MarketMakingPerformanceMonitor(
            fee_calculator=self.fee_calculator,
            symbol=symbol,
            thresholds=self.thresholds,
            max_history_size=5000,
        )

        # Setup alert handling
        self.performance_monitor.base_monitor.add_alert_callback(
            self._handle_performance_alert
        )

        # Simulation state
        self.current_price = Decimal("50000.0")
        self.order_counter = 0
        self.signal_counter = 0

        logger.info("Initialized MarketMakingPerformanceExample for %s", symbol)

    async def run_simulation(self, duration_minutes: int = 60) -> None:
        """
        Run a market making simulation with performance monitoring.

        Args:
            duration_minutes: Duration of simulation in minutes
        """
        logger.info("Starting %d-minute market making simulation", duration_minutes)

        # Start performance monitoring
        await self.performance_monitor.start_monitoring()

        try:
            simulation_start = datetime.now(UTC)
            simulation_end = simulation_start + timedelta(minutes=duration_minutes)

            # Run simulation loop
            while datetime.now(UTC) < simulation_end:
                await self._simulation_tick()
                await asyncio.sleep(5)  # 5-second intervals

            # Generate final performance report
            await self._generate_final_report()

        finally:
            # Stop monitoring
            await self.performance_monitor.stop_monitoring()

        logger.info("Simulation completed")

    async def _simulation_tick(self) -> None:
        """Execute one simulation tick."""
        try:
            # Simulate price movement
            self._update_market_price()

            # Generate VuManChu signals periodically
            if self.signal_counter % 12 == 0:  # Every minute (12 * 5 seconds)
                await self._generate_vumanchu_signal()

            # Simulate order fills
            if self.order_counter % 3 == 0:  # Every 15 seconds
                await self._simulate_order_fills()

            # Update inventory snapshot
            if self.order_counter % 6 == 0:  # Every 30 seconds
                await self._update_inventory_snapshot()

            # Check for performance alerts
            if self.order_counter % 24 == 0:  # Every 2 minutes
                alerts = self.performance_monitor.check_performance_alerts()
                if alerts:
                    logger.info("Generated %d performance alerts", len(alerts))

            # Log real-time metrics periodically
            if self.order_counter % 12 == 0:  # Every minute
                await self._log_real_time_metrics()

            self.order_counter += 1
            self.signal_counter += 1

        except Exception as e:
            logger.exception("Error in simulation tick: %s", e)

    def _update_market_price(self) -> None:
        """Simulate market price movement."""
        import random

        # Random walk with slight upward bias
        change_pct = random.normalvariate(0.001, 0.002)  # 0.1% mean, 0.2% std
        price_change = self.current_price * Decimal(str(change_pct))
        self.current_price = max(Decimal("1000.0"), self.current_price + price_change)

    async def _generate_vumanchu_signal(self) -> None:
        """Generate and record a VuManChu signal."""
        try:
            import random

            # Generate realistic indicator values
            indicators = IndicatorData(
                timestamp=datetime.now(UTC),
                cipher_a_dot=random.uniform(-1.0, 1.0),
                cipher_b_wave=random.uniform(-100.0, 100.0),
                cipher_b_money_flow=random.uniform(0.0, 100.0),
                rsi=random.uniform(20.0, 80.0),
                ema_fast=float(self.current_price * Decimal("1.001")),
                ema_slow=float(self.current_price * Decimal("0.999")),
            )

            # Determine bias based on indicators
            if indicators.cipher_a_dot > 0.3 and indicators.cipher_b_money_flow > 60:
                direction = "bullish"
                strength = 0.7 + random.uniform(0, 0.3)
            elif indicators.cipher_a_dot < -0.3 and indicators.cipher_b_money_flow < 40:
                direction = "bearish"
                strength = 0.7 + random.uniform(0, 0.3)
            else:
                direction = "neutral"
                strength = random.uniform(0.2, 0.5)

            bias = DirectionalBias(
                direction=direction,
                strength=strength,
                confidence=random.uniform(0.5, 0.9),
                signals={
                    "cipher_a_dot": indicators.cipher_a_dot,
                    "cipher_b_wave": indicators.cipher_b_wave,
                    "rsi": indicators.rsi,
                },
            )

            # Record signal
            signal_id = self.performance_monitor.record_vumanchu_signal(
                indicators=indicators,
                bias=bias,
                timestamp=datetime.now(UTC),
            )

            # Record spread target based on signal
            spread_calc = self._calculate_spread_from_bias(bias)
            self.performance_monitor.record_spread_target(
                spread_calc=spread_calc,
                current_price=self.current_price,
                timestamp=datetime.now(UTC),
                vumanchu_bias=bias,
            )

            # Simulate signal outcome after some time (randomly)
            if random.random() < 0.3:  # 30% chance to immediately resolve old signals
                await self._resolve_random_signal()

            logger.debug("Generated VuManChu signal: %s %s", signal_id, direction)

        except Exception as e:
            logger.exception("Error generating VuManChu signal: %s", e)

    def _calculate_spread_from_bias(self, bias: DirectionalBias) -> SpreadCalculation:
        """Calculate spread based on directional bias."""
        base_spread_bps = 10  # 0.1%
        base_spread = self.current_price * Decimal(str(base_spread_bps / 10000))

        # Adjust spread based on bias
        if bias.direction == "bullish":
            bid_adjustment = -base_spread * Decimal(str(bias.strength * 0.3))
            ask_adjustment = base_spread * Decimal(str(bias.strength * 0.1))
        elif bias.direction == "bearish":
            bid_adjustment = base_spread * Decimal(str(bias.strength * 0.1))
            ask_adjustment = -base_spread * Decimal(str(bias.strength * 0.3))
        else:
            bid_adjustment = Decimal(0)
            ask_adjustment = Decimal(0)

        adjusted_spread = base_spread + abs(bid_adjustment) + abs(ask_adjustment)

        # Calculate minimum profitable spread
        fees = self.fee_calculator.calculate_round_trip_cost(self.current_price)
        min_profitable_spread = fees.round_trip_cost * Decimal(2)

        return SpreadCalculation(
            base_spread=base_spread,
            adjusted_spread=max(adjusted_spread, min_profitable_spread),
            bid_adjustment=bid_adjustment,
            ask_adjustment=ask_adjustment,
            min_profitable_spread=min_profitable_spread,
        )

    async def _simulate_order_fills(self) -> None:
        """Simulate order fills and record them."""
        try:
            import random

            # Generate 1-3 random fills
            num_fills = random.randint(1, 3)

            for i in range(num_fills):
                # Create mock order
                order_id = f"order_{self.order_counter}_{i}"
                side = random.choice(["BUY", "SELL"])
                level = random.randint(0, 2)

                # Calculate realistic fill price
                spread_pct = 0.001 * (level + 1)  # 0.1%, 0.2%, 0.3% from mid
                if side == "BUY":
                    fill_price = self.current_price * (
                        Decimal(1) - Decimal(str(spread_pct))
                    )
                else:
                    fill_price = self.current_price * (
                        Decimal(1) + Decimal(str(spread_pct))
                    )

                quantity = Decimal(str(random.uniform(0.1, 2.0)))

                # Create order and managed order
                order = Order(
                    id=order_id,
                    symbol=self.symbol,
                    side=side,  # type: ignore
                    type="LIMIT",
                    quantity=quantity,
                    price=fill_price,
                    status=OrderStatus.FILLED,
                    timestamp=datetime.now(UTC),
                )

                managed_order = ManagedOrder(order, level, fill_price)
                managed_order.state = OrderState.FILLED

                # Calculate fees
                notional_value = quantity * fill_price
                fees = self.fee_calculator.calculate_maker_fee(notional_value)

                # Record fill
                self.performance_monitor.record_order_fill(
                    managed_order=managed_order,
                    fill_quantity=quantity,
                    fill_price=fill_price,
                    fill_timestamp=datetime.now(UTC),
                    fees_paid=fees,
                )

            logger.debug("Simulated %d order fills", num_fills)

        except Exception as e:
            logger.exception("Error simulating order fills: %s", e)

    async def _update_inventory_snapshot(self) -> None:
        """Update inventory snapshot."""
        try:
            import random

            # Calculate current inventory from active positions
            total_base_qty = Decimal(0)
            total_position_value = Decimal(0)

            for position in self.performance_monitor.active_positions.values():
                qty = position["quantity"]
                if position["side"] == "BUY":
                    total_base_qty += qty
                else:
                    total_base_qty -= qty

                total_position_value += qty * position["entry_price"]

            # Calculate imbalance percentage
            max_position = Decimal("10.0")  # Assume max 10 units
            imbalance_pct = (
                float(total_base_qty / max_position * 100) if max_position > 0 else 0.0
            )

            # Create inventory metrics
            inventory_metrics = InventoryMetrics(
                symbol=self.symbol,
                net_position=total_base_qty,
                position_value=total_position_value,
                imbalance_percentage=imbalance_pct,
                risk_score=min(
                    abs(imbalance_pct) / 20.0, 1.0
                ),  # Risk increases with imbalance
                max_position_limit=max_position,
                rebalancing_threshold=15.0,  # 15% rebalancing threshold
                time_weighted_exposure=total_position_value,
                inventory_duration_hours=random.uniform(0.5, 4.0),  # Random duration
            )

            # Record snapshot
            self.performance_monitor.record_inventory_snapshot(inventory_metrics)

            logger.debug(
                "Updated inventory: position=%.3f, value=%.2f, imbalance=%.1f%%",
                float(total_base_qty),
                float(total_position_value),
                imbalance_pct,
            )

        except Exception as e:
            logger.exception("Error updating inventory snapshot: %s", e)

    async def _resolve_random_signal(self) -> None:
        """Resolve a random unresolved signal."""
        try:
            import random

            # Find unresolved signals
            unresolved_signals = [
                signal
                for signal in self.performance_monitor.signal_events
                if not signal["outcome_recorded"]
            ]

            if not unresolved_signals:
                return

            # Pick a random signal to resolve
            signal = random.choice(unresolved_signals)

            # Generate outcome based on bias and random factors
            success_probability = (
                signal["bias_confidence"] * 0.7
            )  # Base success on confidence
            success = random.random() < success_probability

            # Generate realistic P&L
            if success:
                pnl = Decimal(str(random.uniform(10.0, 100.0)))
            else:
                pnl = Decimal(str(random.uniform(-50.0, -5.0)))

            # Record outcome
            self.performance_monitor.record_signal_outcome(
                signal_id=signal["signal_id"],
                pnl_result=pnl,
                success=success,
                outcome_timestamp=datetime.now(UTC),
            )

            logger.debug(
                "Resolved signal %s: success=%s, pnl=%.2f",
                signal["signal_id"],
                success,
                float(pnl),
            )

        except Exception as e:
            logger.exception("Error resolving signal: %s", e)

    async def _log_real_time_metrics(self) -> None:
        """Log real-time performance metrics."""
        try:
            # Get current P&L
            current_prices = {self.symbol: self.current_price}
            pnl_breakdown = self.performance_monitor.get_real_time_pnl(current_prices)

            # Get performance metrics
            metrics = self.performance_monitor.get_performance_metrics(
                timedelta(minutes=15)
            )

            logger.info(
                "Performance Update - Net P&L: %.2f, Trades: %d, Win Rate: %.1f%%, "
                "Fill Efficiency: %.1f%%, Active Positions: %d",
                float(pnl_breakdown["net_pnl"]),
                metrics.get("total_trades", 0),
                metrics.get("win_rate", 0.0) * 100,
                metrics.get("fill_efficiency", 0.0) * 100,
                len(self.performance_monitor.active_positions),
            )

        except Exception as e:
            logger.exception("Error logging real-time metrics: %s", e)

    async def _generate_final_report(self) -> None:
        """Generate final performance report."""
        try:
            logger.info("=" * 60)
            logger.info("FINAL PERFORMANCE REPORT")
            logger.info("=" * 60)

            # Get comprehensive metrics
            metrics_1h = self.performance_monitor.get_performance_metrics(
                timedelta(hours=1)
            )
            current_prices = {self.symbol: self.current_price}
            pnl_breakdown = self.performance_monitor.get_real_time_pnl(current_prices)
            dashboard_data = self.performance_monitor.get_dashboard_data()

            # Display key metrics
            logger.info("Trading Performance:")
            logger.info("  Total Trades: %d", metrics_1h.get("total_trades", 0))
            logger.info("  Win Rate: %.1f%%", metrics_1h.get("win_rate", 0.0) * 100)
            logger.info("  Net P&L: %.2f USDC", float(pnl_breakdown["net_pnl"]))
            logger.info("  Total Fees: %.2f USDC", float(pnl_breakdown["total_fees"]))
            logger.info(
                "  Fee Percentage: %.2f%%", float(pnl_breakdown["fee_percentage"])
            )

            logger.info("\nExecution Performance:")
            logger.info(
                "  Fill Efficiency: %.1f%%",
                metrics_1h.get("fill_efficiency", 0.0) * 100,
            )
            logger.info(
                "  Spread Capture: %.1f%%",
                metrics_1h.get("spread_capture_rate", 0.0) * 100,
            )
            logger.info(
                "  Average Fill Time: %.2fs", metrics_1h.get("average_fill_time", 0.0)
            )

            logger.info("\nSignal Performance:")
            logger.info(
                "  Signal Success Rate: %.1f%%",
                metrics_1h.get("signal_success_rate", 0.0) * 100,
            )
            logger.info(
                "  Average Signal P&L: %.2f USDC",
                metrics_1h.get("average_signal_pnl", 0.0),
            )

            logger.info("\nRisk Metrics:")
            logger.info(
                "  Max Drawdown: %.2f USDC", metrics_1h.get("max_drawdown", 0.0)
            )
            logger.info("  Sharpe Ratio: %.2f", metrics_1h.get("sharpe_ratio", 0.0))
            logger.info("  Profit Factor: %.2f", metrics_1h.get("profit_factor", 0.0))

            logger.info("\nSystem Health:")
            logger.info(
                "  Health Score: %.1f/100", dashboard_data.get("health_score", 0.0)
            )
            logger.info(
                "  Active Positions: %d", dashboard_data.get("active_positions", 0)
            )
            logger.info(
                "  Inventory Imbalance: %.1f%%",
                dashboard_data.get("inventory_imbalance", 0.0),
            )

            # Export data for analysis
            export_data = self.performance_monitor.export_performance_data(
                time_window=timedelta(hours=1),
                include_raw_data=True,
            )

            # Save to file
            filename = f"market_making_performance_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                # Convert Decimal to float for JSON serialization
                json_data = self._convert_decimals_to_float(export_data)
                json.dump(json_data, f, indent=2, default=str)

            logger.info("\nPerformance data exported to: %s", filename)
            logger.info("=" * 60)

        except Exception as e:
            logger.exception("Error generating final report: %s", e)

    def _convert_decimals_to_float(self, data: any) -> any:
        """Convert Decimal objects to float for JSON serialization."""
        if isinstance(data, Decimal):
            return float(data)
        if isinstance(data, dict):
            return {
                key: self._convert_decimals_to_float(value)
                for key, value in data.items()
            }
        if isinstance(data, list):
            return [self._convert_decimals_to_float(item) for item in data]
        return data

    def _handle_performance_alert(self, alert: PerformanceAlert) -> None:
        """Handle performance alerts."""
        logger.warning(
            "PERFORMANCE ALERT [%s]: %s (value: %.3f, threshold: %.3f)",
            alert.level.value.upper(),
            alert.message,
            alert.current_value,
            alert.threshold,
        )


async def main():
    """Run the market making performance example."""
    try:
        # Create and run simulation
        example = MarketMakingPerformanceExample("BTC-PERP")

        logger.info("Starting Market Making Performance Monitor Example")
        logger.info("This simulation will run for 10 minutes and demonstrate:")
        logger.info("- Real-time P&L tracking")
        logger.info("- Order fill monitoring")
        logger.info("- VuManChu signal effectiveness")
        logger.info("- Inventory management")
        logger.info("- Performance alerts")
        logger.info("- Data export capabilities")
        logger.info("")

        # Run 10-minute simulation
        await example.run_simulation(duration_minutes=10)

    except Exception as e:
        logger.exception("Error in main: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
