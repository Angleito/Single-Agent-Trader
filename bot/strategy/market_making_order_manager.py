"""
Market Making Order Management System.

This module provides a sophisticated order management system specifically designed
for market making operations. It handles multiple concurrent orders, tracks fills
in real-time, manages inventory imbalances, and provides emergency risk controls.

Key Features:
- Place and manage ladder orders across multiple price levels
- Real-time order fill tracking and P&L calculation
- Dynamic order replacement on significant price moves
- Inventory imbalance management with bias adjustments
- Emergency position management for risk control
- Integration with Bluefin exchange client
- Comprehensive error handling and recovery
"""

import asyncio
import contextlib
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from bot.exchange.bluefin import BluefinClient
from bot.order_manager import OrderEvent, OrderManager
from bot.trading_types import Order

from .market_making_strategy import OrderLevel

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    """Enhanced order state tracking for market making."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    REPLACING = "REPLACING"  # Order being replaced due to price move


class InventoryPosition:
    """Track inventory position and imbalance."""

    def __init__(self):
        self.total_base_qty = Decimal(0)  # Net base asset position
        self.total_quote_value = Decimal(0)  # Net quote asset exposure
        self.realized_pnl = Decimal(0)
        self.unrealized_pnl = Decimal(0)
        self.last_update = datetime.now(UTC)

    def update_fill(self, side: str, quantity: Decimal, price: Decimal, fees: Decimal):
        """Update inventory based on order fill."""
        if side == "BUY":
            self.total_base_qty += quantity
            self.total_quote_value -= quantity * price + fees
        else:  # SELL
            self.total_base_qty -= quantity
            self.total_quote_value += quantity * price - fees

        self.last_update = datetime.now(UTC)

    def calculate_imbalance(self) -> Decimal:
        """Calculate inventory imbalance ratio (-1 to 1)."""
        if self.total_base_qty == 0:
            return Decimal(0)

        # Normalize by typical position size
        max_position = Decimal(1000)  # Configurable
        imbalance = self.total_base_qty / max_position
        return max(min(imbalance, Decimal(1)), Decimal(-1))


class ManagedOrder:
    """Enhanced order tracking for market making."""

    def __init__(self, order: Order, level: int, target_price: Decimal):
        self.order = order
        self.level = level  # Order book level (0 = best bid/ask)
        self.target_price = target_price  # Original target price
        self.state = OrderState.PENDING
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at
        self.replacement_count = 0
        self.fill_events: list[dict[str, Any]] = []

    def add_fill(self, quantity: Decimal, price: Decimal, timestamp: datetime):
        """Record a fill event."""
        self.fill_events.append(
            {
                "quantity": quantity,
                "price": price,
                "timestamp": timestamp,
                "cumulative_qty": sum(f["quantity"] for f in self.fill_events)
                + quantity,
            }
        )
        self.updated_at = timestamp

    def get_filled_quantity(self) -> Decimal:
        """Get total filled quantity."""
        return sum(f["quantity"] for f in self.fill_events)

    def get_average_fill_price(self) -> Decimal:
        """Calculate volume-weighted average fill price."""
        if not self.fill_events:
            return Decimal(0)

        total_value = sum(f["quantity"] * f["price"] for f in self.fill_events)
        total_quantity = sum(f["quantity"] for f in self.fill_events)

        return total_value / total_quantity if total_quantity > 0 else Decimal(0)


class MarketMakingOrderManager:
    """
    Sophisticated order management system for market making operations.

    Manages multiple limit orders simultaneously across different price levels,
    tracks order states and fills in real-time, handles order replacement on
    significant price moves, and provides comprehensive P&L tracking.
    """

    def __init__(
        self,
        exchange_client: BluefinClient,
        order_manager: OrderManager | None = None,
        symbol: str = "BTC-PERP",
        max_levels: int = 5,
        price_move_threshold: Decimal = Decimal("0.001"),  # 0.1%
        max_position_value: Decimal = Decimal(10000),  # $10k max
        emergency_stop_loss: Decimal = Decimal("0.02"),  # 2% emergency stop
    ):
        """
        Initialize the Market Making Order Manager.

        Args:
            exchange_client: Bluefin exchange client for order operations
            order_manager: Optional existing order manager (creates new if None)
            symbol: Trading symbol (e.g., 'BTC-PERP')
            max_levels: Maximum number of order levels per side
            price_move_threshold: Price change threshold for order replacement
            max_position_value: Maximum position value for risk control
            emergency_stop_loss: Emergency stop loss percentage
        """
        self.exchange_client = exchange_client
        self.order_manager = order_manager or OrderManager()
        self.symbol = symbol
        self.max_levels = max_levels
        self.price_move_threshold = price_move_threshold
        self.max_position_value = max_position_value
        self.emergency_stop_loss = emergency_stop_loss

        # Order tracking
        self.managed_orders: dict[str, ManagedOrder] = {}
        self.orders_by_level: dict[int, dict[str, ManagedOrder]] = defaultdict(
            dict
        )  # level -> {side: order}
        self.active_order_ids: set[str] = set()

        # Price tracking
        self.last_price = Decimal(0)
        self.last_update_time = datetime.now(UTC)
        self.reference_prices: dict[int, Decimal] = {}  # level -> reference price

        # Inventory tracking
        self.inventory = InventoryPosition()
        self.position_limits_breached = False
        self.emergency_mode = False

        # Performance metrics
        self.total_trades = 0
        self.total_volume = Decimal(0)
        self.total_fees_paid = Decimal(0)
        self.round_trip_count = 0

        # Async tasks
        self._monitor_task: asyncio.Task | None = None
        self._running = False

        logger.info(
            "Initialized MarketMakingOrderManager for %s with %d levels, %.3f%% price threshold",
            symbol,
            max_levels,
            float(price_move_threshold * 100),
        )

    async def start(self) -> None:
        """Start the order manager background monitoring."""
        self._running = True
        await self.order_manager.start()

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_orders())

        logger.info("MarketMakingOrderManager started")

    async def stop(self) -> None:
        """Stop the order manager and cancel all orders."""
        self._running = False

        # Cancel monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        # Cancel all active orders
        await self.cancel_all_orders(self.symbol)

        # Stop order manager
        await self.order_manager.stop()

        logger.info("MarketMakingOrderManager stopped")

    async def place_ladder_orders(
        self, levels: list[OrderLevel], symbol: str, current_price: Decimal
    ) -> dict[str, Order | None]:
        """
        Place multiple limit orders across different price levels.

        Args:
            levels: List of OrderLevel objects defining the ladder
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Dictionary mapping level keys to placed orders (or None if failed)
        """
        self.last_price = current_price
        self.last_update_time = datetime.now(UTC)

        placed_orders: dict[str, Order | None] = {}

        try:
            # Group levels by side for organized placement
            bid_levels = [level for level in levels if level.side == "BUY"]
            ask_levels = [level for level in levels if level.side == "SELL"]

            # Sort levels by price (best prices first)
            bid_levels.sort(key=lambda x: x.price, reverse=True)  # Highest bid first
            ask_levels.sort(key=lambda x: x.price)  # Lowest ask first

            # Check inventory imbalance before placing orders
            imbalance = self.inventory.calculate_imbalance()
            logger.info("Current inventory imbalance: %.3f", float(imbalance))

            # Adjust order sizes based on imbalance
            adjusted_levels = self._adjust_levels_for_inventory(
                bid_levels + ask_levels, imbalance
            )

            # Place orders level by level
            for level in adjusted_levels:
                try:
                    # Skip if we already have an order at this level
                    if (
                        level.level in self.orders_by_level
                        and level.side in self.orders_by_level[level.level]
                    ):
                        existing_order = self.orders_by_level[level.level][level.side]
                        if existing_order.state in [
                            OrderState.OPEN,
                            OrderState.PENDING,
                        ]:
                            logger.debug(
                                "Skipping level %d %s - order already exists: %s",
                                level.level,
                                level.side,
                                existing_order.order.id,
                            )
                            continue

                    # Convert size percentage to actual quantity
                    position_value = current_price * Decimal(100)  # Base calculation
                    target_value = position_value * level.size / Decimal(100)
                    quantity = target_value / level.price

                    # Apply minimum size constraints
                    min_quantity = Decimal("0.001")  # Configurable minimum
                    if quantity < min_quantity:
                        logger.warning(
                            "Order quantity %.6f below minimum %.6f for level %d %s",
                            float(quantity),
                            float(min_quantity),
                            level.level,
                            level.side,
                        )
                        continue

                    # Place the limit order
                    order = await self.exchange_client.place_limit_order(
                        symbol=symbol,
                        side=level.side,  # type: ignore
                        quantity=quantity,
                        price=level.price,
                    )

                    if order:
                        # Create managed order wrapper
                        managed_order = ManagedOrder(order, level.level, level.price)
                        managed_order.state = OrderState.SUBMITTED

                        # Track the order
                        self.managed_orders[order.id] = managed_order
                        self.orders_by_level[level.level][level.side] = managed_order
                        self.active_order_ids.add(order.id)
                        self.reference_prices[level.level] = current_price

                        # Register callback for fill monitoring
                        def create_callback():
                            def callback(oid: str, event: OrderEvent) -> None:
                                # Create async task to handle order event
                                asyncio.create_task(
                                    self._handle_order_event(oid, event)
                                )

                            return callback

                        self.order_manager.register_callback(
                            order.id, create_callback()
                        )

                        placed_orders[f"{level.side}_{level.level}"] = order

                        logger.info(
                            "Placed %s order at level %d: %.6f %s @ %.6f (ID: %s)",
                            level.side,
                            level.level,
                            float(quantity),
                            symbol,
                            float(level.price),
                            order.id,
                        )
                    else:
                        placed_orders[f"{level.side}_{level.level}"] = None
                        logger.error(
                            "Failed to place %s order at level %d: %.6f @ %.6f",
                            level.side,
                            level.level,
                            float(quantity),
                            float(level.price),
                        )

                except Exception as e:
                    logger.exception(
                        "Error placing order for level %d %s: %s",
                        level.level,
                        level.side,
                        e,
                    )
                    placed_orders[f"{level.side}_{level.level}"] = None

                    # Add small delay between orders to avoid rate limits
                    await asyncio.sleep(0.1)

            logger.info(
                "Placed %d/%d ladder orders successfully",
                len([o for o in placed_orders.values() if o is not None]),
                len(levels),
            )

        except Exception as e:
            logger.exception("Error placing ladder orders: %s", e)

        return placed_orders

    async def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancel all active orders for the symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if all orders cancelled successfully
        """
        try:
            # Get all active orders for this symbol
            orders_to_cancel = [
                order_id
                for order_id in self.active_order_ids
                if order_id in self.managed_orders
                and self.managed_orders[order_id].order.symbol == symbol
                and self.managed_orders[order_id].state
                in [OrderState.OPEN, OrderState.PENDING, OrderState.PARTIALLY_FILLED]
            ]

            if not orders_to_cancel:
                logger.info("No active orders to cancel for %s", symbol)
                return True

            logger.info("Cancelling %d orders for %s", len(orders_to_cancel), symbol)

            # Cancel orders via exchange
            success_count = 0
            for order_id in orders_to_cancel:
                try:
                    success = await self.exchange_client.cancel_order(order_id)
                    if success:
                        # Update managed order state
                        if order_id in self.managed_orders:
                            self.managed_orders[order_id].state = OrderState.CANCELLED
                            self.managed_orders[order_id].updated_at = datetime.now(UTC)
                        success_count += 1
                        logger.debug("Cancelled order %s", order_id)
                    else:
                        logger.warning("Failed to cancel order %s", order_id)

                except Exception as e:
                    logger.exception("Error cancelling order %s: %s", order_id, e)

            # Clean up tracking
            self._cleanup_cancelled_orders()

            logger.info(
                "Cancelled %d/%d orders for %s",
                success_count,
                len(orders_to_cancel),
                symbol,
            )

            return success_count == len(orders_to_cancel)

        except Exception as e:
            logger.exception("Error cancelling all orders: %s", e)
            return False

    async def update_orders_on_price_move(
        self, new_price: Decimal, threshold: Decimal | None = None
    ) -> bool:
        """
        Update orders when price moves beyond threshold.

        Args:
            new_price: New market price
            threshold: Price change threshold (uses default if None)

        Returns:
            True if orders were successfully updated
        """
        if threshold is None:
            threshold = self.price_move_threshold

        try:
            # Calculate price change
            if self.last_price == 0:
                self.last_price = new_price
                return True

            price_change = abs(new_price - self.last_price) / self.last_price

            if price_change < threshold:
                return True  # No update needed

            logger.info(
                "Price moved %.3f%% (%.6f -> %.6f), updating orders",
                float(price_change * 100),
                float(self.last_price),
                float(new_price),
            )

            # Cancel existing orders that are far from market
            orders_to_replace = []
            for managed_order in self.managed_orders.values():
                if managed_order.state not in [OrderState.OPEN, OrderState.PENDING]:
                    continue

                order = managed_order.order
                price_diff = (
                    abs(order.price - new_price) / new_price
                    if order.price
                    else Decimal(1)
                )

                # Replace orders that are more than 2x threshold away from market
                if price_diff > threshold * 2:
                    orders_to_replace.append(managed_order)

            # Cancel orders that need replacement
            if orders_to_replace:
                logger.info(
                    "Replacing %d orders due to price move", len(orders_to_replace)
                )

                for managed_order in orders_to_replace:
                    try:
                        # Mark as replacing
                        managed_order.state = OrderState.REPLACING
                        managed_order.replacement_count += 1

                        # Cancel the order
                        success = await self.exchange_client.cancel_order(
                            managed_order.order.id
                        )
                        if success:
                            logger.debug(
                                "Cancelled order %s for replacement",
                                managed_order.order.id,
                            )
                        else:
                            logger.warning(
                                "Failed to cancel order %s", managed_order.order.id
                            )

                    except Exception:
                        logger.exception(
                            "Error replacing order %s", managed_order.order.id
                        )

            # Update reference price
            self.last_price = new_price
            self.last_update_time = datetime.now(UTC)

            return True

        except Exception:
            logger.exception("Error updating orders on price move")
            return False

    async def track_order_fills(self) -> dict[str, Any]:
        """
        Track and analyze recent order fills.

        Returns:
            Dictionary with fill statistics and performance metrics
        """
        try:
            current_time = datetime.now(UTC)
            recent_cutoff = current_time - timedelta(hours=1)

            # Collect recent fills
            recent_fills = []
            total_volume_1h = Decimal(0)
            total_fees_1h = Decimal(0)
            buy_volume = Decimal(0)
            sell_volume = Decimal(0)

            for managed_order in self.managed_orders.values():
                for fill in managed_order.fill_events:
                    if fill["timestamp"] >= recent_cutoff:
                        recent_fills.append(
                            {
                                "order_id": managed_order.order.id,
                                "side": managed_order.order.side,
                                "level": managed_order.level,
                                "quantity": fill["quantity"],
                                "price": fill["price"],
                                "timestamp": fill["timestamp"],
                                "value": fill["quantity"] * fill["price"],
                            }
                        )

                        volume = fill["quantity"] * fill["price"]
                        total_volume_1h += volume

                        # Estimate fees (would use actual fees from exchange in production)
                        estimated_fee = volume * Decimal("0.0001")  # 0.01% maker fee
                        total_fees_1h += estimated_fee

                        if managed_order.order.side == "BUY":
                            buy_volume += volume
                        else:
                            sell_volume += volume

            # Calculate metrics
            fill_rate = len(recent_fills) / max(len(self.active_order_ids), 1)
            volume_imbalance = (
                (buy_volume - sell_volume) / (buy_volume + sell_volume)
                if (buy_volume + sell_volume) > 0
                else Decimal(0)
            )

            # Update inventory with recent fills
            for fill in recent_fills:
                self.inventory.update_fill(
                    fill["side"],
                    fill["quantity"],
                    fill["price"],
                    fill["value"] * Decimal("0.0001"),  # Estimated fee
                )

            metrics = {
                "recent_fills_count": len(recent_fills),
                "total_volume_1h": float(total_volume_1h),
                "total_fees_1h": float(total_fees_1h),
                "fill_rate": float(fill_rate),
                "volume_imbalance": float(volume_imbalance),
                "inventory_imbalance": float(self.inventory.calculate_imbalance()),
                "active_orders": len(self.active_order_ids),
                "managed_orders": len(self.managed_orders),
                "last_update": self.last_update_time.isoformat(),
                "unrealized_pnl": float(self._calculate_unrealized_pnl()),
                "realized_pnl": float(self.inventory.realized_pnl),
            }

            logger.debug("Order fill tracking: %s", metrics)
            return metrics

        except Exception as e:
            logger.exception("Error tracking order fills: %s", e)
            return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    async def manage_inventory_imbalance(self) -> bool:
        """
        Manage inventory imbalance through order adjustments.

        Returns:
            True if imbalance was successfully managed
        """
        try:
            imbalance = self.inventory.calculate_imbalance()

            # Threshold for action (10% imbalance)
            action_threshold = Decimal("0.1")

            if abs(imbalance) < action_threshold:
                return True  # No action needed

            logger.info(
                "Managing inventory imbalance: %.3f (threshold: %.3f)",
                float(imbalance),
                float(action_threshold),
            )

            # Adjust future order placement bias
            if imbalance > action_threshold:
                # Too long - reduce bid sizes, increase ask sizes
                bias_adjustment = min(imbalance, Decimal("0.5"))
                logger.info(
                    "Reducing bid sizes by %.1f%% due to long inventory",
                    float(bias_adjustment * 100),
                )

            elif imbalance < -action_threshold:
                # Too short - increase bid sizes, reduce ask sizes
                bias_adjustment = max(imbalance, Decimal("-0.5"))
                logger.info(
                    "Reducing ask sizes by %.1f%% due to short inventory",
                    float(abs(bias_adjustment) * 100),
                )

            # Emergency inventory management
            emergency_threshold = Decimal("0.5")  # 50% imbalance
            if abs(imbalance) > emergency_threshold:
                logger.warning(
                    "EMERGENCY: Inventory imbalance %.3f exceeds threshold %.3f",
                    float(imbalance),
                    float(emergency_threshold),
                )

                # Could trigger emergency position reduction here
                await self._emergency_inventory_management(imbalance)

            return True

        except Exception as e:
            logger.exception("Error managing inventory imbalance: %s", e)
            return False

    def get_current_pnl(self) -> dict[str, Decimal]:
        """
        Calculate current P&L including fees.

        Returns:
            Dictionary with realized and unrealized P&L
        """
        try:
            realized_pnl = self.inventory.realized_pnl
            unrealized_pnl = self._calculate_unrealized_pnl()
            total_pnl = realized_pnl + unrealized_pnl

            return {
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "total_fees": self.total_fees_paid,
                "net_pnl": total_pnl - self.total_fees_paid,
            }

        except Exception as e:
            logger.exception("Error calculating P&L: %s", e)
            return {
                "realized_pnl": Decimal(0),
                "unrealized_pnl": Decimal(0),
                "total_pnl": Decimal(0),
                "total_fees": Decimal(0),
                "net_pnl": Decimal(0),
            }

    async def emergency_position_management(self) -> bool:
        """
        Emergency position management for risk control.

        Returns:
            True if emergency measures were successful
        """
        try:
            # Check if emergency mode is needed
            pnl = self.get_current_pnl()
            position_value = abs(self.inventory.total_base_qty * self.last_price)

            # Emergency triggers
            emergency_conditions = [
                # Large unrealized loss
                pnl["unrealized_pnl"]
                < -self.emergency_stop_loss * self.max_position_value,
                # Position size too large
                position_value > self.max_position_value,
                # Net loss exceeds limit
                pnl["net_pnl"] < -self.emergency_stop_loss * self.max_position_value,
            ]

            if not any(emergency_conditions):
                return True  # No emergency action needed

            logger.critical(
                "EMERGENCY POSITION MANAGEMENT TRIGGERED - "
                "PnL: %.2f, Position Value: %.2f, Max: %.2f",
                float(pnl["net_pnl"]),
                float(position_value),
                float(self.max_position_value),
            )

            self.emergency_mode = True

            # 1. Cancel all open orders immediately
            await self.cancel_all_orders(self.symbol)

            # 2. Close position if needed
            if abs(self.inventory.total_base_qty) > Decimal("0.001"):
                await self._emergency_close_position()

            # 3. Set position limits flag
            self.position_limits_breached = True

            logger.critical("Emergency position management completed")
            return True

        except Exception as e:
            logger.exception("Error in emergency position management: %s", e)
            return False

    # Private helper methods

    def _adjust_levels_for_inventory(
        self, levels: list[OrderLevel], imbalance: Decimal
    ) -> list[OrderLevel]:
        """Adjust order levels based on inventory imbalance."""
        adjusted_levels = []

        for level in levels:
            # Calculate size adjustment based on imbalance
            if level.side == "BUY" and imbalance > Decimal("0.1"):
                # Reduce bid sizes when long
                size_adjustment = max(Decimal("0.5"), 1 - imbalance)
            elif level.side == "SELL" and imbalance < Decimal("-0.1"):
                # Reduce ask sizes when short
                size_adjustment = max(Decimal("0.5"), 1 + imbalance)
            else:
                size_adjustment = Decimal(1)

            adjusted_size = level.size * size_adjustment

            adjusted_levels.append(
                OrderLevel(
                    side=level.side,
                    price=level.price,
                    size=adjusted_size,
                    level=level.level,
                )
            )

        return adjusted_levels

    def _calculate_unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L based on current position and last price."""
        if self.inventory.total_base_qty == 0 or self.last_price == 0:
            return Decimal(0)

        # Simplified calculation - would use mark price in production
        position_value = self.inventory.total_base_qty * self.last_price
        return position_value + self.inventory.total_quote_value

    async def _handle_order_event(self, order_id: str, event: OrderEvent) -> None:
        """Handle order lifecycle events."""
        try:
            if order_id not in self.managed_orders:
                return

            managed_order = self.managed_orders[order_id]

            if event == OrderEvent.FILLED:
                managed_order.state = OrderState.FILLED
                # Record the fill
                order = managed_order.order
                managed_order.add_fill(
                    order.filled_quantity,
                    order.price or self.last_price,
                    datetime.now(UTC),
                )

                self.total_trades += 1
                self.total_volume += order.filled_quantity * (
                    order.price or self.last_price
                )

                logger.info(
                    "Order filled: %s - %.6f %s @ %.6f",
                    order_id,
                    float(order.filled_quantity),
                    order.side,
                    float(order.price or self.last_price),
                )

            elif event == OrderEvent.PARTIALLY_FILLED:
                managed_order.state = OrderState.PARTIALLY_FILLED

            elif event in [
                OrderEvent.CANCELLED,
                OrderEvent.REJECTED,
                OrderEvent.FAILED,
            ]:
                managed_order.state = OrderState.CANCELLED

            managed_order.updated_at = datetime.now(UTC)

        except Exception as e:
            logger.exception(
                "Error handling order event %s for %s: %s", event, order_id, e
            )

    def _cleanup_cancelled_orders(self) -> None:
        """Clean up cancelled and completed orders from tracking."""
        cleanup_states = [
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.FAILED,
        ]

        orders_to_remove = [
            order_id
            for order_id, managed_order in self.managed_orders.items()
            if managed_order.state in cleanup_states
            and managed_order.updated_at < datetime.now(UTC) - timedelta(minutes=5)
        ]

        for order_id in orders_to_remove:
            managed_order = self.managed_orders.pop(order_id, None)
            if managed_order and order_id in self.active_order_ids:
                self.active_order_ids.remove(order_id)

                # Remove from level tracking
                level = managed_order.level
                side = managed_order.order.side
                if (
                    level in self.orders_by_level
                    and side in self.orders_by_level[level]
                ) and self.orders_by_level[level][side] == managed_order:
                    del self.orders_by_level[level][side]

        if orders_to_remove:
            logger.debug("Cleaned up %d completed orders", len(orders_to_remove))

    async def _monitor_orders(self) -> None:
        """Background task to monitor order states and performance."""
        while self._running:
            try:
                await asyncio.sleep(5)  # Monitor every 5 seconds

                # Track fills
                await self.track_order_fills()

                # Check for inventory imbalance
                await self.manage_inventory_imbalance()

                # Check for emergency conditions
                await self.emergency_position_management()

                # Clean up old orders
                self._cleanup_cancelled_orders()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in order monitoring: %s", e)
                await asyncio.sleep(1)

    async def _emergency_close_position(self) -> None:
        """Emergency position closure."""
        try:
            if abs(self.inventory.total_base_qty) < Decimal("0.001"):
                return

            # Determine side for closing
            close_side = "SELL" if self.inventory.total_base_qty > 0 else "BUY"

            close_quantity = abs(self.inventory.total_base_qty)

            logger.critical(
                "EMERGENCY CLOSE: %s %.6f %s at market",
                close_side,
                float(close_quantity),
                self.symbol,
            )

            # Place market order to close
            close_order = await self.exchange_client.place_market_order(
                symbol=self.symbol,
                side=close_side,  # type: ignore
                quantity=close_quantity,
            )

            if close_order:
                logger.critical("Emergency close order placed: %s", close_order.id)
            else:
                logger.critical("FAILED to place emergency close order")

        except Exception as e:
            logger.exception("Error in emergency position closure: %s", e)

    async def _emergency_inventory_management(self, imbalance: Decimal) -> None:
        """Handle emergency inventory imbalance."""
        try:
            # Calculate position to close (50% of imbalance)
            position_to_close = self.inventory.total_base_qty * Decimal("0.5")

            if abs(position_to_close) < Decimal("0.001"):
                return

            logger.warning(
                "Emergency inventory management: closing %.6f position",
                float(abs(position_to_close)),
            )

            # Place market order to reduce imbalance
            side = "SELL" if position_to_close > 0 else "BUY"

            emergency_order = await self.exchange_client.place_market_order(
                symbol=self.symbol,
                side=side,  # type: ignore
                quantity=abs(position_to_close),
            )

            if emergency_order:
                logger.warning(
                    "Emergency inventory order placed: %s", emergency_order.id
                )

        except Exception as e:
            logger.exception("Error in emergency inventory management: %s", e)
