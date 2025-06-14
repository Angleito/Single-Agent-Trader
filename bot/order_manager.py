"""
Order management system for tracking order lifecycle.

This module handles order tracking, fill monitoring, timeout management,
and order state persistence for the trading bot.
"""

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, UTC
from decimal import Decimal
from enum import Enum
from pathlib import Path

from .config import settings
from .types import Order, OrderStatus

logger = logging.getLogger(__name__)


class OrderEvent(str, Enum):
    """Order lifecycle events."""

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class OrderManager:
    """
    Order management system for tracking order lifecycle.

    Manages pending orders, tracks fills, handles timeouts, and provides
    order status monitoring for the trading bot.
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the order manager.

        Args:
            data_dir: Directory for state persistence (default: data/orders)
        """
        self.data_dir = data_dir or Path("data/orders")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe order storage
        self._active_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []
        self._order_callbacks: dict[str, list[Callable]] = {}
        self._lock = threading.RLock()

        # Timeout tracking
        self._timeout_tasks: dict[str, asyncio.Task] = {}
        self._running = False

        # State file paths
        self.orders_file = self.data_dir / "active_orders.json"
        self.history_file = self.data_dir / "order_history.json"

        # Load persisted state
        self._load_state()

        logger.info(
            f"Initialized OrderManager with {len(self._active_orders)} active orders"
        )

    async def start(self) -> None:
        """Start the order manager background tasks."""
        self._running = True
        logger.info("OrderManager started")

    async def stop(self) -> None:
        """Stop the order manager and cancel all background tasks."""
        self._running = False

        # Cancel all timeout tasks
        for task in self._timeout_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._timeout_tasks:
            await asyncio.gather(*self._timeout_tasks.values(), return_exceptions=True)

        self._timeout_tasks.clear()
        logger.info("OrderManager stopped")

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        timeout_seconds: int | None = None,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/STOP/STOP_LIMIT)
            quantity: Order quantity
            price: Limit price (for LIMIT/STOP_LIMIT orders)
            stop_price: Stop price (for STOP/STOP_LIMIT orders)
            timeout_seconds: Order timeout (default: from settings)

        Returns:
            Created order
        """
        with self._lock:
            # Generate unique order ID
            order_id = f"order_{datetime.now(UTC).timestamp()}_{symbol}_{side}"

            # Create order object
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(UTC),
                filled_quantity=Decimal("0"),
            )

            # Add to active orders
            self._active_orders[order_id] = order

            # Set up timeout if specified
            if timeout_seconds is None:
                timeout_seconds = settings.trading.order_timeout_seconds

            if timeout_seconds > 0 and self._running:
                self._schedule_timeout(order_id, timeout_seconds)

            # Trigger callbacks
            self._trigger_callbacks(order_id, OrderEvent.CREATED)

            # Persist state
            self._save_state()

            logger.info(
                f"Created order {order_id}: {side} {quantity} {symbol} ({order_type})"
            )
            return order.copy()

    def add_order(self, order: Order) -> None:
        """
        Add an existing order to the manager.
        This method is primarily for testing and integration scenarios
        where an order object is already created and needs to be tracked.
        Args:
            order: Order object to add
        """
        with self._lock:
            # Add to active orders if not completed
            if order.status in [
                OrderStatus.PENDING,
                OrderStatus.OPEN,
            ]:
                self._active_orders[order.id] = order
                self._trigger_callbacks(order.id, OrderEvent.CREATED)
            else:
                # Add to history if already completed
                self._order_history.append(order)
            # Persist state
            self._save_state()
            logger.info(f"Added existing order {order.id} with status {order.status}")

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: Decimal | None = None,
        fill_price: Decimal | None = None,
    ) -> Order | None:
        """
        Update order status and fill information.

        Args:
            order_id: Order ID
            status: New order status
            filled_quantity: Filled quantity (if partial/full fill)
            fill_price: Fill price

        Returns:
            Updated order or None if not found
        """
        with self._lock:
            if order_id not in self._active_orders:
                logger.warning(f"Order {order_id} not found for status update")
                return None

            order = self._active_orders[order_id]
            old_status = order.status

            # Update order fields
            order.status = status
            order.timestamp = datetime.now(UTC)

            if filled_quantity is not None:
                order.filled_quantity = filled_quantity

            # Determine event type
            event = None
            if status == OrderStatus.FILLED:
                event = OrderEvent.FILLED
            elif status == OrderStatus.CANCELLED:
                event = OrderEvent.CANCELLED
            elif status == OrderStatus.REJECTED:
                event = OrderEvent.REJECTED
            elif status == OrderStatus.FAILED:
                event = OrderEvent.FAILED
            elif status == OrderStatus.OPEN and old_status == OrderStatus.PENDING:
                event = OrderEvent.SUBMITTED
            elif (
                filled_quantity
                and filled_quantity > Decimal("0")
                and filled_quantity < order.quantity
            ):
                event = OrderEvent.PARTIALLY_FILLED

            # Handle order completion
            if status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.FAILED,
            ]:
                # Move to history
                completed_order = self._active_orders.pop(order_id)
                self._order_history.append(completed_order)

                # Cancel timeout task
                if order_id in self._timeout_tasks:
                    self._timeout_tasks[order_id].cancel()
                    del self._timeout_tasks[order_id]

                logger.info(f"Order {order_id} completed with status {status}")

            # Trigger callbacks
            if event:
                self._trigger_callbacks(order_id, event)

            # Persist state
            self._save_state()

            return order.copy()

    def get_order(self, order_id: str) -> Order | None:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        with self._lock:
            if order_id in self._active_orders:
                return self._active_orders[order_id].copy()

            # Check history
            for order in self._order_history:
                if order.id == order_id:
                    return order.copy()

            return None

    def get_active_orders(self, symbol: str | None = None) -> list[Order]:
        """
        Get all active orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of active orders
        """
        with self._lock:
            orders = list(self._active_orders.values())

            if symbol:
                orders = [order for order in orders if order.symbol == symbol]

            return [order.copy() for order in orders]

    def get_orders_by_status(
        self, status: OrderStatus, symbol: str | None = None
    ) -> list[Order]:
        """
        Get orders by status.

        Args:
            status: Order status to filter by
            symbol: Filter by symbol (optional)

        Returns:
            List of orders matching criteria
        """
        with self._lock:
            orders = []

            # Check active orders
            for order in self._active_orders.values():
                if order.status == status:
                    if symbol is None or order.symbol == symbol:
                        orders.append(order.copy())

            # Check recent history for completed orders
            if status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.FAILED,
            ]:
                cutoff_time = datetime.now(UTC) - timedelta(hours=24)
                for order in self._order_history:
                    if order.status == status and order.timestamp >= cutoff_time:
                        if symbol is None or order.symbol == symbol:
                            orders.append(order.copy())

            return orders

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        with self._lock:
            if order_id not in self._active_orders:
                logger.warning(
                    f"Cannot cancel order {order_id}: not found in active orders"
                )
                return False

            order = self._active_orders[order_id]

            if order.status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.FAILED,
            ]:
                logger.warning(
                    f"Cannot cancel order {order_id}: already completed with status {order.status}"
                )
                return False

            # Update status to cancelled
            self.update_order_status(order_id, OrderStatus.CANCELLED)

            logger.info(f"Order {order_id} cancelled")
            return True

    def cancel_all_orders(self, symbol: str | None = None) -> int:
        """
        Cancel all active orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            Number of orders cancelled
        """
        with self._lock:
            orders_to_cancel = []

            for order_id, order in self._active_orders.items():
                if symbol is None or order.symbol == symbol:
                    if order.status not in [
                        OrderStatus.FILLED,
                        OrderStatus.CANCELLED,
                        OrderStatus.REJECTED,
                        OrderStatus.FAILED,
                    ]:
                        orders_to_cancel.append(order_id)

            cancelled_count = 0
            for order_id in orders_to_cancel:
                if self.cancel_order(order_id):
                    cancelled_count += 1

            logger.info(
                f"Cancelled {cancelled_count} orders"
                + (f" for {symbol}" if symbol else "")
            )
            return cancelled_count

    def register_callback(
        self, order_id: str, callback: Callable[[str, OrderEvent], None]
    ) -> None:
        """
        Register a callback for order events.

        Args:
            order_id: Order ID to monitor
            callback: Callback function (order_id, event)
        """
        with self._lock:
            if order_id not in self._order_callbacks:
                self._order_callbacks[order_id] = []
            self._order_callbacks[order_id].append(callback)

    def get_order_statistics(
        self, symbol: str | None = None, hours: int = 24
    ) -> dict[str, any]:
        """
        Get order statistics.

        Args:
            symbol: Filter by symbol (optional)
            hours: Time window in hours

        Returns:
            Dictionary with order statistics
        """
        with self._lock:
            cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

            # Collect orders from history
            recent_orders = []
            for order in self._order_history:
                if order.timestamp >= cutoff_time:
                    if symbol is None or order.symbol == symbol:
                        recent_orders.append(order)

            # Add active orders
            for order in self._active_orders.values():
                if order.timestamp >= cutoff_time:
                    if symbol is None or order.symbol == symbol:
                        recent_orders.append(order)

            # Calculate statistics
            total_orders = len(recent_orders)
            filled_orders = len(
                [o for o in recent_orders if o.status == OrderStatus.FILLED]
            )
            cancelled_orders = len(
                [o for o in recent_orders if o.status == OrderStatus.CANCELLED]
            )
            rejected_orders = len(
                [o for o in recent_orders if o.status == OrderStatus.REJECTED]
            )
            failed_orders = len(
                [o for o in recent_orders if o.status == OrderStatus.FAILED]
            )
            pending_orders = len(
                [
                    o
                    for o in recent_orders
                    if o.status in [OrderStatus.PENDING, OrderStatus.OPEN]
                ]
            )

            # Calculate fill rate
            fill_rate = (
                (filled_orders / total_orders * 100) if total_orders > 0 else 0.0
            )

            # Calculate average fill time for filled orders
            fill_times = []
            for order in recent_orders:
                if order.status == OrderStatus.FILLED:
                    # Simplified - would need more detailed tracking for accurate fill time
                    fill_times.append(30.0)  # Placeholder

            avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0.0

            return {
                "total_orders": total_orders,
                "filled_orders": filled_orders,
                "cancelled_orders": cancelled_orders,
                "rejected_orders": rejected_orders,
                "failed_orders": failed_orders,
                "pending_orders": pending_orders,
                "fill_rate_pct": fill_rate,
                "avg_fill_time_seconds": avg_fill_time,
                "time_window_hours": hours,
                "symbol": symbol,
            }

    def _schedule_timeout(self, order_id: str, timeout_seconds: int) -> None:
        """
        Schedule order timeout.

        Args:
            order_id: Order ID
            timeout_seconds: Timeout in seconds
        """

        async def timeout_handler():
            try:
                await asyncio.sleep(timeout_seconds)

                with self._lock:
                    if order_id in self._active_orders:
                        order = self._active_orders[order_id]
                        if order.status in [OrderStatus.PENDING, OrderStatus.OPEN]:
                            logger.warning(
                                f"Order {order_id} timed out after {timeout_seconds}s"
                            )
                            self.update_order_status(order_id, OrderStatus.CANCELLED)
                            self._trigger_callbacks(order_id, OrderEvent.EXPIRED)

            except asyncio.CancelledError:
                pass  # Expected when order completes before timeout
            finally:
                if order_id in self._timeout_tasks:
                    del self._timeout_tasks[order_id]

        # Schedule the timeout task
        if self._running:
            task = asyncio.create_task(timeout_handler())
            self._timeout_tasks[order_id] = task

    def _trigger_callbacks(self, order_id: str, event: OrderEvent) -> None:
        """
        Trigger registered callbacks for an order event.

        Args:
            order_id: Order ID
            event: Order event
        """
        if order_id in self._order_callbacks:
            for callback in self._order_callbacks[order_id]:
                try:
                    callback(order_id, event)
                except Exception as e:
                    logger.error(f"Order callback error for {order_id}: {e}")

            # Clean up callbacks for completed orders
            if event in [
                OrderEvent.FILLED,
                OrderEvent.CANCELLED,
                OrderEvent.REJECTED,
                OrderEvent.FAILED,
                OrderEvent.EXPIRED,
            ]:
                del self._order_callbacks[order_id]

    def _save_state(self) -> None:
        """Save current state to files."""
        try:
            # Save active orders
            orders_data = {}
            for order_id, order in self._active_orders.items():
                orders_data[order_id] = {
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.type,
                    "quantity": str(order.quantity),
                    "price": str(order.price) if order.price else None,
                    "stop_price": str(order.stop_price) if order.stop_price else None,
                    "status": order.status.value,
                    "timestamp": order.timestamp.isoformat(),
                    "filled_quantity": str(order.filled_quantity),
                }

            with open(self.orders_file, "w") as f:
                json.dump(orders_data, f, indent=2)

            # Save order history (last 500 entries)
            history_data = []
            for order in self._order_history[-500:]:
                history_data.append(
                    {
                        "id": order.id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "type": order.type,
                        "quantity": str(order.quantity),
                        "price": str(order.price) if order.price else None,
                        "stop_price": (
                            str(order.stop_price) if order.stop_price else None
                        ),
                        "status": order.status.value,
                        "timestamp": order.timestamp.isoformat(),
                        "filled_quantity": str(order.filled_quantity),
                    }
                )

            with open(self.history_file, "w") as f:
                json.dump(history_data, f, indent=2)

            logger.debug("Order state saved successfully")

        except Exception as e:
            logger.error(f"Failed to save order state: {e}")

    def _load_state(self) -> None:
        """Load state from files."""
        try:
            # Load active orders
            if self.orders_file.exists():
                with open(self.orders_file) as f:
                    orders_data = json.load(f)

                for order_id, order_data in orders_data.items():
                    self._active_orders[order_id] = Order(
                        id=order_data["id"],
                        symbol=order_data["symbol"],
                        side=order_data["side"],
                        type=order_data["type"],
                        quantity=Decimal(order_data["quantity"]),
                        price=(
                            Decimal(order_data["price"])
                            if order_data["price"]
                            else None
                        ),
                        stop_price=(
                            Decimal(order_data["stop_price"])
                            if order_data["stop_price"]
                            else None
                        ),
                        status=OrderStatus(order_data["status"]),
                        timestamp=datetime.fromisoformat(order_data["timestamp"]),
                        filled_quantity=Decimal(order_data["filled_quantity"]),
                    )

                logger.info(f"Loaded {len(self._active_orders)} active orders")

            # Load order history
            if self.history_file.exists():
                with open(self.history_file) as f:
                    history_data = json.load(f)

                for order_data in history_data:
                    self._order_history.append(
                        Order(
                            id=order_data["id"],
                            symbol=order_data["symbol"],
                            side=order_data["side"],
                            type=order_data["type"],
                            quantity=Decimal(order_data["quantity"]),
                            price=(
                                Decimal(order_data["price"])
                                if order_data["price"]
                                else None
                            ),
                            stop_price=(
                                Decimal(order_data["stop_price"])
                                if order_data["stop_price"]
                                else None
                            ),
                            status=OrderStatus(order_data["status"]),
                            timestamp=datetime.fromisoformat(order_data["timestamp"]),
                            filled_quantity=Decimal(order_data["filled_quantity"]),
                        )
                    )

                logger.info(f"Loaded {len(self._order_history)} historical orders")

        except Exception as e:
            logger.error(f"Failed to load order state: {e}")
            # Continue with empty state

    def clear_old_history(self, days_to_keep: int = 7) -> None:
        """
        Clear old order history.

        Args:
            days_to_keep: Number of days to keep in history
        """
        with self._lock:
            cutoff_date = datetime.now(UTC) - timedelta(days=days_to_keep)

            original_count = len(self._order_history)
            self._order_history = [
                order for order in self._order_history if order.timestamp >= cutoff_date
            ]

            removed_count = original_count - len(self._order_history)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old order records")
                self._save_state()

    def reset_orders(self) -> None:
        """Reset all orders (for testing/emergency use)."""
        with self._lock:
            # Cancel all timeout tasks
            for task in self._timeout_tasks.values():
                task.cancel()
            self._timeout_tasks.clear()

            # Clear order data
            self._active_orders.clear()
            self._order_history.clear()
            self._order_callbacks.clear()

            self._save_state()
            logger.warning("All orders have been reset")
