"""
Backwards-Compatible Order Manager Adapter

This module provides a backwards-compatible adapter that wraps the functional
order management system while preserving the existing imperative APIs.
This allows gradual migration without breaking existing code.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from ..config import settings
from ..trading_types import Order, OrderStatus
from ..utils.path_utils import get_data_directory, get_data_file_path
from .orders import (
    FunctionalOrderManager,
    OrderFill,
    OrderId,
    OrderParameters,
    OrderState,
    create_order,
)

logger = logging.getLogger(__name__)


class CompatibilityOrderManager:
    """
    Backwards-compatible order manager that wraps the functional implementation.

    This adapter maintains the same interface as the original OrderManager class
    while using the new functional order management system underneath.
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the compatibility order manager.

        Args:
            data_dir: Directory for state persistence (default: uses fallback-aware data/orders)
        """
        if data_dir:
            self.data_dir = data_dir
            self.data_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use fallback-aware path utilities
            self.data_dir = get_data_directory() / "orders"
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize functional order manager
        self._functional_manager = FunctionalOrderManager.create()

        # Compatibility layer state
        self._order_callbacks: dict[str, list[Callable]] = {}
        self._timeout_tasks: dict[str, asyncio.Task] = {}
        self._running = False

        # State file paths - use fallback-aware utilities
        if data_dir:
            # Use provided data_dir
            self.orders_file = self.data_dir / "active_orders.json"
            self.history_file = self.data_dir / "order_history.json"
        else:
            # Use fallback-aware file paths
            self.orders_file = get_data_file_path("orders/active_orders.json")
            self.history_file = get_data_file_path("orders/order_history.json")

        # Load persisted state if available
        self._load_legacy_state()

        logger.info(
            "Initialized CompatibilityOrderManager with %s active orders",
            len(self._functional_manager.state.active_orders),
        )

    async def start(self) -> None:
        """Start the order manager background tasks."""
        self._running = True
        logger.info("CompatibilityOrderManager started")

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
        logger.info("CompatibilityOrderManager stopped")

    def create_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        order_type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"],
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        timeout_seconds: int | None = None,
    ) -> Order:
        """
        Create a new order using the legacy API.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/STOP/STOP_LIMIT)
            quantity: Order quantity
            price: Limit price (for LIMIT/STOP_LIMIT orders)
            stop_price: Stop price (for STOP/STOP_LIMIT orders)
            timeout_seconds: Order timeout (default: from settings)

        Returns:
            Created order in legacy format
        """
        # Create functional order parameters
        params_result = OrderParameters.create(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=float(quantity),
            price=float(price) if price else None,
            stop_price=float(stop_price) if stop_price else None,
            timeout_seconds=timeout_seconds,
        )

        if hasattr(params_result, "error"):  # It's a Failure
            raise ValueError(f"Invalid order parameters: {params_result.error}")

        parameters = params_result.value

        # Create order using functional system
        creation_result = self._functional_manager.create_order_effect(parameters)

        try:
            updated_manager, new_order = creation_result.run()
            self._functional_manager = updated_manager
        except Exception as e:
            logger.error("Failed to create order: %s", str(e))
            raise ValueError(f"Order creation failed: {e!s}")

        # Set up timeout if specified
        if timeout_seconds is None:
            timeout_seconds = settings.trading.order_timeout_seconds

        if timeout_seconds > 0 and self._running:
            self._schedule_timeout(str(new_order.id), timeout_seconds)

        # Trigger callbacks
        self._trigger_callbacks(str(new_order.id), "CREATED")

        # Persist state
        self._save_state()

        logger.info(
            "Created order %s: %s %s %s (%s)",
            new_order.id,
            side,
            quantity,
            symbol,
            order_type,
        )

        return self._convert_to_legacy_order(new_order)

    def add_order(self, order: Order) -> None:
        """
        Add an existing order to the manager.
        This method is primarily for testing and integration scenarios.

        Args:
            order: Legacy order object to add
        """
        # Convert legacy order to functional order
        functional_order = self._convert_from_legacy_order(order)

        # Add to functional manager
        updated_state = self._functional_manager.state.add_order(functional_order)
        self._functional_manager = self._functional_manager.with_state(updated_state)

        # Trigger callbacks
        self._trigger_callbacks(order.id, "CREATED")

        # Persist state
        self._save_state()

        logger.info("Added existing order %s with status %s", order.id, order.status)

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: Decimal | None = None,
        _fill_price: Decimal | None = None,
    ) -> Order | None:
        """
        Update order status and fill information.

        Args:
            order_id: Order ID
            status: New order status
            filled_quantity: Filled quantity (if partial/full fill)
            _fill_price: Fill price (legacy parameter, not used)

        Returns:
            Updated order in legacy format or None if not found
        """
        functional_order_id = OrderId(order_id)
        order = self._functional_manager.state.get_order(functional_order_id)

        if order is None:
            logger.warning("Order %s not found for status update", order_id)
            return None

        # Handle fills
        if filled_quantity is not None and filled_quantity > order.filled_quantity:
            fill_quantity = filled_quantity - order.filled_quantity
            fill_price = _fill_price or Decimal(
                50000
            )  # Default fill price for compatibility

            fill = OrderFill(
                quantity=fill_quantity,
                price=fill_price,
                timestamp=datetime.now(UTC),
            )

            # Process fill
            fill_result = self._functional_manager.process_fill_effect(
                functional_order_id, fill
            )
            try:
                self._functional_manager = fill_result.run()
            except Exception as e:
                logger.error("Failed to process fill: %s", str(e))
                return None
        else:
            # Simple status update
            from .orders import transition_order_status

            updated_order = transition_order_status(
                order, self._convert_status_to_functional(status)
            )
            updated_state = self._functional_manager.state.update_order(
                functional_order_id, updated_order
            )
            self._functional_manager = self._functional_manager.with_state(
                updated_state
            )

        # Get updated order
        updated_order = self._functional_manager.state.get_order(functional_order_id)
        if updated_order is None:
            return None

        # Determine event type
        event = self._status_to_event_type(status)

        # Handle order completion
        if status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
        ]:
            # Cancel timeout task
            if order_id in self._timeout_tasks:
                self._timeout_tasks[order_id].cancel()
                del self._timeout_tasks[order_id]

            logger.info("Order %s completed with status %s", order_id, status)

        # Trigger callbacks
        if event:
            self._trigger_callbacks(order_id, event)

        # Persist state
        self._save_state()

        return self._convert_to_legacy_order(updated_order)

    def get_order(self, order_id: str) -> Order | None:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object in legacy format or None if not found
        """
        functional_order_id = OrderId(order_id)
        functional_order = self._functional_manager.state.get_order(functional_order_id)

        if functional_order is None:
            return None

        return self._convert_to_legacy_order(functional_order)

    def get_active_orders(self, symbol: str | None = None) -> list[Order]:
        """
        Get all active orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of active orders in legacy format
        """
        from .orders import Symbol

        symbol_filter = None
        if symbol:
            symbol_result = Symbol.create(symbol)
            if hasattr(symbol_result, "value"):  # It's a Success
                symbol_filter = symbol_result.value

        active_orders = self._functional_manager.state.get_active_orders(symbol_filter)
        return [self._convert_to_legacy_order(order) for order in active_orders]

    def get_orders_by_status(
        self, status: OrderStatus, symbol: str | None = None
    ) -> list[Order]:
        """
        Get orders by status.

        Args:
            status: Order status to filter by
            symbol: Filter by symbol (optional)

        Returns:
            List of orders matching criteria in legacy format
        """
        from .orders import Symbol

        symbol_filter = None
        if symbol:
            symbol_result = Symbol.create(symbol)
            if hasattr(symbol_result, "value"):  # It's a Success
                symbol_filter = symbol_result.value

        functional_status = self._convert_status_to_functional(status)
        orders = self._functional_manager.state.get_orders_by_status(
            functional_status, symbol_filter
        )

        return [self._convert_to_legacy_order(order) for order in orders]

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        functional_order_id = OrderId(order_id)
        order = self._functional_manager.state.get_order(functional_order_id)

        if order is None:
            logger.warning(
                "Cannot cancel order %s: not found in active orders", order_id
            )
            return False

        if order.is_complete:
            logger.warning(
                "Cannot cancel order %s: already completed with status %s",
                order_id,
                order.status,
            )
            return False

        # Cancel using functional system
        cancel_result = self._functional_manager.cancel_order_effect(
            functional_order_id
        )

        try:
            self._functional_manager = cancel_result.run()
            logger.info("Order %s cancelled", order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, str(e))
            return False

    def cancel_all_orders(
        self, symbol: str | None = None, _status: str | None = None
    ) -> int:
        """
        Cancel all active orders.

        Args:
            symbol: Filter by symbol (optional)
            _status: Filter by order status (legacy parameter, not used)

        Returns:
            Number of orders cancelled
        """
        from .orders import Symbol

        symbol_filter = None
        if symbol:
            symbol_result = Symbol.create(symbol)
            if hasattr(symbol_result, "value"):  # It's a Success
                symbol_filter = symbol_result.value

        # Get active orders to cancel
        orders_to_cancel = self._functional_manager.state.get_active_orders(
            symbol_filter
        )

        cancelled_count = 0
        for order in orders_to_cancel:
            if not order.is_complete:
                cancel_result = self._functional_manager.cancel_order_effect(order.id)
                try:
                    self._functional_manager = cancel_result.run()
                    cancelled_count += 1
                except Exception as e:
                    logger.error("Failed to cancel order %s: %s", order.id, str(e))

        logger.info(
            "Cancelled %s orders%s",
            cancelled_count,
            f" for {symbol}" if symbol else "",
        )
        return cancelled_count

    def register_callback(
        self, order_id: str, callback: Callable[[str, str], None]
    ) -> None:
        """
        Register a callback for order events.

        Args:
            order_id: Order ID to monitor
            callback: Callback function (order_id, event)
        """
        if order_id not in self._order_callbacks:
            self._order_callbacks[order_id] = []
        self._order_callbacks[order_id].append(callback)

    def get_order_statistics(
        self, symbol: str | None = None, hours: int = 24
    ) -> dict[str, Any]:
        """
        Get order statistics.

        Args:
            symbol: Filter by symbol (optional)
            hours: Time window in hours

        Returns:
            Dictionary with order statistics
        """
        from .orders import Symbol

        symbol_filter = None
        if symbol:
            symbol_result = Symbol.create(symbol)
            if hasattr(symbol_result, "value"):  # It's a Success
                symbol_filter = symbol_result.value

        stats = self._functional_manager.state.get_statistics(symbol_filter, hours)

        return {
            "total_orders": stats.total_orders,
            "filled_orders": stats.filled_orders,
            "cancelled_orders": stats.cancelled_orders,
            "rejected_orders": stats.rejected_orders,
            "failed_orders": stats.failed_orders,
            "pending_orders": stats.pending_orders,
            "fill_rate_pct": stats.fill_rate_pct,
            "avg_fill_time_seconds": stats.avg_fill_time_seconds,
            "time_window_hours": hours,
            "symbol": symbol,
        }

    def clear_old_history(self, days_to_keep: int = 7) -> None:
        """
        Clear old order history.

        Args:
            days_to_keep: Number of days to keep in history
        """
        # Update functional manager state
        cleaned_state = self._functional_manager.state.cleanup_old_orders(days_to_keep)
        self._functional_manager = self._functional_manager.with_state(cleaned_state)

        # Persist state
        self._save_state()

    def reset_orders(self) -> None:
        """Reset all orders (for testing/emergency use)."""
        # Cancel all timeout tasks
        for task in self._timeout_tasks.values():
            task.cancel()
        self._timeout_tasks.clear()

        # Reset functional manager
        self._functional_manager = FunctionalOrderManager.create()

        # Clear callbacks
        self._order_callbacks.clear()

        self._save_state()
        logger.warning("All orders have been reset")

    # Private helper methods
    def _convert_to_legacy_order(self, functional_order: OrderState) -> Order:
        """Convert functional order to legacy Order format."""
        return Order(
            id=str(functional_order.id),
            symbol=str(functional_order.parameters.symbol),
            side=functional_order.parameters.side.value,
            type=functional_order.parameters.order_type.value,
            quantity=functional_order.parameters.quantity,
            price=functional_order.parameters.price,
            stop_price=functional_order.parameters.stop_price,
            status=self._convert_status_from_functional(functional_order.status),
            timestamp=functional_order.created_at,
            filled_quantity=functional_order.filled_quantity,
        )

    def _convert_from_legacy_order(self, legacy_order: Order) -> OrderState:
        """Convert legacy Order to functional OrderState."""
        # Create parameters
        params_result = OrderParameters.create(
            symbol=legacy_order.symbol,
            side=legacy_order.side,
            order_type=legacy_order.type,
            quantity=float(legacy_order.quantity),
            price=float(legacy_order.price) if legacy_order.price else None,
            stop_price=(
                float(legacy_order.stop_price) if legacy_order.stop_price else None
            ),
        )

        if hasattr(params_result, "error"):  # It's a Failure
            raise ValueError(f"Invalid legacy order: {params_result.error}")

        parameters = params_result.value

        # Create functional order
        order = create_order(parameters)

        # Update with legacy order data
        from .orders import replace

        order = replace(
            order,
            id=OrderId(legacy_order.id),
            status=self._convert_status_to_functional(legacy_order.status),
            created_at=legacy_order.timestamp,
            updated_at=legacy_order.timestamp,
            filled_quantity=legacy_order.filled_quantity,
        )

        return order

    def _convert_status_to_functional(self, legacy_status: OrderStatus):
        """Convert legacy OrderStatus to functional OrderStatus."""
        from .orders import OrderStatus as FunctionalOrderStatus

        status_map = {
            OrderStatus.PENDING: FunctionalOrderStatus.PENDING,
            OrderStatus.OPEN: FunctionalOrderStatus.OPEN,
            OrderStatus.FILLED: FunctionalOrderStatus.FILLED,
            OrderStatus.CANCELLED: FunctionalOrderStatus.CANCELLED,
            OrderStatus.REJECTED: FunctionalOrderStatus.REJECTED,
            OrderStatus.FAILED: FunctionalOrderStatus.FAILED,
        }

        return status_map.get(legacy_status, FunctionalOrderStatus.PENDING)

    def _convert_status_from_functional(self, functional_status):
        """Convert functional OrderStatus to legacy OrderStatus."""
        from .orders import OrderStatus as FunctionalOrderStatus

        status_map = {
            FunctionalOrderStatus.PENDING: OrderStatus.PENDING,
            FunctionalOrderStatus.OPEN: OrderStatus.OPEN,
            FunctionalOrderStatus.PARTIALLY_FILLED: OrderStatus.OPEN,  # Legacy doesn't have partial filled
            FunctionalOrderStatus.FILLED: OrderStatus.FILLED,
            FunctionalOrderStatus.CANCELLED: OrderStatus.CANCELLED,
            FunctionalOrderStatus.REJECTED: OrderStatus.REJECTED,
            FunctionalOrderStatus.FAILED: OrderStatus.FAILED,
            FunctionalOrderStatus.EXPIRED: OrderStatus.CANCELLED,  # Map expired to cancelled
        }

        return status_map.get(functional_status, OrderStatus.PENDING)

    def _status_to_event_type(self, status: OrderStatus) -> str | None:
        """Convert order status to event type."""
        status_to_event = {
            OrderStatus.OPEN: "SUBMITTED",
            OrderStatus.FILLED: "FILLED",
            OrderStatus.CANCELLED: "CANCELLED",
            OrderStatus.REJECTED: "REJECTED",
            OrderStatus.FAILED: "FAILED",
        }
        return status_to_event.get(status)

    def _schedule_timeout(self, order_id: str, timeout_seconds: int) -> None:
        """Schedule order timeout."""

        async def timeout_handler():
            try:
                await asyncio.sleep(timeout_seconds)

                # Check if order still exists and is active
                functional_order_id = OrderId(order_id)
                order = self._functional_manager.state.get_order(functional_order_id)

                if order and not order.is_complete:
                    logger.warning(
                        "Order %s timed out after %ss",
                        order_id,
                        timeout_seconds,
                    )
                    # Try to timeout the order
                    timeout_result = self._functional_manager.timeout_order_effect(
                        functional_order_id
                    )
                    try:
                        self._functional_manager = timeout_result.run()
                        self._trigger_callbacks(order_id, "EXPIRED")
                    except Exception as e:
                        logger.error("Failed to timeout order %s: %s", order_id, str(e))

            except asyncio.CancelledError:
                pass  # Expected when order completes before timeout
            finally:
                if order_id in self._timeout_tasks:
                    del self._timeout_tasks[order_id]

        # Schedule the timeout task
        if self._running:
            task = asyncio.create_task(timeout_handler())
            self._timeout_tasks[order_id] = task

    def _trigger_callbacks(self, order_id: str, event: str) -> None:
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
                except Exception:
                    logger.exception("Order callback error for %s", order_id)

            # Clean up callbacks for completed orders
            if event in ["FILLED", "CANCELLED", "REJECTED", "FAILED", "EXPIRED"]:
                del self._order_callbacks[order_id]

    def _save_state(self) -> None:
        """Save current state to files (legacy format for compatibility)."""
        import json

        try:
            # Save active orders in legacy format
            orders_data = {}
            for order_id, order in self._functional_manager.state.active_orders.items():
                legacy_order = self._convert_to_legacy_order(order)
                orders_data[str(order_id)] = {
                    "id": legacy_order.id,
                    "symbol": legacy_order.symbol,
                    "side": legacy_order.side,
                    "type": legacy_order.type,
                    "quantity": str(legacy_order.quantity),
                    "price": str(legacy_order.price) if legacy_order.price else None,
                    "stop_price": (
                        str(legacy_order.stop_price)
                        if legacy_order.stop_price
                        else None
                    ),
                    "status": (
                        legacy_order.status
                        if isinstance(legacy_order.status, str)
                        else legacy_order.status.value
                    ),
                    "timestamp": legacy_order.timestamp.isoformat(),
                    "filled_quantity": str(legacy_order.filled_quantity),
                }

            with self.orders_file.open("w") as f:
                json.dump(orders_data, f, indent=2)

            # Save order history (last 500 entries) in legacy format
            history_data = []
            for order in self._functional_manager.state.completed_orders[-500:]:
                legacy_order = self._convert_to_legacy_order(order)
                history_data.append(
                    {
                        "id": legacy_order.id,
                        "symbol": legacy_order.symbol,
                        "side": legacy_order.side,
                        "type": legacy_order.type,
                        "quantity": str(legacy_order.quantity),
                        "price": (
                            str(legacy_order.price) if legacy_order.price else None
                        ),
                        "stop_price": (
                            str(legacy_order.stop_price)
                            if legacy_order.stop_price
                            else None
                        ),
                        "status": (
                            legacy_order.status
                            if isinstance(legacy_order.status, str)
                            else legacy_order.status.value
                        ),
                        "timestamp": legacy_order.timestamp.isoformat(),
                        "filled_quantity": str(legacy_order.filled_quantity),
                    }
                )

            with self.history_file.open("w") as f:
                json.dump(history_data, f, indent=2)

            logger.debug("Order state saved successfully")

        except Exception:
            logger.exception("Failed to save order state")

    def _load_legacy_state(self) -> None:
        """Load state from legacy files if they exist."""
        import json

        try:
            # Load active orders
            if self.orders_file.exists():
                with self.orders_file.open() as f:
                    orders_data = json.load(f)

                for order_id, order_data in orders_data.items():
                    legacy_order = Order(
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

                    functional_order = self._convert_from_legacy_order(legacy_order)
                    updated_state = self._functional_manager.state.add_order(
                        functional_order
                    )
                    self._functional_manager = self._functional_manager.with_state(
                        updated_state
                    )

                logger.info(
                    "Loaded %d active orders from legacy state", len(orders_data)
                )

        except Exception:
            logger.exception("Failed to load legacy order state")
            # Continue with empty state
