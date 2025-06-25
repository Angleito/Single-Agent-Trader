"""
Functional Order Management System

This module provides immutable order types and pure functional algorithms
for order lifecycle management, execution tracking, and state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .types.base import Percentage, Symbol
from .types.result import Failure, Result, Success

if TYPE_CHECKING:
    from .effects import IO, AsyncIO, IOEither


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class OrderEventType(Enum):
    """Order lifecycle events."""

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class OrderId:
    """Immutable order identifier."""

    value: str

    @classmethod
    def generate(cls, symbol: str, side: OrderSide) -> OrderId:
        """Generate a unique order ID."""
        timestamp = int(datetime.now().timestamp() * 1000)
        unique_id = str(uuid4())[:8]
        return cls(f"order_{timestamp}_{symbol}_{side.value}_{unique_id}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class OrderParameters:
    """Immutable order parameters."""

    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    timeout_seconds: int | None = None

    @classmethod
    def create(
        cls,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        timeout_seconds: int | None = None,
    ) -> Result[OrderParameters, str]:
        """Create validated order parameters."""
        # Validate symbol
        symbol_result = Symbol.create(symbol)
        if isinstance(symbol_result, Failure):
            return symbol_result

        # Validate side
        try:
            order_side = OrderSide(side.upper())
        except ValueError:
            return Failure(f"Invalid order side: {side}")

        # Validate order type
        try:
            order_type_enum = OrderType(order_type.upper())
        except ValueError:
            return Failure(f"Invalid order type: {order_type}")

        # Validate quantity
        if quantity <= 0:
            return Failure(f"Quantity must be positive, got {quantity}")

        # Validate prices
        if price is not None and price <= 0:
            return Failure(f"Price must be positive, got {price}")

        if stop_price is not None and stop_price <= 0:
            return Failure(f"Stop price must be positive, got {stop_price}")

        # Validate price requirements for order types
        if order_type_enum in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            return Failure(f"Order type {order_type} requires a price")

        if (
            order_type_enum in [OrderType.STOP, OrderType.STOP_LIMIT]
            and stop_price is None
        ):
            return Failure(f"Order type {order_type} requires a stop price")

        return Success(
            cls(
                symbol=symbol_result.success(),
                side=order_side,
                order_type=order_type_enum,
                quantity=Decimal(str(quantity)),
                price=Decimal(str(price)) if price is not None else None,
                stop_price=Decimal(str(stop_price)) if stop_price is not None else None,
                timeout_seconds=timeout_seconds,
            )
        )


@dataclass(frozen=True)
class OrderFill:
    """Immutable order fill record."""

    quantity: Decimal
    price: Decimal
    timestamp: datetime
    fee: Decimal | None = None
    trade_id: str | None = None

    @property
    def value(self) -> Decimal:
        """Calculate fill value."""
        return self.quantity * self.price


@dataclass(frozen=True)
class OrderState:
    """Immutable order state."""

    id: OrderId
    parameters: OrderParameters
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    filled_quantity: Decimal = Decimal(0)
    fills: list[OrderFill] = None
    average_fill_price: Decimal | None = None
    total_fees: Decimal = Decimal(0)
    error_message: str | None = None

    def __post_init__(self):
        """Initialize default values."""
        if self.fills is None:
            object.__setattr__(self, "fills", [])

    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to fill."""
        return self.parameters.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> Percentage:
        """Calculate fill percentage."""
        if self.parameters.quantity == 0:
            return Percentage.create(0.0).value

        fill_ratio = float(self.filled_quantity / self.parameters.quantity)
        return Percentage.create(min(fill_ratio, 1.0)).value

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
            OrderStatus.EXPIRED,
        ]

    @property
    def is_fillable(self) -> bool:
        """Check if order can still receive fills."""
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]

    def with_status(
        self, status: OrderStatus, error_message: str | None = None
    ) -> OrderState:
        """Create new order state with updated status."""
        return replace(
            self,
            status=status,
            updated_at=datetime.now(),
            error_message=error_message,
        )

    def with_fill(self, fill: OrderFill) -> OrderState:
        """Create new order state with additional fill."""
        new_fills = [*self.fills, fill]
        new_filled_quantity = self.filled_quantity + fill.quantity
        new_total_fees = self.total_fees + (fill.fee or Decimal(0))

        # Calculate new average fill price
        total_fill_value = sum(f.quantity * f.price for f in new_fills)
        new_average_price = (
            total_fill_value / new_filled_quantity if new_filled_quantity > 0 else None
        )

        # Determine new status
        new_status = self.status
        if new_filled_quantity >= self.parameters.quantity:
            new_status = OrderStatus.FILLED
        elif new_filled_quantity > 0:
            new_status = OrderStatus.PARTIALLY_FILLED

        return replace(
            self,
            status=new_status,
            updated_at=datetime.now(),
            filled_quantity=new_filled_quantity,
            fills=new_fills,
            average_fill_price=new_average_price,
            total_fees=new_total_fees,
        )


@dataclass(frozen=True)
class OrderEvent:
    """Immutable order event."""

    order_id: OrderId
    event_type: OrderEventType
    timestamp: datetime
    data: dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.data is None:
            object.__setattr__(self, "data", {})


@dataclass(frozen=True)
class OrderStatistics:
    """Immutable order statistics."""

    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    failed_orders: int = 0
    pending_orders: int = 0
    fill_rate_pct: float = 0.0
    avg_fill_time_seconds: float = 0.0
    time_window_hours: int = 24
    symbol: Symbol | None = None

    @classmethod
    def from_orders(
        cls,
        orders: list[OrderState],
        time_window_hours: int = 24,
        symbol: Symbol | None = None,
    ) -> OrderStatistics:
        """Calculate statistics from a list of orders."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

        # Filter orders by time window and symbol
        relevant_orders = [
            order
            for order in orders
            if order.created_at >= cutoff_time
            and (symbol is None or order.parameters.symbol == symbol)
        ]

        if not relevant_orders:
            return cls(time_window_hours=time_window_hours, symbol=symbol)

        total_orders = len(relevant_orders)
        filled_orders = sum(
            1 for o in relevant_orders if o.status == OrderStatus.FILLED
        )
        cancelled_orders = sum(
            1 for o in relevant_orders if o.status == OrderStatus.CANCELLED
        )
        rejected_orders = sum(
            1 for o in relevant_orders if o.status == OrderStatus.REJECTED
        )
        failed_orders = sum(
            1 for o in relevant_orders if o.status == OrderStatus.FAILED
        )
        pending_orders = sum(
            1
            for o in relevant_orders
            if o.status
            in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        )

        fill_rate_pct = (
            (filled_orders / total_orders * 100) if total_orders > 0 else 0.0
        )

        # Calculate average fill time for filled orders
        fill_times = []
        for order in relevant_orders:
            if order.status == OrderStatus.FILLED and order.fills:
                # Use the last fill time as completion time
                completion_time = max(fill.timestamp for fill in order.fills)
                fill_time = (completion_time - order.created_at).total_seconds()
                fill_times.append(fill_time)

        avg_fill_time_seconds = sum(fill_times) / len(fill_times) if fill_times else 0.0

        return cls(
            total_orders=total_orders,
            filled_orders=filled_orders,
            cancelled_orders=cancelled_orders,
            rejected_orders=rejected_orders,
            failed_orders=failed_orders,
            pending_orders=pending_orders,
            fill_rate_pct=fill_rate_pct,
            avg_fill_time_seconds=avg_fill_time_seconds,
            time_window_hours=time_window_hours,
            symbol=symbol,
        )


# Pure functional state transitions
def create_order(parameters: OrderParameters) -> OrderState:
    """Create a new order state."""
    order_id = OrderId.generate(str(parameters.symbol), parameters.side)
    now = datetime.now()

    return OrderState(
        id=order_id,
        parameters=parameters,
        status=OrderStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


def transition_order_status(
    order: OrderState,
    new_status: OrderStatus,
    error_message: str | None = None,
) -> OrderState:
    """Pure function to transition order status."""
    return order.with_status(new_status, error_message)


def add_fill_to_order(order: OrderState, fill: OrderFill) -> OrderState:
    """Pure function to add a fill to an order."""
    if not order.is_fillable:
        raise ValueError(f"Cannot fill order {order.id} with status {order.status}")

    if fill.quantity <= 0:
        raise ValueError("Fill quantity must be positive")

    if fill.quantity > order.remaining_quantity:
        raise ValueError("Fill quantity exceeds remaining order quantity")

    return order.with_fill(fill)


def cancel_order(order: OrderState) -> OrderState:
    """Pure function to cancel an order."""
    if order.is_complete:
        raise ValueError(f"Cannot cancel completed order {order.id}")

    return transition_order_status(order, OrderStatus.CANCELLED)


def expire_order(order: OrderState) -> OrderState:
    """Pure function to expire an order."""
    if order.is_complete:
        return order  # Already complete, no change needed

    return transition_order_status(order, OrderStatus.EXPIRED)


def reject_order(order: OrderState, reason: str) -> OrderState:
    """Pure function to reject an order."""
    return transition_order_status(order, OrderStatus.REJECTED, reason)


def fail_order(order: OrderState, reason: str) -> OrderState:
    """Pure function to fail an order."""
    return transition_order_status(order, OrderStatus.FAILED, reason)


# Functional Order Lifecycle Management
@dataclass(frozen=True)
class OrderManagerState:
    """Immutable order manager state."""

    active_orders: dict[OrderId, OrderState]
    completed_orders: list[OrderState]
    order_events: list[OrderEvent]
    statistics: OrderStatistics

    @classmethod
    def empty(cls) -> OrderManagerState:
        """Create empty order manager state."""
        return cls(
            active_orders={},
            completed_orders=[],
            order_events=[],
            statistics=OrderStatistics(),
        )

    def add_order(self, order: OrderState) -> OrderManagerState:
        """Add a new order to the state."""
        new_active_orders = {**self.active_orders, order.id: order}

        # Create order creation event
        creation_event = OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.CREATED,
            timestamp=order.created_at,
        )

        new_events = [*self.order_events, creation_event]

        # Update statistics
        all_orders = list(new_active_orders.values()) + self.completed_orders
        new_statistics = OrderStatistics.from_orders(all_orders)

        return replace(
            self,
            active_orders=new_active_orders,
            order_events=new_events,
            statistics=new_statistics,
        )

    def update_order(
        self, order_id: OrderId, updated_order: OrderState
    ) -> OrderManagerState:
        """Update an existing order."""
        if order_id not in self.active_orders:
            return self  # Order not found, no change

        old_order = self.active_orders[order_id]

        # Determine event type based on status change
        event_type = None
        if old_order.status != updated_order.status:
            event_type = self._status_to_event_type(updated_order.status)
        elif updated_order.filled_quantity > old_order.filled_quantity:
            event_type = OrderEventType.PARTIALLY_FILLED

        # Update active or completed orders
        new_active_orders = dict(self.active_orders)
        new_completed_orders = list(self.completed_orders)

        if updated_order.is_complete:
            # Move to completed orders
            del new_active_orders[order_id]
            new_completed_orders.append(updated_order)
        else:
            # Update active order
            new_active_orders[order_id] = updated_order

        # Add event if status changed
        new_events = list(self.order_events)
        if event_type:
            event = OrderEvent(
                order_id=order_id,
                event_type=event_type,
                timestamp=updated_order.updated_at,
                data={
                    "old_status": old_order.status.value,
                    "new_status": updated_order.status.value,
                    "filled_quantity": str(updated_order.filled_quantity),
                },
            )
            new_events.append(event)

        # Update statistics
        all_orders = list(new_active_orders.values()) + new_completed_orders
        new_statistics = OrderStatistics.from_orders(all_orders)

        return replace(
            self,
            active_orders=new_active_orders,
            completed_orders=new_completed_orders,
            order_events=new_events,
            statistics=new_statistics,
        )

    def cancel_order(self, order_id: OrderId) -> Result[OrderManagerState, str]:
        """Cancel an order by ID."""
        if order_id not in self.active_orders:
            return Failure(f"Order {order_id} not found")

        order = self.active_orders[order_id]

        try:
            cancelled_order = cancel_order(order)
            return Success(self.update_order(order_id, cancelled_order))
        except ValueError as e:
            return Failure(str(e))

    def cancel_all_orders(self, symbol: Symbol | None = None) -> OrderManagerState:
        """Cancel all active orders, optionally filtered by symbol."""
        new_state = self

        for order_id, order in self.active_orders.items():
            if symbol is None or order.parameters.symbol == symbol:
                if not order.is_complete:
                    try:
                        cancelled_order = cancel_order(order)
                        new_state = new_state.update_order(order_id, cancelled_order)
                    except ValueError:
                        # Order already completed, skip
                        pass

        return new_state

    def get_order(self, order_id: OrderId) -> OrderState | None:
        """Get an order by ID."""
        # Check active orders first
        if order_id in self.active_orders:
            return self.active_orders[order_id]

        # Check completed orders
        for order in self.completed_orders:
            if order.id == order_id:
                return order

        return None

    def get_active_orders(self, symbol: Symbol | None = None) -> list[OrderState]:
        """Get all active orders, optionally filtered by symbol."""
        orders = list(self.active_orders.values())

        if symbol:
            orders = [order for order in orders if order.parameters.symbol == symbol]

        return orders

    def get_orders_by_status(
        self,
        status: OrderStatus,
        symbol: Symbol | None = None,
        hours: int = 24,
    ) -> list[OrderState]:
        """Get orders by status."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        orders = []

        # Check active orders
        for order in self.active_orders.values():
            if (
                order.status == status
                and order.created_at >= cutoff_time
                and (symbol is None or order.parameters.symbol == symbol)
            ):
                orders.append(order)

        # Check completed orders for terminal statuses
        if status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
        ]:
            for order in self.completed_orders:
                if (
                    order.status == status
                    and order.created_at >= cutoff_time
                    and (symbol is None or order.parameters.symbol == symbol)
                ):
                    orders.append(order)

        return orders

    def get_statistics(
        self,
        symbol: Symbol | None = None,
        hours: int = 24,
    ) -> OrderStatistics:
        """Get order statistics."""
        all_orders = list(self.active_orders.values()) + self.completed_orders
        return OrderStatistics.from_orders(all_orders, hours, symbol)

    def cleanup_old_orders(self, days_to_keep: int = 7) -> OrderManagerState:
        """Remove old completed orders."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        new_completed_orders = [
            order for order in self.completed_orders if order.updated_at >= cutoff_date
        ]

        # Also cleanup old events
        new_events = [
            event for event in self.order_events if event.timestamp >= cutoff_date
        ]

        return replace(
            self,
            completed_orders=new_completed_orders,
            order_events=new_events,
        )

    def _status_to_event_type(self, status: OrderStatus) -> OrderEventType:
        """Convert order status to event type."""
        status_to_event = {
            OrderStatus.OPEN: OrderEventType.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED: OrderEventType.PARTIALLY_FILLED,
            OrderStatus.FILLED: OrderEventType.FILLED,
            OrderStatus.CANCELLED: OrderEventType.CANCELLED,
            OrderStatus.REJECTED: OrderEventType.REJECTED,
            OrderStatus.FAILED: OrderEventType.FAILED,
            OrderStatus.EXPIRED: OrderEventType.EXPIRED,
        }
        return status_to_event.get(status, OrderEventType.CREATED)


# Pure functional order lifecycle operations
def process_order_creation(
    state: OrderManagerState,
    parameters: OrderParameters,
) -> Result[tuple[OrderManagerState, OrderState], str]:
    """Process order creation."""
    # Validate parameters
    if not isinstance(parameters, OrderParameters):
        return Failure("Invalid order parameters")

    # Create new order
    new_order = create_order(parameters)

    # Add to state
    new_state = state.add_order(new_order)

    return Success((new_state, new_order))


def process_order_fill(
    state: OrderManagerState,
    order_id: OrderId,
    fill: OrderFill,
) -> Result[OrderManagerState, str]:
    """Process order fill."""
    order = state.get_order(order_id)
    if order is None:
        return Failure(f"Order {order_id} not found")

    try:
        filled_order = add_fill_to_order(order, fill)
        new_state = state.update_order(order_id, filled_order)
        return Success(new_state)
    except ValueError as e:
        return Failure(str(e))


def process_order_timeout(
    state: OrderManagerState,
    order_id: OrderId,
) -> Result[OrderManagerState, str]:
    """Process order timeout."""
    order = state.get_order(order_id)
    if order is None:
        return Failure(f"Order {order_id} not found")

    if order.is_complete:
        return Success(state)  # Already complete

    expired_order = expire_order(order)
    new_state = state.update_order(order_id, expired_order)

    # Add timeout event
    timeout_event = OrderEvent(
        order_id=order_id,
        event_type=OrderEventType.TIMEOUT,
        timestamp=datetime.now(),
    )

    new_state = replace(
        new_state,
        order_events=[*new_state.order_events, timeout_event],
    )

    return Success(new_state)


# Functional Order Execution Effects
class OrderExecutionError(Exception):
    """Order execution error."""

    def __init__(self, message: str, order_id: OrderId | None = None):
        self.message = message
        self.order_id = order_id
        super().__init__(message)


def place_order_effect(order: OrderState) -> IOEither[OrderExecutionError, OrderState]:
    """Effect to place an order on the exchange."""

    def place():
        # Simulate order placement validation
        if order.parameters.quantity <= 0:
            raise OrderExecutionError(
                f"Invalid order quantity: {order.parameters.quantity}", order.id
            )

        if (
            order.parameters.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]
            and order.parameters.price is None
        ):
            raise OrderExecutionError(
                f"Order type {order.parameters.order_type} requires a price", order.id
            )

        # Simulate successful order placement
        return transition_order_status(order, OrderStatus.OPEN)

    from .effects.io import from_try

    return from_try(place)


def cancel_order_effect(
    order_id: OrderId, order: OrderState
) -> IOEither[OrderExecutionError, OrderState]:
    """Effect to cancel an order on the exchange."""

    def cancel():
        if order.is_complete:
            raise OrderExecutionError(
                f"Cannot cancel completed order {order_id}", order_id
            )

        # Simulate order cancellation
        return transition_order_status(order, OrderStatus.CANCELLED)

    from .effects.io import from_try

    return from_try(cancel)


def get_order_status_effect(
    order_id: OrderId,
) -> IOEither[OrderExecutionError, OrderStatus | None]:
    """Effect to get order status from exchange."""

    def get_status():
        # Simulate status retrieval from exchange
        # In real implementation, this would call exchange API
        import random

        statuses = [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED]
        return random.choice(statuses)

    from .effects.io import from_try

    return from_try(get_status)


def monitor_order_fills_effect(order_id: OrderId) -> AsyncIO[list[OrderFill]]:
    """Effect to monitor order fills from exchange."""

    async def monitor():
        # Simulate monitoring fills
        import asyncio
        import random

        await asyncio.sleep(1)  # Simulate network delay

        # Simulate receiving a fill
        if random.random() > 0.5:  # 50% chance of fill
            fill = OrderFill(
                quantity=Decimal(str(random.uniform(0.1, 1.0))),
                price=Decimal(str(random.uniform(50000, 51000))),
                timestamp=datetime.now(),
                fee=Decimal("0.001"),
                trade_id=str(uuid4()),
            )
            return [fill]

        return []

    from .effects.io import AsyncIO

    return AsyncIO.from_coroutine(monitor())


def timeout_order_effect(order_id: OrderId, timeout_seconds: int) -> AsyncIO[OrderId]:
    """Effect to timeout an order after specified seconds."""

    async def timeout():
        import asyncio

        await asyncio.sleep(timeout_seconds)
        return order_id

    from .effects.io import AsyncIO

    return AsyncIO.from_coroutine(timeout())


# Functional Order Manager with Effects
@dataclass(frozen=True)
class FunctionalOrderManager:
    """Functional order manager using effects and immutable state."""

    state: OrderManagerState

    @classmethod
    def create(cls) -> FunctionalOrderManager:
        """Create a new functional order manager."""
        return cls(state=OrderManagerState.empty())

    def with_state(self, new_state: OrderManagerState) -> FunctionalOrderManager:
        """Create a new manager with updated state."""
        return replace(self, state=new_state)

    def create_order_effect(
        self, parameters: OrderParameters
    ) -> IOEither[OrderExecutionError, tuple[FunctionalOrderManager, OrderState]]:
        """Effect to create and place a new order."""

        def create_and_place():
            # Create order
            creation_result = process_order_creation(self.state, parameters)
            if isinstance(creation_result, Failure):
                raise OrderExecutionError(creation_result.error)

            new_state, new_order = creation_result.value

            # Update manager with new state
            updated_manager = self.with_state(new_state)

            return updated_manager, new_order

        from .effects.io import from_try

        return from_try(create_and_place)

    def place_order_effect(
        self, order_id: OrderId
    ) -> IOEither[OrderExecutionError, FunctionalOrderManager]:
        """Effect to place an existing order."""

        def place():
            order = self.state.get_order(order_id)
            if order is None:
                raise OrderExecutionError(f"Order {order_id} not found", order_id)

            if order.status != OrderStatus.PENDING:
                raise OrderExecutionError(
                    f"Cannot place order {order_id} with status {order.status}",
                    order_id,
                )

            # Place order effect
            placed_order = transition_order_status(order, OrderStatus.OPEN)
            new_state = self.state.update_order(order_id, placed_order)

            return self.with_state(new_state)

        from .effects.io import from_try

        return from_try(place)

    def cancel_order_effect(
        self, order_id: OrderId
    ) -> IOEither[OrderExecutionError, FunctionalOrderManager]:
        """Effect to cancel an order."""

        def cancel():
            cancel_result = self.state.cancel_order(order_id)
            if isinstance(cancel_result, Failure):
                raise OrderExecutionError(cancel_result.error, order_id)

            return self.with_state(cancel_result.value)

        from .effects.io import from_try

        return from_try(cancel)

    def process_fill_effect(
        self, order_id: OrderId, fill: OrderFill
    ) -> IOEither[OrderExecutionError, FunctionalOrderManager]:
        """Effect to process an order fill."""

        def process_fill():
            fill_result = process_order_fill(self.state, order_id, fill)
            if isinstance(fill_result, Failure):
                raise OrderExecutionError(fill_result.error, order_id)

            return self.with_state(fill_result.value)

        from .effects.io import from_try

        return from_try(process_fill)

    def timeout_order_effect(
        self, order_id: OrderId
    ) -> IOEither[OrderExecutionError, FunctionalOrderManager]:
        """Effect to timeout an order."""

        def timeout():
            timeout_result = process_order_timeout(self.state, order_id)
            if isinstance(timeout_result, Failure):
                raise OrderExecutionError(timeout_result.error, order_id)

            return self.with_state(timeout_result.value)

        from .effects.io import from_try

        return from_try(timeout)

    def get_order_effect(self, order_id: OrderId) -> IO[OrderState | None]:
        """Effect to get an order."""

        def get_order():
            return self.state.get_order(order_id)

        from .effects.io import IO

        return IO.from_callable(get_order)

    def get_active_orders_effect(
        self, symbol: Symbol | None = None
    ) -> IO[list[OrderState]]:
        """Effect to get active orders."""

        def get_orders():
            return self.state.get_active_orders(symbol)

        from .effects.io import IO

        return IO.from_callable(get_orders)

    def get_statistics_effect(
        self,
        symbol: Symbol | None = None,
        hours: int = 24,
    ) -> IO[OrderStatistics]:
        """Effect to get order statistics."""

        def get_stats():
            return self.state.get_statistics(symbol, hours)

        from .effects.io import IO

        return IO.from_callable(get_stats)


# Effect Combinators for Order Management
def sequence_order_effects(
    effects: list[IOEither[OrderExecutionError, Any]],
) -> IOEither[OrderExecutionError, list[Any]]:
    """Sequence multiple order effects."""
    from .effects.io import sequence

    return sequence(effects)


def parallel_order_effects(
    effects: list[IOEither[OrderExecutionError, Any]],
) -> IOEither[OrderExecutionError, list[Any]]:
    """Run multiple order effects in parallel."""
    from .effects.io import parallel

    return parallel(effects)


def retry_order_effect(
    effect: IOEither[OrderExecutionError, Any],
    max_retries: int = 3,
    delay_seconds: float = 1.0,
) -> IOEither[OrderExecutionError, Any]:
    """Retry an order effect with exponential backoff."""

    def retry_logic():
        import time

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return effect.run()
            except OrderExecutionError as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(delay_seconds * (2**attempt))
                    continue
                raise

        raise last_error or OrderExecutionError("Retry failed")

    from .effects.io import from_try

    return from_try(retry_logic)


def with_timeout_effect(
    effect: IOEither[OrderExecutionError, Any],
    timeout_seconds: float,
) -> IOEither[OrderExecutionError, Any]:
    """Add timeout to an order effect."""

    def with_timeout():
        import signal

        class TimeoutError(Exception):
            pass

        def timeout_handler(_signum, _frame):
            raise TimeoutError("Operation timed out")

        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            result = effect.run()
            signal.alarm(0)  # Cancel timeout
            signal.signal(signal.SIGALRM, old_handler)
            return result
        except TimeoutError:
            signal.signal(signal.SIGALRM, old_handler)
            raise OrderExecutionError("Order effect timed out")

    from .effects.io import from_try

    return from_try(with_timeout)


# Functional Order Event Sourcing
@dataclass(frozen=True)
class OrderEventData:
    """Order-specific event data."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: str
    price: str | None = None
    stop_price: str | None = None
    status: str | None = None
    filled_quantity: str | None = None
    fill_price: str | None = None
    fee: str | None = None
    error_message: str | None = None


class OrderTrackedEvent:
    """Order events that can be tracked through event sourcing."""

    @staticmethod
    def order_created(order: OrderState) -> OrderEvent:
        """Create an order creation event."""
        return OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.CREATED,
            timestamp=order.created_at,
            data={
                "order_id": str(order.id),
                "symbol": str(order.parameters.symbol),
                "side": order.parameters.side.value,
                "order_type": order.parameters.order_type.value,
                "quantity": str(order.parameters.quantity),
                "price": (
                    str(order.parameters.price) if order.parameters.price else None
                ),
                "stop_price": (
                    str(order.parameters.stop_price)
                    if order.parameters.stop_price
                    else None
                ),
                "status": order.status.value,
            },
        )

    @staticmethod
    def order_submitted(order: OrderState) -> OrderEvent:
        """Create an order submission event."""
        return OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.SUBMITTED,
            timestamp=order.updated_at,
            data={
                "order_id": str(order.id),
                "status": order.status.value,
            },
        )

    @staticmethod
    def order_filled(order: OrderState, fill: OrderFill) -> OrderEvent:
        """Create an order fill event."""
        return OrderEvent(
            order_id=order.id,
            event_type=(
                OrderEventType.FILLED
                if order.status == OrderStatus.FILLED
                else OrderEventType.PARTIALLY_FILLED
            ),
            timestamp=fill.timestamp,
            data={
                "order_id": str(order.id),
                "status": order.status.value,
                "filled_quantity": str(order.filled_quantity),
                "fill_price": str(fill.price),
                "fill_quantity": str(fill.quantity),
                "fee": str(fill.fee) if fill.fee else None,
                "trade_id": fill.trade_id,
            },
        )

    @staticmethod
    def order_cancelled(order: OrderState) -> OrderEvent:
        """Create an order cancellation event."""
        return OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.CANCELLED,
            timestamp=order.updated_at,
            data={
                "order_id": str(order.id),
                "status": order.status.value,
                "filled_quantity": str(order.filled_quantity),
            },
        )

    @staticmethod
    def order_rejected(order: OrderState, reason: str) -> OrderEvent:
        """Create an order rejection event."""
        return OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.REJECTED,
            timestamp=order.updated_at,
            data={
                "order_id": str(order.id),
                "status": order.status.value,
                "error_message": reason,
            },
        )

    @staticmethod
    def order_failed(order: OrderState, reason: str) -> OrderEvent:
        """Create an order failure event."""
        return OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.FAILED,
            timestamp=order.updated_at,
            data={
                "order_id": str(order.id),
                "status": order.status.value,
                "error_message": reason,
            },
        )

    @staticmethod
    def order_expired(order: OrderState) -> OrderEvent:
        """Create an order expiration event."""
        return OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.EXPIRED,
            timestamp=order.updated_at,
            data={
                "order_id": str(order.id),
                "status": order.status.value,
                "filled_quantity": str(order.filled_quantity),
            },
        )


@dataclass(frozen=True)
class EventSourcedOrderManager:
    """Event-sourced order manager that tracks all order events."""

    state: OrderManagerState
    event_store: Any  # EventStore protocol

    @classmethod
    def create(cls, event_store: Any) -> EventSourcedOrderManager:
        """Create an event-sourced order manager."""
        return cls(state=OrderManagerState.empty(), event_store=event_store)

    @classmethod
    def replay_from_events(cls, event_store: Any) -> EventSourcedOrderManager:
        """Replay order manager state from stored events."""
        # Start with empty state
        state = OrderManagerState.empty()

        # Replay all order events
        events = event_store.replay()

        for event in events:
            # Only process order events
            if hasattr(event, "data") and "order_id" in event.data:
                state = cls._apply_event_to_state(state, event)

        return cls(state=state, event_store=event_store)

    def with_state(self, new_state: OrderManagerState) -> EventSourcedOrderManager:
        """Create new manager with updated state."""
        return replace(self, state=new_state)

    def create_order_effect(
        self, parameters: OrderParameters
    ) -> IOEither[OrderExecutionError, tuple[EventSourcedOrderManager, OrderState]]:
        """Effect to create an order with event tracking."""

        def create_and_track():
            # Create order
            creation_result = process_order_creation(self.state, parameters)
            if isinstance(creation_result, Failure):
                raise OrderExecutionError(creation_result.error)

            new_state, new_order = creation_result.value

            # Create and store event
            event = OrderTrackedEvent.order_created(new_order)
            self.event_store.append(self._convert_to_trading_event(event))

            return self.with_state(new_state), new_order

        from .effects.io import from_try

        return from_try(create_and_track)

    def place_order_effect(
        self, order_id: OrderId
    ) -> IOEither[OrderExecutionError, EventSourcedOrderManager]:
        """Effect to place an order with event tracking."""

        def place_and_track():
            order = self.state.get_order(order_id)
            if order is None:
                raise OrderExecutionError(f"Order {order_id} not found", order_id)

            # Update order status
            placed_order = transition_order_status(order, OrderStatus.OPEN)
            new_state = self.state.update_order(order_id, placed_order)

            # Create and store event
            event = OrderTrackedEvent.order_submitted(placed_order)
            self.event_store.append(self._convert_to_trading_event(event))

            return self.with_state(new_state)

        from .effects.io import from_try

        return from_try(place_and_track)

    def process_fill_effect(
        self, order_id: OrderId, fill: OrderFill
    ) -> IOEither[OrderExecutionError, EventSourcedOrderManager]:
        """Effect to process a fill with event tracking."""

        def process_and_track():
            fill_result = process_order_fill(self.state, order_id, fill)
            if isinstance(fill_result, Failure):
                raise OrderExecutionError(fill_result.error, order_id)

            new_state = fill_result.value
            updated_order = new_state.get_order(order_id)

            if updated_order:
                # Create and store event
                event = OrderTrackedEvent.order_filled(updated_order, fill)
                self.event_store.append(self._convert_to_trading_event(event))

            return self.with_state(new_state)

        from .effects.io import from_try

        return from_try(process_and_track)

    def cancel_order_effect(
        self, order_id: OrderId
    ) -> IOEither[OrderExecutionError, EventSourcedOrderManager]:
        """Effect to cancel an order with event tracking."""

        def cancel_and_track():
            cancel_result = self.state.cancel_order(order_id)
            if isinstance(cancel_result, Failure):
                raise OrderExecutionError(cancel_result.error, order_id)

            new_state = cancel_result.value
            cancelled_order = new_state.get_order(order_id)

            if cancelled_order:
                # Create and store event
                event = OrderTrackedEvent.order_cancelled(cancelled_order)
                self.event_store.append(self._convert_to_trading_event(event))

            return self.with_state(new_state)

        from .effects.io import from_try

        return from_try(cancel_and_track)

    def get_order_history_effect(self, order_id: OrderId) -> IO[list[OrderEvent]]:
        """Effect to get complete order history from events."""

        def get_history():
            # Query events for this order
            events = self.event_store.query(
                lambda e: hasattr(e, "data") and e.data.get("order_id") == str(order_id)
            )

            # Convert to order events
            order_events = []
            for event in events:
                order_event = self._convert_from_trading_event(event)
                if order_event:
                    order_events.append(order_event)

            return order_events

        from .effects.io import IO

        return IO.from_callable(get_history)

    def get_order_analytics_effect(
        self,
        symbol: Symbol | None = None,
        hours: int = 24,
    ) -> IO[dict[str, Any]]:
        """Effect to get order analytics from event history."""

        def get_analytics():
            cutoff_time = datetime.now() - timedelta(hours=hours)

            # Query recent events
            events = self.event_store.get_events_after(cutoff_time)

            # Filter order events
            order_events = [
                e
                for e in events
                if hasattr(e, "data")
                and "order_id" in e.data
                and (symbol is None or e.data.get("symbol") == str(symbol))
            ]

            # Calculate analytics
            event_counts = {}
            order_ids = set()

            for event in order_events:
                event_type = event.event_type
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                order_ids.add(event.data["order_id"])

            return {
                "total_events": len(order_events),
                "unique_orders": len(order_ids),
                "event_counts": event_counts,
                "time_window_hours": hours,
                "symbol": str(symbol) if symbol else None,
            }

        from .effects.io import IO

        return IO.from_callable(get_analytics)

    def _convert_to_trading_event(self, order_event: OrderEvent) -> Any:
        """Convert OrderEvent to TradingEvent for storage."""
        from .events.store import TradingEvent

        return TradingEvent(
            event_id=str(uuid4()),
            event_type=order_event.event_type.value,
            timestamp=order_event.timestamp,
            data=order_event.data or {},
        )

    def _convert_from_trading_event(self, trading_event: Any) -> OrderEvent | None:
        """Convert TradingEvent back to OrderEvent."""
        try:
            event_type = OrderEventType(trading_event.event_type)
            order_id_str = trading_event.data.get("order_id")

            if not order_id_str:
                return None

            return OrderEvent(
                order_id=OrderId(order_id_str),
                event_type=event_type,
                timestamp=trading_event.timestamp,
                data=trading_event.data,
            )
        except (ValueError, KeyError):
            return None

    @staticmethod
    def _apply_event_to_state(
        state: OrderManagerState, event: Any
    ) -> OrderManagerState:
        """Apply a single event to the order manager state."""
        # This would contain the logic to reconstruct state from events
        # For now, return unchanged state as this is a complex operation
        # that would require full event replay logic
        return state


# Event-sourced order lifecycle functions
def create_order_with_events(
    manager: EventSourcedOrderManager, parameters: OrderParameters
) -> Result[tuple[EventSourcedOrderManager, OrderState], str]:
    """Create an order and record the creation event."""
    try:
        result = manager.create_order_effect(parameters).run()
        return Success(result)
    except OrderExecutionError as e:
        return Failure(e.message)


def process_fill_with_events(
    manager: EventSourcedOrderManager, order_id: OrderId, fill: OrderFill
) -> Result[EventSourcedOrderManager, str]:
    """Process a fill and record the fill event."""
    try:
        result = manager.process_fill_effect(order_id, fill).run()
        return Success(result)
    except OrderExecutionError as e:
        return Failure(e.message)


def cancel_order_with_events(
    manager: EventSourcedOrderManager, order_id: OrderId
) -> Result[EventSourcedOrderManager, str]:
    """Cancel an order and record the cancellation event."""
    try:
        result = manager.cancel_order_effect(order_id).run()
        return Success(result)
    except OrderExecutionError as e:
        return Failure(e.message)


# Pure Functional Order Analytics
@dataclass(frozen=True)
class OrderPerformanceMetrics:
    """Immutable order performance metrics."""

    total_orders: int
    successful_orders: int
    failed_orders: int
    average_fill_time_seconds: float
    fill_rate_percentage: float
    average_slippage_bps: float
    total_fees_paid: Decimal
    volume_traded: Decimal
    best_execution_score: float  # 0-100, higher is better

    @property
    def success_rate(self) -> float:
        """Calculate order success rate."""
        if self.total_orders == 0:
            return 0.0
        return (self.successful_orders / self.total_orders) * 100

    @property
    def average_fee_rate(self) -> float:
        """Calculate average fee rate as percentage of volume."""
        if self.volume_traded == 0:
            return 0.0
        return float(self.total_fees_paid / self.volume_traded) * 100


@dataclass(frozen=True)
class OrderVolumeAnalysis:
    """Immutable order volume analysis."""

    total_volume: Decimal
    buy_volume: Decimal
    sell_volume: Decimal
    average_order_size: Decimal
    largest_order_size: Decimal
    smallest_order_size: Decimal
    volume_weighted_average_price: Decimal

    @property
    def buy_sell_ratio(self) -> float:
        """Calculate buy to sell volume ratio."""
        if self.sell_volume == 0:
            return float("inf") if self.buy_volume > 0 else 0.0
        return float(self.buy_volume / self.sell_volume)

    @property
    def volume_concentration(self) -> float:
        """Calculate volume concentration (largest order / total volume)."""
        if self.total_volume == 0:
            return 0.0
        return float(self.largest_order_size / self.total_volume)


@dataclass(frozen=True)
class OrderTimingAnalysis:
    """Immutable order timing analysis."""

    total_orders: int
    orders_per_hour: float
    peak_hour: int  # Hour of day with most orders (0-23)
    off_peak_hour: int  # Hour of day with least orders
    average_time_between_orders_minutes: float
    fastest_fill_seconds: float
    slowest_fill_seconds: float
    median_fill_time_seconds: float


# Pure functional analytics algorithms
def analyze_order_performance(orders: list[OrderState]) -> OrderPerformanceMetrics:
    """Pure function to analyze order performance."""
    if not orders:
        return OrderPerformanceMetrics(
            total_orders=0,
            successful_orders=0,
            failed_orders=0,
            average_fill_time_seconds=0.0,
            fill_rate_percentage=0.0,
            average_slippage_bps=0.0,
            total_fees_paid=Decimal(0),
            volume_traded=Decimal(0),
            best_execution_score=0.0,
        )

    total_orders = len(orders)
    successful_orders = sum(1 for o in orders if o.status == OrderStatus.FILLED)
    failed_orders = sum(
        1
        for o in orders
        if o.status in [OrderStatus.REJECTED, OrderStatus.FAILED, OrderStatus.EXPIRED]
    )

    # Calculate fill times for successful orders
    fill_times = []
    total_fees = Decimal(0)
    total_volume = Decimal(0)

    for order in orders:
        if order.status == OrderStatus.FILLED and order.fills:
            # Calculate fill time (simplified - using last fill time)
            last_fill_time = max(fill.timestamp for fill in order.fills)
            fill_time = (last_fill_time - order.created_at).total_seconds()
            fill_times.append(fill_time)

            # Sum fees and volume
            total_fees += order.total_fees
            total_volume += order.filled_quantity

    average_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0.0
    fill_rate = (successful_orders / total_orders) * 100 if total_orders > 0 else 0.0

    # Calculate best execution score (simplified metric)
    # Based on fill rate, speed, and fee efficiency
    speed_score = max(0, 100 - (average_fill_time / 60))  # Penalty for slow fills
    fee_score = max(0, 100 - float(total_fees) * 1000)  # Penalty for high fees
    best_execution_score = (fill_rate + speed_score + fee_score) / 3

    return OrderPerformanceMetrics(
        total_orders=total_orders,
        successful_orders=successful_orders,
        failed_orders=failed_orders,
        average_fill_time_seconds=average_fill_time,
        fill_rate_percentage=fill_rate,
        average_slippage_bps=0.0,  # Would need market data to calculate
        total_fees_paid=total_fees,
        volume_traded=total_volume,
        best_execution_score=best_execution_score,
    )


def analyze_order_volume(orders: list[OrderState]) -> OrderVolumeAnalysis:
    """Pure function to analyze order volume patterns."""
    if not orders:
        return OrderVolumeAnalysis(
            total_volume=Decimal(0),
            buy_volume=Decimal(0),
            sell_volume=Decimal(0),
            average_order_size=Decimal(0),
            largest_order_size=Decimal(0),
            smallest_order_size=Decimal(0),
            volume_weighted_average_price=Decimal(0),
        )

    total_volume = Decimal(0)
    buy_volume = Decimal(0)
    sell_volume = Decimal(0)
    order_sizes = []
    weighted_price_sum = Decimal(0)
    total_weighted_volume = Decimal(0)

    for order in orders:
        filled_qty = order.filled_quantity
        order_sizes.append(filled_qty)
        total_volume += filled_qty

        if order.parameters.side == OrderSide.BUY:
            buy_volume += filled_qty
        else:
            sell_volume += filled_qty

        # Calculate volume-weighted average price
        if order.average_fill_price and filled_qty > 0:
            weighted_price_sum += order.average_fill_price * filled_qty
            total_weighted_volume += filled_qty

    average_order_size = total_volume / len(orders) if orders else Decimal(0)
    largest_order_size = max(order_sizes) if order_sizes else Decimal(0)
    smallest_order_size = min(order_sizes) if order_sizes else Decimal(0)
    vwap = (
        weighted_price_sum / total_weighted_volume
        if total_weighted_volume > 0
        else Decimal(0)
    )

    return OrderVolumeAnalysis(
        total_volume=total_volume,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        average_order_size=average_order_size,
        largest_order_size=largest_order_size,
        smallest_order_size=smallest_order_size,
        volume_weighted_average_price=vwap,
    )


def analyze_order_timing(orders: list[OrderState]) -> OrderTimingAnalysis:
    """Pure function to analyze order timing patterns."""
    if not orders:
        return OrderTimingAnalysis(
            total_orders=0,
            orders_per_hour=0.0,
            peak_hour=0,
            off_peak_hour=0,
            average_time_between_orders_minutes=0.0,
            fastest_fill_seconds=0.0,
            slowest_fill_seconds=0.0,
            median_fill_time_seconds=0.0,
        )

    total_orders = len(orders)

    # Analyze order distribution by hour
    hour_counts = [0] * 24
    fill_times = []
    order_timestamps = []

    for order in orders:
        hour = order.created_at.hour
        hour_counts[hour] += 1
        order_timestamps.append(order.created_at)

        # Calculate fill time for filled orders
        if order.status == OrderStatus.FILLED and order.fills:
            last_fill_time = max(fill.timestamp for fill in order.fills)
            fill_time = (last_fill_time - order.created_at).total_seconds()
            fill_times.append(fill_time)

    # Find peak and off-peak hours
    peak_hour = hour_counts.index(max(hour_counts))
    off_peak_hour = hour_counts.index(min(hour_counts))

    # Calculate time between orders
    order_timestamps.sort()
    time_diffs = []
    for i in range(1, len(order_timestamps)):
        diff = (order_timestamps[i] - order_timestamps[i - 1]).total_seconds() / 60
        time_diffs.append(diff)

    avg_time_between = sum(time_diffs) / len(time_diffs) if time_diffs else 0.0

    # Calculate fill time statistics
    fill_times.sort()
    fastest_fill = min(fill_times) if fill_times else 0.0
    slowest_fill = max(fill_times) if fill_times else 0.0
    median_fill = fill_times[len(fill_times) // 2] if fill_times else 0.0

    # Calculate orders per hour
    if len(order_timestamps) >= 2:
        time_span = (order_timestamps[-1] - order_timestamps[0]).total_seconds() / 3600
        orders_per_hour = total_orders / time_span if time_span > 0 else 0.0
    else:
        orders_per_hour = 0.0

    return OrderTimingAnalysis(
        total_orders=total_orders,
        orders_per_hour=orders_per_hour,
        peak_hour=peak_hour,
        off_peak_hour=off_peak_hour,
        average_time_between_orders_minutes=avg_time_between,
        fastest_fill_seconds=fastest_fill,
        slowest_fill_seconds=slowest_fill,
        median_fill_time_seconds=median_fill,
    )


def calculate_order_success_patterns(orders: list[OrderState]) -> dict[str, float]:
    """Pure function to calculate order success patterns by different criteria."""
    if not orders:
        return {}

    patterns = {}

    # Success rate by order type
    for order_type in OrderType:
        type_orders = [o for o in orders if o.parameters.order_type == order_type]
        if type_orders:
            successful = sum(1 for o in type_orders if o.status == OrderStatus.FILLED)
            patterns[f"success_rate_{order_type.value.lower()}"] = (
                successful / len(type_orders)
            ) * 100

    # Success rate by order side
    for side in OrderSide:
        side_orders = [o for o in orders if o.parameters.side == side]
        if side_orders:
            successful = sum(1 for o in side_orders if o.status == OrderStatus.FILLED)
            patterns[f"success_rate_{side.value.lower()}"] = (
                successful / len(side_orders)
            ) * 100

    # Success rate by order size (quartiles)
    sizes = [float(o.parameters.quantity) for o in orders]
    if sizes:
        sizes.sort()
        q1 = sizes[len(sizes) // 4]
        q3 = sizes[3 * len(sizes) // 4]

        small_orders = [o for o in orders if float(o.parameters.quantity) <= q1]
        large_orders = [o for o in orders if float(o.parameters.quantity) >= q3]

        if small_orders:
            successful = sum(1 for o in small_orders if o.status == OrderStatus.FILLED)
            patterns["success_rate_small_orders"] = (
                successful / len(small_orders)
            ) * 100

        if large_orders:
            successful = sum(1 for o in large_orders if o.status == OrderStatus.FILLED)
            patterns["success_rate_large_orders"] = (
                successful / len(large_orders)
            ) * 100

    return patterns


def calculate_order_efficiency_score(orders: list[OrderState]) -> float:
    """Pure function to calculate overall order efficiency score (0-100)."""
    if not orders:
        return 0.0

    performance = analyze_order_performance(orders)
    timing = analyze_order_timing(orders)

    # Weighted components of efficiency
    fill_rate_score = performance.fill_rate_percentage
    speed_score = max(
        0, 100 - (timing.median_fill_time_seconds / 60)
    )  # Penalty for slow fills
    consistency_score = max(
        0, 100 - abs(timing.fastest_fill_seconds - timing.slowest_fill_seconds) / 60
    )

    # Weighted average
    efficiency_score = (
        fill_rate_score * 0.5  # 50% weight on fill rate
        + speed_score * 0.3  # 30% weight on speed
        + consistency_score * 0.2  # 20% weight on consistency
    )

    return min(100.0, max(0.0, efficiency_score))


# Functional analytics composition
def generate_comprehensive_order_analytics(orders: list[OrderState]) -> dict[str, Any]:
    """Generate comprehensive order analytics using pure functional composition."""

    # Apply all analytics functions
    performance = analyze_order_performance(orders)
    volume = analyze_order_volume(orders)
    timing = analyze_order_timing(orders)
    patterns = calculate_order_success_patterns(orders)
    efficiency = calculate_order_efficiency_score(orders)

    return {
        "performance": {
            "total_orders": performance.total_orders,
            "successful_orders": performance.successful_orders,
            "failed_orders": performance.failed_orders,
            "success_rate": performance.success_rate,
            "average_fill_time_seconds": performance.average_fill_time_seconds,
            "fill_rate_percentage": performance.fill_rate_percentage,
            "total_fees_paid": str(performance.total_fees_paid),
            "volume_traded": str(performance.volume_traded),
            "best_execution_score": performance.best_execution_score,
        },
        "volume": {
            "total_volume": str(volume.total_volume),
            "buy_volume": str(volume.buy_volume),
            "sell_volume": str(volume.sell_volume),
            "buy_sell_ratio": volume.buy_sell_ratio,
            "average_order_size": str(volume.average_order_size),
            "largest_order_size": str(volume.largest_order_size),
            "smallest_order_size": str(volume.smallest_order_size),
            "volume_weighted_average_price": str(volume.volume_weighted_average_price),
            "volume_concentration": volume.volume_concentration,
        },
        "timing": {
            "orders_per_hour": timing.orders_per_hour,
            "peak_hour": timing.peak_hour,
            "off_peak_hour": timing.off_peak_hour,
            "average_time_between_orders_minutes": timing.average_time_between_orders_minutes,
            "fastest_fill_seconds": timing.fastest_fill_seconds,
            "slowest_fill_seconds": timing.slowest_fill_seconds,
            "median_fill_time_seconds": timing.median_fill_time_seconds,
        },
        "patterns": patterns,
        "overall_efficiency_score": efficiency,
        "summary": {
            "total_orders": len(orders),
            "efficiency_grade": _efficiency_to_grade(efficiency),
            "primary_recommendation": _generate_recommendation(performance, timing),
        },
    }


def _efficiency_to_grade(score: float) -> str:
    """Convert efficiency score to letter grade."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _generate_recommendation(
    performance: OrderPerformanceMetrics, timing: OrderTimingAnalysis
) -> str:
    """Generate primary recommendation based on analytics."""
    if performance.fill_rate_percentage < 80:
        return "Focus on improving order fill rate"
    if timing.median_fill_time_seconds > 300:  # 5 minutes
        return "Optimize order execution speed"
    if performance.total_fees_paid > performance.volume_traded * Decimal("0.001"):
        return "Reduce trading fees through better execution"
    return "Maintain current execution quality"


# Advanced Functional Error Handling and Timeout Management
@dataclass(frozen=True)
class OrderTimeout:
    """Immutable order timeout configuration."""

    initial_timeout_seconds: int
    max_retries: int
    backoff_multiplier: float
    max_timeout_seconds: int

    def calculate_timeout(self, attempt: int) -> int:
        """Calculate timeout for given attempt with exponential backoff."""
        timeout = self.initial_timeout_seconds * (self.backoff_multiplier**attempt)
        return min(int(timeout), self.max_timeout_seconds)


@dataclass(frozen=True)
class OrderErrorContext:
    """Immutable error context for order operations."""

    order_id: OrderId
    operation: str
    attempt: int
    max_attempts: int
    error: Exception
    timestamp: datetime

    @property
    def is_retryable(self) -> bool:
        """Check if error is retryable."""
        # Define retryable error types
        retryable_types = (
            ConnectionError,
            TimeoutError,
            OSError,
        )

        # Check if it's a specific order execution error
        if isinstance(self.error, OrderExecutionError):
            # Some order errors are retryable (network issues)
            # Others are not (validation errors)
            error_msg = self.error.message.lower()
            non_retryable_keywords = [
                "invalid",
                "validation",
                "insufficient",
                "rejected",
                "unauthorized",
            ]
            return not any(keyword in error_msg for keyword in non_retryable_keywords)

        return isinstance(self.error, retryable_types)

    @property
    def can_retry(self) -> bool:
        """Check if we can still retry."""
        return self.attempt < self.max_attempts and self.is_retryable


def with_order_timeout(
    effect: IOEither[OrderExecutionError, Any],
    timeout_config: OrderTimeout,
    order_id: OrderId,
    operation: str,
) -> IOEither[OrderExecutionError, Any]:
    """Add sophisticated timeout handling to an order effect."""

    def timeout_wrapper():
        import signal
        import time

        def timeout_handler(_signum, _frame):
            raise TimeoutError(
                f"Order {operation} timed out after {timeout_config.initial_timeout_seconds}s"
            )

        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_config.initial_timeout_seconds)

        try:
            start_time = time.time()
            result = effect.run()
            elapsed = time.time() - start_time

            # Log performance metrics
            if elapsed > timeout_config.initial_timeout_seconds * 0.8:
                print(f"Warning: Order {operation} took {elapsed:.2f}s (near timeout)")

            signal.alarm(0)  # Cancel timeout
            signal.signal(signal.SIGALRM, old_handler)
            return result

        except TimeoutError as e:
            signal.signal(signal.SIGALRM, old_handler)
            raise OrderExecutionError(
                f"Timeout during {operation} for order {order_id}: {e!s}", order_id
            )
        except Exception:
            signal.signal(signal.SIGALRM, old_handler)
            raise

    from .effects.io import from_try

    return from_try(timeout_wrapper)


def with_order_retry(
    effect_factory: Callable[[], IOEither[OrderExecutionError, Any]],
    timeout_config: OrderTimeout,
    order_id: OrderId,
    operation: str,
) -> IOEither[OrderExecutionError, Any]:
    """Add sophisticated retry logic with exponential backoff."""

    def retry_wrapper():
        import time

        last_error = None

        for attempt in range(timeout_config.max_retries + 1):
            try:
                # Create fresh effect for each attempt
                effect = effect_factory()

                # Apply timeout for this attempt
                timeout_seconds = timeout_config.calculate_timeout(attempt)
                timed_effect = with_order_timeout(
                    effect,
                    replace(timeout_config, initial_timeout_seconds=timeout_seconds),
                    order_id,
                    operation,
                )

                # Execute with timeout
                result = timed_effect.run()

                # Success - log if we had retries
                if attempt > 0:
                    print(f"Order {operation} succeeded after {attempt + 1} attempts")

                return result

            except OrderExecutionError as e:
                last_error = e

                # Create error context
                error_context = OrderErrorContext(
                    order_id=order_id,
                    operation=operation,
                    attempt=attempt,
                    max_attempts=timeout_config.max_retries,
                    error=e,
                    timestamp=datetime.now(),
                )

                # Check if we should retry
                if not error_context.can_retry:
                    break

                # Calculate backoff delay
                delay = timeout_config.initial_timeout_seconds * (
                    timeout_config.backoff_multiplier**attempt
                )
                delay = min(delay, timeout_config.max_timeout_seconds)

                print(
                    f"Order {operation} failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e.message}"
                )
                time.sleep(delay)
                continue

        # All retries exhausted
        raise last_error or OrderExecutionError(
            f"All retries exhausted for {operation}", order_id
        )

    from .effects.io import from_try

    return from_try(retry_wrapper)


def with_circuit_breaker(
    effect: IOEither[OrderExecutionError, Any],
    failure_threshold: int = 5,
    recovery_timeout_seconds: int = 60,
) -> IOEither[OrderExecutionError, Any]:
    """Add circuit breaker pattern to order effects."""

    # Simple in-memory circuit breaker state
    # In production, this would be shared across processes
    if not hasattr(with_circuit_breaker, "_state"):
        with_circuit_breaker._state = {
            "failure_count": 0,
            "last_failure_time": None,
            "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
        }

    def circuit_wrapper():
        import time

        state = with_circuit_breaker._state
        current_time = time.time()

        # Check if circuit is open
        if state["state"] == "OPEN":
            # Check if recovery timeout has passed
            if (
                state["last_failure_time"]
                and current_time - state["last_failure_time"] > recovery_timeout_seconds
            ):
                state["state"] = "HALF_OPEN"
                print("Circuit breaker moving to HALF_OPEN state")
            else:
                raise OrderExecutionError("Circuit breaker is OPEN - blocking requests")

        try:
            result = effect.run()

            # Success - reset circuit breaker
            if state["state"] == "HALF_OPEN":
                state["state"] = "CLOSED"
                state["failure_count"] = 0
                print("Circuit breaker reset to CLOSED state")

            return result

        except OrderExecutionError:
            # Record failure
            state["failure_count"] += 1
            state["last_failure_time"] = current_time

            # Check if we should open circuit
            if state["failure_count"] >= failure_threshold:
                state["state"] = "OPEN"
                print(f"Circuit breaker opened after {failure_threshold} failures")

            raise

    from .effects.io import from_try

    return from_try(circuit_wrapper)


def with_order_monitoring(
    effect: IOEither[OrderExecutionError, Any], order_id: OrderId, operation: str
) -> IOEither[OrderExecutionError, Any]:
    """Add monitoring and telemetry to order effects."""

    def monitor_wrapper():
        import time

        start_time = time.time()

        try:
            print(f"Starting {operation} for order {order_id}")

            result = effect.run()

            elapsed = time.time() - start_time
            print(f"Completed {operation} for order {order_id} in {elapsed:.3f}s")

            # Record success metrics (in production, would use proper metrics system)
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            print(
                f"Failed {operation} for order {order_id} after {elapsed:.3f}s: {e!s}"
            )

            # Record failure metrics
            raise

    from .effects.io import from_try

    return from_try(monitor_wrapper)


# Effect Composition Combinators
def compose_order_effects(
    base_effect_factory: Callable[[], IOEither[OrderExecutionError, Any]],
    order_id: OrderId,
    operation: str,
    timeout_config: OrderTimeout | None = None,
    use_circuit_breaker: bool = True,
    enable_monitoring: bool = True,
) -> IOEither[OrderExecutionError, Any]:
    """Compose multiple effect enhancers for robust order operations."""

    # Default timeout configuration
    if timeout_config is None:
        timeout_config = OrderTimeout(
            initial_timeout_seconds=30,
            max_retries=3,
            backoff_multiplier=2.0,
            max_timeout_seconds=300,
        )

    def composed_effect():
        # Start with base effect
        effect = base_effect_factory()

        # Add monitoring if enabled
        if enable_monitoring:
            effect = with_order_monitoring(effect, order_id, operation)

        # Add circuit breaker if enabled
        if use_circuit_breaker:
            effect = with_circuit_breaker(effect)

        return effect

    # Add retry with timeout (outermost layer)
    return with_order_retry(composed_effect, timeout_config, order_id, operation)


# Specialized Order Effect Builders
def build_robust_place_order_effect(
    order: OrderState, timeout_config: OrderTimeout | None = None
) -> IOEither[OrderExecutionError, OrderState]:
    """Build a robust place order effect with full error handling."""

    def base_effect_factory():
        return place_order_effect(order)

    return compose_order_effects(
        base_effect_factory,
        order.id,
        "place_order",
        timeout_config,
        use_circuit_breaker=True,
        enable_monitoring=True,
    )


def build_robust_cancel_order_effect(
    order_id: OrderId, order: OrderState, timeout_config: OrderTimeout | None = None
) -> IOEither[OrderExecutionError, OrderState]:
    """Build a robust cancel order effect with full error handling."""

    def base_effect_factory():
        return cancel_order_effect(order_id, order)

    return compose_order_effects(
        base_effect_factory,
        order_id,
        "cancel_order",
        timeout_config,
        use_circuit_breaker=False,  # Cancellation should always be attempted
        enable_monitoring=True,
    )


# Parallel Effect Execution with Error Aggregation
def execute_orders_in_parallel(
    order_effects: list[tuple[OrderId, IOEither[OrderExecutionError, Any]]],
    max_concurrent: int = 5,
) -> IOEither[list[OrderExecutionError], list[Any]]:
    """Execute multiple order effects in parallel with error aggregation."""

    def parallel_execution():
        import concurrent.futures
        import threading

        results = []
        errors = []
        lock = threading.Lock()

        def execute_order_effect(order_id, effect):
            try:
                result = effect.run()
                with lock:
                    results.append((order_id, result))
            except OrderExecutionError as e:
                with lock:
                    errors.append(e)

        # Execute effects in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent
        ) as executor:
            futures = []
            for order_id, effect in order_effects:
                future = executor.submit(execute_order_effect, order_id, effect)
                futures.append(future)

            # Wait for all to complete
            concurrent.futures.wait(futures)

        # Return results or errors
        if errors:
            raise errors[0]  # For simplicity, raise first error

        return [result for _, result in sorted(results)]

    from .effects.io import from_try

    return from_try(parallel_execution)


# Error Recovery Strategies
def with_fallback_strategy(
    primary_effect: IOEither[OrderExecutionError, Any],
    fallback_effect: IOEither[OrderExecutionError, Any],
    order_id: OrderId,
) -> IOEither[OrderExecutionError, Any]:
    """Try primary effect, fall back to secondary on failure."""

    def fallback_wrapper():
        try:
            return primary_effect.run()
        except OrderExecutionError as e:
            print(
                f"Primary effect failed for order {order_id}, trying fallback: {e.message}"
            )
            try:
                return fallback_effect.run()
            except OrderExecutionError as fallback_error:
                # Combine error messages
                combined_message = f"Both primary and fallback failed. Primary: {e.message}, Fallback: {fallback_error.message}"
                raise OrderExecutionError(combined_message, order_id)

    from .effects.io import from_try

    return from_try(fallback_wrapper)
