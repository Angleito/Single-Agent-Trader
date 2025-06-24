"""Immutable position types and state management for functional position tracking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Literal

from bot.fp.types.effects import Maybe, Nothing, Some
from bot.fp.types.result import Failure, Result, Success


# Core Position Types
@dataclass(frozen=True)
class PositionSide:
    """Position side as an algebraic data type."""

    value: Literal["LONG", "SHORT", "FLAT"]

    def is_long(self) -> bool:
        """Check if position is long."""
        return self.value == "LONG"

    def is_short(self) -> bool:
        """Check if position is short."""
        return self.value == "SHORT"

    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.value == "FLAT"

    def opposite(self) -> PositionSide:
        """Get opposite position side."""
        if self.value == "LONG":
            return PositionSide("SHORT")
        if self.value == "SHORT":
            return PositionSide("LONG")
        return PositionSide("FLAT")


# Position side constants
LONG = PositionSide("LONG")
SHORT = PositionSide("SHORT")
FLAT = PositionSide("FLAT")


@dataclass(frozen=True)
class FunctionalLot:
    """Immutable representation of a trading lot for FIFO accounting."""

    lot_id: str
    symbol: str
    quantity: Decimal
    purchase_price: Decimal
    purchase_date: datetime
    remaining_quantity: Decimal

    def __post_init__(self) -> None:
        """Validate lot data."""
        if self.quantity <= 0:
            raise ValueError(f"Lot quantity must be positive, got {self.quantity}")
        if self.purchase_price <= 0:
            raise ValueError(
                f"Purchase price must be positive, got {self.purchase_price}"
            )
        if self.remaining_quantity < 0:
            raise ValueError(
                f"Remaining quantity cannot be negative, got {self.remaining_quantity}"
            )
        if self.remaining_quantity > self.quantity:
            raise ValueError(
                f"Remaining quantity {self.remaining_quantity} cannot exceed total {self.quantity}"
            )

    @property
    def is_fully_consumed(self) -> bool:
        """Check if lot is fully consumed."""
        return self.remaining_quantity == Decimal(0)

    @property
    def consumed_quantity(self) -> Decimal:
        """Get consumed quantity."""
        return self.quantity - self.remaining_quantity

    @property
    def cost_basis(self) -> Decimal:
        """Get total cost basis for the lot."""
        return self.quantity * self.purchase_price

    @property
    def remaining_cost_basis(self) -> Decimal:
        """Get remaining cost basis."""
        return self.remaining_quantity * self.purchase_price

    def consume(self, amount: Decimal) -> Result[str, FunctionalLot]:
        """
        Consume amount from lot, returning new lot.

        Args:
            amount: Amount to consume

        Returns:
            Result containing new lot or error message
        """
        if amount <= 0:
            return Failure(f"Consume amount must be positive, got {amount}")

        if amount > self.remaining_quantity:
            return Failure(
                f"Cannot consume {amount} from lot with only {self.remaining_quantity} remaining"
            )

        new_lot = replace(self, remaining_quantity=self.remaining_quantity - amount)
        return Success(new_lot)


@dataclass(frozen=True)
class LotSale:
    """Immutable representation of a lot sale for realized P&L tracking."""

    lot_id: str
    quantity_sold: Decimal
    sale_price: Decimal
    sale_date: datetime
    cost_basis: Decimal

    def __post_init__(self) -> None:
        """Validate sale data."""
        if self.quantity_sold <= 0:
            raise ValueError(
                f"Sale quantity must be positive, got {self.quantity_sold}"
            )
        if self.sale_price <= 0:
            raise ValueError(f"Sale price must be positive, got {self.sale_price}")
        if self.cost_basis <= 0:
            raise ValueError(f"Cost basis must be positive, got {self.cost_basis}")

    @property
    def realized_pnl(self) -> Decimal:
        """Calculate realized P&L for this sale."""
        return (self.sale_price * self.quantity_sold) - self.cost_basis

    @property
    def sale_value(self) -> Decimal:
        """Get total sale value."""
        return self.sale_price * self.quantity_sold


@dataclass(frozen=True)
class FunctionalPosition:
    """Immutable position representation with functional state management."""

    symbol: str
    side: PositionSide
    lots: tuple[FunctionalLot, ...]
    sales_history: tuple[LotSale, ...]
    unrealized_pnl: Decimal
    last_update: datetime

    def __post_init__(self) -> None:
        """Validate position data."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.side.is_flat() or self.total_quantity == Decimal(0)

    @property
    def total_quantity(self) -> Decimal:
        """Calculate total quantity across all lots."""
        return sum(lot.remaining_quantity for lot in self.lots)

    @property
    def average_price(self) -> Maybe[Decimal]:
        """Calculate weighted average price of remaining lots."""
        if not self.lots or self.total_quantity == Decimal(0):
            return Nothing()

        total_cost = sum(lot.remaining_cost_basis for lot in self.lots)
        return Some(total_cost / self.total_quantity)

    @property
    def total_realized_pnl(self) -> Decimal:
        """Calculate total realized P&L from all sales."""
        return sum(sale.realized_pnl for sale in self.sales_history)

    @property
    def position_value(self) -> Decimal:
        """Calculate current position value."""
        avg_price = self.average_price
        if avg_price.is_nothing():
            return Decimal(0)
        return self.total_quantity * avg_price.value

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.unrealized_pnl

    def add_lot(
        self,
        quantity: Decimal,
        price: Decimal,
        timestamp: datetime,
        lot_id: str | None = None,
    ) -> Result[str, FunctionalPosition]:
        """
        Add a new lot to the position.

        Args:
            quantity: Lot quantity
            price: Purchase price
            timestamp: Purchase timestamp
            lot_id: Optional lot ID (generated if not provided)

        Returns:
            Result containing new position or error message
        """
        try:
            if lot_id is None:
                lot_id = f"{self.symbol}_{timestamp.isoformat()}_{len(self.lots)}"

            new_lot = FunctionalLot(
                lot_id=lot_id,
                symbol=self.symbol,
                quantity=quantity,
                purchase_price=price,
                purchase_date=timestamp,
                remaining_quantity=quantity,
            )

            # Determine new side
            if self.is_flat:
                new_side = LONG  # Adding to flat position makes it long
            elif self.side.is_long():
                new_side = LONG  # Adding to long position keeps it long
            else:
                # This would be covering a short position - handle separately
                return Failure(
                    "Cannot add lot to short position - use sell_fifo instead"
                )

            new_position = replace(
                self, side=new_side, lots=self.lots + (new_lot,), last_update=timestamp
            )

            return Success(new_position)

        except (ValueError, TypeError) as e:
            return Failure(f"Failed to add lot: {e!s}")

    def sell_fifo(
        self, quantity: Decimal, price: Decimal, timestamp: datetime
    ) -> Result[str, tuple[FunctionalPosition, tuple[LotSale, ...]]]:
        """
        Sell quantity using FIFO accounting.

        Args:
            quantity: Quantity to sell
            price: Sale price
            timestamp: Sale timestamp

        Returns:
            Result containing (new_position, sales_made) or error message
        """
        if quantity <= 0:
            return Failure(f"Sell quantity must be positive, got {quantity}")

        if quantity > self.total_quantity:
            return Failure(
                f"Cannot sell {quantity} - only {self.total_quantity} available"
            )

        remaining_to_sell = quantity
        new_lots = []
        sales = []

        # Process lots in FIFO order
        for lot in self.lots:
            if remaining_to_sell <= Decimal(0):
                # Keep remaining lots unchanged
                new_lots.append(lot)
                continue

            # Determine how much to sell from this lot
            sell_from_lot = min(remaining_to_sell, lot.remaining_quantity)

            # Create sale record
            cost_basis = sell_from_lot * lot.purchase_price
            sale = LotSale(
                lot_id=lot.lot_id,
                quantity_sold=sell_from_lot,
                sale_price=price,
                sale_date=timestamp,
                cost_basis=cost_basis,
            )
            sales.append(sale)

            # Update remaining quantity
            remaining_to_sell -= sell_from_lot

            # Update lot
            lot_result = lot.consume(sell_from_lot)
            if lot_result.is_failure():
                return Failure(f"Failed to consume from lot: {lot_result.error}")

            updated_lot = lot_result.value
            if not updated_lot.is_fully_consumed:
                new_lots.append(updated_lot)

        # Determine new side
        total_remaining = sum(lot.remaining_quantity for lot in new_lots)
        if total_remaining == Decimal(0):
            new_side = FLAT
        else:
            new_side = self.side  # Keep same side if quantity remains

        # Create new position
        new_position = replace(
            self,
            side=new_side,
            lots=tuple(new_lots),
            sales_history=self.sales_history + tuple(sales),
            last_update=timestamp,
        )

        return Success((new_position, tuple(sales)))

    def update_unrealized_pnl(self, current_price: Decimal) -> FunctionalPosition:
        """
        Update unrealized P&L based on current price.

        Args:
            current_price: Current market price

        Returns:
            New position with updated unrealized P&L
        """
        if self.is_flat:
            return replace(self, unrealized_pnl=Decimal(0))

        avg_price = self.average_price
        if avg_price.is_nothing():
            return replace(self, unrealized_pnl=Decimal(0))

        price_diff = current_price - avg_price.value

        if self.side.is_long():
            unrealized = self.total_quantity * price_diff
        else:  # SHORT
            unrealized = self.total_quantity * (-price_diff)

        return replace(self, unrealized_pnl=unrealized)

    def with_timestamp(self, timestamp: datetime) -> FunctionalPosition:
        """Update position with new timestamp."""
        return replace(self, last_update=timestamp)


@dataclass(frozen=True)
class PositionSnapshot:
    """Immutable snapshot of position state at a point in time."""

    timestamp: datetime
    positions: tuple[FunctionalPosition, ...]
    total_unrealized_pnl: Decimal
    total_realized_pnl: Decimal

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L across all positions."""
        return self.total_unrealized_pnl + self.total_realized_pnl

    @property
    def position_count(self) -> int:
        """Get number of active positions."""
        return len([pos for pos in self.positions if not pos.is_flat])

    @property
    def symbols(self) -> tuple[str, ...]:
        """Get all symbols with positions."""
        return tuple(pos.symbol for pos in self.positions if not pos.is_flat)

    def get_position(self, symbol: str) -> Maybe[FunctionalPosition]:
        """Get position for a specific symbol."""
        for pos in self.positions:
            if pos.symbol == symbol:
                return Some(pos)
        return Nothing()

    def update_position(self, position: FunctionalPosition) -> PositionSnapshot:
        """Update a position in the snapshot."""
        new_positions = []
        found = False

        for pos in self.positions:
            if pos.symbol == position.symbol:
                new_positions.append(position)
                found = True
            else:
                new_positions.append(pos)

        if not found:
            new_positions.append(position)

        # Recalculate totals
        total_unrealized = sum(pos.unrealized_pnl for pos in new_positions)
        total_realized = sum(pos.total_realized_pnl for pos in new_positions)

        return replace(
            self,
            positions=tuple(new_positions),
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            timestamp=position.last_update,
        )


# Factory functions
def create_empty_position(
    symbol: str, timestamp: datetime | None = None
) -> FunctionalPosition:
    """Create an empty (flat) position."""
    if timestamp is None:
        timestamp = datetime.now()

    return FunctionalPosition(
        symbol=symbol,
        side=FLAT,
        lots=(),
        sales_history=(),
        unrealized_pnl=Decimal(0),
        last_update=timestamp,
    )


def create_position_from_lot(
    symbol: str,
    quantity: Decimal,
    price: Decimal,
    timestamp: datetime | None = None,
    lot_id: str | None = None,
) -> Result[str, FunctionalPosition]:
    """Create a position from a single lot."""
    if timestamp is None:
        timestamp = datetime.now()

    empty_pos = create_empty_position(symbol, timestamp)
    return empty_pos.add_lot(quantity, price, timestamp, lot_id)


def create_empty_snapshot(timestamp: datetime | None = None) -> PositionSnapshot:
    """Create an empty position snapshot."""
    if timestamp is None:
        timestamp = datetime.now()

    return PositionSnapshot(
        timestamp=timestamp,
        positions=(),
        total_unrealized_pnl=Decimal(0),
        total_realized_pnl=Decimal(0),
    )


# Enhanced Position Types for Margin and Leverage Support

from enum import Enum


class MarginType(Enum):
    """Margin type enumeration."""

    CROSS = "CROSS"
    ISOLATED = "ISOLATED"


class MarginHealthStatus(Enum):
    """Margin health status enumeration."""

    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    LIQUIDATION_RISK = "LIQUIDATION_RISK"


@dataclass(frozen=True)
class FuturesMarginInfo:
    """Immutable futures margin information."""

    margin_type: MarginType
    initial_margin: Decimal
    maintenance_margin: Decimal
    margin_used: Decimal
    margin_available: Decimal
    unrealized_pnl: Decimal
    equity: Decimal
    leverage: Decimal
    margin_ratio: Decimal
    liquidation_price: Maybe[Decimal]
    liquidation_threshold: Decimal
    last_updated: datetime

    def __post_init__(self) -> None:
        """Validate margin info."""
        if self.initial_margin < 0:
            raise ValueError("Initial margin cannot be negative")
        if self.maintenance_margin < 0:
            raise ValueError("Maintenance margin cannot be negative")
        if self.leverage < Decimal(1):
            raise ValueError("Leverage must be >= 1")
        if self.liquidation_threshold <= 0:
            raise ValueError("Liquidation threshold must be positive")

    @property
    def health_status(self) -> MarginHealthStatus:
        """Calculate margin health status."""
        if self.margin_ratio <= self.liquidation_threshold:
            return MarginHealthStatus.LIQUIDATION_RISK
        if self.margin_ratio <= self.liquidation_threshold * Decimal("1.2"):
            return MarginHealthStatus.CRITICAL
        if self.margin_ratio <= self.liquidation_threshold * Decimal("1.5"):
            return MarginHealthStatus.WARNING
        return MarginHealthStatus.HEALTHY

    @property
    def is_healthy(self) -> bool:
        """Check if margin is in healthy state."""
        return self.health_status == MarginHealthStatus.HEALTHY

    @property
    def needs_attention(self) -> bool:
        """Check if margin needs attention."""
        return self.health_status in [
            MarginHealthStatus.WARNING,
            MarginHealthStatus.CRITICAL,
            MarginHealthStatus.LIQUIDATION_RISK,
        ]

    def with_updated_equity(
        self, new_equity: Decimal, new_unrealized_pnl: Decimal
    ) -> FuturesMarginInfo:
        """Return new margin info with updated equity and unrealized P&L."""
        new_margin_ratio = (
            new_equity / self.margin_used if self.margin_used > 0 else Decimal(0)
        )
        new_margin_available = max(Decimal(0), new_equity - self.margin_used)

        return replace(
            self,
            equity=new_equity,
            unrealized_pnl=new_unrealized_pnl,
            margin_ratio=new_margin_ratio,
            margin_available=new_margin_available,
            last_updated=datetime.now(),
        )

    def calculate_max_position_size(self, price: Decimal) -> Decimal:
        """Calculate maximum position size based on available margin."""
        if price <= 0:
            return Decimal(0)

        max_value = self.margin_available * self.leverage
        return max_value / price


@dataclass(frozen=True)
class FuturesPosition(FunctionalPosition):
    """Extended position with futures-specific information."""

    margin_info: Maybe[FuturesMarginInfo]
    leverage: Decimal
    mark_price: Maybe[Decimal]
    funding_rate: Maybe[Decimal]
    next_funding_time: Maybe[datetime]

    def __post_init__(self) -> None:
        """Validate futures position."""
        super().__post_init__()
        if self.leverage < Decimal(1):
            raise ValueError("Leverage must be >= 1")

    @property
    def is_futures(self) -> bool:
        """Check if this is a futures position."""
        return True

    @property
    def margin_health(self) -> Maybe[MarginHealthStatus]:
        """Get margin health status."""
        return self.margin_info.map(lambda m: m.health_status)

    @property
    def liquidation_price(self) -> Maybe[Decimal]:
        """Get liquidation price."""
        return self.margin_info.flat_map(lambda m: m.liquidation_price)

    @property
    def position_notional(self) -> Decimal:
        """Calculate position notional value."""
        if self.is_flat or self.total_quantity == Decimal(0):
            return Decimal(0)

        avg_price = self.average_price
        if avg_price.is_nothing():
            return Decimal(0)

        return self.total_quantity * avg_price.value

    @property
    def margin_usage_ratio(self) -> Maybe[Decimal]:
        """Calculate margin usage ratio."""
        return self.margin_info.map(
            lambda m: m.margin_used / m.equity if m.equity > 0 else Decimal(1)
        )

    def calculate_unrealized_pnl_with_mark(self, mark_price: Decimal) -> Decimal:
        """Calculate unrealized P&L using mark price."""
        if self.is_flat or self.total_quantity == Decimal(0):
            return Decimal(0)

        avg_price = self.average_price
        if avg_price.is_nothing():
            return Decimal(0)

        price_diff = mark_price - avg_price.value

        if self.side.is_long():
            return self.total_quantity * price_diff
        if self.side.is_short():
            return self.total_quantity * (-price_diff)
        return Decimal(0)

    def calculate_funding_payment(self) -> Maybe[Decimal]:
        """Calculate funding payment for the position."""
        if self.is_flat or self.funding_rate.is_nothing():
            return Nothing()

        funding_rate = self.funding_rate.value
        notional = self.position_notional

        if self.side.is_long():
            payment = notional * funding_rate
        elif self.side.is_short():
            payment = -(notional * funding_rate)
        else:
            payment = Decimal(0)

        return Some(payment)

    def with_updated_margin(
        self, new_margin_info: FuturesMarginInfo
    ) -> FuturesPosition:
        """Return new position with updated margin info."""
        return replace(self, margin_info=Some(new_margin_info))

    def with_mark_price(self, new_mark_price: Decimal) -> FuturesPosition:
        """Return new position with updated mark price."""
        return replace(self, mark_price=Some(new_mark_price))


@dataclass(frozen=True)
class FuturesOrderRequest:
    """Immutable futures order request."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    order_type: str  # MARKET, LIMIT, STOP_MARKET, etc.
    price: Maybe[Decimal]
    leverage: Decimal
    margin_type: MarginType
    reduce_only: bool
    time_in_force: str

    def __post_init__(self) -> None:
        """Validate order request."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.leverage < Decimal(1):
            raise ValueError("Leverage must be >= 1")

    @property
    def notional_value(self) -> Maybe[Decimal]:
        """Calculate notional value if price is available."""
        return self.price.map(lambda p: self.quantity * p)

    @property
    def required_margin(self) -> Maybe[Decimal]:
        """Calculate required margin for the order."""
        return self.notional_value.map(lambda n: n / self.leverage)


@dataclass(frozen=True)
class FuturesLiquidationEvent:
    """Immutable futures liquidation event."""

    position_id: str
    symbol: str
    side: PositionSide
    liquidated_quantity: Decimal
    liquidation_price: Decimal
    bankruptcy_price: Decimal
    insurance_fund_used: Decimal
    timestamp: datetime
    reason: str

    @property
    def liquidation_value(self) -> Decimal:
        """Calculate liquidation value."""
        return self.liquidated_quantity * self.liquidation_price

    @property
    def was_partial_liquidation(self) -> bool:
        """Check if this was a partial liquidation."""
        return "partial" in self.reason.lower()


# Factory functions for futures positions


def create_futures_position(
    symbol: str,
    side: PositionSide,
    quantity: Decimal,
    entry_price: Decimal,
    leverage: Decimal,
    margin_type: MarginType = MarginType.CROSS,
    timestamp: datetime | None = None,
) -> Result[str, FuturesPosition]:
    """Create a futures position with margin info."""
    if timestamp is None:
        timestamp = datetime.now()

    try:
        # Calculate margin requirements
        notional_value = quantity * entry_price
        required_margin = notional_value / leverage

        # Create margin info
        margin_info = FuturesMarginInfo(
            margin_type=margin_type,
            initial_margin=required_margin,
            maintenance_margin=required_margin * Decimal("0.8"),  # Simplified
            margin_used=required_margin,
            margin_available=Decimal(
                0
            ),  # Will be updated when account info is available
            unrealized_pnl=Decimal(0),
            equity=required_margin,  # Initial equity equals margin used
            leverage=leverage,
            margin_ratio=Decimal(1),  # Initial ratio
            liquidation_price=Nothing(),  # Will be calculated based on position
            liquidation_threshold=Decimal("1.1"),  # 110% maintenance margin
            last_updated=timestamp,
        )

        # Create the lot
        lot = FunctionalLot(
            lot_id=f"{symbol}_{timestamp.isoformat()}",
            symbol=symbol,
            quantity=quantity,
            purchase_price=entry_price,
            purchase_date=timestamp,
            remaining_quantity=quantity,
        )

        # Create futures position
        position = FuturesPosition(
            symbol=symbol,
            side=side,
            lots=(lot,),
            sales_history=(),
            unrealized_pnl=Decimal(0),
            last_update=timestamp,
            margin_info=Some(margin_info),
            leverage=leverage,
            mark_price=Some(entry_price),  # Initially same as entry price
            funding_rate=Nothing(),
            next_funding_time=Nothing(),
        )

        return Success(position)

    except (ValueError, TypeError) as e:
        return Failure(f"Failed to create futures position: {e!s}")


def calculate_liquidation_price(
    position: FuturesPosition, maintenance_margin_ratio: Decimal
) -> Maybe[Decimal]:
    """Calculate liquidation price for a futures position."""
    if position.is_flat or position.margin_info.is_nothing():
        return Nothing()

    margin_info = position.margin_info.value
    avg_price = position.average_price

    if avg_price.is_nothing():
        return Nothing()

    try:
        entry_price = avg_price.value
        quantity = position.total_quantity

        if position.side.is_long():
            # For long positions: liquidation_price = entry_price * (1 - 1/leverage + maintenance_margin_ratio)
            liquidation_price = entry_price * (
                Decimal(1) - (Decimal(1) / position.leverage) + maintenance_margin_ratio
            )
        elif position.side.is_short():
            # For short positions: liquidation_price = entry_price * (1 + 1/leverage - maintenance_margin_ratio)
            liquidation_price = entry_price * (
                Decimal(1) + (Decimal(1) / position.leverage) - maintenance_margin_ratio
            )
        else:
            return Nothing()

        return Some(max(Decimal(0), liquidation_price))

    except (ValueError, ArithmeticError):
        return Nothing()


def update_position_with_funding(
    position: FuturesPosition, funding_rate: Decimal, next_funding_time: datetime
) -> FuturesPosition:
    """Update position with funding information."""
    return replace(
        position,
        funding_rate=Some(funding_rate),
        next_funding_time=Some(next_funding_time),
        last_update=datetime.now(),
    )


@dataclass(frozen=True)
class TradeResult:
    """Result of a trade execution."""

    position_id: str
    symbol: str
    side: PositionSide
    size: Decimal
    price: Decimal
    timestamp: datetime
    fee: Decimal = Decimal(0)
    success: bool = True
    error_message: str = ""

    @classmethod
    def success_trade(
        cls,
        position_id: str,
        symbol: str,
        side: PositionSide,
        size: Decimal,
        price: Decimal,
        fee: Decimal = Decimal(0),
    ) -> TradeResult:
        """Create a successful trade result."""
        return cls(
            position_id=position_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            timestamp=datetime.now(),
            fee=fee,
            success=True,
            error_message="",
        )

    @classmethod
    def failed_trade(
        cls,
        position_id: str,
        symbol: str,
        side: PositionSide,
        size: Decimal,
        price: Decimal,
        error_message: str,
    ) -> TradeResult:
        """Create a failed trade result."""
        return cls(
            position_id=position_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            timestamp=datetime.now(),
            fee=Decimal(0),
            success=False,
            error_message=error_message,
        )
