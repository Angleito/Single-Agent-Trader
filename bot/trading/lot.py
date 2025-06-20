"""FIFO lot tracking for trading positions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class TradeLot:
    """Represents a single lot (batch) of shares/units purchased."""

    lot_id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    quantity: Decimal = Decimal("0")
    purchase_price: Decimal = Decimal("0")
    purchase_date: datetime = field(default_factory=datetime.now)
    remaining_quantity: Decimal = field(default_factory=lambda: Decimal("0"))

    def __post_init__(self):
        """Initialize remaining quantity to match quantity if not set."""
        if self.remaining_quantity == Decimal("0") and self.quantity > Decimal("0"):
            self.remaining_quantity = self.quantity

    @property
    def cost_basis(self) -> Decimal:
        """Total cost of this lot."""
        return self.quantity * self.purchase_price

    @property
    def remaining_cost_basis(self) -> Decimal:
        """Cost basis of remaining shares."""
        return self.remaining_quantity * self.purchase_price

    @property
    def is_fully_sold(self) -> bool:
        """Check if this lot has been completely sold."""
        return self.remaining_quantity <= Decimal("0")

    def sell_from_lot(self, quantity: Decimal) -> tuple[Decimal, Decimal]:
        """
        Sell shares from this lot.

        Args:
            quantity: Number of shares to sell

        Returns:
            Tuple of (shares_sold, remaining_to_sell)
        """
        shares_to_sell = min(self.remaining_quantity, quantity)
        self.remaining_quantity -= shares_to_sell
        remaining_to_sell = quantity - shares_to_sell
        return shares_to_sell, remaining_to_sell


class LotSale(BaseModel):
    """Record of a sale from a specific lot."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lot_id: str
    quantity_sold: Decimal
    sale_price: Decimal
    sale_date: datetime
    cost_basis: Decimal
    realized_pnl: Decimal


class FIFOPosition(BaseModel):
    """Position tracking with FIFO lot accounting."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str
    side: str = Field(pattern="^(LONG|SHORT|FLAT)$")
    lots: list[TradeLot] = Field(default_factory=list)
    sale_history: list[LotSale] = Field(default_factory=list)
    total_realized_pnl: Decimal = Decimal("0")

    @property
    def total_quantity(self) -> Decimal:
        """Total remaining quantity across all lots."""
        return sum(lot.remaining_quantity for lot in self.lots) or Decimal("0")

    @property
    def average_price(self) -> Decimal:
        """Weighted average price of remaining shares."""
        if self.total_quantity == Decimal("0"):
            return Decimal("0")

        total_cost = sum(lot.remaining_cost_basis for lot in self.lots)
        return total_cost / self.total_quantity

    @property
    def total_cost_basis(self) -> Decimal:
        """Total cost basis of remaining shares."""
        return sum(lot.remaining_cost_basis for lot in self.lots) or Decimal("0")

    def add_lot(
        self,
        quantity: Decimal,
        price: Decimal,
        purchase_date: datetime | None = None,
    ) -> TradeLot:
        """
        Add a new lot to the position.

        Args:
            quantity: Number of shares purchased
            price: Purchase price per share
            purchase_date: Date of purchase (defaults to now)

        Returns:
            The created TradeLot
        """
        lot = TradeLot(
            symbol=self.symbol,
            quantity=quantity,
            purchase_price=price,
            purchase_date=purchase_date or datetime.now(),
            remaining_quantity=quantity,
        )
        self.lots.append(lot)

        # Update side if we were flat
        if self.side == "FLAT" and quantity > 0:
            self.side = "LONG"  # Assuming LONG for now, can be extended for SHORT

        return lot

    def sell_fifo(
        self,
        quantity: Decimal,
        sale_price: Decimal,
        sale_date: datetime | None = None,
    ) -> list[LotSale]:
        """
        Sell shares using FIFO accounting.

        Args:
            quantity: Number of shares to sell
            sale_price: Sale price per share
            sale_date: Date of sale (defaults to now)

        Returns:
            List of LotSale records for this transaction
        """
        if quantity > self.total_quantity:
            raise ValueError(
                f"Cannot sell {quantity} shares, only {self.total_quantity} available"
            )

        sales = []
        remaining_to_sell = quantity
        sale_date = sale_date or datetime.now()

        # Sort lots by purchase date (FIFO)
        sorted_lots = sorted(self.lots, key=lambda x: x.purchase_date)

        for lot in sorted_lots:
            if remaining_to_sell <= 0 or lot.is_fully_sold:
                continue

            shares_sold, remaining_to_sell = lot.sell_from_lot(remaining_to_sell)

            if shares_sold > 0:
                cost_basis = shares_sold * lot.purchase_price
                sale_value = shares_sold * sale_price
                realized_pnl = sale_value - cost_basis

                sale_record = LotSale(
                    lot_id=lot.lot_id,
                    quantity_sold=shares_sold,
                    sale_price=sale_price,
                    sale_date=sale_date,
                    cost_basis=cost_basis,
                    realized_pnl=realized_pnl,
                )

                sales.append(sale_record)
                self.sale_history.append(sale_record)
                self.total_realized_pnl += realized_pnl

        # Remove fully sold lots
        self.lots = [lot for lot in self.lots if not lot.is_fully_sold]

        # Update side if position is closed
        if self.total_quantity == Decimal("0"):
            self.side = "FLAT"

        return sales

    def get_tax_lots_report(self) -> dict[str, Any]:
        """
        Generate a report of all tax lots.

        Returns:
            Dictionary with lot details and summary statistics
        """
        active_lots = []
        for lot in sorted(self.lots, key=lambda x: x.purchase_date):
            active_lots.append(
                {
                    "lot_id": lot.lot_id,
                    "purchase_date": lot.purchase_date.isoformat(),
                    "quantity": str(lot.quantity),
                    "remaining_quantity": str(lot.remaining_quantity),
                    "purchase_price": str(lot.purchase_price),
                    "cost_basis": str(lot.cost_basis),
                    "remaining_cost_basis": str(lot.remaining_cost_basis),
                }
            )

        return {
            "symbol": self.symbol,
            "side": self.side,
            "total_quantity": str(self.total_quantity),
            "average_price": str(self.average_price),
            "total_cost_basis": str(self.total_cost_basis),
            "total_realized_pnl": str(self.total_realized_pnl),
            "active_lots": active_lots,
            "total_sales": len(self.sale_history),
        }
