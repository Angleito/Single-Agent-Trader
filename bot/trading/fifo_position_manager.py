"""FIFO-based position manager for the trading bot."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

import aiofiles

from bot.trading.lot import FIFOPosition, LotSale, TradeLot
from bot.trading_types import Order, OrderStatus, Position

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class FIFOPositionManager:
    """Position manager with FIFO lot tracking."""

    def __init__(self, state_file: Path | None = None):
        """Initialize the FIFO position manager."""
        self._positions: dict[str, FIFOPosition] = {}
        self._position_history: list[dict[str, Any]] = []
        self._lock = Lock()
        self._state_file = state_file

        # Flag to avoid async operations during initialization
        self._initializing = True

        if state_file and state_file.exists():
            self._load_state()

        # Initialization complete
        self._initializing = False

    def get_position(self, symbol: str) -> Position:
        """Get current position for a symbol in the legacy format."""
        with self._lock:
            fifo_pos = self._positions.get(symbol)
            if not fifo_pos or fifo_pos.side == "FLAT":
                return Position(
                    symbol=symbol,
                    side="FLAT",
                    size=Decimal("0"),
                    entry_price=None,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=Decimal("0"),
                    timestamp=datetime.now(UTC),
                )

            side_literal: Literal["LONG", "SHORT", "FLAT"] = fifo_pos.side  # type: ignore[assignment]
            return Position(
                symbol=symbol,
                side=side_literal,
                size=fifo_pos.total_quantity,
                entry_price=fifo_pos.average_price,
                unrealized_pnl=Decimal("0"),  # Calculated separately
                realized_pnl=fifo_pos.total_realized_pnl,
                timestamp=datetime.now(UTC),
            )

    def get_fifo_position(self, symbol: str) -> FIFOPosition:
        """Get FIFO position with full lot details."""
        with self._lock:
            if symbol not in self._positions:
                self._positions[symbol] = FIFOPosition(symbol=symbol, side="FLAT")
            return self._positions[symbol]

    def update_position_from_order(
        self, order: Order, current_price: Decimal
    ) -> Position:
        """
        Update position based on a filled order using FIFO accounting.

        Args:
            order: The filled order
            current_price: Current market price for unrealized P&L calculation

        Returns:
            Updated position in legacy format
        """
        if order.status != OrderStatus.FILLED:
            logger.warning(
                "Attempted to update position with non-filled order: %s", order
            )
            return self.get_position(order.symbol)

        with self._lock:
            fifo_pos = self.get_fifo_position(order.symbol)

            # Determine if this is opening or closing a position
            if order.side == "BUY":
                self._handle_buy_order(fifo_pos, order)
            else:  # SELL
                self._handle_sell_order(fifo_pos, order, current_price)

            # Save state after update
            if self._state_file:
                self._save_state()

            # Return position in legacy format
            return self.get_position(order.symbol)

    def _handle_buy_order(self, fifo_pos: FIFOPosition, order: Order) -> None:
        """Handle a buy order (opening or adding to long position)."""
        # For spot trading, BUY means going LONG
        fill_price = (
            order.price if order.price else order.filled_quantity / order.quantity
        )

        if fifo_pos.side == "SHORT":
            # Closing short position first (buy to cover)
            sales = fifo_pos.sell_fifo(
                order.filled_quantity, fill_price, order.timestamp
            )
            logger.info("Closed SHORT position with %s lot sales", len(sales))
        else:
            # Opening or adding to LONG position
            lot = fifo_pos.add_lot(order.filled_quantity, fill_price, order.timestamp)
            logger.info(
                "Added lot %s with %s units at %s",
                lot.lot_id,
                lot.quantity,
                lot.purchase_price,
            )

    def _handle_sell_order(
        self, fifo_pos: FIFOPosition, order: Order, current_price: Decimal
    ) -> None:
        """Handle a sell order (closing long position or opening short)."""
        fill_price = order.price if order.price else current_price

        if fifo_pos.side == "LONG" and fifo_pos.total_quantity > 0:
            # Closing long position using FIFO
            try:
                sales = fifo_pos.sell_fifo(
                    order.filled_quantity, fill_price, order.timestamp
                )

                # Log each lot sale
                for sale in sales:
                    logger.info(
                        "Sold %s units from lot %s " "at %s, realized P&L: %s",
                        sale.quantity_sold,
                        sale.lot_id,
                        sale.sale_price,
                        sale.realized_pnl,
                    )

                # Record to position history
                self._position_history.append(
                    {
                        "timestamp": order.timestamp.isoformat(),
                        "symbol": order.symbol,
                        "action": "SELL",
                        "quantity": str(order.filled_quantity),
                        "price": str(fill_price),
                        "lot_sales": len(sales),
                        "realized_pnl": str(sum(s.realized_pnl for s in sales)),
                    }
                )

            except ValueError:
                logger.exception("Error selling FIFO")
        else:
            # For futures, this would open a SHORT position
            # For spot trading, we typically don't support shorting
            logger.warning(
                "Attempted to sell %s units with no long position",
                order.filled_quantity,
            )

    def get_all_positions(self) -> dict[str, Position]:
        """Get all active positions in legacy format."""
        with self._lock:
            positions = {}
            for symbol, fifo_pos in self._positions.items():
                if fifo_pos.total_quantity > 0:
                    positions[symbol] = self.get_position(symbol)
            return positions

    def get_tax_lots_report(self, symbol: str) -> dict[str, Any]:
        """Get detailed tax lot report for a symbol."""
        with self._lock:
            fifo_pos = self.get_fifo_position(symbol)
            return fifo_pos.get_tax_lots_report()

    def get_realized_pnl(self, symbol: str | None = None) -> Decimal:
        """Get total realized P&L for a symbol or all symbols."""
        with self._lock:
            if symbol:
                fifo_pos = self._positions.get(symbol)
                return fifo_pos.total_realized_pnl if fifo_pos else Decimal("0")

            total_pnl = Decimal("0")
            for pos in self._positions.values():
                total_pnl += pos.total_realized_pnl
            return total_pnl

    async def _save_state_async(self) -> None:
        """Save position state to file asynchronously."""
        if not self._state_file:
            return

        try:
            state: dict[str, Any] = {"positions": {}, "history": self._position_history}

            # Serialize FIFO positions
            for symbol, fifo_pos in self._positions.items():
                state["positions"][symbol] = {
                    "symbol": fifo_pos.symbol,
                    "side": fifo_pos.side,
                    "total_realized_pnl": str(fifo_pos.total_realized_pnl),
                    "lots": [
                        {
                            "lot_id": lot.lot_id,
                            "quantity": str(lot.quantity),
                            "purchase_price": str(lot.purchase_price),
                            "purchase_date": lot.purchase_date.isoformat(),
                            "remaining_quantity": str(lot.remaining_quantity),
                        }
                        for lot in fifo_pos.lots
                    ],
                    "sale_history": [
                        {
                            "lot_id": sale.lot_id,
                            "quantity_sold": str(sale.quantity_sold),
                            "sale_price": str(sale.sale_price),
                            "sale_date": sale.sale_date.isoformat(),
                            "cost_basis": str(sale.cost_basis),
                            "realized_pnl": str(sale.realized_pnl),
                        }
                        for sale in fifo_pos.sale_history
                    ],
                }

            async with aiofiles.open(self._state_file, "w") as f:
                await f.write(json.dumps(state, indent=2))

        except Exception:
            logger.exception("Failed to save position state")

    def _save_state_sync(self) -> None:
        """Save position state to file synchronously (fallback method)."""
        if not self._state_file:
            return

        try:
            state: dict[str, Any] = {"positions": {}, "history": self._position_history}

            # Serialize FIFO positions
            for symbol, fifo_pos in self._positions.items():
                state["positions"][symbol] = {
                    "symbol": fifo_pos.symbol,
                    "side": fifo_pos.side,
                    "total_realized_pnl": str(fifo_pos.total_realized_pnl),
                    "lots": [
                        {
                            "lot_id": lot.lot_id,
                            "quantity": str(lot.quantity),
                            "purchase_price": str(lot.purchase_price),
                            "purchase_date": lot.purchase_date.isoformat(),
                            "remaining_quantity": str(lot.remaining_quantity),
                        }
                        for lot in fifo_pos.lots
                    ],
                    "sale_history": [
                        {
                            "lot_id": sale.lot_id,
                            "quantity_sold": str(sale.quantity_sold),
                            "sale_price": str(sale.sale_price),
                            "sale_date": sale.sale_date.isoformat(),
                            "cost_basis": str(sale.cost_basis),
                            "realized_pnl": str(sale.realized_pnl),
                        }
                        for sale in fifo_pos.sale_history
                    ],
                }

            self._state_file.write_text(json.dumps(state, indent=2))

        except Exception:
            logger.exception("Failed to save position state (sync)")

    def reconcile_position_from_exchange(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        size: Decimal,
        entry_price: Decimal,
    ) -> None:
        """
        Reconcile FIFO position from exchange position data.

        This creates a synthetic position based on exchange data without
        individual lot tracking. Used during startup reconciliation.

        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            size: Position size
            entry_price: Entry price from exchange
        """
        with self._lock:
            # Create a new FIFO position
            fifo_pos = FIFOPosition(
                symbol=symbol,
                side=side,
                total_realized_pnl=Decimal("0"),
            )

            # Create a single synthetic lot representing the entire position
            synthetic_lot = TradeLot(
                lot_id="exchange_reconcile_%s_%s"
                % (symbol, datetime.now(UTC).isoformat()),
                symbol=symbol,
                quantity=size,
                purchase_price=entry_price,
                purchase_date=datetime.now(UTC),
                remaining_quantity=size,
            )

            fifo_pos.lots.append(synthetic_lot)
            self._positions[symbol] = fifo_pos

            # Save state
            self._save_state()

            logger.info(
                "FIFO position reconciled from exchange for %s: " "%s %s @ %s",
                symbol,
                side,
                size,
                entry_price,
            )

    def _save_state(self) -> None:
        """Save position state to file (non-blocking when called from async context)."""
        if not self._state_file:
            return

        # During initialization, always save synchronously to avoid event loop issues
        if getattr(self, "_initializing", False):
            self._save_state_sync()
            return

        try:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context, schedule the save as a fire-and-forget task
                task = asyncio.create_task(self._save_state_async())
                # Add error handling callback
                task.add_done_callback(self._handle_save_error)
            except RuntimeError:
                # No event loop running, do synchronous save
                self._save_state_sync()
        except Exception:
            logger.exception("Failed to save state")

    def _handle_save_error(self, task: asyncio.Task) -> None:
        """Handle errors from async save task."""
        try:
            task.result()
        except Exception:
            logger.exception("Async save failed")

    def _load_state(self) -> None:
        """Load position state from file."""
        if not self._state_file or not self._state_file.exists():
            return

        try:
            content = self._state_file.read_text().strip()
            if not content:
                logger.info("FIFO positions file is empty, starting with clean state")
                return

            try:
                state = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Invalid JSON in FIFO positions file: %s. Starting with clean state",
                    e,
                )
                return

            # Restore position history
            self._position_history = state.get("history", [])

            # Restore FIFO positions
            for symbol, pos_data in state.get("positions", {}).items():
                fifo_pos = FIFOPosition(
                    symbol=symbol,
                    side=pos_data["side"],
                    total_realized_pnl=Decimal(pos_data["total_realized_pnl"]),
                )

                # Restore lots
                for lot_data in pos_data.get("lots", []):
                    lot = TradeLot(
                        lot_id=lot_data["lot_id"],
                        symbol=symbol,
                        quantity=Decimal(lot_data["quantity"]),
                        purchase_price=Decimal(lot_data["purchase_price"]),
                        purchase_date=datetime.fromisoformat(lot_data["purchase_date"]),
                        remaining_quantity=Decimal(lot_data["remaining_quantity"]),
                    )
                    fifo_pos.lots.append(lot)

                # Restore sale history
                for sale_data in pos_data.get("sale_history", []):
                    sale = LotSale(
                        lot_id=sale_data["lot_id"],
                        quantity_sold=Decimal(sale_data["quantity_sold"]),
                        sale_price=Decimal(sale_data["sale_price"]),
                        sale_date=datetime.fromisoformat(sale_data["sale_date"]),
                        cost_basis=Decimal(sale_data["cost_basis"]),
                        realized_pnl=Decimal(sale_data["realized_pnl"]),
                    )
                    fifo_pos.sale_history.append(sale)

                self._positions[symbol] = fifo_pos

            logger.info("Loaded %s positions from state file", len(self._positions))

        except Exception:
            logger.exception("Failed to load position state")
            # Continue with empty state
