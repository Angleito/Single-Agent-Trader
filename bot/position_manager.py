"""
Position management system for tracking active trades.

This module handles position tracking, P&L calculations, risk metrics,
and state persistence for the trading bot.
"""

import asyncio
import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional, Union

import aiofiles

from .config import settings
from .paper_trading import PaperTradingAccount
from .trading.fifo_position_manager import FIFOPositionManager
from .types import Order, Position

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Position management system for tracking active trades.

    Manages current positions, calculates unrealized P&L, tracks entry/exit
    prices, and provides risk metrics for decision making.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        paper_trading_account: Optional[PaperTradingAccount] = None,
        use_fifo: bool = True,
    ) -> None:
        """
        Initialize the position manager.

        Args:
            data_dir: Directory for state persistence (default: data/positions)
            paper_trading_account: Paper trading account for enhanced simulation
            use_fifo: Whether to use FIFO accounting (default: True)
        """
        self.data_dir = data_dir or Path("data/positions")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_fifo = use_fifo

        # Thread-safe position storage
        self._positions: dict[str, Position] = {}
        self._position_history: list[Position] = []
        self._lock = threading.RLock()

        # FIFO position manager
        if self.use_fifo:
            fifo_state_file = self.data_dir / "fifo_positions.json"
            self.fifo_manager = FIFOPositionManager(state_file=fifo_state_file)
            logger.info("Using FIFO position tracking")

        # Paper trading integration
        self.paper_account = paper_trading_account
        if settings.system.dry_run and not self.paper_account:
            # Initialize paper trading account for dry run mode
            self.paper_account = PaperTradingAccount(
                data_dir=self.data_dir.parent / "paper_trading"
            )

        # State file paths
        self.positions_file = self.data_dir / "positions.json"
        self.history_file = self.data_dir / "position_history.json"

        # Load persisted state
        self._load_state()

        logger.info(
            f"Initialized PositionManager with {len(self._positions)} active positions"
        )
        if self.paper_account:
            account_status = self.paper_account.get_account_status()
            logger.info(
                f"Paper trading account: ${account_status['equity']:,.2f} equity, {account_status['open_positions']} open positions"
            )

    def get_position(self, symbol: str) -> Position:
        """
        Get current position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position object (FLAT if no position)
        """
        if self.use_fifo:
            return self.fifo_manager.get_position(symbol)

        with self._lock:
            if symbol in self._positions:
                return self._positions[symbol].copy()

            # Return flat position if none exists
            return Position(
                symbol=symbol,
                side="FLAT",
                size=Decimal("0"),
                entry_price=None,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                timestamp=datetime.now(UTC),
            )

    def get_all_positions(self) -> list[Position]:
        """
        Get all active positions.

        Returns:
            List of Position objects
        """
        with self._lock:
            return [pos.copy() for pos in self._positions.values()]

    def update_position_from_order(self, order: Order, fill_price: Decimal) -> Position:
        """
        Update position based on order fill.

        Args:
            order: Filled order
            fill_price: Actual fill price

        Returns:
            Updated position
        """
        if self.use_fifo:
            # Use FIFO manager for position updates
            position = self.fifo_manager.update_position_from_order(order, fill_price)

            # Update legacy storage for compatibility
            with self._lock:
                if position.side == "FLAT":
                    if order.symbol in self._positions:
                        self._positions.pop(order.symbol)
                else:
                    self._positions[order.symbol] = position
                self._save_state()

            return position

        with self._lock:
            symbol = order.symbol
            current_pos = self.get_position(symbol)

            # Calculate new position after order fill
            new_position = self._calculate_new_position(current_pos, order, fill_price)

            # Update position storage
            if new_position.side == "FLAT":
                # Position closed - move to history and remove from active
                if symbol in self._positions:
                    closed_pos = self._positions.pop(symbol)
                    closed_pos.timestamp = datetime.now(UTC)
                    self._position_history.append(closed_pos)
                    logger.info(
                        f"Position closed for {symbol}: {closed_pos.realized_pnl} PnL"
                    )
            else:
                # Active position - update or create
                self._positions[symbol] = new_position
                logger.info(
                    f"Position updated for {symbol}: {new_position.side} {new_position.size} @ {new_position.entry_price}"
                )

            # Persist state
            self._save_state()

            return new_position.copy()

    def update_position_from_exchange(
        self, symbol: str, side: str, size: Decimal, entry_price: Decimal
    ) -> Position:
        """
        Update position state based on actual exchange position data.

        This method is used during startup to reconcile local position state
        with actual positions found on the exchange.

        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            size: Position size
            entry_price: Entry price from exchange

        Returns:
            Updated position
        """
        with self._lock:
            new_position = Position(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                timestamp=datetime.now(UTC),
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0"),
            )

            # Update position storage
            self._positions[symbol] = new_position

            # Update FIFO manager if enabled
            if self.use_fifo:
                self.fifo_manager.reconcile_position_from_exchange(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                )

            # Persist state
            self._save_state()

            logger.info(
                f"Position reconciled from exchange for {symbol}: {side} {size} @ {entry_price}"
            )

            return new_position.copy()

    def update_unrealized_pnl(self, symbol: str, current_price: Decimal) -> Decimal:
        """
        Update unrealized P&L for a position.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Updated unrealized P&L
        """
        with self._lock:
            if symbol not in self._positions:
                return Decimal("0")

            position = self._positions[symbol]
            if position.side == "FLAT" or position.entry_price is None:
                return Decimal("0")

            # Calculate unrealized P&L
            price_diff = current_price - position.entry_price

            if position.side == "LONG":
                unrealized_pnl = position.size * price_diff
            else:  # SHORT
                unrealized_pnl = position.size * (-price_diff)

            # Update position
            position.unrealized_pnl = unrealized_pnl
            position.timestamp = datetime.utcnow()

            return unrealized_pnl

    def calculate_total_pnl(self) -> tuple[Decimal, Decimal]:
        """
        Calculate total realized and unrealized P&L.

        Returns:
            Tuple of (realized_pnl, unrealized_pnl)
        """
        if self.use_fifo:
            # Get realized P&L from FIFO manager
            realized_total = self.fifo_manager.get_realized_pnl()

            # Calculate unrealized P&L from active positions
            unrealized_total = Decimal("0")
            for symbol in self._positions:
                position = self.get_position(symbol)
                unrealized_total += position.unrealized_pnl

            return realized_total, unrealized_total

        with self._lock:
            realized_total = Decimal("0")
            unrealized_total = Decimal("0")

            # Sum from active positions
            for position in self._positions.values():
                realized_total += position.realized_pnl
                unrealized_total += position.unrealized_pnl

            # Add realized P&L from closed positions
            for position in self._position_history:
                realized_total += position.realized_pnl

            return realized_total, unrealized_total

    def get_position_risk_metrics(self, symbol: str) -> dict[str, float]:
        """
        Get risk metrics for a position.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with risk metrics
        """
        with self._lock:
            position = self.get_position(symbol)

            if position.side == "FLAT":
                return {
                    "position_value": 0.0,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                    "time_in_position_hours": 0.0,
                    "exposure_risk": 0.0,
                }

            # Calculate position value
            if position.entry_price is None:
                position_value = 0.0
            else:
                position_value = float(position.size * position.entry_price)

            # Calculate unrealized P&L percentage
            unrealized_pnl_pct = 0.0
            if position_value > 0:
                unrealized_pnl_pct = (
                    float(position.unrealized_pnl) / position_value * 100
                )

            # Calculate time in position
            time_diff = datetime.utcnow() - position.timestamp
            time_in_position_hours = time_diff.total_seconds() / 3600

            # Calculate exposure risk (position value as % of account)
            account_balance = float(settings.risk.min_account_balance)  # Simplified
            exposure_risk = (
                position_value / account_balance * 100 if account_balance > 0 else 0.0
            )

            return {
                "position_value": position_value,
                "unrealized_pnl": float(position.unrealized_pnl),
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "time_in_position_hours": time_in_position_hours,
                "exposure_risk": exposure_risk,
            }

    def should_close_position(
        self, symbol: str, current_price: Decimal
    ) -> tuple[bool, str]:
        """
        Check if a position should be closed based on risk rules.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Tuple of (should_close, reason)
        """
        with self._lock:
            position = self.get_position(symbol)

            if position.side == "FLAT":
                return False, "No position to close"

            # Update unrealized P&L
            self.update_unrealized_pnl(symbol, current_price)

            # Check maximum hold time
            time_diff = datetime.utcnow() - position.timestamp
            max_hold_hours = settings.risk.max_position_hold_hours

            if time_diff.total_seconds() / 3600 > max_hold_hours:
                return True, f"Maximum hold time exceeded ({max_hold_hours}h)"

            # Check if position is at risk levels that require closing
            risk_metrics = self.get_position_risk_metrics(symbol)

            # Emergency stop loss check
            emergency_stop_pct = settings.risk.emergency_stop_loss_pct
            if risk_metrics["unrealized_pnl_pct"] <= -emergency_stop_pct:
                return True, f"Emergency stop loss triggered ({emergency_stop_pct}%)"

            # Large unrealized loss check
            if risk_metrics["unrealized_pnl_pct"] <= -10.0:  # 10% unrealized loss
                return True, "Large unrealized loss detected"

            return False, "Position within acceptable risk"

    def get_daily_pnl(self, target_date: datetime = None) -> dict[str, float]:
        """
        Get P&L for a specific day.

        Args:
            target_date: Target date (default: today)

        Returns:
            Dictionary with daily P&L metrics
        """
        if target_date is None:
            target_date = datetime.utcnow()

        target_date_start = target_date.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        target_date_end = target_date_start + timedelta(days=1)

        with self._lock:
            daily_realized = Decimal("0")
            daily_unrealized = Decimal("0")
            trades_count = 0

            # Check closed positions from target date
            for position in self._position_history:
                if target_date_start <= position.timestamp < target_date_end:
                    daily_realized += position.realized_pnl
                    trades_count += 1

            # Add unrealized P&L from active positions
            for position in self._positions.values():
                daily_unrealized += position.unrealized_pnl

            return {
                "date": target_date.strftime("%Y-%m-%d"),
                "realized_pnl": float(daily_realized),
                "unrealized_pnl": float(daily_unrealized),
                "total_pnl": float(daily_realized + daily_unrealized),
                "trades_count": trades_count,
            }

    def get_position_summary(self) -> dict[str, Any]:
        """
        Get summary of all positions.

        Returns:
            Dictionary with position summary
        """
        with self._lock:
            active_positions = len(self._positions)
            total_realized, total_unrealized = self.calculate_total_pnl()

            # Calculate total exposure
            total_exposure = Decimal("0")
            for position in self._positions.values():
                if position.entry_price is not None:
                    total_exposure += position.size * position.entry_price

            return {
                "active_positions": active_positions,
                "total_realized_pnl": float(total_realized),
                "total_unrealized_pnl": float(total_unrealized),
                "total_pnl": float(total_realized + total_unrealized),
                "total_exposure": float(total_exposure),
                "closed_positions_today": len(
                    [
                        p
                        for p in self._position_history
                        if p.timestamp.date() == datetime.utcnow().date()
                    ]
                ),
            }

    def _calculate_new_position(
        self, current_pos: Position, order: Order, fill_price: Decimal
    ) -> Position:
        """
        Calculate new position after order fill.

        Args:
            current_pos: Current position
            order: Filled order
            fill_price: Actual fill price

        Returns:
            New position after order
        """
        # Convert order side to position side
        if order.side == "BUY":
            order_side = "LONG"
            order_size = order.filled_quantity
        else:  # SELL
            order_side = "SHORT"
            order_size = -order.filled_quantity

        # Calculate new position
        if current_pos.side == "FLAT":
            # Opening new position
            new_position = Position(
                symbol=order.symbol,
                side=order_side,
                size=abs(order_size),
                entry_price=fill_price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                timestamp=datetime.now(UTC),
            )
        else:
            # Modifying existing position
            current_size = (
                current_pos.size if current_pos.side == "LONG" else -current_pos.size
            )
            new_size = current_size + order_size

            if abs(new_size) < Decimal("0.00001"):  # Position closed
                # Calculate realized P&L
                if current_pos.entry_price is not None:
                    if current_pos.side == "LONG":
                        realized_pnl = current_pos.size * (
                            fill_price - current_pos.entry_price
                        )
                    else:  # SHORT
                        realized_pnl = current_pos.size * (
                            current_pos.entry_price - fill_price
                        )
                else:
                    realized_pnl = Decimal("0")

                new_position = Position(
                    symbol=order.symbol,
                    side="FLAT",
                    size=Decimal("0"),
                    entry_price=None,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=current_pos.realized_pnl + realized_pnl,
                    timestamp=datetime.now(UTC),
                )
            else:
                # Position continues with new size
                new_side = "LONG" if new_size > 0 else "SHORT"

                # Calculate weighted average entry price if adding to position
                if (current_pos.side == "LONG" and order_side == "LONG") or (
                    current_pos.side == "SHORT" and order_side == "SHORT"
                ):
                    # Adding to position
                    total_cost = (current_pos.size * current_pos.entry_price) + (
                        abs(order_size) * fill_price
                    )
                    new_entry_price = total_cost / abs(new_size)
                else:
                    # Reducing position - keep original entry price
                    new_entry_price = current_pos.entry_price

                new_position = Position(
                    symbol=order.symbol,
                    side=new_side,
                    size=abs(new_size),
                    entry_price=new_entry_price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=current_pos.realized_pnl,
                    timestamp=datetime.now(UTC),
                )

        return new_position

    async def _save_state_async(self) -> None:
        """Save current state to files asynchronously."""
        try:
            # Save active positions
            positions_data = {}
            for symbol, position in self._positions.items():
                positions_data[symbol] = {
                    "symbol": position.symbol,
                    "side": position.side,
                    "size": str(position.size),
                    "entry_price": (
                        str(position.entry_price) if position.entry_price else None
                    ),
                    "unrealized_pnl": str(position.unrealized_pnl),
                    "realized_pnl": str(position.realized_pnl),
                    "timestamp": position.timestamp.isoformat(),
                }

            async with aiofiles.open(self.positions_file, "w") as f:
                await f.write(json.dumps(positions_data, indent=2))

            # Save position history (last 100 entries)
            history_data = []
            for position in self._position_history[-100:]:
                history_data.append(
                    {
                        "symbol": position.symbol,
                        "side": position.side,
                        "size": str(position.size),
                        "entry_price": (
                            str(position.entry_price) if position.entry_price else None
                        ),
                        "unrealized_pnl": str(position.unrealized_pnl),
                        "realized_pnl": str(position.realized_pnl),
                        "timestamp": position.timestamp.isoformat(),
                    }
                )

            async with aiofiles.open(self.history_file, "w") as f:
                await f.write(json.dumps(history_data, indent=2))

            logger.debug("Position state saved successfully")

        except Exception as e:
            logger.error(f"Failed to save position state: {e}")

    def _save_state_sync(self) -> None:
        """Save current state to files synchronously (fallback method)."""
        try:
            # Save active positions
            positions_data = {}
            for symbol, position in self._positions.items():
                positions_data[symbol] = {
                    "symbol": position.symbol,
                    "side": position.side,
                    "size": str(position.size),
                    "entry_price": (
                        str(position.entry_price) if position.entry_price else None
                    ),
                    "unrealized_pnl": str(position.unrealized_pnl),
                    "realized_pnl": str(position.realized_pnl),
                    "timestamp": position.timestamp.isoformat(),
                }

            with open(self.positions_file, "w") as f:
                json.dump(positions_data, f, indent=2)

            # Save position history (last 100 entries)
            history_data = []
            for position in self._position_history[-100:]:
                history_data.append(
                    {
                        "symbol": position.symbol,
                        "side": position.side,
                        "size": str(position.size),
                        "entry_price": (
                            str(position.entry_price) if position.entry_price else None
                        ),
                        "unrealized_pnl": str(position.unrealized_pnl),
                        "realized_pnl": str(position.realized_pnl),
                        "timestamp": position.timestamp.isoformat(),
                    }
                )

            with open(self.history_file, "w") as f:
                json.dump(history_data, f, indent=2)

            logger.debug("Position state saved successfully (sync)")

        except Exception as e:
            logger.error(f"Failed to save position state (sync): {e}")

    def _save_state(self) -> None:
        """Save current state to files (non-blocking when called from async context)."""
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
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _handle_save_error(self, task: asyncio.Task) -> None:
        """Handle errors from async save task."""
        try:
            task.result()
        except Exception as e:
            logger.error(f"Async save failed: {e}")

    def _load_state(self) -> None:
        """Load state from files."""
        try:
            # Load active positions
            if self.positions_file.exists():
                with open(self.positions_file) as f:
                    positions_data = json.load(f)

                for symbol, pos_data in positions_data.items():
                    self._positions[symbol] = Position(
                        symbol=pos_data["symbol"],
                        side=pos_data["side"],
                        size=Decimal(pos_data["size"]),
                        entry_price=(
                            Decimal(pos_data["entry_price"])
                            if pos_data["entry_price"]
                            else None
                        ),
                        unrealized_pnl=Decimal(pos_data["unrealized_pnl"]),
                        realized_pnl=Decimal(pos_data["realized_pnl"]),
                        timestamp=datetime.fromisoformat(pos_data["timestamp"]),
                    )

                logger.info(f"Loaded {len(self._positions)} active positions")

            # Load position history
            if self.history_file.exists():
                with open(self.history_file) as f:
                    history_data = json.load(f)

                for pos_data in history_data:
                    self._position_history.append(
                        Position(
                            symbol=pos_data["symbol"],
                            side=pos_data["side"],
                            size=Decimal(pos_data["size"]),
                            entry_price=(
                                Decimal(pos_data["entry_price"])
                                if pos_data["entry_price"]
                                else None
                            ),
                            unrealized_pnl=Decimal(pos_data["unrealized_pnl"]),
                            realized_pnl=Decimal(pos_data["realized_pnl"]),
                            timestamp=datetime.fromisoformat(pos_data["timestamp"]),
                        )
                    )

                logger.info(
                    f"Loaded {len(self._position_history)} historical positions"
                )

        except Exception as e:
            logger.error(f"Failed to load position state: {e}")
            # Continue with empty state

    def get_paper_trading_performance(self, days: int = 7) -> dict[str, Any]:
        """
        Get enhanced paper trading performance metrics.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with comprehensive performance metrics
        """
        if not self.paper_account:
            return {"error": "Paper trading not enabled"}

        # Update daily performance before generating report
        self.paper_account.update_daily_performance()

        # Get performance summary
        performance = self.paper_account.get_performance_summary(days=days)

        # Add account status
        account_status = self.paper_account.get_account_status()
        performance.update(account_status)

        # Add trade history
        performance["recent_trades"] = self.paper_account.get_trade_history(days=days)

        return performance

    def generate_daily_report(self, date: str = None) -> str:
        """
        Generate a comprehensive daily trading report.

        Args:
            date: Date for report (default: today)

        Returns:
            Formatted daily report string
        """
        if not self.paper_account:
            # Fallback to basic report
            return self._generate_basic_daily_report(date)

        return self.paper_account.generate_daily_report(date)

    def _generate_basic_daily_report(self, date: str = None) -> str:
        """Generate basic daily report without paper trading."""
        target_date = date or datetime.utcnow().date().isoformat()
        daily_pnl = self.get_daily_pnl()
        summary = self.get_position_summary()

        report = f"""
🗓️  Daily Trading Report - {target_date}
{'=' * 50}

📊 Position Summary:
   Active Positions: {summary['active_positions']}
   Total Exposure:   ${summary['total_exposure']:,.2f}
   Realized P&L:     ${summary['total_realized_pnl']:,.2f}
   Unrealized P&L:   ${summary['total_unrealized_pnl']:,.2f}
   Total P&L:        ${summary['total_pnl']:,.2f}

🔄 Today's Activity:
   Closed Positions: {summary['closed_positions_today']}
   Daily P&L:        ${daily_pnl['total_pnl']:,.2f}
   Trades Count:     {daily_pnl['trades_count']}
"""
        return report

    def get_weekly_performance_summary(self) -> dict[str, Any]:
        """Get weekly performance summary."""
        if self.paper_account:
            return self.paper_account.get_performance_summary(days=7)

        # Fallback for basic position tracking
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly_positions = [
            pos for pos in self._position_history if pos.timestamp >= week_ago
        ]

        total_realized_pnl = sum(pos.realized_pnl for pos in weekly_positions)
        winning_trades = len([pos for pos in weekly_positions if pos.realized_pnl > 0])
        total_trades = len(weekly_positions)

        return {
            "period_days": 7,
            "total_trades": total_trades,
            "total_realized_pnl": float(total_realized_pnl),
            "win_rate": (
                (winning_trades / total_trades * 100) if total_trades > 0 else 0
            ),
            "active_positions": len(self._positions),
        }

    def get_tax_lots_report(self, symbol: str) -> Optional[dict]:
        """
        Get FIFO tax lots report for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tax lots report or None if not using FIFO
        """
        if not self.use_fifo:
            return None

        return self.fifo_manager.get_tax_lots_report(symbol)

    def export_trade_history(self, days: int = 30, format: str = "json") -> str:
        """
        Export trade history for analysis.

        Args:
            days: Number of days to export
            format: Export format ('json' or 'csv')

        Returns:
            Exported data as string
        """
        if self.paper_account:
            trade_history = self.paper_account.get_trade_history(days=days)
        else:
            # Fallback to basic position history
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            recent_positions = [
                pos for pos in self._position_history if pos.timestamp >= cutoff_date
            ]

            trade_history = [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": float(pos.entry_price) if pos.entry_price else None,
                    "size": float(pos.size),
                    "realized_pnl": float(pos.realized_pnl),
                    "timestamp": pos.timestamp.isoformat(),
                }
                for pos in recent_positions
            ]

        if format == "json":
            return json.dumps(trade_history, indent=2)
        elif format == "csv":
            if not trade_history:
                return "No trade history available"

            import csv
            import io

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=trade_history[0].keys())
            writer.writeheader()
            writer.writerows(trade_history)
            return output.getvalue()

        return "Unsupported format"

    def clear_old_history(self, days_to_keep: int = 30) -> None:
        """
        Clear old position history.

        Args:
            days_to_keep: Number of days to keep in history
        """
        with self._lock:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            original_count = len(self._position_history)
            self._position_history = [
                pos for pos in self._position_history if pos.timestamp >= cutoff_date
            ]

            removed_count = original_count - len(self._position_history)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old position records")
                self._save_state()

    def reset_positions(self) -> None:
        """Reset all positions (for testing/emergency use)."""
        with self._lock:
            self._positions.clear()
            self._position_history.clear()
            if self.paper_account:
                self.paper_account.reset_account()
            self._save_state()
            logger.warning("All positions have been reset")
