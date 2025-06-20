"""
Experience Manager for tracking complete trade lifecycle.

Monitors trades from entry to exit, captures market reactions,
and stores experiences for learning.
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from bot.config import settings
from bot.logging.trade_logger import TradeLogger
from bot.mcp.memory_server import MCPMemoryServer
from bot.trading_types import MarketState, Order, Position, TradeAction

logger = logging.getLogger(__name__)


class ActiveTrade:
    """Tracks an active trade from entry to exit."""

    def __init__(
        self,
        trade_id: str,
        experience_id: str,
        entry_order: Order,
        trade_action: TradeAction,
        market_state_at_entry: MarketState,
    ):
        self.trade_id = trade_id
        self.experience_id = experience_id
        self.entry_order = entry_order
        self.trade_action = trade_action
        self.market_state_at_entry = market_state_at_entry

        # Trade lifecycle tracking
        self.entry_time = datetime.now(UTC)
        self.exit_time: datetime | None = None
        self.exit_order: Order | None = None
        self.exit_price: Decimal | None = None

        # Performance tracking
        self.unrealized_pnl: Decimal = Decimal("0")
        self.realized_pnl: Decimal | None = None
        self.max_favorable_excursion: Decimal = Decimal("0")  # Best unrealized PnL
        self.max_adverse_excursion: Decimal = Decimal("0")  # Worst unrealized PnL

        # Market snapshots
        self.market_snapshots: list[dict[str, Any]] = []
        self.last_snapshot_time = datetime.now(UTC)


class ExperienceManager:
    """
    Manages the complete lifecycle of trading experiences.

    Tracks trades from decision to outcome, captures market data,
    and stores experiences in the MCP memory server for learning.
    """

    def __init__(self, memory_server: MCPMemoryServer):
        """Initialize the experience manager."""
        self.memory_server = memory_server

        # Active trade tracking
        self.active_trades: dict[str, ActiveTrade] = {}

        # Pending experiences awaiting trade execution
        self.pending_experiences: dict[str, str] = {}  # order_id -> experience_id

        # Background task for monitoring trades
        self._monitor_task: asyncio.Task | None = None
        self._running = False

        # Track reflection tasks for proper cleanup
        self._reflection_tasks: list[asyncio.Task] = []

        # Initialize trade logger
        self.trade_logger = TradeLogger()

        logger.info(
            "🎯 Experience Manager: Initialized with enhanced logging and trade tracking"
        )

    async def start(self) -> None:
        """Start the experience manager background monitoring."""
        self._running = True

        # Start trade monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_trades())

        logger.info("✅ Experience Manager: Started background monitoring")

    async def stop(self) -> None:
        """Stop the experience manager."""
        self._running = False

        # Cancel monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        # Cancel reflection tasks
        for task in self._reflection_tasks:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._reflection_tasks.clear()

        # Update any remaining active trades
        for trade in self.active_trades.values():
            if trade.realized_pnl is None:
                logger.warning(
                    "Trade %s still active at shutdown, marking as incomplete",
                    trade.trade_id,
                )

        logger.info("🛑 Experience Manager: Stopped")

    async def record_trading_decision(
        self, market_state: MarketState, trade_action: TradeAction
    ) -> str:
        """
        Record a trading decision before execution.

        Args:
            market_state: Current market state
            trade_action: Trading decision made

        Returns:
            Experience ID for tracking
        """
        # Store the experience in memory
        experience_id = await self.memory_server.store_experience(
            market_state,
            trade_action,
            _additional_context={
                "decision_time": datetime.now(UTC).isoformat(),
                "position_before": {
                    "side": market_state.current_position.side,
                    "size": float(market_state.current_position.size),
                    "entry_price": (
                        float(market_state.current_position.entry_price)
                        if market_state.current_position.entry_price
                        else None
                    ),
                },
            },
        )

        logger.info(
            f"📝 Experience Manager: Recorded {trade_action.action} decision | "
            f"Experience ID: {experience_id[:8]}... | "
            f"Price: ${market_state.current_price} | "
            f"Position: {market_state.current_position.side}"
        )

        # Log structured trade decision
        self.trade_logger.log_trade_decision(
            market_state=market_state,
            trade_action=trade_action,
            experience_id=experience_id,
            memory_context=None,  # Will be populated by memory-enhanced agent
        )

        return experience_id

    def link_order_to_experience(self, order_id: str, experience_id: str) -> None:
        """
        Link an order to its corresponding experience.

        Args:
            order_id: Order ID from exchange
            experience_id: Experience ID from memory
        """
        self.pending_experiences[order_id] = experience_id
        logger.info(
            "🔗 Experience Manager: Linked order %s to experience %s...",
            order_id,
            experience_id[:8],
        )

    def start_tracking_trade(
        self, order: Order, trade_action: TradeAction, market_state: MarketState
    ) -> str | None:
        """
        Start tracking an executed trade.

        Args:
            order: Executed order
            trade_action: Original trade action
            market_state: Market state at execution

        Returns:
            Trade ID if tracking started
        """
        # Check if we have an experience for this order
        experience_id = self.pending_experiences.get(order.id)
        if not experience_id:
            logger.warning(
                "⚠️ Experience Manager: No experience found for order %s", order.id
            )
            return None

        # Create active trade tracking
        trade_id = f"trade_{uuid4().hex[:8]}"
        active_trade = ActiveTrade(
            trade_id=trade_id,
            experience_id=experience_id,
            entry_order=order,
            trade_action=trade_action,
            market_state_at_entry=market_state,
        )

        self.active_trades[trade_id] = active_trade

        # Remove from pending
        del self.pending_experiences[order.id]

        logger.info(
            f"🚀 Experience Manager: Started tracking {trade_action.action} trade | "
            f"Trade ID: {trade_id} | Price: ${order.price} | "
            f"Size: {order.quantity} | Experience: {experience_id[:8]}..."
        )

        # Log detailed trade entry
        logger.debug(
            f"Trade entry details: Symbol={order.symbol}, Side={order.side}, "
            f"Size={order.quantity}, Price=${order.price}, "
            f"Order Type={order.type}, Status={order.status}"
        )

        return trade_id

    async def update_trade_progress(
        self,
        position: Position,
        current_price: Decimal,
        market_state: MarketState | None = None,
    ) -> None:
        """
        Update progress of active trades.

        Args:
            position: Current position
            current_price: Current market price
            market_state: Current market state (for snapshots)
        """
        # Find active trade matching position
        active_trade = None
        for trade in self.active_trades.values():
            if (
                trade.entry_order.symbol == position.symbol
                and trade.entry_order.side == position.side
                and not trade.exit_time
            ):
                active_trade = trade
                break

        if not active_trade:
            return

        # Update unrealized PnL
        entry_price = active_trade.entry_order.price
        size = active_trade.entry_order.quantity

        if entry_price is not None and size is not None:
            if position.side == "LONG":
                unrealized_pnl = (current_price - entry_price) * size
            else:  # SHORT
                unrealized_pnl = (entry_price - current_price) * size
        else:
            unrealized_pnl = Decimal("0")

        active_trade.unrealized_pnl = unrealized_pnl

        # Track max favorable/adverse excursions
        if unrealized_pnl > active_trade.max_favorable_excursion:
            active_trade.max_favorable_excursion = unrealized_pnl
        if unrealized_pnl < active_trade.max_adverse_excursion:
            active_trade.max_adverse_excursion = unrealized_pnl

        # Capture market snapshot if enough time has passed
        if (
            market_state
            and (datetime.now(UTC) - active_trade.last_snapshot_time).seconds > 60
        ):
            snapshot = {
                "timestamp": datetime.now(UTC).isoformat(),
                "price": float(current_price),
                "unrealized_pnl": float(unrealized_pnl),
                "indicators": self._extract_indicators_snapshot(market_state),
            }
            active_trade.market_snapshots.append(snapshot)
            active_trade.last_snapshot_time = datetime.now(UTC)

            # Log position update
            self.trade_logger.log_position_update(
                trade_id=active_trade.trade_id,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                max_favorable=active_trade.max_favorable_excursion,
                max_adverse=active_trade.max_adverse_excursion,
            )

    async def complete_trade(
        self,
        exit_order: Order,
        exit_price: Decimal,
        market_state_at_exit: MarketState | None = None,
    ) -> bool:
        """
        Complete a trade and update its experience with outcome.

        Args:
            exit_order: Exit order that closed the position
            exit_price: Actual exit price
            market_state_at_exit: Market state at exit

        Returns:
            True if trade was completed successfully
        """
        # Find the active trade
        active_trade = None
        for trade in self.active_trades.values():
            if trade.entry_order.symbol == exit_order.symbol and not trade.exit_time:
                active_trade = trade
                break

        if not active_trade:
            logger.warning("No active trade found for exit order %s", exit_order.id)
            return False

        # Mark trade as complete
        active_trade.exit_time = datetime.now(UTC)
        active_trade.exit_order = exit_order
        active_trade.exit_price = exit_price

        # Calculate final PnL
        entry_price = active_trade.entry_order.price
        size = active_trade.entry_order.quantity

        if entry_price is not None and size is not None:
            if active_trade.entry_order.side == "BUY":  # Was LONG
                realized_pnl = (exit_price - entry_price) * size
            else:  # Was SHORT
                realized_pnl = (entry_price - exit_price) * size
        else:
            realized_pnl = Decimal("0")

        active_trade.realized_pnl = realized_pnl

        # Calculate trade duration
        if active_trade.exit_time is not None:
            duration_minutes = (
                active_trade.exit_time - active_trade.entry_time
            ).total_seconds() / 60
        else:
            duration_minutes = 0.0

        # Update the experience with outcome
        await self.memory_server.update_experience_outcome(
            experience_id=active_trade.experience_id,
            pnl=realized_pnl,
            exit_price=exit_price,
            duration_minutes=duration_minutes,
            market_data_at_exit=market_state_at_exit,
        )

        # Calculate percentage return safely
        pnl_percentage = 0.0
        if entry_price is not None and entry_price > 0:
            pnl_percentage = float(realized_pnl) / float(entry_price) * 100

        logger.info(
            f"🏁 Experience Manager: Trade completed | ID: {active_trade.trade_id} | "
            f"PnL: ${realized_pnl:.2f} ({'+' if realized_pnl > 0 else ''}{pnl_percentage:.2f}%) | "
            f"Duration: {duration_minutes:.1f}min | {'✅ WIN' if realized_pnl > 0 else '❌ LOSS'}"
        )

        # Log structured trade outcome
        if entry_price is not None:
            self.trade_logger.log_trade_outcome(
                experience_id=active_trade.experience_id,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=realized_pnl,
                duration_minutes=duration_minutes,
                insights=None,  # Will be populated after reflection
            )

        # Archive the trade
        del self.active_trades[active_trade.trade_id]

        # Schedule reflection analysis after delay
        if settings.mcp.track_trade_lifecycle:
            reflection_task = asyncio.create_task(
                self._schedule_trade_reflection(
                    active_trade, delay_minutes=settings.mcp.reflection_delay_minutes
                )
            )
            self._reflection_tasks.append(reflection_task)

            # Clean up completed tasks periodically
            self._reflection_tasks = [
                task for task in self._reflection_tasks if not task.done()
            ]

        return True

    async def _monitor_trades(self) -> None:
        """Background task to monitor active trades."""
        while self._running:
            try:
                # Check for stale trades
                current_time = datetime.now(UTC)
                stale_trades = []

                for trade_id, trade in self.active_trades.items():
                    # Check if trade has been open too long
                    trade_duration = (
                        current_time - trade.entry_time
                    ).total_seconds() / 3600

                    if trade_duration > 24:  # 24 hours
                        logger.warning(
                            f"Trade {trade_id} has been open for {trade_duration:.1f} hours"
                        )
                        stale_trades.append(trade_id)

                # Clean up very old trades (>48 hours)
                for trade_id in stale_trades:
                    trade = self.active_trades[trade_id]
                    if (current_time - trade.entry_time).total_seconds() / 3600 > 48:
                        logger.error(
                            "Removing stale trade %s - open for over 48 hours", trade_id
                        )
                        del self.active_trades[trade_id]

                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.exception("Error in trade monitoring: %s", e)
                await asyncio.sleep(60)

    async def _schedule_trade_reflection(
        self, completed_trade: ActiveTrade, delay_minutes: int
    ) -> None:
        """
        Schedule post-trade reflection and analysis.

        Args:
            completed_trade: Completed trade to analyze
            delay_minutes: Minutes to wait before analysis
        """
        try:
            # Wait for market to develop after trade
            await asyncio.sleep(delay_minutes * 60)

            # Analyze what happened after the trade
            reflection = self._generate_trade_reflection(completed_trade)

            if reflection:
                logger.info(
                    "Trade reflection for %s: %s", completed_trade.trade_id, reflection
                )

                # Could store this reflection or use it for pattern analysis
                # This is where we'd integrate with the self-improvement engine

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Error in trade reflection: %s", e)

    def _generate_trade_reflection(self, trade: ActiveTrade) -> str:
        """Generate reflection insights from completed trade."""
        insights = []

        # Analyze execution quality
        if trade.realized_pnl and trade.max_favorable_excursion:
            capture_ratio = float(trade.realized_pnl / trade.max_favorable_excursion)

            if capture_ratio < 0.5 and trade.realized_pnl > 0:
                insights.append(
                    f"Only captured {capture_ratio:.1%} of max profit - "
                    "consider trailing stops or partial exits"
                )
            elif capture_ratio > 0.9:
                insights.append("Excellent profit capture - near optimal exit")

        # Analyze drawdown tolerance
        if trade.max_adverse_excursion < -100:  # Significant drawdown
            if trade.realized_pnl and trade.realized_pnl > 0:
                insights.append(
                    f"Endured ${abs(trade.max_adverse_excursion):.2f} drawdown before profit - "
                    "high risk tolerance paid off"
                )
            else:
                insights.append(
                    f"Large drawdown of ${abs(trade.max_adverse_excursion):.2f} - "
                    "consider tighter stops"
                )

        # Analyze hold duration
        if trade.exit_time is not None:
            duration_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            if duration_hours < 1 and trade.realized_pnl and trade.realized_pnl > 0:
                insights.append("Quick profitable scalp - good timing")
            elif duration_hours > 12:
                insights.append("Extended hold period - consider position management")

        return "; ".join(insights) if insights else "Standard trade execution"

    def _extract_indicators_snapshot(
        self, market_state: MarketState
    ) -> dict[str, float]:
        """Extract key indicators for snapshot."""
        snapshot = {}

        if market_state.indicators:
            ind = market_state.indicators
            snapshot.update(
                {
                    "rsi": float(ind.rsi) if ind.rsi else 50.0,
                    "cipher_b_wave": (
                        float(ind.cipher_b_wave) if ind.cipher_b_wave else 0.0
                    ),
                    "ema_trend": (
                        1
                        if (
                            ind.ema_fast
                            and ind.ema_slow
                            and ind.ema_fast > ind.ema_slow
                        )
                        else -1
                    ),
                }
            )

        if market_state.dominance_data:
            snapshot["stablecoin_dominance"] = float(
                market_state.dominance_data.stablecoin_dominance
            )

        return snapshot

    def get_active_trades_summary(self) -> dict[str, Any]:
        """Get summary of all active trades."""
        summary: dict[str, Any] = {
            "active_count": len(self.active_trades),
            "total_unrealized_pnl": Decimal("0"),
            "trades": [],
        }

        for trade in self.active_trades.values():
            summary["total_unrealized_pnl"] = (
                summary["total_unrealized_pnl"] + trade.unrealized_pnl
            )

            trade_info = {
                "trade_id": trade.trade_id,
                "symbol": trade.entry_order.symbol,
                "side": trade.entry_order.side,
                "entry_price": (
                    float(trade.entry_order.price) if trade.entry_order.price else 0.0
                ),
                "size": (
                    float(trade.entry_order.quantity)
                    if trade.entry_order.quantity
                    else 0.0
                ),
                "unrealized_pnl": float(trade.unrealized_pnl),
                "duration_hours": (datetime.now(UTC) - trade.entry_time).total_seconds()
                / 3600,
                "max_profit": float(trade.max_favorable_excursion),
                "max_loss": float(trade.max_adverse_excursion),
            }

            summary["trades"].append(trade_info)

        summary["total_unrealized_pnl"] = float(summary["total_unrealized_pnl"])

        return summary
