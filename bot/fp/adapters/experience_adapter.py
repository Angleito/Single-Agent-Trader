"""
Functional programming adapter for experience management.

This adapter provides pure functional interfaces for trade lifecycle tracking,
outcome recording, and experience analysis while maintaining compatibility
with the existing imperative experience manager.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from bot.fp.types import (
    ExperienceId,
    Failure,
    MarketSnapshot,
    Maybe,
    Nothing,
    PatternTag,
    Result,
    Some,
    Success,
    TradingOutcome,
)
from bot.learning.experience_manager import ExperienceManager
from bot.trading_types import MarketState, Order, TradeAction

logger = logging.getLogger(__name__)


class ActiveTradeFP:
    """Immutable active trade representation for FP."""

    def __init__(
        self,
        trade_id: str,
        experience_id: ExperienceId,
        entry_snapshot: MarketSnapshot,
        trade_decision: Any,  # Would be proper FP trade decision type
        entry_time: datetime,
    ):
        self.trade_id = trade_id
        self.experience_id = experience_id
        self.entry_snapshot = entry_snapshot
        self.trade_decision = trade_decision
        self.entry_time = entry_time
        self.exit_time: datetime | None = None
        self.current_pnl = Decimal(0)
        self.max_profit = Decimal(0)
        self.max_loss = Decimal(0)
        self.market_snapshots: list[dict[str, Any]] = []

    def with_update(
        self,
        current_price: Decimal,
        unrealized_pnl: Decimal,
        market_snapshot: dict[str, Any] | None = None,
    ) -> "ActiveTradeFP":
        """Create updated trade with new market data."""
        new_trade = ActiveTradeFP(
            self.trade_id,
            self.experience_id,
            self.entry_snapshot,
            self.trade_decision,
            self.entry_time,
        )
        new_trade.exit_time = self.exit_time
        new_trade.current_pnl = unrealized_pnl
        new_trade.max_profit = max(self.max_profit, unrealized_pnl)
        new_trade.max_loss = min(self.max_loss, unrealized_pnl)
        new_trade.market_snapshots = list(self.market_snapshots)

        if market_snapshot:
            new_trade.market_snapshots.append(market_snapshot)

        return new_trade

    def with_exit(self, exit_time: datetime, final_pnl: Decimal) -> "ActiveTradeFP":
        """Create completed trade with exit data."""
        new_trade = self.with_update(Decimal(0), final_pnl)
        new_trade.exit_time = exit_time
        return new_trade

    def is_completed(self) -> bool:
        """Check if trade is completed."""
        return self.exit_time is not None

    def duration_minutes(self) -> float | None:
        """Get trade duration in minutes."""
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return None


class ExperienceManagerFP:
    """
    Functional programming adapter for experience management.

    Provides pure functional interfaces for trade lifecycle tracking
    while bridging to the imperative experience manager.
    """

    def __init__(self, experience_manager: ExperienceManager):
        """Initialize with imperative experience manager."""
        self._experience_manager = experience_manager
        self._active_trades: dict[str, ActiveTradeFP] = {}

    async def record_trading_decision_fp(
        self,
        market_snapshot: MarketSnapshot,
        trade_decision: Any,  # Would be proper FP trade decision type
        rationale: str,
        pattern_tags: list[PatternTag] | None = None,
    ) -> Result[ExperienceId, str]:
        """
        Record a trading decision using FP types.

        Args:
            market_snapshot: Current market snapshot
            trade_decision: FP trade decision
            rationale: Decision rationale
            pattern_tags: Optional pattern tags

        Returns:
            Result containing experience ID or error
        """
        try:
            # Convert FP types to imperative format for compatibility
            market_state = self._fp_to_market_state(market_snapshot)
            trade_action = self._fp_to_trade_action(trade_decision, rationale)

            # Record using imperative manager
            experience_id_str = await self._experience_manager.record_trading_decision(
                market_state, trade_action
            )

            experience_id_result = ExperienceId.create(experience_id_str)
            if experience_id_result.is_failure():
                return Failure(experience_id_result.failure())

            logger.info(
                "📝 FP Experience Manager: Recorded decision %s | Action: %s | Patterns: %s",
                experience_id_result.success().short(),
                trade_decision,
                [p.name for p in (pattern_tags or [])],
            )

            return Success(experience_id_result.success())

        except Exception as e:
            error_msg = f"Failed to record FP trading decision: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    def link_order_to_experience_fp(
        self,
        order_id: str,
        experience_id: ExperienceId,
    ) -> Result[None, str]:
        """
        Link an order to its experience using FP types.

        Args:
            order_id: Exchange order ID
            experience_id: Experience ID

        Returns:
            Result indicating success or failure
        """
        try:
            self._experience_manager.link_order_to_experience(
                order_id, experience_id.value
            )

            logger.info(
                "🔗 FP Experience Manager: Linked order %s to experience %s",
                order_id,
                experience_id.short(),
            )

            return Success(None)

        except Exception as e:
            error_msg = f"Failed to link FP order to experience: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    def start_tracking_trade_fp(
        self,
        order_data: dict[str, Any],  # Would be proper FP order type
        trade_decision: Any,
        market_snapshot: MarketSnapshot,
    ) -> Result[str, str]:
        """
        Start tracking a trade using FP types.

        Args:
            order_data: Order execution data
            trade_decision: Original trade decision
            market_snapshot: Market snapshot at execution

        Returns:
            Result containing trade ID or error
        """
        try:
            # Convert to imperative types for compatibility
            order = self._create_order_from_data(order_data)
            trade_action = self._fp_to_trade_action(trade_decision, "FP trade")
            market_state = self._fp_to_market_state(market_snapshot)

            # Start tracking using imperative manager
            trade_id = self._experience_manager.start_tracking_trade(
                order, trade_action, market_state
            )

            if not trade_id:
                return Failure("Failed to start trade tracking")

            # Create FP active trade
            experience_id_result = ExperienceId.create(trade_id)
            if experience_id_result.is_failure():
                return Failure(experience_id_result.failure())

            fp_trade = ActiveTradeFP(
                trade_id=trade_id,
                experience_id=experience_id_result.success(),
                entry_snapshot=market_snapshot,
                trade_decision=trade_decision,
                entry_time=datetime.utcnow(),
            )

            self._active_trades[trade_id] = fp_trade

            logger.info(
                "🏁 FP Experience Manager: Started tracking trade %s | Price: $%s",
                trade_id,
                market_snapshot.price,
            )

            return Success(trade_id)

        except Exception as e:
            error_msg = f"Failed to start FP trade tracking: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    async def update_trade_progress_fp(
        self,
        trade_id: str,
        current_price: Decimal,
        market_snapshot: MarketSnapshot | None = None,
    ) -> Result[ActiveTradeFP, str]:
        """
        Update trade progress using FP types.

        Args:
            trade_id: Trade ID to update
            current_price: Current market price
            market_snapshot: Optional current market snapshot

        Returns:
            Result containing updated trade or error
        """
        try:
            if trade_id not in self._active_trades:
                return Failure(f"Trade {trade_id} not found")

            current_trade = self._active_trades[trade_id]

            # Calculate unrealized PnL (simplified)
            entry_price = current_trade.entry_snapshot.price
            unrealized_pnl = (current_price - entry_price) * Decimal(
                1
            )  # Simplified size

            # Create market snapshot dict if provided
            snapshot_dict = None
            if market_snapshot:
                snapshot_dict = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "price": float(current_price),
                    "unrealized_pnl": float(unrealized_pnl),
                    "indicators": market_snapshot.indicators,
                }

            # Update FP trade
            updated_trade = current_trade.with_update(
                current_price, unrealized_pnl, snapshot_dict
            )

            self._active_trades[trade_id] = updated_trade

            # Also update imperative manager for compatibility
            if market_snapshot:
                position = self._create_position_from_snapshot(market_snapshot)
                market_state = self._fp_to_market_state(market_snapshot)
                await self._experience_manager.update_trade_progress(
                    position, current_price, market_state
                )

            logger.debug(
                "📈 FP Experience Manager: Updated trade %s | PnL: $%.2f | Max Profit: $%.2f",
                trade_id,
                unrealized_pnl,
                updated_trade.max_profit,
            )

            return Success(updated_trade)

        except Exception as e:
            error_msg = f"Failed to update FP trade progress: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    async def complete_trade_fp(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_snapshot: MarketSnapshot | None = None,
    ) -> Result[TradingOutcome, str]:
        """
        Complete a trade using FP types.

        Args:
            trade_id: Trade ID to complete
            exit_price: Final exit price
            exit_snapshot: Optional market snapshot at exit

        Returns:
            Result containing trading outcome or error
        """
        try:
            if trade_id not in self._active_trades:
                return Failure(f"Trade {trade_id} not found")

            current_trade = self._active_trades[trade_id]

            # Calculate final outcome
            entry_price = current_trade.entry_snapshot.price
            final_pnl = (exit_price - entry_price) * Decimal(1)  # Simplified size
            duration = (
                datetime.utcnow() - current_trade.entry_time
            ).total_seconds() / 60

            # Create trading outcome
            outcome_result = TradingOutcome.create(
                pnl=final_pnl,
                exit_price=exit_price,
                entry_price=entry_price,
                duration_minutes=duration,
            )

            if outcome_result.is_failure():
                return Failure(outcome_result.failure())

            outcome = outcome_result.success()

            # Complete FP trade
            completed_trade = current_trade.with_exit(datetime.utcnow(), final_pnl)
            self._active_trades[trade_id] = completed_trade

            # Complete imperative trade for compatibility
            exit_order = self._create_exit_order(exit_price)
            market_state_at_exit = None
            if exit_snapshot:
                market_state_at_exit = self._fp_to_market_state(exit_snapshot)

            await self._experience_manager.complete_trade(
                exit_order, exit_price, market_state_at_exit
            )

            # Remove from active trades
            del self._active_trades[trade_id]

            logger.info(
                "🏁 FP Experience Manager: Completed trade %s | PnL: $%.2f | Duration: %.1fmin | %s",
                trade_id,
                final_pnl,
                duration,
                "✅ WIN" if outcome.is_successful else "❌ LOSS",
            )

            return Success(outcome)

        except Exception as e:
            error_msg = f"Failed to complete FP trade: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    def get_active_trades_fp(self) -> list[ActiveTradeFP]:
        """Get all active FP trades."""
        return list(self._active_trades.values())

    def get_active_trade_fp(self, trade_id: str) -> Maybe[ActiveTradeFP]:
        """Get specific active FP trade."""
        trade = self._active_trades.get(trade_id)
        return Some(trade) if trade else Nothing()

    def analyze_active_trades_fp(self) -> Result[dict[str, Any], str]:
        """
        Analyze current active trades and generate insights.

        Returns:
            Result containing analysis data or error
        """
        try:
            if not self._active_trades:
                return Success(
                    {
                        "total_trades": 0,
                        "total_unrealized_pnl": 0,
                        "insights": [],
                    }
                )

            total_unrealized_pnl = sum(
                trade.current_pnl for trade in self._active_trades.values()
            )

            insights = []

            # Analyze performance distribution
            winning_trades = sum(
                1 for trade in self._active_trades.values() if trade.current_pnl > 0
            )

            total_trades = len(self._active_trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            if win_rate > 0.7:
                insights.append("Strong performance - most active trades profitable")
            elif win_rate < 0.3:
                insights.append("Weak performance - consider risk management review")

            # Analyze duration patterns
            long_running_trades = [
                trade
                for trade in self._active_trades.values()
                if (datetime.utcnow() - trade.entry_time).total_seconds() / 3600 > 4
            ]

            if long_running_trades:
                insights.append(f"{len(long_running_trades)} trades open for >4 hours")

            analysis = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "total_unrealized_pnl": float(total_unrealized_pnl),
                "long_running_count": len(long_running_trades),
                "insights": insights,
            }

            logger.info(
                "📊 FP Experience Manager: Analyzed %d active trades | Win rate: %.1f%% | PnL: $%.2f",
                total_trades,
                win_rate * 100,
                total_unrealized_pnl,
            )

            return Success(analysis)

        except Exception as e:
            error_msg = f"Failed to analyze FP active trades: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    async def cleanup_stale_trades_fp(
        self,
        max_age_hours: float = 48.0,
    ) -> Result[int, str]:
        """
        Clean up stale trades that have been open too long.

        Args:
            max_age_hours: Maximum age in hours before considering stale

        Returns:
            Result containing number of cleaned up trades or error
        """
        try:
            current_time = datetime.utcnow()
            stale_trades = []

            for trade_id, trade in self._active_trades.items():
                age_hours = (current_time - trade.entry_time).total_seconds() / 3600
                if age_hours > max_age_hours:
                    stale_trades.append(trade_id)

            # Remove stale trades
            for trade_id in stale_trades:
                del self._active_trades[trade_id]
                logger.warning(
                    "🧼 FP Experience Manager: Cleaned up stale trade %s",
                    trade_id,
                )

            logger.info(
                "🧼 FP Experience Manager: Cleaned up %d stale trades",
                len(stale_trades),
            )

            return Success(len(stale_trades))

        except Exception as e:
            error_msg = f"Failed to cleanup FP stale trades: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    # Helper methods for conversion between FP and imperative types

    def _fp_to_market_state(self, snapshot: MarketSnapshot) -> MarketState:
        """Convert FP market snapshot to imperative MarketState."""
        from bot.trading_types import IndicatorData, Position

        indicators = IndicatorData(
            timestamp=snapshot.timestamp,
            rsi=snapshot.indicators.get("rsi"),
            cipher_a_dot=snapshot.indicators.get("cipher_a_dot"),
            cipher_b_wave=snapshot.indicators.get("cipher_b_wave"),
            cipher_b_money_flow=snapshot.indicators.get("cipher_b_money_flow"),
            ema_fast=snapshot.indicators.get("ema_fast"),
            ema_slow=snapshot.indicators.get("ema_slow"),
        )

        position = Position(
            symbol=snapshot.symbol.value,
            side=snapshot.position_side,
            size=snapshot.position_size,
            timestamp=snapshot.timestamp,
        )

        return MarketState(
            symbol=snapshot.symbol.value,
            interval="1m",
            timestamp=snapshot.timestamp,
            current_price=snapshot.price,
            ohlcv_data=[],
            indicators=indicators,
            current_position=position,
            dominance_data=None,
        )

    def _fp_to_trade_action(self, decision: Any, rationale: str) -> TradeAction:
        """Convert FP trade decision to imperative TradeAction."""
        # Simplified conversion - would need proper type matching
        return TradeAction(
            action="HOLD",  # Default
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            leverage=1,
            rationale=rationale,
        )

    def _create_order_from_data(self, order_data: dict[str, Any]) -> Order:
        """Create Order object from FP order data."""
        from bot.trading_types import Order

        return Order(
            id=order_data.get("id", str(uuid4())),
            symbol=order_data.get("symbol", "BTC-USD"),
            side=order_data.get("side", "BUY"),
            type=order_data.get("type", "MARKET"),
            quantity=Decimal(str(order_data.get("quantity", 1))),
            price=Decimal(str(order_data.get("price", 50000))),
            status=order_data.get("status", "FILLED"),
            timestamp=datetime.utcnow(),
        )

    def _create_position_from_snapshot(self, snapshot: MarketSnapshot):
        """Create Position from market snapshot."""
        from bot.trading_types import Position

        return Position(
            symbol=snapshot.symbol.value,
            side=snapshot.position_side,
            size=snapshot.position_size,
            timestamp=snapshot.timestamp,
        )

    def _create_exit_order(self, exit_price: Decimal) -> Order:
        """Create exit order for trade completion."""
        from bot.trading_types import Order

        return Order(
            id=str(uuid4()),
            symbol="BTC-USD",  # Simplified
            side="SELL",
            type="MARKET",
            quantity=Decimal(1),
            price=exit_price,
            status="FILLED",
            timestamp=datetime.utcnow(),
        )


__all__ = ["ActiveTradeFP", "ExperienceManagerFP"]
