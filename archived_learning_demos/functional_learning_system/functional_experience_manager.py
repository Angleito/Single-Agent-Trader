"""
Functional Experience Manager with immutable state management.

This module provides a functional alternative to the imperative experience manager,
using pure functions and immutable state while maintaining compatibility with
the existing MCP memory server.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from bot.fp.effects.io import AsyncIO
from bot.fp.types.result import Result, Ok, Err
from bot.logging.trade_logger import TradeLogger
from bot.mcp.memory_server import MCPMemoryServer
from bot.trading_types import MarketState, Order, Position, TradeAction

from .experience import (
    TradeExperience, 
    ExperienceState, 
    LearningContext,
    create_experience,
    update_experience_outcome,
)
from .memory_effects import MemoryEffectInterpreter
from .learning_algorithms import (
    analyze_trading_patterns,
    generate_strategy_insights,
    identify_market_regimes,
)
from .combinators import (
    sequence_async_operations,
    validate_minimum_experiences,
    build_analysis_pipeline,
    tap,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveTradeState:
    """Immutable state for tracking active trades."""
    
    trade_id: str
    experience_id: str
    entry_order: Order
    trade_action: TradeAction
    market_state_at_entry: MarketState
    
    # Trade lifecycle tracking (immutable)
    entry_time: datetime
    exit_time: Optional[datetime]
    exit_order: Optional[Order]
    exit_price: Optional[Decimal]
    
    # Performance tracking (immutable)
    unrealized_pnl: Decimal
    realized_pnl: Optional[Decimal]
    max_favorable_excursion: Decimal
    max_adverse_excursion: Decimal
    
    # Market snapshots (immutable tuple)
    market_snapshots: Tuple[Dict[str, Any], ...]
    last_snapshot_time: datetime
    
    def with_unrealized_pnl(self, pnl: Decimal) -> ActiveTradeState:
        """Return new state with updated unrealized PnL."""
        new_max_favorable = max(self.max_favorable_excursion, pnl)
        new_max_adverse = min(self.max_adverse_excursion, pnl)
        
        return replace(
            self,
            unrealized_pnl=pnl,
            max_favorable_excursion=new_max_favorable,
            max_adverse_excursion=new_max_adverse,
        )
    
    def with_market_snapshot(self, snapshot: Dict[str, Any]) -> ActiveTradeState:
        """Return new state with added market snapshot."""
        new_snapshots = self.market_snapshots + (snapshot,)
        
        return replace(
            self,
            market_snapshots=new_snapshots,
            last_snapshot_time=datetime.now(UTC),
        )
    
    def with_completion(
        self,
        exit_order: Order,
        exit_price: Decimal,
        realized_pnl: Decimal,
    ) -> ActiveTradeState:
        """Return new state with trade completion data."""
        return replace(
            self,
            exit_time=datetime.now(UTC),
            exit_order=exit_order,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
        )
    
    def is_completed(self) -> bool:
        """Check if trade is completed."""
        return self.exit_time is not None


@dataclass(frozen=True)
class FunctionalExperienceManagerState:
    """Immutable state for the functional experience manager."""
    
    experience_state: ExperienceState
    active_trades: Dict[str, ActiveTradeState]
    pending_experiences: Dict[str, str]  # order_id -> experience_id
    learning_context: LearningContext
    last_updated: datetime
    
    @classmethod
    def empty(cls, config: Dict[str, Any]) -> FunctionalExperienceManagerState:
        """Create empty manager state."""
        return cls(
            experience_state=ExperienceState.empty(),
            active_trades={},
            pending_experiences={},
            learning_context=LearningContext.create(config),
            last_updated=datetime.now(UTC),
        )
    
    def add_experience(self, experience: TradeExperience) -> FunctionalExperienceManagerState:
        """Return new state with added experience."""
        new_experience_state = self.experience_state.add_experience(experience)
        new_learning_context = self.learning_context.with_state(new_experience_state)
        
        return replace(
            self,
            experience_state=new_experience_state,
            learning_context=new_learning_context,
            last_updated=datetime.now(UTC),
        )
    
    def add_active_trade(self, active_trade: ActiveTradeState) -> FunctionalExperienceManagerState:
        """Return new state with added active trade."""
        new_active_trades = dict(self.active_trades)
        new_active_trades[active_trade.trade_id] = active_trade
        
        return replace(
            self,
            active_trades=new_active_trades,
            last_updated=datetime.now(UTC),
        )
    
    def update_active_trade(
        self,
        trade_id: str,
        update_fn: callable,
    ) -> FunctionalExperienceManagerState:
        """Return new state with updated active trade."""
        if trade_id not in self.active_trades:
            return self
        
        new_active_trades = dict(self.active_trades)
        new_active_trades[trade_id] = update_fn(self.active_trades[trade_id])
        
        return replace(
            self,
            active_trades=new_active_trades,
            last_updated=datetime.now(UTC),
        )
    
    def remove_active_trade(self, trade_id: str) -> FunctionalExperienceManagerState:
        """Return new state with removed active trade."""
        if trade_id not in self.active_trades:
            return self
        
        new_active_trades = dict(self.active_trades)
        del new_active_trades[trade_id]
        
        return replace(
            self,
            active_trades=new_active_trades,
            last_updated=datetime.now(UTC),
        )
    
    def link_order_to_experience(
        self, 
        order_id: str, 
        experience_id: str
    ) -> FunctionalExperienceManagerState:
        """Return new state with order-experience link."""
        new_pending = dict(self.pending_experiences)
        new_pending[order_id] = experience_id
        
        return replace(
            self,
            pending_experiences=new_pending,
            last_updated=datetime.now(UTC),
        )
    
    def remove_pending_experience(self, order_id: str) -> FunctionalExperienceManagerState:
        """Return new state with removed pending experience."""
        if order_id not in self.pending_experiences:
            return self
        
        new_pending = dict(self.pending_experiences)
        del new_pending[order_id]
        
        return replace(
            self,
            pending_experiences=new_pending,
            last_updated=datetime.now(UTC),
        )


class FunctionalExperienceManager:
    """
    Functional experience manager using immutable state and pure functions.
    
    This manager provides a functional interface while maintaining compatibility
    with the existing MCP memory server and trade logging infrastructure.
    """
    
    def __init__(self, memory_server: MCPMemoryServer, config: Dict[str, Any] = None):
        """Initialize the functional experience manager."""
        self.memory_interpreter = MemoryEffectInterpreter(memory_server)
        self.trade_logger = TradeLogger()
        
        # Immutable state container
        self._state = FunctionalExperienceManagerState.empty(config or {})
        
        # Background task management
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("🎯 Functional Experience Manager: Initialized with immutable state")
    
    @property
    def state(self) -> FunctionalExperienceManagerState:
        """Get current immutable state."""
        return self._state
    
    def _update_state(self, new_state: FunctionalExperienceManagerState) -> None:
        """Internal method to update state (only place where mutation happens)."""
        self._state = new_state
    
    async def start(self) -> None:
        """Start the functional experience manager."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_trades_functional())
        logger.info("✅ Functional Experience Manager: Started with functional monitoring")
    
    async def stop(self) -> None:
        """Stop the functional experience manager."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Functional Experience Manager: Stopped")
    
    async def record_trading_decision(
        self, 
        market_state: MarketState, 
        trade_action: TradeAction
    ) -> Result[str, str]:
        """Record a trading decision using functional approach."""
        # Create experience using pure function
        experience_result = create_experience(market_state, trade_action)
        
        if experience_result.is_failure():
            return experience_result
        
        experience = experience_result.success()
        
        # Update immutable state
        new_state = self._state.add_experience(experience)
        self._update_state(new_state)
        
        # Store in MCP memory server
        store_result = await self.memory_interpreter.interpret_store_experience(experience)
        
        if store_result.is_failure():
            logger.warning("Failed to store experience in MCP: %s", store_result.failure())
        
        # Log the decision
        logger.info(
            "📝 Functional Manager: Recorded %s decision | "
            "Experience ID: %s... | "
            "Price: $%s | "
            "Patterns: %s",
            trade_action.action,
            experience.experience_id[:8],
            market_state.current_price,
            ", ".join(experience.pattern_tags),
        )
        
        # Structured logging
        self.trade_logger.log_trade_decision(
            market_state=market_state,
            trade_action=trade_action,
            experience_id=experience.experience_id,
            memory_context=None,
        )
        
        return Ok(experience.experience_id)
    
    def link_order_to_experience(self, order_id: str, experience_id: str) -> None:
        """Link an order to its corresponding experience."""
        new_state = self._state.link_order_to_experience(order_id, experience_id)
        self._update_state(new_state)
        
        logger.info(
            "🔗 Functional Manager: Linked order %s to experience %s...",
            order_id,
            experience_id[:8],
        )
    
    def start_tracking_trade(
        self,
        order: Order,
        trade_action: TradeAction,
        market_state: MarketState,
    ) -> Optional[str]:
        """Start tracking an executed trade functionally."""
        # Check for corresponding experience
        experience_id = self._state.pending_experiences.get(order.id)
        if not experience_id:
            logger.warning(
                "⚠️ Functional Manager: No experience found for order %s", order.id
            )
            return None
        
        # Create active trade state
        from uuid import uuid4
        trade_id = f"trade_{uuid4().hex[:8]}"
        
        active_trade = ActiveTradeState(
            trade_id=trade_id,
            experience_id=experience_id,
            entry_order=order,
            trade_action=trade_action,
            market_state_at_entry=market_state,
            entry_time=datetime.now(UTC),
            exit_time=None,
            exit_order=None,
            exit_price=None,
            unrealized_pnl=Decimal(0),
            realized_pnl=None,
            max_favorable_excursion=Decimal(0),
            max_adverse_excursion=Decimal(0),
            market_snapshots=(),
            last_snapshot_time=datetime.now(UTC),
        )
        
        # Update state immutably
        new_state = (self._state
                    .add_active_trade(active_trade)
                    .remove_pending_experience(order.id))
        self._update_state(new_state)
        
        logger.info(
            "🚀 Functional Manager: Started tracking %s trade | "
            "Trade ID: %s | Price: $%s | Size: %s",
            trade_action.action,
            trade_id,
            order.price,
            order.quantity,
        )
        
        return trade_id
    
    async def update_trade_progress(
        self,
        position: Position,
        current_price: Decimal,
        market_state: Optional[MarketState] = None,
    ) -> None:
        """Update trade progress functionally."""
        # Find matching active trade
        matching_trade = None
        for trade in self._state.active_trades.values():
            if (trade.entry_order.symbol == position.symbol 
                and trade.entry_order.side == position.side 
                and not trade.is_completed()):
                matching_trade = trade
                break
        
        if not matching_trade:
            return
        
        # Calculate unrealized PnL
        unrealized_pnl = self._calculate_unrealized_pnl(
            matching_trade.entry_order, current_price
        )
        
        # Update trade state immutably
        def update_trade(trade: ActiveTradeState) -> ActiveTradeState:
            updated_trade = trade.with_unrealized_pnl(unrealized_pnl)
            
            # Add market snapshot if enough time has passed
            if (market_state 
                and (datetime.now(UTC) - trade.last_snapshot_time).seconds > 60):
                snapshot = self._create_market_snapshot(current_price, unrealized_pnl, market_state)
                updated_trade = updated_trade.with_market_snapshot(snapshot)
            
            return updated_trade
        
        new_state = self._state.update_active_trade(matching_trade.trade_id, update_trade)
        self._update_state(new_state)
        
        # Log position update
        self.trade_logger.log_position_update(
            trade_id=matching_trade.trade_id,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            max_favorable=new_state.active_trades[matching_trade.trade_id].max_favorable_excursion,
            max_adverse=new_state.active_trades[matching_trade.trade_id].max_adverse_excursion,
        )
    
    async def complete_trade(
        self,
        exit_order: Order,
        exit_price: Decimal,
        market_state_at_exit: Optional[MarketState] = None,
    ) -> bool:
        """Complete a trade functionally."""
        # Find matching active trade
        matching_trade = None
        for trade in self._state.active_trades.values():
            if (trade.entry_order.symbol == exit_order.symbol 
                and not trade.is_completed()):
                matching_trade = trade
                break
        
        if not matching_trade:
            logger.warning("No active trade found for exit order %s", exit_order.id)
            return False
        
        # Calculate realized PnL
        realized_pnl = self._calculate_realized_pnl(
            matching_trade.entry_order, exit_price
        )
        
        # Complete the trade state
        completed_trade = matching_trade.with_completion(
            exit_order, exit_price, realized_pnl
        )
        
        # Calculate duration
        duration_minutes = (
            completed_trade.exit_time - completed_trade.entry_time
        ).total_seconds() / 60
        
        # Update experience with outcome
        experience = self._state.experience_state.get_experience(completed_trade.experience_id)
        if experience:
            outcome_result = update_experience_outcome(
                experience,
                realized_pnl,
                exit_price,
                duration_minutes,
                market_reaction=self._calculate_market_reaction(completed_trade, market_state_at_exit),
                insights=self._generate_trade_insights(completed_trade),
            )
            
            if outcome_result.is_success():
                updated_experience = outcome_result.success()
                
                # Update MCP memory server
                await self.memory_interpreter.interpret_update_experience(
                    completed_trade.experience_id, updated_experience
                )
                
                # Update local state
                new_experience_state = self._state.experience_state.update_experience(
                    completed_trade.experience_id,
                    lambda _: updated_experience,
                )
                
                new_state = (self._state
                             .remove_active_trade(completed_trade.trade_id)
                             .copy(experience_state=new_experience_state))
                self._update_state(new_state)
        
        # Log completion
        pnl_percentage = float(realized_pnl) / float(matching_trade.entry_order.price) * 100 if matching_trade.entry_order.price else 0.0
        
        logger.info(
            "🏁 Functional Manager: Trade completed | ID: %s | "
            "PnL: $%.2f (%s%.2f%%) | Duration: %.1fmin | %s",
            completed_trade.trade_id,
            realized_pnl,
            "+" if realized_pnl > 0 else "",
            pnl_percentage,
            duration_minutes,
            "✅ WIN" if realized_pnl > 0 else "❌ LOSS",
        )
        
        return True
    
    async def analyze_performance(self, hours: int = 24) -> Result[Dict[str, Any], str]:
        """Analyze recent performance using functional algorithms."""
        try:
            # Use functional algorithms for analysis
            pattern_analysis = analyze_trading_patterns(self._state.experience_state)
            strategy_insights = generate_strategy_insights(self._state.experience_state, hours)
            market_regimes = identify_market_regimes(self._state.experience_state)
            
            # Calculate summary metrics
            recent_cutoff = datetime.now(UTC) - timedelta(hours=hours)
            recent_experiences = self._state.experience_state.filter_experiences(
                lambda exp: exp.is_completed() and exp.timestamp >= recent_cutoff
            )
            
            total_trades = len(recent_experiences)
            successful_trades = sum(1 for exp in recent_experiences if exp.is_successful())
            total_pnl = sum(exp.get_pnl() or 0.0 for exp in recent_experiences)
            
            analysis = {
                "period_hours": hours,
                "total_trades": total_trades,
                "success_rate": successful_trades / total_trades if total_trades > 0 else 0,
                "total_pnl": total_pnl,
                "avg_pnl_per_trade": total_pnl / total_trades if total_trades > 0 else 0,
                "pattern_analysis": {
                    name: {
                        "success_rate": analysis.success_rate,
                        "avg_pnl": analysis.avg_pnl,
                        "confidence": analysis.confidence_score,
                        "sample_size": analysis.occurrence_count,
                    }
                    for name, analysis in pattern_analysis.items()
                },
                "strategy_insights": [
                    {
                        "type": insight.insight_type,
                        "description": insight.description,
                        "recommended_action": insight.recommended_action,
                        "confidence": insight.confidence,
                        "expected_improvement": insight.expected_improvement,
                    }
                    for insight in strategy_insights
                ],
                "market_regimes": [
                    {
                        "name": regime.regime_name,
                        "performance": regime.performance_stats,
                    }
                    for regime in market_regimes
                ],
                "active_trades_count": len(self._state.active_trades),
            }
            
            logger.info(
                "📊 Functional Analysis: %s trades, %.1f%% success rate, $%.2f total PnL",
                total_trades,
                analysis["success_rate"] * 100,
                total_pnl,
            )
            
            return Ok(analysis)
            
        except Exception as e:
            return Err(f"Performance analysis failed: {str(e)}")
    
    def get_active_trades_summary(self) -> Dict[str, Any]:
        """Get summary of active trades."""
        total_unrealized_pnl = sum(
            trade.unrealized_pnl for trade in self._state.active_trades.values()
        )
        
        trades_info = []
        for trade in self._state.active_trades.values():
            trade_info = {
                "trade_id": trade.trade_id,
                "symbol": trade.entry_order.symbol,
                "side": trade.entry_order.side,
                "entry_price": float(trade.entry_order.price) if trade.entry_order.price else 0.0,
                "size": float(trade.entry_order.quantity) if trade.entry_order.quantity else 0.0,
                "unrealized_pnl": float(trade.unrealized_pnl),
                "duration_hours": (datetime.now(UTC) - trade.entry_time).total_seconds() / 3600,
                "max_profit": float(trade.max_favorable_excursion),
                "max_loss": float(trade.max_adverse_excursion),
            }
            trades_info.append(trade_info)
        
        return {
            "active_count": len(self._state.active_trades),
            "total_unrealized_pnl": float(total_unrealized_pnl),
            "trades": trades_info,
        }
    
    # Helper methods (pure functions)
    
    def _calculate_unrealized_pnl(self, entry_order: Order, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL (pure function)."""
        if not entry_order.price or not entry_order.quantity:
            return Decimal(0)
        
        if entry_order.side == "BUY":  # LONG position
            return (current_price - entry_order.price) * entry_order.quantity
        else:  # SHORT position
            return (entry_order.price - current_price) * entry_order.quantity
    
    def _calculate_realized_pnl(self, entry_order: Order, exit_price: Decimal) -> Decimal:
        """Calculate realized PnL (pure function)."""
        if not entry_order.price or not entry_order.quantity:
            return Decimal(0)
        
        if entry_order.side == "BUY":  # Was LONG
            return (exit_price - entry_order.price) * entry_order.quantity
        else:  # Was SHORT
            return (entry_order.price - exit_price) * entry_order.quantity
    
    def _create_market_snapshot(
        self, 
        current_price: Decimal, 
        unrealized_pnl: Decimal,
        market_state: MarketState,
    ) -> Dict[str, Any]:
        """Create market snapshot (pure function)."""
        snapshot = {
            "timestamp": datetime.now(UTC).isoformat(),
            "price": float(current_price),
            "unrealized_pnl": float(unrealized_pnl),
        }
        
        if market_state.indicators:
            ind = market_state.indicators
            snapshot["indicators"] = {
                "rsi": float(ind.rsi) if ind.rsi else 50.0,
                "cipher_b_wave": float(ind.cipher_b_wave) if ind.cipher_b_wave else 0.0,
            }
        
        return snapshot
    
    def _calculate_market_reaction(
        self,
        completed_trade: ActiveTradeState,
        market_state_at_exit: Optional[MarketState],
    ) -> Optional[Dict[str, float]]:
        """Calculate market reaction (pure function)."""
        if not market_state_at_exit or not completed_trade.exit_price:
            return None
        
        price_change = float(completed_trade.exit_price - completed_trade.entry_order.price)
        
        return {
            "price_change": price_change,
            "volume_ratio": 1.0,  # Placeholder
        }
    
    def _generate_trade_insights(self, completed_trade: ActiveTradeState) -> str:
        """Generate trade insights (pure function)."""
        insights = []
        
        if completed_trade.realized_pnl and completed_trade.realized_pnl > 0:
            insights.append("Successful trade execution")
            
            # Analyze profit capture
            if completed_trade.max_favorable_excursion > 0:
                capture_ratio = float(completed_trade.realized_pnl / completed_trade.max_favorable_excursion)
                if capture_ratio > 0.8:
                    insights.append("Excellent profit capture")
                elif capture_ratio < 0.5:
                    insights.append("Consider trailing stops for better profit capture")
        else:
            insights.append("Trade resulted in loss")
            
            if abs(completed_trade.max_adverse_excursion) > 100:
                insights.append("Significant drawdown - consider tighter stops")
        
        return "; ".join(insights) if insights else "Standard trade execution"
    
    async def _monitor_trades_functional(self) -> None:
        """Functional trade monitoring loop."""
        while self._running:
            try:
                current_time = datetime.now(UTC)
                
                # Check for stale trades using functional approach
                stale_trade_ids = []
                for trade_id, trade in self._state.active_trades.items():
                    trade_duration_hours = (current_time - trade.entry_time).total_seconds() / 3600
                    
                    if trade_duration_hours > 24:
                        logger.warning(
                            "Trade %s has been open for %.1f hours",
                            trade_id,
                            trade_duration_hours,
                        )
                        
                        if trade_duration_hours > 48:
                            stale_trade_ids.append(trade_id)
                
                # Remove very old trades
                if stale_trade_ids:
                    new_state = self._state
                    for trade_id in stale_trade_ids:
                        logger.error("Removing stale trade %s - open for over 48 hours", trade_id)
                        new_state = new_state.remove_active_trade(trade_id)
                    
                    self._update_state(new_state)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception:
                logger.exception("Error in functional trade monitoring")
                await asyncio.sleep(60)