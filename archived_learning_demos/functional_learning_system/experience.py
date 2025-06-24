"""
Immutable experience state management for functional learning.

This module provides pure functional data structures and operations
for managing trading experiences in an immutable way.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from bot.fp.types.result import Result, Ok, Err
from bot.trading_types import MarketState, TradeAction


@dataclass(frozen=True)
class TradeExperience:
    """Immutable representation of a trading experience."""
    
    experience_id: str
    timestamp: datetime
    
    # Market context (immutable snapshots)
    symbol: str
    price: Decimal
    market_snapshot: Dict[str, Any]  # Serialized market state
    indicators: Dict[str, float]
    dominance_data: Optional[Dict[str, float]]
    
    # Trading decision (immutable)
    decision: Dict[str, Any]  # Serialized trade action
    rationale: str
    
    # Outcome data (None until trade completes)
    outcome: Optional[Dict[str, Any]]
    duration_minutes: Optional[float]
    market_reaction: Optional[Dict[str, float]]
    
    # Learning metadata
    pattern_tags: Tuple[str, ...]  # Immutable tuple
    confidence_score: float
    learned_insights: Optional[str]
    
    @classmethod
    def create(
        cls,
        market_state: MarketState,
        trade_action: TradeAction,
        pattern_tags: List[str] = None,
    ) -> TradeExperience:
        """Create a new immutable trading experience."""
        return cls(
            experience_id=str(uuid4()),
            timestamp=datetime.now(UTC),
            symbol=market_state.symbol,
            price=market_state.current_price,
            market_snapshot=cls._serialize_market_state(market_state),
            indicators=cls._extract_indicators(market_state),
            dominance_data=cls._extract_dominance_data(market_state),
            decision=cls._serialize_trade_action(trade_action),
            rationale=trade_action.rationale,
            outcome=None,
            duration_minutes=None,
            market_reaction=None,
            pattern_tags=tuple(pattern_tags or []),
            confidence_score=0.5,
            learned_insights=None,
        )
    
    def with_outcome(
        self,
        pnl: Decimal,
        exit_price: Decimal,
        duration_minutes: float,
        market_reaction: Optional[Dict[str, float]] = None,
        insights: Optional[str] = None,
    ) -> TradeExperience:
        """Return a new experience with outcome data."""
        entry_price = self.price
        price_change_pct = float((exit_price - entry_price) / entry_price * 100)
        
        outcome = {
            "pnl": float(pnl),
            "exit_price": float(exit_price),
            "price_change_pct": price_change_pct,
            "success": pnl > 0,
            "duration_minutes": duration_minutes,
        }
        
        confidence = self._calculate_confidence(outcome)
        
        return replace(
            self,
            outcome=outcome,
            duration_minutes=duration_minutes,
            market_reaction=market_reaction,
            learned_insights=insights,
            confidence_score=confidence,
        )
    
    def with_insights(self, insights: str) -> TradeExperience:
        """Return a new experience with learning insights."""
        return replace(self, learned_insights=insights)
    
    def with_patterns(self, patterns: List[str]) -> TradeExperience:
        """Return a new experience with updated pattern tags."""
        return replace(self, pattern_tags=tuple(patterns))
    
    def is_completed(self) -> bool:
        """Check if this experience has outcome data."""
        return self.outcome is not None
    
    def is_successful(self) -> bool:
        """Check if this was a successful trade."""
        return self.outcome is not None and self.outcome.get("success", False)
    
    def get_pnl(self) -> Optional[float]:
        """Get the PnL for this experience."""
        return self.outcome.get("pnl") if self.outcome else None
    
    @staticmethod
    def _serialize_market_state(market_state: MarketState) -> Dict[str, Any]:
        """Serialize market state to immutable dict."""
        return {
            "symbol": market_state.symbol,
            "interval": market_state.interval,
            "timestamp": market_state.timestamp.isoformat(),
            "current_price": float(market_state.current_price),
            "ohlcv_count": len(market_state.ohlcv_data),
            "position_side": market_state.current_position.side,
            "position_size": float(market_state.current_position.size),
        }
    
    @staticmethod
    def _serialize_trade_action(trade_action: TradeAction) -> Dict[str, Any]:
        """Serialize trade action to immutable dict."""
        return {
            "action": trade_action.action,
            "size_pct": trade_action.size_pct,
            "take_profit_pct": trade_action.take_profit_pct,
            "stop_loss_pct": trade_action.stop_loss_pct,
            "leverage": trade_action.leverage,
            "reduce_only": trade_action.reduce_only,
        }
    
    @staticmethod
    def _extract_indicators(market_state: MarketState) -> Dict[str, float]:
        """Extract indicators to immutable dict."""
        if not market_state.indicators:
            return {}
        
        ind = market_state.indicators
        return {
            "rsi": float(ind.rsi) if ind.rsi else 50.0,
            "cipher_a_dot": float(ind.cipher_a_dot) if ind.cipher_a_dot else 0.0,
            "cipher_b_wave": float(ind.cipher_b_wave) if ind.cipher_b_wave else 0.0,
            "cipher_b_money_flow": float(ind.cipher_b_money_flow) if ind.cipher_b_money_flow else 50.0,
            "ema_fast": float(ind.ema_fast) if ind.ema_fast else 0.0,
            "ema_slow": float(ind.ema_slow) if ind.ema_slow else 0.0,
        }
    
    @staticmethod
    def _extract_dominance_data(market_state: MarketState) -> Optional[Dict[str, float]]:
        """Extract dominance data to immutable dict."""
        if not market_state.dominance_data:
            return None
        
        dom = market_state.dominance_data
        return {
            "stablecoin_dominance": float(dom.stablecoin_dominance),
            "dominance_24h_change": float(dom.dominance_24h_change),
            "dominance_rsi": float(dom.dominance_rsi) if dom.dominance_rsi else 50.0,
        }
    
    def _calculate_confidence(self, outcome: Dict[str, Any]) -> float:
        """Calculate confidence score based on outcome."""
        base_confidence = 0.6 if outcome.get("success", False) else 0.4
        
        # Adjust based on profit magnitude
        pnl = outcome.get("pnl", 0.0)
        if outcome.get("success", False):
            profit_factor = min(abs(pnl) / 100, 0.3)
            base_confidence += profit_factor
        else:
            loss_factor = min(abs(pnl) / 100, 0.2)
            base_confidence -= loss_factor
        
        return max(0.1, min(0.9, base_confidence))


@dataclass(frozen=True)
class PatternState:
    """Immutable state for pattern analysis."""
    
    pattern_name: str
    occurrence_count: int
    success_count: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    avg_duration_minutes: float
    confidence_score: float
    last_updated: datetime
    
    @classmethod
    def empty(cls, pattern_name: str) -> PatternState:
        """Create empty pattern state."""
        return cls(
            pattern_name=pattern_name,
            occurrence_count=0,
            success_count=0,
            total_pnl=0.0,
            avg_pnl=0.0,
            win_rate=0.0,
            avg_duration_minutes=0.0,
            confidence_score=0.5,
            last_updated=datetime.now(UTC),
        )
    
    def update_with_experience(self, experience: TradeExperience) -> PatternState:
        """Return new state updated with experience."""
        if not experience.is_completed():
            return self
        
        new_count = self.occurrence_count + 1
        new_success_count = (
            self.success_count + 1 if experience.is_successful() else self.success_count
        )
        
        pnl = experience.get_pnl() or 0.0
        new_total_pnl = self.total_pnl + pnl
        new_avg_pnl = new_total_pnl / new_count
        new_win_rate = new_success_count / new_count
        
        # Update average duration
        if experience.duration_minutes:
            total_duration = self.avg_duration_minutes * self.occurrence_count
            new_avg_duration = (total_duration + experience.duration_minutes) / new_count
        else:
            new_avg_duration = self.avg_duration_minutes
        
        # Calculate new confidence
        new_confidence = calculate_pattern_confidence(new_count, new_win_rate)
        
        return replace(
            self,
            occurrence_count=new_count,
            success_count=new_success_count,
            total_pnl=new_total_pnl,
            avg_pnl=new_avg_pnl,
            win_rate=new_win_rate,
            avg_duration_minutes=new_avg_duration,
            confidence_score=new_confidence,
            last_updated=datetime.now(UTC),
        )


@dataclass(frozen=True)
class ExperienceState:
    """Immutable state container for all learning data."""
    
    experiences: Tuple[TradeExperience, ...]
    patterns: Dict[str, PatternState]
    total_experiences: int
    total_completed: int
    total_successful: int
    last_updated: datetime
    
    @classmethod
    def empty(cls) -> ExperienceState:
        """Create empty experience state."""
        return cls(
            experiences=(),
            patterns={},
            total_experiences=0,
            total_completed=0,
            total_successful=0,
            last_updated=datetime.now(UTC),
        )
    
    def add_experience(self, experience: TradeExperience) -> ExperienceState:
        """Return new state with added experience."""
        new_experiences = self.experiences + (experience,)
        new_total = self.total_experiences + 1
        
        # Update pattern states
        new_patterns = dict(self.patterns)
        for pattern in experience.pattern_tags:
            if pattern not in new_patterns:
                new_patterns[pattern] = PatternState.empty(pattern)
            
            new_patterns[pattern] = new_patterns[pattern].update_with_experience(experience)
        
        # Update completion stats
        new_completed = self.total_completed
        new_successful = self.total_successful
        if experience.is_completed():
            new_completed += 1
            if experience.is_successful():
                new_successful += 1
        
        return replace(
            self,
            experiences=new_experiences,
            patterns=new_patterns,
            total_experiences=new_total,
            total_completed=new_completed,
            total_successful=new_successful,
            last_updated=datetime.now(UTC),
        )
    
    def update_experience(
        self, 
        experience_id: str, 
        update_fn: Callable[[TradeExperience], TradeExperience]
    ) -> ExperienceState:
        """Return new state with updated experience."""
        new_experiences = []
        found = False
        
        for exp in self.experiences:
            if exp.experience_id == experience_id:
                updated_exp = update_fn(exp)
                new_experiences.append(updated_exp)
                found = True
            else:
                new_experiences.append(exp)
        
        if not found:
            return self
        
        return replace(
            self,
            experiences=tuple(new_experiences),
            last_updated=datetime.now(UTC),
        )
    
    def get_experience(self, experience_id: str) -> Optional[TradeExperience]:
        """Get experience by ID."""
        for exp in self.experiences:
            if exp.experience_id == experience_id:
                return exp
        return None
    
    def filter_experiences(
        self, 
        predicate: Callable[[TradeExperience], bool]
    ) -> Tuple[TradeExperience, ...]:
        """Filter experiences by predicate."""
        return tuple(exp for exp in self.experiences if predicate(exp))
    
    def get_pattern_performance(self, pattern: str) -> Optional[PatternState]:
        """Get performance for a specific pattern."""
        return self.patterns.get(pattern)


@dataclass(frozen=True)
class LearningContext:
    """Immutable context for learning operations."""
    
    current_state: ExperienceState
    config: Dict[str, Any]
    timestamp: datetime
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> LearningContext:
        """Create new learning context."""
        return cls(
            current_state=ExperienceState.empty(),
            config=config,
            timestamp=datetime.now(UTC),
        )
    
    def with_state(self, state: ExperienceState) -> LearningContext:
        """Return context with new state."""
        return replace(self, current_state=state, timestamp=datetime.now(UTC))


# Pure functions for experience operations

def create_experience(
    market_state: MarketState,
    trade_action: TradeAction,
    patterns: List[str] = None,
) -> Result[TradeExperience, str]:
    """Pure function to create a new trading experience."""
    try:
        experience = TradeExperience.create(market_state, trade_action, patterns)
        return Ok(experience)
    except Exception as e:
        return Err(f"Failed to create experience: {str(e)}")


def update_experience_outcome(
    experience: TradeExperience,
    pnl: Decimal,
    exit_price: Decimal,
    duration_minutes: float,
    market_reaction: Optional[Dict[str, float]] = None,
    insights: Optional[str] = None,
) -> Result[TradeExperience, str]:
    """Pure function to update experience with outcome."""
    try:
        updated_experience = experience.with_outcome(
            pnl, exit_price, duration_minutes, market_reaction, insights
        )
        return Ok(updated_experience)
    except Exception as e:
        return Err(f"Failed to update experience outcome: {str(e)}")


def query_similar_experiences(
    state: ExperienceState,
    target_indicators: Dict[str, float],
    similarity_threshold: float = 0.7,
    max_results: int = 10,
) -> Tuple[TradeExperience, ...]:
    """Pure function to query similar experiences."""
    def similarity_score(exp: TradeExperience) -> float:
        if not exp.indicators:
            return 0.0
        
        # Simple cosine similarity for indicators
        target_values = list(target_indicators.values())
        exp_values = [exp.indicators.get(k, 0.0) for k in target_indicators.keys()]
        
        if not target_values or not exp_values:
            return 0.0
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(target_values, exp_values))
        
        # Magnitudes
        target_mag = sum(a * a for a in target_values) ** 0.5
        exp_mag = sum(b * b for b in exp_values) ** 0.5
        
        if target_mag == 0 or exp_mag == 0:
            return 0.0
        
        return dot_product / (target_mag * exp_mag)
    
    # Filter completed experiences and calculate similarities
    completed_experiences = state.filter_experiences(lambda exp: exp.is_completed())
    similarities = [(exp, similarity_score(exp)) for exp in completed_experiences]
    
    # Filter by threshold and sort by similarity
    filtered = [(exp, score) for exp, score in similarities if score >= similarity_threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    # Return top results
    return tuple(exp for exp, _ in filtered[:max_results])


def analyze_pattern_performance(
    state: ExperienceState,
    pattern: str,
) -> Optional[PatternState]:
    """Pure function to analyze pattern performance."""
    return state.get_pattern_performance(pattern)


def calculate_pattern_confidence(sample_size: int, win_rate: float) -> float:
    """Pure function to calculate pattern confidence."""
    base_confidence = win_rate
    sample_factor = min(sample_size / 20, 1.0)  # Max confidence at 20 samples
    time_decay = 0.95  # Could be calculated from timestamps
    
    confidence = base_confidence * sample_factor * time_decay
    return max(0.1, min(0.95, confidence))