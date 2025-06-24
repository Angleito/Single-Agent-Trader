"""
Functional programming types for learning and memory systems.

This module defines immutable data structures and pure functions for the
AI trading bot's learning system, including memory storage, experience tracking,
and self-improvement features.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union
from uuid import uuid4

from bot.fp.types.base import Maybe, Nothing, Some, Symbol, Timestamp
from bot.fp.types.result import Result, Success, Failure
from bot.fp.types.trading import MarketData, TradeDecision


@dataclass(frozen=True)
class ExperienceId:
    """Unique identifier for a trading experience."""
    
    value: str
    
    @classmethod
    def generate(cls) -> "ExperienceId":
        """Generate a new experience ID."""
        return cls(value=str(uuid4()))
    
    @classmethod
    def create(cls, value: str) -> Result["ExperienceId", str]:
        """Create experience ID with validation."""
        if not value or not value.strip():
            return Failure("Experience ID cannot be empty")
        return Success(cls(value=value.strip()))
    
    def __str__(self) -> str:
        return self.value
    
    def short(self) -> str:
        """Get shortened ID for logging."""
        return self.value[:8]


@dataclass(frozen=True)
class TradingOutcome:
    """Immutable trading outcome data."""
    
    pnl: Decimal
    exit_price: Decimal
    price_change_pct: Decimal
    is_successful: bool
    duration_minutes: Decimal
    
    @classmethod
    def create(
        cls,
        pnl: Decimal,
        exit_price: Decimal,
        entry_price: Decimal,
        duration_minutes: float,
    ) -> Result["TradingOutcome", str]:
        """Create trading outcome with calculations."""
        try:
            if entry_price <= 0:
                return Failure("Entry price must be positive")
            
            price_change_pct = (exit_price - entry_price) / entry_price * 100
            is_successful = pnl > 0
            
            return Success(cls(
                pnl=pnl,
                exit_price=exit_price,
                price_change_pct=price_change_pct,
                is_successful=is_successful,
                duration_minutes=Decimal(str(duration_minutes)),
            ))
        except Exception as e:
            return Failure(f"Failed to create trading outcome: {e}")


@dataclass(frozen=True)
class PatternTag:
    """Immutable pattern tag for experience classification."""
    
    name: str
    
    @classmethod
    def create(cls, name: str) -> Result["PatternTag", str]:
        """Create pattern tag with validation."""
        if not name or not name.strip():
            return Failure("Pattern tag name cannot be empty")
        
        normalized = name.strip().lower().replace(" ", "_")
        return Success(cls(name=normalized))
    
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable market state snapshot."""
    
    symbol: Symbol
    timestamp: datetime
    price: Decimal
    indicators: Dict[str, float]
    dominance_data: Optional[Dict[str, float]]
    position_side: str
    position_size: Decimal
    
    @classmethod
    def from_market_state(cls, market_state) -> "MarketSnapshot":
        """Create snapshot from market state (for adapter compatibility)."""
        indicators = {}
        if market_state.indicators:
            indicators = {
                "rsi": float(market_state.indicators.rsi or 50.0),
                "cipher_a_dot": float(market_state.indicators.cipher_a_dot or 0.0),
                "cipher_b_wave": float(market_state.indicators.cipher_b_wave or 0.0),
                "cipher_b_money_flow": float(market_state.indicators.cipher_b_money_flow or 50.0),
                "ema_fast": float(market_state.indicators.ema_fast or 0.0),
                "ema_slow": float(market_state.indicators.ema_slow or 0.0),
            }
        
        dominance_data = None
        if market_state.dominance_data:
            dominance_data = {
                "stablecoin_dominance": float(market_state.dominance_data.stablecoin_dominance),
                "dominance_24h_change": float(market_state.dominance_data.dominance_24h_change),
                "dominance_rsi": float(market_state.dominance_data.dominance_rsi or 50.0),
            }
        
        symbol_result = Symbol.create(market_state.symbol)
        symbol = symbol_result.success() if symbol_result.is_success() else Symbol.create("BTC-USD").success()
        
        return cls(
            symbol=symbol,
            timestamp=market_state.timestamp,
            price=market_state.current_price,
            indicators=indicators,
            dominance_data=dominance_data,
            position_side=market_state.current_position.side,
            position_size=market_state.current_position.size,
        )


@dataclass(frozen=True)
class TradingExperienceFP:
    """Immutable trading experience for functional programming."""
    
    experience_id: ExperienceId
    timestamp: datetime
    market_snapshot: MarketSnapshot
    trade_decision: TradeDecision
    decision_rationale: str
    pattern_tags: List[PatternTag]
    outcome: Maybe[TradingOutcome]
    learned_insights: Maybe[str]
    confidence_score: Decimal
    
    @classmethod
    def create(
        cls,
        market_snapshot: MarketSnapshot,
        trade_decision: TradeDecision,
        decision_rationale: str,
        pattern_tags: Optional[List[PatternTag]] = None,
    ) -> "TradingExperienceFP":
        """Create a new trading experience."""
        return cls(
            experience_id=ExperienceId.generate(),
            timestamp=datetime.utcnow(),
            market_snapshot=market_snapshot,
            trade_decision=trade_decision,
            decision_rationale=decision_rationale,
            pattern_tags=pattern_tags or [],
            outcome=Nothing(),
            learned_insights=Nothing(),
            confidence_score=Decimal("0.5"),
        )
    
    def with_outcome(self, outcome: TradingOutcome) -> "TradingExperienceFP":
        """Create new experience with outcome added."""
        from dataclasses import replace
        return replace(
            self,
            outcome=Some(outcome),
            confidence_score=self._calculate_confidence(outcome),
        )
    
    def with_insights(self, insights: str) -> "TradingExperienceFP":
        """Create new experience with insights added."""
        from dataclasses import replace
        return replace(self, learned_insights=Some(insights))
    
    def _calculate_confidence(self, outcome: TradingOutcome) -> Decimal:
        """Calculate confidence score based on outcome."""
        base_confidence = Decimal("0.6") if outcome.is_successful else Decimal("0.4")
        
        if outcome.is_successful:
            # Higher profit = higher confidence (capped at 0.3 boost)
            profit_factor = min(outcome.pnl / 100, Decimal("0.3"))
            base_confidence += profit_factor
        else:
            # Larger loss = lower confidence (capped at 0.2 reduction)
            loss_factor = min(abs(outcome.pnl) / 100, Decimal("0.2"))
            base_confidence -= loss_factor
        
        # Ensure within bounds
        return max(Decimal("0.1"), min(Decimal("0.9"), base_confidence))
    
    def is_completed(self) -> bool:
        """Check if experience has an outcome."""
        return self.outcome.is_some()
    
    def get_success_rate(self) -> Maybe[bool]:
        """Get success rate if outcome is available."""
        return self.outcome.map(lambda o: o.is_successful)


@dataclass(frozen=True)
class MemoryQueryFP:
    """Immutable memory query parameters."""
    
    current_price: Maybe[Decimal]
    indicators: Dict[str, float]
    dominance_data: Maybe[Dict[str, float]]
    pattern_tags: List[PatternTag]
    max_results: int
    min_similarity: Decimal
    time_weight: Decimal
    
    @classmethod
    def create(
        cls,
        current_price: Optional[Decimal] = None,
        indicators: Optional[Dict[str, float]] = None,
        dominance_data: Optional[Dict[str, float]] = None,
        pattern_tags: Optional[List[PatternTag]] = None,
        max_results: int = 10,
        min_similarity: float = 0.7,
        time_weight: float = 0.2,
    ) -> Result["MemoryQueryFP", str]:
        """Create memory query with validation."""
        try:
            if max_results <= 0 or max_results > 50:
                return Failure("max_results must be between 1 and 50")
            
            if not (0.0 <= min_similarity <= 1.0):
                return Failure("min_similarity must be between 0.0 and 1.0")
            
            if not (0.0 <= time_weight <= 1.0):
                return Failure("time_weight must be between 0.0 and 1.0")
            
            return Success(cls(
                current_price=Some(current_price) if current_price is not None else Nothing(),
                indicators=indicators or {},
                dominance_data=Some(dominance_data) if dominance_data is not None else Nothing(),
                pattern_tags=pattern_tags or [],
                max_results=max_results,
                min_similarity=Decimal(str(min_similarity)),
                time_weight=Decimal(str(time_weight)),
            ))
        except Exception as e:
            return Failure(f"Failed to create memory query: {e}")


@dataclass(frozen=True)
class PatternStatistics:
    """Immutable pattern performance statistics."""
    
    pattern: PatternTag
    total_occurrences: int
    successful_trades: int
    total_pnl: Decimal
    average_pnl: Decimal
    success_rate: Decimal
    
    @classmethod
    def calculate(
        cls,
        pattern: PatternTag,
        experiences: List[TradingExperienceFP],
    ) -> Result["PatternStatistics", str]:
        """Calculate statistics for a pattern from experiences."""
        try:
            # Filter experiences with this pattern and outcomes
            relevant_experiences = [
                exp for exp in experiences
                if pattern in exp.pattern_tags and exp.outcome.is_some()
            ]
            
            if not relevant_experiences:
                return Success(cls(
                    pattern=pattern,
                    total_occurrences=0,
                    successful_trades=0,
                    total_pnl=Decimal("0"),
                    average_pnl=Decimal("0"),
                    success_rate=Decimal("0"),
                ))
            
            total_occurrences = len(relevant_experiences)
            successful_trades = sum(
                1 for exp in relevant_experiences
                if exp.outcome.value.is_successful
            )
            total_pnl = sum(
                exp.outcome.value.pnl for exp in relevant_experiences
            )
            average_pnl = total_pnl / len(relevant_experiences)
            success_rate = Decimal(str(successful_trades)) / Decimal(str(total_occurrences))
            
            return Success(cls(
                pattern=pattern,
                total_occurrences=total_occurrences,
                successful_trades=successful_trades,
                total_pnl=total_pnl,
                average_pnl=average_pnl,
                success_rate=success_rate,
            ))
        except Exception as e:
            return Failure(f"Failed to calculate pattern statistics: {e}")
    
    def is_profitable(self) -> bool:
        """Check if pattern is profitable overall."""
        return self.average_pnl > 0
    
    def is_reliable(self, min_occurrences: int = 5) -> bool:
        """Check if pattern has enough data to be reliable."""
        return self.total_occurrences >= min_occurrences


@dataclass(frozen=True)
class LearningInsight:
    """Immutable learning insight from trade analysis."""
    
    insight_type: str
    description: str
    confidence: Decimal
    supporting_evidence: List[str]
    related_patterns: List[PatternTag]
    
    @classmethod
    def create(
        cls,
        insight_type: str,
        description: str,
        confidence: float = 0.5,
        supporting_evidence: Optional[List[str]] = None,
        related_patterns: Optional[List[PatternTag]] = None,
    ) -> Result["LearningInsight", str]:
        """Create learning insight with validation."""
        try:
            if not insight_type or not insight_type.strip():
                return Failure("Insight type cannot be empty")
            
            if not description or not description.strip():
                return Failure("Description cannot be empty")
            
            if not (0.0 <= confidence <= 1.0):
                return Failure("Confidence must be between 0.0 and 1.0")
            
            return Success(cls(
                insight_type=insight_type.strip(),
                description=description.strip(),
                confidence=Decimal(str(confidence)),
                supporting_evidence=supporting_evidence or [],
                related_patterns=related_patterns or [],
            ))
        except Exception as e:
            return Failure(f"Failed to create learning insight: {e}")


@dataclass(frozen=True)
class MemoryStorage:
    """Immutable memory storage state."""
    
    experiences: List[TradingExperienceFP]
    pattern_index: Dict[str, List[ExperienceId]]
    total_experiences: int
    completed_experiences: int
    
    @classmethod
    def empty(cls) -> "MemoryStorage":
        """Create empty memory storage."""
        return cls(
            experiences=[],
            pattern_index={},
            total_experiences=0,
            completed_experiences=0,
        )
    
    def add_experience(self, experience: TradingExperienceFP) -> "MemoryStorage":
        """Add experience to storage (returns new storage)."""
        new_experiences = self.experiences + [experience]
        
        # Update pattern index
        new_pattern_index = dict(self.pattern_index)
        for pattern in experience.pattern_tags:
            pattern_key = pattern.name
            if pattern_key not in new_pattern_index:
                new_pattern_index[pattern_key] = []
            new_pattern_index[pattern_key] = new_pattern_index[pattern_key] + [experience.experience_id]
        
        new_completed = self.completed_experiences + (1 if experience.is_completed() else 0)
        
        from dataclasses import replace
        return replace(
            self,
            experiences=new_experiences,
            pattern_index=new_pattern_index,
            total_experiences=self.total_experiences + 1,
            completed_experiences=new_completed,
        )
    
    def update_experience(
        self,
        experience_id: ExperienceId,
        updater: callable,
    ) -> Result["MemoryStorage", str]:
        """Update an experience (returns new storage)."""
        try:
            # Find experience index
            experience_index = None
            for i, exp in enumerate(self.experiences):
                if exp.experience_id.value == experience_id.value:
                    experience_index = i
                    break
            
            if experience_index is None:
                return Failure(f"Experience {experience_id} not found")
            
            # Apply update
            old_experience = self.experiences[experience_index]
            new_experience = updater(old_experience)
            
            # Replace in list
            new_experiences = list(self.experiences)
            new_experiences[experience_index] = new_experience
            
            # Update completion count if needed
            completion_change = 0
            if not old_experience.is_completed() and new_experience.is_completed():
                completion_change = 1
            elif old_experience.is_completed() and not new_experience.is_completed():
                completion_change = -1
            
            from dataclasses import replace
            return Success(replace(
                self,
                experiences=new_experiences,
                completed_experiences=self.completed_experiences + completion_change,
            ))
        except Exception as e:
            return Failure(f"Failed to update experience: {e}")
    
    def find_by_id(self, experience_id: ExperienceId) -> Maybe[TradingExperienceFP]:
        """Find experience by ID."""
        for exp in self.experiences:
            if exp.experience_id.value == experience_id.value:
                return Some(exp)
        return Nothing()
    
    def find_by_pattern(self, pattern: PatternTag) -> List[TradingExperienceFP]:
        """Find experiences by pattern."""
        pattern_key = pattern.name
        if pattern_key not in self.pattern_index:
            return []
        
        experience_ids = self.pattern_index[pattern_key]
        experiences = []
        for exp_id in experience_ids:
            exp = self.find_by_id(exp_id)
            if exp.is_some():
                experiences.append(exp.value)
        
        return experiences
    
    def get_completed_experiences(self) -> List[TradingExperienceFP]:
        """Get all completed experiences."""
        return [exp for exp in self.experiences if exp.is_completed()]


# Export main types
__all__ = [
    "ExperienceId",
    "TradingOutcome",
    "PatternTag",
    "MarketSnapshot",
    "TradingExperienceFP",
    "MemoryQueryFP",
    "PatternStatistics",
    "LearningInsight",
    "MemoryStorage",
]
