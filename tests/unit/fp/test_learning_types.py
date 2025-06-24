"""
Unit tests for FP learning system types.

These tests validate the immutable data structures, pure functions,
and type safety of the functional programming learning system components.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot.fp.types import (
    Result, Success, Failure,
    Maybe, Some, Nothing,
    Symbol, ExperienceId, TradingExperienceFP, TradingOutcome,
    MarketSnapshot, PatternTag, MemoryQueryFP, PatternStatistics,
    LearningInsight, MemoryStorage,
)


# ExperienceId Tests

class TestExperienceId:
    """Test ExperienceId type functionality."""
    
    def test_experience_id_generate(self):
        """Test experience ID generation."""
        exp_id = ExperienceId.generate()
        assert isinstance(exp_id, ExperienceId)
        assert len(exp_id.value) > 0
        assert exp_id.short() == exp_id.value[:8]
        
        # Generate multiple IDs - should be unique
        exp_id2 = ExperienceId.generate()
        assert exp_id.value != exp_id2.value
    
    def test_experience_id_create_valid(self):
        """Test valid experience ID creation."""
        result = ExperienceId.create("exp_123456789")
        assert result.is_success()
        exp_id = result.success()
        assert exp_id.value == "exp_123456789"
        assert exp_id.short() == "exp_1234"
    
    def test_experience_id_create_invalid(self):
        """Test invalid experience ID creation."""
        invalid_cases = ["", "   ", "\t\n"]
        for invalid_id in invalid_cases:
            result = ExperienceId.create(invalid_id)
            assert result.is_failure()
            assert "empty" in result.failure().lower()
    
    def test_experience_id_string_representation(self):
        """Test experience ID string representation."""
        exp_id = ExperienceId.create("test_id_123").success()
        assert str(exp_id) == "test_id_123"


# TradingOutcome Tests

class TestTradingOutcome:
    """Test TradingOutcome type functionality."""
    
    def test_trading_outcome_create_profitable(self):
        """Test profitable trading outcome creation."""
        result = TradingOutcome.create(
            pnl=Decimal("100"),
            exit_price=Decimal("51000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        )
        
        assert result.is_success()
        outcome = result.success()
        assert outcome.pnl == Decimal("100")
        assert outcome.exit_price == Decimal("51000")
        assert outcome.price_change_pct == Decimal("2")  # 2% gain
        assert outcome.is_successful is True
        assert outcome.duration_minutes == Decimal("30")
    
    def test_trading_outcome_create_losing(self):
        """Test losing trading outcome creation."""
        result = TradingOutcome.create(
            pnl=Decimal("-75"),
            exit_price=Decimal("49000"),
            entry_price=Decimal("50000"),
            duration_minutes=45.0,
        )
        
        assert result.is_success()
        outcome = result.success()
        assert outcome.pnl == Decimal("-75")
        assert outcome.exit_price == Decimal("49000")
        assert outcome.price_change_pct == Decimal("-2")  # 2% loss
        assert outcome.is_successful is False
        assert outcome.duration_minutes == Decimal("45")
    
    def test_trading_outcome_create_invalid_entry_price(self):
        """Test trading outcome creation with invalid entry price."""
        result = TradingOutcome.create(
            pnl=Decimal("100"),
            exit_price=Decimal("51000"),
            entry_price=Decimal("0"),  # Invalid
            duration_minutes=30.0,
        )
        
        assert result.is_failure()
        assert "positive" in result.failure().lower()


# PatternTag Tests

class TestPatternTag:
    """Test PatternTag type functionality."""
    
    def test_pattern_tag_create_valid(self):
        """Test valid pattern tag creation."""
        valid_patterns = [
            "uptrend",
            "Oversold RSI",
            "High Volume",
            "BULLISH_DIVERGENCE",
        ]
        
        for pattern in valid_patterns:
            result = PatternTag.create(pattern)
            assert result.is_success()
            tag = result.success()
            # Should be normalized to lowercase with underscores
            assert tag.name == pattern.lower().replace(" ", "_")
            assert str(tag) == tag.name
    
    def test_pattern_tag_create_invalid(self):
        """Test invalid pattern tag creation."""
        invalid_patterns = ["", "   ", "\t\n"]
        for pattern in invalid_patterns:
            result = PatternTag.create(pattern)
            assert result.is_failure()
            assert "empty" in result.failure().lower()
    
    def test_pattern_tag_equality(self):
        """Test pattern tag equality."""
        tag1 = PatternTag.create("uptrend").success()
        tag2 = PatternTag.create("uptrend").success()
        tag3 = PatternTag.create("downtrend").success()
        
        assert tag1 == tag2
        assert tag1 != tag3


# MarketSnapshot Tests

class TestMarketSnapshot:
    """Test MarketSnapshot type functionality."""
    
    def test_market_snapshot_creation(self):
        """Test market snapshot creation."""
        symbol = Symbol.create("BTC-USD").success()
        timestamp = datetime.utcnow()
        price = Decimal("50000")
        indicators = {"rsi": 45.0, "ema_fast": 49900.0}
        dominance_data = {"stablecoin_dominance": 8.5}
        
        snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            indicators=indicators,
            dominance_data=dominance_data,
            position_side="FLAT",
            position_size=Decimal("0"),
        )
        
        assert snapshot.symbol == symbol
        assert snapshot.timestamp == timestamp
        assert snapshot.price == price
        assert snapshot.indicators == indicators
        assert snapshot.dominance_data == dominance_data
        assert snapshot.position_side == "FLAT"
        assert snapshot.position_size == Decimal("0")
    
    def test_market_snapshot_from_market_state(self):
        """Test creating snapshot from market state."""
        # Mock market state
        market_state = MagicMock()
        market_state.symbol = "BTC-USD"
        market_state.timestamp = datetime.utcnow()
        market_state.current_price = Decimal("50000")
        
        # Mock indicators
        market_state.indicators = MagicMock()
        market_state.indicators.rsi = 45.0
        market_state.indicators.cipher_a_dot = 5.0
        market_state.indicators.cipher_b_wave = -10.0
        market_state.indicators.cipher_b_money_flow = 48.0
        market_state.indicators.ema_fast = 49900.0
        market_state.indicators.ema_slow = 49800.0
        
        # Mock dominance data
        market_state.dominance_data = MagicMock()
        market_state.dominance_data.stablecoin_dominance = 8.5
        market_state.dominance_data.dominance_24h_change = -0.5
        market_state.dominance_data.dominance_rsi = 40.0
        
        # Mock position
        market_state.current_position = MagicMock()
        market_state.current_position.side = "FLAT"
        market_state.current_position.size = Decimal("0")
        
        # Create snapshot
        snapshot = MarketSnapshot.from_market_state(market_state)
        
        assert snapshot.symbol.value == "BTC-USD"
        assert snapshot.price == Decimal("50000")
        assert snapshot.indicators["rsi"] == 45.0
        assert snapshot.dominance_data["stablecoin_dominance"] == 8.5
        assert snapshot.position_side == "FLAT"


# TradingExperienceFP Tests

class TestTradingExperienceFP:
    """Test TradingExperienceFP type functionality."""
    
    @pytest.fixture
    def sample_market_snapshot(self):
        """Create sample market snapshot."""
        symbol = Symbol.create("BTC-USD").success()
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price=Decimal("50000"),
            indicators={"rsi": 45.0},
            dominance_data=None,
            position_side="FLAT",
            position_size=Decimal("0"),
        )
    
    @pytest.fixture
    def sample_pattern_tags(self):
        """Create sample pattern tags."""
        return [
            PatternTag.create("uptrend").success(),
            PatternTag.create("oversold").success(),
        ]
    
    def test_trading_experience_create(self, sample_market_snapshot, sample_pattern_tags):
        """Test trading experience creation."""
        experience = TradingExperienceFP.create(
            market_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            decision_rationale="Test rationale",
            pattern_tags=sample_pattern_tags,
        )
        
        assert isinstance(experience.experience_id, ExperienceId)
        assert experience.market_snapshot == sample_market_snapshot
        assert experience.trade_decision == "LONG"
        assert experience.decision_rationale == "Test rationale"
        assert experience.pattern_tags == sample_pattern_tags
        assert experience.outcome.is_nothing()
        assert experience.learned_insights.is_nothing()
        assert experience.confidence_score == Decimal("0.5")
        assert not experience.is_completed()
    
    def test_trading_experience_with_outcome(self, sample_market_snapshot):
        """Test adding outcome to trading experience."""
        experience = TradingExperienceFP.create(
            market_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            decision_rationale="Test rationale",
        )
        
        outcome = TradingOutcome.create(
            pnl=Decimal("100"),
            exit_price=Decimal("51000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        ).success()
        
        # Original experience should be unchanged (immutable)
        experience_with_outcome = experience.with_outcome(outcome)
        
        assert experience.outcome.is_nothing()  # Original unchanged
        assert experience_with_outcome.outcome.is_some()  # New has outcome
        assert experience_with_outcome.outcome.value == outcome
        assert experience_with_outcome.is_completed()
        
        # Confidence should be updated for successful trade
        assert experience_with_outcome.confidence_score > Decimal("0.5")
    
    def test_trading_experience_with_insights(self, sample_market_snapshot):
        """Test adding insights to trading experience."""
        experience = TradingExperienceFP.create(
            market_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            decision_rationale="Test rationale",
        )
        
        insights = "Quick profitable scalp - good timing"
        experience_with_insights = experience.with_insights(insights)
        
        assert experience.learned_insights.is_nothing()  # Original unchanged
        assert experience_with_insights.learned_insights.is_some()  # New has insights
        assert experience_with_insights.learned_insights.value == insights
    
    def test_trading_experience_get_success_rate(self, sample_market_snapshot):
        """Test getting success rate from trading experience."""
        experience = TradingExperienceFP.create(
            market_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            decision_rationale="Test rationale",
        )
        
        # No outcome - should return Nothing
        success_rate = experience.get_success_rate()
        assert success_rate.is_nothing()
        
        # Add successful outcome
        outcome = TradingOutcome.create(
            pnl=Decimal("100"),
            exit_price=Decimal("51000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        ).success()
        
        experience_with_outcome = experience.with_outcome(outcome)
        success_rate = experience_with_outcome.get_success_rate()
        
        assert success_rate.is_some()
        assert success_rate.value is True
    
    def test_confidence_calculation(self, sample_market_snapshot):
        """Test confidence score calculation logic."""
        experience = TradingExperienceFP.create(
            market_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            decision_rationale="Test rationale",
        )
        
        # High profit outcome
        high_profit_outcome = TradingOutcome.create(
            pnl=Decimal("500"),  # High profit
            exit_price=Decimal("55000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        ).success()
        
        high_profit_exp = experience.with_outcome(high_profit_outcome)
        assert high_profit_exp.confidence_score > Decimal("0.6")
        
        # High loss outcome
        high_loss_outcome = TradingOutcome.create(
            pnl=Decimal("-500"),  # High loss
            exit_price=Decimal("45000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        ).success()
        
        high_loss_exp = experience.with_outcome(high_loss_outcome)
        assert high_loss_exp.confidence_score < Decimal("0.4")
        
        # Confidence should be bounded
        assert high_profit_exp.confidence_score <= Decimal("0.9")
        assert high_loss_exp.confidence_score >= Decimal("0.1")


# MemoryQueryFP Tests

class TestMemoryQueryFP:
    """Test MemoryQueryFP type functionality."""
    
    def test_memory_query_create_valid(self):
        """Test valid memory query creation."""
        result = MemoryQueryFP.create(
            current_price=Decimal("50000"),
            indicators={"rsi": 45.0, "ema_fast": 49900.0},
            dominance_data={"stablecoin_dominance": 8.5},
            pattern_tags=[PatternTag.create("uptrend").success()],
            max_results=15,
            min_similarity=0.8,
            time_weight=0.3,
        )
        
        assert result.is_success()
        query = result.success()
        
        assert query.current_price.is_some()
        assert query.current_price.value == Decimal("50000")
        assert query.indicators["rsi"] == 45.0
        assert query.dominance_data.is_some()
        assert query.dominance_data.value["stablecoin_dominance"] == 8.5
        assert len(query.pattern_tags) == 1
        assert query.max_results == 15
        assert query.min_similarity == Decimal("0.8")
        assert query.time_weight == Decimal("0.3")
    
    def test_memory_query_create_minimal(self):
        """Test memory query creation with minimal parameters."""
        result = MemoryQueryFP.create()
        
        assert result.is_success()
        query = result.success()
        
        assert query.current_price.is_nothing()
        assert query.indicators == {}
        assert query.dominance_data.is_nothing()
        assert query.pattern_tags == []
        assert query.max_results == 10  # Default
        assert query.min_similarity == Decimal("0.7")  # Default
        assert query.time_weight == Decimal("0.2")  # Default
    
    def test_memory_query_create_invalid_params(self):
        """Test memory query creation with invalid parameters."""
        # Invalid max_results
        result = MemoryQueryFP.create(max_results=0)
        assert result.is_failure()
        
        result = MemoryQueryFP.create(max_results=100)
        assert result.is_failure()
        
        # Invalid min_similarity
        result = MemoryQueryFP.create(min_similarity=-0.1)
        assert result.is_failure()
        
        result = MemoryQueryFP.create(min_similarity=1.5)
        assert result.is_failure()
        
        # Invalid time_weight
        result = MemoryQueryFP.create(time_weight=-0.1)
        assert result.is_failure()
        
        result = MemoryQueryFP.create(time_weight=1.5)
        assert result.is_failure()


# PatternStatistics Tests

class TestPatternStatistics:
    """Test PatternStatistics type functionality."""
    
    def test_pattern_statistics_calculate_success(self):
        """Test pattern statistics calculation with successful experiences."""
        pattern = PatternTag.create("uptrend").success()
        
        # Create experiences with this pattern
        experiences = []
        for i in range(10):
            exp_id = ExperienceId.create(f"exp_{i}").success()
            
            # 70% success rate
            pnl = Decimal("100") if i < 7 else Decimal("-50")
            outcome = TradingOutcome.create(
                pnl=pnl,
                exit_price=Decimal("51000") if pnl > 0 else Decimal("49000"),
                entry_price=Decimal("50000"),
                duration_minutes=30.0,
            ).success()
            
            experience = TradingExperienceFP(
                experience_id=exp_id,
                timestamp=datetime.utcnow(),
                market_snapshot=MagicMock(),
                trade_decision="LONG",
                decision_rationale="Test",
                pattern_tags=[pattern],
                outcome=Some(outcome),
                learned_insights=Nothing(),
                confidence_score=Decimal("0.5"),
            )
            experiences.append(experience)
        
        # Calculate statistics
        result = PatternStatistics.calculate(pattern, experiences)
        assert result.is_success()
        
        stats = result.success()
        assert stats.pattern == pattern
        assert stats.total_occurrences == 10
        assert stats.successful_trades == 7
        assert stats.success_rate == Decimal("0.7")
        assert stats.total_pnl == Decimal("550")  # 7*100 - 3*50
        assert stats.average_pnl == Decimal("55")
        assert stats.is_profitable()
        assert stats.is_reliable(min_occurrences=5)
    
    def test_pattern_statistics_calculate_empty(self):
        """Test pattern statistics calculation with no experiences."""
        pattern = PatternTag.create("nonexistent").success()
        experiences = []
        
        result = PatternStatistics.calculate(pattern, experiences)
        assert result.is_success()
        
        stats = result.success()
        assert stats.pattern == pattern
        assert stats.total_occurrences == 0
        assert stats.successful_trades == 0
        assert stats.success_rate == Decimal("0")
        assert stats.total_pnl == Decimal("0")
        assert stats.average_pnl == Decimal("0")
        assert not stats.is_profitable()
        assert not stats.is_reliable()
    
    def test_pattern_statistics_calculate_no_outcomes(self):
        """Test pattern statistics calculation with experiences without outcomes."""
        pattern = PatternTag.create("incomplete").success()
        
        # Create experiences without outcomes
        experiences = []
        for i in range(5):
            exp_id = ExperienceId.create(f"exp_{i}").success()
            experience = TradingExperienceFP(
                experience_id=exp_id,
                timestamp=datetime.utcnow(),
                market_snapshot=MagicMock(),
                trade_decision="LONG",
                decision_rationale="Test",
                pattern_tags=[pattern],
                outcome=Nothing(),  # No outcome
                learned_insights=Nothing(),
                confidence_score=Decimal("0.5"),
            )
            experiences.append(experience)
        
        result = PatternStatistics.calculate(pattern, experiences)
        assert result.is_success()
        
        stats = result.success()
        assert stats.total_occurrences == 0  # Only counts completed experiences
        assert not stats.is_reliable()


# LearningInsight Tests

class TestLearningInsight:
    """Test LearningInsight type functionality."""
    
    def test_learning_insight_create_valid(self):
        """Test valid learning insight creation."""
        pattern = PatternTag.create("uptrend").success()
        
        result = LearningInsight.create(
            insight_type="pattern_performance",
            description="Uptrend pattern shows high success rate",
            confidence=0.8,
            supporting_evidence=["10 occurrences", "70% success rate"],
            related_patterns=[pattern],
        )
        
        assert result.is_success()
        insight = result.success()
        
        assert insight.insight_type == "pattern_performance"
        assert insight.description == "Uptrend pattern shows high success rate"
        assert insight.confidence == Decimal("0.8")
        assert len(insight.supporting_evidence) == 2
        assert len(insight.related_patterns) == 1
        assert insight.related_patterns[0] == pattern
    
    def test_learning_insight_create_minimal(self):
        """Test learning insight creation with minimal parameters."""
        result = LearningInsight.create(
            insight_type="timing",
            description="Quick exits tend to be more profitable",
        )
        
        assert result.is_success()
        insight = result.success()
        
        assert insight.insight_type == "timing"
        assert insight.description == "Quick exits tend to be more profitable"
        assert insight.confidence == Decimal("0.5")  # Default
        assert insight.supporting_evidence == []
        assert insight.related_patterns == []
    
    def test_learning_insight_create_invalid(self):
        """Test invalid learning insight creation."""
        # Empty insight type
        result = LearningInsight.create(
            insight_type="",
            description="Valid description",
        )
        assert result.is_failure()
        
        # Empty description
        result = LearningInsight.create(
            insight_type="valid_type",
            description="",
        )
        assert result.is_failure()
        
        # Invalid confidence
        result = LearningInsight.create(
            insight_type="valid_type",
            description="Valid description",
            confidence=1.5,
        )
        assert result.is_failure()
        
        result = LearningInsight.create(
            insight_type="valid_type",
            description="Valid description",
            confidence=-0.1,
        )
        assert result.is_failure()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
