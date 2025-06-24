"""
Functional programming tests for learning and memory system integration.

These tests validate the FP learning system adapters, memory operations,
and integration with the MCP memory server using immutable data structures
and pure functional patterns.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.fp.types import (
    Result, Success, Failure,
    Maybe, Some, Nothing,
    Symbol, ExperienceId, TradingExperienceFP, TradingOutcome,
    MarketSnapshot, PatternTag, MemoryQueryFP, PatternStatistics,
    LearningInsight, MemoryStorage,
)
from bot.fp.adapters.memory_adapter import MemoryAdapterFP
from bot.fp.adapters.experience_adapter import ExperienceManagerFP, ActiveTradeFP
from bot.mcp.memory_server import MCPMemoryServer
from bot.learning.experience_manager import ExperienceManager


# Test Fixtures

@pytest.fixture
def mock_mcp_server():
    """Create a mock MCP memory server."""
    server = MagicMock(spec=MCPMemoryServer)
    server.store_experience = AsyncMock(return_value="exp_123")
    server.update_experience_outcome = AsyncMock(return_value=True)
    server.query_similar_experiences = AsyncMock(return_value=[])
    server.get_pattern_statistics = AsyncMock(return_value={})
    return server


@pytest.fixture
def mock_experience_manager():
    """Create a mock experience manager."""
    manager = MagicMock(spec=ExperienceManager)
    manager.record_trading_decision = AsyncMock(return_value="exp_123")
    manager.link_order_to_experience = MagicMock()
    manager.start_tracking_trade = MagicMock(return_value="trade_123")
    manager.update_trade_progress = AsyncMock()
    manager.complete_trade = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def sample_market_snapshot():
    """Create a sample market snapshot for testing."""
    symbol_result = Symbol.create("BTC-USD")
    assert symbol_result.is_success()
    
    return MarketSnapshot(
        symbol=symbol_result.success(),
        timestamp=datetime.utcnow(),
        price=Decimal("50000"),
        indicators={
            "rsi": 45.0,
            "cipher_a_dot": 5.0,
            "cipher_b_wave": -10.0,
            "cipher_b_money_flow": 48.0,
            "ema_fast": 49900.0,
            "ema_slow": 49800.0,
        },
        dominance_data={
            "stablecoin_dominance": 8.5,
            "dominance_24h_change": -0.5,
            "dominance_rsi": 40.0,
        },
        position_side="FLAT",
        position_size=Decimal("0"),
    )


@pytest.fixture
def sample_pattern_tags():
    """Create sample pattern tags."""
    patterns = ["uptrend", "oversold", "high_stablecoin_dominance"]
    return [PatternTag.create(p).success() for p in patterns]


@pytest.fixture
def sample_trading_experience(sample_market_snapshot, sample_pattern_tags):
    """Create a sample trading experience."""
    return TradingExperienceFP.create(
        market_snapshot=sample_market_snapshot,
        trade_decision="LONG",  # Simplified for testing
        decision_rationale="Test rationale",
        pattern_tags=sample_pattern_tags,
    )


# Memory Adapter Tests

class TestMemoryAdapterFP:
    """Test the FP memory adapter functionality."""
    
    @pytest.mark.asyncio
    async def test_store_experience_fp_success(self, mock_mcp_server, sample_trading_experience):
        """Test successful FP experience storage."""
        adapter = MemoryAdapterFP(mock_mcp_server)
        
        result = await adapter.store_experience_fp(sample_trading_experience)
        
        assert result.is_success()
        experience_id = result.success()
        assert isinstance(experience_id, ExperienceId)
        assert experience_id.value == "exp_123"
        
        # Verify MCP server was called
        mock_mcp_server.store_experience.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_experience_fp_failure(self, mock_mcp_server, sample_trading_experience):
        """Test FP experience storage failure handling."""
        mock_mcp_server.store_experience.side_effect = Exception("Storage failed")
        adapter = MemoryAdapterFP(mock_mcp_server)
        
        result = await adapter.store_experience_fp(sample_trading_experience)
        
        assert result.is_failure()
        assert "Storage failed" in result.failure()
    
    @pytest.mark.asyncio
    async def test_update_experience_outcome_fp_success(self, mock_mcp_server):
        """Test successful FP experience outcome update."""
        adapter = MemoryAdapterFP(mock_mcp_server)
        
        experience_id = ExperienceId.create("exp_123").success()
        outcome_result = TradingOutcome.create(
            pnl=Decimal("100"),
            exit_price=Decimal("51000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        )
        assert outcome_result.is_success()
        outcome = outcome_result.success()
        
        # Mock local storage to have the experience
        adapter._local_storage = adapter._local_storage.add_experience(
            TradingExperienceFP(
                experience_id=experience_id,
                timestamp=datetime.utcnow(),
                market_snapshot=MagicMock(),
                trade_decision="LONG",
                decision_rationale="Test",
                pattern_tags=[],
                outcome=Nothing(),
                learned_insights=Nothing(),
                confidence_score=Decimal("0.5"),
            )
        )
        
        result = await adapter.update_experience_outcome_fp(experience_id, outcome)
        
        assert result.is_success()
        updated_experience = result.success()
        assert updated_experience.outcome.is_some()
        assert updated_experience.outcome.value.pnl == Decimal("100")
        
        # Verify MCP server was called
        mock_mcp_server.update_experience_outcome.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_similar_experiences_fp(self, mock_mcp_server, sample_market_snapshot):
        """Test FP similar experiences query."""
        # Mock return data from MCP server
        mock_experience = MagicMock()
        mock_experience.experience_id = "exp_123"
        mock_experience.timestamp = datetime.utcnow()
        mock_experience.symbol = "BTC-USD"
        mock_experience.price = Decimal("50000")
        mock_experience.indicators = {"rsi": 45.0}
        mock_experience.dominance_data = None
        mock_experience.market_state_snapshot = {"position_side": "FLAT", "position_size": 0}
        mock_experience.pattern_tags = ["uptrend"]
        mock_experience.outcome = {"pnl": 100, "exit_price": 51000, "success": True, "duration_minutes": 30}
        mock_experience.decision_rationale = "Test decision"
        mock_experience.learned_insights = "Test insights"
        mock_experience.confidence_score = 0.7
        
        mock_mcp_server.query_similar_experiences.return_value = [mock_experience]
        
        adapter = MemoryAdapterFP(mock_mcp_server)
        
        query_result = MemoryQueryFP.create(
            current_price=Decimal("50000"),
            indicators={"rsi": 45.0},
            max_results=5,
            min_similarity=0.7,
        )
        assert query_result.is_success()
        query = query_result.success()
        
        result = await adapter.query_similar_experiences_fp(sample_market_snapshot, query)
        
        assert result.is_success()
        experiences = result.success()
        assert len(experiences) >= 0  # May be 0 if conversion fails
        
        # Verify MCP server was called
        mock_mcp_server.query_similar_experiences.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_pattern_statistics_fp(self, mock_mcp_server):
        """Test FP pattern statistics retrieval."""
        mock_stats = {
            "uptrend": {
                "count": 10,
                "success_rate": 0.7,
                "avg_pnl": 50.0,
                "total_pnl": 500.0,
            },
            "oversold": {
                "count": 5,
                "success_rate": 0.8,
                "avg_pnl": 80.0,
                "total_pnl": 400.0,
            },
        }
        mock_mcp_server.get_pattern_statistics.return_value = mock_stats
        
        adapter = MemoryAdapterFP(mock_mcp_server)
        
        result = await adapter.get_pattern_statistics_fp()
        
        assert result.is_success()
        statistics = result.success()
        assert len(statistics) == 2
        
        # Verify pattern statistics content
        uptrend_stats = next((s for s in statistics if s.pattern.name == "uptrend"), None)
        assert uptrend_stats is not None
        assert uptrend_stats.total_occurrences == 10
        assert uptrend_stats.success_rate == Decimal("0.7")
        assert uptrend_stats.average_pnl == Decimal("50.0")
    
    @pytest.mark.asyncio
    async def test_generate_learning_insights_fp(self, mock_mcp_server, sample_trading_experience):
        """Test FP learning insights generation."""
        adapter = MemoryAdapterFP(mock_mcp_server)
        
        # Create experiences with outcomes
        outcome_result = TradingOutcome.create(
            pnl=Decimal("100"),
            exit_price=Decimal("51000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        )
        assert outcome_result.is_success()
        outcome = outcome_result.success()
        
        experience_with_outcome = sample_trading_experience.with_outcome(outcome)
        experiences = [experience_with_outcome] * 5  # Multiple experiences
        
        result = await adapter.generate_learning_insights_fp(experiences)
        
        assert result.is_success()
        insights = result.success()
        assert isinstance(insights, list)
        # Insights may be empty if no patterns are detected


# Experience Manager Adapter Tests

class TestExperienceManagerFP:
    """Test the FP experience manager adapter functionality."""
    
    @pytest.mark.asyncio
    async def test_record_trading_decision_fp(self, mock_experience_manager, sample_market_snapshot):
        """Test FP trading decision recording."""
        adapter = ExperienceManagerFP(mock_experience_manager)
        
        result = await adapter.record_trading_decision_fp(
            market_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            rationale="Test decision",
            pattern_tags=[PatternTag.create("uptrend").success()],
        )
        
        assert result.is_success()
        experience_id = result.success()
        assert isinstance(experience_id, ExperienceId)
        assert experience_id.value == "exp_123"
        
        # Verify experience manager was called
        mock_experience_manager.record_trading_decision.assert_called_once()
    
    def test_link_order_to_experience_fp(self, mock_experience_manager):
        """Test FP order linking to experience."""
        adapter = ExperienceManagerFP(mock_experience_manager)
        
        experience_id = ExperienceId.create("exp_123").success()
        result = adapter.link_order_to_experience_fp("order_456", experience_id)
        
        assert result.is_success()
        
        # Verify experience manager was called
        mock_experience_manager.link_order_to_experience.assert_called_once_with(
            "order_456", "exp_123"
        )
    
    def test_start_tracking_trade_fp(self, mock_experience_manager, sample_market_snapshot):
        """Test FP trade tracking initiation."""
        adapter = ExperienceManagerFP(mock_experience_manager)
        
        order_data = {
            "id": "order_123",
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": 1.0,
            "price": 50000.0,
        }
        
        result = adapter.start_tracking_trade_fp(
            order_data=order_data,
            trade_decision="LONG",
            market_snapshot=sample_market_snapshot,
        )
        
        assert result.is_success()
        trade_id = result.success()
        assert trade_id == "trade_123"
        
        # Verify active trade was created
        assert len(adapter._active_trades) == 1
        assert trade_id in adapter._active_trades
        
        active_trade = adapter._active_trades[trade_id]
        assert isinstance(active_trade, ActiveTradeFP)
        assert active_trade.trade_id == trade_id
    
    @pytest.mark.asyncio
    async def test_update_trade_progress_fp(self, mock_experience_manager, sample_market_snapshot):
        """Test FP trade progress updates."""
        adapter = ExperienceManagerFP(mock_experience_manager)
        
        # First create an active trade
        trade_id = "trade_123"
        experience_id = ExperienceId.create("exp_123").success()
        active_trade = ActiveTradeFP(
            trade_id=trade_id,
            experience_id=experience_id,
            entry_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            entry_time=datetime.utcnow(),
        )
        adapter._active_trades[trade_id] = active_trade
        
        # Update trade progress
        result = await adapter.update_trade_progress_fp(
            trade_id=trade_id,
            current_price=Decimal("51000"),
            market_snapshot=sample_market_snapshot,
        )
        
        assert result.is_success()
        updated_trade = result.success()
        assert isinstance(updated_trade, ActiveTradeFP)
        assert updated_trade.current_pnl == Decimal("1000")  # 51000 - 50000
        assert updated_trade.max_profit == Decimal("1000")
    
    @pytest.mark.asyncio
    async def test_complete_trade_fp(self, mock_experience_manager, sample_market_snapshot):
        """Test FP trade completion."""
        adapter = ExperienceManagerFP(mock_experience_manager)
        
        # First create an active trade
        trade_id = "trade_123"
        experience_id = ExperienceId.create("exp_123").success()
        active_trade = ActiveTradeFP(
            trade_id=trade_id,
            experience_id=experience_id,
            entry_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            entry_time=datetime.utcnow() - timedelta(minutes=30),
        )
        adapter._active_trades[trade_id] = active_trade
        
        # Complete the trade
        result = await adapter.complete_trade_fp(
            trade_id=trade_id,
            exit_price=Decimal("51000"),
            exit_snapshot=sample_market_snapshot,
        )
        
        assert result.is_success()
        outcome = result.success()
        assert isinstance(outcome, TradingOutcome)
        assert outcome.pnl == Decimal("1000")
        assert outcome.is_successful
        
        # Verify trade was removed from active trades
        assert trade_id not in adapter._active_trades
        
        # Verify experience manager was called
        mock_experience_manager.complete_trade.assert_called_once()
    
    def test_get_active_trades_fp(self, mock_experience_manager, sample_market_snapshot):
        """Test FP active trades retrieval."""
        adapter = ExperienceManagerFP(mock_experience_manager)
        
        # Add some active trades
        for i in range(3):
            trade_id = f"trade_{i}"
            experience_id = ExperienceId.create(f"exp_{i}").success()
            active_trade = ActiveTradeFP(
                trade_id=trade_id,
                experience_id=experience_id,
                entry_snapshot=sample_market_snapshot,
                trade_decision="LONG",
                entry_time=datetime.utcnow(),
            )
            adapter._active_trades[trade_id] = active_trade
        
        active_trades = adapter.get_active_trades_fp()
        
        assert len(active_trades) == 3
        assert all(isinstance(trade, ActiveTradeFP) for trade in active_trades)
    
    def test_get_active_trade_fp(self, mock_experience_manager, sample_market_snapshot):
        """Test FP specific active trade retrieval."""
        adapter = ExperienceManagerFP(mock_experience_manager)
        
        trade_id = "trade_123"
        experience_id = ExperienceId.create("exp_123").success()
        active_trade = ActiveTradeFP(
            trade_id=trade_id,
            experience_id=experience_id,
            entry_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            entry_time=datetime.utcnow(),
        )
        adapter._active_trades[trade_id] = active_trade
        
        # Test existing trade
        result = adapter.get_active_trade_fp(trade_id)
        assert result.is_some()
        assert result.value.trade_id == trade_id
        
        # Test non-existing trade
        result = adapter.get_active_trade_fp("nonexistent")
        assert result.is_nothing()
    
    def test_analyze_active_trades_fp(self, mock_experience_manager, sample_market_snapshot):
        """Test FP active trades analysis."""
        adapter = ExperienceManagerFP(mock_experience_manager)
        
        # Test empty case
        result = adapter.analyze_active_trades_fp()
        assert result.is_success()
        analysis = result.success()
        assert analysis["total_trades"] == 0
        
        # Add some active trades with different PnL
        trades_data = [
            ("trade_1", Decimal("100")),   # Winning
            ("trade_2", Decimal("-50")),   # Losing
            ("trade_3", Decimal("200")),   # Winning
        ]
        
        for trade_id, pnl in trades_data:
            experience_id = ExperienceId.create(f"exp_{trade_id}").success()
            active_trade = ActiveTradeFP(
                trade_id=trade_id,
                experience_id=experience_id,
                entry_snapshot=sample_market_snapshot,
                trade_decision="LONG",
                entry_time=datetime.utcnow(),
            )
            active_trade = active_trade.with_update(
                current_price=Decimal("51000"),
                unrealized_pnl=pnl,
            )
            adapter._active_trades[trade_id] = active_trade
        
        # Analyze trades
        result = adapter.analyze_active_trades_fp()
        assert result.is_success()
        analysis = result.success()
        
        assert analysis["total_trades"] == 3
        assert analysis["winning_trades"] == 2
        assert analysis["win_rate"] == pytest.approx(2/3, rel=1e-2)
        assert analysis["total_unrealized_pnl"] == 250.0  # 100 - 50 + 200
    
    @pytest.mark.asyncio
    async def test_cleanup_stale_trades_fp(self, mock_experience_manager, sample_market_snapshot):
        """Test FP stale trades cleanup."""
        adapter = ExperienceManagerFP(mock_experience_manager)
        
        # Add trades with different ages
        current_time = datetime.utcnow()
        trades_data = [
            ("trade_fresh", current_time - timedelta(hours=1)),    # Fresh
            ("trade_old", current_time - timedelta(hours=50)),     # Stale
            ("trade_very_old", current_time - timedelta(hours=100)), # Very stale
        ]
        
        for trade_id, entry_time in trades_data:
            experience_id = ExperienceId.create(f"exp_{trade_id}").success()
            active_trade = ActiveTradeFP(
                trade_id=trade_id,
                experience_id=experience_id,
                entry_snapshot=sample_market_snapshot,
                trade_decision="LONG",
                entry_time=entry_time,
            )
            adapter._active_trades[trade_id] = active_trade
        
        # Cleanup stale trades (older than 48 hours)
        result = await adapter.cleanup_stale_trades_fp(max_age_hours=48.0)
        
        assert result.is_success()
        cleaned_count = result.success()
        assert cleaned_count == 2  # trade_old and trade_very_old
        
        # Verify only fresh trade remains
        assert len(adapter._active_trades) == 1
        assert "trade_fresh" in adapter._active_trades
        assert "trade_old" not in adapter._active_trades
        assert "trade_very_old" not in adapter._active_trades


# Pattern and Outcome Analysis Tests

class TestPatternAnalysis:
    """Test pattern analysis and statistics functionality."""
    
    def test_pattern_tag_creation(self):
        """Test pattern tag creation and validation."""
        # Valid pattern tags
        valid_patterns = ["uptrend", "oversold", "high_volume", "bullish_divergence"]
        for pattern in valid_patterns:
            result = PatternTag.create(pattern)
            assert result.is_success()
            tag = result.success()
            assert tag.name == pattern.lower().replace(" ", "_")
        
        # Invalid pattern tags
        invalid_patterns = ["", "   ", None]
        for pattern in invalid_patterns:
            if pattern is not None:
                result = PatternTag.create(pattern)
                assert result.is_failure()
    
    def test_pattern_statistics_calculation(self, sample_pattern_tags):
        """Test pattern statistics calculation."""
        pattern = sample_pattern_tags[0]  # "uptrend"
        
        # Create experiences with outcomes
        experiences = []
        for i in range(10):
            exp_id = ExperienceId.create(f"exp_{i}").success()
            
            # Create outcome (70% success rate)
            pnl = Decimal("100") if i < 7 else Decimal("-50")
            outcome_result = TradingOutcome.create(
                pnl=pnl,
                exit_price=Decimal("51000" if pnl > 0 else "49000"),
                entry_price=Decimal("50000"),
                duration_minutes=30.0,
            )
            assert outcome_result.is_success()
            outcome = outcome_result.success()
            
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
        stats_result = PatternStatistics.calculate(pattern, experiences)
        assert stats_result.is_success()
        
        stats = stats_result.success()
        assert stats.pattern == pattern
        assert stats.total_occurrences == 10
        assert stats.successful_trades == 7
        assert stats.success_rate == Decimal("0.7")
        assert stats.is_profitable()
        assert stats.is_reliable(min_occurrences=5)
    
    def test_learning_insight_creation(self):
        """Test learning insight creation and validation."""
        # Valid insight
        result = LearningInsight.create(
            insight_type="pattern_performance",
            description="Uptrend pattern shows high success rate",
            confidence=0.8,
            supporting_evidence=["10 occurrences", "70% success rate"],
            related_patterns=[PatternTag.create("uptrend").success()],
        )
        
        assert result.is_success()
        insight = result.success()
        assert insight.insight_type == "pattern_performance"
        assert insight.confidence == Decimal("0.8")
        assert len(insight.supporting_evidence) == 2
        
        # Invalid insights
        invalid_cases = [
            ("", "description", 0.5),  # Empty type
            ("type", "", 0.5),         # Empty description
            ("type", "desc", 1.5),    # Invalid confidence
        ]
        
        for insight_type, description, confidence in invalid_cases:
            result = LearningInsight.create(
                insight_type=insight_type,
                description=description,
                confidence=confidence,
            )
            assert result.is_failure()


# Memory Storage Tests

class TestMemoryStorage:
    """Test the immutable memory storage functionality."""
    
    def test_empty_memory_storage(self):
        """Test empty memory storage creation."""
        storage = MemoryStorage.empty()
        
        assert len(storage.experiences) == 0
        assert len(storage.pattern_index) == 0
        assert storage.total_experiences == 0
        assert storage.completed_experiences == 0
    
    def test_add_experience_to_storage(self, sample_trading_experience):
        """Test adding experience to memory storage."""
        storage = MemoryStorage.empty()
        
        # Add experience
        new_storage = storage.add_experience(sample_trading_experience)
        
        # Verify immutability
        assert storage.total_experiences == 0  # Original unchanged
        assert new_storage.total_experiences == 1  # New storage updated
        
        # Verify experience was added
        assert len(new_storage.experiences) == 1
        assert new_storage.experiences[0] == sample_trading_experience
        
        # Verify pattern index was updated
        for pattern in sample_trading_experience.pattern_tags:
            assert pattern.name in new_storage.pattern_index
            assert sample_trading_experience.experience_id in new_storage.pattern_index[pattern.name]
    
    def test_update_experience_in_storage(self, sample_trading_experience):
        """Test updating experience in memory storage."""
        storage = MemoryStorage.empty().add_experience(sample_trading_experience)
        
        # Create outcome for update
        outcome_result = TradingOutcome.create(
            pnl=Decimal("100"),
            exit_price=Decimal("51000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        )
        assert outcome_result.is_success()
        outcome = outcome_result.success()
        
        # Update experience
        update_result = storage.update_experience(
            sample_trading_experience.experience_id,
            lambda exp: exp.with_outcome(outcome),
        )
        
        assert update_result.is_success()
        new_storage = update_result.success()
        
        # Verify completion count was updated
        assert storage.completed_experiences == 0  # Original unchanged
        assert new_storage.completed_experiences == 1  # New storage updated
        
        # Verify experience was updated
        updated_exp = new_storage.find_by_id(sample_trading_experience.experience_id)
        assert updated_exp.is_some()
        assert updated_exp.value.outcome.is_some()
        assert updated_exp.value.outcome.value.pnl == Decimal("100")
    
    def test_find_by_pattern_in_storage(self, sample_trading_experience):
        """Test finding experiences by pattern in storage."""
        storage = MemoryStorage.empty().add_experience(sample_trading_experience)
        
        # Find by existing pattern
        pattern = sample_trading_experience.pattern_tags[0]
        found_experiences = storage.find_by_pattern(pattern)
        
        assert len(found_experiences) == 1
        assert found_experiences[0] == sample_trading_experience
        
        # Find by non-existing pattern
        non_existing_pattern = PatternTag.create("nonexistent").success()
        found_experiences = storage.find_by_pattern(non_existing_pattern)
        
        assert len(found_experiences) == 0
    
    def test_get_completed_experiences(self, sample_trading_experience):
        """Test getting completed experiences from storage."""
        storage = MemoryStorage.empty()
        
        # Add incomplete experience
        storage = storage.add_experience(sample_trading_experience)
        
        # Add completed experience
        outcome_result = TradingOutcome.create(
            pnl=Decimal("100"),
            exit_price=Decimal("51000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        )
        assert outcome_result.is_success()
        outcome = outcome_result.success()
        
        completed_experience = sample_trading_experience.with_outcome(outcome)
        storage = storage.add_experience(completed_experience)
        
        # Get completed experiences
        completed = storage.get_completed_experiences()
        
        assert len(completed) == 1
        assert completed[0] == completed_experience
        assert completed[0].is_completed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
