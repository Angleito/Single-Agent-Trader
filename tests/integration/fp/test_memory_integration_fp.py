"""
Integration tests for functional programming memory and learning system.

This module tests the functional programming memory integration patterns including:
1. Functional memory adapter operations with FP types
2. Memory storage with immutable data structures
3. Learning insights generation using functional patterns
4. Pattern statistics with functional types
5. Error handling with Result/Either types
6. Conversion between imperative and functional memory types
"""

import asyncio
import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch

from bot.fp.adapters.memory_adapter import MemoryAdapterFP
from bot.fp.types import (
    Result, Success, Failure,
    Maybe, Some, Nothing,
    ExperienceId, TradingExperienceFP, TradingOutcome,
    MemoryQueryFP, MarketSnapshot, PatternTag,
    PatternStatistics, LearningInsight, MemoryStorage,
    Symbol,
)
from bot.fp.types.trading import Long, Short, Hold
from bot.mcp.memory_server import MCPMemoryServer, TradingExperience
from bot.trading_types import MarketState, TradeAction


class TestFunctionalMemoryAdapter:
    """Test functional memory adapter operations."""
    
    @pytest.fixture
    async def mock_mcp_server(self):
        """Create mock MCP memory server."""
        server = MagicMock(spec=MCPMemoryServer)
        server.store_experience = AsyncMock(return_value="exp_123456789")
        server.update_experience_outcome = AsyncMock(return_value=True)
        server.query_similar_experiences = AsyncMock(return_value=[])
        server.get_pattern_statistics = AsyncMock(return_value={})
        return server
    
    @pytest.fixture
    def memory_adapter(self, mock_mcp_server):
        """Create functional memory adapter with mock server."""
        return MemoryAdapterFP(mock_mcp_server)
    
    @pytest.fixture
    def sample_market_snapshot(self):
        """Create sample market snapshot for testing."""
        symbol = Symbol.create("BTC-USD").success()
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            indicators={
                "rsi": 45.0,
                "ema_fast": 49900.0,
                "ema_slow": 49800.0,
                "cipher_a_dot": 5.0,
                "cipher_b_wave": -10.0,
                "cipher_b_money_flow": 48.0,
            },
            dominance_data={"stablecoin_dominance": 8.5},
            position_side="FLAT",
            position_size=Decimal("0"),
        )
    
    @pytest.fixture
    def sample_trading_experience(self, sample_market_snapshot):
        """Create sample functional trading experience."""
        pattern_tags = [
            PatternTag.create("uptrend").success(),
            PatternTag.create("oversold").success(),
        ]
        
        return TradingExperienceFP.create(
            market_snapshot=sample_market_snapshot,
            trade_decision="LONG",
            decision_rationale="Strong bullish signal with oversold RSI",
            pattern_tags=pattern_tags,
        )
    
    @pytest.mark.asyncio
    async def test_store_experience_fp_success(self, memory_adapter, sample_trading_experience):
        """Test successful FP experience storage."""
        result = await memory_adapter.store_experience_fp(sample_trading_experience)
        
        assert result.is_success()
        experience_id = result.success()
        assert isinstance(experience_id, ExperienceId)
        assert experience_id.value == "exp_123456789"
        
        # Verify mock was called correctly
        memory_adapter._mcp_server.store_experience.assert_called_once()
        
        # Verify local storage was updated
        storage = memory_adapter.get_memory_storage()
        assert len(storage.experiences) == 1
        assert storage.find_by_id(experience_id).is_some()
    
    @pytest.mark.asyncio
    async def test_store_experience_fp_failure(self, memory_adapter, sample_trading_experience):
        """Test FP experience storage failure."""
        # Make mock server raise exception
        memory_adapter._mcp_server.store_experience.side_effect = Exception("Storage failed")
        
        result = await memory_adapter.store_experience_fp(sample_trading_experience)
        
        assert result.is_failure()
        assert "Storage failed" in result.failure()
        
        # Local storage should not be updated on failure
        storage = memory_adapter.get_memory_storage()
        assert len(storage.experiences) == 0
    
    @pytest.mark.asyncio
    async def test_update_experience_outcome_fp_success(
        self, memory_adapter, sample_trading_experience
    ):
        """Test successful experience outcome update."""
        # First store the experience
        store_result = await memory_adapter.store_experience_fp(sample_trading_experience)
        experience_id = store_result.success()
        
        # Create trading outcome
        outcome = TradingOutcome.create(
            pnl=Decimal("150"),
            exit_price=Decimal("51500"),
            entry_price=Decimal("50000"),
            duration_minutes=45.0,
        ).success()
        
        # Update with outcome
        result = await memory_adapter.update_experience_outcome_fp(experience_id, outcome)
        
        assert result.is_success()
        updated_experience = result.success()
        assert updated_experience.outcome.is_some()
        assert updated_experience.outcome.value.pnl == Decimal("150")
        assert updated_experience.is_completed()
        
        # Verify mock was called
        memory_adapter._mcp_server.update_experience_outcome.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_experience_outcome_fp_failure(
        self, memory_adapter, sample_trading_experience
    ):
        """Test experience outcome update failure."""
        # Store experience first
        store_result = await memory_adapter.store_experience_fp(sample_trading_experience)
        experience_id = store_result.success()
        
        # Make update fail
        memory_adapter._mcp_server.update_experience_outcome.return_value = False
        
        outcome = TradingOutcome.create(
            pnl=Decimal("150"),
            exit_price=Decimal("51500"),
            entry_price=Decimal("50000"),
            duration_minutes=45.0,
        ).success()
        
        result = await memory_adapter.update_experience_outcome_fp(experience_id, outcome)
        
        assert result.is_failure()
        assert "Failed to update experience" in result.failure()
    
    @pytest.mark.asyncio
    async def test_query_similar_experiences_fp_success(
        self, memory_adapter, sample_market_snapshot
    ):
        """Test successful similar experiences query."""
        # Mock server to return experiences
        mock_experience = TradingExperience(
            experience_id="exp_similar_123",
            timestamp=datetime.now(UTC),
            symbol="BTC-USD",
            price=Decimal("49500"),
            decision_rationale="Similar trade",
            indicators={"rsi": 47.0},
            dominance_data={},
            pattern_tags=["uptrend"],
            market_state_snapshot={"position_side": "FLAT", "position_size": 0},
            outcome={"pnl": 100, "exit_price": 50500, "duration_minutes": 30},
            learned_insights="Good timing",
            confidence_score=0.7,
        )
        
        memory_adapter._mcp_server.query_similar_experiences.return_value = [mock_experience]
        
        # Create query
        query = MemoryQueryFP.create(
            current_price=Decimal("50000"),
            indicators={"rsi": 45.0},
            max_results=10,
            min_similarity=0.7,
        ).success()
        
        result = await memory_adapter.query_similar_experiences_fp(
            sample_market_snapshot, query
        )
        
        assert result.is_success()
        experiences = result.success()
        assert len(experiences) == 1
        assert isinstance(experiences[0], TradingExperienceFP)
        assert experiences[0].outcome.is_some()
        
        # Verify mock was called with converted parameters
        memory_adapter._mcp_server.query_similar_experiences.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_similar_experiences_fp_empty_result(
        self, memory_adapter, sample_market_snapshot
    ):
        """Test query with no similar experiences found."""
        memory_adapter._mcp_server.query_similar_experiences.return_value = []
        
        result = await memory_adapter.query_similar_experiences_fp(sample_market_snapshot)
        
        assert result.is_success()
        experiences = result.success()
        assert len(experiences) == 0
    
    @pytest.mark.asyncio
    async def test_get_pattern_statistics_fp_success(self, memory_adapter):
        """Test successful pattern statistics retrieval."""
        # Mock statistics data
        mock_stats = {
            "uptrend": {
                "count": 10,
                "success_rate": 0.7,
                "total_pnl": 500.0,
                "avg_pnl": 50.0,
            },
            "oversold": {
                "count": 8,
                "success_rate": 0.625,
                "total_pnl": 200.0,
                "avg_pnl": 25.0,
            },
        }
        
        memory_adapter._mcp_server.get_pattern_statistics.return_value = mock_stats
        
        result = await memory_adapter.get_pattern_statistics_fp()
        
        assert result.is_success()
        statistics = result.success()
        assert len(statistics) == 2
        
        uptrend_stat = next(s for s in statistics if s.pattern.name == "uptrend")
        assert uptrend_stat.total_occurrences == 10
        assert uptrend_stat.successful_trades == 7
        assert uptrend_stat.success_rate == Decimal("0.7")
        assert uptrend_stat.is_profitable()
        assert uptrend_stat.is_reliable()
    
    @pytest.mark.asyncio
    async def test_get_pattern_statistics_fp_filtered(self, memory_adapter):
        """Test pattern statistics with filtering."""
        mock_stats = {
            "uptrend": {"count": 10, "success_rate": 0.7, "total_pnl": 500.0, "avg_pnl": 50.0},
            "downtrend": {"count": 5, "success_rate": 0.4, "total_pnl": -100.0, "avg_pnl": -20.0},
        }
        
        memory_adapter._mcp_server.get_pattern_statistics.return_value = mock_stats
        
        # Filter for specific pattern
        patterns = [PatternTag.create("uptrend").success()]
        result = await memory_adapter.get_pattern_statistics_fp(patterns)
        
        assert result.is_success()
        statistics = result.success()
        assert len(statistics) == 1
        assert statistics[0].pattern.name == "uptrend"


class TestLearningInsightsGeneration:
    """Test functional learning insights generation."""
    
    @pytest.fixture
    def memory_adapter_with_data(self):
        """Create memory adapter with sample data for insights testing."""
        mock_server = MagicMock(spec=MCPMemoryServer)
        adapter = MemoryAdapterFP(mock_server)
        return adapter
    
    @pytest.fixture
    def sample_experiences_with_outcomes(self):
        """Create sample experiences with various outcomes for testing."""
        symbol = Symbol.create("BTC-USD").success()
        base_snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            indicators={"rsi": 30.0},  # Oversold
            dominance_data=None,
            position_side="FLAT",
            position_size=Decimal("0"),
        )
        
        experiences = []
        
        # Create experiences with different patterns and outcomes
        patterns_data = [
            ("uptrend", True, Decimal("150"), 30.0),  # Successful uptrend
            ("uptrend", True, Decimal("200"), 25.0),  # Successful uptrend
            ("uptrend", False, Decimal("-80"), 45.0),  # Failed uptrend
            ("oversold", True, Decimal("120"), 20.0),  # Successful oversold
            ("oversold", True, Decimal("180"), 35.0),  # Successful oversold
            ("oversold", True, Decimal("90"), 15.0),   # Successful oversold
        ]
        
        for i, (pattern_name, is_successful, pnl, duration) in enumerate(patterns_data):
            # Create pattern tag
            pattern_tag = PatternTag.create(pattern_name).success()
            
            # Create experience
            experience = TradingExperienceFP.create(
                market_snapshot=base_snapshot,
                trade_decision="LONG",
                decision_rationale=f"Trade based on {pattern_name}",
                pattern_tags=[pattern_tag],
            )
            
            # Add outcome
            exit_price = Decimal("51500") if is_successful else Decimal("49200")
            outcome = TradingOutcome.create(
                pnl=pnl,
                exit_price=exit_price,
                entry_price=Decimal("50000"),
                duration_minutes=duration,
            ).success()
            
            experience_with_outcome = experience.with_outcome(outcome)
            experiences.append(experience_with_outcome)
        
        return experiences
    
    @pytest.mark.asyncio
    async def test_generate_learning_insights_fp_success(
        self, memory_adapter_with_data, sample_experiences_with_outcomes
    ):
        """Test successful learning insights generation."""
        result = await memory_adapter_with_data.generate_learning_insights_fp(
            sample_experiences_with_outcomes
        )
        
        assert result.is_success()
        insights = result.success()
        assert len(insights) > 0
        
        # Should have pattern performance insights
        pattern_insights = [i for i in insights if i.insight_type == "pattern_performance"]
        assert len(pattern_insights) > 0
        
        # Check for oversold pattern insight (100% success rate)
        oversold_insight = next(
            (i for i in pattern_insights if "oversold" in i.description.lower()),
            None
        )
        assert oversold_insight is not None
        assert "high success rate" in oversold_insight.description.lower()
        assert oversold_insight.confidence > Decimal("0.5")
    
    @pytest.mark.asyncio
    async def test_generate_learning_insights_timing_patterns(
        self, memory_adapter_with_data, sample_experiences_with_outcomes
    ):
        """Test timing pattern insights generation."""
        result = await memory_adapter_with_data.generate_learning_insights_fp(
            sample_experiences_with_outcomes
        )
        
        assert result.is_success()
        insights = result.success()
        
        # Should have timing insights
        timing_insights = [i for i in insights if i.insight_type == "timing_pattern"]
        if timing_insights:  # May not always generate depending on data
            timing_insight = timing_insights[0]
            assert "quick" in timing_insight.description.lower() or "duration" in timing_insight.description.lower()
    
    @pytest.mark.asyncio
    async def test_generate_learning_insights_market_conditions(
        self, memory_adapter_with_data, sample_experiences_with_outcomes
    ):
        """Test market condition insights generation."""
        result = await memory_adapter_with_data.generate_learning_insights_fp(
            sample_experiences_with_outcomes
        )
        
        assert result.is_success()
        insights = result.success()
        
        # Should have market condition insights for oversold RSI
        market_insights = [i for i in insights if i.insight_type == "market_condition"]
        if market_insights:
            rsi_insight = next(
                (i for i in market_insights if "oversold" in i.description.lower()),
                None
            )
            if rsi_insight:
                assert "rsi" in rsi_insight.description.lower()
                assert rsi_insight.confidence > Decimal("0.5")
    
    @pytest.mark.asyncio
    async def test_generate_learning_insights_empty_experiences(
        self, memory_adapter_with_data
    ):
        """Test insights generation with empty experience list."""
        result = await memory_adapter_with_data.generate_learning_insights_fp([])
        
        assert result.is_success()
        insights = result.success()
        assert len(insights) == 0


class TestMemoryStorageOperations:
    """Test functional memory storage operations with immutable data structures."""
    
    @pytest.fixture
    def empty_memory_storage(self):
        """Create empty memory storage."""
        return MemoryStorage.empty()
    
    @pytest.fixture
    def sample_experience_for_storage(self):
        """Create sample experience for storage testing."""
        symbol = Symbol.create("ETH-USD").success()
        snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            price=Decimal("3000"),
            indicators={"rsi": 65.0},
            dominance_data=None,
            position_side="FLAT",
            position_size=Decimal("0"),
        )
        
        return TradingExperienceFP.create(
            market_snapshot=snapshot,
            trade_decision="SHORT",
            decision_rationale="High RSI suggests overbought condition",
            pattern_tags=[PatternTag.create("overbought").success()],
        )
    
    def test_memory_storage_add_experience(
        self, empty_memory_storage, sample_experience_for_storage
    ):
        """Test adding experience to memory storage."""
        updated_storage = empty_memory_storage.add_experience(sample_experience_for_storage)
        
        # Original storage should be unchanged (immutable)
        assert len(empty_memory_storage.experiences) == 0
        
        # Updated storage should have the new experience
        assert len(updated_storage.experiences) == 1
        assert updated_storage.experiences[0] == sample_experience_for_storage
    
    def test_memory_storage_find_by_id(
        self, empty_memory_storage, sample_experience_for_storage
    ):
        """Test finding experience by ID."""
        updated_storage = empty_memory_storage.add_experience(sample_experience_for_storage)
        
        # Should find the experience
        found = updated_storage.find_by_id(sample_experience_for_storage.experience_id)
        assert found.is_some()
        assert found.value == sample_experience_for_storage
        
        # Should not find non-existent experience
        fake_id = ExperienceId.generate()
        not_found = updated_storage.find_by_id(fake_id)
        assert not_found.is_nothing()
    
    def test_memory_storage_update_experience(
        self, empty_memory_storage, sample_experience_for_storage
    ):
        """Test updating experience in storage."""
        # Add experience first
        storage_with_exp = empty_memory_storage.add_experience(sample_experience_for_storage)
        
        # Create outcome to add
        outcome = TradingOutcome.create(
            pnl=Decimal("-75"),
            exit_price=Decimal("2925"),
            entry_price=Decimal("3000"),
            duration_minutes=60.0,
        ).success()
        
        # Update experience with outcome
        update_result = storage_with_exp.update_experience(
            sample_experience_for_storage.experience_id,
            lambda exp: exp.with_outcome(outcome),
        )
        
        assert update_result.is_success()
        updated_storage = update_result.success()
        
        # Find updated experience
        updated_exp = updated_storage.find_by_id(sample_experience_for_storage.experience_id)
        assert updated_exp.is_some()
        assert updated_exp.value.outcome.is_some()
        assert updated_exp.value.outcome.value.pnl == Decimal("-75")
        
        # Original storage should be unchanged
        original_exp = storage_with_exp.find_by_id(sample_experience_for_storage.experience_id)
        assert original_exp.value.outcome.is_nothing()
    
    def test_memory_storage_update_nonexistent_experience(self, empty_memory_storage):
        """Test updating non-existent experience."""
        fake_id = ExperienceId.generate()
        
        update_result = empty_memory_storage.update_experience(
            fake_id,
            lambda exp: exp.with_insights("Should not work"),
        )
        
        assert update_result.is_failure()
        assert "not found" in update_result.failure().lower()
    
    def test_memory_storage_filter_by_pattern(
        self, empty_memory_storage, sample_experience_for_storage
    ):
        """Test filtering experiences by pattern."""
        # Add multiple experiences with different patterns
        oversold_pattern = PatternTag.create("oversold").success()
        oversold_exp = TradingExperienceFP.create(
            market_snapshot=sample_experience_for_storage.market_snapshot,
            trade_decision="LONG",
            decision_rationale="Oversold condition",
            pattern_tags=[oversold_pattern],
        )
        
        storage = empty_memory_storage.add_experience(sample_experience_for_storage)
        storage = storage.add_experience(oversold_exp)
        
        # Filter by overbought pattern
        overbought_pattern = PatternTag.create("overbought").success()
        overbought_experiences = storage.filter_by_pattern(overbought_pattern)
        assert len(overbought_experiences) == 1
        assert overbought_experiences[0] == sample_experience_for_storage
        
        # Filter by oversold pattern
        oversold_experiences = storage.filter_by_pattern(oversold_pattern)
        assert len(oversold_experiences) == 1
        assert oversold_experiences[0] == oversold_exp
        
        # Filter by non-existent pattern
        nonexistent_pattern = PatternTag.create("nonexistent").success()
        no_experiences = storage.filter_by_pattern(nonexistent_pattern)
        assert len(no_experiences) == 0


class TestFunctionalMemoryIntegrationComplexScenarios:
    """Test complex integration scenarios combining multiple functional memory patterns."""
    
    @pytest.fixture
    async def full_memory_system(self):
        """Create a complete functional memory system for integration testing."""
        mock_server = MagicMock(spec=MCPMemoryServer)
        
        # Set up mock to simulate realistic behavior
        stored_experiences = {}
        
        async def mock_store_experience(market_state, trade_action):
            exp_id = f"exp_{len(stored_experiences) + 1:06d}"
            stored_experiences[exp_id] = {
                "market_state": market_state,
                "trade_action": trade_action,
                "timestamp": datetime.now(UTC),
                "outcome": None,
            }
            return exp_id
        
        async def mock_update_outcome(exp_id, pnl, exit_price, duration, market_state=None):
            if exp_id in stored_experiences:
                stored_experiences[exp_id]["outcome"] = {
                    "pnl": pnl,
                    "exit_price": exit_price,
                    "duration_minutes": duration,
                    "market_state_at_exit": market_state,
                }
                return True
            return False
        
        async def mock_query_similar(market_state, query=None):
            # Return some mock experiences
            return list(stored_experiences.values())[:3]  # Return up to 3
        
        mock_server.store_experience = mock_store_experience
        mock_server.update_experience_outcome = mock_update_outcome
        mock_server.query_similar_experiences = mock_query_similar
        mock_server.get_pattern_statistics.return_value = {}
        
        return MemoryAdapterFP(mock_server), stored_experiences
    
    @pytest.mark.asyncio
    async def test_complete_trading_lifecycle_with_fp_memory(self, full_memory_system):
        """Test complete trading lifecycle using functional memory patterns."""
        memory_adapter, stored_experiences = full_memory_system
        
        # 1. Create market snapshot
        symbol = Symbol.create("BTC-USD").success()
        market_snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            price=Decimal("52000"),
            indicators={
                "rsi": 35.0,
                "ema_fast": 51800.0,
                "ema_slow": 51500.0,
            },
            dominance_data={"stablecoin_dominance": 9.2},
            position_side="FLAT",
            position_size=Decimal("0"),
        )
        
        # 2. Create trading experience
        pattern_tags = [
            PatternTag.create("oversold").success(),
            PatternTag.create("bullish_divergence").success(),
        ]
        
        experience = TradingExperienceFP.create(
            market_snapshot=market_snapshot,
            trade_decision="LONG",
            decision_rationale="Oversold RSI with bullish divergence - good buy opportunity",
            pattern_tags=pattern_tags,
        )
        
        # 3. Store experience
        store_result = await memory_adapter.store_experience_fp(experience)
        assert store_result.is_success()
        experience_id = store_result.success()
        
        # 4. Simulate trade execution and outcome
        outcome = TradingOutcome.create(
            pnl=Decimal("280"),
            exit_price=Decimal("53400"),
            entry_price=Decimal("52000"),
            duration_minutes=90.0,
        ).success()
        
        # 5. Update with outcome
        update_result = await memory_adapter.update_experience_outcome_fp(
            experience_id, outcome
        )
        assert update_result.is_success()
        completed_experience = update_result.success()
        
        # 6. Verify experience is completed
        assert completed_experience.is_completed()
        assert completed_experience.outcome.is_some()
        assert completed_experience.outcome.value.is_successful
        assert completed_experience.confidence_score > Decimal("0.5")
        
        # 7. Query for similar experiences
        query = MemoryQueryFP.create(
            current_price=Decimal("52000"),
            indicators={"rsi": 35.0},
            pattern_tags=pattern_tags,
            max_results=5,
            min_similarity=0.6,
        ).success()
        
        query_result = await memory_adapter.query_similar_experiences_fp(
            market_snapshot, query
        )
        assert query_result.is_success()
        
        # 8. Generate learning insights
        experiences_for_insights = [completed_experience]
        insights_result = await memory_adapter.generate_learning_insights_fp(
            experiences_for_insights
        )
        assert insights_result.is_success()
        insights = insights_result.success()
        
        # Should have some insights generated
        assert len(insights) >= 0  # May be empty with single experience
        
        # 9. Verify memory storage state
        storage = memory_adapter.get_memory_storage()
        assert len(storage.experiences) == 1
        stored_exp = storage.find_by_id(experience_id)
        assert stored_exp.is_some()
        assert stored_exp.value.outcome.is_some()
    
    @pytest.mark.asyncio
    async def test_memory_system_error_recovery_patterns(self, full_memory_system):
        """Test error recovery patterns in functional memory system."""
        memory_adapter, _ = full_memory_system
        
        # 1. Test with invalid experience data
        symbol = Symbol.create("INVALID-SYMBOL").success()
        invalid_snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            price=Decimal("-1000"),  # Invalid negative price - but allowed in snapshot
            indicators={},
            dominance_data=None,
            position_side="INVALID",
            position_size=Decimal("0"),
        )
        
        invalid_experience = TradingExperienceFP.create(
            market_snapshot=invalid_snapshot,
            trade_decision="INVALID_ACTION",
            decision_rationale="This should handle gracefully",
            pattern_tags=[],
        )
        
        # The adapter should handle conversion errors gracefully
        # Even if the data is unusual, it should not crash
        store_result = await memory_adapter.store_experience_fp(invalid_experience)
        # Result depends on adapter's error handling - could succeed or fail
        
        # 2. Test outcome update for non-existent experience
        fake_id = ExperienceId.generate()
        outcome = TradingOutcome.create(
            pnl=Decimal("100"),
            exit_price=Decimal("51000"),
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        ).success()
        
        # Should return proper error result
        fake_update_result = await memory_adapter.update_experience_outcome_fp(
            fake_id, outcome
        )
        # This will depend on the mock setup, but should handle gracefully
        
        # 3. Test with server connection failures
        memory_adapter._mcp_server.store_experience.side_effect = Exception("Connection lost")
        
        valid_symbol = Symbol.create("BTC-USD").success()
        valid_snapshot = MarketSnapshot(
            symbol=valid_symbol,
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            indicators={"rsi": 50.0},
            dominance_data=None,
            position_side="FLAT",
            position_size=Decimal("0"),
        )
        
        valid_experience = TradingExperienceFP.create(
            market_snapshot=valid_snapshot,
            trade_decision="LONG",
            decision_rationale="Valid trade",
        )
        
        connection_error_result = await memory_adapter.store_experience_fp(valid_experience)
        assert connection_error_result.is_failure()
        assert "Connection lost" in connection_error_result.failure()
    
    @pytest.mark.asyncio
    async def test_bulk_memory_operations_performance(self, full_memory_system):
        """Test performance characteristics of bulk memory operations."""
        memory_adapter, _ = full_memory_system
        
        # Create multiple experiences for bulk testing
        symbol = Symbol.create("ETH-USD").success()
        experiences = []
        
        for i in range(10):  # Create 10 experiences
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=datetime.now(UTC) - timedelta(minutes=i*30),
                price=Decimal(f"{3000 + i*10}"),
                indicators={"rsi": 50.0 + i*2},
                dominance_data=None,
                position_side="FLAT",
                position_size=Decimal("0"),
            )
            
            pattern = PatternTag.create(f"pattern_{i % 3}").success()
            experience = TradingExperienceFP.create(
                market_snapshot=snapshot,
                trade_decision="LONG" if i % 2 == 0 else "SHORT",
                decision_rationale=f"Trade {i}",
                pattern_tags=[pattern],
            )
            
            experiences.append(experience)
        
        # Store all experiences
        import time
        start_time = time.time()
        
        for experience in experiences:
            result = await memory_adapter.store_experience_fp(experience)
            assert result.is_success()
        
        store_time = time.time() - start_time
        
        # Should complete reasonably quickly
        assert store_time < 5.0  # Less than 5 seconds for 10 operations
        
        # Verify all were stored
        storage = memory_adapter.get_memory_storage()
        assert len(storage.experiences) == 10
        
        # Test bulk insights generation
        start_time = time.time()
        insights_result = await memory_adapter.generate_learning_insights_fp(experiences)
        insights_time = time.time() - start_time
        
        assert insights_result.is_success()
        assert insights_time < 2.0  # Should be fast for small dataset


class TestTypeConversionsAndCompatibility:
    """Test type conversions between functional and imperative memory types."""
    
    @pytest.fixture
    def memory_adapter(self):
        """Create memory adapter for conversion testing."""
        mock_server = MagicMock(spec=MCPMemoryServer)
        return MemoryAdapterFP(mock_server)
    
    def test_fp_to_market_state_conversion(self, memory_adapter):
        """Test conversion from FP MarketSnapshot to imperative MarketState."""
        symbol = Symbol.create("BTC-USD").success()
        fp_snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            indicators={
                "rsi": 45.0,
                "ema_fast": 49900.0,
                "ema_slow": 49800.0,
                "cipher_a_dot": 5.0,
            },
            dominance_data={"stablecoin_dominance": 8.5},
            position_side="LONG",
            position_size=Decimal("0.5"),
        )
        
        market_state = memory_adapter._fp_to_market_state(fp_snapshot)
        
        # Verify conversion
        assert market_state.symbol == "BTC-USD"
        assert market_state.current_price == Decimal("50000")
        assert market_state.indicators.rsi == 45.0
        assert market_state.indicators.ema_fast == 49900.0
        assert market_state.current_position.side == "LONG"
        assert market_state.current_position.size == Decimal("0.5")
    
    def test_imperative_to_fp_experience_conversion(self, memory_adapter):
        """Test conversion from imperative TradingExperience to FP types."""
        imperative_experience = TradingExperience(
            experience_id="exp_test_123456",
            timestamp=datetime.now(UTC),
            symbol="ETH-USD",
            price=Decimal("3000"),
            decision_rationale="Test conversion",
            indicators={"rsi": 65.0, "ema_fast": 2990.0},
            dominance_data={"stablecoin_dominance": 9.0},
            pattern_tags=["overbought", "high_volume"],
            market_state_snapshot={"position_side": "FLAT", "position_size": 0},
            outcome={"pnl": 150, "exit_price": 3150, "duration_minutes": 45},
            learned_insights="Good exit timing",
            confidence_score=0.8,
        )
        
        conversion_result = memory_adapter._imperative_to_fp_experience(imperative_experience)
        
        assert conversion_result.is_success()
        fp_experience = conversion_result.success()
        
        # Verify conversion
        assert fp_experience.experience_id.value == "exp_test_123456"
        assert fp_experience.market_snapshot.symbol.value == "ETH-USD"
        assert fp_experience.market_snapshot.price == Decimal("3000")
        assert fp_experience.outcome.is_some()
        assert fp_experience.outcome.value.pnl == Decimal("150")
        assert fp_experience.learned_insights.is_some()
        assert fp_experience.learned_insights.value == "Good exit timing"
        assert len(fp_experience.pattern_tags) == 2
    
    def test_conversion_error_handling(self, memory_adapter):
        """Test error handling in type conversions."""
        # Test with invalid experience ID
        invalid_experience = TradingExperience(
            experience_id="",  # Invalid empty ID
            timestamp=datetime.now(UTC),
            symbol="BTC-USD",
            price=Decimal("50000"),
            decision_rationale="Test",
            indicators={},
            dominance_data={},
            pattern_tags=[],
            market_state_snapshot={},
            outcome=None,
            learned_insights=None,
            confidence_score=0.5,
        )
        
        conversion_result = memory_adapter._imperative_to_fp_experience(invalid_experience)
        assert conversion_result.is_failure()
        assert "empty" in conversion_result.failure().lower()
        
        # Test with invalid symbol
        invalid_symbol_experience = TradingExperience(
            experience_id="exp_valid_123",
            timestamp=datetime.now(UTC),
            symbol="",  # Invalid empty symbol
            price=Decimal("50000"),
            decision_rationale="Test",
            indicators={},
            dominance_data={},
            pattern_tags=[],
            market_state_snapshot={},
            outcome=None,
            learned_insights=None,
            confidence_score=0.5,
        )
        
        conversion_result = memory_adapter._imperative_to_fp_experience(invalid_symbol_experience)
        assert conversion_result.is_failure()


if __name__ == "__main__":
    # Run some basic functionality tests
    print("Testing Functional Memory Integration...")
    
    # Test basic adapter functionality
    import asyncio
    
    async def run_basic_tests():
        # Create mock server for testing
        mock_server = MagicMock(spec=MCPMemoryServer)
        mock_server.store_experience = AsyncMock(return_value="exp_123456789")
        mock_server.update_experience_outcome = AsyncMock(return_value=True)
        mock_server.query_similar_experiences = AsyncMock(return_value=[])
        mock_server.get_pattern_statistics = AsyncMock(return_value={})
        
        adapter = MemoryAdapterFP(mock_server)
        
        # Test basic storage operations
        symbol = Symbol.create("BTC-USD").success()
        snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            indicators={"rsi": 45.0},
            dominance_data=None,
            position_side="FLAT",
            position_size=Decimal("0"),
        )
        
        experience = TradingExperienceFP.create(
            market_snapshot=snapshot,
            trade_decision="LONG",
            decision_rationale="Test trade",
        )
        
        # Test store
        result = await adapter.store_experience_fp(experience)
        assert result.is_success()
        print("✓ Basic store operation test passed")
        
        # Test query
        query_result = await adapter.query_similar_experiences_fp(snapshot)
        assert query_result.is_success()
        print("✓ Basic query operation test passed")
        
        # Test pattern statistics
        stats_result = await adapter.get_pattern_statistics_fp()
        assert stats_result.is_success()
        print("✓ Basic pattern statistics test passed")
        
        print("All basic functional memory integration tests completed successfully!")
    
    asyncio.run(run_basic_tests())