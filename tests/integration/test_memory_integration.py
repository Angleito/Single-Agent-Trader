"""Integration tests for the memory and learning system."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from bot.config import settings
from bot.learning.experience_manager import ExperienceManager
from bot.learning.self_improvement import SelfImprovementEngine
from bot.mcp.memory_server import MCPMemoryServer
from bot.strategy.memory_enhanced_agent import MemoryEnhancedLLMAgent
from bot.trading_types import (
    IndicatorData,
    MarketData,
    MarketState,
    Position,
    StablecoinDominance,
    TradeAction,
)


@pytest.fixture
async def memory_server():
    """Create a memory server instance."""
    server = MCPMemoryServer()
    await server.connect()
    yield server
    await server.disconnect()


@pytest.fixture
async def experience_manager(memory_server):
    """Create an experience manager instance."""
    manager = ExperienceManager(memory_server)
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
def self_improvement_engine(memory_server):
    """Create a self-improvement engine instance."""
    return SelfImprovementEngine(memory_server)


@pytest.fixture
def sample_market_state():
    """Create a sample market state for testing."""
    return MarketState(
        symbol="BTC-USD",
        interval="3m",
        timestamp=datetime.now(UTC),
        current_price=Decimal(50000),
        ohlcv_data=[
            MarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(UTC) - timedelta(minutes=3),
                open=Decimal(49800),
                high=Decimal(50100),
                low=Decimal(49700),
                close=Decimal(50000),
                volume=Decimal(100),
            )
        ],
        indicators=IndicatorData(
            timestamp=datetime.now(UTC),
            rsi=45.0,
            ema_fast=49900.0,
            ema_slow=49800.0,
            cipher_a_dot=5.0,
            cipher_b_wave=-10.0,
            cipher_b_money_flow=48.0,
            market_sentiment="NEUTRAL",
        ),
        dominance_data=StablecoinDominance(
            timestamp=datetime.now(UTC),
            stablecoin_dominance=8.5,
            usdt_dominance=5.0,
            usdc_dominance=3.5,
            dominance_24h_change=-0.5,
            dominance_rsi=40.0,
        ),
        current_position=Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal(0),
            timestamp=datetime.now(UTC),
        ),
    )


@pytest.mark.asyncio
async def test_memory_server_store_and_retrieve(memory_server, sample_market_state):
    """Test storing and retrieving experiences."""
    # Store an experience
    trade_action = TradeAction(
        action="LONG",
        size_pct=10,
        take_profit_pct=2.0,
        stop_loss_pct=1.0,
        leverage=2,
        rationale="Test trade",
    )

    experience_id = await memory_server.store_experience(
        sample_market_state, trade_action
    )

    assert experience_id is not None
    assert experience_id in memory_server.memory_cache

    # Query similar experiences
    similar = await memory_server.query_similar_experiences(sample_market_state)
    assert len(similar) == 0  # No completed trades yet

    # Update with outcome
    await memory_server.update_experience_outcome(
        experience_id,
        pnl=Decimal(100),
        exit_price=Decimal(51000),
        duration_minutes=30.0,
    )

    # Now should find it
    similar = await memory_server.query_similar_experiences(sample_market_state)
    assert len(similar) == 1
    assert similar[0].experience_id == experience_id


@pytest.mark.asyncio
async def test_experience_manager_lifecycle(
    experience_manager, memory_server, sample_market_state
):
    """Test complete trade lifecycle tracking."""
    # Record trading decision
    trade_action = TradeAction(
        action="SHORT",
        size_pct=15,
        take_profit_pct=1.5,
        stop_loss_pct=0.8,
        leverage=3,
        rationale="Bearish divergence",
    )

    experience_id = await experience_manager.record_trading_decision(
        sample_market_state, trade_action
    )

    assert experience_id is not None

    # Link to order
    experience_manager.link_order_to_experience("order123", experience_id)
    assert "order123" in experience_manager.pending_experiences

    # Get active trades summary
    summary = experience_manager.get_active_trades_summary()
    assert summary["active_count"] == 0  # No active trades yet


@pytest.mark.asyncio
async def test_self_improvement_analysis(
    self_improvement_engine, memory_server, sample_market_state
):
    """Test self-improvement engine analysis."""
    # Store some sample experiences
    for i in range(5):
        trade_action = TradeAction(
            action="LONG" if i % 2 == 0 else "SHORT",
            size_pct=10 + i,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            leverage=2,
            rationale=f"Test trade {i}",
        )

        exp_id = await memory_server.store_experience(sample_market_state, trade_action)

        # Update with outcome
        await memory_server.update_experience_outcome(
            exp_id,
            pnl=Decimal(50) if i % 2 == 0 else Decimal(-30),
            exit_price=Decimal(51000) if i % 2 == 0 else Decimal(49000),
            duration_minutes=30.0 + i * 10,
        )

    # Analyze performance
    analysis = await self_improvement_engine.analyze_recent_performance(hours=1)

    assert "total_trades" in analysis
    assert analysis["total_trades"] == 5
    assert "success_rate" in analysis
    assert "pattern_performance" in analysis

    # Get recommendations
    recommendations = await self_improvement_engine.get_recommendations_for_market(
        sample_market_state.indicators.__dict__,
        (
            sample_market_state.dominance_data.__dict__
            if sample_market_state.dominance_data
            else None
        ),
    )

    assert "suggested_actions" in recommendations
    assert "confidence_factors" in recommendations


@pytest.mark.asyncio
async def test_memory_enhanced_agent(memory_server, sample_market_state):
    """Test memory-enhanced LLM agent."""
    # Skip if no LLM configured
    if not settings.llm.openai_api_key:
        pytest.skip("No OpenAI API key configured")

    # Create agent with memory
    agent = MemoryEnhancedLLMAgent(memory_server=memory_server)

    # Store some historical data
    for _ in range(3):
        trade_action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            leverage=2,
            rationale="Historical trade",
        )

        exp_id = await memory_server.store_experience(sample_market_state, trade_action)

        await memory_server.update_experience_outcome(
            exp_id,
            pnl=Decimal(100),
            exit_price=Decimal(51000),
            duration_minutes=45.0,
        )

    # Get agent status
    status = agent.get_status()
    assert "memory_enabled" in status
    assert status["memory_enabled"] is True

    # Note: We don't actually call analyze_market here to avoid real API calls
    # In production tests, you'd mock the LLM responses


@pytest.mark.asyncio
async def test_pattern_statistics(memory_server, sample_market_state):
    """Test pattern statistics generation."""
    # Store experiences with different patterns
    patterns = ["uptrend", "oversold", "high_stablecoin_dominance"]

    for i, pattern in enumerate(patterns * 2):  # 2 of each pattern
        # Modify market state to match pattern
        if pattern == "oversold":
            sample_market_state.indicators.rsi = 25.0
        elif pattern == "high_stablecoin_dominance":
            sample_market_state.dominance_data.stablecoin_dominance = 12.0

        trade_action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            leverage=2,
            rationale=f"Trade with {pattern}",
        )

        exp_id = await memory_server.store_experience(sample_market_state, trade_action)

        # Make half successful
        pnl = Decimal(50) if i % 2 == 0 else Decimal(-30)
        await memory_server.update_experience_outcome(
            exp_id,
            pnl=pnl,
            exit_price=Decimal(51000) if pnl > 0 else Decimal(49000),
            duration_minutes=30.0,
        )

    # Get pattern statistics
    stats = await memory_server.get_pattern_statistics()

    assert len(stats) > 0
    assert "oversold" in stats
    assert stats["oversold"]["count"] >= 2
    assert "success_rate" in stats["oversold"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
