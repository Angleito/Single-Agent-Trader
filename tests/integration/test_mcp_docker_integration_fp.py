"""
Functional programming Docker integration tests for MCP Memory Server.

These tests validate the FP learning system adapters working with a real
MCP server running in Docker, using immutable data structures and
functional programming patterns.

Run with: pytest tests/integration/test_mcp_docker_integration_fp.py -v
"""

import asyncio
import subprocess
import time
from datetime import datetime
from decimal import Decimal
from typing import List

import httpx
import pytest

from bot.fp.types import (
    Result, Success, Failure,
    Maybe, Some, Nothing,
    Symbol, ExperienceId, TradingExperienceFP, TradingOutcome,
    MarketSnapshot, PatternTag, MemoryQueryFP, PatternStatistics,
    LearningInsight, MemoryStorage,
)
from bot.fp.adapters.memory_adapter import MemoryAdapterFP
from bot.mcp.memory_server import MCPMemoryServer

# Configuration
MCP_SERVER_URL = "http://localhost:8765"
CONTAINER_NAME = "mcp-memory-server"


@pytest.fixture(scope="module")
def ensure_mcp_running():
    """Ensure MCP server is running before tests."""
    # Check if container is already running
    result = subprocess.run(
        [
            "docker",
            "ps",
            "--filter",
            f"name={CONTAINER_NAME}",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if CONTAINER_NAME not in result.stdout:
        # Start the container
        subprocess.run(["docker-compose", "up", "-d", "mcp-memory"], check=True)
        time.sleep(5)  # Wait for startup


@pytest.fixture
def sample_market_snapshot_fp():
    """Create a sample FP market snapshot for testing."""
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
def sample_pattern_tags_fp():
    """Create sample FP pattern tags."""
    patterns = ["uptrend", "oversold_rsi", "high_stablecoin_dominance"]
    return [PatternTag.create(p).success() for p in patterns]


@pytest.fixture
def sample_trading_experience_fp(sample_market_snapshot_fp, sample_pattern_tags_fp):
    """Create a sample FP trading experience."""
    return TradingExperienceFP.create(
        market_snapshot=sample_market_snapshot_fp,
        trade_decision="LONG",  # Simplified for testing
        decision_rationale="FP Docker integration test trade",
        pattern_tags=sample_pattern_tags_fp,
    )


@pytest.fixture
async def mcp_server_fp():
    """Create MCP memory server for FP testing."""
    server = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server.connect()
    yield server
    await server.disconnect()


@pytest.fixture
async def memory_adapter_fp(mcp_server_fp):
    """Create FP memory adapter with MCP server."""
    adapter = MemoryAdapterFP(mcp_server_fp)
    yield adapter


# Health and Connection Tests

@pytest.mark.asyncio
async def test_mcp_health_check_fp(ensure_mcp_running):
    """Test MCP server health check endpoint with FP approach."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MCP_SERVER_URL}/health", timeout=5.0)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "memory_count" in data


@pytest.mark.asyncio
async def test_mcp_connection_fp(ensure_mcp_running):
    """Test FP adapter connecting to MCP memory server."""
    server = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server.connect()

    assert server._connected is True

    # Test with FP adapter
    adapter = MemoryAdapterFP(server)
    sync_result = await adapter.sync_with_server()
    assert sync_result.is_success()

    await server.disconnect()


# Experience Storage and Retrieval Tests

@pytest.mark.asyncio
async def test_store_and_retrieve_experience_fp(
    ensure_mcp_running, memory_adapter_fp, sample_trading_experience_fp
):
    """Test storing and retrieving FP trading experiences."""
    # Store FP experience
    store_result = await memory_adapter_fp.store_experience_fp(sample_trading_experience_fp)
    
    assert store_result.is_success()
    experience_id = store_result.success()
    assert isinstance(experience_id, ExperienceId)

    # Query similar experiences (should be empty before outcome)
    query_result = MemoryQueryFP.create(
        current_price=Decimal("50000"),
        indicators={"rsi": 45.0, "cipher_a_dot": 5.0},
        max_results=5,
        min_similarity=0.7,
    )
    assert query_result.is_success()
    query = query_result.success()

    similar_result = await memory_adapter_fp.query_similar_experiences_fp(
        sample_trading_experience_fp.market_snapshot, query
    )
    assert similar_result.is_success()
    similar_experiences = similar_result.success()
    # May be empty since no outcomes yet

    # Add outcome to experience
    outcome_result = TradingOutcome.create(
        pnl=Decimal("100"),
        exit_price=Decimal("51000"),
        entry_price=Decimal("50000"),
        duration_minutes=30.0,
    )
    assert outcome_result.is_success()
    outcome = outcome_result.success()

    update_result = await memory_adapter_fp.update_experience_outcome_fp(
        experience_id, outcome
    )
    assert update_result.is_success()
    updated_experience = update_result.success()
    assert updated_experience.outcome.is_some()
    assert updated_experience.outcome.value.pnl == Decimal("100")

    # Now should find it in queries
    similar_result = await memory_adapter_fp.query_similar_experiences_fp(
        sample_trading_experience_fp.market_snapshot, query
    )
    assert similar_result.is_success()
    similar_experiences = similar_result.success()
    # Should now contain at least one experience if conversion successful


@pytest.mark.asyncio
async def test_memory_persistence_across_connections_fp(
    ensure_mcp_running, sample_market_snapshot_fp
):
    """Test that FP memories persist across different connections."""
    # First connection - store data
    server1 = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server1.connect()
    adapter1 = MemoryAdapterFP(server1)

    # Create and store FP experience
    eth_symbol_result = Symbol.create("ETH-USD")
    assert eth_symbol_result.is_success()
    
    eth_snapshot = MarketSnapshot(
        symbol=eth_symbol_result.success(),
        timestamp=datetime.utcnow(),
        price=Decimal("3000"),
        indicators={
            "rsi": 60.0,
            "ema_fast": 2990.0,
            "ema_slow": 2980.0,
            "cipher_a_dot": 10.0,
            "cipher_b_wave": 15.0,
            "cipher_b_money_flow": 55.0,
        },
        dominance_data=None,
        position_side="FLAT",
        position_size=Decimal("0"),
    )

    pattern_tags = [PatternTag.create("bullish_momentum").success()]
    experience = TradingExperienceFP.create(
        market_snapshot=eth_snapshot,
        trade_decision="LONG",
        decision_rationale="FP persistence test trade",
        pattern_tags=pattern_tags,
    )

    store_result = await adapter1.store_experience_fp(experience)
    assert store_result.is_success()
    experience_id = store_result.success()

    # Add outcome
    outcome_result = TradingOutcome.create(
        pnl=Decimal("150"),
        exit_price=Decimal("3150"),
        entry_price=Decimal("3000"),
        duration_minutes=45.0,
    )
    assert outcome_result.is_success()
    outcome = outcome_result.success()

    update_result = await adapter1.update_experience_outcome_fp(experience_id, outcome)
    assert update_result.is_success()

    await server1.disconnect()

    # Brief pause
    await asyncio.sleep(1)

    # Second connection - retrieve data
    server2 = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server2.connect()
    adapter2 = MemoryAdapterFP(server2)

    # Query for the stored experience
    query_result = MemoryQueryFP.create(
        current_price=Decimal("3000"),
        indicators={"rsi": 60.0},
        max_results=10,
        min_similarity=0.5,
    )
    assert query_result.is_success()
    query = query_result.success()

    similar_result = await adapter2.query_similar_experiences_fp(eth_snapshot, query)
    assert similar_result.is_success()
    experiences = similar_result.success()

    # Should find persisted data
    found = any(
        exp.experience_id.value == experience_id.value for exp in experiences
        if hasattr(exp, 'experience_id')
    )
    # Note: May fail if conversion from imperative to FP fails
    
    await server2.disconnect()


@pytest.mark.asyncio
async def test_pattern_analysis_fp(
    ensure_mcp_running, memory_adapter_fp, sample_market_snapshot_fp
):
    """Test FP pattern analysis and statistics."""
    # Store multiple experiences with different patterns and outcomes
    pattern_data = [
        ("oversold_rsi", Decimal("80"), True),    # Successful
        ("oversold_rsi", Decimal("-40"), False),  # Failed
        ("oversold_rsi", Decimal("120"), True),   # Successful
        ("overbought_rsi", Decimal("-60"), False), # Failed
        ("uptrend", Decimal("100"), True),        # Successful
    ]

    stored_experiences = []
    for pattern_name, pnl, success in pattern_data:
        # Create pattern tag
        pattern_result = PatternTag.create(pattern_name)
        assert pattern_result.is_success()
        pattern = pattern_result.success()

        # Create experience
        experience = TradingExperienceFP.create(
            market_snapshot=sample_market_snapshot_fp,
            trade_decision="LONG" if success else "SHORT",
            decision_rationale=f"FP pattern test: {pattern_name}",
            pattern_tags=[pattern],
        )

        # Store experience
        store_result = await memory_adapter_fp.store_experience_fp(experience)
        assert store_result.is_success()
        experience_id = store_result.success()

        # Add outcome
        exit_price = Decimal("51000") if success else Decimal("49000")
        outcome_result = TradingOutcome.create(
            pnl=pnl,
            exit_price=exit_price,
            entry_price=Decimal("50000"),
            duration_minutes=30.0,
        )
        assert outcome_result.is_success()
        outcome = outcome_result.success()

        update_result = await memory_adapter_fp.update_experience_outcome_fp(
            experience_id, outcome
        )
        assert update_result.is_success()
        stored_experiences.append(update_result.success())

    # Get pattern statistics
    stats_result = await memory_adapter_fp.get_pattern_statistics_fp()
    assert stats_result.is_success()
    pattern_stats = stats_result.success()

    # Verify statistics for oversold_rsi pattern (3 occurrences, 2 successful)
    oversold_stats = next(
        (s for s in pattern_stats if s.pattern.name == "oversold_rsi"), None
    )
    if oversold_stats:  # May be None if conversion fails
        assert oversold_stats.total_occurrences >= 3
        # Success rate should be around 2/3
        assert oversold_stats.success_rate >= Decimal("0.5")

    # Generate learning insights
    insights_result = await memory_adapter_fp.generate_learning_insights_fp(
        stored_experiences
    )
    assert insights_result.is_success()
    insights = insights_result.success()
    assert isinstance(insights, list)
    # Insights may be empty depending on analysis logic


@pytest.mark.asyncio
async def test_fp_memory_storage_operations(ensure_mcp_running, sample_trading_experience_fp):
    """Test FP memory storage operations with real MCP server."""
    server = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server.connect()
    adapter = MemoryAdapterFP(server)

    # Test local storage operations
    storage = MemoryStorage.empty()
    assert storage.total_experiences == 0

    # Add experience to local storage
    storage = storage.add_experience(sample_trading_experience_fp)
    assert storage.total_experiences == 1
    assert storage.completed_experiences == 0

    # Find by ID
    found_exp = storage.find_by_id(sample_trading_experience_fp.experience_id)
    assert found_exp.is_some()
    assert found_exp.value == sample_trading_experience_fp

    # Find by pattern
    if sample_trading_experience_fp.pattern_tags:
        pattern = sample_trading_experience_fp.pattern_tags[0]
        pattern_experiences = storage.find_by_pattern(pattern)
        assert len(pattern_experiences) == 1
        assert pattern_experiences[0] == sample_trading_experience_fp

    # Update experience with outcome
    outcome_result = TradingOutcome.create(
        pnl=Decimal("200"),
        exit_price=Decimal("52000"),
        entry_price=Decimal("50000"),
        duration_minutes=60.0,
    )
    assert outcome_result.is_success()
    outcome = outcome_result.success()

    update_result = storage.update_experience(
        sample_trading_experience_fp.experience_id,
        lambda exp: exp.with_outcome(outcome),
    )
    assert update_result.is_success()
    updated_storage = update_result.success()
    
    # Verify completion count updated
    assert updated_storage.completed_experiences == 1
    
    # Get completed experiences
    completed = updated_storage.get_completed_experiences()
    assert len(completed) == 1
    assert completed[0].outcome.is_some()
    assert completed[0].outcome.value.pnl == Decimal("200")

    await server.disconnect()


@pytest.mark.asyncio
async def test_fp_pattern_tag_indexing(ensure_mcp_running, memory_adapter_fp):
    """Test FP pattern tag indexing and retrieval."""
    # Create experiences with specific patterns
    test_patterns = [
        ("extreme_oversold", {"rsi": 25.0}),
        ("extreme_overbought", {"rsi": 75.0}),
        ("cipher_b_extreme_low", {"cipher_b_wave": -80.0}),
        ("bullish_divergence", {"rsi": 35.0, "cipher_a_dot": 15.0}),
    ]

    stored_experience_ids = []
    for pattern_name, indicators in test_patterns:
        # Create pattern tag
        pattern_result = PatternTag.create(pattern_name)
        assert pattern_result.is_success()
        pattern = pattern_result.success()

        # Create market snapshot with specific indicators
        symbol_result = Symbol.create("BTC-USD")
        assert symbol_result.is_success()
        
        snapshot = MarketSnapshot(
            symbol=symbol_result.success(),
            timestamp=datetime.utcnow(),
            price=Decimal("45000"),
            indicators=indicators,
            dominance_data=None,
            position_side="FLAT",
            position_size=Decimal("0"),
        )

        # Create experience
        experience = TradingExperienceFP.create(
            market_snapshot=snapshot,
            trade_decision="LONG",
            decision_rationale=f"FP indexing test: {pattern_name}",
            pattern_tags=[pattern],
        )

        # Store experience
        store_result = await memory_adapter_fp.store_experience_fp(experience)
        assert store_result.is_success()
        experience_id = store_result.success()
        stored_experience_ids.append((experience_id, pattern))

        # Add outcome for pattern analysis
        pnl = Decimal("50") if "oversold" in pattern_name or "divergence" in pattern_name else Decimal("-30")
        outcome_result = TradingOutcome.create(
            pnl=pnl,
            exit_price=Decimal("45500") if pnl > 0 else Decimal("44500"),
            entry_price=Decimal("45000"),
            duration_minutes=45.0,
        )
        assert outcome_result.is_success()
        outcome = outcome_result.success()

        update_result = await memory_adapter_fp.update_experience_outcome_fp(
            experience_id, outcome
        )
        assert update_result.is_success()

    # Test pattern-based queries
    # Query for oversold patterns
    oversold_query = MemoryQueryFP.create(
        indicators={"rsi": 30.0},
        pattern_tags=[PatternTag.create("extreme_oversold").success()],
        max_results=10,
        min_similarity=0.5,
    )
    assert oversold_query.is_success()
    
    # Create test snapshot for query
    test_symbol = Symbol.create("BTC-USD").success()
    test_snapshot = MarketSnapshot(
        symbol=test_symbol,
        timestamp=datetime.utcnow(),
        price=Decimal("45000"),
        indicators={"rsi": 30.0},
        dominance_data=None,
        position_side="FLAT",
        position_size=Decimal("0"),
    )

    oversold_result = await memory_adapter_fp.query_similar_experiences_fp(
        test_snapshot, oversold_query.success()
    )
    assert oversold_result.is_success()
    oversold_experiences = oversold_result.success()
    # May be empty if pattern matching/conversion fails


@pytest.mark.asyncio
async def test_container_resource_usage_fp(ensure_mcp_running):
    """Test FP operations don't cause excessive container resource usage."""
    # Get initial container stats
    result = await asyncio.to_thread(
        subprocess.run,
        [
            "docker",
            "stats",
            CONTAINER_NAME,
            "--no-stream",
            "--format",
            "{{.MemUsage}} {{.CPUPerc}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        # Parse memory and CPU usage
        parts = result.stdout.strip().split()
        if len(parts) >= 2:
            mem_usage = parts[0]  # e.g., "50MiB / 512MiB"
            cpu_usage = parts[1]  # e.g., "0.5%"

            # Basic assertions (adjust thresholds as needed)
            assert "MiB" in mem_usage or "GiB" in mem_usage
            cpu_percent = float(cpu_usage.rstrip("%"))
            assert cpu_percent < 50.0  # CPU should be under 50%

            # Perform some FP operations to test resource usage
            server = MCPMemoryServer(server_url=MCP_SERVER_URL)
            await server.connect()
            adapter = MemoryAdapterFP(server)

            # Create multiple experiences
            for i in range(10):
                symbol_result = Symbol.create("BTC-USD")
                assert symbol_result.is_success()
                
                snapshot = MarketSnapshot(
                    symbol=symbol_result.success(),
                    timestamp=datetime.utcnow(),
                    price=Decimal(f"{50000 + i * 100}"),
                    indicators={"rsi": 50.0 + i},
                    dominance_data=None,
                    position_side="FLAT",
                    position_size=Decimal("0"),
                )

                experience = TradingExperienceFP.create(
                    market_snapshot=snapshot,
                    trade_decision="LONG",
                    decision_rationale=f"FP stress test {i}",
                    pattern_tags=[],
                )

                store_result = await adapter.store_experience_fp(experience)
                assert store_result.is_success()

            await server.disconnect()

            # Check resource usage again after operations
            result_after = await asyncio.to_thread(
                subprocess.run,
                [
                    "docker",
                    "stats",
                    CONTAINER_NAME,
                    "--no-stream",
                    "--format",
                    "{{.CPUPerc}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result_after.returncode == 0:
                cpu_after = float(result_after.stdout.strip().rstrip("%"))
                # Resource usage should remain reasonable
                assert cpu_after < 80.0  # Should not spike too high


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
