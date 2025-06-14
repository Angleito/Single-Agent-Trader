"""
Docker integration tests for MCP Memory Server.

Run with: pytest tests/integration/test_mcp_docker_integration.py -v
"""

import asyncio
import subprocess
import time
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from bot.data.dominance import DominanceData
from bot.mcp.memory_server import MCPMemoryServer
from bot.types import (
    IndicatorData,
    MarketData,
    MarketState,
    Position,
    TradeAction,
)

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
    )

    if CONTAINER_NAME not in result.stdout:
        # Start the container
        subprocess.run(["docker-compose", "up", "-d", "mcp-memory"], check=True)
        time.sleep(5)  # Wait for startup

    yield

    # Optionally stop container after tests (comment out to keep running)
    # subprocess.run(["docker-compose", "stop", "mcp-memory"])


@pytest.mark.asyncio
async def test_mcp_health_check(ensure_mcp_running):
    """Test MCP server health check endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MCP_SERVER_URL}/health", timeout=5.0)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "memory_count" in data


@pytest.mark.asyncio
async def test_mcp_connection(ensure_mcp_running):
    """Test connecting to MCP memory server."""
    server = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server.connect()

    assert server._connected is True

    await server.disconnect()


@pytest.mark.asyncio
async def test_store_and_retrieve_experience(ensure_mcp_running):
    """Test storing and retrieving trading experiences."""
    server = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server.connect()

    # Create test market state
    now = datetime.now(UTC)
    market_state = MarketState(
        symbol="BTC-USD",
        interval="3m",
        timestamp=now,
        current_price=Decimal("50000"),
        ohlcv_data=[
            MarketData(
                symbol="BTC-USD",
                timestamp=now,
                open=Decimal("49800"),
                high=Decimal("50100"),
                low=Decimal("49700"),
                close=Decimal("50000"),
                volume=Decimal("100"),
            )
        ],
        indicators=IndicatorData(
            timestamp=now,
            rsi=45.0,
            ema_fast=49900.0,
            ema_slow=49800.0,
            cipher_a_dot=5.0,
            cipher_b_wave=-10.0,
            cipher_b_money_flow=48.0,
            market_sentiment="NEUTRAL",
        ),
        dominance_data=DominanceData(
            timestamp=now,
            usdt_market_cap=Decimal("95000000000"),
            usdc_market_cap=Decimal("45000000000"),
            total_stablecoin_cap=Decimal("140000000000"),
            crypto_total_market_cap=Decimal("1650000000000"),
            usdt_dominance=5.76,
            usdc_dominance=2.73,
            stablecoin_dominance=8.5,
            dominance_24h_change=-0.5,
            dominance_7d_change=-1.2,
            dominance_rsi=40.0,
        ),
        current_position=Position(
            symbol="BTC-USD", side="FLAT", size=Decimal("0"), timestamp=now
        ),
    )

    # Create trade action
    trade_action = TradeAction(
        action="LONG",
        size_pct=10,
        take_profit_pct=2.0,
        stop_loss_pct=1.0,
        leverage=2,
        rationale="Docker integration test trade",
    )

    # Store experience
    experience_id = await server.store_experience(
        market_state, trade_action, insights="Docker test successful"
    )

    assert experience_id is not None

    # Retrieve similar experiences
    similar_experiences = await server.retrieve_similar_experiences(
        current_price=Decimal("50000"),
        indicators={
            "rsi": 45.0,
            "cipher_a_dot": 5.0,
            "cipher_b_wave": -10.0,
        },
        max_results=5,
    )

    # Verify we can find our stored experience
    found = any(exp.experience_id == experience_id for exp in similar_experiences)
    assert found, "Could not retrieve stored experience"

    await server.disconnect()


@pytest.mark.asyncio
async def test_memory_persistence_across_connections(ensure_mcp_running):
    """Test that memories persist across different connections."""
    # First connection - store data
    server1 = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server1.connect()

    now = datetime.now(UTC)
    market_state = MarketState(
        symbol="ETH-USD",
        interval="3m",
        timestamp=now,
        current_price=Decimal("3000"),
        ohlcv_data=[],
        indicators=IndicatorData(
            timestamp=now,
            rsi=60.0,
            ema_fast=2990.0,
            ema_slow=2980.0,
            cipher_a_dot=10.0,
            cipher_b_wave=15.0,
            cipher_b_money_flow=55.0,
            market_sentiment="BULLISH",
        ),
        dominance_data=None,
        current_position=Position(
            symbol="ETH-USD", side="FLAT", size=Decimal("0"), timestamp=now
        ),
    )

    trade_action = TradeAction(
        action="LONG",
        size_pct=15,
        take_profit_pct=3.0,
        stop_loss_pct=1.5,
        leverage=3,
        rationale="Persistence test trade",
    )

    experience_id = await server1.store_experience(
        market_state, trade_action, insights="Testing persistence in Docker"
    )

    await server1.disconnect()

    # Brief pause
    await asyncio.sleep(1)

    # Second connection - retrieve data
    server2 = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server2.connect()

    experiences = await server2.retrieve_similar_experiences(
        current_price=Decimal("3000"),
        indicators={"rsi": 60.0},
        max_results=10,
    )

    found = any(exp.experience_id == experience_id for exp in experiences)
    assert found, "Data not persisted across connections"

    await server2.disconnect()


@pytest.mark.asyncio
async def test_pattern_indexing(ensure_mcp_running):
    """Test pattern-based retrieval."""
    server = MCPMemoryServer(server_url=MCP_SERVER_URL)
    await server.connect()

    # Store experience with specific patterns
    now = datetime.now(UTC)
    market_state = MarketState(
        symbol="BTC-USD",
        interval="3m",
        timestamp=now,
        current_price=Decimal("45000"),
        ohlcv_data=[],
        indicators=IndicatorData(
            timestamp=now,
            rsi=25.0,  # Oversold
            ema_fast=44900.0,
            ema_slow=45100.0,
            cipher_a_dot=-20.0,
            cipher_b_wave=-80.0,  # Extreme low
            cipher_b_money_flow=30.0,
            market_sentiment="BEARISH",
        ),
        dominance_data=None,
        current_position=Position(
            symbol="BTC-USD", side="FLAT", size=Decimal("0"), timestamp=now
        ),
    )

    trade_action = TradeAction(
        action="LONG",
        size_pct=20,
        take_profit_pct=3.0,
        stop_loss_pct=1.5,
        leverage=2,
        rationale="Oversold bounce play",
    )

    # Store with pattern tags
    experience_id = await server.store_experience(
        market_state,
        trade_action,
        insights="Testing pattern indexing",
        pattern_tags=["oversold_rsi", "cipher_b_extreme_low", "bearish_sentiment"],
    )

    # Retrieve by pattern
    pattern_experiences = await server.retrieve_by_pattern(
        ["oversold_rsi", "cipher_b_extreme_low"], limit=5
    )

    assert len(pattern_experiences) > 0
    found = any(exp.experience_id == experience_id for exp in pattern_experiences)
    assert found, "Could not retrieve experience by pattern"

    await server.disconnect()


@pytest.mark.asyncio
async def test_container_resource_usage(ensure_mcp_running):
    """Test container resource usage is within limits."""
    # Get container stats
    result = subprocess.run(
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
    )

    if result.returncode == 0:
        # Parse memory and CPU usage
        parts = result.stdout.strip().split()
        if len(parts) >= 2:
            mem_usage = parts[0]  # e.g., "50MiB / 512MiB"
            cpu_usage = parts[1]  # e.g., "0.5%"

            # Basic assertions (adjust thresholds as needed)
            assert "MiB" in mem_usage or "GiB" in mem_usage
            assert float(cpu_usage.rstrip("%")) < 50.0  # CPU should be under 50%
