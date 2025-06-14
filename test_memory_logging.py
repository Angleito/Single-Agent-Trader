#!/usr/bin/env python3
"""
Test script to verify MCP memory logging is working correctly.
"""

import asyncio
import os
import sys
from decimal import Decimal
from datetime import datetime, UTC

# Add the bot module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.mcp.memory_server import MCPMemoryServer
from bot.types import MarketState, TradeAction, Position, IndicatorData, OHLCV


async def test_memory_logging():
    """Test that memory operations are properly logged."""
    
    print("🧪 Testing MCP Memory Logging...")
    
    # Initialize memory server
    memory_server = MCPMemoryServer(
        server_url="http://localhost:8765",
        api_key=None
    )
    
    # Try to connect
    connected = await memory_server.connect()
    print(f"Connection status: {'✅ Connected' if connected else '❌ Failed'}")
    
    # Create test market state
    market_state = MarketState(
        symbol="ETH-USD",
        interval="3m",
        timestamp=datetime.now(UTC),
        current_price=Decimal("2500.00"),
        ohlcv_data=[
            OHLCV(
                timestamp=datetime.now(UTC),
                open=Decimal("2490"),
                high=Decimal("2510"),
                low=Decimal("2485"),
                close=Decimal("2500"),
                volume=Decimal("1000")
            )
        ],
        indicators=IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=1.0,
            cipher_b_wave=5.5,
            rsi=60.0,
            ema_fast=Decimal("2505"),
            ema_slow=Decimal("2495"),
            market_sentiment="BULLISH"
        ),
        current_position=Position(
            symbol="ETH-USD",
            side="FLAT",
            size=Decimal("0"),
            entry_price=None,
            unrealized_pnl=Decimal("0")
        )
    )
    
    # Create test trade action
    trade_action = TradeAction(
        action="LONG",
        size_pct=15,
        rationale="Test trade for memory logging verification"
    )
    
    print("\n📝 Storing test experience...")
    
    # Store experience
    experience_id = await memory_server.store_experience(
        market_state,
        trade_action,
        additional_context={"test": True}
    )
    
    print(f"Experience stored with ID: {experience_id}")
    
    print("\n🔍 Querying similar experiences...")
    
    # Query for similar experiences
    similar_experiences = await memory_server.query_similar_experiences(market_state)
    
    print(f"Found {len(similar_experiences)} similar experiences")
    
    print("\n📊 Getting pattern statistics...")
    
    # Get pattern statistics
    pattern_stats = await memory_server.get_pattern_statistics()
    
    print(f"Found {len(pattern_stats)} patterns")
    
    # Disconnect
    await memory_server.disconnect()
    
    print("\n✅ Memory logging test completed!")
    print("Check docker logs for detailed memory operation logs:")
    print("  docker-compose logs mcp-memory | grep MCP")
    print("  docker-compose logs ai-trading-bot | grep MCP")


if __name__ == "__main__":
    asyncio.run(test_memory_logging())