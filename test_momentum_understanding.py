#!/usr/bin/env python3
"""Test script to verify LLM understands momentum-based trading."""

import asyncio
from decimal import Decimal
from datetime import datetime, UTC

from bot.strategy.llm_agent import LLMAgent
from bot.types import MarketState, Position, MarketData, IndicatorData


async def test_momentum_understanding():
    """Test the LLM's understanding of momentum trading with positions."""
    
    # Initialize LLM agent
    agent = LLMAgent()
    
    # Test 1: No position - should be able to open LONG or SHORT
    print("\n=== TEST 1: No Position (FLAT) ===")
    market_state_1 = MarketState(
        symbol="ETH-USD",
        interval="5m",
        timestamp=datetime.now(UTC),
        current_price=Decimal("2500.00"),
        ohlcv_data=[],
        indicators=IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=1.0,
            cipher_b_wave=2.5,
            cipher_b_money_flow=65.0,
            rsi=55.0,
            ema_fast=2495.0,
            ema_slow=2490.0
        ),
        current_position=Position(
            symbol="ETH-USD",
            side="FLAT",
            size=Decimal("0"),
            timestamp=datetime.now(UTC)
        )
    )
    
    action_1 = await agent.analyze_market(market_state_1)
    print(f"LLM Decision: {action_1.action}")
    print(f"Rationale: {action_1.rationale}")
    
    # Test 2: Already have LONG position - should only HOLD or CLOSE
    print("\n=== TEST 2: Existing LONG Position ===")
    market_state_2 = MarketState(
        symbol="ETH-USD",
        interval="5m",
        timestamp=datetime.now(UTC),
        current_price=Decimal("2510.00"),
        ohlcv_data=[],
        indicators=IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=1.0,
            cipher_b_wave=2.5,
            cipher_b_money_flow=65.0,
            rsi=55.0,
            ema_fast=2505.0,
            ema_slow=2500.0
        ),
        current_position=Position(
            symbol="ETH-USD",
            side="LONG",
            size=Decimal("2.0"),
            entry_price=Decimal("2500.00"),
            timestamp=datetime.now(UTC)
        )
    )
    
    action_2 = await agent.analyze_market(market_state_2)
    print(f"LLM Decision: {action_2.action}")
    print(f"Rationale: {action_2.rationale}")
    
    # Test 3: Have LONG but bearish signals - should CLOSE, not SHORT
    print("\n=== TEST 3: LONG Position with Bearish Signals ===")
    market_state_3 = MarketState(
        symbol="ETH-USD",
        interval="5m",
        timestamp=datetime.now(UTC),
        current_price=Decimal("2480.00"),
        ohlcv_data=[],
        indicators=IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=-1.0,
            cipher_b_wave=-2.5,
            cipher_b_money_flow=35.0,
            rsi=40.0,
            ema_fast=2485.0,
            ema_slow=2490.0
        ),
        current_position=Position(
            symbol="ETH-USD",
            side="LONG",
            size=Decimal("2.0"),
            entry_price=Decimal("2500.00"),
            timestamp=datetime.now(UTC)
        )
    )
    
    action_3 = await agent.analyze_market(market_state_3)
    print(f"LLM Decision: {action_3.action}")
    print(f"Rationale: {action_3.rationale}")
    
    # Test 4: Already have SHORT position
    print("\n=== TEST 4: Existing SHORT Position ===")
    market_state_4 = MarketState(
        symbol="ETH-USD",
        interval="5m",
        timestamp=datetime.now(UTC),
        current_price=Decimal("2470.00"),
        ohlcv_data=[],
        indicators=IndicatorData(
            timestamp=datetime.now(UTC),
            cipher_a_dot=-1.0,
            cipher_b_wave=-3.0,
            cipher_b_money_flow=30.0,
            rsi=35.0,
            ema_fast=2475.0,
            ema_slow=2480.0
        ),
        current_position=Position(
            symbol="ETH-USD",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("2490.00"),
            timestamp=datetime.now(UTC)
        )
    )
    
    action_4 = await agent.analyze_market(market_state_4)
    print(f"LLM Decision: {action_4.action}")
    print(f"Rationale: {action_4.rationale}")
    
    print("\n=== MOMENTUM UNDERSTANDING TEST COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(test_momentum_understanding())