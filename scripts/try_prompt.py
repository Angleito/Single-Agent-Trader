#!/usr/bin/env python3
"""
Utility script to test the LLM prompt with sample data.

This script allows testing the trading prompt template and LLM agent
without running the full trading bot.
"""

import asyncio
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.llm_agent import LLMAgent
from bot.types import IndicatorData, MarketData, MarketState, Position


def create_sample_market_state() -> MarketState:
    """Create sample market state for testing."""

    # Sample OHLCV data (last 5 candles)
    sample_ohlcv = []
    base_price = 50000

    for i in range(5):
        sample_ohlcv.append(
            MarketData(
                symbol="BTC-USD",
                timestamp=datetime.utcnow(),
                open=Decimal(str(base_price + i * 100)),
                high=Decimal(str(base_price + i * 100 + 200)),
                low=Decimal(str(base_price + i * 100 - 150)),
                close=Decimal(str(base_price + i * 100 + 50)),
                volume=Decimal("25.5"),
            )
        )

    # Sample indicators
    indicators = IndicatorData(
        timestamp=datetime.utcnow(),
        cipher_a_dot=1.2,  # Bullish signal
        cipher_b_wave=2.5,  # Positive wave
        cipher_b_money_flow=65.0,  # Strong buying
        rsi=45.0,  # Neutral RSI
        ema_fast=50250.0,
        ema_slow=50100.0,
        vwap=50200.0,
    )

    # Current position (flat)
    current_position = Position(
        symbol="BTC-USD", side="FLAT", size=Decimal("0"), timestamp=datetime.utcnow()
    )

    return MarketState(
        symbol="BTC-USD",
        interval="1m",
        timestamp=datetime.utcnow(),
        current_price=Decimal("50250"),
        ohlcv_data=sample_ohlcv,
        indicators=indicators,
        current_position=current_position,
    )


def create_bearish_market_state() -> MarketState:
    """Create bearish market state for testing."""

    sample_ohlcv = []
    base_price = 50000

    for i in range(5):
        sample_ohlcv.append(
            MarketData(
                symbol="BTC-USD",
                timestamp=datetime.utcnow(),
                open=Decimal(str(base_price - i * 150)),
                high=Decimal(str(base_price - i * 150 + 100)),
                low=Decimal(str(base_price - i * 150 - 300)),
                close=Decimal(str(base_price - i * 150 - 200)),
                volume=Decimal("35.2"),
            )
        )

    indicators = IndicatorData(
        timestamp=datetime.utcnow(),
        cipher_a_dot=-0.8,  # Bearish signal
        cipher_b_wave=-1.5,  # Negative wave
        cipher_b_money_flow=35.0,  # Selling pressure
        rsi=55.0,  # Declining from high
        ema_fast=49800.0,
        ema_slow=50100.0,  # Fast below slow
        vwap=50000.0,
    )

    current_position = Position(
        symbol="BTC-USD", side="FLAT", size=Decimal("0"), timestamp=datetime.utcnow()
    )

    return MarketState(
        symbol="BTC-USD",
        interval="1m",
        timestamp=datetime.utcnow(),
        current_price=Decimal("49800"),
        ohlcv_data=sample_ohlcv,
        indicators=indicators,
        current_position=current_position,
    )


async def test_prompt_with_sample_data():
    """Test the prompt with sample market data."""

    print("🤖 AI Trading Bot - Prompt Tester")
    print("=" * 50)

    # Initialize LLM agent
    llm_agent = LLMAgent()

    print(f"📊 LLM Agent Status: {llm_agent.get_status()}")
    print(f"🔗 LLM Available: {llm_agent.is_available()}")
    print()

    # Test scenarios
    scenarios = [
        ("Bullish Market Conditions", create_sample_market_state()),
        ("Bearish Market Conditions", create_bearish_market_state()),
    ]

    for scenario_name, market_state in scenarios:
        print(f"📈 Testing Scenario: {scenario_name}")
        print("-" * 30)

        try:
            # Get trading decision
            trade_action = await llm_agent.analyze_market(market_state)

            # Display results
            print(f"Decision: {trade_action.action}")
            print(f"Size: {trade_action.size_pct}%")
            print(f"Take Profit: {trade_action.take_profit_pct}%")
            print(f"Stop Loss: {trade_action.stop_loss_pct}%")
            print(f"Rationale: {trade_action.rationale}")
            print()

            # Show formatted JSON
            print("📄 Full Response JSON:")
            print(json.dumps(trade_action.dict(), indent=2, default=str))
            print()

        except Exception as e:
            print(f"❌ Error testing scenario: {e}")
            print()

    print("✅ Prompt testing completed!")


def test_prompt_formatting():
    """Test prompt template formatting without LLM call."""

    print("🔧 Testing Prompt Template Formatting")
    print("=" * 50)

    llm_agent = LLMAgent()
    market_state = create_sample_market_state()

    # Get formatted input
    llm_input = llm_agent._prepare_llm_input(market_state)

    print("📝 Formatted Prompt Input:")
    print(json.dumps(llm_input, indent=2, default=str))
    print()

    # Show actual prompt if available
    if llm_agent.prompt_text:
        print("📄 Prompt Template:")
        print(llm_agent.prompt_text.format(**llm_input))

    print("✅ Prompt formatting test completed!")


async def main():
    """Main function."""

    if len(sys.argv) > 1 and sys.argv[1] == "--format-only":
        test_prompt_formatting()
    else:
        await test_prompt_with_sample_data()


if __name__ == "__main__":
    asyncio.run(main())
