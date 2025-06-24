#!/usr/bin/env python3
"""
Test script for functional strategy adapters.

This script tests the functional strategy adapters to ensure they work correctly
before running the migration.
"""

import asyncio
import logging
import sys
from datetime import UTC, datetime
from decimal import Decimal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_market_state():
    """Create a test MarketState for testing."""
    from bot.trading_types import IndicatorData, MarketData, MarketState, Position

    # Create test OHLCV data
    test_candle = MarketData(
        timestamp=datetime.now(UTC),
        open=Decimal(50000),
        high=Decimal(51000),
        low=Decimal(49500),
        close=Decimal(50500),
        volume=Decimal("100.5"),
    )

    # Create test indicators
    test_indicators = IndicatorData(
        timestamp=datetime.now(UTC),
        rsi=55.0,
        cipher_a_dot=0.5,
        cipher_b_wave=10.0,
        cipher_b_money_flow=60.0,
        ema_fast=50200.0,
        ema_slow=50100.0,
    )

    # Create test position
    test_position = Position(
        symbol="BTC-USD",
        side="FLAT",
        size=Decimal(0),
        timestamp=datetime.now(UTC),
    )

    # Create test market state
    market_state = MarketState(
        symbol="BTC-USD",
        interval="5m",
        timestamp=datetime.now(UTC),
        current_price=Decimal(50500),
        ohlcv_data=[test_candle],
        indicators=test_indicators,
        current_position=test_position,
    )

    return market_state


async def test_llm_agent_adapter():
    """Test the LLMAgentAdapter."""
    logger.info("🧪 Testing LLMAgentAdapter")

    try:
        from bot.fp.adapters.strategy_adapter import LLMAgentAdapter

        # Initialize adapter
        agent = LLMAgentAdapter(
            model_provider="openai", model_name="gpt-4", omnisearch_client=None
        )

        # Test is_available
        assert agent.is_available() == True, "Agent should be available"
        logger.info("✅ is_available() test passed")

        # Test get_status
        status = agent.get_status()
        assert isinstance(status, dict), "Status should be a dictionary"
        assert "model_provider" in status, "Status should include model_provider"
        assert "strategy_type" in status, "Status should include strategy_type"
        assert status["strategy_type"] == "functional", "Should use functional strategy"
        logger.info("✅ get_status() test passed")

        # Test analyze_market
        market_state = create_test_market_state()
        trade_action = await agent.analyze_market(market_state)

        # Verify trade action structure
        assert hasattr(trade_action, "action"), "TradeAction should have action"
        assert hasattr(trade_action, "size_pct"), "TradeAction should have size_pct"
        assert hasattr(trade_action, "rationale"), "TradeAction should have rationale"
        assert trade_action.action in [
            "LONG",
            "SHORT",
            "CLOSE",
            "HOLD",
        ], "Action should be valid"

        logger.info(
            f"✅ analyze_market() test passed - Decision: {trade_action.action}"
        )
        logger.info(f"   Rationale: {trade_action.rationale}")

        return True

    except Exception as e:
        logger.error(f"❌ LLMAgentAdapter test failed: {e}")
        return False


async def test_memory_enhanced_agent_adapter():
    """Test the MemoryEnhancedLLMAgentAdapter."""
    logger.info("🧪 Testing MemoryEnhancedLLMAgentAdapter")

    try:
        from bot.fp.adapters.strategy_adapter import MemoryEnhancedLLMAgentAdapter

        # Initialize adapter
        agent = MemoryEnhancedLLMAgentAdapter(
            model_provider="openai",
            model_name="gpt-4",
            memory_server=None,  # No memory server for testing
            omnisearch_client=None,
        )

        # Test is_available
        assert agent.is_available() == True, "Agent should be available"
        logger.info("✅ is_available() test passed")

        # Test get_status
        status = agent.get_status()
        assert isinstance(status, dict), "Status should be a dictionary"
        assert "model_provider" in status, "Status should include model_provider"
        assert "strategy_type" in status, "Status should include strategy_type"
        assert "memory_enabled" in status, "Status should include memory_enabled"
        assert (
            status["strategy_type"] == "functional_memory_enhanced"
        ), "Should use functional memory strategy"
        logger.info("✅ get_status() test passed")

        # Test analyze_market
        market_state = create_test_market_state()
        trade_action = await agent.analyze_market(market_state)

        # Verify trade action structure
        assert hasattr(trade_action, "action"), "TradeAction should have action"
        assert hasattr(trade_action, "size_pct"), "TradeAction should have size_pct"
        assert hasattr(trade_action, "rationale"), "TradeAction should have rationale"
        assert trade_action.action in [
            "LONG",
            "SHORT",
            "CLOSE",
            "HOLD",
        ], "Action should be valid"

        # Test memory context attribute compatibility
        assert hasattr(
            agent, "_last_memory_context"
        ), "Should have _last_memory_context attribute"

        logger.info(
            f"✅ analyze_market() test passed - Decision: {trade_action.action}"
        )
        logger.info(f"   Rationale: {trade_action.rationale}")
        logger.info(f"   Memory available: {status['memory_enabled']}")

        return True

    except Exception as e:
        logger.error(f"❌ MemoryEnhancedLLMAgentAdapter test failed: {e}")
        return False


async def test_type_converter():
    """Test the TypeConverter functionality."""
    logger.info("🧪 Testing TypeConverter")

    try:
        from bot.fp.adapters.strategy_adapter import TypeConverter
        from bot.fp.strategies.llm_functional import LLMResponse
        from bot.fp.types import Signal

        # Test create_trading_params
        params = TypeConverter.create_trading_params()
        assert hasattr(params, "risk_per_trade"), "Should have risk_per_trade"
        assert hasattr(params, "max_leverage"), "Should have max_leverage"
        logger.info("✅ create_trading_params() test passed")

        # Test llm_response_to_trade_action
        market_state = create_test_market_state()

        # Create test LLMResponse
        test_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.8,
            reasoning="Test reasoning for LONG signal",
            risk_assessment={
                "risk_level": "MEDIUM",
                "key_risks": ["Market volatility"],
                "mitigation": "Use stop loss",
            },
            suggested_params={
                "position_size": 0.25,
                "stop_loss": 49000,
                "take_profit": 52000,
            },
        )

        trade_action = TypeConverter.llm_response_to_trade_action(
            test_response, market_state
        )

        assert (
            trade_action.action == "LONG"
        ), "Should convert LONG signal to LONG action"
        assert (
            trade_action.rationale == "Test reasoning for LONG signal"
        ), "Should preserve reasoning"
        assert trade_action.size_pct > 0, "Should have positive size for LONG"

        logger.info("✅ llm_response_to_trade_action() test passed")

        return True

    except Exception as e:
        logger.error(f"❌ TypeConverter test failed: {e}")
        return False


async def test_functional_llm_strategy():
    """Test the FunctionalLLMStrategy."""
    logger.info("🧪 Testing FunctionalLLMStrategy")

    try:
        from bot.fp.adapters.strategy_adapter import FunctionalLLMStrategy

        # Initialize strategy
        strategy = FunctionalLLMStrategy(
            model_provider="openai", model_name="gpt-4", omnisearch_client=None
        )

        # Test analyze_market_functional
        market_state = create_test_market_state()
        llm_response = await strategy.analyze_market_functional(market_state)

        # Verify response structure
        assert hasattr(llm_response, "signal"), "Response should have signal"
        assert hasattr(llm_response, "confidence"), "Response should have confidence"
        assert hasattr(llm_response, "reasoning"), "Response should have reasoning"
        assert hasattr(
            llm_response, "risk_assessment"
        ), "Response should have risk_assessment"

        logger.info(
            f"✅ analyze_market_functional() test passed - Signal: {llm_response.signal}"
        )
        logger.info(f"   Confidence: {llm_response.confidence}")
        logger.info(f"   Reasoning: {llm_response.reasoning}")

        return True

    except Exception as e:
        logger.error(f"❌ FunctionalLLMStrategy test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests and report results."""
    logger.info("🚀 Starting functional strategy adapter tests")
    logger.info("=" * 50)

    tests = [
        ("TypeConverter", test_type_converter()),
        ("FunctionalLLMStrategy", test_functional_llm_strategy()),
        ("LLMAgentAdapter", test_llm_agent_adapter()),
        ("MemoryEnhancedLLMAgentAdapter", test_memory_enhanced_agent_adapter()),
    ]

    results = []
    for test_name, test_coro in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    # Report results
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")

    passed = 0
    failed = 0

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1

    logger.info(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info(
            "🎉 All tests passed! Functional strategies are ready for migration."
        )
        return True
    logger.error("❌ Some tests failed. Please fix issues before migration.")
    return False


async def main():
    """Main test function."""
    success = await run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
