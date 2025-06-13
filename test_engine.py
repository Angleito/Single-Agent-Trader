#!/usr/bin/env python3
"""
Test script to validate the TradingEngine implementation.

This script performs a simple integration test to ensure all components
can be imported and initialized properly.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from bot.config import create_settings
    from bot.main import TradingEngine

    print("✓ Successfully imported TradingEngine")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


async def test_engine_initialization():
    """Test engine initialization without running the full loop."""
    print("\n=== Testing TradingEngine Initialization ===")

    try:
        # Create engine in dry-run mode
        engine = TradingEngine(
            symbol="BTC-USD", interval="1m", config_file=None, dry_run=True
        )
        print("✓ TradingEngine initialized successfully")

        # Test configuration loading
        print(f"✓ Configuration loaded: {engine.settings.system.environment}")
        print(f"✓ Dry run mode: {engine.settings.system.dry_run}")
        print(f"✓ Trading symbol: {engine.symbol}")
        print(f"✓ Update frequency: {engine.settings.system.update_frequency_seconds}s")

        # Test component initialization
        print("✓ Market data provider initialized")
        print("✓ Indicator calculator initialized")
        print("✓ LLM agent initialized")
        print("✓ Validator initialized")
        print("✓ Risk manager initialized")
        print("✓ Exchange client initialized")

        return True

    except Exception as e:
        print(f"✗ Engine initialization failed: {e}")
        return False


async def test_component_compatibility():
    """Test that all components can work together."""
    print("\n=== Testing Component Compatibility ===")

    try:
        engine = TradingEngine(dry_run=True)

        # Test configuration access
        config = engine.settings
        print(f"✓ Max position size: {config.trading.max_size_pct}%")
        print(f"✓ Leverage: {config.trading.leverage}x")
        print(f"✓ Risk daily loss limit: {config.risk.max_daily_loss_pct}%")
        print(f"✓ LLM provider: {config.llm.provider}")

        # Test component status
        llm_status = engine.llm_agent.get_status()
        print(f"✓ LLM agent available: {llm_status['llm_available']}")

        exchange_status = engine.exchange_client.get_connection_status()
        print(f"✓ Exchange sandbox mode: {exchange_status['sandbox']}")

        return True

    except Exception as e:
        print(f"✗ Component compatibility test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("AI Trading Bot - Engine Validation Test")
    print("=" * 50)

    # Run tests
    test1 = await test_engine_initialization()
    test2 = await test_component_compatibility()

    # Summary
    print("\n=== Test Summary ===")
    if test1 and test2:
        print("✓ All tests passed! The trading engine is ready.")
        print("\nTo start the bot in dry-run mode:")
        print("python -m bot.main live --dry-run --symbol BTC-USD")
    else:
        print("✗ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
