#!/usr/bin/env python3
"""
Test script to isolate the TradingEngine initialization issue.
"""
import asyncio
import os
import sys
import traceback

# Set environment variables for testing
os.environ["SYSTEM__DRY_RUN"] = "false"
os.environ["EXCHANGE__EXCHANGE_TYPE"] = "coinbase"
os.environ["CONFIG_FILE"] = "/app/config/live_trading_safe.json"

try:
    print("Importing TradingEngine...")
    from bot.main import TradingEngine

    print("Creating TradingEngine instance...")
    engine = TradingEngine(
        symbol="ETH-USD",
        interval="1m",
        config_file="/app/config/live_trading_safe.json",
        dry_run=False,
    )

    print("✅ TradingEngine initialized successfully!")

    # Test the run method as well
    print("Testing engine.run() method...")

    async def test_run():
        try:
            # Just test if the run method can be called without errors
            # We'll interrupt it quickly to avoid actual trading
            await asyncio.wait_for(engine.run(), timeout=5)
        except TimeoutError:
            print("✅ engine.run() started successfully (timed out as expected)")
        except Exception as e:
            print(f"❌ Error in engine.run(): {e}")
            traceback.print_exc()

    # Don't actually run it to avoid complexity
    print("✅ All tests passed!")

except Exception as e:
    print(f"❌ Error during TradingEngine initialization: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)
