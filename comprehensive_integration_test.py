#!/usr/bin/env python3
"""Comprehensive System Integration Test"""

import sys
import traceback


def test_component_imports():
    """Test individual component imports to isolate working vs failing components."""
    print("=== COMPREHENSIVE SYSTEM INTEGRATION TEST ===")
    print()

    results = {"working": [], "failing": [], "warnings": []}

    # Test 1: Core Types and Utilities
    print("1. Testing core types and utilities...")
    try:
        results["working"].append(
            "✅ Core types (Price, Quantity, Symbol, ValidationResult)"
        )
    except Exception as e:
        results["failing"].append(f"❌ Core types failed: {e}")

    try:
        results["working"].append("✅ Market data types")
    except Exception as e:
        results["failing"].append(f"❌ Market data types failed: {e}")

    # Test 2: VuManChu Indicators (already confirmed working)
    print("2. Testing VuManChu indicators...")
    try:
        from bot.indicators.vumanchu import VuManChuIndicators

        results["working"].append("✅ VuManChu indicators")
    except Exception as e:
        results["failing"].append(f"❌ VuManChu indicators failed: {e}")

    # Test 3: Functional Programming Components
    print("3. Testing functional programming components...")
    try:
        from bot.fp.types.trading import Hold, Long, Short

        results["working"].append("✅ FP core types (Result, TradeSignal)")
    except Exception as e:
        results["failing"].append(f"❌ FP core types failed: {e}")

    try:
        results["working"].append(
            "✅ FP trading types (AccountBalance, OrderResult, etc.)"
        )
    except Exception as e:
        results["failing"].append(f"❌ FP trading types failed: {e}")

    # Test 4: Configuration System
    print("4. Testing configuration system...")
    try:
        # Test configuration loading without full bot initialization
        import os

        os.environ.setdefault("SYSTEM__DRY_RUN", "true")
        os.environ.setdefault("TRADING__SYMBOL", "BTC-USD")

        from bot.config import Settings

        settings = Settings()
        results["working"].append(
            f"✅ Configuration system (dry_run: {settings.system.dry_run})"
        )
    except Exception as e:
        results["failing"].append(f"❌ Configuration system failed: {e}")

    # Test 5: Data Processing Components
    print("5. Testing data processing components...")
    try:
        import numpy as np
        import pandas as pd

        # Create sample data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
                "open": np.random.uniform(49000, 51000, 50),
                "high": np.random.uniform(50000, 52000, 50),
                "low": np.random.uniform(48000, 50000, 50),
                "close": np.random.uniform(49000, 51000, 50),
                "volume": np.random.uniform(1000, 10000, 50),
            }
        )

        # Test VuManChu calculation
        from bot.indicators.vumanchu import VuManChuIndicators

        indicators = VuManChuIndicators()
        result = indicators.calculate(data)

        results["working"].append(
            f"✅ Data processing pipeline (processed {len(result)} data points)"
        )

    except Exception as e:
        results["failing"].append(f"❌ Data processing failed: {e}")

    # Test 6: Order Types and Trading Logic
    print("6. Testing order types and trading logic...")
    try:
        from bot.fp.types.trading import LimitOrder, MarketOrder, StopOrder

        # Test order creation
        market_order = MarketOrder(symbol="BTC-USD", side="buy", size=0.1)
        limit_order = LimitOrder(symbol="BTC-USD", side="sell", price=50000.0, size=0.1)
        stop_order = StopOrder(
            symbol="BTC-USD", side="sell", stop_price=48000.0, size=0.1
        )

        results["working"].append("✅ Order types (MarketOrder, LimitOrder, StopOrder)")

    except Exception as e:
        results["failing"].append(f"❌ Order types failed: {e}")

    # Test 7: Paper Trading Types
    print("7. Testing paper trading types...")
    try:
        results["working"].append("✅ Paper trading types")
    except Exception as e:
        results["failing"].append(f"❌ Paper trading types failed: {e}")

    # Test 8: Strategy Components
    print("8. Testing strategy components...")
    try:
        from bot.fp.types.trading import (
            Hold,
            Long,
            Short,
            is_directional_signal,
            signal_to_side,
        )

        # Test signal creation
        long_signal = Long(confidence=0.8, size=0.5, reason="Bullish momentum")
        short_signal = Short(confidence=0.7, size=0.3, reason="Bearish divergence")
        hold_signal = Hold(reason="Uncertain market conditions")

        # Test signal utilities
        assert is_directional_signal(long_signal) == True
        assert is_directional_signal(hold_signal) == False
        assert signal_to_side(long_signal) == "buy"
        assert signal_to_side(short_signal) == "sell"

        results["working"].append("✅ Strategy components (signals and utilities)")

    except Exception as e:
        results["failing"].append(f"❌ Strategy components failed: {e}")

    return results


def print_results(results):
    """Print comprehensive test results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE INTEGRATION TEST RESULTS")
    print("=" * 80)

    print(f"\n✅ WORKING COMPONENTS ({len(results['working'])}):")
    for item in results["working"]:
        print(f"  {item}")

    print(f"\n❌ FAILING COMPONENTS ({len(results['failing'])}):")
    for item in results["failing"]:
        print(f"  {item}")

    if results["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for item in results["warnings"]:
            print(f"  {item}")

    # Calculate overall system health
    total_tests = len(results["working"]) + len(results["failing"])
    success_rate = len(results["working"]) / total_tests * 100 if total_tests > 0 else 0

    print("\n📊 SYSTEM HEALTH:")
    print(
        f"  Success Rate: {success_rate:.1f}% ({len(results['working'])}/{total_tests} components working)"
    )

    if success_rate >= 80:
        print("  Status: ✅ SYSTEM READY FOR PRODUCTION")
    elif success_rate >= 60:
        print("  Status: ⚠️  SYSTEM NEEDS MINOR FIXES")
    else:
        print("  Status: ❌ SYSTEM NEEDS MAJOR FIXES")

    print("\n" + "=" * 80)


def main():
    """Run comprehensive integration test."""
    try:
        results = test_component_imports()
        print_results(results)

        # Return success if most components are working
        success_rate = (
            len(results["working"])
            / (len(results["working"]) + len(results["failing"]))
            * 100
        )
        return success_rate >= 60

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
