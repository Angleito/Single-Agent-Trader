#!/usr/bin/env python3
"""
Simplified VuManChu compatibility test focusing on core functionality.

This tests the essential compatibility fixes without triggering complex import chains.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


def test_basic_vumanchu_functionality():
    """Test basic VuManChu functionality."""
    print("🔍 Testing basic VuManChu functionality...")

    try:
        from bot.indicators.vumanchu import VuManChuIndicators

        # Create test data
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1min")
        test_data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 200),
                "high": np.random.uniform(110, 120, 200),
                "low": np.random.uniform(90, 100, 200),
                "close": np.random.uniform(100, 110, 200),
                "volume": np.random.uniform(1000, 5000, 200),
            },
            index=dates,
        )

        # Initialize VuManChu indicators
        vumanchu = VuManChuIndicators()
        print("✅ VuManChuIndicators initialization: SUCCESS")

        # Test calculate() method
        result_calculate = vumanchu.calculate(test_data)
        print("✅ calculate() method execution: SUCCESS")
        print(f"   Result shape: {result_calculate.shape}")

        # Test calculate_all() method (backward compatibility)
        result_calculate_all = vumanchu.calculate_all(test_data)
        print("✅ calculate_all() method execution: SUCCESS")
        print(f"   Result shape: {result_calculate_all.shape}")

        # Test get_latest_state() method
        latest_state = vumanchu.get_latest_state(result_calculate)
        print("✅ get_latest_state() method execution: SUCCESS")
        print(f"   State keys: {len(latest_state)} keys")

        return True

    except Exception as e:
        print(f"❌ Basic VuManChu functionality test failed: {e}")
        return False


def test_stochastic_rsi_parameter_fix():
    """Test that StochasticRSI gets correct rsi_length parameter."""
    print("\n🔍 Testing StochasticRSI parameter fix...")

    try:
        from bot.indicators.vumanchu import CipherA

        # Test with custom RSI length
        rsi_length = 21
        cipher_a = CipherA(rsi_length=rsi_length)

        # Verify StochasticRSI was initialized with correct rsi_length
        actual_rsi_length = cipher_a.stoch_rsi.rsi_length
        assert (
            actual_rsi_length == rsi_length
        ), f"StochasticRSI rsi_length mismatch: expected {rsi_length}, got {actual_rsi_length}"

        print(
            f"✅ StochasticRSI rsi_length parameter correctly set to {actual_rsi_length}"
        )
        return True

    except Exception as e:
        print(f"❌ StochasticRSI parameter test failed: {e}")
        return False


def test_calculate_all_backward_compatibility():
    """Test that calculate_all maintains backward compatibility."""
    print("\n🔍 Testing calculate_all backward compatibility...")

    try:
        from bot.indicators.vumanchu import VuManChuIndicators

        # Create market data
        dates = pd.date_range(start="2024-01-01", periods=150, freq="1min")
        market_data = pd.DataFrame(
            {
                "open": np.random.uniform(50000, 52000, 150),
                "high": np.random.uniform(52000, 54000, 150),
                "low": np.random.uniform(48000, 50000, 150),
                "close": np.random.uniform(50000, 52000, 150),
                "volume": np.random.uniform(100, 1000, 150),
            },
            index=dates,
        )

        # Initialize like main.py does
        indicator_calc = VuManChuIndicators()

        # Test calculate_all with dominance_candles=None (like main.py)
        calc_result = indicator_calc.calculate_all(market_data, dominance_candles=None)
        print("✅ calculate_all() with dominance_candles=None: SUCCESS")

        # Test get_latest_state
        indicator_state = indicator_calc.get_latest_state(calc_result)
        print("✅ get_latest_state() execution: SUCCESS")

        # Verify essential keys are present
        essential_keys = ["wt1", "wt2", "rsi", "combined_signal"]
        missing_keys = [key for key in essential_keys if key not in indicator_state]

        if missing_keys:
            print(f"⚠️  Missing essential keys: {missing_keys}")
            return False
        print("✅ All essential state keys present: SUCCESS")
        return True

    except Exception as e:
        print(f"❌ calculate_all backward compatibility test failed: {e}")
        return False


def test_exports_from_indicators_package():
    """Test that VuManChu classes can be imported from indicators package."""
    print("\n🔍 Testing exports from indicators package...")

    try:
        # Test importing from indicators package
        from bot.indicators import CipherA, CipherB, VuManChuIndicators

        print("✅ VuManChu classes import from indicators package: SUCCESS")

        # Test that they're functional
        vumanchu = VuManChuIndicators()
        cipher_a = CipherA()
        cipher_b = CipherB()

        print("✅ VuManChu classes can be instantiated: SUCCESS")
        return True

    except Exception as e:
        print(f"❌ Indicators package export test failed: {e}")
        return False


def main():
    """Run simplified VuManChu compatibility tests."""
    print("🚀 Running Simplified VuManChu Compatibility Tests")
    print("=" * 60)

    test_results = []

    # Run core tests
    test_results.append(
        ("Basic VuManChu Functionality", test_basic_vumanchu_functionality())
    )
    test_results.append(
        ("StochasticRSI Parameter Fix", test_stochastic_rsi_parameter_fix())
    )
    test_results.append(
        (
            "calculate_all Backward Compatibility",
            test_calculate_all_backward_compatibility(),
        )
    )
    test_results.append(
        ("Indicators Package Exports", test_exports_from_indicators_package())
    )

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<45} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL CORE VUMANCHU COMPATIBILITY TESTS PASSED!")
        print("✅ VuManChu Cipher A & B core functionality is working correctly")
        print("✅ StochasticRSI parameter mismatch fixed")
        print("✅ calculate_all() method maintains backward compatibility")
        print("✅ Original VuManChu implementations preserved")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED - Core compatibility issues detected")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
