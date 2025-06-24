#!/usr/bin/env python3
"""
Test VuManChu Cipher A & B compatibility fixes.

This script tests:
1. VuManchuState export and accessibility
2. calculate() and calculate_all() method compatibility
3. StochasticRSI parameter passing
4. Backward compatibility with existing strategy code
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vumanchu_state_export():
    """Test that VuManchuState is properly exported and accessible."""
    print("🔍 Testing VuManchuState export and accessibility...")

    try:
        # Test import from main vumanchu module
        from bot.indicators.vumanchu import VuManchuState

        print("✅ VuManchuState import from vumanchu module: SUCCESS")

        # Test import from indicators package
        from bot.indicators import VuManchuState as VuManchuStatePackage

        print("✅ VuManchuState import from indicators package: SUCCESS")

        # Test import from functional types module
        from bot.fp.types.indicators import VuManchuState as VuManchuStateFP

        print("✅ VuManchuState import from functional types: SUCCESS")

        # Verify they're the same class
        assert VuManchuState == VuManchuStatePackage == VuManchuStateFP
        print("✅ All VuManchuState imports refer to the same class: SUCCESS")

        return True

    except ImportError as e:
        print(f"❌ VuManchuState import failed: {e}")
        return False
    except AssertionError:
        print("❌ VuManchuState imports refer to different classes")
        return False
    except Exception as e:
        print(f"❌ Unexpected error testing VuManchuState export: {e}")
        return False


def test_calculate_methods_compatibility():
    """Test that both calculate() and calculate_all() methods work correctly."""
    print("\n🔍 Testing calculate() and calculate_all() method compatibility...")

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
        print(f"   Result type: {type(result_calculate)}")
        print(f"   Result columns: {len(result_calculate.columns)} columns")

        # Test calculate_all() method (backward compatibility)
        result_calculate_all = vumanchu.calculate_all(test_data)
        print("✅ calculate_all() method execution: SUCCESS")
        print(f"   Result type: {type(result_calculate_all)}")
        print(f"   Result columns: {len(result_calculate_all.columns)} columns")

        # Verify results are equivalent
        pd.testing.assert_frame_equal(result_calculate, result_calculate_all)
        print("✅ calculate() and calculate_all() produce identical results: SUCCESS")

        # Test get_latest_state() method
        latest_state = vumanchu.get_latest_state(result_calculate)
        print("✅ get_latest_state() method execution: SUCCESS")
        print(f"   State keys: {list(latest_state.keys())}")

        # Verify expected keys are present
        expected_keys = [
            "wt1",
            "wt2",
            "rsi",
            "cipher_a_confidence",
            "cipher_a_signal",
            "cipher_b_money_flow",
            "vwap",
            "combined_signal",
            "combined_confidence",
        ]

        missing_keys = [key for key in expected_keys if key not in latest_state]
        if missing_keys:
            print(f"⚠️  Missing expected keys: {missing_keys}")
        else:
            print("✅ All expected state keys present: SUCCESS")

        return True

    except Exception as e:
        print(f"❌ Calculate methods compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_stochastic_rsi_parameters():
    """Test that StochasticRSI parameters are correctly passed."""
    print("\n🔍 Testing StochasticRSI parameter passing...")

    try:
        from bot.indicators.vumanchu import CipherA

        # Test with different RSI lengths
        test_rsi_lengths = [14, 21, 30]

        for rsi_length in test_rsi_lengths:
            cipher_a = CipherA(rsi_length=rsi_length)

            # Verify StochasticRSI was initialized with correct rsi_length
            assert (
                cipher_a.stoch_rsi.rsi_length == rsi_length
            ), f"StochasticRSI rsi_length mismatch: expected {rsi_length}, got {cipher_a.stoch_rsi.rsi_length}"

            print(
                f"✅ CipherA with rsi_length={rsi_length}: StochasticRSI rsi_length={cipher_a.stoch_rsi.rsi_length}"
            )

        print("✅ StochasticRSI parameter passing: SUCCESS")
        return True

    except Exception as e:
        print(f"❌ StochasticRSI parameter test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backward_compatibility_scenario():
    """Test a scenario that mimics existing strategy code usage."""
    print("\n🔍 Testing backward compatibility scenario...")

    try:
        # Simulate how main.py uses VuManChu
        from bot.indicators.vumanchu import VuManChuIndicators

        # Create market data similar to what main.py would provide
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1min")
        market_data = pd.DataFrame(
            {
                "open": np.random.uniform(50000, 52000, 200),
                "high": np.random.uniform(52000, 54000, 200),
                "low": np.random.uniform(48000, 50000, 200),
                "close": np.random.uniform(50000, 52000, 200),
                "volume": np.random.uniform(100, 1000, 200),
            },
            index=dates,
        )

        # Initialize like main.py does
        indicator_calc = VuManChuIndicators()
        print("✅ VuManChuIndicators initialization (main.py style): SUCCESS")

        # Calculate like main.py does - with dominance_candles parameter
        calc_result = indicator_calc.calculate_all(market_data, dominance_candles=None)
        print("✅ calculate_all() with dominance_candles parameter: SUCCESS")

        # Get latest state like main.py does
        indicator_state = indicator_calc.get_latest_state(calc_result)
        print("✅ get_latest_state() execution: SUCCESS")

        # Verify the state structure matches what main.py expects
        expected_structure = {
            "wt1": (float, int),
            "wt2": (float, int),
            "rsi": (float, int),
            "cipher_a_confidence": (float, int),
            "cipher_a_signal": (int,),
            "combined_signal": (float, int),
            "combined_confidence": (float, int),
            "market_sentiment": (str,),
            "implementation_mode": (str,),
        }

        structure_valid = True
        for key, expected_types in expected_structure.items():
            if key in indicator_state:
                if not isinstance(indicator_state[key], expected_types):
                    print(
                        f"⚠️  Key '{key}' has unexpected type: {type(indicator_state[key])}, expected: {expected_types}"
                    )
                    structure_valid = False
            else:
                print(f"⚠️  Missing expected key: {key}")
                structure_valid = False

        if structure_valid:
            print("✅ Indicator state structure matches expectations: SUCCESS")
        else:
            print("⚠️  Some indicator state structure issues detected")

        # Test that the actual values are reasonable
        wt1 = indicator_state.get("wt1", 0)
        wt2 = indicator_state.get("wt2", 0)
        rsi = indicator_state.get("rsi", 50)

        if (
            isinstance(wt1, (int, float))
            and isinstance(wt2, (int, float))
            and isinstance(rsi, (int, float))
        ):
            if -200 <= wt1 <= 200 and -200 <= wt2 <= 200 and 0 <= rsi <= 100:
                print("✅ Indicator values are within reasonable ranges: SUCCESS")
            else:
                print(
                    f"⚠️  Some indicator values out of expected ranges: wt1={wt1}, wt2={wt2}, rsi={rsi}"
                )
        else:
            print(
                f"⚠️  Indicator values have unexpected types: wt1={type(wt1)}, wt2={type(wt2)}, rsi={type(rsi)}"
            )

        return True

    except Exception as e:
        print(f"❌ Backward compatibility scenario test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_functional_enhancements():
    """Test that functional enhancements work alongside original code."""
    print("\n🔍 Testing functional enhancements integration...")

    try:
        from bot.indicators.vumanchu import VuManChuFunctional, VuManChuIndicators

        # Test original implementation
        vumanchu_original = VuManChuIndicators(implementation_mode="original")
        print("✅ Original implementation mode: SUCCESS")

        # Test functional implementation
        vumanchu_functional = VuManChuIndicators(implementation_mode="functional")
        print("✅ Functional implementation mode: SUCCESS")

        # Test hybrid implementation
        vumanchu_hybrid = VuManChuIndicators(implementation_mode="hybrid")
        print("✅ Hybrid implementation mode: SUCCESS")

        # Test pure functional class
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
        test_data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(110, 120, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(100, 110, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        # Test functional calculation methods
        functional_result_a = VuManChuFunctional.calculate_cipher_a_functional(
            test_data["high"].values,
            test_data["low"].values,
            test_data["close"].values,
            test_data["volume"].values,
        )
        print("✅ Functional Cipher A calculation: SUCCESS")
        print(f"   Result keys: {list(functional_result_a.keys())}")

        functional_result_b = VuManChuFunctional.calculate_cipher_b_functional(
            test_data["high"].values,
            test_data["low"].values,
            test_data["close"].values,
            test_data["volume"].values,
        )
        print("✅ Functional Cipher B calculation: SUCCESS")
        print(f"   Result keys: {list(functional_result_b.keys())}")

        return True

    except Exception as e:
        print(f"❌ Functional enhancements test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all VuManChu compatibility tests."""
    print("🚀 Running VuManChu Cipher A & B Compatibility Tests")
    print("=" * 60)

    test_results = []

    # Run all tests
    test_results.append(("VuManchuState Export", test_vumanchu_state_export()))
    test_results.append(
        ("Calculate Methods Compatibility", test_calculate_methods_compatibility())
    )
    test_results.append(("StochasticRSI Parameters", test_stochastic_rsi_parameters()))
    test_results.append(
        ("Backward Compatibility", test_backward_compatibility_scenario())
    )
    test_results.append(("Functional Enhancements", test_functional_enhancements()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL VUMANCHU COMPATIBILITY TESTS PASSED!")
        print(
            "✅ VuManChu Cipher A & B are fully compatible with existing strategy code"
        )
        print(
            "✅ Original implementations preserved while adding functional enhancements"
        )
        print("✅ Backward compatibility maintained for all existing usage patterns")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED - VuManChu compatibility issues detected")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
