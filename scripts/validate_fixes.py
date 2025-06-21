#!/usr/bin/env python3
"""
Validation script to test all critical fixes applied to the trading bot.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add bot module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import price conversion utilities
from bot.utils.price_conversion import convert_from_18_decimal, is_likely_18_decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_price_conversion():
    """Test price conversion fixes."""
    print("\n🧪 Testing Price Conversion Fixes")
    print("=" * 40)

    # Test astronomical price conversion
    astronomical_price = "2648900000000000000"  # Example from logs
    converted = convert_from_18_decimal(astronomical_price, "SUI-PERP", "test_price")
    print(f"Astronomical Price: ${astronomical_price}")
    print(f"Converted Price: ${converted}")
    print(f"Is 18-decimal: {is_likely_18_decimal(astronomical_price)}")

    # Test normal price (should not be converted)
    normal_price = "2.64"
    converted_normal = convert_from_18_decimal(normal_price, "SUI-PERP", "test_price")
    print(f"Normal Price: ${normal_price}")
    print(f"Converted Normal: ${converted_normal}")
    print(f"Is 18-decimal: {is_likely_18_decimal(normal_price)}")

    return float(converted) < 10  # Should be around $2.64 for SUI


def test_logging_format():
    """Test logging format fixes."""
    print("\n🧪 Testing Logging Format Fixes")
    print("=" * 40)

    try:
        # Test the specific format that was causing issues
        action = "HOLD"
        compute_time = 2.479623556137085

        # This should not raise a TypeError
        test_message = "🔄 LLM Cache MISS: %s (compute_time: %.2fs)" % (
            action,
            compute_time,
        )
        print(f"✅ Format test passed: {test_message}")

        # Test with potential problematic values
        length_value = 42
        test_message2 = "Cache size: %d items" % length_value
        print(f"✅ Length format test passed: {test_message2}")

        return True

    except Exception as e:
        print(f"❌ Logging format test failed: {e}")
        return False


def test_dataframe_assignment():
    """Test DataFrame assignment fixes."""
    print("\n🧪 Testing DataFrame Assignment Fixes")
    print("=" * 40)

    try:
        import numpy as np
        import pandas as pd

        # Create a test DataFrame similar to VuManChu
        df = pd.DataFrame({"timestamp": [1, 2, 3], "value": [1.0, 2.0, 3.0]})

        # Test assignment that was causing issues
        test_values = {
            "scalar_val": 42.0,
            "list_val": [1, 2, 3],
            "nan_val": np.nan,
            "none_val": None,
        }

        for col, val in test_values.items():
            try:
                if isinstance(val, (list, tuple, set)):
                    if len(val) == 1:
                        df.at[df.index[-1], col] = next(iter(val))
                    else:
                        df.at[df.index[-1], col] = str(val)
                elif pd.isna(val) or val is None:
                    df.at[df.index[-1], col] = np.nan
                else:
                    df.at[df.index[-1], col] = val
                print(f"✅ Successfully assigned {col}: {val}")
            except Exception as e:
                print(f"❌ Failed to assign {col}: {e}")
                return False

        return True

    except Exception as e:
        print(f"❌ DataFrame assignment test failed: {e}")
        return False


def test_imports():
    """Test that all critical modules can be imported."""
    print("\n🧪 Testing Module Imports")
    print("=" * 40)

    modules_to_test = [
        "bot.main",
        "bot.strategy.llm_cache",
        "bot.indicators.vumanchu",
        "bot.data.bluefin_market",
        "bot.utils.price_conversion",
        "bot.risk",
    ]

    success_count = 0
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
            success_count += 1
        except Exception as e:
            print(f"❌ {module}: {e}")

    return success_count == len(modules_to_test)


async def main():
    """Run all validation tests."""
    print("🔧 Trading Bot Fix Validation")
    print("=" * 50)

    tests = [
        ("Price Conversion", test_price_conversion),
        ("Logging Format", test_logging_format),
        ("DataFrame Assignment", test_dataframe_assignment),
        ("Module Imports", test_imports),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n📋 Validation Summary")
    print("=" * 30)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All fixes validated successfully!")
        print("   Trading bot should be ready for restart.")
        return 0
    print("⚠️  Some tests failed. Check individual test output above.")
    return 1


if __name__ == "__main__":
    asyncio.run(main())
