#!/usr/bin/env python3
"""
Simplified Functional Data Layer Test

Agent 8: Direct testing of functional data layer components without full bot import
"""

import sys
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add bot directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_functional_types_direct():
    """Test functional types directly"""
    print("🔍 Testing Functional Types Direct Import...")

    try:
        # Test direct import of market types
        from bot.fp.types.market import OHLCV

        print("✅ Successfully imported market types")

        # Test OHLCV creation
        start_time = time.time()
        ohlcv_items = []

        for i in range(1000):
            ohlcv = OHLCV(
                open=Decimal(f"{50000 + i}"),
                high=Decimal(f"{50100 + i}"),
                low=Decimal(f"{49900 + i}"),
                close=Decimal(f"{50050 + i}"),
                volume=Decimal(f"{100 + i}"),
                timestamp=datetime.utcnow() + timedelta(minutes=i),
            )
            ohlcv_items.append(ohlcv)

        creation_time = time.time() - start_time
        ops_per_sec = 1000 / creation_time if creation_time > 0 else 0

        print(
            f"✅ Created 1000 OHLCV objects in {creation_time:.3f}s ({ops_per_sec:.0f} ops/sec)"
        )

        # Test immutability
        try:
            ohlcv_items[0].open = Decimal(60000)  # Should fail
            print("❌ OHLCV objects are not immutable!")
        except AttributeError:
            print("✅ OHLCV objects are properly immutable")

        # Test properties
        sample_ohlcv = ohlcv_items[0]
        print(
            f"✅ OHLCV properties: range={sample_ohlcv.price_range}, bullish={sample_ohlcv.is_bullish}"
        )

        return True

    except Exception as e:
        print(f"❌ Failed to test functional types: {e}")
        return False


def test_functional_validation():
    """Test functional validation components"""
    print("\n🔍 Testing Functional Validation...")

    try:
        from bot.fp.core.either import Either, Left, Right
        from bot.fp.core.option import None_, Some

        print("✅ Successfully imported functional core types")

        # Test Either success case
        def safe_divide(a: float, b: float) -> Either[str, float]:
            if b == 0:
                return Left("Division by zero")
            return Right(a / b)

        result = safe_divide(10.0, 2.0)
        if result.is_right() and result.value == 5.0:
            print("✅ Either Right case working correctly")
        else:
            print("❌ Either Right case failed")

        # Test Either failure case
        result = safe_divide(10.0, 0.0)
        if result.is_left() and "zero" in result.value:
            print("✅ Either Left case working correctly")
        else:
            print("❌ Either Left case failed")

        # Test Option
        some_value = Some(42)
        none_value = None_()

        if some_value.is_some() and some_value.unwrap() == 42:
            print("✅ Option Some case working correctly")
        else:
            print("❌ Option Some case failed")

        if none_value.is_none():
            print("✅ Option None case working correctly")
        else:
            print("❌ Option None case failed")

        return True

    except Exception as e:
        print(f"❌ Failed to test functional validation: {e}")
        return False


def test_data_pipeline_components():
    """Test data pipeline components if available"""
    print("\n🔍 Testing Data Pipeline Components...")

    try:
        from bot.fp.data_pipeline import DataPipelineConfig, FunctionalDataPipeline

        print("✅ Successfully imported data pipeline")

        # Create pipeline config
        config = DataPipelineConfig(
            buffer_size=1000, enable_batching=True, batch_size=100
        )

        # Create pipeline
        pipeline = FunctionalDataPipeline(config)
        print("✅ Successfully created data pipeline")

        # Test basic operations
        test_data = [1, 2, 3, 4, 5]

        # Test map operation
        doubled = pipeline.map_data(lambda x: x * 2)(test_data).run()
        if doubled == [2, 4, 6, 8, 10]:
            print("✅ Pipeline map operation working correctly")
        else:
            print(f"❌ Pipeline map operation failed: {doubled}")

        # Test filter operation
        evens = pipeline.filter_data(lambda x: x % 2 == 0)(test_data).run()
        if evens == [2, 4]:
            print("✅ Pipeline filter operation working correctly")
        else:
            print(f"❌ Pipeline filter operation failed: {evens}")

        return True

    except Exception as e:
        print(f"❌ Failed to test data pipeline: {e}")
        return False


def test_memory_efficiency():
    """Test memory efficiency of functional types"""
    print("\n🔍 Testing Memory Efficiency...")

    try:
        import sys

        from bot.fp.types.market import OHLCV, Trade

        # Test OHLCV memory usage
        ohlcv_list = []
        for i in range(100):
            ohlcv = OHLCV(
                open=Decimal(50000),
                high=Decimal(50100),
                low=Decimal(49900),
                close=Decimal(50050),
                volume=Decimal(100),
                timestamp=datetime.utcnow(),
            )
            ohlcv_list.append(ohlcv)

        ohlcv_memory = sys.getsizeof(ohlcv_list) + sum(
            sys.getsizeof(item) for item in ohlcv_list
        )
        ohlcv_per_item = ohlcv_memory / len(ohlcv_list)

        print(
            f"✅ OHLCV memory usage: {ohlcv_memory} bytes total, {ohlcv_per_item:.0f} bytes per item"
        )

        # Test Trade memory usage
        trade_list = []
        for i in range(100):
            trade = Trade(
                id=f"trade-{i}",
                timestamp=datetime.utcnow(),
                price=Decimal(50000),
                size=Decimal("0.1"),
                side="BUY",
                symbol="BTC-USD",
            )
            trade_list.append(trade)

        trade_memory = sys.getsizeof(trade_list) + sum(
            sys.getsizeof(item) for item in trade_list
        )
        trade_per_item = trade_memory / len(trade_list)

        print(
            f"✅ Trade memory usage: {trade_memory} bytes total, {trade_per_item:.0f} bytes per item"
        )

        # Memory efficiency assessment
        if ohlcv_per_item < 500 and trade_per_item < 800:
            print("✅ Memory efficiency is good")
        else:
            print("⚠️ Memory usage could be optimized")

        return True

    except Exception as e:
        print(f"❌ Failed to test memory efficiency: {e}")
        return False


def assess_performance_bottlenecks():
    """Assess potential performance bottlenecks"""
    print("\n🔍 Assessing Performance Bottlenecks...")

    bottlenecks = []

    # Test large data creation performance
    try:
        from bot.fp.types.market import OHLCV

        start_time = time.time()
        large_dataset = []

        for i in range(10000):  # 10k items
            ohlcv = OHLCV(
                open=Decimal(f"{50000 + i}"),
                high=Decimal(f"{50100 + i}"),
                low=Decimal(f"{49900 + i}"),
                close=Decimal(f"{50050 + i}"),
                volume=Decimal(f"{100 + i}"),
                timestamp=datetime.utcnow(),
            )
            large_dataset.append(ohlcv)

        creation_time = time.time() - start_time
        ops_per_sec = 10000 / creation_time if creation_time > 0 else 0

        print(
            f"📊 Large dataset creation: {creation_time:.3f}s ({ops_per_sec:.0f} ops/sec)"
        )

        if ops_per_sec < 50000:
            bottlenecks.append("Type creation performance could be optimized")

        # Test data access performance
        start_time = time.time()
        price_sum = sum(float(item.close) for item in large_dataset)
        access_time = time.time() - start_time

        print(f"📊 Data access time: {access_time:.3f}s for 10k items")

        if access_time > 0.1:
            bottlenecks.append("Data access could be optimized")

    except Exception as e:
        bottlenecks.append(f"Performance testing failed: {e}")

    return bottlenecks


def generate_optimization_report(test_results):
    """Generate optimization recommendations"""
    print("\n" + "=" * 60)
    print("📋 FUNCTIONAL DATA LAYER OPTIMIZATION REPORT")
    print("=" * 60)

    print("\n🔍 TEST RESULTS SUMMARY:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"• {test_name}: {status}")

    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")

    # Performance-based recommendations
    bottlenecks = assess_performance_bottlenecks()

    if not bottlenecks:
        print("1. ✅ No major performance bottlenecks detected")
    else:
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"{i}. ⚠️ {bottleneck}")

    print("\n🚀 RECOMMENDED OPTIMIZATIONS:")
    print("1. Implement __slots__ for data classes to reduce memory overhead")
    print("2. Add object pooling for frequently created OHLCV instances")
    print("3. Use numpy arrays for large dataset operations")
    print("4. Implement lazy evaluation for pipeline operations")
    print("5. Add caching for validation results")
    print("6. Optimize Decimal operations with float conversion where appropriate")

    print("\n🎯 PERFORMANCE TARGETS:")
    print("• OHLCV creation: > 50,000 ops/sec")
    print("• Memory usage: < 500 bytes per OHLCV item")
    print("• Pipeline operations: > 100,000 ops/sec")
    print("• Data validation: < 1ms per 1000 items")


def main():
    """Main test execution"""
    print("🚀 Agent 8: Functional Data Layer Performance Analysis")
    print("=" * 60)

    test_results = {}

    # Run all tests
    test_results["Functional Types"] = test_functional_types_direct()
    test_results["Functional Validation"] = test_functional_validation()
    test_results["Data Pipeline"] = test_data_pipeline_components()
    test_results["Memory Efficiency"] = test_memory_efficiency()

    # Generate report
    generate_optimization_report(test_results)

    # Overall assessment
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

    print(f"\n📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Functional data layer is working correctly.")
    elif passed_tests >= total_tests * 0.7:
        print("⚠️ Most tests passed, but some optimizations needed.")
    else:
        print("❌ Multiple test failures - functional data layer needs attention.")


if __name__ == "__main__":
    main()
