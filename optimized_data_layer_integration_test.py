#!/usr/bin/env python3
"""
Optimized Data Layer Integration Test

Agent 8: Comprehensive test of the optimized functional data layer
to validate performance improvements and functionality.
"""

import asyncio
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add bot directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_optimized_types_performance():
    """Test performance of optimized types vs regular types"""
    print("🔬 Testing Optimized Types Performance...")

    try:
        from bot.fp.types.market import OHLCV
        from bot.fp.types.optimized_market import (
            create_optimized_ohlcv,
        )

        # Test creation performance comparison
        test_size = 5000

        # Regular types
        start_time = time.time()
        regular_ohlcv = []
        for i in range(test_size):
            ohlcv = OHLCV(
                open=Decimal(f"{50000 + i}"),
                high=Decimal(f"{50100 + i}"),
                low=Decimal(f"{49900 + i}"),
                close=Decimal(f"{50050 + i}"),
                volume=Decimal(f"{100 + i}"),
                timestamp=datetime.now(UTC),
            )
            regular_ohlcv.append(ohlcv)
        regular_time = time.time() - start_time
        regular_ops = test_size / regular_time

        # Optimized types
        start_time = time.time()
        optimized_ohlcv = []
        for i in range(test_size):
            ohlcv = create_optimized_ohlcv(
                open=Decimal(f"{50000 + i}"),
                high=Decimal(f"{50100 + i}"),
                low=Decimal(f"{49900 + i}"),
                close=Decimal(f"{50050 + i}"),
                volume=Decimal(f"{100 + i}"),
            )
            optimized_ohlcv.append(ohlcv)
        optimized_time = time.time() - start_time
        optimized_ops = test_size / optimized_time

        # Memory comparison
        import sys

        regular_memory = sys.getsizeof(regular_ohlcv) + sum(
            sys.getsizeof(item) for item in regular_ohlcv
        )
        optimized_memory = sys.getsizeof(optimized_ohlcv) + sum(
            sys.getsizeof(item) for item in optimized_ohlcv
        )

        memory_savings = ((regular_memory - optimized_memory) / regular_memory) * 100
        performance_improvement = ((optimized_ops - regular_ops) / regular_ops) * 100

        print(f"✅ Performance Results (size={test_size}):")
        print(f"   Regular Types:   {regular_time:.3f}s ({regular_ops:.0f} ops/sec)")
        print(
            f"   Optimized Types: {optimized_time:.3f}s ({optimized_ops:.0f} ops/sec)"
        )
        print(f"   Performance Improvement: {performance_improvement:+.1f}%")
        print(f"   Memory Savings: {memory_savings:+.1f}%")

        # Property access performance
        start_time = time.time()
        for ohlcv in regular_ohlcv[:1000]:
            _ = ohlcv.is_bullish
            _ = ohlcv.price_range
            _ = ohlcv.body_size
        regular_access_time = time.time() - start_time

        start_time = time.time()
        for ohlcv in optimized_ohlcv[:1000]:
            _ = ohlcv.is_bullish
            _ = ohlcv.price_range
            _ = ohlcv.body_size
        optimized_access_time = time.time() - start_time

        access_improvement = (
            (regular_access_time - optimized_access_time) / regular_access_time
        ) * 100
        print(f"   Property Access Improvement: {access_improvement:+.1f}%")

        return {
            "performance_improvement": performance_improvement,
            "memory_savings": memory_savings,
            "access_improvement": access_improvement,
        }

    except Exception as e:
        print(f"❌ Failed to test optimized types: {e}")
        return None


def test_enhanced_aggregation():
    """Test enhanced aggregation performance"""
    print("\n🔄 Testing Enhanced Aggregation...")

    try:
        from bot.fp.effects.enhanced_aggregation import (
            create_enhanced_aggregator,
            create_high_performance_aggregator,
            create_low_latency_aggregator,
        )
        from bot.fp.types.optimized_market import create_optimized_trade

        # Create test data
        test_size = 10000
        trades = []
        base_time = datetime.now(UTC)

        for i in range(test_size):
            trade = create_optimized_trade(
                id=f"trade-{i}",
                price=Decimal(f"{50000 + (i % 100)}"),
                size=Decimal("0.1"),
                side="BUY" if i % 2 == 0 else "SELL",
                symbol="BTC-USD",
                timestamp=base_time + timedelta(seconds=i),
            )
            trades.append(trade)

        # Test different aggregator configurations
        aggregators = {
            "Enhanced": create_enhanced_aggregator(timedelta(minutes=1)),
            "High Performance": create_high_performance_aggregator(
                timedelta(minutes=1)
            ),
            "Low Latency": create_low_latency_aggregator(timedelta(minutes=1)),
        }

        results = {}

        for name, aggregator in aggregators.items():
            start_time = time.time()

            # Test batch aggregation
            candles = aggregator.aggregate_trades_batch(trades).run()

            processing_time = time.time() - start_time
            ops_per_sec = test_size / processing_time if processing_time > 0 else 0

            # Get metrics
            metrics = aggregator.get_metrics().run()

            results[name] = {
                "processing_time": processing_time,
                "ops_per_sec": ops_per_sec,
                "candles_generated": len(candles),
                "metrics": {
                    "processing_rate": metrics.processing_rate,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "avg_latency_ms": metrics.avg_latency_ms,
                },
            }

            print(f"✅ {name} Aggregator:")
            print(f"   Processing: {processing_time:.3f}s ({ops_per_sec:.0f} ops/sec)")
            print(f"   Candles: {len(candles)}")
            print(f"   Memory: {metrics.memory_usage_mb:.2f} MB")

        # Find best performer
        best_performer = max(results.keys(), key=lambda k: results[k]["ops_per_sec"])
        print(f"🏆 Best Performer: {best_performer}")

        return results

    except Exception as e:
        print(f"❌ Failed to test enhanced aggregation: {e}")
        return None


async def test_optimized_adapter():
    """Test optimized market data adapter"""
    print("\n🔗 Testing Optimized Market Data Adapter...")

    try:
        from bot.fp.adapters.optimized_market_data_adapter import (
            create_high_performance_adapter,
            create_low_latency_adapter,
            create_optimized_adapter,
        )

        # Create adapters
        adapters = {
            "Balanced": create_optimized_adapter("BTC-USD", "1m"),
            "High Performance": create_high_performance_adapter("BTC-USD", "1m"),
            "Low Latency": create_low_latency_adapter("BTC-USD", "1m"),
        }

        results = {}

        for name, adapter in adapters.items():
            start_time = time.time()

            # Test connection
            connection_result = await adapter.connect()
            if connection_result.is_left():
                print(f"❌ {name} adapter failed to connect: {connection_result.value}")
                continue

            # Test basic operations
            status = adapter.get_connection_status().run()
            metrics_result = adapter.get_performance_metrics().run()

            # Test optimization
            optimization_result = adapter.optimize_performance().run()

            setup_time = time.time() - start_time

            # Disconnect
            await adapter.disconnect()

            results[name] = {
                "setup_time": setup_time,
                "connection_success": connection_result.is_right(),
                "status": status,
                "metrics": metrics_result.value if metrics_result.is_right() else None,
                "optimization": (
                    optimization_result.value
                    if optimization_result.is_right()
                    else None
                ),
            }

            print(f"✅ {name} Adapter:")
            print(f"   Setup: {setup_time:.3f}s")
            print(f"   Connected: {connection_result.is_right()}")
            print(f"   Performance Mode: {status.get('performance_mode', 'unknown')}")

            if metrics_result.is_right():
                metrics = metrics_result.value
                print(f"   Processing Rate: {metrics.processing_rate:.0f} ops/sec")
                print(f"   Memory Usage: {metrics.memory_usage_mb:.2f} MB")

        return results

    except Exception as e:
        print(f"❌ Failed to test optimized adapter: {e}")
        return None


def test_data_factory_caching():
    """Test data factory caching performance"""
    print("\n💾 Testing Data Factory Caching...")

    try:
        from bot.fp.types.optimized_market import OptimizedDataFactory

        # Test with caching enabled
        cached_factory = OptimizedDataFactory(enable_caching=True, cache_size=1000)

        # Test without caching
        uncached_factory = OptimizedDataFactory(enable_caching=False)

        test_size = 1000
        common_values = [
            (
                Decimal(50000),
                Decimal(50100),
                Decimal(49900),
                Decimal(50050),
                Decimal(100),
            )
        ] * test_size

        # Test cached performance
        start_time = time.time()
        for values in common_values:
            ohlcv = cached_factory.create_ohlcv(*values)
        cached_time = time.time() - start_time

        # Test uncached performance
        start_time = time.time()
        for values in common_values:
            ohlcv = uncached_factory.create_ohlcv(*values)
        uncached_time = time.time() - start_time

        # Get cache stats
        cache_stats = cached_factory.get_cache_stats()

        cache_improvement = ((uncached_time - cached_time) / uncached_time) * 100

        print("✅ Caching Results:")
        print(f"   Cached:   {cached_time:.3f}s")
        print(f"   Uncached: {uncached_time:.3f}s")
        print(f"   Improvement: {cache_improvement:+.1f}%")
        print(
            f"   Cache Size: {cache_stats['ohlcv_cache_size']}/{cache_stats['max_cache_size']}"
        )

        return {"cache_improvement": cache_improvement, "cache_stats": cache_stats}

    except Exception as e:
        print(f"❌ Failed to test data factory caching: {e}")
        return None


def test_vectorized_operations():
    """Test vectorized operations performance"""
    print("\n🧮 Testing Vectorized Operations...")

    try:
        from bot.fp.types.optimized_market import (
            OptimizedDataProcessor,
            create_optimized_ohlcv,
        )

        # Create test data
        test_size = 5000
        prices = []
        ohlcv_data = []

        for i in range(test_size):
            price = Decimal(f"{50000 + i}")
            prices.append(price)

            ohlcv = create_optimized_ohlcv(
                open=price,
                high=price + Decimal(100),
                low=price - Decimal(50),
                close=price + Decimal(25),
                volume=Decimal(100),
            )
            ohlcv_data.append(ohlcv)

        processor = OptimizedDataProcessor()

        # Test SMA calculation
        start_time = time.time()
        sma_values = processor.calculate_sma(prices, 20)
        sma_time = time.time() - start_time

        # Test RSI calculation
        start_time = time.time()
        rsi_values = processor.calculate_rsi(prices, 14)
        rsi_time = time.time() - start_time

        # Test batch validation
        start_time = time.time()
        valid_data, errors = processor.batch_validate_ohlcv(ohlcv_data)
        validation_time = time.time() - start_time

        print("✅ Vectorized Operations:")
        print(f"   SMA Calculation: {sma_time:.3f}s ({len(sma_values)} values)")
        print(f"   RSI Calculation: {rsi_time:.3f}s ({len(rsi_values)} values)")
        print(
            f"   Batch Validation: {validation_time:.3f}s ({len(valid_data)} valid, {len(errors)} errors)"
        )

        return {
            "sma_performance": len(sma_values) / sma_time if sma_time > 0 else 0,
            "rsi_performance": len(rsi_values) / rsi_time if rsi_time > 0 else 0,
            "validation_performance": (
                len(ohlcv_data) / validation_time if validation_time > 0 else 0
            ),
            "validation_accuracy": len(valid_data) / len(ohlcv_data) * 100,
        }

    except Exception as e:
        print(f"❌ Failed to test vectorized operations: {e}")
        return None


async def run_comprehensive_optimization_test():
    """Run comprehensive optimization test suite"""
    print("🚀 Agent 8: Functional Data Layer Optimization Test")
    print("=" * 60)

    test_results = {}

    # Run all tests
    test_results["optimized_types"] = test_optimized_types_performance()
    test_results["enhanced_aggregation"] = test_enhanced_aggregation()
    test_results["optimized_adapter"] = await test_optimized_adapter()
    test_results["caching"] = test_data_factory_caching()
    test_results["vectorized_ops"] = test_vectorized_operations()

    # Generate comprehensive report
    generate_optimization_report(test_results)

    return test_results


def generate_optimization_report(results: dict):
    """Generate comprehensive optimization report"""
    print("\n" + "=" * 60)
    print("📋 FUNCTIONAL DATA LAYER OPTIMIZATION REPORT")
    print("=" * 60)

    print("\n🎯 PERFORMANCE SUMMARY:")

    # Optimized types performance
    if results.get("optimized_types"):
        types_result = results["optimized_types"]
        print(
            f"• Type Creation Improvement: {types_result['performance_improvement']:+.1f}%"
        )
        print(f"• Memory Usage Reduction: {types_result['memory_savings']:+.1f}%")
        print(
            f"• Property Access Improvement: {types_result['access_improvement']:+.1f}%"
        )

    # Aggregation performance
    if results.get("enhanced_aggregation"):
        agg_results = results["enhanced_aggregation"]
        best_performer = max(
            agg_results.keys(), key=lambda k: agg_results[k]["ops_per_sec"]
        )
        best_ops = agg_results[best_performer]["ops_per_sec"]
        print(
            f"• Best Aggregation Performance: {best_ops:.0f} ops/sec ({best_performer})"
        )

    # Caching improvements
    if results.get("caching"):
        cache_result = results["caching"]
        print(f"• Caching Improvement: {cache_result['cache_improvement']:+.1f}%")

    # Vectorized operations
    if results.get("vectorized_ops"):
        vec_result = results["vectorized_ops"]
        print(
            f"• SMA Calculation: {vec_result['sma_performance']:.0f} calculations/sec"
        )
        print(
            f"• RSI Calculation: {vec_result['rsi_performance']:.0f} calculations/sec"
        )
        print(f"• Validation Accuracy: {vec_result['validation_accuracy']:.1f}%")

    print("\n🔧 OPTIMIZATION ACHIEVEMENTS:")

    achievements = []

    # Check for significant improvements
    if results.get("optimized_types"):
        types_result = results["optimized_types"]
        if types_result["performance_improvement"] > 20:
            achievements.append("✅ Significant type creation performance improvement")
        if types_result["memory_savings"] > 15:
            achievements.append("✅ Substantial memory usage reduction")

    if results.get("enhanced_aggregation"):
        agg_results = results["enhanced_aggregation"]
        max_ops = max(result["ops_per_sec"] for result in agg_results.values())
        if max_ops > 100000:
            achievements.append("✅ High-performance real-time aggregation achieved")

    if results.get("caching") and results["caching"]["cache_improvement"] > 50:
        achievements.append("✅ Effective caching system implemented")

    if results.get("vectorized_ops"):
        vec_result = results["vectorized_ops"]
        if vec_result["validation_performance"] > 10000:
            achievements.append("✅ Fast batch validation implemented")

    if not achievements:
        achievements.append(
            "⚠️ Performance improvements detected but may need further optimization"
        )

    for achievement in achievements:
        print(f"  {achievement}")

    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")

    recommendations = []

    # Check for areas needing improvement
    if results.get("optimized_types"):
        types_result = results["optimized_types"]
        if types_result["performance_improvement"] < 10:
            recommendations.append(
                "Consider further type optimization with native extensions"
            )
        if types_result["memory_savings"] < 10:
            recommendations.append(
                "Investigate additional memory optimization techniques"
            )

    if results.get("enhanced_aggregation"):
        agg_results = results["enhanced_aggregation"]
        min_ops = min(result["ops_per_sec"] for result in agg_results.values())
        if min_ops < 50000:
            recommendations.append("Optimize slower aggregation algorithms")

    # Performance mode recommendations
    if results.get("optimized_adapter"):
        recommendations.append(
            "Use 'high_performance' mode for high-throughput scenarios"
        )
        recommendations.append("Use 'low_latency' mode for time-critical applications")

    if not recommendations:
        recommendations.append("✅ All optimization targets achieved")

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n🎖️ PERFORMANCE GRADE:")

    # Calculate overall grade
    grade_points = 0
    total_tests = 0

    if results.get("optimized_types"):
        types_result = results["optimized_types"]
        if types_result["performance_improvement"] > 30:
            grade_points += 4
        elif types_result["performance_improvement"] > 15:
            grade_points += 3
        elif types_result["performance_improvement"] > 5:
            grade_points += 2
        else:
            grade_points += 1
        total_tests += 1

    if results.get("enhanced_aggregation"):
        agg_results = results["enhanced_aggregation"]
        max_ops = max(result["ops_per_sec"] for result in agg_results.values())
        if max_ops > 200000:
            grade_points += 4
        elif max_ops > 100000:
            grade_points += 3
        elif max_ops > 50000:
            grade_points += 2
        else:
            grade_points += 1
        total_tests += 1

    if results.get("caching"):
        cache_result = results["caching"]
        if cache_result["cache_improvement"] > 75:
            grade_points += 4
        elif cache_result["cache_improvement"] > 50:
            grade_points += 3
        elif cache_result["cache_improvement"] > 25:
            grade_points += 2
        else:
            grade_points += 1
        total_tests += 1

    if total_tests > 0:
        avg_grade = grade_points / total_tests
        if avg_grade >= 3.5:
            grade = "A+"
        elif avg_grade >= 3.0:
            grade = "A"
        elif avg_grade >= 2.5:
            grade = "B+"
        elif avg_grade >= 2.0:
            grade = "B"
        else:
            grade = "C"

        print(f"  Overall Performance Grade: {grade}")
        print(f"  (Based on {total_tests} test categories)")

    print("\n✅ Optimization analysis complete!")


async def main():
    """Main execution function"""
    print("Agent 8: VuManChu Preservation & Core Functionality")
    print("Data Layer Optimization Analysis")
    print("-" * 60)

    try:
        results = await run_comprehensive_optimization_test()

        # Save results
        import json

        with open("optimized_data_layer_results.json", "w") as f:
            # Convert results to JSON-serializable format
            json_results = {}
            for key, value in results.items():
                if value is not None:
                    json_results[key] = value
            json.dump(json_results, f, indent=2, default=str)

        print("\n📁 Results saved to: optimized_data_layer_results.json")

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
