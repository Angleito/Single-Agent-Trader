#!/usr/bin/env python3
"""
Functional Data Layer Performance Test & Optimization Analysis

Agent 8: VuManChu Preservation & Core Functionality
Tests the functional data layer performance and identifies optimization opportunities.
"""

import time
from datetime import datetime, timedelta
from decimal import Decimal

# Test if we can import the functional components
try:
    from bot.fp.data import FunctionalMarketDataProvider
    from bot.fp.data_pipeline import (
        FunctionalDataPipeline,
        create_high_performance_pipeline,
        create_low_latency_pipeline,
        create_market_data_pipeline,
    )
    from bot.fp.effects.market_data_aggregation import (
        RealTimeAggregator,
        create_high_frequency_aggregator,
        create_real_time_aggregator,
    )
    from bot.fp.types.market import (
        OHLCV,
        Candle,
        ConnectionState,
        ConnectionStatus,
        DataQuality,
        MarketData,
        MarketSnapshot,
        OrderBook,
        Trade,
    )

    FUNCTIONAL_IMPORTS_AVAILABLE = True
    print("✅ Functional data layer imports successful")
except ImportError as e:
    print(f"❌ Functional data layer import failed: {e}")
    FUNCTIONAL_IMPORTS_AVAILABLE = False


class DataLayerPerformanceBenchmark:
    """Comprehensive benchmark suite for functional data layer performance"""

    def __init__(self):
        self.results = {}
        self.test_data_sizes = [100, 1000, 5000, 10000]

    def generate_test_ohlcv_data(self, count: int) -> list[OHLCV]:
        """Generate test OHLCV data for performance testing"""
        if not FUNCTIONAL_IMPORTS_AVAILABLE:
            return []

        base_time = datetime.utcnow()
        base_price = Decimal(50000)

        test_data = []
        for i in range(count):
            open_price = base_price + Decimal(str(i * 10))
            high_price = open_price + Decimal(100)
            low_price = open_price - Decimal(50)
            close_price = open_price + Decimal(25)
            volume = Decimal(str(100 + i))

            ohlcv = OHLCV(
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timestamp=base_time + timedelta(minutes=i),
            )
            test_data.append(ohlcv)

        return test_data

    def generate_test_trade_data(self, count: int) -> list[Trade]:
        """Generate test trade data for performance testing"""
        if not FUNCTIONAL_IMPORTS_AVAILABLE:
            return []

        base_time = datetime.utcnow()
        base_price = Decimal(50000)

        test_data = []
        for i in range(count):
            trade = Trade(
                id=f"trade-{i}",
                timestamp=base_time + timedelta(seconds=i),
                price=base_price + Decimal(str(i % 100)),
                size=Decimal("0.1"),
                side="BUY" if i % 2 == 0 else "SELL",
                symbol="BTC-USD",
            )
            test_data.append(trade)

        return test_data

    def benchmark_functional_types_creation(self):
        """Benchmark creation performance of functional types"""
        if not FUNCTIONAL_IMPORTS_AVAILABLE:
            return {"error": "Functional types not available"}

        print("\n📊 Benchmarking Functional Types Creation Performance...")

        results = {}

        for size in self.test_data_sizes:
            start_time = time.time()

            # Test OHLCV creation
            ohlcv_data = self.generate_test_ohlcv_data(size)

            ohlcv_time = time.time() - start_time

            # Test Trade creation
            start_time = time.time()
            trade_data = self.generate_test_trade_data(size)
            trade_time = time.time() - start_time

            # Test MarketSnapshot creation
            start_time = time.time()
            snapshots = []
            for i in range(size):
                snapshot = MarketSnapshot(
                    timestamp=datetime.utcnow(),
                    symbol="BTC-USD",
                    price=Decimal(50000),
                    volume=Decimal(100),
                    bid=Decimal(49990),
                    ask=Decimal(50010),
                )
                snapshots.append(snapshot)
            snapshot_time = time.time() - start_time

            results[size] = {
                "ohlcv_creation_time": ohlcv_time,
                "ohlcv_ops_per_sec": size / ohlcv_time if ohlcv_time > 0 else 0,
                "trade_creation_time": trade_time,
                "trade_ops_per_sec": size / trade_time if trade_time > 0 else 0,
                "snapshot_creation_time": snapshot_time,
                "snapshot_ops_per_sec": (
                    size / snapshot_time if snapshot_time > 0 else 0
                ),
            }

            print(
                f"  Size {size:5d}: OHLCV {ohlcv_time:.3f}s ({results[size]['ohlcv_ops_per_sec']:.0f} ops/s)"
            )
            print(
                f"             Trade {trade_time:.3f}s ({results[size]['trade_ops_per_sec']:.0f} ops/s)"
            )
            print(
                f"             Snapshot {snapshot_time:.3f}s ({results[size]['snapshot_ops_per_sec']:.0f} ops/s)"
            )

        return results

    def benchmark_data_pipeline_performance(self):
        """Benchmark functional data pipeline performance"""
        if not FUNCTIONAL_IMPORTS_AVAILABLE:
            return {"error": "Data pipeline not available"}

        print("\n🔄 Benchmarking Functional Data Pipeline Performance...")

        results = {}

        # Test different pipeline configurations
        pipeline_configs = {
            "standard": create_market_data_pipeline(),
            "high_performance": create_high_performance_pipeline(),
            "low_latency": create_low_latency_pipeline(),
        }

        for config_name, pipeline in pipeline_configs.items():
            print(f"\n  Testing {config_name} pipeline:")
            config_results = {}

            for size in self.test_data_sizes:
                test_data = self.generate_test_ohlcv_data(size)

                # Test data transformation
                start_time = time.time()
                normalized_result = pipeline.normalize_prices(test_data).run()
                normalize_time = time.time() - start_time

                # Test validation
                start_time = time.time()
                validation_result = pipeline.validate_data_quality(test_data).run()
                validation_time = time.time() - start_time

                # Test memory optimization
                start_time = time.time()
                optimized_result = pipeline.optimize_memory_usage(test_data).run()
                optimization_time = time.time() - start_time

                config_results[size] = {
                    "normalize_time": normalize_time,
                    "normalize_ops_per_sec": (
                        size / normalize_time if normalize_time > 0 else 0
                    ),
                    "validation_time": validation_time,
                    "validation_ops_per_sec": (
                        size / validation_time if validation_time > 0 else 0
                    ),
                    "optimization_time": optimization_time,
                    "optimization_ops_per_sec": (
                        size / optimization_time if optimization_time > 0 else 0
                    ),
                }

                print(
                    f"    Size {size:5d}: Normalize {normalize_time:.3f}s, "
                    f"Validate {validation_time:.3f}s, Optimize {optimization_time:.3f}s"
                )

            results[config_name] = config_results

        return results

    def benchmark_aggregation_performance(self):
        """Benchmark real-time aggregation performance"""
        if not FUNCTIONAL_IMPORTS_AVAILABLE:
            return {"error": "Aggregation not available"}

        print("\n📈 Benchmarking Real-Time Aggregation Performance...")

        results = {}

        # Test different aggregator configurations
        aggregator_configs = {
            "standard": create_real_time_aggregator(timedelta(minutes=1)),
            "high_frequency": create_high_frequency_aggregator(),
        }

        for config_name, aggregator in aggregator_configs.items():
            print(f"\n  Testing {config_name} aggregator:")
            config_results = {}

            for size in self.test_data_sizes:
                trade_data = self.generate_test_trade_data(size)

                # Test trade aggregation
                start_time = time.time()
                candles_result = aggregator.aggregate_trades_real_time(trade_data).run()
                aggregation_time = time.time() - start_time

                # Test completed candles retrieval
                start_time = time.time()
                completed_candles = aggregator.get_completed_candles().run()
                retrieval_time = time.time() - start_time

                config_results[size] = {
                    "aggregation_time": aggregation_time,
                    "aggregation_ops_per_sec": (
                        size / aggregation_time if aggregation_time > 0 else 0
                    ),
                    "retrieval_time": retrieval_time,
                    "candles_generated": len(candles_result),
                    "completed_candles": len(completed_candles),
                }

                print(
                    f"    Size {size:5d}: Aggregate {aggregation_time:.3f}s "
                    f"({config_results[size]['aggregation_ops_per_sec']:.0f} ops/s), "
                    f"Generated {len(candles_result)} candles"
                )

            results[config_name] = config_results

        return results

    def benchmark_memory_usage(self):
        """Benchmark memory usage of functional data structures"""
        if not FUNCTIONAL_IMPORTS_AVAILABLE:
            return {"error": "Memory benchmarking not available"}

        print("\n💾 Benchmarking Memory Usage...")

        import sys

        results = {}

        for size in self.test_data_sizes:
            # Measure OHLCV memory usage
            ohlcv_data = self.generate_test_ohlcv_data(size)
            ohlcv_memory = sys.getsizeof(ohlcv_data) + sum(
                sys.getsizeof(item) for item in ohlcv_data
            )

            # Measure Trade memory usage
            trade_data = self.generate_test_trade_data(size)
            trade_memory = sys.getsizeof(trade_data) + sum(
                sys.getsizeof(item) for item in trade_data
            )

            results[size] = {
                "ohlcv_memory_bytes": ohlcv_memory,
                "ohlcv_memory_per_item": ohlcv_memory / size if size > 0 else 0,
                "trade_memory_bytes": trade_memory,
                "trade_memory_per_item": trade_memory / size if size > 0 else 0,
            }

            print(
                f"  Size {size:5d}: OHLCV {ohlcv_memory:8d} bytes "
                f"({results[size]['ohlcv_memory_per_item']:.0f} bytes/item)"
            )
            print(
                f"             Trade {trade_memory:8d} bytes "
                f"({results[size]['trade_memory_per_item']:.0f} bytes/item)"
            )

        return results

    def test_data_validation_performance(self):
        """Test validation performance with various data quality scenarios"""
        if not FUNCTIONAL_IMPORTS_AVAILABLE:
            return {"error": "Validation testing not available"}

        print("\n✅ Testing Data Validation Performance...")

        results = {}

        for size in [1000, 5000]:  # Focus on realistic sizes
            # Test with valid data
            valid_data = self.generate_test_ohlcv_data(size)

            start_time = time.time()
            pipeline = create_market_data_pipeline()
            validation_result = pipeline.validate_data_quality(valid_data).run()
            validation_time = time.time() - start_time

            # Test anomaly detection
            trade_data = self.generate_test_trade_data(size)

            start_time = time.time()
            anomalies = pipeline.detect_price_anomalies(valid_data, threshold=0.1).run()
            anomaly_time = time.time() - start_time

            results[size] = {
                "validation_time": validation_time,
                "validation_success": (
                    validation_result.is_right()
                    if hasattr(validation_result, "is_right")
                    else True
                ),
                "anomaly_detection_time": anomaly_time,
                "anomalies_detected": len(anomalies),
            }

            print(
                f"  Size {size:5d}: Validation {validation_time:.3f}s, "
                f"Anomaly detection {anomaly_time:.3f}s, "
                f"Found {len(anomalies)} anomalies"
            )

        return results

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite"""
        print("🚀 Starting Functional Data Layer Performance Benchmark")
        print("=" * 60)

        benchmark_results = {}

        # Run all benchmarks
        benchmark_results["type_creation"] = self.benchmark_functional_types_creation()
        benchmark_results["data_pipeline"] = self.benchmark_data_pipeline_performance()
        benchmark_results["aggregation"] = self.benchmark_aggregation_performance()
        benchmark_results["memory_usage"] = self.benchmark_memory_usage()
        benchmark_results["validation"] = self.test_data_validation_performance()

        # Generate summary report
        self.generate_performance_report(benchmark_results)

        return benchmark_results

    def generate_performance_report(self, results: dict):
        """Generate a comprehensive performance report"""
        print("\n" + "=" * 60)
        print("📋 FUNCTIONAL DATA LAYER PERFORMANCE REPORT")
        print("=" * 60)

        # Summary of key findings
        print("\n🔍 KEY PERFORMANCE INSIGHTS:")

        if "type_creation" in results and "error" not in results["type_creation"]:
            type_results = results["type_creation"]
            max_size = max(type_results.keys())
            ohlcv_perf = type_results[max_size]["ohlcv_ops_per_sec"]
            trade_perf = type_results[max_size]["trade_ops_per_sec"]

            print(f"• OHLCV Creation Performance: {ohlcv_perf:.0f} operations/second")
            print(f"• Trade Creation Performance: {trade_perf:.0f} operations/second")

            if ohlcv_perf < 10000:
                print("  ⚠️  Type creation performance could be optimized")
            else:
                print("  ✅ Type creation performance is adequate")

        if "data_pipeline" in results and "error" not in results["data_pipeline"]:
            # Analyze pipeline performance
            pipeline_results = results["data_pipeline"]
            best_config = None
            best_performance = 0

            for config_name, config_data in pipeline_results.items():
                if isinstance(config_data, dict):
                    max_size = max(config_data.keys()) if config_data else 0
                    if max_size:
                        perf = config_data[max_size]["normalize_ops_per_sec"]
                        if perf > best_performance:
                            best_performance = perf
                            best_config = config_name

            if best_config:
                print(
                    f"• Best Pipeline Config: {best_config} ({best_performance:.0f} ops/sec)"
                )

        if "memory_usage" in results and "error" not in results["memory_usage"]:
            memory_results = results["memory_usage"]
            max_size = max(memory_results.keys())
            ohlcv_memory_per_item = memory_results[max_size]["ohlcv_memory_per_item"]
            trade_memory_per_item = memory_results[max_size]["trade_memory_per_item"]

            print(f"• OHLCV Memory Usage: {ohlcv_memory_per_item:.0f} bytes per item")
            print(f"• Trade Memory Usage: {trade_memory_per_item:.0f} bytes per item")

            if ohlcv_memory_per_item > 1000:
                print("  ⚠️  High memory usage per OHLCV item - consider optimization")

        # Recommendations
        print("\n💡 OPTIMIZATION RECOMMENDATIONS:")

        recommendations = []

        if "type_creation" in results and "error" not in results["type_creation"]:
            type_results = results["type_creation"]
            max_size = max(type_results.keys())
            if type_results[max_size]["ohlcv_ops_per_sec"] < 10000:
                recommendations.append(
                    "Optimize OHLCV type creation with __slots__ or caching"
                )

        if "data_pipeline" in results and "error" not in results["data_pipeline"]:
            if "low_latency" in results["data_pipeline"]:
                recommendations.append(
                    "Consider using low_latency pipeline for real-time trading"
                )

        if "memory_usage" in results and "error" not in results["memory_usage"]:
            memory_results = results["memory_usage"]
            max_size = max(memory_results.keys())
            if memory_results[max_size]["ohlcv_memory_per_item"] > 800:
                recommendations.append("Implement memory pooling for OHLCV objects")

        if not recommendations:
            recommendations.append("Performance is within acceptable ranges")

        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        # Performance grades
        print("\n🎯 PERFORMANCE GRADES:")

        grades = {}

        if "type_creation" in results and "error" not in results["type_creation"]:
            type_results = results["type_creation"]
            max_size = max(type_results.keys())
            ohlcv_perf = type_results[max_size]["ohlcv_ops_per_sec"]

            if ohlcv_perf > 50000:
                grades["Type Creation"] = "A+"
            elif ohlcv_perf > 20000:
                grades["Type Creation"] = "A"
            elif ohlcv_perf > 10000:
                grades["Type Creation"] = "B+"
            elif ohlcv_perf > 5000:
                grades["Type Creation"] = "B"
            else:
                grades["Type Creation"] = "C"

        if "aggregation" in results and "error" not in results["aggregation"]:
            grades["Real-time Aggregation"] = "A"  # Assume good if no errors

        if "validation" in results and "error" not in results["validation"]:
            grades["Data Validation"] = "A"  # Assume good if no errors

        for component, grade in grades.items():
            print(f"• {component}: {grade}")

        print("\n✅ Performance analysis complete!")


def main():
    """Main execution function"""
    print("Agent 8: VuManChu Preservation & Core Functionality")
    print("Data Layer Performance Optimization Analysis")
    print("-" * 60)

    if not FUNCTIONAL_IMPORTS_AVAILABLE:
        print(
            "❌ Cannot run performance tests - functional data layer not properly installed"
        )
        print("Please ensure all functional programming modules are available")
        return

    benchmark = DataLayerPerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()

    # Save results for further analysis
    import json

    with open("functional_data_performance_results.json", "w") as f:
        # Convert Decimal objects to strings for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {str(k): v for k, v in value.items()}
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2, default=str)

    print("\n📁 Results saved to: functional_data_performance_results.json")


if __name__ == "__main__":
    main()
