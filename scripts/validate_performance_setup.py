#!/usr/bin/env python3
"""
Validation script for the performance testing setup.

This script validates that all performance testing components are properly
installed and configured, and runs a quick validation test.
"""

import asyncio
import logging
import sys
from pathlib import Path


def check_imports():
    """Check that all required imports are available."""
    print("🔍 Checking imports...")

    required_modules = [
        ("psutil", "System monitoring"),
        ("numpy", "Numerical operations"),
        ("pandas", "Data processing"),
        ("pandas_ta", "Technical analysis"),
        ("asyncio", "Async operations"),
    ]

    missing_modules = []

    for module_name, description in required_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name} - {description}")
        except ImportError:
            print(f"  ❌ {module_name} - {description} (MISSING)")
            missing_modules.append(module_name)

    if missing_modules:
        print(f"\n❌ Missing required modules: {', '.join(missing_modules)}")
        print("Install with: pip install " + " ".join(missing_modules))
        return False

    return True


def check_bot_components():
    """Check that bot components can be imported."""
    print("\n🔍 Checking bot components...")

    try:
        # Add the project root to Python path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from bot.indicators.vumanchu import VuManChuIndicators

        print("  ✅ VuManChuIndicators")

        from bot.strategy.llm_agent import LLMAgent

        print("  ✅ LLMAgent")

        from bot.types import IndicatorData, MarketState, Position

        print("  ✅ Trading types")

        from bot.performance_monitor import PerformanceMonitor, track

        print("  ✅ PerformanceMonitor")

        return True

    except ImportError as e:
        print(f"  ❌ Failed to import bot components: {e}")
        return False


def check_performance_tests():
    """Check that performance test modules can be imported."""
    print("\n🔍 Checking performance test modules...")

    try:
        # Add the tests directory to Python path
        tests_dir = Path(__file__).parent.parent / "tests"
        sys.path.insert(0, str(tests_dir))

        from performance.benchmark_suite import PerformanceBenchmarks

        print("  ✅ Benchmark suite")

        from performance.load_tests import LoadTestSuite

        print("  ✅ Load test suite")

        return True

    except ImportError as e:
        print(f"  ❌ Failed to import performance tests: {e}")
        return False


async def run_quick_validation():
    """Run a quick validation test."""
    print("\n🧪 Running quick validation test...")

    try:
        # Import required components
        import numpy as np
        import pandas as pd

        from bot.indicators.vumanchu import VuManChuIndicators
        from bot.performance_monitor import PerformanceMonitor, track

        # Initialize performance monitor
        monitor = PerformanceMonitor()
        await monitor.start_monitoring()

        # Create test data
        data = {
            "open": np.random.uniform(49000, 51000, 50),
            "high": np.random.uniform(49500, 51500, 50),
            "low": np.random.uniform(48500, 50500, 50),
            "close": np.random.uniform(49000, 51000, 50),
            "volume": np.random.uniform(1000, 10000, 50),
        }
        df = pd.DataFrame(data)

        # Initialize indicator calculator
        indicator_calc = VuManChuIndicators()

        # Test indicator calculation with monitoring
        with track("validation_test"):
            result = indicator_calc.calculate_all(df)

        # Test that result contains expected indicators
        expected_indicators = ["ema_fast", "ema_slow", "rsi"]
        for indicator in expected_indicators:
            if indicator not in result.columns:
                print(f"  ❌ Missing indicator: {indicator}")
                return False

        print("  ✅ Indicator calculations")

        # Stop monitoring and get summary
        await monitor.stop_monitoring()
        summary = monitor.get_performance_summary()

        if summary["health_score"] > 0:
            print(
                f"  ✅ Performance monitoring (Health Score: {summary['health_score']:.1f})"
            )
        else:
            print("  ⚠️  Performance monitoring (No metrics collected)")

        return True

    except Exception as e:
        print(f"  ❌ Validation test failed: {e}")
        return False


async def test_benchmark_suite():
    """Test that benchmark suite can run."""
    print("\n🏃 Testing benchmark suite (quick run)...")

    try:
        from performance.benchmark_suite import PerformanceBenchmarks

        benchmarks = PerformanceBenchmarks()

        # Run a single benchmark test
        result = benchmarks.benchmark_cipher_a_calculation()

        if result.execution_time_ms > 0:
            print(f"  ✅ Cipher A benchmark: {result.execution_time_ms:.2f}ms")
        else:
            print("  ❌ Cipher A benchmark returned no timing")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Benchmark test failed: {e}")
        return False


async def test_load_suite():
    """Test that load test suite can initialize."""
    print("\n🔥 Testing load test suite (initialization only)...")

    try:
        from performance.load_tests import LoadTestConfig, LoadTestSuite

        config = LoadTestConfig(duration_seconds=1, concurrent_users=1)
        load_tests = LoadTestSuite(config)

        # Test market data simulator
        simulator = load_tests.market_simulator
        test_data = simulator.generate_tick_data(10)

        if len(test_data) == 10:
            print("  ✅ Market data simulation")
        else:
            print("  ❌ Market data simulation failed")
            return False

        # Test indicator calculation with timing
        response_time = load_tests._calculate_indicators_with_timing(
            load_tests.market_simulator._generate_test_data(100)
        )

        if response_time > 0:
            print(f"  ✅ Load test timing: {response_time:.2f}ms")
        else:
            print("  ❌ Load test timing failed")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Load test failed: {e}")
        return False


def check_file_structure():
    """Check that all required files exist."""
    print("\n📁 Checking file structure...")

    project_root = Path(__file__).parent.parent

    required_files = [
        "bot/performance_monitor.py",
        "tests/performance/benchmark_suite.py",
        "tests/performance/load_tests.py",
        "tests/performance/run_performance_tests.py",
        "docs/Performance_Optimization.md",
        "examples/performance_integration_example.py",
    ]

    missing_files = []

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (MISSING)")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n❌ Missing files: {len(missing_files)}")
        return False

    return True


async def main():
    """Main validation function."""
    print("🎯 AI Trading Bot Performance Testing Validation")
    print("=" * 60)

    # Track validation results
    checks = []

    # Run all validation checks
    checks.append(("Import Check", check_imports()))
    checks.append(("File Structure", check_file_structure()))
    checks.append(("Bot Components", check_bot_components()))
    checks.append(("Performance Tests", check_performance_tests()))
    checks.append(("Quick Validation", await run_quick_validation()))
    checks.append(("Benchmark Suite", await test_benchmark_suite()))
    checks.append(("Load Test Suite", await test_load_suite()))

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n🎉 All validation checks passed!")
        print("\nYou can now run performance tests with:")
        print("  python tests/performance/run_performance_tests.py --all")
        print("\nOr integrate monitoring with:")
        print("  python examples/performance_integration_example.py")
        return True
    else:
        print(f"\n❌ {failed} validation check(s) failed.")
        print("\nPlease fix the issues above before running performance tests.")
        return False


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during validation
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)
