#!/usr/bin/env python3
"""
Comprehensive Error Testing Test Runner

This script runs all error handling and edge case tests with detailed reporting.
It provides organized test execution for:

1. Core error handling tests
2. Functional programming error tests
3. Integration error tests
4. Stress testing scenarios
5. Performance benchmarks

Usage:
    python tests/run_error_tests.py [options]

Options:
    --quick         Run only fast tests (exclude stress tests)
    --stress-only   Run only stress tests
    --fp-only      Run only functional programming tests
    --benchmark    Run performance benchmarks
    --report       Generate detailed HTML report
    --verbose      Enable verbose output
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class ErrorTestRunner:
    """Comprehensive error test runner with reporting."""

    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None

    def run_test_suite(
        self, test_path: str, suite_name: str, pytest_args: list[str] = None
    ) -> dict:
        """Run a test suite and collect results."""
        print(f"\n{'=' * 60}")
        print(f"Running {suite_name}")
        print(f"{'=' * 60}")

        start_time = time.time()

        # Prepare pytest arguments
        args = [test_path, "-v"]
        if pytest_args:
            args.extend(pytest_args)

        # Run tests
        exit_code = pytest.main(args)

        end_time = time.time()
        duration = end_time - start_time

        result = {
            "suite_name": suite_name,
            "test_path": test_path,
            "exit_code": exit_code,
            "duration": duration,
            "status": "PASSED" if exit_code == 0 else "FAILED",
            "timestamp": datetime.now().isoformat(),
        }

        self.test_results[suite_name] = result

        print(f"\n{suite_name}: {result['status']} (Duration: {duration:.2f}s)")

        return result

    def run_comprehensive_tests(self, options):
        """Run comprehensive error handling tests."""
        self.start_time = time.time()

        # Define test suites
        test_suites = []

        if not options.stress_only and not options.fp_only:
            # Core error handling tests
            test_suites.append(
                {
                    "path": "tests/unit/test_comprehensive_error_handling.py",
                    "name": "Core Error Handling Tests",
                    "args": [],
                }
            )

            # Integration error tests
            test_suites.append(
                {
                    "path": "tests/integration/test_error_handling.py",
                    "name": "Integration Error Tests",
                    "args": [],
                }
            )

        if not options.stress_only:
            # Functional programming error tests
            test_suites.append(
                {
                    "path": "tests/unit/fp/test_comprehensive_fp_error_handling.py",
                    "name": "Functional Programming Error Tests",
                    "args": [],
                }
            )

            test_suites.append(
                {
                    "path": "tests/unit/fp/test_error_simulation_functional.py",
                    "name": "FP Error Simulation Tests",
                    "args": [],
                }
            )

        if not options.quick and not options.fp_only:
            # Stress tests
            test_suites.append(
                {
                    "path": "tests/stress/test_error_stress_scenarios.py",
                    "name": "Error Stress Tests",
                    "args": ["--tb=short"],
                }
            )

        if options.benchmark:
            # Performance benchmarks (subset of stress tests)
            test_suites.append(
                {
                    "path": "tests/stress/test_error_stress_scenarios.py::TestErrorHandlerPerformanceBenchmarks",
                    "name": "Error Handling Performance Benchmarks",
                    "args": ["--tb=short", "-s"],
                }
            )

        # Run each test suite
        for suite in test_suites:
            try:
                self.run_test_suite(suite["path"], suite["name"], suite["args"])
            except KeyboardInterrupt:
                print("\n\nTest execution interrupted by user.")
                break
            except Exception as e:
                print(f"\nError running {suite['name']}: {e}")
                self.test_results[suite["name"]] = {
                    "suite_name": suite["name"],
                    "test_path": suite["path"],
                    "exit_code": -1,
                    "duration": 0,
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        self.end_time = time.time()

    def generate_summary_report(self):
        """Generate a summary report of test results."""
        if not self.test_results:
            print("\nNo test results to report.")
            return

        total_duration = (
            self.end_time - self.start_time if self.start_time and self.end_time else 0
        )

        print(f"\n{'=' * 80}")
        print("ERROR HANDLING TESTS SUMMARY REPORT")
        print(f"{'=' * 80}")
        print(f"Total Execution Time: {total_duration:.2f} seconds")
        print(f"Test Suites Run: {len(self.test_results)}")

        passed_suites = [
            r for r in self.test_results.values() if r["status"] == "PASSED"
        ]
        failed_suites = [
            r for r in self.test_results.values() if r["status"] in ["FAILED", "ERROR"]
        ]

        print(f"Passed: {len(passed_suites)}")
        print(f"Failed: {len(failed_suites)}")
        print(f"Success Rate: {len(passed_suites) / len(self.test_results) * 100:.1f}%")

        print(f"\n{'Test Suite':<40} {'Status':<10} {'Duration':<10}")
        print("-" * 60)

        for result in self.test_results.values():
            status = result["status"]
            duration = f"{result['duration']:.2f}s"
            name = (
                result["suite_name"][:38] + "..."
                if len(result["suite_name"]) > 38
                else result["suite_name"]
            )

            print(f"{name:<40} {status:<10} {duration:<10}")

        if failed_suites:
            print("\nFAILED SUITES:")
            for suite in failed_suites:
                print(
                    f"  - {suite['suite_name']}: {suite.get('error', 'Tests failed')}"
                )

        # Error testing specific recommendations
        print("\nERROR HANDLING TEST INSIGHTS:")

        if len(passed_suites) == len(self.test_results):
            print("✅ All error handling tests passed!")
            print("   - Error boundaries are functioning correctly")
            print("   - Circuit breakers are working as expected")
            print("   - Recovery mechanisms are operational")
            print("   - Functional error patterns are implemented properly")

        elif len(passed_suites) > 0:
            print("⚠️  Some error handling tests failed:")
            print("   - Review failed test output for specific issues")
            print("   - Check error propagation and logging mechanisms")
            print("   - Verify circuit breaker thresholds and timeouts")
            print("   - Ensure fallback strategies are properly implemented")

        else:
            print("❌ Critical: All error handling tests failed!")
            print("   - Error handling infrastructure may be broken")
            print("   - Review error handling imports and dependencies")
            print("   - Check logging configuration and error boundaries")
            print("   - Verify functional programming error types are available")

        # Performance insights (if benchmark was run)
        benchmark_result = self.test_results.get(
            "Error Handling Performance Benchmarks"
        )
        if benchmark_result:
            if benchmark_result["status"] == "PASSED":
                print("   - Error handling performance is within acceptable limits")
            else:
                print("   - Error handling performance may be degraded")
                print("   - Consider optimizing error logging and aggregation")

        print(f"\n{'=' * 80}")

    def generate_json_report(self, output_file: str):
        """Generate detailed JSON report."""
        report = {
            "test_run": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": (
                    self.end_time - self.start_time
                    if self.start_time and self.end_time
                    else 0
                ),
                "timestamp": datetime.now().isoformat(),
            },
            "summary": {
                "total_suites": len(self.test_results),
                "passed_suites": len(
                    [r for r in self.test_results.values() if r["status"] == "PASSED"]
                ),
                "failed_suites": len(
                    [
                        r
                        for r in self.test_results.values()
                        if r["status"] in ["FAILED", "ERROR"]
                    ]
                ),
                "success_rate": (
                    len(
                        [
                            r
                            for r in self.test_results.values()
                            if r["status"] == "PASSED"
                        ]
                    )
                    / len(self.test_results)
                    * 100
                    if self.test_results
                    else 0
                ),
            },
            "test_suites": self.test_results,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed JSON report saved to: {output_file}")


def setup_logging(verbose: bool = False):
    """Setup logging for test runner."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive error handling tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_error_tests.py                    # Run all tests
  python tests/run_error_tests.py --quick            # Skip stress tests
  python tests/run_error_tests.py --stress-only      # Only stress tests
  python tests/run_error_tests.py --fp-only          # Only FP tests
  python tests/run_error_tests.py --benchmark        # Include benchmarks
  python tests/run_error_tests.py --report           # Generate JSON report
        """,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only fast tests (exclude stress tests)",
    )

    parser.add_argument(
        "--stress-only", action="store_true", help="Run only stress tests"
    )

    parser.add_argument(
        "--fp-only",
        action="store_true",
        help="Run only functional programming error tests",
    )

    parser.add_argument(
        "--benchmark", action="store_true", help="Include performance benchmarks"
    )

    parser.add_argument(
        "--report", action="store_true", help="Generate detailed JSON report"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--output",
        default="error_test_report.json",
        help="Output file for JSON report (default: error_test_report.json)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Change to project root directory
    os.chdir(PROJECT_ROOT)

    # Initialize test runner
    runner = ErrorTestRunner()

    print("🧪 COMPREHENSIVE ERROR HANDLING TEST SUITE")
    print("=" * 50)

    if args.quick:
        print("Running QUICK tests (excluding stress tests)")
    elif args.stress_only:
        print("Running STRESS TESTS only")
    elif args.fp_only:
        print("Running FUNCTIONAL PROGRAMMING ERROR TESTS only")
    else:
        print("Running ALL error handling tests")

    if args.benchmark:
        print("Including PERFORMANCE BENCHMARKS")

    print()

    try:
        # Run tests
        runner.run_comprehensive_tests(args)

        # Generate reports
        runner.generate_summary_report()

        if args.report:
            runner.generate_json_report(args.output)

        # Return appropriate exit code
        failed_suites = [
            r
            for r in runner.test_results.values()
            if r["status"] in ["FAILED", "ERROR"]
        ]
        if failed_suites:
            return 1
        return 0

    except KeyboardInterrupt:
        print("\n\nTest execution interrupted.")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.exception("Unexpected error in test runner")
        return 1


if __name__ == "__main__":
    exit(main())
