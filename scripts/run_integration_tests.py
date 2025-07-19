#!/usr/bin/env python3
"""
Comprehensive Integration Test Runner

This script runs the complete suite of integration tests for the orderbook flow,
providing detailed reporting, performance metrics, and test environment management.

Usage:
    python run_integration_tests.py [options]

Options:
    --suite SUITE       Run specific test suite (orderbook, market_making, sdk_service, all)
    --performance       Run performance tests only
    --parallel          Run tests in parallel
    --report            Generate detailed HTML report
    --verbose           Enable verbose output
    --timeout SECONDS   Set test timeout (default: 300)
    --markers MARKERS   Run tests with specific pytest markers
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


class IntegrationTestRunner:
    """Comprehensive integration test runner."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.test_results = {}
        self.performance_metrics = {}

    def run_test_suite(
        self,
        suite: str = "all",
        performance_only: bool = False,
        parallel: bool = False,
        generate_report: bool = False,
        verbose: bool = False,
        timeout: int = 300,
        markers: str | None = None,
    ) -> bool:
        """Run integration test suite with specified options."""

        self.start_time = datetime.now(UTC)
        print(f"🚀 Starting Integration Test Suite at {self.start_time}")
        print("📋 Configuration:")
        print(f"   Suite: {suite}")
        print(f"   Performance Only: {performance_only}")
        print(f"   Parallel: {parallel}")
        print(f"   Generate Report: {generate_report}")
        print(f"   Timeout: {timeout}s")

        try:
            # Setup test environment
            self._setup_test_environment()

            # Build pytest command
            pytest_args = self._build_pytest_command(
                suite=suite,
                performance_only=performance_only,
                parallel=parallel,
                generate_report=generate_report,
                verbose=verbose,
                timeout=timeout,
                markers=markers,
            )

            print(f"\n🧪 Running pytest with args: {' '.join(pytest_args)}")

            # Run tests
            result = subprocess.run(
                ["python", "-m", "pytest"] + pytest_args,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout + 60,  # Add buffer for pytest overhead
            )

            # Process results
            success = result.returncode == 0
            self._process_test_results(result, success)

            if generate_report:
                self._generate_detailed_report()

            return success

        except subprocess.TimeoutExpired:
            print(f"❌ Tests timed out after {timeout + 60} seconds")
            return False
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
        finally:
            self.end_time = datetime.now(UTC)
            self._cleanup_test_environment()
            self._print_summary()

    def _setup_test_environment(self):
        """Set up test environment."""
        print("\n🔧 Setting up test environment...")

        # Ensure test directories exist
        test_dirs = [
            "tests/integration",
            "tests/integration/reports",
            "tests/integration/logs",
        ]

        for test_dir in test_dirs:
            Path(test_dir).mkdir(parents=True, exist_ok=True)

        # Set test environment variables
        test_env = {
            "TRADING_MODE": "paper",
            "EXCHANGE_TYPE": "coinbase",
            "LOG_LEVEL": "DEBUG",
            "ENABLE_WEBSOCKET": "false",  # Disable for testing
            "ENABLE_MEMORY": "false",
            "ENABLE_RISK_MANAGEMENT": "true",
            "PYTEST_RUNNING": "true",
        }

        for key, value in test_env.items():
            os.environ[key] = value

        print("✅ Test environment configured")

    def _build_pytest_command(
        self,
        suite: str,
        performance_only: bool,
        parallel: bool,
        generate_report: bool,
        verbose: bool,
        timeout: int,
        markers: str | None,
    ) -> list[str]:
        """Build pytest command arguments."""

        args = []

        # Test discovery
        if suite == "all":
            args.append("tests/integration/")
        elif suite == "orderbook":
            args.append("tests/integration/test_orderbook_integration.py")
        elif suite == "market_making":
            args.append("tests/integration/test_market_making_orderbook_integration.py")
        elif suite == "sdk_service":
            args.append("tests/integration/test_sdk_service_integration.py")
        else:
            args.append(f"tests/integration/test_{suite}_integration.py")

        # Verbosity
        if verbose:
            args.extend(["-v", "-s"])

        # Parallel execution
        if parallel:
            args.extend(["-n", "auto"])

        # Performance tests only
        if performance_only:
            args.extend(["-m", "performance"])

        # Custom markers
        if markers:
            args.extend(["-m", markers])

        # Timeout
        args.extend(["--timeout", str(timeout)])

        # Coverage
        args.extend(
            [
                "--cov=bot",
                "--cov-report=term-missing",
                "--cov-report=html:tests/integration/reports/coverage",
            ]
        )

        # Output formats
        if generate_report:
            args.extend(
                [
                    "--html=tests/integration/reports/report.html",
                    "--self-contained-html",
                    "--junit-xml=tests/integration/reports/junit.xml",
                ]
            )

        # Additional options
        args.extend(["--tb=short", "--strict-markers", "--disable-warnings"])

        return args

    def _process_test_results(self, result: subprocess.CompletedProcess, success: bool):
        """Process test results and extract metrics."""

        print("\n📊 Processing test results...")

        self.test_results = {
            "success": success,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        # Parse pytest output for metrics
        output_lines = result.stdout.split("\n")

        for line in output_lines:
            if "passed" in line and "failed" in line:
                # Parse test counts
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed":
                        self.test_results["passed"] = int(parts[i - 1])
                    elif part == "failed":
                        self.test_results["failed"] = int(parts[i - 1])
                    elif part == "error":
                        self.test_results["errors"] = int(parts[i - 1])
                    elif part == "skipped":
                        self.test_results["skipped"] = int(parts[i - 1])

        # Extract performance metrics
        self._extract_performance_metrics(output_lines)

        print(
            f"✅ Results processed: {self.test_results.get('passed', 0)} passed, {self.test_results.get('failed', 0)} failed"
        )

    def _extract_performance_metrics(self, output_lines: list[str]):
        """Extract performance metrics from test output."""

        metrics = {
            "orderbook_operations": [],
            "market_making_calculations": [],
            "websocket_processing": [],
            "sdk_requests": [],
        }

        for line in output_lines:
            # Look for performance-related log messages
            if "Average request time:" in line:
                try:
                    time_str = line.split(":")[1].strip().replace("s", "")
                    metrics["sdk_requests"].append(float(time_str))
                except:
                    pass
            elif "OrderBook creation time:" in line:
                try:
                    time_str = line.split(":")[1].strip().replace("ms", "")
                    metrics["orderbook_operations"].append(float(time_str) / 1000)
                except:
                    pass
            elif "Quote calculation time:" in line:
                try:
                    time_str = line.split(":")[1].strip().replace("ms", "")
                    metrics["market_making_calculations"].append(float(time_str) / 1000)
                except:
                    pass

        self.performance_metrics = metrics

    def _generate_detailed_report(self):
        """Generate detailed HTML and JSON reports."""

        print("\n📝 Generating detailed reports...")

        # Create comprehensive report data
        report_data = {
            "test_run": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": (
                    (self.end_time - self.start_time).total_seconds()
                    if self.start_time and self.end_time
                    else None
                ),
                "environment": dict(os.environ),
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "results": self.test_results,
            "performance_metrics": self.performance_metrics,
            "test_coverage": self._get_coverage_data(),
            "recommendations": self._generate_recommendations(),
        }

        # Save JSON report
        json_report_path = "tests/integration/reports/detailed_report.json"
        with open(json_report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"✅ Detailed report saved to {json_report_path}")

        # Generate performance summary
        self._generate_performance_summary()

    def _get_coverage_data(self) -> dict:
        """Get test coverage data."""

        coverage_file = "tests/integration/reports/coverage/.coverage"
        if os.path.exists(coverage_file):
            try:
                # This would parse coverage data in a real implementation
                return {"coverage_available": True, "file": coverage_file}
            except:
                pass

        return {"coverage_available": False}

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results."""

        recommendations = []

        if self.test_results.get("failed", 0) > 0:
            recommendations.append(
                "🔴 Some tests failed - review error logs and fix issues"
            )

        # Performance recommendations
        avg_orderbook_time = 0
        if self.performance_metrics.get("orderbook_operations"):
            avg_orderbook_time = sum(
                self.performance_metrics["orderbook_operations"]
            ) / len(self.performance_metrics["orderbook_operations"])

        if avg_orderbook_time > 0.1:
            recommendations.append(
                "⚠️  OrderBook operations are slow - consider optimization"
            )

        avg_sdk_time = 0
        if self.performance_metrics.get("sdk_requests"):
            avg_sdk_time = sum(self.performance_metrics["sdk_requests"]) / len(
                self.performance_metrics["sdk_requests"]
            )

        if avg_sdk_time > 0.5:
            recommendations.append(
                "⚠️  SDK requests are slow - check network or add caching"
            )

        if len(recommendations) == 0:
            recommendations.append("✅ All tests passed and performance is good!")

        return recommendations

    def _generate_performance_summary(self):
        """Generate performance summary report."""

        summary_path = "tests/integration/reports/performance_summary.txt"

        with open(summary_path, "w") as f:
            f.write("Integration Tests Performance Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Test Run Date: {self.start_time}\n")
            f.write(
                f"Total Duration: {(self.end_time - self.start_time).total_seconds():.2f}s\n\n"
            )

            for metric_name, values in self.performance_metrics.items():
                if values:
                    avg_time = sum(values) / len(values)
                    max_time = max(values)
                    min_time = min(values)

                    f.write(f"{metric_name.replace('_', ' ').title()}:\n")
                    f.write(f"  Average: {avg_time:.3f}s\n")
                    f.write(f"  Min: {min_time:.3f}s\n")
                    f.write(f"  Max: {max_time:.3f}s\n")
                    f.write(f"  Samples: {len(values)}\n\n")

        print(f"✅ Performance summary saved to {summary_path}")

    def _cleanup_test_environment(self):
        """Clean up test environment."""
        print("\n🧹 Cleaning up test environment...")

        # Remove test-specific environment variables
        test_vars = ["PYTEST_RUNNING"]
        for var in test_vars:
            os.environ.pop(var, None)

        print("✅ Test environment cleaned up")

    def _print_summary(self):
        """Print test run summary."""

        if not self.start_time or not self.end_time:
            return

        duration = self.end_time - self.start_time

        print("\n" + "=" * 60)
        print("🎯 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"⏱️  Duration: {duration.total_seconds():.2f} seconds")
        print(f"✅ Passed: {self.test_results.get('passed', 0)}")
        print(f"❌ Failed: {self.test_results.get('failed', 0)}")
        print(f"⚠️  Errors: {self.test_results.get('errors', 0)}")
        print(f"⏭️  Skipped: {self.test_results.get('skipped', 0)}")

        if self.test_results.get("success"):
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n💥 SOME TESTS FAILED!")

        # Performance summary
        if self.performance_metrics:
            print("\n📊 Performance Highlights:")
            for metric_name, values in self.performance_metrics.items():
                if values:
                    avg = sum(values) / len(values)
                    print(f"   {metric_name}: {avg:.3f}s avg")

        print("\n📋 Recommendations:")
        for rec in self._generate_recommendations():
            print(f"   {rec}")

        print("=" * 60)


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Run comprehensive integration tests for orderbook flow"
    )

    parser.add_argument(
        "--suite",
        choices=["all", "orderbook", "market_making", "sdk_service"],
        default="all",
        help="Test suite to run",
    )

    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests only"
    )

    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")

    parser.add_argument(
        "--report", action="store_true", help="Generate detailed HTML report"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--timeout", type=int, default=300, help="Test timeout in seconds"
    )

    parser.add_argument(
        "--markers", type=str, help="Run tests with specific pytest markers"
    )

    args = parser.parse_args()

    # Create and run test runner
    runner = IntegrationTestRunner()

    success = runner.run_test_suite(
        suite=args.suite,
        performance_only=args.performance,
        parallel=args.parallel,
        generate_report=args.report,
        verbose=args.verbose,
        timeout=args.timeout,
        markers=args.markers,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
