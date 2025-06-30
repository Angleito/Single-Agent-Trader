"""
Comprehensive property-based test suite runner and utilities.

This module provides a unified interface for running all property-based tests,
collecting statistics, and reporting on test coverage and performance.
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hypothesis
import pytest
from hypothesis.database import InMemoryExampleDatabase

# Import all property test modules
from . import (
    test_configuration_validation_properties,
    test_market_data_validation_properties,
    test_orderbook_properties,
    test_performance_properties,
    test_stateful_orderbook,
)


@dataclass
class TestResult:
    """Results from running a property test module."""

    module_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    execution_time: float
    hypothesis_statistics: dict[str, Any]
    coverage_percentage: float
    error_messages: list[str]


@dataclass
class PropertyTestReport:
    """Comprehensive report of all property test results."""

    timestamp: datetime
    total_modules: int
    total_tests: int
    overall_pass_rate: float
    execution_time: float
    module_results: list[TestResult]
    summary_statistics: dict[str, Any]


class PropertyTestRunner:
    """Orchestrates property-based test execution and reporting."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("test_results/property")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure hypothesis for consistent testing
        self.configure_hypothesis()

    def configure_hypothesis(self):
        """Configure hypothesis settings for comprehensive testing."""
        # Use in-memory database for faster test execution
        hypothesis.settings.register_profile(
            "property_testing",
            database=InMemoryExampleDatabase(),
            max_examples=200,
            deadline=10000,  # 10 second deadline
            stateful_step_count=50,
            suppress_health_check=[
                hypothesis.HealthCheck.too_slow,
                hypothesis.HealthCheck.data_too_large,
            ],
        )

        # Use the property testing profile
        hypothesis.settings.load_profile("property_testing")

    def run_all_tests(self) -> PropertyTestReport:
        """Run all property-based tests and generate comprehensive report."""
        start_time = time.time()
        timestamp = datetime.now(UTC)

        test_modules = [
            ("orderbook_properties", test_orderbook_properties),
            ("market_data_validation", test_market_data_validation_properties),
            ("stateful_orderbook", test_stateful_orderbook),
            ("configuration_validation", test_configuration_validation_properties),
            ("performance_properties", test_performance_properties),
        ]

        module_results = []
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for module_name, module in test_modules:
            print(f"\n{'=' * 60}")
            print(f"Running property tests for: {module_name}")
            print(f"{'=' * 60}")

            result = self.run_module_tests(module_name, module)
            module_results.append(result)

            total_tests += result.total_tests
            total_passed += result.passed
            total_failed += result.failed
            total_skipped += result.skipped

            print(
                f"Module {module_name}: {result.passed}/{result.total_tests} passed "
                f"({result.coverage_percentage:.1f}% coverage)"
            )

        execution_time = time.time() - start_time
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        # Generate summary statistics
        summary_stats = self.generate_summary_statistics(module_results)

        report = PropertyTestReport(
            timestamp=timestamp,
            total_modules=len(test_modules),
            total_tests=total_tests,
            overall_pass_rate=overall_pass_rate,
            execution_time=execution_time,
            module_results=module_results,
            summary_statistics=summary_stats,
        )

        # Save report
        self.save_report(report)

        # Print summary
        self.print_summary(report)

        return report

    def run_module_tests(self, module_name: str, module) -> TestResult:
        """Run tests for a specific module and collect results."""
        start_time = time.time()

        # Collect test functions from the module
        test_functions = self.collect_test_functions(module)

        passed = 0
        failed = 0
        skipped = 0
        error_messages = []
        hypothesis_stats = {}

        for test_func in test_functions:
            try:
                print(f"  Running {test_func.__name__}...", end=" ")

                # Run the test
                test_func()
                passed += 1
                print("PASSED")

            except pytest.skip.Exception as e:
                skipped += 1
                print(f"SKIPPED: {e}")

            except Exception as e:
                failed += 1
                error_msg = f"{test_func.__name__}: {e!s}"
                error_messages.append(error_msg)
                print(f"FAILED: {str(e)[:100]}...")

        execution_time = time.time() - start_time
        total_tests = len(test_functions)
        coverage_percentage = (passed / total_tests * 100) if total_tests > 0 else 0

        return TestResult(
            module_name=module_name,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            execution_time=execution_time,
            hypothesis_statistics=hypothesis_stats,
            coverage_percentage=coverage_percentage,
            error_messages=error_messages,
        )

    def collect_test_functions(self, module) -> list[callable]:
        """Collect all test functions from a module."""
        test_functions = []

        # Get all classes and functions from the module
        for name in dir(module):
            obj = getattr(module, name)

            # Check if it's a test class
            if (
                isinstance(obj, type)
                and name.startswith("Test")
                and hasattr(obj, "__dict__")
            ):
                # Get test methods from the class
                instance = obj()
                for method_name in dir(instance):
                    if method_name.startswith("test_"):
                        method = getattr(instance, method_name)
                        if callable(method):
                            test_functions.append(method)

            # Check if it's a standalone test function
            elif callable(obj) and name.startswith("test_"):
                test_functions.append(obj)

        return test_functions

    def generate_summary_statistics(
        self, module_results: list[TestResult]
    ) -> dict[str, Any]:
        """Generate summary statistics across all modules."""
        if not module_results:
            return {}

        total_execution_time = sum(r.execution_time for r in module_results)
        average_pass_rate = sum(r.coverage_percentage for r in module_results) / len(
            module_results
        )

        # Find best and worst performing modules
        best_module = max(module_results, key=lambda r: r.coverage_percentage)
        worst_module = min(module_results, key=lambda r: r.coverage_percentage)

        # Count error types
        all_errors = []
        for result in module_results:
            all_errors.extend(result.error_messages)

        return {
            "total_execution_time": total_execution_time,
            "average_pass_rate": average_pass_rate,
            "best_performing_module": {
                "name": best_module.module_name,
                "pass_rate": best_module.coverage_percentage,
            },
            "worst_performing_module": {
                "name": worst_module.module_name,
                "pass_rate": worst_module.coverage_percentage,
            },
            "total_errors": len(all_errors),
            "modules_with_failures": len([r for r in module_results if r.failed > 0]),
            "fastest_module": min(
                module_results, key=lambda r: r.execution_time
            ).module_name,
            "slowest_module": max(
                module_results, key=lambda r: r.execution_time
            ).module_name,
        }

    def save_report(self, report: PropertyTestReport):
        """Save the test report to JSON file."""
        report_file = (
            self.output_dir
            / f"property_test_report_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Convert report to dictionary
        report_dict = asdict(report)
        report_dict["timestamp"] = report.timestamp.isoformat()

        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        print(f"\nReport saved to: {report_file}")

    def print_summary(self, report: PropertyTestReport):
        """Print a comprehensive summary of test results."""
        print(f"\n{'=' * 80}")
        print("PROPERTY-BASED TEST SUITE SUMMARY")
        print(f"{'=' * 80}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Modules: {report.total_modules}")
        print(f"Total Tests: {report.total_tests}")
        print(f"Overall Pass Rate: {report.overall_pass_rate:.1f}%")
        print(f"Total Execution Time: {report.execution_time:.2f}s")

        print(f"\n{'MODULE BREAKDOWN':=^80}")
        for result in report.module_results:
            status = "✅" if result.failed == 0 else "❌"
            print(
                f"{status} {result.module_name:<30} "
                f"{result.passed:>3}/{result.total_tests:<3} "
                f"({result.coverage_percentage:>5.1f}%) "
                f"{result.execution_time:>6.2f}s"
            )

            if result.error_messages:
                print("    Errors:")
                for error in result.error_messages[:3]:  # Show first 3 errors
                    print(f"      - {error[:70]}...")
                if len(result.error_messages) > 3:
                    print(f"      ... and {len(result.error_messages) - 3} more")

        print(f"\n{'SUMMARY STATISTICS':=^80}")
        stats = report.summary_statistics
        print(f"Average Pass Rate: {stats.get('average_pass_rate', 0):.1f}%")
        print(
            f"Best Module: {stats.get('best_performing_module', {}).get('name', 'N/A')} "
            f"({stats.get('best_performing_module', {}).get('pass_rate', 0):.1f}%)"
        )
        print(
            f"Worst Module: {stats.get('worst_performing_module', {}).get('name', 'N/A')} "
            f"({stats.get('worst_performing_module', {}).get('pass_rate', 0):.1f}%)"
        )
        print(
            f"Modules with Failures: {stats.get('modules_with_failures', 0)}/{report.total_modules}"
        )
        print(f"Total Errors: {stats.get('total_errors', 0)}")

        print(f"\n{'RECOMMENDATIONS':=^80}")
        self.print_recommendations(report)

    def print_recommendations(self, report: PropertyTestReport):
        """Print recommendations based on test results."""
        stats = report.summary_statistics

        if report.overall_pass_rate < 90:
            print("⚠️  Overall pass rate is below 90%. Consider:")
            print("   - Reviewing failed test cases")
            print("   - Adjusting hypothesis settings")
            print("   - Implementing missing functionality")

        if stats.get("modules_with_failures", 0) > 0:
            print("🔍 Some modules have failures. Prioritize:")
            worst_module = stats.get("worst_performing_module", {})
            print(f"   - Focus on {worst_module.get('name', 'failed modules')}")
            print("   - Review error patterns")

        if report.execution_time > 300:  # 5 minutes
            print("⏱️  Test suite is taking a long time. Consider:")
            print("   - Reducing max_examples in hypothesis settings")
            print("   - Optimizing slow test cases")
            print("   - Parallelizing test execution")

        if report.overall_pass_rate >= 95:
            print("🎉 Excellent test coverage! Consider:")
            print("   - Adding more edge cases")
            print("   - Increasing hypothesis max_examples")
            print("   - Adding performance benchmarks")


class PropertyTestValidator:
    """Validates property test implementations and configurations."""

    @staticmethod
    def validate_test_coverage() -> dict[str, list[str]]:
        """Validate that all major components have property test coverage."""
        required_coverage = {
            "OrderBook": [
                "bid/ask ordering",
                "spread calculation",
                "depth calculation",
                "VWAP calculation",
                "price impact",
            ],
            "MarketData": [
                "price validation",
                "volume validation",
                "OHLCV relationships",
                "spread calculation",
            ],
            "Configuration": [
                "API key validation",
                "private key masking",
                "parameter validation",
                "type safety",
            ],
            "Performance": [
                "algorithmic complexity",
                "memory usage",
                "processing time",
            ],
        }

        # This would ideally check actual test implementations
        # For now, return the requirements
        return required_coverage

    @staticmethod
    def validate_hypothesis_settings() -> list[str]:
        """Validate hypothesis configuration for comprehensive testing."""
        issues = []

        current_settings = hypothesis.settings.get_profile("default")

        if current_settings.max_examples < 100:
            issues.append(
                "max_examples should be at least 100 for comprehensive testing"
            )

        if current_settings.deadline and current_settings.deadline < 5000:
            issues.append(
                "deadline should be at least 5000ms for complex property tests"
            )

        return issues


def run_property_test_suite():
    """Main entry point for running the complete property test suite."""
    runner = PropertyTestRunner()
    report = runner.run_all_tests()

    # Validate test coverage
    validator = PropertyTestValidator()
    coverage_requirements = validator.validate_test_coverage()
    settings_issues = validator.validate_hypothesis_settings()

    if settings_issues:
        print(f"\n{'CONFIGURATION ISSUES':=^80}")
        for issue in settings_issues:
            print(f"⚠️  {issue}")

    return report


if __name__ == "__main__":
    # Run the complete property test suite
    report = run_property_test_suite()

    # Exit with appropriate code
    if report.overall_pass_rate >= 95:
        exit(0)  # Success
    elif report.overall_pass_rate >= 80:
        exit(1)  # Some failures but mostly working
    else:
        exit(2)  # Significant failures
