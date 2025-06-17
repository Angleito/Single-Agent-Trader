#!/usr/bin/env python3
"""
Test runner script for OmniSearch MCP integration tests.

This script provides various test execution modes and reporting options
for the comprehensive OmniSearch test suite.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"✅ {description} completed successfully in {elapsed:.2f}s")
        return True
    else:
        print(f"❌ {description} failed after {elapsed:.2f}s")
        return False


def run_unit_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run unit tests for OmniSearch components."""
    cmd = ["pytest", "tests/unit/"]

    if verbose:
        cmd.extend(["-v", "-s"])

    if coverage:
        cmd.extend(["--cov=bot", "--cov-report=html", "--cov-report=term"])

    cmd.extend(["-m", "not slow"])  # Skip slow tests by default

    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run integration tests for OmniSearch."""
    cmd = ["pytest", "tests/integration/test_omnisearch_integration.py"]

    if verbose:
        cmd.extend(["-v", "-s"])

    if coverage:
        cmd.extend(
            ["--cov=bot", "--cov-report=html", "--cov-report=term", "--cov-append"]
        )

    return run_command(cmd, "Integration Tests")


def run_omnisearch_specific_tests(
    verbose: bool = False, coverage: bool = False
) -> bool:
    """Run OmniSearch-specific tests only."""
    cmd = ["pytest", "-m", "omnisearch"]

    if verbose:
        cmd.extend(["-v", "-s"])

    if coverage:
        cmd.extend(["--cov=bot", "--cov-report=html", "--cov-report=term"])

    return run_command(cmd, "OmniSearch Specific Tests")


def run_performance_tests(verbose: bool = False) -> bool:
    """Run performance benchmarks for OmniSearch components."""
    cmd = ["pytest", "-m", "slow", "--benchmark-only"]

    if verbose:
        cmd.extend(["-v", "-s"])

    return run_command(cmd, "Performance Tests")


def run_all_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run all OmniSearch tests."""
    cmd = ["pytest", "tests/"]

    if verbose:
        cmd.extend(["-v", "-s"])

    if coverage:
        cmd.extend(
            [
                "--cov=bot",
                "--cov-report=html:htmlcov",
                "--cov-report=term",
                "--cov-report=xml",
            ]
        )

    # Add test markers
    cmd.extend(["--tb=short"])

    return run_command(cmd, "All Tests")


def run_type_checking() -> bool:
    """Run type checking with mypy."""
    cmd = ["mypy", "bot/", "--ignore-missing-imports"]
    return run_command(cmd, "Type Checking")


def run_linting() -> bool:
    """Run code linting with ruff."""
    cmd = ["ruff", "check", "bot/", "tests/"]
    return run_command(cmd, "Code Linting")


def run_formatting_check() -> bool:
    """Check code formatting with black."""
    cmd = ["black", "--check", "--diff", "bot/", "tests/"]
    return run_command(cmd, "Code Formatting Check")


def generate_test_report() -> bool:
    """Generate comprehensive test report."""
    cmd = [
        "pytest",
        "tests/",
        "--html=reports/test_report.html",
        "--self-contained-html",
        "--json-report",
        "--json-report-file=reports/test_report.json",
        "--cov=bot",
        "--cov-report=html:reports/coverage",
        "--cov-report=xml:reports/coverage.xml",
    ]

    # Create reports directory
    Path("reports").mkdir(exist_ok=True)

    return run_command(cmd, "Test Report Generation")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run OmniSearch MCP integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with coverage
  python scripts/run_omnisearch_tests.py --all --coverage --verbose

  # Run only unit tests
  python scripts/run_omnisearch_tests.py --unit

  # Run integration tests with verbose output
  python scripts/run_omnisearch_tests.py --integration --verbose

  # Run OmniSearch-specific tests only
  python scripts/run_omnisearch_tests.py --omnisearch

  # Run performance benchmarks
  python scripts/run_omnisearch_tests.py --performance

  # Generate comprehensive test report
  python scripts/run_omnisearch_tests.py --report

  # Run quality checks (type checking, linting, formatting)
  python scripts/run_omnisearch_tests.py --quality
        """,
    )

    # Test execution modes
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument(
        "--omnisearch", action="store_true", help="Run OmniSearch-specific tests"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--quality", action="store_true", help="Run quality checks")
    parser.add_argument("--report", action="store_true", help="Generate test report")

    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )

    args = parser.parse_args()

    # If no specific test type is specified, run all tests
    if not any(
        [
            args.unit,
            args.integration,
            args.omnisearch,
            args.performance,
            args.all,
            args.quality,
            args.report,
        ]
    ):
        args.all = True

    success = True

    print("🧪 OmniSearch MCP Integration Test Runner")
    print("=" * 50)

    # Run quality checks if requested
    if args.quality:
        print("\n📋 Running Quality Checks...")
        success &= run_type_checking()
        success &= run_linting()
        success &= run_formatting_check()

    # Run specific test suites
    if args.unit:
        success &= run_unit_tests(args.verbose, args.coverage)

    if args.integration:
        success &= run_integration_tests(args.verbose, args.coverage)

    if args.omnisearch:
        success &= run_omnisearch_specific_tests(args.verbose, args.coverage)

    if args.performance:
        success &= run_performance_tests(args.verbose)

    if args.all:
        success &= run_all_tests(args.verbose, args.coverage)

    if args.report:
        success &= generate_test_report()

    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests and checks completed successfully!")
        print("\nNext steps:")
        print("  - Review coverage report in htmlcov/index.html")
        print("  - Check test report in reports/test_report.html")
        print("  - Ensure all OmniSearch integration points are tested")
    else:
        print("💥 Some tests or checks failed!")
        print("\nPlease:")
        print("  - Review the output above for specific failures")
        print("  - Fix any issues and re-run the tests")
        print("  - Ensure all dependencies are installed")
        sys.exit(1)


if __name__ == "__main__":
    main()
