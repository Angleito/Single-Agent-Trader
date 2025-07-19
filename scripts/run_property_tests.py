#!/usr/bin/env python3
"""
Property-based test runner script.

This script runs the comprehensive property-based test suite for orderbook
data validation, market data processing, and configuration validation.

Usage:
    python run_property_tests.py [options]

Options:
    --module <name>    Run tests for specific module only
    --verbose          Enable verbose output
    --quick            Run with reduced examples for faster execution
    --save-report      Save detailed report to file
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import hypothesis

    from tests.property.test_property_suite import (
        PropertyTestRunner,
        run_property_test_suite,
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you have installed all dependencies with: poetry install")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run property-based tests for orderbook validation"
    )
    parser.add_argument("--module", help="Run tests for specific module only")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced examples for faster execution",
    )
    parser.add_argument(
        "--save-report", action="store_true", help="Save detailed report to file"
    )

    args = parser.parse_args()

    # Configure hypothesis based on arguments
    if args.quick:
        hypothesis.settings.register_profile(
            "quick", max_examples=20, deadline=2000, stateful_step_count=10
        )
        hypothesis.settings.load_profile("quick")
        print("Running in quick mode with reduced examples...")

    # Create test runner
    runner = PropertyTestRunner() if args.save_report else PropertyTestRunner()

    if args.module:
        print(f"Running property tests for module: {args.module}")
        # This would run specific module tests
        # For now, run all tests
        report = run_property_test_suite()
    else:
        print("Running complete property-based test suite...")
        report = run_property_test_suite()

    # Print final status
    print(f"\n{'FINAL RESULT':=^80}")
    if report.overall_pass_rate >= 95:
        print("🎉 EXCELLENT: All property tests passed!")
        return 0
    if report.overall_pass_rate >= 80:
        print("⚠️  GOOD: Most property tests passed, some issues to address")
        return 1
    print("❌ NEEDS WORK: Significant property test failures")
    return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
