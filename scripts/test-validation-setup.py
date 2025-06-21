#!/usr/bin/env python3
"""
Simple test script to validate that the market making validation setup is correctly installed.

This script performs basic checks without requiring the full trading bot environment.
"""

import os
import sys
from pathlib import Path


def test_validation_setup():
    """Test that validation scripts are properly set up."""
    print("🔍 Testing Market Making Validation Setup")
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    # Test 1: Check validation script exists
    validation_script = project_root / "scripts" / "validate-market-making-setup.py"
    if validation_script.exists():
        print("✅ Main validation script found")
    else:
        print("❌ Main validation script missing")
        return False

    # Test 2: Check quick check script exists
    quick_check_script = project_root / "scripts" / "quick-market-making-check.sh"
    if quick_check_script.exists():
        print("✅ Quick check script found")
    else:
        print("❌ Quick check script missing")
        return False

    # Test 3: Check scripts are executable
    if os.access(validation_script, os.X_OK):
        print("✅ Main validation script is executable")
    else:
        print("⚠️ Main validation script not executable")

    if os.access(quick_check_script, os.X_OK):
        print("✅ Quick check script is executable")
    else:
        print("⚠️ Quick check script not executable")

    # Test 4: Check documentation exists
    docs_file = project_root / "docs" / "MARKET_MAKING_VALIDATION_GUIDE.md"
    if docs_file.exists():
        print("✅ Validation documentation found")
    else:
        print("❌ Validation documentation missing")

    # Test 5: Check market making config exists
    config_file = project_root / "config" / "market_making.json"
    if config_file.exists():
        print("✅ Market making configuration found")
    else:
        print("⚠️ Market making configuration missing")

    # Test 6: Check basic script structure
    try:
        with open(validation_script) as f:
            content = f.read()
            if "MarketMakingValidator" in content:
                print("✅ Main validation class found in script")
            else:
                print("❌ Main validation class missing from script")
                return False
    except Exception as e:
        print(f"❌ Error reading validation script: {e}")
        return False

    # Test 7: Check script has all required methods
    required_methods = [
        "run_full_validation",
        "run_pre_deployment_validation",
        "run_health_checks",
        "run_configuration_validation",
        "run_performance_benchmarks",
        "run_connectivity_tests",
        "run_indicator_validation",
        "run_fee_calculation_tests",
        "run_emergency_procedure_tests",
    ]

    missing_methods = []
    for method in required_methods:
        if method not in content:
            missing_methods.append(method)

    if missing_methods:
        print(f"❌ Missing validation methods: {', '.join(missing_methods)}")
        return False
    print("✅ All required validation methods found")

    print("\n" + "=" * 50)
    print("✅ Market Making Validation Setup Complete!")
    print("\nNext steps:")
    print("1. Install dependencies: poetry install")
    print("2. Configure .env file with your settings")
    print("3. Run quick check: ./scripts/quick-market-making-check.sh")
    print(
        "4. Run full validation: poetry run python scripts/validate-market-making-setup.py --full"
    )

    return True


if __name__ == "__main__":
    success = test_validation_setup()
    sys.exit(0 if success else 1)
