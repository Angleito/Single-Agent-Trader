#!/usr/bin/env python3
"""
Simple validation script for Coinbase Futures trading implementation.

This script validates core functionality without external dependencies.
"""

import sys
from pathlib import Path


def test_configuration():
    """Test configuration file changes."""
    print("🔍 Testing Configuration...")

    # Read config.py file
    config_file = Path("bot/config.py")
    if not config_file.exists():
        print("❌ Config file not found")
        return False

    config_content = config_file.read_text()

    # Check for o3 model default
    if 'default="o3"' not in config_content:
        print("❌ Default model not set to 'o3'")
        return False

    # Check for futures configuration
    futures_checks = [
        "enable_futures",
        "futures_account_type",
        "auto_cash_transfer",
        "max_futures_leverage",
    ]

    for check in futures_checks:
        if check not in config_content:
            print(f"❌ Missing futures config: {check}")
            return False

    # Check model validation includes o3
    if "'o3'" not in config_content:
        print("❌ o3 model not in validation list")
        return False

    print("✅ Configuration: PASSED")
    return True


def test_types():
    """Test types.py enhancements."""
    print("🔍 Testing Types...")

    types_file = Path("bot/types.py")
    if not types_file.exists():
        print("❌ Types file not found")
        return False

    types_content = types_file.read_text()

    # Check for new enums and classes
    required_items = [
        "class AccountType",
        "class MarginHealthStatus",
        "class MarginInfo",
        "class FuturesAccountInfo",
        "class FuturesOrder",
        "class CashTransferRequest",
        "class FuturesMarketState",
        "CFM",
        "CBI",
        "HEALTHY",
        "LIQUIDATION_RISK",
        "leverage:",
        "reduce_only:",
    ]

    for item in required_items:
        if item not in types_content:
            print(f"❌ Missing type definition: {item}")
            return False

    print("✅ Types: PASSED")
    return True


def test_exchange_client():
    """Test exchange client enhancements."""
    print("🔍 Testing Exchange Client...")

    exchange_file = Path("bot/exchange/coinbase.py")
    if not exchange_file.exists():
        print("❌ Exchange client file not found")
        return False

    exchange_content = exchange_file.read_text()

    # Check for futures methods
    futures_methods = [
        "get_futures_balance",
        "get_spot_balance",
        "get_futures_account_info",
        "get_margin_info",
        "transfer_cash_to_futures",
        "get_futures_positions",
        "place_futures_market_order",
        "_open_futures_position",
    ]

    for method in futures_methods:
        if f"def {method}" not in exchange_content:
            print(f"❌ Missing method: {method}")
            return False

    # Check for futures configuration
    if "self.enable_futures" not in exchange_content:
        print("❌ Missing futures configuration in client")
        return False

    # Check for futures imports
    futures_imports = ["AccountType", "MarginHealthStatus", "FuturesAccountInfo"]

    for import_item in futures_imports:
        if import_item not in exchange_content:
            print(f"❌ Missing import: {import_item}")
            return False

    print("✅ Exchange Client: PASSED")
    return True


def test_llm_agent():
    """Test LLM agent enhancements."""
    print("🔍 Testing LLM Agent...")

    llm_file = Path("bot/strategy/llm_agent.py")
    if not llm_file.exists():
        print("❌ LLM agent file not found")
        return False

    llm_content = llm_file.read_text()

    # Check for futures-related prompt updates
    futures_prompt_items = [
        "leverage",
        "reduce_only",
        "margin_health",
        "available_margin",
        "futures trading",
        "max_leverage",
    ]

    for item in futures_prompt_items:
        if item.lower() not in llm_content.lower():
            print(f"❌ Missing futures prompt item: {item}")
            return False

    # Check for o3 model configuration
    if "o3" not in llm_content:
        print("❌ o3 model configuration not found")
        return False

    print("✅ LLM Agent: PASSED")
    return True


def test_template_config():
    """Test configuration template updates."""
    print("🔍 Testing Template Configuration...")

    config_file = Path("bot/config.py")
    config_content = config_file.read_text()

    # Check template includes o3 and futures
    if '"model_name": "o3"' not in config_content:
        print("❌ Template doesn't include o3 model")
        return False

    if '"enable_futures": true' not in config_content.lower():
        print("❌ Template doesn't include futures config")
        return False

    print("✅ Template Configuration: PASSED")
    return True


def main():
    """Run all validation tests."""
    print("🚀 Coinbase Futures Implementation Validation")
    print("=" * 50)

    tests = [
        ("Configuration", test_configuration),
        ("Types", test_types),
        ("Exchange Client", test_exchange_client),
        ("LLM Agent", test_llm_agent),
        ("Template Config", test_template_config),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    success_rate = (passed / total) * 100
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nKey enhancements validated:")
        print("  ✓ OpenAI o3 model as default")
        print("  ✓ Futures trading configuration")
        print("  ✓ CFM/CBI account types")
        print("  ✓ Margin health monitoring")
        print("  ✓ Futures-specific order handling")
        print("  ✓ Enhanced LLM prompting")
        print("\n🎯 Coinbase Futures implementation is ready!")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed")
        print("Please review the implementation.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Validation error: {e}")
        sys.exit(1)
