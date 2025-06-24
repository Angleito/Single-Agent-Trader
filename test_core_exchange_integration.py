#!/usr/bin/env python3
"""
Core Exchange Integration Test

This test directly validates the core exchange functionality without
complex dependencies that might have import issues.
"""

import asyncio
import logging
import sys
from decimal import Decimal

# Add the project root to sys.path to import modules
sys.path.insert(0, "/Users/angel/Documents/Projects/cursorprod")

# Direct imports to avoid complex dependency chains
from bot.exchange.base import BaseExchange
from bot.exchange.factory import ExchangeFactory

logger = logging.getLogger(__name__)


async def test_exchange_factory():
    """Test the exchange factory functionality."""
    print("Testing Exchange Factory...")

    results = []

    # Test supported exchanges
    try:
        supported = ExchangeFactory.get_supported_exchanges()
        expected = ["coinbase", "bluefin"]

        if set(supported) == set(expected):
            results.append(("✅", "Supported exchanges", f"Found: {supported}"))
        else:
            results.append(
                ("❌", "Supported exchanges", f"Expected {expected}, got {supported}")
            )
    except Exception as e:
        results.append(("❌", "Supported exchanges", f"Error: {e}"))

    return results


async def test_coinbase_creation():
    """Test Coinbase exchange creation."""
    print("Testing Coinbase Creation...")

    results = []

    try:
        # Create Coinbase client with dry_run=True
        coinbase = ExchangeFactory.create_exchange(
            exchange_type="coinbase", dry_run=True
        )

        if isinstance(coinbase, BaseExchange):
            results.append(
                ("✅", "Coinbase creation", "Successfully created CoinbaseClient")
            )

            # Test basic properties
            if hasattr(coinbase, "exchange_name"):
                results.append(
                    ("✅", "Coinbase properties", f"Name: {coinbase.exchange_name}")
                )
            else:
                results.append(("❌", "Coinbase properties", "Missing exchange_name"))

            # Test dry_run property
            if hasattr(coinbase, "dry_run"):
                results.append(
                    ("✅", "Coinbase dry_run", f"Dry run: {coinbase.dry_run}")
                )
            else:
                results.append(("❌", "Coinbase dry_run", "Missing dry_run property"))

            # Test connection status
            try:
                connected = coinbase.is_connected()
                results.append(
                    ("✅", "Coinbase connection check", f"Connected: {connected}")
                )
            except Exception as e:
                results.append(("❌", "Coinbase connection check", f"Error: {e}"))

            # Test connection status structure
            try:
                status = coinbase.get_connection_status()
                if isinstance(status, dict) and "connected" in status:
                    results.append(
                        (
                            "✅",
                            "Coinbase status structure",
                            f"Keys: {list(status.keys())}",
                        )
                    )
                else:
                    results.append(
                        ("❌", "Coinbase status structure", f"Invalid: {type(status)}")
                    )
            except Exception as e:
                results.append(("❌", "Coinbase status structure", f"Error: {e}"))

        else:
            results.append(("❌", "Coinbase creation", f"Wrong type: {type(coinbase)}"))

    except Exception as e:
        results.append(("❌", "Coinbase creation", f"Failed: {e}"))

    return results


async def test_bluefin_creation():
    """Test Bluefin exchange creation."""
    print("Testing Bluefin Creation...")

    results = []

    try:
        # Create Bluefin client with dry_run=True
        bluefin = ExchangeFactory.create_exchange(exchange_type="bluefin", dry_run=True)

        if isinstance(bluefin, BaseExchange):
            results.append(
                ("✅", "Bluefin creation", "Successfully created BluefinClient")
            )

            # Test basic properties
            if hasattr(bluefin, "exchange_name"):
                results.append(
                    ("✅", "Bluefin properties", f"Name: {bluefin.exchange_name}")
                )
            else:
                results.append(("❌", "Bluefin properties", "Missing exchange_name"))

            # Test dry_run property
            if hasattr(bluefin, "dry_run"):
                results.append(("✅", "Bluefin dry_run", f"Dry run: {bluefin.dry_run}"))
            else:
                results.append(("❌", "Bluefin dry_run", "Missing dry_run property"))

            # Test DEX properties
            if hasattr(bluefin, "is_decentralized"):
                results.append(
                    (
                        "✅",
                        "Bluefin DEX properties",
                        f"Is DEX: {bluefin.is_decentralized}",
                    )
                )
            else:
                results.append(
                    ("❌", "Bluefin DEX properties", "Missing is_decentralized")
                )

            # Test connection status
            try:
                connected = bluefin.is_connected()
                results.append(
                    ("✅", "Bluefin connection check", f"Connected: {connected}")
                )
            except Exception as e:
                results.append(("❌", "Bluefin connection check", f"Error: {e}"))

            # Test connection status structure
            try:
                status = bluefin.get_connection_status()
                if isinstance(status, dict) and "connected" in status:
                    results.append(
                        (
                            "✅",
                            "Bluefin status structure",
                            f"Keys: {list(status.keys())}",
                        )
                    )
                else:
                    results.append(
                        ("❌", "Bluefin status structure", f"Invalid: {type(status)}")
                    )
            except Exception as e:
                results.append(("❌", "Bluefin status structure", f"Error: {e}"))

        else:
            results.append(("❌", "Bluefin creation", f"Wrong type: {type(bluefin)}"))

    except Exception as e:
        results.append(("❌", "Bluefin creation", f"Failed: {e}"))

    return results


async def test_balance_operations():
    """Test balance operations on both exchanges."""
    print("Testing Balance Operations...")

    results = []

    for exchange_type in ["coinbase", "bluefin"]:
        try:
            exchange = ExchangeFactory.create_exchange(
                exchange_type=exchange_type, dry_run=True
            )

            # Test balance query with error handling
            try:
                balance = await exchange.get_account_balance_with_error_handling()
                if isinstance(balance, Decimal):
                    results.append(
                        ("✅", f"{exchange_type} balance query", f"Balance: ${balance}")
                    )
                else:
                    results.append(
                        (
                            "❌",
                            f"{exchange_type} balance query",
                            f"Invalid type: {type(balance)}",
                        )
                    )
            except Exception as e:
                results.append(("❌", f"{exchange_type} balance query", f"Error: {e}"))

            # Test balance validation
            if hasattr(exchange, "validate_balance_update"):
                try:
                    validation = await exchange.validate_balance_update(
                        Decimal("100.00"), "test_validation"
                    )
                    if validation.get("valid"):
                        results.append(
                            (
                                "✅",
                                f"{exchange_type} balance validation",
                                "Validation passed",
                            )
                        )
                    else:
                        results.append(
                            (
                                "❌",
                                f"{exchange_type} balance validation",
                                "Validation failed",
                            )
                        )
                except Exception as e:
                    results.append(
                        ("❌", f"{exchange_type} balance validation", f"Error: {e}")
                    )

        except Exception as e:
            results.append(
                ("❌", f"{exchange_type} balance operations", f"Setup failed: {e}")
            )

    return results


async def test_position_operations():
    """Test position operations on both exchanges."""
    print("Testing Position Operations...")

    results = []

    for exchange_type in ["coinbase", "bluefin"]:
        try:
            exchange = ExchangeFactory.create_exchange(
                exchange_type=exchange_type, dry_run=True
            )

            # Test position query with error handling
            try:
                positions = await exchange.get_positions_with_error_handling()
                if isinstance(positions, list):
                    results.append(
                        (
                            "✅",
                            f"{exchange_type} position query",
                            f"Positions: {len(positions)}",
                        )
                    )
                else:
                    results.append(
                        (
                            "❌",
                            f"{exchange_type} position query",
                            f"Invalid type: {type(positions)}",
                        )
                    )
            except Exception as e:
                results.append(("❌", f"{exchange_type} position query", f"Error: {e}"))

            # Test futures positions if available
            if hasattr(exchange, "get_futures_positions"):
                try:
                    futures_pos = await exchange.get_futures_positions()
                    if isinstance(futures_pos, list):
                        results.append(
                            (
                                "✅",
                                f"{exchange_type} futures positions",
                                f"Positions: {len(futures_pos)}",
                            )
                        )
                    else:
                        results.append(
                            (
                                "❌",
                                f"{exchange_type} futures positions",
                                f"Invalid type: {type(futures_pos)}",
                            )
                        )
                except Exception as e:
                    results.append(
                        ("❌", f"{exchange_type} futures positions", f"Error: {e}")
                    )

        except Exception as e:
            results.append(
                ("❌", f"{exchange_type} position operations", f"Setup failed: {e}")
            )

    return results


async def test_error_handling():
    """Test error handling capabilities."""
    print("Testing Error Handling...")

    results = []

    for exchange_type in ["coinbase", "bluefin"]:
        try:
            exchange = ExchangeFactory.create_exchange(
                exchange_type=exchange_type, dry_run=True
            )

            # Test error boundary status
            if hasattr(exchange, "get_error_boundary_status"):
                try:
                    status = exchange.get_error_boundary_status()
                    if isinstance(status, dict):
                        results.append(
                            (
                                "✅",
                                f"{exchange_type} error boundary",
                                f"Status keys: {list(status.keys())}",
                            )
                        )
                    else:
                        results.append(
                            (
                                "❌",
                                f"{exchange_type} error boundary",
                                f"Invalid type: {type(status)}",
                            )
                        )
                except Exception as e:
                    results.append(
                        ("❌", f"{exchange_type} error boundary", f"Error: {e}")
                    )
            else:
                results.append(
                    ("⚠️", f"{exchange_type} error boundary", "Method not found")
                )

            # Test balance validation status
            if hasattr(exchange, "get_balance_validation_status"):
                try:
                    validation_status = exchange.get_balance_validation_status()
                    if isinstance(validation_status, dict):
                        enabled = validation_status.get("validation_enabled", False)
                        results.append(
                            (
                                "✅",
                                f"{exchange_type} validation status",
                                f"Enabled: {enabled}",
                            )
                        )
                    else:
                        results.append(
                            (
                                "❌",
                                f"{exchange_type} validation status",
                                f"Invalid type: {type(validation_status)}",
                            )
                        )
                except Exception as e:
                    results.append(
                        ("❌", f"{exchange_type} validation status", f"Error: {e}")
                    )
            else:
                results.append(
                    ("⚠️", f"{exchange_type} validation status", "Method not found")
                )

        except Exception as e:
            results.append(
                ("❌", f"{exchange_type} error handling", f"Setup failed: {e}")
            )

    return results


def print_results(test_name: str, results: list[tuple]):
    """Print test results in a formatted way."""
    print(f"\n{test_name}:")
    print("-" * 50)

    for status, test, details in results:
        print(f"  {status} {test}: {details}")

    passed = sum(1 for status, _, _ in results if status == "✅")
    total = len(results)
    print(f"\n  Summary: {passed}/{total} tests passed")


async def main():
    """Main test function."""
    print("=" * 80)
    print("CORE EXCHANGE INTEGRATION VALIDATION")
    print("=" * 80)

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise
        format="%(levelname)s: %(message)s",
    )

    all_results = []

    # Run all tests
    tests = [
        ("Exchange Factory", test_exchange_factory),
        ("Coinbase Creation", test_coinbase_creation),
        ("Bluefin Creation", test_bluefin_creation),
        ("Balance Operations", test_balance_operations),
        ("Position Operations", test_position_operations),
        ("Error Handling", test_error_handling),
    ]

    for test_name, test_func in tests:
        try:
            results = await test_func()
            print_results(test_name, results)
            all_results.extend(results)
        except Exception as e:
            print(f"\n{test_name}:")
            print("-" * 50)
            print(f"  ❌ Test execution failed: {e}")
            all_results.append(("❌", test_name, f"Execution failed: {e}"))

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    passed = sum(1 for status, _, _ in all_results if status == "✅")
    warnings = sum(1 for status, _, _ in all_results if status == "⚠️")
    failed = sum(1 for status, _, _ in all_results if status == "❌")
    total = len(all_results)

    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"❌ Failed: {failed}")

    # Check for critical failures
    critical_failures = [
        (test, details)
        for status, test, details in all_results
        if status == "❌"
        and any(
            keyword in test.lower() for keyword in ["creation", "factory", "connection"]
        )
    ]

    if critical_failures:
        print(f"\n❌ CRITICAL FAILURES ({len(critical_failures)}):")
        for test, details in critical_failures:
            print(f"  - {test}: {details}")
        return 1
    print("\n🎉 CORE EXCHANGE INTEGRATIONS VALIDATED!")
    if failed > 0:
        print(f"   Note: {failed} non-critical tests failed")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
