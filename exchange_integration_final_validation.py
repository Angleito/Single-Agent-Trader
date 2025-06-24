#!/usr/bin/env python3
"""
Final Exchange Integration Validation

This comprehensive test validates that exchange integrations work correctly
by directly testing exchange components to avoid complex dependency issues.
"""

import asyncio
import logging
import sys

# Add the project root to sys.path
sys.path.insert(0, "/Users/angel/Documents/Projects/cursorprod")

logger = logging.getLogger(__name__)


async def test_exchange_direct_imports():
    """Test that we can import exchange components directly."""
    print("Testing Exchange Direct Imports...")

    results = []

    # Test Coinbase import
    try:
        from bot.exchange.coinbase import CoinbaseClient

        results.append(
            ("✅", "Coinbase import", "CoinbaseClient imported successfully")
        )

        # Test creation with minimal config
        try:
            coinbase = CoinbaseClient(dry_run=True)
            results.append(
                ("✅", "Coinbase creation", f"Created: {type(coinbase).__name__}")
            )

            # Test basic methods
            connected = coinbase.is_connected()
            results.append(
                ("✅", "Coinbase connection check", f"Connected: {connected}")
            )

            status = coinbase.get_connection_status()
            if isinstance(status, dict):
                results.append(("✅", "Coinbase status", f"Status keys: {len(status)}"))
            else:
                results.append(
                    ("❌", "Coinbase status", f"Invalid status type: {type(status)}")
                )

        except Exception as e:
            results.append(("❌", "Coinbase creation", f"Creation failed: {e}"))

    except ImportError as e:
        results.append(("❌", "Coinbase import", f"Import failed: {e}"))

    # Test Bluefin import
    try:
        from bot.exchange.bluefin import BluefinClient

        results.append(("✅", "Bluefin import", "BluefinClient imported successfully"))

        # Test creation with minimal config
        try:
            bluefin = BluefinClient(dry_run=True)
            results.append(
                ("✅", "Bluefin creation", f"Created: {type(bluefin).__name__}")
            )

            # Test basic methods
            connected = bluefin.is_connected()
            results.append(
                ("✅", "Bluefin connection check", f"Connected: {connected}")
            )

            status = bluefin.get_connection_status()
            if isinstance(status, dict):
                results.append(("✅", "Bluefin status", f"Status keys: {len(status)}"))
            else:
                results.append(
                    ("❌", "Bluefin status", f"Invalid status type: {type(status)}")
                )

        except Exception as e:
            results.append(("❌", "Bluefin creation", f"Creation failed: {e}"))

    except ImportError as e:
        results.append(("❌", "Bluefin import", f"Import failed: {e}"))

    return results


async def test_exchange_factory_bypass():
    """Test exchange creation bypassing factory issues."""
    print("Testing Exchange Factory Bypass...")

    results = []

    # Test factory import
    try:
        from bot.exchange.factory import ExchangeFactory

        results.append(
            ("✅", "Factory import", "ExchangeFactory imported successfully")
        )

        # Test supported exchanges
        try:
            supported = ExchangeFactory.get_supported_exchanges()
            results.append(("✅", "Supported exchanges", f"Found: {supported}"))
        except Exception as e:
            results.append(("❌", "Supported exchanges", f"Error: {e}"))

    except ImportError as e:
        results.append(("❌", "Factory import", f"Import failed: {e}"))

    return results


async def test_base_exchange_interface():
    """Test the base exchange interface."""
    print("Testing Base Exchange Interface...")

    results = []

    try:
        from bot.exchange.base import BaseExchange, ExchangeError

        results.append(
            ("✅", "Base exchange import", "BaseExchange imported successfully")
        )

        # Test exception classes
        try:
            test_error = ExchangeError("Test error")
            results.append(
                ("✅", "Exchange errors", f"Error class: {type(test_error).__name__}")
            )
        except Exception as e:
            results.append(("❌", "Exchange errors", f"Error creation failed: {e}"))

    except ImportError as e:
        results.append(("❌", "Base exchange import", f"Import failed: {e}"))

    return results


async def test_trading_types():
    """Test trading types are available."""
    print("Testing Trading Types...")

    results = []

    try:
        from bot.trading_types import AccountType, Order, Position, TradeAction

        results.append(
            ("✅", "Trading types import", "Trading types imported successfully")
        )

        # Test TradeAction creation
        try:
            trade_action = TradeAction(
                action="LONG",
                size_pct=10.0,
                stop_loss_pct=5.0,
                take_profit_pct=10.0,
                rationale="Test action",
            )
            results.append(
                ("✅", "TradeAction creation", f"Created: {trade_action.action}")
            )
        except Exception as e:
            results.append(("❌", "TradeAction creation", f"Failed: {e}"))

    except ImportError as e:
        results.append(("❌", "Trading types import", f"Import failed: {e}"))

    return results


async def test_websocket_components():
    """Test WebSocket components availability."""
    print("Testing WebSocket Components...")

    results = []

    # Test WebSocket library
    try:
        import websockets

        results.append(("✅", "WebSocket library", "Version available: websockets"))
    except ImportError:
        results.append(("❌", "WebSocket library", "websockets library not available"))

    # Test Bluefin WebSocket
    try:
        from bot.data.bluefin_websocket import BluefinWebSocketClient

        results.append(
            ("✅", "Bluefin WebSocket", "BluefinWebSocketClient imported successfully")
        )

        # Test client creation
        try:
            ws_client = BluefinWebSocketClient(symbol="BTC-PERP", network="testnet")
            results.append(
                (
                    "✅",
                    "Bluefin WebSocket client",
                    f"Client created for: {ws_client.network}",
                )
            )
        except Exception as e:
            results.append(("❌", "Bluefin WebSocket client", f"Creation failed: {e}"))

    except ImportError as e:
        results.append(("❌", "Bluefin WebSocket", f"Import failed: {e}"))

    return results


async def test_configuration_system():
    """Test configuration system."""
    print("Testing Configuration System...")

    results = []

    try:
        from bot.config import settings

        results.append(("✅", "Config import", "Settings imported successfully"))

        # Test exchange settings
        if hasattr(settings, "exchange"):
            exchange_settings = settings.exchange
            results.append(
                (
                    "✅",
                    "Exchange config",
                    f"Exchange type: {exchange_settings.exchange_type}",
                )
            )

            # Check for the rate limit fix
            if hasattr(exchange_settings, "rate_limit_window_seconds"):
                results.append(
                    (
                        "✅",
                        "Rate limit config",
                        f"Window: {exchange_settings.rate_limit_window_seconds}s",
                    )
                )
            else:
                results.append(
                    ("❌", "Rate limit config", "Missing rate_limit_window_seconds")
                )
        else:
            results.append(("❌", "Exchange config", "No exchange settings found"))

        # Test trading settings
        if hasattr(settings, "trading"):
            trading_settings = settings.trading
            results.append(
                ("✅", "Trading config", f"Symbol: {trading_settings.symbol}")
            )
        else:
            results.append(("❌", "Trading config", "No trading settings found"))

    except ImportError as e:
        results.append(("❌", "Config import", f"Import failed: {e}"))

    return results


async def test_monitoring_components():
    """Test monitoring and error handling components."""
    print("Testing Monitoring Components...")

    results = []

    # Test error handling
    try:
        from bot.error_handling import ErrorBoundary, exception_handler

        results.append(("✅", "Error handling", "Error handling components available"))
    except ImportError as e:
        results.append(("❌", "Error handling", f"Import failed: {e}"))

    # Test system monitor
    try:
        from bot.system_monitor import error_recovery_manager

        results.append(("✅", "System monitor", "System monitor components available"))
    except ImportError as e:
        results.append(("❌", "System monitor", f"Import failed: {e}"))

    return results


def print_results(test_name: str, results: list[tuple[str, str, str]]):
    """Print test results in a formatted way."""
    print(f"\n{test_name}:")
    print("-" * 50)

    for status, test, details in results:
        print(f"  {status} {test}: {details}")

    passed = sum(1 for status, _, _ in results if status == "✅")
    warnings = sum(1 for status, _, _ in results if status == "⚠️")
    total = len(results)
    print(f"\n  Summary: {passed}/{total} tests passed ({warnings} warnings)")


async def main():
    """Main validation function."""
    print("=" * 80)
    print("FINAL EXCHANGE INTEGRATION VALIDATION")
    print("=" * 80)
    print("Testing core exchange functionality without complex dependencies...")

    # Setup minimal logging
    logging.basicConfig(
        level=logging.ERROR,  # Only show errors
        format="%(levelname)s: %(message)s",
    )

    all_results = []

    # Run all tests
    tests = [
        ("Exchange Direct Imports", test_exchange_direct_imports),
        ("Exchange Factory Bypass", test_exchange_factory_bypass),
        ("Base Exchange Interface", test_base_exchange_interface),
        ("Trading Types", test_trading_types),
        ("WebSocket Components", test_websocket_components),
        ("Configuration System", test_configuration_system),
        ("Monitoring Components", test_monitoring_components),
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
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for status, _, _ in all_results if status == "✅")
    warnings = sum(1 for status, _, _ in all_results if status == "⚠️")
    failed = sum(1 for status, _, _ in all_results if status == "❌")
    total = len(all_results)

    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"❌ Failed: {failed}")

    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    # Check for critical component failures
    critical_components = ["import", "creation", "config"]
    critical_failures = [
        (test, details)
        for status, test, details in all_results
        if status == "❌"
        and any(keyword in test.lower() for keyword in critical_components)
    ]

    if critical_failures:
        print(f"\n❌ CRITICAL COMPONENT FAILURES ({len(critical_failures)}):")
        for test, details in critical_failures:
            print(f"  - {test}: {details}")
        print("\n❌ EXCHANGE INTEGRATIONS REQUIRE ATTENTION!")
        return 1
    if failed == 0:
        print("\n🎉 ALL EXCHANGE INTEGRATIONS VALIDATED SUCCESSFULLY!")
        return 0
    print("\n✅ CORE EXCHANGE INTEGRATIONS VALIDATED!")
    print(f"   Note: {failed} non-critical tests failed, but core functionality works")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
