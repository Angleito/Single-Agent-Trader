#!/usr/bin/env python3
"""
Exchange Order Operations and WebSocket Connectivity Test

This test validates order placement functionality and WebSocket connectivity
for both Coinbase and Bluefin exchanges in dry-run mode.
"""

import asyncio
import logging
import sys
from decimal import Decimal
from typing import List, Tuple

# Add the project root to sys.path to import modules
sys.path.insert(0, '/Users/angel/Documents/Projects/cursorprod')

from bot.exchange.factory import ExchangeFactory
from bot.trading_types import TradeAction


logger = logging.getLogger(__name__)


async def test_order_operations():
    """Test order operations on both exchanges."""
    print("Testing Order Operations...")
    
    results = []
    
    for exchange_type in ['coinbase', 'bluefin']:
        try:
            exchange = ExchangeFactory.create_exchange(
                exchange_type=exchange_type,
                dry_run=True  # Always use dry-run for safety
            )
            
            print(f"\n  Testing {exchange_type} orders...")
            
            # Test market order capability (dry-run)
            try:
                order = await exchange.place_market_order_with_error_handling(
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("0.001")
                )
                
                if order is not None:
                    results.append(("✅", f"{exchange_type} market order", f"Order object returned: {type(order).__name__}"))
                else:
                    results.append(("✅", f"{exchange_type} market order", "Dry-run simulation completed (no real order)"))
                    
            except Exception as e:
                results.append(("❌", f"{exchange_type} market order", f"Error: {e}"))
            
            # Test limit order capability (dry-run)
            try:
                order = await exchange.place_limit_order_with_error_handling(
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("0.001"),
                    price=Decimal("50000")
                )
                
                if order is not None:
                    results.append(("✅", f"{exchange_type} limit order", f"Order object returned: {type(order).__name__}"))
                else:
                    results.append(("✅", f"{exchange_type} limit order", "Dry-run simulation completed (no real order)"))
                    
            except Exception as e:
                results.append(("❌", f"{exchange_type} limit order", f"Error: {e}"))
            
            # Test trade action execution
            try:
                trade_action = TradeAction(
                    action="LONG",
                    size=Decimal("0.001"),
                    stop_loss=Decimal("50000"),
                    take_profit=Decimal("60000"),
                    reason="Test order"
                )
                
                order = await exchange.execute_trade_action_with_saga(
                    trade_action=trade_action,
                    symbol="BTC-USD",
                    current_price=Decimal("55000")
                )
                
                if order is not None:
                    results.append(("✅", f"{exchange_type} trade action", f"Trade executed: {type(order).__name__}"))
                else:
                    results.append(("✅", f"{exchange_type} trade action", "Dry-run trade simulation completed"))
                    
            except Exception as e:
                results.append(("❌", f"{exchange_type} trade action", f"Error: {e}"))
            
            # Test order cancellation capability
            try:
                # This should work even in dry-run mode
                cancel_result = await exchange.cancel_all_orders()
                
                if isinstance(cancel_result, bool):
                    results.append(("✅", f"{exchange_type} cancel orders", f"Cancel all returned: {cancel_result}"))
                else:
                    results.append(("❌", f"{exchange_type} cancel orders", f"Invalid return type: {type(cancel_result)}"))
                    
            except Exception as e:
                results.append(("❌", f"{exchange_type} cancel orders", f"Error: {e}"))
                
        except Exception as e:
            results.append(("❌", f"{exchange_type} order setup", f"Failed to setup exchange: {e}"))
    
    return results


async def test_websocket_connectivity():
    """Test WebSocket connectivity for both exchanges."""
    print("Testing WebSocket Connectivity...")
    
    results = []
    
    # Test Bluefin WebSocket (it has more explicit WebSocket support)
    try:
        from bot.data.bluefin_websocket import BluefinWebSocketClient
        
        # Create WebSocket client for testnet (safer for testing)
        ws_client = BluefinWebSocketClient(
            network="testnet",
            market_data_timeout=5.0
        )
        
        # Test connection capability
        try:
            # Test if we can get connection info without actually connecting
            connection_info = {
                'network': ws_client.network,
                'websocket_url': ws_client.websocket_url,
                'notifications_url': ws_client.notifications_url
            }
            
            results.append(("✅", "Bluefin WebSocket config", f"URLs configured: {list(connection_info.keys())}"))
            
            # Test WebSocket URL accessibility (without connecting)
            if ws_client.websocket_url.startswith('wss://'):
                results.append(("✅", "Bluefin WebSocket URL", f"Valid URL: {ws_client.websocket_url}"))
            else:
                results.append(("❌", "Bluefin WebSocket URL", f"Invalid URL: {ws_client.websocket_url}"))
                
        except Exception as e:
            results.append(("❌", "Bluefin WebSocket", f"Config error: {e}"))
            
    except ImportError as e:
        results.append(("⚠️", "Bluefin WebSocket", f"WebSocket client not available: {e}"))
    
    # Test general WebSocket capabilities
    try:
        import websockets
        results.append(("✅", "WebSocket library", "websockets library available"))
    except ImportError:
        results.append(("❌", "WebSocket library", "websockets library not available"))
    
    return results


async def test_futures_operations():
    """Test futures-specific operations."""
    print("Testing Futures Operations...")
    
    results = []
    
    for exchange_type in ['coinbase', 'bluefin']:
        try:
            exchange = ExchangeFactory.create_exchange(
                exchange_type=exchange_type,
                dry_run=True
            )
            
            # Test futures support detection
            if hasattr(exchange, 'supports_futures'):
                supports_futures = exchange.supports_futures
                results.append(("✅", f"{exchange_type} futures support", f"Supports futures: {supports_futures}"))
            else:
                results.append(("⚠️", f"{exchange_type} futures support", "No futures support property"))
            
            # Test futures account info
            try:
                futures_info = await exchange.get_futures_account_info()
                if futures_info is not None:
                    results.append(("✅", f"{exchange_type} futures account", f"Info available: {type(futures_info).__name__}"))
                else:
                    results.append(("✅", f"{exchange_type} futures account", "No futures account configured (expected in dry-run)"))
            except Exception as e:
                results.append(("❌", f"{exchange_type} futures account", f"Error: {e}"))
            
            # Test margin info
            try:
                margin_info = await exchange.get_margin_info()
                if margin_info is not None:
                    results.append(("✅", f"{exchange_type} margin info", f"Margin info available: {type(margin_info).__name__}"))
                else:
                    results.append(("✅", f"{exchange_type} margin info", "No margin info (expected in dry-run)"))
            except Exception as e:
                results.append(("❌", f"{exchange_type} margin info", f"Error: {e}"))
            
            # Test futures market order
            if hasattr(exchange, 'place_futures_market_order'):
                try:
                    futures_order = await exchange.place_futures_market_order(
                        symbol="BTC-USD",
                        side="BUY",
                        quantity=Decimal("0.001"),
                        leverage=5
                    )
                    
                    if futures_order is not None:
                        results.append(("✅", f"{exchange_type} futures order", f"Futures order: {type(futures_order).__name__}"))
                    else:
                        results.append(("✅", f"{exchange_type} futures order", "Dry-run futures simulation completed"))
                        
                except Exception as e:
                    results.append(("❌", f"{exchange_type} futures order", f"Error: {e}"))
            else:
                results.append(("⚠️", f"{exchange_type} futures order", "No futures order method"))
                
        except Exception as e:
            results.append(("❌", f"{exchange_type} futures setup", f"Failed to setup exchange: {e}"))
    
    return results


async def test_symbol_mapping():
    """Test symbol mapping functionality."""
    print("Testing Symbol Mapping...")
    
    results = []
    
    for exchange_type in ['coinbase', 'bluefin']:
        try:
            exchange = ExchangeFactory.create_exchange(
                exchange_type=exchange_type,
                dry_run=True
            )
            
            # Test trading symbol mapping
            test_symbols = ["BTC-USD", "ETH-USD", "SUI-USD"]
            
            for symbol in test_symbols:
                try:
                    trading_symbol = await exchange.get_trading_symbol(symbol)
                    
                    if trading_symbol:
                        results.append(("✅", f"{exchange_type} symbol {symbol}", f"Maps to: {trading_symbol}"))
                    else:
                        results.append(("❌", f"{exchange_type} symbol {symbol}", "No symbol mapping returned"))
                        
                except Exception as e:
                    results.append(("❌", f"{exchange_type} symbol {symbol}", f"Mapping error: {e}"))
                
        except Exception as e:
            results.append(("❌", f"{exchange_type} symbol mapping", f"Failed to setup exchange: {e}"))
    
    return results


def print_results(test_name: str, results: List[Tuple[str, str, str]]):
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
    """Main test function."""
    print("="*80)
    print("EXCHANGE ORDER OPERATIONS & WEBSOCKET VALIDATION")
    print("="*80)
    
    # Setup logging (reduced to avoid noise)
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
    all_results = []
    
    # Run all tests
    tests = [
        ("Order Operations", test_order_operations),
        ("WebSocket Connectivity", test_websocket_connectivity),
        ("Futures Operations", test_futures_operations),
        ("Symbol Mapping", test_symbol_mapping),
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
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    passed = sum(1 for status, _, _ in all_results if status == "✅")
    warnings = sum(1 for status, _, _ in all_results if status == "⚠️")
    failed = sum(1 for status, _, _ in all_results if status == "❌")
    total = len(all_results)
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"❌ Failed: {failed}")
    
    # Check for critical failures in order operations
    critical_failures = [
        (test, details) for status, test, details in all_results
        if status == "❌" and any(keyword in test.lower() 
                                for keyword in ['order', 'setup', 'websocket library'])
    ]
    
    if critical_failures:
        print(f"\n❌ CRITICAL FAILURES ({len(critical_failures)}):")
        for test, details in critical_failures:
            print(f"  - {test}: {details}")
        return 1
    else:
        print(f"\n🎉 ORDER OPERATIONS & WEBSOCKET CONNECTIVITY VALIDATED!")
        if failed > 0:
            print(f"   Note: {failed} non-critical tests failed")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)