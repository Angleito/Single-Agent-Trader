#!/usr/bin/env python3
"""
Simple test script for Bluefin integration
Tests the basic functionality without Docker complications
"""

import asyncio
import os
import sys
from decimal import Decimal

# Set environment variables for testing
os.environ['EXCHANGE__EXCHANGE_TYPE'] = 'bluefin'
os.environ['SYSTEM__DRY_RUN'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after setting env vars
from bot.exchange.factory import create_exchange
from bot.types import TradeAction


async def test_bluefin_integration():
    """Test Bluefin integration for perpetual trading."""
    print("=" * 60)
    print("BLUEFIN PERPETUAL TRADING TEST")
    print("=" * 60)
    print()
    
    # Test 1: Connection and basic info
    print("Test 1: Testing Bluefin connection...")
    try:
        exchange = create_exchange(exchange_type='bluefin', dry_run=True)
        connected = await exchange.connect()
        
        if not connected:
            print("❌ Failed to connect to Bluefin")
            return False
            
        print("✅ Connected successfully")
        
        # Get connection status
        status = exchange.get_connection_status()
        print("\nConnection Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Verify it's configured for perpetuals
        assert exchange.supports_futures == True, "Bluefin should support futures"
        assert exchange.is_decentralized == True, "Bluefin should be decentralized"
        print("\n✅ Perpetual futures support confirmed")
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Symbol conversion
    print("\n" + "-" * 40)
    print("Test 2: Symbol conversion for perpetuals...")
    try:
        from bot.exchange.bluefin import BluefinClient
        client = BluefinClient(dry_run=True)
        
        test_symbols = {
            'BTC-USD': 'BTC-PERP',
            'ETH-USD': 'ETH-PERP',
            'SOL-USD': 'SOL-PERP',
            'SUI-USD': 'SUI-PERP'
        }
        
        print("Symbol conversions:")
        for standard, expected in test_symbols.items():
            converted = client._convert_symbol(standard)
            status = "✅" if converted == expected else "❌"
            print(f"  {standard} -> {converted} {status}")
            assert converted == expected, f"Expected {expected}, got {converted}"
        
        print("\n✅ Symbol conversion test passed")
        
    except Exception as e:
        print(f"❌ Symbol conversion test failed: {e}")
        return False
    
    # Test 3: Paper trading perpetual order
    print("\n" + "-" * 40)
    print("Test 3: Testing perpetual order placement (paper trading)...")
    try:
        # Create a test trade action
        trade_action = TradeAction(
            action='LONG',
            size_pct=10.0,
            leverage=5,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            confidence=0.85,
            reasoning='Test perpetual trade on Bluefin'
        )
        
        # Mock current price
        current_price = Decimal('3500.00')  # ETH price
        
        print(f"\nPlacing test order:")
        print(f"  Action: {trade_action.action}")
        print(f"  Size: {trade_action.size_pct}%")
        print(f"  Leverage: {trade_action.leverage}x")
        print(f"  Stop Loss: {trade_action.stop_loss_pct}%")
        print(f"  Take Profit: {trade_action.take_profit_pct}%")
        
        # Execute trade
        order = await exchange.execute_trade_action(
            trade_action,
            'ETH-USD',  # Will be converted to ETH-PERP
            current_price
        )
        
        if order:
            print(f"\n✅ Order placed successfully:")
            print(f"  Order ID: {order.id}")
            print(f"  Symbol: {order.symbol}")
            print(f"  Side: {order.side}")
            print(f"  Type: {order.type}")
            print(f"  Quantity: {order.quantity}")
            print(f"  Status: {order.status}")
        else:
            print("❌ Failed to place order")
            return False
            
    except Exception as e:
        print(f"❌ Order placement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Get positions (should be empty in paper trading)
    print("\n" + "-" * 40)
    print("Test 4: Testing position retrieval...")
    try:
        positions = await exchange.get_positions()
        print(f"Positions found: {len(positions)}")
        
        # Also test futures-specific method
        futures_positions = await exchange.get_futures_positions()
        print(f"Futures positions found: {len(futures_positions)}")
        
        print("✅ Position retrieval test passed")
        
    except Exception as e:
        print(f"❌ Position retrieval test failed: {e}")
        return False
    
    # Disconnect
    await exchange.disconnect()
    print("\n✅ Disconnected successfully")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("Bluefin is properly configured for perpetual trading")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    # Check if bluefin SDK is available
    try:
        import bluefin_v2_client
        print("✅ Bluefin SDK is installed")
    except ImportError:
        print("⚠️  Bluefin SDK not installed - running in mock mode")
        print("   The bot will work in paper trading mode")
    
    print()
    
    # Run the test
    success = asyncio.run(test_bluefin_integration())
    sys.exit(0 if success else 1)