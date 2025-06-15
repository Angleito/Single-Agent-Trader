#!/usr/bin/env python3
"""
Quick test script to verify position checking functionality.
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

# Add bot module to path
sys.path.append(str(Path(__file__).parent))

from bot.config import settings
from bot.exchange.coinbase import CoinbaseClient


async def test_position_check():
    """Test the position checking functionality."""
    print("Testing position check functionality...")
    
    # Initialize exchange client
    exchange_client = CoinbaseClient()
    
    # Connect to exchange
    connected = await exchange_client.connect()
    if not connected:
        print("❌ Failed to connect to exchange")
        return
    
    print("✅ Connected to exchange")
    
    try:
        # Get trading symbol
        symbol = "ETH-USD"
        actual_trading_symbol = await exchange_client.get_trading_symbol(symbol)
        print(f"📈 Trading symbol: {actual_trading_symbol}")
        
        # Check for futures positions
        print("🔍 Checking for futures positions...")
        futures_positions = await exchange_client.get_futures_positions(actual_trading_symbol)
        
        if futures_positions:
            print(f"📊 Found {len(futures_positions)} futures positions:")
            for pos in futures_positions:
                # Handle both dict and object formats
                if hasattr(pos, 'get'):
                    # Dictionary format
                    symbol_key = pos.get('symbol') or pos.get('product_id')
                    size = pos.get('size', 0)
                    side = pos.get('side', 'UNKNOWN')
                    entry_price = pos.get('entry_price', 0)
                    unrealized_pnl = pos.get('unrealized_pnl', 0)
                else:
                    # Object format
                    symbol_key = getattr(pos, 'symbol', None) or getattr(pos, 'product_id', None)
                    size = getattr(pos, 'size', 0)
                    side = getattr(pos, 'side', 'UNKNOWN')
                    entry_price = getattr(pos, 'entry_price', 0)
                    unrealized_pnl = getattr(pos, 'unrealized_pnl', 0)
                
                print(f"  • {symbol_key}: {side} {size} @ ${entry_price} (PnL: ${unrealized_pnl})")
                
                if symbol_key == actual_trading_symbol and float(size) > 0:
                    print(f"⚠️  Found matching position for {actual_trading_symbol}")
                    return True
        else:
            print("✅ No futures positions found")
        
        # Check for spot positions (if not using futures)
        print("🔍 Checking for spot positions...")
        spot_positions = await exchange_client.get_positions(actual_trading_symbol)
        
        if spot_positions:
            print(f"📊 Found {len(spot_positions)} spot positions:")
            for pos in spot_positions:
                # Handle both dict and object formats
                if hasattr(pos, 'get'):
                    # Dictionary format
                    symbol_key = pos.get('symbol') or pos.get('product_id')
                    size = pos.get('size', 0)
                    side = pos.get('side', 'UNKNOWN')
                else:
                    # Object format
                    symbol_key = getattr(pos, 'symbol', None) or getattr(pos, 'product_id', None)
                    size = getattr(pos, 'size', 0)
                    side = getattr(pos, 'side', 'UNKNOWN')
                
                print(f"  • {symbol_key}: {side} {size}")
                
                if symbol_key == actual_trading_symbol and float(size) > 0:
                    print(f"⚠️  Found matching position for {actual_trading_symbol}")
                    return True
        else:
            print("✅ No spot positions found")
            
        print("✅ Position check completed - no existing positions")
        return False
        
    except Exception as e:
        print(f"❌ Error during position check: {e}")
        return False
    
    finally:
        await exchange_client.disconnect()


if __name__ == "__main__":
    result = asyncio.run(test_position_check())
    print(f"\n🏁 Test completed: {'POSITIONS FOUND' if result else 'NO POSITIONS'}")