#!/usr/bin/env python3
"""
Test script to verify Bluefin exchange connection and basic functionality.

Usage:
    python scripts/test_bluefin_connection.py
"""

import asyncio
import logging
import sys
from decimal import Decimal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import settings
from bot.exchange.factory import ExchangeFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_bluefin_connection():
    """Test Bluefin exchange connection and basic operations."""
    
    print("\n" + "="*50)
    print("BLUEFIN EXCHANGE CONNECTION TEST")
    print("="*50 + "\n")
    
    # Create Bluefin client in dry-run mode for safety
    try:
        exchange = ExchangeFactory.create_exchange(
            exchange_type="bluefin",
            dry_run=True  # Always use dry-run for this test
        )
        print("✓ Bluefin client created successfully")
    except Exception as e:
        print(f"✗ Failed to create Bluefin client: {e}")
        return
    
    # Test connection
    try:
        connected = await exchange.connect()
        if connected:
            print("✓ Connected to Bluefin successfully")
        else:
            print("✗ Failed to connect to Bluefin")
            return
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return
    
    # Get connection status
    try:
        status = exchange.get_connection_status()
        print("\nConnection Status:")
        print(f"  Exchange: {status.get('exchange', 'Unknown')}")
        print(f"  Network: {status.get('network', 'Unknown')}")
        print(f"  Blockchain: {status.get('blockchain', 'Unknown')}")
        print(f"  Connected: {status.get('connected', False)}")
        print(f"  Trading Mode: {status.get('trading_mode', 'Unknown')}")
        print(f"  Account: {status.get('account_address', 'Not available')}")
    except Exception as e:
        print(f"✗ Failed to get connection status: {e}")
    
    # Test account balance (dry-run will return mock balance)
    try:
        balance = await exchange.get_account_balance()
        print(f"\n✓ Account Balance: ${balance}")
    except Exception as e:
        print(f"✗ Failed to get account balance: {e}")
    
    # Test getting positions
    try:
        positions = await exchange.get_positions()
        print(f"\n✓ Current Positions: {len(positions)}")
        for pos in positions:
            print(f"  - {pos.symbol}: {pos.side} {pos.size} @ {pos.entry_price}")
    except Exception as e:
        print(f"✗ Failed to get positions: {e}")
    
    # Test market order placement (dry-run)
    try:
        print("\nTesting order placement (DRY RUN):")
        order = await exchange.place_market_order(
            symbol="ETH-PERP",
            side="BUY",
            quantity=Decimal("0.1")
        )
        if order:
            print(f"✓ Market order placed: {order.id}")
            print(f"  Symbol: {order.symbol}")
            print(f"  Side: {order.side}")
            print(f"  Quantity: {order.quantity}")
            print(f"  Status: {order.status}")
        else:
            print("✗ Failed to place market order")
    except Exception as e:
        print(f"✗ Order placement error: {e}")
    
    # Disconnect
    try:
        await exchange.disconnect()
        print("\n✓ Disconnected from Bluefin")
    except Exception as e:
        print(f"✗ Disconnect error: {e}")
    
    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)
    
    # Summary
    print("\nNOTE: This test ran in DRY RUN mode.")
    print("No real trades were executed.")
    print("\nTo use Bluefin for live trading:")
    print("1. Set EXCHANGE__EXCHANGE_TYPE=bluefin in your .env")
    print("2. Add your Sui wallet private key to EXCHANGE__BLUEFIN_PRIVATE_KEY")
    print("3. Ensure your wallet has USDC and SUI for gas")
    print("4. Run the bot with appropriate risk settings")


if __name__ == "__main__":
    asyncio.run(test_bluefin_connection())