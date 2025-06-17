#!/usr/bin/env python3
"""
Test script to verify Bluefin position query functionality.
"""

import asyncio
import logging
import os
import sys
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.exchange.bluefin import BluefinClient
from bot.types import Position


# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_bluefin_positions():
    """Test Bluefin position query functionality."""
    print("Testing Bluefin position query...")
    
    try:
        # Initialize BluefinClient with environment configuration
        # Using dry_run=False to connect to real exchange
        client = BluefinClient(
            private_key=os.getenv("EXCHANGE__BLUEFIN_PRIVATE_KEY", ""),
            network=os.getenv("EXCHANGE__BLUEFIN_NETWORK", "mainnet"),
            dry_run=False
        )
        
        # Connect to the exchange
        print("Connecting to Bluefin service...")
        await client.connect()
        print("Connected to Bluefin service")
        
        # Query positions
        print("\nQuerying positions...")
        positions: List[Position] = await client.get_futures_positions()
        
        # Display results
        if positions:
            print(f"\nFound {len(positions)} positions:")
            for i, position in enumerate(positions, 1):
                print(f"\n- Position {i}:")
                print(f"  Symbol: {position.symbol}")
                print(f"  Side: {position.side}")
                print(f"  Size: {position.size}")
                print(f"  Entry Price: {position.entry_price}")
                print(f"  Unrealized PnL: ${position.unrealized_pnl:.2f}")
                print(f"  Realized PnL: ${position.realized_pnl:.2f}")
                if position.margin_used:
                    print(f"  Margin Used: ${position.margin_used:.2f}")
                if position.leverage:
                    print(f"  Leverage: {position.leverage}x")
                if position.liquidation_price:
                    print(f"  Liquidation Price: ${position.liquidation_price:.2f}")
        else:
            print("\nNo open positions found.")
        
        # Also try to get raw position data for debugging
        print("\n\nFetching raw position data for debugging...")
        try:
            raw_positions = await client._get_positions()
            if raw_positions:
                print(f"Raw positions data: {raw_positions}")
            else:
                print("No raw position data returned")
        except Exception as e:
            print(f"Error fetching raw positions: {e}")
        
    except Exception as e:
        logger.error(f"Error during position query: {type(e).__name__}: {e}")
        print(f"\nError: {type(e).__name__}: {e}")
        
        # Print additional debug info
        if hasattr(e, '__dict__'):
            print(f"Error details: {e.__dict__}")
        
        # Check if it's a configuration issue
        if not os.getenv("EXCHANGE__BLUEFIN_PRIVATE_KEY"):
            print("\n⚠️  EXCHANGE__BLUEFIN_PRIVATE_KEY environment variable not set!")
            print("Please set your Bluefin private key in the .env file or environment")
        
        raise
    
    finally:
        # Clean up
        if 'client' in locals():
            await client.disconnect()
            print("\nDisconnected from Bluefin service")


async def main():
    """Main entry point."""
    try:
        await test_bluefin_positions()
        print("\n✅ Test completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())