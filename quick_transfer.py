#!/usr/bin/env python3
"""Quick script to transfer funds from spot to futures."""

import os
import sys
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.exchange.coinbase import CoinbaseClient
from bot.types import AccountType

async def transfer_funds():
    """Transfer funds from spot to futures."""
    # Initialize client
    client = CoinbaseClient()
    
    print("Connecting to Coinbase...")
    if not await client.connect():
        print("Failed to connect to Coinbase")
        return
    
    print("✅ Connected successfully")
    
    # Get current balances
    print("\nFetching current balances...")
    spot_balance = await client.get_account_balance(AccountType.CBI)
    futures_balance = await client.get_account_balance(AccountType.CFM)
    
    print(f"\nCurrent Balances:")
    print(f"  Spot (CBI):    ${spot_balance:,.2f}")
    print(f"  Futures (CFM): ${futures_balance:,.2f}")
    print(f"  Total:         ${spot_balance + futures_balance:,.2f}")
    
    # Suggest transfer amount (leave some in spot for fees)
    suggested_amount = min(spot_balance * Decimal("0.9"), Decimal("500"))
    
    if spot_balance < Decimal("100"):
        print(f"\n⚠️  Spot balance too low for transfer (minimum $100)")
        return
    
    # Transfer suggested amount
    transfer_amount = suggested_amount
    print(f"\nTransferring ${transfer_amount:,.2f} to futures account...")
    
    success = await client.transfer_cash_to_futures(transfer_amount, "BOT_STARTUP")
    
    if success:
        print("✅ Transfer successful!")
        
        # Show updated balances
        print("\nFetching updated balances...")
        new_spot = await client.get_account_balance(AccountType.CBI)
        new_futures = await client.get_account_balance(AccountType.CFM)
        
        print(f"\nUpdated Balances:")
        print(f"  Spot (CBI):    ${new_spot:,.2f}")
        print(f"  Futures (CFM): ${new_futures:,.2f}")
        print(f"  Total:         ${new_spot + new_futures:,.2f}")
    else:
        print("❌ Transfer failed!")
    
    await client.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(transfer_funds())