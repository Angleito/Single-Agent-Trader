#!/usr/bin/env python3
"""
Script to transfer REAL funds from Coinbase spot account to futures account.
This script explicitly overrides dry_run mode to execute real transfers.
"""

import asyncio
import os
from decimal import Decimal

from dotenv import load_dotenv

# Override dry_run before importing anything else
os.environ['SYSTEM__DRY_RUN'] = 'false'

from bot.config import settings
from bot.exchange.coinbase import CoinbaseClient
from bot.types import AccountType


async def main():
    """Main function to execute the transfer."""
    # Load environment variables
    load_dotenv()
    
    # Force live trading mode
    settings.system.dry_run = False
    
    print("⚠️  WARNING: This will transfer REAL MONEY!")
    print("Running in LIVE mode - real funds will be moved\n")
    
    # Initialize client
    print("Initializing Coinbase client for LIVE trading...")
    client = CoinbaseClient()
    
    try:
        # Connect to Coinbase
        print("Connecting to Coinbase (LIVE mode)...")
        connected = await client.connect()
        if not connected:
            print("Failed to connect to Coinbase")
            return
        
        print("Successfully connected to Coinbase LIVE account\n")
        
        # Get current balances
        print("Fetching REAL account balances...")
        spot_balance = await client.get_account_balance(AccountType.CBI)
        futures_balance = await client.get_account_balance(AccountType.CFM)
        
        print(f"\nCurrent REAL Balances:")
        print(f"  Spot (CBI):    ${spot_balance:,.2f}")
        print(f"  Futures (CFM): ${futures_balance:,.2f}")
        print(f"  Total:         ${spot_balance + futures_balance:,.2f}")
        
        # Check if there's anything to transfer
        if spot_balance <= 0:
            print("\nNo funds available in spot account to transfer.")
            return
        
        # Transfer all spot balance
        transfer_amount = spot_balance
        
        print(f"\n💸 TRANSFERRING ALL REAL SPOT HOLDINGS TO FUTURES")
        print(f"  Amount to transfer: ${transfer_amount:,.2f} (REAL USD)")
        print(f"\nExpected balances after transfer:")
        print(f"  Spot:    $0.00")
        print(f"  Futures: ${futures_balance + transfer_amount:,.2f}")
        
        # Final confirmation
        print("\n⚠️  THIS IS A REAL MONEY TRANSFER!")
        confirm = input("Type 'TRANSFER' to proceed with real money transfer: ")
        if confirm != "TRANSFER":
            print("Transfer cancelled")
            return
        
        # Execute transfer
        print(f"\nExecuting REAL transfer of ${transfer_amount:,.2f} to futures...")
        success = await client.transfer_cash_to_futures(
            amount=transfer_amount,
            reason="MANUAL"
        )
        
        if success:
            print("✅ REAL Transfer successful!")
            
            # Show updated balances
            print("\nWaiting for transfer to process...")
            await asyncio.sleep(5)  # Wait longer for real transfer
            new_spot = await client.get_account_balance(AccountType.CBI)
            new_futures = await client.get_account_balance(AccountType.CFM)
            
            print(f"\nUpdated REAL Balances:")
            print(f"  Spot (CBI):    ${new_spot:,.2f}")
            print(f"  Futures (CFM): ${new_futures:,.2f}")
            print(f"  Total:         ${new_spot + new_futures:,.2f}")
            
            print(f"\n💰 REAL CASH AMOUNT TRANSFERRED: ${transfer_amount:,.2f}")
        else:
            print("❌ Transfer failed")
            print("\nPossible reasons:")
            print("- Insufficient permissions on API key")
            print("- Transfer limits exceeded")
            print("- Minimum transfer amount not met")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Disconnect
        await client.disconnect()
        print("\nDisconnected from Coinbase")


if __name__ == "__main__":
    print("=" * 60)
    print("COINBASE REAL MONEY TRANSFER SCRIPT")
    print("=" * 60)
    print("\n⚠️  WARNING: This script transfers REAL MONEY!")
    print("Make sure you want to move real funds from spot to futures.\n")
    
    confirm = input("Type 'YES' to continue with real money transfer: ")
    if confirm == "YES":
        # Run the async main function
        asyncio.run(main())
    else:
        print("Transfer cancelled")