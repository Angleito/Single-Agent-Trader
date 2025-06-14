#!/usr/bin/env python3
"""
Script to transfer ALL funds from Coinbase spot account to futures account.
"""

import asyncio
from decimal import Decimal

from dotenv import load_dotenv

from bot.exchange.coinbase import CoinbaseClient
from bot.types import AccountType


async def main():
    """Main function to execute the transfer."""
    # Load environment variables
    load_dotenv()
    
    # Initialize client
    print("Initializing Coinbase client...")
    client = CoinbaseClient()
    
    try:
        # Connect to Coinbase
        print("Connecting to Coinbase...")
        connected = await client.connect()
        if not connected:
            print("Failed to connect to Coinbase")
            return
        
        print("Successfully connected to Coinbase\n")
        
        # Get current balances
        print("Fetching current balances...")
        spot_balance = await client.get_account_balance(AccountType.CBI)
        futures_balance = await client.get_account_balance(AccountType.CFM)
        
        print(f"\nCurrent Balances:")
        print(f"  Spot (CBI):    ${spot_balance:,.2f}")
        print(f"  Futures (CFM): ${futures_balance:,.2f}")
        print(f"  Total:         ${spot_balance + futures_balance:,.2f}")
        
        # Check if there's anything to transfer
        if spot_balance <= 0:
            print("\nNo funds available in spot account to transfer.")
            return
        
        # Transfer all spot balance
        transfer_amount = spot_balance
        
        print(f"\n📊 TRANSFERRING ALL SPOT HOLDINGS TO FUTURES")
        print(f"  Amount to transfer: ${transfer_amount:,.2f}")
        print(f"\nExpected balances after transfer:")
        print(f"  Spot:    $0.00")
        print(f"  Futures: ${futures_balance + transfer_amount:,.2f}")
        
        # Execute transfer
        print(f"\nExecuting transfer of ${transfer_amount:,.2f} to futures...")
        success = await client.transfer_cash_to_futures(
            amount=transfer_amount,
            reason="MANUAL"
        )
        
        if success:
            print("✅ Transfer successful!")
            
            # Show updated balances
            print("\nFetching updated balances...")
            await asyncio.sleep(3)  # Wait for transfer to process
            new_spot = await client.get_account_balance(AccountType.CBI)
            new_futures = await client.get_account_balance(AccountType.CFM)
            
            print(f"\nUpdated Balances:")
            print(f"  Spot (CBI):    ${new_spot:,.2f}")
            print(f"  Futures (CFM): ${new_futures:,.2f}")
            print(f"  Total:         ${new_spot + new_futures:,.2f}")
            
            print(f"\n💰 CASH AMOUNT TRANSFERRED: ${transfer_amount:,.2f}")
        else:
            print("❌ Transfer failed")
            print("\nNote: Transfers might fail if:")
            print("- Auto cash transfer is disabled in config")
            print("- API permissions don't allow transfers")
            print("- There's a minimum transfer amount")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Disconnect
        await client.disconnect()
        print("\nDisconnected from Coinbase")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())