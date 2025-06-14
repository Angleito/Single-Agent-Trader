#!/usr/bin/env python3
"""
Script to transfer funds from Coinbase spot account to futures account.

Usage:
    python transfer_to_futures.py [amount]
    
If amount is not provided, the script will prompt for it.
"""

import asyncio
import os
import sys
from decimal import Decimal
from typing import Optional

from dotenv import load_dotenv

from bot.exchange.coinbase import CoinbaseClient
from bot.types import AccountType


async def get_balances(client: CoinbaseClient) -> tuple[Decimal, Decimal]:
    """Get spot and futures balances."""
    spot_balance = await client.get_account_balance(AccountType.CBI)
    futures_balance = await client.get_account_balance(AccountType.CFM)
    return spot_balance, futures_balance


async def main(amount: Optional[Decimal] = None):
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
        spot_balance, futures_balance = await get_balances(client)
        
        print(f"\nCurrent Balances:")
        print(f"  Spot (CBI):    ${spot_balance:,.2f}")
        print(f"  Futures (CFM): ${futures_balance:,.2f}")
        print(f"  Total:         ${spot_balance + futures_balance:,.2f}")
        
        # Get transfer amount if not provided
        if amount is None:
            print(f"\nAvailable for transfer: ${spot_balance:,.2f}")
            amount_str = input("Enter amount to transfer to futures (USD): $")
            try:
                amount = Decimal(amount_str.strip())
            except Exception:
                print("Invalid amount entered")
                return
        
        # Validate amount
        if amount <= 0:
            print("Amount must be greater than 0")
            return
        
        if amount > spot_balance:
            print(f"Insufficient funds. Available: ${spot_balance:,.2f}")
            return
        
        # Confirm transfer
        print(f"\nTransfer Summary:")
        print(f"  From: Spot (CBI)")
        print(f"  To:   Futures (CFM)")
        print(f"  Amount: ${amount:,.2f}")
        print(f"\nNew balances after transfer:")
        print(f"  Spot:    ${spot_balance - amount:,.2f}")
        print(f"  Futures: ${futures_balance + amount:,.2f}")
        
        confirm = input("\nProceed with transfer? (y/n): ")
        if confirm.lower() != 'y':
            print("Transfer cancelled")
            return
        
        # Execute transfer
        print(f"\nExecuting transfer of ${amount:,.2f} to futures...")
        success = await client.transfer_cash_to_futures(
            amount=amount,
            reason="MANUAL_TRANSFER"
        )
        
        if success:
            print("✅ Transfer successful!")
            
            # Show updated balances
            print("\nFetching updated balances...")
            await asyncio.sleep(2)  # Wait for transfer to process
            new_spot, new_futures = await get_balances(client)
            
            print(f"\nUpdated Balances:")
            print(f"  Spot (CBI):    ${new_spot:,.2f}")
            print(f"  Futures (CFM): ${new_futures:,.2f}")
            print(f"  Total:         ${new_spot + new_futures:,.2f}")
        else:
            print("❌ Transfer failed")
            print("\nNote: Transfers might fail if:")
            print("- Auto cash transfer is disabled in config")
            print("- API permissions don't allow transfers")
            print("- There's a minimum transfer amount")
    
    except KeyboardInterrupt:
        print("\n\nTransfer cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Disconnect
        await client.disconnect()
        print("\nDisconnected from Coinbase")


if __name__ == "__main__":
    # Get amount from command line if provided
    amount = None
    if len(sys.argv) > 1:
        try:
            amount = Decimal(sys.argv[1])
        except Exception:
            print(f"Invalid amount: {sys.argv[1]}")
            sys.exit(1)
    
    # Run the async main function
    asyncio.run(main(amount))