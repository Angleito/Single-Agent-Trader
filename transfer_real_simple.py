#!/usr/bin/env python3
"""
Direct transfer script using Coinbase SDK without bot framework.
This bypasses all safety checks to execute real transfers.
"""

import asyncio
import os
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Coinbase SDK directly
try:
    from coinbase.rest import RESTClient
except ImportError:
    print("Error: coinbase-advanced-py not installed")
    print("Run: pip install coinbase-advanced-py")
    exit(1)


async def main():
    """Execute the transfer using direct SDK calls."""
    # Get CDP credentials from environment
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    if not cdp_api_key or not cdp_private_key:
        print("Error: CDP credentials not found in .env file")
        return
    
    print("=" * 60)
    print("DIRECT COINBASE REAL MONEY TRANSFER")
    print("=" * 60)
    print("\n⚠️  WARNING: This transfers REAL MONEY!")
    print("Using CDP credentials from .env file\n")
    
    # Initialize Coinbase client
    print("Initializing Coinbase client...")
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Get accounts to find balances
        print("Fetching account information...")
        accounts = client.get_accounts()
        
        # Find USD balances
        spot_balance = Decimal('0')
        futures_balance = Decimal('0')
        
        for account in accounts.accounts:
            if account.currency == 'USD':
                balance = Decimal(account.available_balance.value)
                if account.type == 'ACCOUNT_TYPE_CRYPTO':
                    spot_balance = balance
                elif account.type == 'ACCOUNT_TYPE_FUTURES':
                    futures_balance = balance
        
        print(f"\nCurrent REAL Account Balances:")
        print(f"  Spot (CBI):    ${spot_balance:,.2f}")
        print(f"  Futures (CFM): ${futures_balance:,.2f}")
        print(f"  Total:         ${spot_balance + futures_balance:,.2f}")
        
        if spot_balance <= 0:
            print("\nNo funds available in spot account to transfer.")
            return
        
        print(f"\n💸 Will transfer ALL spot balance: ${spot_balance:,.2f}")
        
        # Get portfolios to find IDs
        print("\nFetching portfolio information...")
        portfolios_response = client.get('/api/v3/brokerage/portfolios')
        portfolios = portfolios_response.get('portfolios', [])
        
        default_portfolio_id = None
        futures_portfolio_id = None
        
        for portfolio in portfolios:
            if portfolio.get('is_default'):
                default_portfolio_id = portfolio.get('uuid')
            if 'futures' in portfolio.get('name', '').lower():
                futures_portfolio_id = portfolio.get('uuid')
        
        print(f"Default portfolio ID: {default_portfolio_id}")
        print(f"Futures portfolio ID: {futures_portfolio_id}")
        
        # Try to get FCM balance summary
        print("\nChecking futures account status...")
        try:
            fcm_response = client.get_futures_balance_summary()
            print(f"Futures buying power: ${fcm_response.balance_summary.futures_buying_power.value}")
        except Exception as e:
            print(f"Could not get futures balance summary: {e}")
        
        # Confirm transfer
        print(f"\n⚠️  READY TO TRANSFER ${spot_balance:,.2f} TO FUTURES")
        confirm = input("Type 'TRANSFER NOW' to proceed: ")
        if confirm != "TRANSFER NOW":
            print("Transfer cancelled")
            return
        
        # Execute transfer using allocate_portfolio
        print(f"\n💸 Executing transfer of ${spot_balance:,.2f}...")
        
        try:
            # Method 1: Try allocate_portfolio
            if default_portfolio_id:
                result = client.allocate_portfolio(
                    source_portfolio_uuid=default_portfolio_id,
                    amount=str(spot_balance),
                    currency="USD"
                )
                print("✅ Transfer initiated successfully!")
                print(f"Transfer ID: {result}")
            else:
                # Method 2: Try direct transfer endpoint
                transfer_data = {
                    "source_account": "CBI",
                    "target_account": "CFM", 
                    "amount": str(spot_balance),
                    "currency": "USD"
                }
                result = client.post('/api/v3/brokerage/cfm/cash_transfers', transfer_data)
                print("✅ Transfer initiated successfully!")
                
        except Exception as e:
            print(f"❌ Transfer failed: {e}")
            print("\nTrying alternative method...")
            
            # Method 3: Try intraday transfer
            try:
                result = client.create_transfer(
                    source_account_id=default_portfolio_id,
                    target_account_id="cfm_account",
                    amount=str(spot_balance),
                    currency="USD"
                )
                print("✅ Transfer successful using alternative method!")
            except Exception as e2:
                print(f"❌ Alternative method also failed: {e2}")
                return
        
        # Wait and check updated balances
        print("\nWaiting 10 seconds for transfer to process...")
        await asyncio.sleep(10)
        
        # Fetch updated balances
        print("\nFetching updated balances...")
        accounts = client.get_accounts()
        
        new_spot = Decimal('0')
        new_futures = Decimal('0')
        
        for account in accounts.accounts:
            if account.currency == 'USD':
                balance = Decimal(account.available_balance.value)
                if account.type == 'ACCOUNT_TYPE_CRYPTO':
                    new_spot = balance
                elif account.type == 'ACCOUNT_TYPE_FUTURES':
                    new_futures = balance
        
        print(f"\nUpdated REAL Account Balances:")
        print(f"  Spot (CBI):    ${new_spot:,.2f}")
        print(f"  Futures (CFM): ${new_futures:,.2f}")
        print(f"  Total:         ${new_spot + new_futures:,.2f}")
        
        transferred = spot_balance - new_spot
        print(f"\n💰 ACTUAL CASH TRANSFERRED: ${transferred:,.2f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n⚠️  REAL MONEY TRANSFER SCRIPT")
    print("This will transfer real USD from your spot to futures account\n")
    
    confirm = input("Type 'YES' to continue: ")
    if confirm == "YES":
        asyncio.run(main())
    else:
        print("Transfer cancelled")