#!/usr/bin/env python3
"""
Execute transfer immediately without confirmations.
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
    exit(1)


async def main():
    """Execute the transfer."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("EXECUTING REAL MONEY TRANSFER")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Get current balances
        accounts = client.get_accounts()
        spot_balance = Decimal('0')
        
        for account in accounts.accounts:
            if account.currency == 'USD' and account.type == 'ACCOUNT_TYPE_CRYPTO':
                spot_balance = Decimal(account.available_balance.value)
        
        print(f"\nSpot Balance to Transfer: ${spot_balance:,.2f}")
        
        if spot_balance <= 0:
            print("No funds available to transfer.")
            return
        
        # Get default portfolio ID
        portfolios_response = client.get('/api/v3/brokerage/portfolios')
        portfolios = portfolios_response.get('portfolios', [])
        
        default_portfolio_id = None
        for portfolio in portfolios:
            if portfolio.get('is_default'):
                default_portfolio_id = portfolio.get('uuid')
                break
        
        if not default_portfolio_id:
            # Use hardcoded default
            default_portfolio_id = "1f3ed8bf-a65c-5022-8258-87ce50c517f6"
        
        print(f"Using portfolio ID: {default_portfolio_id}")
        
        # Execute transfer
        print(f"\nTransferring ${spot_balance:,.2f} to futures...")
        
        result = client.allocate_portfolio(
            source_portfolio_uuid=default_portfolio_id,
            amount=str(spot_balance),
            currency="USD"
        )
        
        print("✅ Transfer executed!")
        
        # Wait and check results
        await asyncio.sleep(5)
        
        # Get updated balances
        accounts = client.get_accounts()
        new_spot = Decimal('0')
        futures_balance = Decimal('0')
        
        for account in accounts.accounts:
            if account.currency == 'USD':
                if account.type == 'ACCOUNT_TYPE_CRYPTO':
                    new_spot = Decimal(account.available_balance.value)
                elif account.type == 'ACCOUNT_TYPE_FUTURES':
                    futures_balance = Decimal(account.available_balance.value)
        
        transferred = spot_balance - new_spot
        
        print(f"\n💰 CASH AMOUNT TRANSFERRED: ${transferred:,.2f}")
        print(f"\nFinal Balances:")
        print(f"  Spot:    ${new_spot:,.2f}")
        print(f"  Futures: ${futures_balance:,.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())