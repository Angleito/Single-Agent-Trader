#!/usr/bin/env python3
"""
Check all account balances on Coinbase.
"""

import os
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Coinbase SDK
try:
    from coinbase.rest import RESTClient
except ImportError:
    print("Error: coinbase-advanced-py not installed")
    exit(1)


def main():
    """Check all balances."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("COINBASE ACCOUNT BALANCES")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Get all accounts
        print("\nFetching all accounts...")
        accounts = client.get_accounts()
        
        print(f"\nFound {len(accounts.accounts)} accounts:\n")
        
        usd_accounts = []
        other_accounts = []
        
        for account in accounts.accounts:
            # Handle both object and dict formats
            if isinstance(account, dict):
                currency = account.get('currency')
                account_type = account.get('type')
                balance = Decimal(account.get('available_balance', {}).get('value', '0'))
                hold = Decimal(account.get('hold', {}).get('value', '0'))
            else:
                currency = account.currency
                account_type = account.type
                balance = Decimal(account.available_balance.value)
                hold = Decimal(account.hold.value if hasattr(account, 'hold') else '0')
            
            account_info = {
                'currency': currency,
                'type': account_type,
                'balance': balance,
                'hold': hold,
                'name': getattr(account, 'name', 'N/A'),
                'uuid': getattr(account, 'uuid', 'N/A')
            }
            
            if currency == 'USD':
                usd_accounts.append(account_info)
            else:
                other_accounts.append(account_info)
        
        # Display USD accounts
        print("USD ACCOUNTS:")
        print("-" * 60)
        total_usd = Decimal('0')
        for acc in usd_accounts:
            print(f"Type: {acc['type']}")
            print(f"  Balance: ${acc['balance']:,.2f}")
            print(f"  On Hold: ${acc['hold']:,.2f}")
            print(f"  UUID: {acc['uuid']}")
            print()
            total_usd += acc['balance']
        
        print(f"TOTAL USD: ${total_usd:,.2f}\n")
        
        # Display other currencies if any
        if other_accounts:
            print("\nOTHER CURRENCIES:")
            print("-" * 60)
            for acc in other_accounts:
                if acc['balance'] > 0:
                    print(f"{acc['currency']}: {acc['balance']} (Type: {acc['type']})")
        
        # Try to get futures balance specifically
        print("\n\nFUTURES ACCOUNT CHECK:")
        print("-" * 60)
        try:
            fcm_response = client.get_futures_balance_summary()
            balance_summary = fcm_response.balance_summary
            
            print(f"CBI USD Balance: ${balance_summary.cbi_usd_balance.value}")
            print(f"CFM USD Balance: ${balance_summary.cfm_usd_balance.value}")
            print(f"Total USD Balance: ${balance_summary.total_usd_balance.value}")
            print(f"Futures Buying Power: ${balance_summary.futures_buying_power.value}")
            
            # This is likely where your funds are
            cash_to_transfer = Decimal(balance_summary.cbi_usd_balance.value)
            print(f"\n💰 AVAILABLE TO TRANSFER TO FUTURES: ${cash_to_transfer:,.2f}")
            
        except Exception as e:
            print(f"Could not get futures balance summary: {e}")
        
        # Check portfolios
        print("\n\nPORTFOLIOS:")
        print("-" * 60)
        try:
            portfolios = client.get_portfolios()
            for portfolio in portfolios.portfolios:
                print(f"Name: {portfolio.name}")
                print(f"  UUID: {portfolio.uuid}")
                print(f"  Type: {portfolio.type}")
                print(f"  Deleted: {portfolio.deleted}")
                print()
        except Exception as e:
            print(f"Could not get portfolios: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()