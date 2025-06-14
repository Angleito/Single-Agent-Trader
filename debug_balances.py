#!/usr/bin/env python3
"""
Debug account structure and find USD balances.
"""

import os
import json
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Coinbase SDK
from coinbase.rest import RESTClient


def main():
    """Debug balances."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("DEBUGGING COINBASE ACCOUNTS")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Get all accounts
        accounts_response = client.get_accounts()
        
        # Check what type of response we got
        print(f"Response type: {type(accounts_response)}")
        print(f"Has 'accounts' attr: {hasattr(accounts_response, 'accounts')}")
        
        # Try to access accounts
        if hasattr(accounts_response, 'accounts'):
            accounts_list = accounts_response.accounts
        else:
            accounts_list = accounts_response.get('accounts', [])
        
        print(f"\nFound {len(accounts_list)} accounts")
        
        # Look at first account structure
        if accounts_list:
            first_account = accounts_list[0]
            print(f"\nFirst account type: {type(first_account)}")
            if hasattr(first_account, '__dict__'):
                print(f"First account attributes: {list(first_account.__dict__.keys())}")
            
        # Find USD accounts
        print("\n\nUSD ACCOUNTS:")
        print("-" * 60)
        
        total_usd = Decimal('0')
        for i, account in enumerate(accounts_list):
            try:
                # Try different ways to access currency
                currency = None
                if hasattr(account, 'currency'):
                    currency = account.currency
                elif isinstance(account, dict):
                    currency = account.get('currency')
                
                if currency == 'USD':
                    # Try to get balance
                    balance_value = '0'
                    if hasattr(account, 'available_balance'):
                        if hasattr(account.available_balance, 'value'):
                            balance_value = account.available_balance.value
                        elif isinstance(account.available_balance, dict):
                            balance_value = account.available_balance.get('value', '0')
                    elif isinstance(account, dict) and 'available_balance' in account:
                        balance_value = account['available_balance'].get('value', '0')
                    
                    balance = Decimal(balance_value)
                    
                    # Get account type
                    acc_type = 'Unknown'
                    if hasattr(account, 'type'):
                        acc_type = account.type
                    elif isinstance(account, dict):
                        acc_type = account.get('type', 'Unknown')
                    
                    print(f"Account {i+1}: Type={acc_type}, Balance=${balance:,.2f}")
                    total_usd += balance
                    
            except Exception as e:
                print(f"Error processing account {i+1}: {e}")
        
        print(f"\nTOTAL USD: ${total_usd:,.2f}")
        
        # Try futures balance summary
        print("\n\nFUTURES BALANCE SUMMARY:")
        print("-" * 60)
        try:
            fcm = client.get_futures_balance_summary()
            
            # Try to access balance data
            if hasattr(fcm, 'balance_summary'):
                bs = fcm.balance_summary
                print(f"Balance summary type: {type(bs)}")
                
                # Try different ways to get CBI balance
                cbi_balance = '0'
                if hasattr(bs, 'cbi_usd_balance'):
                    if hasattr(bs.cbi_usd_balance, 'value'):
                        cbi_balance = bs.cbi_usd_balance.value
                    elif isinstance(bs.cbi_usd_balance, dict):
                        cbi_balance = bs.cbi_usd_balance.get('value', '0')
                
                print(f"CBI (Spot) Balance: ${Decimal(cbi_balance):,.2f}")
                print(f"\n💰 THIS IS THE AMOUNT AVAILABLE TO TRANSFER: ${Decimal(cbi_balance):,.2f}")
                
        except Exception as e:
            print(f"Error getting futures balance: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()