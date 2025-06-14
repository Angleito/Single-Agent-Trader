#!/usr/bin/env python3
"""Check USD accounts specifically."""

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)

try:
    # Get accounts
    accounts_resp = client.get_accounts()
    
    print("Looking for USD accounts...")
    for acc in accounts_resp.accounts:
        print(f"\nCurrency: {acc.currency}")
        print(f"Available balance: {acc.available_balance}")
        print(f"Type of available_balance: {type(acc.available_balance)}")
        
        if acc.currency == "USD":
            print("Found USD account!")
            # Check structure
            if hasattr(acc.available_balance, 'value'):
                print(f"USD Balance (attr): ${acc.available_balance.value}")
            elif isinstance(acc.available_balance, dict):
                print(f"USD Balance (dict): ${acc.available_balance.get('value', 'N/A')}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()