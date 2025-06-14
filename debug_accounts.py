#!/usr/bin/env python3
"""Debug account structure."""

import os
import json
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
    
    print("Type of accounts_resp:", type(accounts_resp))
    print("\nAttributes:", dir(accounts_resp))
    
    # Try to access accounts
    if hasattr(accounts_resp, 'accounts'):
        print("\nFound .accounts attribute")
        accounts = accounts_resp.accounts
        print("Type of accounts:", type(accounts))
        if accounts:
            print("\nFirst account:")
            first = accounts[0]
            print("Type:", type(first))
            print("Dir:", [x for x in dir(first) if not x.startswith('_')])
            
            # Try to print account details
            if hasattr(first, 'to_dict'):
                print("\nAccount dict:", json.dumps(first.to_dict(), indent=2))
            elif hasattr(first, '__dict__'):
                print("\nAccount __dict__:", first.__dict__)
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()