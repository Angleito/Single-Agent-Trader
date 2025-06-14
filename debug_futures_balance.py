#!/usr/bin/env python3
"""Debug futures balance structure."""

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
    # Get futures balance
    print("Getting futures balance...")
    fcm = client.get_futures_balance_summary()
    
    print("\nType of fcm:", type(fcm))
    print("Attributes:", [x for x in dir(fcm) if not x.startswith('_')])
    
    if hasattr(fcm, 'balance_summary'):
        bs = fcm.balance_summary
        print("\nType of balance_summary:", type(bs))
        print("Attributes:", [x for x in dir(bs) if not x.startswith('_')])
        
        # Try to access specific fields
        if hasattr(bs, 'cfm_usd_balance'):
            print("\nType of cfm_usd_balance:", type(bs.cfm_usd_balance))
            if hasattr(bs.cfm_usd_balance, 'value'):
                print("CFM USD Balance value:", bs.cfm_usd_balance.value)
            else:
                print("cfm_usd_balance:", bs.cfm_usd_balance)
                
        # Try to_dict
        if hasattr(bs, 'to_dict'):
            print("\nBalance summary dict:", json.dumps(bs.to_dict(), indent=2))
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()