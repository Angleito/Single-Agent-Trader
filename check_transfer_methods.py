#!/usr/bin/env python3
"""Check transfer-related methods."""

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

print("Transfer-related methods:")
methods = [x for x in dir(client) if 'transfer' in x.lower() or 'allocate' in x.lower() or 'move' in x.lower()]
for m in sorted(methods):
    print(f"  - {m}")
    
# Check allocate_portfolio signature
import inspect
print("\nallocate_portfolio signature:")
sig = inspect.signature(client.allocate_portfolio)
print(f"  {sig}")

# Look for futures-specific transfer methods
print("\nFutures/CFM related methods:")
methods = [x for x in dir(client) if 'futures' in x.lower() or 'cfm' in x.lower() or 'fcm' in x.lower()]
for m in sorted(methods):
    print(f"  - {m}")