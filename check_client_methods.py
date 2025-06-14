#!/usr/bin/env python3
"""Check available methods on client."""

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

print("Client methods:")
methods = [x for x in dir(client) if not x.startswith('_') and callable(getattr(client, x))]
for m in sorted(methods):
    print(f"  - {m}")