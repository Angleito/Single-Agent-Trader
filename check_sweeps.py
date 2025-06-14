#!/usr/bin/env python3
"""Check pending sweeps."""

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
    # List futures sweeps
    print("Checking futures sweeps...")
    sweeps = client.list_futures_sweeps()
    
    print(f"\nType: {type(sweeps)}")
    if hasattr(sweeps, 'sweeps'):
        print(f"Number of sweeps: {len(sweeps.sweeps)}")
        for sweep in sweeps.sweeps:
            print(f"\nSweep:")
            if hasattr(sweep, 'to_dict'):
                import json
                print(json.dumps(sweep.to_dict(), indent=2))
            else:
                print(f"  ID: {sweep.id if hasattr(sweep, 'id') else 'N/A'}")
                print(f"  Status: {sweep.status if hasattr(sweep, 'status') else 'N/A'}")
                print(f"  Amount: {sweep.amount if hasattr(sweep, 'amount') else 'N/A'}")
    
    # Try to cancel pending sweeps
    print("\n\nTrying to cancel pending sweeps...")
    result = client.cancel_pending_futures_sweep()
    print(f"Cancel result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()