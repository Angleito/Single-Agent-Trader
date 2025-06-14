#!/usr/bin/env python3
"""Test order response structure."""

import os
import uuid
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
    # Place a tiny spot order
    print("Placing tiny spot order...")
    result = client.create_order(
        client_order_id=str(uuid.uuid4()),
        product_id="ETH-USD",
        side="BUY",
        order_configuration={
            "market_market_ioc": {"base_size": "0.001"}
        }
    )
    
    print(f"\nType of result: {type(result)}")
    print(f"Dir: {[x for x in dir(result) if not x.startswith('_')]}")
    
    # Try to access common fields
    if hasattr(result, 'success'):
        print(f"\nSuccess: {result.success}")
    if hasattr(result, 'order'):
        print(f"Has order attribute")
        order = result.order
        print(f"Order type: {type(order)}")
        print(f"Order dir: {[x for x in dir(order) if not x.startswith('_')]}")
        if hasattr(order, 'order_id'):
            print(f"Order ID: {order.order_id}")
    
    # Try to_dict
    if hasattr(result, 'to_dict'):
        import json
        print(f"\nResult dict: {json.dumps(result.to_dict(), indent=2)}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()