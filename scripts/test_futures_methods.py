#!/usr/bin/env python3
"""Test various futures-related methods to understand the API."""

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
import inspect

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)

print("TESTING FUTURES API METHODS")
print("=" * 80)

# Find all methods that might be futures-related
print("\n1. Available futures-related methods:")
futures_methods = []
for method_name in dir(client):
    if any(keyword in method_name.lower() for keyword in ['futures', 'cfm', 'fcm', 'derivative', 'perpetual']):
        if not method_name.startswith('_'):
            futures_methods.append(method_name)
            print(f"   - {method_name}")

# Test each method
print("\n2. Testing futures methods:")

for method_name in futures_methods:
    print(f"\n   Testing {method_name}...")
    try:
        method = getattr(client, method_name)
        sig = inspect.signature(method)
        print(f"   Signature: {sig}")
        
        # Try to call methods with no required params
        if len([p for p in sig.parameters.values() if p.default == inspect.Parameter.empty]) == 0:
            result = method()
            print(f"   ✓ Success: {type(result)}")
            
            # Show some details if it's a list
            if hasattr(result, '__iter__') and not isinstance(result, str):
                print(f"   Items: {len(list(result))}")
        else:
            print(f"   - Requires parameters, skipping")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")

# Try to place a test futures order with different approaches
print("\n3. Testing futures order placement approaches:")

# Approach 1: ETH-USD with leverage
print("\n   Approach 1: ETH-USD with leverage parameter")
try:
    import uuid
    # Don't actually place the order, just test the parameters
    print("   Would place order with:")
    print("   - product_id: ETH-USD")
    print("   - leverage: 2")
    print("   - side: BUY")
    print("   - size: 0.001")
except Exception as e:
    print(f"   Error: {e}")

# Check if we can get contract specifications
print("\n4. Looking for contract specifications...")
try:
    # Try different product IDs
    test_symbols = ['ETH-USD', 'BTC-USD', 'ETH-PERP', 'BTC-PERP', 'ETH/USD', 'BTC/USD']
    for symbol in test_symbols:
        try:
            product = client.get_product(symbol)
            if product:
                print(f"\n   Found product: {symbol}")
                if hasattr(product, 'to_dict'):
                    details = product.to_dict()
                    for key, value in details.items():
                        print(f"     {key}: {value}")
                break
        except:
            continue
except Exception as e:
    print(f"   Error: {e}")