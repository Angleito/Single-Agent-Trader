#!/usr/bin/env python3
"""Find the correct futures contract symbols."""

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from datetime import datetime

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)

print("SEARCHING FOR FUTURES CONTRACTS")
print("=" * 60)

# Test various futures symbol formats
test_symbols = [
    "ETH-27JUN25-CDE",
    "BTC-27JUN25-CDE",
    "ETH-28JUN24-CDE",
    "ETH-26JUL24-CDE",
    "ETH-30AUG24-CDE",
    "ETH-27SEP24-CDE",
    "ETH-25OCT24-CDE",
    "ETH-29NOV24-CDE",
    "ETH-27DEC24-CDE",
    "ETH-31JAN25-CDE",
    "ETH-28FEB25-CDE",
    "ETH-28MAR25-CDE",
    "ETH-25APR25-CDE",
    "ETH-30MAY25-CDE",
    "ETH-27JUN25-CDE",
]

print("\n1. Testing futures contract symbols...")
valid_contracts = []

for symbol in test_symbols:
    try:
        product = client.get_product(symbol)
        if product:
            print(f"\n✅ Found valid contract: {symbol}")
            if hasattr(product, 'to_dict'):
                details = product.to_dict()
            else:
                details = {
                    'product_id': product.product_id,
                    'status': getattr(product, 'status', 'unknown'),
                    'product_type': getattr(product, 'product_type', 'unknown'),
                    'base_currency': getattr(product, 'base_currency', 'unknown'),
                    'quote_currency': getattr(product, 'quote_currency', 'unknown'),
                }
            
            print(f"   Status: {details.get('status')}")
            print(f"   Type: {details.get('product_type')}")
            print(f"   Base: {details.get('base_currency')}")
            print(f"   Quote: {details.get('quote_currency')}")
            
            valid_contracts.append(symbol)
    except Exception as e:
        if "404" not in str(e):
            print(f"   Error checking {symbol}: {e}")

if not valid_contracts:
    print("\n❌ No valid futures contracts found with -CDE suffix")
    print("\n2. Checking if we need different format...")
    
    # Try without -CDE suffix
    alt_symbols = ["ETH-27JUN25", "BTC-27JUN25", "ETH 27 JUN 25"]
    for symbol in alt_symbols:
        try:
            product = client.get_product(symbol)
            if product:
                print(f"\n✅ Found with alternate format: {symbol}")
                valid_contracts.append(symbol)
        except:
            pass

# Try to place a test order with the correct symbol
if valid_contracts:
    print(f"\n3. Valid contracts found: {valid_contracts}")
    print("\nTo use these contracts, update the bot configuration.")
else:
    print("\n3. No futures contracts found. ETH-USD with leverage may be the only option.")