#!/usr/bin/env python3
"""List all available futures products on Coinbase."""

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

print("COINBASE FUTURES PRODUCTS")
print("=" * 80)

try:
    # Get all products
    print("\n1. Fetching all products...")
    products = client.get_products()
    
    # Filter for futures products
    futures_products = []
    spot_products = []
    
    # Handle the products response
    if hasattr(products, 'products'):
        product_list = products.products
    else:
        product_list = products
    
    for product in product_list:
        # Get product details
        if hasattr(product, 'to_dict'):
            p = product.to_dict()
        elif hasattr(product, 'product_id'):
            # Direct attribute access
            p = {
                'product_id': product.product_id,
                'product_type': getattr(product, 'product_type', ''),
                'base_currency': getattr(product, 'base_currency', ''),
                'quote_currency': getattr(product, 'quote_currency', ''),
                'status': getattr(product, 'status', ''),
                'trading_disabled': getattr(product, 'trading_disabled', False)
            }
        else:
            continue
            
        product_id = p.get('product_id', '')
        product_type = p.get('product_type', '')
        
        # Look for futures indicators
        if 'FUTURE' in product_type.upper() or any(month in product_id for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']):
            futures_products.append(p)
        elif product_id in ['ETH-USD', 'BTC-USD']:
            spot_products.append(p)
    
    # Display futures products
    if futures_products:
        print(f"\n2. Found {len(futures_products)} futures products:")
        for p in futures_products[:10]:  # Show first 10
            print(f"\n   Product ID: {p.get('product_id')}")
            print(f"   Type: {p.get('product_type')}")
            print(f"   Base: {p.get('base_currency')}")
            print(f"   Quote: {p.get('quote_currency')}")
            print(f"   Status: {p.get('status')}")
            print(f"   Trading Disabled: {p.get('trading_disabled', False)}")
    else:
        print("\n2. No futures products found with FUTURE type or month names")
    
    # Show spot products for comparison
    print(f"\n3. Spot products (for comparison):")
    for p in spot_products:
        print(f"\n   Product ID: {p.get('product_id')}")
        print(f"   Type: {p.get('product_type')}")
        print(f"   Status: {p.get('status')}")
    
    # Try to get futures-specific endpoints
    print("\n4. Checking futures positions endpoint...")
    try:
        positions = client.list_futures_positions()
        print(f"   ✓ Futures positions accessible: {len(positions.positions)} positions")
    except Exception as e:
        print(f"   ✗ Error accessing futures positions: {e}")
    
    # Check futures balance
    print("\n5. Checking futures balance...")
    try:
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        print(f"   CFM Balance: ${bs.cfm_usd_balance['value']}")
        print(f"   CBI Balance: ${bs.cbi_usd_balance['value']}")
        print(f"   Futures Buying Power: ${bs.futures_buying_power['value']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()