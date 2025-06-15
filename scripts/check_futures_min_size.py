#!/usr/bin/env python3
"""Check minimum order size for futures contract."""

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

FUTURES_SYMBOL = "ET-27JUN25-CDE"

print(f"CHECKING {FUTURES_SYMBOL} SPECIFICATIONS")
print("=" * 60)

try:
    product = client.get_product(FUTURES_SYMBOL)
    
    print(f"\nProduct Details:")
    print(f"  Product ID: {product.product_id}")
    print(f"  Status: {product.status}")
    print(f"  Product Type: {product.product_type}")
    print(f"  Current Price: ${product.price}")
    
    print(f"\nSize Limits:")
    print(f"  Base Min Size: {product.base_min_size}")
    print(f"  Base Max Size: {product.base_max_size}")
    print(f"  Base Increment: {product.base_increment}")
    
    print(f"\nQuote Limits:")
    print(f"  Quote Min Size: {product.quote_min_size}")
    print(f"  Quote Max Size: {product.quote_max_size}")
    print(f"  Quote Increment: {product.quote_increment}")
    
    # Nano contract = 0.1 ETH
    print(f"\nFor Futures Trading:")
    print(f"  Minimum contracts: {float(product.base_min_size) / 0.1:.1f} contracts")
    print(f"  Minimum ETH: {product.base_min_size} ETH")
    print(f"  At current price: ${float(product.base_min_size) * float(product.price):.2f}")
    
except Exception as e:
    print(f"Error: {e}")