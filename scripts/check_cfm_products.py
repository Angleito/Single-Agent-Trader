#!/usr/bin/env python3
"""Check for CFM-specific products or trading sessions."""

import os

from coinbase.rest import RESTClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv("EXCHANGE__CDP_API_KEY_NAME")
cdp_private_key = os.getenv("EXCHANGE__CDP_PRIVATE_KEY")

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)

print("CHECKING CFM/FUTURES PRODUCT DETAILS")
print("=" * 60)

# Check ETH-USD product details
print("\n1. ETH-USD Product Details:")
try:
    product = client.get_product("ETH-USD")
    if hasattr(product, "fcm_trading_session_details"):
        print(f"   FCM Trading Session Details: {product.fcm_trading_session_details}")
    if hasattr(product, "product_venue"):
        print(f"   Product Venue: {product.product_venue}")
    if hasattr(product, "is_futures"):
        print(f"   Is Futures: {product.is_futures}")

    # Check all attributes
    print("\n   All attributes:")
    for attr in dir(product):
        if not attr.startswith("_"):
            value = getattr(product, attr)
            if value is not None and not callable(value):
                print(f"   - {attr}: {value}")
except Exception as e:
    print(f"   Error: {e}")

# Check if placing order with leverage creates a futures position
print("\n2. Recent Futures Positions:")
try:
    positions = client.list_futures_positions()
    if hasattr(positions, "positions"):
        print(f"   Found {len(positions.positions)} futures positions")
        for pos in positions.positions:
            print("\n   Position:")
            print(f"   - Product: {pos.product_id}")
            print(f"   - Side: {pos.side}")
            print(f"   - Contracts: {pos.number_of_contracts}")
            print(f"   - Entry Price: {pos.avg_entry_price}")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Conclusion:")
print("   Coinbase CFM appears to use standard spot symbols (ETH-USD)")
print("   with the 'leverage' parameter to route orders to futures.")
print("   The dated contracts may only be visible in the web UI.")
