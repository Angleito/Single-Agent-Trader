#!/usr/bin/env python3
"""Test placing order with actual futures contract symbol."""

import os
import uuid
import asyncio
from decimal import Decimal
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)

async def test_futures_order():
    print("TESTING REAL FUTURES CONTRACT SYMBOL")
    print("=" * 60)
    
    # The actual futures contract symbol from your position
    FUTURES_SYMBOL = "ET-27JUN25-CDE"
    
    try:
        # 1. Check if we can get the product
        print(f"\n1. Checking if {FUTURES_SYMBOL} is accessible...")
        try:
            product = client.get_product(FUTURES_SYMBOL)
            print(f"   ✅ Product found!")
            if hasattr(product, 'price'):
                print(f"   Current price: ${product.price}")
        except Exception as e:
            print(f"   ❌ Cannot access product: {e}")
            
        # 2. Check current position
        print(f"\n2. Current futures positions:")
        positions = client.list_futures_positions()
        for pos in positions.positions:
            print(f"   - {pos.product_id}: {pos.side} {pos.number_of_contracts} contracts @ ${pos.avg_entry_price}")
        
        # 3. Try to place an order with the real futures symbol
        print(f"\n3. Attempting to place order with {FUTURES_SYMBOL}...")
        try:
            # Very small order to close part of the short position
            order_result = client.create_order(
                client_order_id=str(uuid.uuid4()),
                product_id=FUTURES_SYMBOL,  # Use actual futures symbol
                side="BUY",  # Buy to reduce short position
                order_configuration={
                    "market_market_ioc": {"base_size": "0.01"}  # Small size
                }
                # Note: NO leverage parameter for actual futures contracts
            )
            
            if order_result.success:
                print("   ✅ ORDER PLACED SUCCESSFULLY!")
                resp = order_result.success_response
                order_id = resp.order_id if hasattr(resp, 'order_id') else resp.get('order_id', 'unknown')
                print(f"   Order ID: {order_id}")
            else:
                print("   ❌ Order failed")
                
        except Exception as e:
            print(f"   ❌ Order error: {e}")
            
        # 4. Also test with leverage parameter
        print(f"\n4. Testing {FUTURES_SYMBOL} WITH leverage parameter...")
        try:
            order_result = client.create_order(
                client_order_id=str(uuid.uuid4()),
                product_id=FUTURES_SYMBOL,
                side="BUY",
                order_configuration={
                    "market_market_ioc": {"base_size": "0.01"}
                },
                leverage="5"  # Add leverage parameter
            )
            
            if order_result.success:
                print("   ✅ With leverage worked!")
            else:
                print("   ❌ With leverage failed")
                
        except Exception as e:
            print(f"   ❌ Error with leverage: {e}")
            
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(test_futures_order())