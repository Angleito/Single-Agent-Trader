#!/usr/bin/env python3
"""Test futures order with correct minimum size."""

import os
import uuid
import asyncio
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
    FUTURES_SYMBOL = "ET-27JUN25-CDE"
    
    print("TESTING FUTURES ORDER WITH CORRECT SIZE")
    print("=" * 60)
    
    try:
        # Check balance
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        buying_power = float(bs.futures_buying_power['value'])
        
        print(f"\nFutures Buying Power: ${buying_power}")
        
        # Check current position
        print(f"\nCurrent Position:")
        positions = client.list_futures_positions()
        for pos in positions.positions:
            print(f"  {pos.product_id}: {pos.side} {pos.number_of_contracts} contracts")
        
        # Place order with minimum size (1 ETH)
        print(f"\nPlacing BUY order for 1 ETH on {FUTURES_SYMBOL}...")
        print("  This will reduce your SHORT position by 1 ETH")
        
        order_result = client.create_order(
            client_order_id=str(uuid.uuid4()),
            product_id=FUTURES_SYMBOL,
            side="BUY",  # Buy to reduce short
            order_configuration={
                "market_market_ioc": {"base_size": "1"}  # Minimum 1 ETH
            }
            # NO leverage parameter for real futures
        )
        
        if order_result.success:
            print("\n✅ FUTURES ORDER PLACED SUCCESSFULLY!")
            resp = order_result.success_response
            if isinstance(resp, dict):
                order_id = resp.get('order_id')
            else:
                order_id = resp.order_id if hasattr(resp, 'order_id') else None
            print(f"  Order ID: {order_id}")
            
            # Check updated position
            await asyncio.sleep(3)
            print("\nUpdated Position:")
            positions = client.list_futures_positions()
            for pos in positions.positions:
                print(f"  {pos.product_id}: {pos.side} {pos.number_of_contracts} contracts")
        else:
            print("\n❌ Order failed")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_futures_order())