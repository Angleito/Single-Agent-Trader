#!/usr/bin/env python3
"""Test futures order using buying power (not requiring CFM balance)."""

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
    print("FUTURES ORDER TEST (Using Buying Power)")
    print("=" * 60)
    
    try:
        # 1. Check futures buying power
        print("\n1. Checking futures buying power...")
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        cfm_balance = Decimal(bs.cfm_usd_balance['value'])
        cbi_balance = Decimal(bs.cbi_usd_balance['value'])
        buying_power = Decimal(bs.futures_buying_power['value'])
        
        print(f"   CFM Balance: ${cfm_balance}")
        print(f"   CBI Balance: ${cbi_balance}")
        print(f"   Futures Buying Power: ${buying_power}")
        
        if buying_power < 50:
            print("\n❌ Insufficient buying power. Need at least $50.")
            return
        
        # 2. Place a VERY small futures order
        print("\n2. Placing minimal futures order...")
        
        # Try the absolute minimum order
        order_result = client.create_order(
            client_order_id=str(uuid.uuid4()),
            product_id="ETH-USD",
            side="BUY",
            order_configuration={
                "market_market_ioc": {"base_size": "0.001"}  # $2.50 worth at ~$2500/ETH
            },
            leverage="5"  # 5x leverage
        )
        
        if order_result.success:
            print("\n✅ ORDER PLACED SUCCESSFULLY!")
            resp = order_result.success_response
            if isinstance(resp, dict):
                order_id = resp.get('order_id', 'unknown')
            else:
                order_id = resp.order_id if hasattr(resp, 'order_id') else 'unknown'
            print(f"   Order ID: {order_id}")
            
            # 3. Check for futures position
            print("\n3. Waiting for position...")
            await asyncio.sleep(5)
            
            positions = client.list_futures_positions()
            if hasattr(positions, 'positions') and len(positions.positions) > 0:
                print("\n🎉 FUTURES POSITION CONFIRMED!")
                for pos in positions.positions:
                    print(f"\n   Position Details:")
                    print(f"   - Product: {pos.product_id}")
                    print(f"   - Side: {pos.side}")
                    print(f"   - Contracts: {pos.number_of_contracts}")
                    if hasattr(pos, 'avg_entry_price'):
                        print(f"   - Entry Price: ${pos.avg_entry_price}")
            else:
                print("\n⚠️  No futures position visible yet")
                
        else:
            print("\n❌ Order failed - not successful")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        # If it's an insufficient funds error, show more detail
        if "INSUFFICIENT" in str(e).upper():
            print("\n💡 This suggests the leverage parameter is working,")
            print("   but the sweep hasn't completed yet.")

if __name__ == "__main__":
    asyncio.run(test_futures_order())