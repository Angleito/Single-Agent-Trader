#!/usr/bin/env python3
"""Test placing a real futures order once sweep completes."""

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
    print("FUTURES ORDER TEST")
    print("=" * 60)
    
    try:
        # 1. Check CFM balance
        print("\n1. Checking CFM balance...")
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        cfm_balance = Decimal(bs.cfm_usd_balance['value'])
        buying_power = Decimal(bs.futures_buying_power['value'])
        
        print(f"   CFM Balance: ${cfm_balance}")
        print(f"   Buying Power: ${buying_power}")
        
        if cfm_balance == 0:
            print("\n❌ No CFM balance available. Please wait for sweep to complete.")
            return
        
        # 2. Calculate safe order size
        print("\n2. Calculating order size...")
        # Get current ETH price
        product = client.get_product('ETH-USD')
        current_price = Decimal(product.price)
        print(f"   Current ETH price: ${current_price}")
        
        # Calculate minimum order
        # For 5x leverage, $50 margin controls $250 notional
        margin_to_use = Decimal("50")  # Use $50 margin
        leverage = 5
        notional_value = margin_to_use * leverage
        eth_quantity = notional_value / current_price
        
        # Round to appropriate precision
        eth_quantity = round(eth_quantity, 6)
        
        print(f"   Margin: ${margin_to_use}")
        print(f"   Leverage: {leverage}x")
        print(f"   Notional: ${notional_value}")
        print(f"   ETH quantity: {eth_quantity}")
        
        # 3. Place futures order
        print("\n3. Placing futures order...")
        print("   Parameters:")
        print(f"   - Product: ETH-USD")
        print(f"   - Side: BUY")
        print(f"   - Size: {eth_quantity} ETH")
        print(f"   - Leverage: {leverage}")
        
        order_result = client.create_order(
            client_order_id=str(uuid.uuid4()),
            product_id="ETH-USD",
            side="BUY",
            order_configuration={
                "market_market_ioc": {"base_size": str(eth_quantity)}
            },
            leverage=str(leverage)  # This should route to futures
        )
        
        if order_result.success:
            print("\n✅ ORDER PLACED SUCCESSFULLY!")
            resp = order_result.success_response
            order_id = resp.order_id if hasattr(resp, 'order_id') else resp.get('order_id')
            print(f"   Order ID: {order_id}")
            
            # 4. Wait and check for futures position
            print("\n4. Checking for futures position...")
            await asyncio.sleep(3)
            
            positions = client.list_futures_positions()
            if hasattr(positions, 'positions') and len(positions.positions) > 0:
                print("\n🎉 FUTURES POSITION CREATED!")
                for pos in positions.positions:
                    print(f"\n   Position Details:")
                    print(f"   - Product: {pos.product_id}")
                    print(f"   - Side: {pos.side}")
                    print(f"   - Contracts: {pos.number_of_contracts}")
                    print(f"   - Entry Price: ${pos.avg_entry_price}")
                    print(f"   - Unrealized PnL: ${pos.unrealized_pnl}")
            else:
                print("\n⚠️  No futures position found yet")
                print("   Order may still be processing or was too small")
                
            # 5. Check updated balance
            print("\n5. Updated balances...")
            fcm = client.get_futures_balance_summary()
            bs = fcm.balance_summary
            print(f"   CFM Balance: ${bs.cfm_usd_balance['value']}")
            print(f"   Initial Margin: ${bs.initial_margin['value']}")
            print(f"   Available Margin: ${bs.available_margin['value']}")
            
        else:
            print("\n❌ Order failed")
            print(f"   Reason: Order was not successful")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_futures_order())