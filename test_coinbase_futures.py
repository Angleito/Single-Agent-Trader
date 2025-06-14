#!/usr/bin/env python3
"""Test Coinbase futures trading to understand the correct approach."""

import os
import asyncio
import uuid
from decimal import Decimal
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()

async def test_futures():
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("COINBASE FUTURES TEST")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # 1. Check account structure
        print("\n1. Checking account balances...")
        accounts_resp = client.get_accounts()
        
        total_usd = Decimal("0")
        for acc in accounts_resp.accounts:
            if acc.currency == "USD":
                # available_balance is a dict
                balance = Decimal(acc.available_balance['value'])
                print(f"  {acc.currency} balance: ${balance}")
                total_usd += balance
        
        # 2. Check futures balance
        print("\n2. Checking futures balance...")
        try:
            fcm = client.get_futures_balance_summary()
            bs = fcm.balance_summary
            # Handle dict format for balance fields
            print(f"  CFM USD Balance: ${bs.cfm_usd_balance['value']}")
            print(f"  CBI USD Balance: ${bs.cbi_usd_balance['value']}")
            print(f"  Total USD Balance: ${bs.total_usd_balance['value']}")
            print(f"  Futures Buying Power: ${bs.futures_buying_power['value']}")
            print(f"  Available Margin: ${bs.available_margin['value']}")
        except Exception as e:
            print(f"  Error getting futures balance: {e}")
        
        # 2.5 Transfer funds from CBI to CFM if needed
        if Decimal(bs.cfm_usd_balance['value']) == 0 and Decimal(bs.cbi_usd_balance['value']) > 0:
            print("\n2.5. Transferring funds from CBI to CFM...")
            try:
                transfer_amount = "100"  # Transfer $100 to CFM
                # Try using schedule_futures_sweep
                sweep = client.schedule_futures_sweep(
                    usd_amount=transfer_amount
                )
                print(f"  ✅ Sweep scheduled: ${transfer_amount} USD")
                print("  Waiting 5 seconds for transfer to complete...")
                await asyncio.sleep(5)
                
                # Check balance again
                fcm = client.get_futures_balance_summary()
                bs = fcm.balance_summary
                print(f"  Updated CFM USD Balance: ${bs.cfm_usd_balance['value']}")
            except Exception as e:
                print(f"  ❌ Transfer failed: {e}")
                print(f"  Note: You may need to manually transfer funds from CBI to CFM on the Coinbase website")
        
        # 3. Test tiny spot order first
        print("\n3. Testing tiny SPOT order (0.001 ETH)...")
        try:
            spot_order = client.create_order(
                client_order_id=str(uuid.uuid4()),
                product_id="ETH-USD",
                side="BUY",
                order_configuration={
                    "market_market_ioc": {"base_size": "0.001"}
                }
                # NO leverage parameter = spot order
            )
            # Handle object response
            if spot_order.success:
                print(f"  ✅ Spot order successful!")
                resp = spot_order.success_response
                # Handle both dict and object formats
                if isinstance(resp, dict):
                    print(f"    Order ID: {resp['order_id']}")
                    print(f"    Product: {resp['product_id']}")
                    print(f"    Side: {resp['side']}")
                else:
                    print(f"    Order ID: {resp.order_id}")
                    print(f"    Product: {resp.product_id}")
                    print(f"    Side: {resp.side}")
            else:
                print(f"  ❌ Spot order failed")
        except Exception as e:
            print(f"  ❌ Spot order failed: {e}")
        
        # 4. Test futures order with leverage
        print("\n4. Testing FUTURES order with leverage...")
        
        # Try different approaches
        approaches = [
            {
                "name": "Minimal futures (0.01 ETH, 2x leverage)",
                "base_size": "0.01",
                "leverage": "2"
            },
            {
                "name": "Tiny futures (0.001 ETH, 5x leverage)",
                "base_size": "0.001",
                "leverage": "5"
            },
            {
                "name": "With margin_type CROSS",
                "base_size": "0.01",
                "leverage": "2",
                "margin_type": "CROSS"
            },
            {
                "name": "With margin_type ISOLATED",
                "base_size": "0.01",
                "leverage": "2",
                "margin_type": "ISOLATED"
            }
        ]
        
        for approach in approaches:
            print(f"\n  Testing: {approach['name']}")
            try:
                order_data = {
                    "client_order_id": str(uuid.uuid4()),
                    "product_id": "ETH-USD",
                    "side": "BUY",
                    "order_configuration": {
                        "market_market_ioc": {"base_size": approach['base_size']}
                    },
                    "leverage": approach['leverage']
                }
                
                # Add optional parameters
                if 'margin_type' in approach:
                    order_data['margin_type'] = approach['margin_type']
                
                result = client.create_order(**order_data)
                # Handle object response
                if result.success:
                    print(f"    ✅ Success!")
                    resp = result.success_response
                    # Handle both dict and object formats
                    if isinstance(resp, dict):
                        print(f"      Order ID: {resp['order_id']}")
                        print(f"      Product: {resp['product_id']}")
                        print(f"      Side: {resp['side']}")
                    else:
                        print(f"      Order ID: {resp.order_id}")
                        print(f"      Product: {resp.product_id}")
                        print(f"      Side: {resp.side}")
                else:
                    print(f"    ❌ Order failed")
                
                # Check if this created a futures position
                await asyncio.sleep(2)
                positions = client.list_futures_positions()
                if len(positions.positions) > 0:
                    print("    ✅ CONFIRMED: Futures position created!")
                    for pos in positions.positions:
                        print(f"      {pos.product_id}: {pos.number_of_contracts} contracts")
                    break  # Stop testing once we find what works
                else:
                    print("    ⚠️  No futures position found")
                    
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        
        # 5. Check final state
        print("\n5. Final account state...")
        positions = client.list_futures_positions()
        print(f"  Futures positions: {len(positions.positions)}")
        for pos in positions.positions:
            print(f"    {pos.product_id}: {pos.side} {pos.number_of_contracts} contracts")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_futures())