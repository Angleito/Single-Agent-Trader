#!/usr/bin/env python3
"""Final test of futures trading with all fixes applied."""

import os
import asyncio
import uuid
from decimal import Decimal
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()

async def test_futures_trading():
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("COINBASE FUTURES TRADING TEST - FINAL")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # 1. Check futures balance before
        print("\n1. Checking futures balance BEFORE...")
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        cfm_before = Decimal(bs.cfm_usd_balance['value'])
        cbi_before = Decimal(bs.cbi_usd_balance['value'])
        buying_power = Decimal(bs.futures_buying_power['value'])
        
        print(f"  CFM USD Balance: ${cfm_before}")
        print(f"  CBI USD Balance: ${cbi_before}")
        print(f"  Futures Buying Power: ${buying_power}")
        
        # 2. If CFM balance is 0, schedule a sweep
        if cfm_before == 0 and cbi_before > 0:
            print("\n2. Scheduling futures sweep...")
            try:
                # Cancel any pending sweeps
                try:
                    client.cancel_pending_futures_sweep()
                    print("  Cancelled pending sweep")
                except:
                    pass
                
                # Schedule new sweep
                transfer_amount = min(50, cbi_before)  # Transfer up to $50
                sweep = client.schedule_futures_sweep(usd_amount=str(transfer_amount))
                print(f"  ✅ Scheduled sweep for ${transfer_amount}")
                
                # Wait for sweep
                print("  Waiting 10 seconds for sweep to process...")
                await asyncio.sleep(10)
                
                # Check balance again
                fcm = client.get_futures_balance_summary()
                bs = fcm.balance_summary
                cfm_after = Decimal(bs.cfm_usd_balance['value'])
                print(f"  CFM balance after sweep: ${cfm_after}")
                
            except Exception as e:
                print(f"  ❌ Sweep failed: {e}")
                return
        
        # 3. Try to place a futures order with leverage
        print("\n3. Placing FUTURES order with leverage...")
        try:
            order_result = client.create_order(
                client_order_id=str(uuid.uuid4()),
                product_id="ETH-USD",
                side="BUY",
                order_configuration={
                    "market_market_ioc": {"base_size": "0.001"}  # Very small order
                },
                leverage="2"  # This parameter makes it a futures order
            )
            
            if order_result.success:
                print("  ✅ Futures order placed successfully!")
                resp = order_result.success_response
                if isinstance(resp, dict):
                    order_id = resp['order_id']
                else:
                    order_id = resp.order_id
                print(f"  Order ID: {order_id}")
                
                # 4. Check if we have a futures position
                await asyncio.sleep(3)
                print("\n4. Checking futures positions...")
                positions = client.list_futures_positions()
                if hasattr(positions, 'positions') and len(positions.positions) > 0:
                    print("  ✅ FUTURES POSITION CREATED!")
                    for pos in positions.positions:
                        print(f"    Product: {pos.product_id}")
                        print(f"    Side: {pos.side}")
                        print(f"    Contracts: {pos.number_of_contracts}")
                        print(f"    Average Price: ${pos.avg_entry_price}")
                else:
                    print("  ⚠️  No futures positions found (order may be too small)")
            else:
                print("  ❌ Order failed")
                
        except Exception as e:
            print(f"  ❌ Order error: {e}")
            
        # 5. Final balance check
        print("\n5. Final futures balance...")
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        print(f"  CFM USD Balance: ${bs.cfm_usd_balance['value']}")
        print(f"  Unrealized PnL: ${bs.unrealized_pnl['value']}")
        print(f"  Daily Realized PnL: ${bs.daily_realized_pnl['value']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_futures_trading())