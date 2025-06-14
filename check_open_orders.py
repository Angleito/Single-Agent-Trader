#!/usr/bin/env python3
"""
Check for open orders and recent trades.
"""

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()


def main():
    """Check orders and trades."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("CHECKING ORDERS AND TRADES")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Check current balance first
        print("\n1. Current Balance Status:")
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        print(f"  CBI Balance: ${bs.cbi_usd_balance['value']}")
        print(f"  Futures Buying Power: ${bs.futures_buying_power['value']}")
        print(f"  Total USD: ${bs.total_usd_balance['value']}")
        
        # Check open orders
        print("\n2. Open Orders:")
        try:
            orders = client.list_orders(order_status="OPEN")
            if orders.orders:
                for order in orders.orders:
                    print(f"\n  Order ID: {order.order_id}")
                    print(f"  Product: {order.product_id}")
                    print(f"  Side: {order.side}")
                    print(f"  Size: {order.size}")
                    print(f"  Status: {order.status}")
            else:
                print("  No open orders")
        except Exception as e:
            print(f"  Error checking orders: {e}")
        
        # Check recent fills
        print("\n3. Recent Fills (last 24 hours):")
        try:
            # Get fills from last 24 hours
            start_time = datetime.utcnow() - timedelta(hours=24)
            fills = client.list_fills(
                product_id="ETH-USD",
                start_sequence_timestamp=start_time.isoformat() + "Z"
            )
            
            if hasattr(fills, 'fills') and fills.fills:
                total_eth = 0
                total_usd = 0
                for fill in fills.fills:
                    print(f"\n  Fill ID: {fill.trade_id}")
                    print(f"  Product: {fill.product_id}")
                    print(f"  Side: {fill.side}")
                    print(f"  Size: {fill.size} ETH")
                    print(f"  Price: ${fill.price}")
                    print(f"  Fee: ${fill.commission}")
                    print(f"  Time: {fill.trade_time}")
                    
                    if fill.side == "BUY":
                        total_eth += float(fill.size)
                        total_usd -= float(fill.size) * float(fill.price)
                    else:
                        total_eth -= float(fill.size)
                        total_usd += float(fill.size) * float(fill.price)
                
                print(f"\n  Summary: Net ETH: {total_eth:.6f}, Net USD: ${total_usd:.2f}")
            else:
                print("  No recent fills")
                
        except Exception as e:
            print(f"  Error checking fills: {e}")
        
        # Check portfolio breakdown
        print("\n4. Portfolio Breakdown:")
        try:
            breakdown = client.get_portfolio_breakdown(
                portfolio_uuid="1f3ed8bf-a65c-5022-8258-87ce50c517f6"
            )
            
            if hasattr(breakdown, 'breakdown'):
                bd = breakdown.breakdown
                print(f"  Total Balance: ${bd.portfolio_balances.total_balance.value}")
                print(f"  Total Cash: ${bd.portfolio_balances.total_cash_value.value}")
                if hasattr(bd, 'futures_positions'):
                    print(f"  Futures Positions Value: ${bd.futures_positions_value.value}")
            
        except Exception as e:
            print(f"  Error getting breakdown: {e}")
        
        # Check ETH balance
        print("\n5. ETH Holdings:")
        accounts = client.get_accounts()
        for acc in accounts.accounts:
            if acc.currency == "ETH":
                print(f"  ETH Balance: {acc.available_balance.value} ETH")
                if hasattr(acc, 'hold'):
                    print(f"  ETH On Hold: {acc.hold.value} ETH")
                break
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()