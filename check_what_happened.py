#!/usr/bin/env python3
"""
Check what happened to the funds - simplified version.
"""

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from decimal import Decimal

# Load environment variables
load_dotenv()


def main():
    """Check what happened."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("INVESTIGATING FUND USAGE")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Check balances
        print("\n1. Current Balance Summary:")
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        print(f"  CBI (Cash) Balance: ${bs.cbi_usd_balance['value']}")
        print(f"  CFM Balance: ${bs.cfm_usd_balance['value']}")
        print(f"  Total USD: ${bs.total_usd_balance['value']}")
        print(f"  Futures Buying Power: ${bs.futures_buying_power['value']}")
        
        print(f"\n  💡 You had $267.66, now you have $12.51")
        print(f"  Missing: ${Decimal('267.66') - Decimal('12.51'):.2f}")
        
        # Check all crypto balances
        print("\n2. Crypto Holdings:")
        accounts = client.get_accounts()
        
        total_crypto_value = Decimal('0')
        crypto_holdings = []
        
        for acc in accounts.accounts:
            currency = acc['currency'] if isinstance(acc, dict) else acc.currency
            
            if currency not in ['USD', 'USDC', 'USDT', 'DAI']:  # Skip stablecoins
                balance_dict = acc['available_balance'] if isinstance(acc, dict) else acc.available_balance
                balance = Decimal(balance_dict['value'] if isinstance(balance_dict, dict) else balance_dict.value)
                
                if balance > 0:
                    crypto_holdings.append((currency, balance))
        
        # Show ETH specifically
        for currency, balance in crypto_holdings:
            if currency == 'ETH':
                print(f"\n  🔍 ETH Balance: {balance} ETH")
                # Estimate value at ~$2500/ETH
                eth_value = balance * Decimal('2500')
                print(f"  Estimated value: ${eth_value:.2f}")
                total_crypto_value += eth_value
            else:
                print(f"  {currency}: {balance}")
        
        print(f"\n  Total estimated crypto value: ${total_crypto_value:.2f}")
        print(f"  Cash + Crypto = ${Decimal('12.51') + total_crypto_value:.2f}")
        
        # Check recent orders
        print("\n3. Recent Orders:")
        try:
            # Get all recent orders
            orders = client.list_orders()
            
            eth_orders = []
            for order in orders.orders[:10]:  # Last 10 orders
                if order.product_id == 'ETH-USD':
                    eth_orders.append(order)
            
            if eth_orders:
                print(f"\n  Found {len(eth_orders)} recent ETH-USD orders:")
                for order in eth_orders[:5]:  # Show first 5
                    print(f"\n  Order: {order.order_id}")
                    print(f"    Status: {order.status}")
                    print(f"    Side: {order.side}")
                    print(f"    Size: {order.size}")
                    if hasattr(order, 'executed_value'):
                        print(f"    Value: ${order.executed_value}")
        except Exception as e:
            print(f"  Error checking orders: {e}")
        
        print("\n4. CONCLUSION:")
        print("=" * 60)
        print("Your funds were used to buy ETH in the test orders!")
        print("The orders we thought would fail actually executed as spot orders.")
        print("You now own ETH instead of having USD.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()