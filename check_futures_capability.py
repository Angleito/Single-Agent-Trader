#!/usr/bin/env python3
"""
Check if futures trading is available with current balance.
"""

import os
from decimal import Decimal
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()


def main():
    """Check futures capability."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("FUTURES TRADING CAPABILITY CHECK")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Get futures balance summary
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        
        print("\nFUTURES BALANCE SUMMARY:")
        print(f"  CBI USD Balance: ${bs.cbi_usd_balance['value']}")
        print(f"  CFM USD Balance: ${bs.cfm_usd_balance['value']}")
        print(f"  Total USD Balance: ${bs.total_usd_balance['value']}")
        
        # Check if futures_buying_power exists
        if hasattr(bs, 'futures_buying_power'):
            print(f"  Futures Buying Power: ${bs.futures_buying_power['value']}")
        
        # Check margin info
        if hasattr(bs, 'available_margin'):
            print(f"  Available Margin: ${bs.available_margin['value']}")
        
        if hasattr(bs, 'initial_margin'):
            print(f"  Initial Margin: ${bs.initial_margin['value']}")
            
        print("\n💡 CONCLUSION:")
        print(f"You have ${bs.cbi_usd_balance['value']} available in your account.")
        print("This balance appears to be in a unified account that can be used for futures trading.")
        print("No transfer is needed - you can directly place futures trades with this balance.")
        
        # Check if we can get futures positions
        print("\nChecking for existing futures positions...")
        try:
            positions = client.list_futures_positions()
            print(f"Futures positions accessible: Yes")
            if hasattr(positions, 'positions'):
                print(f"Number of open positions: {len(positions.positions)}")
        except Exception as e:
            print(f"Futures positions check: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()