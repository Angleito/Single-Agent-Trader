#!/usr/bin/env python3
"""
Transfer funds from spot to futures using the correct API method.
"""

import os
from decimal import Decimal
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()


def main():
    """Execute the transfer."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("EXECUTING SPOT TO FUTURES TRANSFER")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Get current balances
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        
        cbi_balance = Decimal(bs.cbi_usd_balance['value'])
        cfm_balance = Decimal(bs.cfm_usd_balance['value'])
        
        print(f"\nCurrent Balances:")
        print(f"  CBI (Spot):    ${cbi_balance:,.2f}")
        print(f"  CFM (Futures): ${cfm_balance:,.2f}")
        
        if cbi_balance <= 0:
            print("\nNo funds available to transfer.")
            return
        
        # Default portfolio UUID (from the bot code)
        default_portfolio_uuid = "1f3ed8bf-a65c-5022-8258-87ce50c517f6"
        
        print(f"\n💸 Transferring ${cbi_balance:,.2f} to futures...")
        
        # Use allocate_portfolio with to_fcm_account=True
        result = client.allocate_portfolio(
            source_portfolio_uuid=default_portfolio_uuid,
            amount=str(cbi_balance),
            currency="USD",
            to_fcm_account=True
        )
        
        print(f"✅ Transfer initiated successfully!")
        print(f"Result: {result}")
        
        # Wait and check balance
        print("\nWaiting for transfer to complete...")
        import time
        time.sleep(10)
        
        # Check updated balances
        fcm_new = client.get_futures_balance_summary()
        bs_new = fcm_new.balance_summary
        
        new_cbi = Decimal(bs_new.cbi_usd_balance['value'])
        new_cfm = Decimal(bs_new.cfm_usd_balance['value'])
        
        print(f"\nUpdated Balances:")
        print(f"  CBI (Spot):    ${new_cbi:,.2f}")
        print(f"  CFM (Futures): ${new_cfm:,.2f}")
        
        transferred = cbi_balance - new_cbi
        print(f"\n💰 CASH AMOUNT TRANSFERRED: ${transferred:,.2f}")
        
        if transferred > 0:
            print(f"\n✅ Successfully transferred ${transferred:,.2f} from spot to futures!")
        else:
            print("\n⚠️  No funds were transferred. The transfer may still be processing.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()