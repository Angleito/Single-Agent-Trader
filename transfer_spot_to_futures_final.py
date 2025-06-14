#!/usr/bin/env python3
"""
Transfer $267.66 from spot (CBI) to futures (CFM).
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
    
    print("TRANSFERRING SPOT BALANCE TO FUTURES")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Get current CBI balance from futures balance summary
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        
        # Handle dict format
        if isinstance(bs.cbi_usd_balance, dict):
            cbi_balance = Decimal(bs.cbi_usd_balance['value'])
            cfm_balance = Decimal(bs.cfm_usd_balance['value'])
        else:
            cbi_balance = Decimal(bs.cbi_usd_balance.value)
            cfm_balance = Decimal(bs.cfm_usd_balance.value)
        
        print(f"\nCurrent Balances:")
        print(f"  CBI (Spot):    ${cbi_balance:,.2f}")
        print(f"  CFM (Futures): ${cfm_balance:,.2f}")
        print(f"  Total:         ${cbi_balance + cfm_balance:,.2f}")
        
        if cbi_balance <= 0:
            print("\nNo funds available to transfer.")
            return
        
        # Execute transfer using direct endpoint
        print(f"\n💸 Transferring ${cbi_balance:,.2f} to futures account...")
        
        # Try method 1: Post to cash transfers endpoint
        try:
            transfer_request = {
                "source_account": "CBI",
                "target_account": "CFM",
                "amount": str(cbi_balance),
                "currency": "USD"
            }
            
            result = client.post('/api/v3/brokerage/cfm/cash_transfers', transfer_request)
            print(f"✅ Transfer successful! Result: {result}")
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
            
            # Try method 2: Allocate portfolio
            print("\nTrying alternative method...")
            try:
                # Get default portfolio ID
                portfolios = client.get_portfolios()
                default_id = None
                for p in portfolios.portfolios:
                    if p.is_default:
                        default_id = p.uuid
                        break
                
                if not default_id:
                    default_id = "1f3ed8bf-a65c-5022-8258-87ce50c517f6"
                
                result = client.allocate_portfolio(
                    source_portfolio_uuid=default_id,
                    amount=str(cbi_balance),
                    currency="USD"
                )
                print(f"✅ Transfer successful using allocate! Result: {result}")
                
            except Exception as e2:
                print(f"Method 2 also failed: {e2}")
                
                # Try method 3: Direct transfer
                print("\nTrying third method...")
                try:
                    result = client.create_transfer(
                        source_portfolio_id=default_id,
                        target_portfolio_id="futures",
                        amount=str(cbi_balance),
                        currency="USD"
                    )
                    print(f"✅ Transfer successful! Result: {result}")
                except Exception as e3:
                    print(f"All methods failed. Error: {e3}")
                    return
        
        # Check updated balance
        print("\nChecking updated balances...")
        import time
        time.sleep(5)
        
        fcm_new = client.get_futures_balance_summary()
        bs_new = fcm_new.balance_summary
        
        if isinstance(bs_new.cbi_usd_balance, dict):
            new_cbi = Decimal(bs_new.cbi_usd_balance['value'])
            new_cfm = Decimal(bs_new.cfm_usd_balance['value'])
        else:
            new_cbi = Decimal(bs_new.cbi_usd_balance.value)
            new_cfm = Decimal(bs_new.cfm_usd_balance.value)
        
        print(f"\nUpdated Balances:")
        print(f"  CBI (Spot):    ${new_cbi:,.2f}")
        print(f"  CFM (Futures): ${new_cfm:,.2f}")
        print(f"  Total:         ${new_cbi + new_cfm:,.2f}")
        
        transferred = cbi_balance - new_cbi
        print(f"\n💰 CASH AMOUNT TRANSFERRED: ${transferred:,.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()