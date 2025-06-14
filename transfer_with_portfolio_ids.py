#!/usr/bin/env python3
"""
Transfer funds using move_portfolio_funds with correct portfolio IDs.
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
    
    print("SPOT TO FUTURES TRANSFER")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Get current balances
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        cbi_balance = Decimal(bs.cbi_usd_balance['value'])
        
        print(f"\nAvailable to transfer: ${cbi_balance:,.2f}")
        
        if cbi_balance <= 0:
            print("No funds available.")
            return
        
        # Get portfolios to find futures portfolio
        print("\nFinding portfolio IDs...")
        portfolios = client.get_portfolios()
        
        default_portfolio_id = None
        futures_portfolio_id = None
        
        for portfolio in portfolios.portfolios:
            print(f"Portfolio: {portfolio.name} (UUID: {portfolio.uuid}, Type: {portfolio.type})")
            
            if hasattr(portfolio, 'is_default') and portfolio.is_default:
                default_portfolio_id = portfolio.uuid
            elif portfolio.name and ('futures' in portfolio.name.lower() or 'intx' in portfolio.name.lower()):
                futures_portfolio_id = portfolio.uuid
            elif portfolio.type and portfolio.type.lower() == 'intx':
                futures_portfolio_id = portfolio.uuid
        
        # Use defaults if not found
        if not default_portfolio_id:
            default_portfolio_id = "1f3ed8bf-a65c-5022-8258-87ce50c517f6"
        
        print(f"\nSource portfolio (default): {default_portfolio_id}")
        print(f"Target portfolio (futures): {futures_portfolio_id}")
        
        if not futures_portfolio_id:
            print("\n⚠️  Could not find futures portfolio ID")
            print("Your account may not have futures enabled or the portfolio structure is different.")
            return
        
        # Execute transfer
        print(f"\n💸 Transferring ${cbi_balance:,.2f} to futures...")
        
        result = client.move_portfolio_funds(
            value=str(cbi_balance),
            currency="USD",
            source_portfolio_uuid=default_portfolio_id,
            target_portfolio_uuid=futures_portfolio_id
        )
        
        print(f"✅ Transfer executed!")
        print(f"Result: {result}")
        
        # Check balance after a short wait
        import time
        time.sleep(5)
        
        fcm_new = client.get_futures_balance_summary()
        new_cbi = Decimal(fcm_new.balance_summary.cbi_usd_balance['value'])
        new_cfm = Decimal(fcm_new.balance_summary.cfm_usd_balance['value'])
        
        print(f"\nUpdated Balances:")
        print(f"  CBI (Spot):    ${new_cbi:,.2f}")
        print(f"  CFM (Futures): ${new_cfm:,.2f}")
        
        transferred = cbi_balance - new_cbi
        print(f"\n💰 CASH AMOUNT TRANSFERRED: ${transferred:,.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()