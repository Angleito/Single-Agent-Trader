#!/usr/bin/env python3
"""
Test if ETH-USD can be traded as futures by specifying account/portfolio type.
"""

import os
from decimal import Decimal
from dotenv import load_dotenv
from coinbase.rest import RESTClient
import uuid

# Load environment variables
load_dotenv()


def main():
    """Test futures trading with account type specifications."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("TESTING FUTURES TRADING WITH ACCOUNT TYPE")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # First, let's understand the account structure better
        print("\n1. Analyzing account structure...")
        
        # Get accounts
        accounts = client.get_accounts()
        print(f"\nFound {len(accounts.accounts)} accounts:")
        
        futures_account_id = None
        for acc in accounts.accounts:
            if hasattr(acc, 'type'):
                print(f"  Account: {acc.currency}, Type: {acc.type}, UUID: {getattr(acc, 'uuid', 'N/A')}")
                if 'FUTURES' in str(acc.type).upper() or 'CFM' in str(acc.type).upper():
                    futures_account_id = getattr(acc, 'uuid', None)
        
        # Get portfolios with more detail
        print("\n2. Analyzing portfolios...")
        portfolios = client.get_portfolios()
        
        for portfolio in portfolios.portfolios:
            print(f"\nPortfolio: {portfolio.name}")
            print(f"  UUID: {portfolio.uuid}")
            print(f"  Type: {portfolio.type}")
            
            # Check if portfolio has trading type info
            if hasattr(portfolio, 'trading_type'):
                print(f"  Trading Type: {portfolio.trading_type}")
            
        # Check futures balance with more detail
        print("\n3. Checking futures capabilities...")
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        
        print(f"  Futures Buying Power: ${bs.futures_buying_power['value']}")
        print(f"  CBI Balance: ${bs.cbi_usd_balance['value']}")
        print(f"  CFM Balance: ${bs.cfm_usd_balance['value']}")
        
        # Key insight: If CFM balance is 0 but futures buying power exists,
        # it might mean we need to specify the order differently
        
        # Test different order configurations
        print("\n4. Testing order configurations...")
        
        # Configuration 1: Specify margin_type
        print("\n  a) Testing with margin_type=ISOLATED...")
        try:
            result = client.create_order(
                client_order_id=f"test-{uuid.uuid4().hex[:8]}",
                product_id="ETH-USD",
                side="BUY",
                order_configuration={
                    "market_market_ioc": {"base_size": "0.1"}
                },
                margin_type="ISOLATED",
                leverage="5"
            )
            print(f"     Result: {result}")
        except Exception as e:
            print(f"     Failed: {e}")
        
        # Configuration 2: Try with post_only flag (sometimes indicates futures)
        print("\n  b) Testing with different order configuration...")
        try:
            result = client.create_order(
                client_order_id=f"test-{uuid.uuid4().hex[:8]}",
                product_id="ETH-USD",
                side="BUY",
                order_configuration={
                    "limit_limit_fok": {
                        "base_size": "0.1",
                        "limit_price": "2000",  # Below market
                        "post_only": False
                    }
                },
                leverage="5"
            )
            print(f"     Result: {result}")
        except Exception as e:
            print(f"     Failed: {e}")
        
        # Configuration 3: Check if we need to specify account
        print("\n  c) Testing with explicit source account...")
        if futures_account_id:
            try:
                result = client.create_order(
                    client_order_id=f"test-{uuid.uuid4().hex[:8]}",
                    product_id="ETH-USD",
                    side="BUY",
                    order_configuration={
                        "market_market_ioc": {"base_size": "0.1"}
                    },
                    source_account_id=futures_account_id,
                    leverage="5"
                )
                print(f"     Result: {result}")
            except Exception as e:
                print(f"     Failed: {e}")
        
        # Configuration 4: Try INTX parameters (Coinbase International Exchange)
        print("\n  d) Testing with INTX parameters...")
        try:
            # INTX might use different endpoints or parameters
            result = client.create_order(
                client_order_id=f"test-{uuid.uuid4().hex[:8]}",
                product_id="ETH-USD",
                side="BUY", 
                order_configuration={
                    "market_market_ioc": {"base_size": "0.1"}
                },
                retail_portfolio_id="intx",  # Try INTX portfolio
                leverage="5"
            )
            print(f"     Result: {result}")
        except Exception as e:
            print(f"     Failed: {e}")
            
        print("\n5. IMPORTANT DISCOVERY:")
        print("=" * 60)
        print("Based on the tests, it appears that:")
        print("1. You have futures buying power ($267.56)")
        print("2. But orders with 'leverage' parameter fail with insufficient funds")
        print("3. This suggests your account type might require a different approach")
        print("\nPossible solutions:")
        print("- You might need to use Coinbase International Exchange (INTX)")
        print("- Or use a different API endpoint for futures")
        print("- Or the futures feature might not be fully enabled on your account")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()