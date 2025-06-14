#!/usr/bin/env python3
"""
Test placing a small futures order to diagnose the insufficient funds issue.
"""

import os
from decimal import Decimal
from dotenv import load_dotenv
from coinbase.rest import RESTClient
import uuid

# Load environment variables
load_dotenv()


def main():
    """Test futures order."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("FUTURES ORDER TEST")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Check current balance and futures info
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        
        print(f"\nAvailable Futures Buying Power: ${bs.futures_buying_power['value']}")
        print(f"CBI Balance: ${bs.cbi_usd_balance['value']}")
        
        # Get current ETH price
        print("\nGetting current ETH-USD price...")
        ticker = client.get_product("ETH-USD")
        current_price = Decimal(ticker['price'])
        print(f"Current ETH-USD price: ${current_price}")
        
        # Calculate minimal order size
        # Coinbase futures: 1 contract = 0.1 ETH
        contract_size = Decimal("0.1")
        min_contracts = 1
        order_size = contract_size * min_contracts  # 0.1 ETH
        
        # Calculate notional value
        notional_value = order_size * current_price
        print(f"\nOrder details:")
        print(f"  Contracts: {min_contracts}")
        print(f"  Size: {order_size} ETH")
        print(f"  Notional value: ${notional_value:.2f}")
        
        # Try different order configurations
        print("\n" + "="*60)
        print("TESTING DIFFERENT ORDER METHODS:")
        print("="*60)
        
        # Test 1: Basic futures order
        print("\n1. Testing basic futures market order...")
        client_order_id = f"test-{uuid.uuid4().hex[:8]}"
        
        order_data = {
            "product_id": "ETH-USD",
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": str(order_size)
                }
            }
        }
        
        try:
            result = client.create_order(client_order_id, **order_data)
            print(f"✅ Order successful: {result}")
        except Exception as e:
            print(f"❌ Basic order failed: {e}")
        
        # Test 2: With leverage parameter
        print("\n2. Testing with leverage parameter...")
        client_order_id = f"test-{uuid.uuid4().hex[:8]}"
        
        order_data_with_leverage = {
            "product_id": "ETH-USD",
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": str(order_size)
                }
            },
            "leverage": "5"
        }
        
        try:
            result = client.create_order(client_order_id, **order_data_with_leverage)
            print(f"✅ Order with leverage successful: {result}")
        except Exception as e:
            print(f"❌ Order with leverage failed: {e}")
        
        # Test 3: With portfolio ID
        print("\n3. Testing with portfolio ID...")
        
        # Get portfolios
        portfolios = client.get_portfolios()
        portfolio_id = None
        for p in portfolios.portfolios:
            print(f"  Found portfolio: {p.name} (ID: {p.uuid})")
            portfolio_id = p.uuid
        
        if portfolio_id:
            client_order_id = f"test-{uuid.uuid4().hex[:8]}"
            
            order_data_with_portfolio = {
                "product_id": "ETH-USD",
                "side": "BUY",
                "order_configuration": {
                    "market_market_ioc": {
                        "base_size": str(order_size)
                    }
                },
                "retail_portfolio_id": portfolio_id
            }
            
            try:
                result = client.create_order(client_order_id, **order_data_with_portfolio)
                print(f"✅ Order with portfolio ID successful: {result}")
            except Exception as e:
                print(f"❌ Order with portfolio ID failed: {e}")
        
        # Test 4: Try perpetuals endpoint
        print("\n4. Testing perpetuals-specific order...")
        
        # First check if we have access to perpetuals
        try:
            perps_portfolios = client.get_perps_portfolio_summary(portfolio_id)
            print(f"  Perpetuals portfolio accessible: {perps_portfolios}")
        except Exception as e:
            print(f"  No perpetuals access: {e}")
        
        # Check product details
        print("\n5. Checking product details for ETH-USD...")
        try:
            product = client.get_product("ETH-USD")
            print(f"  Product type: {product.get('product_type', 'Unknown')}")
            print(f"  Quote increment: {product.get('quote_increment')}")
            print(f"  Base increment: {product.get('base_increment')}")
            print(f"  Min order size: {product.get('base_min_size')}")
        except Exception as e:
            print(f"  Error getting product details: {e}")
        
        # Test with smaller amount
        print("\n6. Testing with very small order...")
        tiny_order_size = Decimal("0.001")  # Much smaller
        client_order_id = f"test-{uuid.uuid4().hex[:8]}"
        
        tiny_order_data = {
            "product_id": "ETH-USD",
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": str(tiny_order_size)
                }
            }
        }
        
        try:
            result = client.create_order(client_order_id, **tiny_order_data)
            print(f"✅ Tiny order successful: {result}")
        except Exception as e:
            print(f"❌ Tiny order failed: {e}")
            
            # If it's about product type, try spot order
            if "product" in str(e).lower() or "spot" in str(e).lower():
                print("\n7. Testing as SPOT order instead...")
                # This might reveal if ETH-USD is spot-only
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()