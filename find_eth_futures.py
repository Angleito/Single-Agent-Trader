#!/usr/bin/env python3
"""
Find ETH futures contracts with different search methods.
"""

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()


def main():
    """Find ETH futures."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("SEARCHING FOR ETH FUTURES CONTRACTS")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Method 1: Check all products for futures indicators
        print("\n1. Searching all products for futures indicators...")
        products = client.get_products()
        
        futures_candidates = []
        
        for product in products.products:
            product_id = product.product_id
            
            # Look for various futures indicators
            if any(indicator in product_id.upper() for indicator in ['FUTURES', 'FUT', 'PERP', 'SWAP', '2024', '2025', 'DEC', 'JAN', 'MAR', 'JUN', 'SEP']):
                if 'ETH' in product_id:
                    futures_candidates.append(product)
            
            # Also check if it has ETH and a date
            if 'ETH' in product_id and any(year in product_id for year in ['24', '25', '2024', '2025']):
                if product not in futures_candidates:
                    futures_candidates.append(product)
        
        if futures_candidates:
            print(f"\nFound {len(futures_candidates)} potential ETH futures:")
            for p in futures_candidates:
                print(f"  {p.product_id} - Status: {getattr(p, 'status', 'unknown')}")
        
        # Method 2: Try common futures naming conventions
        print("\n2. Testing common futures product IDs...")
        test_ids = [
            "ETH-USD-FUTURES",
            "ETH-FUTURES-USD", 
            "ETH-USD-PERP",
            "ETH-PERP",
            "ETHUSDT-PERP",
            "ETH-USD-SWAP",
            "ETH-25JAN25",  # Dated futures
            "ETH-31JAN25",
            "ETH-28MAR25",
            "ETH-27JUN25",
            "ETH-USD-250131",  # Alternative date format
            "ETH-USD-20250131",
            "BIT-ETH-USD",  # Sometimes prefixed
            "CFM-ETH-USD",
        ]
        
        for test_id in test_ids:
            try:
                product = client.get_product(test_id)
                print(f"  ✅ Found: {test_id}")
                # Print details
                if hasattr(product, '__dict__'):
                    for key, value in product.__dict__.items():
                        if key not in ['_client']:
                            print(f"     {key}: {value}")
            except Exception:
                # Product doesn't exist
                pass
        
        # Method 3: Check futures-specific endpoints
        print("\n3. Checking futures-specific data...")
        
        # Get futures positions to see product format
        try:
            positions = client.list_futures_positions()
            print(f"\nFutures positions response:")
            if hasattr(positions, 'positions') and positions.positions:
                for pos in positions.positions:
                    print(f"  Position product_id: {pos.product_id}")
            else:
                print("  No open futures positions to examine")
        except Exception as e:
            print(f"  Error getting futures positions: {e}")
        
        # Method 4: Check futures balance summary for clues
        print("\n4. Checking futures balance summary...")
        try:
            fcm = client.get_futures_balance_summary()
            bs = fcm.balance_summary
            
            # Sometimes the response includes supported products
            if hasattr(bs, 'supported_products'):
                print("  Supported products:", bs.supported_products)
            
            # Check if there are any other attributes that might help
            attrs = [attr for attr in dir(bs) if not attr.startswith('_')]
            print(f"  Balance summary attributes: {attrs}")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # Method 5: Try to place a futures order with ETH-USD and see the error
        print("\n5. Testing order placement to see specific error...")
        try:
            import uuid
            result = client.create_order(
                client_order_id=f"test-{uuid.uuid4().hex[:8]}",
                product_id="ETH-USD",
                side="BUY",
                order_configuration={
                    "market_market_ioc": {"base_size": "0.1"}
                },
                leverage="5",
                margin_type="CROSS"  # Try adding margin type
            )
            print(f"  Order result: {result}")
        except Exception as e:
            print(f"  Order error (this might give us clues): {e}")
        
        # Method 6: Look for contract specifications
        print("\n6. Checking if ETH-USD supports margin/leverage...")
        try:
            product = client.get_product("ETH-USD")
            # Check various attributes
            check_attrs = ['margin_enabled', 'futures_enabled', 'leverage_enabled', 
                          'contract_size', 'contract_display_name', 'future_product_details']
            
            for attr in check_attrs:
                if hasattr(product, attr):
                    print(f"  {attr}: {getattr(product, attr)}")
        except Exception as e:
            print(f"  Error: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()