#!/usr/bin/env python3
"""
List all available products and identify which are futures.
"""

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()


def main():
    """List products."""
    # Get CDP credentials
    cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
    cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')
    
    print("AVAILABLE PRODUCTS ANALYSIS")
    print("=" * 60)
    
    # Initialize client
    client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)
    
    try:
        # Get all products
        print("\nFetching all products...")
        products = client.get_products()
        
        spot_products = []
        futures_products = []
        perp_products = []
        
        # Analyze each product
        for product in products.products:
            product_id = product.product_id
            product_type = getattr(product, 'product_type', 'UNKNOWN')
            
            # Check for futures indicators
            is_futures = False
            if hasattr(product, 'product_type'):
                if 'FUTURE' in product_type or 'PERP' in product_type:
                    is_futures = True
            
            # Check by naming convention
            if '-PERP' in product_id or 'PERP-' in product_id:
                perp_products.append(product_id)
            elif is_futures:
                futures_products.append(product_id)
            else:
                spot_products.append(product_id)
        
        print(f"\nFound {len(products.products)} total products:")
        print(f"  Spot products: {len(spot_products)}")
        print(f"  Futures products: {len(futures_products)}")
        print(f"  Perpetual products: {len(perp_products)}")
        
        # Show ETH products specifically
        print("\nETH-RELATED PRODUCTS:")
        print("-" * 40)
        
        eth_products = [p for p in products.products if 'ETH' in p.product_id]
        for product in eth_products[:10]:  # Show first 10
            product_type = getattr(product, 'product_type', 'SPOT')
            status = getattr(product, 'status', 'UNKNOWN')
            print(f"{product.product_id}: Type={product_type}, Status={status}")
        
        # Look for futures-specific products
        print("\nFUTURES/PERP PRODUCTS:")
        print("-" * 40)
        
        futures_found = False
        for product in products.products:
            product_id = product.product_id
            product_type = getattr(product, 'product_type', 'UNKNOWN')
            
            if 'FUTURE' in product_type or 'PERP' in product_id or 'INTX' in product_id:
                futures_found = True
                status = getattr(product, 'status', 'UNKNOWN')
                print(f"{product_id}: Type={product_type}, Status={status}")
        
        if not futures_found:
            print("No futures or perpetual products found.")
        
        # Check account capabilities
        print("\n\nACCOUNT CAPABILITIES:")
        print("-" * 40)
        
        # Try to list futures positions (will fail if no access)
        try:
            positions = client.list_futures_positions()
            print("✅ Futures positions API: Accessible")
        except Exception as e:
            print(f"❌ Futures positions API: {e}")
        
        # Try to get perpetuals info
        try:
            # This might not exist in the API
            print("Checking for perpetuals access...")
            # Different APIs might have different endpoints
        except:
            pass
        
        print("\n💡 CONCLUSION:")
        print("Your account appears to only have access to SPOT trading.")
        print("ETH-USD is a spot product, not a futures product.")
        print("To trade with leverage, you would need:")
        print("1. Access to perpetual futures (ETH-PERP or similar)")
        print("2. Proper permissions on your API key")
        print("3. A different account type that supports margin/futures")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()