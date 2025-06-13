#!/usr/bin/env python3
"""
Direct comparison test to isolate the JWT generation issue.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from bot.config import settings

try:
    from coinbase import jwt_generator
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

def test_direct_comparison():
    """Test JWT generation both ways."""
    if not SDK_AVAILABLE:
        print("❌ SDK not available")
        return
    
    # Method 1: Direct extraction (working)
    print("Method 1: Direct extraction from settings")
    cdp_api_key_obj = getattr(settings.exchange, 'cdp_api_key_name', None)
    cdp_private_key_obj = getattr(settings.exchange, 'cdp_private_key', None)
    
    cdp_api_key_direct = cdp_api_key_obj.get_secret_value()
    cdp_private_key_direct = cdp_private_key_obj.get_secret_value()
    
    print(f"API key: {cdp_api_key_direct[:50]}... (len: {len(cdp_api_key_direct)})")
    print(f"Private key starts: {cdp_private_key_direct[:30]}... (len: {len(cdp_private_key_direct)})")
    
    jwt_direct = jwt_generator.build_ws_jwt(cdp_api_key_direct, cdp_private_key_direct)
    print(f"Result: {jwt_direct is not None} (len: {len(jwt_direct) if jwt_direct else 'None'})")
    
    print("\n" + "="*50 + "\n")
    
    # Method 2: Using the provider's method (failing)
    print("Method 2: Using MarketDataProvider method")
    from bot.data.market import MarketDataProvider
    
    provider = MarketDataProvider()
    
    # Extract the same way as the provider
    cdp_api_key_provider = cdp_api_key_obj.get_secret_value() if hasattr(cdp_api_key_obj, 'get_secret_value') else str(cdp_api_key_obj)
    cdp_private_key_provider = cdp_private_key_obj.get_secret_value() if hasattr(cdp_private_key_obj, 'get_secret_value') else str(cdp_private_key_obj)
    
    print(f"API key: {cdp_api_key_provider[:50]}... (len: {len(cdp_api_key_provider)})")
    print(f"Private key starts: {cdp_private_key_provider[:30]}... (len: {len(cdp_private_key_provider)})")
    
    # Test if they're identical
    print(f"API keys identical: {cdp_api_key_direct == cdp_api_key_provider}")
    print(f"Private keys identical: {cdp_private_key_direct == cdp_private_key_provider}")
    
    jwt_provider = jwt_generator.build_ws_jwt(cdp_api_key_provider, cdp_private_key_provider)
    print(f"Result: {jwt_provider is not None} (len: {len(jwt_provider) if jwt_provider else 'None'})")
    
    print("\n" + "="*50 + "\n")
    
    # Method 3: Using provider's internal method
    print("Method 3: Using provider's _build_websocket_jwt method")
    jwt_internal = provider._build_websocket_jwt()
    print(f"Result: {jwt_internal is not None} (len: {len(jwt_internal) if jwt_internal else 'None'})")
    
    # Show raw bytes comparison if they differ
    if cdp_api_key_direct == cdp_api_key_provider and cdp_private_key_direct == cdp_private_key_provider:
        print("✅ All credential extractions are identical")
        if jwt_direct != jwt_provider:
            print("❌ But JWT results differ - this shouldn't happen!")
        else:
            print("✅ JWT results are identical")

if __name__ == "__main__":
    test_direct_comparison()