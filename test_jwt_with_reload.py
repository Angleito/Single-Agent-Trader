#!/usr/bin/env python3
"""
Test JWT generation with module reload to ensure changes are picked up.
"""

import os
import sys
import importlib

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

# Clear any cached imports
for module_name in list(sys.modules.keys()):
    if module_name.startswith('bot.'):
        del sys.modules[module_name]

from bot.config import settings

try:
    from coinbase import jwt_generator
    SDK_AVAILABLE = True
    print(f"SDK jwt_generator type: {type(jwt_generator)}")
    print(f"SDK jwt_generator.build_ws_jwt: {jwt_generator.build_ws_jwt}")
except ImportError:
    SDK_AVAILABLE = False

def test_provider_method():
    """Test the provider method with debugging."""
    if not SDK_AVAILABLE:
        print("❌ SDK not available")
        return
    
    # Import the market module fresh
    from bot.data.market import MarketDataProvider, jwt_generator as market_jwt_generator
    
    print(f"Market module jwt_generator type: {type(market_jwt_generator)}")
    print(f"Market module jwt_generator.build_ws_jwt: {market_jwt_generator.build_ws_jwt}")
    print(f"Are they the same object? {jwt_generator is market_jwt_generator}")
    
    # Create provider and test
    provider = MarketDataProvider()
    
    # Manually call the method and debug step by step
    print("\nStep-by-step debugging:")
    
    cdp_api_key_obj = getattr(settings.exchange, 'cdp_api_key_name', None)
    cdp_private_key_obj = getattr(settings.exchange, 'cdp_private_key', None)
    
    print(f"1. CDP credentials found: {cdp_api_key_obj is not None and cdp_private_key_obj is not None}")
    
    if cdp_api_key_obj and cdp_private_key_obj:
        cdp_api_key = cdp_api_key_obj.get_secret_value()
        cdp_private_key = cdp_private_key_obj.get_secret_value()
        
        print(f"2. Extracted credentials: API key len={len(cdp_api_key)}, Private key len={len(cdp_private_key)}")
        
        # Test direct call to market module's jwt_generator
        print("3. Testing market module's jwt_generator directly:")
        jwt_result = market_jwt_generator.build_ws_jwt(cdp_api_key, cdp_private_key)
        print(f"   Result: {jwt_result is not None} (len: {len(jwt_result) if jwt_result else 'None'})")
        
        # Test the provider method
        print("4. Testing provider's _build_websocket_jwt method:")
        provider_result = provider._build_websocket_jwt()
        print(f"   Result: {provider_result is not None} (len: {len(provider_result) if provider_result else 'None'})")

if __name__ == "__main__":
    test_provider_method()