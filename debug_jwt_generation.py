#!/usr/bin/env python3
"""
Debug script to test JWT generation with different approaches.
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
    print("❌ Coinbase SDK not available")

def test_jwt_generation():
    """Test JWT generation with detailed logging."""
    print("=" * 60)
    print("JWT Generation Debug Test")
    print("=" * 60)
    
    if not SDK_AVAILABLE:
        print("❌ Cannot test - SDK not available")
        return
    
    # Get credentials
    cdp_api_key_obj = getattr(settings.exchange, 'cdp_api_key_name', None)
    cdp_private_key_obj = getattr(settings.exchange, 'cdp_private_key', None)
    
    print(f"API key object: {type(cdp_api_key_obj)}")
    print(f"Private key object: {type(cdp_private_key_obj)}")
    
    if not cdp_api_key_obj or not cdp_private_key_obj:
        print("❌ CDP credentials not found")
        return
    
    # Extract values
    cdp_api_key = cdp_api_key_obj.get_secret_value()
    cdp_private_key = cdp_private_key_obj.get_secret_value()
    
    print(f"API key length: {len(cdp_api_key)}")
    print(f"API key format check: {'organizations/' in cdp_api_key}")
    print(f"Private key length: {len(cdp_private_key)}")
    print(f"Private key format check: {'BEGIN EC PRIVATE KEY' in cdp_private_key}")
    
    # Test JWT generation
    try:
        print("\nTesting SDK jwt_generator.build_ws_jwt...")
        jwt_token = jwt_generator.build_ws_jwt(cdp_api_key, cdp_private_key)
        
        if jwt_token:
            print(f"✅ JWT generation successful!")
            print(f"JWT length: {len(jwt_token)}")
            print(f"JWT preview: {jwt_token[:50]}...")
            
            # Validate JWT structure
            parts = jwt_token.split('.')
            if len(parts) == 3:
                print("✅ JWT has correct structure (3 parts)")
                
                # Try to decode header and payload
                import base64
                import json
                
                try:
                    header = json.loads(base64.urlsafe_b64decode(parts[0] + '=='))
                    payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))
                    
                    print(f"JWT header: {json.dumps(header, indent=2)}")
                    print(f"JWT payload: {json.dumps(payload, indent=2)}")
                    
                except Exception as e:
                    print(f"❌ Failed to decode JWT: {e}")
            else:
                print(f"❌ JWT has wrong structure: {len(parts)} parts")
        else:
            print("❌ JWT generation returned None")
            
            # Try to determine why
            print("\nDebugging why JWT generation failed...")
            
            # Test API key format
            if not cdp_api_key.startswith('organizations/'):
                print(f"❌ API key doesn't start with 'organizations/': {cdp_api_key[:30]}...")
            else:
                print("✅ API key format looks correct")
                
            # Test private key format
            if not cdp_private_key.startswith('-----BEGIN'):
                print(f"❌ Private key doesn't start with '-----BEGIN': {cdp_private_key[:30]}...")
            else:
                print("✅ Private key format looks correct")
            
            # Try to load the private key directly
            try:
                from cryptography.hazmat.primitives import serialization
                private_key_bytes = cdp_private_key.encode('utf-8')
                private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
                print("✅ Private key loads successfully with cryptography")
            except Exception as e:
                print(f"❌ Failed to load private key: {e}")
            
    except Exception as e:
        print(f"❌ JWT generation failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_jwt_generation()