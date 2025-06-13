#!/usr/bin/env python3
"""
Test script to determine the correct RESTClient initialization parameters.
"""

import sys
import traceback

# Test different ways to initialize RESTClient
def test_restclient_init():
    try:
        from coinbase.rest import RESTClient
        print("Successfully imported RESTClient from coinbase.rest")
        
        # Check the constructor signature
        import inspect
        sig = inspect.signature(RESTClient.__init__)
        print(f"RESTClient.__init__ signature: {sig}")
        
        # Test different initialization methods
        print("\n--- Testing different initialization methods ---")
        
        # Test 1: Empty initialization
        try:
            client = RESTClient()
            print("✓ Empty initialization works")
        except Exception as e:
            print(f"✗ Empty initialization failed: {e}")
        
        # Test 2: Base URL only
        try:
            client = RESTClient(base_url="https://api.coinbase.com")
            print("✓ Base URL initialization works")
        except Exception as e:
            print(f"✗ Base URL initialization failed: {e}")
            
        # Test 3: Legacy parameters
        try:
            client = RESTClient(
                api_key="test_key",
                api_secret="test_secret",
                passphrase="test_passphrase"
            )
            print("✓ Legacy parameters work")
        except Exception as e:
            print(f"✗ Legacy parameters failed: {e}")
            
        # Test 4: CDP parameters - test different variations
        try:
            client = RESTClient(
                api_key_name="organizations/test/apiKeys/test",
                private_key="test_key"
            )
            print("✓ CDP parameters (api_key_name, private_key) work")
        except Exception as e:
            print(f"✗ CDP parameters (api_key_name, private_key) failed: {e}")
            
        # Test 5: Try common CDP parameter names
        try:
            client = RESTClient(
                key_name="organizations/test/apiKeys/test",
                private_key="test_key"
            )
            print("✓ CDP parameters (key_name, private_key) work")
        except Exception as e:
            print(f"✗ CDP parameters (key_name, private_key) failed: {e}")
            
        # Test 6: Try other CDP parameter names
        try:
            client = RESTClient(
                cdp_api_key_name="organizations/test/apiKeys/test",
                cdp_private_key="test_key"
            )
            print("✓ CDP parameters (cdp_api_key_name, cdp_private_key) work")
        except Exception as e:
            print(f"✗ CDP parameters (cdp_api_key_name, cdp_private_key) failed: {e}")
            
    except ImportError as e:
        print(f"Failed to import RESTClient: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

def check_environment_variables():
    """Check what environment variables the SDK expects"""
    import os
    
    print("\n--- Environment variable checking ---")
    
    # Common Coinbase environment variables
    env_vars = [
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET", 
        "COINBASE_PASSPHRASE",
        "CDP_API_KEY_NAME",
        "CDP_PRIVATE_KEY",
        "COINBASE_CDP_API_KEY_NAME",
        "COINBASE_CDP_PRIVATE_KEY",
        "CB_API_KEY",
        "CB_API_SECRET",
        "CB_PASSPHRASE"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        status = "✓ SET" if value else "✗ NOT SET"
        print(f"{status}: {var}")

if __name__ == "__main__":
    print("Testing coinbase-advanced-py RESTClient initialization\n")
    test_restclient_init()
    check_environment_variables()