#!/usr/bin/env python3
"""
Test script to determine what environment variables the SDK uses for CDP authentication.
"""

import os
import sys
import traceback

def test_with_env_vars():
    """Test RESTClient with various environment variables set"""
    
    print("--- Testing with environment variables ---")
    
    # Test 1: Standard Coinbase env vars
    print("\nTest 1: Standard Coinbase environment variables")
    os.environ["COINBASE_API_KEY"] = "test_key"
    os.environ["COINBASE_API_SECRET"] = "test_secret"
    
    try:
        from coinbase.rest import RESTClient
        client = RESTClient()
        print("✓ RESTClient works with COINBASE_API_KEY/SECRET")
        # Try to access auth info
        print(f"Client has api_key: {hasattr(client, 'api_key')}")
        print(f"Client has api_secret: {hasattr(client, 'api_secret')}")
    except Exception as e:
        print(f"✗ Failed with COINBASE_API_KEY/SECRET: {e}")
    
    # Clean up
    del os.environ["COINBASE_API_KEY"]
    del os.environ["COINBASE_API_SECRET"]
    
    # Test 2: CDP environment variables
    print("\nTest 2: CDP environment variables")
    test_private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEILJD7T8VJMW036QfbbuuF49lAwYsHvtlL56tiU0RRg2toAoGCCqGSM49
AwEHoUQDQgAEM4vnRtudsM0bDWv9b7MFhG7FBoYq5LU+/YPK9hTpY25C6i1odJa4
Yus+Ym+vCLL/CiBkERaBaixT9HZjRyi40g==
-----END EC PRIVATE KEY-----"""
    
    # Try different CDP env var patterns
    cdp_env_patterns = [
        ("CDP_API_KEY_NAME", "CDP_PRIVATE_KEY"),
        ("COINBASE_CDP_API_KEY_NAME", "COINBASE_CDP_PRIVATE_KEY"),
        ("API_KEY_NAME", "PRIVATE_KEY"),
        ("COINBASE_API_KEY_NAME", "COINBASE_PRIVATE_KEY"),
    ]
    
    for key_var, private_var in cdp_env_patterns:
        print(f"\nTesting {key_var} + {private_var}")
        
        os.environ[key_var] = "organizations/test-org/apiKeys/test-key"
        os.environ[private_var] = test_private_key
        
        try:
            from coinbase.rest import RESTClient
            client = RESTClient()
            print(f"✓ RESTClient works with {key_var}/{private_var}")
            # Check what attributes the client has
            attrs = [attr for attr in dir(client) if not attr.startswith('_')]
            auth_attrs = [attr for attr in attrs if 'key' in attr.lower() or 'auth' in attr.lower()]
            print(f"Auth-related attributes: {auth_attrs}")
        except Exception as e:
            print(f"✗ Failed with {key_var}/{private_var}: {e}")
        
        # Clean up
        if key_var in os.environ:
            del os.environ[key_var]
        if private_var in os.environ:
            del os.environ[private_var]

def test_key_file_parameter():
    """Test the key_file parameter for CDP authentication"""
    print("\n--- Testing key_file parameter ---")
    
    # Create a temporary key file
    test_private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEILJD7T8VJMW036QfbbuuF49lAwYsHvtlL56tiU0RRg2toAoGCCqGSM49
AwEHoUQDQgAEM4vnRtudsM0bDWv9b7MFhG7FBoYq5LU+/YPK9hTpY25C6i1odJa4
Yus+Ym+vCLL/CiBkERaBaixT9HZjRyi40g==
-----END EC PRIVATE KEY-----"""
    
    key_file_path = "/tmp/test_private_key.pem"
    
    try:
        with open(key_file_path, 'w') as f:
            f.write(test_private_key)
        
        from coinbase.rest import RESTClient
        
        # Test with key_file parameter
        client = RESTClient(key_file=key_file_path)
        print("✓ RESTClient works with key_file parameter")
        
        # Test with key_file as string content
        client2 = RESTClient(key_file=test_private_key)
        print("✓ RESTClient works with key_file as string content")
        
    except Exception as e:
        print(f"✗ key_file parameter failed: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        try:
            os.remove(key_file_path)
        except:
            pass

if __name__ == "__main__":
    print("Testing coinbase-advanced-py environment variables and key_file parameter\n")
    test_with_env_vars()
    test_key_file_parameter()