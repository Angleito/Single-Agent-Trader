#!/usr/bin/env python3
"""
Test environment variable setting and SDK detection
"""

import os
import sys

def check_environment_variables():
    """Check current environment variables"""
    print("=== Environment Variables Check ===\n")
    
    # Check all relevant env vars
    env_vars = [
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET", 
        "COINBASE_PASSPHRASE",
        "CDP_API_KEY_NAME",
        "CDP_PRIVATE_KEY",
        "EXCHANGE__CDP_API_KEY_NAME",
        "EXCHANGE__CDP_PRIVATE_KEY",
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Show first 50 chars for security
            display_value = value[:50] + "..." if len(value) > 50 else value
            print(f"✓ {var}: {display_value}")
        else:
            print(f"✗ {var}: NOT SET")

def test_sdk_authentication():
    """Test SDK authentication with current env vars"""
    print("\n=== SDK Authentication Test ===\n")
    
    try:
        from coinbase.rest import RESTClient
        
        # Try to create client
        client = RESTClient()
        print("✓ RESTClient created successfully")
        
        # Check if client thinks it's authenticated
        try:
            is_auth = client.is_authenticated()
            print(f"✓ SDK authentication status: {is_auth}")
        except AttributeError:
            print("ℹ SDK doesn't have is_authenticated method")
        
        # Try to make a simple API call
        try:
            accounts = client.get_accounts()
            print("✓ API call successful!")
            print(f"  Found {len(accounts.accounts)} accounts")
        except Exception as e:
            print(f"✗ API call failed: {e}")
            
    except Exception as e:
        print(f"✗ SDK test failed: {e}")

def test_manual_env_setting():
    """Test manually setting env vars and then creating SDK client"""
    print("\n=== Manual Environment Variable Test ===\n")
    
    # Get our internal config values
    sys.path.insert(0, '/app')
    from bot.config import settings
    
    if settings.exchange.cdp_api_key_name and settings.exchange.cdp_private_key:
        key_name = settings.exchange.cdp_api_key_name.get_secret_value()
        private_key = settings.exchange.cdp_private_key.get_secret_value()
        
        print("Setting environment variables manually...")
        os.environ["CDP_API_KEY_NAME"] = key_name
        os.environ["CDP_PRIVATE_KEY"] = private_key
        
        print(f"✓ Set CDP_API_KEY_NAME: {key_name[:50]}...")
        print(f"✓ Set CDP_PRIVATE_KEY: {len(private_key)} chars")
        
        # Now test SDK
        try:
            from coinbase.rest import RESTClient
            
            # Force reimport to pick up new env vars
            import importlib
            import coinbase.rest
            importlib.reload(coinbase.rest)
            
            client = RESTClient()
            print("✓ RESTClient created with manual env vars")
            
            # Try API call
            try:
                accounts = client.get_accounts()
                print("✓ API call successful with manual env vars!")
                print(f"  Found {len(accounts.accounts)} accounts")
            except Exception as e:
                print(f"✗ API call failed with manual env vars: {e}")
                
        except Exception as e:
            print(f"✗ Manual env var test failed: {e}")
    else:
        print("✗ No CDP credentials available in settings")

if __name__ == "__main__":
    check_environment_variables()
    test_sdk_authentication()
    test_manual_env_setting()