#!/usr/bin/env python3
"""
Test CDP authentication using key_file parameter
"""

import os
import sys
import tempfile

def test_keyfile_authentication():
    """Test CDP authentication using the key_file parameter"""
    print("=== Testing CDP Authentication via key_file Parameter ===\n")
    
    sys.path.insert(0, '/app')
    from bot.config import settings
    
    if not (settings.exchange.cdp_api_key_name and settings.exchange.cdp_private_key):
        print("✗ No CDP credentials available in settings")
        return
    
    key_name = settings.exchange.cdp_api_key_name.get_secret_value()
    private_key = settings.exchange.cdp_private_key.get_secret_value()
    
    print(f"CDP API Key Name: {key_name}")
    print(f"Private Key Length: {len(private_key)} chars")
    
    # Clear any existing env vars that might interfere
    for var in ["COINBASE_API_KEY", "COINBASE_API_SECRET", "CDP_API_KEY_NAME", "CDP_PRIVATE_KEY"]:
        if var in os.environ:
            del os.environ[var]
    
    try:
        from coinbase.rest import RESTClient
        
        # Test 1: key_file as string content
        print("\n--- Test 1: key_file as string content ---")
        try:
            client = RESTClient(key_file=private_key)
            print("✓ RESTClient created with key_file as string")
            
            # Try API call
            accounts = client.get_accounts()
            print("✓ API call successful!")
            print(f"  Found {len(accounts.accounts)} accounts")
            
        except Exception as e:
            print(f"✗ key_file as string failed: {e}")
        
        # Test 2: key_file as file path
        print("\n--- Test 2: key_file as file path ---")
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as tmp_file:
                tmp_file.write(private_key)
                key_file_path = tmp_file.name
            
            client = RESTClient(key_file=key_file_path)
            print("✓ RESTClient created with key_file as path")
            
            # Try API call
            accounts = client.get_accounts()
            print("✓ API call successful!")
            print(f"  Found {len(accounts.accounts)} accounts")
            
        except Exception as e:
            print(f"✗ key_file as path failed: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(key_file_path)
            except:
                pass
        
        # Test 3: Environment variables with key_file
        print("\n--- Test 3: Environment variables + key_file ---")
        try:
            os.environ["CDP_API_KEY_NAME"] = key_name
            
            client = RESTClient(key_file=private_key)
            print("✓ RESTClient created with env var + key_file")
            
            # Try API call
            accounts = client.get_accounts()
            print("✓ API call successful!")
            print(f"  Found {len(accounts.accounts)} accounts")
            
        except Exception as e:
            print(f"✗ env var + key_file failed: {e}")
        
        # Test 4: Check what authentication info the client has
        print("\n--- Test 4: Client authentication info ---")
        try:
            client = RESTClient(key_file=private_key)
            
            # Check various attributes
            attrs_to_check = ['api_key', 'api_secret', 'key_file', '_private_key', '_key_file']
            for attr in attrs_to_check:
                if hasattr(client, attr):
                    value = getattr(client, attr)
                    if value:
                        if 'key' in attr.lower() and len(str(value)) > 50:
                            print(f"  {attr}: {str(value)[:50]}... (length: {len(str(value))})")
                        else:
                            print(f"  {attr}: {value}")
                    else:
                        print(f"  {attr}: None/Empty")
                else:
                    print(f"  {attr}: Not found")
                    
        except Exception as e:
            print(f"✗ client info check failed: {e}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_keyfile_authentication()