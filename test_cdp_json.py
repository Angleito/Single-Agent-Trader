#!/usr/bin/env python3
"""
Test CDP authentication using correct JSON format
"""

import os
import sys
import json
import tempfile

def test_cdp_json_authentication():
    """Test CDP authentication using the correct JSON format"""
    print("=== Testing CDP Authentication via JSON key_file ===\n")
    
    sys.path.insert(0, '/app')
    from bot.config import settings
    
    if not (settings.exchange.cdp_api_key_name and settings.exchange.cdp_private_key):
        print("✗ No CDP credentials available in settings")
        return
    
    key_name = settings.exchange.cdp_api_key_name.get_secret_value()
    private_key = settings.exchange.cdp_private_key.get_secret_value()
    
    print(f"CDP API Key Name: {key_name}")
    print(f"Private Key Length: {len(private_key)} chars")
    
    # Create the correct JSON format
    cdp_json = {
        "name": key_name,
        "privateKey": private_key
    }
    
    print(f"\nCreated JSON structure:")
    print(f"  name: {cdp_json['name']}")
    print(f"  privateKey: {cdp_json['privateKey'][:50]}...")
    
    # Clear any conflicting env vars
    for var in ["COINBASE_API_KEY", "COINBASE_API_SECRET", "CDP_API_KEY_NAME", "CDP_PRIVATE_KEY"]:
        if var in os.environ:
            del os.environ[var]
    
    try:
        from coinbase.rest import RESTClient
        
        # Test with JSON file
        print("\n--- Test: JSON file as key_file ---")
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(cdp_json, tmp_file, indent=2)
                json_file_path = tmp_file.name
            
            print(f"Created temporary JSON file: {json_file_path}")
            
            # Read and verify the file content
            with open(json_file_path, 'r') as f:
                file_content = f.read()
            print(f"File content preview: {file_content[:200]}...")
            
            client = RESTClient(key_file=json_file_path)
            print("✓ RESTClient created with JSON key_file")
            
            # Try API call
            accounts = client.get_accounts()
            print("✓ API call successful!")
            print(f"  Found {len(accounts.accounts)} accounts")
            
            # Show some account info (safely)
            for i, account in enumerate(accounts.accounts[:3]):  # Show first 3 accounts
                print(f"  Account {i+1}: {account.currency} - {account.name if hasattr(account, 'name') else 'N/A'}")
            
        except Exception as e:
            print(f"✗ JSON key_file failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up temp file
            try:
                os.unlink(json_file_path)
            except:
                pass
                
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cdp_json_authentication()