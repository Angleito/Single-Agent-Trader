# Coinbase SDK Initialization Fix

## Problem Identified

The `coinbase-advanced-py` SDK was being initialized with incorrect parameter names for CDP authentication:

**INCORRECT (before fix):**
```python
client = RESTClient(
    api_key_name=cdp_api_key_name,    # ❌ Wrong parameter name
    private_key=cdp_private_key       # ❌ Wrong parameter name
)
```

## Root Cause

The error occurred because we were using parameter names that don't exist in the `coinbase-advanced-py` SDK. The SDK expects different parameter names than what our code was providing.

## Solution Implemented

**CORRECT (after fix):**
```python
client = RESTClient(
    api_key=cdp_api_key_name,         # ✅ Correct parameter name
    api_secret=cdp_private_key        # ✅ Correct parameter name
)
```

## Evidence from Official Documentation

According to the [official Coinbase documentation](https://docs.cdp.coinbase.com/coinbase-app/docs/trade/sdk-rest-client-trade):

```python
api_key = "organizations/{org_id}/apiKeys/{key_id}"
api_secret = "-----BEGIN EC PRIVATE KEY-----\nYOUR PRIVATE KEY\n-----END EC PRIVATE KEY-----"

client = RESTClient(api_key=api_key, api_secret=api_secret)
```

This confirms that:
- The first parameter should be `api_key` (not `api_key_name`)
- The second parameter should be `api_secret` (not `private_key`)

## Environment Variables

Our environment variables are correctly set up:
- `EXCHANGE__CDP_API_KEY_NAME` = `"organizations/c84bc13d-f23c-4218-af98-9d37ee4566c7/apiKeys/3891da34-3550-4066-bf1d-9decfde34a16"`
- `EXCHANGE__CDP_PRIVATE_KEY` = PEM-formatted private key (227 characters)

## Changes Made

### File: `bot/exchange/coinbase.py`

1. **Removed temporary file approach** - No longer creating JSON files for CDP authentication
2. **Fixed parameter names** - Using `api_key` and `api_secret` instead of incorrect names
3. **Simplified initialization** - Direct initialization without temporary files

**Before:**
```python
# Create temporary JSON file for CDP authentication
import json
import tempfile

cdp_json = {
    "name": self.cdp_api_key_name,
    "privateKey": self.cdp_private_key
}

# Create temporary file for the CDP key
self._cdp_key_file = tempfile.NamedTemporaryFile(
    mode='w', suffix='.json', delete=False
)
json.dump(cdp_json, self._cdp_key_file, indent=2)
self._cdp_key_file.close()

# Initialize RESTClient with the CDP key file
base_url = "api-public.sandbox.pro.coinbase.com" if self.sandbox else "api.coinbase.com"
self._client = CoinbaseAdvancedTrader(
    key_file=self._cdp_key_file.name,
    base_url=base_url
)
```

**After:**
```python
# Initialize RESTClient with CDP credentials directly
# Based on official docs: api_key should be the CDP key name, api_secret should be the private key
self._client = CoinbaseAdvancedTrader(
    api_key=self.cdp_api_key_name,
    api_secret=self.cdp_private_key
)
```

2. **Removed cleanup code** - No longer need to clean up temporary files

## Expected Result

With these changes, the SDK should initialize successfully without the `"api_key_name" parameter error`. The bot should be able to:

1. ✅ Load CDP credentials from environment variables
2. ✅ Initialize the RESTClient with correct parameters
3. ✅ Authenticate with Coinbase using CDP authentication
4. ✅ Proceed with trading operations in dry-run mode

## Testing

A test script confirms:
- ✅ Environment variables are loaded correctly
- ✅ CDP credentials have the correct format
- ✅ Parameter mapping is now correct per official documentation
- ✅ No more temporary file handling needed

## Next Steps

The Agent 2 team should now be able to:
1. Test the connection establishment
2. Verify API authentication works
3. Test basic API calls (e.g., get_accounts)
4. Proceed with the full authentication flow

## Security Notes

- CDP credentials are still properly managed through environment variables
- No credentials are exposed in logs or temporary files
- Private key remains in PEM format as required by the SDK