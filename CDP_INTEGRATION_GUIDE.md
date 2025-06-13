# Coinbase CDP API Key Integration Guide

This guide explains how to use the new Coinbase Cloud Development Platform (CDP) API keys with the trading bot. CDP keys are the newer format and are recommended for new applications.

## Overview

The bot now supports both legacy Coinbase Advanced Trade API keys and the newer CDP API keys:

- **Legacy Keys**: Traditional API key, secret, and passphrase format
- **CDP Keys**: Newer format with API key name and private key (PEM format)

## What Was Updated

### 1. Environment Configuration (`.env.example`)
- Added CDP_API_KEY_NAME and CDP_PRIVATE_KEY variables
- Added instructions for extracting keys from CDP JSON files
- Clear documentation about using either legacy OR CDP keys (not both)

### 2. Configuration System (`bot/config.py`)
- Added CDP API key settings to ExchangeSettings class
- Added validation to prevent using both legacy and CDP keys simultaneously
- Added private key format validation (PEM format)
- Updated export functions to handle CDP keys securely

### 3. Exchange Client (`bot/exchange/coinbase.py`)
- Auto-detection of authentication method (legacy vs CDP vs none)
- Support for both authentication types in the same client
- Updated connection status reporting to show authentication method
- Maintains backward compatibility with existing legacy setups

### 4. Helper Tools
- **`scripts/extract_cdp_keys.py`**: Extracts CDP credentials from JSON files
- **`scripts/test_cdp_simple.py`**: Tests CDP integration functionality

## Setting Up CDP API Keys

### Step 1: Download Your CDP API Key File

1. Go to the [Coinbase Developer Platform](https://portal.cdp.coinbase.com/)
2. Create or select your project
3. Generate a new API key
4. Download the JSON file (it will look like this):

```json
{
   "name": "organizations/your-org-id/apiKeys/your-key-id",
   "privateKey": "-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----\n"
}
```

### Step 2: Extract Credentials

Use the helper script to extract the credentials:

```bash
python3 scripts/extract_cdp_keys.py /path/to/your/cdp-api-key.json
```

This will output:
```
# CDP API Keys - Copy these to your .env file
#============================================================

CDP_API_KEY_NAME=organizations/your-org-id/apiKeys/your-key-id
CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----\n"

# Additional Information:
# Organization ID: your-org-id
# Key ID: your-key-id
# Authentication Method: CDP
```

### Step 3: Update Your .env File

1. Copy the output from step 2 to your `.env` file
2. **Important**: Comment out or remove any existing legacy credentials:

```bash
# Legacy credentials (comment these out when using CDP)
# COINBASE_API_KEY=your_old_api_key
# COINBASE_API_SECRET=your_old_api_secret
# COINBASE_PASSPHRASE=your_old_passphrase

# CDP credentials (use these instead)
CDP_API_KEY_NAME=organizations/your-org-id/apiKeys/your-key-id
CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----\n"
```

### Step 4: Test Your Setup

Run the test script to verify everything works:

```bash
python3 scripts/test_cdp_simple.py
```

## Authentication Method Detection

The bot automatically detects which authentication method to use:

1. **CDP Keys**: If CDP_API_KEY_NAME and CDP_PRIVATE_KEY are provided
2. **Legacy Keys**: If COINBASE_API_KEY, COINBASE_API_SECRET, and COINBASE_PASSPHRASE are provided
3. **No Authentication**: If no credentials are provided (dry-run mode only)

The bot will NOT allow both legacy and CDP credentials to be set simultaneously.

## Security Considerations

### CDP Key Security
- CDP private keys are in PEM format and contain sensitive cryptographic material
- Never commit your `.env` file or CDP JSON file to version control
- Store CDP JSON files securely and delete them after extracting credentials
- Use environment variables in production instead of files

### Key Rotation
- CDP keys can be rotated through the Coinbase Developer Platform
- Legacy keys are rotated through the regular Coinbase interface
- Update your `.env` file when rotating keys

## Troubleshooting

### Common Issues

**1. "Cannot use both legacy and CDP credentials" error**
- Solution: Use only one authentication method. Comment out the unused credentials.

**2. "CDP private key must be in PEM format" error**
- Solution: Ensure the private key starts with `-----BEGIN EC PRIVATE KEY-----`
- Use the extraction script to get the correct format

**3. "Missing CDP credentials" error in live trading**
- Solution: Ensure both CDP_API_KEY_NAME and CDP_PRIVATE_KEY are set
- Check that the credentials are not empty strings

**4. Connection fails with CDP keys**
- Solution: Verify your CDP API key has the correct permissions
- Check that you're using the sandbox setting correctly
- Ensure the API key is active in the Coinbase Developer Platform

### Checking Your Setup

You can check your authentication method by looking at the bot logs:
```
INFO - Initialized CoinbaseClient (auth: cdp, sandbox: true, futures: true, account_type: CFM)
```

Or programmatically:
```python
from bot.exchange.coinbase import CoinbaseClient
client = CoinbaseClient()
status = client.get_connection_status()
print(f"Authentication method: {status['auth_method']}")
print(f"Has credentials: {status['has_credentials']}")
```

## Migration from Legacy to CDP

If you're currently using legacy API keys and want to switch to CDP:

1. **Keep your legacy setup working** - don't change anything yet
2. **Get CDP credentials** following the setup steps above
3. **Test with CDP** in a separate environment if possible
4. **Switch over** by updating your `.env` file:
   - Comment out legacy credentials
   - Add CDP credentials
5. **Verify** the bot connects successfully

## Benefits of CDP Keys

- **Improved Security**: Private key-based authentication vs shared secrets
- **Better Integration**: Designed for modern cloud applications
- **Future-Proof**: Recommended by Coinbase for new applications
- **Enhanced Permissions**: More granular permission control

## Support

If you encounter issues with CDP integration:

1. Check the troubleshooting section above
2. Run the test script: `python3 scripts/test_cdp_simple.py`
3. Check the bot logs for authentication method and errors
4. Verify your CDP API key permissions and status

## File Reference

### Updated Files
- `/Users/angel/Documents/Projects/cursorprod/.env.example` - Environment template
- `/Users/angel/Documents/Projects/cursorprod/bot/config.py` - Configuration settings
- `/Users/angel/Documents/Projects/cursorprod/bot/exchange/coinbase.py` - Exchange client

### New Files  
- `/Users/angel/Documents/Projects/cursorprod/scripts/extract_cdp_keys.py` - Key extraction helper
- `/Users/angel/Documents/Projects/cursorprod/scripts/test_cdp_simple.py` - Integration tests
- `/Users/angel/Documents/Projects/cursorprod/CDP_INTEGRATION_GUIDE.md` - This guide