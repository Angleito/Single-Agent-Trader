# Bluefin Private Key Auto-Conversion Fix

## Problem
Your VPS deployment was failing because the bot detected a mnemonic phrase but wasn't automatically converting it to the required hex format, despite having created a converter utility for this exact purpose.

## Solution
The `_convert_sui_private_key_legacy` method in `bot/config.py` was updated to:

1. **Automatically convert mnemonic phrases** to hex format using the existing `sui_key_converter` utility
2. **Automatically convert bech32 format** (suiprivkey...) to hex format
3. **Provide better error messages** with fallback instructions if conversion fails

## What Changed

### bot/config.py
- Added imports for `mnemonic_to_hex` and `bech32_to_hex` from the converter utility
- Modified the validation logic to attempt automatic conversion before raising errors
- Added success/failure messages to track conversion process

### New Scripts
1. **scripts/fix-bluefin-key-conversion.sh** - Helps migrate your VPS configuration to use either:
   - FP configuration (recommended) with auto-conversion support
   - Legacy configuration with the auto-conversion fix

2. **scripts/test-sui-converter.py** - Test utility to verify the converter is working properly

## How to Deploy the Fix on Your VPS

### Option 1: Quick Fix (Keep Current Config)
```bash
# SSH to your VPS
ssh your-vps

# Pull the latest changes
cd /path/to/your/bot
git pull

# Restart containers
docker-compose down
docker-compose up -d

# The bot should now automatically convert your mnemonic phrase!
```

### Option 2: Migrate to FP Configuration (Recommended)
```bash
# SSH to your VPS
ssh your-vps

# Pull the latest changes
cd /path/to/your/bot
git pull

# Run the migration script
./scripts/fix-bluefin-key-conversion.sh
# Choose option 1 to migrate to FP configuration

# Restart containers
docker-compose down
docker-compose up -d
```

## Verification

After deploying, check the logs for these messages:
```
🔄 Mnemonic phrase detected, attempting automatic conversion...
✅ Successfully converted mnemonic to hex format
```

If you see these, the conversion worked! The bot will now use the converted hex key automatically.

## Benefits of FP Configuration

If you migrate to FP configuration (Option 2), you'll get:
- Automatic key format conversion
- Better validation and error messages
- Enhanced security with opaque types (keys are automatically masked in logs)
- More flexible configuration options

## Troubleshooting

If conversion still fails:
1. Check that your mnemonic phrase is valid (12 or 24 words)
2. Ensure all words are lowercase and correctly spelled
3. Try the test script: `python scripts/test-sui-converter.py`
4. As a last resort, manually convert using Sui CLI and update your .env

## Security Note

The converter runs locally in your container - your private keys never leave your server and are automatically masked in all logs for security.
