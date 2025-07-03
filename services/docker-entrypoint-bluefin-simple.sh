#!/bin/sh
# Simplified Bluefin entrypoint for legacy Docker compatibility
# Enhanced with Python path setup and error handling

set -e  # Exit on any error

echo "Starting Bluefin SDK Service..."

# Ensure Python path includes parent directory for bot imports
export PYTHONPATH="/app:${PYTHONPATH:-}"

# Set development-friendly environment variables if not set
export SYSTEM__DRY_RUN="${SYSTEM__DRY_RUN:-true}"
export SYSTEM__ENVIRONMENT="${SYSTEM__ENVIRONMENT:-development}"
export BLUEFIN_PRIVATE_KEY="${BLUEFIN_PRIVATE_KEY:-dummy_key_for_testing}"
export BLUEFIN_NETWORK="${BLUEFIN_NETWORK:-testnet}"

# Validate Python environment
if ! python3 -c "import sys; print('Python path:', sys.path)" > /dev/null 2>&1; then
    echo "ERROR: Python environment validation failed"
    exit 1
fi

# Check if required bot modules can be imported (optional - warn but don't fail)
if ! python3 -c "import bot.utils.secure_logging" > /dev/null 2>&1; then
    echo "WARNING: bot.utils.secure_logging not available - using fallback logging"
fi

echo "Environment: ${SYSTEM__ENVIRONMENT}"
echo "Dry Run Mode: ${SYSTEM__DRY_RUN}"
echo "Bluefin Network: ${BLUEFIN_NETWORK}"
echo "Python Path: ${PYTHONPATH}"

exec python3 -u -O bluefin_sdk_service.py
