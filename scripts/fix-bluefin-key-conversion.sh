#!/bin/bash
# Fix Bluefin private key conversion on VPS

echo "🔧 Bluefin Private Key Conversion Fix Script"
echo "==========================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📝 Current Configuration:"
echo "------------------------"

# Check which environment variables are set
if [ -n "$EXCHANGE__BLUEFIN_PRIVATE_KEY" ]; then
    echo "✓ Found EXCHANGE__BLUEFIN_PRIVATE_KEY (Legacy format)"
fi

if [ -n "$BLUEFIN_PRIVATE_KEY" ]; then
    echo "✓ Found BLUEFIN_PRIVATE_KEY (FP format)"
fi

# Check .env file
if [ -f ".env" ]; then
    echo ""
    echo "📄 Checking .env file..."

    if grep -q "^EXCHANGE__BLUEFIN_PRIVATE_KEY=" .env; then
        echo "✓ Found EXCHANGE__BLUEFIN_PRIVATE_KEY in .env (Legacy format)"
    fi

    if grep -q "^BLUEFIN_PRIVATE_KEY=" .env; then
        echo "✓ Found BLUEFIN_PRIVATE_KEY in .env (FP format)"
    fi
fi

echo ""
echo "🔄 Migration Options:"
echo "--------------------"
echo "1. Use Functional Programming config (Recommended - includes auto-conversion)"
echo "2. Keep Legacy config but enable auto-conversion"
echo ""
read -p "Choose option (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Migrating to FP Configuration..."
        echo ""

        # Create backup
        if [ -f ".env" ]; then
            cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
            echo "✓ Created backup of .env"
        fi

        # Update .env to use FP format
        if grep -q "^EXCHANGE__BLUEFIN_PRIVATE_KEY=" .env; then
            # Extract the current key value
            KEY_VALUE=$(grep "^EXCHANGE__BLUEFIN_PRIVATE_KEY=" .env | cut -d'=' -f2-)

            # Add FP format variable
            echo "" >> .env
            echo "# Functional Programming Configuration (with auto-conversion)" >> .env
            echo "BLUEFIN_PRIVATE_KEY=$KEY_VALUE" >> .env
            echo "BLUEFIN_NETWORK=mainnet" >> .env
            echo "TRADING_MODE=paper" >> .env
            echo "EXCHANGE_TYPE=bluefin" >> .env
            echo "STRATEGY_TYPE=llm" >> .env
            echo "LOG_LEVEL=INFO" >> .env

            # Comment out legacy variable
            sed -i 's/^EXCHANGE__BLUEFIN_PRIVATE_KEY=/#EXCHANGE__BLUEFIN_PRIVATE_KEY=/' .env
            sed -i 's/^EXCHANGE__EXCHANGE_TYPE=/#EXCHANGE__EXCHANGE_TYPE=/' .env

            echo "✓ Updated .env to use FP configuration"
            echo ""
            echo "📋 Added FP configuration variables:"
            echo "   - BLUEFIN_PRIVATE_KEY (with auto-conversion support)"
            echo "   - BLUEFIN_NETWORK=mainnet"
            echo "   - TRADING_MODE=paper"
            echo "   - EXCHANGE_TYPE=bluefin"
            echo "   - STRATEGY_TYPE=llm"
            echo ""
        fi
        ;;

    2)
        echo ""
        echo "🔧 Keeping Legacy Configuration..."
        echo ""
        echo "The auto-conversion has been added to the legacy config path."
        echo "Your bot should now automatically convert mnemonic phrases to hex format."
        echo ""
        ;;

    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "🧪 Testing Configuration..."
echo "--------------------------"

# Test the configuration
docker-compose run --rm ai-trading-bot python -c "
import os
import sys

# Test FP configuration
try:
    from bot.fp.types.config import build_exchange_config_from_env
    result = build_exchange_config_from_env()
    if hasattr(result, 'is_success') and result.is_success():
        print('✅ FP Configuration: Valid')
    else:
        error_msg = result.failure() if hasattr(result, 'failure') else 'Unknown error'
        print(f'❌ FP Configuration: {error_msg}')
except Exception as e:
    print(f'⚠️  FP Configuration: Not available ({str(e)})')

# Test legacy configuration
try:
    from bot.config import settings
    if settings.exchange.bluefin_private_key:
        print('✅ Legacy Configuration: Valid')
    else:
        print('❌ Legacy Configuration: Missing private key')
except Exception as e:
    print(f'❌ Legacy Configuration: {str(e)}')
"

echo ""
echo "📝 Next Steps:"
echo "-------------"
echo "1. Restart your containers:"
echo "   docker-compose down"
echo "   docker-compose up -d"
echo ""
echo "2. Check the logs:"
echo "   docker-compose logs -f ai-trading-bot"
echo ""
echo "3. Look for these success messages:"
echo "   - '🔄 Mnemonic phrase detected, attempting automatic conversion...'"
echo "   - '✅ Successfully converted mnemonic to hex format'"
echo ""

echo "✅ Script completed!"
