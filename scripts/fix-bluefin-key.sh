#!/bin/bash

# Quick fix for Bluefin private key validation error
# This script updates the environment to use a valid dummy hex key

echo "🔧 Fixing Bluefin private key configuration..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file from Bluefin template..."
    cp .env.bluefin .env
else
    echo "✅ .env file found"
fi

# Update the private key to use a valid hex format
echo "🔑 Setting valid dummy private key..."
if grep -q "EXCHANGE__BLUEFIN_PRIVATE_KEY" .env; then
    # Update existing key
    sed -i 's/EXCHANGE__BLUEFIN_PRIVATE_KEY=.*/EXCHANGE__BLUEFIN_PRIVATE_KEY=0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef/' .env
else
    # Add the key if it doesn't exist
    echo "EXCHANGE__BLUEFIN_PRIVATE_KEY=0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" >> .env
fi

# Ensure other critical settings are correct
echo "⚙️  Ensuring safe development settings..."

# Set dry run mode
if grep -q "SYSTEM__DRY_RUN" .env; then
    sed -i 's/SYSTEM__DRY_RUN=.*/SYSTEM__DRY_RUN=true/' .env
else
    echo "SYSTEM__DRY_RUN=true" >> .env
fi

# Set development environment
if grep -q "SYSTEM__ENVIRONMENT" .env; then
    sed -i 's/SYSTEM__ENVIRONMENT=.*/SYSTEM__ENVIRONMENT=development/' .env
else
    echo "SYSTEM__ENVIRONMENT=development" >> .env
fi

# Set exchange type
if grep -q "EXCHANGE__EXCHANGE_TYPE" .env; then
    sed -i 's/EXCHANGE__EXCHANGE_TYPE=.*/EXCHANGE__EXCHANGE_TYPE=bluefin/' .env
else
    echo "EXCHANGE__EXCHANGE_TYPE=bluefin" >> .env
fi

# Set testnet network
if grep -q "EXCHANGE__BLUEFIN_NETWORK" .env; then
    sed -i 's/EXCHANGE__BLUEFIN_NETWORK=.*/EXCHANGE__BLUEFIN_NETWORK=testnet/' .env
else
    echo "EXCHANGE__BLUEFIN_NETWORK=testnet" >> .env
fi

echo ""
echo "✅ Configuration fixed! Key settings:"
echo "   • Private Key: Valid 64-character hex dummy key (safe for testing)"
echo "   • Dry Run: ENABLED (paper trading only)"
echo "   • Environment: development"
echo "   • Exchange: bluefin"
echo "   • Network: testnet"
echo ""
echo "🚀 You can now restart the trading bot:"
echo "   docker-compose restart ai-trading-bot"
echo ""
echo "📊 To monitor logs:"
echo "   docker-compose logs -f ai-trading-bot"
