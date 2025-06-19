#!/bin/bash
# API Key Configuration Script for Bluefin Trading Bot

set -e

echo "🔐 BLUEFIN TRADING BOT - API KEY CONFIGURATION"
echo "=============================================="
echo ""
echo "This script will help you configure the required API keys for your trading bot."
echo ""

# Check if .env exists
if [[ ! -f ".env" ]]; then
    echo "❌ .env file not found!"
    exit 1
fi

echo "📋 Current configuration status:"
echo ""

# Check OpenAI API Key
if grep -q "sk-PLEASE_REPLACE" .env; then
    echo "❌ OpenAI API Key: NOT CONFIGURED"
    echo "   Get your API key from: https://platform.openai.com/api-keys"
else
    echo "✅ OpenAI API Key: CONFIGURED"
fi

# Check Bluefin Private Key
if grep -q "0xPLEASE_REPLACE" .env; then
    echo "❌ Bluefin Private Key: NOT CONFIGURED"
    echo "   This should be your SUI wallet private key in hex format"
else
    echo "✅ Bluefin Private Key: CONFIGURED"
fi

echo ""
echo "🔧 TO CONFIGURE YOUR API KEYS:"
echo ""
echo "1. Get your OpenAI API key:"
echo "   • Visit: https://platform.openai.com/api-keys"
echo "   • Create a new API key"
echo "   • Copy the key (starts with 'sk-')"
echo ""
echo "2. Get your SUI wallet private key:"
echo "   • Export from your SUI wallet (Sui Wallet, Suiet, etc.)"
echo "   • Should be in hex format (starts with '0x')"
echo "   • ⚠️  KEEP THIS PRIVATE - Never share this key!"
echo ""
echo "3. Edit .env and replace the placeholder values:"
echo "   nano .env"
echo ""
echo "   Replace:"
echo "   LLM__OPENAI_API_KEY=sk-PLEASE_REPLACE_WITH_YOUR_OPENAI_API_KEY"
echo "   EXCHANGE__BLUEFIN_PRIVATE_KEY=0xPLEASE_REPLACE_WITH_YOUR_SUI_WALLET_PRIVATE_KEY"
echo ""
echo "   With your actual keys:"
echo "   LLM__OPENAI_API_KEY=sk-your_actual_openai_key"
echo "   EXCHANGE__BLUEFIN_PRIVATE_KEY=0xYourActualSuiPrivateKey"
echo ""
echo "4. After configuring, run the bot:"
echo "   ./quick-launch-bluefin.sh"
echo ""
echo "🛡️  SECURITY REMINDERS:"
echo "• Never commit your .env file to git"
echo "• Never share your private keys"
echo "• Start with paper trading mode (SYSTEM__DRY_RUN=true)"
echo "• Test thoroughly before enabling live trading"
echo ""

# Offer to open the file for editing
read -p "Would you like to open .env for editing now? (y/n): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v code &> /dev/null; then
        echo "Opening in VS Code..."
        code .env
    elif command -v nano &> /dev/null; then
        echo "Opening in nano..."
        nano .env
    else
        echo "Please edit .env manually with your preferred text editor"
    fi
fi