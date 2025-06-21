# Bluefin Quick Start Guide

Get your AI trading bot running on Bluefin DEX in under 10 minutes.

## What is Bluefin?

Bluefin is a decentralized perpetual futures exchange built on Sui blockchain offering:
- **Low fees** (~0.05% vs 0.5%+ on CEX)
- **Fast settlement** (550ms on-chain)
- **High leverage** (up to 100x)
- **Non-custodial** (you control your keys)
- **No KYC** required

## Prerequisites

- Docker and Docker Compose installed
- Sui wallet with some SUI tokens for gas
- USDC tokens for trading
- OpenAI API key

## Step 1: Get a Sui Wallet

1. **Install Sui Wallet**:
   - Chrome/Brave: [Sui Wallet Extension](https://chrome.google.com/webstore/detail/sui-wallet/opcgpfmipidbgpenhmajoajpbobppdil)
   - Alternative: [Suiet Wallet](https://suiet.app/)

2. **Create New Wallet** and save your recovery phrase securely

3. **Export Private Key**:
   - Go to Settings → Export Private Key
   - Copy the hex string (starts with `0x`)

4. **Fund Your Wallet**:
   - Get SUI tokens for gas (~1 SUI is enough)
   - Transfer USDC for trading (your capital)
   - Use [Wormhole](https://wormhole.com/) or [Portal Bridge](https://portalbridge.com/) to bridge from other chains

## Step 2: Quick Launch

1. **Clone and Configure**:
```bash
git clone <your-repo>
cd ai-trading-bot

# Copy environment template
cp .env.bluefin.example .env
```

2. **Edit `.env` file**:
```bash
# Required - Your Sui wallet private key
EXCHANGE__BLUEFIN_PRIVATE_KEY=0x1234567890abcdef...

# Required - Your OpenAI API key
LLM__OPENAI_API_KEY=sk-...

# Optional - Change trading symbol
TRADING__SYMBOL=SUI-PERP

# Safety first - paper trading
SYSTEM__DRY_RUN=true
```

3. **Launch with Docker**:
```bash
# Start paper trading (safe mode)
docker-compose -f docker-compose.bluefin.yml up

# Or use quick launch script
./quick-launch-bluefin.sh
```

## Step 3: Verify Setup

You should see logs like:
```
✓ Bluefin Service    Connected    mainnet
✓ Exchange (Bluefin) Connected    mainnet network (Sui)
✓ LLM Agent         OpenAI GPT-4 Ready
🎯 PAPER TRADING MODE - No real trades will be executed
```

## Step 4: Monitor Trading

1. **View Logs**:
```bash
docker-compose -f docker-compose.bluefin.yml logs -f ai-trading-bot-bluefin
```

2. **Optional Dashboard** (recommended):
```bash
# Start with dashboard
docker-compose -f docker-compose.bluefin.yml --profile dashboard up
```
Then visit: http://localhost:3000

3. **Check Bluefin Interface**:
   - Visit [trade.bluefin.io](https://trade.bluefin.io)
   - Connect your wallet to see positions

## Common Paper Trading Output

```
🎯 PAPER TRADING DECISION: LONG | Symbol: SUI-PERP | Current Price: $3.45 | Size: 25%
📊 TRADE SIMULATION DETAILS:
  • Current Real Price: $3.45
  • Position Size: 724.64 SUI
  • Position Value: $2,500.00
  • Required Margin: $500.00
✅ PAPER TRADING EXECUTION COMPLETE
```

## Going Live (Advanced)

⚠️ **WARNING**: This uses real money!

1. **Start Small**:
```bash
# Edit .env
SYSTEM__DRY_RUN=false
TRADING__LEVERAGE=2  # Conservative leverage
```

2. **Monitor Closely**:
   - Watch your actual positions on [trade.bluefin.io](https://trade.bluefin.io)
   - Check your wallet balance regularly
   - Set up alerts for large moves

## Next Steps

- Read the [Comprehensive Setup Guide](BLUEFIN_SETUP_GUIDE.md)
- Learn about [Security Best Practices](BLUEFIN_SECURITY_GUIDE.md)
- Check [Troubleshooting Guide](BLUEFIN_TROUBLESHOOTING.md) if issues arise
- Optimize with [Performance Tuning](BLUEFIN_PERFORMANCE_OPTIMIZATION.md)

## Support

- **Documentation**: Check other guides in `/docs`
- **Bluefin Docs**: [bluefin-exchange.readme.io](https://bluefin-exchange.readme.io)
- **Sui Docs**: [docs.sui.io](https://docs.sui.io)
- **Discord**: [discord.gg/bluefin](https://discord.gg/bluefin)

## Safety Reminders

- ✅ Always test in paper trading first (`SYSTEM__DRY_RUN=true`)
- ✅ Start with small amounts and low leverage
- ✅ Keep your private keys secure
- ✅ Monitor your trades actively
- ⚠️ Never commit your private key to git
- ⚠️ DeFi has smart contract risk
- ⚠️ Only trade what you can afford to lose
