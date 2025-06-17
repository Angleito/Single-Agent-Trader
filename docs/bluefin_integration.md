# Bluefin Exchange Integration Guide

This guide explains how to set up and use Bluefin exchange as an alternative to Coinbase for perpetual futures trading.

## Overview

Bluefin is a decentralized perpetual futures exchange built on the Sui blockchain. It offers:
- High-performance trading with 30ms transaction finalization
- Low fees (<$0.005 per trade)
- Up to 100x leverage
- Non-custodial trading (you control your keys)
- Perpetual futures contracts (BTC-PERP, ETH-PERP, etc.)

## Setup Requirements

### 1. Sui Wallet Setup

First, you need a Sui wallet:

1. **Install Sui Wallet Extension**
   - Chrome/Brave: [Sui Wallet](https://chrome.google.com/webstore/detail/sui-wallet/opcgpfmipidbgpenhmajoajpbobppdil)
   - Or use [Suiet Wallet](https://suiet.app/)

2. **Create a New Wallet**
   - Save your recovery phrase securely
   - Export your private key (you'll need this for the bot)

3. **Fund Your Wallet**
   - Get SUI tokens for gas fees (small amount needed, ~1 SUI)
   - Transfer USDC to your Sui wallet (this is your trading capital)
   - Bridge assets from other chains using [Wormhole](https://wormhole.com/) or [Portal Bridge](https://portalbridge.com/)

### 2. Bot Configuration

Add these settings to your `.env` file:

```bash
# Exchange Selection
EXCHANGE__EXCHANGE_TYPE=bluefin

# Bluefin Configuration
EXCHANGE__BLUEFIN_PRIVATE_KEY=your_sui_wallet_private_key_here
EXCHANGE__BLUEFIN_NETWORK=mainnet  # or testnet for testing
EXCHANGE__BLUEFIN_RPC_URL=https://sui-mainnet.nodeinfra.com  # optional custom RPC

# Disable Coinbase futures if using Bluefin
TRADING__ENABLE_FUTURES=true  # Bluefin only supports futures
```

### 3. Install Dependencies

Due to dependency conflicts, the Bluefin SDK needs to be installed separately:

```bash
# Install main dependencies
poetry install

# Install Bluefin SDK in a separate virtual environment or globally
pip install bluefin-v2-client==3.2.4

# Or use a dedicated virtualenv for Bluefin
python -m venv bluefin-env
source bluefin-env/bin/activate  # On Windows: bluefin-env\Scripts\activate
pip install bluefin-v2-client==3.2.4
```

**Note**: The Bluefin SDK has specific version requirements for aiohttp and websocket-client that conflict with other dependencies. Consider using Docker or a separate environment for production deployments.

## Trading Configuration

### Symbol Mapping

Bluefin uses different symbol formats:
- Coinbase: `ETH-USD`, `BTC-USD`
- Bluefin: `ETH-PERP`, `BTC-PERP`

The bot automatically handles this conversion.

### Supported Markets

Common perpetual futures on Bluefin:
- `BTC-PERP` - Bitcoin perpetual
- `ETH-PERP` - Ethereum perpetual
- `SOL-PERP` - Solana perpetual
- `SUI-PERP` - Sui perpetual

### Leverage Settings

Bluefin supports up to 100x leverage, but be cautious:

```bash
# Conservative
TRADING__LEVERAGE=2
TRADING__MAX_FUTURES_LEVERAGE=5

# Moderate
TRADING__LEVERAGE=5
TRADING__MAX_FUTURES_LEVERAGE=10

# Aggressive (not recommended)
TRADING__LEVERAGE=10
TRADING__MAX_FUTURES_LEVERAGE=20
```

## Running the Bot

### Paper Trading (Recommended First)

Test with paper trading first:

```bash
# Paper trading with Bluefin
python -m bot.main live --dry-run
```

### Live Trading

⚠️ **WARNING**: This uses real money on Bluefin!

```bash
# Live trading on Bluefin mainnet
python -m bot.main live

# Or use testnet first
EXCHANGE__BLUEFIN_NETWORK=testnet python -m bot.main live
```

## Key Differences from Coinbase

### 1. **Decentralized Nature**
- All trades are on-chain (Sui blockchain)
- You control your private keys
- No KYC required
- Cannot be shut down or censored

### 2. **Fee Structure**
- Trading fees: ~0.05% (much lower than CEX)
- Gas fees: Minimal on Sui (<$0.01)
- No deposit/withdrawal fees

### 3. **Settlement**
- Trades settle on-chain in ~550ms
- All positions are perpetual futures
- USDC is the settlement currency

### 4. **Risk Considerations**
- Smart contract risk (audited but still DeFi)
- Liquidity may be lower than CEX
- Network congestion possible during high activity
- No customer support (decentralized)

## Monitoring Your Trades

### 1. **Bluefin Web Interface**
Visit [trade.bluefin.io](https://trade.bluefin.io) to see your positions

### 2. **Sui Explorer**
Track your transactions: [suiscan.xyz](https://suiscan.xyz)

### 3. **Bot Logs**
The bot will show exchange type in startup:
```
Exchange (Bluefin)    ✓ Connected    mainnet network (Sui)
```

## Troubleshooting

### Connection Issues

1. **"Failed to connect to Bluefin"**
   - Check your private key format (should be hex string)
   - Ensure you have SUI for gas
   - Try a different RPC endpoint

2. **"Insufficient funds"**
   - Check USDC balance in your Sui wallet
   - Ensure you have SUI for gas fees

3. **"Order failed"**
   - Check minimum order size (usually $10 notional)
   - Verify the trading pair is active
   - Check if you have open positions at max leverage

### Network Issues

If you experience slow performance, try alternative RPC endpoints:
- `https://fullnode.mainnet.sui.io`
- `https://sui-rpc.publicnode.com`
- `https://mainnet.suiet.app`

## Security Best Practices

1. **Private Key Storage**
   - Never commit your private key to git
   - Use environment variables or `.env` file
   - Consider using a hardware wallet for large amounts

2. **Risk Management**
   - Start with small position sizes
   - Use conservative leverage (2-5x)
   - Always set stop losses
   - Monitor gas balance

3. **Backup**
   - Keep your recovery phrase secure
   - Have backup RPC endpoints configured
   - Monitor smart contract upgrades

## Advanced Configuration

### Custom Contract Addresses

If Bluefin updates their contracts, you can override:

```python
# In bot/exchange/bluefin.py
CONTRACT_ADDRESSES = {
    "mainnet": {
        "exchange": "0x...",
        "clearing_house": "0x...",
    }
}
```

### WebSocket Optimization

For better real-time performance:

```bash
# Increase WebSocket timeout
EXCHANGE__WEBSOCKET_TIMEOUT=60

# Reduce rate limiting for Bluefin
EXCHANGE__RATE_LIMIT_REQUESTS=30
```

## Switching Between Exchanges

You can easily switch between Coinbase and Bluefin:

```bash
# Use Coinbase
EXCHANGE__EXCHANGE_TYPE=coinbase python -m bot.main live

# Use Bluefin
EXCHANGE__EXCHANGE_TYPE=bluefin python -m bot.main live
```

## Support

- Bluefin Documentation: [bluefin-exchange.readme.io](https://bluefin-exchange.readme.io)
- Sui Developer Docs: [docs.sui.io](https://docs.sui.io)
- Bluefin Discord: [discord.gg/bluefin](https://discord.gg/bluefin)

Remember: Bluefin is a DeFi protocol. Always DYOR (Do Your Own Research) and never trade more than you can afford to lose.
