# Bluefin DEX Trading Bot on OrbStack

This guide explains how to launch the AI trading bot specifically configured for Bluefin DEX on OrbStack.

## 🚀 Quick Start

1. **Install OrbStack** (if not already installed):
   ```bash
   # Download from https://orbstack.dev/
   ```

2. **Configure API Keys**:
   ```bash
   # Edit .env.bluefin and add your API keys:
   LLM__OPENAI_API_KEY=your_openai_api_key_here
   EXCHANGE__BLUEFIN_PRIVATE_KEY=your_sui_wallet_private_key_here
   ```

3. **Launch the Bot**:
   ```bash
   ./orbstack-launch-bluefin.sh
   ```

## 📋 What This Deployment Includes

- **AI Trading Bot**: Configured for Bluefin DEX (SUI-PERP)
- **Bluefin Service**: SDK service for Bluefin DEX operations
- **Real Market Data**: Live price feeds from Bluefin DEX
- **Paper Trading**: Safe simulation mode using real prices
- **Comprehensive Logging**: Detailed trade analysis and logging

## 🔧 Configuration Details

### Trading Configuration
- **Exchange**: Bluefin DEX (Sui Network)
- **Symbol**: SUI-PERP (SUI Perpetual Futures)
- **Leverage**: 5x
- **Interval**: 1 minute
- **Mode**: Paper trading (SYSTEM__DRY_RUN=true)

### Data Sources
- **Market Data**: Real-time from Bluefin DEX
- **No Mock Data**: All data comes from live exchange
- **Real Price Feeds**: WebSocket and REST API integration

## 📊 Monitoring and Logs

### View Bot Logs
```bash
docker-compose -f docker-compose.bluefin.yml logs -f ai-trading-bot-bluefin
```

### View All Services
```bash
docker-compose -f docker-compose.bluefin.yml logs -f
```

### Check Service Status
```bash
docker-compose -f docker-compose.bluefin.yml ps
```

## 🛑 Stopping the Bot

```bash
docker-compose -f docker-compose.bluefin.yml down
```

## 📈 Expected Log Output

When the bot is running, you'll see logs like:

```
🎯 PAPER TRADING DECISION: LONG | Symbol: SUI-PERP | Current Price: $3.45 | Size: 25% | Reason: Strong bullish signals
📊 TRADE SIMULATION DETAILS:
  • Symbol: SUI-PERP
  • Action: LONG  
  • Current Real Price: $3.45
  • Position Size: 724.64 SUI
  • Position Value: $2,500.00
  • Leverage: 5x
  • Required Margin: $500.00
✅ PAPER TRADING EXECUTION COMPLETE:
  🎯 Action: LONG
  📊 Size: 724.64 SUI-PERP
  💵 Price: $3.4517
  💸 Value: $2,501.23
  🏷️ Fees: $3.75 @ 0.1500%
  🔴 Stop Loss: $3.28 (-$123.45)
  🟢 Take Profit: $3.62 (+$123.45)
```

## 🛡️ Safety Features

- **Paper Trading Mode**: Uses real prices but simulates trades (no real money)
- **Real Market Data**: Accurate price feeds for realistic simulation
- **Security Hardening**: Containers run as non-root with minimal privileges
- **Resource Limits**: Memory and CPU limits prevent system overload

## ⚙️ Advanced Configuration

### Enable Live Trading (Real Money)
⚠️ **WARNING: This trades with real money!**

```bash
# Edit .env.bluefin
SYSTEM__DRY_RUN=false
```

### Enable MCP Memory (AI Learning)
```bash
# Edit .env.bluefin  
MCP_ENABLED=true

# Launch with memory profile
docker-compose -f docker-compose.bluefin.yml --profile memory up -d
```

### Enable Dashboard
```bash
# Launch with dashboard profile
docker-compose -f docker-compose.bluefin.yml --profile dashboard up -d

# Access dashboard at:
# http://localhost:3000
```

## 🔧 Troubleshooting

### Check OrbStack Status
```bash
orb status
```

### Rebuild Containers
```bash
docker-compose -f docker-compose.bluefin.yml build --no-cache
```

### View Container Resources
```bash
docker stats
```

### Check Network Connectivity
```bash
docker-compose -f docker-compose.bluefin.yml exec ai-trading-bot-bluefin ping bluefin-service
```

## 📁 File Structure

```
.
├── docker-compose.bluefin.yml    # Bluefin-specific Docker Compose
├── .env.bluefin                  # Bluefin environment configuration
├── orbstack-launch-bluefin.sh    # OrbStack launch script
├── logs/bluefin/                 # Bluefin service logs
├── data/                         # Trading data storage
└── README-OrbStack-Bluefin.md    # This file
```

## 🎯 Next Steps

1. Monitor the logs to see real-time trading decisions
2. Analyze paper trading performance
3. Adjust configuration as needed
4. Consider enabling live trading only after thorough testing

---

**Remember**: This bot uses real market data from Bluefin DEX but trades in paper mode by default. Perfect for testing strategies with live market conditions!