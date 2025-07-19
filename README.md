# AI Trading Bot

An AI-powered crypto trading bot supporting both Coinbase (CEX) and Bluefin (DEX) with VuManChu Cipher indicators and LangChain-powered decision making.

**Perfect for macOS with OrbStack** - Simple Docker setup, no Kubernetes complexity.

## Features

- **Multi-Exchange Support**: Trade on Coinbase (CEX) or Bluefin (DEX on Sui blockchain)
- **AI-Powered Decisions**: Uses OpenAI GPT models for intelligent trading decisions
- **VuManChu Cipher Indicators**: Custom technical indicators implemented in Python
- **Stablecoin Dominance Analysis**: Real-time USDT/USDC dominance tracking for market sentiment
  - Tracks stablecoin market share to gauge risk-on/risk-off sentiment
  - Adjusts position sizes based on dominance levels
  - Integrated into both LLM and fallback trading logic
- **Risk Management**: Built-in position sizing, stop-loss, and take-profit mechanisms
- **Real-time Data**: Live market data via WebSocket and REST APIs
- **Backtesting**: Historical strategy testing capabilities
- **Paper Trading Mode**: Test strategies safely without real money (single toggle control)
- **Memory & Learning**: MCP-powered experience tracking for continuous improvement
- **Functional Programming**: Enhanced security with opaque types and Result-based error handling

## 📊 Trading Features

### Exchange Options

#### Coinbase (Centralized Exchange)
- **Spot Trading**: Full support with volume-based fee tiers
- **Futures Trading**: Perpetual contracts available
- **Regulated**: KYC required, USD on/off ramps
- **High Liquidity**: Deep order books

#### Bluefin (Decentralized Exchange on Sui)
- **Perpetual Futures Only**: BTC-PERP, ETH-PERP, SOL-PERP, SUI-PERP
- **Non-Custodial**: You control your private keys
- **Low Fees**: ~0.05% trading fees, <$0.01 gas fees
- **High Performance**: 30ms transaction finality
- **Up to 100x Leverage**: Use with caution

### Exchange Selection
```bash
# In your .env file - Choose your exchange
EXCHANGE_TYPE=coinbase  # or "bluefin" for DEX trading
```

### Spot Trading Support (Coinbase)
- **Optimized for Coinbase Spot Markets**: Full support for spot trading with accurate fee calculations
- **Volume-Based Fee Tiers**: Automatically detects your 30-day trading volume and applies correct fees
- **Fee-Adjusted Position Sizing**: Positions are automatically sized to account for trading fees
- **Real-Time Fee Display**: Dashboard shows current fee tier, rates, and minimum profitable moves

### Fee Structure (Coinbase Spot)
- **Basic Tier** (< $10K): 0.6% maker, 1.2% taker
- **Active Tier** ($10K+): 0.25% maker, 0.40% taker
- **Standard Tier** ($50K+): 0.15% maker, 0.25% taker
- **Plus Tier** ($100K+): 0.10% maker, 0.20% taker
- **Pro Tier** ($1M+): 0.07% maker, 0.12% taker
- **Advanced Tier** ($15M+): 0.04% maker, 0.08% taker
- **VIP Tier** ($75M+): 0.02% maker, 0.05% taker
- **VIP Ultra** ($250M+): 0.00% maker, 0.05% taker

### Futures Trading Configuration
```bash
# For Coinbase futures
TRADING__ENABLE_FUTURES=true
TRADING__SYMBOL=BTC-USD

# For Bluefin (always futures)
EXCHANGE_TYPE=bluefin
TRADING_PAIRS=BTC-PERP,ETH-PERP  # Bluefin format
```

## Quick Start (3 Steps)

### 1. Setup Your Environment

```bash
# Clone the repository
git clone <repository-url>
cd ai-trading-bot

# Copy and configure environment
cp .env.example .env
```

### 2. Add Your API Keys

Edit `.env` with your credentials:

```bash
# Choose your exchange (pick one)
EXCHANGE_TYPE=coinbase  # or "bluefin"

# For Coinbase (CEX)
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_PRIVATE_KEY=your_coinbase_private_key_here

# For Bluefin (DEX)
BLUEFIN_PRIVATE_KEY=your_sui_wallet_private_key_here
BLUEFIN_NETWORK=mainnet  # or "testnet"

# Required: OpenAI API key
LLM_OPENAI_API_KEY=your_openai_api_key_here
```

**Get your API keys:**
- **Coinbase**: [Advanced Trade API](https://www.coinbase.com/cloud/api-keys)
- **Bluefin**: Export private key from your [Sui Wallet](https://chrome.google.com/webstore/detail/sui-wallet/opcgpfmipidbgpenhmajoajpbobppdil)
- **OpenAI**: [Platform API Keys](https://platform.openai.com/api-keys)

### 3. Start Trading

```bash
# Start the bot (in safe paper trading mode)
docker-compose up

# For development (with live code reloading)
docker-compose --profile dev up
```

**That's it!** The bot starts in paper trading mode (`SYSTEM__DRY_RUN=true`) for safety. This single environment variable controls all paper trading features - no need for CLI flags or separate settings.

## Trading Modes

### Paper Trading (Recommended First)
```bash
# Safe paper trading - no real money
# This is the default mode when SYSTEM__DRY_RUN=true
docker-compose up
```

### Live Trading (When Ready)
```bash
# 1. Edit .env and set SYSTEM__DRY_RUN=false
# 2. Start with small positions
# 3. WARNING: This uses real money!
docker-compose up
```

### Development Mode
```bash
# Live code reloading for development
docker-compose --profile dev up
```

## Configuration

All configuration is done via the `.env` file:

```bash
# Trading Mode Control (Single Toggle)
SYSTEM__DRY_RUN=true       # true = Paper trading (safe, simulated trades)
                           # false = Live trading (real money!)

# Trading settings
TRADING__SYMBOL=BTC-USD    # Trading pair
MAX_POSITION_SIZE=0.1      # 10% of portfolio max
RISK_PER_TRADE=0.02        # 2% risk per trade

# Safety settings
MAX_DAILY_LOSS=0.05        # Stop if 5% daily loss
STOP_LOSS_ENABLED=true     # Enable stop losses
TAKE_PROFIT_ENABLED=true   # Enable take profits
```

**Note:** The `SYSTEM__DRY_RUN` variable is the single master toggle for paper trading. When set to `true`, all trading operations are simulated. There are no additional CLI flags or settings needed - this one variable controls everything.

## 🌊 Bluefin DEX Integration

### What is Bluefin?

Bluefin is a high-performance decentralized perpetual futures exchange built on the Sui blockchain, offering:

- **Non-Custodial Trading**: You control your private keys - no exchange holds your funds
- **Lightning Fast**: 30ms transaction finality on Sui blockchain
- **Ultra-Low Fees**: ~0.05% trading fees + minimal gas (<$0.01)
- **High Leverage**: Up to 100x leverage (use responsibly!)
- **No KYC**: Trade without identity verification
- **Perpetual Futures**: BTC-PERP, ETH-PERP, SOL-PERP, SUI-PERP

### Quick Setup for Bluefin

1. **Get a Sui Wallet**
   - Install [Sui Wallet](https://chrome.google.com/webstore/detail/sui-wallet/opcgpfmipidbgpenhmajoajpbobppdil)
   - Create wallet and save recovery phrase
   - Export your private key

2. **Fund Your Wallet**
   - Get SUI tokens for gas (need ~1 SUI)
   - Transfer USDC to your Sui wallet (your trading capital)
   - Use [Wormhole Bridge](https://wormhole.com/) to bridge from other chains

3. **Configure the Bot**
   ```bash
   # In your .env file
   EXCHANGE_TYPE=bluefin
   BLUEFIN_PRIVATE_KEY=your_sui_private_key_here
   BLUEFIN_NETWORK=mainnet  # or testnet for testing
   TRADING_PAIRS=ETH-PERP,BTC-PERP  # Bluefin uses -PERP format
   ```

4. **Start Trading**
   ```bash
   # Paper trading on Bluefin (recommended first)
   TRADING_MODE=paper docker-compose up
   
   # Live trading (real money!)
   TRADING_MODE=live docker-compose up
   ```

### Bluefin vs Coinbase

| Feature | Coinbase (CEX) | Bluefin (DEX) |
|---------|----------------|---------------|
| **Custody** | Exchange holds funds | You hold funds |
| **KYC Required** | Yes | No |
| **Trading Fees** | 0.05-0.6% | ~0.05% |
| **Settlement** | Instant | ~550ms on-chain |
| **Leverage** | Up to 10x | Up to 100x |
| **Markets** | Spot & Futures | Futures only |
| **Liquidity** | Very High | Growing |
| **Security** | Regulated, insured | Smart contract risk |

### Enhanced Security Features

The bot includes enhanced security for Bluefin:

- **Opaque Private Keys**: Keys are automatically masked in all logs
- **Result-Based Validation**: Clear error messages for configuration issues
- **Rate Limiting**: Built-in protection against API abuse
- **Type-Safe Configuration**: Prevents common setup mistakes

See [docs/bluefin_integration.md](docs/bluefin_integration.md) for detailed setup instructions.

## Memory & Learning System (MCP)

The bot includes an advanced memory system that learns from every trade to improve future decisions.

### How It Works

**Automatic Memory Collection:**
- Records every trading decision with full market context
- Tracks trade outcomes (profit/loss, duration)
- Monitors market conditions throughout trades
- Generates insights from completed trades

**Memory-Enhanced Decisions:**
- Automatically retrieves similar past trades before each decision
- Shows the LLM what worked (or didn't) in similar conditions
- Identifies winning and losing patterns over time
- Adapts strategies based on accumulated experience

### Setup Options

#### Option 1: Local Memory (Default - No API Key Required)
```bash
# In your .env file:
MCP_ENABLED=true
MCP_SERVER_URL=http://localhost:8765

# Start with memory enabled:
docker-compose up -d mcp-memory
docker-compose up ai-trading-bot
```

All memories stored locally in `./data/mcp_memory/`

#### Option 2: With Cloud Backup (Optional)
```bash
# Sign up at https://mem0.ai for an API key
# In your .env file:
MCP_ENABLED=true
MCP_SERVER_URL=http://localhost:8765
MEM0_API_KEY=your-api-key-here
```

### What Gets Tracked

Every trade captures:
- Market conditions at entry (price, indicators, dominance)
- Trading decision and reasoning
- Position changes throughout the trade
- Final outcome and performance metrics
- Learned insights for future trades

### Memory in Action

The LLM sees past experiences in its prompts:
```
PAST TRADING EXPERIENCES:
1. 2024-01-15: LONG at $45,230 (RSI=62, Cipher B=15)
   Outcome: Profit +$523 (+1.2% in 45min)
   Insight: Quick profit when RSI > 60 with positive cipher

2. 2024-01-10: SHORT at $45,500 (RSI=68, Cipher B=-5)
   Outcome: Loss -$312 (-0.7% in 120min)
   Insight: Shorting in uptrend was premature
```

## Monitoring

### View Logs
```bash
# Follow bot logs
docker-compose logs -f ai-trading-bot

# View specific logs
docker-compose logs ai-trading-bot

# View memory server logs
docker-compose logs -f mcp-memory
```

### Health Check
```bash
# Check if bot is running
docker-compose ps

# Check container health
docker inspect ai-trading-bot --format='{{.State.Health.Status}}'

# Check memory server health
curl http://localhost:8765/health
```

## Data Persistence

Your data is automatically saved in:
- `./logs/` - Trading logs and events
- `./data/` - Market data and trading history
- `./data/mcp_memory/` - Trading experiences and learned patterns
- `./config/` - Configuration files

## Troubleshooting

### Common Issues

**Bot won't start?**
```bash
# Check your .env file has valid API keys
cat .env

# Check container logs
docker-compose logs ai-trading-bot
```

**No trading activity?**
```bash
# Ensure SYSTEM__DRY_RUN=true for paper trading
# Check market hours (crypto trades 24/7)
# Verify API keys are correct
```

**Docker issues on macOS?**
```bash
# Make sure OrbStack or Docker Desktop is running
docker ps

# Restart Docker if needed
docker-compose down && docker-compose up
```

## Development

### Local Development (No Docker)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell

# Run bot locally (paper trading mode by default)
python -m bot.main live
```

### Code Quality

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Type checking
poetry run mypy bot/

# Run tests
poetry run pytest
```

## Project Structure

```
bot/                      # Main application code
├── main.py              # CLI entry point
├── config.py            # Configuration management
├── data/market.py       # Market data ingestion
├── indicators/          # Technical indicators
├── strategy/            # AI trading strategy
├── exchange/coinbase.py # Coinbase integration
├── risk.py              # Risk management
└── backtest/            # Backtesting engine

config/                  # Configuration files
├── development.json     # Dev settings
└── conservative_config.json

logs/                    # Trading logs (auto-created)
data/                    # Market data (auto-created)
```

## Safety Features

🛡️ **Built-in Safety:**
- Starts in paper trading mode (`SYSTEM__DRY_RUN=true`) by default
- Position size limits
- Daily loss limits
- Stop-loss protection
- Health checks and monitoring

## Architecture

**Simple & Clean:**
- **Data Layer**: Real-time market data from Coinbase or Bluefin
- **AI Engine**: OpenAI GPT-powered trading decisions with memory enhancement
- **Memory System**: MCP server for experience tracking and learning
- **Risk Manager**: Position sizing and loss protection
- **Exchange Layer**: Multi-exchange support (Coinbase CEX & Bluefin DEX)
- **Validator**: JSON schema validation with fallback to HOLD
- **Functional Programming**: Type-safe configuration with Result-based error handling

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and run tests
4. Commit changes (`git commit -m 'Add feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## License

MIT License - see LICENSE file for details.

## Risk Disclaimer

⚠️ **IMPORTANT**: This is experimental trading software. Cryptocurrency trading involves significant risk:

- **Always start with paper trading (`SYSTEM__DRY_RUN=true`)**
- **Use small position sizes initially**
- **Never risk more than you can afford to lose**
- **Thoroughly test before live trading**
- **Monitor closely during initial weeks**

## Support

- 📚 Check the [docs/](docs/) directory
- 🐛 Open an issue on GitHub
- 💬 Review architecture guide in `docs/AI_Trading_Bot_Architecture.md`

---

**Ready to start?** Run `cp .env.example .env`, add your API keys, and `docker-compose up`!
