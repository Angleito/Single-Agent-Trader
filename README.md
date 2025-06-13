# AI Trading Bot

An AI-powered crypto trading bot for Coinbase with VuManChu Cipher indicators and LangChain-powered decision making.

**Perfect for macOS with OrbStack** - Simple Docker setup, no Kubernetes complexity.

## Features

- **AI-Powered Decisions**: Uses OpenAI GPT models for intelligent trading decisions
- **VuManChu Cipher Indicators**: Custom technical indicators implemented in Python
- **Stablecoin Dominance Analysis**: Real-time USDT/USDC dominance tracking for market sentiment
  - Tracks stablecoin market share to gauge risk-on/risk-off sentiment
  - Adjusts position sizes based on dominance levels
  - Integrated into both LLM and fallback trading logic
- **Risk Management**: Built-in position sizing, stop-loss, and take-profit mechanisms
- **Real-time Data**: Live market data via Coinbase WebSocket and REST APIs
- **Backtesting**: Historical strategy testing capabilities
- **Dry Run Mode**: Test strategies safely without real money

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
# Required: Coinbase API credentials
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# Required: OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

**Get your API keys:**
- **Coinbase**: [Advanced Trade API](https://www.coinbase.com/cloud/api-keys)
- **OpenAI**: [Platform API Keys](https://platform.openai.com/api-keys)

### 3. Start Trading

```bash
# Start the bot (in safe dry-run mode)
docker-compose up

# For development (with live code reloading)
docker-compose --profile dev up
```

**That's it!** The bot starts in `DRY_RUN=true` mode for safety.

## Trading Modes

### Dry Run (Recommended First)
```bash
# Safe paper trading - no real money
docker-compose up
```

### Live Trading (When Ready)
```bash
# 1. Edit .env and set DRY_RUN=false
# 2. Start with small positions
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
# Trading settings
SYMBOL=BTC-USD              # Trading pair
DRY_RUN=true               # Safe mode (no real money)
MAX_POSITION_SIZE=0.1      # 10% of portfolio max
RISK_PER_TRADE=0.02        # 2% risk per trade

# Safety settings
MAX_DAILY_LOSS=0.05        # Stop if 5% daily loss
STOP_LOSS_ENABLED=true     # Enable stop losses
TAKE_PROFIT_ENABLED=true   # Enable take profits
```

## Monitoring

### View Logs
```bash
# Follow bot logs
docker-compose logs -f ai-trading-bot

# View specific logs
docker-compose logs ai-trading-bot
```

### Health Check
```bash
# Check if bot is running
docker-compose ps

# Check container health
docker inspect ai-trading-bot --format='{{.State.Health.Status}}'
```

## Data Persistence

Your data is automatically saved in:
- `./logs/` - Trading logs and events
- `./data/` - Market data and trading history
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
# Ensure DRY_RUN=true for testing
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

# Run bot locally
python -m bot.main live --dry-run
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
- Starts in `DRY_RUN=true` mode by default
- Position size limits
- Daily loss limits
- Stop-loss protection
- Health checks and monitoring

## Architecture

**Simple & Clean:**
- **Data Layer**: Real-time Coinbase market data
- **AI Engine**: OpenAI GPT-powered trading decisions
- **Risk Manager**: Position sizing and loss protection
- **Exchange**: Coinbase order execution
- **Validator**: JSON schema validation with fallback to HOLD

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

- **Always start with `DRY_RUN=true`**
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