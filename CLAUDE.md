# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-assisted crypto futures trading bot that supports both Coinbase (CEX) and Bluefin (DEX on Sui) exchanges. It uses LangChain-powered decision making with custom VuManChu Cipher indicators. The bot is built with Python 3.12+ using Poetry for dependency management.

## Development Commands

### Setup
```bash
# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Activate virtual environment
poetry shell
```

### Running the Bot
```bash
# Paper trading mode (safe - no real money)
python -m bot.main live
poetry run ai-trading-bot live

# Live trading with real money (dangerous)
# Set SYSTEM__DRY_RUN=false in .env first!
python -m bot.main live

# Backtesting
poetry run ai-trading-bot backtest --from 2024-01-01 --to 2024-12-31
poetry run ai-trading-bot backtest --symbol ETH-USD
```

### Code Quality Tools
```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Type checking
poetry run mypy bot/

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=bot

# Run specific test file
poetry run pytest tests/unit/test_indicators.py
```

### Docker Operations (Simplified for macOS)
```bash
# Quick start (recommended)
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f ai-trading-bot

# Stop and restart
docker-compose restart ai-trading-bot
docker-compose down
```

## Architecture

### Core Components
- **Data Layer** (`bot/data/market.py`): Real-time market data from exchange WebSocket/REST
- **Indicators** (`bot/indicators/vumanchu.py`): VuManChu Cipher A & B indicators
- **LLM Agent** (`bot/strategy/llm_agent.py`): LangChain-powered trading decisions  
- **Memory-Enhanced Agent** (`bot/strategy/memory_enhanced_agent.py`): LLM with learning from past trades
- **MCP Memory Server** (`bot/mcp/memory_server.py`): Persistent memory storage for experiences
- **Experience Manager** (`bot/learning/experience_manager.py`): Trade lifecycle tracking
- **Self-Improvement Engine** (`bot/learning/self_improvement.py`): Pattern analysis and strategy optimization
- **Validator** (`bot/validator.py`): JSON schema validation with fallback to HOLD
- **Risk Manager** (`bot/risk.py`): Position sizing, leverage control, loss limits
- **Exchange Layer** (`bot/exchange/`): Multi-exchange support
  - **Base Interface** (`base.py`): Abstract exchange interface
  - **Coinbase** (`coinbase.py`): Centralized exchange implementation
  - **Bluefin** (`bluefin.py`): Decentralized exchange on Sui
  - **Factory** (`factory.py`): Exchange instantiation based on config
- **CLI** (`bot/main.py`): Command-line interface and orchestration

### Data Flow
Market Data → Indicators → LLM Agent (with Memory) → Validator → Risk Manager → Exchange → Experience Tracking → Learning

### Configuration System
- Environment variables in `.env` file
- JSON config files in `config/` directory (development.json, production.json, conservative_config.json)
- Pydantic models for type-safe settings in `bot/config.py`

### Key Technologies
- **Poetry** for dependency management
- **LangChain** + OpenAI for AI decision making
- **coinbase-advanced-py** for exchange integration
- **pandas/numpy** for technical analysis
- **pydantic** v2 for data validation
- **click** for CLI interface
- **rich** for terminal UI

## Development Guidelines

### Technical Indicators
- Implement using vectorized numpy/pandas operations (no for-loops)
- Return both latest state and full series for backtesting
- Follow the VuManChu Cipher reference in `docs/vumanchu_cipher_reference.md`

### LLM Integration
- Prompts located in `prompts/` directory
- JSON schema validation is critical - invalid responses default to HOLD
- Use `TradeAction` pydantic model for structured output

### Safety Measures
- Always test in paper trading mode first (SYSTEM__DRY_RUN=true)
- Validate all trade actions through risk management layer
- Environment variables for API keys - never commit secrets
- Default to conservative settings in development

### Testing Strategy
- Unit tests for indicators, validator, risk management
- Integration tests for strategy flow
- Backtest coverage on multiple time periods and symbols
- Use pytest fixtures for consistent test data

### Memory and Learning System (MCP)
When MCP is enabled, the bot:
- Stores every trading decision with full market context
- Tracks trade outcomes from entry to exit
- Analyzes patterns in successful/failed trades
- Retrieves similar past experiences before making decisions
- Generates insights and strategy adjustments based on performance
- Maintains a persistent memory across bot restarts

To enable memory features:
1. Set `MCP_ENABLED=true` in your `.env` file
2. Run `docker-compose up mcp-memory` to start the memory server
3. The bot will automatically use the memory-enhanced agent

The memory system includes:
- **TradingExperience**: Complete record of a trade with context and outcome
- **MemoryQuery**: Similarity-based retrieval of relevant past trades
- **PatternPerformance**: Statistical tracking of pattern success rates
- **StrategyAdjustment**: Recommended parameter changes based on analysis
- **LearningInsight**: Discovered correlations and market behaviors

## Important Files

- `bot/main.py` - CLI entry point and orchestration
- `bot/config.py` - Configuration management with pydantic (includes MCP settings)
- `bot/strategy/llm_agent.py` - Core AI decision making
- `bot/strategy/memory_enhanced_agent.py` - Memory-enhanced LLM with learning capabilities
- `bot/indicators/vumanchu.py` - Custom technical indicators
- `bot/risk.py` - Risk management and position sizing
- `bot/validator.py` - JSON schema validation
- `bot/mcp/memory_server.py` - MCP memory server for persistent learning
- `bot/learning/experience_manager.py` - Trade lifecycle tracking
- `bot/learning/self_improvement.py` - Pattern analysis and improvement engine
- `pyproject.toml` - Poetry dependencies and tool configuration
- `docker-compose.yml` - Container orchestration (includes MCP memory server)
- `docs/AI_Trading_Bot_Architecture.md` - Detailed architecture guide
- `docs/bluefin_integration.md` - Bluefin DEX setup and usage guide

## Environment Setup

Required environment variables:

### Exchange Selection
- `EXCHANGE__EXCHANGE_TYPE` - Exchange to use: "coinbase" or "bluefin" (default: "coinbase")

### Coinbase Configuration
- `EXCHANGE__CDP_API_KEY_NAME` - Coinbase CDP API key name
- `EXCHANGE__CDP_PRIVATE_KEY` - Coinbase CDP private key (PEM format)  

### Bluefin Configuration
- `EXCHANGE__BLUEFIN_PRIVATE_KEY` - Sui wallet private key (hex format)
- `EXCHANGE__BLUEFIN_NETWORK` - Network: "mainnet" or "testnet" (default: "mainnet")

### Trading Configuration
- `LLM__OPENAI_API_KEY` - OpenAI API key
- `SYSTEM__DRY_RUN` - Set to "false" for live trading, "true" for paper trading (default: "true")
- `TRADING__SYMBOL` - Trading pair (default: "BTC-USD")
- `TRADING__LEVERAGE` - Futures leverage (default: 5)

Optional MCP/Learning environment variables:
- `MCP_ENABLED` - Enable memory and learning features (default: "false")
- `MCP_SERVER_URL` - MCP memory server URL (default: "http://localhost:8765")
- `MEM0_API_KEY` - API key for Mem0 if using cloud storage
- `MCP_MEMORY_RETENTION_DAYS` - Days to retain memories (default: 90)

## Security Notes

- API keys are managed through environment variables
- Container runs as non-root user for security
- Paper trading mode (SYSTEM__DRY_RUN=true) is the default to prevent accidental live trading
- All trading actions go through risk management validation