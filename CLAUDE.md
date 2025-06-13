# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-assisted crypto futures trading bot for Coinbase that uses LangChain-powered decision making with custom VuManChu Cipher indicators. The bot is built with Python 3.12+ using Poetry for dependency management.

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
# Live trading (dry-run mode - safe)
python -m bot.main live
poetry run ai-trading-bot live

# Live trading with real money (dangerous)
poetry run ai-trading-bot live --dry-run false

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
- **Data Layer** (`bot/data/market.py`): Real-time market data from Coinbase WebSocket/REST
- **Indicators** (`bot/indicators/vumanchu.py`): VuManChu Cipher A & B indicators
- **LLM Agent** (`bot/strategy/llm_agent.py`): LangChain-powered trading decisions  
- **Validator** (`bot/validator.py`): JSON schema validation with fallback to HOLD
- **Risk Manager** (`bot/risk.py`): Position sizing, leverage control, loss limits
- **Exchange** (`bot/exchange/coinbase.py`): Coinbase order execution
- **CLI** (`bot/main.py`): Command-line interface and orchestration

### Data Flow
Market Data → Indicators → LLM Agent → Validator → Risk Manager → Exchange

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
- Always test in dry-run mode first
- Validate all trade actions through risk management layer
- Environment variables for API keys - never commit secrets
- Default to conservative settings in development

### Testing Strategy
- Unit tests for indicators, validator, risk management
- Integration tests for strategy flow
- Backtest coverage on multiple time periods and symbols
- Use pytest fixtures for consistent test data

## Important Files

- `bot/main.py` - CLI entry point and orchestration
- `bot/config.py` - Configuration management with pydantic
- `bot/strategy/llm_agent.py` - Core AI decision making
- `bot/indicators/vumanchu.py` - Custom technical indicators
- `bot/risk.py` - Risk management and position sizing
- `bot/validator.py` - JSON schema validation
- `pyproject.toml` - Poetry dependencies and tool configuration
- `docker-compose.yml` - Container orchestration
- `docs/AI_Trading_Bot_Architecture.md` - Detailed architecture guide

## Environment Setup

Required environment variables:
- `CB_API_KEY` - Coinbase API key
- `CB_API_SECRET` - Coinbase API secret  
- `OPENAI_API_KEY` - OpenAI API key
- `DRY_RUN` - Set to "false" for live trading (default: "true")
- `SYMBOL` - Trading pair (default: "BTC-USD")
- `LEVERAGE` - Futures leverage (default: 5)

## Security Notes

- API keys are managed through environment variables
- Container runs as non-root user for security
- Dry-run mode is the default to prevent accidental live trading
- All trading actions go through risk management validation