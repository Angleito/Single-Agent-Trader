# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**FUNCTIONAL PROGRAMMING TRANSFORMATION COMPLETE**: This repository now supports both legacy and functional programming (FP) configuration patterns. The FP system provides enhanced type safety, security, and error handling while maintaining full backward compatibility.

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

#### Configuration Validation (FP System)
```bash
# Validate FP configuration before running
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print('✅ Config valid' if result.is_success() else f'❌ Config error: {result.failure()}')"

# Validate specific components
python -c "from bot.fp.types.config import build_exchange_config_from_env; result = build_exchange_config_from_env(); print('✅ Exchange valid' if result.is_success() else f'❌ Exchange error: {result.failure()}')"
```

#### Paper Trading (Recommended)
```bash
# Paper trading mode (safe - no real money) - FP Configuration
TRADING_MODE=paper python -m bot.main live

# Paper trading mode - Legacy Configuration
python -m bot.main live
poetry run ai-trading-bot live

# Paper trading with real market data (default behavior)
# Uses live prices but doesn't place real trades - logs what would happen
python -m bot.main live

# Configure with environment file
cp .env.example .env
python -m bot.main live
```

#### Live Trading (Use with Caution)
```bash
# Live trading with real money (dangerous) - FP Configuration
TRADING_MODE=live python -m bot.main live

# Live trading - Legacy Configuration
# Set SYSTEM__DRY_RUN=false in .env first!
python -m bot.main live
```

#### Backtesting
```bash
# Backtesting with FP Configuration
TRADING_MODE=backtest python -m bot.main backtest --from 2024-01-01 --to 2024-12-31

# Legacy backtesting
poetry run ai-trading-bot backtest --from 2024-01-01 --to 2024-12-31
poetry run ai-trading-bot backtest --symbol ETH-USD
```

### Code Quality Tools
```bash
# Format code
poetry run black .

# Lint and fix code issues
poetry run ruff check . --fix

# Format code with Ruff (alternative to black)
poetry run ruff format .

# Type checking
poetry run mypy bot/

# Dead code detection
poetry run vulture bot/

# Security vulnerability scanning
poetry run bandit -r bot/

# HTML/template linting and formatting
poetry run djlint --check --reformat .

# Run all pre-commit hooks
poetry run pre-commit run --all-files

# Full code quality check (recommended)
poetry run ruff check . --fix && poetry run ruff format . && poetry run mypy bot/ && poetry run vulture bot/ && poetry run bandit -r bot/

# Automated code quality script
./scripts/code-quality.sh
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

### Docker Permissions Setup (IMPORTANT - Run First!)
```bash
# Fix Docker volume permission issues (run once)
./scripts/setup-docker-permissions.sh

# This script:
# - Creates required directories (logs/, data/, etc.)
# - Sets proper permissions and ownership for your OS user
# - Updates .env with correct HOST_UID/HOST_GID settings
# - Prevents "permission denied" errors during container startup
# - Tests write permissions and provides troubleshooting guidance

# If the script fails or you encounter issues, manually set ownership:
sudo chown -R 1000:1000 ./logs ./data ./tmp
```

### Docker Operations with Functional Programming Support

#### Docker Setup with Configuration Validation
```bash
# FIRST TIME: Set up permissions (prevents permission errors)
./scripts/setup-docker-permissions.sh

# Validate configuration before starting Docker
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print('✅ Config valid' if result.is_success() else f'❌ Config error: {result.failure()}')"

# Quick start (recommended)
docker-compose up

# Run in background with configuration validation
docker-compose up -d

# View logs with configuration info
docker-compose logs -f ai-trading-bot

# Stop and restart with config check
docker-compose restart ai-trading-bot
docker-compose down
```

#### Docker Environment Configuration

**Docker Compose Environment Variables**:
```yaml
# docker-compose.override.yml
services:
  ai-trading-bot:
    environment:
      # Functional Programming Configuration
      - STRATEGY_TYPE=llm
      - TRADING_MODE=paper
      - EXCHANGE_TYPE=coinbase
      - LOG_LEVEL=INFO

      # Security (credentials from .env file)
      - COINBASE_API_KEY=${COINBASE_API_KEY}
      - COINBASE_PRIVATE_KEY=${COINBASE_PRIVATE_KEY}
      - LLM_OPENAI_API_KEY=${LLM_OPENAI_API_KEY}

      # Feature flags
      - ENABLE_WEBSOCKET=true
      - ENABLE_RISK_MANAGEMENT=true
      - ENABLE_METRICS=true
```

**Configuration Validation in Docker**:
```dockerfile
# Add to Dockerfile for configuration validation
RUN python -c "from bot.fp.types.config import Config; \
    result = Config.from_env(); \
    assert result.is_success(), f'Config validation failed: {result.failure() if result.is_failure() else None}'"
```

## Architecture

### Core Components
- **Functional Programming Layer** (`bot/fp/`): Type-safe functional programming foundation
  - **Core Types** (`bot/fp/types/`): Result/Either, Option, and domain types
  - **Configuration** (`bot/fp/types/config.py`): Functional configuration with Result-based validation
  - **Adapters** (`bot/fp/adapters/`): Compatibility layer for legacy systems
- **Data Layer** (`bot/data/market.py`): Real-time market data from exchange WebSocket/REST
  - **Functional Data Pipeline** (`bot/fp/data_pipeline.py`): Functional composition for data processing
- **Indicators** (`bot/indicators/vumanchu.py`): VuManChu Cipher A & B indicators
  - **Functional Indicators** (`bot/fp/indicators/`): Pure functional indicator implementations
- **LLM Agent** (`bot/strategy/llm_agent.py`): LangChain-powered trading decisions
  - **Functional LLM** (`bot/fp/strategies/llm_functional.py`): Functional LLM strategy implementation
- **Memory-Enhanced Agent** (`bot/strategy/memory_enhanced_agent.py`): LLM with learning from past trades
- **MCP Memory Server** (`bot/mcp/memory_server.py`): Persistent memory storage for experiences
- **Experience Manager** (`bot/learning/experience_manager.py`): Trade lifecycle tracking
- **Self-Improvement Engine** (`bot/learning/self_improvement.py`): Pattern analysis and strategy optimization
- **Validator** (`bot/validator.py`): JSON schema validation with fallback to HOLD
- **Risk Manager** (`bot/risk.py`): Position sizing, leverage control, loss limits
  - **Functional Risk** (`bot/fp/strategies/risk_management.py`): Functional risk management with Result types
- **Exchange Layer** (`bot/exchange/`): Multi-exchange support
  - **Base Interface** (`base.py`): Abstract exchange interface
  - **Coinbase** (`coinbase.py`): Centralized exchange implementation
  - **Bluefin** (`bluefin.py`): Decentralized exchange on Sui
  - **Factory** (`factory.py`): Exchange instantiation based on config
  - **Functional Adapters** (`bot/fp/adapters/`): FP adapters for exchanges
- **CLI** (`bot/main.py`): Command-line interface and orchestration
  - **Functional Runtime** (`bot/fp/runtime/cli.py`): Functional program interpretation

### Data Flow
**Legacy Flow**: Market Data → Indicators → LLM Agent (with Memory) → Validator → Risk Manager → Exchange → Experience Tracking → Learning

**Functional Flow**: Market Data Effects → Functional Indicators → Strategy Composition → Risk Validation → Exchange Effects → Result Aggregation

### Configuration System

#### Dual Configuration Architecture
The bot now supports both legacy and functional programming configuration patterns:

**Legacy Configuration (Backward Compatible)**:
- Environment variables in `.env` file
- JSON config files in `config/` directory (development.json, production.json, conservative_config.json)
- Pydantic models for type-safe settings in `bot/config.py`

**Functional Programming Configuration (Recommended)**:
- **Result-based validation** with `bot/fp/types/result.py` (Success/Failure)
- **Opaque types** for sensitive data (APIKey, PrivateKey) with automatic masking
- **Sum types** for strategy configuration (Momentum/MeanReversion/LLM)
- **Comprehensive validation** with detailed error messages
- **Environment variable builders** in `bot/fp/types/config.py`
- **Compatibility adapters** in `bot/fp/adapters/compatibility_layer.py`

#### Configuration Security Features
- **Opaque types** prevent accidental logging of sensitive data
- **Automatic credential masking** in logs and error messages
- **Result-based error handling** prevents configuration failures from crashing
- **Validation pipeline** with security checks and warnings

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

### Configuration Architecture Overview

The bot supports two configuration systems that can be used together:

1. **Legacy Configuration**: Original environment variables (backward compatible)
2. **Functional Programming Configuration**: New FP-based config with enhanced validation

### Quick Start Environment Variables

**Essential Configuration**:
```bash
# Trading Mode
SYSTEM__DRY_RUN=true  # false for live trading (DANGEROUS!)

# Exchange Selection
EXCHANGE__EXCHANGE_TYPE=coinbase  # or "bluefin"

# Trading Configuration
TRADING__SYMBOL=BTC-USD
TRADING__INTERVAL=1m
TRADING__LEVERAGE=5

# LLM Configuration
LLM__OPENAI_API_KEY=your_openai_api_key_here
LLM__MODEL_NAME=gpt-4
LLM__TEMPERATURE=0.1
```

### Exchange Configuration

#### Exchange Selection
- `EXCHANGE__EXCHANGE_TYPE` - Exchange to use: "coinbase" or "bluefin" (default: "coinbase")
- `EXCHANGE_TYPE` - **Functional**: Exchange type for FP config system

#### Coinbase Configuration
**Legacy Format**:
- `EXCHANGE__CDP_API_KEY_NAME` - Coinbase CDP API key name
- `EXCHANGE__CDP_PRIVATE_KEY` - Coinbase CDP private key (PEM format)

**Functional Format** (Enhanced Security):
- `COINBASE_API_KEY` - Coinbase API key (automatically masked in logs)
- `COINBASE_PRIVATE_KEY` - Coinbase private key (opaque type, never logged)
- `COINBASE_API_URL` - API endpoint (default: "https://api.coinbase.com")
- `COINBASE_WS_URL` - WebSocket endpoint (default: "wss://ws.coinbase.com")

#### Bluefin Configuration
**Legacy Format**:
- `EXCHANGE__BLUEFIN_PRIVATE_KEY` - Sui wallet private key (hex format)
- `EXCHANGE__BLUEFIN_NETWORK` - Network: "mainnet" or "testnet" (default: "mainnet")

**Functional Format** (Enhanced Security):
- `BLUEFIN_PRIVATE_KEY` - Sui wallet private key (opaque type, automatically masked)
- `BLUEFIN_NETWORK` - Network: "mainnet" or "testnet" (default: "mainnet")
- `BLUEFIN_RPC_URL` - RPC endpoint (default: "https://sui-mainnet.bluefin.io")

#### Rate Limiting Configuration
**Functional Format** (Consistent across exchanges):
- `RATE_LIMIT_RPS` - Requests per second (default: 10)
- `RATE_LIMIT_RPM` - Requests per minute (default: 100)
- `RATE_LIMIT_RPH` - Requests per hour (default: 1000)

#### Exchange Advanced Configuration
- `EXCHANGE__USE_TRADE_AGGREGATION` - Enable trade aggregation for sub-minute intervals (default: "true")
  - Required for trading intervals: 1s, 5s, 15s, 30s
  - When enabled, individual trades are aggregated into candles at the specified interval
  - Can be disabled for standard intervals (1m+) to reduce processing overhead

### Strategy Configuration

#### Strategy Selection
- `STRATEGY_TYPE` - **Functional**: Strategy type: "momentum", "mean_reversion", or "llm" (default: "llm")

#### LLM Strategy Configuration
**Legacy Format**:
- `LLM__OPENAI_API_KEY` - OpenAI API key
- `LLM__MODEL_NAME` - Model name (default: "gpt-4")
- `LLM__TEMPERATURE` - Temperature (default: 0.1)

**Functional Format** (Enhanced Validation):
- `LLM_MODEL` - Model name with validation (default: "gpt-4")
- `LLM_TEMPERATURE` - Temperature (0.0-2.0, default: 0.7)
- `LLM_MAX_CONTEXT` - Max context length (default: 4000)
- `LLM_USE_MEMORY` - Enable memory features (default: "false")
- `LLM_CONFIDENCE_THRESHOLD` - Confidence threshold (0.0-1.0, default: 0.7)

#### Momentum Strategy Configuration
**Functional Format**:
- `MOMENTUM_LOOKBACK` - Lookback period in candles (default: 20)
- `MOMENTUM_ENTRY_THRESHOLD` - Entry threshold percentage (default: 0.02)
- `MOMENTUM_EXIT_THRESHOLD` - Exit threshold percentage (default: 0.01)
- `MOMENTUM_USE_VOLUME` - Use volume confirmation (default: "true")

#### Mean Reversion Strategy Configuration
**Functional Format**:
- `MEAN_REVERSION_WINDOW` - Window size for mean calculation (default: 50)
- `MEAN_REVERSION_STD_DEV` - Standard deviations for entry (default: 2.0)
- `MEAN_REVERSION_MIN_VOL` - Minimum volatility threshold (default: 0.001)
- `MEAN_REVERSION_MAX_HOLD` - Maximum holding period (default: 100)

### Trading Configuration

#### Basic Trading Settings
**Legacy Format**:
- `TRADING__SYMBOL` - Trading pair (default: "BTC-USD")
- `TRADING__INTERVAL` - Candle interval for analysis (default: "1m")
- `TRADING__LEVERAGE` - Futures leverage (default: 5)

**Functional Format** (Enhanced Validation):
- `TRADING_PAIRS` - Comma-separated trading pairs (default: "BTC-USD")
- `TRADING_INTERVAL` - Candle interval with validation (default: "1m")
  - Standard intervals: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d
  - Sub-minute intervals: 1s, 5s, 15s, 30s (require `EXCHANGE__USE_TRADE_AGGREGATION=true`)
- `TRADING_MODE` - Trading mode: "paper", "live", or "backtest" (default: "paper")

### System Configuration

#### Core System Settings
**Legacy Format**:
- `SYSTEM__DRY_RUN` - Set to "false" for live trading, "true" for paper trading (default: "true")
- `SYSTEM__LOG_LEVEL` - Log level (default: "INFO")

**Functional Format** (Enhanced Features):
- `LOG_LEVEL` - Log level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" (default: "INFO")
- `MAX_CONCURRENT_POSITIONS` - Maximum concurrent positions (default: 3)
- `DEFAULT_POSITION_SIZE` - Default position size as decimal (default: 0.1)

#### Feature Flags
**Functional Format**:
- `ENABLE_WEBSOCKET` - Enable WebSocket connections (default: "true")
- `ENABLE_MEMORY` - Enable memory/learning features (default: "false")
- `ENABLE_BACKTESTING` - Enable backtesting capabilities (default: "true")
- `ENABLE_PAPER_TRADING` - Enable paper trading (default: "true")
- `ENABLE_RISK_MANAGEMENT` - Enable risk management (default: "true")
- `ENABLE_NOTIFICATIONS` - Enable notifications (default: "false")
- `ENABLE_METRICS` - Enable metrics collection (default: "true")

### Backtest Configuration

**Functional Format**:
- `BACKTEST_START_DATE` - Start date (ISO format, default: "2024-01-01")
- `BACKTEST_END_DATE` - End date (ISO format, default: "2024-12-31")
- `BACKTEST_INITIAL_CAPITAL` - Initial capital amount (default: 10000.0)
- `BACKTEST_CURRENCY` - Currency (default: "USD")
- `BACKTEST_MAKER_FEE` - Maker fee percentage (default: 0.001)
- `BACKTEST_TAKER_FEE` - Taker fee percentage (default: 0.002)
- `BACKTEST_SLIPPAGE` - Slippage percentage (default: 0.0005)
- `BACKTEST_USE_LIMIT_ORDERS` - Use limit orders (default: "true")

### Real Market Data
The bot exclusively uses real market data from exchanges - no mock data is generated:

All trading modes use live market data:

The trading system will:
- ✅ Use real-time market prices from your configured exchange for all operations
- ✅ Fetch historical data from live exchange APIs
- ✅ Process real WebSocket feeds for live price updates
- ✅ Calculate accurate fees, slippage, and margin requirements
- ✅ Provide realistic backtesting and paper trading simulation
- ⚠️ **Paper trading** (SYSTEM__DRY_RUN=true): Uses real prices, simulates trades
- ⚠️ **Live trading** (SYSTEM__DRY_RUN=false): Uses real prices, places real trades

Example paper trading log output:
```
🎯 PAPER TRADING DECISION: LONG | Symbol: SUI-PERP | Current Price: $3.45 | Size: 25%
📊 TRADE SIMULATION DETAILS:
  • Current Real Price: $3.45
  • Position Size: 724.64 SUI
  • Position Value: $2,500.00
  • Required Margin: $500.00
✅ PAPER TRADING EXECUTION COMPLETE:
  🎯 Action: LONG
  💵 Price: $3.4517
  🔴 Stop Loss: $3.28 (-$123.45)
  🟢 Take Profit: $3.62 (+$123.45)
```

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

## Python Code Quality Guidelines

### Linting and Security Tools
- **Ruff**: Fast modern linter for style and quality issues
- **Black**: Code formatting for consistent style
- **isort**: Import sorting and organization
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanner that detects:
  - Hardcoded passwords and secrets
  - SQL injection vulnerabilities
  - Shell injection risks
  - Insecure cryptographic practices
  - Use of dangerous functions
- **Vulture**: Dead code detection for:
  - Unused functions, classes, and variables
  - Unreachable code blocks
  - Unused imports

### Security Best Practices
- Run `bandit -r bot/ -ll` for low-confidence security scanning
- Generate security reports: `bandit -r bot/ -f json -o security-report.json`
- Never commit API keys, passwords, or sensitive data to version control
- Use environment variables for all secrets and configuration
- Validate all external inputs and API responses
- Follow principle of least privilege for file permissions and access

### Code Quality Pipeline
```bash
# Daily development workflow
poetry run black .
poetry run isort .
poetry run ruff check . --fix
poetry run bandit -r bot/ -ll
poetry run vulture bot/ --min-confidence 80
poetry run mypy bot/
poetry run pytest --cov=bot

# Pre-commit/CI complete pipeline
poetry run black . && \
poetry run isort . && \
poetry run ruff check . --fix && \
poetry run ruff format . && \
poetry run bandit -r bot/ -ll && \
poetry run vulture bot/ --min-confidence 80 && \
poetry run mypy bot/ && \
poetry run pytest --cov=bot
```

## Functional Programming Configuration Quick Reference

### Configuration System Summary

| Feature | Legacy System | Functional Programming System |
|---------|---------------|-------------------------------|
| **Error Handling** | Silent failures, crashes | Result types, explicit errors |
| **Security** | Basic env vars | Opaque types, auto-masking |
| **Validation** | Runtime errors | Compile-time + runtime validation |
| **Type Safety** | Pydantic models | Sum types, opaque types |
| **Compatibility** | Original system | Backward compatible adapter |
| **Performance** | Standard loading | Lazy loading, caching |
| **Testing** | Manual testing | Property-based testing |

### Environment Variable Quick Reference

```bash
# Essential Configuration
STRATEGY_TYPE=llm                    # Strategy: momentum, mean_reversion, llm
TRADING_MODE=paper                   # Mode: paper, live, backtest
EXCHANGE_TYPE=coinbase              # Exchange: coinbase, bluefin
TRADING_PAIRS=BTC-USD               # Trading pairs (comma-separated)
TRADING_INTERVAL=5m                 # Interval: 1m, 5m, 1h, etc.

# Security (automatically masked in logs)
COINBASE_API_KEY=your_api_key       # Will show as APIKey(***key)
COINBASE_PRIVATE_KEY=your_key        # Will show as PrivateKey(***)
LLM_OPENAI_API_KEY=your_openai_key  # OpenAI API key

# Risk Management
MAX_CONCURRENT_POSITIONS=3          # Max positions (1-10)
DEFAULT_POSITION_SIZE=0.1           # Position size (0.01-1.0)
ENABLE_RISK_MANAGEMENT=true         # Enable risk controls

# System Features
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
ENABLE_WEBSOCKET=true               # Real-time data
ENABLE_MEMORY=false                 # Learning features
ENABLE_METRICS=true                 # Performance metrics
```

### Configuration Profiles

```bash
# Conservative Profile
python -m bot.main live --profile conservative
# Sets: leverage=2, max_size_pct=10, temperature=0.05

# Aggressive Profile
python -m bot.main live --profile aggressive
# Sets: leverage=10, max_size_pct=50, temperature=0.2

# Balanced Profile (default)
python -m bot.main live --profile balanced
# Sets: leverage=5, max_size_pct=25, temperature=0.1
```

### Configuration Migration Checklist

- [ ] **Backup existing configuration** (.env files, config/ directory)
- [ ] **Update environment variables** to FP format (optional, both work)
- [ ] **Test configuration loading** with validation commands
- [ ] **Verify credentials are masked** in logs and error messages
- [ ] **Update deployment scripts** to use new validation
- [ ] **Test paper trading** before any live trading
- [ ] **Review security settings** and rate limits
- [ ] **Update monitoring** to use new configuration structure
