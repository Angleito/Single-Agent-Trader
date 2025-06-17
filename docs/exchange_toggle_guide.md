# Exchange Toggle Guide

This guide explains how to configure and switch between Coinbase (CEX) and Bluefin (DEX) exchanges in the AI Trading Bot.

## Table of Contents

1. [How the Exchange Toggle Works](#how-the-exchange-toggle-works)
2. [Step-by-Step Setup Instructions](#step-by-step-setup-instructions)
3. [Configuration Verification](#configuration-verification)
4. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
5. [Example .env Configurations](#example-env-configurations)
6. [Running Tests](#running-tests)
7. [Architecture Overview](#architecture-overview)

## How the Exchange Toggle Works

The exchange toggle feature allows seamless switching between centralized (Coinbase) and decentralized (Bluefin) exchanges through a single environment variable: `EXCHANGE__EXCHANGE_TYPE`.

### Key Components

1. **Exchange Factory**: `bot/exchange/factory.py` creates the appropriate exchange instance based on configuration
2. **Base Interface**: `bot/exchange/base.py` defines a common interface both exchanges implement
3. **Configuration**: `bot/config.py` contains exchange-specific settings
4. **Environment Variables**: Exchange selection and credentials are configured via `.env`

### Switching Process

When you change `EXCHANGE__EXCHANGE_TYPE` in your `.env` file:
- The factory automatically instantiates the correct exchange class
- All trading operations use the same interface regardless of exchange
- Risk management and strategy layers work identically with both exchanges

## Step-by-Step Setup Instructions

### Prerequisites

1. Python 3.12+ installed
2. Poetry for dependency management
3. API credentials for your chosen exchange

### Coinbase Setup

1. **Create Coinbase CDP API Keys**
   ```bash
   # Visit: https://portal.cdp.coinbase.com/
   # Create a new API key with trading permissions
   # Download the JSON credentials file
   ```

2. **Extract Credentials**
   - Open the downloaded JSON file
   - Copy the `name` field → This is your `CDP_API_KEY_NAME`
   - Copy the `privateKey` field → This is your `CDP_PRIVATE_KEY`

3. **Configure Environment**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add Coinbase credentials
   EXCHANGE__EXCHANGE_TYPE=coinbase
   EXCHANGE__CDP_API_KEY_NAME="your-key-name"
   EXCHANGE__CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----
   your-private-key-content
   -----END EC PRIVATE KEY-----"
   ```

### Bluefin Setup

1. **Create Sui Wallet**
   ```bash
   # Install Sui CLI (if not already installed)
   brew install sui
   
   # Create new wallet
   sui client new-address ed25519
   
   # Export private key
   sui keytool export --address <your-address>
   ```

2. **Fund Your Wallet**
   - For testnet: Use [Sui Testnet Faucet](https://faucet.sui.io/)
   - For mainnet: Transfer SUI tokens to your wallet address

3. **Configure Environment**
   ```bash
   # Edit .env and add Bluefin credentials
   EXCHANGE__EXCHANGE_TYPE=bluefin
   EXCHANGE__BLUEFIN_PRIVATE_KEY="0x1234...your-private-key"
   EXCHANGE__BLUEFIN_NETWORK=mainnet  # or testnet
   ```

### Common Configuration

Both exchanges require:
```bash
# OpenAI API key for LLM trading decisions
LLM__OPENAI_API_KEY="sk-..."

# Trading parameters
TRADING__SYMBOL=BTC-USD
TRADING__LEVERAGE=5
SYSTEM__DRY_RUN=true  # Set to false for live trading
```

## Configuration Verification

### Using the Verification Script

Run the exchange configuration verification script:

```bash
# Verify current configuration
python scripts/verify_exchange_config.py

# Test specific exchange
python scripts/verify_exchange_config.py --exchange coinbase
python scripts/verify_exchange_config.py --exchange bluefin

# Enable debug output
python scripts/verify_exchange_config.py --debug
```

### Expected Output

Successful Coinbase verification:
```
Exchange Configuration Verification
==================================

Current Exchange Type: coinbase

Checking Coinbase Configuration...
✓ CDP API Key Name is set
✓ CDP Private Key is set
✓ Private key format appears valid

Coinbase configuration is valid!
```

Successful Bluefin verification:
```
Exchange Configuration Verification
==================================

Current Exchange Type: bluefin

Checking Bluefin Configuration...
✓ Private key is set
✓ Network is set: mainnet
✓ Private key format appears valid

Bluefin configuration is valid!
```

### Manual Verification

You can also verify by checking exchange connectivity:

```bash
# Test exchange connection
python -c "
from bot.exchange.factory import ExchangeFactory
from bot.config import Settings
settings = Settings()
exchange = ExchangeFactory.create_exchange(settings.exchange)
print(f'Connected to {exchange.__class__.__name__}')
"
```

## Common Issues and Troubleshooting

### Issue 1: Invalid API Credentials

**Symptoms**: Authentication errors when starting the bot

**Coinbase Solution**:
```bash
# Verify key format - should be multi-line PEM
echo "$EXCHANGE__CDP_PRIVATE_KEY" | openssl ec -check

# Common fix: Ensure newlines are preserved
EXCHANGE__CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----
MHcCAQEE...your-key...
-----END EC PRIVATE KEY-----"
```

**Bluefin Solution**:
```bash
# Verify key format - should be 0x-prefixed hex
# Length should be 66 characters (0x + 64 hex chars)
echo "${#EXCHANGE__BLUEFIN_PRIVATE_KEY}"  # Should output 66

# Common fix: Add 0x prefix if missing
EXCHANGE__BLUEFIN_PRIVATE_KEY="0x1234567890abcdef..."
```

### Issue 2: Exchange Type Not Recognized

**Symptoms**: Error: "Unknown exchange type"

**Solution**:
```bash
# Check spelling (case-sensitive)
EXCHANGE__EXCHANGE_TYPE=coinbase  # Correct
EXCHANGE__EXCHANGE_TYPE=Coinbase  # Wrong

# Valid values: "coinbase" or "bluefin"
```

### Issue 3: Network Connectivity

**Symptoms**: Connection timeouts or network errors

**Coinbase Solution**:
```bash
# Test API endpoint
curl https://api.coinbase.com/api/v3/brokerage/accounts \
  -H "Authorization: Bearer $YOUR_JWT"

# Check firewall/proxy settings
```

**Bluefin Solution**:
```bash
# Test RPC endpoint
curl https://fullnode.mainnet.sui.io:443 \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"sui_getChainIdentifier","params":[]}'

# For testnet, use: https://fullnode.testnet.sui.io:443
```

### Issue 4: Insufficient Balance

**Symptoms**: "Insufficient funds" errors

**Solution**:
- Coinbase: Ensure USD balance for trading
- Bluefin: Ensure SUI balance for gas and USDC for trading

### Issue 5: Wrong Network (Bluefin)

**Symptoms**: Transactions fail or wallet not found

**Solution**:
```bash
# Ensure network matches your wallet
EXCHANGE__BLUEFIN_NETWORK=mainnet  # For production
EXCHANGE__BLUEFIN_NETWORK=testnet  # For testing
```

## Example .env Configurations

### Coinbase Configuration

```bash
# Exchange Selection
EXCHANGE__EXCHANGE_TYPE=coinbase

# Coinbase Credentials
EXCHANGE__CDP_API_KEY_NAME="organizations/abc123/apiKeys/def456"
EXCHANGE__CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIBJ6n3IpYEfKaem2cKnGDH0HJMqK3FmXurH8jTZge+ScoAoGCCqGSM49
AwEHoUQDQgAE4OPRndpBIBzNEo5a0R2l5bE9r6jEp5RqUPRPWPH0sfDk5x3o8H5i
I2R5XRRFc6OjqgkGDLwCuPZUmBQ8QHjYfg==
-----END EC PRIVATE KEY-----"

# Trading Configuration
LLM__OPENAI_API_KEY="sk-proj-abc123..."
TRADING__SYMBOL=BTC-USD
TRADING__LEVERAGE=5
TRADING__POSITION_SIZE=0.001
SYSTEM__DRY_RUN=true

# Risk Management
RISK__MAX_POSITION_SIZE=0.01
RISK__MAX_LEVERAGE=10
RISK__STOP_LOSS_PERCENTAGE=2.0
RISK__TAKE_PROFIT_PERCENTAGE=5.0

# Optional: Memory/Learning
MCP_ENABLED=false
```

### Bluefin Configuration

```bash
# Exchange Selection
EXCHANGE__EXCHANGE_TYPE=bluefin

# Bluefin Credentials
EXCHANGE__BLUEFIN_PRIVATE_KEY="0x4c0883a69102937d6231471b5dbb6204fe512961708279f9d3e4e72b8e1c9cb8"
EXCHANGE__BLUEFIN_NETWORK=mainnet

# Trading Configuration
LLM__OPENAI_API_KEY="sk-proj-xyz789..."
TRADING__SYMBOL=BTC-USD
TRADING__LEVERAGE=5
TRADING__POSITION_SIZE=100  # USDC
SYSTEM__DRY_RUN=true

# Risk Management
RISK__MAX_POSITION_SIZE=1000  # USDC
RISK__MAX_LEVERAGE=10
RISK__STOP_LOSS_PERCENTAGE=2.0
RISK__TAKE_PROFIT_PERCENTAGE=5.0

# Optional: Memory/Learning
MCP_ENABLED=false
```

### Testnet Configuration (Bluefin)

```bash
# For testing without real funds
EXCHANGE__EXCHANGE_TYPE=bluefin
EXCHANGE__BLUEFIN_NETWORK=testnet
EXCHANGE__BLUEFIN_PRIVATE_KEY="0xtest_private_key..."

# Use testnet faucet to get test tokens
# Visit: https://faucet.sui.io/
```

## Running Tests

### Unit Tests

Test individual exchange implementations:

```bash
# Run all exchange tests
poetry run pytest tests/unit/exchange/

# Test specific exchange
poetry run pytest tests/unit/exchange/test_coinbase.py
poetry run pytest tests/unit/exchange/test_bluefin.py

# Test exchange factory
poetry run pytest tests/unit/exchange/test_factory.py
```

### Integration Tests

Test end-to-end exchange operations:

```bash
# Run integration tests (requires valid credentials)
poetry run pytest tests/integration/test_exchange_integration.py

# Test specific exchange integration
poetry run pytest tests/integration/test_exchange_integration.py -k coinbase
poetry run pytest tests/integration/test_exchange_integration.py -k bluefin
```

### Mock Testing

Test without real exchange connections:

```bash
# Run tests with mocked exchanges
poetry run pytest tests/unit/ -m "not requires_exchange"

# Test strategy with mock exchange
poetry run pytest tests/unit/strategy/test_exchange_agnostic.py
```

### Coverage Report

```bash
# Generate coverage report
poetry run pytest --cov=bot.exchange --cov-report=html

# View report
open htmlcov/index.html
```

## Architecture Overview

### Exchange System Design

```
bot/exchange/
├── __init__.py
├── base.py          # Abstract base class defining exchange interface
├── coinbase.py      # Coinbase CEX implementation
├── bluefin.py       # Bluefin DEX implementation
└── factory.py       # Factory for creating exchange instances
```

### Key Interfaces

#### BaseExchange Abstract Class

```python
class BaseExchange(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to exchange"""
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Get account balances"""
    
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place a trading order"""
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
```

### Exchange Factory Pattern

The factory pattern enables:
- Runtime exchange selection based on configuration
- Consistent interface across different exchange types
- Easy addition of new exchanges
- Proper dependency injection

```python
# Factory usage
exchange = ExchangeFactory.create_exchange(settings.exchange)
# Returns either CoinbaseExchange or BluefinExchange instance
```

### Data Flow

1. **Configuration Loading**
   - Settings loaded from environment variables
   - Exchange type determined from `EXCHANGE__EXCHANGE_TYPE`

2. **Exchange Instantiation**
   - Factory creates appropriate exchange instance
   - Credentials passed to exchange constructor

3. **Trading Operations**
   - Strategy layer calls exchange methods
   - Same interface regardless of exchange type
   - Results normalized to common data structures

4. **Error Handling**
   - Exchange-specific errors caught and wrapped
   - Consistent error types across exchanges
   - Graceful fallbacks for network issues

### Security Considerations

1. **Credential Storage**
   - Never commit credentials to git
   - Use environment variables exclusively
   - Rotate keys regularly

2. **Network Security**
   - All API calls use HTTPS
   - Websocket connections use WSS
   - Request signing for authentication

3. **Error Handling**
   - Never log sensitive credentials
   - Sanitize error messages
   - Implement rate limiting

### Adding New Exchanges

To add support for a new exchange:

1. Create new file in `bot/exchange/`
2. Implement `BaseExchange` interface
3. Add configuration in `bot/config.py`
4. Update factory in `bot/exchange/factory.py`
5. Add tests in `tests/unit/exchange/`
6. Update this documentation

Example structure:
```python
# bot/exchange/new_exchange.py
from bot.exchange.base import BaseExchange

class NewExchange(BaseExchange):
    def __init__(self, config: NewExchangeConfig):
        self.config = config
    
    async def connect(self) -> None:
        # Implementation
        pass
    
    # Implement all abstract methods
```

### Performance Considerations

- Connection pooling for REST APIs
- Websocket reconnection logic
- Async/await for non-blocking operations
- Caching for frequently accessed data
- Rate limit compliance

### Monitoring and Logging

The exchange system includes:
- Detailed logging for debugging
- Performance metrics collection
- Error tracking and alerting
- Trade execution monitoring

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m bot.main live
```