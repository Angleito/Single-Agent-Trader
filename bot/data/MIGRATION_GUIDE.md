# Market Data Provider Migration Guide

## Overview

The market data provider system has been refactored to use a unified architecture that reduces code duplication while maintaining all functionality. The new system provides:

1. **AbstractMarketDataProvider** - Base class with common functionality
2. **Exchange-specific providers** - CoinbaseMarketDataProvider and BluefinMarketDataProvider
3. **Factory pattern** - Automatic provider selection based on exchange type
4. **Backward compatibility** - Existing code continues to work without changes

## Architecture Changes

### Before
```
bot/data/
├── market.py          # Coinbase-specific implementation (2400+ lines)
├── bluefin_market.py  # Bluefin-specific implementation (3400+ lines)
└── __init__.py
```

### After
```
bot/data/
├── base_market_provider.py    # Abstract base class (common functionality)
├── providers/
│   ├── coinbase_provider.py  # Coinbase-specific logic only
│   └── bluefin_provider.py   # Bluefin-specific logic only
├── factory.py                 # Provider factory
├── market.py                  # Legacy Coinbase implementation (preserved)
├── bluefin_market.py         # Legacy Bluefin implementation (preserved)
└── __init__.py               # Exports with backward compatibility
```

## Usage Examples

### Basic Usage (Backward Compatible)

```python
# This still works exactly as before
from bot.data import MarketDataProvider

# Automatically creates the correct provider based on settings
provider = MarketDataProvider()  # Uses settings.exchange.exchange_type
```

### Explicit Exchange Selection

```python
from bot.data import create_market_data_provider

# Create Coinbase provider
coinbase_provider = create_market_data_provider(
    exchange_type="coinbase",
    symbol="BTC-USD",
    interval="5m"
)

# Create Bluefin provider
bluefin_provider = create_market_data_provider(
    exchange_type="bluefin",
    symbol="ETH-PERP",
    interval="1m"
)
```

### Using Base Class for Type Hints

```python
from bot.data import AbstractMarketDataProvider

async def process_market_data(provider: AbstractMarketDataProvider) -> None:
    """Works with any market data provider."""
    await provider.connect()
    data = await provider.fetch_historical_data()
    current_price = await provider.fetch_latest_price()
```

## Benefits

1. **Reduced Code Duplication**: Common functionality (caching, WebSocket management, subscribers) is implemented once in the base class.

2. **Easier Maintenance**: Bug fixes and improvements in common functionality automatically apply to all providers.

3. **Consistent Interface**: All providers share the same interface, making it easy to switch between exchanges.

4. **Type Safety**: Using AbstractMarketDataProvider for type hints ensures compatibility.

5. **Extensibility**: Adding new exchanges is straightforward - just create a new provider class.

## Migration Checklist

- [x] No changes needed for existing code using `MarketDataProvider`
- [x] The factory automatically selects the correct provider based on `settings.exchange.exchange_type`
- [x] All functionality is preserved (WebSocket, REST API, caching, etc.)
- [x] Legacy imports continue to work for backward compatibility

## Implementation Details

### Common Functionality (Base Class)
- Data caching with TTL
- WebSocket connection management
- Subscriber pattern for real-time updates
- Background task management
- Message queue processing
- Data validation
- Connection status tracking

### Exchange-Specific Functionality
- API endpoint configuration
- Authentication methods
- Message parsing
- Symbol conversion
- Interval formatting
- Price decimal conversion (Bluefin)

## Testing

Run the following to verify the new system:

```bash
# Test Coinbase provider
EXCHANGE__EXCHANGE_TYPE=coinbase python -m pytest tests/unit/test_market_data.py

# Test Bluefin provider  
EXCHANGE__EXCHANGE_TYPE=bluefin python -m pytest tests/unit/test_market_data.py
```