# Type System Documentation

This directory contains type definitions and stubs for the AI Trading Bot project.

## Directory Structure

```
bot/types/
├── README.md           # This file
├── __init__.py        # Package marker
├── base_types.py      # Core type definitions
├── exceptions.py      # Custom exception types
├── guards.py          # Type guard functions
└── stubs/            # Type stubs for external libraries
    ├── __init__.py
    ├── aiohttp.pyi
    ├── ccxt.pyi
    ├── coinbase_advanced_py.pyi
    ├── docker.pyi
    ├── pandas_ta.pyi
    ├── psutil.pyi
    └── websockets.pyi
```

## Type Checking Configuration

### MyPy (pyproject.toml)
- **Strict mode enabled** with comprehensive checks
- **No implicit Any** - all types must be explicit
- **Full coverage** - untyped functions are errors
- **Per-module overrides** for gradual adoption

### Pyright (pyrightconfig.json)
- **Strict type checking mode**
- **Complete error reporting**
- **Custom stub path** configured
- **Multi-environment support**

## Type Stubs

Custom type stubs are provided for libraries without official type support:

### websockets.pyi
- WebSocket client/server protocols
- Connection management types
- Exception hierarchy

### pandas_ta.pyi
- Technical analysis indicators
- Overloaded functions for Series/DataFrame
- Common TA functions (RSI, EMA, MACD, etc.)

### coinbase_advanced_py.pyi
- REST client interface
- WebSocket client types
- Order/trade type definitions

### aiohttp.pyi
- Async HTTP client/server types
- Session management
- WebSocket support

### docker.pyi
- Container management
- Image operations
- Network/volume types

### psutil.pyi
- System monitoring types
- Process management
- Resource usage types

### ccxt.pyi
- Exchange base class
- Common exchange methods
- Exception hierarchy

## Usage Guidelines

### 1. Import Types
```python
from bot.types.base_types import OrderType, PositionSide
from bot.types.exceptions import TradingError
from bot.types.guards import is_valid_order
```

### 2. Type Annotations
```python
from typing import Optional, List, Dict, Any

async def place_order(
    symbol: str,
    side: PositionSide,
    amount: float,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """Place an order with full type safety."""
    ...
```

### 3. Type Guards
```python
from bot.types.guards import is_valid_price

def process_price(value: Any) -> float:
    if not is_valid_price(value):
        raise ValueError(f"Invalid price: {value}")
    return float(value)
```

### 4. Generic Types
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class TradingResult(Generic[T]):
    def __init__(self, value: T, success: bool) -> None:
        self.value = value
        self.success = success
```

## Running Type Checks

### Quick Check
```bash
poetry run mypy bot/
```

### Comprehensive Check
```bash
./scripts/test-type-checking.sh
```

### Pre-commit Hook
```bash
poetry run pre-commit run mypy --all-files
```

### IDE Integration
- **VS Code**: Install Pylance extension
- **PyCharm**: Built-in type checking
- **Vim/Neovim**: Use coc-pyright

## Common Issues and Solutions

### Missing Import
```python
# Error: Cannot find module 'some_module'
# Solution: Add to mypy overrides in pyproject.toml
[[tool.mypy.overrides]]
module = "some_module.*"
ignore_missing_imports = true
```

### Type Ignore
```python
# Use sparingly, document why
result = some_untyped_function()  # type: ignore[no-untyped-call]
```

### Gradual Typing
```python
# Start with simple annotations
def calculate(value: float) -> float:
    return value * 2

# Add complexity gradually
def calculate(
    value: float,
    multiplier: float = 2.0,
    precision: int = 2,
) -> float:
    return round(value * multiplier, precision)
```

## Best Practices

1. **Always annotate function signatures**
2. **Use Optional[] for nullable values**
3. **Prefer Union[] over Any**
4. **Create custom types for domain concepts**
5. **Use TypedDict for structured dicts**
6. **Leverage Literal[] for fixed values**
7. **Document complex type relationships**
8. **Use Protocol for duck typing**

## Contributing

When adding new types:
1. Place in appropriate module
2. Add comprehensive docstrings
3. Include usage examples
4. Update this README
5. Run type checks before committing
