# Type Checking Guide for AI Trading Bot

This guide covers the comprehensive type checking setup for the AI Trading Bot project, ensuring type safety across the entire codebase.

## Overview

The project uses a multi-layered type checking approach:
- **MyPy**: Primary type checker with strict configuration
- **Pyright**: Secondary type checker for additional coverage
- **Custom Type Stubs**: For external libraries without type support
- **Pre-commit Hooks**: Automated type checking before commits

## Configuration Files

### 1. MyPy Configuration (pyproject.toml)
```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
```

### 2. Pyright Configuration (pyrightconfig.json)
```json
{
  "typeCheckingMode": "strict",
  "pythonVersion": "3.12",
  "stubPath": "bot/types/stubs",
  "reportMissingImports": true,
  "reportMissingTypeStubs": false
}
```

## Running Type Checks

### Quick Check
```bash
# MyPy check
poetry run mypy bot/

# Pyright check
pyright

# Both checks with detailed output
./scripts/test-type-checking.sh
```

### Pre-commit Hook
```bash
# Install hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run mypy --all-files
poetry run pre-commit run pyright --all-files
```

### Continuous Integration
```yaml
# In your CI/CD pipeline
- name: Type Check
  run: |
    poetry run mypy bot/ --config-file pyproject.toml
    pyright --project .
```

## Type Annotation Examples

### Basic Function Annotations
```python
from typing import Optional, List, Dict, Union, Tuple

def calculate_position_size(
    balance: float,
    risk_percentage: float,
    leverage: int = 1,
) -> float:
    """Calculate position size with type safety."""
    return (balance * risk_percentage / 100) * leverage

async def fetch_market_data(
    symbol: str,
    interval: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Union[str, float]]]:
    """Fetch market data with proper typing."""
    ...
```

### Class Annotations
```python
from typing import Optional, ClassVar
from datetime import datetime

class Trade:
    """Fully typed trade class."""

    # Class variables
    MAX_LEVERAGE: ClassVar[int] = 10

    def __init__(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        amount: float,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.price = price
        self.timestamp = timestamp or datetime.now()

    def calculate_value(self) -> float:
        """Calculate trade value."""
        return self.amount * self.price
```

### Generic Types
```python
from typing import TypeVar, Generic, Optional

T = TypeVar('T')

class Result(Generic[T]):
    """Generic result container."""

    def __init__(
        self,
        value: Optional[T] = None,
        error: Optional[str] = None,
    ) -> None:
        self.value = value
        self.error = error
        self._success = error is None

    @property
    def success(self) -> bool:
        """Check if result is successful."""
        return self._success

    def unwrap(self) -> T:
        """Unwrap value or raise exception."""
        if self.error:
            raise ValueError(self.error)
        if self.value is None:
            raise ValueError("No value present")
        return self.value
```

### Protocol Types
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Exchange(Protocol):
    """Protocol for exchange implementations."""

    name: str

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]: ...
```

### TypedDict for Structured Data
```python
from typing import TypedDict, Optional, List

class MarketData(TypedDict):
    """Typed dictionary for market data."""
    symbol: str
    price: float
    volume: float
    timestamp: int
    bid: float
    ask: float

class OrderResponse(TypedDict, total=False):
    """Order response with optional fields."""
    order_id: str
    status: str
    filled_amount: float
    remaining_amount: float
    fees: Optional[float]
    trades: Optional[List[Dict[str, Any]]]
```

## Common Type Patterns

### Optional Parameters
```python
# Good: Clear optional typing
def process_data(
    data: List[float],
    normalize: bool = False,
    scale: Optional[float] = None,
) -> List[float]:
    if scale is not None:
        data = [x * scale for x in data]
    if normalize:
        data = normalize_data(data)
    return data

# Bad: Implicit None
def process_data(data, normalize=False, scale=None):  # Missing types!
    ...
```

### Union Types
```python
from typing import Union

# Good: Explicit union
def parse_value(value: Union[str, int, float]) -> float:
    if isinstance(value, str):
        return float(value.strip())
    return float(value)

# Better: Use overload for clarity
from typing import overload

@overload
def parse_value(value: str) -> float: ...

@overload
def parse_value(value: Union[int, float]) -> float: ...

def parse_value(value: Union[str, int, float]) -> float:
    if isinstance(value, str):
        return float(value.strip())
    return float(value)
```

### Async Context Managers
```python
from typing import AsyncContextManager, Optional
from types import TracebackType

class AsyncClient:
    """Async client with proper typing."""

    async def __aenter__(self) -> "AsyncClient":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.disconnect()
```

## Type Guards

### Custom Type Guards
```python
from typing import Any, TypeGuard

def is_valid_price(value: Any) -> TypeGuard[float]:
    """Type guard for price validation."""
    try:
        price = float(value)
        return price > 0 and price < 1_000_000
    except (TypeError, ValueError):
        return False

def process_price(value: Any) -> float:
    """Process price with type narrowing."""
    if not is_valid_price(value):
        raise ValueError(f"Invalid price: {value}")
    # Type checker knows value is float here
    return round(value, 2)
```

### Using assert for Type Narrowing
```python
from typing import Optional

def get_required_value(value: Optional[str]) -> str:
    """Get required value with assertion."""
    assert value is not None, "Value is required"
    # Type checker knows value is str here
    return value.upper()
```

## Handling External Libraries

### Libraries with Type Support
```python
# These have built-in type support
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Use directly with full type safety
df: pd.DataFrame = pd.read_csv("data.csv")
arr: np.ndarray = np.array([1, 2, 3])
```

### Libraries without Type Support
```python
# Use custom stubs or type: ignore
import pandas_ta as ta  # type: ignore[import-untyped]

# Or create minimal stub
# bot/types/stubs/pandas_ta.pyi
def rsi(close: pd.Series, length: int = 14) -> pd.Series: ...
```

## Best Practices

### 1. Start with Function Signatures
Always type function parameters and return values first:
```python
def calculate_profit(
    entry_price: float,
    exit_price: float,
    amount: float,
    leverage: int = 1,
) -> float:
    return (exit_price - entry_price) * amount * leverage
```

### 2. Use Specific Types
```python
# Bad: Too generic
def process_data(data: Any) -> Any: ...

# Good: Specific types
def process_data(data: List[float]) -> np.ndarray: ...
```

### 3. Avoid Type Ignores
```python
# Bad: Suppressing errors
result = untyped_function()  # type: ignore

# Good: Add proper types or stubs
from typing import cast
result = cast(Dict[str, Any], untyped_function())
```

### 4. Document Complex Types
```python
from typing import TypeAlias, Dict, List, Union

# Type alias with documentation
OrderBook: TypeAlias = Dict[str, List[tuple[float, float]]]
"""Order book structure: {"bids": [(price, amount), ...], "asks": [...]}"""

def parse_orderbook(data: Dict[str, Any]) -> OrderBook:
    """Parse raw order book data into typed structure."""
    ...
```

### 5. Progressive Enhancement
Start simple and add complexity:
```python
# Phase 1: Basic types
def calculate(x: float, y: float) -> float: ...

# Phase 2: Add optionals
def calculate(x: float, y: float, precision: Optional[int] = None) -> float: ...

# Phase 3: Add overloads
@overload
def calculate(x: float, y: float) -> float: ...
@overload
def calculate(x: float, y: float, precision: int) -> Decimal: ...
```

## Troubleshooting

### Common Errors and Solutions

#### 1. Import Errors
```python
# Error: Cannot find module 'some_module'
# Solution 1: Add to mypy ignore list
# pyproject.toml
[[tool.mypy.overrides]]
module = "some_module.*"
ignore_missing_imports = true

# Solution 2: Create stub file
# bot/types/stubs/some_module.pyi
```

#### 2. Incompatible Types
```python
# Error: Incompatible types in assignment
# Fix: Use proper type conversions
# Bad
value: int = "123"  # Error!

# Good
value: int = int("123")
```

#### 3. Missing Type Parameters
```python
# Error: Missing type parameters for generic type
# Bad
results: List = []  # Missing parameter

# Good
results: List[str] = []
```

## Resources

- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pyright Documentation](https://github.com/microsoft/pyright)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 526 - Variable Annotations](https://peps.python.org/pep-0526/)
- [PEP 544 - Protocols](https://peps.python.org/pep-0544/)
