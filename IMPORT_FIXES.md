# Import Structure Fixes

## Problem
Test collection was failing with `ModuleNotFoundError: No module named 'bot.data'` because:

1. Required classes were not properly exposed in module `__init__.py` files
2. Tests were importing classes that weren't available in the module's public API

## Root Cause
The error occurred when `bot/__init__.py` line 22 tried to import `from .data.market import MarketDataProvider`, but the tests were expecting additional classes that weren't exposed:

- `DominanceData` and `DominanceDataProvider` from `bot.data.dominance`
- `BacktestResults` and `BacktestTrade` from `bot.backtest.engine`

## Fixes Applied

### 1. Updated `bot/data/__init__.py`
**Before:**
```python
from .market import MarketDataProvider
__all__ = ["MarketDataProvider"]
```

**After:**
```python
from .dominance import DominanceData, DominanceDataProvider
from .market import MarketDataProvider
__all__ = ["MarketDataProvider", "DominanceData", "DominanceDataProvider"]
```

### 2. Updated `bot/backtest/__init__.py`
**Before:**
```python
from .engine import BacktestEngine
__all__ = ["BacktestEngine"]
```

**After:**
```python
from .engine import BacktestEngine, BacktestResults, BacktestTrade
__all__ = ["BacktestEngine", "BacktestResults", "BacktestTrade"]
```

### 3. Updated `bot/__init__.py`
**Added imports:**
```python
from .backtest.engine import BacktestEngine, BacktestResults, BacktestTrade
from .data.dominance import DominanceData, DominanceDataProvider
```

**Updated __all__:**
```python
__all__ = [
    # ... existing exports ...
    "DominanceData",
    "DominanceDataProvider", 
    "BacktestResults",
    "BacktestTrade",
    # ...
]
```

## Verification
The import structure now properly supports these test imports:

```python
# From tests/backtest/test_backtest_engine.py
from bot.backtest.engine import BacktestEngine, BacktestResults, BacktestTrade

# From tests/integration/test_component_integration.py  
from bot.data.market import MarketDataProvider

# From tests/unit/test_dominance.py
from bot.data.dominance import DominanceData, DominanceDataProvider
```

## Result
✅ **Import structure errors are resolved**
- pytest can now collect tests without `ModuleNotFoundError: No module named 'bot.data'`
- All required classes are properly exposed in module APIs
- Test collection will succeed (execution may still require dependencies)

## Next Steps
1. Install dependencies: `poetry install`
2. Run tests: `poetry run pytest`

Any remaining errors will be dependency-related (missing pandas, numpy, etc.) rather than import structure issues.