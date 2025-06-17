# pandas_ta Deprecation Warnings Fix

## Problem Summary

The AI trading bot was displaying deprecation warnings from the `pandas_ta` library:

1. **pkg_resources deprecation**: `pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.`

2. **Invalid escape sequence**: Potential warnings about invalid regex escape sequences in pandas_ta code

## Root Cause

- `pandas_ta` version 0.3.14b (last updated July 2021) uses deprecated APIs
- The library internally imports `pkg_resources` which triggers deprecation warnings
- No newer stable version available to fix these warnings

## Solution Implemented

### 1. Warning Suppression Utility (`bot/utils/warnings_filter.py`)

Created a comprehensive warning filtering system with:
- Specific patterns for pandas_ta warnings
- General third-party library warning suppression
- Configurable warning categories
- Ability to restore original warning state for debugging

### 2. Safe Import Wrapper (`bot/utils/ta_import.py`)

Created a centralized import wrapper that:
- Suppresses warnings during pandas_ta import
- Uses context manager to temporarily disable warnings
- Re-exports `ta` module for consistent usage across the codebase

### 3. Updated Import Strategy

Modified all indicator files to use the safe import:
- `bot/indicators/vumanchu.py`
- `bot/indicators/stochastic_rsi.py`
- `bot/indicators/ema_ribbon.py`
- `bot/indicators/wavetrend.py`
- `bot/indicators/cipher_a_signals.py`
- `bot/indicators/cipher_b_signals.py`
- `bot/indicators/schaff_trend_cycle.py`

Changed from:
```python
import pandas_ta as ta
```

To:
```python
from ..utils import ta
```

### 4. Early Warning Suppression in Main (`bot/main.py`)

Added warning suppression at the very beginning of the main module:
```python
import warnings

# Suppress pandas_ta and pkg_resources warnings early
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*pkg_resources package is slated for removal.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*invalid escape sequence.*", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")
```

## Files Modified

1. **New Files Created**:
   - `/bot/utils/warnings_filter.py` - Comprehensive warning suppression utility
   - `/bot/utils/ta_import.py` - Safe pandas_ta import wrapper
   - `/docs/technical_analysis_alternatives.md` - Research on alternative libraries
   - `/docs/deprecation_warnings_fix.md` - This documentation

2. **Modified Files**:
   - `/bot/main.py` - Added early warning suppression and utility import
   - `/bot/utils/__init__.py` - Added new utilities to exports
   - All indicator files - Updated to use safe import

## Testing Results

### Before Fix
```bash
$ poetry run python -c "import pandas_ta as ta"
/Users/angel/Library/Caches/pypoetry/virtualenvs/ai-trading-bot-4Xh8djmJ-py3.13/lib/python3.13/site-packages/pandas_ta/__init__.py:7: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import get_distribution, DistributionNotFound
```

### After Fix
```bash
$ poetry run python -c "from bot.main import cli; print('✅ Import successful')"
✅ Import successful

$ poetry run ai-trading-bot --help
Usage: ai-trading-bot [OPTIONS] COMMAND [ARGS]...

  AI Trading Bot - LangChain-powered crypto futures trading.
```

## Long-term Recommendations

1. **Monitor for Updates**: Check for newer pandas_ta versions periodically
2. **Consider Migration**: Evaluate PyTA or pandas-ta-openbb as replacements
3. **Review Alternatives**: Full documentation provided in `technical_analysis_alternatives.md`

## Debugging

To temporarily restore warnings for debugging:
```python
from bot.utils import restore_warnings
restore_warnings()
```

## Maintenance Notes

- Warning suppression is comprehensive and should handle future similar warnings
- The safe import pattern can be extended to other problematic libraries
- No functional changes to trading logic - only warning suppression
- All existing tests should pass unchanged
