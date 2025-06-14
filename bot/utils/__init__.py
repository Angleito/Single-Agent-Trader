"""
Utility modules for the AI Trading Bot.

This package contains various utility functions and classes used throughout
the trading bot application.
"""

# Import warnings suppression utilities (always available)
from .warnings_filter import (
    init_all_suppressions,
    initialize_early_warning_suppression,
    restore_warnings,
    setup_warnings_suppression,
    suppress_langchain_warnings,
    suppress_pandas_ta_warnings,
    warnings_filter,
)

# Import ta with graceful fallback if pandas_ta is not available
try:
    from .ta_import import ta

    _ta_available = True
except ImportError:
    ta = None
    _ta_available = False

__all__ = [
    "init_all_suppressions",
    "initialize_early_warning_suppression",
    "restore_warnings",
    "setup_warnings_suppression",
    "suppress_langchain_warnings",
    "suppress_pandas_ta_warnings",
    "warnings_filter",
]

# Only add ta to __all__ if it's available
if _ta_available:
    __all__.append("ta")
