"""
Safe pandas_ta import wrapper that suppresses known deprecation warnings.

This module provides a centralized way to import pandas_ta while suppressing
the known deprecation warnings that pollute the application output.
Designed to work in Docker environments where warnings may persist.
"""

import warnings
import sys

# Set up warning registry for this module
if not hasattr(sys.modules[__name__], '__warningregistry__'):
    sys.modules[__name__].__warningregistry__ = {}

# Comprehensive pandas_ta import with maximum warning suppression
with warnings.catch_warnings():
    # Ignore all common warning categories
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", DeprecationWarning) 
    warnings.simplefilter("ignore", SyntaxWarning)
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", ImportWarning)
    
    # Specific message-based filters
    message_patterns = [
        ".*pkg_resources.*",
        ".*deprecated.*",
        ".*escape sequence.*",
        ".*setup.py.*",
        ".*distutils.*",
        ".*setuptools.*"
    ]
    
    for pattern in message_patterns:
        warnings.filterwarnings("ignore", message=pattern)
    
    # Module-based filters
    problematic_modules = [
        "pkg_resources", 
        "pandas_ta",
        "setuptools",
        "distutils", 
        "importlib_metadata"
    ]
    
    for module_name in problematic_modules:
        warnings.filterwarnings("ignore", module=module_name)
        warnings.filterwarnings("ignore", module=f"{module_name}.*")
        warnings.filterwarnings("ignore", module=f".*{module_name}.*")
    
    # Force reload warning filters
    warnings.resetwarnings()
    for pattern in message_patterns:
        warnings.filterwarnings("ignore", message=pattern)
    for module_name in problematic_modules:
        warnings.filterwarnings("ignore", module=module_name)
    
    # Now safely import pandas_ta
    import pandas_ta as ta

# Re-export ta for use by other modules
__all__ = ["ta"]