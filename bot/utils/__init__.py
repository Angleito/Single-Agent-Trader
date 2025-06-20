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

# Import web search formatter
try:
    from .web_search_formatter import (
        ContentPriority,
        FormattedContent,
        WebSearchFormatter,
    )

    _web_search_formatter_available = True
except ImportError:
    WebSearchFormatter = None
    ContentPriority = None
    FormattedContent = None
    _web_search_formatter_available = False

# Import path utilities
try:
    from .path_utils import (
        ensure_directory_exists,
        get_bluefin_logs_directory,
        get_bluefin_logs_file_path,
        get_config_directory,
        get_config_file_path,
        get_data_directory,
        get_data_file_path,
        get_logs_directory,
        get_logs_file_path,
        get_mcp_logs_directory,
        get_mcp_logs_file_path,
        get_mcp_memory_directory,
        get_mcp_memory_file_path,
        get_omnisearch_cache_directory,
        get_omnisearch_cache_file_path,
        get_safe_file_path,
        get_tmp_directory,
        get_tmp_file_path,
    )

    _path_utils_available = True
except ImportError:
    _path_utils_available = False

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

# Only add web search formatter to __all__ if it's available
if _web_search_formatter_available:
    __all__.extend(["ContentPriority", "FormattedContent", "WebSearchFormatter"])

# Only add path utilities to __all__ if they're available
if _path_utils_available:
    __all__.extend(
        [
            "ensure_directory_exists",
            "get_bluefin_logs_directory",
            "get_bluefin_logs_file_path",
            "get_config_directory",
            "get_config_file_path",
            "get_data_directory",
            "get_data_file_path",
            "get_logs_directory",
            "get_logs_file_path",
            "get_mcp_logs_directory",
            "get_mcp_logs_file_path",
            "get_mcp_memory_directory",
            "get_mcp_memory_file_path",
            "get_omnisearch_cache_directory",
            "get_omnisearch_cache_file_path",
            "get_safe_file_path",
            "get_tmp_directory",
            "get_tmp_file_path",
        ]
    )
