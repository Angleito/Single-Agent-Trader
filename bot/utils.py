"""
Centralized utility functions for the AI Trading Bot.

This module provides core utility functions including comprehensive warning suppression
that works reliably in Docker environments and catches all pandas_ta related warnings.
"""

import warnings
import sys
from typing import List, Optional


def setup_warnings_suppression() -> None:
    """
    Set up comprehensive warning suppression for the AI Trading Bot.
    
    This function provides maximum warning suppression for pandas_ta and related
    dependencies, designed to work in both local and Docker environments.
    Should be called early in the application startup process.
    """
    # Set up warning registry for comprehensive capture
    if not hasattr(sys.modules[__name__], '__warningregistry__'):
        sys.modules[__name__].__warningregistry__ = {}
    
    # Comprehensive message patterns that cover all known problematic warnings
    message_patterns = [
        # pkg_resources warnings (most common)
        r".*pkg_resources.*deprecated.*",
        r".*pkg_resources.*slated.*removal.*", 
        r".*pkg_resources is deprecated as an API.*",
        r".*pkg_resources package is slated for removal.*",
        
        # Setup/build system warnings
        r".*setup\.py.*deprecated.*",
        r".*distutils.*deprecated.*",
        r".*setuptools.*deprecated.*",
        r".*importlib.*deprecated.*",
        
        # Escape sequence warnings
        r".*invalid escape sequence.*",
        r".*invalid escape sequence '\\g'.*",
        
        # General pandas_ta warnings
        r".*pandas_ta.*",
        
        # Catch-all patterns
        r".*deprecated as an API.*",
        r".*slated for removal.*"
    ]
    
    # All warning categories to suppress
    warning_categories = [
        UserWarning,
        DeprecationWarning,
        FutureWarning,
        SyntaxWarning,
        ImportWarning,
        RuntimeWarning
    ]
    
    # Apply message-based filters (most specific)
    for pattern in message_patterns:
        for category in warning_categories:
            warnings.filterwarnings(
                "ignore",
                message=pattern,
                category=category
            )
    
    # Module-based filtering for problematic modules
    problematic_modules = [
        "pkg_resources",
        "pandas_ta",
        "setuptools", 
        "distutils",
        "importlib_metadata",
        "setuptools._distutils",
        "_distutils_hack"
    ]
    
    for module_name in problematic_modules:
        for category in warning_categories:
            # Exact module match
            warnings.filterwarnings(
                "ignore",
                category=category,
                module=module_name
            )
            # Submodule match  
            warnings.filterwarnings(
                "ignore",
                category=category,
                module=f"{module_name}.*"
            )
            # Parent module match
            warnings.filterwarnings(
                "ignore",
                category=category,
                module=f".*{module_name}.*"
            )
    
    # Global catch-all filters (broadest suppression)
    warnings.filterwarnings("ignore", message=r".*pkg_resources.*")
    warnings.filterwarnings("ignore", module=r".*pkg_resources.*")
    warnings.filterwarnings("ignore", module=r".*pandas_ta.*")
    
    # Apply additional Docker-specific suppression
    _apply_docker_warning_suppression()


def _apply_docker_warning_suppression() -> None:
    """
    Apply additional warning suppression specifically for Docker environments.
    
    Docker environments may have different module loading behavior that requires
    additional warning suppression patterns.
    """
    # Docker-specific patterns that may not be caught by regular filters
    docker_patterns = [
        r".*site-packages.*pkg_resources.*",
        r".*dist-packages.*pkg_resources.*", 
        r".*egg-info.*deprecated.*",
        r".*\.egg.*deprecated.*"
    ]
    
    warning_categories = [UserWarning, DeprecationWarning, SyntaxWarning, ImportWarning]
    
    for pattern in docker_patterns:
        for category in warning_categories:
            warnings.filterwarnings("ignore", message=pattern, category=category)
    
    # Force ignore all warnings from any path containing problematic modules
    path_patterns = [
        r".*pkg_resources.*",
        r".*setuptools.*", 
        r".*distutils.*"
    ]
    
    for pattern in path_patterns:
        warnings.filterwarnings("ignore", filename=pattern)


def initialize_early_warning_suppression() -> None:
    """
    Initialize warning suppression that must happen before any library imports.
    
    This is the most aggressive warning suppression function that should be called
    at the very beginning of the application, before importing any modules that
    might trigger deprecation warnings. Specifically designed for Docker environments.
    """
    # Clear any existing warning registry to start fresh
    if hasattr(sys.modules[__name__], '__warningregistry__'):
        sys.modules[__name__].__warningregistry__.clear()
    
    # Set warnings to ignore by default for problematic categories
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", SyntaxWarning)
    
    # Apply setup_warnings_suppression for comprehensive coverage
    setup_warnings_suppression()
    
    # Add extra aggressive patterns for early import warnings
    early_patterns = [
        r".*",  # Nuclear option - catch everything (can be refined if needed)
    ]
    
    # Temporarily apply nuclear suppression during critical imports
    for pattern in early_patterns:
        warnings.filterwarnings("ignore", message=pattern, category=UserWarning)
        warnings.filterwarnings("ignore", message=pattern, category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=pattern, category=SyntaxWarning)


def restore_warnings() -> None:
    """
    Restore warnings to their default state.
    
    Useful for debugging when you need to see warnings.
    """
    warnings.resetwarnings()


def suppress_specific_warning(
    message_pattern: str, 
    categories: Optional[List[type]] = None,
    module_pattern: Optional[str] = None
) -> None:
    """
    Suppress a specific warning pattern.
    
    Args:
        message_pattern: Regular expression pattern to match warning messages
        categories: List of warning categories to suppress (defaults to common ones)
        module_pattern: Optional module pattern to match
    """
    if categories is None:
        categories = [UserWarning, DeprecationWarning, FutureWarning, SyntaxWarning]
    
    for category in categories:
        if module_pattern:
            warnings.filterwarnings(
                "ignore",
                message=message_pattern,
                category=category,
                module=module_pattern
            )
        else:
            warnings.filterwarnings(
                "ignore",
                message=message_pattern,
                category=category
            )


# Export key functions
__all__ = [
    "setup_warnings_suppression",
    "initialize_early_warning_suppression", 
    "restore_warnings",
    "suppress_specific_warning"
]