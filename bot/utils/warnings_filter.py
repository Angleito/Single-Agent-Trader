"""
Warning suppression utilities for the AI Trading Bot.

This module provides utilities to suppress specific deprecation warnings
and other non-critical warnings that come from third-party libraries,
particularly pandas_ta, LangChain, and their dependencies.
"""

import sys
import warnings


class WarningsFilter:
    """
    A utility class to manage warning suppression for third-party libraries.

    This class provides a centralized way to suppress known deprecation warnings
    and other non-critical warnings that pollute the application output.
    """

    # Known warning patterns to suppress
    PANDAS_TA_WARNINGS = [
        # pkg_resources deprecation warning
        r"pkg_resources is deprecated as an API",
        r"The pkg_resources package is slated for removal",
        r".*pkg_resources.*deprecated.*",
        r".*pkg_resources.*slated.*removal.*",
        # Invalid escape sequence warnings
        r"invalid escape sequence",
        r"invalid escape sequence '\\g'",
        r".*invalid escape sequence.*",
        # Other pandas_ta related warnings
        r"pandas_ta.*FutureWarning",
        r".*pandas_ta.*",
        # Setuptools warnings
        r"setup\.py install is deprecated",
        r".*setup\.py.*deprecated.*",
        # Docker-specific warnings
        r".*pkg_resources.*",
        r".*setuptools.*deprecated.*",
    ]

    # LangChain-specific warning patterns
    LANGCHAIN_WARNINGS = [
        # LangChain deprecation warnings
        r".*langchain.*deprecated.*",
        r".*langchain_core.*deprecated.*",
        r".*langchain_community.*deprecated.*",
        r".*langchain_openai.*deprecated.*",
        # LangChain configuration warnings
        r".*LangChain.*UserWarning.*",
        r".*temperature.*o3.*",
        r".*max_completion_tokens.*",
        # OpenAI integration warnings
        r".*openai.*deprecated.*",
        r".*pydantic.*deprecated.*",
        # LangChain model warnings
        r".*ChatOpenAI.*deprecated.*",
        r".*PromptTemplate.*deprecated.*",
    ]

    # Additional third-party library warnings
    GENERAL_WARNINGS = [
        # Setuptools warnings
        r"setup\.py install is deprecated",
        # Other common deprecation warnings
        r"numpy\..*is deprecated",
        r"pandas\..*is deprecated",
        # HTTP client warnings
        r".*urllib3.*deprecated.*",
        r".*requests.*deprecated.*",
        # Async warnings
        r".*asyncio.*deprecated.*",
    ]

    def __init__(self):
        """Initialize the warnings filter."""
        self._original_warnings_state = warnings.filters.copy()
        self._suppressed_patterns: list[str] = []

    def suppress_pandas_ta_warnings(self) -> None:
        """
        Suppress all known pandas_ta related warnings.

        This method adds warning filters for common pandas_ta deprecation
        warnings and invalid escape sequence warnings.
        """
        for pattern in self.PANDAS_TA_WARNINGS:
            warnings.filterwarnings("ignore", message=pattern, category=UserWarning)
            warnings.filterwarnings(
                "ignore", message=pattern, category=DeprecationWarning
            )
            warnings.filterwarnings("ignore", message=pattern, category=FutureWarning)
            warnings.filterwarnings("ignore", message=pattern, category=SyntaxWarning)
            self._suppressed_patterns.append(pattern)

    def suppress_langchain_warnings(self) -> None:
        """
        Suppress all known LangChain related warnings.

        This method adds warning filters for LangChain deprecation warnings,
        model configuration warnings, and OpenAI integration warnings.
        """
        for pattern in self.LANGCHAIN_WARNINGS:
            warnings.filterwarnings("ignore", message=pattern, category=UserWarning)
            warnings.filterwarnings(
                "ignore", message=pattern, category=DeprecationWarning
            )
            warnings.filterwarnings("ignore", message=pattern, category=FutureWarning)
            warnings.filterwarnings("ignore", message=pattern, category=SyntaxWarning)
            warnings.filterwarnings("ignore", message=pattern, category=RuntimeWarning)
            self._suppressed_patterns.append(pattern)

    def suppress_general_warnings(self) -> None:
        """
        Suppress general third-party library warnings.

        This method adds warning filters for common deprecation warnings
        from various third-party libraries used in the project.
        """
        for pattern in self.GENERAL_WARNINGS:
            warnings.filterwarnings(
                "ignore", message=pattern, category=DeprecationWarning
            )
            warnings.filterwarnings("ignore", message=pattern, category=FutureWarning)
            self._suppressed_patterns.append(pattern)

    def suppress_custom_pattern(
        self, pattern: str, categories: list[type[Warning]] | None = None
    ) -> None:
        """
        Suppress warnings matching a custom pattern.

        Args:
            pattern: Regular expression pattern to match warning messages
            categories: List of warning categories to suppress (default: common ones)
        """
        if categories is None:
            categories = [UserWarning, DeprecationWarning, FutureWarning, SyntaxWarning]

        for category in categories:
            warnings.filterwarnings("ignore", message=pattern, category=category)

        self._suppressed_patterns.append(pattern)

    def apply_all_filters(self) -> None:
        """
        Apply all warning filters at once.

        This is the main method to call during application startup to
        suppress all known problematic warnings.
        """
        self.suppress_pandas_ta_warnings()
        self.suppress_langchain_warnings()
        self.suppress_general_warnings()

        # Apply comprehensive module-level filtering
        self._apply_module_level_filters()

        # Apply comprehensive category-level filtering
        self._apply_category_level_filters()

    def _apply_module_level_filters(self) -> None:
        """
        Apply warning filters at the module level for known problematic modules.
        """
        problematic_modules = [
            "pkg_resources",
            "pandas_ta",
            "setuptools",
            "distutils",
            "importlib_metadata",
            "langchain",
            "langchain_core",
            "langchain_community",
            "langchain_openai",
            "openai",
            "pydantic",
        ]

        warning_categories = [
            UserWarning,
            DeprecationWarning,
            FutureWarning,
            SyntaxWarning,
            ImportWarning,
            RuntimeWarning,
        ]

        for module_name in problematic_modules:
            for category in warning_categories:
                warnings.filterwarnings("ignore", category=category, module=module_name)
                # Also catch submodules
                warnings.filterwarnings(
                    "ignore", category=category, module=f"{module_name}.*"
                )

    def _apply_category_level_filters(self) -> None:
        """
        Apply warning filters at the category level for specific message patterns.
        """
        # Comprehensive message patterns that should be suppressed
        message_patterns = [
            r".*pkg_resources.*",
            r".*deprecated.*API.*",
            r".*slated.*removal.*",
            r".*escape sequence.*",
            r".*setup\.py.*deprecated.*",
            r".*distutils.*deprecated.*",
            r".*importlib.*deprecated.*",
            r".*langchain.*deprecated.*",
            r".*openai.*deprecated.*",
            r".*pydantic.*deprecated.*",
            r".*temperature.*o3.*",
            r".*max_completion_tokens.*",
        ]

        warning_categories = [
            UserWarning,
            DeprecationWarning,
            FutureWarning,
            SyntaxWarning,
            ImportWarning,
            RuntimeWarning,
        ]

        for pattern in message_patterns:
            for category in warning_categories:
                warnings.filterwarnings("ignore", message=pattern, category=category)

    def restore_warnings(self) -> None:
        """
        Restore the original warnings state.

        This method can be used to restore warnings if needed for debugging.
        """
        # Reset to original state by recreating the list
        warnings.filters[:] = list(self._original_warnings_state)  # type: ignore[index]
        self._suppressed_patterns.clear()

    def get_suppressed_patterns(self) -> list[str]:
        """
        Get a list of all suppressed warning patterns.

        Returns:
            List of regex patterns that are being suppressed
        """
        return self._suppressed_patterns.copy()

    def show_suppressed_count(self) -> int:
        """
        Get the count of suppressed warning patterns.

        Returns:
            Number of warning patterns being suppressed
        """
        return len(self._suppressed_patterns)


# Global instance for easy access
warnings_filter = WarningsFilter()


def setup_warnings_suppression() -> None:
    """
    Convenience function to set up all warning suppressions.

    This function should be called early in the application startup
    process to ensure warnings are suppressed before any problematic
    imports occur.
    """
    warnings_filter.apply_all_filters()


def restore_warnings() -> None:
    """
    Convenience function to restore original warning state.

    Useful for debugging when you need to see all warnings.
    """
    warnings_filter.restore_warnings()


def suppress_pandas_ta_warnings() -> None:
    """
    Convenience function to suppress pandas_ta warnings.

    This function can be imported and called from anywhere in the application
    to suppress pandas_ta related warnings.
    """
    warnings_filter.suppress_pandas_ta_warnings()


def suppress_langchain_warnings() -> None:
    """
    Convenience function to suppress LangChain warnings.

    This function can be imported and called from anywhere in the application
    to suppress LangChain and OpenAI related warnings.
    """
    warnings_filter.suppress_langchain_warnings()


def suppress_pandas_ta_import_warnings() -> warnings.catch_warnings:
    """
    Context manager style function to temporarily suppress warnings
    during pandas_ta import.

    Usage:
        with suppress_pandas_ta_import_warnings():
            import pandas_ta as ta
    """
    return warnings.catch_warnings()


def init_all_suppressions() -> None:
    """
    Initialize all warning suppressions using the WarningsFilter class.

    This is the recommended function to call for complete warning suppression
    across the entire application. It applies pandas_ta, LangChain, and general
    library warning suppressions.
    """
    warnings_filter.apply_all_filters()


def initialize_early_warning_suppression() -> None:
    """
    Initialize comprehensive warning suppression that must happen before any library imports.

    This function should be called at the very beginning of the application,
    before importing any modules that might trigger deprecation warnings.
    This is specifically designed to work in Docker environments.
    """
    # Set warning registry to capture import-time warnings
    if not hasattr(sys.modules[__name__], "__warningregistry__"):
        sys.modules[__name__].__warningregistry__ = {}

    # Comprehensive message-based filtering including LangChain
    message_patterns = [
        r".*pkg_resources.*deprecated.*",
        r".*pkg_resources.*slated.*removal.*",
        r".*pkg_resources is deprecated as an API.*",
        r".*setup\.py.*deprecated.*",
        r".*invalid escape sequence.*",
        r".*distutils.*deprecated.*",
        r".*importlib.*deprecated.*",
        r".*setuptools.*deprecated.*",
        r".*langchain.*deprecated.*",
        r".*openai.*deprecated.*",
        r".*temperature.*o3.*",
        r".*max_completion_tokens.*",
    ]

    # All warning categories to suppress
    warning_categories = [
        UserWarning,
        DeprecationWarning,
        FutureWarning,
        SyntaxWarning,
        ImportWarning,
        RuntimeWarning,
    ]

    # Apply message-based filters
    for pattern in message_patterns:
        for category in warning_categories:
            warnings.filterwarnings("ignore", message=pattern, category=category)

    # Module-based filtering for problematic modules
    problematic_modules = [
        "pkg_resources",
        "pandas_ta",
        "setuptools",
        "distutils",
        "importlib_metadata",
        "setuptools._distutils",
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
        "openai",
        "pydantic",
    ]

    for module_name in problematic_modules:
        for category in warning_categories:
            # Exact module match
            warnings.filterwarnings("ignore", category=category, module=module_name)
            # Submodule match
            warnings.filterwarnings(
                "ignore", category=category, module=f"{module_name}.*"
            )
            # Regex match for dynamic module names
            warnings.filterwarnings(
                "ignore", category=category, module=f".*{module_name}.*"
            )

    # Catch-all for any remaining warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=".*pandas_ta.*")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module=".*pandas_ta.*"
    )
    warnings.filterwarnings("ignore", category=SyntaxWarning, module=".*pandas_ta.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=".*langchain.*")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module=".*langchain.*"
    )

    # Global catch-all patterns
    warnings.filterwarnings("ignore", message=r".*pkg_resources.*")
    warnings.filterwarnings("ignore", module=r".*pkg_resources.*")
    warnings.filterwarnings("ignore", module=r".*langchain.*")
