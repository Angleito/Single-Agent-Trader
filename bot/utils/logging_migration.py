"""
Migration utilities for upgrading components to use the centralized logging system.

This module provides utilities to help migrate existing logging code to use
the new centralized logging factory with proper fallback support.
"""

import logging
import re
from pathlib import Path
from typing import Any

from bot.utils.logging_factory import LoggingFactory


class LoggingMigrationHelper:
    """
    Helper class for migrating components to centralized logging.

    This class provides methods to identify and suggest improvements
    for logging code that could benefit from the centralized system.
    """

    @staticmethod
    def analyze_file_logging_patterns(file_path: Path) -> dict[str, Any]:
        """
        Analyze a Python file for logging patterns that could be improved.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary with analysis results and improvement suggestions
        """
        if not file_path.exists() or file_path.suffix != ".py":
            return {"error": "File does not exist or is not a Python file"}

        content = file_path.read_text(encoding="utf-8")

        analysis = {
            "file_path": str(file_path),
            "issues": [],
            "suggestions": [],
            "severity": "low",
        }

        # Check for hardcoded log file paths
        hardcoded_paths = re.findall(r'["\']([^"\']*\.log)["\']', content)
        if hardcoded_paths:
            analysis["issues"].append(
                {
                    "type": "hardcoded_paths",
                    "description": "Found hardcoded log file paths",
                    "paths": hardcoded_paths,
                    "severity": "high",
                }
            )
            analysis["suggestions"].append(
                "Replace hardcoded paths with centralized path resolution from bot.utils.path_utils"
            )
            analysis["severity"] = "high"

        # Check for direct FileHandler/RotatingFileHandler usage
        file_handlers = re.findall(
            r"(logging\.handlers\.)?(RotatingFileHandler|FileHandler)", content
        )
        if file_handlers:
            analysis["issues"].append(
                {
                    "type": "direct_file_handlers",
                    "description": "Found direct file handler creation",
                    "handlers": [handler[1] for handler in file_handlers],
                    "severity": "medium",
                }
            )
            analysis["suggestions"].append(
                "Use LoggingFactory.create_logger() for consistent file handler creation with fallback support"
            )
            if analysis["severity"] == "low":
                analysis["severity"] = "medium"

        # Check for logging.basicConfig usage
        if "logging.basicConfig" in content:
            analysis["issues"].append(
                {
                    "type": "basic_config",
                    "description": "Found logging.basicConfig usage",
                    "severity": "medium",
                }
            )
            analysis["suggestions"].append(
                "Replace logging.basicConfig with centralized logging configuration"
            )
            if analysis["severity"] == "low":
                analysis["severity"] = "medium"

        # Check for missing security filtering
        if "logging" in content and "SensitiveDataFilter" not in content:
            analysis["issues"].append(
                {
                    "type": "no_security_filter",
                    "description": "Logging present but no sensitive data filtering detected",
                    "severity": "medium",
                }
            )
            analysis["suggestions"].append(
                "Consider adding SensitiveDataFilter or use LoggingFactory which includes it by default"
            )

        # Check for proper error handling in logging setup
        setup_patterns = [
            "logging.getLogger",
            "addHandler",
            "setLevel",
            "setFormatter",
        ]
        has_logging_setup = any(pattern in content for pattern in setup_patterns)
        has_error_handling = any(
            pattern in content for pattern in ["try:", "except", "Exception"]
        )

        if has_logging_setup and not has_error_handling:
            analysis["issues"].append(
                {
                    "type": "no_error_handling",
                    "description": "Logging setup without proper error handling",
                    "severity": "low",
                }
            )
            analysis["suggestions"].append(
                "Add error handling for logging setup operations"
            )

        return analysis

    @staticmethod
    def generate_migration_code(
        component_name: str, logger_type: str = "component"
    ) -> str:
        """
        Generate example code for migrating to centralized logging.

        Args:
            component_name: Name of the component
            logger_type: Type of logger (component, trade, exchange, strategy, mcp)

        Returns:
            Example code string
        """
        if logger_type == "trade":
            return f"""
# BEFORE: Manual logging setup
import logging
logger = logging.getLogger("{component_name}")
# ... manual handler setup ...

# AFTER: Using centralized logging factory
from bot.utils.logging_factory import get_trade_logger

logger = get_trade_logger()
# Automatically includes security filtering, fallback support, and proper formatting
"""

        if logger_type == "exchange":
            return f"""
# BEFORE: Manual logging setup
import logging
logger = logging.getLogger("bot.exchange.{component_name}")

# AFTER: Using centralized logging factory
from bot.utils.logging_factory import get_exchange_logger

logger = get_exchange_logger("{component_name}")
# Automatically includes exchange-specific configuration and fallback support
"""

        if logger_type == "strategy":
            return f"""
# BEFORE: Manual logging setup
import logging
logger = logging.getLogger("bot.strategy.{component_name}")

# AFTER: Using centralized logging factory
from bot.utils.logging_factory import get_strategy_logger

logger = get_strategy_logger("{component_name}")
# Automatically includes strategy-specific configuration
"""

        if logger_type == "mcp":
            return """
# BEFORE: Manual logging setup
import logging
logger = logging.getLogger("bot.mcp")

# AFTER: Using centralized logging factory
from bot.utils.logging_factory import get_mcp_logger

logger = get_mcp_logger()
# Automatically includes MCP-specific configuration and file paths
"""

        # component
        return f"""
# BEFORE: Manual logging setup
import logging
logger = logging.getLogger("bot.{component_name}")

# AFTER: Using centralized logging factory
from bot.utils.logging_factory import LoggingFactory

logger = LoggingFactory.create_component_logger("{component_name}")
# Automatically includes proper fallback support, security filtering, and formatting
"""

    @staticmethod
    def create_upgrade_logger_function(
        component_name: str,
        file_log_name: str | None = None,
    ) -> logging.Logger:
        """
        Create a logger using the new centralized system for a specific component.

        This is a convenience function for immediate upgrades during migration.

        Args:
            component_name: Name of the component
            file_log_name: Optional specific log file name

        Returns:
            Configured logger with fallback support
        """
        return LoggingFactory.create_component_logger(
            component_name=component_name,
            log_file=file_log_name,
        )


def quick_logger_upgrade(
    name: str,
    log_file: str | None = None,
    **kwargs: Any,
) -> logging.Logger:
    """
    Quick upgrade function to replace manual logger creation.

    This function can be used as a drop-in replacement for manual
    logger creation to immediately gain fallback support.

    Args:
        name: Logger name
        log_file: Optional log file name
        **kwargs: Additional arguments for logger configuration

    Returns:
        Configured logger with centralized features

    Example:
        # Replace this:
        logger = logging.getLogger("my_component")

        # With this:
        logger = quick_logger_upgrade("my_component")
    """
    return LoggingFactory.create_logger(name=name, log_file=log_file, **kwargs)


def analyze_codebase_logging(
    base_path: Path,
    file_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze the entire codebase for logging improvement opportunities.

    Args:
        base_path: Base directory to analyze
        file_patterns: List of file patterns to include (default: ['*.py'])

    Returns:
        Comprehensive analysis results
    """
    if file_patterns is None:
        file_patterns = ["*.py"]

    results = {
        "summary": {
            "total_files": 0,
            "files_with_issues": 0,
            "high_severity_files": 0,
            "medium_severity_files": 0,
        },
        "files": [],
        "common_issues": {},
    }

    # Find all Python files
    python_files = []
    for pattern in file_patterns:
        python_files.extend(base_path.rglob(pattern))

    results["summary"]["total_files"] = len(python_files)

    # Analyze each file
    for file_path in python_files:
        analysis = LoggingMigrationHelper.analyze_file_logging_patterns(file_path)

        if "error" not in analysis and analysis["issues"]:
            results["summary"]["files_with_issues"] += 1

            if analysis["severity"] == "high":
                results["summary"]["high_severity_files"] += 1
            elif analysis["severity"] == "medium":
                results["summary"]["medium_severity_files"] += 1

            results["files"].append(analysis)

            # Track common issues
            for issue in analysis["issues"]:
                issue_type = issue["type"]
                if issue_type not in results["common_issues"]:
                    results["common_issues"][issue_type] = 0
                results["common_issues"][issue_type] += 1

    return results


# Example usage and testing functions
def demo_migration_examples() -> None:
    """Demonstrate migration examples for different component types."""
    helper = LoggingMigrationHelper()

    print("=== Logging Migration Examples ===")
    print("\n1. Trading Component:")
    print(helper.generate_migration_code("trade_executor", "trade"))

    print("\n2. Exchange Component:")
    print(helper.generate_migration_code("coinbase", "exchange"))

    print("\n3. Strategy Component:")
    print(helper.generate_migration_code("llm_agent", "strategy"))

    print("\n4. MCP Component:")
    print(helper.generate_migration_code("memory", "mcp"))


if __name__ == "__main__":
    # Run demo if executed directly
    demo_migration_examples()
