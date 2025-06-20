"""
Path utilities for consistent fallback directory resolution.

This module provides utilities for resolving directory paths with fallback support,
ensuring the bot can handle permission issues gracefully by falling back to
alternative directories when needed.
"""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file's directory and walk up to find the project root
    current = Path(__file__).parent
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "bot").is_dir():
            return current
        current = current.parent
    # Fallback to current working directory
    return Path.cwd()


def _is_directory_writable(directory: Path) -> bool:
    """
    Check if a directory is writable by attempting to create a test file.

    Args:
        directory: The directory to test

    Returns:
        True if directory is writable, False otherwise
    """
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            return False

    try:
        # Try to create a temporary file to test write permissions
        test_file = directory / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False


def _resolve_directory_with_fallback(
    primary_path: Path, fallback_env_var: str, directory_name: str
) -> Path:
    """
    Resolve a directory path with fallback support.

    Args:
        primary_path: The primary directory path to try
        fallback_env_var: Environment variable for fallback directory
        directory_name: Human-readable name for logging

    Returns:
        Path object for the resolved directory

    Raises:
        OSError: If both primary and fallback directories are not writable
    """
    # Try primary path first
    if _is_directory_writable(primary_path):
        logger.debug(f"Using primary {directory_name} directory: {primary_path}")
        return primary_path

    # Check for fallback environment variable
    fallback_dir = os.getenv(fallback_env_var)
    if fallback_dir:
        fallback_path = Path(fallback_dir)
        if _is_directory_writable(fallback_path):
            logger.debug(f"Using fallback {directory_name} directory: {fallback_path}")
            return fallback_path
        else:
            logger.warning(
                f"Fallback {directory_name} directory is not writable: {fallback_path}"
            )

    # Try system temp directory as last resort
    try:
        temp_dir = (
            Path(tempfile.gettempdir()) / f"ai_trading_bot_{directory_name.lower()}"
        )
        if _is_directory_writable(temp_dir):
            logger.warning(
                f"Using system temp directory for {directory_name}: {temp_dir}"
            )
            return temp_dir
    except Exception as e:
        logger.error(f"Failed to create temp directory for {directory_name}: {e}")

    # If everything fails, raise an error
    raise OSError(
        f"Cannot find writable directory for {directory_name}. "
        f"Primary path: {primary_path}, "
        f"Fallback env var: {fallback_env_var}={fallback_dir}"
    )


def get_logs_directory() -> Path:
    """
    Get logs directory with fallback support.

    Returns:
        Path object for the logs directory

    Raises:
        OSError: If no writable logs directory can be found
    """
    project_root = _get_project_root()
    primary_path = project_root / "logs"
    return _resolve_directory_with_fallback(
        primary_path=primary_path,
        fallback_env_var="FALLBACK_LOGS_DIR",
        directory_name="logs",
    )


def get_data_directory() -> Path:
    """
    Get data directory with fallback support.

    Returns:
        Path object for the data directory

    Raises:
        OSError: If no writable data directory can be found
    """
    project_root = _get_project_root()
    primary_path = project_root / "data"
    return _resolve_directory_with_fallback(
        primary_path=primary_path,
        fallback_env_var="FALLBACK_DATA_DIR",
        directory_name="data",
    )


def get_config_directory() -> Path:
    """
    Get config directory with fallback support.

    Returns:
        Path object for the config directory

    Raises:
        OSError: If no writable config directory can be found
    """
    project_root = _get_project_root()
    primary_path = project_root / "config"
    return _resolve_directory_with_fallback(
        primary_path=primary_path,
        fallback_env_var="FALLBACK_CONFIG_DIR",
        directory_name="config",
    )


def get_tmp_directory() -> Path:
    """
    Get tmp directory with fallback support.

    Returns:
        Path object for the tmp directory

    Raises:
        OSError: If no writable tmp directory can be found
    """
    project_root = _get_project_root()
    primary_path = project_root / "tmp"
    return _resolve_directory_with_fallback(
        primary_path=primary_path,
        fallback_env_var="FALLBACK_TMP_DIR",
        directory_name="tmp",
    )


def get_mcp_memory_directory() -> Path:
    """
    Get MCP memory directory with fallback support.

    Returns:
        Path object for the MCP memory directory

    Raises:
        OSError: If no writable MCP memory directory can be found
    """
    data_dir = get_data_directory()
    primary_path = data_dir / "mcp_memory"
    return _resolve_directory_with_fallback(
        primary_path=primary_path,
        fallback_env_var="FALLBACK_MCP_MEMORY_DIR",
        directory_name="MCP memory",
    )


def get_mcp_logs_directory() -> Path:
    """
    Get MCP logs directory with fallback support.

    Returns:
        Path object for the MCP logs directory

    Raises:
        OSError: If no writable MCP logs directory can be found
    """
    logs_dir = get_logs_directory()
    primary_path = logs_dir / "mcp"
    return _resolve_directory_with_fallback(
        primary_path=primary_path,
        fallback_env_var="FALLBACK_MCP_DIR",
        directory_name="MCP logs",
    )


def get_bluefin_logs_directory() -> Path:
    """
    Get Bluefin logs directory with fallback support.

    Returns:
        Path object for the Bluefin logs directory

    Raises:
        OSError: If no writable Bluefin logs directory can be found
    """
    logs_dir = get_logs_directory()
    primary_path = logs_dir / "bluefin"
    return _resolve_directory_with_fallback(
        primary_path=primary_path,
        fallback_env_var="FALLBACK_BLUEFIN_DIR",
        directory_name="Bluefin logs",
    )


def get_omnisearch_cache_directory() -> Path:
    """
    Get OmniSearch cache directory with fallback support.

    Returns:
        Path object for the OmniSearch cache directory

    Raises:
        OSError: If no writable OmniSearch cache directory can be found
    """
    data_dir = get_data_directory()
    primary_path = data_dir / "omnisearch_cache"
    return _resolve_directory_with_fallback(
        primary_path=primary_path,
        fallback_env_var="FALLBACK_OMNISEARCH_CACHE_DIR",
        directory_name="OmniSearch cache",
    )


def ensure_directory_exists(directory: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: The directory path to ensure exists

    Returns:
        The directory path

    Raises:
        OSError: If the directory cannot be created
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    except (OSError, PermissionError) as e:
        raise OSError(f"Cannot create directory {directory}: {e}") from e


def get_safe_file_path(directory: Path, filename: str) -> Path:
    """
    Get a safe file path within a directory, ensuring the directory exists.

    Args:
        directory: The directory containing the file
        filename: The filename

    Returns:
        The full file path

    Raises:
        OSError: If the directory cannot be created or is not writable
    """
    ensure_directory_exists(directory)
    return directory / filename


# Convenience functions for commonly used file paths


def get_logs_file_path(filename: str) -> Path:
    """Get a file path in the logs directory."""
    return get_safe_file_path(get_logs_directory(), filename)


def get_data_file_path(filename: str) -> Path:
    """Get a file path in the data directory."""
    return get_safe_file_path(get_data_directory(), filename)


def get_config_file_path(filename: str) -> Path:
    """Get a file path in the config directory."""
    return get_safe_file_path(get_config_directory(), filename)


def get_tmp_file_path(filename: str) -> Path:
    """Get a file path in the tmp directory."""
    return get_safe_file_path(get_tmp_directory(), filename)


def get_mcp_memory_file_path(filename: str) -> Path:
    """Get a file path in the MCP memory directory."""
    return get_safe_file_path(get_mcp_memory_directory(), filename)


def get_mcp_logs_file_path(filename: str) -> Path:
    """Get a file path in the MCP logs directory."""
    return get_safe_file_path(get_mcp_logs_directory(), filename)


def get_bluefin_logs_file_path(filename: str) -> Path:
    """Get a file path in the Bluefin logs directory."""
    return get_safe_file_path(get_bluefin_logs_directory(), filename)


def get_omnisearch_cache_file_path(filename: str) -> Path:
    """Get a file path in the OmniSearch cache directory."""
    return get_safe_file_path(get_omnisearch_cache_directory(), filename)
