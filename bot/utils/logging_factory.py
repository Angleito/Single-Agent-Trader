"""
Centralized logging factory for consistent logging configuration with robust fallback support.

This module provides a unified interface for creating loggers across the entire application,
ensuring consistent behavior, proper fallback directory resolution, and secure filtering.
"""

import logging
import logging.handlers
import sys
from typing import Any

from bot.config import settings
from bot.utils.path_utils import get_logs_file_path
from bot.utils.secure_logging import SensitiveDataFilter


class LoggingFactory:
    """
    Factory for creating standardized loggers with fallback support.

    This factory ensures all loggers use:
    - Consistent formatting
    - Proper fallback directory resolution
    - Security filtering for sensitive data
    - Graceful degradation when file logging fails
    """

    _initialized = False
    _console_formatter = None
    _file_formatter = None
    _security_filter = None

    @classmethod
    def _initialize(cls) -> None:
        """Initialize shared formatters and filters."""
        if cls._initialized:
            return

        # Create formatters
        cls._console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        cls._file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create security filter
        cls._security_filter = SensitiveDataFilter()

        cls._initialized = True

    @classmethod
    def create_logger(
        cls,
        name: str,
        level: str | int | None = None,
        log_file: str | None = None,
        max_bytes: int = 50 * 1024 * 1024,  # 50MB
        backup_count: int = 5,
        enable_console: bool | None = None,
        enable_security_filter: bool = True,
    ) -> logging.Logger:
        """
        Create a standardized logger with fallback support.

        Args:
            name: Logger name
            level: Logging level (defaults to settings.system.log_level)
            log_file: Log file name (optional, relative to logs directory)
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            enable_console: Enable console logging (defaults to settings.system.log_to_console)
            enable_security_filter: Enable sensitive data filtering

        Returns:
            Configured logger instance
        """
        cls._initialize()

        # Get logger and clear existing handlers
        logger = logging.getLogger(name)
        logger.handlers.clear()

        # Set level
        if level is None:
            level = settings.system.log_level
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)

        # Determine console logging
        if enable_console is None:
            enable_console = settings.system.log_to_console

        # Add console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(level)
            console_handler.setFormatter(cls._console_formatter)

            if enable_security_filter:
                console_handler.addFilter(cls._security_filter)

            logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            file_handler = cls._create_file_handler(
                log_file=log_file,
                level=level,
                max_bytes=max_bytes,
                backup_count=backup_count,
                enable_security_filter=enable_security_filter,
            )

            if file_handler:
                logger.addHandler(file_handler)

        # Add security filter to logger itself if enabled
        if enable_security_filter:
            logger.addFilter(cls._security_filter)

        return logger

    @classmethod
    def _create_file_handler(
        cls,
        log_file: str,
        level: int,
        max_bytes: int,
        backup_count: int,
        enable_security_filter: bool,
    ) -> logging.Handler | None:
        """
        Create a file handler with fallback support.

        Args:
            log_file: Log file name
            level: Logging level
            max_bytes: Maximum file size
            backup_count: Number of backup files
            enable_security_filter: Enable security filtering

        Returns:
            File handler or None if creation failed
        """
        try:
            # Use centralized path resolution with fallback support
            log_path = get_logs_file_path(log_file)

            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )

            file_handler.setLevel(level)
            file_handler.setFormatter(cls._file_formatter)

            if enable_security_filter:
                file_handler.addFilter(cls._security_filter)

            # Test write access
            test_record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Logging initialization test",
                args=(),
                exc_info=None,
            )
            file_handler.emit(test_record)
            file_handler.flush()

            return file_handler

        except Exception as e:
            # Log to console about file logging failure
            console_logger = logging.getLogger("logging_factory")
            console_logger.warning(
                "Failed to create file handler for %s: %s. Continuing with console logging only.",
                log_file,
                e,
            )
            return None

    @classmethod
    def create_component_logger(
        cls,
        component_name: str,
        subcomponent: str | None = None,
        **kwargs: Any,
    ) -> logging.Logger:
        """
        Create a logger for a specific component with standardized naming.

        Args:
            component_name: Main component name (e.g., 'exchange', 'strategy')
            subcomponent: Optional subcomponent (e.g., 'coinbase', 'bluefin')
            **kwargs: Additional arguments passed to create_logger

        Returns:
            Configured logger instance
        """
        # Build logger name
        logger_name = f"bot.{component_name}"
        if subcomponent:
            logger_name += f".{subcomponent}"

        # Build log file name if not specified
        if "log_file" not in kwargs:
            log_file = f"{component_name}"
            if subcomponent:
                log_file += f"_{subcomponent}"
            log_file += ".log"
            kwargs["log_file"] = log_file

        return cls.create_logger(name=logger_name, **kwargs)

    @classmethod
    def create_trade_logger(cls, **kwargs: Any) -> logging.Logger:
        """Create a logger specifically for trading operations."""
        return cls.create_component_logger(
            component_name="trading",
            subcomponent="decisions",
            **kwargs,
        )

    @classmethod
    def create_exchange_logger(
        cls, exchange_name: str, **kwargs: Any
    ) -> logging.Logger:
        """Create a logger for exchange operations."""
        return cls.create_component_logger(
            component_name="exchange",
            subcomponent=exchange_name,
            **kwargs,
        )

    @classmethod
    def create_strategy_logger(
        cls, strategy_name: str, **kwargs: Any
    ) -> logging.Logger:
        """Create a logger for strategy operations."""
        return cls.create_component_logger(
            component_name="strategy",
            subcomponent=strategy_name,
            **kwargs,
        )

    @classmethod
    def create_mcp_logger(cls, **kwargs: Any) -> logging.Logger:
        """Create a logger for MCP operations."""
        return cls.create_component_logger(
            component_name="mcp",
            log_file="mcp/mcp.log",
            **kwargs,
        )

    @classmethod
    def create_performance_logger(cls, **kwargs: Any) -> logging.Logger:
        """Create a logger for performance monitoring."""
        return cls.create_component_logger(
            component_name="performance",
            log_file="performance.log",
            **kwargs,
        )


def get_logger(
    name: str,
    **kwargs: Any,
) -> logging.Logger:
    """
    Convenience function to get a logger using the factory.

    Args:
        name: Logger name
        **kwargs: Arguments passed to LoggingFactory.create_logger

    Returns:
        Configured logger instance
    """
    return LoggingFactory.create_logger(name=name, **kwargs)


def setup_root_logging() -> None:
    """
    Setup root logging configuration for the entire application.

    This should be called once at application startup to ensure
    consistent logging behavior across all components.
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Set level from settings
    level = getattr(logging, settings.system.log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Create main application logger
    main_logger = LoggingFactory.create_logger(
        name="bot",
        level=level,
        log_file="bot.log",
        enable_console=settings.system.log_to_console,
    )

    # Add a handler to root that forwards to our main logger
    class RootToMainHandler(logging.Handler):
        """Handler that forwards root logger messages to main logger."""

        def emit(self, record: logging.LogRecord) -> None:
            # Only forward if it's not already from our loggers
            if not record.name.startswith("bot."):
                main_logger.handle(record)

    root_handler = RootToMainHandler()
    root_handler.setLevel(level)
    root_logger.addHandler(root_handler)


def log_system_info() -> None:
    """Log system and configuration information."""
    logger = get_logger("bot.system")

    logger.info("=== AI Trading Bot Logging System Initialized ===")
    logger.info("Log Level: %s", settings.system.log_level)
    logger.info("Console Logging: %s", settings.system.log_to_console)
    logger.info("Dry Run Mode: %s", settings.system.dry_run)
    logger.info("Exchange: %s", settings.exchange.exchange_type)

    # Test path resolution
    from bot.utils.path_utils import get_logs_directory

    try:
        logs_dir = get_logs_directory()
        logger.info("Logs Directory: %s", logs_dir)
    except Exception as e:
        logger.warning("Logs directory resolution failed: %s", e)

    logger.info("=== Logging System Ready ===")


# Convenience functions for common logger types
def get_trade_logger(**kwargs: Any) -> logging.Logger:
    """Get a trade operations logger."""
    return LoggingFactory.create_trade_logger(**kwargs)


def get_exchange_logger(exchange_name: str, **kwargs: Any) -> logging.Logger:
    """Get an exchange operations logger."""
    return LoggingFactory.create_exchange_logger(exchange_name, **kwargs)


def get_strategy_logger(strategy_name: str, **kwargs: Any) -> logging.Logger:
    """Get a strategy operations logger."""
    return LoggingFactory.create_strategy_logger(strategy_name, **kwargs)


def get_mcp_logger(**kwargs: Any) -> logging.Logger:
    """Get an MCP operations logger."""
    return LoggingFactory.create_mcp_logger(**kwargs)
