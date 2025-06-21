"""
Advanced logging configuration with robust fallback strategies and error recovery.

This module provides comprehensive logging configuration management that handles
Docker volume permission issues, fallback directory resolution, and graceful
degradation when primary logging mechanisms fail.
"""

import logging
import logging.config
import os
import tempfile
from pathlib import Path
from typing import Any

from bot.config import settings
from bot.utils.path_utils import get_logs_directory


class RobustLoggingConfig:
    """
    Robust logging configuration that handles permission failures gracefully.

    This class manages logging configuration with multiple fallback strategies:
    1. Primary logs directory (from path_utils with fallback support)
    2. Environment variable fallback directories
    3. System temporary directory
    4. Console-only logging as final fallback
    """

    def __init__(self):
        self.config_applied = False
        self.fallback_mode = False
        self.logs_directory = None
        self.available_handlers = []

    def setup_logging(self) -> dict[str, Any]:
        """
        Setup comprehensive logging configuration with fallback support.

        Returns:
            Dictionary with setup results and diagnostics
        """
        setup_results = {
            "status": "success",
            "logs_directory": None,
            "handlers_created": [],
            "fallback_mode": False,
            "warnings": [],
            "errors": [],
        }

        try:
            # Step 1: Resolve logs directory with fallback support
            logs_dir = self._resolve_logs_directory()
            setup_results["logs_directory"] = str(logs_dir)
            self.logs_directory = logs_dir

            # Step 2: Create logging configuration
            config = self._create_logging_config(logs_dir)

            # Step 3: Apply configuration with error handling
            self._apply_config_safely(config, setup_results)

            # Step 4: Verify handlers are working
            self._verify_handlers(setup_results)

            self.config_applied = True

        except Exception as e:
            setup_results["status"] = "fallback"
            setup_results["errors"].append(f"Primary setup failed: {e}")
            self._setup_emergency_logging(setup_results)

        return setup_results

    def _resolve_logs_directory(self) -> Path:
        """
        Resolve logs directory using multiple fallback strategies.

        Returns:
            Path to usable logs directory

        Raises:
            OSError: If no writable directory can be found
        """
        try:
            # Use the centralized path resolution system
            return get_logs_directory()
        except Exception as e:
            # If centralized system fails, try manual fallback resolution
            logging.getLogger(__name__).warning(
                "Centralized logs directory resolution failed: %s. Trying manual fallback.",
                e,
            )

            return self._manual_fallback_resolution()

    def _manual_fallback_resolution(self) -> Path:
        """
        Manual fallback directory resolution when centralized system fails.

        Returns:
            Path to usable directory

        Raises:
            OSError: If no directory can be found
        """
        # Try standard logs directory
        standard_logs = Path("logs")
        if self._test_directory_writable(standard_logs):
            return standard_logs

        # Try fallback environment variables
        for env_var in ["FALLBACK_LOGS_DIR", "FALLBACK_DIR", "TEMP_LOGS_DIR"]:
            fallback_dir = os.getenv(env_var)
            if fallback_dir:
                fallback_path = Path(fallback_dir)
                if self._test_directory_writable(fallback_path):
                    self.fallback_mode = True
                    return fallback_path

        # Try system temp directory
        temp_dir = Path(tempfile.gettempdir()) / "ai_trading_bot_logs"
        if self._test_directory_writable(temp_dir):
            self.fallback_mode = True
            return temp_dir

        raise OSError("No writable directory found for logs")

    def _test_directory_writable(self, directory: Path) -> bool:
        """Test if a directory is writable."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            test_file = directory / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False

    def _create_logging_config(self, logs_dir: Path) -> dict[str, Any]:
        """
        Create comprehensive logging configuration.

        Args:
            logs_dir: Directory for log files

        Returns:
            Logging configuration dictionary
        """
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "simple": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "console": {
                    "format": "%(levelname)s - %(name)s - %(message)s",
                },
            },
            "filters": {
                "security_filter": {
                    "()": "bot.utils.secure_logging.SensitiveDataFilter",
                },
            },
            "handlers": {},
            "loggers": {},
            "root": {
                "level": settings.system.log_level.upper(),
                "handlers": [],
            },
        }

        # Add console handler if enabled
        if settings.system.log_to_console:
            config["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "level": settings.system.log_level.upper(),
                "formatter": "console",
                "filters": ["security_filter"],
                "stream": "ext://sys.stderr",
            }
            config["root"]["handlers"].append("console")

        # Add file handlers with fallback
        file_handlers = self._create_file_handlers_config(logs_dir)
        config["handlers"].update(file_handlers)

        # Add specific loggers
        config["loggers"].update(self._create_logger_configs())

        return config

    def _create_file_handlers_config(self, logs_dir: Path) -> dict[str, Any]:
        """
        Create file handler configurations with fallback support.

        Args:
            logs_dir: Directory for log files

        Returns:
            Dictionary of file handler configurations
        """
        handlers = {}

        # Main application log
        try:
            main_log_path = logs_dir / "bot.log"
            handlers["file_main"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.system.log_level.upper(),
                "formatter": "detailed",
                "filters": ["security_filter"],
                "filename": str(main_log_path),
                "maxBytes": 50 * 1024 * 1024,  # 50MB
                "backupCount": 5,
                "encoding": "utf-8",
            }
            self.available_handlers.append("file_main")
        except Exception:
            pass

        # Trading decisions log
        try:
            trade_log_path = logs_dir / "trades" / "decisions.log"
            trade_log_path.parent.mkdir(exist_ok=True)
            handlers["file_trading"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filters": ["security_filter"],
                "filename": str(trade_log_path),
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 10,
                "encoding": "utf-8",
            }
            self.available_handlers.append("file_trading")
        except Exception:
            pass

        # Error log
        try:
            error_log_path = logs_dir / "errors.log"
            handlers["file_errors"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filters": ["security_filter"],
                "filename": str(error_log_path),
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8",
            }
            self.available_handlers.append("file_errors")
        except Exception:
            pass

        return handlers

    def _create_logger_configs(self) -> dict[str, Any]:
        """Create specific logger configurations."""
        loggers = {}

        # Bot root logger
        loggers["bot"] = {
            "level": settings.system.log_level.upper(),
            "handlers": self.available_handlers
            + (["console"] if settings.system.log_to_console else []),
            "propagate": False,
        }

        # Trading logger
        trading_handlers = (
            ["file_trading"] if "file_trading" in self.available_handlers else []
        )
        if settings.system.log_to_console:
            trading_handlers.append("console")

        loggers["bot.trading"] = {
            "level": "INFO",
            "handlers": trading_handlers,
            "propagate": False,
        }

        # Error logger
        error_handlers = (
            ["file_errors"] if "file_errors" in self.available_handlers else []
        )
        if settings.system.log_to_console:
            error_handlers.append("console")

        loggers["bot.errors"] = {
            "level": "ERROR",
            "handlers": error_handlers,
            "propagate": False,
        }

        return loggers

    def _apply_config_safely(
        self, config: dict[str, Any], results: dict[str, Any]
    ) -> None:
        """
        Apply logging configuration with error handling.

        Args:
            config: Logging configuration
            results: Results dictionary to update
        """
        try:
            logging.config.dictConfig(config)
            results["handlers_created"] = list(config["handlers"].keys())
        except Exception as e:
            results["errors"].append(f"Config application failed: {e}")
            # Try applying a minimal configuration
            self._apply_minimal_config(results)

    def _apply_minimal_config(self, results: dict[str, Any]) -> None:
        """Apply minimal logging configuration as fallback."""
        try:
            logging.basicConfig(
                level=getattr(logging, settings.system.log_level.upper(), logging.INFO),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            results["handlers_created"] = ["basic_config"]
            results["warnings"].append("Using basic logging configuration")
        except Exception as e:
            results["errors"].append(f"Even basic config failed: {e}")

    def _setup_emergency_logging(self, results: dict[str, Any]) -> None:
        """Setup emergency console-only logging."""
        try:
            # Clear any existing configuration
            logging.getLogger().handlers.clear()

            # Setup basic console logging
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
            console_handler.setFormatter(formatter)

            root_logger = logging.getLogger()
            root_logger.addHandler(console_handler)
            root_logger.setLevel(logging.INFO)

            results["handlers_created"] = ["emergency_console"]
            results["fallback_mode"] = True
            self.fallback_mode = True

        except Exception as e:
            results["errors"].append(f"Emergency logging setup failed: {e}")

    def _verify_handlers(self, results: dict[str, Any]) -> None:
        """Verify that logging handlers are working correctly."""
        test_logger = logging.getLogger("bot.system.test")

        working_handlers = []
        failed_handlers = []

        for handler_name in results["handlers_created"]:
            try:
                # Try to log a test message
                test_logger.info("Logging system verification test")
                working_handlers.append(handler_name)
            except Exception as e:
                failed_handlers.append((handler_name, str(e)))

        if failed_handlers:
            results["warnings"].extend(
                [
                    f"Handler {name} failed verification: {error}"
                    for name, error in failed_handlers
                ]
            )

        results["working_handlers"] = working_handlers
        results["failed_handlers"] = [name for name, _ in failed_handlers]

    def get_system_info(self) -> dict[str, Any]:
        """Get current logging system information."""
        return {
            "config_applied": self.config_applied,
            "fallback_mode": self.fallback_mode,
            "logs_directory": str(self.logs_directory) if self.logs_directory else None,
            "available_handlers": self.available_handlers,
            "root_logger_level": logging.getLogger().level,
            "root_logger_handlers": [
                type(h).__name__ for h in logging.getLogger().handlers
            ],
        }


# Global instance
_logging_config = RobustLoggingConfig()


def setup_application_logging() -> dict[str, Any]:
    """
    Setup application-wide logging configuration.

    This function should be called once at application startup.

    Returns:
        Dictionary with setup results and diagnostics
    """
    return _logging_config.setup_logging()


def get_logging_system_info() -> dict[str, Any]:
    """Get information about the current logging system state."""
    return _logging_config.get_system_info()


def is_fallback_mode() -> bool:
    """Check if logging is running in fallback mode."""
    return _logging_config.fallback_mode


def get_logs_directory_info() -> dict[str, Any]:
    """Get information about the logs directory."""
    return {
        "logs_directory": (
            str(_logging_config.logs_directory)
            if _logging_config.logs_directory
            else None
        ),
        "fallback_mode": _logging_config.fallback_mode,
        "writable": (
            _logging_config._test_directory_writable(_logging_config.logs_directory)
            if _logging_config.logs_directory
            else False
        ),
    }
