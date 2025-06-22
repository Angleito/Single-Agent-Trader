"""Type-safe logger factory for automatic format string conversion."""

import logging
from typing import Any

from .log_format_parser import FormatSpec


class TypeSafeLogger:
    """Logger wrapper with automatic type conversion for format strings."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _log(self, level: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Internal logging with type safety."""
        if not args:
            # No arguments, just log the message
            getattr(self._logger, level)(msg, **kwargs)
            return

        try:
            # Convert arguments to match format string
            safe_args = FormatSpec.convert_args(msg, args)
            getattr(self._logger, level)(msg, *safe_args, **kwargs)
        except Exception as e:
            # Fallback: stringify everything and try again
            try:
                str_args = tuple(str(arg) for arg in args)
                getattr(self._logger, level)(msg, *str_args, **kwargs)
            except Exception:
                # Last resort: log without formatting
                error_msg = f"{msg} [Format Error: {e}]"
                getattr(self._logger, level)(error_msg, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message with type safety."""
        self._log("debug", msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message with type safety."""
        self._log("info", msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message with type safety."""
        self._log("warning", msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message with type safety."""
        self._log("error", msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message with type safety."""
        self._log("critical", msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log exception with type safety."""
        # exception() needs special handling
        if not args:
            self._logger.exception(msg, **kwargs)
            return

        try:
            safe_args = FormatSpec.convert_args(msg, args)
            self._logger.exception(msg, *safe_args, **kwargs)
        except Exception:
            str_args = tuple(str(arg) for arg in args)
            self._logger.exception(msg, *str_args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log with custom level and type safety."""
        level_name = logging.getLevelName(level).lower()
        self._log(level_name, msg, *args, **kwargs)

    # Pass through other logger attributes
    def __getattr__(self, name: str) -> Any:
        """Pass through to underlying logger."""
        return getattr(self._logger, name)


# Cache for logger instances
_logger_cache: dict[str, TypeSafeLogger] = {}


def get_logger(name: str) -> TypeSafeLogger:
    """
    Get a type-safe logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        TypeSafeLogger instance
    """
    if name not in _logger_cache:
        _logger_cache[name] = TypeSafeLogger(logging.getLogger(name))
    return _logger_cache[name]


def clear_logger_cache() -> None:
    """Clear the logger cache (useful for testing)."""
    _logger_cache.clear()
