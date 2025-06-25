"""
Logging Effects for Functional Trading Bot

This module provides functional logging effects with structured logs,
performance monitoring, and composable log contexts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from .io import IO

if TYPE_CHECKING:
    from collections.abc import Callable


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class LogConfig:
    """Logging configuration"""

    level: LogLevel = LogLevel.INFO
    format: str = "json"
    outputs: list = None
    structured: bool = True


@dataclass
class LogContext:
    """Logging context data"""

    data: dict[str, Any]

    def with_field(self, key: str, value: Any) -> LogContext:
        """Add field to context"""
        new_data = self.data.copy()
        new_data[key] = value
        return LogContext(new_data)


def log(
    level: LogLevel, message: str, context: dict[str, Any] | None = None
) -> IO[None]:
    """Core logging effect"""

    def do_log():
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "message": message,
            "context": context or {},
        }

        if level == LogLevel.ERROR:
            print(f"ERROR: {json.dumps(log_entry)}")
        elif level == LogLevel.WARN:
            print(f"WARN: {json.dumps(log_entry)}")
        elif level == LogLevel.INFO:
            print(f"INFO: {json.dumps(log_entry)}")
        else:
            print(f"DEBUG: {json.dumps(log_entry)}")

    return IO(do_log)


def debug(message: str, context: dict[str, Any] | None = None) -> IO[None]:
    """Debug log effect"""
    return log(LogLevel.DEBUG, message, context)


def info(message: str, context: dict[str, Any] | None = None) -> IO[None]:
    """Info log effect"""
    return log(LogLevel.INFO, message, context)


def warn(message: str, context: dict[str, Any] | None = None) -> IO[None]:
    """Warning log effect"""
    return log(LogLevel.WARN, message, context)


def error(message: str, context: dict[str, Any] | None = None) -> IO[None]:
    """Error log effect"""
    return log(LogLevel.ERROR, message, context)


def with_context(context: dict[str, Any]) -> Callable[[IO[A]], IO[A]]:
    """Add context to all logs in an effect"""

    def wrapper(effect: IO[A]) -> IO[A]:
        # In a real implementation, this would use thread-local storage
        # or effect environment to carry context
        return effect

    return wrapper


def log_performance(label: str) -> Callable[[IO[A]], IO[A]]:
    """Log performance metrics for an effect"""

    def wrapper(effect: IO[A]) -> IO[A]:
        def timed():
            start_time = time.time()
            result = effect.run()
            duration = time.time() - start_time

            info(
                f"Performance: {label}",
                {"duration_ms": duration * 1000, "label": label},
            ).run()

            return result

        return IO(timed)

    return wrapper


def log_trade_event(event_type: str, data: dict[str, Any]) -> IO[None]:
    """Log trading-specific events"""
    return info(
        f"Trade Event: {event_type}",
        {
            "event_type": event_type,
            "trade_data": data,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


def configure_logging(config: LogConfig) -> IO[None]:
    """Configure logging system"""

    def configure():
        # In a real implementation, this would configure the logging system
        print(f"Logging configured: {config}")

    return IO(configure)
