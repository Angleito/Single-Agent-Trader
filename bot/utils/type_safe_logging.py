"""Type-safe logging utilities to prevent format string errors."""

import logging
from functools import wraps
from typing import Any, Protocol


class LoggerProtocol(Protocol):
    """Protocol for logger methods."""

    def __call__(self, msg: str, *args: Any) -> None: ...


def ensure_numeric(*indices: int):
    """Decorator to ensure arguments at specific indices are numeric."""

    def decorator(func: LoggerProtocol) -> LoggerProtocol:
        @wraps(func)
        def wrapper(msg: str, *args: Any) -> None:
            safe_args = list(args)
            for i in indices:
                if i < len(safe_args):
                    try:
                        # Check if it's a float by looking for decimal point
                        if "." in str(safe_args[i]):
                            safe_args[i] = float(safe_args[i])
                        else:
                            safe_args[i] = int(float(str(safe_args[i])))
                    except (ValueError, TypeError):
                        # Keep original value if conversion fails
                        pass
            return func(msg, *safe_args)

        return wrapper

    return decorator


# Type-safe logging methods
@ensure_numeric(0, 1, 2, 3)
def log_reconnect(
    logger: logging.Logger, delay: float, retry: int, max_retry: int, failures: int
) -> None:
    """Log websocket reconnection attempt with type safety."""
    logger.warning(
        "Reconnecting in %.1fs (attempt %d/%d, failures: %d)",
        delay,
        retry,
        max_retry,
        failures,
    )


@ensure_numeric(0)
def log_service_wait(logger: logging.Logger, delay: float, service: str) -> None:
    """Log service startup wait with type safety."""
    logger.info("Waiting %.1fs before starting %s...", delay, service)


@ensure_numeric(1)
def log_service_start(logger: logging.Logger, name: str, startup_time: float) -> None:
    """Log successful service start with type safety."""
    logger.info("✓ %s service started successfully (%.1fs)", name, startup_time)


@ensure_numeric(1)
def log_service_available(
    logger: logging.Logger, name: str, startup_time: float
) -> None:
    """Log service availability with type safety."""
    logger.info("✓ %-20s: Available (%.1fs)", name, startup_time)


@ensure_numeric(0, 1)
def log_service_count(
    logger: logging.Logger, available: int, total: int, percentage: float
) -> None:
    """Log service availability count with type safety."""
    logger.info("Services available: %s/%s (%.0f%%)", available, total, percentage)


@ensure_numeric(0)
def log_required_failures(logger: logging.Logger, count: int) -> None:
    """Log required service failures with type safety."""
    logger.error("ERROR: %s required service(s) failed to start!", count)


@ensure_numeric(0, 1)
def log_startup_attempt(
    logger: logging.Logger, attempt: int, max_retries: int, error: str
) -> None:
    """Log service startup attempt with type safety."""
    logger.warning(
        "Service startup failed (attempt %s/%s): %s", attempt, max_retries, error
    )
