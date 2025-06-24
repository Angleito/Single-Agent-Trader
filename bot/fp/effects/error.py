"""
Error Handling Effects for Functional Trading Bot

This module provides functional error handling effects with recovery strategies,
retry logic, and circuit breaker patterns.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from .io import IO

A = TypeVar("A")
E = TypeVar("E")


class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    delay: float = 1.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_delay: float = 60.0


@dataclass
class CircuitConfig:
    failure_threshold: int = 5
    timeout: float = 60.0
    half_open_max_calls: int = 3


# Exchange-specific error types
class ExchangeError(Exception):
    """Base class for exchange-related errors."""
    pass


class ConnectionError(ExchangeError):
    """Exchange connection error."""
    pass


class AuthenticationError(ExchangeError):
    """Exchange authentication error."""
    pass


class RateLimitError(ExchangeError):
    """Exchange rate limit error."""
    pass


class InsufficientFundsError(ExchangeError):
    """Insufficient funds error."""
    pass


class OrderError(ExchangeError):
    """Order-related error."""
    pass


class NetworkError(ExchangeError):
    """Network-related error."""
    pass


class TimeoutError(ExchangeError):
    """Timeout error."""
    pass


class ValidationError(ExchangeError):
    """Validation error."""
    pass


class DataError(ExchangeError):
    """Data-related error."""
    pass


def retry(policy: RetryPolicy, effect: IO[A]) -> IO[A]:
    """Retry effect with policy"""

    def retried():
        attempts = 0
        while attempts < policy.max_attempts:
            try:
                return effect.run()
            except Exception as e:
                attempts += 1
                if attempts >= policy.max_attempts:
                    raise e

                if policy.strategy == RetryStrategy.EXPONENTIAL:
                    delay = min(policy.delay * (2**attempts), policy.max_delay)
                else:
                    delay = policy.delay

                time.sleep(delay)

        raise Exception("Max retry attempts exceeded")

    return IO(retried)


def fallback(default: A, effect: IO[A]) -> IO[A]:
    """Provide fallback value on error"""

    def with_fallback():
        try:
            return effect.run()
        except:
            return default

    return IO(with_fallback)


def recover(handler: Callable[[Exception], A], effect: IO[A]) -> IO[A]:
    """Recover from errors with handler"""

    def recovered():
        try:
            return effect.run()
        except Exception as e:
            return handler(e)

    return IO(recovered)


def with_retry(policy: RetryPolicy):
    """Decorator for retrying operations with policy."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return retry(policy, IO(lambda: func(*args, **kwargs)))
        return wrapper
    return decorator


def with_fallback(default_value):
    """Decorator for providing fallback value on error."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return fallback(default_value, IO(lambda: func(*args, **kwargs)))
        return wrapper
    return decorator


def with_circuit_breaker(config: CircuitConfig):
    """Decorator for circuit breaker pattern."""
    # Simple implementation - can be enhanced with actual state management
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # In a real implementation, this would track failures and open/close the circuit
                raise e
        return wrapper
    return decorator


def create_error_recovery_strategy(retry_policy: RetryPolicy, fallback_value=None):
    """Create a comprehensive error recovery strategy."""
    def strategy(func):
        def wrapper(*args, **kwargs):
            effect = IO(lambda: func(*args, **kwargs))
            effect_with_retry = retry(retry_policy, effect)
            if fallback_value is not None:
                return fallback(fallback_value, effect_with_retry)
            return effect_with_retry
        return wrapper
    return strategy
