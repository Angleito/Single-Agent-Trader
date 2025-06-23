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
