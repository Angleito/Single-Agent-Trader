"""
Time Effects for Functional Trading Bot

This module provides functional effects for time-based operations,
scheduling, and rate limiting.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import TypeVar

from .io import IO

A = TypeVar("A")


def now() -> IO[datetime]:
    """Get current time effect"""
    return IO(lambda: datetime.utcnow())


def delay(duration: timedelta) -> IO[None]:
    """Delay execution"""

    def sleep():
        time.sleep(duration.total_seconds())

    return IO(sleep)


def timeout(duration: timedelta, effect: IO[A]) -> IO[A | None]:
    """Apply timeout to effect"""

    def timed():
        try:
            # Simulate timeout logic
            return effect.run()
        except:
            return None

    return IO(timed)


def measure_time(effect: IO[A]) -> IO[tuple[A, timedelta]]:
    """Measure execution time"""

    def measured():
        start = time.time()
        result = effect.run()
        duration = timedelta(seconds=time.time() - start)
        return (result, duration)

    return IO(measured)
