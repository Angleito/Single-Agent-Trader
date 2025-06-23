"""
Monitoring Effects for Functional Trading Bot

This module provides functional effects for metrics collection,
health checks, and performance monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from .io import IO

A = TypeVar("A")


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class Span:
    name: str
    start_time: float
    tags: dict[str, Any]


def increment_counter(name: str, tags: dict[str, Any] | None = None) -> IO[None]:
    """Increment a counter metric"""

    def increment():
        print(f"Counter {name} incremented with tags: {tags or {}}")

    return IO(increment)


def record_gauge(
    name: str, value: float, tags: dict[str, Any] | None = None
) -> IO[None]:
    """Record a gauge metric"""

    def record():
        print(f"Gauge {name} = {value} with tags: {tags or {}}")

    return IO(record)


def record_histogram(
    name: str, value: float, tags: dict[str, Any] | None = None
) -> IO[None]:
    """Record a histogram metric"""

    def record():
        print(f"Histogram {name} = {value} with tags: {tags or {}}")

    return IO(record)


def health_check(name: str) -> IO[HealthStatus]:
    """Perform a health check"""

    def check():
        # Simulate health check
        return HealthStatus.HEALTHY

    return IO(check)


def start_span(name: str) -> IO[Span]:
    """Start a tracing span"""

    def start():
        import time

        return Span(name=name, start_time=time.time(), tags={})

    return IO(start)


def alert(level: AlertLevel, message: str) -> IO[None]:
    """Send an alert"""

    def send_alert():
        print(f"ALERT [{level.value.upper()}]: {message}")

    return IO(send_alert)
