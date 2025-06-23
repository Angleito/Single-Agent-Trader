"""
Type definitions for risk management module.

This module contains data structures used across risk management components.
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal


@dataclass
class DailyPnL:
    """Daily P&L tracking."""

    date: date
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    trades_count: int = 0
    max_drawdown: Decimal = Decimal(0)


@dataclass
class FailureRecord:
    """Record of trading failures for circuit breaker."""

    timestamp: datetime
    failure_type: str
    error_message: str
    severity: str = "medium"  # low, medium, high, critical
