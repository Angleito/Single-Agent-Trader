"""
Base types for the functional programming configuration system.

This module provides fundamental opaque types and value objects used throughout
the trading bot configuration system.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Generic, NewType, TypeVar

from bot.fp.types.result import Failure, Result, Success

T = TypeVar("T")


# Opaque types for type safety
Timestamp = NewType("Timestamp", int)  # Unix timestamp in milliseconds


@dataclass(frozen=True)
class Money:
    """Immutable money type with currency."""

    amount: Decimal
    currency: str

    @classmethod
    def create(cls, amount: float, currency: str) -> Result["Money", str]:
        """Create Money with validation."""
        if amount < 0:
            return Failure("Amount cannot be negative")
        if not currency or len(currency) < 2:
            return Failure("Invalid currency")

        return Success(cls(amount=Decimal(str(amount)), currency=currency.upper()))

    def __str__(self) -> str:
        return f"{self.amount} {self.currency}"


@dataclass(frozen=True)
class Percentage:
    """Immutable percentage type (0.0 to 1.0)."""

    value: Decimal

    @classmethod
    def create(cls, value: float) -> Result["Percentage", str]:
        """Create Percentage with validation."""
        if not 0 <= value <= 1:
            return Failure(f"Percentage must be between 0 and 1, got {value}")

        return Success(cls(value=Decimal(str(value))))

    def __str__(self) -> str:
        return f"{float(self.value) * 100:.2f}%"

    def as_ratio(self) -> float:
        """Get as ratio (0.0 to 1.0)."""
        return float(self.value)

    def as_percent(self) -> float:
        """Get as percentage (0.0 to 100.0)."""
        return float(self.value) * 100


@dataclass(frozen=True)
class Symbol:
    """Trading symbol (e.g., BTC-USD)."""

    value: str

    @classmethod
    def create(cls, symbol: str) -> Result["Symbol", str]:
        """Create Symbol with validation."""
        if not symbol:
            return Failure("Symbol cannot be empty")

        # Basic symbol format validation
        if not re.match(r"^[A-Z0-9]+-[A-Z0-9]+$", symbol.upper()):
            return Failure(f"Invalid symbol format: {symbol}")

        return Success(cls(value=symbol.upper()))

    def __str__(self) -> str:
        return self.value

    @property
    def base(self) -> str:
        """Get base currency (e.g., 'BTC' from 'BTC-USD')."""
        return self.value.split("-")[0]

    @property
    def quote(self) -> str:
        """Get quote currency (e.g., 'USD' from 'BTC-USD')."""
        return self.value.split("-")[1]


class TimeIntervalUnit(Enum):
    """Time interval units."""

    SECOND = "s"
    MINUTE = "m"
    HOUR = "h"
    DAY = "d"


@dataclass(frozen=True)
class TimeInterval:
    """Time interval (e.g., 1m, 5m, 1h)."""

    value: int
    unit: TimeIntervalUnit

    @classmethod
    def create(cls, interval: str) -> Result["TimeInterval", str]:
        """Create TimeInterval with validation."""
        if not interval:
            return Failure("Interval cannot be empty")

        # Parse interval string (e.g., "1m", "5s", "1h")
        match = re.match(r"^(\d+)([smhd])$", interval.lower())
        if not match:
            return Failure(f"Invalid interval format: {interval}")

        value_str, unit_str = match.groups()
        value = int(value_str)

        if value <= 0:
            return Failure("Interval value must be positive")

        unit_map = {
            "s": TimeIntervalUnit.SECOND,
            "m": TimeIntervalUnit.MINUTE,
            "h": TimeIntervalUnit.HOUR,
            "d": TimeIntervalUnit.DAY,
        }

        unit = unit_map[unit_str]

        return Success(cls(value=value, unit=unit))

    def __str__(self) -> str:
        return f"{self.value}{self.unit.value}"

    def to_seconds(self) -> int:
        """Convert to total seconds."""
        multipliers = {
            TimeIntervalUnit.SECOND: 1,
            TimeIntervalUnit.MINUTE: 60,
            TimeIntervalUnit.HOUR: 3600,
            TimeIntervalUnit.DAY: 86400,
        }
        return self.value * multipliers[self.unit]

    def to_milliseconds(self) -> int:
        """Convert to total milliseconds."""
        return self.to_seconds() * 1000


class TradingMode(Enum):
    """Trading modes."""

    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


# Maybe types for optional values
class Maybe(Generic[T]):
    """Base class for Maybe monad (Option type)."""

    def __init__(self, value: T | None = None) -> None:
        self._value = value

    def is_some(self) -> bool:
        """Check if this has a value."""
        return self._value is not None

    def is_nothing(self) -> bool:
        """Check if this has no value."""
        return self._value is None

    @property
    def value(self) -> T:
        """Get the value (only valid if is_some)."""
        if self._value is None:
            raise ValueError("Cannot get value from Nothing")
        return self._value

    def map(self, func) -> "Maybe":
        """Apply function to value if present."""
        if self.is_nothing():
            return Nothing()
        try:
            result = func(self._value)
            return Some(result) if result is not None else Nothing()
        except Exception:
            return Nothing()

    def flat_map(self, func) -> "Maybe":
        """Apply function that returns Maybe."""
        if self.is_nothing():
            return Nothing()
        try:
            return func(self._value)
        except Exception:
            return Nothing()

    def get_or_else(self, default: T) -> T:
        """Get value or return default."""
        return self._value if self._value is not None else default


class Some(Maybe[T]):
    """Maybe with a value."""

    def __init__(self, value: T) -> None:
        if value is None:
            raise ValueError("Some cannot contain None")
        super().__init__(value)

    def __str__(self) -> str:
        return f"Some({self._value})"

    def __repr__(self) -> str:
        return f"Some({self._value!r})"


class Nothing(Maybe[T]):
    """Maybe with no value."""

    def __init__(self) -> None:
        super().__init__(None)

    def __str__(self) -> str:
        return "Nothing"

    def __repr__(self) -> str:
        return "Nothing()"


# Export main types
__all__ = [
    "Maybe",
    "Money",
    "Nothing",
    "Percentage",
    "Some",
    "Symbol",
    "TimeInterval",
    "TimeIntervalUnit",
    "Timestamp",
    "TradingMode",
]
