"""Effect system types including monads for functional programming.

This module provides fundamental monadic types for handling effects in a pure functional way:
- Result[T, E]: For computations that may fail with an error
- Maybe[T]: For computations that may return no value
- IO[T]: For wrapping side-effectful computations
- Effect: Sum type for various side effects
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
)

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


# Result Monad
# Monad Laws:
# 1. Left Identity: Result.ok(a).flat_map(f) == f(a)
# 2. Right Identity: m.flat_map(Result.ok) == m
# 3. Associativity: m.flat_map(f).flat_map(g) == m.flat_map(lambda x: f(x).flat_map(g))


@dataclass(frozen=True)
class Ok(Generic[T, E]):
    """Successful result containing a value."""

    value: T

    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        """Apply function to the contained value."""
        return Ok(f(self.value))

    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Apply function that returns a Result."""
        return f(self.value)

    def map_error(self, f: Callable[[E], F]) -> Result[T, F]:
        """Transform the error type (no-op for Ok)."""
        return Ok(self.value)

    def unwrap(self) -> T:
        """Extract the value."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Return the value."""
        return self.value

    def is_ok(self) -> bool:
        """Check if this is an Ok variant."""
        return True

    def is_err(self) -> bool:
        """Check if this is an Err variant."""
        return False


@dataclass(frozen=True)
class Err(Generic[T, E]):
    """Failed result containing an error."""

    error: E

    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        """No-op for Err."""
        return Err(self.error)

    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """No-op for Err."""
        return Err(self.error)

    def map_error(self, f: Callable[[E], F]) -> Result[T, F]:
        """Transform the error."""
        return Err(f(self.error))

    def unwrap(self) -> T:
        """Raise an exception with the error."""
        raise ValueError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Return the default value."""
        return default

    def is_ok(self) -> bool:
        """Check if this is an Ok variant."""
        return False

    def is_err(self) -> bool:
        """Check if this is an Err variant."""
        return True


# Type alias for Result
Result = Union[Ok[T, E], Err[T, E]]


# Maybe Monad
# Monad Laws:
# 1. Left Identity: Maybe.some(a).flat_map(f) == f(a)
# 2. Right Identity: m.flat_map(Maybe.some) == m
# 3. Associativity: m.flat_map(f).flat_map(g) == m.flat_map(lambda x: f(x).flat_map(g))


@dataclass(frozen=True)
class Some(Generic[T]):
    """Value is present."""

    value: T

    def map(self, f: Callable[[T], U]) -> Maybe[U]:
        """Apply function to the contained value."""
        return Some(f(self.value))

    def flat_map(self, f: Callable[[T], Maybe[U]]) -> Maybe[U]:
        """Apply function that returns a Maybe."""
        return f(self.value)

    def or_else(self, default: Callable[[], Maybe[T]]) -> Maybe[T]:
        """Return self (value is present)."""
        return self

    def unwrap(self) -> T:
        """Extract the value."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Return the value."""
        return self.value

    def is_some(self) -> bool:
        """Check if this is a Some variant."""
        return True

    def is_nothing(self) -> bool:
        """Check if this is a Nothing variant."""
        return False


@dataclass(frozen=True)
class Nothing(Generic[T]):
    """No value is present."""

    def map(self, f: Callable[[T], U]) -> Maybe[U]:
        """No-op for Nothing."""
        return Nothing()

    def flat_map(self, f: Callable[[T], Maybe[U]]) -> Maybe[U]:
        """No-op for Nothing."""
        return Nothing()

    def or_else(self, default: Callable[[], Maybe[T]]) -> Maybe[T]:
        """Return the result of calling default."""
        return default()

    def unwrap(self) -> T:
        """Raise an exception."""
        raise ValueError("Called unwrap on Nothing")

    def unwrap_or(self, default: T) -> T:
        """Return the default value."""
        return default

    def is_some(self) -> bool:
        """Check if this is a Some variant."""
        return False

    def is_nothing(self) -> bool:
        """Check if this is a Nothing variant."""
        return True


# Type alias for Maybe
Maybe = Union[Some[T], Nothing[T]]


# IO Monad
# Monad Laws:
# 1. Left Identity: IO.pure(a).flat_map(f) == f(a)
# 2. Right Identity: m.flat_map(IO.pure) == m
# 3. Associativity: m.flat_map(f).flat_map(g) == m.flat_map(lambda x: f(x).flat_map(g))


@dataclass(frozen=True)
class IO(Generic[T]):
    """Wrapper for side-effectful computations."""

    _computation: Callable[[], T]

    def map(self, f: Callable[[T], U]) -> IO[U]:
        """Transform the result of the computation."""
        return IO(lambda: f(self._computation()))

    def flat_map(self, f: Callable[[T], IO[U]]) -> IO[U]:
        """Sequence IO computations."""
        return IO(lambda: f(self._computation()).run())

    def run(self) -> T:
        """Execute the IO computation and return the result."""
        return self._computation()

    @staticmethod
    def pure(value: T) -> IO[T]:
        """Lift a pure value into IO."""
        return IO(lambda: value)

    @staticmethod
    def from_effect(computation: Callable[[], T]) -> IO[T]:
        """Create an IO from an effectful computation."""
        return IO(computation)


# Effect Sum Type
@dataclass(frozen=True)
class ReadFile:
    """Effect for reading a file."""

    path: Path


@dataclass(frozen=True)
class WriteFile:
    """Effect for writing to a file."""

    path: Path
    content: str


@dataclass(frozen=True)
class HttpRequest:
    """Effect for making HTTP requests."""

    method: str
    url: str
    headers: dict[str, str]
    body: str | None = None


@dataclass(frozen=True)
class DbQuery:
    """Effect for database queries."""

    query: str
    params: tuple[Any, ...]


@dataclass(frozen=True)
class Log:
    """Effect for logging."""

    level: str
    message: str


# WebSocket and API Types
@dataclass(frozen=True)
class WebSocketConnection:
    """WebSocket connection state"""

    websocket: Any  # WebSocket connection object
    config: Any  # Connection configuration
    is_connected: bool
    subscriptions: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class RateLimit:
    """Rate limiting configuration"""

    requests_per_second: float
    max_burst: int = 10


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy configuration"""

    max_attempts: int
    delay: float
    backoff_multiplier: float = 2.0


# Exchange Operation Types
@dataclass(frozen=True)
class CancelResult:
    """Result of order cancellation"""

    order_id: str
    success: bool
    message: str = ""


@dataclass(frozen=True)
class PositionUpdate:
    """Position update notification"""

    symbol: str
    side: str
    size: Any  # Decimal
    entry_price: Any  # Decimal
    unrealized_pnl: Any  # Decimal
    timestamp: Any  # datetime


# Effect sum type
Effect = Union[ReadFile, WriteFile, HttpRequest, DbQuery, Log]


# Helper Functions


def lift(value: T) -> Result[T, Any]:
    """Lift a value into the Result monad as Ok."""
    return Ok(value)


def lift_maybe(value: T | None) -> Maybe[T]:
    """Convert an Optional to Maybe."""
    return Some(value) if value is not None else Nothing()


def sequence_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """Convert a list of Results to a Result of list.

    Returns Ok with all values if all are Ok, otherwise returns the first Err.
    """
    values: list[T] = []
    for result in results:
        if isinstance(result, Ok):
            values.append(result.value)
        else:
            return Err(result.error)
    return Ok(values)


def sequence_maybes(maybes: list[Maybe[T]]) -> Maybe[list[T]]:
    """Convert a list of Maybes to a Maybe of list.

    Returns Some with all values if all are Some, otherwise returns Nothing.
    """
    values: list[T] = []
    for maybe in maybes:
        if isinstance(maybe, Some):
            values.append(maybe.value)
        else:
            return Nothing()
    return Some(values)


def sequence_io(ios: list[IO[T]]) -> IO[list[T]]:
    """Convert a list of IO computations to an IO of list."""
    return IO(lambda: [io.run() for io in ios])


def traverse_result(
    func: Callable[[T], Result[U, E]], items: list[T]
) -> Result[list[U], E]:
    """Map a function returning Result over a list and collect results."""
    return sequence_results([func(item) for item in items])


def traverse_maybe(func: Callable[[T], Maybe[U]], items: list[T]) -> Maybe[list[U]]:
    """Map a function returning Maybe over a list and collect results."""
    return sequence_maybes([func(item) for item in items])


def traverse_io(func: Callable[[T], IO[U]], items: list[T]) -> IO[list[U]]:
    """Map a function returning IO over a list and collect results."""
    return sequence_io([func(item) for item in items])


# Monadic composition helpers


def compose_results(
    f: Callable[[T], Result[U, E]], g: Callable[[U], Result[V, E]]
) -> Callable[[T], Result[V, E]]:
    """Compose two functions that return Results."""
    return lambda x: f(x).flat_map(g)


def compose_maybes(
    f: Callable[[T], Maybe[U]], g: Callable[[U], Maybe[V]]
) -> Callable[[T], Maybe[V]]:
    """Compose two functions that return Maybes."""
    return lambda x: f(x).flat_map(g)


def compose_io(
    f: Callable[[T], IO[U]], g: Callable[[U], IO[V]]
) -> Callable[[T], IO[V]]:
    """Compose two functions that return IO."""
    return lambda x: f(x).flat_map(g)
