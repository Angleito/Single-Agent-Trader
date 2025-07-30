"""
Minimal Result monad implementation for functional programming.

This provides a basic Result type without requiring external dependencies.
"""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")


class Result[T, E]:
    """Base class for Result monad."""

    def __init__(self) -> None:
        pass

    def is_success(self) -> bool:
        """Check if this is a success result."""
        return isinstance(self, Success)

    def is_failure(self) -> bool:
        """Check if this is a failure result."""
        return isinstance(self, Failure)

    def is_ok(self) -> bool:
        """Check if this is a success result (alias for is_success)."""
        return self.is_success()

    def is_err(self) -> bool:
        """Check if this is a failure result (alias for is_failure)."""
        return self.is_failure()

    def success(self) -> T:
        """Get success value - only valid for Success instances."""
        if isinstance(self, Success):
            return self._value  # type: ignore[no-any-return]
        raise ValueError("Cannot get success value from Failure")

    def failure(self) -> E:
        """Get failure value - only valid for Failure instances."""
        if isinstance(self, Failure):
            return self._error  # type: ignore[no-any-return]
        raise ValueError("Cannot get failure value from Success")

    def map(self, func: Callable[[T], U]) -> "Result[U, E]":
        """Map function over success value."""
        if isinstance(self, Success):
            try:
                return Success(func(self._value))
            except Exception as e:
                return Failure(str(e))  # type: ignore
        return self  # type: ignore

    def flat_map(self, func: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        """Flat map function over success value."""
        if isinstance(self, Success):
            return func(self._value)
        return self  # type: ignore


class Success(Result[T, E]):
    """Success result containing a value."""

    def __init__(self, value: T) -> None:
        super().__init__()
        self._value = value

    def __str__(self) -> str:
        return f"Success({self._value})"

    def __repr__(self) -> str:
        return f"Success({self._value!r})"


class Failure(Result[T, E]):
    """Failure result containing an error."""

    def __init__(self, error: E) -> None:
        super().__init__()
        self._error = error

    def __str__(self) -> str:
        return f"Failure({self._error})"

    def __repr__(self) -> str:
        return f"Failure({self._error!r})"


# Type aliases for convenience
def Ok[T, E](value: T) -> Success[T, E]:
    """Create a success result."""
    return Success(value)


def Err[T, E](error: E) -> Failure[T, E]:
    """Create a failure result."""
    return Failure(error)


__all__ = ["Err", "Failure", "Ok", "Result", "Success"]
