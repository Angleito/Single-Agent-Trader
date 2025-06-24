"""
Either monad for handling success/error cases in functional programming.

This module provides an Either type for representing computations that may fail,
with Left representing failure and Right representing success.
"""

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U") 
E = TypeVar("E")


class Either(Generic[E, T], ABC):
    """Abstract base class for Either monad."""

    @abstractmethod
    def is_left(self) -> bool:
        """Check if this is a Left (error) value."""

    @abstractmethod
    def is_right(self) -> bool:
        """Check if this is a Right (success) value."""

    @abstractmethod
    def map(self, func: Callable[[T], U]) -> "Either[E, U]":
        """Map function over Right value, preserving Left."""

    @abstractmethod
    def flat_map(self, func: Callable[[T], "Either[E, U]"]) -> "Either[E, U]":
        """Flat map function over Right value."""

    @abstractmethod
    def map_left(self, func: Callable[[E], U]) -> "Either[U, T]":
        """Map function over Left value, preserving Right."""

    @abstractmethod
    def fold(
        self, 
        left_func: Callable[[E], U], 
        right_func: Callable[[T], U]
    ) -> U:
        """Fold Either by applying appropriate function."""

    def get_or_else(self, default: T) -> T:
        """Get Right value or return default."""
        if self.is_right():
            return cast("Right[E, T]", self).value
        return default

    def or_else(self, other: "Either[E, T]") -> "Either[E, T]":
        """Return this if Right, otherwise return other."""
        return self if self.is_right() else other


class Left(Either[E, T]):
    """Left side of Either representing an error/failure."""

    def __init__(self, value: E) -> None:
        self.value = value

    def is_left(self) -> bool:
        return True

    def is_right(self) -> bool:
        return False

    def map(self, func: Callable[[T], U]) -> "Either[E, U]":
        return cast("Either[E, U]", self)

    def flat_map(self, func: Callable[[T], "Either[E, U]"]) -> "Either[E, U]":
        return cast("Either[E, U]", self)

    def map_left(self, func: Callable[[E], U]) -> "Either[U, T]":
        return Left(func(self.value))

    def fold(
        self, 
        left_func: Callable[[E], U], 
        right_func: Callable[[T], U]
    ) -> U:
        return left_func(self.value)

    def __str__(self) -> str:
        return f"Left({self.value})"

    def __repr__(self) -> str:
        return f"Left({self.value!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Left) and self.value == other.value


class Right(Either[E, T]):
    """Right side of Either representing success."""

    def __init__(self, value: T) -> None:
        self.value = value

    def is_left(self) -> bool:
        return False

    def is_right(self) -> bool:
        return True

    def map(self, func: Callable[[T], U]) -> "Either[E, U]":
        try:
            return Right(func(self.value))
        except Exception as e:
            return Left(cast(E, str(e)))

    def flat_map(self, func: Callable[[T], "Either[E, U]"]) -> "Either[E, U]":
        try:
            return func(self.value)
        except Exception as e:
            return Left(cast(E, str(e)))

    def map_left(self, func: Callable[[E], U]) -> "Either[U, T]":
        return cast("Either[U, T]", self)

    def fold(
        self, 
        left_func: Callable[[E], U], 
        right_func: Callable[[T], U]
    ) -> U:
        return right_func(self.value)

    def __str__(self) -> str:
        return f"Right({self.value})"

    def __repr__(self) -> str:
        return f"Right({self.value!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Right) and self.value == other.value


# Utility functions for creating Either instances
def left(value: E) -> Either[E, T]:
    """Create a Left Either."""
    return Left(value)


def right(value: T) -> Either[E, T]:
    """Create a Right Either."""
    return Right(value)


def try_either(func: Callable[[], T]) -> Either[str, T]:
    """Try to execute a function, catching exceptions as Left."""
    try:
        return Right(func())
    except Exception as e:
        return Left(str(e))


def sequence_either(eithers: list[Either[E, T]]) -> Either[E, list[T]]:
    """Transform a list of Eithers into an Either of list.
    
    Returns Left with the first error found, or Right with all values.
    """
    results = []
    for either in eithers:
        if either.is_left():
            return cast("Either[E, list[T]]", either)
        results.append(cast("Right[E, T]", either).value)
    return Right(results)


def traverse_either(
    items: list[T], 
    func: Callable[[T], Either[E, U]]
) -> Either[E, list[U]]:
    """Apply function to each item and sequence results."""
    return sequence_either([func(item) for item in items])


__all__ = [
    "Either",
    "Left", 
    "Right",
    "left",
    "right", 
    "try_either",
    "sequence_either",
    "traverse_either",
]