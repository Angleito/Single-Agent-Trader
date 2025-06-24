"""
Option monad for handling nullable values in functional programming.

This module provides an Option type for representing values that may or may not exist,
with Some representing a value and Empty representing absence.
"""

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")


class Option(Generic[T], ABC):
    """Abstract base class for Option monad."""

    @abstractmethod
    def is_some(self) -> bool:
        """Check if this is a Some (has value)."""

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if this is Empty (no value)."""

    @abstractmethod
    def map(self, func: Callable[[T], U]) -> "Option[U]":
        """Map function over Some value, preserving Empty."""

    @abstractmethod
    def flat_map(self, func: Callable[[T], "Option[U]"]) -> "Option[U]":
        """Flat map function over Some value."""

    @abstractmethod
    def filter(self, predicate: Callable[[T], bool]) -> "Option[T]":
        """Filter value based on predicate."""

    @abstractmethod
    def fold(self, empty_value: U, some_func: Callable[[T], U]) -> U:
        """Fold Option by providing value for Empty and function for Some."""

    def get_or_else(self, default: T) -> T:
        """Get Some value or return default."""
        if self.is_some():
            return cast("Some[T]", self).value
        return default

    def or_else(self, other: "Option[T]") -> "Option[T]":
        """Return this if Some, otherwise return other."""
        return self if self.is_some() else other

    def to_list(self) -> list[T]:
        """Convert Option to list (empty list for Empty, single-item list for Some)."""
        return [cast("Some[T]", self).value] if self.is_some() else []


class Some(Option[T]):
    """Some variant of Option representing a value."""

    def __init__(self, value: T) -> None:
        if value is None:
            raise ValueError("Some cannot contain None - use Empty instead")
        self.value = value

    def is_some(self) -> bool:
        return True

    def is_empty(self) -> bool:
        return False

    def map(self, func: Callable[[T], U]) -> "Option[U]":
        try:
            result = func(self.value)
            return Some(result) if result is not None else Empty()
        except Exception:
            return Empty()

    def flat_map(self, func: Callable[[T], "Option[U]"]) -> "Option[U]":
        try:
            return func(self.value)
        except Exception:
            return Empty()

    def filter(self, predicate: Callable[[T], bool]) -> "Option[T]":
        try:
            return self if predicate(self.value) else Empty()
        except Exception:
            return Empty()

    def fold(self, empty_value: U, some_func: Callable[[T], U]) -> U:
        return some_func(self.value)

    def __str__(self) -> str:
        return f"Some({self.value})"

    def __repr__(self) -> str:
        return f"Some({self.value!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Some) and self.value == other.value


class Empty(Option[T]):
    """Empty variant of Option representing absence of value."""

    def is_some(self) -> bool:
        return False

    def is_empty(self) -> bool:
        return True

    def map(self, func: Callable[[T], U]) -> "Option[U]":
        return cast("Option[U]", self)

    def flat_map(self, func: Callable[[T], "Option[U]"]) -> "Option[U]":
        return cast("Option[U]", self)

    def filter(self, predicate: Callable[[T], bool]) -> "Option[T]":
        return self

    def fold(self, empty_value: U, some_func: Callable[[T], U]) -> U:
        return empty_value

    def __str__(self) -> str:
        return "Empty"

    def __repr__(self) -> str:
        return "Empty()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Empty)


# Utility functions for creating Option instances
def some(value: T) -> Option[T]:
    """Create a Some Option."""
    return Some(value)


def empty() -> Option[T]:
    """Create an Empty Option."""
    return Empty()


def option_from_nullable(value: T | None) -> Option[T]:
    """Create Option from potentially null value."""
    return Some(value) if value is not None else Empty()


def try_option(func: Callable[[], T]) -> Option[T]:
    """Try to execute a function, catching exceptions as Empty."""
    try:
        result = func()
        return Some(result) if result is not None else Empty()
    except Exception:
        return Empty()


def sequence_option(options: list[Option[T]]) -> Option[list[T]]:
    """Transform a list of Options into an Option of list.
    
    Returns Empty if any option is Empty, otherwise Some with all values.
    """
    results = []
    for option in options:
        if option.is_empty():
            return Empty()
        results.append(cast("Some[T]", option).value)
    return Some(results)


def traverse_option(
    items: list[T], 
    func: Callable[[T], Option[U]]
) -> Option[list[U]]:
    """Apply function to each item and sequence results."""
    return sequence_option([func(item) for item in items])


def first_some(options: list[Option[T]]) -> Option[T]:
    """Return the first Some option, or Empty if all are Empty."""
    for option in options:
        if option.is_some():
            return option
    return Empty()


def combine_options(
    option1: Option[T], 
    option2: Option[U], 
    combiner: Callable[[T, U], "V"]
) -> "Option[V]":
    """Combine two options using a combiner function."""
    if option1.is_some() and option2.is_some():
        value1 = cast("Some[T]", option1).value
        value2 = cast("Some[U]", option2).value
        return Some(combiner(value1, value2))
    return Empty()


class OptionalChain(Generic[T]):
    """Builder for chaining optional operations."""

    def __init__(self, option: Option[T]) -> None:
        self._option = option

    def map(self, func: Callable[[T], U]) -> "OptionalChain[U]":
        """Map a function over the optional value."""
        return OptionalChain(self._option.map(func))

    def flat_map(self, func: Callable[[T], Option[U]]) -> "OptionalChain[U]":
        """Flat map a function over the optional value."""
        return OptionalChain(self._option.flat_map(func))

    def filter(self, predicate: Callable[[T], bool]) -> "OptionalChain[T]":
        """Filter the optional value."""
        return OptionalChain(self._option.filter(predicate))

    def get_or_else(self, default: T) -> T:
        """Get the value or return default."""
        return self._option.get_or_else(default)

    def to_option(self) -> Option[T]:
        """Convert back to Option."""
        return self._option


def optional_chain(option: Option[T]) -> OptionalChain[T]:
    """Create an optional chain for fluent composition."""
    return OptionalChain(option)


__all__ = [
    "Option",
    "Some",
    "Empty", 
    "some",
    "empty",
    "option_from_nullable",
    "try_option",
    "sequence_option",
    "traverse_option",
    "first_some",
    "combine_options",
    "OptionalChain",
    "optional_chain",
]