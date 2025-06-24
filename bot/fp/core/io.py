"""
IO monad for managing side effects in functional programming.

This module provides an IO type for wrapping side-effectful computations
and composing them in a pure way.
"""

from typing import Callable, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class IO(Generic[T]):
    """IO monad for wrapping side-effectful computations."""

    def __init__(self, computation: Callable[[], T]) -> None:
        """Initialize IO with a computation."""
        self._computation = computation

    def run(self) -> T:
        """Execute the IO computation."""
        return self._computation()

    def map(self, func: Callable[[T], U]) -> "IO[U]":
        """Map a function over the IO value."""
        return IO(lambda: func(self.run()))

    def flat_map(self, func: Callable[[T], "IO[U]"]) -> "IO[U]":
        """Flat map a function that returns IO over the IO value."""
        return IO(lambda: func(self.run()).run())

    def filter(self, predicate: Callable[[T], bool]) -> "IO[T | None]":
        """Filter IO value based on predicate."""
        def filtered_computation() -> T | None:
            value = self.run()
            return value if predicate(value) else None
        return IO(filtered_computation)

    def zip(self, other: "IO[U]") -> "IO[tuple[T, U]]":
        """Combine two IO computations into a tuple."""
        return IO(lambda: (self.run(), other.run()))

    def __str__(self) -> str:
        return f"IO(<computation>)"

    def __repr__(self) -> str:
        return f"IO({self._computation})"


def pure(value: T) -> IO[T]:
    """Create an IO that returns the given value."""
    return IO(lambda: value)


def unit() -> IO[None]:
    """Create an IO that does nothing."""
    return IO(lambda: None)


def delay(computation: Callable[[], T]) -> IO[T]:
    """Delay a computation by wrapping it in IO."""
    return IO(computation)


def sequence_io(ios: list[IO[T]]) -> IO[list[T]]:
    """Execute a list of IO computations and collect results."""
    def run_all() -> list[T]:
        return [io.run() for io in ios]
    return IO(run_all)


def traverse_io(items: list[T], func: Callable[[T], IO[U]]) -> IO[list[U]]:
    """Apply IO-returning function to each item and sequence results."""
    return sequence_io([func(item) for item in items])


def for_each_io(items: list[T], action: Callable[[T], IO[None]]) -> IO[None]:
    """Execute an IO action for each item in the list."""
    def run_actions() -> None:
        for item in items:
            action(item).run()
    return IO(run_actions)


def while_io(condition: IO[bool], action: IO[None]) -> IO[None]:
    """Repeat action while condition is true."""
    def loop() -> None:
        while condition.run():
            action.run()
    return IO(loop)


def if_io(condition: IO[bool], then_action: IO[T], else_action: IO[T]) -> IO[T]:
    """Conditional IO execution."""
    def conditional() -> T:
        if condition.run():
            return then_action.run()
        return else_action.run()
    return IO(conditional)


def try_io(io: IO[T], error_handler: Callable[[Exception], T]) -> IO[T]:
    """Try to run IO, handling exceptions."""
    def safe_run() -> T:
        try:
            return io.run()
        except Exception as e:
            return error_handler(e)
    return IO(safe_run)


def unsafe_run_io_sync(io: IO[T]) -> T:
    """Unsafely run IO synchronously - use sparingly."""
    return io.run()


class IOBuilder(Generic[T]):
    """Builder for composing IO operations."""

    def __init__(self, initial: IO[T]) -> None:
        self._io = initial

    def then(self, func: Callable[[T], IO[U]]) -> "IOBuilder[U]":
        """Chain another IO operation."""
        return IOBuilder(self._io.flat_map(func))

    def map(self, func: Callable[[T], U]) -> "IOBuilder[U]":
        """Map a function over the IO value."""
        return IOBuilder(self._io.map(func))

    def filter(self, predicate: Callable[[T], bool]) -> "IOBuilder[T | None]":
        """Filter the IO value."""
        return IOBuilder(self._io.filter(predicate))

    def build(self) -> IO[T]:
        """Build the final IO computation."""
        return self._io


def io_builder(initial: IO[T]) -> IOBuilder[T]:
    """Create an IO builder for fluent composition."""
    return IOBuilder(initial)


__all__ = [
    "IO",
    "pure",
    "unit", 
    "delay",
    "sequence_io",
    "traverse_io",
    "for_each_io",
    "while_io",
    "if_io",
    "try_io",
    "unsafe_run_io_sync",
    "IOBuilder",
    "io_builder",
]