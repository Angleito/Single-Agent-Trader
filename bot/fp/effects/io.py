"""
IO Monad and Effect Composition for Functional Trading Bot

This module provides the core IO monad implementation with effect composition
utilities for managing side effects in a purely functional way.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

A = TypeVar("A")
B = TypeVar("B")
E = TypeVar("E")


class IO[A]:
    """IO monad for lazy evaluation of side effects"""

    def __init__(self, computation: Callable[[], A]):
        self._computation = computation

    def run(self) -> A:
        """Execute the IO computation"""
        return self._computation()

    def map(self, f: Callable[[A], B]) -> IO[B]:
        """Map a pure function over the IO value"""
        return IO(lambda: f(self.run()))

    def flat_map(self, f: Callable[[A], IO[B]]) -> IO[B]:
        """Monadic bind operation"""
        return IO(lambda: f(self.run()).run())

    def chain(self, other: IO[B]) -> IO[B]:
        """Chain two IO operations, discarding first result"""
        return IO(lambda: (self.run(), other.run())[1])

    def and_then(self, f: Callable[[A], IO[B]]) -> IO[B]:
        """Alias for flat_map"""
        return self.flat_map(f)

    @staticmethod
    def pure(value: A) -> IO[A]:
        """Lift a pure value into IO"""
        return IO(lambda: value)

    @staticmethod
    def from_callable(f: Callable[[], A]) -> IO[A]:
        """Create IO from callable"""
        return IO(f)


class AsyncIO[A]:
    """Async IO monad for asynchronous effects"""

    def __init__(self, computation: Callable[[], asyncio.Task[A]]):
        self._computation = computation

    async def run(self) -> A:
        """Execute the async IO computation"""
        task = self._computation()
        return await task

    def map(self, f: Callable[[A], B]) -> AsyncIO[B]:
        """Map a pure function over the async IO value"""

        async def mapped():
            result = await self.run()
            return f(result)

        return AsyncIO(lambda: asyncio.create_task(mapped()))

    def flat_map(self, f: Callable[[A], AsyncIO[B]]) -> AsyncIO[B]:
        """Async monadic bind"""

        async def bound():
            result = await self.run()
            return await f(result).run()

        return AsyncIO(lambda: asyncio.create_task(bound()))

    @staticmethod
    def pure(value: A) -> AsyncIO[A]:
        """Lift a pure value into AsyncIO"""
        return AsyncIO(lambda: asyncio.create_task(asyncio.coroutine(lambda: value)()))


@dataclass
class Either(Generic[E, A], ABC):
    """Either type for error handling"""

    @abstractmethod
    def is_left(self) -> bool:
        pass

    @abstractmethod
    def is_right(self) -> bool:
        pass

    @abstractmethod
    def map(self, f: Callable[[A], B]) -> Either[E, B]:
        pass

    @abstractmethod
    def flat_map(self, f: Callable[[A], Either[E, B]]) -> Either[E, B]:
        pass


@dataclass
class Left(Either[E, A]):
    """Left side of Either (error case)"""

    value: E

    def is_left(self) -> bool:
        return True

    def is_right(self) -> bool:
        return False

    def map(self, f: Callable[[A], B]) -> Either[E, B]:
        return Left(self.value)

    def flat_map(self, f: Callable[[A], Either[E, B]]) -> Either[E, B]:
        return Left(self.value)


@dataclass
class Right(Either[E, A]):
    """Right side of Either (success case)"""

    value: A

    def is_left(self) -> bool:
        return False

    def is_right(self) -> bool:
        return True

    def map(self, f: Callable[[A], B]) -> Either[E, B]:
        return Right(f(self.value))

    def flat_map(self, f: Callable[[A], Either[E, B]]) -> Either[E, B]:
        return f(self.value)


class IOEither[E, A]:
    """IO that can fail with error type E"""

    def __init__(self, computation: Callable[[], Either[E, A]]):
        self._computation = computation

    def run(self) -> Either[E, A]:
        """Execute the computation"""
        return self._computation()

    def map(self, f: Callable[[A], B]) -> IOEither[E, B]:
        """Map over the success value"""
        return IOEither(lambda: self.run().map(f))

    def flat_map(self, f: Callable[[A], IOEither[E, B]]) -> IOEither[E, B]:
        """Monadic bind for IOEither"""

        def bound():
            result = self.run()
            if result.is_left():
                return Left(result.value)
            return f(result.value).run()

        return IOEither(bound)

    def recover(self, f: Callable[[E], A]) -> IO[A]:
        """Recover from error with default value"""

        def recovered():
            result = self.run()
            if result.is_left():
                return f(result.value)
            return result.value

        return IO(recovered)

    @staticmethod
    def pure(value: A) -> IOEither[E, A]:
        """Lift pure value into IOEither"""
        return IOEither(lambda: Right(value))

    @staticmethod
    def left(error: E) -> IOEither[E, A]:
        """Create failed IOEither"""
        return IOEither(lambda: Left(error))


# Effect Combinators


def sequence[A](ios: list[IO[A]]) -> IO[list[A]]:
    """Execute IOs in sequence, collect results"""

    def sequenced():
        results = []
        for io in ios:
            results.append(io.run())
        return results

    return IO(sequenced)


def traverse(items: list[A], f: Callable[[A], IO[B]]) -> IO[list[B]]:
    """Map function over list and sequence results"""
    return sequence([f(item) for item in items])


def parallel[A](ios: list[IO[A]], max_workers: int = 4) -> IO[list[A]]:
    """Execute IOs in parallel using ThreadPoolExecutor"""

    def paralleled():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(io.run) for io in ios]
            return [future.result() for future in futures]

    return IO(paralleled)


async def async_parallel[A](async_ios: list[AsyncIO[A]]) -> list[A]:
    """Execute AsyncIOs in parallel"""
    tasks = [aio.run() for aio in async_ios]
    return await asyncio.gather(*tasks)


def race[A](ios: list[AsyncIO[A]]) -> AsyncIO[A]:
    """Race multiple AsyncIOs, return first to complete"""

    async def raced():
        tasks = [aio.run() for aio in ios]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        return next(iter(done)).result()

    return AsyncIO(lambda: asyncio.create_task(raced()))


# Lifting Utilities


def from_try[A](f: Callable[[], A]) -> IOEither[Exception, A]:
    """Lift function that may throw into IOEither"""

    def safe():
        try:
            return Right(f())
        except Exception as e:
            return Left(e)

    return IOEither(safe)


def from_option[A, E](option: A | None, error: E) -> IOEither[E, A]:
    """Convert Option to IOEither"""
    if option is None:
        return IOEither.left(error)
    return IOEither.pure(option)


def from_future[A](future: asyncio.Future[A]) -> AsyncIO[A]:
    """Convert Future to AsyncIO"""
    return AsyncIO(lambda: future)


# Utility Functions


def void[A](io: IO[A]) -> IO[None]:
    """Discard the result of an IO"""
    return io.map(lambda _: None)


def when[A](condition: bool, io: IO[A]) -> IO[A | None]:
    """Execute IO only if condition is true"""
    if condition:
        return io.map(lambda x: x)
    return IO.pure(None)


def unless[A](condition: bool, io: IO[A]) -> IO[A | None]:
    """Execute IO only if condition is false"""
    return when(not condition, io)
