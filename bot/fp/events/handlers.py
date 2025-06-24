"""Event handler registration and dispatch system with FP patterns."""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
    Protocol,
    TypeVar,
    cast,
)

from bot.fp.core.either import Either, Left, Right
from bot.fp.core.io import IO
from bot.fp.core.option import Empty, Option, Some
from bot.fp.events.base import Event

T = TypeVar("T")
E = TypeVar("E", bound=Event)
R = TypeVar("R")


class EventHandler(Protocol[E, R]):
    """Protocol for event handlers."""

    @abstractmethod
    def handle(self, event: E) -> IO[Either[str, R]]:
        """Handle an event and return IO-wrapped result."""
        ...


class AsyncEventHandler(Protocol[E, R]):
    """Protocol for async event handlers."""

    @abstractmethod
    async def handle(self, event: E) -> Either[str, R]:
        """Handle an event asynchronously."""
        ...


@dataclass
class HandlerConfig:
    """Configuration for event handlers."""

    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float | None = None
    error_handler: Callable[[Exception], None] | None = None


@dataclass
class HandlerRegistration:
    """Registration info for a handler."""

    handler: EventHandler[Any, Any] | AsyncEventHandler[Any, Any]
    config: HandlerConfig
    priority: int = 0
    enabled: bool = True


class HandlerRegistry:
    """Registry for event handlers with type-based dispatch."""

    def __init__(self) -> None:
        """Initialize handler registry."""
        self._handlers: dict[type[Event], list[HandlerRegistration]] = defaultdict(list)
        self._global_handlers: list[HandlerRegistration] = []

    def register(
        self,
        event_type: type[E],
        handler: EventHandler[E, R] | AsyncEventHandler[E, R],
        config: HandlerConfig | None = None,
        priority: int = 0,
    ) -> HandlerRegistry:
        """Register a handler for an event type."""
        registration = HandlerRegistration(
            handler=cast(
                "EventHandler[Any, Any] | AsyncEventHandler[Any, Any]", handler
            ),
            config=config or HandlerConfig(),
            priority=priority,
        )
        self._handlers[event_type].append(registration)
        # Sort by priority (higher first)
        self._handlers[event_type].sort(key=lambda r: r.priority, reverse=True)
        return self

    def register_global(
        self,
        handler: EventHandler[Event, R] | AsyncEventHandler[Event, R],
        config: HandlerConfig | None = None,
        priority: int = 0,
    ) -> HandlerRegistry:
        """Register a global handler for all events."""
        registration = HandlerRegistration(
            handler=cast(
                "EventHandler[Any, Any] | AsyncEventHandler[Any, Any]", handler
            ),
            config=config or HandlerConfig(),
            priority=priority,
        )
        self._global_handlers.append(registration)
        self._global_handlers.sort(key=lambda r: r.priority, reverse=True)
        return self

    def unregister(
        self,
        event_type: type[E],
        handler: EventHandler[E, R] | AsyncEventHandler[E, R],
    ) -> bool:
        """Unregister a handler."""
        if event_type in self._handlers:
            initial_count = len(self._handlers[event_type])
            self._handlers[event_type] = [
                r for r in self._handlers[event_type] if r.handler != handler
            ]
            return len(self._handlers[event_type]) < initial_count
        return False

    def get_handlers(self, event: Event) -> list[HandlerRegistration]:
        """Get all handlers for an event."""
        handlers = []

        # Add type-specific handlers
        event_type = type(event)
        if event_type in self._handlers:
            handlers.extend(r for r in self._handlers[event_type] if r.enabled)

        # Add global handlers
        handlers.extend(r for r in self._global_handlers if r.enabled)

        # Sort by priority
        handlers.sort(key=lambda r: r.priority, reverse=True)
        return handlers


class EventDispatcher:
    """Event dispatcher with side effect management."""

    def __init__(self, registry: HandlerRegistry) -> None:
        """Initialize dispatcher with registry."""
        self._registry = registry
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def dispatch(self, event: Event) -> IO[list[Either[str, Any]]]:
        """Dispatch event to all handlers with IO monad."""

        def _dispatch() -> list[Either[str, Any]]:
            handlers = self._registry.get_handlers(event)
            results = []

            for registration in handlers:
                result = self._handle_with_retry(event, registration)
                results.append(result)

            return results

        return IO(_dispatch)

    async def dispatch_async(self, event: Event) -> list[Either[str, Any]]:
        """Dispatch event asynchronously."""
        handlers = self._registry.get_handlers(event)
        tasks = []

        for registration in handlers:
            task = self._handle_async_with_retry(event, registration)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def _handle_with_retry(
        self,
        event: Event,
        registration: HandlerRegistration,
    ) -> Either[str, Any]:
        """Handle event with retry logic for sync handlers."""
        handler = registration.handler
        config = registration.config

        if isinstance(handler, AsyncEventHandler):
            # Run async handler in sync context
            loop = self._get_or_create_event_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._handle_async_with_retry(event, registration), loop
            )
            return future.result(timeout=config.timeout)

        for attempt in range(config.max_retries):
            try:
                # Sync handler returns IO[Either[str, R]]
                io_result = handler.handle(event)
                return io_result.run()
            except Exception as e:
                if config.error_handler:
                    config.error_handler(e)

                if attempt < config.max_retries - 1:
                    import time

                    time.sleep(config.retry_delay)
                else:
                    return Left(
                        f"Handler failed after {config.max_retries} attempts: {e!s}"
                    )

        return Left("Handler failed: Unknown error")

    async def _handle_async_with_retry(
        self,
        event: Event,
        registration: HandlerRegistration,
    ) -> Either[str, Any]:
        """Handle event with retry logic for async handlers."""
        handler = registration.handler
        config = registration.config

        for attempt in range(config.max_retries):
            try:
                if isinstance(handler, AsyncEventHandler):
                    # Async handler returns Either[str, R]
                    if config.timeout:
                        return await asyncio.wait_for(
                            handler.handle(event), timeout=config.timeout
                        )
                    return await handler.handle(event)
                # Sync handler in async context
                io_result = handler.handle(event)
                return await asyncio.to_thread(io_result.run)
            except Exception as e:
                if config.error_handler:
                    config.error_handler(e)

                if attempt < config.max_retries - 1:
                    await asyncio.sleep(config.retry_delay)
                else:
                    return Left(
                        f"Handler failed after {config.max_retries} attempts: {e!s}"
                    )

        return Left("Handler failed: Unknown error")

    def _get_or_create_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for async operations."""
        if self._event_loop is None or self._event_loop.is_closed():
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
        return self._event_loop


# Handler composition utilities
def compose_handlers(*handlers: EventHandler[E, Any]) -> EventHandler[E, list[Any]]:
    """Compose multiple handlers into one."""

    class ComposedHandler:
        def __init__(self, handlers: tuple[EventHandler[E, Any], ...]):
            self._handlers = handlers

        def handle(self, event: E) -> IO[Either[str, list[Any]]]:
            def _handle() -> Either[str, list[Any]]:
                results = []
                for handler in self._handlers:
                    result = handler.handle(event).run()
                    if isinstance(result, Left):
                        return result
                    results.append(result.value)
                return Right(results)

            return IO(_handle)

    return ComposedHandler(handlers)


def chain_handlers(
    first: EventHandler[E, R],
    second: EventHandler[R, T],
) -> EventHandler[E, T]:
    """Chain two handlers where output of first feeds into second."""

    class ChainedHandler:
        def __init__(
            self,
            first: EventHandler[E, R],
            second: EventHandler[R, T],
        ):
            self._first = first
            self._second = second

        def handle(self, event: E) -> IO[Either[str, T]]:
            def _handle() -> Either[str, T]:
                first_result = self._first.handle(event).run()
                if isinstance(first_result, Left):
                    return first_result

                # Create a synthetic event from the result
                synthetic_event = cast("R", first_result.value)
                return self._second.handle(synthetic_event).run()

            return IO(_handle)

    return ChainedHandler(first, second)


def filter_handler(
    predicate: Callable[[E], bool],
    handler: EventHandler[E, R],
) -> EventHandler[E, Option[R]]:
    """Create a handler that only runs if predicate is true."""

    class FilteredHandler:
        def __init__(
            self,
            predicate: Callable[[E], bool],
            handler: EventHandler[E, R],
        ):
            self._predicate = predicate
            self._handler = handler

        def handle(self, event: E) -> IO[Either[str, Option[R]]]:
            def _handle() -> Either[str, Option[R]]:
                if self._predicate(event):
                    result = self._handler.handle(event).run()
                    if isinstance(result, Left):
                        return result
                    return Right(Some(result.value))
                return Right(Empty())

            return IO(_handle)

    return FilteredHandler(predicate, handler)


# Example handlers
class LoggingHandler(EventHandler[Event, None]):
    """Simple logging handler for any event."""

    def __init__(self, logger: Callable[[str], None]):
        self._logger = logger

    def handle(self, event: Event) -> IO[Either[str, None]]:
        def _handle() -> Either[str, None]:
            try:
                self._logger(f"Event: {event.event_type.value} at {event.timestamp}")
                return Right(None)
            except Exception as e:
                return Left(f"Logging failed: {e!s}")

        return IO(_handle)


class MetricsHandler(AsyncEventHandler[Event, None]):
    """Async metrics collection handler."""

    def __init__(self, metrics_client: Any):
        self._metrics = metrics_client

    async def handle(self, event: Event) -> Either[str, None]:
        try:
            await self._metrics.increment(
                f"events.{event.event_type.value}", tags={"source": event.source}
            )
            return Right(None)
        except Exception as e:
            return Left(f"Metrics failed: {e!s}")


# Factory function
def create_event_system(
    handlers: (
        list[tuple[type[Event], EventHandler[Any, Any], HandlerConfig | None]] | None
    ) = None,
) -> tuple[HandlerRegistry, EventDispatcher]:
    """Create a complete event system."""
    registry = HandlerRegistry()

    if handlers:
        for event_type, handler, config in handlers:
            registry.register(event_type, handler, config)

    dispatcher = EventDispatcher(registry)
    return registry, dispatcher
