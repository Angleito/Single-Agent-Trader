"""
Effect Interpreter for Functional Trading Bot

This module provides the main effect interpreter that executes
IO effects and manages the runtime environment.
"""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from bot.fp.effects.logging import error, info
from bot.fp.effects.monitoring import increment_counter, record_gauge

if TYPE_CHECKING:
    from bot.fp.effects.io import IO, AsyncIO, IOEither

A = TypeVar("A")


@dataclass
class RuntimeConfig:
    """Runtime configuration"""

    max_concurrent_effects: int = 100
    effect_timeout: float = 30.0
    error_recovery: bool = True
    metrics_enabled: bool = True


@dataclass
class RuntimeContext:
    """Runtime execution context"""

    config: RuntimeConfig
    metrics: dict[str, Any]
    active_effects: int = 0

    def increment_active(self) -> None:
        self.active_effects += 1

    def decrement_active(self) -> None:
        self.active_effects = max(0, self.active_effects - 1)


class EffectInterpreter:
    """Main effect interpreter"""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.context = RuntimeContext(config=config, metrics={}, active_effects=0)

    def run_effect(self, effect: IO[A]) -> A:
        """Execute an IO effect"""
        try:
            self.context.increment_active()

            if self.config.metrics_enabled:
                increment_counter("effects.executed").run()

            result = effect.run()

            if self.config.metrics_enabled:
                increment_counter("effects.succeeded").run()

            return result

        except Exception as e:
            if self.config.metrics_enabled:
                increment_counter("effects.failed").run()

            if self.config.error_recovery:
                error(
                    f"Effect execution failed: {e!s}",
                    {
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                    },
                ).run()

            raise

        finally:
            self.context.decrement_active()

            if self.config.metrics_enabled:
                record_gauge("effects.active", self.context.active_effects).run()

    async def run_async_effect(self, effect: AsyncIO[A]) -> A:
        """Execute an async IO effect"""
        try:
            self.context.increment_active()

            if self.config.metrics_enabled:
                increment_counter("async_effects.executed").run()

            # Apply timeout if configured
            if self.config.effect_timeout > 0:
                result = await asyncio.wait_for(
                    effect.run(), timeout=self.config.effect_timeout
                )
            else:
                result = await effect.run()

            if self.config.metrics_enabled:
                increment_counter("async_effects.succeeded").run()

            return result

        except TimeoutError:
            if self.config.metrics_enabled:
                increment_counter("async_effects.timeout").run()

            error(
                "Async effect timed out", {"timeout": self.config.effect_timeout}
            ).run()
            raise

        except Exception as e:
            if self.config.metrics_enabled:
                increment_counter("async_effects.failed").run()

            if self.config.error_recovery:
                error(
                    f"Async effect execution failed: {e!s}",
                    {
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                    },
                ).run()

            raise

        finally:
            self.context.decrement_active()

            if self.config.metrics_enabled:
                record_gauge("async_effects.active", self.context.active_effects).run()

    def run_either_effect(self, effect: IOEither[Exception, A]) -> A:
        """Execute an IOEither effect with error handling"""
        try:
            self.context.increment_active()

            result = effect.run()

            if result.is_left():
                # Handle error case
                exception = result.value
                if self.config.metrics_enabled:
                    increment_counter("either_effects.failed").run()

                error(
                    f"IOEither effect failed: {exception!s}",
                    {"error_type": type(exception).__name__},
                ).run()

                raise exception
            # Handle success case
            if self.config.metrics_enabled:
                increment_counter("either_effects.succeeded").run()

            return result.value

        except Exception:
            if self.config.metrics_enabled:
                increment_counter("either_effects.error").run()
            raise

        finally:
            self.context.decrement_active()

    def get_runtime_stats(self) -> dict[str, Any]:
        """Get runtime statistics"""
        return {
            "active_effects": self.context.active_effects,
            "config": {
                "max_concurrent_effects": self.config.max_concurrent_effects,
                "effect_timeout": self.config.effect_timeout,
                "error_recovery": self.config.error_recovery,
                "metrics_enabled": self.config.metrics_enabled,
            },
            "metrics": self.context.metrics,
        }

    def shutdown(self) -> None:
        """Shutdown the interpreter"""
        info(
            "Effect interpreter shutting down",
            {"final_active_effects": self.context.active_effects},
        ).run()


# Global interpreter instance
_interpreter: EffectInterpreter | None = None


def get_interpreter() -> EffectInterpreter:
    """Get the global effect interpreter"""
    global _interpreter
    if _interpreter is None:
        _interpreter = EffectInterpreter(RuntimeConfig())
    return _interpreter


def run[A](effect: IO[A]) -> A:
    """Run an effect using the global interpreter"""
    return get_interpreter().run_effect(effect)


async def run_async[A](effect: AsyncIO[A]) -> A:
    """Run an async effect using the global interpreter"""
    return await get_interpreter().run_async_effect(effect)


def run_either[A](effect: IOEither[Exception, A]) -> A:
    """Run an IOEither effect using the global interpreter"""
    return get_interpreter().run_either_effect(effect)
