"""
Functional combinators for learning operations.

This module provides composable functions for building complex learning
pipelines from simple, pure functions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import TypeVar

from bot.fp.effects.io import AsyncIO
from bot.fp.types.result import Err, Ok, Result

from .experience import ExperienceState, TradeExperience
from .learning_algorithms import LearningResult, StrategyInsight

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


# Function composition combinators


def compose_learning_functions(
    f: Callable[[A], B],
    g: Callable[[B], C],
) -> Callable[[A], C]:
    """Compose two learning functions."""

    def composed(x: A) -> C:
        return g(f(x))

    return composed


def pipe(*functions: Callable[[A], A]) -> Callable[[A], A]:
    """Pipe data through a sequence of functions."""

    def piped(initial: A) -> A:
        result = initial
        for func in functions:
            result = func(result)
        return result

    return piped


# Sequencing combinators


def sequence_learning_operations(
    operations: list[Callable[[ExperienceState], Result[ExperienceState, str]]],
) -> Callable[[ExperienceState], Result[ExperienceState, str]]:
    """Sequence learning operations, stopping on first error."""

    def sequenced(initial_state: ExperienceState) -> Result[ExperienceState, str]:
        current_state = initial_state

        for operation in operations:
            result = operation(current_state)
            if result.is_failure():
                return result
            current_state = result.success()

        return Ok(current_state)

    return sequenced


def sequence_async_operations(
    operations: list[
        Callable[[ExperienceState], AsyncIO[Result[ExperienceState, str]]]
    ],
) -> Callable[[ExperienceState], AsyncIO[Result[ExperienceState, str]]]:
    """Sequence async learning operations."""

    def sequenced(
        initial_state: ExperienceState,
    ) -> AsyncIO[Result[ExperienceState, str]]:
        async def run_sequence():
            current_state = initial_state

            for operation in operations:
                result = await operation(current_state).run()
                if result.is_failure():
                    return result
                current_state = result.success()

            return Ok(current_state)

        return AsyncIO(lambda: asyncio.create_task(run_sequence()))

    return sequenced


# Parallel execution combinators


def parallel_learning_analysis(
    analyses: list[Callable[[ExperienceState], Result[LearningResult, str]]],
) -> Callable[[ExperienceState], Result[list[LearningResult], str]]:
    """Run multiple learning analyses in parallel."""

    def parallel_analysis(state: ExperienceState) -> Result[list[LearningResult], str]:
        results = []
        errors = []

        for analysis in analyses:
            result = analysis(state)
            if result.is_success():
                results.append(result.success())
            else:
                errors.append(result.failure())

        if errors:
            return Err(f"Parallel analysis errors: {'; '.join(errors)}")

        return Ok(results)

    return parallel_analysis


def parallel_async_analysis(
    analyses: list[Callable[[ExperienceState], AsyncIO[Result[LearningResult, str]]]],
) -> Callable[[ExperienceState], AsyncIO[Result[list[LearningResult], str]]]:
    """Run async learning analyses in parallel."""

    def parallel_async(
        state: ExperienceState,
    ) -> AsyncIO[Result[list[LearningResult], str]]:
        async def run_parallel():
            tasks = [analysis(state).run() for analysis in analyses]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_results = []
            errors = []

            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                elif isinstance(result, Result) and result.is_success():
                    success_results.append(result.success())
                elif isinstance(result, Result) and result.is_failure():
                    errors.append(result.failure())

            if errors:
                return Err(f"Parallel async analysis errors: {'; '.join(errors)}")

            return Ok(success_results)

        return AsyncIO(lambda: asyncio.create_task(run_parallel()))

    return parallel_async


# Data transformation combinators


def fold_experiences(
    initial: A,
    folder: Callable[[A, TradeExperience], A],
) -> Callable[[ExperienceState], A]:
    """Fold over all experiences to accumulate a result."""

    def fold(state: ExperienceState) -> A:
        result = initial
        for experience in state.experiences:
            result = folder(result, experience)
        return result

    return fold


def filter_experiences(
    predicate: Callable[[TradeExperience], bool],
) -> Callable[[ExperienceState], tuple[TradeExperience, ...]]:
    """Filter experiences by predicate."""

    def filter_func(state: ExperienceState) -> tuple[TradeExperience, ...]:
        return state.filter_experiences(predicate)

    return filter_func


def map_experiences(
    mapper: Callable[[TradeExperience], B],
) -> Callable[[ExperienceState], tuple[B, ...]]:
    """Map function over all experiences."""

    def map_func(state: ExperienceState) -> tuple[B, ...]:
        return tuple(mapper(exp) for exp in state.experiences)

    return map_func


def group_experiences_by(
    key_func: Callable[[TradeExperience], str],
) -> Callable[[ExperienceState], dict[str, tuple[TradeExperience, ...]]]:
    """Group experiences by a key function."""

    def group_func(state: ExperienceState) -> dict[str, tuple[TradeExperience, ...]]:
        groups: dict[str, list[TradeExperience]] = {}

        for experience in state.experiences:
            key = key_func(experience)
            if key not in groups:
                groups[key] = []
            groups[key].append(experience)

        return {key: tuple(experiences) for key, experiences in groups.items()}

    return group_func


# Conditional combinators


def when(
    condition: Callable[[ExperienceState], bool],
    operation: Callable[[ExperienceState], Result[ExperienceState, str]],
) -> Callable[[ExperienceState], Result[ExperienceState, str]]:
    """Apply operation only when condition is true."""

    def conditional(state: ExperienceState) -> Result[ExperienceState, str]:
        if condition(state):
            return operation(state)
        return Ok(state)

    return conditional


def unless(
    condition: Callable[[ExperienceState], bool],
    operation: Callable[[ExperienceState], Result[ExperienceState, str]],
) -> Callable[[ExperienceState], Result[ExperienceState, str]]:
    """Apply operation only when condition is false."""

    def conditional(state: ExperienceState) -> Result[ExperienceState, str]:
        if not condition(state):
            return operation(state)
        return Ok(state)

    return conditional


# Error handling combinators


def with_fallback(
    primary: Callable[[ExperienceState], Result[A, str]],
    fallback: Callable[[ExperienceState], Result[A, str]],
) -> Callable[[ExperienceState], Result[A, str]]:
    """Try primary operation, fallback to secondary on error."""

    def fallback_operation(state: ExperienceState) -> Result[A, str]:
        result = primary(state)
        if result.is_failure():
            return fallback(state)
        return result

    return fallback_operation


def with_retry(
    operation: Callable[[ExperienceState], Result[A, str]],
    max_attempts: int = 3,
) -> Callable[[ExperienceState], Result[A, str]]:
    """Retry operation on failure."""

    def retry_operation(state: ExperienceState) -> Result[A, str]:
        last_error = "No attempts made"

        for attempt in range(max_attempts):
            result = operation(state)
            if result.is_success():
                return result
            last_error = result.failure()

        return Err(f"Operation failed after {max_attempts} attempts: {last_error}")

    return retry_operation


# Validation combinators


def validate_state(
    validator: Callable[[ExperienceState], bool],
    error_message: str,
) -> Callable[[ExperienceState], Result[ExperienceState, str]]:
    """Validate state and return error if invalid."""

    def validation(state: ExperienceState) -> Result[ExperienceState, str]:
        if validator(state):
            return Ok(state)
        return Err(error_message)

    return validation


def validate_minimum_experiences(
    min_count: int,
) -> Callable[[ExperienceState], Result[ExperienceState, str]]:
    """Validate state has minimum number of experiences."""
    return validate_state(
        lambda state: state.total_experiences >= min_count,
        f"Insufficient experiences: need at least {min_count}",
    )


def validate_minimum_completed(
    min_count: int,
) -> Callable[[ExperienceState], Result[ExperienceState, str]]:
    """Validate state has minimum number of completed experiences."""
    return validate_state(
        lambda state: state.total_completed >= min_count,
        f"Insufficient completed experiences: need at least {min_count}",
    )


# Learning pipeline builders


def build_analysis_pipeline(
    *operations: Callable[[ExperienceState], Result[ExperienceState, str]],
) -> Callable[[ExperienceState], Result[ExperienceState, str]]:
    """Build a complete analysis pipeline from operations."""
    return sequence_learning_operations(list(operations))


def build_insight_pipeline(
    state: ExperienceState,
    insights_generators: list[Callable[[ExperienceState], list[StrategyInsight]]],
) -> Result[list[StrategyInsight], str]:
    """Build a pipeline to generate insights."""
    try:
        all_insights = []
        for generator in insights_generators:
            insights = generator(state)
            all_insights.extend(insights)

        # Sort by confidence and expected improvement
        all_insights.sort(
            key=lambda x: (x.confidence, x.expected_improvement),
            reverse=True,
        )

        return Ok(all_insights)
    except Exception as e:
        return Err(f"Insight pipeline failed: {e!s}")


# Utility combinators


def tap(side_effect: Callable[[A], None]) -> Callable[[A], A]:
    """Apply side effect but pass through original value."""

    def tap_func(value: A) -> A:
        side_effect(value)
        return value

    return tap_func


def debug_log(message: str) -> Callable[[A], A]:
    """Log debug message and pass through value."""

    def debug_func(value: A) -> A:
        print(f"DEBUG: {message} - {type(value).__name__}")
        return value

    return debug_func


def memoize(
    func: Callable[[A], B],
    cache_size: int = 100,
) -> Callable[[A], B]:
    """Memoize function results with LRU cache."""
    cache: dict[A, B] = {}
    access_order: list[A] = []

    def memoized(arg: A) -> B:
        if arg in cache:
            # Move to end (most recently used)
            access_order.remove(arg)
            access_order.append(arg)
            return cache[arg]

        # Compute result
        result = func(arg)

        # Add to cache
        cache[arg] = result
        access_order.append(arg)

        # Evict oldest if cache is full
        if len(cache) > cache_size:
            oldest = access_order.pop(0)
            del cache[oldest]

        return result

    return memoized


# Pattern matching helpers


def match_pattern(pattern: str) -> Callable[[TradeExperience], bool]:
    """Create predicate to match experiences with specific pattern."""

    def matcher(experience: TradeExperience) -> bool:
        return pattern in experience.pattern_tags

    return matcher


def match_successful() -> Callable[[TradeExperience], bool]:
    """Create predicate to match successful experiences."""

    def matcher(experience: TradeExperience) -> bool:
        return experience.is_successful()

    return matcher


def match_timeframe(
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> Callable[[TradeExperience], bool]:
    """Create predicate to match experiences within timeframe."""

    def matcher(experience: TradeExperience) -> bool:
        if start_time and experience.timestamp < start_time:
            return False
        if end_time and experience.timestamp > end_time:
            return False
        return True

    return matcher
