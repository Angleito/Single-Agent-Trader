"""
Functional validation pipeline system for composable validation chains.

This module provides:
- Composable validation pipelines
- Pipeline stages with functional composition
- Error aggregation and reporting
- Conditional validation flows
- Parallel validation execution
- Pipeline monitoring and metrics
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from bot.fp.core.functional_validation import (
    FieldError,
    FPFailure,
    FPResult,
    FPSuccess,
    ValidatorError,
)
from bot.validation.data_integrity import FunctionalIntegrityValidator, IntegrityLevel

# Type variables
T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")


class PipelineStage(Enum):
    """Pipeline validation stages."""

    PREPROCESSING = "preprocessing"
    TYPE_VALIDATION = "type_validation"
    RANGE_VALIDATION = "range_validation"
    BUSINESS_RULES = "business_rules"
    INTEGRITY_CHECKS = "integrity_checks"
    CROSS_VALIDATION = "cross_validation"
    POSTPROCESSING = "postprocessing"


class ExecutionMode(Enum):
    """Pipeline execution modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    FAIL_FAST = "fail_fast"
    COLLECT_ALL = "collect_all"


@dataclass(frozen=True)
class PipelineStep:
    """Immutable pipeline step definition."""

    name: str
    stage: PipelineStage
    validator: Callable[[Any], FPResult[Any, ValidatorError]]
    description: str = ""
    optional: bool = False
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: float | None = None
    retry_count: int = 0

    def execute(self, data: Any) -> FPResult[Any, ValidatorError]:
        """Execute this pipeline step."""
        try:
            result = self.validator(data)
            return result
        except Exception as e:
            return FPFailure(
                FieldError(
                    field=self.name,
                    message=f"Pipeline step execution failed: {e}",
                    validation_rule="pipeline_step_error",
                    context={"stage": self.stage.value, "error": str(e)},
                )
            )


@dataclass(frozen=True)
class PipelineMetrics:
    """Pipeline execution metrics."""

    total_steps: int
    successful_steps: int
    failed_steps: int
    execution_time_ms: float
    step_timings: dict[str, float] = field(default_factory=dict)
    error_count_by_stage: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate pipeline success rate."""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps

    @property
    def average_step_time(self) -> float:
        """Calculate average step execution time."""
        if not self.step_timings:
            return 0.0
        return sum(self.step_timings.values()) / len(self.step_timings)


@dataclass(frozen=True)
class PipelineResult:
    """Pipeline execution result."""

    success: bool
    data: Any
    errors: list[ValidatorError]
    metrics: PipelineMetrics
    execution_path: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def get_errors_by_stage(self) -> dict[PipelineStage, list[ValidatorError]]:
        """Group errors by pipeline stage."""
        error_map = {}
        for error in self.errors:
            if isinstance(error, FieldError) and "stage" in error.context:
                stage_name = error.context["stage"]
                try:
                    stage = PipelineStage(stage_name)
                    if stage not in error_map:
                        error_map[stage] = []
                    error_map[stage].append(error)
                except ValueError:
                    # Unknown stage, put in a default category
                    pass
        return error_map


class FunctionalValidationPipeline:
    """Advanced functional validation pipeline with composable stages."""

    def __init__(
        self,
        name: str = "validation_pipeline",
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    ):
        self.name = name
        self.mode = mode
        self.steps: list[PipelineStep] = []
        self.conditional_steps: dict[str, Callable[[Any], bool]] = {}
        self.parallel_groups: list[list[str]] = []

    def add_step(
        self,
        name: str,
        validator: Callable[[Any], FPResult[Any, ValidatorError]],
        stage: PipelineStage = PipelineStage.BUSINESS_RULES,
        description: str = "",
        optional: bool = False,
        dependencies: list[str] = None,
        timeout_seconds: float | None = None,
    ) -> "FunctionalValidationPipeline":
        """Add a validation step to the pipeline."""
        step = PipelineStep(
            name=name,
            stage=stage,
            validator=validator,
            description=description,
            optional=optional,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
        )

        # Create new pipeline with added step
        new_pipeline = FunctionalValidationPipeline(self.name, self.mode)
        new_pipeline.steps = self.steps + [step]
        new_pipeline.conditional_steps = self.conditional_steps.copy()
        new_pipeline.parallel_groups = self.parallel_groups.copy()

        return new_pipeline

    def add_conditional_step(
        self,
        name: str,
        validator: Callable[[Any], FPResult[Any, ValidatorError]],
        condition: Callable[[Any], bool],
        stage: PipelineStage = PipelineStage.BUSINESS_RULES,
        description: str = "",
    ) -> "FunctionalValidationPipeline":
        """Add a conditional validation step."""

        def conditional_validator(data: Any) -> FPResult[Any, ValidatorError]:
            if condition(data):
                return validator(data)
            return FPSuccess(data)

        new_pipeline = self.add_step(name, conditional_validator, stage, description)
        new_pipeline.conditional_steps[name] = condition

        return new_pipeline

    def add_parallel_group(
        self, step_names: list[str]
    ) -> "FunctionalValidationPipeline":
        """Add a group of steps to be executed in parallel."""
        new_pipeline = FunctionalValidationPipeline(self.name, self.mode)
        new_pipeline.steps = self.steps.copy()
        new_pipeline.conditional_steps = self.conditional_steps.copy()
        new_pipeline.parallel_groups = self.parallel_groups + [step_names]

        return new_pipeline

    def execute(self, data: Any) -> PipelineResult:
        """Execute the validation pipeline."""
        start_time = datetime.now()
        errors = []
        warnings = []
        execution_path = []
        step_timings = {}
        successful_steps = 0

        current_data = data

        # Group steps by stage for organized execution
        stages = self._group_steps_by_stage()

        # Execute stages in order
        for stage, stage_steps in stages.items():
            stage_start = datetime.now()

            if self.mode == ExecutionMode.PARALLEL:
                stage_result = self._execute_stage_parallel(stage_steps, current_data)
            else:
                stage_result = self._execute_stage_sequential(stage_steps, current_data)

            stage_time = (datetime.now() - stage_start).total_seconds() * 1000

            # Process stage results
            if stage_result.success:
                current_data = stage_result.data
                successful_steps += len(stage_steps)
                execution_path.extend([step.name for step in stage_steps])
            else:
                errors.extend(stage_result.errors)
                warnings.extend(stage_result.warnings)

                # Fail fast if configured
                if self.mode == ExecutionMode.FAIL_FAST:
                    break

            # Record timings
            for step in stage_steps:
                step_timings[step.name] = stage_time / len(stage_steps)

        # Calculate metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        error_count_by_stage = {}

        for error in errors:
            if isinstance(error, FieldError) and "stage" in error.context:
                stage_name = error.context["stage"]
                error_count_by_stage[stage_name] = (
                    error_count_by_stage.get(stage_name, 0) + 1
                )

        metrics = PipelineMetrics(
            total_steps=len(self.steps),
            successful_steps=successful_steps,
            failed_steps=len(self.steps) - successful_steps,
            execution_time_ms=execution_time,
            step_timings=step_timings,
            error_count_by_stage=error_count_by_stage,
        )

        return PipelineResult(
            success=len(errors) == 0,
            data=current_data if len(errors) == 0 else data,
            errors=errors,
            metrics=metrics,
            execution_path=execution_path,
            warnings=warnings,
        )

    async def execute_async(self, data: Any) -> PipelineResult:
        """Execute the validation pipeline asynchronously."""
        start_time = datetime.now()
        errors = []
        warnings = []
        execution_path = []
        step_timings = {}
        successful_steps = 0

        current_data = data

        # Group steps by stage
        stages = self._group_steps_by_stage()

        # Execute stages asynchronously
        for stage, stage_steps in stages.items():
            stage_start = datetime.now()

            # Execute all steps in stage concurrently
            tasks = []
            for step in stage_steps:
                task = asyncio.create_task(self._execute_step_async(step, current_data))
                tasks.append((step.name, task))

            # Wait for all tasks to complete
            step_results = []
            for step_name, task in tasks:
                try:
                    result = await task
                    step_results.append((step_name, result))
                except Exception as e:
                    error_result = FPFailure(
                        FieldError(
                            field=step_name,
                            message=f"Async execution failed: {e}",
                            validation_rule="async_error",
                        )
                    )
                    step_results.append((step_name, error_result))

            stage_time = (datetime.now() - stage_start).total_seconds() * 1000

            # Process results
            stage_success = True
            for step_name, result in step_results:
                step_timings[step_name] = stage_time / len(stage_steps)

                if result.is_success():
                    current_data = result.success()
                    successful_steps += 1
                    execution_path.append(step_name)
                else:
                    errors.append(result.failure())
                    stage_success = False

            # Fail fast if configured
            if not stage_success and self.mode == ExecutionMode.FAIL_FAST:
                break

        # Calculate metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        error_count_by_stage = {}

        for error in errors:
            if isinstance(error, FieldError) and "stage" in error.context:
                stage_name = error.context["stage"]
                error_count_by_stage[stage_name] = (
                    error_count_by_stage.get(stage_name, 0) + 1
                )

        metrics = PipelineMetrics(
            total_steps=len(self.steps),
            successful_steps=successful_steps,
            failed_steps=len(self.steps) - successful_steps,
            execution_time_ms=execution_time,
            step_timings=step_timings,
            error_count_by_stage=error_count_by_stage,
        )

        return PipelineResult(
            success=len(errors) == 0,
            data=current_data if len(errors) == 0 else data,
            errors=errors,
            metrics=metrics,
            execution_path=execution_path,
            warnings=warnings,
        )

    def _group_steps_by_stage(self) -> dict[PipelineStage, list[PipelineStep]]:
        """Group pipeline steps by stage."""
        stages = {}

        for step in self.steps:
            if step.stage not in stages:
                stages[step.stage] = []
            stages[step.stage].append(step)

        # Return stages in execution order
        ordered_stages = {}
        for stage in PipelineStage:
            if stage in stages:
                ordered_stages[stage] = stages[stage]

        return ordered_stages

    def _execute_stage_sequential(
        self, steps: list[PipelineStep], data: Any
    ) -> PipelineResult:
        """Execute stage steps sequentially."""
        errors = []
        warnings = []
        execution_path = []
        current_data = data

        for step in steps:
            # Check dependencies
            if not self._check_dependencies(step, execution_path):
                if not step.optional:
                    errors.append(
                        FieldError(
                            field=step.name,
                            message=f"Step dependencies not met: {step.dependencies}",
                            validation_rule="dependency_error",
                        )
                    )
                continue

            # Execute step
            result = step.execute(current_data)

            if result.is_success():
                current_data = result.success()
                execution_path.append(step.name)
            else:
                error = result.failure()
                if step.optional:
                    warnings.append(f"Optional step '{step.name}' failed: {error}")
                else:
                    errors.append(error)
                    if self.mode == ExecutionMode.FAIL_FAST:
                        break

        return PipelineResult(
            success=len(errors) == 0,
            data=current_data,
            errors=errors,
            metrics=PipelineMetrics(0, 0, 0, 0),  # Simplified for stage execution
            execution_path=execution_path,
            warnings=warnings,
        )

    def _execute_stage_parallel(
        self, steps: list[PipelineStep], data: Any
    ) -> PipelineResult:
        """Execute stage steps in parallel."""
        errors = []
        warnings = []
        execution_path = []

        with ThreadPoolExecutor(max_workers=min(len(steps), 4)) as executor:
            # Submit all steps
            future_to_step = {
                executor.submit(step.execute, data): step for step in steps
            }

            # Collect results
            for future in as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    result = future.result()

                    if result.is_success():
                        execution_path.append(step.name)
                    else:
                        error = result.failure()
                        if step.optional:
                            warnings.append(
                                f"Optional step '{step.name}' failed: {error}"
                            )
                        else:
                            errors.append(error)

                except Exception as e:
                    errors.append(
                        FieldError(
                            field=step.name,
                            message=f"Parallel execution failed: {e}",
                            validation_rule="parallel_error",
                        )
                    )

        return PipelineResult(
            success=len(errors) == 0,
            data=data,  # In parallel mode, data doesn't transform between steps
            errors=errors,
            metrics=PipelineMetrics(0, 0, 0, 0),  # Simplified for stage execution
            execution_path=execution_path,
            warnings=warnings,
        )

    async def _execute_step_async(
        self, step: PipelineStep, data: Any
    ) -> FPResult[Any, ValidatorError]:
        """Execute a single step asynchronously."""
        try:
            # Apply timeout if specified
            if step.timeout_seconds:
                result = await asyncio.wait_for(
                    asyncio.create_task(asyncio.to_thread(step.execute, data)),
                    timeout=step.timeout_seconds,
                )
            else:
                result = await asyncio.to_thread(step.execute, data)

            return result
        except TimeoutError:
            return FPFailure(
                FieldError(
                    field=step.name,
                    message=f"Step execution timed out after {step.timeout_seconds}s",
                    validation_rule="timeout_error",
                )
            )
        except Exception as e:
            return FPFailure(
                FieldError(
                    field=step.name,
                    message=f"Async step execution failed: {e}",
                    validation_rule="async_error",
                )
            )

    def _check_dependencies(
        self, step: PipelineStep, execution_path: list[str]
    ) -> bool:
        """Check if step dependencies are satisfied."""
        return all(dep in execution_path for dep in step.dependencies)


# Pre-built Pipeline Factories


def create_trade_action_pipeline() -> FunctionalValidationPipeline:
    """Create a validation pipeline for trade actions."""

    pipeline = FunctionalValidationPipeline("trade_action_validation")

    # Type validation
    def type_validator(
        data: dict[str, Any],
    ) -> FPResult[dict[str, Any], ValidatorError]:
        required_fields = [
            "action",
            "size_pct",
            "take_profit_pct",
            "stop_loss_pct",
            "rationale",
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return FPFailure(
                FieldError(
                    field="type_validation",
                    message=f"Missing required fields: {missing}",
                    validation_rule="required_fields",
                )
            )
        return FPSuccess(data)

    # Business rules validator
    def business_rules_validator(
        data: dict[str, Any],
    ) -> FPResult[dict[str, Any], ValidatorError]:
        # Risk/reward ratio check
        tp = float(data.get("take_profit_pct", 0))
        sl = float(data.get("stop_loss_pct", 0))

        if tp > 0 and sl > 0 and tp / sl < 0.5:
            return FPFailure(
                FieldError(
                    field="risk_reward",
                    message="Risk/reward ratio too low",
                    validation_rule="business_rule",
                    context={"take_profit": tp, "stop_loss": sl, "ratio": tp / sl},
                )
            )

        return FPSuccess(data)

    # Integrity validator
    integrity_validator = FunctionalIntegrityValidator(IntegrityLevel.STRICT)

    def integrity_check(
        data: dict[str, Any],
    ) -> FPResult[dict[str, Any], ValidatorError]:
        result = integrity_validator.validate_data(data)
        if result.is_failure():
            return FPFailure(result.failure())
        return FPSuccess(data)

    return (
        pipeline.add_step(
            "type_validation", type_validator, PipelineStage.TYPE_VALIDATION
        )
        .add_step(
            "business_rules", business_rules_validator, PipelineStage.BUSINESS_RULES
        )
        .add_step("integrity_check", integrity_check, PipelineStage.INTEGRITY_CHECKS)
    )


def create_market_data_pipeline() -> FunctionalValidationPipeline:
    """Create a validation pipeline for market data."""

    pipeline = FunctionalValidationPipeline("market_data_validation")

    def ohlc_validator(
        data: dict[str, Any],
    ) -> FPResult[dict[str, Any], ValidatorError]:
        """Validate OHLC relationships."""
        required = ["open", "high", "low", "close"]
        missing = [f for f in required if f not in data]
        if missing:
            return FPFailure(
                FieldError(
                    field="ohlc",
                    message=f"Missing OHLC fields: {missing}",
                    validation_rule="required_fields",
                )
            )

        o, h, l, c = data["open"], data["high"], data["low"], data["close"]

        if not (l <= o <= h and l <= c <= h):
            return FPFailure(
                FieldError(
                    field="ohlc",
                    message="Invalid OHLC relationships",
                    validation_rule="ohlc_validation",
                    context={"open": o, "high": h, "low": l, "close": c},
                )
            )

        return FPSuccess(data)

    def volume_validator(
        data: dict[str, Any],
    ) -> FPResult[dict[str, Any], ValidatorError]:
        """Validate volume is non-negative."""
        if "volume" in data:
            volume = data["volume"]
            if volume < 0:
                return FPFailure(
                    FieldError(
                        field="volume",
                        message="Volume cannot be negative",
                        value=str(volume),
                        validation_rule="non_negative",
                    )
                )
        return FPSuccess(data)

    return pipeline.add_step(
        "ohlc_validation", ohlc_validator, PipelineStage.TYPE_VALIDATION
    ).add_step("volume_validation", volume_validator, PipelineStage.RANGE_VALIDATION)


# Export pipeline system
__all__ = [
    # Core types
    "PipelineStage",
    "ExecutionMode",
    "PipelineStep",
    "PipelineMetrics",
    "PipelineResult",
    # Main pipeline class
    "FunctionalValidationPipeline",
    # Factory functions
    "create_trade_action_pipeline",
    "create_market_data_pipeline",
]
