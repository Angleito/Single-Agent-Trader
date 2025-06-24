"""
Enhanced functional validation module for type safety and data conversion.

This module provides:
- Pure functional validation primitives
- Composable validation combinators
- Monadic error handling with detailed error information
- Type-safe data converters
- Validation pipelines for complex data flows
- Trading-specific validators
- Parallel and batch validation support
- Backward compatibility with existing validation APIs
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Any, TypeVar, Union

from pydantic import BaseModel, Field

from bot.fp.types.base import Money, Percentage
from bot.fp.types.result import Failure as FPFailure
from bot.fp.types.result import Result as FPResult
from bot.fp.types.result import Success as FPSuccess
from bot.trading_types import TradeAction

# Type variables for functional programming
T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")

# Type alias for validation results
ValidationResult = FPResult


# Enhanced Validation Error Types
class FieldError(BaseModel):
    """Individual field validation error."""

    field: str
    message: str
    value: str | None = None
    expected_type: str | None = None
    validation_rule: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        context_str = f" (context: {self.context})" if self.context else ""
        return f"Field '{self.field}': {self.message}{context_str}"


class SchemaError(BaseModel):
    """Schema validation error."""

    schema: str
    errors: list[FieldError]
    path: list[str] = Field(default_factory=list)
    severity: str = "error"  # "error", "warning", "info"

    def __str__(self) -> str:
        path_str = ".".join(self.path) if self.path else "root"
        error_summary = ", ".join([str(e) for e in self.errors])
        return f"Schema '{self.schema}' at '{path_str}': {error_summary}"


class ConversionError(BaseModel):
    """Type conversion error."""

    source_type: str
    target_type: str
    message: str
    source_value: Any = None
    conversion_path: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        path_str = (
            " -> ".join(self.conversion_path) if self.conversion_path else "direct"
        )
        return f"Conversion {self.source_type} -> {self.target_type} ({path_str}): {self.message}"


class ValidationChainError(BaseModel):
    """Error in validation chain execution."""

    chain_step: str
    errors: list[FieldError | SchemaError | ConversionError]
    partial_result: Any = None

    def __str__(self) -> str:
        error_count = len(self.errors)
        return f"Validation chain failed at step '{self.chain_step}' with {error_count} error(s)"


# Union type for all validation errors
ValidatorError = Union[FieldError, SchemaError, ConversionError, ValidationChainError]


# Functional Validation Primitives


def validate_positive(
    value: float, field_name: str = "value"
) -> FPResult[float, FieldError]:
    """Pure function to validate positive numbers."""
    if value <= 0:
        return FPFailure(
            FieldError(
                field=field_name,
                message=f"Must be positive, got {value}",
                value=str(value),
                validation_rule="positive",
            )
        )
    return FPSuccess(value)


def validate_range(
    min_val: float, max_val: float
) -> Callable[[float], FPResult[float, FieldError]]:
    """Higher-order function to create range validators."""

    def validator(
        value: float, field_name: str = "value"
    ) -> FPResult[float, FieldError]:
        if not (min_val <= value <= max_val):
            return FPFailure(
                FieldError(
                    field=field_name,
                    message=f"Must be between {min_val} and {max_val}, got {value}",
                    value=str(value),
                    validation_rule="range",
                    context={"min": min_val, "max": max_val},
                )
            )
        return FPSuccess(value)

    return validator


def validate_string_length(
    min_len: int = 0, max_len: int = 1000
) -> Callable[[str], FPResult[str, FieldError]]:
    """Higher-order function to create string length validators."""

    def validator(value: str, field_name: str = "value") -> FPResult[str, FieldError]:
        if not (min_len <= len(value) <= max_len):
            return FPFailure(
                FieldError(
                    field=field_name,
                    message=f"Length must be between {min_len} and {max_len}, got {len(value)}",
                    value=value[:50] + "..." if len(value) > 50 else value,
                    validation_rule="length",
                    context={
                        "min_length": min_len,
                        "max_length": max_len,
                        "actual_length": len(value),
                    },
                )
            )
        return FPSuccess(value)

    return validator


def validate_pattern(
    pattern: str, description: str = "pattern"
) -> Callable[[str], FPResult[str, FieldError]]:
    """Higher-order function to create regex pattern validators."""
    import re

    compiled_pattern = re.compile(pattern)

    def validator(value: str, field_name: str = "value") -> FPResult[str, FieldError]:
        if not compiled_pattern.match(value):
            return FPFailure(
                FieldError(
                    field=field_name,
                    message=f"Must match {description} pattern",
                    value=value,
                    validation_rule="pattern",
                    context={"pattern": pattern, "description": description},
                )
            )
        return FPSuccess(value)

    return validator


def validate_enum(
    valid_values: set[Any], case_sensitive: bool = True
) -> Callable[[Any], FPResult[Any, FieldError]]:
    """Higher-order function to create enum validators."""

    def validator(value: Any, field_name: str = "value") -> FPResult[Any, FieldError]:
        check_value = value if case_sensitive else str(value).upper()
        check_set = (
            valid_values if case_sensitive else {str(v).upper() for v in valid_values}
        )

        if check_value not in check_set:
            return FPFailure(
                FieldError(
                    field=field_name,
                    message=f"Must be one of {valid_values}, got {value}",
                    value=str(value),
                    validation_rule="enum",
                    context={
                        "valid_values": list(valid_values),
                        "case_sensitive": case_sensitive,
                    },
                )
            )
        return FPSuccess(value)

    return validator


# Validation Chain Combinators


def chain_validators(
    *validators: Callable[[T], FPResult[T, E]],
) -> Callable[[T], FPResult[T, list[E]]]:
    """Chain multiple validators, collecting all errors."""

    def chained_validator(value: T) -> FPResult[T, list[E]]:
        errors = []
        current_value = value

        for validator in validators:
            result = validator(current_value)
            if result.is_failure():
                errors.append(result.failure())
            else:
                current_value = result.success()

        if errors:
            return FPFailure(errors)
        return FPSuccess(current_value)

    return chained_validator


def validate_all(
    validators: dict[str, Callable[[Any], FPResult[Any, FieldError]]],
) -> Callable[[dict[str, Any]], FPResult[dict[str, Any], SchemaError]]:
    """Validate all fields in a dictionary."""

    def validator(data: dict[str, Any]) -> FPResult[dict[str, Any], SchemaError]:
        errors = []
        validated_data = {}

        for field_name, field_validator in validators.items():
            if field_name in data:
                result = field_validator(data[field_name], field_name)
                if result.is_failure():
                    errors.append(result.failure())
                else:
                    validated_data[field_name] = result.success()
            else:
                errors.append(
                    FieldError(
                        field=field_name,
                        message="Required field missing",
                        validation_rule="required",
                    )
                )

        if errors:
            return FPFailure(SchemaError(schema="object", errors=errors))

        return FPSuccess(validated_data)

    return validator


def optional_field(
    validator: Callable[[T], FPResult[T, E]], default: T = None
) -> Callable[[T | None], FPResult[T | None, E]]:
    """Make a validator optional with a default value."""

    def optional_validator(value: T | None) -> FPResult[T | None, E]:
        if value is None:
            return FPSuccess(default)
        return validator(value)

    return optional_validator


# Functional Data Converters


def convert_decimal(value: Any) -> FPResult[Decimal, ConversionError]:
    """Pure function to convert values to Decimal."""
    try:
        if isinstance(value, Decimal):
            return FPSuccess(value)
        return FPSuccess(Decimal(str(value)))
    except (ValueError, TypeError, ArithmeticError) as e:
        return FPFailure(
            ConversionError(
                source_type=type(value).__name__,
                target_type="Decimal",
                message=str(e),
                source_value=str(value) if value is not None else None,
            )
        )


def convert_percentage(value: Any) -> FPResult[Percentage, ConversionError]:
    """Pure function to convert values to Percentage."""
    try:
        if isinstance(value, Percentage):
            return FPSuccess(value)

        # Handle percentage values > 1 (assume they're in 0-100 format)
        numeric_value = float(value)
        if numeric_value > 1:
            numeric_value = numeric_value / 100

        result = Percentage.create(numeric_value)
        if result.is_success():
            return FPSuccess(result.success())
        return FPFailure(
            ConversionError(
                source_type=type(value).__name__,
                target_type="Percentage",
                message=result.failure(),
                source_value=str(value),
            )
        )
    except (ValueError, TypeError) as e:
        return FPFailure(
            ConversionError(
                source_type=type(value).__name__,
                target_type="Percentage",
                message=str(e),
                source_value=str(value) if value is not None else None,
            )
        )


def convert_money(
    amount: Any, currency: str = "USD"
) -> FPResult[Money, ConversionError]:
    """Pure function to convert values to Money."""
    try:
        if isinstance(amount, Money):
            return FPSuccess(amount)

        result = Money.create(float(amount), currency)
        if result.is_success():
            return FPSuccess(result.success())
        return FPFailure(
            ConversionError(
                source_type=type(amount).__name__,
                target_type="Money",
                message=result.failure(),
                source_value=str(amount),
            )
        )
    except (ValueError, TypeError) as e:
        return FPFailure(
            ConversionError(
                source_type=type(amount).__name__,
                target_type="Money",
                message=str(e),
                source_value=str(amount) if amount is not None else None,
            )
        )


# Validation Pipeline


class ValidationPipeline:
    """Functional validation pipeline for composable validation chains."""

    def __init__(self):
        self.steps: list[tuple[str, Callable[[Any], FPResult[Any, ValidatorError]]]] = (
            []
        )

    def add_step(
        self, name: str, validator: Callable[[Any], FPResult[Any, ValidatorError]]
    ) -> "ValidationPipeline":
        """Add a validation step to the pipeline."""
        self.steps.append((name, validator))
        return self

    def validate(self, data: Any) -> FPResult[Any, ValidationChainError]:
        """Execute the validation pipeline."""
        current_data = data

        for step_name, validator in self.steps:
            result = validator(current_data)
            if result.is_failure():
                error = result.failure()
                return FPFailure(
                    ValidationChainError(
                        chain_step=step_name,
                        errors=[error] if not isinstance(error, list) else error,
                        partial_result=current_data,
                    )
                )
            current_data = result.success()

        return FPSuccess(current_data)

    def validate_collect_errors(
        self, data: Any
    ) -> FPResult[Any, list[ValidationChainError]]:
        """Execute pipeline collecting all errors instead of failing fast."""
        current_data = data
        errors = []

        for step_name, validator in self.steps:
            result = validator(current_data)
            if result.is_failure():
                error = result.failure()
                errors.append(
                    ValidationChainError(
                        chain_step=step_name,
                        errors=[error] if not isinstance(error, list) else error,
                        partial_result=current_data,
                    )
                )
            else:
                current_data = result.success()

        if errors:
            return FPFailure(errors)
        return FPSuccess(current_data)


# Trading-Specific Validators


def validate_trade_action_functional(
    action_data: dict[str, Any],
) -> FPResult[TradeAction, SchemaError]:
    """Functional validation for trade actions."""
    validators = {
        "action": validate_enum(
            {"LONG", "SHORT", "CLOSE", "HOLD"}, case_sensitive=False
        ),
        "size_pct": chain_validators(validate_positive, validate_range(0, 100)),
        "take_profit_pct": validate_positive,
        "stop_loss_pct": validate_positive,
        "leverage": chain_validators(validate_positive, validate_range(1, 100)),
        "rationale": validate_string_length(min_len=1, max_len=500),
    }

    schema_validator = validate_all(validators)
    result = schema_validator(action_data)

    if result.is_failure():
        return result

    validated_data = result.success()

    try:
        # Convert to TradeAction object
        trade_action = TradeAction(
            action=validated_data["action"].upper(),
            size_pct=validated_data["size_pct"],
            take_profit_pct=validated_data["take_profit_pct"],
            stop_loss_pct=validated_data["stop_loss_pct"],
            leverage=int(validated_data.get("leverage", 1)),
            rationale=validated_data["rationale"],
        )
        return FPSuccess(trade_action)
    except Exception as e:
        return FPFailure(
            SchemaError(
                schema="TradeAction",
                errors=[
                    FieldError(
                        field="construction",
                        message=f"Failed to construct TradeAction: {e}",
                        validation_rule="object_creation",
                    )
                ],
            )
        )


def validate_position_functional(
    position_data: dict[str, Any],
) -> FPResult[dict[str, Any], SchemaError]:
    """Functional validation for position data."""
    validators = {
        "symbol": validate_string_length(min_len=1, max_len=20),
        "side": validate_enum({"LONG", "SHORT", "FLAT"}, case_sensitive=False),
        "size": validate_positive,
        "entry_price": optional_field(validate_positive),
        "unrealized_pnl": optional_field(lambda x: FPSuccess(x)),  # Allow any value
        "realized_pnl": optional_field(lambda x: FPSuccess(x)),  # Allow any value
    }

    return validate_all(validators)(position_data)


# Enhanced Batch Validation


def validate_batch_functional(
    items: list[T], validator_fn: Callable[[T], FPResult[U, E]]
) -> FPResult[list[U], list[tuple[int, E]]]:
    """Functionally validate a batch of items, collecting errors with indices."""
    results = []
    errors = []

    for i, item in enumerate(items):
        result = validator_fn(item)
        if result.is_success():
            results.append(result.success())
        else:
            errors.append((i, result.failure()))

    if errors:
        return FPFailure(errors)
    return FPSuccess(results)


def validate_batch_parallel(
    items: list[T], validator_fn: Callable[[T], FPResult[U, E]], max_workers: int = 4
) -> FPResult[list[U], list[tuple[int, E]]]:
    """Validate batch items in parallel using thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(items)
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all validation tasks
        future_to_index = {
            executor.submit(validator_fn, item): i for i, item in enumerate(items)
        }

        # Collect results
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                if result.is_success():
                    results[index] = result.success()
                else:
                    errors.append((index, result.failure()))
            except Exception as e:
                errors.append((index, str(e)))

    if errors:
        return FPFailure(errors)
    return FPSuccess([r for r in results if r is not None])


# Validation Composition Utilities


def compose_validators(
    *validators: Callable[[T], FPResult[T, E]],
) -> Callable[[T], FPResult[T, list[E]]]:
    """Compose multiple validators using function composition."""

    def composed(value: T) -> FPResult[T, list[E]]:
        errors = []
        current_value = value

        for validator in validators:
            result = validator(current_value)
            if result.is_failure():
                errors.append(result.failure())
            else:
                current_value = result.success()

        if errors:
            return FPFailure(errors)
        return FPSuccess(current_value)

    return composed


def any_validator(
    *validators: Callable[[T], FPResult[T, E]],
) -> Callable[[T], FPResult[T, list[E]]]:
    """Validator that succeeds if any of the provided validators succeed."""

    def any_valid(value: T) -> FPResult[T, list[E]]:
        errors = []
        for validator in validators:
            result = validator(value)
            if result.is_success():
                return FPSuccess(result.success())
            errors.append(result.failure())
        return FPFailure(errors)

    return any_valid


def conditional_validator(
    condition: Callable[[T], bool], validator: Callable[[T], FPResult[T, E]]
) -> Callable[[T], FPResult[T, E | None]]:
    """Apply validator only if condition is met."""

    def conditional(value: T) -> FPResult[T, E | None]:
        if condition(value):
            result = validator(value)
            if result.is_success():
                return FPSuccess(result.success())
            return FPFailure(result.failure())
        return FPSuccess(value)

    return conditional


# Data Integrity Validators


def validate_data_integrity(
    data: dict[str, Any], integrity_rules: dict[str, Callable[[Any], bool]]
) -> FPResult[dict[str, Any], SchemaError]:
    """Validate data integrity using custom rules."""
    errors = []

    for field_name, rule in integrity_rules.items():
        if field_name in data:
            try:
                if not rule(data[field_name]):
                    errors.append(
                        FieldError(
                            field=field_name,
                            message="Data integrity check failed",
                            value=str(data[field_name]),
                            validation_rule="integrity",
                        )
                    )
            except Exception as e:
                errors.append(
                    FieldError(
                        field=field_name,
                        message=f"Integrity check error: {e}",
                        value=str(data[field_name]),
                        validation_rule="integrity_error",
                    )
                )

    if errors:
        return FPFailure(SchemaError(schema="data_integrity", errors=errors))

    return FPSuccess(data)


def validate_cross_field_constraints(
    data: dict[str, Any], constraints: list[Callable[[dict[str, Any]], bool]]
) -> FPResult[dict[str, Any], SchemaError]:
    """Validate cross-field constraints."""
    errors = []

    for i, constraint in enumerate(constraints):
        try:
            if not constraint(data):
                errors.append(
                    FieldError(
                        field="cross_field",
                        message=f"Cross-field constraint {i + 1} failed",
                        validation_rule="cross_field_constraint",
                    )
                )
        except Exception as e:
            errors.append(
                FieldError(
                    field="cross_field",
                    message=f"Cross-field constraint {i + 1} error: {e}",
                    validation_rule="cross_field_error",
                )
            )

    if errors:
        return FPFailure(SchemaError(schema="cross_field_constraints", errors=errors))

    return FPSuccess(data)


# Export enhanced validation functions
__all__ = [
    # Core validation primitives
    "validate_positive",
    "validate_range",
    "validate_string_length",
    "validate_pattern",
    "validate_enum",
    # Validation combinators
    "chain_validators",
    "validate_all",
    "optional_field",
    "compose_validators",
    "any_validator",
    "conditional_validator",
    # Data converters
    "convert_decimal",
    "convert_percentage",
    "convert_money",
    # Validation pipeline
    "ValidationPipeline",
    # Trading-specific validators
    "validate_trade_action_functional",
    "validate_position_functional",
    # Batch validation
    "validate_batch_functional",
    "validate_batch_parallel",
    # Data integrity
    "validate_data_integrity",
    "validate_cross_field_constraints",
    # Error types
    "FieldError",
    "SchemaError",
    "ConversionError",
    "ValidationChainError",
    "ValidatorError",
]
