"""
Enhanced functional validation decorators with pure function patterns.

This module provides functional programming enhanced validation decorators that:
- Use pure functions for validation logic
- Support composable validation chains
- Provide monadic error handling
- Enable functional composition of validators
- Maintain backward compatibility with existing decorators
"""

import functools
import inspect
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any, ParamSpec, Protocol, TypeVar, cast, overload, Union

from bot.fp.core.functional_validation import (
    FieldError, SchemaError, ValidationChainError, ValidatorError,
    validate_positive, validate_range, validate_string_length, validate_enum,
    chain_validators, validate_all, ValidationPipeline,
    FPResult, FPSuccess, FPFailure
)
from bot.fp.types.base import Money, Percentage
from bot.trading_types import Position, TradeAction
from bot.types.exceptions import (
    BalanceValidationError,
    PositionValidationError,
    TradeValidationError,
    ValidationError,
)

# Type variables for decorated functions
P = ParamSpec("P")
T = TypeVar("T")


# Protocol for functions that can be sync or async
class SyncOrAsyncCallable(Protocol[P, T]):
    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...
    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[T]: ...


# Functional Validation Decorator Base

def functional_validator(
    validator_fn: Callable[[Any], FPResult[Any, ValidatorError]],
    arg_name: str = "value",
    error_converter: Callable[[ValidatorError], Exception] = lambda e: ValidationError(str(e))
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Generic functional validator decorator.
    
    Args:
        validator_fn: Pure validation function that returns FPResult
        arg_name: Name of the argument to validate
        error_converter: Function to convert validation errors to exceptions
    
    Returns:
        Decorated function with functional validation
    """
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Handle synchronous functions
        if not inspect.iscoroutinefunction(func):
            
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Extract value to validate
                value, arg_index = _extract_argument(func, args, kwargs, arg_name)
                
                # Apply functional validation
                result = validator_fn(value)
                if result.is_failure():
                    raise error_converter(result.failure())
                
                # Update argument with validated value
                args, kwargs = _update_argument(args, kwargs, arg_name, arg_index, result.success())
                
                return func(*args, **kwargs)
            
            return cast("Callable[P, T]", sync_wrapper)
        
        # Handle asynchronous functions
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Extract value to validate
            value, arg_index = _extract_argument(func, args, kwargs, arg_name)
            
            # Apply functional validation
            result = validator_fn(value)
            if result.is_failure():
                raise error_converter(result.failure())
            
            # Update argument with validated value
            args, kwargs = _update_argument(args, kwargs, arg_name, arg_index, result.success())
            
            return await func(*args, **kwargs)  # type: ignore[misc]
        
        return cast("Callable[P, T]", async_wrapper)
    
    return decorator


def functional_chain_validator(
    validators: list[Callable[[Any], FPResult[Any, ValidatorError]]],
    arg_name: str = "value",
    error_converter: Callable[[list[ValidatorError]], Exception] = lambda errors: ValidationError(f"Multiple validation errors: {errors}")
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Functional validator decorator that chains multiple validation functions.
    
    Args:
        validators: List of validation functions to chain
        arg_name: Name of the argument to validate
        error_converter: Function to convert validation errors to exceptions
    
    Returns:
        Decorated function with chained functional validation
    """
    
    def chain_validator(value: Any) -> FPResult[Any, list[ValidatorError]]:
        return chain_validators(*validators)(value)
    
    def multi_error_converter(errors: list[ValidatorError]) -> Exception:
        return error_converter(errors)
    
    return functional_validator(chain_validator, arg_name, multi_error_converter)


# Enhanced Balance Validation with Functional Programming

def functional_balance_validator(
    require_positive: bool = True,
    min_value: float | None = None,
    max_value: float | None = None,
    precision: int = 2,
    balance_arg_name: str = "balance"
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Functional balance validator with composable validation rules.
    
    Args:
        require_positive: Whether the balance must be positive
        min_value: Minimum allowed balance value
        max_value: Maximum allowed balance value
        precision: Decimal precision for rounding
        balance_arg_name: Name of the balance argument to validate
    
    Returns:
        Decorated function with functional balance validation
    """
    
    # Build validation chain
    validators = []
    
    if require_positive:
        validators.append(validate_positive)
    
    if min_value is not None and max_value is not None:
        validators.append(validate_range(min_value, max_value))
    elif min_value is not None:
        validators.append(validate_range(min_value, float('inf')))
    elif max_value is not None:
        validators.append(validate_range(float('-inf'), max_value))
    
    def balance_validator(value: Any) -> FPResult[Decimal, ValidatorError]:
        # Convert to Decimal first
        try:
            decimal_value = Decimal(str(value))
        except (ValueError, TypeError) as e:
            return FPFailure(FieldError(
                field=balance_arg_name,
                message=f"Cannot convert to Decimal: {e}",
                value=str(value),
                validation_rule="decimal_conversion"
            ))
        
        # Apply validation chain
        float_value = float(decimal_value)
        chain_result = chain_validators(*validators)(float_value)
        
        if chain_result.is_failure():
            return FPFailure(chain_result.failure())
        
        # Round to specified precision
        precision_str = f"0.{'0' * precision}"
        rounded_value = decimal_value.quantize(Decimal(precision_str))
        
        return FPSuccess(rounded_value)
    
    def error_converter(error: ValidatorError) -> BalanceValidationError:
        if isinstance(error, FieldError):
            return BalanceValidationError(
                error.message,
                field_name=error.field,
                invalid_value=error.value,
                validation_rule=error.validation_rule
            )
        return BalanceValidationError(str(error))
    
    return functional_validator(balance_validator, balance_arg_name, error_converter)


# Enhanced Trade Action Validation

def functional_trade_action_validator(
    trade_action_arg_name: str = "trade_action",
    allow_none: bool = False,
    validate_risk_params: bool = True,
    size_range: tuple[float, float] = (0, 100),
    tp_range: tuple[float, float] = (0, 50),
    sl_range: tuple[float, float] = (0, 25)
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Functional trade action validator with configurable constraints.
    
    Args:
        trade_action_arg_name: Name of the trade action argument to validate
        allow_none: Whether None values are allowed
        validate_risk_params: Whether to validate stop loss and take profit
        size_range: Allowed range for position size percentage
        tp_range: Allowed range for take profit percentage
        sl_range: Allowed range for stop loss percentage
    
    Returns:
        Decorated function with functional trade action validation
    """
    
    def trade_action_validator(value: Any) -> FPResult[TradeAction, ValidatorError]:
        # Handle None values
        if value is None:
            if allow_none:
                return FPSuccess(None)
            return FPFailure(FieldError(
                field=trade_action_arg_name,
                message="Trade action cannot be None",
                validation_rule="not_null"
            ))
        
        # Validate type
        if not isinstance(value, TradeAction):
            return FPFailure(FieldError(
                field=trade_action_arg_name,
                message=f"Expected TradeAction, got {type(value).__name__}",
                value=str(value),
                validation_rule="type_check"
            ))
        
        # Validate action
        valid_actions = {"LONG", "SHORT", "CLOSE", "HOLD"}
        if value.action not in valid_actions:
            return FPFailure(FieldError(
                field="action",
                message=f"Invalid action: {value.action}",
                value=value.action,
                validation_rule="enum",
                context={"valid_values": list(valid_actions)}
            ))
        
        # Validate size percentage
        if not (size_range[0] <= value.size_pct <= size_range[1]):
            return FPFailure(FieldError(
                field="size_pct",
                message=f"Size percentage must be between {size_range[0]} and {size_range[1]}",
                value=str(value.size_pct),
                validation_rule="range",
                context={"min": size_range[0], "max": size_range[1]}
            ))
        
        if validate_risk_params and value.action in ["LONG", "SHORT"]:
            # Validate take profit
            if not (tp_range[0] < value.take_profit_pct <= tp_range[1]):
                return FPFailure(FieldError(
                    field="take_profit_pct",
                    message=f"Take profit must be between {tp_range[0]} and {tp_range[1]}",
                    value=str(value.take_profit_pct),
                    validation_rule="range",
                    context={"min": tp_range[0], "max": tp_range[1]}
                ))
            
            # Validate stop loss
            if not (sl_range[0] < value.stop_loss_pct <= sl_range[1]):
                return FPFailure(FieldError(
                    field="stop_loss_pct",
                    message=f"Stop loss must be between {sl_range[0]} and {sl_range[1]}",
                    value=str(value.stop_loss_pct),
                    validation_rule="range",
                    context={"min": sl_range[0], "max": sl_range[1]}
                ))
        
        return FPSuccess(value)
    
    def error_converter(error: ValidatorError) -> TradeValidationError:
        if isinstance(error, FieldError):
            return TradeValidationError(
                error.message,
                field_name=error.field,
                invalid_value=error.value,
                validation_rule=error.validation_rule
            )
        return TradeValidationError(str(error))
    
    return functional_validator(trade_action_validator, trade_action_arg_name, error_converter)


# Enhanced Position Validation

def functional_position_validator(
    position_arg_name: str = "position",
    allow_none: bool = False,
    require_open: bool = False,
    min_size: float = 0.0,
    max_size: float = float('inf')
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Functional position validator with configurable constraints.
    
    Args:
        position_arg_name: Name of the position argument to validate
        allow_none: Whether None values are allowed
        require_open: Whether the position must be open (not FLAT)
        min_size: Minimum position size
        max_size: Maximum position size
    
    Returns:
        Decorated function with functional position validation
    """
    
    def position_validator(value: Any) -> FPResult[Position, ValidatorError]:
        # Handle None values
        if value is None:
            if allow_none:
                return FPSuccess(None)
            return FPFailure(FieldError(
                field=position_arg_name,
                message="Position cannot be None",
                validation_rule="not_null"
            ))
        
        # Validate type
        if not isinstance(value, Position):
            return FPFailure(FieldError(
                field=position_arg_name,
                message=f"Expected Position, got {type(value).__name__}",
                value=str(value),
                validation_rule="type_check"
            ))
        
        # Validate position side
        valid_sides = {"LONG", "SHORT", "FLAT"}
        if value.side not in valid_sides:
            return FPFailure(FieldError(
                field="side",
                message=f"Invalid position side: {value.side}",
                value=value.side,
                validation_rule="enum",
                context={"valid_values": list(valid_sides)}
            ))
        
        # Check if position is open when required
        if require_open and value.side == "FLAT":
            return FPFailure(FieldError(
                field="side",
                message="Position must be open (not FLAT)",
                value=value.side,
                validation_rule="open_position_required"
            ))
        
        # Validate position size
        if not (min_size <= value.size <= max_size):
            return FPFailure(FieldError(
                field="size",
                message=f"Position size must be between {min_size} and {max_size}",
                value=str(value.size),
                validation_rule="range",
                context={"min": min_size, "max": max_size}
            ))
        
        return FPSuccess(value)
    
    def error_converter(error: ValidatorError) -> PositionValidationError:
        if isinstance(error, FieldError):
            return PositionValidationError(
                error.message,
                field_name=error.field,
                invalid_value=error.value,
                validation_rule=error.validation_rule
            )
        return PositionValidationError(str(error))
    
    return functional_validator(position_validator, position_arg_name, error_converter)


# Functional Percentage Validation

def functional_percentage_validator(
    percentage_arg_name: str = "percentage",
    min_value: float = 0.0,
    max_value: float = 100.0,
    allow_none: bool = False,
    normalize_format: bool = True  # Convert 0-100 to 0-1 if needed
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Functional percentage validator with automatic format normalization.
    
    Args:
        percentage_arg_name: Name of the percentage argument to validate
        min_value: Minimum allowed percentage
        max_value: Maximum allowed percentage
        allow_none: Whether None values are allowed
        normalize_format: Whether to normalize 0-100 format to 0-1
    
    Returns:
        Decorated function with functional percentage validation
    """
    
    def percentage_validator(value: Any) -> FPResult[float, ValidatorError]:
        # Handle None values
        if value is None:
            if allow_none:
                return FPSuccess(None)
            return FPFailure(FieldError(
                field=percentage_arg_name,
                message="Percentage cannot be None",
                validation_rule="not_null"
            ))
        
        # Convert to float
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            return FPFailure(FieldError(
                field=percentage_arg_name,
                message=f"Cannot convert to float: {e}",
                value=str(value),
                validation_rule="type_conversion"
            ))
        
        # Normalize format if needed
        if normalize_format and float_value > 1.0 and max_value <= 1.0:
            float_value = float_value / 100.0
        
        # Validate range
        if not (min_value <= float_value <= max_value):
            return FPFailure(FieldError(
                field=percentage_arg_name,
                message=f"Percentage must be between {min_value} and {max_value}",
                value=str(float_value),
                validation_rule="range",
                context={"min": min_value, "max": max_value}
            ))
        
        return FPSuccess(float_value)
    
    def error_converter(error: ValidatorError) -> ValidationError:
        if isinstance(error, FieldError):
            return ValidationError(
                error.message,
                field_name=error.field,
                invalid_value=error.value,
                validation_rule=error.validation_rule
            )
        return ValidationError(str(error))
    
    return functional_validator(percentage_validator, percentage_arg_name, error_converter)


# Functional Pipeline Validator

def functional_pipeline_validator(
    pipeline: ValidationPipeline,
    arg_name: str = "value",
    error_converter: Callable[[ValidationChainError], Exception] = lambda e: ValidationError(str(e))
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator that applies a validation pipeline to an argument.
    
    Args:
        pipeline: ValidationPipeline to apply
        arg_name: Name of the argument to validate
        error_converter: Function to convert validation errors to exceptions
    
    Returns:
        Decorated function with pipeline validation
    """
    
    def pipeline_validator(value: Any) -> FPResult[Any, ValidationChainError]:
        return pipeline.validate(value)
    
    return functional_validator(pipeline_validator, arg_name, error_converter)


# Utility Functions

def _extract_argument(func: Callable, args: tuple, kwargs: dict, arg_name: str) -> tuple[Any, int]:
    """Extract argument value and position from function call."""
    arg_index = -1
    
    # Check kwargs first
    if arg_name in kwargs:
        return kwargs[arg_name], arg_index
    
    # Get function signature to find argument position
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    if arg_name in params:
        arg_index = params.index(arg_name)
        if arg_index < len(args):
            return args[arg_index], arg_index
    
    raise ValueError(f"Argument '{arg_name}' not found in function call")


def _update_argument(args: tuple, kwargs: dict, arg_name: str, arg_index: int, new_value: Any) -> tuple[tuple, dict]:
    """Update argument with validated value."""
    # Update kwargs if present
    if arg_name in kwargs:
        kwargs = kwargs.copy()
        kwargs[arg_name] = new_value
        return args, kwargs
    
    # Update positional args
    if arg_index >= 0 and arg_index < len(args):
        args_list = list(args)
        args_list[arg_index] = new_value
        return tuple(args_list), kwargs
    
    return args, kwargs


# Export functional validation decorators
__all__ = [
    # Core functional decorators
    "functional_validator",
    "functional_chain_validator",
    "functional_pipeline_validator",
    
    # Enhanced specific validators
    "functional_balance_validator",
    "functional_trade_action_validator", 
    "functional_position_validator",
    "functional_percentage_validator",
    
    # Utility functions
    "_extract_argument",
    "_update_argument",
]