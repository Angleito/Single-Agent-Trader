"""
Validation decorators for common validation patterns.

This module provides decorators that can be applied to functions to automatically
validate inputs and outputs, reducing boilerplate code and ensuring consistency.
"""

import functools
import inspect
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any, ParamSpec, Protocol, TypeVar, Union, cast, overload

from bot.trading_types import Position, TradeAction
from bot.types.exceptions import (
    BalanceValidationError,
    PositionValidationError,
    TradeValidationError,
    ValidationError,
)
from bot.types.guards import (
    ensure_decimal,
    ensure_positive_decimal,
    is_valid_price,
    is_valid_quantity,
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


def validate_balance(
    balance_arg_name: str = "balance",
    require_positive: bool = True,
    max_value: Decimal | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to validate Decimal balance values.

    Args:
        balance_arg_name: Name of the balance argument to validate
        require_positive: Whether the balance must be positive
        max_value: Maximum allowed balance value

    Returns:
        Decorated function with balance validation

    Example:
        ```python
        @validate_balance("available_balance", require_positive=True)
        def calculate_position_size(self, available_balance: Decimal) -> Decimal:
            # Function implementation
            pass
        ```

    Raises:
        BalanceValidationError: If balance validation fails
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the balance value from args or kwargs
            balance_value = None

            # Check kwargs first
            if balance_arg_name in kwargs:
                balance_value = kwargs[balance_arg_name]
            else:
                # Get function signature to find argument position

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                if balance_arg_name in params:
                    arg_index = params.index(balance_arg_name)
                    if arg_index < len(args):
                        balance_value = args[arg_index]

            if balance_value is None:
                raise BalanceValidationError(
                    f"Balance argument '{balance_arg_name}' not provided",
                    field_name=balance_arg_name,
                )

            # Validate the balance
            try:
                if require_positive:
                    validated_balance = ensure_positive_decimal(
                        balance_value, balance_arg_name
                    )
                else:
                    validated_balance = ensure_decimal(balance_value, balance_arg_name)

                # Check max value if specified
                if max_value is not None and validated_balance > max_value:
                    raise BalanceValidationError(
                        f"Balance exceeds maximum allowed value of {max_value}",
                        field_name=balance_arg_name,
                        invalid_value=validated_balance,
                        validation_rule="max_value",
                        required_balance=max_value,
                        available_balance=validated_balance,
                    )

                # Update the argument with validated value
                if balance_arg_name in kwargs:
                    kwargs[balance_arg_name] = validated_balance
                else:
                    args = list(args)
                    args[arg_index] = validated_balance
                    args = tuple(args)

            except ValidationError as e:
                # Convert to BalanceValidationError if needed
                if not isinstance(e, BalanceValidationError):
                    raise BalanceValidationError(
                        e.message,
                        field_name=balance_arg_name,
                        invalid_value=balance_value,
                        validation_rule=e.validation_rule,
                    ) from e
                raise

            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def validate_trade_action(
    trade_action_arg_name: str = "trade_action",
    allow_none: bool = False,
    validate_risk_params: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to validate TradeAction objects.

    Args:
        trade_action_arg_name: Name of the trade action argument to validate
        allow_none: Whether None values are allowed
        validate_risk_params: Whether to validate stop loss and take profit

    Returns:
        Decorated function with trade action validation

    Example:
        ```python
        @validate_trade_action("action", validate_risk_params=True)
        def execute_trade(self, action: TradeAction) -> Order:
            # Function implementation
            pass
        ```

    Raises:
        TradeValidationError: If trade action validation fails
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the trade action value
            trade_action_value = None

            # Check kwargs first
            if trade_action_arg_name in kwargs:
                trade_action_value = kwargs[trade_action_arg_name]
            else:
                # Get function signature to find argument position

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                if trade_action_arg_name in params:
                    arg_index = params.index(trade_action_arg_name)
                    if arg_index < len(args):
                        trade_action_value = args[arg_index]

            # Handle None values
            if trade_action_value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise TradeValidationError(
                    f"Trade action argument '{trade_action_arg_name}' not provided",
                    field_name=trade_action_arg_name,
                )

            # Validate the trade action
            if not isinstance(trade_action_value, TradeAction):
                raise TradeValidationError(
                    f"Expected TradeAction object, got {type(trade_action_value).__name__}",
                    field_name=trade_action_arg_name,
                    invalid_value=str(trade_action_value),
                    validation_rule="type_check",
                )

            # Additional validation based on action type
            if validate_risk_params and trade_action_value.action in ["LONG", "SHORT"]:
                if trade_action_value.stop_loss_pct <= 0:
                    raise TradeValidationError(
                        f"Stop loss must be greater than 0 for {trade_action_value.action} action",
                        field_name="stop_loss_pct",
                        invalid_value=trade_action_value.stop_loss_pct,
                        validation_rule="positive_value",
                        trade_action=trade_action_value.action,
                    )

                if trade_action_value.take_profit_pct <= 0:
                    raise TradeValidationError(
                        f"Take profit must be greater than 0 for {trade_action_value.action} action",
                        field_name="take_profit_pct",
                        invalid_value=trade_action_value.take_profit_pct,
                        validation_rule="positive_value",
                        trade_action=trade_action_value.action,
                    )

                if (
                    trade_action_value.size_pct <= 0
                    or trade_action_value.size_pct > 100
                ):
                    raise TradeValidationError(
                        "Position size must be between 0 and 100 percent",
                        field_name="size_pct",
                        invalid_value=trade_action_value.size_pct,
                        validation_rule="percentage_range",
                        trade_action=trade_action_value.action,
                    )

            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def validate_position(
    position_arg_name: str = "position",
    allow_none: bool = False,
    require_open: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to validate Position objects.

    Args:
        position_arg_name: Name of the position argument to validate
        allow_none: Whether None values are allowed
        require_open: Whether the position must be open (not FLAT)

    Returns:
        Decorated function with position validation

    Example:
        ```python
        @validate_position("current_position", require_open=True)
        def calculate_pnl(self, current_position: Position) -> Decimal:
            # Function implementation
            pass
        ```

    Raises:
        PositionValidationError: If position validation fails
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the position value
            position_value = None

            # Check kwargs first
            if position_arg_name in kwargs:
                position_value = kwargs[position_arg_name]
            else:
                # Get function signature to find argument position

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                if position_arg_name in params:
                    arg_index = params.index(position_arg_name)
                    if arg_index < len(args):
                        position_value = args[arg_index]

            # Handle None values
            if position_value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise PositionValidationError(
                    f"Position argument '{position_arg_name}' not provided",
                    field_name=position_arg_name,
                )

            # Validate the position
            if not isinstance(position_value, Position):
                raise PositionValidationError(
                    f"Expected Position object, got {type(position_value).__name__}",
                    field_name=position_arg_name,
                    invalid_value=str(position_value),
                    validation_rule="type_check",
                )

            # Validate position fields
            if not is_valid_quantity(position_value.size):
                raise PositionValidationError(
                    "Invalid position size",
                    field_name="size",
                    invalid_value=position_value.size,
                    validation_rule="quantity_validation",
                    position_side=position_value.side,
                    position_size=position_value.size,
                )

            # Check if position is open when required
            if require_open and position_value.side == "FLAT":
                raise PositionValidationError(
                    "Position must be open (not FLAT)",
                    field_name="side",
                    invalid_value=position_value.side,
                    validation_rule="open_position_required",
                    position_side=position_value.side,
                )

            # Validate entry price for open positions
            if (
                position_value.side != "FLAT"
                and position_value.entry_price is not None
                and not is_valid_price(position_value.entry_price)
            ):
                raise PositionValidationError(
                    "Invalid entry price",
                    field_name="entry_price",
                    invalid_value=position_value.entry_price,
                    validation_rule="price_validation",
                    position_side=position_value.side,
                    position_size=position_value.size,
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def validate_percentage(
    percentage_arg_name: str = "percentage",
    min_value: float = 0.0,
    max_value: float = 100.0,
    allow_none: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to validate percentage values.

    Args:
        percentage_arg_name: Name of the percentage argument to validate
        min_value: Minimum allowed percentage (default: 0.0)
        max_value: Maximum allowed percentage (default: 100.0)
        allow_none: Whether None values are allowed

    Returns:
        Decorated function with percentage validation

    Example:
        ```python
        @validate_percentage("risk_pct", min_value=0.1, max_value=5.0)
        def set_risk_percentage(self, risk_pct: float) -> None:
            # Function implementation
            pass
        ```

    Raises:
        ValidationError: If percentage validation fails
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the percentage value
            percentage_value = None

            # Check kwargs first
            if percentage_arg_name in kwargs:
                percentage_value = kwargs[percentage_arg_name]
            else:
                # Get function signature to find argument position

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                if percentage_arg_name in params:
                    arg_index = params.index(percentage_arg_name)
                    if arg_index < len(args):
                        percentage_value = args[arg_index]

            # Handle None values
            if percentage_value is None:
                if allow_none:
                    return func(*args, **kwargs)
                raise ValidationError(
                    f"Percentage argument '{percentage_arg_name}' not provided",
                    field_name=percentage_arg_name,
                )

            # Validate the percentage
            try:
                float_value = float(percentage_value)
            except (ValueError, TypeError) as e:
                raise ValidationError(
                    f"Invalid percentage value for {percentage_arg_name}",
                    field_name=percentage_arg_name,
                    invalid_value=percentage_value,
                    validation_rule="type_conversion",
                ) from e

            if not (min_value <= float_value <= max_value):
                raise ValidationError(
                    f"Percentage must be between {min_value} and {max_value}",
                    field_name=percentage_arg_name,
                    invalid_value=float_value,
                    validation_rule="range_check",
                )

            # Update the argument with validated value
            if percentage_arg_name in kwargs:
                kwargs[percentage_arg_name] = float_value
            else:
                args = list(args)
                args[arg_index] = float_value
                args = tuple(args)

            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
