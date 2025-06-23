"""
Type guard functions for runtime type validation.

This module provides type narrowing functions and runtime validators
to ensure type safety when dealing with external data.
"""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, TypeGuard

from .base_types import Percentage, Price, Quantity, Symbol
from .exceptions import ValidationError


def is_valid_price(value: Any) -> TypeGuard[Price]:
    """
    Check if value is a valid price (positive Decimal).

    Args:
        value: Value to check

    Returns:
        True if value is a valid price
    """
    try:
        if value is None:
            return False
        decimal_value = Decimal(str(value))
        return decimal_value > 0 and decimal_value.is_finite()
    except (InvalidOperation, ValueError, TypeError):
        return False


def is_valid_quantity(value: Any) -> TypeGuard[Quantity]:
    """
    Check if value is a valid quantity (non-negative Decimal).

    Args:
        value: Value to check

    Returns:
        True if value is a valid quantity
    """
    try:
        if value is None:
            return False
        decimal_value = Decimal(str(value))
        return decimal_value >= 0 and decimal_value.is_finite()
    except (InvalidOperation, ValueError, TypeError):
        return False


def is_valid_percentage(value: Any) -> TypeGuard[Percentage]:
    """
    Check if value is a valid percentage (0-100).

    Args:
        value: Value to check

    Returns:
        True if value is a valid percentage
    """
    try:
        if value is None:
            return False
        float_value = float(value)
        return 0 <= float_value <= 100
    except (ValueError, TypeError):
        return False


def is_valid_symbol(value: Any) -> TypeGuard[Symbol]:
    """
    Check if value is a valid trading symbol.

    Args:
        value: Value to check

    Returns:
        True if value is a valid symbol
    """
    if not isinstance(value, str):
        return False

    # Check for common symbol patterns
    if not value or len(value) < 3:
        return False

    # Must contain hyphen for pairs like BTC-USD
    if "-" not in value:
        return False

    parts = value.split("-")
    if len(parts) != 2:
        return False

    # Both parts should be non-empty and alphanumeric
    return all(part.isalnum() and len(part) > 0 for part in parts)


def ensure_decimal(value: Any, field_name: str = "value") -> Decimal:
    """
    Ensure value is converted to Decimal or raise ValidationError.

    Args:
        value: Value to convert
        field_name: Name of field for error messages

    Returns:
        Decimal value

    Raises:
        ValidationError: If conversion fails
    """
    try:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as e:
        raise ValidationError(
            f"Invalid decimal value for {field_name}",
            field_name=field_name,
            invalid_value=value,
            validation_rule="decimal_conversion",
        ) from e


def ensure_positive_decimal(value: Any, field_name: str = "value") -> Decimal:
    """
    Ensure value is a positive Decimal or raise ValidationError.

    Args:
        value: Value to convert
        field_name: Name of field for error messages

    Returns:
        Positive Decimal value

    Raises:
        ValidationError: If conversion fails or value is not positive
    """
    decimal_value = ensure_decimal(value, field_name)

    if decimal_value <= 0:
        raise ValidationError(
            f"{field_name} must be positive",
            field_name=field_name,
            invalid_value=value,
            validation_rule="positive_value",
        )

    return decimal_value


def is_valid_timestamp(value: Any) -> TypeGuard[datetime]:
    """
    Check if value is a valid timestamp.

    Args:
        value: Value to check

    Returns:
        True if value is a valid timestamp
    """
    return isinstance(value, datetime)


def validate_dict_keys(
    data: dict[str, Any], required_keys: set[str], optional_keys: set[str] | None = None
) -> None:
    """
    Validate dictionary has required keys and no extra keys.

    Args:
        data: Dictionary to validate
        required_keys: Set of required keys
        optional_keys: Set of optional keys

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError(
            "Expected dictionary",
            invalid_value=type(data).__name__,
            validation_rule="type_check",
        )

    data_keys = set(data.keys())
    missing_keys = required_keys - data_keys

    if missing_keys:
        raise ValidationError(
            f"Missing required keys: {missing_keys}",
            invalid_value=list(missing_keys),
            validation_rule="required_keys",
        )

    if optional_keys is not None:
        allowed_keys = required_keys | optional_keys
        extra_keys = data_keys - allowed_keys

        if extra_keys:
            raise ValidationError(
                f"Unexpected keys: {extra_keys}",
                invalid_value=list(extra_keys),
                validation_rule="allowed_keys",
            )


def validate_enum_value(
    value: Any, enum_values: set[str], field_name: str = "value"
) -> str:
    """
    Validate value is in allowed enum values.

    Args:
        value: Value to validate
        enum_values: Set of allowed values
        field_name: Name of field for error messages

    Returns:
        Validated string value

    Raises:
        ValidationError: If value not in enum
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} must be a string",
            field_name=field_name,
            invalid_value=value,
            validation_rule="type_check",
        )

    if value not in enum_values:
        raise ValidationError(
            f"{field_name} must be one of {enum_values}",
            field_name=field_name,
            invalid_value=value,
            validation_rule="enum_value",
        )

    return value
