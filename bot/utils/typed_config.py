"""Type-safe configuration access utilities."""

from typing import Protocol, TypeVar, Union, overload

T = TypeVar("T")

# Common configuration value types
ConfigValue = Union[str, int, float, bool, None]


class DictLike(Protocol):
    """Protocol for dict-like objects."""

    def get(self, key: str, default: ConfigValue = None) -> ConfigValue: ...


@overload
def get_typed(
    obj: dict[str, ConfigValue] | DictLike | object, attr: str, default: bool
) -> bool: ...
@overload
def get_typed(
    obj: dict[str, ConfigValue] | DictLike | object, attr: str, default: int
) -> int: ...
@overload
def get_typed(
    obj: dict[str, ConfigValue] | DictLike | object, attr: str, default: float
) -> float: ...
@overload
def get_typed(
    obj: dict[str, ConfigValue] | DictLike | object, attr: str, default: str
) -> str: ...


def get_typed[
    T
](obj: dict[str, ConfigValue] | DictLike | object, attr: str, default: T) -> T:
    """
    Type-safe config getter with automatic conversion.

    Args:
        obj: Object to get attribute from (can be dict, BaseModel, or any object)
        attr: Attribute name
        default: Default value (also determines return type)

    Returns:
        Value of the correct type
    """
    # Handle dict-like objects
    if isinstance(obj, dict):
        value = obj.get(attr, default)
    else:
        value = getattr(obj, attr, default)

    if value is None:
        return default

    target_type = type(default)

    # Already correct type
    if isinstance(value, target_type):
        return value

    try:
        # Handle string to numeric conversions
        if target_type == int:
            # Convert through float to handle decimal strings
            return int(float(str(value)))  # type: ignore[return-value]
        if target_type == float:
            return float(str(value))  # type: ignore[return-value]
        if target_type == bool:
            # Handle string booleans
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")  # type: ignore[return-value]
            return bool(value)  # type: ignore[return-value]
        if target_type == str:
            return str(value)  # type: ignore[return-value]
        # For other types, try direct conversion (skip object and type)
        if callable(target_type) and target_type.__name__ not in ("object", "type"):
            try:
                return target_type(value)  # type: ignore[call-arg]
            except (ValueError, TypeError):
                return default
        return default
    except (ValueError, TypeError):
        return default


def ensure_int(value: ConfigValue, default: int = 0) -> int:
    """Ensure a value is an integer."""
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default


def ensure_float(value: ConfigValue, default: float = 0.0) -> float:
    """Ensure a value is a float."""
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return default


def ensure_str(value: ConfigValue, default: str = "") -> str:
    """Ensure a value is a string."""
    if value is None:
        return default
    return str(value)
