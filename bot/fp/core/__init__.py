"""
Functional Programming Core Components

This module provides core functional programming utilities including
validation, error handling, and functional composition primitives.
"""

# Core monads
from .either import Either, Left, Right, left, right, try_either, sequence_either
from .io import IO, pure, delay, sequence_io, traverse_io, IOBuilder
from .option import Option, Some, Empty, some, empty, option_from_nullable, try_option

# Core validation components
from .functional_validation import (
    FieldError,
    SchemaError,
    ConversionError,
    ValidationChainError,
    ValidatorError,
    ValidationPipeline,
    validate_trade_action_functional,
    validate_position_functional,
    validate_positive,
    validate_range,
    compose_validators,
)

# Core utilities
from .validation import (
    validate_decimal,
    validate_timestamp,
    validate_non_negative,
    validate_positive,
)

__all__ = [
    # Core monads
    "Either", "Left", "Right", "left", "right", "try_either", "sequence_either",
    "IO", "pure", "delay", "sequence_io", "traverse_io", "IOBuilder",
    "Option", "Some", "Empty", "some", "empty", "option_from_nullable", "try_option",
    # Functional validation
    "FieldError",
    "SchemaError",
    "ConversionError", 
    "ValidationChainError",
    "ValidatorError",
    "ValidationPipeline",
    "validate_trade_action_functional",
    "validate_position_functional",
    "compose_validators",
    # Core validation utilities
    "validate_decimal",
    "validate_timestamp", 
    "validate_non_negative",
    "validate_positive",
    "validate_range",
]