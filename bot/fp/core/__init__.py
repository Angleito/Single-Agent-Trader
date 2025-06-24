"""
Functional Programming Core Components

This module provides core functional programming utilities including
validation, error handling, and functional composition primitives.
"""

# Core monads
from .either import Either, Left, Right, left, right, sequence_either, try_either

# Core validation components
from .functional_validation import (
    ConversionError,
    FieldError,
    SchemaError,
    ValidationChainError,
    ValidationPipeline,
    ValidatorError,
    compose_validators,
    validate_position_functional,
    validate_positive,
    validate_range,
    validate_trade_action_functional,
)
from .io import IO, IOBuilder, delay, pure, sequence_io, traverse_io
from .option import Empty, Option, Some, empty, option_from_nullable, some, try_option

# Core utilities
from .validation import (
    validate_decimal,
    validate_non_negative,
    validate_positive,
    validate_timestamp,
)

__all__ = [
    # Core monads
    "Either",
    "Left",
    "Right",
    "left",
    "right",
    "try_either",
    "sequence_either",
    "IO",
    "pure",
    "delay",
    "sequence_io",
    "traverse_io",
    "IOBuilder",
    "Option",
    "Some",
    "Empty",
    "some",
    "empty",
    "option_from_nullable",
    "try_option",
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
