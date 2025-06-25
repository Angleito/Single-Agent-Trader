"""
Functional Programming Core Components

This module provides core functional programming utilities including
validation, error handling, and functional composition primitives.
"""

# Core monads
# Strategy and Signal types (moved to shared constants to break circular imports)
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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


# Define shared constants to avoid circular imports
class SignalType:
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


# Use TYPE_CHECKING for forward references
if TYPE_CHECKING:
    from bot.fp.strategies.base import Strategy as StrategyImpl
    from bot.fp.strategies.signals import Signal as SignalImpl
    from bot.fp.types.trading import MarketState as MarketStateImpl

    Strategy = StrategyImpl
    Signal = SignalImpl
    MarketState = MarketStateImpl
    STRATEGY_TYPES_AVAILABLE = True
else:
    # Runtime fallback types
    class Signal:
        pass

    class MarketState:
        pass

    Strategy = Callable[[Any], Any]
    STRATEGY_TYPES_AVAILABLE = False

__all__ = [
    "IO",
    "STRATEGY_TYPES_AVAILABLE",
    "ConversionError",
    # Core monads
    "Either",
    "Empty",
    # Functional validation
    "FieldError",
    "IOBuilder",
    "Left",
    "MarketState",
    "Option",
    "Right",
    "SchemaError",
    "Signal",
    "SignalType",
    "Some",
    # Strategy and Signal types
    "Strategy",
    "ValidationChainError",
    "ValidationPipeline",
    "ValidatorError",
    "compose_validators",
    "delay",
    "empty",
    "left",
    "option_from_nullable",
    "pure",
    "right",
    "sequence_either",
    "sequence_io",
    "some",
    "traverse_io",
    "try_either",
    "try_option",
    # Core validation utilities
    "validate_decimal",
    "validate_non_negative",
    "validate_position_functional",
    "validate_positive",
    "validate_range",
    "validate_timestamp",
    "validate_trade_action_functional",
]
