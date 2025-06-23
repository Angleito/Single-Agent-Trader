"""
Type system foundation for the trading bot.

This module provides centralized type definitions, exception hierarchy,
and type guards for improved type safety across the codebase.
"""

from .base_types import (
    AccountId,
    IndicatorDict,
    MarketDataDict,
    OrderId,
    Percentage,
    Price,
    Quantity,
    Symbol,
    Timestamp,
    ValidationResult,
)
from .exceptions import (
    BalanceValidationError,
    ExchangeAuthError,
    ExchangeConnectionError,
    ExchangeError,
    IndicatorError,
    LLMError,
    OrderExecutionError,
    PositionValidationError,
    StrategyError,
    TradeValidationError,
    TradingBotError,
    ValidationError,
)
from .guards import (
    ensure_decimal,
    ensure_positive_decimal,
    is_valid_percentage,
    is_valid_price,
    is_valid_quantity,
    is_valid_symbol,
)

__all__ = [
    # Base types
    "Price",
    "Quantity",
    "Percentage",
    "Timestamp",
    "Symbol",
    "OrderId",
    "AccountId",
    "MarketDataDict",
    "IndicatorDict",
    "ValidationResult",
    # Exceptions
    "TradingBotError",
    "ValidationError",
    "TradeValidationError",
    "BalanceValidationError",
    "PositionValidationError",
    "ExchangeError",
    "ExchangeConnectionError",
    "ExchangeAuthError",
    "OrderExecutionError",
    "StrategyError",
    "LLMError",
    "IndicatorError",
    # Guards
    "is_valid_price",
    "is_valid_quantity",
    "is_valid_percentage",
    "is_valid_symbol",
    "ensure_decimal",
    "ensure_positive_decimal",
]
