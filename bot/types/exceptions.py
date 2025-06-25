"""
Centralized exception hierarchy for the trading bot.

This module consolidates all exception classes into a clear hierarchy,
replacing the 97 scattered error classes found across the codebase.
"""

from datetime import UTC, datetime
from decimal import Decimal

# Type alias for error context data
type ErrorContextData = str | int | float | bool | Decimal | datetime | None
type ErrorContextDict = dict[str, ErrorContextData]


class TradingBotError(Exception):
    """
    Base exception for all trading bot errors.

    Provides common attributes and methods for error handling,
    context tracking, and recovery strategies.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: ErrorContextDict | None = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now(UTC)

    def get_error_context(self) -> ErrorContextDict:
        """Get structured error context for logging and debugging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": str(self.context),  # Convert to string for serialization
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
        }


# Validation Errors
class ValidationError(TradingBotError):
    """Base class for all validation errors."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        invalid_value: ErrorContextData = None,
        validation_rule: str | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.validation_rule = validation_rule

    def get_error_context(self) -> ErrorContextDict:
        context = super().get_error_context()
        context.update(
            {
                "field_name": self.field_name,
                "invalid_value": (
                    str(self.invalid_value) if self.invalid_value is not None else None
                ),
                "validation_rule": self.validation_rule,
            }
        )
        return context


class TradeValidationError(ValidationError):
    """Error in trade action validation."""

    def __init__(
        self,
        message: str,
        trade_action: str | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.trade_action = trade_action


class BalanceValidationError(ValidationError):
    """Error in balance validation."""

    def __init__(
        self,
        message: str,
        required_balance: Decimal | None = None,
        available_balance: Decimal | None = None,
        account_type: str | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.required_balance = required_balance
        self.available_balance = available_balance
        self.account_type = account_type
        self.shortfall = None
        if required_balance is not None and available_balance is not None:
            self.shortfall = required_balance - available_balance


class PositionValidationError(ValidationError):
    """Error in position validation."""

    def __init__(
        self,
        message: str,
        position_side: str | None = None,
        position_size: Decimal | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.position_side = position_side
        self.position_size = position_size


# Exchange Errors
class ExchangeError(TradingBotError):
    """Base class for all exchange-related errors."""

    def __init__(
        self,
        message: str,
        exchange_name: str | None = None,
        symbol: str | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.exchange_name = exchange_name
        self.symbol = symbol


class ExchangeConnectionError(ExchangeError):
    """Connection-related exchange errors."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ExchangeAuthError(ExchangeError):
    """Authentication-related exchange errors."""

    def __init__(self, message: str, **kwargs: ErrorContextData):
        super().__init__(message, recoverable=False, **kwargs)


class OrderExecutionError(ExchangeError):
    """Order execution errors."""

    def __init__(
        self,
        message: str,
        order_id: str | None = None,
        order_type: str | None = None,
        order_side: str | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.order_id = order_id
        self.order_type = order_type
        self.order_side = order_side


class InsufficientFundsError(ExchangeError):
    """Insufficient funds for trading operation."""

    def __init__(
        self,
        message: str,
        required_amount: Decimal | None = None,
        available_amount: Decimal | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.required_amount = required_amount
        self.available_amount = available_amount


# Strategy Errors
class StrategyError(TradingBotError):
    """Base class for strategy-related errors."""


class LLMError(StrategyError):
    """LLM-related errors."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        prompt_size: int | None = None,
        response_time: float | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.prompt_size = prompt_size
        self.response_time = response_time


class IndicatorError(StrategyError):
    """Technical indicator calculation errors."""

    def __init__(
        self,
        message: str,
        indicator_name: str | None = None,
        data_points: int | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.indicator_name = indicator_name
        self.data_points = data_points


# Risk Management Errors
class RiskManagementError(TradingBotError):
    """Base class for risk management errors."""


class PositionSizingError(RiskManagementError):
    """Position sizing calculation errors."""


class MarginError(RiskManagementError):
    """Margin-related errors."""

    def __init__(
        self,
        message: str,
        margin_required: Decimal | None = None,
        margin_available: Decimal | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, **kwargs)
        self.margin_required = margin_required
        self.margin_available = margin_available


class DailyLossLimitError(RiskManagementError):
    """Daily loss limit exceeded."""

    def __init__(
        self,
        message: str,
        daily_loss: Decimal | None = None,
        loss_limit: Decimal | None = None,
        **kwargs: ErrorContextData,
    ):
        super().__init__(message, recoverable=False, **kwargs)
        self.daily_loss = daily_loss
        self.loss_limit = loss_limit
