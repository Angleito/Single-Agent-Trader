"""
Base exchange interface for abstracting exchange operations.

This module provides the abstract base class that all exchange implementations
must inherit from, ensuring consistent interface across different exchanges.
Includes enterprise-grade error handling with error boundaries and recovery mechanisms.
"""

import decimal
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, NoReturn

from bot.error_handling import (
    ErrorBoundary,
    TradeSaga,
    exception_handler,
    graceful_degradation,
)
from bot.system_monitor import error_recovery_manager
from bot.trading_types import (
    AccountType,
    FuturesAccountInfo,
    MarginInfo,
    Order,
    Position,
    TradeAction,
)

# Validation functionality is implemented inline in this module

# Configure logger
logger = logging.getLogger(__name__)


class ExchangeError(Exception):
    """Base exception for exchange errors."""


class ExchangeConnectionError(ExchangeError):
    """Connection-related errors."""


class ExchangeAuthError(ExchangeError):
    """Authentication-related errors."""


class ExchangeOrderError(ExchangeError):
    """Order execution errors."""


class ExchangeInsufficientFundsError(ExchangeError):
    """Insufficient funds errors."""


# Balance-specific exception hierarchy
class BalanceRetrievalError(ExchangeError):
    """
    Base exception for balance retrieval operations.

    This is the parent class for all balance-related errors,
    providing common attributes and methods for error handling.
    """

    def __init__(
        self,
        message: str,
        account_type: str | None = None,
        symbol: str | None = None,
        balance_context: dict | None = None,
    ):
        super().__init__(message)
        self.account_type = account_type
        self.symbol = symbol
        self.balance_context = balance_context or {}
        self.timestamp = datetime.now(UTC)

    def get_error_context(self) -> dict:
        """Get structured error context for logging and debugging."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "account_type": self.account_type,
            "symbol": self.symbol,
            "balance_context": self.balance_context,
            "timestamp": self.timestamp.isoformat(),
        }


class InsufficientBalanceError(BalanceRetrievalError):
    """
    Raised when account balance is insufficient for the requested operation.

    This exception includes information about required vs available balance
    for detailed error reporting and recovery strategies.
    """

    def __init__(
        self,
        message: str,
        required_balance: float | None = None,
        available_balance: float | None = None,
        account_type: str | None = None,
        symbol: str | None = None,
        operation_context: dict | None = None,
    ):
        super().__init__(message, account_type, symbol, operation_context)
        self.required_balance = required_balance
        self.available_balance = available_balance
        self.shortfall = None
        if required_balance is not None and available_balance is not None:
            self.shortfall = required_balance - available_balance

    def get_error_context(self) -> dict:
        """Get enhanced error context including balance details."""
        context = super().get_error_context()
        context.update(
            {
                "required_balance": self.required_balance,
                "available_balance": self.available_balance,
                "shortfall": self.shortfall,
            }
        )
        return context


class BalanceValidationError(BalanceRetrievalError):
    """
    Raised when balance data fails validation checks.

    This includes negative balances, invalid decimal values,
    or other data integrity issues.
    """

    def __init__(
        self,
        message: str,
        invalid_value: Any = None,
        validation_rule: str | None = None,
        account_type: str | None = None,
    ):
        super().__init__(message, account_type)
        self.invalid_value = invalid_value
        self.validation_rule = validation_rule

    def get_error_context(self) -> dict:
        """Get validation-specific error context."""
        context = super().get_error_context()
        context.update(
            {
                "invalid_value": (
                    str(self.invalid_value) if self.invalid_value is not None else None
                ),
                "validation_rule": self.validation_rule,
            }
        )
        return context


class BalanceServiceUnavailableError(BalanceRetrievalError):
    """
    Raised when balance service or API is temporarily unavailable.

    This indicates a service-level issue that may be retryable
    after a backoff period.
    """

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.service_name = service_name
        self.status_code = status_code
        self.retry_after = retry_after
        self.is_retryable = True

    def get_error_context(self) -> dict:
        """Get service availability error context."""
        context = super().get_error_context()
        context.update(
            {
                "service_name": self.service_name,
                "status_code": self.status_code,
                "retry_after": self.retry_after,
                "is_retryable": self.is_retryable,
            }
        )
        return context


class BalanceTimeoutError(BalanceRetrievalError):
    """
    Raised when balance API requests timeout.

    This indicates network or API performance issues that may
    require retry with exponential backoff.
    """

    def __init__(
        self,
        message: str,
        timeout_duration: float | None = None,
        endpoint: str | None = None,
        account_type: str | None = None,
    ):
        super().__init__(message, account_type)
        self.timeout_duration = timeout_duration
        self.endpoint = endpoint
        self.is_retryable = True

    def get_error_context(self) -> dict:
        """Get timeout-specific error context."""
        context = super().get_error_context()
        context.update(
            {
                "timeout_duration": self.timeout_duration,
                "endpoint": self.endpoint,
                "is_retryable": self.is_retryable,
            }
        )
        return context


class BaseExchange(ABC):
    """
    Abstract base class for exchange implementations with enterprise-grade error handling.

    All exchange clients must implement this interface to ensure
    compatibility with the trading engine. Includes error boundaries,
    automatic recovery, and comprehensive error tracking.
    """

    def __init__(self, dry_run: bool = True):
        """
        Initialize the exchange client with error handling capabilities.

        Args:
            dry_run: Whether to run in paper trading mode
        """
        self.dry_run = dry_run
        self._connected = False
        self._last_health_check: Any | None = None
        self._last_validated_balance: Decimal | None = None

        # Balance validation - using inline validation with proper exceptions

        # Error handling components
        self.exchange_name = self.__class__.__name__.replace("Client", "").replace(
            "Exchange", ""
        )
        self._error_boundary = ErrorBoundary(
            component_name=f"{self.exchange_name}_exchange",
            fallback_behavior=self._exchange_error_fallback,
            max_retries=3,
            retry_delay=2.0,
        )

        # Register this exchange with graceful degradation
        graceful_degradation.register_service(
            f"{self.exchange_name}_connection",
            self._connection_fallback,
            degradation_threshold=3,
        )

        graceful_degradation.register_service(
            f"{self.exchange_name}_trading",
            self._trading_fallback,
            degradation_threshold=2,
        )

        logger.info(
            "Initialized %s exchange with error handling and balance validation",
            self.exchange_name,
        )

    async def validate_balance_update(
        self,
        new_balance: Decimal,
        operation_type: str = "balance_update",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Validate balance update with comprehensive checks.

        Args:
            new_balance: New balance to validate
            operation_type: Type of operation causing the update
            metadata: Additional context information

        Returns:
            Validation result dictionary

        Raises:
            BalanceValidationError: If validation fails
        """

        def _raise_null_balance_error() -> None:
            raise BalanceValidationError(
                "Balance value cannot be None",
                invalid_value=None,
                validation_rule="null_check",
            )

        def _raise_balance_format_error(balance: Any) -> None:
            raise BalanceValidationError(
                f"Invalid balance format: {balance}",
                invalid_value=balance,
                validation_rule="decimal_conversion",
            )

        def _raise_balance_too_large_error(balance: Decimal) -> None:
            raise BalanceValidationError(
                f"Balance value too large: ${balance}",
                invalid_value=float(balance),
                validation_rule="max_balance_check",
            )

        try:
            # Inline balance validation using our new exception architecture
            if new_balance is None:
                _raise_null_balance_error()

            if not isinstance(new_balance, Decimal):
                try:
                    new_balance = Decimal(str(new_balance))
                except (ValueError, TypeError, decimal.InvalidOperation) as e:
                    raise ValueError(f"Invalid balance format: {new_balance}") from e

            # Check for negative balance in most cases (some DEXs allow negative due to unrealized PnL)
            if new_balance < Decimal(0) and not (
                metadata and metadata.get("allow_negative", False)
            ):
                logger.warning(
                    "Negative balance detected: $%s for operation %s",
                    new_balance,
                    operation_type,
                )

            # Check for unrealistic balance values
            max_reasonable_balance = Decimal(1000000000)  # $1B threshold
            if new_balance > max_reasonable_balance:
                _raise_balance_too_large_error(new_balance)

            # Update last validated balance
            self._last_validated_balance = new_balance
            logger.debug(
                "✅ Balance validation passed: $%s (%s)", new_balance, operation_type
            )

            return {
                "valid": True,
                "balance": new_balance,
                "operation_type": operation_type,
                "validation_timestamp": datetime.now(UTC).isoformat(),
            }

        except BalanceValidationError:
            # Re-raise balance validation errors
            raise
        except Exception as e:
            logger.exception("Balance validation error in %s", self.exchange_name)
            raise BalanceValidationError(
                f"Balance validation failed for {operation_type}: {e}",
                invalid_value=str(new_balance) if new_balance is not None else None,
                validation_rule="comprehensive_validation",
            ) from e

    async def validate_margin_requirements(
        self,
        balance: Decimal,
        used_margin: Decimal,
        _position_value: Decimal | None = None,
        _leverage: int | None = None,
    ) -> dict[str, Any]:
        """
        Validate margin requirements and calculations.

        Args:
            balance: Account balance
            used_margin: Currently used margin
            position_value: Total position value (optional)
            leverage: Leverage ratio (optional)

        Returns:
            Margin validation result

        Raises:
            BalanceValidationError: If margin validation fails
        """

        def _raise_negative_balance_error(balance: Decimal) -> None:
            raise BalanceValidationError(
                "Account balance cannot be negative for margin calculation",
                invalid_value=float(balance),
                validation_rule="negative_balance_check",
            )

        def _raise_negative_margin_error(used_margin: Decimal) -> None:
            raise BalanceValidationError(
                "Used margin cannot be negative",
                invalid_value=float(used_margin),
                validation_rule="negative_used_margin_check",
            )

        try:
            # Inline margin validation
            if balance < Decimal(0):
                _raise_negative_balance_error(balance)

            if used_margin < Decimal(0):
                _raise_negative_margin_error(used_margin)

            if used_margin > balance:
                logger.warning(
                    "Used margin ($%s) exceeds balance ($%s)", used_margin, balance
                )

            available_margin = balance - used_margin
            margin_ratio = (
                (used_margin / balance * 100) if balance > 0 else Decimal(100)
            )

            return {
                "valid": True,
                "balance": balance,
                "used_margin": used_margin,
                "available_margin": available_margin,
                "margin_ratio_pct": float(margin_ratio),
                "validation_timestamp": datetime.now(UTC).isoformat(),
            }

        except BalanceValidationError:
            # Re-raise balance validation errors
            raise
        except Exception as e:
            logger.exception("Margin validation error in %s", self.exchange_name)
            raise BalanceValidationError(
                f"Margin validation failed: {e}",
                invalid_value=f"balance:{balance}, used_margin:{used_margin}",
                validation_rule="margin_calculation",
            ) from e

    async def validate_balance_reconciliation(
        self,
        calculated_balance: Decimal,
        exchange_reported_balance: Decimal,
        tolerance_pct: float = 0.1,
    ) -> dict[str, Any]:
        """
        Validate balance reconciliation between internal calculations and exchange.

        Args:
            calculated_balance: Balance calculated from our records
            exchange_reported_balance: Balance reported by exchange
            tolerance_pct: Acceptable difference percentage

        Returns:
            Reconciliation validation result

        Raises:
            BalanceValidationError: If reconciliation fails
        """

        def _raise_null_reconciliation_error() -> NoReturn:
            raise BalanceValidationError(
                "Balance values cannot be None for reconciliation",
                invalid_value="null_balance",
                validation_rule="null_balance_check",
            )

        try:
            # Inline balance reconciliation validation
            if calculated_balance is None or exchange_reported_balance is None:
                _raise_null_reconciliation_error()

            # Calculate difference
            difference = abs(calculated_balance - exchange_reported_balance)
            relative_difference = 0.0

            if exchange_reported_balance != Decimal(0):
                relative_difference = float(
                    difference / abs(exchange_reported_balance) * 100
                )
            elif calculated_balance != Decimal(0):
                # If reported balance is 0 but calculated isn't, this is a significant discrepancy
                relative_difference = 100.0

            is_within_tolerance = relative_difference <= tolerance_pct

            validation_result = {
                "valid": is_within_tolerance,
                "calculated_balance": calculated_balance,
                "exchange_reported_balance": exchange_reported_balance,
                "difference": difference,
                "relative_difference_pct": relative_difference,
                "tolerance_pct": tolerance_pct,
                "validation_timestamp": datetime.now(UTC).isoformat(),
            }

            if not is_within_tolerance:
                logger.warning(
                    "Balance reconciliation outside tolerance: "
                    "calculated=$%.2f, reported=$%.2f, "
                    "difference=%.2f%% (tolerance: %.2f%%)",
                    calculated_balance,
                    exchange_reported_balance,
                    relative_difference,
                    tolerance_pct,
                )

            return validation_result

        except BalanceValidationError:
            # Re-raise balance validation errors
            raise
        except Exception as e:
            logger.exception("Balance reconciliation error in %s", self.exchange_name)
            raise BalanceValidationError(
                f"Balance reconciliation failed: {e}",
                invalid_value=f"calculated:{calculated_balance}, reported:{exchange_reported_balance}",
                validation_rule="reconciliation",
            ) from e

    def get_balance_validation_status(self) -> dict[str, Any]:
        """
        Get current balance validation status and statistics.

        Returns:
            Dictionary with validation status and statistics
        """
        return {
            "exchange_name": self.exchange_name,
            "last_validated_balance": (
                float(self._last_validated_balance)
                if self._last_validated_balance
                else None
            ),
            "validation_enabled": True,
            "validation_method": "inline_validation_with_balance_exceptions",
        }

    async def _exchange_error_fallback(self, error: Exception, _context: dict) -> None:
        """Fallback behavior for exchange errors."""
        logger.warning("Exchange error fallback triggered: %s", error)

        # Attempt recovery based on error type
        error_type = type(error).__name__
        if "Connection" in error_type:
            await error_recovery_manager.recover_from_error(
                "network_error",
                {"error": str(error), "component": "exchange"},
                self.exchange_name,
            )
        elif "Auth" in error_type:
            await error_recovery_manager.recover_from_error(
                "auth_error",
                {"error": str(error), "component": "exchange"},
                self.exchange_name,
            )

    async def _connection_fallback(self, *_args, **_kwargs) -> bool:
        """Fallback for connection failures."""
        logger.info("Using connection fallback for %s", self.exchange_name)
        return False  # Indicate connection is unavailable

    async def _trading_fallback(self, *_args, **_kwargs) -> Order | None:
        """Fallback for trading operations."""
        logger.info(
            "Using trading fallback for %s - returning None", self.exchange_name
        )
        return None  # No order placed in fallback mode

    async def connect_with_error_handling(self) -> bool:
        """
        Connect with error boundary protection.

        Returns:
            True if connection successful
        """
        return await graceful_degradation.execute_with_fallback(
            f"{self.exchange_name}_connection", self.connect
        )

    async def execute_trade_action_with_saga(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Execute trade action using transaction saga pattern for consistency.

        Args:
            trade_action: Trade action to execute
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Order object if successful, None otherwise
        """
        # Create saga for trade execution
        saga = TradeSaga(f"trade_{trade_action.action}_{symbol}")

        try:
            # Add main trade step
            saga.add_step(
                lambda: self.execute_trade_action(trade_action, symbol, current_price),
                step_name="execute_trade",
            )

            # Execute saga with automatic compensation
            success = await saga.execute()

            if success and saga.completed_steps:
                # Return the order from the executed step
                _, _, order_result, _ = saga.completed_steps[0]
                return order_result
            # Saga succeeded but no steps completed, or saga failed
            return None

        except Exception as e:
            # Log saga failure with enhanced context
            exception_handler.log_exception_with_context(
                e,
                {
                    "trade_action": trade_action.__dict__,
                    "symbol": symbol,
                    "current_price": float(current_price),
                    "saga_status": saga.get_status(),
                },
                component=f"{self.exchange_name}_trading",
                operation="execute_trade_saga",
            )

            # Attempt recovery
            await error_recovery_manager.recover_from_error(
                "position_error",
                {"symbol": symbol, "action": trade_action.action},
                self.exchange_name,
            )

            return None

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect and authenticate with the exchange.

        Returns:
            True if connection successful
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""

    @abstractmethod
    async def execute_trade_action(
        self, trade_action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Execute a trade action on the exchange.

        Args:
            trade_action: Trade action to execute
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Order object if successful, None otherwise
        """

    @abstractmethod
    async def place_market_order(
        self, symbol: str, side: Literal["BUY", "SELL"], quantity: Decimal
    ) -> Order | None:
        """
        Place a market order.

        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity

        Returns:
            Order object if successful
        """

    @abstractmethod
    async def place_limit_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        price: Decimal,
    ) -> Order | None:
        """
        Place a limit order.

        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            price: Limit price

        Returns:
            Order object if successful
        """

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get current positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of Position objects
        """

    @abstractmethod
    async def get_account_balance(
        self, account_type: AccountType | None = None
    ) -> Decimal:
        """
        Get account balance in USD.

        Args:
            account_type: Specific account type or None for total

        Returns:
            Account balance in USD (normalized to 2 decimal places)

        Note:
            Implementations should normalize USD balances to 2 decimal places
            and crypto amounts to 8 decimal places for consistency.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """

    @abstractmethod
    async def cancel_all_orders(
        self, symbol: str | None = None, status: str | None = None
    ) -> bool:
        """
        Cancel all open orders.

        Args:
            symbol: Optional trading symbol filter
            status: Optional order status filter (for SDK compatibility)

        Returns:
            True if successful
        """

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if client is connected and authenticated.

        Returns:
            True if connected
        """

    @abstractmethod
    def get_connection_status(self) -> dict[str, Any]:
        """
        Get connection status information.

        Returns:
            Dictionary with connection details
        """

    # Optional methods for futures trading
    async def get_futures_positions(self, _symbol: str | None = None) -> list[Position]:
        """
        Get current futures positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of futures Position objects
        """
        return []

    async def get_futures_account_info(
        self, _refresh: bool = False
    ) -> FuturesAccountInfo | None:
        """
        Get comprehensive futures account information.

        Args:
            refresh: Force refresh of cached data

        Returns:
            FuturesAccountInfo object or None if not available
        """
        return None

    async def get_margin_info(self) -> MarginInfo | None:
        """
        Get futures margin information and health status.

        Returns:
            MarginInfo object with current margin status
        """
        return None

    async def place_futures_market_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        _leverage: int | None = None,
        _reduce_only: bool = False,
    ) -> Order | None:
        """
        Place a futures market order with leverage.

        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            leverage: Leverage multiplier
            reduce_only: True if this order should only reduce position

        Returns:
            Order object if successful
        """
        # Default to regular market order if not implemented
        return await self.place_market_order(symbol, side, quantity)

    # Error-wrapped convenience methods
    async def place_market_order_with_error_handling(
        self, symbol: str, side: Literal["BUY", "SELL"], quantity: Decimal
    ) -> Order | None:
        """Place market order with error boundary protection."""
        async with self._error_boundary:
            return await graceful_degradation.execute_with_fallback(
                f"{self.exchange_name}_trading",
                self.place_market_order,
                symbol,
                side,
                quantity,
            )

    async def place_limit_order_with_error_handling(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        price: Decimal,
    ) -> Order | None:
        """Place limit order with error boundary protection."""
        async with self._error_boundary:
            return await graceful_degradation.execute_with_fallback(
                f"{self.exchange_name}_trading",
                self.place_limit_order,
                symbol,
                side,
                quantity,
                price,
            )

    async def get_account_balance_with_error_handling(
        self, account_type: AccountType | None = None
    ) -> Decimal:
        """Get account balance with error boundary protection and validation."""
        async with self._error_boundary:
            try:
                # Get balance from exchange
                balance = await self.get_account_balance(account_type)

                # Validate balance
                try:
                    validation_result = await self.validate_balance_update(
                        new_balance=balance,
                        operation_type="balance_fetch",
                        metadata={
                            "account_type": str(account_type) if account_type else "all"
                        },
                    )

                    if not validation_result["valid"]:
                        logger.warning(
                            "⚠️ Balance validation failed for %s: %s",
                            self.exchange_name,
                            validation_result.get("error", {}).get(
                                "message", "Unknown error"
                            ),
                        )
                        # Return the balance anyway but log the issue
                        # In production, you might want to trigger alerts here

                    return validation_result.get("balance", balance)

                except BalanceValidationError:
                    logger.exception("Balance validation error")
                    # Return original balance if validation fails but log the issue
                    return balance

            except Exception as e:
                # Log error with context
                exception_handler.log_exception_with_context(
                    e,
                    {"account_type": str(account_type) if account_type else "all"},
                    component=f"{self.exchange_name}_balance",
                    operation="get_balance",
                )

                # Return safe default
                return Decimal(0)

    async def get_positions_with_error_handling(
        self, symbol: str | None = None
    ) -> list[Position]:
        """Get positions with error boundary protection."""
        async with self._error_boundary:
            try:
                return await self.get_positions(symbol)
            except Exception as e:
                # Log error with context
                exception_handler.log_exception_with_context(
                    e,
                    {"symbol": symbol},
                    component=f"{self.exchange_name}_positions",
                    operation="get_positions",
                )

                # Return empty list as safe default
                return []

    def get_error_boundary_status(self) -> dict[str, Any]:
        """Get error boundary status and health information."""
        return {
            "exchange_name": self.exchange_name,
            "error_boundary_degraded": self._error_boundary.is_degraded(),
            "error_count": self._error_boundary.error_count,
            "last_error": (
                str(self._error_boundary.last_error)
                if self._error_boundary.last_error
                else None
            ),
            "service_health": {
                service_name: (
                    graceful_degradation.get_service_status(service_name).__dict__
                    if graceful_degradation.get_service_status(service_name)
                    else None
                )
                for service_name in [
                    f"{self.exchange_name}_connection",
                    f"{self.exchange_name}_trading",
                ]
            },
        }

    @property
    def exchange_name_property(self) -> str:
        """Get the exchange name."""
        return self.__class__.__name__.replace("Client", "").replace("Exchange", "")

    @property
    def supports_futures(self) -> bool:
        """Check if exchange supports futures trading."""
        return False

    @property
    def is_decentralized(self) -> bool:
        """Check if this is a decentralized exchange."""
        return False

    @property
    @abstractmethod
    def enable_futures(self) -> bool:
        """
        Check if futures trading is enabled for this exchange instance.

        This should be implemented as a property that returns True if the exchange
        instance is configured for futures trading, False for spot trading.
        """

    @abstractmethod
    async def get_trading_symbol(self, symbol: str) -> str:
        """
        Get the actual trading symbol for the given base symbol.

        This method handles the conversion from base symbols (like "BTC-USD")
        to the actual trading symbols used by the exchange (e.g., spot symbols,
        futures contract symbols, perpetual symbols, etc.).

        Args:
            symbol: Base trading symbol (e.g., "BTC-USD", "ETH-USD")

        Returns:
            The actual trading symbol used by the exchange
        """
