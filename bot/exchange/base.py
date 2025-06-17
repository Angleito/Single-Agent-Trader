"""
Base exchange interface for abstracting exchange operations.

This module provides the abstract base class that all exchange implementations
must inherit from, ensuring consistent interface across different exchanges.
Includes enterprise-grade error handling with error boundaries and recovery mechanisms.
"""

import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Literal

from ..error_handling import (
    ErrorBoundary,
    TradeSaga,
    exception_handler,
    graceful_degradation,
)
from ..system_monitor import error_recovery_manager
from ..types import (
    AccountType,
    FuturesAccountInfo,
    MarginInfo,
    Order,
    Position,
    TradeAction,
)

# Configure logger
logger = logging.getLogger(__name__)


class ExchangeError(Exception):
    """Base exception for exchange errors."""

    pass


class ExchangeConnectionError(ExchangeError):
    """Connection-related errors."""

    pass


class ExchangeAuthError(ExchangeError):
    """Authentication-related errors."""

    pass


class ExchangeOrderError(ExchangeError):
    """Order execution errors."""

    pass


class ExchangeInsufficientFundsError(ExchangeError):
    """Insufficient funds errors."""

    pass


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

        logger.info(f"Initialized {self.exchange_name} exchange with error handling")

    async def _exchange_error_fallback(self, error: Exception, context: dict) -> None:
        """Fallback behavior for exchange errors."""
        logger.warning(f"Exchange error fallback triggered: {error}")

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

    async def _connection_fallback(self, *args, **kwargs) -> bool:
        """Fallback for connection failures."""
        logger.info(f"Using connection fallback for {self.exchange_name}")
        return False  # Indicate connection is unavailable

    async def _trading_fallback(self, *args, **kwargs) -> Order | None:
        """Fallback for trading operations."""
        logger.info(f"Using trading fallback for {self.exchange_name} - returning None")
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
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass

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
        pass

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
        pass

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
        pass

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get current positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    async def get_account_balance(
        self, account_type: AccountType | None = None
    ) -> Decimal:
        """
        Get account balance in USD.

        Args:
            account_type: Specific account type or None for total

        Returns:
            Account balance in USD
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        pass

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
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if client is connected and authenticated.

        Returns:
            True if connected
        """
        pass

    @abstractmethod
    def get_connection_status(self) -> dict[str, Any]:
        """
        Get connection status information.

        Returns:
            Dictionary with connection details
        """
        pass

    # Optional methods for futures trading
    async def get_futures_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get current futures positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of futures Position objects
        """
        return []

    async def get_futures_account_info(
        self, refresh: bool = False
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
        leverage: int | None = None,
        reduce_only: bool = False,
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
        """Get account balance with error boundary protection."""
        async with self._error_boundary:
            try:
                return await self.get_account_balance(account_type)
            except Exception as e:
                # Log error with context
                exception_handler.log_exception_with_context(
                    e,
                    {"account_type": str(account_type) if account_type else "all"},
                    component=f"{self.exchange_name}_balance",
                    operation="get_balance",
                )

                # Return safe default
                return Decimal("0")

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
        pass

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
        pass
