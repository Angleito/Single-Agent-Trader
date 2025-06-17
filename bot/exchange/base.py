"""
Base exchange interface for abstracting exchange operations.

This module provides the abstract base class that all exchange implementations
must inherit from, ensuring consistent interface across different exchanges.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..types import (
    AccountType,
    FuturesAccountInfo,
    MarginInfo,
    Order,
    Position,
    TradeAction,
)


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
    Abstract base class for exchange implementations.
    
    All exchange clients must implement this interface to ensure
    compatibility with the trading engine.
    """
    
    def __init__(self, dry_run: bool = True):
        """
        Initialize the exchange client.
        
        Args:
            dry_run: Whether to run in paper trading mode
        """
        self.dry_run = dry_run
        self._connected = False
        self._last_health_check = None
        
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
    ) -> Optional[Order]:
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
        self, symbol: str, side: str, quantity: Decimal
    ) -> Optional[Order]:
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
        self, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> Optional[Order]:
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
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
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
        self, account_type: Optional[AccountType] = None
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
    async def cancel_all_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> bool:
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
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status information.
        
        Returns:
            Dictionary with connection details
        """
        pass
        
    # Optional methods for futures trading
    async def get_futures_positions(self, symbol: Optional[str] = None) -> List[Position]:
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
    ) -> Optional[FuturesAccountInfo]:
        """
        Get comprehensive futures account information.
        
        Args:
            refresh: Force refresh of cached data
            
        Returns:
            FuturesAccountInfo object or None if not available
        """
        return None
        
    async def get_margin_info(self) -> Optional[MarginInfo]:
        """
        Get futures margin information and health status.
        
        Returns:
            MarginInfo object with current margin status
        """
        return None
        
    async def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        leverage: Optional[int] = None,
        reduce_only: bool = False,
    ) -> Optional[Order]:
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
        
    @property
    def exchange_name(self) -> str:
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