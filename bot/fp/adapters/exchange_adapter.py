"""
Exchange Interface Adapters for Functional Trading Bot

This module provides adapters for different exchange interfaces,
bridging between the functional effects system and exchange APIs.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from bot.fp.effects.io import IOEither
    from bot.fp.types.trading import AccountBalance, Order, OrderResult, Position


class ExchangeAdapter(Protocol):
    """Protocol for exchange adapters"""

    @abstractmethod
    def place_order_impl(self, order: Order) -> IOEither[Exception, OrderResult]:
        """Implementation-specific order placement"""

    @abstractmethod
    def cancel_order_impl(self, order_id: str) -> IOEither[Exception, bool]:
        """Implementation-specific order cancellation"""

    @abstractmethod
    def get_positions_impl(self) -> IOEither[Exception, list[Position]]:
        """Implementation-specific position retrieval"""

    @abstractmethod
    def get_balance_impl(self) -> IOEither[Exception, AccountBalance]:
        """Implementation-specific balance retrieval"""


@dataclass
class UnifiedExchangeAdapter:
    """Unified adapter that delegates to specific exchange implementations"""

    adapters: dict[str, ExchangeAdapter]
    default_exchange: str

    def get_adapter(self, exchange: str | None = None) -> ExchangeAdapter:
        """Get adapter for specific exchange"""
        exchange_name = exchange or self.default_exchange

        if exchange_name not in self.adapters:
            raise ValueError(f"No adapter found for exchange: {exchange_name}")

        return self.adapters[exchange_name]

    def place_order(
        self, order: Order, exchange: str | None = None
    ) -> IOEither[Exception, OrderResult]:
        """Place order on specified exchange"""
        adapter = self.get_adapter(exchange)
        return adapter.place_order_impl(order)

    def cancel_order(
        self, order_id: str, exchange: str | None = None
    ) -> IOEither[Exception, bool]:
        """Cancel order on specified exchange"""
        adapter = self.get_adapter(exchange)
        return adapter.cancel_order_impl(order_id)

    def get_positions(
        self, exchange: str | None = None
    ) -> IOEither[Exception, list[Position]]:
        """Get positions from specified exchange"""
        adapter = self.get_adapter(exchange)
        return adapter.get_positions_impl()

    def get_balance(
        self, exchange: str | None = None
    ) -> IOEither[Exception, AccountBalance]:
        """Get balance from specified exchange"""
        adapter = self.get_adapter(exchange)
        return adapter.get_balance_impl()


# Global unified adapter
_unified_adapter: UnifiedExchangeAdapter | None = None


def get_exchange_adapter() -> UnifiedExchangeAdapter:
    """Get the global exchange adapter"""
    global _unified_adapter
    if _unified_adapter is None:
        # Will be initialized with specific adapters
        _unified_adapter = UnifiedExchangeAdapter(
            adapters={}, default_exchange="coinbase"
        )
    return _unified_adapter


def register_exchange_adapter(name: str, adapter: ExchangeAdapter) -> None:
    """Register an exchange adapter"""
    unified = get_exchange_adapter()
    unified.adapters[name] = adapter
