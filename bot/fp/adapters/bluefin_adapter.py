"""
Bluefin Exchange Functional Adapter

This module provides a functional adapter for the Bluefin DEX exchange,
implementing the ExchangeAdapter protocol with IOEither effects.
"""

import logging
from typing import Any

from bot.exchange.bluefin import BluefinClient
from bot.fp.effects.io import IOEither, Left, Right
from bot.fp.types.trading import Order, OrderResult, Position

from .type_converters import (
    create_fp_account_balance,
    create_order_result,
    current_position_to_fp_position,
    fp_order_to_current_order,
)

logger = logging.getLogger(__name__)


class BluefinExchangeAdapter:
    """Functional adapter for Bluefin DEX exchange operations."""

    def __init__(self, bluefin_client: BluefinClient):
        """Initialize adapter with Bluefin client instance."""
        self.client = bluefin_client
        self._logger = logger

    def place_order_impl(self, order: Order) -> IOEither[Exception, OrderResult]:
        """Place order using Bluefin client with functional effects."""

        def _place_order() -> OrderResult | Exception:
            try:
                # Convert FP order to current order type
                current_order = fp_order_to_current_order(order)

                # Execute using existing Bluefin client methods
                if current_order.type == "MARKET":
                    result = self.client.place_market_order(
                        symbol=current_order.symbol,
                        side=current_order.side,
                        quantity=current_order.quantity,
                    )
                elif current_order.type == "LIMIT":
                    if current_order.price is None:
                        raise ValueError("Limit order missing price")
                    result = self.client.place_limit_order(
                        symbol=current_order.symbol,
                        side=current_order.side,
                        quantity=current_order.quantity,
                        price=current_order.price,
                    )
                else:
                    raise ValueError(
                        f"Unsupported order type for Bluefin: {current_order.type}"
                    )

                if result is None:
                    raise Exception("Order placement failed - no result returned")

                # Convert result to functional OrderResult
                return create_order_result(result, success=True)

            except Exception as e:
                self._logger.exception(f"Bluefin order placement failed: {e}")
                return e

        def safe_place_order():
            try:
                result = _place_order()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_place_order)

    def cancel_order_impl(self, order_id: str) -> IOEither[Exception, bool]:
        """Cancel order using Bluefin client with functional effects."""

        def _cancel_order() -> bool | Exception:
            try:
                result = self.client.cancel_order(order_id)
                return result if result is not None else False
            except Exception as e:
                self._logger.exception(f"Bluefin order cancellation failed: {e}")
                return e

        def safe_cancel_order():
            try:
                result = _cancel_order()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_cancel_order)

    def get_positions_impl(self) -> IOEither[Exception, list[Position]]:
        """Get positions using Bluefin client with functional effects."""

        def _get_positions() -> list[Position] | Exception:
            try:
                # Get positions from Bluefin client
                current_positions = self.client.get_positions()

                # Convert to functional positions
                fp_positions = []
                for pos in current_positions:
                    if pos.side != "FLAT":  # Only include non-flat positions
                        try:
                            fp_pos = current_position_to_fp_position(pos)
                            fp_positions.append(fp_pos)
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to convert Bluefin position {pos.symbol}: {e}"
                            )
                            continue

                return fp_positions

            except Exception as e:
                self._logger.exception(f"Failed to get Bluefin positions: {e}")
                return e

        def safe_get_positions():
            try:
                result = _get_positions()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_get_positions)

    def get_balance_impl(self) -> IOEither[Exception, dict[str, Any]]:
        """Get account balance using Bluefin client with functional effects."""

        def _get_balance() -> dict[str, Any] | Exception:
            try:
                # Get balance from Bluefin client
                balance = self.client.get_account_balance()

                # Convert to functional AccountBalance format
                return create_fp_account_balance(balance)

            except Exception as e:
                self._logger.exception(f"Failed to get Bluefin balance: {e}")
                return e

        def safe_get_balance():
            try:
                result = _get_balance()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_get_balance)

    async def connect_impl(self) -> IOEither[Exception, bool]:
        """Connect to Bluefin with functional effects."""

        def _connect() -> bool | Exception:
            try:
                result = self.client.connect()
                return result if result is not None else False
            except Exception as e:
                self._logger.exception(f"Bluefin connection failed: {e}")
                return e

        def safe_connect():
            try:
                result = _connect()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_connect)

    def disconnect_impl(self) -> IOEither[Exception, bool]:
        """Disconnect from Bluefin with functional effects."""

        def _disconnect() -> bool | Exception:
            try:
                self.client.disconnect()
                return True
            except Exception as e:
                self._logger.exception(f"Bluefin disconnection failed: {e}")
                return e

        def safe_disconnect():
            try:
                result = _disconnect()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_disconnect)

    def cancel_all_orders_impl(
        self, symbol: str | None = None
    ) -> IOEither[Exception, bool]:
        """Cancel all orders using Bluefin client with functional effects."""

        def _cancel_all_orders() -> bool | Exception:
            try:
                result = self.client.cancel_all_orders(symbol=symbol)
                return result if result is not None else False
            except Exception as e:
                self._logger.exception(f"Bluefin cancel all orders failed: {e}")
                return e

        def safe_cancel_all_orders():
            try:
                result = _cancel_all_orders()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_cancel_all_orders)

    def get_futures_positions_impl(self) -> IOEither[Exception, list[Position]]:
        """Get futures positions (same as regular positions for Bluefin)."""
        # Bluefin is primarily a perpetuals DEX, so futures = regular positions
        return self.get_positions_impl()

    def place_futures_order_impl(
        self, order: Order, leverage: int | None = None
    ) -> IOEither[Exception, OrderResult]:
        """Place futures order (same as regular order for Bluefin)."""
        # Bluefin is primarily a perpetuals DEX, so this is the same as place_order
        return self.place_order_impl(order)

    def get_trading_symbol_impl(self, symbol: str) -> IOEither[Exception, str]:
        """Get trading symbol using Bluefin client with functional effects."""

        def _get_trading_symbol() -> str | Exception:
            try:
                return self.client.get_trading_symbol(symbol)
            except Exception as e:
                self._logger.exception(f"Failed to get Bluefin trading symbol: {e}")
                return e

        def safe_get_trading_symbol():
            try:
                result = _get_trading_symbol()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_get_trading_symbol)


def create_bluefin_adapter(
    private_key: str | None = None,
    network: str = "mainnet",
    dry_run: bool = True,
) -> BluefinExchangeAdapter:
    """Factory function to create a Bluefin exchange adapter."""
    # Create the underlying Bluefin client
    bluefin_client = BluefinClient(
        private_key=private_key,
        network=network,
        dry_run=dry_run,
    )

    # Return the functional adapter
    return BluefinExchangeAdapter(bluefin_client)
