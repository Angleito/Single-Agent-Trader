"""
Coinbase Exchange Functional Adapter

This module provides a functional adapter for the Coinbase exchange,
implementing the ExchangeAdapter protocol with IOEither effects.
"""

import logging
from typing import Any

from bot.exchange.coinbase import CoinbaseClient

from ..effects.io import IOEither, Left, Right
from ..types.trading import Order, OrderResult, Position
from .type_converters import (
    create_fp_account_balance,
    create_order_result,
    current_position_to_fp_position,
    fp_order_to_current_order,
)

logger = logging.getLogger(__name__)


class CoinbaseExchangeAdapter:
    """Functional adapter for Coinbase exchange operations."""

    def __init__(self, coinbase_client: CoinbaseClient):
        """Initialize adapter with Coinbase client instance."""
        self.client = coinbase_client
        self._logger = logger

    def place_order_impl(self, order: Order) -> IOEither[Exception, OrderResult]:
        """Place order using Coinbase client with functional effects."""

        def _place_order() -> OrderResult | Exception:
            try:
                # Convert FP order to current order type
                current_order = fp_order_to_current_order(order)

                # Execute using existing Coinbase client methods
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
                    raise ValueError(f"Unsupported order type: {current_order.type}")

                if result is None:
                    raise Exception("Order placement failed - no result returned")

                # Convert result to functional OrderResult
                order_result = create_order_result(result, success=True)
                return order_result

            except Exception as e:
                self._logger.error(f"Order placement failed: {e}")
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
        """Cancel order using Coinbase client with functional effects."""

        def _cancel_order() -> bool | Exception:
            try:
                result = self.client.cancel_order(order_id)
                return result if result is not None else False
            except Exception as e:
                self._logger.error(f"Order cancellation failed: {e}")
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
        """Get positions using Coinbase client with functional effects."""

        def _get_positions() -> list[Position] | Exception:
            try:
                # Get positions from Coinbase client
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
                                f"Failed to convert position {pos.symbol}: {e}"
                            )
                            continue

                return fp_positions

            except Exception as e:
                self._logger.error(f"Failed to get positions: {e}")
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
        """Get account balance using Coinbase client with functional effects."""

        def _get_balance() -> dict[str, Any] | Exception:
            try:
                # Get balance from Coinbase client
                balance = self.client.get_account_balance()

                # Convert to functional AccountBalance format
                fp_balance = create_fp_account_balance(balance)
                return fp_balance

            except Exception as e:
                self._logger.error(f"Failed to get balance: {e}")
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
        """Connect to Coinbase with functional effects."""

        def _connect() -> bool | Exception:
            try:
                result = self.client.connect()
                return result if result is not None else False
            except Exception as e:
                self._logger.error(f"Connection failed: {e}")
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
        """Disconnect from Coinbase with functional effects."""

        def _disconnect() -> bool | Exception:
            try:
                self.client.disconnect()
                return True
            except Exception as e:
                self._logger.error(f"Disconnection failed: {e}")
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

    def get_futures_positions_impl(self) -> IOEither[Exception, list[Position]]:
        """Get futures positions using Coinbase client with functional effects."""

        def _get_futures_positions() -> list[Position] | Exception:
            try:
                # Get futures positions from Coinbase client
                current_positions = self.client.get_futures_positions()

                # Convert to functional positions
                fp_positions = []
                for pos in current_positions:
                    if pos.side != "FLAT":  # Only include non-flat positions
                        try:
                            fp_pos = current_position_to_fp_position(pos)
                            fp_positions.append(fp_pos)
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to convert futures position {pos.symbol}: {e}"
                            )
                            continue

                return fp_positions

            except Exception as e:
                self._logger.error(f"Failed to get futures positions: {e}")
                return e

        def safe_get_futures_positions():
            try:
                result = _get_futures_positions()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_get_futures_positions)

    def place_futures_order_impl(
        self, order: Order, leverage: int | None = None
    ) -> IOEither[Exception, OrderResult]:
        """Place futures order using Coinbase client with functional effects."""

        def _place_futures_order() -> OrderResult | Exception:
            try:
                # Convert FP order to current order type
                current_order = fp_order_to_current_order(order)

                # Execute using existing Coinbase futures client methods
                result = self.client.place_futures_market_order(
                    symbol=current_order.symbol,
                    side=current_order.side,
                    quantity=current_order.quantity,
                    leverage=leverage,
                    reduce_only=False,
                )

                if result is None:
                    raise Exception(
                        "Futures order placement failed - no result returned"
                    )

                # Convert result to functional OrderResult
                order_result = create_order_result(result, success=True)
                return order_result

            except Exception as e:
                self._logger.error(f"Futures order placement failed: {e}")
                return e

        def safe_place_futures_order():
            try:
                result = _place_futures_order()
                if isinstance(result, Exception):
                    return Left(result)
                return Right(result)
            except Exception as e:
                return Left(e)

        return IOEither(safe_place_futures_order)


def create_coinbase_adapter(
    cdp_api_key_name: str | None = None,
    cdp_private_key: str | None = None,
    dry_run: bool = True,
) -> CoinbaseExchangeAdapter:
    """Factory function to create a Coinbase exchange adapter."""
    # Create the underlying Coinbase client
    coinbase_client = CoinbaseClient(
        cdp_api_key_name=cdp_api_key_name,
        cdp_private_key=cdp_private_key,
        dry_run=dry_run,
    )

    # Return the functional adapter
    return CoinbaseExchangeAdapter(coinbase_client)
