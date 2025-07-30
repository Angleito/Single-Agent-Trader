"""Factory for creating market data providers based on exchange configuration."""

import logging
from typing import Literal

from bot.config import settings
from bot.data.base_market_provider import AbstractMarketDataProvider
from bot.data.providers.bluefin_provider import BluefinMarketDataProvider
from bot.data.providers.coinbase_provider import CoinbaseMarketDataProvider

logger = logging.getLogger(__name__)


def create_market_data_provider(
    exchange_type: Literal["coinbase", "bluefin"] | None = None,
    symbol: str | None = None,
    interval: str | None = None,
) -> AbstractMarketDataProvider:
    """
    Create the appropriate market data provider based on exchange type.

    Args:
        exchange_type: Exchange type ('coinbase' or 'bluefin'). If None, uses settings.
        symbol: Trading symbol. If None, uses settings.
        interval: Candle interval. If None, uses settings.

    Returns:
        Market data provider instance

    Raises:
        ValueError: If exchange type is not supported
    """
    # Use settings if not provided
    exchange_type = exchange_type or settings.exchange.exchange_type

    if exchange_type == "coinbase":
        logger.info("Creating Coinbase market data provider")
        return CoinbaseMarketDataProvider(symbol=symbol, interval=interval)
    if exchange_type == "bluefin":
        logger.info("Creating Bluefin market data provider")
        return BluefinMarketDataProvider(symbol=symbol, interval=interval)
    raise ValueError(f"Unsupported exchange type: {exchange_type}")


# Legacy alias for backward compatibility
MarketDataProvider = create_market_data_provider
