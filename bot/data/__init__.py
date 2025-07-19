"""Data ingestion and market data handling modules."""

from .dominance import DominanceData, DominanceDataProvider
from .factory import create_market_data_provider
from .base_market_provider import AbstractMarketDataProvider

# Import legacy classes for backward compatibility
from .market import (
    MarketDataProvider as LegacyMarketDataProvider,
    MarketDataClient,
    MarketDataFeed,
    create_market_data_client,
)

# Export the factory function as MarketDataProvider for backward compatibility
MarketDataProvider = create_market_data_provider

__all__ = [
    "DominanceData",
    "DominanceDataProvider",
    "MarketDataProvider",  # This is now the factory function
    "create_market_data_provider",
    "AbstractMarketDataProvider",
    "LegacyMarketDataProvider",
    "MarketDataClient",
    "MarketDataFeed",
    "create_market_data_client",
]
