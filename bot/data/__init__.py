"""Data ingestion and market data handling modules."""

from .base_market_provider import AbstractMarketDataProvider
from .dominance import DominanceData, DominanceDataProvider
from .factory import create_market_data_provider
from .market import (
    MarketDataClient,
    MarketDataFeed,
    create_market_data_client,
)

# Import legacy classes for backward compatibility
from .market import (
    MarketDataProvider as LegacyMarketDataProvider,
)

# Export the factory function as MarketDataProvider for backward compatibility
MarketDataProvider = create_market_data_provider

__all__ = [
    "AbstractMarketDataProvider",
    "DominanceData",
    "DominanceDataProvider",
    "LegacyMarketDataProvider",
    "MarketDataClient",
    "MarketDataFeed",
    "MarketDataProvider",  # This is now the factory function
    "create_market_data_client",
    "create_market_data_provider",
]
