"""Data ingestion and market data handling modules."""

from .dominance import DominanceData, DominanceDataProvider

# Temporarily disable FP imports to test core functionality
# Import functional market data provider by default
# try:
#     from ..fp.data import MarketDataProvider, MarketDataClient
#     # Also make the imperative version available for backward compatibility
#     from .market import MarketDataProvider as ImperativeMarketDataProvider
#     __all__ = [
#         "DominanceData",
#         "DominanceDataProvider",
#         "MarketDataProvider",
#         "MarketDataClient",
#         "ImperativeMarketDataProvider"
#     ]
# except ImportError:
# Fallback to imperative version if functional version isn't available
from .market import MarketDataProvider

__all__ = ["DominanceData", "DominanceDataProvider", "MarketDataProvider"]
