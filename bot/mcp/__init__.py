"""
MCP (Model Context Protocol) Memory Module.

Provides persistent memory and learning capabilities for the AI trading bot.
"""

# Import omnisearch components (always available)
from .omnisearch_client import (
    FinancialNewsResult,
    MarketCorrelation,
    OmniSearchClient,
    SearchResult,
    SentimentAnalysis,
)

# Try to import memory server components (optional)
try:
    from .memory_server import MCPMemoryServer, MemoryQuery, TradingExperience

    _memory_server_available = True
except ImportError:
    # Memory server components not available - define placeholder None values
    MCPMemoryServer = None
    MemoryQuery = None
    TradingExperience = None
    _memory_server_available = False

__all__ = [
    "FinancialNewsResult",
    "MCPMemoryServer",
    "MarketCorrelation",
    "MemoryQuery",
    "OmniSearchClient",
    "SearchResult",
    "SentimentAnalysis",
    "TradingExperience",
]
