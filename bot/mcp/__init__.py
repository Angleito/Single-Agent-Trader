"""
MCP (Model Context Protocol) Memory Module.

Provides persistent memory and learning capabilities for the AI trading bot.
"""

from .memory_server import MCPMemoryServer, MemoryQuery, TradingExperience
from .omnisearch_client import (
    FinancialNewsResult,
    MarketCorrelation,
    OmniSearchClient,
    SearchResult,
    SentimentAnalysis,
)

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
