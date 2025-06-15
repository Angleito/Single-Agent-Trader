"""
MCP (Model Context Protocol) Memory Module.

Provides persistent memory and learning capabilities for the AI trading bot.
"""

from .memory_server import MCPMemoryServer, MemoryQuery, TradingExperience
from .omnisearch_client import (
    FinancialNews,
    MarketCorrelation,
    OmniSearchClient,
    SearchResult,
    SentimentAnalysis,
    analyze_correlation,
    get_market_sentiment,
    search_crypto_news,
)

__all__ = [
    "MCPMemoryServer",
    "MemoryQuery",
    "TradingExperience",
    "OmniSearchClient",
    "FinancialNews",
    "SentimentAnalysis",
    "MarketCorrelation",
    "SearchResult",
    "analyze_correlation",
    "get_market_sentiment",
    "search_crypto_news",
]
