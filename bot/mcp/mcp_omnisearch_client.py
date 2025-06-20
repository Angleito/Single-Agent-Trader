"""
MCP-OmniSearch Client for AI Trading Bot.

This client connects to the MCP-OmniSearch server to provide web searching,
AI responses, content processing, and market intelligence for trading decisions.
"""

import asyncio
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field


class MCPConnectionError(Exception):
    """Raised when MCP server connection fails."""


logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Individual search result item."""

    result_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    url: str
    snippet: str
    source: str
    published_date: datetime | None = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        json_encoders: ClassVar[dict[type, Any]] = {datetime: lambda v: v.isoformat()}


class FinancialNewsResult(BaseModel):
    """Financial news search result with additional metadata."""

    base_result: SearchResult
    sentiment: str | None = None  # "positive", "negative", "neutral"
    mentioned_symbols: list[str] = Field(default_factory=list)
    news_category: str | None = None  # "earnings", "regulation", "adoption", etc.
    impact_level: str | None = None  # "high", "medium", "low"


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result for a symbol or market."""

    symbol: str
    overall_sentiment: str  # "bullish", "bearish", "neutral"
    sentiment_score: float = Field(
        ge=-1.0, le=1.0
    )  # -1 (very bearish) to 1 (very bullish)
    confidence: float = Field(ge=0.0, le=1.0)
    source_count: int
    timeframe: str = "24h"

    # Detailed breakdown
    news_sentiment: float | None = None
    social_sentiment: float | None = None
    technical_sentiment: float | None = None

    # Key insights
    key_drivers: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)


class MarketCorrelation(BaseModel):
    """Market correlation analysis between assets."""

    primary_symbol: str
    secondary_symbol: str
    correlation_coefficient: float = Field(ge=-1.0, le=1.0)
    timeframe: str = "30d"
    strength: str  # "strong", "moderate", "weak"
    direction: str  # "positive", "negative", "neutral"

    # Additional metrics
    beta: float | None = None
    r_squared: float | None = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MCPOmniSearchClient:
    """
    MCP-OmniSearch client for enhanced market intelligence.

    Connects to the MCP-OmniSearch server via subprocess and MCP protocol
    to provide comprehensive search, AI responses, and content processing.
    """

    def __init__(
        self,
        server_path: str | None = None,
        enable_cache: bool = True,
        cache_ttl: int = 900,
    ):
        """Initialize the MCP-OmniSearch client."""
        # Server configuration
        self.server_path = server_path or "/app/bot/mcp/omnisearch-server/dist/index.js"

        # Client state
        self._process: subprocess.Popen | None = None
        self._connected = False
        self._request_id = 0

        # Cache configuration
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self._cache: dict[str, dict[str, Any]] = {}

        logger.info("🔍 MCP-OmniSearch Client: Initialized for %s", self.server_path)

    async def connect(self) -> bool:
        """Connect to the MCP-OmniSearch server."""
        try:
            # Security: Validate that node executable exists and server path is safe
            node_executable = shutil.which("node")
            if not node_executable:
                logger.error("Node.js executable not found in PATH")
                return False

            # Security: Validate server path exists and is not attempting path traversal
            from pathlib import Path

            if not Path(self.server_path).exists() or ".." in self.server_path:
                logger.error("Invalid or unsafe server path: %s", self.server_path)
                return False

            # Start the MCP server process with validated paths
            self._process = await asyncio.to_thread(
                subprocess.Popen,
                [node_executable, self.server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
            )

            # Initialize the connection
            await self._send_initialize()
            self._connected = True

            logger.info("✅ MCP-OmniSearch: Successfully connected")
            return True

        except Exception:
            logger.exception("Failed to connect to MCP-OmniSearch server")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP-OmniSearch server."""
        try:
            if self._process:
                self._process.terminate()
                self._process.wait(timeout=5)
                self._process = None

            self._connected = False
            logger.info("Disconnected from MCP-OmniSearch server")
        except Exception:
            logger.exception("Error during disconnect")

    async def _send_initialize(self) -> None:
        """Send initialization message to MCP server."""
        init_message = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "ai-trading-bot", "version": "1.0.0"},
            },
        }

        await self._send_message(init_message)
        response = await self._read_message()

        if response.get("error"):
            raise MCPConnectionError(f"MCP initialization failed: {response['error']}")

    async def _send_message(self, message: dict[str, Any]) -> None:
        """Send a message to the MCP server."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("MCP server not connected")

        message_str = json.dumps(message) + "\n"
        self._process.stdin.write(message_str)
        self._process.stdin.flush()

    async def _read_message(self) -> dict[str, Any]:
        """Read a message from the MCP server."""
        if not self._process or not self._process.stdout:
            raise MCPConnectionError("MCP server not connected")

        line = self._process.stdout.readline()
        if not line:
            raise MCPConnectionError("MCP server connection closed")

        return json.loads(line.strip())

    def _get_next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    async def _call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on the MCP server."""
        message = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        await self._send_message(message)
        response = await self._read_message()

        if response.get("error"):
            raise MCPConnectionError(f"Tool call failed: {response['error']}")

        return response.get("result", {})

    async def search_financial_news(
        self,
        query: str,
        limit: int = 5,
        timeframe: str = "24h",
        include_sentiment: bool = True,
    ) -> list[FinancialNewsResult]:
        """
        Search for financial news using MCP-OmniSearch.

        Args:
            query: Search query (e.g., "Bitcoin ETF approval", "Ethereum regulation")
            limit: Maximum number of results to return
            timeframe: Time range for news ("1h", "24h", "7d", "30d")
            include_sentiment: Whether to include sentiment analysis

        Returns:
            List of financial news results with metadata
        """
        try:
            # Use Tavily search for financial news
            result = await self._call_tool(
                "search_tavily",
                {
                    "query": f"{query} financial news {timeframe}",
                },
            )

            # Process results into financial news format
            financial_results = []
            content = result.get("content", [])

            if isinstance(content, list):
                for item in content[:limit]:
                    if isinstance(item, dict):
                        base_result = SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("content", "")[:200],
                            source=(
                                item.get("url", "").split("/")[2]
                                if item.get("url")
                                else ""
                            ),
                            relevance_score=item.get("score", 0.0),
                        )

                        # Simple sentiment analysis based on keywords
                        sentiment = self._analyze_sentiment(item.get("content", ""))

                        financial_result = FinancialNewsResult(
                            base_result=base_result,
                            sentiment=sentiment if include_sentiment else None,
                            mentioned_symbols=self._extract_symbols(
                                item.get("content", "")
                            ),
                            news_category="market_news",
                            impact_level="medium",
                        )
                        financial_results.append(financial_result)

            logger.info(
                "🔍 MCP-OmniSearch: Found %s financial news results for '%s'",
                len(financial_results),
                query,
            )
            return financial_results

        except Exception:
            logger.exception("Financial news search failed for '%s'", query)
            return []

    async def search_crypto_sentiment(self, symbol: str) -> SentimentAnalysis:
        """
        Analyze sentiment for a specific cryptocurrency using AI response.

        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH", "BTC-USD")

        Returns:
            Comprehensive sentiment analysis
        """
        # Normalize symbol
        base_symbol = symbol.split("-")[0].upper()

        try:
            # Use Perplexity AI for sentiment analysis
            result = await self._call_tool(
                "ai_perplexity",
                {
                    "query": f"What is the current market sentiment for {base_symbol} cryptocurrency? Include bullish/bearish indicators, news sentiment, and technical analysis sentiment. Provide a numerical sentiment score from -1 (very bearish) to 1 (very bullish)."
                },
            )

            content = result.get("content", [])
            response_text = ""

            if isinstance(content, list) and content:
                response_text = (
                    content[0].get("text", "")
                    if isinstance(content[0], dict)
                    else str(content[0])
                )
            elif isinstance(content, str):
                response_text = content

            # Parse AI response for sentiment
            sentiment_score = self._extract_sentiment_score(response_text)
            overall_sentiment = self._score_to_sentiment(sentiment_score)

            sentiment = SentimentAnalysis(
                symbol=base_symbol,
                overall_sentiment=overall_sentiment,
                sentiment_score=sentiment_score,
                confidence=0.7,  # Medium confidence for AI-based analysis
                source_count=1,
                key_drivers=self._extract_key_drivers(response_text),
                risk_factors=self._extract_risk_factors(response_text),
            )

            logger.info(
                "🔍 MCP-OmniSearch: %s sentiment - %s (score: %.2f)",
                base_symbol,
                sentiment.overall_sentiment,
                sentiment.sentiment_score,
            )
            return sentiment

        except Exception:
            logger.exception("Crypto sentiment search failed for %s", base_symbol)
            return self._get_fallback_sentiment(base_symbol)

    async def search_nasdaq_sentiment(self) -> SentimentAnalysis:
        """
        Analyze overall NASDAQ/stock market sentiment.

        Returns:
            NASDAQ market sentiment analysis
        """
        try:
            # Use Kagi FastGPT for quick market sentiment
            result = await self._call_tool(
                "ai_kagi_fastgpt",
                {
                    "query": "What is the current NASDAQ and overall stock market sentiment? Include key market indicators, news sentiment, and provide a sentiment score from -1 (very bearish) to 1 (very bullish)."
                },
            )

            content = result.get("content", [])
            response_text = ""

            if isinstance(content, list) and content:
                response_text = (
                    content[0].get("text", "")
                    if isinstance(content[0], dict)
                    else str(content[0])
                )
            elif isinstance(content, str):
                response_text = content

            # Parse AI response for sentiment
            sentiment_score = self._extract_sentiment_score(response_text)
            overall_sentiment = self._score_to_sentiment(sentiment_score)

            sentiment = SentimentAnalysis(
                symbol="NASDAQ",
                overall_sentiment=overall_sentiment,
                sentiment_score=sentiment_score,
                confidence=0.7,
                source_count=1,
                key_drivers=self._extract_key_drivers(response_text),
                risk_factors=self._extract_risk_factors(response_text),
            )

            logger.info(
                "🔍 MCP-OmniSearch: NASDAQ sentiment - %s (score: %.2f)",
                sentiment.overall_sentiment,
                sentiment.sentiment_score,
            )
            return sentiment

        except Exception:
            logger.exception("NASDAQ sentiment search failed")
            return self._get_fallback_sentiment("NASDAQ")

    async def search_market_correlation(
        self, crypto_symbol: str, nasdaq_symbol: str = "QQQ", timeframe: str = "30d"
    ) -> MarketCorrelation:
        """
        Analyze correlation between crypto and traditional markets.

        Args:
            crypto_symbol: Crypto symbol (e.g., "BTC", "ETH")
            nasdaq_symbol: NASDAQ symbol to correlate with (default: "QQQ")
            timeframe: Analysis timeframe ("7d", "30d", "90d")

        Returns:
            Market correlation analysis
        """
        # Normalize symbols
        crypto_base = crypto_symbol.split("-")[0].upper()
        nasdaq_base = nasdaq_symbol.upper()

        try:
            # Use search to find correlation information
            await self._call_tool(
                "search_kagi",
                {
                    "query": f"{crypto_base} {nasdaq_base} correlation analysis {timeframe} market relationship",
                    "language": "en",
                },
            )

            # For now, return a neutral correlation as we'd need specialized financial APIs
            # for accurate correlation calculations
            correlation = MarketCorrelation(
                primary_symbol=crypto_base,
                secondary_symbol=nasdaq_base,
                correlation_coefficient=0.0,
                timeframe=timeframe,
                strength="weak",
                direction="neutral",
            )

            logger.info(
                "🔍 MCP-OmniSearch: %s-%s correlation - neutral weak (0.000)",
                crypto_base,
                nasdaq_base,
            )
            return correlation

        except Exception:
            logger.exception(
                "Market correlation search failed for %s-%s",
                crypto_base,
                nasdaq_base,
            )
            return self._get_fallback_correlation(crypto_base, nasdaq_base, timeframe)

    def _analyze_sentiment(self, text: str) -> str:
        """Simple keyword-based sentiment analysis."""
        positive_words = [
            "bullish",
            "positive",
            "growth",
            "increase",
            "up",
            "gains",
            "rally",
            "surge",
        ]
        negative_words = [
            "bearish",
            "negative",
            "decline",
            "decrease",
            "down",
            "losses",
            "crash",
            "fall",
        ]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return "positive"
        if neg_count > pos_count:
            return "negative"
        return "neutral"

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract crypto symbols from text."""
        symbols = []
        common_symbols = ["BTC", "ETH", "ADA", "SOL", "DOGE", "XRP", "MATIC", "AVAX"]

        text_upper = text.upper()
        for symbol in common_symbols:
            if symbol in text_upper:
                symbols.append(symbol)

        return symbols

    def _extract_sentiment_score(self, text: str) -> float:
        """Extract sentiment score from AI response text."""
        import re

        # Look for numerical sentiment scores
        patterns = [
            r"sentiment score.*?(-?\d+\.?\d*)",
            r"score.*?(-?\d+\.?\d*)",
            r"(-?\d+\.?\d*).*?sentiment",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to -1 to 1 range
                    return max(-1.0, min(1.0, score))
                except ValueError:
                    continue

        # Fallback: keyword-based scoring
        sentiment = self._analyze_sentiment(text)
        if sentiment == "positive":
            return 0.3
        if sentiment == "negative":
            return -0.3
        return 0.0

    def _score_to_sentiment(self, score: float) -> str:
        """Convert sentiment score to sentiment label."""
        if score > 0.2:
            return "bullish"
        if score < -0.2:
            return "bearish"
        return "neutral"

    def _extract_key_drivers(self, text: str) -> list[str]:
        """Extract key market drivers from text."""
        drivers = []
        driver_keywords = [
            "ETF",
            "regulation",
            "institutional",
            "adoption",
            "technology",
            "partnership",
            "earnings",
            "fed",
            "interest rates",
            "inflation",
            "macro",
        ]

        text_lower = text.lower()
        for keyword in driver_keywords:
            if keyword.lower() in text_lower:
                drivers.append(keyword)

        return drivers[:3]  # Limit to top 3

    def _extract_risk_factors(self, text: str) -> list[str]:
        """Extract risk factors from text."""
        risks = []
        risk_keywords = [
            "volatility",
            "regulation",
            "security",
            "hack",
            "market crash",
            "liquidity",
            "manipulation",
            "uncertainty",
        ]

        text_lower = text.lower()
        for keyword in risk_keywords:
            if keyword.lower() in text_lower:
                risks.append(keyword)

        return risks[:3]  # Limit to top 3

    def _get_fallback_sentiment(self, symbol: str) -> SentimentAnalysis:
        """Provide fallback sentiment when analysis fails."""
        return SentimentAnalysis(
            symbol=symbol,
            overall_sentiment="neutral",
            sentiment_score=0.0,
            confidence=0.1,
            source_count=0,
            key_drivers=["MCP server unavailable"],
            risk_factors=["Limited sentiment data available"],
        )

    def _get_fallback_correlation(
        self, crypto_symbol: str, nasdaq_symbol: str, timeframe: str
    ) -> MarketCorrelation:
        """Provide fallback correlation when analysis fails."""
        return MarketCorrelation(
            primary_symbol=crypto_symbol,
            secondary_symbol=nasdaq_symbol,
            correlation_coefficient=0.0,
            timeframe=timeframe,
            strength="weak",
            direction="neutral",
        )

    async def health_check(self) -> dict[str, Any]:
        """Check the health and status of the MCP-OmniSearch client."""
        return {
            "connected": self._connected,
            "server_path": self.server_path,
            "cache_enabled": self.enable_cache,
            "process_alive": self._process is not None and self._process.poll() is None,
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Example usage and testing
async def main():
    """Example usage of the MCPOmniSearchClient."""
    client = MCPOmniSearchClient()

    try:
        # Connect to service
        connected = await client.connect()
        if not connected:
            logger.warning("Could not connect to MCP-OmniSearch service")
            return

        # Test financial news search
        news_results = await client.search_financial_news(
            "Bitcoin ETF approval", limit=3
        )
        print(f"Found {len(news_results)} news results")

        # Test crypto sentiment
        btc_sentiment = await client.search_crypto_sentiment("BTC-USD")
        print(
            f"BTC sentiment: {btc_sentiment.overall_sentiment} ({btc_sentiment.sentiment_score:.2f})"
        )

        # Test NASDAQ sentiment
        nasdaq_sentiment = await client.search_nasdaq_sentiment()
        print(
            f"NASDAQ sentiment: {nasdaq_sentiment.overall_sentiment} ({nasdaq_sentiment.sentiment_score:.2f})"
        )

        # Health check
        health = await client.health_check()
        print(f"Client health: {health}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
