"""
OmniSearch MCP Client for AI Trading Bot.

Provides web searching capabilities for financial data, news, and market sentiment
analysis to enhance trading decisions with real-time external information.
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import aiohttp
from pydantic import BaseModel, Field

from bot.config import settings


class OmniSearchError(Exception):
    """Base exception for OmniSearch client errors."""


class OmniSearchConnectionError(OmniSearchError):
    """Raised when connection to OmniSearch service fails."""


class OmniSearchAPIError(OmniSearchError):
    """Raised when OmniSearch API returns an error."""


class OmniSearchTimeoutError(OmniSearchError):
    """Raised when OmniSearch requests timeout."""


logger = logging.getLogger(__name__)


def get_data_directory(subdirectory: str = "omnisearch_cache") -> Path:
    """
    Get the appropriate data directory with fallback support.

    Args:
        subdirectory: The subdirectory name within the data directory

    Returns:
        Path to the data directory that can be written to

    Raises:
        PermissionError: If neither the original nor fallback directory can be created
    """
    # Try original data directory first
    original_path = Path("data") / subdirectory

    def try_create_directory(path: Path) -> bool:
        """Try to create a directory and test write permissions."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test write permissions by creating a temporary file
            test_file = path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except (PermissionError, OSError):
            return False

    # Try original path first
    if try_create_directory(original_path):
        logger.debug("Using original data directory: %s", original_path)
        return original_path

    # Try fallback directory from environment variable
    fallback_dir = os.getenv("FALLBACK_DATA_DIR")
    if fallback_dir:
        fallback_path = Path(fallback_dir) / subdirectory
        if try_create_directory(fallback_path):
            logger.info("Using fallback data directory: %s", fallback_path)
            return fallback_path
        logger.warning("Could not create fallback directory: %s", fallback_path)

    # Try system temporary directory as last resort
    import tempfile

    temp_path = Path(tempfile.gettempdir()) / "ai_trading_bot" / subdirectory
    if try_create_directory(temp_path):
        logger.warning("Using temporary directory for data storage: %s", temp_path)
        return temp_path

    # If all else fails, raise an error
    raise PermissionError(
        f"Cannot create data directory. Tried: {original_path}, "
        f"{fallback_path if fallback_dir else 'no fallback set'}, {temp_path}. "
        "Set FALLBACK_DATA_DIR environment variable to specify an alternative location."
    )


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


class SearchCache:
    """Simple in-memory cache with TTL for search results."""

    def __init__(self, default_ttl: int = 900):  # 15 minutes default
        self.cache: dict[str, dict[str, Any]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if entry["expires_at"] > time.time():
                return entry["value"]
            del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set cached value with TTL."""
        expires_at = time.time() + (ttl or self.default_ttl)
        self.cache[key] = {"value": value, "expires_at": expires_at}

    def clear_expired(self) -> int:
        """Clear expired entries and return count removed."""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self.cache.items()
            if entry["expires_at"] <= current_time
        ]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []

    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        current_time = time.time()

        # Remove old requests outside the window
        self.requests = [
            req_time
            for req_time in self.requests
            if current_time - req_time < self.window_seconds
        ]

        if len(self.requests) >= self.max_requests:
            # Calculate wait time until oldest request expires
            wait_time = self.window_seconds - (current_time - self.requests[0])
            if wait_time > 0:
                logger.warning("Rate limit reached, waiting %.1fs", wait_time)
                await asyncio.sleep(wait_time)
                return await self.acquire()  # Retry after waiting

        self.requests.append(current_time)
        return True


class OmniSearchClient:
    """
    MCP-based OmniSearch client for financial data and news searching.

    Provides intelligent web searching capabilities with caching, rate limiting,
    and financial-specific result processing for enhanced trading decisions.
    """

    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        enable_cache: bool = True,
        cache_ttl: int = 900,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 3600,
    ):
        """Initialize the OmniSearch client."""
        # Server configuration
        self.server_url = server_url or getattr(
            settings.omnisearch, "server_url", "https://api.omnisearch.dev/v1"
        )
        self.api_key = api_key or (
            settings.omnisearch.api_key.get_secret_value()
            if settings.omnisearch.api_key
            else None
        )

        # Client state
        self._session: aiohttp.ClientSession | None = None
        self._connected = False

        # Cache and rate limiting
        self.cache = SearchCache(cache_ttl) if enable_cache else None
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)

        # Local storage for fallback with permission handling
        try:
            self.local_storage_path = get_data_directory("omnisearch_cache")
        except PermissionError:
            logger.exception("Failed to create data directory")
            # Fall back to a minimal path that won't be used for actual storage
            import tempfile

            self.local_storage_path = (
                Path(tempfile.gettempdir()) / "omnisearch_cache_fallback"
            )
            logger.warning("Using minimal fallback path: %s", self.local_storage_path)

        logger.info("🔍 OmniSearch Client: Initialized for %s", self.server_url)

    async def connect(self) -> bool:
        """Connect to the OmniSearch service."""
        try:
            if self._session is None:
                timeout = aiohttp.ClientTimeout(total=30)
                self._session = aiohttp.ClientSession(timeout=timeout)

            # Test connection with a simple health check
            headers = self._get_headers()

            async with self._session.get(
                f"{self.server_url}/health", headers=headers
            ) as response:
                if response.status == 200:
                    self._connected = True
                    logger.info("✅ OmniSearch: Successfully connected")
                    return True
                logger.warning("OmniSearch server returned status %s", response.status)
                return False

        except Exception:
            logger.exception("Failed to connect to OmniSearch service")
            # Set connected to False but don't raise - allows graceful degradation
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the OmniSearch service."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

        # Clean up expired cache entries
        if self.cache:
            expired_count = self.cache.clear_expired()
            if expired_count > 0:
                logger.info("Cleaned up %s expired cache entries", expired_count)

        logger.info("Disconnected from OmniSearch service")

    async def search_financial_news(
        self,
        query: str,
        limit: int = 5,
        timeframe: str = "24h",
        include_sentiment: bool = True,
    ) -> list[FinancialNewsResult]:
        """
        Search for financial news related to the query.

        Args:
            query: Search query (e.g., "Bitcoin ETF approval", "Ethereum regulation")
            limit: Maximum number of results to return
            timeframe: Time range for news ("1h", "24h", "7d", "30d")
            include_sentiment: Whether to include sentiment analysis

        Returns:
            List of financial news results with metadata
        """
        cache_key = f"financial_news:{query}:{limit}:{timeframe}:{include_sentiment}"

        # Try cache first
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(
                    "OmniSearch: Cache hit for financial news query: %s", query
                )
                return [FinancialNewsResult(**item) for item in cached_result]

        # Rate limiting
        await self.rate_limiter.acquire()

        try:
            results = await self._search_with_fallback(
                endpoint="financial-news",
                params={
                    "q": query,
                    "limit": limit,
                    "timeframe": timeframe,
                    "include_sentiment": str(include_sentiment).lower(),
                },
            )

            # Process results into structured format
            financial_results = []
            for item in results.get("results", []):
                base_result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source=item.get("source", ""),
                    published_date=self._parse_date(item.get("published_date")),
                    relevance_score=item.get("relevance_score", 0.0),
                )

                financial_result = FinancialNewsResult(
                    base_result=base_result,
                    sentiment=item.get("sentiment"),
                    mentioned_symbols=item.get("mentioned_symbols", []),
                    news_category=item.get("category"),
                    impact_level=item.get("impact_level"),
                )
                financial_results.append(financial_result)

            # Cache results
            if self.cache and financial_results:
                cache_data = [result.dict() for result in financial_results]
                self.cache.set(cache_key, cache_data)

            logger.info(
                "🔍 OmniSearch: Found %s financial news results for '%s'",
                len(financial_results),
                query,
            )

            return financial_results

        except Exception:
            logger.exception("Financial news search failed for '%s'", query)
            return await self._get_fallback_news(query, limit)

    async def search_crypto_sentiment(self, symbol: str) -> SentimentAnalysis:
        """
        Analyze sentiment for a specific cryptocurrency.

        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH", "BTC-USD")

        Returns:
            Comprehensive sentiment analysis
        """
        # Normalize symbol
        base_symbol = symbol.split("-")[0].upper()
        cache_key = f"crypto_sentiment:{base_symbol}"

        # Try cache first
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(
                    "OmniSearch: Cache hit for crypto sentiment: %s", base_symbol
                )
                return SentimentAnalysis(**cached_result)

        # Rate limiting
        await self.rate_limiter.acquire()

        try:
            results = await self._search_with_fallback(
                endpoint="crypto-sentiment",
                params={
                    "symbol": base_symbol,
                    "include_social": "true",
                    "include_news": "true",
                    "include_technical": "true",
                },
            )

            sentiment_data = results.get("sentiment", {})

            sentiment = SentimentAnalysis(
                symbol=base_symbol,
                overall_sentiment=sentiment_data.get("overall", "neutral"),
                sentiment_score=sentiment_data.get("score", 0.0),
                confidence=sentiment_data.get("confidence", 0.5),
                source_count=sentiment_data.get("source_count", 0),
                news_sentiment=sentiment_data.get("news_sentiment"),
                social_sentiment=sentiment_data.get("social_sentiment"),
                technical_sentiment=sentiment_data.get("technical_sentiment"),
                key_drivers=sentiment_data.get("key_drivers", []),
                risk_factors=sentiment_data.get("risk_factors", []),
            )

            # Cache result
            if self.cache:
                self.cache.set(cache_key, sentiment.dict())

            logger.info(
                "🔍 OmniSearch: %s sentiment - %s (score: %.2f, confidence: %.2f)",
                base_symbol,
                sentiment.overall_sentiment,
                sentiment.sentiment_score,
                sentiment.confidence,
            )

            return sentiment

        except Exception:
            logger.exception("Crypto sentiment search failed for %s", base_symbol)
            return await self._get_fallback_sentiment(base_symbol)

    async def search_nasdaq_sentiment(self) -> SentimentAnalysis:
        """
        Analyze overall NASDAQ/stock market sentiment.

        Returns:
            NASDAQ market sentiment analysis
        """
        cache_key = "nasdaq_sentiment"

        # Try cache first
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug("OmniSearch: Cache hit for NASDAQ sentiment")
                return SentimentAnalysis(**cached_result)

        # Rate limiting
        await self.rate_limiter.acquire()

        try:
            results = await self._search_with_fallback(
                endpoint="market-sentiment",
                params={
                    "market": "nasdaq",
                    "include_indices": "true",
                    "include_sectors": "true",
                },
            )

            sentiment_data = results.get("sentiment", {})

            sentiment = SentimentAnalysis(
                symbol="NASDAQ",
                overall_sentiment=sentiment_data.get("overall", "neutral"),
                sentiment_score=sentiment_data.get("score", 0.0),
                confidence=sentiment_data.get("confidence", 0.5),
                source_count=sentiment_data.get("source_count", 0),
                news_sentiment=sentiment_data.get("news_sentiment"),
                social_sentiment=sentiment_data.get("social_sentiment"),
                technical_sentiment=sentiment_data.get("technical_sentiment"),
                key_drivers=sentiment_data.get("key_drivers", []),
                risk_factors=sentiment_data.get("risk_factors", []),
            )

            # Cache result (shorter TTL for market sentiment)
            if self.cache:
                self.cache.set(cache_key, sentiment.dict(), ttl=300)  # 5 minutes

            logger.info(
                "🔍 OmniSearch: NASDAQ sentiment - %s (score: %.2f)",
                sentiment.overall_sentiment,
                sentiment.sentiment_score,
            )

            return sentiment

        except Exception:
            logger.exception("NASDAQ sentiment search failed")
            return await self._get_fallback_sentiment("NASDAQ")

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

        cache_key = f"market_correlation:{crypto_base}:{nasdaq_base}:{timeframe}"

        # Try cache first
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(
                    "OmniSearch: Cache hit for correlation %s-%s",
                    crypto_base,
                    nasdaq_base,
                )
                return MarketCorrelation(**cached_result)

        # Rate limiting
        await self.rate_limiter.acquire()

        try:
            results = await self._search_with_fallback(
                endpoint="market-correlation",
                params={
                    "symbol1": crypto_base,
                    "symbol2": nasdaq_base,
                    "timeframe": timeframe,
                    "include_beta": "true",
                },
            )

            correlation_data = results.get("correlation", {})
            raw_correlation_coeff = correlation_data.get("coefficient", 0.0)

            # Debug logging for API response
            logger.debug(
                "API correlation response for %s-%s: %s",
                crypto_base,
                nasdaq_base,
                correlation_data,
            )

            # Validate and clamp correlation coefficient to valid range
            try:
                correlation_coeff = float(raw_correlation_coeff)

                # Check for extremely large values that might be percentages, timeframe values, or other units
                if abs(correlation_coeff) > 10.0:
                    logger.warning(
                        "Extremely large correlation value %s from API for %s-%s. Timeframe: %s. Likely incorrect unit or data mixup. Raw API response: %s",
                        correlation_coeff,
                        crypto_base,
                        nasdaq_base,
                        timeframe,
                        correlation_data,
                    )

                    # Check if this might be a timeframe value (e.g., 30 from "30d")
                    timeframe_numeric = None
                    if timeframe and timeframe.endswith("d"):
                        with contextlib.suppress(ValueError):
                            timeframe_numeric = int(timeframe[:-1])

                    if (
                        timeframe_numeric
                        and abs(correlation_coeff) == timeframe_numeric
                    ):
                        logger.error(
                            "Correlation coefficient %s matches timeframe %s! API likely returned timeframe value instead of correlation. Defaulting to 0.0.",
                            correlation_coeff,
                            timeframe,
                        )
                        correlation_coeff = 0.0
                    elif abs(correlation_coeff) <= 100.0:
                        # If it looks like a percentage (e.g., 30 meaning 30%), normalize it
                        correlation_coeff = correlation_coeff / 100.0
                        logger.warning(
                            "Normalized correlation from %s to %s (divided by 100)",
                            raw_correlation_coeff,
                            correlation_coeff,
                        )
                    else:
                        # For even larger values, default to neutral correlation
                        logger.warning(
                            "Value %s too large even for percentage, defaulting to 0.0",
                            correlation_coeff,
                        )
                        correlation_coeff = 0.0

                # Final clamp to valid correlation range [-1, 1]
                if abs(correlation_coeff) > 1.0:
                    logger.warning(
                        "Clamping correlation coefficient %s to valid range [-1, 1]",
                        correlation_coeff,
                    )
                    correlation_coeff = max(-1.0, min(1.0, correlation_coeff))

            except (ValueError, TypeError):
                logger.exception(
                    "Invalid correlation coefficient type %s: %s from API for %s-%s. Using fallback value 0.0",
                    type(raw_correlation_coeff),
                    raw_correlation_coeff,
                    crypto_base,
                    nasdaq_base,
                )
                correlation_coeff = 0.0

            # Determine correlation strength and direction
            abs_corr = abs(correlation_coeff)
            if abs_corr >= 0.7:
                strength = "strong"
            elif abs_corr >= 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            if correlation_coeff > 0.1:
                direction = "positive"
            elif correlation_coeff < -0.1:
                direction = "negative"
            else:
                direction = "neutral"

            # Validate beta value
            raw_beta = correlation_data.get("beta")
            beta = None
            if raw_beta is not None:
                try:
                    beta = float(raw_beta)
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid beta value %s from API. Setting to None.", raw_beta
                    )
                    beta = None

            # Validate r_squared value (should be between 0 and 1)
            raw_r_squared = correlation_data.get("r_squared")
            r_squared = None
            if raw_r_squared is not None:
                try:
                    r_squared = float(raw_r_squared)
                    if r_squared < 0.0 or r_squared > 1.0:
                        logger.warning(
                            "Invalid r_squared value %s from API. Should be between 0 and 1. Setting to None.",
                            r_squared,
                        )
                        r_squared = None
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid r_squared value %s from API. Setting to None.",
                        raw_r_squared,
                    )
                    r_squared = None

            correlation = MarketCorrelation(
                primary_symbol=crypto_base,
                secondary_symbol=nasdaq_base,
                correlation_coefficient=correlation_coeff,
                timeframe=timeframe,
                strength=strength,
                direction=direction,
                beta=beta,
                r_squared=r_squared,
            )

            # Cache result
            if self.cache:
                self.cache.set(cache_key, correlation.dict(), ttl=1800)  # 30 minutes

            logger.info(
                "🔍 OmniSearch: %s-%s correlation - %s %s (%.3f)",
                crypto_base,
                nasdaq_base,
                direction,
                strength,
                correlation_coeff,
            )

            return correlation

        except Exception:
            logger.exception(
                "Market correlation search failed for %s-%s",
                crypto_base,
                nasdaq_base,
            )
            return await self._get_fallback_correlation(
                crypto_base, nasdaq_base, timeframe
            )

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-Trading-Bot/1.0",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def _search_with_fallback(
        self, endpoint: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute search with fallback handling.

        Args:
            endpoint: API endpoint to call
            params: Query parameters

        Returns:
            Search results or fallback data
        """
        if not self._connected or not self._session:
            raise OmniSearchConnectionError("Not connected to OmniSearch service")

        url = f"{self.server_url}/{endpoint}"
        headers = self._get_headers()

        try:
            async with self._session.get(
                url, headers=headers, params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                if response.status == 429:  # Rate limited
                    logger.warning("OmniSearch API rate limit hit")
                    raise OmniSearchAPIError("Rate limit exceeded")
                logger.warning("OmniSearch API returned status %s", response.status)
                raise OmniSearchAPIError(f"API error: {response.status}")

        except TimeoutError as e:
            logger.warning("OmniSearch request timed out for %s", endpoint)
            raise OmniSearchTimeoutError("Request timeout") from e
        except aiohttp.ClientError as e:
            logger.warning("OmniSearch network error: %s", e)
            raise OmniSearchConnectionError(f"Network error: {e}") from e

    async def _get_fallback_news(
        self, query: str, limit: int
    ) -> list[FinancialNewsResult]:
        """Provide fallback news results when API is unavailable."""
        logger.info("Using fallback news results for: %s", query)

        # Simple fallback with basic results
        fallback_results = []

        # Check local cache files if storage path is available
        try:
            local_file = self.local_storage_path / f"news_{hash(query) % 1000}.json"
            if local_file.exists() and local_file.is_file():
                try:
                    with local_file.open() as f:
                        cached_data = json.load(f)
                        for item in cached_data[:limit]:
                            fallback_results.append(FinancialNewsResult(**item))
                except (OSError, PermissionError, json.JSONDecodeError):
                    logger.debug("Failed to load local cache from %s", local_file)
        except (OSError, PermissionError):
            logger.debug(
                "Cannot access local storage directory: %s", self.local_storage_path
            )

        # If no cached results, return empty with warning
        if not fallback_results:
            logger.warning("No fallback news available for: %s", query)

        return fallback_results

    async def _get_fallback_sentiment(self, symbol: str) -> SentimentAnalysis:
        """Provide fallback sentiment when API is unavailable."""
        logger.info("Using fallback sentiment for: %s", symbol)

        return SentimentAnalysis(
            symbol=symbol,
            overall_sentiment="neutral",
            sentiment_score=0.0,
            confidence=0.1,  # Low confidence for fallback
            source_count=0,
            key_drivers=["API unavailable - using fallback neutral sentiment"],
            risk_factors=["Limited sentiment data available"],
        )

    async def _get_fallback_correlation(
        self, crypto_symbol: str, nasdaq_symbol: str, timeframe: str
    ) -> MarketCorrelation:
        """Provide fallback correlation when API is unavailable."""
        logger.info(
            "Using fallback correlation for: %s-%s", crypto_symbol, nasdaq_symbol
        )

        return MarketCorrelation(
            primary_symbol=crypto_symbol,
            secondary_symbol=nasdaq_symbol,
            correlation_coefficient=0.0,
            timeframe=timeframe,
            strength="weak",
            direction="neutral",
        )

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse date string to datetime object."""
        if not date_str:
            return None

        try:
            # Try common date formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    return datetime.strptime(date_str, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue

            logger.debug("Could not parse date: %s", date_str)
            return None

        except Exception:
            logger.debug("Date parsing error")
            return None

    async def search(self, query: str, limit: int = 5) -> list[SearchResult] | None:
        """
        Generic search method for testing and basic queries.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of search results or None if search fails
        """
        try:
            # Use financial news search as the generic search endpoint
            news_results = await self.search_financial_news(
                query=query, 
                limit=limit, 
                timeframe="24h",
                include_sentiment=False
            )
            
            # Convert to basic SearchResult objects
            basic_results = []
            for news in news_results:
                basic_results.append(news.base_result)
                
            return basic_results
            
        except Exception as e:
            logger.warning("Generic search failed for '%s': %s", query, str(e))
            return None

    async def health_check(self) -> dict[str, Any]:
        """Check the health and status of the OmniSearch client."""
        return {
            "connected": self._connected,
            "server_url": self.server_url,
            "cache_enabled": self.cache is not None,
            "cache_size": len(self.cache.cache) if self.cache else 0,
            "rate_limit_remaining": max(
                0, self.rate_limiter.max_requests - len(self.rate_limiter.requests)
            ),
            "local_storage": str(self.local_storage_path),
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Example usage and testing
async def main():
    """Example usage of the OmniSearchClient."""
    client = OmniSearchClient()

    try:
        # Connect to service
        connected = await client.connect()
        if not connected:
            logger.warning(
                "Could not connect to OmniSearch service, using fallback mode"
            )

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

        # Test market correlation
        correlation = await client.search_market_correlation("BTC", "QQQ")
        print(
            f"BTC-QQQ correlation: {correlation.direction} {correlation.strength} ({correlation.correlation_coefficient:.3f})"
        )

        # Health check
        health = await client.health_check()
        print(f"Client health: {health}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
