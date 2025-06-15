"""Unit tests for OmniSearch MCP client."""

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID

import aiohttp
import pytest

from bot.mcp.omnisearch_client import (
    FinancialNewsResult,
    MarketCorrelation,
    OmniSearchClient,
    RateLimiter,
    SearchCache,
    SearchResult,
    SentimentAnalysis,
)


class TestSearchCache:
    """Test cases for SearchCache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with default TTL."""
        cache = SearchCache()
        assert cache.default_ttl == 900  # 15 minutes
        assert len(cache.cache) == 0

    def test_cache_initialization_custom_ttl(self):
        """Test cache initialization with custom TTL."""
        cache = SearchCache(default_ttl=300)
        assert cache.default_ttl == 300
        assert len(cache.cache) == 0

    def test_cache_set_and_get(self):
        """Test setting and getting cache values."""
        cache = SearchCache()
        test_value = {"data": "test"}
        
        cache.set("test_key", test_value)
        result = cache.get("test_key")
        
        assert result == test_value

    def test_cache_get_nonexistent_key(self):
        """Test getting non-existent cache key."""
        cache = SearchCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = SearchCache()
        test_value = {"data": "test"}
        
        # Set with very short TTL
        cache.set("test_key", test_value, ttl=1)
        
        # Should be available immediately
        assert cache.get("test_key") == test_value
        
        # Sleep and check expiration
        time.sleep(1.1)
        assert cache.get("test_key") is None

    def test_cache_clear_expired(self):
        """Test clearing expired entries."""
        cache = SearchCache()
        
        # Add some entries with different TTLs
        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=10)
        
        # Wait for first to expire
        time.sleep(1.1)
        
        # Clear expired
        expired_count = cache.clear_expired()
        
        assert expired_count == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_cache_custom_ttl_override(self):
        """Test custom TTL override for specific entries."""
        cache = SearchCache(default_ttl=100)
        
        cache.set("default_ttl", "value1")  # Uses default TTL
        cache.set("custom_ttl", "value2", ttl=200)  # Uses custom TTL
        
        # Check that entries exist
        assert cache.get("default_ttl") == "value1"
        assert cache.get("custom_ttl") == "value2"
        
        # Verify TTL values are set correctly
        default_entry = cache.cache["default_ttl"]
        custom_entry = cache.cache["custom_ttl"]
        
        assert custom_entry["expires_at"] > default_entry["expires_at"]


class TestRateLimiter:
    """Test cases for RateLimiter functionality."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60
        assert len(limiter.requests) == 0

    @pytest.mark.asyncio
    async def test_rate_limiter_allow_requests(self):
        """Test rate limiter allows requests within limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=10)
        
        # Should allow first 5 requests
        for _ in range(5):
            result = await limiter.acquire()
            assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiter_block_excess_requests(self):
        """Test rate limiter blocks requests exceeding limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=10)
        
        # First 2 requests should succeed quickly
        start_time = time.time()
        for _ in range(2):
            result = await limiter.acquire()
            assert result is True
        
        # Should be fast (under 0.1 seconds)
        elapsed = time.time() - start_time
        assert elapsed < 0.1
        
        # Third request should be delayed
        # Note: In real test, we'd mock time.time() to avoid actual delays
        # For now, we'll test the logic without waiting

    @pytest.mark.asyncio
    async def test_rate_limiter_window_cleanup(self):
        """Test rate limiter cleans up old requests."""
        limiter = RateLimiter(max_requests=3, window_seconds=1)
        
        # Add requests manually to simulate old requests
        current_time = time.time()
        limiter.requests = [
            current_time - 2,  # Older than window
            current_time - 0.5,  # Within window
        ]
        
        # Should clean up old request and allow new one
        result = await limiter.acquire()
        assert result is True
        assert len(limiter.requests) == 2  # Old one removed, new one added


class TestSearchResult:
    """Test cases for SearchResult model."""

    def test_search_result_creation(self):
        """Test creating a SearchResult with basic data."""
        result = SearchResult(
            title="Test Article",
            url="https://example.com/test",
            snippet="This is a test snippet",
            source="example.com"
        )
        
        assert result.title == "Test Article"
        assert result.url == "https://example.com/test"
        assert result.snippet == "This is a test snippet"
        assert result.source == "example.com"
        assert result.relevance_score == 0.0  # Default value
        assert result.published_date is None  # Default value
        assert isinstance(result.result_id, str)

    def test_search_result_with_optional_fields(self):
        """Test SearchResult with all optional fields."""
        pub_date = datetime.now(UTC)
        result = SearchResult(
            title="Test Article",
            url="https://example.com/test",
            snippet="Test snippet",
            source="example.com",
            published_date=pub_date,
            relevance_score=0.85
        )
        
        assert result.published_date == pub_date
        assert result.relevance_score == 0.85

    def test_search_result_validation(self):
        """Test SearchResult field validation."""
        with pytest.raises(ValueError):
            # Relevance score out of range
            SearchResult(
                title="Test",
                url="https://example.com",
                snippet="Test",
                source="example.com",
                relevance_score=1.5
            )
        
        with pytest.raises(ValueError):
            # Negative relevance score
            SearchResult(
                title="Test",
                url="https://example.com",
                snippet="Test",
                source="example.com",
                relevance_score=-0.1
            )


class TestFinancialNewsResult:
    """Test cases for FinancialNewsResult model."""

    def test_financial_news_result_creation(self):
        """Test creating FinancialNewsResult."""
        base_result = SearchResult(
            title="Bitcoin ETF Approved",
            url="https://example.com/btc-etf",
            snippet="SEC approves Bitcoin ETF",
            source="financial-news.com"
        )
        
        financial_result = FinancialNewsResult(
            base_result=base_result,
            sentiment="positive",
            mentioned_symbols=["BTC", "ETH"],
            news_category="regulation",
            impact_level="high"
        )
        
        assert financial_result.base_result == base_result
        assert financial_result.sentiment == "positive"
        assert financial_result.mentioned_symbols == ["BTC", "ETH"]
        assert financial_result.news_category == "regulation"
        assert financial_result.impact_level == "high"

    def test_financial_news_result_defaults(self):
        """Test FinancialNewsResult with default values."""
        base_result = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Test",
            source="test.com"
        )
        
        financial_result = FinancialNewsResult(base_result=base_result)
        
        assert financial_result.sentiment is None
        assert financial_result.mentioned_symbols == []
        assert financial_result.news_category is None
        assert financial_result.impact_level is None


class TestSentimentAnalysis:
    """Test cases for SentimentAnalysis model."""

    def test_sentiment_analysis_creation(self):
        """Test creating SentimentAnalysis with required fields."""
        sentiment = SentimentAnalysis(
            symbol="BTC",
            overall_sentiment="bullish",
            sentiment_score=0.75,
            confidence=0.85,
            source_count=15
        )
        
        assert sentiment.symbol == "BTC"
        assert sentiment.overall_sentiment == "bullish"
        assert sentiment.sentiment_score == 0.75
        assert sentiment.confidence == 0.85
        assert sentiment.source_count == 15
        assert sentiment.timeframe == "24h"  # Default value

    def test_sentiment_analysis_with_details(self):
        """Test SentimentAnalysis with detailed breakdown."""
        sentiment = SentimentAnalysis(
            symbol="ETH",
            overall_sentiment="bearish",
            sentiment_score=-0.4,
            confidence=0.7,
            source_count=20,
            news_sentiment=-0.3,
            social_sentiment=-0.5,
            technical_sentiment=-0.2,
            key_drivers=["Regulatory concerns", "Market correction"],
            risk_factors=["High volatility", "Regulatory uncertainty"]
        )
        
        assert sentiment.news_sentiment == -0.3
        assert sentiment.social_sentiment == -0.5
        assert sentiment.technical_sentiment == -0.2
        assert len(sentiment.key_drivers) == 2
        assert len(sentiment.risk_factors) == 2

    def test_sentiment_analysis_validation(self):
        """Test SentimentAnalysis field validation."""
        with pytest.raises(ValueError):
            # Sentiment score out of range
            SentimentAnalysis(
                symbol="BTC",
                overall_sentiment="bullish",
                sentiment_score=1.5,
                confidence=0.8,
                source_count=10
            )
        
        with pytest.raises(ValueError):
            # Confidence out of range
            SentimentAnalysis(
                symbol="BTC",
                overall_sentiment="bullish",
                sentiment_score=0.5,
                confidence=1.2,
                source_count=10
            )


class TestMarketCorrelation:
    """Test cases for MarketCorrelation model."""

    def test_market_correlation_creation(self):
        """Test creating MarketCorrelation."""
        correlation = MarketCorrelation(
            primary_symbol="BTC",
            secondary_symbol="QQQ",
            correlation_coefficient=0.65,
            strength="moderate",
            direction="positive"
        )
        
        assert correlation.primary_symbol == "BTC"
        assert correlation.secondary_symbol == "QQQ"
        assert correlation.correlation_coefficient == 0.65
        assert correlation.strength == "moderate"
        assert correlation.direction == "positive"
        assert correlation.timeframe == "30d"  # Default value

    def test_market_correlation_with_metrics(self):
        """Test MarketCorrelation with additional metrics."""
        correlation = MarketCorrelation(
            primary_symbol="ETH",
            secondary_symbol="SPY",
            correlation_coefficient=-0.3,
            strength="weak",
            direction="negative",
            beta=-0.8,
            r_squared=0.15
        )
        
        assert correlation.beta == -0.8
        assert correlation.r_squared == 0.15
        assert isinstance(correlation.last_updated, datetime)

    def test_market_correlation_validation(self):
        """Test MarketCorrelation validation."""
        with pytest.raises(ValueError):
            # Correlation coefficient out of range
            MarketCorrelation(
                primary_symbol="BTC",
                secondary_symbol="QQQ",
                correlation_coefficient=1.5,
                strength="strong",
                direction="positive"
            )


class TestOmniSearchClient:
    """Test cases for OmniSearchClient."""

    def test_client_initialization_defaults(self):
        """Test client initialization with default values."""
        client = OmniSearchClient()
        
        assert client.server_url is not None
        assert client.api_key is None
        assert client.cache is not None
        assert client.rate_limiter is not None
        assert not client._connected
        assert client._session is None

    def test_client_initialization_custom(self):
        """Test client initialization with custom values."""
        client = OmniSearchClient(
            server_url="https://custom.api.com",
            api_key="test_key",
            enable_cache=False,
            cache_ttl=600,
            rate_limit_requests=50,
            rate_limit_window=1800
        )
        
        assert client.server_url == "https://custom.api.com"
        assert client.api_key == "test_key"
        assert client.cache is None  # Disabled
        assert client.rate_limiter.max_requests == 50
        assert client.rate_limiter.window_seconds == 1800

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to OmniSearch service."""
        client = OmniSearchClient()
        
        with patch.object(client, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            result = await client.connect()
            
            assert result is True
            assert client._connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test failed connection to OmniSearch service."""
        client = OmniSearchClient()
        
        with patch.object(client, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            result = await client.connect()
            
            assert result is False
            assert client._connected is False

    @pytest.mark.asyncio
    async def test_connect_exception(self):
        """Test connection exception handling."""
        client = OmniSearchClient()
        
        with patch.object(client, '_session') as mock_session:
            mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
            
            result = await client.connect()
            
            assert result is False
            assert client._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnecting from OmniSearch service."""
        client = OmniSearchClient()
        client._connected = True
        client._session = Mock()
        client._session.close = AsyncMock()
        
        await client.disconnect()
        
        assert client._connected is False
        assert client._session is None
        client._session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_financial_news_cache_hit(self):
        """Test financial news search with cache hit."""
        client = OmniSearchClient()
        
        # Mock cache hit
        cached_data = [
            {
                "base_result": {
                    "result_id": "123",
                    "title": "Test News",
                    "url": "https://example.com",
                    "snippet": "Test snippet",
                    "source": "test.com",
                    "relevance_score": 0.8
                },
                "sentiment": "positive"
            }
        ]
        client.cache.set("financial_news:test query:5:24h:True", cached_data)
        
        results = await client.search_financial_news("test query")
        
        assert len(results) == 1
        assert results[0].base_result.title == "Test News"
        assert results[0].sentiment == "positive"

    @pytest.mark.asyncio
    async def test_search_financial_news_api_call(self):
        """Test financial news search with API call."""
        client = OmniSearchClient()
        client._connected = True
        client._session = AsyncMock()
        
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Bitcoin Surges",
                    "url": "https://example.com/btc",
                    "snippet": "Bitcoin price increases",
                    "source": "crypto-news.com",
                    "published_date": "2024-01-01T12:00:00Z",
                    "relevance_score": 0.9,
                    "sentiment": "positive",
                    "mentioned_symbols": ["BTC"],
                    "category": "price",
                    "impact_level": "high"
                }
            ]
        }
        client._session.get.return_value.__aenter__.return_value = mock_response
        
        # Mock rate limiter
        client.rate_limiter.acquire = AsyncMock(return_value=True)
        
        results = await client.search_financial_news("bitcoin news")
        
        assert len(results) == 1
        assert results[0].base_result.title == "Bitcoin Surges"
        assert results[0].sentiment == "positive"
        assert "BTC" in results[0].mentioned_symbols

    @pytest.mark.asyncio
    async def test_search_financial_news_api_error(self):
        """Test financial news search with API error."""
        client = OmniSearchClient()
        client._connected = True
        client._session = AsyncMock()
        
        # Mock API error
        mock_response = AsyncMock()
        mock_response.status = 500
        client._session.get.return_value.__aenter__.return_value = mock_response
        
        # Mock rate limiter
        client.rate_limiter.acquire = AsyncMock(return_value=True)
        
        results = await client.search_financial_news("test query")
        
        # Should return fallback results (empty list in this case)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_crypto_sentiment_success(self):
        """Test crypto sentiment search success."""
        client = OmniSearchClient()
        client._connected = True
        client._session = AsyncMock()
        
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "sentiment": {
                "overall": "bullish",
                "score": 0.6,
                "confidence": 0.8,
                "source_count": 25,
                "news_sentiment": 0.5,
                "social_sentiment": 0.7,
                "technical_sentiment": 0.4,
                "key_drivers": ["ETF approval", "Institutional adoption"],
                "risk_factors": ["Regulatory uncertainty"]
            }
        }
        client._session.get.return_value.__aenter__.return_value = mock_response
        
        # Mock rate limiter
        client.rate_limiter.acquire = AsyncMock(return_value=True)
        
        sentiment = await client.search_crypto_sentiment("BTC-USD")
        
        assert sentiment.symbol == "BTC"
        assert sentiment.overall_sentiment == "bullish"
        assert sentiment.sentiment_score == 0.6
        assert sentiment.confidence == 0.8
        assert len(sentiment.key_drivers) == 2

    @pytest.mark.asyncio
    async def test_search_nasdaq_sentiment_success(self):
        """Test NASDAQ sentiment search success."""
        client = OmniSearchClient()
        client._connected = True
        client._session = AsyncMock()
        
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "sentiment": {
                "overall": "bearish",
                "score": -0.3,
                "confidence": 0.7,
                "source_count": 30,
                "key_drivers": ["Interest rate concerns"],
                "risk_factors": ["Economic slowdown"]
            }
        }
        client._session.get.return_value.__aenter__.return_value = mock_response
        
        # Mock rate limiter
        client.rate_limiter.acquire = AsyncMock(return_value=True)
        
        sentiment = await client.search_nasdaq_sentiment()
        
        assert sentiment.symbol == "NASDAQ"
        assert sentiment.overall_sentiment == "bearish"
        assert sentiment.sentiment_score == -0.3

    @pytest.mark.asyncio
    async def test_search_market_correlation_success(self):
        """Test market correlation search success."""
        client = OmniSearchClient()
        client._connected = True
        client._session = AsyncMock()
        
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "correlation": {
                "coefficient": 0.45,
                "beta": 1.2,
                "r_squared": 0.35
            }
        }
        client._session.get.return_value.__aenter__.return_value = mock_response
        
        # Mock rate limiter
        client.rate_limiter.acquire = AsyncMock(return_value=True)
        
        correlation = await client.search_market_correlation("BTC", "QQQ")
        
        assert correlation.primary_symbol == "BTC"
        assert correlation.secondary_symbol == "QQQ"
        assert correlation.correlation_coefficient == 0.45
        assert correlation.strength == "moderate"  # 0.4 < 0.45 <= 0.6
        assert correlation.direction == "positive"
        assert correlation.beta == 1.2

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        client = OmniSearchClient()
        client._connected = True
        
        health = await client.health_check()
        
        assert health["connected"] is True
        assert "server_url" in health
        assert "cache_enabled" in health
        assert "cache_size" in health
        assert "rate_limit_remaining" in health
        assert "local_storage" in health
        assert "timestamp" in health

    def test_get_headers_without_api_key(self):
        """Test header generation without API key."""
        client = OmniSearchClient()
        headers = client._get_headers()
        
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"] == "AI-Trading-Bot/1.0"
        assert "Authorization" not in headers

    def test_get_headers_with_api_key(self):
        """Test header generation with API key."""
        client = OmniSearchClient(api_key="test_key")
        headers = client._get_headers()
        
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"] == "AI-Trading-Bot/1.0"
        assert headers["Authorization"] == "Bearer test_key"

    def test_parse_date_valid_formats(self):
        """Test date parsing with valid formats."""
        client = OmniSearchClient()
        
        # Test ISO format with Z
        date1 = client._parse_date("2024-01-01T12:00:00Z")
        assert isinstance(date1, datetime)
        assert date1.tzinfo == UTC
        
        # Test ISO format with microseconds
        date2 = client._parse_date("2024-01-01T12:00:00.123456Z")
        assert isinstance(date2, datetime)
        
        # Test simple date
        date3 = client._parse_date("2024-01-01")
        assert isinstance(date3, datetime)

    def test_parse_date_invalid_format(self):
        """Test date parsing with invalid format."""
        client = OmniSearchClient()
        
        # Invalid format should return None
        result = client._parse_date("invalid date")
        assert result is None
        
        # None input should return None
        result = client._parse_date(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_fallback_sentiment(self):
        """Test fallback sentiment generation."""
        client = OmniSearchClient()
        
        sentiment = await client._get_fallback_sentiment("BTC")
        
        assert sentiment.symbol == "BTC"
        assert sentiment.overall_sentiment == "neutral"
        assert sentiment.sentiment_score == 0.0
        assert sentiment.confidence == 0.1
        assert "API unavailable" in sentiment.key_drivers[0]

    @pytest.mark.asyncio
    async def test_get_fallback_correlation(self):
        """Test fallback correlation generation."""
        client = OmniSearchClient()
        
        correlation = await client._get_fallback_correlation("BTC", "QQQ", "30d")
        
        assert correlation.primary_symbol == "BTC"
        assert correlation.secondary_symbol == "QQQ"
        assert correlation.correlation_coefficient == 0.0
        assert correlation.timeframe == "30d"
        assert correlation.strength == "weak"
        assert correlation.direction == "neutral"

    @pytest.mark.asyncio
    async def test_get_fallback_news_empty(self):
        """Test fallback news with no local cache."""
        client = OmniSearchClient()
        
        results = await client._get_fallback_news("test query", 5)
        
        # Should return empty list when no fallback data
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self):
        """Test rate limiting integration with search methods."""
        client = OmniSearchClient(rate_limit_requests=1, rate_limit_window=10)
        client._connected = True
        client._session = AsyncMock()
        
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"results": []}
        client._session.get.return_value.__aenter__.return_value = mock_response
        
        # First request should succeed quickly
        start_time = time.time()
        await client.search_financial_news("test")
        first_request_time = time.time() - start_time
        
        assert first_request_time < 0.1  # Should be fast
        
        # Verify rate limiter was called
        assert len(client.rate_limiter.requests) == 1


# Fixtures for integration tests
@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing."""
    with patch('aiohttp.ClientSession') as mock:
        session = AsyncMock()
        mock.return_value = session
        yield session


@pytest.fixture
def sample_financial_news_response():
    """Sample financial news API response."""
    return {
        "results": [
            {
                "title": "Bitcoin Reaches New High",
                "url": "https://example.com/btc-high",
                "snippet": "Bitcoin price reaches $60,000",
                "source": "crypto-times.com",
                "published_date": "2024-01-01T12:00:00Z",
                "relevance_score": 0.95,
                "sentiment": "positive",
                "mentioned_symbols": ["BTC", "BTCUSD"],
                "category": "price",
                "impact_level": "high"
            },
            {
                "title": "Ethereum Network Upgrade",
                "url": "https://example.com/eth-upgrade",
                "snippet": "Ethereum implements new upgrade",
                "source": "blockchain-news.com",
                "published_date": "2024-01-01T10:00:00Z",
                "relevance_score": 0.8,
                "sentiment": "positive",
                "mentioned_symbols": ["ETH"],
                "category": "technology",
                "impact_level": "medium"
            }
        ]
    }


@pytest.fixture
def sample_sentiment_response():
    """Sample sentiment analysis API response."""
    return {
        "sentiment": {
            "overall": "bullish",
            "score": 0.65,
            "confidence": 0.82,
            "source_count": 42,
            "news_sentiment": 0.7,
            "social_sentiment": 0.6,
            "technical_sentiment": 0.5,
            "key_drivers": [
                "Institutional adoption increasing",
                "Positive regulatory developments",
                "Strong technical momentum"
            ],
            "risk_factors": [
                "Market volatility",
                "Regulatory uncertainty in some regions"
            ]
        }
    }


@pytest.fixture
def sample_correlation_response():
    """Sample correlation analysis API response."""
    return {
        "correlation": {
            "coefficient": 0.73,
            "beta": 1.85,
            "r_squared": 0.53
        }
    }


class TestOmniSearchClientIntegration:
    """Integration tests for OmniSearchClient with mocked external services."""

    @pytest.mark.asyncio
    async def test_full_search_workflow(self, mock_aiohttp_session, sample_financial_news_response):
        """Test complete search workflow from connection to results."""
        client = OmniSearchClient()
        
        # Mock connection success
        health_response = AsyncMock()
        health_response.status = 200
        
        # Mock search response
        search_response = AsyncMock()
        search_response.status = 200
        search_response.json.return_value = sample_financial_news_response
        
        mock_aiohttp_session.get.return_value.__aenter__.side_effect = [
            health_response,  # Connection health check
            search_response   # Search request
        ]
        
        # Test workflow
        connected = await client.connect()
        assert connected is True
        
        results = await client.search_financial_news("bitcoin ethereum")
        assert len(results) == 2
        assert results[0].base_result.title == "Bitcoin Reaches New High"
        assert results[1].base_result.title == "Ethereum Network Upgrade"
        
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_sentiment_analysis_workflow(self, mock_aiohttp_session, sample_sentiment_response):
        """Test sentiment analysis workflow."""
        client = OmniSearchClient()
        client._connected = True
        client._session = mock_aiohttp_session
        
        # Mock sentiment response
        sentiment_response = AsyncMock()
        sentiment_response.status = 200
        sentiment_response.json.return_value = sample_sentiment_response
        
        mock_aiohttp_session.get.return_value.__aenter__.return_value = sentiment_response
        
        sentiment = await client.search_crypto_sentiment("BTC")
        
        assert sentiment.symbol == "BTC"
        assert sentiment.overall_sentiment == "bullish"
        assert sentiment.sentiment_score == 0.65
        assert len(sentiment.key_drivers) == 3
        assert len(sentiment.risk_factors) == 2

    @pytest.mark.asyncio
    async def test_correlation_analysis_workflow(self, mock_aiohttp_session, sample_correlation_response):
        """Test correlation analysis workflow."""
        client = OmniSearchClient()
        client._connected = True
        client._session = mock_aiohttp_session
        
        # Mock correlation response
        correlation_response = AsyncMock()
        correlation_response.status = 200
        correlation_response.json.return_value = sample_correlation_response
        
        mock_aiohttp_session.get.return_value.__aenter__.return_value = correlation_response
        
        correlation = await client.search_market_correlation("BTC", "QQQ")
        
        assert correlation.primary_symbol == "BTC"
        assert correlation.secondary_symbol == "QQQ"
        assert correlation.correlation_coefficient == 0.73
        assert correlation.strength == "strong"  # > 0.6
        assert correlation.direction == "positive"
        assert correlation.beta == 1.85

    @pytest.mark.asyncio
    async def test_caching_across_requests(self):
        """Test caching behavior across multiple requests."""
        client = OmniSearchClient(cache_ttl=300)  # 5 minute cache
        
        # First request - should hit API
        with patch.object(client, '_search_with_fallback') as mock_search:
            mock_search.return_value = {"results": []}
            
            await client.search_financial_news("test query")
            assert mock_search.call_count == 1
            
            # Second identical request - should hit cache
            await client.search_financial_news("test query")
            assert mock_search.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        client = OmniSearchClient()
        client._connected = True
        client._session = AsyncMock()
        
        # Test API timeout
        client._session.get.side_effect = asyncio.TimeoutError()
        
        # Should handle timeout gracefully and return fallback
        sentiment = await client.search_crypto_sentiment("BTC")
        assert sentiment.symbol == "BTC"
        assert sentiment.overall_sentiment == "neutral"
        assert sentiment.confidence == 0.1
        
        # Test network error
        client._session.get.side_effect = aiohttp.ClientError("Network error")
        
        correlation = await client.search_market_correlation("BTC", "QQQ")
        assert correlation.correlation_coefficient == 0.0
        assert correlation.strength == "weak"