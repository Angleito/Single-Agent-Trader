#!/usr/bin/env python
"""Test OmniSearch with mock server."""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Override exchange type to avoid config issues
os.environ["EXCHANGE__EXCHANGE_TYPE"] = "coinbase"

from bot.mcp.omnisearch_client import (
    ComprehensiveMarketAnalysis,
    FinancialNewsResult,
    MarketCorrelation,
    OmniSearchClient,
    SearchResult,
    SentimentAnalysis,
)


async def test_with_mocked_api():
    """Test OmniSearch functionality with mocked API responses."""
    print("\n=== OmniSearch Mock Test ===")
    
    # Create client
    client = OmniSearchClient()
    
    # Mock the aiohttp session
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"content-type": "application/json"}
    
    with patch.object(client.session, 'get', new_callable=AsyncMock) as mock_get:
        # Test 1: Mock connection test
        mock_response.json = AsyncMock(return_value={"status": "ok", "version": "1.0"})
        mock_get.return_value.__aenter__.return_value = mock_response
        
        print("\n1. Testing connection...")
        connected = await client.connect()
        print(f"   Connected: {connected}")
        
        # Test 2: Mock financial news search
        print("\n2. Testing financial news search...")
        news_data = {
            "results": [
                {
                    "title": "Bitcoin Surges Past $50,000 on ETF Optimism",
                    "url": "https://example.com/btc-surge",
                    "snippet": "Bitcoin rallied to a three-month high as investors bet on imminent ETF approval...",
                    "source": "CoinDesk",
                    "published_date": datetime.utcnow().isoformat(),
                    "relevance_score": 0.95,
                    "sentiment": "positive",
                    "mentioned_symbols": ["BTC", "ETH"],
                    "news_category": "market_movement",
                    "impact_level": "high"
                },
                {
                    "title": "Federal Reserve Signals Potential Rate Cuts",
                    "url": "https://example.com/fed-rates",
                    "snippet": "Fed Chair Powell hints at possible rate cuts if inflation continues to moderate...",
                    "source": "Reuters",
                    "published_date": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    "relevance_score": 0.88,
                    "sentiment": "positive",
                    "mentioned_symbols": [],
                    "news_category": "monetary_policy",
                    "impact_level": "high"
                }
            ]
        }
        mock_response.json = AsyncMock(return_value=news_data)
        
        news_results = await client.search_financial_news("Bitcoin ETF", max_results=5)
        print(f"   Found {len(news_results)} news articles:")
        for idx, news in enumerate(news_results):
            print(f"   [{idx+1}] {news.base_result.title}")
            print(f"       Sentiment: {news.sentiment}, Impact: {news.impact_level}")
        
        # Test 3: Mock crypto sentiment
        print("\n3. Testing crypto sentiment analysis...")
        sentiment_data = {
            "symbol": "BTC",
            "sentiment_score": 0.72,
            "overall_sentiment": "bullish",
            "confidence": 0.86,
            "sources_analyzed": 142,
            "key_themes": ["ETF", "institutional", "halving"],
            "bullish_indicators": [
                "ETF approval momentum",
                "Institutional accumulation",
                "Technical breakout"
            ],
            "bearish_indicators": [
                "Overbought RSI",
                "Resistance at $52K"
            ]
        }
        mock_response.json = AsyncMock(return_value=sentiment_data)
        
        sentiment = await client.get_crypto_sentiment("BTC")
        print(f"   BTC Sentiment Score: {sentiment.sentiment_score}")
        print(f"   Overall Sentiment: {sentiment.overall_sentiment}")
        print(f"   Confidence: {sentiment.confidence}")
        print(f"   Key Themes: {', '.join(sentiment.key_themes[:3])}")
        
        # Test 4: Mock market correlation
        print("\n4. Testing market correlation...")
        correlation_data = {
            "symbol1": "BTC",
            "symbol2": "SPY",
            "correlation_coefficient": 0.68,
            "period": "30d",
            "relationship_strength": "moderate",
            "direction": "positive",
            "p_value": 0.001,
            "interpretation": "Bitcoin shows moderate positive correlation with S&P 500"
        }
        mock_response.json = AsyncMock(return_value=correlation_data)
        
        correlation = await client.get_market_correlation("BTC", "SPY")
        print(f"   BTC-SPY Correlation: {correlation.correlation_coefficient}")
        print(f"   Relationship: {correlation.relationship_strength} {correlation.direction}")
        print(f"   Interpretation: {correlation.interpretation}")
        
        # Test 5: Mock comprehensive analysis
        print("\n5. Testing comprehensive market analysis...")
        analysis_data = {
            "symbol": "BTC",
            "timestamp": datetime.utcnow().isoformat(),
            "recent_news": news_data["results"],
            "sentiment_analysis": sentiment_data,
            "market_correlations": [correlation_data],
            "summary": {
                "market_condition": "bullish",
                "key_factors": [
                    "Strong positive sentiment driven by ETF optimism",
                    "Moderate correlation with traditional markets",
                    "Technical indicators suggesting continued momentum"
                ],
                "risk_factors": [
                    "Overbought conditions on shorter timeframes",
                    "Potential resistance at psychological levels"
                ]
            }
        }
        mock_response.json = AsyncMock(return_value=analysis_data)
        
        analysis = await client.get_comprehensive_market_analysis("BTC")
        print(f"   Market Condition: {analysis.summary['market_condition']}")
        print(f"   News Articles: {len(analysis.recent_news)}")
        print(f"   Sentiment Score: {analysis.sentiment_analysis.sentiment_score}")
        print(f"   Correlations Analyzed: {len(analysis.market_correlations)}")
        print("\n   Key Factors:")
        for factor in analysis.summary['key_factors'][:2]:
            print(f"   - {factor}")
        
        # Test caching
        print("\n6. Testing cache functionality...")
        # First call should hit the API
        await client.search_financial_news("Bitcoin", max_results=3)
        print("   First call completed (should hit API)")
        
        # Second call should use cache
        cached_results = await client.search_financial_news("Bitcoin", max_results=3)
        print(f"   Second call completed (should use cache)")
        print(f"   Cache contains {len(client._cache.cache)} entries")
        
        # Test rate limiting
        print("\n7. Testing rate limiter...")
        print(f"   Request count: {client._rate_limiter.request_count}")
        print(f"   Rate limit: {client._rate_limiter.max_requests} requests per {client._rate_limiter.window_seconds}s")
    
    await client.disconnect()
    print("\n✓ All mock tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_with_mocked_api())