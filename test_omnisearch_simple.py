#!/usr/bin/env python
"""Simple OmniSearch test script."""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Override exchange type to avoid config issues
os.environ["EXCHANGE__EXCHANGE_TYPE"] = "coinbase"

from bot.mcp.omnisearch_client import OmniSearchClient
from bot.config import settings


async def test_omnisearch():
    """Test OmniSearch functionality."""
    print(f"\n=== OmniSearch Test ===")
    print(f"Enabled: {settings.omnisearch.enabled}")
    print(f"Server URL: {settings.omnisearch.server_url}")
    print(f"API Key: {'[SET]' if settings.omnisearch.api_key != 'your_omnisearch_api_key_here' else '[NOT SET]'}")
    
    if not settings.omnisearch.enabled:
        print("\nOmniSearch is disabled. Enable it by setting OMNISEARCH__ENABLED=true")
        return
    
    # Create client
    client = OmniSearchClient()
    
    print("\n1. Testing connection...")
    is_connected = await client.connect()
    print(f"   Connected: {is_connected}")
    
    if not is_connected:
        print("   Failed to connect. Check your API key and server URL.")
        return
    
    print("\n2. Testing financial news search...")
    try:
        news_results = await client.search_financial_news("BTC Bitcoin cryptocurrency", max_results=3)
        print(f"   Found {len(news_results)} news articles")
        for idx, result in enumerate(news_results):
            print(f"   [{idx+1}] {result.base_result.title[:60]}...")
            if result.sentiment:
                print(f"       Sentiment: {result.sentiment}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Testing crypto sentiment analysis...")
    try:
        sentiment = await client.get_crypto_sentiment("BTC")
        print(f"   BTC Sentiment Score: {sentiment.sentiment_score}")
        print(f"   Overall Sentiment: {sentiment.overall_sentiment}")
        print(f"   Confidence: {sentiment.confidence}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. Testing market correlation...")
    try:
        correlation = await client.get_market_correlation("BTC", "SPY")
        print(f"   BTC-SPY Correlation: {correlation.correlation_coefficient}")
        print(f"   Relationship Strength: {correlation.relationship_strength}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n5. Testing comprehensive market analysis...")
    try:
        analysis = await client.get_comprehensive_market_analysis("BTC")
        print(f"   News count: {len(analysis.recent_news)}")
        print(f"   Sentiment score: {analysis.sentiment_analysis.sentiment_score}")
        print(f"   Market correlations: {len(analysis.market_correlations)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Disconnect
    await client.disconnect()
    print("\n✓ Test completed")


if __name__ == "__main__":
    asyncio.run(test_omnisearch())