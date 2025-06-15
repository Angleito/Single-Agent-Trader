#!/usr/bin/env python3
"""
Test script for OmniSearch MCP client.
This script tests the functionality without requiring external dependencies.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.mcp.omnisearch_client import (
    OmniSearchClient,
    search_crypto_news,
    get_market_sentiment,
    analyze_correlation
)


async def test_omnisearch_client():
    """Test the OmniSearch client functionality."""
    print("🔍 Testing OmniSearch MCP Client...")
    
    # Test client initialization
    client = OmniSearchClient(
        api_key="test_key",
        base_url="https://api.example.com/v1",
        cache_ttl=60
    )
    
    print(f"✅ Client initialized with base URL: {client.base_url}")
    
    # Test connection (will fail but should handle gracefully)
    print("\n📡 Testing connection...")
    connected = await client.connect()
    print(f"Connection result: {connected} (expected False for mock API)")
    
    # Test financial news search (will use fallback data)
    print("\n📰 Testing financial news search...")
    news = await client.search_financial_news("Bitcoin BTC", limit=3)
    print(f"Found {len(news.results)} news articles")
    for i, article in enumerate(news.results, 1):
        print(f"  {i}. {article.title}")
        print(f"     Source: {article.source} | Score: {article.relevance_score:.2f}")
    
    # Test crypto sentiment analysis
    print("\n💭 Testing crypto sentiment analysis...")
    sentiment = await client.search_crypto_sentiment("BTC")
    print(f"BTC Sentiment: {sentiment.sentiment_label} (score: {sentiment.sentiment_score:.2f})")
    print(f"Confidence: {sentiment.confidence:.2f} | Themes: {', '.join(sentiment.key_themes)}")
    
    # Test NASDAQ sentiment
    print("\n📈 Testing NASDAQ sentiment analysis...")
    nasdaq_sentiment = await client.search_nasdaq_sentiment()
    print(f"NASDAQ Sentiment: {nasdaq_sentiment.sentiment_label} (score: {nasdaq_sentiment.sentiment_score:.2f})")
    
    # Test market correlation
    print("\n🔗 Testing market correlation analysis...")
    correlation = await client.search_market_correlation("BTC", "QQQ", "30d")
    print(f"BTC vs QQQ Correlation: {correlation.correlation_coefficient:.3f} ({correlation.correlation_strength})")
    print(f"Timeframe: {correlation.timeframe} | P-value: {correlation.p_value}")
    
    # Test cache functionality
    print("\n💾 Testing cache functionality...")
    cache_stats = await client.get_cache_stats()
    print(f"Cache entries: {cache_stats['total_entries']}")
    
    # Test cleanup
    cleaned = await client.cleanup_cache()
    print(f"Cleaned up {cleaned} expired cache entries")
    
    # Close client
    await client.disconnect()
    print("\n✅ All tests completed successfully!")


async def test_convenience_functions():
    """Test the convenience functions."""
    print("\n🚀 Testing convenience functions...")
    
    # Test crypto news search
    print("Searching crypto news...")
    news = await search_crypto_news("ETH", limit=2)
    print(f"Found {len(news.results)} ETH news articles")
    
    # Test market sentiment
    print("Getting market sentiment...")
    sentiment = await get_market_sentiment("ETH")
    print(f"ETH sentiment: {sentiment.sentiment_label} ({sentiment.sentiment_score:.2f})")
    
    # Test correlation analysis
    print("Analyzing correlation...")
    correlation = await analyze_correlation("ETH", "SPY")
    print(f"ETH vs SPY correlation: {correlation.correlation_coefficient:.3f}")


async def main():
    """Main test function."""
    try:
        await test_omnisearch_client()
        await test_convenience_functions()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)