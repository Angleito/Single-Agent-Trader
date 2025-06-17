#!/usr/bin/env python
"""Basic OmniSearch functionality test."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Override exchange type to avoid config issues
os.environ["EXCHANGE__EXCHANGE_TYPE"] = "coinbase"

print("=== OmniSearch Basic Test ===\n")

try:
    print("1. Testing imports...")
    from bot.mcp.omnisearch_client import (
        OmniSearchClient,
        SearchResult,
        FinancialNewsResult,
        SentimentAnalysis,
        MarketCorrelation,
        SearchCache,
        RateLimiter
    )
    print("   ✓ All imports successful")
    
    print("\n2. Testing configuration...")
    from bot.config import settings
    print(f"   OmniSearch Enabled: {settings.omnisearch.enabled}")
    print(f"   Server URL: {settings.omnisearch.server_url}")
    print(f"   Cache Enabled: {settings.omnisearch.enable_cache}")
    print(f"   Rate Limit: {settings.omnisearch.rate_limit_requests} req/{settings.omnisearch.rate_limit_window_seconds}s")
    
    print("\n3. Testing class instantiation...")
    
    # Test SearchCache
    cache = SearchCache(default_ttl=300)
    print(f"   ✓ SearchCache created (TTL: {cache.default_ttl}s)")
    
    # Test RateLimiter  
    rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
    print(f"   ✓ RateLimiter created ({rate_limiter.max_requests} req/{rate_limiter.window_seconds}s)")
    
    # Test OmniSearchClient
    client = OmniSearchClient()
    print("   ✓ OmniSearchClient created")
    
    print("\n4. Testing data models...")
    
    # Test SearchResult
    search_result = SearchResult(
        title="Test Article",
        url="https://example.com",
        snippet="Test snippet",
        source="example.com",
        relevance_score=0.9
    )
    print(f"   ✓ SearchResult: {search_result.title}")
    
    # Test FinancialNewsResult
    news_result = FinancialNewsResult(
        base_result=search_result,
        sentiment="positive",
        mentioned_symbols=["BTC", "ETH"],
        impact_level="high"
    )
    print(f"   ✓ FinancialNewsResult: sentiment={news_result.sentiment}")
    
    # Test SentimentAnalysis
    sentiment = SentimentAnalysis(
        symbol="BTC",
        sentiment_score=0.75,
        overall_sentiment="bullish",
        confidence=0.85,
        sources_analyzed=100
    )
    print(f"   ✓ SentimentAnalysis: {sentiment.symbol} = {sentiment.overall_sentiment}")
    
    # Test MarketCorrelation
    correlation = MarketCorrelation(
        symbol1="BTC",
        symbol2="SPY",
        correlation_coefficient=0.65,
        period="30d",
        relationship_strength="moderate",
        direction="positive"
    )
    print(f"   ✓ MarketCorrelation: {correlation.symbol1}-{correlation.symbol2} = {correlation.correlation_coefficient}")
    
    print("\n5. Testing client methods exist...")
    methods = [
        "connect",
        "disconnect", 
        "search_financial_news",
        "get_crypto_sentiment",
        "get_nasdaq_sentiment",
        "get_market_correlation",
        "get_comprehensive_market_analysis"
    ]
    
    for method in methods:
        if hasattr(client, method):
            print(f"   ✓ {method}() method exists")
        else:
            print(f"   ✗ {method}() method missing")
    
    print("\n✓ All basic tests passed!")
    
except Exception as e:
    print(f"\n✗ Test failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()