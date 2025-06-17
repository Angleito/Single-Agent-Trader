#!/usr/bin/env python3
"""
Test script to validate MCP-OmniSearch integration.
"""

import asyncio
import sys
import os
sys.path.append('/Users/angel/Documents/Projects/cursorprod')

from bot.mcp.mcp_omnisearch_client import MCPOmniSearchClient


async def test_mcp_omnisearch():
    """Test the MCP-OmniSearch client functionality."""
    print("🔍 Testing MCP-OmniSearch Client...")
    
    # Use the Docker container's node server path
    client = MCPOmniSearchClient(
        server_path="/app/dist/index.js",
        enable_cache=True,
        cache_ttl=900
    )
    
    try:
        print("📡 Attempting to connect to MCP server...")
        connected = await client.connect()
        
        if not connected:
            print("❌ Failed to connect to MCP server")
            return False
            
        print("✅ Successfully connected to MCP server!")
        
        # Test health check
        print("\n🔍 Testing health check...")
        health = await client.health_check()
        print(f"Health status: {health}")
        
        # Test financial news search
        print("\n📰 Testing financial news search...")
        news_results = await client.search_financial_news(
            "Bitcoin ETF approval", 
            limit=2,
            timeframe="24h"
        )
        print(f"Found {len(news_results)} news results")
        for result in news_results:
            print(f"  - {result.base_result.title}")
        
        # Test crypto sentiment
        print("\n📊 Testing crypto sentiment analysis...")
        btc_sentiment = await client.search_crypto_sentiment("BTC-USD")
        print(f"BTC sentiment: {btc_sentiment.overall_sentiment} (score: {btc_sentiment.sentiment_score:.2f})")
        
        # Test NASDAQ sentiment
        print("\n📈 Testing NASDAQ sentiment analysis...")
        nasdaq_sentiment = await client.search_nasdaq_sentiment()
        print(f"NASDAQ sentiment: {nasdaq_sentiment.overall_sentiment} (score: {nasdaq_sentiment.sentiment_score:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
        
    finally:
        print("\n🔌 Disconnecting...")
        await client.disconnect()


async def test_fallback_mode():
    """Test the fallback mode functionality."""
    print("\n🛡️ Testing fallback mode...")
    
    # Create client that will fail to connect
    client = MCPOmniSearchClient(
        server_path="/nonexistent/path",
        enable_cache=True,
        cache_ttl=900
    )
    
    try:
        # This should fail and trigger fallback
        connected = await client.connect()
        if not connected:
            print("✅ Fallback mode activated as expected")
            
            # Test fallback sentiment
            fallback_sentiment = client._get_fallback_sentiment("BTC")
            print(f"Fallback BTC sentiment: {fallback_sentiment.overall_sentiment}")
            
            return True
            
    except Exception as e:
        print(f"Error in fallback test: {e}")
        return False
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    print("🚀 Starting MCP-OmniSearch Integration Tests\n")
    
    async def run_tests():
        # Test 1: MCP Client functionality
        mcp_success = await test_mcp_omnisearch()
        
        # Test 2: Fallback mode
        fallback_success = await test_fallback_mode()
        
        print(f"\n📋 Test Results:")
        print(f"  MCP Integration: {'✅ PASS' if mcp_success else '❌ FAIL'}")
        print(f"  Fallback Mode: {'✅ PASS' if fallback_success else '❌ FAIL'}")
        
        if mcp_success and fallback_success:
            print("\n🎉 All tests passed! MCP-OmniSearch integration is working correctly.")
            return 0
        else:
            print("\n⚠️ Some tests failed. Check the output above for details.")
            return 1
    
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)