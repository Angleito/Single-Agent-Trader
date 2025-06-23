#!/usr/bin/env python3
"""
Test script to verify OmniSearch client fixes.

Tests:
1. Generic search method implementation
2. Graceful degradation when service unavailable
3. Error handling for missing methods
"""

import asyncio
import logging
from bot.mcp.omnisearch_client import OmniSearchClient
from bot.mcp.mcp_omnisearch_client import MCPOmniSearchClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_omnisearch_client():
    """Test the standard OmniSearchClient."""
    logger.info("Testing OmniSearchClient...")
    
    # Create client without server URL to simulate unavailable service
    client = OmniSearchClient(server_url="http://localhost:9999/nonexistent")
    
    # Test connection (should fail gracefully)
    logger.info("Testing connection...")
    connected = await client.connect()
    logger.info(f"Connection result: {connected}")
    
    # Test generic search method
    logger.info("Testing generic search method...")
    try:
        results = await client.search("test query", limit=3)
        logger.info(f"Search results: {results}")
        if results is None:
            logger.info("Search returned None (expected when service unavailable)")
        else:
            logger.info(f"Got {len(results)} results")
    except AttributeError as e:
        logger.error(f"AttributeError: {e}")
        raise
    except Exception as e:
        logger.warning(f"Search failed with: {e}")
    
    # Test specific search methods with graceful degradation
    logger.info("Testing crypto sentiment...")
    sentiment = await client.search_crypto_sentiment("BTC")
    logger.info(f"Sentiment: {sentiment.overall_sentiment} (score: {sentiment.sentiment_score})")
    
    # Test health check
    logger.info("Testing health check...")
    health = await client.health_check()
    logger.info(f"Health status: {health}")
    
    await client.disconnect()
    logger.info("OmniSearchClient test completed")


async def test_mcp_omnisearch_client():
    """Test the MCPOmniSearchClient."""
    logger.info("\nTesting MCPOmniSearchClient...")
    
    # Create client with invalid server path
    client = MCPOmniSearchClient(server_path="/nonexistent/path/to/server.js")
    
    # Test connection (should fail gracefully)
    logger.info("Testing connection...")
    connected = await client.connect()
    logger.info(f"Connection result: {connected}")
    
    # Test generic search method
    logger.info("Testing generic search method...")
    try:
        results = await client.search("test query", limit=3)
        logger.info(f"Search results: {results}")
        if results is None:
            logger.info("Search returned None (expected when service unavailable)")
        else:
            logger.info(f"Got {len(results)} results")
    except AttributeError as e:
        logger.error(f"AttributeError: {e}")
        raise
    except Exception as e:
        logger.warning(f"Search failed with: {e}")
    
    # Test specific methods with graceful degradation
    logger.info("Testing financial news search...")
    news = await client.search_financial_news("Bitcoin", limit=2)
    logger.info(f"News results: {len(news)} items")
    
    # Test health check
    logger.info("Testing health check...")
    health = await client.health_check()
    logger.info(f"Health status: {health}")
    
    await client.disconnect()
    logger.info("MCPOmniSearchClient test completed")


async def test_service_startup_integration():
    """Test the service startup integration."""
    logger.info("\nTesting service startup integration...")
    
    from bot.utils.service_startup import ServiceStartupManager
    from bot.config import Settings
    
    # Create settings with OmniSearch enabled
    settings = Settings()
    if hasattr(settings, 'omnisearch'):
        settings.omnisearch.enabled = True
    
    # Test service startup
    manager = ServiceStartupManager(settings)
    
    try:
        # Test OmniSearch startup with timeout
        logger.info("Testing OmniSearch startup with timeout...")
        omnisearch = await manager._start_omnisearch(timeout=5.0)
        
        if omnisearch:
            logger.info("OmniSearch client created successfully")
            # Test search method
            results = await omnisearch.search("test", limit=1)
            logger.info(f"Test search result: {results}")
        else:
            logger.info("OmniSearch client creation returned None")
            
    except Exception as e:
        logger.warning(f"Service startup test failed: {e}")
    
    logger.info("Service startup integration test completed")


async def main():
    """Run all tests."""
    logger.info("Starting OmniSearch fix verification tests...")
    
    try:
        await test_omnisearch_client()
    except Exception as e:
        logger.error(f"OmniSearchClient test failed: {e}")
    
    try:
        await test_mcp_omnisearch_client()
    except Exception as e:
        logger.error(f"MCPOmniSearchClient test failed: {e}")
    
    try:
        await test_service_startup_integration()
    except Exception as e:
        logger.error(f"Service startup integration test failed: {e}")
    
    logger.info("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())