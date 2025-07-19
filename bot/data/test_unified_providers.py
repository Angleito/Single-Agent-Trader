"""Test script to verify the unified market data provider system works correctly."""

import asyncio
import logging
from datetime import datetime, UTC

from bot.data import create_market_data_provider, MarketDataProvider
from bot.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_factory_creation():
    """Test that the factory creates the correct provider based on exchange type."""
    logger.info("Testing factory creation...")
    
    # Test Coinbase provider
    coinbase_provider = create_market_data_provider(
        exchange_type="coinbase",
        symbol="BTC-USD",
        interval="5m"
    )
    assert coinbase_provider.__class__.__name__ == "CoinbaseMarketDataProvider"
    logger.info("✓ Coinbase provider created successfully")
    
    # Test Bluefin provider
    bluefin_provider = create_market_data_provider(
        exchange_type="bluefin", 
        symbol="ETH-PERP",
        interval="1m"
    )
    assert bluefin_provider.__class__.__name__ == "BluefinMarketDataProvider"
    logger.info("✓ Bluefin provider created successfully")
    
    # Test backward compatibility
    # This should create provider based on settings.exchange.exchange_type
    default_provider = MarketDataProvider()
    logger.info(f"✓ Default provider created: {default_provider.__class__.__name__}")


async def test_common_functionality():
    """Test that common functionality works for both providers."""
    logger.info("\nTesting common functionality...")
    
    # Create both providers
    providers = {
        "coinbase": create_market_data_provider("coinbase", "BTC-USD", "1h"),
        "bluefin": create_market_data_provider("bluefin", "SUI-PERP", "1h")
    }
    
    for exchange, provider in providers.items():
        logger.info(f"\nTesting {exchange} provider...")
        
        # Test cache operations
        provider.clear_cache()
        assert len(provider._ohlcv_cache) == 0
        logger.info(f"  ✓ Cache cleared")
        
        # Test subscriber management
        def dummy_callback(data):
            pass
            
        provider.subscribe_to_updates(dummy_callback)
        assert len(provider._subscribers) == 1
        provider.unsubscribe_from_updates(dummy_callback) 
        assert len(provider._subscribers) == 0
        logger.info(f"  ✓ Subscriber management works")
        
        # Test interval conversion
        assert provider._interval_to_seconds("1m") == 60
        assert provider._interval_to_seconds("5m") == 300
        assert provider._interval_to_seconds("1h") == 3600
        assert provider._interval_to_seconds("30s") == 30
        logger.info(f"  ✓ Interval conversion works")
        
        # Test data status
        status = provider.get_data_status()
        assert isinstance(status, dict)
        assert "symbol" in status
        assert "interval" in status
        logger.info(f"  ✓ Data status works")


async def test_exchange_specific_features():
    """Test exchange-specific features."""
    logger.info("\nTesting exchange-specific features...")
    
    # Test Coinbase symbol conversion
    from bot.data.providers.coinbase_provider import CoinbaseMarketDataProvider
    cb_provider = CoinbaseMarketDataProvider("BTC-241227", "1h")
    assert cb_provider._data_symbol == "BTC-USD"  # Futures converted to spot
    logger.info("✓ Coinbase futures symbol conversion works")
    
    # Test Bluefin symbol conversion
    from bot.data.providers.bluefin_provider import BluefinMarketDataProvider
    bf_provider = BluefinMarketDataProvider("BTC-USD", "1h")
    assert bf_provider.symbol == "BTC-PERP"  # USD converted to PERP
    logger.info("✓ Bluefin symbol conversion works")
    
    # Test Bluefin interval validation
    bf_provider_sub = BluefinMarketDataProvider("ETH-PERP", "30s")
    assert bf_provider_sub._is_sub_minute_interval("30s") == True
    assert bf_provider_sub._is_sub_minute_interval("1m") == False
    logger.info("✓ Bluefin sub-minute interval detection works")


async def main():
    """Run all tests."""
    logger.info("Starting unified market data provider tests...\n")
    
    try:
        await test_factory_creation()
        await test_common_functionality()
        await test_exchange_specific_features()
        
        logger.info("\n✅ All tests passed! The unified market data provider system is working correctly.")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())