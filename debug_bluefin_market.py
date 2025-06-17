#!/usr/bin/env python3
"""Debug script to test Bluefin market data provider directly."""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import the required modules
from bot.data.bluefin_market import BluefinMarketDataProvider
from bot.config import create_settings

async def test_bluefin_market_data():
    """Test the Bluefin market data provider directly."""
    
    # Create settings
    settings = create_settings()
    
    # Initialize the provider
    provider = BluefinMarketDataProvider(symbol="ETH-USD", interval="5m")
    
    try:
        # Connect to the provider
        logger.info("Connecting to Bluefin market data provider...")
        await provider.connect()
        
        # Check connection status
        logger.info(f"Connection status: {provider.is_connected()}")
        
        # Test fetching historical data
        logger.info("Fetching historical data...")
        historical_data = await provider.fetch_historical_data(force_refresh=True)
        
        logger.info(f"Historical data count: {len(historical_data)}")
        
        if historical_data:
            # Show first few candles
            for i, candle in enumerate(historical_data[:5]):
                logger.info(f"Candle {i+1}: {candle.timestamp} - O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} V:{candle.volume}")
            
            # Show last few candles
            for i, candle in enumerate(historical_data[-5:]):
                logger.info(f"Recent candle {i+1}: {candle.timestamp} - O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} V:{candle.volume}")
        
        # Test get_latest_ohlcv method
        logger.info("Testing get_latest_ohlcv method...")
        ohlcv_data = await provider.get_latest_ohlcv(limit=5)
        logger.info(f"OHLCV data count: {len(ohlcv_data)}")
        
        if ohlcv_data:
            for i, candle in enumerate(ohlcv_data):
                logger.info(f"OHLCV {i+1}: {candle.timestamp} - Close: {candle.close}")
        
        # Test DataFrame conversion
        logger.info("Testing DataFrame conversion...")
        df = provider.to_dataframe(limit=5)
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        if not df.empty:
            logger.info(f"DataFrame tail:\n{df.tail()}")
        
        # Test status
        status = provider.get_status()
        logger.info(f"Provider status: {status}")
        
    except Exception as e:
        logger.error(f"Error testing provider: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect
        await provider.disconnect()

if __name__ == "__main__":
    asyncio.run(test_bluefin_market_data())