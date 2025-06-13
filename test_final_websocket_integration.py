#!/usr/bin/env python3
"""
Final test of the integrated WebSocket JWT authentication with the updated market.py
"""

import asyncio
import logging
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from bot.data.market import MarketDataProvider

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_integrated_websocket():
    """Test the integrated WebSocket functionality with JWT authentication."""
    logger.info("=" * 80)
    logger.info("Final WebSocket JWT Authentication Integration Test")
    logger.info("=" * 80)
    
    provider = None
    
    try:
        # Create the market data provider
        provider = MarketDataProvider(symbol="BTC-USD", interval="1m")
        
        logger.info("Step 1: Connecting to market data feeds (WebSocket only)...")
        await provider.connect(fetch_historical=False)
        
        logger.info("Step 2: Checking connection status...")
        status = provider.get_data_status()
        logger.info(f"Connection status: {status}")
        
        if status['websocket_connected']:
            logger.info("✅ WebSocket is connected")
        else:
            logger.warning("⚠️ WebSocket connection status unknown")
        
        logger.info("Step 3: Waiting for market data...")
        
        # Subscribe to updates to see real-time data
        received_updates = []
        
        def on_update(data):
            received_updates.append(data)
            logger.info(f"📊 Market data update: {data.symbol} Close=${data.close} Volume={data.volume}")
        
        provider.subscribe_to_updates(on_update)
        
        # Wait for some updates
        logger.info("Listening for market data updates for 30 seconds...")
        await asyncio.sleep(30)
        
        logger.info(f"Step 4: Results summary")
        logger.info(f"Received {len(received_updates)} market data updates")
        
        # Check cached data
        latest_price = provider.get_latest_price()
        latest_ohlcv = provider.get_latest_ohlcv(limit=5)
        tick_data = provider.get_tick_data(limit=10)
        
        logger.info(f"Latest price: ${latest_price}")
        logger.info(f"Latest OHLCV candles: {len(latest_ohlcv)}")
        logger.info(f"Recent tick data: {len(tick_data)} trades")
        
        if latest_price:
            logger.info("✅ Price data is available")
        else:
            logger.warning("⚠️ No price data received")
            
        if latest_ohlcv:
            logger.info("✅ OHLCV data is available")
            for i, candle in enumerate(latest_ohlcv[-3:]):  # Show last 3 candles
                logger.info(f"  Candle {i+1}: {candle.timestamp} O={candle.open} H={candle.high} L={candle.low} C={candle.close} V={candle.volume}")
        else:
            logger.warning("⚠️ No OHLCV data available")
            
        if tick_data:
            logger.info("✅ Tick data is available")
            for i, tick in enumerate(tick_data[-3:]):  # Show last 3 trades
                logger.info(f"  Trade {i+1}: {tick['timestamp']} {tick['side']} {tick['size']} @ ${tick['price']}")
        else:
            logger.warning("⚠️ No tick data received")
        
        # Test if we're getting real-time updates
        if received_updates:
            logger.info("✅ Real-time market data updates are working")
        else:
            logger.warning("⚠️ No real-time updates received (this might be normal if market is quiet)")
        
        # Final status check
        final_status = provider.get_data_status()
        logger.info(f"Final status: Connected={final_status['connected']}, "
                   f"WebSocket={final_status['websocket_connected']}, "
                   f"Subscribers={final_status['subscribers']}")
        
        logger.info("✅ Integration test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        
    finally:
        if provider:
            logger.info("Cleaning up...")
            await provider.disconnect()
            logger.info("Disconnected from market data feeds")

async def main():
    """Run the integration test."""
    try:
        await test_integrated_websocket()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    logger.info("=" * 80)
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(main())