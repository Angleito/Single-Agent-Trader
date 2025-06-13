#!/usr/bin/env python3
"""
Test only the WebSocket JWT authentication without historical data.
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_websocket_only():
    """Test only the WebSocket functionality with JWT authentication."""
    logger.info("=" * 80)
    logger.info("WebSocket JWT Authentication Test (No Historical Data)")
    logger.info("=" * 80)
    
    provider = None
    
    try:
        # Create the market data provider
        provider = MarketDataProvider(symbol="BTC-USD", interval="1m")
        
        logger.info("Step 1: Initializing clients...")
        await provider._initialize_clients()
        
        logger.info("Step 2: Testing JWT generation...")
        jwt_token = provider._build_websocket_jwt()
        
        if jwt_token:
            logger.info(f"✅ JWT token generated successfully (length: {len(jwt_token)})")
        else:
            logger.warning("⚠️ JWT token generation failed or returned None")
        
        logger.info("Step 3: Starting WebSocket connection...")
        await provider._start_websocket()
        
        # Wait a moment for connection to establish
        await asyncio.sleep(2)
        
        logger.info("Step 4: Checking connection status...")
        status = provider.get_data_status()
        logger.info(f"WebSocket connected: {status['websocket_connected']}")
        
        if status['websocket_connected']:
            logger.info("✅ WebSocket connection established")
        else:
            logger.warning("⚠️ WebSocket connection status unknown")
        
        # Subscribe to updates to see real-time data
        received_updates = []
        
        def on_update(data):
            received_updates.append(data)
            logger.info(f"📊 Market data update: {data.symbol} Close=${data.close} Volume={data.volume}")
        
        provider.subscribe_to_updates(on_update)
        provider._is_connected = True  # Mark as connected to allow WebSocket to continue
        
        # Wait for some updates
        logger.info("Step 5: Listening for market data updates for 30 seconds...")
        await asyncio.sleep(30)
        
        logger.info(f"Step 6: Results summary")
        logger.info(f"Received {len(received_updates)} market data updates")
        
        # Check cached data
        latest_price = provider.get_latest_price()
        tick_data = provider.get_tick_data(limit=10)
        
        logger.info(f"Latest price: ${latest_price}")
        logger.info(f"Recent tick data: {len(tick_data)} trades")
        
        if latest_price:
            logger.info("✅ Price data is available")
        else:
            logger.warning("⚠️ No price data received")
            
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
        logger.info(f"Final status: WebSocket={final_status['websocket_connected']}, "
                   f"Subscribers={final_status['subscribers']}")
        
        logger.info("✅ WebSocket test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ WebSocket test failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        
    finally:
        if provider:
            logger.info("Cleaning up...")
            await provider.disconnect()
            logger.info("Disconnected from market data feeds")

async def main():
    """Run the WebSocket test."""
    try:
        await test_websocket_only()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    logger.info("=" * 80)
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(main())