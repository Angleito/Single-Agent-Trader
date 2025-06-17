"""Example usage of the BluefinWebSocketClient for standalone WebSocket streaming."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

from bot.data.bluefin_websocket import BluefinWebSocketClient
from bot.types import MarketData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def on_new_candle(candle: MarketData):
    """Callback function called when a new candle is completed."""
    logger.info(
        f"New {candle.symbol} candle: "
        f"O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} "
        f"V:{candle.volume} @ {candle.timestamp}"
    )


async def main():
    """Example of using BluefinWebSocketClient."""
    # Create WebSocket client
    ws_client = BluefinWebSocketClient(
        symbol="SUI-PERP",
        interval="1m",
        candle_limit=500,
        on_candle_update=on_new_candle
    )
    
    try:
        # Connect to WebSocket
        logger.info("Connecting to Bluefin WebSocket...")
        await ws_client.connect()
        
        # Wait for some data
        logger.info("Waiting for market data...")
        await asyncio.sleep(10)
        
        # Get current status
        status = ws_client.get_status()
        logger.info(f"WebSocket Status: {status}")
        
        # Get latest price
        latest_price = ws_client.get_latest_price()
        if latest_price:
            logger.info(f"Latest price: {latest_price}")
        
        # Get historical candles
        candles = ws_client.get_candles(limit=10)
        logger.info(f"Retrieved {len(candles)} candles")
        
        # Get recent ticks
        ticks = ws_client.get_ticks(limit=20)
        logger.info(f"Retrieved {len(ticks)} ticks")
        
        # Keep running for a while
        logger.info("Streaming market data... (Press Ctrl+C to stop)")
        await asyncio.sleep(300)  # Run for 5 minutes
        
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        # Disconnect
        await ws_client.disconnect()
        logger.info("Disconnected from WebSocket")


async def integrate_with_existing_provider():
    """Example of integrating WebSocket with existing BluefinMarketDataProvider."""
    from bot.data.bluefin_market import BluefinMarketDataProvider
    
    # Create market data provider
    provider = BluefinMarketDataProvider(symbol="ETH-PERP", interval="5m")
    
    # Override the connect method to use our WebSocket client
    async def enhanced_connect(self, fetch_historical: bool = True):
        """Enhanced connect method using BluefinWebSocketClient."""
        # Initialize HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Create WebSocket client
        self._ws_client = BluefinWebSocketClient(
            symbol=self.symbol,
            interval=self.interval,
            candle_limit=self.candle_limit,
            on_candle_update=lambda candle: asyncio.create_task(self._notify_subscribers(candle))
        )
        
        # Connect WebSocket
        await self._ws_client.connect()
        
        # Fetch historical data if requested
        if fetch_historical:
            try:
                await self.fetch_historical_data()
            except Exception as e:
                logger.warning(f"Failed to fetch historical data: {e}")
        
        self._is_connected = True
        logger.info("Connected with enhanced WebSocket support")
    
    # Replace the connect method
    provider.connect = enhanced_connect.__get__(provider, BluefinMarketDataProvider)
    
    # Use the provider
    await provider.connect()
    
    # Stream data for a while
    await asyncio.sleep(60)
    
    # Disconnect
    await provider.disconnect()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
    
    # Or run the integration example
    # asyncio.run(integrate_with_existing_provider())