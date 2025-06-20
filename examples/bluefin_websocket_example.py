#!/usr/bin/env python3
"""
Example script demonstrating Bluefin WebSocket integration.

This shows how to use the BluefinWebSocketClient for real-time market data streaming.
"""

import asyncio
import logging
from datetime import datetime

from bot.data.bluefin_websocket import BluefinWebSocketClient
from bot.types import MarketData

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def on_candle_update(candle: MarketData):
    """Callback function called when a new candle is completed."""
    logger.info(
        "New candle: %s @ %s | " "O: %s H: %s L: %s C: %s V: %s",
        candle.symbol,
        candle.timestamp,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
    )


async def main():
    """Main function to demonstrate WebSocket usage."""
    # Symbol to track (SUI perpetual futures)
    symbol = "SUI-PERP"
    interval = "1m"  # 1-minute candles

    logger.info("Starting Bluefin WebSocket client for %s", symbol)

    # Create WebSocket client with network specification
    # Network can be "mainnet" or "testnet", or None to use environment variable
    ws_client = BluefinWebSocketClient(
        symbol=symbol,
        interval=interval,
        candle_limit=500,
        on_candle_update=on_candle_update,
        network="mainnet",  # Change to "testnet" for staging environment
    )

    try:
        # Connect to WebSocket
        await ws_client.connect()

        logger.info("WebSocket connected! Streaming real-time data...")

        # Run for a while to collect data
        run_duration = 300  # 5 minutes
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < run_duration:
            # Get current status
            status = ws_client.get_status()

            # Log status every 30 seconds
            if int((datetime.now() - start_time).total_seconds()) % 30 == 0:
                logger.info("Status: %s", status)

                # Get latest price
                latest_price = ws_client.get_latest_price()
                if latest_price:
                    logger.info("Latest price: %s = $%s", symbol, latest_price)

                # Get recent candles
                candles = ws_client.get_candles(limit=5)
                logger.info("Recent %s candles in buffer", len(candles))

            await asyncio.sleep(1)

        # Get final statistics
        logger.info("\n=== Final Statistics ===")
        final_status = ws_client.get_status()
        logger.info("Messages received: %s", final_status["message_count"])
        logger.info("Errors: %s", final_status["error_count"])
        logger.info("Candles collected: %s", final_status["candles_buffered"])
        logger.info("Ticks collected: %s", final_status["ticks_buffered"])

        # Display last few candles
        last_candles = ws_client.get_candles(limit=10)
        logger.info("\nLast %s candles:", len(last_candles))
        for candle in last_candles:
            logger.info(
                "  %s: O=%s H=%s " "L=%s C=%s V=%s",
                candle.timestamp,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Error: %s", e)
    finally:
        # Disconnect
        logger.info("Disconnecting from WebSocket...")
        await ws_client.disconnect()
        logger.info("Disconnected")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
