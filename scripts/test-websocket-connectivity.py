#!/usr/bin/env python3
"""
Quick test script to verify WebSocket connectivity and service availability.

This script tests WebSocket connections to dashboard and Bluefin services
without starting the full trading bot.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import Settings
from bot.data.bluefin_websocket import BluefinWebSocketClient
from bot.websocket_publisher import WebSocketPublisher

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_websocket_publisher():
    """Test WebSocket publisher connection to dashboard."""
    logger.info("=" * 60)
    logger.info("Testing WebSocket Publisher Connection")
    logger.info("=" * 60)

    try:
        settings = Settings()
        publisher = WebSocketPublisher(settings)

        logger.info("Initializing WebSocket publisher...")
        success = await publisher.initialize()

        if success:
            logger.info("✓ WebSocket publisher connected successfully!")

            # Try sending a test message
            await publisher.publish_system_status(
                status="test", health=True, message="WebSocket connectivity test"
            )
            logger.info("✓ Test message sent successfully!")

            # Wait a bit to ensure message is processed
            await asyncio.sleep(2)

            # Check connection status
            if publisher.connected:
                logger.info("✓ Connection is stable")
            else:
                logger.warning("⚠ Connection lost after initialization")

        else:
            logger.error("✗ Failed to initialize WebSocket publisher")

        # Cleanup
        await publisher.close()
        logger.info("WebSocket publisher closed")

        return success

    except Exception as e:
        logger.error("Error testing WebSocket publisher: %s", str(e))
        return False


async def test_bluefin_websocket():
    """Test Bluefin WebSocket connection."""
    logger.info("\n%s", "=" * 60)
    logger.info("Testing Bluefin WebSocket Connection")
    logger.info("=" * 60)

    try:
        # Test with SUI-PERP symbol
        symbol = "SUI-PERP"
        interval = "1m"

        logger.info("Creating Bluefin WebSocket client for %s", symbol)
        client = BluefinWebSocketClient(
            symbol=symbol, interval=interval, candle_limit=100, network="mainnet"
        )

        logger.info("Connecting to Bluefin WebSocket...")
        await client.connect()

        # Wait for some data
        logger.info("Waiting for market data...")
        await asyncio.sleep(5)

        # Check status
        status = client.get_status()
        logger.info("Connection status: %s", status)

        if status["connected"]:
            logger.info("✓ Bluefin WebSocket connected successfully!")
            logger.info("  - Message count: %d", status["message_count"])
            logger.info("  - Candles buffered: %d", status["candles_buffered"])
            logger.info("  - Latest price: %s", status["latest_price"])
            logger.info("  - Subscribed channels: %s", status["subscribed_channels"])
        else:
            logger.error("✗ Bluefin WebSocket not connected")

        # Cleanup
        await client.disconnect()
        logger.info("Bluefin WebSocket disconnected")

        return status["connected"]

    except Exception as e:
        logger.error("Error testing Bluefin WebSocket: %s", str(e))
        return False


async def main():
    """Main test function."""
    results = {"websocket_publisher": False, "bluefin_websocket": False}

    # Test WebSocket publisher
    try:
        results["websocket_publisher"] = await test_websocket_publisher()
    except Exception as e:
        logger.error("WebSocket publisher test failed: %s", str(e))

    # Test Bluefin WebSocket
    try:
        results["bluefin_websocket"] = await test_bluefin_websocket()
    except Exception as e:
        logger.error("Bluefin WebSocket test failed: %s", str(e))

    # Summary
    logger.info("\n%s", "=" * 60)
    logger.info("CONNECTIVITY TEST SUMMARY")
    logger.info("=" * 60)

    for service, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info("%-25s: %s", service, status)

    # Overall result
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ All connectivity tests passed!")
        return 0
    logger.error("\n✗ Some connectivity tests failed!")
    return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
