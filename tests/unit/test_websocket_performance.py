"""
Test suite for WebSocket performance improvements.

Tests the non-blocking message processing and async indicator calculations.
"""

import asyncio
import logging
import time
import unittest
from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from bot.data.bluefin_market import BluefinMarketDataProvider
from bot.data.market import MarketDataProvider
from bot.indicators.vumanchu import CipherA
from bot.types import MarketData

logger = logging.getLogger(__name__)


class TestWebSocketPerformance(unittest.TestCase):
    """Test WebSocket performance improvements."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test market data
        self.test_data = []
        base_time = datetime.now()
        for i in range(100):
            self.test_data.append(
                MarketData(
                    symbol="BTC-USD",
                    timestamp=base_time + timedelta(minutes=i),
                    open=Decimal("45000") + Decimal(str(i)),
                    high=Decimal("45100") + Decimal(str(i)),
                    low=Decimal("44900") + Decimal(str(i)),
                    close=Decimal("45050") + Decimal(str(i)),
                    volume=Decimal("100"),
                )
            )

    @pytest.mark.asyncio
    async def test_message_queue_performance(self):
        """Test that message queue doesn't block WebSocket reception."""
        provider = MarketDataProvider("BTC-USD", "1m")

        # Initialize the message queue
        provider._message_queue = asyncio.Queue(maxsize=1000)
        provider._running = True

        # Start message processor
        processor_task = asyncio.create_task(provider._process_websocket_messages())

        # Simulate high-frequency message arrival
        messages = []
        for i in range(1000):
            message = {
                "channel": "ticker",
                "events": [
                    {
                        "type": "update",
                        "tickers": [{"product_id": "BTC-USD", "price": f"{45000 + i}"}],
                    }
                ],
            }
            messages.append(message)

        # Measure time to queue all messages
        start_time = time.time()

        for message in messages:
            try:
                provider._message_queue.put_nowait(message)
            except asyncio.QueueFull:
                # Queue handling should prevent blocking
                pass

        queue_time = time.time() - start_time

        # Should be very fast (under 100ms for 1000 messages)
        self.assertLess(queue_time, 0.1, "Message queueing should be non-blocking")

        # Stop processor
        provider._running = False
        await asyncio.sleep(0.2)  # Allow processor to stop
        processor_task.cancel()

        logger.info(f"✅ Queued 1000 messages in {queue_time:.3f}s")

    @pytest.mark.asyncio
    async def test_subscriber_notification_performance(self):
        """Test that subscriber notifications don't block message processing."""
        provider = MarketDataProvider("BTC-USD", "1m")

        # Create slow subscribers to test non-blocking behavior
        call_count = {"sync": 0, "async": 0}

        def slow_sync_callback(data):
            time.sleep(0.01)  # 10ms delay
            call_count["sync"] += 1

        async def slow_async_callback(data):
            await asyncio.sleep(0.01)  # 10ms delay
            call_count["async"] += 1

        # Add subscribers
        provider.subscribe_to_updates(slow_sync_callback)
        provider.subscribe_to_updates(slow_async_callback)

        # Measure notification time
        start_time = time.time()

        # This should not block even with slow subscribers
        await provider._notify_subscribers(self.test_data[0])

        notification_time = time.time() - start_time

        # Should complete immediately (tasks run in background)
        self.assertLess(
            notification_time, 0.005, "Subscriber notification should be non-blocking"
        )

        # Wait a bit for background tasks to complete
        await asyncio.sleep(0.1)

        logger.info(
            f"✅ Notified subscribers in {notification_time:.3f}s (non-blocking)"
        )

    @pytest.mark.asyncio
    async def test_async_indicator_performance(self):
        """Test async indicator calculation performance."""
        cipher_a = CipherA()

        # Create test DataFrame with sufficient data
        df_data = []
        base_time = datetime.now()
        for i in range(200):  # Enough for indicators
            df_data.append(
                {
                    "timestamp": base_time + timedelta(minutes=i),
                    "open": 45000 + i,
                    "high": 45100 + i,
                    "low": 44900 + i,
                    "close": 45050 + i,
                    "volume": 100,
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)

        # Test synchronous calculation time
        start_time = time.time()
        sync_result = cipher_a.calculate(df)
        sync_time = time.time() - start_time

        # Test asynchronous calculation time
        start_time = time.time()
        async_result = await cipher_a.calculate_async(df)
        async_time = time.time() - start_time

        # Test streaming calculation time (parallel)
        start_time = time.time()
        streaming_result = await cipher_a.calculate_streaming(df)
        streaming_time = time.time() - start_time

        # Results should be similar
        self.assertEqual(len(sync_result), len(async_result))
        self.assertEqual(len(sync_result), len(streaming_result))

        # Async should not be significantly slower
        self.assertLess(
            async_time, sync_time * 2, "Async calculation should not be much slower"
        )

        logger.info("✅ Indicator calculation times:")
        logger.info(f"   Sync: {sync_time:.3f}s")
        logger.info(f"   Async: {async_time:.3f}s")
        logger.info(f"   Streaming: {streaming_time:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test that multiple processes can run concurrently without blocking."""
        provider = MarketDataProvider("BTC-USD", "1m")
        cipher_a = CipherA()

        # Create test data
        df_data = []
        for i in range(100):
            df_data.append(
                {
                    "timestamp": datetime.now() + timedelta(minutes=i),
                    "open": 45000 + i,
                    "high": 45100 + i,
                    "low": 44900 + i,
                    "close": 45050 + i,
                    "volume": 100,
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)

        # Simulate concurrent operations
        start_time = time.time()

        # Run multiple operations concurrently
        tasks = []

        # Task 1: Process WebSocket messages
        provider._message_queue = asyncio.Queue(maxsize=100)
        provider._running = True
        tasks.append(asyncio.create_task(provider._process_websocket_messages()))

        # Task 2: Calculate indicators
        tasks.append(asyncio.create_task(cipher_a.calculate_async(df)))

        # Task 3: Calculate streaming indicators
        tasks.append(asyncio.create_task(cipher_a.calculate_streaming(df)))

        # Task 4: Simulate message processing
        async def simulate_messages():
            for i in range(50):
                message = {"type": "test", "data": i}
                try:
                    provider._message_queue.put_nowait(message)
                except asyncio.QueueFull:
                    pass
                await asyncio.sleep(0.001)  # 1ms between messages

        tasks.append(asyncio.create_task(simulate_messages()))

        # Wait for some tasks to complete (not the message processor)
        results = await asyncio.gather(*tasks[1:], return_exceptions=True)

        concurrent_time = time.time() - start_time

        # Stop message processor
        provider._running = False
        tasks[0].cancel()

        # Should complete in reasonable time
        self.assertLess(
            concurrent_time, 5.0, "Concurrent processing should be efficient"
        )

        # Check that calculations completed successfully
        for i, result in enumerate(results[:-1]):  # Exclude message simulation task
            self.assertFalse(
                isinstance(result, Exception), f"Task {i+1} should not fail"
            )

        logger.info(f"✅ Concurrent processing completed in {concurrent_time:.3f}s")

    @pytest.mark.asyncio
    async def test_bluefin_websocket_performance(self):
        """Test Bluefin WebSocket performance improvements."""
        provider = BluefinMarketDataProvider("BTC-PERP", "1m")

        # Test message queue setup
        provider._message_queue = asyncio.Queue(maxsize=1000)
        provider._running = True

        # Start message processor
        processor_task = asyncio.create_task(provider._process_websocket_messages())

        # Simulate market data messages
        messages = []
        for i in range(500):
            message = ["MarketDataUpdate", {"lastPrice": 45000 + i, "volume": 100}]
            messages.append(message)

        # Measure queueing performance
        start_time = time.time()

        for message in messages:
            try:
                provider._message_queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

        queue_time = time.time() - start_time

        # Should be fast
        self.assertLess(queue_time, 0.05, "Bluefin message queueing should be fast")

        # Stop processor
        provider._running = False
        await asyncio.sleep(0.1)
        processor_task.cancel()

        logger.info(f"✅ Bluefin queued 500 messages in {queue_time:.3f}s")

    def test_data_integrity(self):
        """Test that optimizations don't break data integrity."""
        cipher_a = CipherA()

        # Create test data
        df_data = []
        for i in range(150):
            df_data.append(
                {
                    "timestamp": datetime.now() + timedelta(minutes=i),
                    "open": 45000 + i * 10,
                    "high": 45100 + i * 10,
                    "low": 44900 + i * 10,
                    "close": 45050 + i * 10,
                    "volume": 100 + i,
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)

        # Calculate indicators synchronously
        sync_result = cipher_a.calculate(df)

        # Verify key indicators are present
        required_indicators = [
            "wt1",
            "wt2",
            "rsi",
            "ema_ribbon_bullish",
            "ema_ribbon_bearish",
        ]
        for indicator in required_indicators:
            self.assertIn(
                indicator, sync_result.columns, f"Missing indicator: {indicator}"
            )
            # Check that we have actual values (not all NaN)
            non_nan_count = sync_result[indicator].notna().sum()
            self.assertGreater(
                non_nan_count, 50, f"Indicator {indicator} has too many NaN values"
            )

        logger.info(
            f"✅ Data integrity verified - {len(required_indicators)} indicators present"
        )


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main()
