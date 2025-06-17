#!/usr/bin/env python3
"""
Test script for real-time WebSocket data streaming.

This script tests the complete WebSocket data pipeline:
1. Bluefin service WebSocket endpoints
2. BluefinServiceClient WebSocket consumption
3. RealtimeMarketDataProvider with tick aggregation
4. Real-time candle generation for high-frequency trading

Usage:
    python test_realtime_websocket.py
"""

import asyncio
import logging
import time
from datetime import datetime, UTC

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_websocket_streaming():
    """Test WebSocket streaming functionality"""
    logger.info("Starting WebSocket streaming test...")
    
    try:
        # Import our real-time market data provider
        from bot.data.realtime_market import RealtimeMarketDataProvider
        
        # Test parameters
        symbol = "ETH-PERP"  # or SUI-PERP, BTC-PERP
        intervals = [1, 5, 15]  # 1s, 5s, 15s intervals
        test_duration = 60  # Run test for 60 seconds
        
        logger.info(f"Testing real-time data for {symbol} with intervals: {intervals}s")
        
        # Create real-time provider
        provider = RealtimeMarketDataProvider(symbol=symbol, intervals=intervals)
        
        # Set up callback to monitor new candles
        candle_count = {}
        
        def on_new_candle(candle):
            interval_key = f"{candle.symbol}_{(candle.timestamp.replace(second=0, microsecond=0) - candle.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()}"
            candle_count[interval_key] = candle_count.get(interval_key, 0) + 1
            logger.info(
                f"New candle: {candle.symbol} {candle.timestamp} "
                f"OHLC: {candle.open:.6f}/{candle.high:.6f}/{candle.low:.6f}/{candle.close:.6f} "
                f"Volume: {candle.volume:.4f}"
            )
        
        # Subscribe to candle updates
        provider.subscribe_to_candles(on_new_candle)
        
        # Connect to data sources
        logger.info("Connecting to data sources...")
        connected = await provider.connect()
        
        if not connected:
            logger.error("Failed to connect to data sources")
            return False
        
        logger.info("Connected successfully!")
        
        # Wait for WebSocket connection
        logger.info("Waiting for WebSocket connection...")
        start_time = time.time()
        while not provider.is_websocket_connected() and time.time() - start_time < 30:
            await asyncio.sleep(1)
        
        if not provider.is_websocket_connected():
            logger.warning("WebSocket connection not established within 30 seconds")
        else:
            logger.info("WebSocket connected successfully!")
        
        # Monitor real-time data for test duration
        logger.info(f"Monitoring real-time data for {test_duration} seconds...")
        
        start_time = time.time()
        last_status_time = start_time
        
        while time.time() - start_time < test_duration:
            current_time = time.time()
            
            # Log status every 10 seconds
            if current_time - last_status_time >= 10:
                status = provider.get_status()
                performance = provider.get_performance_stats()
                
                logger.info("=== Status Update ===")
                logger.info(f"Connected: {status['connected']}")
                logger.info(f"WebSocket: {status['websocket_connected']}")
                logger.info(f"Total ticks: {performance['total_ticks']}")
                logger.info(f"Tick rate: {performance.get('tick_rate_per_second', 0):.2f} ticks/sec")
                logger.info(f"Current price: ${performance.get('current_price', 'N/A')}")
                
                # Show current candles being built
                current_candles = provider.get_current_candles()
                for interval, candle in current_candles.items():
                    logger.info(
                        f"  {interval}s candle: {candle.tick_count} ticks, "
                        f"Volume: {candle.volume:.4f}, "
                        f"Price: {candle.open:.6f} -> {candle.close:.6f}"
                    )
                
                # Show completed candles count
                for interval in intervals:
                    history_count = len(provider.get_candle_history(interval, 1000))
                    logger.info(f"  {interval}s history: {history_count} completed candles")
                
                last_status_time = current_time
            
            await asyncio.sleep(1)
        
        logger.info("=== Final Test Results ===")
        
        # Get final statistics
        final_performance = provider.get_performance_stats()
        logger.info(f"Total test duration: {test_duration} seconds")
        logger.info(f"Total ticks received: {final_performance['total_ticks']}")
        logger.info(f"Average tick rate: {final_performance.get('tick_rate_per_second', 0):.2f} ticks/sec")
        
        # Check completed candles for each interval
        all_intervals_working = True
        for interval in intervals:
            history = provider.get_candle_history(interval, 1000)
            completed_candles = len(history)
            expected_candles = test_duration // interval  # Rough estimate
            
            logger.info(f"{interval}s interval: {completed_candles} completed candles (expected ~{expected_candles})")
            
            if completed_candles == 0:
                all_intervals_working = False
                logger.warning(f"No completed candles for {interval}s interval!")
            else:
                # Show sample of recent candles
                recent_candles = history[-min(3, len(history)):]
                for candle in recent_candles:
                    logger.info(
                        f"  Sample {interval}s candle: {candle.timestamp} "
                        f"OHLC: {candle.open:.6f}/{candle.high:.6f}/{candle.low:.6f}/{candle.close:.6f}"
                    )
        
        # Test recent ticks
        recent_ticks = provider.get_recent_ticks(10)
        logger.info(f"Recent ticks: {len(recent_ticks)} available")
        for tick in recent_ticks[-3:]:  # Show last 3 ticks
            logger.info(
                f"  Tick: {tick.timestamp} {tick.side} {tick.volume:.4f} @ ${tick.price:.6f}"
            )
        
        # Disconnect
        await provider.disconnect()
        logger.info("Disconnected from data sources")
        
        # Determine test result
        success = (
            final_performance['total_ticks'] > 0 and
            all_intervals_working and
            len(recent_ticks) > 0
        )
        
        if success:
            logger.info("✅ WebSocket streaming test PASSED")
            logger.info("Real-time data pipeline is working correctly!")
        else:
            logger.error("❌ WebSocket streaming test FAILED")
            logger.error("Issues detected in real-time data pipeline")
        
        return success
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_bluefin_service_health():
    """Test Bluefin service health and WebSocket endpoints"""
    logger.info("Testing Bluefin service health...")
    
    try:
        import aiohttp
        
        # Test service health
        async with aiohttp.ClientSession() as session:
            # Test basic health endpoint
            async with session.get("http://localhost:8080/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"Service health: {health_data}")
                else:
                    logger.error(f"Health check failed: {response.status}")
                    return False
            
            # Test streaming status endpoint
            async with session.get("http://localhost:8080/streaming/status") as response:
                if response.status == 200:
                    streaming_data = await response.json()
                    logger.info(f"Streaming status: {streaming_data}")
                else:
                    logger.warning(f"Streaming status failed: {response.status}")
        
        logger.info("✅ Bluefin service health check PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Service health check failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting comprehensive WebSocket streaming tests")
    
    # Test 1: Service health
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Bluefin Service Health")
    logger.info("="*50)
    service_ok = await test_bluefin_service_health()
    
    if not service_ok:
        logger.error("Bluefin service is not available. Please start it first:")
        logger.error("  cd bluefin-service && python bluefin_service.py")
        return
    
    # Test 2: WebSocket streaming
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Real-time WebSocket Streaming")
    logger.info("="*50)
    streaming_ok = await test_websocket_streaming()
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Service Health: {'✅ PASS' if service_ok else '❌ FAIL'}")
    logger.info(f"WebSocket Streaming: {'✅ PASS' if streaming_ok else '❌ FAIL'}")
    
    if service_ok and streaming_ok:
        logger.info("\n🎉 All tests PASSED! Real-time WebSocket system is ready for high-frequency trading!")
        logger.info("\nYou can now run the bot with high-frequency intervals:")
        logger.info("  python -m bot.main live --symbol ETH-PERP --interval 15s")
        logger.info("  python -m bot.main live --symbol SUI-PERP --interval 5s")
        logger.info("  python -m bot.main live --symbol BTC-PERP --interval 1s")
    else:
        logger.error("\n❌ Some tests failed. Please check the logs and fix any issues.")

if __name__ == "__main__":
    asyncio.run(main())