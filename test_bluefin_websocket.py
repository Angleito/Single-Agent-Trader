#!/usr/bin/env python3
"""Test script to verify Bluefin WebSocket connection and data flow."""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add the bot module to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.data.bluefin_market import BluefinMarketDataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_bluefin_websocket():
    """Test Bluefin WebSocket connection and data reception."""
    print("🚀 Testing Bluefin WebSocket connection...")
    
    # Force real data mode
    os.environ["BLUEFIN_USE_REAL_DATA"] = "true"
    
    # Create market data provider
    provider = BluefinMarketDataProvider(symbol="ETH-PERP", interval="1m")
    
    # Track received data
    tick_count = 0
    candle_count = 0
    start_time = datetime.now()
    
    def on_market_data(data):
        nonlocal candle_count
        candle_count += 1
        print(f"📊 New candle: {data.symbol} @ {data.close} (O:{data.open} H:{data.high} L:{data.low} V:{data.volume})")
    
    # Subscribe to updates
    provider.subscribe_to_updates(on_market_data)
    
    try:
        # Connect to WebSocket
        await provider.connect(fetch_historical=False)
        
        print("✅ Connected to Bluefin WebSocket")
        print(f"📈 Monitoring {provider.symbol} with {provider.interval} candles")
        print("⏳ Waiting for real-time data (press Ctrl+C to stop)...\n")
        
        # Monitor for 60 seconds
        for i in range(60):
            await asyncio.sleep(1)
            
            # Get status
            status = provider.get_data_status()
            ws_connected = status.get("connected", False) and provider.has_websocket_data()
            
            # Count ticks
            current_ticks = len(provider._tick_buffer)
            new_ticks = current_ticks - tick_count
            tick_count = current_ticks
            
            # Print status every 5 seconds
            if i % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"\n📡 Status after {elapsed:.0f}s:")
                print(f"   WebSocket: {'✅ Connected' if ws_connected else '❌ Disconnected'}")
                print(f"   Ticks received: {tick_count}")
                print(f"   Candles built: {candle_count}")
                print(f"   Latest price: ${status.get('latest_price', 'N/A')}")
                
                if new_ticks > 0:
                    print(f"   📈 {new_ticks} new ticks in last second")
        
        print("\n✅ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.exception("Test error:")
    finally:
        # Cleanup
        await provider.disconnect()
        print("🔌 Disconnected from Bluefin")

if __name__ == "__main__":
    asyncio.run(test_bluefin_websocket())