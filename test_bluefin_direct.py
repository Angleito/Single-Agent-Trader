#!/usr/bin/env python3
"""Direct test of Bluefin WebSocket integration."""

import asyncio
import os
import logging
from bot.data.bluefin_websocket import BluefinWebSocketClient
from bot.types import MarketData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Track received data
candles_received = 0
last_price = None

async def on_candle(candle: MarketData):
    """Handle new candle data."""
    global candles_received, last_price
    candles_received += 1
    last_price = candle.close
    print(f"✅ Candle #{candles_received}: SUI-PERP @ ${candle.close} | "
          f"O:{candle.open} H:{candle.high} L:{candle.low} V:{candle.volume}")

async def main():
    """Test Bluefin WebSocket."""
    print("🚀 Testing Bluefin WebSocket for SUI-PERP...")
    
    # Create WebSocket client
    ws_client = BluefinWebSocketClient(
        symbol="SUI-PERP",
        interval="15s",  # 15 second candles for faster testing
        on_candle_update=on_candle
    )
    
    try:
        # Connect
        print("📡 Connecting to Bluefin WebSocket...")
        await ws_client.connect()
        print("✅ Connected!")
        
        # Wait for data
        print("⏳ Waiting for real-time data...")
        
        # Monitor for 60 seconds
        for i in range(60):
            await asyncio.sleep(1)
            
            status = ws_client.get_status()
            if i % 10 == 0:
                print(f"\n📊 Status at {i}s:")
                print(f"  Messages: {status['message_count']}")
                print(f"  Ticks: {status['ticks_buffered']}")
                print(f"  Candles: {status['candles_buffered']}")
                print(f"  Latest: ${status['latest_price']}")
        
        print(f"\n✅ Test complete! Received {candles_received} candles")
        print(f"💰 Final SUI-PERP price: ${last_price}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await ws_client.disconnect()
        print("🔌 Disconnected")

if __name__ == "__main__":
    asyncio.run(main())