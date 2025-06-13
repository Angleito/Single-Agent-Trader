#!/usr/bin/env python3
"""
Direct WebSocket test to see what's happening with connection.
"""

import asyncio
import json
import logging
import os
import sys
import websockets

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from bot.config import settings

try:
    from coinbase import jwt_generator
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"

async def test_direct_websocket():
    """Test direct WebSocket connection to see what's happening."""
    if not SDK_AVAILABLE:
        logger.error("❌ SDK not available")
        return

    # Generate JWT token
    cdp_api_key_obj = getattr(settings.exchange, 'cdp_api_key_name', None)
    cdp_private_key_obj = getattr(settings.exchange, 'cdp_private_key', None)
    
    if not cdp_api_key_obj or not cdp_private_key_obj:
        logger.error("❌ CDP credentials not found")
        return
    
    cdp_api_key = cdp_api_key_obj.get_secret_value()
    cdp_private_key = cdp_private_key_obj.get_secret_value()
    
    logger.info("Generating JWT token...")
    jwt_token = jwt_generator.build_ws_jwt(cdp_api_key, cdp_private_key)
    
    if not jwt_token:
        logger.error("❌ Failed to generate JWT token")
        return
    
    logger.info(f"✅ JWT token generated successfully (length: {len(jwt_token)})")
    
    # Test different subscription combinations exactly like market.py uses
    subscriptions = [
        {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channel": "ticker",
            "jwt": jwt_token
        },
        {
            "type": "subscribe", 
            "product_ids": ["BTC-USD"],
            "channel": "market_trades",
            "jwt": jwt_token
        },
        {
            "type": "subscribe",
            "channel": "heartbeats",
            "jwt": jwt_token
        }
    ]
    
    try:
        logger.info(f"Connecting to {COINBASE_WS_URL}")
        
        async with websockets.connect(
            COINBASE_WS_URL,
            timeout=30  # Use same timeout as market.py
        ) as websocket:
            logger.info("✅ WebSocket connected successfully")
            logger.info(f"WebSocket state: {websocket.state}")
            
            # Send all subscriptions
            for i, subscription in enumerate(subscriptions):
                logger.info(f"📤 Sending subscription {i+1}/{len(subscriptions)}: {subscription.get('channel', 'unknown')}")
                logger.debug(f"Full subscription: {json.dumps(subscription, indent=2)}")
                await websocket.send(json.dumps(subscription))
                logger.info(f"✅ Subscription {i+1} sent successfully")
            
            logger.info(f"📡 Sent {len(subscriptions)} subscriptions, listening for responses...")
            
            # Listen for messages with detailed logging
            message_count = 0
            start_time = asyncio.get_event_loop().time()
            
            async for message in websocket:
                try:
                    message_count += 1
                    logger.info(f"📨 Message #{message_count} received ({len(message)} bytes)")
                    
                    data = json.loads(message)
                    msg_type = data.get('type', 'no-type')
                    channel = data.get('channel', 'no-channel')
                    
                    logger.info(f"📨 Message type: {msg_type}, Channel: {channel}")
                    
                    if msg_type == 'subscriptions':
                        logger.info(f"✅ Subscription confirmed: {json.dumps(data, indent=2)}")
                    elif msg_type == 'error':
                        logger.error(f"❌ WebSocket error: {json.dumps(data, indent=2)}")
                    elif channel == 'heartbeats':
                        events = data.get('events', [])
                        if events:
                            counter = events[0].get('heartbeat_counter', 'unknown')
                            logger.info(f"💓 Heartbeat #{counter}")
                    elif channel == 'ticker':
                        events = data.get('events', [])
                        logger.info(f"📊 Ticker update with {len(events)} events")
                        for event in events:
                            tickers = event.get('tickers', [])
                            for ticker in tickers:
                                if ticker.get('product_id') == 'BTC-USD':
                                    logger.info(f"📊 BTC-USD price: ${ticker.get('price', 'unknown')}")
                    elif channel == 'market_trades':
                        events = data.get('events', [])
                        logger.info(f"💰 Trade update with {len(events)} events")
                        for event in events:
                            trades = event.get('trades', [])
                            for trade in trades:
                                if trade.get('product_id') == 'BTC-USD':
                                    logger.info(f"💰 BTC-USD trade: {trade.get('side', 'unknown')} {trade.get('size', 'unknown')} @ ${trade.get('price', 'unknown')}")
                    else:
                        logger.debug(f"📨 Other message: {json.dumps(data, indent=2)}")
                    
                    # Stop after receiving some messages or timeout
                    if message_count >= 10 or (asyncio.get_event_loop().time() - start_time) > 30:
                        logger.info(f"✅ Test completed - received {message_count} messages")
                        break
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Non-JSON message: {message}")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"❌ WebSocket connection closed: {e}")
    except asyncio.TimeoutError:
        logger.error(f"❌ WebSocket connection timed out")
    except Exception as e:
        logger.error(f"❌ WebSocket connection failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")

async def main():
    """Run the direct WebSocket test."""
    logger.info("=" * 80)
    logger.info("Direct WebSocket Connection Test")
    logger.info("=" * 80)
    
    try:
        await test_direct_websocket()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    logger.info("=" * 80)
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(main())