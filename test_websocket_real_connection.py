#!/usr/bin/env python3
"""
Test real WebSocket connection with JWT authentication.
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"

async def test_websocket_connection():
    """Test real WebSocket connection with JWT authentication."""
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
    
    # Prepare subscription messages
    subscription_with_auth = {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channels": ["ticker", "matches"],
        "jwt": jwt_token
    }
    
    subscription_without_auth = {
        "type": "subscribe", 
        "product_ids": ["BTC-USD"],
        "channels": ["ticker", "matches"]
    }
    
    logger.info("Testing WebSocket connection...")
    
    # Test 1: With authentication
    logger.info("\n1. Testing authenticated WebSocket connection:")
    await test_subscription(subscription_with_auth, "Authenticated")
    
    # Test 2: Without authentication (for comparison)
    logger.info("\n2. Testing public WebSocket connection:")
    await test_subscription(subscription_without_auth, "Public")

async def test_subscription(subscription, test_name):
    """Test a WebSocket subscription."""
    try:
        logger.info(f"Connecting to {COINBASE_WS_URL}")
        logger.info(f"Subscription: {json.dumps(subscription, indent=2)}")
        
        async with websockets.connect(COINBASE_WS_URL, timeout=10) as websocket:
            logger.info(f"✅ {test_name} WebSocket connected")
            
            # Send subscription
            await websocket.send(json.dumps(subscription))
            logger.info(f"📤 {test_name} subscription sent")
            
            # Listen for messages
            message_count = 0
            start_time = asyncio.get_event_loop().time()
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'subscriptions':
                        logger.info(f"✅ {test_name} subscription confirmed: {data.get('channels', [])}")
                    elif msg_type == 'error':
                        logger.error(f"❌ {test_name} WebSocket error: {data.get('message', 'Unknown error')}")
                        logger.debug(f"Full error: {json.dumps(data, indent=2)}")
                        break
                    elif msg_type in ['ticker', 'match']:
                        message_count += 1
                        logger.info(f"📨 {test_name} received {msg_type} message #{message_count}")
                        
                        if message_count == 1:
                            logger.debug(f"First message: {json.dumps(data, indent=2)}")
                    else:
                        logger.debug(f"📨 {test_name} received {msg_type}: {data}")
                    
                    # Stop after 5 messages or 10 seconds
                    if message_count >= 5 or (asyncio.get_event_loop().time() - start_time) > 10:
                        logger.info(f"✅ {test_name} test completed successfully - received {message_count} data messages")
                        break
                        
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ {test_name} received non-JSON message")
                except Exception as e:
                    logger.error(f"❌ {test_name} error processing message: {e}")
                    
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"❌ {test_name} WebSocket connection closed: {e}")
    except asyncio.TimeoutError:
        logger.error(f"❌ {test_name} WebSocket connection timed out")
    except Exception as e:
        logger.error(f"❌ {test_name} WebSocket connection failed: {e}")

async def main():
    """Run the WebSocket connection test."""
    logger.info("=" * 60)
    logger.info("Real WebSocket Connection Test with JWT Authentication")
    logger.info("=" * 60)
    
    try:
        await test_websocket_connection()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    logger.info("=" * 60)
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(main())