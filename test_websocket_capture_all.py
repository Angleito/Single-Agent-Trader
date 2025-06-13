#!/usr/bin/env python3
"""
Test WebSocket with complete message capture to see what's really happening.
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

async def test_complete_message_capture():
    """Test WebSocket with complete message capture."""
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
    
    # Test a simple heartbeats subscription first
    test_subscription = {
        "type": "subscribe",
        "channel": "heartbeats"
    }
    
    logger.info("Testing heartbeats subscription with complete message capture...")
    await capture_all_messages(test_subscription, "Heartbeats Test")
    
    # Test with authentication  
    test_subscription_auth = {
        "type": "subscribe",
        "channel": "heartbeats",
        "jwt": jwt_token
    }
    
    logger.info("\nTesting authenticated heartbeats subscription...")
    await capture_all_messages(test_subscription_auth, "Authenticated Heartbeats Test")

async def capture_all_messages(subscription, test_name):
    """Capture all WebSocket messages with detailed logging."""
    try:
        logger.info(f"=== {test_name} ===")
        logger.info(f"Connecting to {COINBASE_WS_URL}")
        logger.info(f"Subscription: {json.dumps(subscription, indent=2)}")
        
        async with websockets.connect(
            COINBASE_WS_URL, 
            timeout=10,
            # Add more detailed logging
            logger=logger
        ) as websocket:
            logger.info(f"✅ {test_name} WebSocket connected successfully")
            logger.debug(f"WebSocket state: {websocket.state}")
            logger.debug(f"WebSocket remote address: {websocket.remote_address}")
            
            # Send subscription
            subscription_json = json.dumps(subscription)
            logger.info(f"📤 Sending subscription: {subscription_json}")
            await websocket.send(subscription_json)
            logger.info(f"📤 {test_name} subscription sent successfully")
            
            # Capture ALL messages for 10 seconds
            message_count = 0
            start_time = asyncio.get_event_loop().time()
            
            logger.info(f"👂 Listening for messages for 10 seconds...")
            
            while (asyncio.get_event_loop().time() - start_time) < 10:
                try:
                    # Wait for message with 1 second timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message_count += 1
                    
                    logger.info(f"📨 Message #{message_count} received ({len(message)} bytes)")
                    logger.debug(f"Raw message: {message}")
                    
                    try:
                        data = json.loads(message)
                        logger.info(f"📨 Parsed message: {json.dumps(data, indent=2)}")
                        
                        msg_type = data.get('type', 'no-type')
                        channel = data.get('channel', 'no-channel')
                        logger.info(f"📨 Message type: {msg_type}, Channel: {channel}")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Non-JSON message received: {message}")
                        logger.warning(f"JSON decode error: {e}")
                        
                except asyncio.TimeoutError:
                    logger.debug(f"⏰ No message received in last 1 second")
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"❌ Connection closed: {e}")
                    break
                except Exception as e:
                    logger.error(f"❌ Error receiving message: {e}")
                    break
            
            logger.info(f"✅ {test_name} completed - received {message_count} total messages")
            
            if message_count == 0:
                logger.warning(f"⚠️ {test_name} - No messages received at all!")
                logger.warning("This suggests:")
                logger.warning("  1. The subscription was silently ignored")
                logger.warning("  2. The channel name is incorrect")
                logger.warning("  3. There's a subscription format issue")
                logger.warning("  4. The WebSocket is connected but not subscribed")
            
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"❌ {test_name} WebSocket connection closed: {e}")
    except asyncio.TimeoutError:
        logger.error(f"❌ {test_name} WebSocket connection timed out")
    except Exception as e:
        logger.error(f"❌ {test_name} WebSocket connection failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")

async def main():
    """Run the complete message capture test."""
    logger.info("=" * 80)
    logger.info("Complete WebSocket Message Capture Test")
    logger.info("=" * 80)
    
    try:
        await test_complete_message_capture()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    logger.info("=" * 80)
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(main())