#!/usr/bin/env python3
"""
Test WebSocket with correct channel names and formats from documentation.
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

async def test_websocket_channels():
    """Test different WebSocket channel combinations."""
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
    
    # Test different subscription combinations
    test_cases = [
        {
            "name": "Heartbeats (Public, No Product IDs)",
            "subscription": {
                "type": "subscribe",
                "channel": "heartbeats"
            }
        },
        {
            "name": "Heartbeats (Authenticated, No Product IDs)",
            "subscription": {
                "type": "subscribe",
                "channel": "heartbeats",
                "jwt": jwt_token
            }
        },
        {
            "name": "Ticker (Public)",
            "subscription": {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channel": "ticker"
            }
        },
        {
            "name": "Ticker (Authenticated)",
            "subscription": {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channel": "ticker",
                "jwt": jwt_token
            }
        },
        {
            "name": "Market Trades (Public)",
            "subscription": {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channel": "market_trades"
            }
        },
        {
            "name": "Market Trades (Authenticated)",
            "subscription": {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channel": "market_trades",
                "jwt": jwt_token
            }
        },
        {
            "name": "Multiple Channels (Authenticated - Channels Array)",
            "subscription": {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channels": ["ticker", "market_trades"],
                "jwt": jwt_token
            }
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n{'='*20} {test_case['name']} {'='*20}")
        await test_subscription(test_case['subscription'], test_case['name'])
        await asyncio.sleep(1)  # Brief pause between tests

async def test_subscription(subscription, test_name):
    """Test a specific WebSocket subscription."""
    try:
        logger.info(f"Connecting to {COINBASE_WS_URL}")
        logger.info(f"Subscription: {json.dumps(subscription, indent=2)}")
        
        async with websockets.connect(COINBASE_WS_URL, timeout=10) as websocket:
            logger.info(f"✅ {test_name} WebSocket connected")
            
            # Send subscription
            await websocket.send(json.dumps(subscription))
            logger.info(f"📤 {test_name} subscription sent")
            
            # Listen for initial messages
            message_count = 0
            start_time = asyncio.get_event_loop().time()
            
            while message_count < 3 and (asyncio.get_event_loop().time() - start_time) < 5:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'subscriptions':
                        logger.info(f"✅ {test_name} subscription confirmed: {data.get('channels', [])}")
                    elif msg_type == 'error':
                        logger.error(f"❌ {test_name} WebSocket error: {data.get('message', 'Unknown error')}")
                        logger.debug(f"Full error: {json.dumps(data, indent=2)}")
                        break
                    elif msg_type in ['heartbeats', 'ticker', 'market_trades']:
                        message_count += 1
                        logger.info(f"📨 {test_name} received {msg_type} message #{message_count}")
                    elif msg_type == 'snapshot':
                        message_count += 1
                        channel = data.get('channel', 'unknown')
                        logger.info(f"📨 {test_name} received snapshot for {channel} channel")
                    else:
                        logger.debug(f"📨 {test_name} received {msg_type}: {json.dumps(data, indent=2)}")
                        
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ {test_name} no messages received in timeout period")
                    break
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ {test_name} received non-JSON message")
                except Exception as e:
                    logger.error(f"❌ {test_name} error processing message: {e}")
                    break
            
            if message_count > 0:
                logger.info(f"✅ {test_name} test successful - received {message_count} messages")
            else:
                logger.warning(f"⚠️ {test_name} test completed but no data messages received")
                
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"❌ {test_name} WebSocket connection closed: {e}")
    except asyncio.TimeoutError:
        logger.error(f"❌ {test_name} WebSocket connection timed out")
    except Exception as e:
        logger.error(f"❌ {test_name} WebSocket connection failed: {e}")

async def main():
    """Run all WebSocket channel tests."""
    logger.info("=" * 80)
    logger.info("Coinbase Advanced Trade WebSocket Channel Tests")
    logger.info("=" * 80)
    
    try:
        await test_websocket_channels()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    logger.info("=" * 80)
    logger.info("All tests completed")

if __name__ == "__main__":
    asyncio.run(main())