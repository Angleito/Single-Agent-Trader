#!/usr/bin/env python3
"""
Test script to verify WebSocket JWT authentication using the SDK's jwt_generator.
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from bot.config import settings
from bot.data.market import MarketDataProvider

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_jwt_generation():
    """Test JWT generation using the SDK's jwt_generator."""
    logger.info("Testing WebSocket JWT generation with SDK")
    
    # Create market data provider
    provider = MarketDataProvider(symbol="BTC-USD", interval="1m")
    
    # Test JWT generation
    jwt_token = provider._build_websocket_jwt()
    
    if jwt_token:
        logger.info("✅ JWT generation successful!")
        logger.info(f"JWT token length: {len(jwt_token)}")
        logger.info(f"JWT token preview: {jwt_token[:50]}...")
        
        # Parse JWT to verify structure (basic validation)
        try:
            import base64
            import json
            
            # Split the JWT into header, payload, signature
            parts = jwt_token.split('.')
            if len(parts) == 3:
                header = json.loads(base64.urlsafe_b64decode(parts[0] + '=='))
                payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))
                
                logger.info(f"JWT header: {json.dumps(header, indent=2)}")
                logger.info(f"JWT payload: {json.dumps(payload, indent=2)}")
                
                # Check required fields
                required_fields = ['sub', 'iss', 'nbf', 'exp']
                missing_fields = [field for field in required_fields if field not in payload]
                
                if missing_fields:
                    logger.warning(f"Missing required JWT fields: {missing_fields}")
                else:
                    logger.info("✅ JWT structure validation passed")
                    
                # Check expiration
                exp = payload.get('exp', 0)
                exp_time = datetime.fromtimestamp(exp)
                logger.info(f"JWT expires at: {exp_time}")
                
            else:
                logger.error("❌ Invalid JWT format - should have 3 parts")
                
        except Exception as e:
            logger.error(f"❌ Failed to parse JWT: {e}")
    else:
        logger.error("❌ JWT generation failed")
        
        # Check CDP credentials
        cdp_api_key = getattr(settings.exchange, 'cdp_api_key_name', None)
        cdp_private_key = getattr(settings.exchange, 'cdp_private_key', None)
        
        if not cdp_api_key:
            logger.error("❌ CDP API key not configured")
        else:
            logger.info("✅ CDP API key found")
            
        if not cdp_private_key:
            logger.error("❌ CDP private key not configured")  
        else:
            logger.info("✅ CDP private key found")


async def test_websocket_connection():
    """Test WebSocket connection with JWT authentication."""
    logger.info("Testing WebSocket connection with JWT authentication")
    
    try:
        # Create market data provider
        provider = MarketDataProvider(symbol="BTC-USD", interval="1m")
        
        # Initialize clients without starting WebSocket
        await provider._initialize_clients()
        
        # Test WebSocket connection (will timeout after a few seconds)
        logger.info("Attempting WebSocket connection...")
        
        # Start the websocket connection with a timeout
        try:
            await asyncio.wait_for(provider._start_websocket(), timeout=10.0)
            logger.info("✅ WebSocket connection started successfully")
            
            # Wait a bit to see if we get any messages
            await asyncio.sleep(5)
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ WebSocket connection test timed out (this is expected)")
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
        finally:
            # Clean up
            await provider.disconnect()
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("WebSocket JWT Authentication Test")
    logger.info("=" * 60)
    
    # Test 1: JWT Generation
    await test_jwt_generation()
    
    logger.info("\n" + "=" * 60)
    
    # Test 2: WebSocket Connection (optional, might timeout)
    try:
        await test_websocket_connection()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
    
    logger.info("=" * 60)
    logger.info("Test completed")


if __name__ == "__main__":
    asyncio.run(main())