#!/usr/bin/env python3
"""
Simple WebSocket client to test dashboard backend WebSocket endpoint.

This script can be run locally or inside Docker containers to verify
WebSocket connectivity and message flow.
"""

import asyncio
import json
import sys
from datetime import datetime
import websockets
import argparse


async def test_websocket_connection(url: str, duration: int = 10):
    """Test WebSocket connection and listen for messages."""
    print(f"Connecting to WebSocket: {url}")
    
    try:
        async with websockets.connect(url, timeout=5) as websocket:
            print("✅ Connected successfully!")
            print(f"Will listen for messages for {duration} seconds...")
            print("-" * 50)
            
            # Send initial test message
            test_msg = {
                "type": "test_connection",
                "timestamp": datetime.now().isoformat(),
                "source": "test_client",
                "message": "WebSocket test client connected"
            }
            await websocket.send(json.dumps(test_msg))
            print(f"→ Sent: {json.dumps(test_msg, indent=2)}")
            
            # Listen for messages
            start_time = asyncio.get_event_loop().time()
            message_count = 0
            
            while asyncio.get_event_loop().time() - start_time < duration:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message_count += 1
                    
                    # Try to parse as JSON
                    try:
                        data = json.loads(message)
                        print(f"\n← Received [{message_count}]: {json.dumps(data, indent=2)}")
                        
                        # Analyze message type
                        msg_type = data.get("type", "unknown")
                        if msg_type == "trading_loop":
                            print(f"  Trading Loop: {data.get('data', {}).get('action')} @ ${data.get('data', {}).get('price')}")
                        elif msg_type == "ai_decision":
                            print(f"  AI Decision: {data.get('data', {}).get('action')} (confidence: {data.get('data', {}).get('confidence')})")
                        elif msg_type == "market_data":
                            print(f"  Market Data: {data.get('data', {}).get('symbol')} @ ${data.get('data', {}).get('price')}")
                            
                    except json.JSONDecodeError:
                        print(f"\n← Received [{message_count}] (raw): {message[:100]}...")
                        
                except asyncio.TimeoutError:
                    # No message received in 1 second, continue listening
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("\n⚠️  WebSocket connection closed by server")
                    break
            
            print("\n" + "-" * 50)
            print(f"✅ Test completed. Received {message_count} messages in {duration} seconds")
            
            # Send disconnect message
            disconnect_msg = {
                "type": "test_disconnect",
                "timestamp": datetime.now().isoformat(),
                "source": "test_client",
                "message": "WebSocket test client disconnecting"
            }
            await websocket.send(json.dumps(disconnect_msg))
            
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def send_test_messages(url: str, message_type: str = "all"):
    """Send specific test messages to the WebSocket."""
    print(f"Connecting to send test messages: {url}")
    
    try:
        async with websockets.connect(url, timeout=5) as websocket:
            print("✅ Connected successfully!")
            
            # Define test messages
            test_messages = {
                "market_data": {
                    "type": "market_data",
                    "data": {
                        "symbol": "BTC-USD",
                        "price": 45000.0,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                "ai_decision": {
                    "type": "ai_decision",
                    "data": {
                        "action": "BUY",
                        "reasoning": "Strong bullish signals detected",
                        "confidence": 0.85,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                "trading_loop": {
                    "type": "trading_loop",
                    "data": {
                        "price": 45000.0,
                        "action": "BUY",
                        "confidence": 0.85,
                        "timestamp": datetime.now().isoformat(),
                        "symbol": "BTC-USD"
                    }
                },
                "system_status": {
                    "type": "system_status",
                    "data": {
                        "status": "healthy",
                        "health": True,
                        "errors": [],
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
            
            # Send requested messages
            if message_type == "all":
                messages_to_send = test_messages.items()
            else:
                messages_to_send = [(message_type, test_messages.get(message_type))]
                
            for msg_type, message in messages_to_send:
                if message:
                    await websocket.send(json.dumps(message))
                    print(f"→ Sent {msg_type}: {json.dumps(message, indent=2)}")
                    await asyncio.sleep(0.5)  # Small delay between messages
                    
            print("\n✅ Test messages sent successfully")
            
    except Exception as e:
        print(f"❌ Failed to send test messages: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test WebSocket connection to dashboard backend")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws",
        help="WebSocket URL to connect to (default: ws://localhost:8000/ws)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duration to listen for messages in seconds (default: 10)"
    )
    parser.add_argument(
        "--send",
        choices=["market_data", "ai_decision", "trading_loop", "system_status", "all"],
        help="Send specific test messages"
    )
    parser.add_argument(
        "--docker-url",
        action="store_true",
        help="Use Docker internal URL (ws://dashboard-backend:8000/ws)"
    )
    
    args = parser.parse_args()
    
    # Override URL if docker flag is set
    if args.docker_url:
        args.url = "ws://dashboard-backend:8000/ws"
    
    # Run the appropriate test
    if args.send:
        success = asyncio.run(send_test_messages(args.url, args.send))
    else:
        success = asyncio.run(test_websocket_connection(args.url, args.duration))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()