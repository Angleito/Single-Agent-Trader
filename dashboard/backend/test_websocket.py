#!/usr/bin/env python3
"""
Test WebSocket connection and LLM event streaming
"""

import asyncio
import json
import websockets
from datetime import datetime


async def test_websocket():
    """Test WebSocket connection and receive LLM events"""
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to WebSocket at {uri}")
            print("Listening for LLM events...\n")
            
            # Send a test message
            await websocket.send("Hello from test client")
            
            # Listen for messages
            message_count = 0
            llm_event_count = 0
            
            while message_count < 20:  # Listen for up to 20 messages
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    # Check message type
                    msg_type = data.get("type", "unknown")
                    timestamp = data.get("timestamp", "")
                    
                    if msg_type in ["llm_event", "llm_decision"]:
                        llm_event_count += 1
                        event_type = data.get("event_type", "unknown")
                        
                        print(f"📊 LLM Event #{llm_event_count}")
                        print(f"   Type: {msg_type} / {event_type}")
                        print(f"   Time: {timestamp}")
                        
                        if event_type == "trading_decision":
                            print(f"   Action: {data.get('action', 'N/A')}")
                            print(f"   Symbol: {data.get('symbol', 'N/A')}")
                            print(f"   Rationale: {data.get('rationale', 'N/A')}")
                        elif event_type == "performance_metrics":
                            print(f"   Completions: {data.get('total_completions', 0)}")
                            print(f"   Avg Response: {data.get('avg_response_time_ms', 0)}ms")
                        
                        print()
                    else:
                        print(f"📨 Message #{message_count}: {msg_type} at {timestamp}")
                        
                except asyncio.TimeoutError:
                    print("⏱️  No message received in 5 seconds, continuing...")
                    continue
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    break
            
            print(f"\n📊 Summary:")
            print(f"   Total messages received: {message_count}")
            print(f"   LLM events received: {llm_event_count}")
            
    except Exception as e:
        print(f"❌ WebSocket connection error: {e}")


async def test_api_endpoints():
    """Test LLM API endpoints"""
    import aiohttp
    
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # Test LLM status endpoint
        try:
            async with session.get(f"{base_url}/llm/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("\n✅ LLM Status Endpoint:")
                    print(f"   Monitoring active: {data.get('monitoring_active', False)}")
                    print(f"   Total decisions: {data.get('total_parsed', {}).get('decisions', 0)}")
                    print(f"   Recent activity: {data.get('recent_activity', {}).get('decisions_last_hour', 0)} decisions in last hour")
                else:
                    print(f"\n❌ LLM Status endpoint returned {resp.status}")
        except Exception as e:
            print(f"\n❌ Error testing LLM status: {e}")
        
        # Test LLM decisions endpoint
        try:
            async with session.get(f"{base_url}/llm/decisions?limit=5") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("\n✅ LLM Decisions Endpoint:")
                    print(f"   Total decisions: {data.get('total_decisions', 0)}")
                    print(f"   Action distribution: {data.get('action_distribution', {})}")
                    
                    decisions = data.get('decisions', [])
                    if decisions:
                        print(f"\n   Recent decisions:")
                        for i, decision in enumerate(decisions[:3], 1):
                            print(f"   {i}. {decision.get('action')} - {decision.get('rationale', 'N/A')}")
                else:
                    print(f"\n❌ LLM Decisions endpoint returned {resp.status}")
        except Exception as e:
            print(f"\n❌ Error testing LLM decisions: {e}")
        
        # Test LLM activity endpoint
        try:
            async with session.get(f"{base_url}/llm/activity?limit=10") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("\n✅ LLM Activity Endpoint:")
                    print(f"   Total events: {data.get('total_events', 0)}")
                else:
                    print(f"\n❌ LLM Activity endpoint returned {resp.status}")
        except Exception as e:
            print(f"\n❌ Error testing LLM activity: {e}")


async def main():
    """Run all tests"""
    print("🧪 Testing AI Trading Bot Dashboard WebSocket and API...\n")
    
    # Test API endpoints first
    await test_api_endpoints()
    
    # Then test WebSocket
    print("\n🔌 Testing WebSocket connection...")
    await test_websocket()


if __name__ == "__main__":
    asyncio.run(main())