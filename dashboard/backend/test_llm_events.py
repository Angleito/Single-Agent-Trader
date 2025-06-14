#!/usr/bin/env python3
"""
Test script to generate sample LLM events for testing the dashboard
"""

import asyncio
import json
from datetime import datetime
import websockets
import random

# Sample LLM events
def create_llm_request():
    return {
        "type": "llm_event",
        "data": {
            "event_type": "llm_request",
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session-001",
            "request_id": f"req-{random.randint(1000, 9999)}",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "prompt_tokens": random.randint(500, 1500),
            "prompt_length": random.randint(1000, 3000),
            "context": {
                "market_data": {
                    "price": 45678.90,
                    "volume": 1234567
                },
                "indicators": {
                    "cipher_a": 0.75,
                    "cipher_b": -0.25
                }
            }
        }
    }

def create_llm_response(request_id):
    success = random.random() > 0.1  # 90% success rate
    return {
        "type": "llm_event",
        "data": {
            "event_type": "llm_response",
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session-001",
            "request_id": request_id,
            "success": success,
            "response_time_ms": random.randint(500, 3000),
            "completion_tokens": random.randint(100, 500),
            "total_tokens": random.randint(600, 2000),
            "cost_estimate_usd": random.uniform(0.01, 0.10),
            "error": None if success else "API rate limit exceeded"
        }
    }

def create_trading_decision(request_id):
    actions = ["LONG", "SHORT", "HOLD", "CLOSE"]
    action = random.choice(actions)
    
    reasoning_map = {
        "LONG": "Strong bullish momentum detected with VuManChu Cipher A showing positive divergence. Market structure indicates potential upward movement with support at current levels.",
        "SHORT": "Bearish divergence on multiple timeframes. VuManChu Cipher B showing overbought conditions with weakening momentum. Risk/reward favors short position.",
        "HOLD": "Mixed signals in current market conditions. Waiting for clearer confirmation before entering position. VuManChu indicators showing neutral stance.",
        "CLOSE": "Target reached with significant profit. Momentum indicators showing exhaustion. Prudent to secure gains and wait for next opportunity."
    }
    
    return {
        "type": "llm_event",
        "data": {
            "event_type": "trading_decision",
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session-001",
            "request_id": request_id,
            "action": action,
            "confidence": random.uniform(0.6, 0.95),
            "reasoning": reasoning_map[action],
            "rationale": reasoning_map[action],
            "symbol": "BTC-USD",
            "price": random.uniform(45000, 46000),
            "current_price": random.uniform(45000, 46000),
            "leverage": random.choice([1, 5, 10]),
            "indicators": {
                "cipher_a": random.uniform(-1, 1),
                "cipher_b": random.uniform(-1, 1),
                "wave_trend_1": random.uniform(-100, 100),
                "wave_trend_2": random.uniform(-100, 100)
            },
            "risk_analysis": {
                "stop_loss": random.uniform(44000, 45000),
                "take_profit": random.uniform(46000, 48000),
                "risk_reward_ratio": random.uniform(1.5, 3.0)
            }
        }
    }

def create_alert():
    levels = ["info", "warning", "critical"]
    level = random.choice(levels)
    
    alert_messages = {
        "info": "LLM performance within normal parameters",
        "warning": "Response time approaching threshold limits",
        "critical": "Multiple consecutive failures detected"
    }
    
    return {
        "type": "llm_event",
        "data": {
            "event_type": "alert",
            "timestamp": datetime.now().isoformat(),
            "alert_level": level,
            "alert_category": random.choice(["performance", "cost", "reliability"]),
            "alert_message": alert_messages[level],
            "details": {
                "metric": "response_time_ms",
                "value": random.randint(5000, 30000),
                "threshold": 10000
            }
        }
    }

async def send_test_events():
    """Send test events to the WebSocket server"""
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Send initial burst of events
            for i in range(5):
                # Create request
                request = create_llm_request()
                request_id = request["data"]["request_id"]
                await websocket.send(json.dumps(request))
                print(f"Sent LLM request: {request_id}")
                await asyncio.sleep(0.5)
                
                # Create response
                response = create_llm_response(request_id)
                await websocket.send(json.dumps(response))
                print(f"Sent LLM response: {request_id}")
                await asyncio.sleep(0.5)
                
                # Create decision
                decision = create_trading_decision(request_id)
                await websocket.send(json.dumps(decision))
                print(f"Sent trading decision: {decision['data']['action']}")
                await asyncio.sleep(1)
            
            # Send periodic events
            while True:
                # Random event type
                event_type = random.choice(["request_response_decision", "alert"])
                
                if event_type == "request_response_decision":
                    # Full cycle
                    request = create_llm_request()
                    request_id = request["data"]["request_id"]
                    await websocket.send(json.dumps(request))
                    print(f"Sent LLM request: {request_id}")
                    await asyncio.sleep(1)
                    
                    response = create_llm_response(request_id)
                    await websocket.send(json.dumps(response))
                    print(f"Sent LLM response: {request_id}")
                    await asyncio.sleep(1)
                    
                    if response["data"]["success"]:
                        decision = create_trading_decision(request_id)
                        await websocket.send(json.dumps(decision))
                        print(f"Sent trading decision: {decision['data']['action']}")
                    
                else:
                    # Alert
                    alert = create_alert()
                    await websocket.send(json.dumps(alert))
                    print(f"Sent alert: {alert['data']['alert_level']}")
                
                # Wait before next event
                await asyncio.sleep(random.uniform(3, 10))
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting LLM event test generator...")
    print("Make sure the dashboard backend is running on http://localhost:8000")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(send_test_events())
    except KeyboardInterrupt:
        print("\nStopping test generator...")