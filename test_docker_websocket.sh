#!/bin/bash
# Test script for WebSocket connectivity in Docker environment

echo "=== Docker WebSocket Integration Test ==="
echo
echo "This script tests WebSocket communication between Docker services"
echo

# Function to check if container is running
check_container() {
    local container=$1
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "✅ Container $container is running"
        return 0
    else
        echo "❌ Container $container is not running"
        return 1
    fi
}

# Function to test WebSocket connection
test_websocket() {
    local container=$1
    local ws_url=$2
    local test_name=$3
    
    echo
    echo "Testing: $test_name"
    echo "Container: $container"
    echo "WebSocket URL: $ws_url"
    
    docker exec $container python -c "
import asyncio
import websockets
import json
import sys

async def test_ws():
    try:
        async with websockets.connect('$ws_url', timeout=5) as websocket:
            print('✅ WebSocket connection established')
            
            # Send a test message
            test_msg = {'type': 'test', 'message': 'Docker WebSocket Test'}
            await websocket.send(json.dumps(test_msg))
            print('✅ Test message sent')
            
            # Try to receive a message (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2)
                print(f'✅ Received response: {response[:100]}...')
            except asyncio.TimeoutError:
                print('⚠️  No response received (timeout) - this is normal for some endpoints')
            
            return True
    except Exception as e:
        print(f'❌ WebSocket test failed: {e}')
        return False

success = asyncio.run(test_ws())
sys.exit(0 if success else 1)
" 2>&1
    
    return $?
}

# Step 1: Check if services are running
echo "Step 1: Checking Docker services..."
echo

services=("ai-trading-bot" "dashboard-backend" "dashboard-frontend")
all_running=true

for service in "${services[@]}"; do
    if ! check_container $service; then
        all_running=false
    fi
done

if [ "$all_running" = false ]; then
    echo
    echo "❌ Not all required services are running."
    echo "Please start the services with: docker-compose up -d"
    exit 1
fi

# Step 2: Test dashboard backend WebSocket endpoint
echo
echo "Step 2: Testing Dashboard Backend WebSocket..."

# Test from trading bot to dashboard backend
if test_websocket "ai-trading-bot" "ws://dashboard-backend:8000/ws" "Trading Bot → Dashboard Backend"; then
    echo "✅ Trading bot can connect to dashboard backend"
else
    echo "❌ Trading bot cannot connect to dashboard backend"
fi

# Step 3: Test WebSocket publisher configuration
echo
echo "Step 3: Checking WebSocket Publisher Configuration..."

docker exec ai-trading-bot python -c "
import os
import sys

# Check environment variables
ws_enabled = os.getenv('SYSTEM__ENABLE_WEBSOCKET_PUBLISHING', 'false').lower() == 'true'
ws_url = os.getenv('SYSTEM__WEBSOCKET_DASHBOARD_URL', 'not set')

print(f'WebSocket Publishing Enabled: {ws_enabled}')
print(f'WebSocket Dashboard URL: {ws_url}')

if ws_enabled and ws_url.startswith('ws://dashboard-backend:'):
    print('✅ WebSocket publisher is properly configured')
    sys.exit(0)
else:
    print('❌ WebSocket publisher configuration needs adjustment')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ WebSocket publisher configuration is correct"
else
    echo "❌ WebSocket publisher configuration needs fixing"
fi

# Step 4: Test inter-service connectivity
echo
echo "Step 4: Testing Inter-Service Network Connectivity..."

# Test network connectivity from trading bot to dashboard backend
docker exec ai-trading-bot python -c "
import socket
import sys

try:
    # Test DNS resolution
    addr = socket.gethostbyname('dashboard-backend')
    print(f'✅ DNS resolution successful: dashboard-backend → {addr}')
    
    # Test TCP connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((addr, 8000))
    sock.close()
    
    if result == 0:
        print('✅ TCP connection to dashboard-backend:8000 successful')
        sys.exit(0)
    else:
        print('❌ TCP connection to dashboard-backend:8000 failed')
        sys.exit(1)
except Exception as e:
    print(f'❌ Network test failed: {e}')
    sys.exit(1)
"

# Step 5: Test WebSocket message flow
echo
echo "Step 5: Testing WebSocket Message Flow..."

# Create a test Python script that simulates the WebSocket publisher
docker exec ai-trading-bot python -c "
import asyncio
import json
from datetime import datetime

async def test_websocket_publisher():
    try:
        # Import the WebSocket publisher
        from bot.websocket_publisher import WebSocketPublisher
        from bot.config import Settings
        
        # Create settings with WebSocket enabled
        import os
        os.environ['SYSTEM__ENABLE_WEBSOCKET_PUBLISHING'] = 'true'
        os.environ['SYSTEM__WEBSOCKET_DASHBOARD_URL'] = 'ws://dashboard-backend:8000/ws'
        
        settings = Settings()
        publisher = WebSocketPublisher(settings)
        
        # Initialize connection
        print('Initializing WebSocket publisher...')
        connected = await publisher.initialize()
        
        if connected:
            print('✅ WebSocket publisher initialized')
            
            # Send test messages
            await publisher.publish_system_status('testing', True)
            print('✅ System status message sent')
            
            await publisher.publish_market_data('BTC-USD', 45000.0)
            print('✅ Market data message sent')
            
            await publisher.publish_ai_decision('HOLD', 'Test decision', 0.85)
            print('✅ AI decision message sent')
            
            # Close connection
            await publisher.close()
            print('✅ WebSocket publisher closed cleanly')
            return True
        else:
            print('❌ Failed to initialize WebSocket publisher')
            return False
            
    except Exception as e:
        print(f'❌ WebSocket publisher test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

success = asyncio.run(test_websocket_publisher())
exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    echo "✅ WebSocket message flow test passed"
else
    echo "❌ WebSocket message flow test failed"
fi

# Step 6: Check dashboard backend logs for WebSocket activity
echo
echo "Step 6: Checking Dashboard Backend Logs..."

echo "Recent WebSocket-related logs:"
docker logs dashboard-backend 2>&1 | grep -i websocket | tail -10

echo
echo "=== WebSocket Integration Test Summary ==="
echo
echo "If all tests passed, WebSocket communication is working correctly."
echo "If any tests failed, check the following:"
echo "1. Ensure all services are running: docker-compose up -d"
echo "2. Check environment variables in docker-compose.yml"
echo "3. Verify network connectivity between containers"
echo "4. Review service logs for errors: docker-compose logs -f"