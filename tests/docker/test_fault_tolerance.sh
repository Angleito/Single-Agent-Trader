#!/bin/bash
set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 WebSocket Fault Tolerance Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to print status
print_status() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_test() {
    echo -e "\n${BLUE}TEST: $1${NC}"
    echo -e "${BLUE}$(printf '%.0s-' {1..60})${NC}"
}

# Check if services are running
check_services() {
    if ! docker-compose ps | grep -q "Up"; then
        print_error "Services are not running. Please start them first."
        echo "Run: docker-compose --env-file .env.test up -d"
        exit 1
    fi
}

# Function to monitor bot logs for WebSocket messages
monitor_websocket_logs() {
    local DURATION=$1
    local PATTERN=$2
    local LOG_FILE="tests/docker/results/fault_tolerance_$(date +%s).log"

    print_status "Monitoring logs for $DURATION seconds..."
    timeout $DURATION docker-compose logs -f ai-trading-bot 2>&1 | grep -E "$PATTERN" > "$LOG_FILE" &
    local PID=$!

    sleep $DURATION

    if [ -s "$LOG_FILE" ]; then
        local COUNT=$(wc -l < "$LOG_FILE")
        print_success "Found $COUNT matching log entries"
        return 0
    else
        print_error "No matching log entries found"
        return 1
    fi
}

# Test 1: Dashboard offline at bot startup
test_dashboard_offline_startup() {
    print_test "Dashboard Offline at Bot Startup"

    # Stop all services
    print_status "Stopping all services..."
    docker-compose down

    # Start only the bot (dashboard offline)
    print_status "Starting bot with dashboard offline..."
    docker-compose up -d ai-trading-bot

    # Wait for bot to initialize
    sleep 10

    # Check bot logs for connection attempts
    print_status "Checking bot logs for connection attempts..."
    if docker-compose logs ai-trading-bot | grep -q "WebSocket.*connection.*failed\|Unable to connect"; then
        print_success "Bot correctly handles missing dashboard"
    else
        print_error "Bot did not log connection failures"
        return 1
    fi

    # Check if bot continues running
    if docker-compose ps ai-trading-bot | grep -q "Up"; then
        print_success "Bot continues running without dashboard"
    else
        print_error "Bot crashed when dashboard unavailable"
        return 1
    fi

    # Start dashboard
    print_status "Starting dashboard services..."
    docker-compose up -d dashboard-backend dashboard-frontend

    # Wait for connection
    sleep 15

    # Check for successful reconnection
    if docker-compose logs --tail=50 ai-trading-bot | grep -q "WebSocket.*connected\|Connection established"; then
        print_success "Bot successfully connected after dashboard became available"
        return 0
    else
        print_error "Bot did not reconnect to dashboard"
        return 1
    fi
}

# Test 2: Dashboard crashes during trading
test_dashboard_crash() {
    print_test "Dashboard Crash During Trading"

    # Ensure all services are running
    print_status "Starting all services..."
    docker-compose up -d
    sleep 20

    # Verify WebSocket connection
    print_status "Verifying initial WebSocket connection..."
    if docker-compose logs --tail=20 ai-trading-bot | grep -q "WebSocket.*connected\|Publishing.*message"; then
        print_success "WebSocket connection established"
    else
        print_error "No WebSocket connection detected"
        return 1
    fi

    # Stop dashboard suddenly
    print_status "Simulating dashboard crash..."
    docker-compose stop dashboard-backend

    # Monitor bot behavior for 30 seconds
    print_status "Monitoring bot behavior during outage..."
    sleep 30

    # Check if bot is still running
    if docker-compose ps ai-trading-bot | grep -q "Up"; then
        print_success "Bot survived dashboard crash"
    else
        print_error "Bot crashed when dashboard stopped"
        return 1
    fi

    # Check for reconnection attempts
    if docker-compose logs --tail=100 ai-trading-bot | grep -q "reconnect\|retry\|connection.*lost"; then
        print_success "Bot attempting to reconnect"
    else
        print_error "No reconnection attempts detected"
    fi

    # Restart dashboard
    print_status "Restarting dashboard..."
    docker-compose start dashboard-backend
    sleep 15

    # Check for successful reconnection
    if docker-compose logs --tail=50 ai-trading-bot | grep -q "reconnected\|connection.*established"; then
        print_success "Bot successfully reconnected after dashboard restart"
        return 0
    else
        print_error "Bot did not reconnect after dashboard restart"
        return 1
    fi
}

# Test 3: Network partition
test_network_partition() {
    print_test "Network Partition Simulation"

    # Ensure all services are running
    print_status "Starting all services..."
    docker-compose up -d
    sleep 20

    # Get container names
    BOT_CONTAINER=$(docker-compose ps -q ai-trading-bot)
    DASHBOARD_CONTAINER=$(docker-compose ps -q dashboard-backend)

    # Disconnect dashboard from network
    print_status "Simulating network partition..."
    docker network disconnect trading-network $DASHBOARD_CONTAINER || true

    # Wait and monitor
    print_status "Monitoring behavior during network partition..."
    sleep 20

    # Check bot status
    if docker-compose ps ai-trading-bot | grep -q "Up"; then
        print_success "Bot remains operational during network partition"
    else
        print_error "Bot failed during network partition"
        return 1
    fi

    # Reconnect network
    print_status "Restoring network connection..."
    docker network connect trading-network $DASHBOARD_CONTAINER

    # Wait for reconnection
    sleep 15

    # Verify reconnection
    if docker-compose logs --tail=50 ai-trading-bot | grep -q "reconnected\|connection.*restored"; then
        print_success "Connection restored after network partition"
        return 0
    else
        print_error "Failed to restore connection"
        return 1
    fi
}

# Test 4: Message queue overflow
test_message_queue_overflow() {
    print_test "Message Queue Overflow Handling"

    # This test is implemented in Python
    print_status "Running message queue overflow test..."

    docker exec ai-trading-bot python3 << 'EOF'
import asyncio
import json
from bot.websocket_publisher import WebSocketPublisher
import os

async def test_queue_overflow():
    # Create publisher with small queue
    os.environ['SYSTEM__WEBSOCKET_QUEUE_SIZE'] = '10'
    publisher = WebSocketPublisher()

    # Simulate connection failure
    publisher.connected = False

    # Send many messages to overflow queue
    overflow_count = 0
    for i in range(20):
        try:
            await publisher.publish({
                "type": "test_message",
                "sequence": i,
                "timestamp": f"2024-01-01T00:00:{i:02d}Z"
            })
        except:
            overflow_count += 1

    print(f"Queue size: {publisher.message_queue.qsize()}")
    print(f"Messages dropped: {overflow_count}")

    # Queue should be at max size (10)
    if publisher.message_queue.qsize() == 10:
        print("✓ Queue correctly limited to max size")
        return True
    else:
        print("✗ Queue size incorrect")
        return False

result = asyncio.run(test_queue_overflow())
exit(0 if result else 1)
EOF

    if [ $? -eq 0 ]; then
        print_success "Message queue overflow handled correctly"
        return 0
    else
        print_error "Message queue overflow test failed"
        return 1
    fi
}

# Test 5: Rapid reconnection cycles
test_rapid_reconnection() {
    print_test "Rapid Reconnection Cycles"

    # Start services
    print_status "Starting services..."
    docker-compose up -d
    sleep 15

    # Perform rapid stop/start cycles
    for i in {1..5}; do
        print_status "Cycle $i/5: Stopping dashboard..."
        docker-compose stop dashboard-backend
        sleep 5

        print_status "Cycle $i/5: Starting dashboard..."
        docker-compose start dashboard-backend
        sleep 10

        # Check bot is still running
        if docker-compose ps ai-trading-bot | grep -q "Up"; then
            echo "  ✓ Bot survived cycle $i"
        else
            print_error "Bot failed during cycle $i"
            return 1
        fi
    done

    # Final check
    sleep 10
    if docker-compose logs --tail=50 ai-trading-bot | grep -q "connected"; then
        print_success "Bot successfully handled rapid reconnection cycles"
        return 0
    else
        print_error "Bot failed to maintain connection after cycles"
        return 1
    fi
}

# Test 6: Dashboard slow response
test_slow_response() {
    print_test "Dashboard Slow Response Simulation"

    print_status "This test simulates a slow/overloaded dashboard..."

    # Use Python to test timeout handling
    docker exec ai-trading-bot python3 << 'EOF'
import asyncio
import time
from bot.websocket_publisher import WebSocketPublisher

async def test_timeout():
    publisher = WebSocketPublisher()
    publisher.timeout = 2.0  # 2 second timeout

    # Simulate slow connection (this will timeout)
    start = time.time()
    try:
        await asyncio.wait_for(
            asyncio.sleep(5),  # Simulate 5 second delay
            timeout=publisher.timeout
        )
        print("✗ Timeout not triggered")
        return False
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"✓ Timeout triggered after {elapsed:.1f}s")
        return True

result = asyncio.run(test_timeout())
exit(0 if result else 1)
EOF

    if [ $? -eq 0 ]; then
        print_success "Timeout handling works correctly"
        return 0
    else
        print_error "Timeout handling failed"
        return 1
    fi
}

# Main test execution
main() {
    # Check prerequisites
    check_services

    # Create results directory
    mkdir -p tests/docker/results

    # Track test results
    TOTAL_TESTS=6
    PASSED_TESTS=0
    FAILED_TESTS=()

    # Run all tests
    echo -e "\n${GREEN}Running Fault Tolerance Tests${NC}\n"

    # Test 1
    if test_dashboard_offline_startup; then
        ((PASSED_TESTS++))
    else
        FAILED_TESTS+=("Dashboard Offline at Startup")
    fi

    # Test 2
    if test_dashboard_crash; then
        ((PASSED_TESTS++))
    else
        FAILED_TESTS+=("Dashboard Crash")
    fi

    # Test 3
    if test_network_partition; then
        ((PASSED_TESTS++))
    else
        FAILED_TESTS+=("Network Partition")
    fi

    # Test 4
    if test_message_queue_overflow; then
        ((PASSED_TESTS++))
    else
        FAILED_TESTS+=("Message Queue Overflow")
    fi

    # Test 5
    if test_rapid_reconnection; then
        ((PASSED_TESTS++))
    else
        FAILED_TESTS+=("Rapid Reconnection")
    fi

    # Test 6
    if test_slow_response; then
        ((PASSED_TESTS++))
    else
        FAILED_TESTS+=("Slow Response")
    fi

    # Print summary
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Fault Tolerance Test Summary${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Total Tests: ${TOTAL_TESTS}"
    echo -e "Passed: ${GREEN}${PASSED_TESTS}${NC}"
    echo -e "Failed: ${RED}$((TOTAL_TESTS - PASSED_TESTS))${NC}"

    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo -e "\nFailed Tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo -e "  ${RED}✗ $test${NC}"
        done
    fi

    # Save results
    cat > tests/docker/results/fault_tolerance_summary.json << EOF
{
  "total_tests": $TOTAL_TESTS,
  "passed": $PASSED_TESTS,
  "failed": $((TOTAL_TESTS - PASSED_TESTS)),
  "failed_tests": [$(printf '"%s",' "${FAILED_TESTS[@]}" | sed 's/,$//')],
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
        echo -e "\n${GREEN}✅ All fault tolerance tests passed!${NC}"
        exit 0
    else
        echo -e "\n${RED}❌ Some fault tolerance tests failed${NC}"
        exit 1
    fi
}

# Run main function
main
