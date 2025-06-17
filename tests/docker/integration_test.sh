#!/bin/bash
set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 WebSocket Integration Test - Full Trading Cycle${NC}"
echo -e "${GREEN}==================================================${NC}"

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

print_info() {
    echo -e "${CYAN}[INFO] $1${NC}"
}

# Configuration
TEST_DURATION=${TEST_DURATION:-120}  # 2 minutes default
SYMBOL=${TRADING__SYMBOL:-BTC-USD}
LOG_DIR="tests/docker/results/integration_$(date +%Y%m%d_%H%M%S)"
METRICS_FILE="$LOG_DIR/metrics.json"

# Create log directory
mkdir -p "$LOG_DIR"

# Phase 1: Environment Preparation
print_status "Phase 1: Preparing test environment..."

# Check if .env.test exists
if [ ! -f ".env.test" ]; then
    print_error ".env.test not found. Please run setup_test_environment.sh first."
    exit 1
fi

# Stop any running services
docker-compose down >/dev/null 2>&1 || true

# Phase 2: Service Startup
print_status "Phase 2: Starting all services..."

# Start services with test configuration
docker-compose --env-file .env.test up -d

# Wait for services to be healthy
print_status "Waiting for services to be healthy..."

# Function to check service health
check_service_health() {
    local SERVICE=$1
    local MAX_ATTEMPTS=30
    local ATTEMPT=0
    
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if docker-compose ps $SERVICE 2>/dev/null | grep -q "healthy\|Up"; then
            return 0
        fi
        sleep 2
        ((ATTEMPT++))
    done
    
    return 1
}

# Check each service
SERVICES=("dashboard-backend" "dashboard-frontend" "ai-trading-bot")
for SERVICE in "${SERVICES[@]}"; do
    if check_service_health $SERVICE; then
        print_success "$SERVICE is healthy"
    else
        print_error "$SERVICE failed to start"
        docker-compose logs $SERVICE
        exit 1
    fi
done

# Phase 3: WebSocket Connection Verification
print_status "Phase 3: Verifying WebSocket connections..."

# Check WebSocket status endpoint
WS_STATUS=$(curl -s http://localhost:8000/api/websocket/status || echo "{}")
print_info "WebSocket Status: $WS_STATUS"

# Test WebSocket connectivity
./tests/docker/test_ws_connection.sh
if [ $? -eq 0 ]; then
    print_success "WebSocket connectivity verified"
else
    print_error "WebSocket connectivity test failed"
    exit 1
fi

# Phase 4: Trading Activity Simulation
print_status "Phase 4: Starting trading simulation..."

# Create a monitoring script
cat > "$LOG_DIR/monitor.py" << 'EOF'
import asyncio
import json
import time
import websockets
from datetime import datetime
from collections import defaultdict

class TradingMonitor:
    def __init__(self, duration=120):
        self.duration = duration
        self.metrics = defaultdict(int)
        self.messages = []
        self.start_time = time.time()
        
    async def monitor(self):
        uri = "ws://localhost:8000/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                print(f"Connected to dashboard WebSocket")
                
                while time.time() - self.start_time < self.duration:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        # Track message types
                        msg_type = data.get("type", "unknown")
                        self.metrics[msg_type] += 1
                        
                        # Store message
                        data["received_at"] = datetime.utcnow().isoformat()
                        self.messages.append(data)
                        
                        # Print key events
                        if msg_type == "ai_decision":
                            print(f"AI Decision: {data.get('action')} - {data.get('reasoning', '')[:50]}...")
                        elif msg_type == "trade_execution":
                            print(f"Trade Executed: {data.get('side')} {data.get('size')} @ {data.get('price')}")
                        elif msg_type == "position_update":
                            print(f"Position Update: P&L {data.get('pnl', 0):.2f} ({data.get('pnl_percentage', 0):.2%})")
                            
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"Error: {e}")
                        
        except Exception as e:
            print(f"Connection failed: {e}")
            
        # Save results
        self.save_results()
        
    def save_results(self):
        results = {
            "duration": time.time() - self.start_time,
            "message_counts": dict(self.metrics),
            "total_messages": sum(self.metrics.values()),
            "messages_per_second": sum(self.metrics.values()) / (time.time() - self.start_time),
            "unique_message_types": list(self.metrics.keys()),
            "sample_messages": self.messages[-10:] if self.messages else []
        }
        
        with open("metrics.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nMonitoring complete:")
        print(f"Total messages: {results['total_messages']}")
        print(f"Message rate: {results['messages_per_second']:.2f} msg/s")
        print(f"Message types: {', '.join(results['unique_message_types'])}")

if __name__ == "__main__":
    monitor = TradingMonitor(duration=int(os.environ.get("TEST_DURATION", "120")))
    asyncio.run(monitor.monitor())
EOF

# Start monitoring in background
print_status "Starting WebSocket monitor..."
cd "$LOG_DIR" && python monitor.py &
MONITOR_PID=$!
cd - >/dev/null

# Capture trading bot logs
print_status "Capturing trading activity..."
docker-compose logs -f ai-trading-bot > "$LOG_DIR/bot.log" 2>&1 &
BOT_LOG_PID=$!

# Capture dashboard logs
docker-compose logs -f dashboard-backend > "$LOG_DIR/dashboard.log" 2>&1 &
DASHBOARD_LOG_PID=$!

# Let the system run
print_status "Running trading simulation for ${TEST_DURATION} seconds..."
print_info "Monitoring real-time data flow between bot and dashboard..."

# Show real-time status
for i in $(seq 1 $TEST_DURATION); do
    if [ $((i % 10)) -eq 0 ]; then
        # Get current metrics
        MSG_COUNT=$(docker-compose logs dashboard-backend 2>/dev/null | grep -c "Broadcasting message" || echo "0")
        print_info "Time: ${i}s - Messages broadcast: $MSG_COUNT"
    fi
    sleep 1
done

# Phase 5: Data Collection
print_status "Phase 5: Collecting test data..."

# Stop monitoring processes
kill $MONITOR_PID 2>/dev/null || true
kill $BOT_LOG_PID 2>/dev/null || true
kill $DASHBOARD_LOG_PID 2>/dev/null || true

# Wait for processes to finish
sleep 5

# Analyze logs
print_status "Analyzing collected data..."

# Create analysis script
cat > "$LOG_DIR/analyze.py" << 'EOF'
import json
import re
from collections import defaultdict

def analyze_logs():
    results = {
        "websocket_events": defaultdict(int),
        "trading_events": defaultdict(int),
        "errors": [],
        "warnings": []
    }
    
    # Analyze bot logs
    try:
        with open("bot.log", "r") as f:
            for line in f:
                # WebSocket events
                if "WebSocket" in line:
                    if "connected" in line.lower():
                        results["websocket_events"]["connections"] += 1
                    elif "disconnect" in line.lower():
                        results["websocket_events"]["disconnections"] += 1
                    elif "publish" in line.lower():
                        results["websocket_events"]["messages_published"] += 1
                
                # Trading events
                if "action" in line.lower() and ("buy" in line.lower() or "sell" in line.lower() or "hold" in line.lower()):
                    results["trading_events"]["decisions"] += 1
                
                # Errors and warnings
                if "ERROR" in line:
                    results["errors"].append(line.strip()[:200])
                elif "WARNING" in line:
                    results["warnings"].append(line.strip()[:200])
    except Exception as e:
        print(f"Error analyzing bot logs: {e}")
    
    # Analyze dashboard logs
    try:
        with open("dashboard.log", "r") as f:
            for line in f:
                if "WebSocket client connected" in line:
                    results["websocket_events"]["client_connections"] += 1
                elif "Broadcasting message" in line:
                    results["websocket_events"]["messages_broadcast"] += 1
    except Exception as e:
        print(f"Error analyzing dashboard logs: {e}")
    
    # Load metrics from monitor
    try:
        with open("metrics.json", "r") as f:
            monitor_metrics = json.load(f)
            results["monitor_metrics"] = monitor_metrics
    except Exception as e:
        print(f"Error loading monitor metrics: {e}")
    
    # Save analysis
    with open("analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Integration Test Analysis ===")
    print(f"\nWebSocket Events:")
    for event, count in results["websocket_events"].items():
        print(f"  {event}: {count}")
    
    print(f"\nTrading Events:")
    for event, count in results["trading_events"].items():
        print(f"  {event}: {count}")
    
    if "monitor_metrics" in results:
        print(f"\nMessage Flow:")
        print(f"  Total messages: {results['monitor_metrics']['total_messages']}")
        print(f"  Message rate: {results['monitor_metrics']['messages_per_second']:.2f} msg/s")
        print(f"  Message types: {len(results['monitor_metrics']['unique_message_types'])}")
    
    print(f"\nErrors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")
    
    return results

if __name__ == "__main__":
    analyze_logs()
EOF

cd "$LOG_DIR" && python analyze.py
cd - >/dev/null

# Phase 6: Validation
print_status "Phase 6: Validating test results..."

# Check if key files exist
VALIDATION_PASSED=true
VALIDATION_ERRORS=()

# Check metrics file
if [ -f "$METRICS_FILE" ]; then
    print_success "Metrics collected successfully"
    
    # Validate message flow
    MSG_COUNT=$(jq -r '.total_messages // 0' "$METRICS_FILE")
    if [ "$MSG_COUNT" -gt 0 ]; then
        print_success "Message flow verified: $MSG_COUNT messages"
    else
        print_error "No messages captured"
        VALIDATION_PASSED=false
        VALIDATION_ERRORS+=("No WebSocket messages captured")
    fi
else
    print_error "Metrics file not found"
    VALIDATION_PASSED=false
    VALIDATION_ERRORS+=("Metrics collection failed")
fi

# Check for critical errors in logs
ERROR_COUNT=$(grep -c "ERROR" "$LOG_DIR/bot.log" 2>/dev/null || echo "0")
if [ "$ERROR_COUNT" -gt 10 ]; then
    print_error "High error count in bot logs: $ERROR_COUNT"
    VALIDATION_PASSED=false
    VALIDATION_ERRORS+=("High error count: $ERROR_COUNT")
fi

# Phase 7: Cleanup and Summary
print_status "Phase 7: Test cleanup and summary..."

# Stop services
docker-compose down

# Generate final report
cat > "$LOG_DIR/report.md" << EOF
# WebSocket Integration Test Report

**Test Date:** $(date)
**Duration:** ${TEST_DURATION} seconds
**Symbol:** ${SYMBOL}

## Test Results

### Service Health
- Dashboard Backend: ✓
- Dashboard Frontend: ✓
- AI Trading Bot: ✓

### WebSocket Connectivity
- Initial Connection: ✓
- Message Flow: $([ "$MSG_COUNT" -gt 0 ] && echo "✓" || echo "✗")
- Error Rate: ${ERROR_COUNT} errors

### Collected Data
- Log Directory: $LOG_DIR
- Bot Logs: bot.log
- Dashboard Logs: dashboard.log
- Metrics: metrics.json
- Analysis: analysis.json

### Validation Status
$(if [ "$VALIDATION_PASSED" = true ]; then
    echo "**✅ All validations passed**"
else
    echo "**❌ Some validations failed:**"
    for error in "${VALIDATION_ERRORS[@]}"; do
        echo "- $error"
    done
fi)

## Recommendations
$(if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "- Investigate high error count in bot logs"
fi)
$(if [ "$MSG_COUNT" -eq 0 ]; then
    echo "- Check WebSocket publisher configuration"
    echo "- Verify SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true"
fi)
EOF

# Print summary
echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}Integration Test Complete${NC}"
echo -e "${GREEN}======================================${NC}"

echo -e "\nResults saved to: ${CYAN}$LOG_DIR${NC}"
echo -e "View report: ${CYAN}cat $LOG_DIR/report.md${NC}"

if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "\n${GREEN}✅ Integration test PASSED${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Integration test FAILED${NC}"
    echo "Errors:"
    for error in "${VALIDATION_ERRORS[@]}"; do
        echo "  - $error"
    done
    exit 1
fi