# WebSocket Integration Test Suite

Comprehensive Docker-based testing for the real-time WebSocket data bridge between the AI trading bot and dashboard.

## 🚀 Quick Start

```bash
# Run all tests (recommended)
./tests/docker/run_all_tests.sh

# View results
cat tests/docker/results/full_suite_*/test_report.md
```

## 📋 Test Components

### 1. **Environment Setup** (`setup_test_environment.sh`)
- Creates test configuration files
- Builds Docker images
- Validates network setup
- Checks port availability

### 2. **Connectivity Tests** (`test_websocket_connectivity.py`)
- Basic WebSocket handshake
- Message exchange verification
- Ping/pong latency
- Concurrent connections
- Large message handling

### 3. **Message Flow Validation** (`test_message_flow.py`)
- Schema validation for all message types
- Message ordering verification
- Data integrity checks
- End-to-end flow testing

### 4. **Fault Tolerance** (`test_fault_tolerance.sh`)
- Dashboard offline at startup
- Dashboard crash simulation
- Network partition testing
- Message queue overflow
- Rapid reconnection cycles

### 5. **Load Testing** (`test_websocket_load.py`)
- Sustained message load
- Burst patterns
- Concurrent connections
- Resource monitoring
- Performance metrics

### 6. **Integration Testing** (`integration_test.sh`)
- Full trading cycle simulation
- Real-time monitoring
- Log analysis
- Metrics collection

### 7. **Performance Monitoring** (`measure_latency.py`)
- End-to-end latency measurement
- Container resource tracking
- Message rate analysis
- Bottleneck identification

## 🛠️ Individual Test Execution

```bash
# Setup only
./tests/docker/setup_test_environment.sh

# Start services with test config
docker-compose --env-file .env.test up -d

# Run specific tests
python tests/docker/test_websocket_connectivity.py
python tests/docker/test_message_flow.py
./tests/docker/test_fault_tolerance.sh
python tests/docker/test_websocket_load.py
./tests/docker/integration_test.sh
python tests/docker/measure_latency.py --duration 60

# Stop services
docker-compose down
```

## 📊 Test Configuration

### Environment Variables

Control which tests run:
```bash
export RUN_CONNECTIVITY=true
export RUN_MESSAGE_FLOW=true
export RUN_FAULT_TOLERANCE=true
export RUN_LOAD_TESTS=true
export RUN_INTEGRATION=true
export RUN_PERFORMANCE=true

./tests/docker/run_all_tests.sh
```

### Test Parameters

Configure via `.env.test`:
```bash
# WebSocket settings
SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true
SYSTEM__WEBSOCKET_DASHBOARD_URL=ws://dashboard-backend:8000/ws
SYSTEM__WEBSOCKET_MAX_RETRIES=5
SYSTEM__WEBSOCKET_RETRY_DELAY=5
SYSTEM__WEBSOCKET_QUEUE_SIZE=100

# Test settings
LOG_LEVEL=DEBUG
TEST_MODE=true
```

## 📈 Success Criteria

All tests pass when:
- ✅ Services start and connect successfully
- ✅ Messages flow with <100ms latency (P99)
- ✅ System handles failures gracefully
- ✅ No message loss during reconnections
- ✅ Trading continues if dashboard fails
- ✅ Resource usage remains stable
- ✅ All message schemas validated
- ✅ Security tests pass

## 🔍 Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check ports
   lsof -i :3000,8000,8765,8767,8081
   
   # Kill processes
   docker-compose down
   docker system prune -f
   ```

2. **WebSocket connection failures**
   ```bash
   # Check logs
   docker-compose logs dashboard-backend
   docker-compose logs ai-trading-bot
   
   # Verify config
   grep WEBSOCKET .env.test
   ```

3. **Test failures**
   ```bash
   # View specific test logs
   cat tests/docker/results/*/fault_tolerance.log
   
   # Check container health
   docker-compose ps
   docker stats
   ```

## 🚨 CI/CD Integration

### GitHub Actions Example

```yaml
name: WebSocket Integration Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Docker
        uses: docker/setup-buildx-action@v1
      
      - name: Run WebSocket tests
        run: |
          ./tests/docker/run_all_tests.sh
        
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: tests/docker/results/
```

## 📝 Test Results

Results are saved to timestamped directories:
```
tests/docker/results/
├── full_suite_YYYYMMDD_HHMMSS/
│   ├── test_summary.json         # Machine-readable summary
│   ├── test_report.md           # Human-readable report
│   ├── connectivity.log         # Individual test logs
│   ├── message_flow.log
│   ├── fault_tolerance.log
│   ├── load_tests.log
│   ├── integration.log
│   └── performance.log
```

## 🎯 Next Steps

After successful tests:

1. **Deploy to production**
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

2. **Monitor WebSocket health**
   ```bash
   curl http://localhost:8000/api/websocket/status
   ```

3. **Set up alerts**
   - High latency (>1s)
   - Connection failures
   - Message queue overflow
   - High error rates

4. **Performance tuning**
   - Adjust queue sizes
   - Optimize message batching
   - Scale horizontally if needed