# AI Trading Bot Dashboard - Integration Tests

This directory contains comprehensive integration tests for the AI Trading Bot Dashboard, focusing on WebSocket connections, API endpoints, and UI components running in Docker environment.

## Test Suite Overview

### 🧪 Test Components

1. **WebSocket Integration Tests** (`websocket-test.js`)
   - Connection establishment and stability
   - Real-time message handling
   - Echo ping-pong functionality
   - Error handling and recovery
   - Mock data injection

2. **REST API Integration Tests** (`api-test.js`)
   - All API endpoint validation
   - Response format verification
   - Error handling testing
   - CORS header validation
   - Performance testing

3. **UI Component Tests** (`ui-test.html`)
   - Component rendering validation
   - Data binding verification
   - Event handling testing
   - TradingView integration
   - Responsive design checks

4. **Docker Environment Validation** (`docker-validator.js`)
   - Container health checks
   - Network connectivity validation
   - Volume mounting verification
   - Resource usage monitoring
   - Log accessibility testing

5. **Mock Data Generation** (`mock-data-generator.js`)
   - Realistic trading data simulation
   - LLM decision event generation
   - Market condition simulation
   - Technical indicator calculation
   - Performance metrics generation

## 🚀 Quick Start

### Run All Tests
```bash
# Main test script that orchestrates everything
./test-docker.sh

# Or run the comprehensive test suite
node test/run-all-tests.js
```

### Run Individual Test Suites
```bash
# WebSocket tests only
node test/websocket-test.js

# API tests only
node test/api-test.js

# Docker validation only
node test/docker-validator.js

# Generate mock data
node test/mock-data-generator.js session 3600000
```

## 📋 Prerequisites

- Docker and Docker Compose installed
- Node.js (18+ recommended)
- curl (for HTTP testing)
- Dashboard containers running

## 🔧 Test Configuration

Edit `test/test-config.json` to customize:

```json
{
  "services": {
    "backend": {
      "url": "http://localhost:8000",
      "timeout": 10000
    },
    "frontend": {
      "url": "http://localhost:3000"
    }
  },
  "websocket": {
    "url": "ws://localhost:8000/ws",
    "expectedMessageTypes": ["echo", "llm_decision", "tradingview_update"]
  },
  "thresholds": {
    "minimumPassRate": 80,
    "maxResponseTime": 5000
  }
}
```

## 📊 Test Reports

Tests generate comprehensive reports in multiple formats:

- **JSON Reports**: `test-reports/integration-test-report-{timestamp}.json`
- **HTML Reports**: `test-reports/integration-test-report-{timestamp}.html`
- **Console Output**: Real-time test execution logs

### Report Structure
```json
{
  "testRun": {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "duration": 120000,
    "environment": "docker"
  },
  "summary": {
    "totalSuites": 4,
    "passedSuites": 3,
    "failedSuites": 1,
    "successRate": 75
  },
  "results": {
    "websocket": { "success": true },
    "api": { "success": true },
    "ui": { "success": true },
    "docker": { "success": false, "error": "Container not running" }
  }
}
```

## 🎯 Test Coverage

### WebSocket Tests
- ✅ Connection establishment
- ✅ Message echo functionality
- ✅ Real-time data flow
- ✅ Connection stability
- ✅ Error handling
- ✅ Mock data injection

### API Tests
- ✅ Health check endpoint
- ✅ Status endpoint
- ✅ Trading data endpoint
- ✅ Logs endpoint
- ✅ LLM monitoring endpoints
- ✅ TradingView UDF endpoints
- ✅ Error response handling
- ✅ CORS headers
- ✅ Response validation

### UI Tests
- ✅ Component rendering
- ✅ Data binding
- ✅ Event handling
- ✅ Chart integration
- ✅ Real-time updates

### Docker Tests
- ✅ Container status
- ✅ Network connectivity
- ✅ Volume mounting
- ✅ Resource usage
- ✅ Inter-service communication
- ✅ Log accessibility

## 🔍 Mock Data Features

The mock data generator creates realistic:

- **OHLCV Market Data**: Candlestick data with proper high/low/open/close relationships
- **Technical Indicators**: RSI, MACD, moving averages, Bollinger bands
- **LLM Decisions**: Context-aware trading decisions with rationales
- **Position Updates**: Realistic position sizing and P&L calculations
- **Performance Metrics**: Win rates, profit factors, Sharpe ratios
- **Market Conditions**: Bull, bear, sideways, and volatile market simulations

### Mock Data Commands
```bash
# Generate LLM trading decision
node test/mock-data-generator.js llm-decision BTC-USD

# Generate OHLCV data
node test/mock-data-generator.js ohlcv BTC-USD 5m 100

# Generate complete trading session
node test/mock-data-generator.js session 3600000

# Generate technical indicators
node test/mock-data-generator.js indicators BTC-USD

# Stream multiple events
node test/mock-data-generator.js stream 10
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection Refused Errors**
   ```bash
   # Check if containers are running
   docker-compose ps
   
   # Restart containers
   docker-compose restart
   ```

2. **WebSocket Connection Fails**
   ```bash
   # Check backend logs
   docker-compose logs dashboard-backend
   
   # Verify WebSocket endpoint
   curl -v http://localhost:8000/health
   ```

3. **UI Tests Not Accessible**
   ```bash
   # Check frontend status
   curl -I http://localhost:3000
   
   # Copy UI test file manually
   cp test/ui-test.html frontend/public/test/
   ```

4. **Docker Validation Fails**
   ```bash
   # Check Docker daemon
   docker info
   
   # Validate compose file
   docker-compose config
   ```

### Test Debugging

Enable verbose output:
```bash
./test-docker.sh --verbose

# Or set debug environment
DEBUG=* node test/run-all-tests.js
```

View detailed logs:
```bash
# Follow test execution
tail -f test-reports/integration-test-report-*.json

# Check individual test status files
ls test/.{*}_test_passed
```

## 📈 Performance Benchmarks

Expected performance thresholds:

- **API Response Time**: < 1000ms for health checks
- **WebSocket Connection**: < 5s establishment time
- **UI Load Time**: < 3s for main dashboard
- **Container Startup**: < 30s for all services
- **Test Suite Execution**: < 5 minutes total

## 🔒 Security Testing

Basic security validations included:

- CORS header verification
- Error message sanitization
- Input validation testing
- Network isolation checks

## 🤝 Contributing

To add new tests:

1. Create test file in `test/` directory
2. Follow naming convention: `{component}-test.js`
3. Update `test-config.json` with new test suite
4. Add test to `run-all-tests.js` execution flow
5. Document test in this README

### Test Template
```javascript
#!/usr/bin/env node

// Test configuration
const CONFIG = {
    // Test-specific configuration
};

// Test results tracking
const TEST_RESULTS = {
    // Individual test results
};

// Test functions
async function testSomething() {
    // Test implementation
}

// Main test execution
async function runTests() {
    // Orchestrate test execution
}

if (require.main === module) {
    runTests().catch(error => {
        console.error('Test failed:', error.message);
        process.exit(1);
    });
}
```

## 📝 Test Maintenance

Regular maintenance tasks:

- Update mock data generation for new features
- Adjust performance thresholds based on infrastructure
- Review and update API endpoint tests for new endpoints
- Validate Docker configuration changes
- Update UI tests for new components

## 🎨 Customization

### Environment-Specific Configuration

Create environment-specific config files:

```bash
# Development environment
cp test-config.json test-config.dev.json

# Production environment  
cp test-config.json test-config.prod.json
```

Use with:
```bash
TEST_CONFIG=test-config.prod.json node test/run-all-tests.js
```

### Custom Test Suites

Disable/enable specific test suites in `test-config.json`:

```json
{
  "testSuites": {
    "websocket": { "enabled": true },
    "api": { "enabled": true },
    "ui": { "enabled": false },
    "docker": { "enabled": true }
  }
}
```

This comprehensive test suite ensures the AI Trading Bot Dashboard works correctly in the Docker environment with full validation of WebSocket connections, API endpoints, and UI components.