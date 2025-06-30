# Docker Test Environment for Orderbook Testing

This directory contains a comprehensive Docker-based testing environment specifically designed for orderbook functionality testing. It provides isolated, reproducible test environments with mock services, databases, and performance monitoring.

## 🏗️ Architecture Overview

```
Docker Test Environment
├── Test Runner Containers
│   ├── Main Test Runner (unit, integration, comprehensive)
│   ├── Performance Test Runner (benchmarks, stress tests)
│   ├── Property Test Runner (hypothesis-based testing)
│   └── Stress Test Runner (high-load testing)
├── Mock Services
│   ├── Mock Bluefin Service (DEX simulation)
│   ├── Mock Coinbase Service (CEX simulation)
│   └── Mock Exchange WebSocket (high-frequency data)
├── Infrastructure Services
│   ├── PostgreSQL (test data storage)
│   ├── Redis (caching and pub/sub)
│   ├── Log Aggregator (centralized logging)
│   └── Metrics Collector (performance monitoring)
└── Test Data & Results
    ├── Test Results Volume
    ├── Test Cache Volume
    └── Persistent Data Volumes
```

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ available RAM
- 10GB+ available disk space

### Running All Tests

```bash
# Run comprehensive orderbook test suite
./tests/docker/scripts/run_orderbook_tests.sh

# Run specific test categories
./tests/docker/scripts/run_orderbook_tests.sh unit
./tests/docker/scripts/run_orderbook_tests.sh integration
./tests/docker/scripts/run_orderbook_tests.sh performance
./tests/docker/scripts/run_orderbook_tests.sh stress
```

### Manual Docker Compose Operations

```bash
# Start all services
docker-compose -f docker-compose.test.yml up --build

# Run tests manually
docker-compose -f docker-compose.test.yml run test-runner

# Run specific test profiles
docker-compose -f docker-compose.test.yml --profile unit-tests up
docker-compose -f docker-compose.test.yml --profile performance-tests up
docker-compose -f docker-compose.test.yml --profile stress-tests up

# Stop and cleanup
docker-compose -f docker-compose.test.yml down -v
```

## 📦 Components

### Test Runner Containers

#### Main Test Runner (`test-runner`)
- **Purpose**: Orchestrates comprehensive orderbook testing
- **Image**: `orderbook-test-runner:latest`
- **Features**:
  - Python 3.12 with all testing dependencies
  - Pytest with multiple plugins and extensions
  - Coverage reporting and code quality tools
  - Property-based testing with Hypothesis
  - Performance benchmarking capabilities
- **Resources**: 2GB RAM, 1 CPU core
- **Command**: Runs all orderbook-related tests

#### Specialized Test Runners
- **Unit Test Runner**: Isolated unit test execution
- **Integration Test Runner**: Cross-component integration testing
- **Performance Test Runner**: Benchmarking and performance analysis
- **Stress Test Runner**: High-load and stress testing

### Mock Services

#### Mock Bluefin Service (`mock-bluefin`)
- **Purpose**: Simulates Bluefin DEX API and WebSocket feeds
- **Port**: 8080 (HTTP), 8083 (WebSocket)
- **Features**:
  - RESTful API endpoints matching Bluefin interface
  - Real-time orderbook WebSocket feeds
  - Configurable market data simulation
  - Error injection for fault tolerance testing
- **Configuration**:
  ```env
  ORDERBOOK_DEPTH=100
  PRICE_VOLATILITY=0.02
  UPDATE_FREQUENCY=10.0
  SIMULATE_LATENCY=true
  SIMULATE_ERRORS=false
  ```

#### Mock Coinbase Service (`mock-coinbase`)
- **Purpose**: Simulates Coinbase Advanced Trade API
- **Port**: 8081 (HTTP), 8084 (WebSocket)
- **Features**:
  - Coinbase Advanced Trade API compatibility
  - Realistic orderbook data with proper formatting
  - Product listings and ticker data
  - Order placement simulation
- **Configuration**:
  ```env
  ORDERBOOK_DEPTH=50
  TICK_SIZE=0.01
  SIMULATE_LATENCY=true
  ```

#### Mock Exchange WebSocket (`mock-exchange`)
- **Purpose**: High-frequency orderbook data simulation
- **Port**: 8082 (WebSocket)
- **Features**:
  - Configurable update frequency (up to 1000 Hz)
  - Multiple symbol support
  - Realistic price movement with random walk
  - Connection stress testing capabilities
- **Configuration**:
  ```env
  UPDATE_FREQUENCY=100
  ORDERBOOK_LEVELS=20
  SYMBOLS=BTC-USD,ETH-USD,SOL-USD
  PRICE_VOLATILITY=0.015
  ```

### Infrastructure Services

#### Test PostgreSQL (`test-postgres`)
- **Purpose**: Test data storage and analytics
- **Port**: 5433 (mapped to avoid conflicts)
- **Database**: `orderbook_test`
- **Features**:
  - Test result storage
  - Performance benchmark tracking
  - Mock data persistence
  - Test analytics and reporting
- **Schema**: See `tests/sql/init.sql`

#### Test Redis (`test-redis`)
- **Purpose**: Caching and message queue testing
- **Port**: 6380 (mapped to avoid conflicts)
- **Features**:
  - Test data caching
  - Pub/sub testing
  - Session storage simulation
- **Configuration**: 128MB memory limit with LRU eviction

#### Log Aggregator (`test-log-aggregator`)
- **Purpose**: Centralized log collection and analysis
- **Port**: 8090 (web UI)
- **Features**:
  - Real-time log aggregation
  - Log analysis and pattern detection
  - Search and filtering capabilities
- **Profile**: `logging` (optional)

#### Metrics Collector (`test-metrics`)
- **Purpose**: Performance and system metrics collection
- **Port**: 8091 (web UI)
- **Features**:
  - Real-time metrics collection
  - Performance trend analysis
  - Resource usage monitoring
- **Profile**: `metrics` (optional)

## 🧪 Test Categories

### Unit Tests
- **Location**: `tests/unit/fp/test_orderbook_*.py`
- **Focus**: Individual orderbook components and functions
- **Coverage**: OrderBook types, validation, calculations
- **Execution**: Fast, isolated, no external dependencies

### Integration Tests
- **Location**: `tests/integration/test_orderbook_*.py`
- **Focus**: Component interaction and data flow
- **Coverage**: WebSocket → OrderBook → Strategy integration
- **Dependencies**: Mock services, database

### Performance Tests
- **Location**: `tests/performance/test_orderbook_performance.py`
- **Focus**: Performance benchmarking and optimization
- **Metrics**: Latency, throughput, memory usage
- **Tools**: pytest-benchmark, memory profiler

### Stress Tests
- **Location**: `tests/stress/test_orderbook_stress.py`
- **Focus**: High-load and edge case testing
- **Scenarios**: Concurrent connections, message floods, resource exhaustion
- **Duration**: Configurable (default: 5 minutes)

### Property-Based Tests
- **Location**: `tests/property/test_orderbook_properties.py`
- **Focus**: Invariant validation and edge case discovery
- **Framework**: Hypothesis
- **Coverage**: OrderBook mathematical properties, data consistency

### Docker-Specific Tests
- **Location**: `tests/docker/test_orderbook_docker.py`
- **Focus**: Container environment validation
- **Coverage**: Service connectivity, environment configuration, resource availability

## 📊 Test Configuration

### Environment Variables

#### Core Testing
```env
TEST_MODE=true
LOG_LEVEL=DEBUG
PYTHONPATH=/app
PYTEST_TIMEOUT=300
```

#### Mock Services
```env
MOCK_BLUEFIN_URL=http://mock-bluefin:8080
MOCK_COINBASE_URL=http://mock-coinbase:8081
MOCK_EXCHANGE_WS_URL=ws://mock-exchange:8082/ws
```

#### Database
```env
TEST_DB_HOST=test-postgres
TEST_DB_PORT=5432
TEST_DB_NAME=orderbook_test
TEST_DB_USER=test_user
TEST_DB_PASSWORD=test_password
```

#### Performance Testing
```env
BENCHMARK_ITERATIONS=1000
LOAD_TEST_DURATION=60
STRESS_TEST_CONNECTIONS=100
STRESS_TEST_MESSAGES_PER_SECOND=1000
```

### Pytest Configuration
- **File**: `tests/docker/config/pytest.ini`
- **Features**:
  - Comprehensive marker system
  - Coverage reporting
  - Performance benchmarking
  - Asyncio support
  - Detailed logging

## 📈 Monitoring and Observability

### Test Results
- **Location**: `/app/test-results/` (in containers)
- **Formats**: HTML, JSON, XML (JUnit)
- **Coverage**: HTML reports with line-by-line analysis
- **Persistence**: Results volume for external access

### Performance Metrics
- **Benchmarks**: JSON format with statistical analysis
- **Profiling**: Memory and CPU profiling reports
- **Trends**: Historical performance tracking

### Logging
- **Level**: DEBUG (configurable)
- **Format**: Structured JSON logging
- **Aggregation**: Centralized log collection
- **Analysis**: Pattern detection and alerting

### Resource Monitoring
- **CPU**: Per-container usage tracking
- **Memory**: Detailed memory profiling
- **Network**: Connection and bandwidth monitoring
- **Disk**: I/O performance analysis

## 🔧 Customization

### Adding New Mock Services
1. Create Dockerfile in `tests/docker/mocks/`
2. Implement service in Python with FastAPI
3. Add service to `docker-compose.test.yml`
4. Update test configuration and scripts

### Custom Test Categories
1. Create test files with appropriate markers
2. Add pytest configuration in `pytest.ini`
3. Update test runner scripts
4. Configure Docker Compose profiles

### Performance Tuning
- **Resource Limits**: Adjust in `docker-compose.test.yml`
- **Concurrency**: Configure pytest workers and async settings
- **Timeouts**: Adjust test and connection timeouts
- **Cache Settings**: Configure Redis and filesystem caching

## 🚨 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check for conflicting processes
lsof -i :8080,8081,8082,5433,6380

# Stop conflicting services
docker-compose down
docker system prune -f
```

#### Memory Issues
```bash
# Check container resource usage
docker stats

# Increase resource limits in docker-compose.test.yml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```

#### Network Issues
```bash
# Check network connectivity
docker-compose exec test-runner ping mock-bluefin
docker-compose exec test-runner nc -zv mock-coinbase 8081

# Rebuild network
docker-compose down
docker network prune -f
docker-compose up
```

#### Test Failures
```bash
# View detailed logs
docker-compose logs test-runner
docker-compose logs mock-bluefin

# Run tests with debug output
docker-compose run test-runner pytest tests/docker/ -vvv --tb=long

# Check health status
docker-compose ps
docker-compose exec test-runner /app/healthcheck.sh
```

### Performance Issues
- **Slow Tests**: Reduce iteration counts or test scope
- **High Memory Usage**: Implement test isolation and cleanup
- **Network Latency**: Configure mock services for faster responses
- **Disk I/O**: Use tmpfs for temporary data

## 📝 Best Practices

### Test Development
1. **Isolation**: Each test should be independent
2. **Cleanup**: Properly tear down resources after tests
3. **Determinism**: Use fixed seeds for reproducible results
4. **Documentation**: Comment complex test scenarios

### Container Management
1. **Resource Limits**: Set appropriate CPU and memory limits
2. **Health Checks**: Implement comprehensive health checks
3. **Graceful Shutdown**: Handle SIGTERM for clean shutdown
4. **Security**: Run containers with non-root users

### Performance Optimization
1. **Parallel Execution**: Use pytest-xdist for parallel tests
2. **Caching**: Leverage Redis and filesystem caching
3. **Connection Pooling**: Reuse database and HTTP connections
4. **Batch Operations**: Group related operations together

## 🔍 Monitoring Dashboard

Access the monitoring interfaces:

- **Test Results**: `http://localhost:8090` (Log Aggregator)
- **Performance Metrics**: `http://localhost:8091` (Metrics Collector)
- **Database Admin**: Connect to `localhost:5433` with any PostgreSQL client
- **Redis CLI**: `redis-cli -h localhost -p 6380`

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

For questions or issues, refer to the main project documentation or create an issue in the project repository.