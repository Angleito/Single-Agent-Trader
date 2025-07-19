# Low-Resource Environment Stress Testing Suite

This comprehensive stress testing suite validates the stability and responsiveness of both the AI Trading Bot and Bluefin service under resource-constrained conditions, simulating real-world deployment scenarios.

## 🎯 Objectives

- **Stability Verification**: Ensure services remain operational under memory and CPU constraints
- **Performance Validation**: Confirm acceptable response times under load
- **Resource Monitoring**: Track memory usage, CPU utilization, and I/O patterns
- **Recovery Testing**: Validate system recovery after stress conditions
- **Bottleneck Identification**: Identify performance limits and optimization opportunities

## 🏗️ Architecture

### Core Components

1. **AI Trading Bot** (256MB memory limit, 0.5 CPU cores)
2. **Bluefin Service** (128MB memory limit, 0.3 CPU cores)
3. **Dashboard Backend** (128MB memory limit, 0.3 CPU cores)
4. **Stress Test Runner** (512MB memory limit, 1.0 CPU cores)
5. **Resource Monitor** (64MB memory limit, 0.1 CPU cores)

### Test Scenarios

1. **Bluefin Service Load Test**
   - High-frequency API calls
   - Multiple concurrent operations
   - Error rate monitoring
   - Response time tracking

2. **WebSocket Load Test**
   - Multiple persistent connections
   - Sustained message throughput
   - Connection stability validation
   - Queue overflow handling

3. **Memory Pressure Test**
   - Gradual memory allocation
   - Service responsiveness under pressure
   - Memory leak detection
   - Recovery validation

4. **Post-Stress Recovery Test**
   - Service health verification
   - Performance restoration
   - Stability confirmation

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for host-based testing)
- 2GB+ available system memory
- jq (for result parsing, optional)

### Basic Usage

```bash
# Run with default settings (5-minute test)
./run_stress_tests.sh

# Quick test (1-minute duration)
TEST_DURATION=60 ./run_stress_tests.sh

# Extended test with more concurrent users
TEST_DURATION=600 CONCURRENT_USERS=5 ./run_stress_tests.sh

# Enable resource monitoring
ENABLE_MONITORING=true ./run_stress_tests.sh
```

### Advanced Configuration

```bash
# Use custom configuration
python3 stress_test_low_resource.py stress_test_config.json

# Docker-based execution
docker-compose -f docker-compose.stress-test.yml up --profile stress-test

# Monitor resources during testing
docker-compose -f docker-compose.stress-test.yml up --profile monitoring
```

## 📊 Test Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STRESS_TEST_ENV` | `low_resource` | Test environment identifier |
| `TEST_DURATION` | `300` | Test duration in seconds |
| `CONCURRENT_USERS` | `3` | Number of concurrent operations |
| `OPERATIONS_PER_SECOND` | `5` | Target operations per second |
| `MEMORY_LIMIT_MB` | `512` | Memory limit for testing |
| `CPU_LIMIT_PERCENT` | `80` | CPU usage threshold |

### Configuration File

The `stress_test_config.json` file provides detailed control over:

- Resource constraints and limits
- Test scenario parameters
- Service endpoint configuration
- Monitoring thresholds
- Reporting options

Example configuration:

```json
{
  "resource_constraints": {
    "memory_limit_mb": 512,
    "cpu_limit_percent": 80,
    "max_concurrent_operations": 3
  },
  "test_parameters": {
    "test_duration_seconds": 300,
    "concurrent_users": 3,
    "operations_per_second": 5
  }
}
```

## 📈 Monitoring and Metrics

### Resource Metrics

- **Memory Usage**: RSS, peak usage, growth rate
- **CPU Utilization**: Average, peak, sustained load
- **I/O Operations**: Disk read/write, network send/receive
- **Service Health**: Response times, error rates, availability

### Performance Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Response Time | <1000ms | 95th percentile API response time |
| Error Rate | <10% | Maximum acceptable failure rate |
| Memory Growth | <100MB | Maximum memory increase during test |
| CPU Usage | <90% | Sustained CPU utilization limit |

### Health Checks

- Service availability (HTTP 200 responses)
- WebSocket connectivity
- Container health status
- Docker resource usage

## 📋 Test Results

### Report Structure

```json
{
  "test_suite": "Low-Resource Environment Stress Test",
  "timestamp": "2024-01-15T10:30:00Z",
  "duration_seconds": 300,
  "total_tests": 4,
  "successful_tests": 4,
  "failed_tests": 0,
  "test_results": [...],
  "summary": {
    "overall_success": true,
    "success_rate_percent": 100,
    "avg_response_time_ms": 250,
    "max_memory_mb": 180,
    "recommendations": [...]
  }
}
```

### Success Criteria

- ✅ All tests complete without critical failures
- ✅ Error rate below 10% for most scenarios
- ✅ Response times within acceptable limits
- ✅ Memory usage within constraints
- ✅ Services recover properly after stress

### Output Files

- `stress_test_report_YYYYMMDD_HHMMSS.json` - Detailed test results
- `stress_test_results/*.log` - Container logs
- `stress_test.log` - Test execution log

## 🔧 Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check container status
docker ps -a --filter "name=stress"

# View container logs
docker logs ai-trading-bot-stress
docker logs bluefin-service-stress
docker logs dashboard-backend-stress
```

#### Memory Limit Exceeded

```bash
# Reduce concurrent users
CONCURRENT_USERS=2 ./run_stress_tests.sh

# Shorter test duration
TEST_DURATION=120 ./run_stress_tests.sh
```

#### High Error Rates

```bash
# Lower operations per second
OPERATIONS_PER_SECOND=3 ./run_stress_tests.sh

# Check service configuration
docker exec ai-trading-bot-stress env | grep -E "(MEMORY|CPU|LIMIT)"
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python3 stress_test_low_resource.py

# Monitor real-time resources
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## 🎛️ Customization

### Adding New Test Scenarios

1. Extend the `LowResourceStressTester` class
2. Implement new test method following the pattern:
   ```python
   async def test_custom_scenario(self) -> StressTestResult:
       # Test implementation
       return result
   ```
3. Add to test sequence in `run_comprehensive_stress_test()`

### Modifying Resource Limits

Edit `docker-compose.stress-test.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 256M  # Adjust as needed
      cpus: '0.5'   # Adjust as needed
```

### Custom Endpoints

Set environment variables:

```bash
export BLUEFIN_ENDPOINT="http://custom-bluefin:8080"
export DASHBOARD_ENDPOINT="http://custom-dashboard:8000"
export WEBSOCKET_ENDPOINT="ws://custom-dashboard:8000/ws"
```

## 📊 Performance Baselines

### Expected Performance (Low-Resource Environment)

| Scenario | Target Throughput | Max Response Time | Max Error Rate |
|----------|------------------|-------------------|----------------|
| Bluefin Load | 5 ops/sec | 500ms | 10% |
| WebSocket | 2 connections | 1000ms | 15% |
| Memory Pressure | N/A | N/A | 20% |
| Recovery | 80% success | 2000ms | 20% |

### Resource Usage Targets

| Component | Memory | CPU | Description |
|-----------|--------|-----|-------------|
| Trading Bot | <200MB | <50% | Core application |
| Bluefin Service | <100MB | <30% | DEX integration |
| Dashboard | <100MB | <30% | Web interface |
| **Total System** | **<400MB** | **<110%** | **Combined usage** |

## 🚨 Alerts and Notifications

### Failure Conditions

- Service health check failures (>3 consecutive)
- Memory usage exceeding 90% of limit
- CPU usage >95% for >30 seconds
- Error rate >20% for any test scenario
- Response time >2000ms for >50% of operations

### Recovery Procedures

1. **Service Restart**: Automatic container restart on health check failure
2. **Resource Cleanup**: Memory pressure release after test completion
3. **Graceful Degradation**: Reduced operation rate on resource exhaustion
4. **Monitoring Alerts**: Log warnings for threshold violations

## 📚 Additional Resources

- [Docker Resource Management](https://docs.docker.com/config/containers/resource_constraints/)
- [Performance Testing Best Practices](./docs/performance-testing.md)
- [System Monitoring Guide](./docs/monitoring.md)
- [Troubleshooting Guide](./docs/troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your test scenarios or improvements
4. Ensure all tests pass
5. Submit a pull request

### Test Development Guidelines

- Follow existing patterns for test methods
- Include comprehensive error handling
- Document test objectives and success criteria
- Validate resource cleanup
- Provide meaningful metrics and logging

---

## 📞 Support

For issues or questions regarding the stress testing suite:

1. Check the troubleshooting section above
2. Review container logs for error details
3. Validate configuration parameters
4. Open an issue with test results and logs

**Happy Stress Testing! 🚀**
