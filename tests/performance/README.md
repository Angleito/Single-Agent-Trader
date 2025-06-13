# Performance Testing Suite

This directory contains comprehensive performance testing tools for the AI Trading Bot, including benchmarks, load tests, and performance monitoring capabilities.

## Components

### 1. Benchmark Suite (`benchmark_suite.py`)
Comprehensive performance benchmarks for all critical components:
- Indicator calculation performance
- LLM response time benchmarks  
- Market data processing speed tests
- Memory usage profiling
- Database operation benchmarks

### 2. Load Testing (`load_tests.py`)
Stress testing under various load conditions:
- High-frequency market data loads
- Concurrent trading decision stress tests
- Memory leak detection over time
- API rate limit testing
- Error recovery under load

### 3. Performance Monitor (`../bot/performance_monitor.py`)
Real-time performance monitoring:
- Real-time metrics collection
- Latency tracking for critical operations
- Resource usage monitoring
- Performance alerting system
- Bottleneck identification tools

### 4. Test Runner (`run_performance_tests.py`)
Convenient test runner with reporting:
- Run individual or all test suites
- Generate HTML and JSON reports
- Configurable test parameters
- Verbose logging options

## Quick Start

### Install Dependencies
```bash
# Install the performance testing dependencies
pip install psutil memory-profiler

# Or with poetry
poetry install --with dev
```

### Run All Tests
```bash
# Run complete performance test suite
python tests/performance/run_performance_tests.py --all

# Run with custom configuration
python tests/performance/run_performance_tests.py --all \
  --duration 120 \
  --concurrent-users 10 \
  --ops-per-second 15
```

### Run Individual Test Suites

#### Benchmarks Only
```bash
python tests/performance/run_performance_tests.py --benchmarks
```

#### Load Tests Only
```bash
python tests/performance/run_performance_tests.py --load-tests --duration 60
```

### Generate Reports
```bash
# Generate JSON report
python tests/performance/run_performance_tests.py --all --output results.json

# Generate HTML report
python tests/performance/run_performance_tests.py --all --html-report report.html

# Both JSON and HTML
python tests/performance/run_performance_tests.py --all \
  --output results.json \
  --html-report report.html
```

## Performance Monitoring Integration

### Basic Usage
```python
from bot.performance_monitor import track, track_async

# Context manager
with track("operation_name"):
    # Your code here
    pass

# Async decorator
@track_async("llm_response")
async def analyze_market(market_state):
    # Your async code here
    pass

# Sync decorator  
@track_sync("indicator_calculation")
def calculate_indicators(df):
    # Your sync code here
    pass
```

### Advanced Usage
```python
from bot.performance_monitor import (
    PerformanceMonitor, 
    PerformanceThresholds
)

# Initialize with custom thresholds
thresholds = PerformanceThresholds()
thresholds.indicator_calculation_ms = 50  # Stricter threshold

monitor = PerformanceMonitor(thresholds)

# Start monitoring
await monitor.start_monitoring()

# Add alert callback
def handle_alert(alert):
    print(f"Alert: {alert.message}")

monitor.add_alert_callback(handle_alert)

# Get performance summary
summary = monitor.get_performance_summary()
print(f"Health Score: {summary['health_score']}")
```

## Interpreting Results

### Benchmark Results
- **Execution Time**: Total time for all iterations
- **Avg Time Per Iteration**: Average time per single operation
- **Throughput**: Operations per second
- **Memory Usage**: Memory consumed during test
- **Peak Memory**: Maximum memory usage observed

### Load Test Results
- **Operations Per Second**: Sustained throughput under load
- **Response Time Percentiles**: P95/P99 latency distribution
- **Error Rate**: Percentage of failed operations
- **Resource Usage**: CPU and memory consumption
- **Scaling Metrics**: Performance across different load levels

### Performance Monitoring
- **Health Score**: Overall system health (0-100)
- **Latency Metrics**: Response time statistics
- **Resource Metrics**: CPU/memory usage patterns
- **Bottleneck Analysis**: Identified performance issues
- **Recommendations**: Optimization suggestions

## Performance Targets

### Latency Targets
| Operation | Target | Warning | Critical |
|-----------|--------|---------|----------|
| Indicator Calculation | <100ms | >100ms | >500ms |
| LLM Response | <5s | >5s | >10s |
| Market Data Processing | <50ms | >50ms | >200ms |
| Trade Execution | <1s | >1s | >5s |

### Resource Targets
| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Memory Usage | <2GB | >2GB | >4GB |
| CPU Usage | <70% | >70% | >90% |
| Error Rate | <1% | >1% | >5% |

### Throughput Targets
| Operation | Minimum | Target | Peak |
|-----------|---------|--------|------|
| Market Data Updates | 5/sec | 10/sec | 25/sec |
| Indicator Calculations | 2/sec | 5/sec | 12/sec |
| Trading Decisions | 0.5/sec | 1/sec | 4/sec |

## Optimization Guidelines

### If Benchmarks Are Slow
1. **Indicator Calculations**: Vectorize operations, use caching
2. **LLM Responses**: Implement caching, optimize prompts
3. **Data Processing**: Use efficient data structures, batch operations

### If Load Tests Fail
1. **High Error Rate**: Improve error handling, add circuit breakers
2. **High Latency**: Optimize bottlenecks, add async processing  
3. **Memory Issues**: Implement cleanup, use memory pools
4. **CPU Issues**: Optimize algorithms, distribute load

### If Memory Usage Is High
1. **Data Cleanup**: Implement periodic cleanup routines
2. **Caching**: Set cache size limits with TTL
3. **Object Pools**: Reuse objects instead of creating new ones
4. **Streaming**: Process data in streams rather than batches

## Continuous Integration

Add performance testing to your CI/CD pipeline:

```yaml
# .github/workflows/performance.yml
name: Performance Tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install Dependencies
        run: |
          pip install poetry
          poetry install --with dev
      - name: Run Performance Tests
        run: |
          poetry run python tests/performance/run_performance_tests.py \
            --benchmarks \
            --duration 30 \
            --output performance_results.json
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: performance-results
          path: performance_results.json
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Errors**: Reduce test data size or duration
3. **Timeout Errors**: Increase test timeouts for slow systems
4. **Permission Errors**: Ensure write permissions for report files

### Debug Mode
```bash
# Run with verbose logging
python tests/performance/run_performance_tests.py --all --verbose

# Run individual components for debugging
python tests/performance/benchmark_suite.py
python tests/performance/load_tests.py
```

### Performance Issues
1. **Check System Resources**: Ensure adequate CPU/memory
2. **Network Connectivity**: Verify stable internet connection  
3. **Background Processes**: Close unnecessary applications
4. **Test Environment**: Use consistent testing conditions

## Contributing

When adding new performance tests:

1. Follow the existing patterns for metrics collection
2. Add appropriate documentation and examples
3. Update this README with new test descriptions
4. Ensure tests are deterministic and reproducible
5. Add proper error handling and cleanup

For questions or issues, refer to the main project documentation or create an issue in the repository.