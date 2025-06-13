# VuManChu E2E Docker Testing Suite

This comprehensive end-to-end testing suite validates the complete VuManChu implementation in a Docker environment, ensuring production readiness through extensive testing scenarios.

## Overview

The E2E testing suite provides:

- **Complete Pipeline Testing**: Full data flow from market data input to signal output
- **Performance Benchmarking**: Load testing and scalability validation
- **Memory Testing**: Long-running memory consumption monitoring
- **Signal Quality Validation**: Comprehensive signal generation accuracy testing
- **Integration Testing**: Component interaction validation
- **Error Recovery Testing**: Fault tolerance and resilience testing
- **Docker Environment Testing**: Container-specific validation

## Architecture

```
tests/
├── test_e2e_vumanchu_docker.py          # Main E2E test suite
├── data/
│   ├── generate_test_market_data.py     # Test data generator
│   └── __init__.py                      # Data package init
└── README_E2E_DOCKER_TESTING.md        # This documentation

docker/
├── test-compose.yml                     # Docker test configuration
└── test-config.yml                      # Test parameters and thresholds

scripts/
└── run_docker_tests.py                 # Test orchestration script
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.12+ with Poetry
- At least 2GB RAM and 5GB disk space available

### Running Tests

1. **Setup Environment**:
   ```bash
   # Validate Docker environment
   python scripts/run_docker_tests.py setup
   ```

2. **Generate Test Data**:
   ```bash
   # Generate test data for all scenarios
   python scripts/run_docker_tests.py generate-data
   ```

3. **Run Comprehensive Test Suite**:
   ```bash
   # Run all tests (recommended)
   python scripts/run_docker_tests.py comprehensive
   ```

4. **Run Specific Test Profile**:
   ```bash
   # Quick tests (minimal)
   python scripts/run_docker_tests.py run-tests --profile quick
   
   # Standard tests (recommended for development)
   python scripts/run_docker_tests.py run-tests --profile standard
   
   # Performance tests only
   python scripts/run_docker_tests.py run-tests --profile performance
   ```

## Test Profiles

### Quick Profile
- **Duration**: ~5 minutes
- **Data**: 1,000 data points, default scenario
- **Coverage**: Basic functionality testing
- **Use Case**: Rapid development validation

### Standard Profile
- **Duration**: ~15 minutes
- **Data**: 1,000-5,000 data points, multiple scenarios
- **Coverage**: Comprehensive functionality + basic performance
- **Use Case**: Pre-commit validation

### Comprehensive Profile
- **Duration**: ~45 minutes
- **Data**: 1,000-10,000 data points, all scenarios
- **Coverage**: Full test suite including performance and integration
- **Use Case**: Release validation

### Performance Profile
- **Duration**: ~30 minutes
- **Data**: 1,000-20,000 data points
- **Coverage**: Scalability, memory, and throughput testing
- **Use Case**: Performance regression testing

## Test Scenarios

### Market Data Scenarios

1. **Default**: Normal market conditions with moderate volatility
2. **Trending**: Strong bullish/bearish trending markets
3. **Ranging**: Sideways market conditions
4. **Volatile**: High volatility periods with clustering
5. **Gap Data**: Markets with gaps and extreme moves
6. **Low Volume**: Low volume trading conditions
7. **Crypto Weekend**: Weekend trading pattern simulation

### Test Categories

#### 1. Full Pipeline Tests (`TestE2EFullPipeline`)
- Complete data flow validation
- Multi-symbol processing
- LLM integration (if API keys available)
- Risk management integration

#### 2. Performance Tests (`TestE2EPerformance`)
- Scalability with increasing data sizes
- Memory usage under load
- Concurrent processing capabilities
- Throughput measurement

#### 3. Signal Quality Tests (`TestE2ESignalQuality`)
- Signal type coverage across market conditions
- Signal timing accuracy
- Signal frequency validation
- Confidence score validation

#### 4. Integration Tests (`TestE2EIntegration`)
- Complete bot integration
- Configuration loading
- Component interaction validation

#### 5. Error Recovery Tests (`TestE2EErrorRecovery`)
- Malformed data handling
- Resource exhaustion recovery
- Network timeout simulation

#### 6. Docker Environment Tests (`TestE2EDockerEnvironment`)
- Docker health checks
- Volume mount accessibility
- Network configuration validation

## Performance Criteria

### Throughput Requirements
- **Small datasets** (≤500 rows): >1,000 rows/second
- **Medium datasets** (501-2,000 rows): >500 rows/second
- **Large datasets** (>2,000 rows): >200 rows/second

### Memory Requirements
- **Peak memory usage**: <500MB
- **Memory increase per operation**: <200MB
- **Memory leak threshold**: <50MB per iteration

### Signal Quality Requirements
- **Signal frequency**: 1-30% of periods
- **Signal confidence**: 0-100 range
- **Signal interval**: 10-500 minutes between signals

## Usage Examples

### Running Individual Test Categories

```bash
# Performance tests only
docker-compose -f docker/test-compose.yml --profile performance up

# Signal quality tests only  
docker-compose -f docker/test-compose.yml --profile signals up

# Integration tests only
docker-compose -f docker/test-compose.yml --profile integration up

# Error recovery tests only
docker-compose -f docker/test-compose.yml --profile errors up
```

### Custom Test Data Generation

```bash
# Generate specific scenarios and sizes
python scripts/run_docker_tests.py generate-data \
    --scenarios "trending,volatile" \
    --sizes "5000,10000"

# Or use the data generator directly
python tests/data/generate_test_market_data.py \
    --scenarios trending,ranging,volatile \
    --sizes 1000,5000,10000 \
    --output-dir ./custom_test_data
```

### Collecting and Analyzing Results

```bash
# Collect all test results
python scripts/run_docker_tests.py collect-results

# View test logs
python scripts/run_docker_tests.py logs

# View logs for specific service
python scripts/run_docker_tests.py logs --service vumanchu-e2e-tests
```

## Docker Services

### Main Test Services

- **e2e-test-runner**: Main comprehensive test suite
- **performance-test-runner**: Performance and scalability tests
- **memory-test-runner**: Memory usage and leak detection
- **signal-quality-test-runner**: Signal generation validation
- **integration-test-runner**: Component integration tests
- **error-recovery-test-runner**: Error handling and recovery
- **test-data-generator**: Test data generation service
- **test-results-aggregator**: Results collection and aggregation

### Resource Configuration

```yaml
# Standard test runner
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'

# Performance test runner (higher limits)
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

## Test Data

### Generated Data Structure

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,50000.0,50125.5,49875.2,50050.3,150.5
2024-01-01 00:01:00,50050.3,50200.1,49950.0,50100.7,275.8
...
```

### Data Validation

All generated data includes:
- Proper OHLC relationships (High ≥ max(Open,Close), Low ≤ min(Open,Close))
- Realistic volume patterns correlated with price movements
- Positive price values (minimum 0.01)
- Consistent timestamp intervals

## Results and Reporting

### Output Files

```
test_results/
├── e2e_results.xml                    # JUnit XML results
├── performance_results.xml            # Performance test results
├── signal_results.xml                # Signal quality results
├── integration_results.xml           # Integration test results
├── memory_results.xml                # Memory test results
├── error_results.xml                 # Error recovery results
├── coverage/                         # HTML coverage reports
├── e2e_test_report.json             # Comprehensive test report
├── test_summary.json                # Test execution summary
└── collection_summary.json          # Results collection summary
```

### Report Analysis

```python
import json

# Load comprehensive test report
with open('test_results/e2e_test_report.json', 'r') as f:
    report = json.load(f)

print(f"Test Suite: {report['test_suite']}")
print(f"System Info: {report['system_info']}")
print(f"Performance Summary: {report['performance_summary']}")
```

## Troubleshooting

### Common Issues

1. **Docker Memory Issues**:
   ```bash
   # Increase Docker memory limit to 4GB+
   # Check Docker Desktop settings
   ```

2. **Test Data Generation Fails**:
   ```bash
   # Ensure sufficient disk space (5GB+)
   # Check volume mount permissions
   ```

3. **Performance Tests Timeout**:
   ```bash
   # Reduce test data sizes
   # Increase container resource limits
   ```

4. **Integration Tests Fail**:
   ```bash
   # Verify API keys are set (optional)
   # Check network connectivity
   ```

### Debug Mode

```bash
# Enable verbose logging
python scripts/run_docker_tests.py run-tests --profile standard --verbose

# View detailed container logs
docker-compose -f docker/test-compose.yml logs -f vumanchu-e2e-tests

# Access container for debugging
docker-compose -f docker/test-compose.yml run --rm e2e-test-runner bash
```

### Performance Optimization

```bash
# Pre-build images to save time
docker-compose -f docker/test-compose.yml build

# Use cached test data
python scripts/run_docker_tests.py run-tests --no-build

# Run tests in parallel (if system allows)
docker-compose -f docker/test-compose.yml up --scale e2e-test-runner=2
```

## Continuous Integration

### GitHub Actions Integration

```yaml
name: VuManChu E2E Tests
on: [push, pull_request]
jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run E2E Tests
        run: |
          python scripts/run_docker_tests.py comprehensive
      - name: Upload Test Results
        uses: actions/upload-artifact@v2
        with:
          name: e2e-test-results
          path: test_results/
```

### GitLab CI Integration

```yaml
e2e_tests:
  stage: test
  script:
    - python scripts/run_docker_tests.py comprehensive
  artifacts:
    reports:
      junit: test_results/*.xml
    paths:
      - test_results/
```

## Configuration

### Environment Variables

```bash
# Test configuration
export DOCKER_CONTAINER=true
export TESTING=true
export DRY_RUN=true
export LOG_LEVEL=INFO

# Performance testing
export PERFORMANCE_TESTING=true
export MEMORY_TESTING=true

# Integration testing (optional)
export OPENAI_API_KEY=your_key_here
export CB_API_KEY=your_key_here
export CB_API_SECRET=your_secret_here
```

### Test Configuration File

The `docker/test-config.yml` file contains detailed test parameters:

```yaml
performance:
  scalability:
    small_dataset_min_throughput: 1000
    medium_dataset_min_throughput: 500
    large_dataset_min_throughput: 200
  memory:
    max_memory_increase: 200
    max_peak_memory: 500
```

## Contributing

### Adding New Tests

1. Add test methods to appropriate test class in `test_e2e_vumanchu_docker.py`
2. Follow naming convention: `test_feature_description`
3. Include comprehensive logging and assertions
4. Update this documentation

### Adding New Test Scenarios

1. Add scenario to `generate_test_market_data.py`
2. Update `test-config.yml` configuration
3. Add scenario to `test-compose.yml` environment variables
4. Test with multiple data sizes

### Performance Test Guidelines

- Test with increasing data sizes: 1000, 5000, 10000, 20000+
- Measure memory usage and throughput
- Set appropriate performance thresholds
- Include real-time simulation scenarios

## Support

For issues with the E2E testing suite:

1. Check Docker logs: `docker-compose -f docker/test-compose.yml logs`
2. Verify system requirements (memory, disk space)
3. Ensure all required files exist
4. Check test configuration in `docker/test-config.yml`
5. Run individual test categories to isolate issues

## Production Readiness Checklist

- [ ] All E2E tests pass
- [ ] Performance benchmarks met
- [ ] Memory efficiency validated
- [ ] Signal quality confirmed
- [ ] Error handling complete
- [ ] Docker environment validated
- [ ] Integration tests successful
- [ ] Documentation updated

The E2E testing suite ensures the VuManChu implementation is production-ready with comprehensive validation across all components and scenarios.