# Docker E2E Testing Configuration

This directory contains Docker configuration files for the VuManChu E2E testing suite.

## Files

- **test-compose.yml**: Docker Compose configuration for E2E testing
- **test-config.yml**: Test parameters, thresholds, and validation criteria
- **README.md**: This documentation

## Usage

### Quick Start

```bash
# Run comprehensive E2E tests
python ../scripts/run_docker_tests.py comprehensive

# Run specific test profile
docker-compose -f test-compose.yml --profile performance up
```

### Available Profiles

- `datagen`: Generate test data
- `performance`: Performance and scalability tests
- `memory`: Memory usage tests
- `signals`: Signal quality tests
- `integration`: Integration tests
- `errors`: Error recovery tests

### Resource Configuration

The test environment is configured with resource limits appropriate for testing:

- **Memory**: 1-4GB depending on test type
- **CPU**: 0.5-2.0 cores depending on test type
- **Storage**: Persistent volumes for test data and results

### Environment Variables

Key environment variables for testing:

```bash
DOCKER_CONTAINER=true
TESTING=true
DRY_RUN=true
LOG_LEVEL=INFO
TEST_DATA_DIR=/app/test_data
TEST_RESULTS_DIR=/app/test_results
```

For more detailed documentation, see:
- `../tests/README_E2E_DOCKER_TESTING.md`
- `../scripts/run_docker_tests.py --help`