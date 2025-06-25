# Ubuntu Docker Deployment Testing Guide

This guide covers the comprehensive testing of Ubuntu Docker environments for the AI Trading Bot using the automated deployment test script.

## Overview

The `test-ubuntu-deployment.sh` script provides thorough validation of:
- Docker build process with Ubuntu optimizations
- Container startup and initialization
- Bot module import validation
- Configuration loading and FP runtime
- Log file creation and permissions
- Virtual environment activation
- Docker Compose configurations (simple and full)
- API connectivity testing
- Health check validation

## Quick Start

### Prerequisites

1. **Docker installed and running**:
   ```bash
   docker --version  # Should be 20.10+ recommended
   docker info       # Should show running daemon
   ```

2. **Docker Compose available**:
   ```bash
   docker-compose --version  # Or 'docker compose version'
   ```

3. **Sufficient disk space** (at least 2GB free):
   ```bash
   df -h .  # Check available space
   ```

4. **Network connectivity** for pulling base images and testing APIs

### Basic Usage

```bash
# Run complete test suite
./scripts/test-ubuntu-deployment.sh

# Quick validation (build + startup only)
./scripts/test-ubuntu-deployment.sh quick

# Test Docker build process only
./scripts/test-ubuntu-deployment.sh build-only

# Test health checks and container functionality
./scripts/test-ubuntu-deployment.sh health-only

# Test Docker Compose configurations
./scripts/test-ubuntu-deployment.sh compose-only
```

## Test Categories

### 1. Prerequisites Check
**What it tests:**
- Docker installation and version
- Docker daemon accessibility
- Docker Compose availability
- Available disk space
- Ubuntu container support

**Success criteria:**
- Docker version 20.10 or higher
- Docker daemon responsive
- At least 2GB free disk space
- Can run Ubuntu 22.04 containers

### 2. Docker Build Process
**What it tests:**
- Ubuntu-optimized Dockerfile build
- Multi-stage build process
- Functional Programming runtime setup
- Virtual environment creation
- Dependency installation

**Key features tested:**
- Platform targeting (linux/amd64)
- User ID/Group ID mapping
- FP runtime environment variables
- Poetry dependency management
- Ubuntu package optimization

### 3. Container Startup
**What it tests:**
- Container initialization process
- Environment variable handling
- Volume mount permissions
- Network connectivity setup
- Process startup sequence

**Timeout settings:**
- Container start: 120 seconds
- Health check: 60 seconds
- Overall test: 300 seconds

### 4. Module Import Validation
**What it tests:**
- Core bot module imports
- Functional Programming imports
- Dependency availability
- Python path configuration

**Modules tested:**
```python
# Core modules
from bot.config import load_settings
from bot.main import main
from bot.health import HealthCheckEndpoints

# FP modules
from bot.fp.runtime.interpreter import get_interpreter
from bot.fp.adapters.compatibility_layer import CompatibilityLayer
```

### 5. Configuration Loading
**What it tests:**
- Environment variable processing
- Settings validation
- FP configuration system
- Error handling

**Test configuration:**
```bash
SYSTEM__DRY_RUN=true
EXCHANGE__EXCHANGE_TYPE=bluefin
FP_RUNTIME_ENABLED=true
FP_RUNTIME_MODE=hybrid
```

### 6. Health Check Validation
**What it tests:**
- Basic health check script
- FP runtime health monitoring
- System resource checks
- Process monitoring

**Health check modes tested:**
- `quick` - Process verification
- `fp-debug` - FP runtime debugging
- `system-only` - Resource checks

### 7. Log File Creation
**What it tests:**
- Directory structure creation
- File permission settings
- Write access validation
- FP-specific log directories

**Directories validated:**
```
/app/logs/
├── fp/           # FP runtime logs
├── trades/       # Trading decision logs
└── mcp/          # MCP memory logs

/app/data/
├── fp_runtime/   # FP state persistence
├── paper_trading/# Paper trading data
└── positions/    # Position tracking
```

### 8. Virtual Environment
**What it tests:**
- Python virtual environment activation
- Package availability
- Path configuration
- Pip functionality

**Validation commands:**
```bash
/app/.venv/bin/python --version
/app/.venv/bin/pip list
```

### 9. Docker Compose Testing

#### Simple Compose (`docker-compose.simple.yml`)
**Services tested:**
- `ai-trading-bot` - Main trading service
- `bluefin-service` - DEX trading service

**Configuration:**
- Minimal service setup
- Basic networking
- Essential volume mounts

#### Full Compose (`docker-compose.yml`)
**Services tested:**
- Core trading services
- Optional MCP services
- Network configuration
- Advanced security settings

### 10. API Connectivity
**What it tests:**
- DNS resolution
- HTTPS connectivity
- External API reachability
- Network timeout handling

**APIs tested:**
- Google DNS (connectivity test)
- HTTPBin (HTTPS test)
- Coinbase API (trading API reachability)

## Test Output and Reports

### Real-time Output
The script provides color-coded real-time output:
- 🟢 **Green**: Successful tests
- 🟡 **Yellow**: Warnings (non-critical issues)
- 🔴 **Red**: Errors (critical failures)
- 🔵 **Blue**: Informational messages

### Log Files
**Main log file:** `logs/ubuntu_deployment_test.log`
- Timestamped entries
- Complete test execution log
- Error details and stack traces

**Temporary test files:** `/tmp/ubuntu_deploy_test_TIMESTAMP/`
- Individual test logs
- Build outputs
- Configuration files

### Test Report
**Generated report:** `logs/ubuntu_deployment_test_report_TIMESTAMP.md`

Contains:
- Test summary table
- Detailed results
- Log excerpts
- Recommendations
- Next steps

## Troubleshooting

### Common Issues

#### 1. Docker Build Failures
```bash
# Check Docker daemon
sudo systemctl status docker

# Clean up Docker system
docker system prune -a

# Check disk space
df -h

# Retry with verbose output
./scripts/test-ubuntu-deployment.sh build-only
```

#### 2. Container Startup Issues
```bash
# Check container logs
docker logs ubuntu-test-bot-TIMESTAMP

# Test with debug mode
docker run -it --rm ai-trading-bot:ubuntu-test-TIMESTAMP /bin/bash

# Verify volume permissions
ls -la logs/ data/
```

#### 3. Permission Problems
```bash
# Fix directory permissions
sudo chown -R $(id -u):$(id -g) logs/ data/ tmp/

# Run permission setup script
./setup-docker-permissions.sh

# Check user mapping
id
echo "HOST_UID=$(id -u)" >> .env
echo "HOST_GID=$(id -g)" >> .env
```

#### 4. Network Connectivity Issues
```bash
# Test Docker networking
docker network ls
docker network inspect bridge

# Test DNS resolution
docker run --rm ubuntu:22.04 nslookup google.com

# Check firewall settings
sudo ufw status
```

#### 5. FP Runtime Issues
```bash
# Test FP imports manually
docker exec -it container_name python -c "from bot.fp.runtime.interpreter import get_interpreter; print('OK')"

# Enable FP debug mode
export FP_DEBUG_MODE=true

# Check FP configuration
docker exec container_name python -c "from bot.fp.types.config import Config; print(Config.from_env())"
```

### Advanced Debugging

#### Enable Debug Logging
```bash
# Set debug environment
export LOG_LEVEL=DEBUG
export FP_DEBUG_MODE=true
export TESTING=true

# Run with debug output
./scripts/test-ubuntu-deployment.sh 2>&1 | tee debug.log
```

#### Manual Container Inspection
```bash
# Start container in interactive mode
docker run -it --rm --name debug-container \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  ai-trading-bot:ubuntu-test-TIMESTAMP /bin/bash

# Inside container, run manual tests
cd /app
python -c "from bot.config import load_settings; print(load_settings())"
/app/healthcheck.sh fp-debug
```

## Performance Considerations

### Resource Usage
**Typical resource requirements:**
- **Memory**: 1.5GB per container
- **CPU**: 0.8 cores per container
- **Disk**: 2GB for images + logs/data
- **Network**: Moderate bandwidth for API calls

### Optimization Tips
1. **Use build cache**: Keep Docker layer cache warm
2. **Parallel testing**: Run tests on dedicated test systems
3. **Cleanup regularly**: Remove test containers and images
4. **Monitor resources**: Use `docker stats` during tests

### CI/CD Integration
```yaml
# Example GitHub Actions workflow
- name: Test Ubuntu Deployment
  run: |
    ./scripts/test-ubuntu-deployment.sh quick

- name: Archive test results
  uses: actions/upload-artifact@v3
  with:
    name: ubuntu-test-results
    path: logs/ubuntu_deployment_test_report_*.md
```

## Security Considerations

### Test Environment Safety
- Uses `SYSTEM__DRY_RUN=true` (paper trading mode)
- Mock API keys for testing
- Isolated test networks
- Temporary containers (auto-cleanup)

### Production Validation
```bash
# Before production deployment
1. Run full test suite: ./scripts/test-ubuntu-deployment.sh
2. Review test report thoroughly
3. Test with production-like environment variables
4. Validate with staging environment
5. Monitor resource usage patterns
```

## Integration with Existing Scripts

### Relationship to Other Scripts
- **`setup-docker-permissions.sh`**: Run before deployment testing
- **`ubuntu-docker-validate.sh`**: Complementary validation
- **`validate-deployment.sh`**: Production deployment validation
- **`docker-health-check.sh`**: Runtime health monitoring

### Workflow Integration
```bash
# Complete deployment validation workflow
1. ./setup-docker-permissions.sh      # Setup permissions
2. ./scripts/test-ubuntu-deployment.sh # Test deployment
3. ./scripts/validate-deployment.sh   # Validate production readiness
4. docker-compose up -d               # Deploy services
5. ./scripts/monitor-deployment.sh    # Monitor running services
```

## Next Steps

After successful testing:

1. **Review the generated test report**
2. **Address any warnings or recommendations**
3. **Test with production environment variables** (in secure environment)
4. **Validate with real API keys** (never in version control)
5. **Deploy to staging environment** for integration testing
6. **Monitor production deployment** with health checks
7. **Set up automated testing** in CI/CD pipeline

## Support and Maintenance

### Regular Testing Schedule
- **Daily**: Quick validation (`quick` mode)
- **Weekly**: Full test suite
- **Before deployments**: Complete validation
- **After changes**: Relevant test categories

### Maintenance Tasks
- Update test configurations for new features
- Monitor test execution times
- Clean up old test artifacts
- Review and update documentation

---

For additional support or questions about Ubuntu Docker deployment testing, refer to the main project documentation or check the generated test reports for specific guidance.
