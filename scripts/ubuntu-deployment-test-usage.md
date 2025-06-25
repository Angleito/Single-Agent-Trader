# Ubuntu Deployment Test Script - Quick Usage Guide

## Quick Commands

```bash
# Make script executable (first time only)
chmod +x ./scripts/test-ubuntu-deployment.sh

# Run complete test suite (recommended)
./scripts/test-ubuntu-deployment.sh

# Quick validation (5-10 minutes)
./scripts/test-ubuntu-deployment.sh quick

# Test specific components
./scripts/test-ubuntu-deployment.sh build-only
./scripts/test-ubuntu-deployment.sh health-only
./scripts/test-ubuntu-deployment.sh compose-only
```

## What Gets Tested

| Component | Test Coverage |
|-----------|---------------|
| **Docker Build** | ✅ Ubuntu optimization, FP runtime, dependencies |
| **Container Startup** | ✅ Initialization, environment, volumes |
| **Module Imports** | ✅ Core bot modules, FP imports, dependencies |
| **Configuration** | ✅ Environment loading, FP config, validation |
| **Health Checks** | ✅ Basic health, FP runtime, system resources |
| **Logging** | ✅ File creation, permissions, FP logs |
| **Virtual Environment** | ✅ Python activation, pip, package access |
| **Simple Compose** | ✅ Basic Docker Compose configuration |
| **Full Compose** | ✅ Complete Docker Compose with all services |
| **API Connectivity** | ✅ Network access, DNS, external APIs |

## Expected Output

### Successful Run
```
[2024-01-15 10:30:00] Starting Ubuntu Docker Deployment Test Suite
[2024-01-15 10:30:00] ================================================
✅ Test environment initialized
✅ Prerequisites check passed
✅ Docker build completed in 45 seconds
✅ Container started successfully
✅ Module imports test passed
✅ Configuration loading test passed
✅ Basic health check passed
✅ Log file creation test passed
✅ Virtual environment test passed
✅ Simple compose started successfully
✅ Full compose core services started
✅ All tests passed! Ubuntu Docker deployment is ready
```

### Generated Files
- **Test Log**: `logs/ubuntu_deployment_test.log`
- **Test Report**: `logs/ubuntu_deployment_test_report_TIMESTAMP.md`
- **Temporary Files**: `/tmp/ubuntu_deploy_test_TIMESTAMP/`

## Time Estimates

| Test Mode | Duration | Use Case |
|-----------|----------|----------|
| **quick** | 5-10 min | Daily validation, CI/CD |
| **build-only** | 3-5 min | Build process validation |
| **health-only** | 7-12 min | Health check validation |
| **compose-only** | 8-15 min | Docker Compose testing |
| **main** (full) | 15-25 min | Complete validation |

## Prerequisites

1. **Docker** (20.10+) installed and running
2. **Docker Compose** available
3. **2GB+ disk space** available
4. **Network connectivity** for image pulls
5. **Proper permissions** for Docker operations

## Quick Troubleshooting

### Build Fails
```bash
# Clean Docker system
docker system prune -a

# Check disk space
df -h

# Verify Docker daemon
sudo systemctl status docker
```

### Permission Issues
```bash
# Fix permissions
sudo chown -R $(id -u):$(id -g) logs/ data/

# Run setup script
./setup-docker-permissions.sh
```

### Container Won't Start
```bash
# Check logs
docker logs CONTAINER_NAME

# Test manually
docker run -it --rm IMAGE_NAME /bin/bash
```

## Integration Commands

```bash
# Complete validation workflow
./setup-docker-permissions.sh           # Setup
./scripts/test-ubuntu-deployment.sh     # Test
./scripts/validate-deployment.sh        # Validate
docker-compose up -d                     # Deploy
```

## Next Steps After Successful Testing

1. Review the generated test report
2. Address any warnings
3. Test with production environment variables
4. Deploy to staging environment
5. Set up automated testing in CI/CD

---

**Need help?** Check the full documentation: `docs/UBUNTU_DOCKER_DEPLOYMENT_TESTING.md`
