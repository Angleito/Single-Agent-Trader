# Log Validation Suite Installation Guide

## Overview
The log validation suite requires additional Python packages for full functionality. This guide covers installation options for different use cases.

## Quick Installation

### Option 1: Install Required Packages for Full Functionality
```bash
# Install additional packages for log validation suite
pip install docker flask requests pyyaml psutil

# Or using poetry (recommended)
poetry add docker flask requests pyyaml psutil --group dev
```

### Option 2: Core Functionality Only (No Docker Integration)
The scripts can run with limited functionality using only standard library packages:
- Log file analysis (without Docker container integration)
- Basic alerting (console output only)
- Simple test result tracking

## Package Requirements by Feature

### Core Log Analysis (`log_analysis_validator.py`)
- **Required**: Standard library only (json, re, pathlib, datetime, etc.)
- **Optional**: `pyyaml` for YAML output format

### Docker Log Monitor (`docker_log_monitor.py`)
- **Required**: `docker` package for container integration
- **Optional**: `aiohttp` for metrics endpoint

### Test Result Dashboard (`test_result_dashboard.py`)
- **Required**: `flask` for web interface
- **Optional**: `sqlite3` (included in standard library)

### Alert Manager (`alert_manager.py`)
- **Required**: `requests` for webhook notifications, `pyyaml` for configuration
- **Optional**: `smtplib` (standard library) for email alerts

## Installation Commands

### Using Poetry (Recommended)
```bash
# Core packages
poetry add pyyaml requests flask

# Docker integration
poetry add docker

# Optional monitoring packages
poetry add psutil aiohttp
```

### Using pip
```bash
# Core packages
pip install pyyaml requests flask

# Docker integration
pip install docker

# Optional monitoring packages
pip install psutil aiohttp
```

### Using Conda
```bash
# Core packages
conda install pyyaml requests flask

# Docker integration
conda install docker-py

# Optional monitoring packages
conda install psutil aiohttp
```

## Docker-based Installation (Recommended for Production)

### Dockerfile Addition
Add to your existing Dockerfile:
```dockerfile
# Add log validation dependencies
RUN pip install --no-cache-dir \
    docker \
    flask \
    requests \
    pyyaml \
    psutil \
    aiohttp
```

### Docker Compose Service
```yaml
services:
  log-validator:
    build: .
    command: >
      bash -c "
        python scripts/start_log_validation_suite.sh start &&
        tail -f /dev/null
      "
    volumes:
      - ./logs:/app/logs
      - /var/run/docker.sock:/var/run/docker.sock
    ports:
      - "8080:8080"  # Dashboard port
    environment:
      - LOGS_DIR=/app/logs
      - DASHBOARD_PORT=8080
      - LOG_LEVEL=INFO
```

## Verification

### Test Installation
```bash
# Test core functionality
python3 scripts/log_analysis_validator.py --help

# Test Docker integration (requires docker package)
python3 scripts/docker_log_monitor.py --help

# Test dashboard (requires flask)
python3 scripts/test_result_dashboard.py --help

# Test alerts (requires requests, pyyaml)
python3 scripts/alert_manager.py --help
```

### Test Suite Startup
```bash
# Start with dependency check
./scripts/start_log_validation_suite.sh start
```

The startup script will check for missing dependencies and provide installation suggestions.

## Fallback Options

### Limited Functionality Mode
If some packages are unavailable, the scripts provide fallback options:

1. **No Docker Package**: Scripts will analyze log files directly instead of connecting to Docker
2. **No Flask**: Dashboard functionality will be disabled, but analysis continues
3. **No External Packages**: Core analysis using only standard library

### Example: Analysis Without Docker
```bash
# Create sample logs directory with test data
mkdir -p ./logs
echo '{"timestamp": "2024-01-15T10:30:00", "level": "INFO", "message": "Test log entry"}' > ./logs/test-logs.jsonl

# Run analysis on log files only
python3 scripts/log_analysis_validator.py --logs-dir ./logs --format text
```

## Production Deployment

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: log-validator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: log-validator
  template:
    metadata:
      labels:
        app: log-validator
    spec:
      containers:
      - name: log-validator
        image: your-registry/trading-bot:latest
        command: ["./scripts/start_log_validation_suite.sh", "start"]
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: logs-volume
          mountPath: /app/logs
        - name: docker-socket
          mountPath: /var/run/docker.sock
        env:
        - name: DASHBOARD_PORT
          value: "8080"
        - name: LOG_LEVEL
          value: "INFO"
      volumes:
      - name: logs-volume
        persistentVolumeClaim:
          claimName: logs-pvc
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock
```

### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: log-validator-service
spec:
  selector:
    app: log-validator
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
  type: LoadBalancer
```

## Troubleshooting

### Common Installation Issues

1. **Permission Denied on Docker Socket**
   ```bash
   # Add user to docker group (Linux)
   sudo usermod -aG docker $USER
   
   # Or use sudo for testing
   sudo python3 scripts/docker_log_monitor.py
   ```

2. **Module Not Found Errors**
   ```bash
   # Verify Python environment
   which python3
   python3 -m pip list
   
   # Install missing packages
   python3 -m pip install <package_name>
   ```

3. **Port Already in Use**
   ```bash
   # Use different dashboard port
   DASHBOARD_PORT=8081 ./scripts/start_log_validation_suite.sh start
   
   # Or find and kill process using port
   lsof -ti:8080 | xargs kill -9
   ```

4. **Log Directory Permissions**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER ./logs
   chmod 755 ./logs
   ```

### Environment-Specific Notes

#### macOS
- Docker Desktop required for Docker socket access
- May need to install Xcode command line tools: `xcode-select --install`

#### Linux
- Docker must be installed and running
- User must be in docker group or use sudo
- SELinux may require additional configuration

#### Windows
- Use WSL2 for best compatibility
- Docker Desktop with WSL2 backend
- PowerShell or Git Bash for running scripts

## Integration with Existing Project

### Add to pyproject.toml
```toml
[tool.poetry.group.monitoring.dependencies]
docker = "^7.0.0"
flask = "^3.0.0"
requests = "^2.32.0"
pyyaml = "^6.0.1"
psutil = "^6.1.0"
aiohttp = "^3.12.0"
```

### Environment Variables
Add to your `.env` file:
```bash
# Log validation configuration
LOGS_DIR=./logs
DASHBOARD_PORT=8080
LOG_LEVEL=INFO
ENABLE_LOG_VALIDATION=true

# Alert configuration
ALERT_CONFIG=./scripts/alerts_config.yaml
ENABLE_ALERTS=true

# Monitoring thresholds
ALERT_CPU_WARNING=70
ALERT_CPU_CRITICAL=85
ALERT_MEMORY_WARNING=75
ALERT_MEMORY_CRITICAL=90
```

This installation guide ensures you can set up the log validation suite according to your specific requirements and environment constraints.