# Log Validation and Monitoring System

This comprehensive log validation system provides real-time monitoring, analysis, and alerting for Docker container logs with a focus on test execution monitoring.

## Components Overview

### 1. Log Analysis Validator (`log_analysis_validator.py`)
- **Purpose**: Analyzes log files and Docker container logs for quality, coverage, and issues
- **Features**:
  - Pattern matching for different log formats (JSON, detailed, simple, performance)
  - Test result extraction from pytest and custom test formats
  - Performance metrics extraction
  - Log quality validation with scoring
  - Comprehensive reporting with actionable recommendations

### 2. Docker Log Monitor (`docker_log_monitor.py`)
- **Purpose**: Real-time streaming and analysis of Docker container logs
- **Features**:
  - Auto-detection of project containers
  - Real-time test execution tracking
  - Performance monitoring integration
  - Container health status monitoring
  - Structured log event storage (JSONL format)

### 3. Test Result Dashboard (`test_result_dashboard.py`)
- **Purpose**: Web-based dashboard for visualizing test results and system metrics
- **Features**:
  - Interactive charts for performance trends
  - Test execution summaries with pass/fail rates
  - Real-time container status monitoring
  - Log activity visualization
  - SQLite database for historical data

### 4. Alert Manager (`alert_manager.py`)
- **Purpose**: Intelligent alerting system for test failures and system issues
- **Features**:
  - Configurable alert rules and thresholds
  - Multiple notification channels (email, Slack, webhook, console)
  - Alert deduplication and cooldown management
  - Escalation policies
  - Alert history and status tracking

## Quick Start

### 1. Basic Log Analysis
```bash
# Analyze Docker container logs from the last hour
./scripts/log_analysis_validator.py --since 1h

# Analyze specific containers
./scripts/log_analysis_validator.py --containers ai-trading-bot dashboard --since 2h

# Generate detailed report
./scripts/log_analysis_validator.py --output /tmp/log_report.md --format text

# JSON output for programmatic processing
./scripts/log_analysis_validator.py --format json --output /tmp/log_data.json
```

### 2. Real-time Log Monitoring
```bash
# Start monitoring all project containers
./scripts/docker_log_monitor.py

# Monitor specific containers with custom log directory
./scripts/docker_log_monitor.py --containers ai-trading-bot mcp-memory --logs-dir ./logs

# Verbose monitoring
./scripts/docker_log_monitor.py --verbose
```

### 3. Test Result Dashboard
```bash
# Start web dashboard on default port 8080
./scripts/test_result_dashboard.py

# Custom configuration
./scripts/test_result_dashboard.py --port 3000 --host 0.0.0.0 --logs-dir ./logs

# Load existing log data on startup
./scripts/test_result_dashboard.py --load-data --db-path ./test_results.db
```

Access the dashboard at: `http://localhost:8080`

### 4. Alert Management
```bash
# Create sample configuration
./scripts/alert_manager.py --create-config alerts_config.yaml

# Start alert monitoring
./scripts/alert_manager.py --config alerts_config.yaml --logs-dir ./logs

# Check current alert status
./scripts/alert_manager.py --status

# Verbose alerting
./scripts/alert_manager.py --config alerts_config.yaml --verbose
```

## Configuration Files

### 1. Log Validation Rules (`log_validation_rules.yaml`)
- Defines validation thresholds and requirements
- Configures required loggers and error patterns
- Sets performance and security validation rules
- Customizable for different environments

### 2. Logging Configuration (`logging-config.yaml`)
- Structured logging setup for all components
- Multiple formatters (detailed, JSON, performance, alerts)
- Filtering and routing configuration
- Rotation and retention policies

### 3. Alert Configuration
Create with: `./scripts/alert_manager.py --create-config alerts_config.yaml`

Example alert rules:
- Test failure rate > 20%
- Container health checks
- CPU usage > 90%
- Memory usage > 85%
- Excessive error log entries

## Docker Integration

### Environment Variables
```bash
# Container identification
export CONTAINER_NAME="ai-trading-bot"
export HOSTNAME="trading-container-01"

# Monitoring configuration
export MONITOR_INTERVAL=30
export ENABLE_METRICS_ENDPOINT=true
export METRICS_PORT=9090

# Alert thresholds
export ALERT_CPU_WARNING=70
export ALERT_CPU_CRITICAL=85
export ALERT_MEMORY_WARNING=75
export ALERT_MEMORY_CRITICAL=90

# Log levels
export LOG_LEVEL=INFO
```

### Docker Compose Integration
```yaml
services:
  ai-trading-bot:
    environment:
      - CONTAINER_NAME=ai-trading-bot
      - MONITOR_INTERVAL=30
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./scripts/logging-config.yaml:/app/logging-config.yaml

  log-monitor:
    build: .
    command: python scripts/docker_log_monitor.py
    volumes:
      - ./logs:/app/logs
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - ai-trading-bot
```

## Monitoring Areas Covered

### 1. Test Execution
- **Test Discovery**: Automatic detection of pytest and custom test formats
- **Result Tracking**: Pass/fail/skip status with duration and error messages
- **Performance Analysis**: Slow test identification and optimization suggestions
- **Coverage Validation**: Test coverage analysis and reporting

### 2. Error Detection and Debugging
- **Log Level Analysis**: Distribution of log levels across containers
- **Error Pattern Recognition**: Detection of common error patterns
- **Critical Error Alerting**: Immediate notification for critical failures
- **Debug Information Extraction**: Structured error data for troubleshooting

### 3. Performance Metrics
- **Resource Utilization**: CPU, memory, disk usage monitoring
- **Application Performance**: Response times, throughput, latency tracking
- **Container Health**: Docker container status and health checks
- **Trend Analysis**: Historical performance data and pattern recognition

### 4. Configuration Validation
- **Environment Validation**: Configuration parameter verification
- **Service Connectivity**: WebSocket, database, and API connection monitoring
- **Security Compliance**: Sensitive data detection and masking validation
- **Configuration Drift**: Detection of configuration changes

## Output Examples

### Log Analysis Report
```markdown
# Log Analysis and Validation Report
Generated: 2024-01-15T10:30:00

## Executive Summary
- Total log entries analyzed: 15,432
- Total errors found: 23
- Total warnings found: 156
- Average coverage score: 87.5%
- Containers analyzed: 4

## Container: ai-trading-bot
### Metrics
- Log entries: 8,432
- Errors: 12
- Warnings: 67
- Coverage score: 92.1%

### Test Results
- Total tests: 145
- Passed: 142
- Failed: 3

### Failed Tests
- test_exchange_connection (2.45s)
- test_risk_validation (1.23s)
- test_llm_timeout (30.12s)

### Recommendations
- Investigate 3 failed tests: test_exchange_connection, test_risk_validation, test_llm_timeout
- Review and address logged errors
- Optimize 1 slow tests (>30s): test_llm_timeout
```

### Dashboard Metrics
- Real-time test pass/fail rates
- Container resource utilization charts
- Log activity timeline
- Performance trend graphs
- Alert status indicators

### Alert Notifications
```
🚨 ALERT [HIGH]
Title: High Test Failure Rate
Source: test_monitor
Time: 2024-01-15T10:45:23
Message: Test failure rate is 23.5% (4/17) in the last 1h, exceeding threshold of 20.0%
```

## Best Practices

### 1. Log Quality
- Use structured logging (JSON format preferred)
- Include contextual information (container, module, function)
- Implement proper log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Avoid logging sensitive information

### 2. Performance Monitoring
- Set appropriate alert thresholds for your environment
- Monitor trends rather than just point-in-time values
- Include business metrics alongside technical metrics
- Regularly review and adjust monitoring configuration

### 3. Test Monitoring
- Ensure comprehensive test coverage
- Monitor test execution time and optimize slow tests
- Track test reliability and flakiness
- Implement proper test isolation and cleanup

### 4. Alert Management
- Configure appropriate notification channels
- Implement alert deduplication and cooldown periods
- Use escalation policies for critical alerts
- Regularly review alert effectiveness and reduce noise

## Troubleshooting

### Common Issues

1. **No Docker Containers Found**
   ```bash
   # Check Docker daemon is running
   docker ps

   # Verify container names
   docker ps --format "table {{.Names}}\t{{.Status}}"
   ```

2. **Permission Denied on Log Files**
   ```bash
   # Fix log directory permissions
   sudo chown -R $USER:$USER ./logs
   chmod 755 ./logs
   ```

3. **Dashboard Not Loading Data**
   ```bash
   # Manually load data from logs
   ./scripts/test_result_dashboard.py --load-data

   # Check log file formats
   head -n 5 ./logs/*-tests.jsonl
   ```

4. **Alerts Not Triggering**
   ```bash
   # Check alert status
   ./scripts/alert_manager.py --status

   # Test with verbose output
   ./scripts/alert_manager.py --verbose
   ```

### Log Format Requirements

For optimal analysis, ensure your logs follow these patterns:

**JSON Format (Recommended):**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "level": "INFO",
  "logger": "bot.strategy",
  "module": "llm_agent",
  "function": "make_decision",
  "line": 123,
  "message": "Trading decision: LONG BTC-USD"
}
```

**Detailed Format:**
```
2024-01-15 10:30:00 - bot.strategy - INFO - [llm_agent.py:123] - make_decision() - Trading decision: LONG BTC-USD
```

**Test Result Format:**
```
test_file.py::test_function PASSED [1.23s]
test_file.py::test_function FAILED [2.45s]
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Analyze Test Logs
  run: |
    ./scripts/log_analysis_validator.py --format json --output test_analysis.json

- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-analysis
    path: test_analysis.json
```

### Jenkins Pipeline Example
```groovy
stage('Log Analysis') {
    steps {
        sh './scripts/log_analysis_validator.py --format json --output test_analysis.json'
        archiveArtifacts artifacts: 'test_analysis.json'
    }
}
```

This comprehensive log validation system provides the foundation for maintaining high-quality, reliable test execution and system monitoring in your Docker-based development environment.
