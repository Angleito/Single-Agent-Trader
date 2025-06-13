# Docker Testing & Validation Infrastructure

This directory contains comprehensive Docker testing tools for the AI Trading Bot that provide:

- **Container lifecycle management and testing**
- **Log validation and analysis**
- **Performance monitoring and optimization**
- **Automated reporting and alerts**

## 🚀 Quick Start

### Basic Validation (1 minute)
```bash
# Quick health check and basic validation
python scripts/run_docker_tests.py --quick

# View logs from trading bot container
python scripts/validate_docker_logs.py --container ai-trading-bot --since 1h

# Monitor performance in real-time
python scripts/monitor_test_performance.py --realtime --duration 5m
```

### Full Testing Suite (5 minutes)
```bash
# Complete end-to-end testing with reports
python scripts/run_docker_tests.py --full --generate-report

# Comprehensive log analysis
python scripts/validate_docker_logs.py --analyze-patterns --export-analysis

# Create performance baseline
python scripts/monitor_test_performance.py --baseline --duration 10m
```

### Performance Benchmarking (30 minutes)
```bash
# Extended performance testing
python scripts/run_docker_tests.py --benchmark --duration=30m --generate-report

# Compare current performance with baseline
python scripts/monitor_test_performance.py --compare-baseline --container ai-trading-bot

# Generate optimization recommendations
python scripts/monitor_test_performance.py --generate-recommendations --container ai-trading-bot
```

## 📊 Tools Overview

### 1. Docker Test Runner (`run_docker_tests.py`)

**Purpose:** Orchestrates comprehensive Docker container testing

**Key Features:**
- Container lifecycle management (start, health check, stop)
- VuManChu indicator validation
- Signal generation testing
- Performance monitoring during tests
- Comprehensive reporting
- Resource usage validation

**Usage Examples:**
```bash
# Quick validation tests (1 minute)
python scripts/run_docker_tests.py --quick

# Full end-to-end tests (5 minutes)
python scripts/run_docker_tests.py --full --generate-report

# Performance benchmark (30 minutes)
python scripts/run_docker_tests.py --benchmark --duration=30m

# Custom duration
python scripts/run_docker_tests.py --full --duration=10m --generate-report
```

**Test Phases:**
1. **Environment Validation** - Docker daemon, compose files, dependencies
2. **Container Startup** - Health checks, service availability
3. **Indicator Validation** - VuManChu Cipher A/B, WaveTrend, EMA Ribbon
4. **Performance Monitoring** - CPU, memory, I/O tracking
5. **Log Validation** - Structure, errors, patterns
6. **Report Generation** - JSON, markdown summaries

### 2. Log Validator (`validate_docker_logs.py`)

**Purpose:** Comprehensive log analysis and validation

**Key Features:**
- Structured log compliance checking
- Pattern analysis for trading signals
- Error and warning categorization
- Performance metric extraction
- Log quality assessment
- Export capabilities (JSON, CSV, Markdown)

**Usage Examples:**
```bash
# Analyze specific container logs
python scripts/validate_docker_logs.py --container ai-trading-bot --since 1h

# Full pattern analysis for all containers
python scripts/validate_docker_logs.py --analyze-patterns

# Export analysis results
python scripts/validate_docker_logs.py --export-analysis --format json

# Analyze logs from specific time range
python scripts/validate_docker_logs.py --since "2024-01-01T10:00:00" --until "2024-01-01T12:00:00"
```

**Analysis Categories:**
- **Basic Statistics** - Line counts, timestamps, character analysis
- **Structured Logs** - JSON compliance, schema validation
- **Pattern Analysis** - Trading signals, indicators, errors
- **Performance Metrics** - Response times, calculation durations
- **Error Analysis** - Categorization, frequency, context

### 3. Performance Monitor (`monitor_test_performance.py`)

**Purpose:** Real-time performance monitoring and optimization

**Key Features:**
- Real-time container performance tracking
- Performance baseline creation and comparison
- Resource usage alerts and thresholds
- Historical performance database
- Optimization recommendations
- Live monitoring dashboard

**Usage Examples:**
```bash
# Real-time monitoring with live dashboard
python scripts/monitor_test_performance.py --realtime --duration 1h

# Create performance baseline
python scripts/monitor_test_performance.py --baseline --duration 1h --baseline-type normal

# Compare current performance with baseline
python scripts/monitor_test_performance.py --compare-baseline --container ai-trading-bot

# Generate optimization recommendations
python scripts/monitor_test_performance.py --generate-recommendations --container ai-trading-bot
```

**Monitoring Metrics:**
- **CPU Usage** - Percentage, spikes, averages
- **Memory Usage** - Current, growth rate, limits
- **Network I/O** - RX/TX bytes, throughput
- **Block I/O** - Read/write operations
- **Container Health** - Status, restart counts, PIDs

## 🔧 Configuration

### Performance Thresholds

Default alert thresholds (configurable in scripts):
```python
alert_thresholds = {
    'cpu_percent': 80.0,              # CPU usage warning
    'memory_percent': 90.0,           # Memory usage warning
    'memory_growth_rate_mb_per_min': 50.0,  # Memory growth alert
    'restart_count_increase': 1       # Container restart alert
}
```

### Log Pattern Validation

The tools validate these log patterns:
- **Startup Patterns** - Service initialization
- **Indicator Patterns** - VuManChu calculations
- **Signal Patterns** - BUY/SELL signals, patterns
- **Trading Patterns** - Position management
- **Error Patterns** - Exceptions, failures

### Test Durations

Pre-configured test durations:
- **Quick:** 60 seconds (basic validation)
- **Full:** 300 seconds (comprehensive testing)
- **Benchmark:** 1800 seconds (performance baseline)

## 📈 Reports and Output

### Test Reports

Generated in `/reports/` directory:
- **JSON Reports** - Machine-readable detailed results
- **Markdown Summaries** - Human-readable summaries
- **CSV Exports** - Tabular data for analysis

**Sample Report Structure:**
```json
{
  "test_metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "test_duration_seconds": 300,
    "docker_version": "24.0.0"
  },
  "test_results": {
    "indicators": {
      "VuManChu_Cipher_A_calculated": true,
      "VuManChu_Cipher_B_calculated": true
    }
  },
  "performance_data": {
    "analysis": {
      "violations": [],
      "warnings": [],
      "summary": {...}
    }
  }
}
```

### Performance Database

SQLite database (`data/performance.db`) stores:
- **Historical metrics** - Time-series performance data
- **Baselines** - Performance benchmarks for comparison
- **Alert history** - Performance violations and warnings

### Log Analysis Exports

Available formats:
- **JSON** - Complete analysis data
- **CSV** - Tabular metrics summary
- **Markdown** - Human-readable reports

## 🚨 Alerts and Monitoring

### Alert Types

1. **CPU Alerts**
   - High CPU usage (>80%)
   - CPU spikes (>95%)

2. **Memory Alerts**
   - High memory usage (>90%)
   - Rapid memory growth (>50MB/min)

3. **Container Alerts**
   - Container restarts
   - Health check failures

### Alert Actions

- **Console Display** - Real-time alerts in monitoring
- **Log Recording** - Alert history in database
- **Report Integration** - Alerts included in test reports

## 🔍 Validation Checks

### VuManChu Indicator Validation

Validates that all indicators calculate correctly:
- **Cipher A** - WaveTrend, EMA Ribbon, RSI+MFI
- **Cipher B** - Signal patterns, divergence detection
- **Support Indicators** - Stochastic RSI, Schaff Trend Cycle

### Signal Quality Validation

Checks for proper signal generation:
- **Diamond Patterns** - Red/Green diamonds
- **Yellow Cross** - Precise timing signals
- **Bull/Bear Patterns** - Candle pattern recognition
- **Divergence Signals** - Price/indicator divergence

### Performance Validation

Ensures containers meet performance requirements:
- **Memory Limits** - Stays within 1GB limit
- **CPU Efficiency** - Below 50% average usage
- **Startup Time** - Containers start within 30 seconds
- **Response Time** - API responses under 5 seconds

## 🛠️ Troubleshooting

### Common Issues

**1. Container Not Found**
```bash
# Check running containers
docker ps -a

# Start containers if needed
docker-compose up -d
```

**2. Permission Denied**
```bash
# Ensure scripts are executable
chmod +x scripts/*.py

# Check Docker daemon access
docker info
```

**3. High Memory Usage**
```bash
# Check container stats
docker stats

# Generate optimization recommendations
python scripts/monitor_test_performance.py --generate-recommendations --container ai-trading-bot
```

**4. Log Analysis Errors**
```bash
# Check container logs directly
docker logs ai-trading-bot --tail 100

# Validate log structure
python scripts/validate_docker_logs.py --container ai-trading-bot --since 10m
```

### Performance Optimization

**If CPU usage is high:**
- Review indicator calculation efficiency
- Check log level settings (DEBUG is CPU intensive)
- Consider increasing CPU limits

**If memory usage is high:**
- Check for memory leaks in indicators
- Review data retention policies
- Monitor memory growth patterns

**If containers restart frequently:**
- Check health check configuration
- Review resource limits
- Investigate crash logs

## 📋 Best Practices

### Regular Testing

1. **Daily Quick Tests**
   ```bash
   python scripts/run_docker_tests.py --quick
   ```

2. **Weekly Full Tests**
   ```bash
   python scripts/run_docker_tests.py --full --generate-report
   ```

3. **Monthly Benchmarks**
   ```bash
   python scripts/run_docker_tests.py --benchmark --generate-report
   ```

### Performance Monitoring

1. **Create Baselines** after major changes
2. **Monitor Trends** with regular performance checks
3. **Review Recommendations** for optimization opportunities

### Log Management

1. **Regular Log Validation** to ensure quality
2. **Pattern Analysis** to catch new issues
3. **Export Analysis** for historical tracking

## 🔗 Integration

### CI/CD Integration

Add to your CI pipeline:
```yaml
# Example GitHub Actions step
- name: Docker Tests
  run: |
    python scripts/run_docker_tests.py --full --generate-report
    python scripts/validate_docker_logs.py --export-analysis
```

### Monitoring Integration

For production monitoring:
```bash
# Continuous performance monitoring
python scripts/monitor_test_performance.py --realtime --duration 24h

# Daily baseline comparison
python scripts/monitor_test_performance.py --compare-baseline --container ai-trading-bot
```

## 📚 Additional Resources

- **Docker Compose Configuration:** `docker-compose.yml`
- **Container Health Checks:** `healthcheck.sh`
- **Performance Database Schema:** Auto-created in `data/performance.db`
- **Log Pattern Definitions:** Embedded in validation scripts

For questions or issues, check the container logs and performance metrics first, then use the validation and monitoring tools to diagnose specific problems.