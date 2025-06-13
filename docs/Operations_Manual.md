# AI Trading Bot - Operations Manual

*Version: 1.0.0 | Updated: 2025-06-11*

This manual provides comprehensive guidance for day-to-day operations, monitoring, troubleshooting, and maintenance of the AI Trading Bot in production environments.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Troubleshooting Guide](#troubleshooting-guide)
4. [Performance Tuning](#performance-tuning)
5. [Backup and Recovery](#backup-and-recovery)
6. [Log Analysis](#log-analysis)
7. [Incident Response](#incident-response)
8. [Maintenance Procedures](#maintenance-procedures)

## Daily Operations

### 1. Daily Checklist

#### Morning Startup (8:00 AM)

- [ ] **System Health Check**
  ```bash
  # Check bot status
  docker ps | grep ai-trading-bot
  
  # Verify health endpoints
  curl -s http://localhost:8080/health | jq '.status'
  
  # Check recent logs for errors
  docker logs ai-trading-bot --since=24h | grep -E "(ERROR|CRITICAL)"
  ```

- [ ] **Trading Performance Review**
  ```bash
  # Get daily P&L summary
  docker exec ai-trading-bot python scripts/daily_report.py
  
  # Check position status
  curl -s http://localhost:8080/positions | jq '.'
  
  # Review risk metrics
  curl -s http://localhost:8080/risk/metrics | jq '.'
  ```

- [ ] **Market Conditions Assessment**
  ```bash
  # Check market data connectivity
  curl -s http://localhost:8080/market/status | jq '.'
  
  # Verify indicator calculations
  curl -s http://localhost:8080/indicators/status | jq '.'
  ```

#### Evening Review (6:00 PM)

- [ ] **Performance Analysis**
  - Review daily trading summary
  - Analyze successful vs failed trades
  - Check risk limit adherence
  - Evaluate LLM decision quality

- [ ] **System Maintenance**
  - Check log file sizes
  - Verify backup completion
  - Review resource usage trends
  - Plan any necessary maintenance

#### Weekly Tasks (Sundays)

- [ ] **Comprehensive Review**
  - Weekly P&L analysis
  - Risk parameter optimization
  - Performance metric trends
  - System health assessment

- [ ] **Maintenance Tasks**
  - Log rotation and cleanup
  - Configuration backup
  - Security updates check
  - Documentation updates

### 2. Operational Scripts

#### Daily Report Generator

Create `scripts/daily_report.py`:

```python
#!/usr/bin/env python3
"""Generate daily trading performance report."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from bot.config import create_settings
from bot.health import create_health_endpoints


def generate_daily_report():
    """Generate comprehensive daily report."""
    settings = create_settings()
    health_endpoints = create_health_endpoints(settings)
    
    # Get current metrics
    health = health_endpoints.get_health_detailed()
    metrics = health_endpoints.get_metrics()
    
    report = {
        "date": datetime.now().isoformat(),
        "system_health": health,
        "performance_metrics": metrics,
        "trading_summary": get_trading_summary(),
        "risk_analysis": get_risk_analysis(),
        "recommendations": get_recommendations()
    }
    
    # Save report
    report_file = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
    Path(report_file).parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary to console
    print_report_summary(report)
    
    return report


def get_trading_summary():
    """Get trading activity summary for the day."""
    # Implementation would analyze trading logs and extract:
    # - Number of trades executed
    # - Success rate
    # - P&L for the day
    # - Position changes
    return {
        "trades_executed": 0,
        "success_rate": 0.0,
        "daily_pnl": 0.0,
        "current_positions": []
    }


def get_risk_analysis():
    """Analyze risk metrics and compliance."""
    # Implementation would check:
    # - Daily loss limits
    # - Position size compliance
    # - Risk-adjusted returns
    return {
        "daily_loss_used": 0.0,
        "max_daily_loss": 3.0,
        "position_size_compliance": True,
        "risk_score": "LOW"
    }


def get_recommendations():
    """Generate operational recommendations."""
    recommendations = []
    
    # Add logic to generate recommendations based on:
    # - Performance trends
    # - Risk metrics
    # - System health
    # - Market conditions
    
    return recommendations


def print_report_summary(report):
    """Print formatted report summary."""
    print("=" * 50)
    print(f"AI Trading Bot - Daily Report")
    print(f"Date: {report['date']}")
    print("=" * 50)
    
    # System Health
    health = report['system_health']
    print(f"System Health: {health['status'].upper()}")
    print(f"Uptime: {health.get('uptime', 'N/A')}")
    
    # Trading Performance
    trading = report['trading_summary']
    print(f"\nTrading Performance:")
    print(f"  Trades Executed: {trading['trades_executed']}")
    print(f"  Success Rate: {trading['success_rate']:.1f}%")
    print(f"  Daily P&L: ${trading['daily_pnl']:.2f}")
    
    # Risk Analysis
    risk = report['risk_analysis']
    print(f"\nRisk Analysis:")
    print(f"  Daily Loss Used: {risk['daily_loss_used']:.1f}%")
    print(f"  Risk Score: {risk['risk_score']}")
    
    # Recommendations
    if report['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("=" * 50)


if __name__ == "__main__":
    try:
        generate_daily_report()
    except Exception as e:
        print(f"Error generating daily report: {e}")
        sys.exit(1)
```

#### System Health Monitor

Create `scripts/health_monitor.py`:

```python
#!/usr/bin/env python3
"""Continuous system health monitoring."""

import time
import json
import logging
from datetime import datetime
from bot.health import create_health_endpoints
from bot.config import create_settings


class HealthMonitor:
    """Continuous health monitoring system."""
    
    def __init__(self, check_interval=300):  # 5 minutes
        self.check_interval = check_interval
        self.settings = create_settings()
        self.health_endpoints = create_health_endpoints(self.settings)
        self.logger = logging.getLogger(__name__)
        
        # Health thresholds
        self.thresholds = {
            "memory_usage_mb": 1500,
            "cpu_usage_percent": 80,
            "disk_usage_percent": 85,
            "api_error_rate": 0.1,
            "response_time_ms": 5000
        }
    
    def run_monitoring_loop(self):
        """Run continuous monitoring loop."""
        self.logger.info("Starting health monitoring loop...")
        
        while True:
            try:
                # Perform health check
                health_status = self.perform_health_check()
                
                # Check thresholds and alert if necessary
                alerts = self.check_thresholds(health_status)
                
                if alerts:
                    self.send_alerts(alerts)
                
                # Log health status
                self.log_health_status(health_status)
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            health = self.health_endpoints.get_health_detailed()
            metrics = self.health_endpoints.get_metrics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "health": health,
                "metrics": metrics,
                "status": health.get("status", "unknown")
            }
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }
    
    def check_thresholds(self, health_status):
        """Check if any health metrics exceed thresholds."""
        alerts = []
        
        if health_status.get("status") == "error":
            alerts.append({
                "severity": "critical",
                "message": f"Health check failed: {health_status.get('error')}"
            })
            return alerts
        
        health = health_status.get("health", {})
        metrics = health_status.get("metrics", {})
        
        # Check memory usage
        memory_mb = metrics.get("memory_usage_mb", 0)
        if memory_mb > self.thresholds["memory_usage_mb"]:
            alerts.append({
                "severity": "warning",
                "message": f"High memory usage: {memory_mb}MB (threshold: {self.thresholds['memory_usage_mb']}MB)"
            })
        
        # Check CPU usage
        cpu_percent = metrics.get("cpu_usage_percent", 0)
        if cpu_percent > self.thresholds["cpu_usage_percent"]:
            alerts.append({
                "severity": "warning",
                "message": f"High CPU usage: {cpu_percent}% (threshold: {self.thresholds['cpu_usage_percent']}%)"
            })
        
        # Check API error rate
        error_rate = metrics.get("api_error_rate", 0)
        if error_rate > self.thresholds["api_error_rate"]:
            alerts.append({
                "severity": "critical",
                "message": f"High API error rate: {error_rate:.2%} (threshold: {self.thresholds['api_error_rate']:.2%})"
            })
        
        return alerts
    
    def send_alerts(self, alerts):
        """Send alerts to configured endpoints."""
        for alert in alerts:
            # Send to webhook if configured
            webhook_url = self.settings.system.alert_webhook_url
            if webhook_url:
                self.send_webhook_alert(webhook_url, alert)
            
            # Send to email if configured
            alert_email = self.settings.system.alert_email
            if alert_email:
                self.send_email_alert(alert_email, alert)
            
            # Log the alert
            self.logger.warning(f"ALERT: {alert['message']}")
    
    def send_webhook_alert(self, webhook_url, alert):
        """Send alert to webhook (Slack, Discord, etc.)."""
        import requests
        
        payload = {
            "text": f"🚨 Trading Bot Alert ({alert['severity'].upper()}): {alert['message']}",
            "username": "Trading Bot Monitor"
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def send_email_alert(self, email, alert):
        """Send alert via email."""
        # Implementation would send email using SMTP
        pass
    
    def log_health_status(self, health_status):
        """Log health status for analysis."""
        log_file = "logs/health_monitor.log"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(health_status, default=str) + "\n")


if __name__ == "__main__":
    monitor = HealthMonitor()
    monitor.run_monitoring_loop()
```

## Monitoring and Alerting

### 1. Key Metrics to Monitor

#### System Metrics
- **CPU Usage**: Target <70%, Alert >80%
- **Memory Usage**: Target <80%, Alert >90%
- **Disk Space**: Target <80%, Alert >90%
- **Network Latency**: Target <100ms, Alert >500ms

#### Trading Metrics
- **Daily P&L**: Monitor against targets and limits
- **Trade Success Rate**: Target >60%, Alert <40%
- **Position Size**: Monitor against risk limits
- **API Response Times**: Target <1s, Alert >5s

#### Application Metrics
- **Health Check Status**: Must be "healthy"
- **API Error Rate**: Target <1%, Alert >5%
- **LLM Response Rate**: Target >95%, Alert <90%
- **Market Data Latency**: Target <5s, Alert >30s

### 2. Alert Configuration

#### Slack Integration

Create `scripts/slack_alerter.py`:

```python
#!/usr/bin/env python3
"""Slack alerting integration."""

import json
import requests
from datetime import datetime
from bot.config import create_settings


class SlackAlerter:
    """Send alerts to Slack channels."""
    
    def __init__(self):
        self.settings = create_settings()
        self.webhook_url = self.settings.system.alert_webhook_url
    
    def send_alert(self, message, severity="info", channel=None):
        """Send alert to Slack."""
        if not self.webhook_url:
            print("No webhook URL configured")
            return
        
        # Choose emoji and color based on severity
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
            "success": "✅"
        }
        
        color_map = {
            "info": "#36a64f",
            "warning": "#ff9500",
            "critical": "#ff0000",
            "success": "#00ff00"
        }
        
        payload = {
            "username": "Trading Bot",
            "icon_emoji": ":robot_face:",
            "channel": channel or "#trading-alerts",
            "attachments": [
                {
                    "color": color_map.get(severity, "#36a64f"),
                    "fields": [
                        {
                            "title": f"{emoji_map.get(severity, 'ℹ️')} {severity.upper()} Alert",
                            "value": message,
                            "short": False
                        },
                        {
                            "title": "Timestamp",
                            "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"Alert sent to Slack: {message}")
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
    
    def send_trading_summary(self, summary):
        """Send daily trading summary to Slack."""
        message = f"""
📊 *Daily Trading Summary*

💰 P&L: ${summary.get('daily_pnl', 0):.2f}
📈 Trades: {summary.get('trades_executed', 0)}
✅ Success Rate: {summary.get('success_rate', 0):.1f}%
📊 Open Positions: {len(summary.get('current_positions', []))}

🎯 Risk Usage: {summary.get('daily_loss_used', 0):.1f}% / {summary.get('max_daily_loss', 0):.1f}%
        """
        
        self.send_alert(message, "info", "#trading-reports")
    
    def send_system_alert(self, component, status, details=""):
        """Send system status alert."""
        if status == "healthy":
            message = f"✅ {component} is healthy {details}"
            self.send_alert(message, "success")
        else:
            message = f"🚨 {component} is {status} {details}"
            self.send_alert(message, "critical")


if __name__ == "__main__":
    alerter = SlackAlerter()
    alerter.send_alert("Trading bot monitoring system started", "info")
```

### 3. Grafana Dashboard Configuration

#### Trading Bot Dashboard JSON

Create detailed Grafana dashboard with panels for:

```json
{
  "dashboard": {
    "title": "AI Trading Bot - Operations Dashboard",
    "tags": ["trading", "bot", "operations"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "System Health Overview",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "trading_bot_health_status",
            "legendFormat": "Health Status"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        }
      },
      {
        "title": "Daily P&L",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
        "targets": [
          {
            "expr": "trading_bot_daily_pnl",
            "legendFormat": "Daily P&L"
          }
        ],
        "yAxes": [
          {
            "label": "USD",
            "min": null,
            "max": null
          }
        ]
      },
      {
        "title": "Active Positions",
        "type": "table",
        "gridPos": {"h": 6, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "trading_bot_positions",
            "format": "table"
          }
        ]
      },
      {
        "title": "API Response Times",
        "type": "graph",
        "gridPos": {"h": 6, "w": 12, "x": 12, "y": 6},
        "targets": [
          {
            "expr": "rate(trading_bot_api_request_duration_seconds_sum[5m]) / rate(trading_bot_api_request_duration_seconds_count[5m])",
            "legendFormat": "{{api_endpoint}}"
          }
        ]
      },
      {
        "title": "Resource Usage",
        "type": "graph",
        "gridPos": {"h": 6, "w": 12, "x": 0, "y": 12},
        "targets": [
          {
            "expr": "trading_bot_memory_usage_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)"
          },
          {
            "expr": "trading_bot_cpu_usage_percent",
            "legendFormat": "CPU (%)"
          }
        ]
      },
      {
        "title": "Trading Activity",
        "type": "graph",
        "gridPos": {"h": 6, "w": 12, "x": 12, "y": 12},
        "targets": [
          {
            "expr": "rate(trading_bot_trades_total[1h])",
            "legendFormat": "Trades per hour"
          },
          {
            "expr": "trading_bot_trade_success_rate",
            "legendFormat": "Success rate"
          }
        ]
      }
    ],
    "time": {
      "from": "now-24h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## Troubleshooting Guide

### 1. Common Issues and Solutions

#### Bot Won't Start

**Symptoms:**
- Container fails to start
- Health checks failing
- API connectivity errors

**Diagnosis:**
```bash
# Check container status
docker ps -a | grep ai-trading-bot

# Check logs for startup errors
docker logs ai-trading-bot --since=5m

# Validate configuration
docker exec ai-trading-bot python scripts/validate_config.py

# Test API connectivity
docker exec ai-trading-bot python -c "
from bot.exchange.coinbase import CoinbaseClient
client = CoinbaseClient()
print(client.test_connection())
"
```

**Solutions:**
1. **Configuration Issues:**
   ```bash
   # Check environment variables
   docker exec ai-trading-bot env | grep -E "(API_KEY|SECRET)"
   
   # Validate configuration format
   python -c "from bot.config import create_settings; print('Config OK')"
   ```

2. **API Key Problems:**
   ```bash
   # Test Coinbase API directly
   curl -H "CB-ACCESS-KEY: $CB_API_KEY" \
        -H "CB-ACCESS-SIGN: $CB_ACCESS_SIGN" \
        -H "CB-ACCESS-TIMESTAMP: $CB_ACCESS_TIMESTAMP" \
        -H "CB-ACCESS-PASSPHRASE: $CB_ACCESS_PASSPHRASE" \
        https://api.coinbase.com/api/v3/brokerage/accounts
   ```

3. **Network Connectivity:**
   ```bash
   # Test external connectivity
   docker exec ai-trading-bot ping -c 3 api.coinbase.com
   docker exec ai-trading-bot ping -c 3 api.openai.com
   
   # Check DNS resolution
   docker exec ai-trading-bot nslookup api.coinbase.com
   ```

#### Trading Bot Not Making Trades

**Symptoms:**
- Bot is running but no trades executed
- LLM always returns HOLD action
- Risk manager rejecting all trades

**Diagnosis:**
```bash
# Check LLM agent status
curl -s http://localhost:8080/llm/status | jq '.'

# Check risk manager metrics
curl -s http://localhost:8080/risk/metrics | jq '.'

# Review recent decision logs
docker logs ai-trading-bot | grep -E "(LLM|decision|action)" | tail -20

# Check market data quality
curl -s http://localhost:8080/market/data | jq '.[-5:]'
```

**Solutions:**
1. **LLM Issues:**
   ```bash
   # Test LLM connectivity
   docker exec ai-trading-bot python -c "
   from bot.strategy.llm_agent import LLMAgent
   agent = LLMAgent()
   print(f'LLM Available: {agent.is_available()}')
   print(f'Test Response: {agent.test_connection()}')
   "
   ```

2. **Risk Manager Too Restrictive:**
   - Review risk parameters in configuration
   - Check if daily loss limits have been reached
   - Verify position size calculations

3. **Market Data Issues:**
   ```bash
   # Check data freshness
   docker exec ai-trading-bot python -c "
   from bot.data.market import MarketDataProvider
   provider = MarketDataProvider('BTC-USD', '1m')
   print(f'Last update: {provider.get_last_update()}')
   print(f'Data points: {len(provider.get_latest_ohlcv(10))}')
   "
   ```

#### High Memory Usage

**Symptoms:**
- Memory usage consistently >80%
- Out of memory errors
- Slow performance

**Diagnosis:**
```bash
# Check memory usage
docker stats ai-trading-bot --no-stream

# Check Python memory usage
docker exec ai-trading-bot python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
print(f'CPU: {process.cpu_percent()}%')
"

# Check for memory leaks
docker exec ai-trading-bot python -c "
import gc
print(f'Objects in memory: {len(gc.get_objects())}')
gc.collect()
print(f'After cleanup: {len(gc.get_objects())}')
"
```

**Solutions:**
1. **Data Caching Issues:**
   - Reduce `candle_limit` in configuration
   - Implement data rotation
   - Clear cached indicators periodically

2. **Memory Leaks:**
   - Restart bot periodically
   - Monitor object creation patterns
   - Implement garbage collection

3. **Resource Limits:**
   ```yaml
   # docker-compose.yml
   services:
     ai-trading-bot:
       deploy:
         resources:
           limits:
             memory: 2G
           reservations:
             memory: 1G
   ```

### 2. Debug Mode Operations

#### Enable Debug Logging

```bash
# Temporarily enable debug logging
docker exec ai-trading-bot python -c "
import logging
logging.getLogger().setLevel(logging.DEBUG)
print('Debug logging enabled')
"

# Or restart with debug environment
docker-compose down
SYSTEM__LOG_LEVEL=DEBUG docker-compose up -d
```

#### Interactive Debugging

```bash
# Open Python shell in container
docker exec -it ai-trading-bot python

# Run specific components for testing
docker exec ai-trading-bot python -c "
from bot.indicators.vumanchu import IndicatorCalculator
calc = IndicatorCalculator()
# Test indicator calculations
"
```

### 3. Performance Debugging

#### Profiling Script

Create `scripts/profile_performance.py`:

```python
#!/usr/bin/env python3
"""Performance profiling for trading bot."""

import cProfile
import pstats
import io
from bot.main import TradingEngine


def profile_trading_loop():
    """Profile the main trading loop."""
    pr = cProfile.Profile()
    
    # Create trading engine
    engine = TradingEngine(dry_run=True)
    
    # Profile a single iteration
    pr.enable()
    
    # Simulate trading loop components
    data = engine.market_data.get_latest_ohlcv(200)
    df = engine.market_data.to_dataframe(200)
    indicators = engine.indicator_calc.calculate_all(df)
    
    pr.disable()
    
    # Print results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    print(s.getvalue())


if __name__ == "__main__":
    profile_trading_loop()
```

## Performance Tuning

### 1. System Optimization

#### Resource Allocation

```yaml
# docker-compose.yml - Optimized resource limits
services:
  ai-trading-bot:
    deploy:
      resources:
        limits:
          memory: 2G      # Generous memory limit
          cpus: '1.0'     # Full CPU core
        reservations:
          memory: 1G      # Minimum guaranteed memory
          cpus: '0.5'     # Minimum guaranteed CPU
```

#### Environment Variables for Performance

```env
# Performance tuning
SYSTEM__PARALLEL_PROCESSING=true
SYSTEM__MAX_WORKER_THREADS=4
SYSTEM__UPDATE_FREQUENCY_SECONDS=60.0

# Data optimization
DATA__CANDLE_LIMIT=200
DATA__DATA_CACHE_TTL_SECONDS=30
DATA__INDICATOR_WARMUP=50

# LLM optimization
LLM__MAX_TOKENS=800
LLM__REQUEST_TIMEOUT=20
LLM__ENABLE_CACHING=true
LLM__CACHE_TTL_SECONDS=300
```

### 2. Database Optimization

#### Query Optimization

```python
# Example optimized data queries
class OptimizedDataProvider:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 30  # seconds
    
    def get_ohlcv_cached(self, symbol, limit=200):
        """Get OHLCV data with intelligent caching."""
        cache_key = f"{symbol}_{limit}"
        now = time.time()
        
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return data
        
        # Fetch fresh data
        data = self.fetch_ohlcv(symbol, limit)
        self.cache[cache_key] = (data, now)
        
        return data
```

### 3. Network Optimization

#### Connection Pooling

```python
# Optimized HTTP client with connection pooling
import aiohttp
import asyncio

class OptimizedHTTPClient:
    def __init__(self):
        self.session = None
        self.connector = aiohttp.TCPConnector(
            limit=100,          # Total connection pool size
            limit_per_host=30,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=60
        )
    
    async def get_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout
            )
        return self.session
```

### 4. Application Performance

#### Indicator Calculation Optimization

```python
# Vectorized indicator calculations
import numpy as np
import pandas as pd

class OptimizedIndicators:
    def __init__(self):
        self.cache = {}
    
    def calculate_ema_vectorized(self, data, period):
        """Optimized EMA calculation using numpy."""
        cache_key = f"ema_{period}_{hash(str(data.values.tobytes()))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Vectorized EMA calculation
        alpha = 2.0 / (period + 1.0)
        result = data.ewm(alpha=alpha).mean()
        
        self.cache[cache_key] = result
        return result
    
    def calculate_all_optimized(self, df):
        """Calculate all indicators with optimization."""
        # Use pandas operations instead of loops
        # Cache intermediate results
        # Vectorize calculations where possible
        pass
```

## Backup and Recovery

### 1. Automated Backup System

#### Comprehensive Backup Script

Create `scripts/automated_backup.py`:

```python
#!/usr/bin/env python3
"""Automated backup system for trading bot."""

import os
import json
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
import boto3  # For S3 uploads


class BackupManager:
    """Manage automated backups of trading bot data."""
    
    def __init__(self):
        self.backup_dir = Path("/backups/trading-bot")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup retention policy
        self.retention_days = {
            "daily": 30,
            "weekly": 12,  # 12 weeks
            "monthly": 12  # 12 months
        }
    
    def create_backup(self, backup_type="daily"):
        """Create a complete backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"trading_bot_{backup_type}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        backup_path.mkdir(exist_ok=True)
        
        # Backup configuration
        self.backup_configuration(backup_path)
        
        # Backup logs
        self.backup_logs(backup_path)
        
        # Backup data
        self.backup_data(backup_path)
        
        # Backup database (if applicable)
        self.backup_database(backup_path)
        
        # Create compressed archive
        archive_path = self.create_archive(backup_path)
        
        # Upload to cloud storage
        if os.getenv("AWS_S3_BUCKET"):
            self.upload_to_s3(archive_path)
        
        # Cleanup temporary directory
        shutil.rmtree(backup_path)
        
        # Cleanup old backups
        self.cleanup_old_backups(backup_type)
        
        return archive_path
    
    def backup_configuration(self, backup_path):
        """Backup configuration files."""
        config_backup = backup_path / "config"
        config_backup.mkdir(exist_ok=True)
        
        # Copy config files (without secrets)
        for config_file in Path("config").glob("*.json"):
            shutil.copy2(config_file, config_backup)
        
        # Export current settings (sanitized)
        from bot.config import create_settings
        settings = create_settings()
        settings.save_to_file(config_backup / "current_settings.json", include_secrets=False)
    
    def backup_logs(self, backup_path, days=7):
        """Backup recent log files."""
        logs_backup = backup_path / "logs"
        logs_backup.mkdir(exist_ok=True)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for log_file in Path("logs").glob("*.log"):
            if log_file.stat().st_mtime > cutoff_date.timestamp():
                shutil.copy2(log_file, logs_backup)
    
    def backup_data(self, backup_path):
        """Backup data directory."""
        if Path("data").exists():
            shutil.copytree("data", backup_path / "data")
    
    def backup_database(self, backup_path):
        """Backup database if configured."""
        # Implementation would backup PostgreSQL/SQLite data
        pass
    
    def create_archive(self, backup_path):
        """Create compressed archive of backup."""
        archive_path = backup_path.with_suffix(".tar.gz")
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_path, arcname=backup_path.name)
        
        return archive_path
    
    def upload_to_s3(self, archive_path):
        """Upload backup to S3."""
        try:
            s3_client = boto3.client('s3')
            bucket = os.getenv("AWS_S3_BUCKET")
            key = f"backups/{archive_path.name}"
            
            s3_client.upload_file(str(archive_path), bucket, key)
            print(f"Backup uploaded to S3: s3://{bucket}/{key}")
            
        except Exception as e:
            print(f"Failed to upload to S3: {e}")
    
    def cleanup_old_backups(self, backup_type):
        """Remove old backups based on retention policy."""
        retention_days = self.retention_days.get(backup_type, 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        pattern = f"trading_bot_{backup_type}_*.tar.gz"
        for backup_file in self.backup_dir.glob(pattern):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()
                print(f"Removed old backup: {backup_file}")
    
    def restore_backup(self, backup_path, restore_type="config"):
        """Restore from backup."""
        if not Path(backup_path).exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Extract backup
        with tarfile.open(backup_path, "r:gz") as tar:
            tar.extractall(self.backup_dir)
        
        extracted_dir = self.backup_dir / Path(backup_path).stem.replace(".tar", "")
        
        try:
            if restore_type == "config":
                self.restore_configuration(extracted_dir)
            elif restore_type == "data":
                self.restore_data(extracted_dir)
            elif restore_type == "full":
                self.restore_configuration(extracted_dir)
                self.restore_data(extracted_dir)
            
            print(f"Restore completed: {restore_type}")
            
        finally:
            # Cleanup extracted files
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)
    
    def restore_configuration(self, backup_dir):
        """Restore configuration from backup."""
        config_backup = backup_dir / "config"
        if config_backup.exists():
            # Stop trading bot before restore
            os.system("docker-compose down")
            
            # Backup current config
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move("config", f"config_backup_{timestamp}")
            
            # Restore config
            shutil.copytree(config_backup, "config")
            
            print("Configuration restored. Please restart the trading bot.")
    
    def restore_data(self, backup_dir):
        """Restore data from backup."""
        data_backup = backup_dir / "data"
        if data_backup.exists():
            if Path("data").exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.move("data", f"data_backup_{timestamp}")
            
            shutil.copytree(data_backup, "data")
            print("Data restored.")


if __name__ == "__main__":
    import sys
    
    backup_manager = BackupManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "backup":
            backup_type = sys.argv[2] if len(sys.argv) > 2 else "daily"
            archive_path = backup_manager.create_backup(backup_type)
            print(f"Backup created: {archive_path}")
            
        elif command == "restore":
            if len(sys.argv) < 3:
                print("Usage: backup.py restore <backup_file> [config|data|full]")
                sys.exit(1)
            
            backup_file = sys.argv[2]
            restore_type = sys.argv[3] if len(sys.argv) > 3 else "config"
            backup_manager.restore_backup(backup_file, restore_type)
            
        else:
            print("Usage: backup.py [backup|restore] ...")
    else:
        # Default: create daily backup
        archive_path = backup_manager.create_backup("daily")
        print(f"Daily backup created: {archive_path}")
```

### 2. Backup Scheduling

#### Cron Configuration

```bash
# /etc/cron.d/trading-bot-backup

# Daily backup at 2 AM
0 2 * * * /opt/trading-bot/scripts/automated_backup.py backup daily

# Weekly backup on Sundays at 3 AM
0 3 * * 0 /opt/trading-bot/scripts/automated_backup.py backup weekly

# Monthly backup on the 1st at 4 AM
0 4 1 * * /opt/trading-bot/scripts/automated_backup.py backup monthly
```

## Log Analysis

### 1. Log Analysis Tools

#### Comprehensive Log Analyzer

Create `scripts/log_analyzer.py`:

```python
#!/usr/bin/env python3
"""Advanced log analysis for trading bot."""

import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd


class LogAnalyzer:
    """Analyze trading bot logs for insights and issues."""
    
    def __init__(self, log_file="logs/bot.log"):
        self.log_file = Path(log_file)
        self.patterns = {
            "error": re.compile(r"ERROR.*"),
            "warning": re.compile(r"WARNING.*"),
            "trade": re.compile(r"Trade executed: (\w+) ([\d.]+)% @ \$([\d.]+)"),
            "api_call": re.compile(r"API call to (\w+): (\d+)ms"),
            "position": re.compile(r"Position: (\w+) ([\d.]+)"),
            "pnl": re.compile(r"P&L: \$([\d.-]+)")
        }
    
    def analyze_logs(self, hours=24):
        """Perform comprehensive log analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        results = {
            "summary": self.get_log_summary(cutoff_time),
            "errors": self.analyze_errors(cutoff_time),
            "trading_activity": self.analyze_trading_activity(cutoff_time),
            "performance": self.analyze_performance(cutoff_time),
            "api_metrics": self.analyze_api_metrics(cutoff_time)
        }
        
        return results
    
    def get_log_summary(self, cutoff_time):
        """Get basic log statistics."""
        total_lines = 0
        error_count = 0
        warning_count = 0
        info_count = 0
        
        with open(self.log_file, 'r') as f:
            for line in f:
                total_lines += 1
                
                if "ERROR" in line:
                    error_count += 1
                elif "WARNING" in line:
                    warning_count += 1
                elif "INFO" in line:
                    info_count += 1
        
        return {
            "total_lines": total_lines,
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "error_rate": error_count / max(total_lines, 1)
        }
    
    def analyze_errors(self, cutoff_time):
        """Analyze error patterns and frequency."""
        errors = []
        error_types = Counter()
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if "ERROR" in line or "CRITICAL" in line:
                    # Extract error type
                    if "API" in line:
                        error_types["API Error"] += 1
                    elif "connection" in line.lower():
                        error_types["Connection Error"] += 1
                    elif "timeout" in line.lower():
                        error_types["Timeout Error"] += 1
                    else:
                        error_types["Other Error"] += 1
                    
                    errors.append(line.strip())
        
        return {
            "total_errors": len(errors),
            "error_types": dict(error_types),
            "recent_errors": errors[-10:] if errors else []
        }
    
    def analyze_trading_activity(self, cutoff_time):
        """Analyze trading performance and patterns."""
        trades = []
        pnl_values = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                # Extract trade information
                trade_match = self.patterns["trade"].search(line)
                if trade_match:
                    action, size_pct, price = trade_match.groups()
                    trades.append({
                        "action": action,
                        "size_pct": float(size_pct),
                        "price": float(price),
                        "timestamp": self.extract_timestamp(line)
                    })
                
                # Extract P&L information
                pnl_match = self.patterns["pnl"].search(line)
                if pnl_match:
                    pnl_values.append(float(pnl_match.group(1)))
        
        return {
            "total_trades": len(trades),
            "trade_types": Counter([t["action"] for t in trades]),
            "average_position_size": sum([t["size_pct"] for t in trades]) / max(len(trades), 1),
            "pnl_summary": {
                "current": pnl_values[-1] if pnl_values else 0,
                "max": max(pnl_values) if pnl_values else 0,
                "min": min(pnl_values) if pnl_values else 0,
                "average": sum(pnl_values) / max(len(pnl_values), 1) if pnl_values else 0
            }
        }
    
    def analyze_performance(self, cutoff_time):
        """Analyze system performance metrics."""
        api_times = []
        memory_usage = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                # Extract API response times
                api_match = self.patterns["api_call"].search(line)
                if api_match:
                    endpoint, response_time = api_match.groups()
                    api_times.append({
                        "endpoint": endpoint,
                        "response_time": int(response_time)
                    })
                
                # Extract memory usage
                if "Memory usage:" in line:
                    memory_match = re.search(r"Memory usage: ([\d.]+)", line)
                    if memory_match:
                        memory_usage.append(float(memory_match.group(1)))
        
        return {
            "api_performance": {
                "average_response_time": sum([t["response_time"] for t in api_times]) / max(len(api_times), 1),
                "max_response_time": max([t["response_time"] for t in api_times]) if api_times else 0,
                "slow_requests": len([t for t in api_times if t["response_time"] > 5000])
            },
            "memory_usage": {
                "current": memory_usage[-1] if memory_usage else 0,
                "average": sum(memory_usage) / max(len(memory_usage), 1) if memory_usage else 0,
                "max": max(memory_usage) if memory_usage else 0
            }
        }
    
    def analyze_api_metrics(self, cutoff_time):
        """Analyze API call patterns and success rates."""
        api_calls = Counter()
        api_errors = Counter()
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if "API call" in line:
                    # Count API calls by endpoint
                    for endpoint in ["coinbase", "openai", "anthropic"]:
                        if endpoint in line.lower():
                            api_calls[endpoint] += 1
                            
                            if "ERROR" in line:
                                api_errors[endpoint] += 1
        
        return {
            "api_calls": dict(api_calls),
            "api_errors": dict(api_errors),
            "success_rates": {
                endpoint: 1 - (api_errors.get(endpoint, 0) / max(calls, 1))
                for endpoint, calls in api_calls.items()
            }
        }
    
    def extract_timestamp(self, log_line):
        """Extract timestamp from log line."""
        timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_line)
        if timestamp_match:
            return datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
        return datetime.now()
    
    def generate_report(self, analysis_results):
        """Generate formatted analysis report."""
        report = []
        report.append("=" * 60)
        report.append("AI Trading Bot - Log Analysis Report")
        report.append("=" * 60)
        
        # Summary
        summary = analysis_results["summary"]
        report.append(f"\n📊 Log Summary:")
        report.append(f"  Total Log Lines: {summary['total_lines']:,}")
        report.append(f"  Errors: {summary['error_count']:,}")
        report.append(f"  Warnings: {summary['warning_count']:,}")
        report.append(f"  Error Rate: {summary['error_rate']:.2%}")
        
        # Trading Activity
        trading = analysis_results["trading_activity"]
        report.append(f"\n💰 Trading Activity:")
        report.append(f"  Total Trades: {trading['total_trades']}")
        report.append(f"  Trade Types: {dict(trading['trade_types'])}")
        report.append(f"  Average Position Size: {trading['average_position_size']:.1f}%")
        
        pnl = trading["pnl_summary"]
        report.append(f"  P&L Summary:")
        report.append(f"    Current: ${pnl['current']:.2f}")
        report.append(f"    Max: ${pnl['max']:.2f}")
        report.append(f"    Min: ${pnl['min']:.2f}")
        
        # Performance
        performance = analysis_results["performance"]
        api_perf = performance["api_performance"]
        report.append(f"\n⚡ Performance Metrics:")
        report.append(f"  Average API Response: {api_perf['average_response_time']:.0f}ms")
        report.append(f"  Max API Response: {api_perf['max_response_time']:.0f}ms")
        report.append(f"  Slow Requests (>5s): {api_perf['slow_requests']}")
        
        memory = performance["memory_usage"]
        report.append(f"  Memory Usage: {memory['current']:.1f}MB (avg: {memory['average']:.1f}MB)")
        
        # API Metrics
        api_metrics = analysis_results["api_metrics"]
        report.append(f"\n🔌 API Metrics:")
        for endpoint, success_rate in api_metrics["success_rates"].items():
            calls = api_metrics["api_calls"].get(endpoint, 0)
            report.append(f"  {endpoint.title()}: {calls} calls, {success_rate:.1%} success rate")
        
        # Recent Errors
        errors = analysis_results["errors"]
        if errors["recent_errors"]:
            report.append(f"\n🚨 Recent Errors:")
            for error in errors["recent_errors"][-5:]:
                report.append(f"  • {error}")
        
        return "\n".join(report)


if __name__ == "__main__":
    analyzer = LogAnalyzer()
    results = analyzer.analyze_logs(hours=24)
    report = analyzer.generate_report(results)
    print(report)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/log_analysis_{timestamp}.txt"
    Path(report_file).parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")
```

## Incident Response

### 1. Incident Response Procedures

#### Emergency Response Plan

**Severity Levels:**

1. **Critical (P0)**: System down, trading stopped, financial impact
2. **High (P1)**: Degraded performance, potential financial impact
3. **Medium (P2)**: Non-critical issues, monitoring alerts
4. **Low (P3)**: Maintenance items, optimization opportunities

#### Response Templates

Create `scripts/incident_response.py`:

```python
#!/usr/bin/env python3
"""Incident response automation and templates."""

import json
from datetime import datetime
from enum import Enum
from bot.config import create_settings


class Severity(Enum):
    CRITICAL = "P0"
    HIGH = "P1"
    MEDIUM = "P2"
    LOW = "P3"


class IncidentResponse:
    """Automated incident response procedures."""
    
    def __init__(self):
        self.settings = create_settings()
        self.response_procedures = {
            Severity.CRITICAL: self.critical_response,
            Severity.HIGH: self.high_response,
            Severity.MEDIUM: self.medium_response,
            Severity.LOW: self.low_response
        }
    
    def handle_incident(self, incident_type, severity, details):
        """Handle incident based on severity level."""
        incident = {
            "id": f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": incident_type,
            "severity": severity.value,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "status": "OPEN"
        }
        
        # Log incident
        self.log_incident(incident)
        
        # Execute response procedure
        response_actions = self.response_procedures[severity](incident)
        incident["actions"] = response_actions
        
        # Send notifications
        self.send_notifications(incident)
        
        return incident
    
    def critical_response(self, incident):
        """Critical incident response (P0)."""
        actions = []
        
        # Immediate actions
        actions.append("IMMEDIATE: Stop trading bot")
        self.execute_emergency_stop()
        
        actions.append("IMMEDIATE: Cancel all open orders")
        self.cancel_all_orders()
        
        actions.append("IMMEDIATE: Notify on-call team")
        self.notify_oncall_team(incident)
        
        # Assessment actions
        actions.append("ASSESS: Determine root cause")
        actions.append("ASSESS: Evaluate financial impact")
        actions.append("ASSESS: Check system integrity")
        
        return actions
    
    def high_response(self, incident):
        """High severity incident response (P1)."""
        actions = []
        
        actions.append("MONITOR: Increase monitoring frequency")
        actions.append("ANALYZE: Review recent logs and metrics")
        actions.append("NOTIFY: Alert operations team")
        
        # If trading performance degraded
        if "trading" in incident["type"].lower():
            actions.append("CONSIDER: Reduce position sizes")
            actions.append("CONSIDER: Increase risk limits")
        
        return actions
    
    def medium_response(self, incident):
        """Medium severity incident response (P2)."""
        actions = []
        
        actions.append("LOG: Document issue for analysis")
        actions.append("MONITOR: Track metrics for trends")
        actions.append("SCHEDULE: Review in next maintenance window")
        
        return actions
    
    def low_response(self, incident):
        """Low severity incident response (P3)."""
        actions = []
        
        actions.append("TRACK: Add to backlog")
        actions.append("OPTIMIZE: Consider for next optimization cycle")
        
        return actions
    
    def execute_emergency_stop(self):
        """Execute emergency stop procedure."""
        import os
        
        # Stop trading bot
        os.system("docker-compose down")
        
        # Send emergency notification
        self.send_emergency_alert("Trading bot emergency stop executed")
    
    def cancel_all_orders(self):
        """Cancel all open orders across all symbols."""
        try:
            # This would integrate with the exchange client
            # to cancel all open orders
            pass
        except Exception as e:
            print(f"Error cancelling orders: {e}")
    
    def notify_oncall_team(self, incident):
        """Notify on-call team of critical incident."""
        message = f"""
🚨 CRITICAL INCIDENT - {incident['id']}

Type: {incident['type']}
Severity: {incident['severity']}
Time: {incident['timestamp']}

Details: {incident['details']}

Immediate actions have been taken:
- Trading bot stopped
- Open orders cancelled

Please respond immediately.
        """
        
        # Send to on-call notification system
        self.send_alert(message, "critical")
    
    def send_notifications(self, incident):
        """Send appropriate notifications based on severity."""
        if incident["severity"] in ["P0", "P1"]:
            # High priority notifications
            self.send_alert(f"Incident {incident['id']}: {incident['type']}", "critical")
        else:
            # Standard notifications
            self.send_alert(f"Incident {incident['id']}: {incident['type']}", "warning")
    
    def send_alert(self, message, severity):
        """Send alert via configured channels."""
        # Implementation would send to Slack, email, PagerDuty, etc.
        print(f"ALERT ({severity}): {message}")
    
    def send_emergency_alert(self, message):
        """Send emergency alert to all channels."""
        # Implementation would send emergency notifications
        print(f"EMERGENCY: {message}")
    
    def log_incident(self, incident):
        """Log incident for tracking and analysis."""
        incident_file = f"logs/incidents/{incident['id']}.json"
        
        from pathlib import Path
        Path(incident_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(incident_file, 'w') as f:
            json.dump(incident, f, indent=2, default=str)


# Example usage
if __name__ == "__main__":
    response = IncidentResponse()
    
    # Example critical incident
    incident = response.handle_incident(
        incident_type="Trading Bot Crash",
        severity=Severity.CRITICAL,
        details="Bot container crashed with out-of-memory error during trading session"
    )
    
    print(f"Incident {incident['id']} handled with {len(incident['actions'])} actions")
```

## Maintenance Procedures

### 1. Regular Maintenance Tasks

#### Weekly Maintenance Script

Create `scripts/weekly_maintenance.py`:

```python
#!/usr/bin/env python3
"""Weekly maintenance procedures."""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from bot.config import create_settings


class MaintenanceManager:
    """Manage routine maintenance tasks."""
    
    def __init__(self):
        self.settings = create_settings()
        self.maintenance_log = "logs/maintenance.log"
    
    def run_weekly_maintenance(self):
        """Run all weekly maintenance tasks."""
        self.log_maintenance("Starting weekly maintenance")
        
        tasks = [
            ("Log rotation and cleanup", self.rotate_logs),
            ("Configuration backup", self.backup_configuration),
            ("Performance analysis", self.analyze_performance),
            ("Security check", self.security_check),
            ("System cleanup", self.system_cleanup),
            ("Update check", self.check_updates)
        ]
        
        for task_name, task_func in tasks:
            try:
                self.log_maintenance(f"Starting: {task_name}")
                task_func()
                self.log_maintenance(f"Completed: {task_name}")
            except Exception as e:
                self.log_maintenance(f"Failed: {task_name} - {e}")
        
        self.log_maintenance("Weekly maintenance completed")
    
    def rotate_logs(self):
        """Rotate and compress old log files."""
        logs_dir = Path("logs")
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for log_file in logs_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                # Compress old log files
                compressed_name = f"{log_file}.{datetime.now().strftime('%Y%m%d')}.gz"
                os.system(f"gzip -c {log_file} > {compressed_name}")
                
                # Clear original log file
                with open(log_file, 'w') as f:
                    f.write("")
    
    def backup_configuration(self):
        """Create configuration backup."""
        timestamp = datetime.now().strftime("%Y%m%d")
        backup_dir = Path(f"backups/config_{timestamp}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy configuration files
        for config_file in Path("config").glob("*.json"):
            shutil.copy2(config_file, backup_dir)
        
        # Export current settings
        settings = create_settings()
        settings.save_to_file(backup_dir / "current_settings.json")
    
    def analyze_performance(self):
        """Analyze system performance trends."""
        # Run log analysis
        os.system("python scripts/log_analyzer.py > reports/weekly_performance.txt")
        
        # Generate performance report
        os.system("python scripts/performance_report.py")
    
    def security_check(self):
        """Perform security checks."""
        checks = []
        
        # Check file permissions
        for sensitive_file in [".env", "config/"]:
            if Path(sensitive_file).exists():
                stat = Path(sensitive_file).stat()
                if stat.st_mode & 0o077:  # World or group readable
                    checks.append(f"WARNING: {sensitive_file} has loose permissions")
        
        # Check for secrets in logs
        for log_file in Path("logs").glob("*.log"):
            with open(log_file, 'r') as f:
                content = f.read()
                if any(keyword in content for keyword in ["password", "secret", "key"]):
                    checks.append(f"WARNING: Potential secret exposure in {log_file}")
        
        # Save security report
        with open("reports/security_check.txt", 'w') as f:
            f.write(f"Security Check - {datetime.now()}\n")
            f.write("=" * 40 + "\n")
            if checks:
                for check in checks:
                    f.write(f"{check}\n")
            else:
                f.write("No security issues found.\n")
    
    def system_cleanup(self):
        """Clean up temporary files and optimize storage."""
        # Remove old temporary files
        temp_dirs = ["tmp/", ".cache/", "__pycache__/"]
        for temp_dir in temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        
        # Clean Docker artifacts
        os.system("docker system prune -f")
        
        # Optimize database (if applicable)
        # os.system("vacuumdb trading_bot")
    
    def check_updates(self):
        """Check for available updates."""
        # Check for Python package updates
        os.system("pip list --outdated > reports/package_updates.txt")
        
        # Check for security updates
        os.system("apt list --upgradable > reports/system_updates.txt")
        
        # Generate update report
        with open("reports/update_check.txt", 'w') as f:
            f.write(f"Update Check - {datetime.now()}\n")
            f.write("=" * 40 + "\n")
            f.write("Check package_updates.txt and system_updates.txt for available updates.\n")
    
    def log_maintenance(self, message):
        """Log maintenance activity."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.maintenance_log, 'a') as f:
            f.write(f"{timestamp} - {message}\n")
        print(f"{timestamp} - {message}")


if __name__ == "__main__":
    manager = MaintenanceManager()
    manager.run_weekly_maintenance()
```

### 2. Configuration Management

#### Configuration Drift Detection

Create `scripts/config_drift_detector.py`:

```python
#!/usr/bin/env python3
"""Detect configuration drift from baseline."""

import json
import hashlib
from pathlib import Path
from bot.config import create_settings


class ConfigDriftDetector:
    """Detect changes in configuration from baseline."""
    
    def __init__(self):
        self.baseline_file = "config/baseline_config.json"
        self.settings = create_settings()
    
    def create_baseline(self):
        """Create configuration baseline."""
        baseline = self.get_config_snapshot()
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2, default=str)
        
        print(f"Baseline created: {self.baseline_file}")
    
    def check_drift(self):
        """Check for configuration drift."""
        if not Path(self.baseline_file).exists():
            print("No baseline found. Creating baseline...")
            self.create_baseline()
            return
        
        # Load baseline
        with open(self.baseline_file, 'r') as f:
            baseline = json.load(f)
        
        # Get current configuration
        current = self.get_config_snapshot()
        
        # Compare configurations
        drift_report = self.compare_configs(baseline, current)
        
        if drift_report["changes"]:
            print("⚠️  Configuration drift detected!")
            self.print_drift_report(drift_report)
        else:
            print("✅ No configuration drift detected")
        
        return drift_report
    
    def get_config_snapshot(self):
        """Get current configuration snapshot."""
        # Export configuration without secrets
        config_dict = self.settings.model_dump(exclude={
            'llm': {'openai_api_key', 'anthropic_api_key'},
            'exchange': {'cb_api_key', 'cb_api_secret', 'cb_passphrase'},
            'system': {'api_secret_key', 'instance_id'}
        })
        
        return {
            "config": config_dict,
            "hash": self.calculate_config_hash(config_dict),
            "timestamp": self.settings.model_dump()['system']['instance_id']  # Placeholder
        }
    
    def calculate_config_hash(self, config_dict):
        """Calculate hash of configuration."""
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def compare_configs(self, baseline, current):
        """Compare baseline and current configurations."""
        changes = []
        
        def compare_dicts(baseline_dict, current_dict, path=""):
            for key in set(baseline_dict.keys()) | set(current_dict.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in baseline_dict:
                    changes.append({
                        "type": "added",
                        "path": current_path,
                        "value": current_dict[key]
                    })
                elif key not in current_dict:
                    changes.append({
                        "type": "removed",
                        "path": current_path,
                        "value": baseline_dict[key]
                    })
                elif isinstance(baseline_dict[key], dict) and isinstance(current_dict[key], dict):
                    compare_dicts(baseline_dict[key], current_dict[key], current_path)
                elif baseline_dict[key] != current_dict[key]:
                    changes.append({
                        "type": "changed",
                        "path": current_path,
                        "old_value": baseline_dict[key],
                        "new_value": current_dict[key]
                    })
        
        compare_dicts(baseline["config"], current["config"])
        
        return {
            "baseline_hash": baseline["hash"],
            "current_hash": current["hash"],
            "changes": changes,
            "drift_detected": len(changes) > 0
        }
    
    def print_drift_report(self, drift_report):
        """Print formatted drift report."""
        print(f"\nConfiguration Drift Report")
        print(f"=" * 40)
        print(f"Baseline Hash: {drift_report['baseline_hash'][:16]}...")
        print(f"Current Hash:  {drift_report['current_hash'][:16]}...")
        print(f"Changes: {len(drift_report['changes'])}")
        print()
        
        for change in drift_report['changes']:
            if change['type'] == 'changed':
                print(f"CHANGED: {change['path']}")
                print(f"  Old: {change['old_value']}")
                print(f"  New: {change['new_value']}")
            elif change['type'] == 'added':
                print(f"ADDED: {change['path']} = {change['value']}")
            elif change['type'] == 'removed':
                print(f"REMOVED: {change['path']} = {change['value']}")
            print()


if __name__ == "__main__":
    detector = ConfigDriftDetector()
    detector.check_drift()
```

---

This Operations Manual provides comprehensive guidance for managing the AI Trading Bot in production, covering monitoring, troubleshooting, performance optimization, backup procedures, and maintenance tasks. The next document will focus on user-facing guidance and best practices.