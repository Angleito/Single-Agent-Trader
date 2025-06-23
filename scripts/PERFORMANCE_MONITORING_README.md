# Performance Monitoring System

## Overview

The `performance-monitor.py` script provides a comprehensive, high-performance monitoring solution for the AI Trading Bot. It tracks all critical performance metrics with minimal overhead and exports them in Prometheus format for easy integration with monitoring stacks.

## Features

### 1. **Trading Bot Response Times**
- Tracks decision-making latency
- Monitors end-to-end response times
- Provides percentile breakdowns (p50, p95, p99)
- Ring buffer storage for 10,000 most recent measurements

### 2. **Order Execution Latency**
- Measures time from order placement to confirmation
- Tracks success/failure rates by order type
- Millisecond precision timing
- Histogram buckets: 1ms to 10s

### 3. **Memory Usage Patterns**
- System-wide memory monitoring
- Process-specific RSS and VMS tracking
- Leak detection with growth alerts
- Swap usage monitoring

### 4. **CPU Utilization**
- Overall and per-core CPU tracking
- Thread-based monitoring for accuracy
- 5-second update intervals
- Historical trend analysis

### 5. **Network I/O Metrics**
- Bytes sent/received per second
- Packet rates by interface
- Excludes loopback interfaces
- Per-interface breakdown

### 6. **Database Query Performance**
- Operation-level timing (SELECT, INSERT, UPDATE)
- Table-specific metrics
- Min/max/avg query times
- Query count tracking

## Architecture

### Design Principles

1. **Minimal Overhead**
   - Ring buffers for time-series data
   - Memory-mapped files for zero-copy sharing
   - Atomic operations for thread safety
   - Sampling strategies to reduce load

2. **High Performance**
   - Separate thread for CPU monitoring
   - Async coroutines for I/O operations
   - Efficient data structures (deque with maxlen)
   - Batched metric updates

3. **Production Ready**
   - Prometheus exposition format
   - Health check endpoint
   - Graceful shutdown handling
   - Resource cleanup on exit

## Usage

### Standalone Monitoring

```bash
# Start the performance monitor
python scripts/performance-monitor.py

# Metrics will be available at:
# http://localhost:9090/metrics (Prometheus format)
# http://localhost:9090/health (JSON health check)
```

### Integration with Trading Bot

#### Method 1: Using Decorators

```python
from scripts.performance_monitor import (
    monitor_response_time,
    monitor_order_execution,
    monitor_db_query
)

@monitor_response_time("trading_decision")
async def make_decision(market_data):
    # Your decision logic
    return decision

@monitor_order_execution("market_order")
async def place_order(params):
    # Your order logic
    return order_result

@monitor_db_query("select", "experiences")
async def query_memories(conditions):
    # Your query logic
    return results
```

#### Method 2: Using Context Managers

```python
from scripts.performance_monitor import get_monitor

monitor = get_monitor()

async def complex_operation():
    with monitor.measure_response_time("operation_name"):
        # Your code here
        pass

    with monitor.measure_order_latency("order_type"):
        # Order execution code
        pass

    with monitor.measure_query_time("select", "table_name"):
        # Database query code
        pass
```

#### Method 3: Direct Integration in main.py

```python
# In bot/main.py
from scripts.performance_monitor import get_monitor

# Start monitoring when bot starts
monitor = get_monitor()
monitor.start_monitoring()

# Use throughout the codebase
# The decorators can be added to existing functions
```

## Metrics Reference

### Response Time Metrics
- `trading_bot_response_seconds` - Histogram of response times
- `trading_bot_response_summary_seconds` - Statistical summary

### Order Execution Metrics
- `order_execution_latency_milliseconds` - Histogram of latencies
- `orders_total` - Counter with labels: action, status

### Memory Metrics
- `memory_usage_bytes` - Gauge with labels: type
- `memory_usage_mb_histogram` - Distribution histogram

### CPU Metrics
- `cpu_usage_percent` - Gauge with labels: core
- `cpu_usage_summary_percent` - Statistical summary

### Network Metrics
- `network_io_bytes_per_second` - Gauge with labels: direction, interface
- `network_packets_total` - Counter with labels: direction, interface

### Database Metrics
- `database_query_duration_milliseconds` - Histogram with labels: operation, table
- `database_queries_total` - Counter with labels: operation, table, status

### System Metrics
- `bot_uptime_seconds` - Gauge of bot uptime
- `bot_health_status` - Overall health score (0-1)

## Prometheus Configuration

Add this job to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'trading_bot'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

## Grafana Dashboard

Import the following queries for a comprehensive dashboard:

```promql
# Response Time
rate(trading_bot_response_seconds_sum[5m]) / rate(trading_bot_response_seconds_count[5m])

# Order Success Rate
sum(rate(orders_total{status="success"}[5m])) / sum(rate(orders_total[5m]))

# Memory Usage
memory_usage_bytes{type="process_rss"} / 1024 / 1024

# CPU Usage
avg(cpu_usage_percent{core!="all"})

# Network I/O
sum(network_io_bytes_per_second) by (direction)

# Query Performance
histogram_quantile(0.95, rate(database_query_duration_milliseconds_bucket[5m]))
```

## Performance Impact

The monitoring system is designed for minimal impact:
- CPU overhead: <1% in normal operation
- Memory overhead: ~10MB for metric storage
- Network overhead: Negligible (metrics endpoint only)
- Latency added: <0.1ms per measurement

## Troubleshooting

### High Memory Usage
- Check for memory leaks in monitored functions
- Reduce ring buffer sizes if needed
- Monitor the `memory_growth` warnings in logs

### Missing Metrics
- Ensure decorators are properly applied
- Check that `monitor.start_monitoring()` was called
- Verify no exceptions in decorated functions

### Performance Degradation
- Reduce monitoring frequency for high-volume operations
- Use sampling for extremely frequent calls
- Check CPU usage of monitoring thread

## Advanced Features

### Custom Metrics

```python
from prometheus_client import Counter, Gauge, Histogram

# Add custom metrics to the monitor
monitor = get_monitor()

# Custom gauge
my_gauge = Gauge('custom_metric', 'Description', registry=monitor.registry)
my_gauge.set(42.5)

# Custom counter
my_counter = Counter('custom_events', 'Description', ['event_type'], registry=monitor.registry)
my_counter.labels(event_type='special').inc()
```

### Shared Memory Access

For ultra-low latency access to metrics:

```python
import mmap
import struct

# Read latest response time from shared memory
with open('/dev/shm/trading_bot_metrics/perf_metrics_*.dat', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 4096, access=mmap.ACCESS_READ)
    latest_response_time = struct.unpack('d', mm[:8])[0]
    mm.close()
```

## Security Considerations

- Metrics endpoint binds to 0.0.0.0:9090 by default
- No authentication on metrics endpoint (add reverse proxy for production)
- Shared memory files have process-specific names to prevent conflicts
- Resource limits prevent unbounded memory growth
