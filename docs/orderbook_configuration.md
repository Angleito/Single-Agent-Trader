# Orderbook Configuration Guide

This document provides comprehensive documentation for the orderbook configuration parameters added to the AI trading bot. These parameters allow fine-tuning of market data collection, processing, and validation for enhanced trading performance.

## Overview

The orderbook configuration system provides advanced controls for:
- Real-time market data collection and processing
- Data quality and validation
- Performance optimization
- Connection management
- Order flow analysis

## Configuration Structure

Orderbook settings are configured in the `orderbook` section of `config/market_making.json` and can be overridden via environment variables using the prefix `MARKET_MAKING__ORDERBOOK__`.

## Core Parameters

### Data Collection

#### `depth_levels` (integer, default: 20)
Number of price levels to fetch from the orderbook on each side (bid/ask).
- **Conservative**: 10 levels
- **Moderate**: 20 levels
- **Aggressive**: 30+ levels
- **Range**: 1-100
- **Environment Variable**: `MARKET_MAKING__ORDERBOOK__DEPTH_LEVELS`

```json
"depth_levels": 20
```

#### `refresh_interval_ms` (integer, default: 100)
How frequently to refresh orderbook data in milliseconds.
- **Conservative**: 200ms (less CPU usage)
- **Moderate**: 100ms (balanced)
- **Aggressive**: 50ms (high frequency)
- **Range**: 10-10000ms
- **Environment Variable**: `MARKET_MAKING__ORDERBOOK__REFRESH_INTERVAL_MS`

```json
"refresh_interval_ms": 100
```

#### `aggregation_levels` (array, default: [1, 5, 10, 20])
Price level aggregation groups for analysis.
- Used for calculating liquidity bands and market depth
- Can be customized based on trading strategy needs

```json
"aggregation_levels": [1, 5, 10, 20]
```

### Data Quality and Validation

#### `quality_threshold` (float, default: 0.8)
Minimum quality score (0.0-1.0) for orderbook data to be considered reliable.
- **Range**: 0.1-1.0
- **Conservative**: 0.9 (very strict)
- **Moderate**: 0.8 (balanced)
- **Aggressive**: 0.7 (more permissive)

```json
"quality_threshold": 0.8
```

#### `max_age_ms` (integer, default: 1000)
Maximum age of orderbook data before considering it stale.

```json
"max_age_ms": 1000
```

#### `staleness_threshold_ms` (integer, default: 2000)
Maximum time before orderbook data is considered too old to use.

```json
"staleness_threshold_ms": 2000
```

#### Market Data Validation

##### Price Validation
- `enable_price_validation` (boolean, default: true): Enable price validation
- `max_price_deviation_pct` (float, default: 5.0): Maximum allowed price deviation percentage

##### Size Validation
- `enable_size_validation` (boolean, default: true): Enable order size validation
- `min_order_size` (string, default: "10"): Minimum valid order size
- `max_order_size` (string, default: "50000"): Maximum valid order size

##### Time Validation
- `enable_time_validation` (boolean, default: true): Enable timestamp validation
- `max_timestamp_drift_ms` (integer, default: 5000): Maximum allowed timestamp drift

### Trading Parameters

#### `min_liquidity_threshold` (string, default: "500")
Minimum size required at a price level to be considered for trading.

```json
"min_liquidity_threshold": "500"
```

#### `max_spread_bps` (integer, default: 200)
Maximum spread in basis points before considering the market illiquid.
- **Conservative**: 100 bps
- **Moderate**: 200 bps
- **Aggressive**: 300 bps

```json
"max_spread_bps": 200
```

#### Liquidity Bands
Predefined liquidity analysis bands with spread and minimum size requirements.

```json
"liquidity_bands": {
  "tight": {"bps": 5, "min_size": "100"},
  "normal": {"bps": 10, "min_size": "250"},
  "wide": {"bps": 25, "min_size": "500"}
}
```

### Advanced Features

#### Order Flow Analysis

##### `enable_order_flow_analysis` (boolean, default: true)
Enable advanced order flow and market microstructure analysis.

```json
"enable_order_flow_analysis": true
```

##### `imbalance_detection_threshold` (float, default: 0.3)
Threshold for detecting order book imbalance (0.0-1.0).
- **Range**: 0.1-1.0
- Higher values = less sensitive to imbalances
- Lower values = more sensitive detection

```json
"imbalance_detection_threshold": 0.3
```

### Performance and Connection Management

#### Real-time Updates

##### `enable_incremental_updates` (boolean, default: true)
Enable incremental orderbook updates for better performance.

```json
"enable_incremental_updates": true
```

##### `enable_snapshot_recovery` (boolean, default: true)
Enable automatic snapshot recovery when incremental updates fail.

```json
"enable_snapshot_recovery": true
```

##### `snapshot_recovery_interval_ms` (integer, default: 5000)
Interval for automatic snapshot recovery attempts.

```json
"snapshot_recovery_interval_ms": 5000
```

#### Buffer and Memory Management

##### `buffer_size` (integer, default: 1000)
Internal buffer size for orderbook data.
- Higher values = more memory usage but better performance
- Lower values = less memory but potential data loss

```json
"buffer_size": 1000
```

##### `compression_enabled` (boolean, default: false)
Enable data compression for orderbook storage (reduces memory usage).

```json
"compression_enabled": false
```

#### WebSocket Connection Settings

##### Connection Timeouts
- `websocket_timeout_ms` (integer, default: 30000): WebSocket connection timeout
- `heartbeat_interval_ms` (integer, default: 15000): Heartbeat ping interval
- `reconnect_delay_ms` (integer, default: 1000): Delay between reconnection attempts
- `max_reconnect_attempts` (integer, default: 10): Maximum reconnection attempts

```json
"websocket_timeout_ms": 30000,
"heartbeat_interval_ms": 15000,
"reconnect_delay_ms": 1000,
"max_reconnect_attempts": 10
```

### Precision Settings

#### `price_precision` (integer, default: 6)
Decimal precision for price formatting and calculations.

```json
"price_precision": 6
```

#### `size_precision` (integer, default: 4)
Decimal precision for order size formatting and calculations.

```json
"size_precision": 4
```

## Configuration Profiles

### Conservative Profile
Optimized for stability and lower resource usage:

```json
"orderbook": {
  "depth_levels": 10,
  "refresh_interval_ms": 200,
  "max_spread_bps": 100,
  "quality_threshold": 0.9,
  "staleness_threshold_ms": 1000,
  "imbalance_detection_threshold": 0.2,
  "enable_order_flow_analysis": false
}
```

### Moderate Profile
Balanced performance and resource usage:

```json
"orderbook": {
  "depth_levels": 20,
  "refresh_interval_ms": 100,
  "max_spread_bps": 200,
  "quality_threshold": 0.8,
  "staleness_threshold_ms": 2000,
  "imbalance_detection_threshold": 0.3,
  "enable_order_flow_analysis": true
}
```

### Aggressive Profile
High-frequency trading optimized:

```json
"orderbook": {
  "depth_levels": 30,
  "refresh_interval_ms": 50,
  "max_spread_bps": 300,
  "quality_threshold": 0.7,
  "staleness_threshold_ms": 3000,
  "imbalance_detection_threshold": 0.4,
  "enable_order_flow_analysis": true,
  "enable_incremental_updates": true,
  "aggregation_levels": [1, 2, 5, 10, 15, 30]
}
```

## Environment Variable Examples

```bash
# Basic orderbook configuration
MARKET_MAKING__ORDERBOOK__DEPTH_LEVELS=20
MARKET_MAKING__ORDERBOOK__REFRESH_INTERVAL_MS=100
MARKET_MAKING__ORDERBOOK__QUALITY_THRESHOLD=0.8

# Advanced features
MARKET_MAKING__ORDERBOOK__ENABLE_ORDER_FLOW_ANALYSIS=true
MARKET_MAKING__ORDERBOOK__IMBALANCE_DETECTION_THRESHOLD=0.3

# Performance tuning
MARKET_MAKING__ORDERBOOK__BUFFER_SIZE=2000
MARKET_MAKING__ORDERBOOK__ENABLE_INCREMENTAL_UPDATES=true

# Validation settings
MARKET_MAKING__ORDERBOOK__ENABLE_PRICE_VALIDATION=true
MARKET_MAKING__ORDERBOOK__MAX_PRICE_DEVIATION_PCT=5.0
```

## Performance Considerations

### High-Frequency Trading
- Use `refresh_interval_ms`: 50ms or lower
- Enable `enable_incremental_updates`
- Increase `buffer_size` to 2000+
- Set `depth_levels` to 30+

### Resource Constrained Environments
- Use `refresh_interval_ms`: 200ms or higher
- Disable `enable_order_flow_analysis`
- Reduce `depth_levels` to 10
- Enable `compression_enabled`

### Balanced Performance
- Use default values as starting point
- Monitor system resources and adjust accordingly
- Enable all validation features for data integrity

## Validation and Error Handling

The configuration system includes comprehensive validation:

- **Range Validation**: All numeric parameters are validated against reasonable ranges
- **Type Validation**: Ensures correct data types for all parameters
- **Logical Validation**: Checks for parameter combinations that make sense
- **Performance Warnings**: Alerts for configurations that may impact performance

## Integration with Trading Strategies

Orderbook parameters integrate with:
- **Market Making**: Depth levels and spread detection for optimal quote placement
- **Arbitrage**: Real-time data quality for price discrepancy detection
- **Momentum Trading**: Order flow analysis for trend detection
- **Mean Reversion**: Imbalance detection for reversal signals

## Monitoring and Diagnostics

The orderbook system provides metrics for:
- Data quality scores
- Update frequencies and latencies
- Connection health and stability
- Order flow imbalance measurements
- Performance statistics

## Best Practices

1. **Start Conservative**: Begin with conservative settings and gradually optimize
2. **Monitor Performance**: Watch system resources and trading performance
3. **Validate Data Quality**: Use validation features to ensure data integrity
4. **Test Thoroughly**: Use paper trading to validate configuration changes
5. **Document Changes**: Keep track of configuration modifications and their impact

## Troubleshooting

### Common Issues

#### High CPU Usage
- Increase `refresh_interval_ms`
- Reduce `depth_levels`
- Disable `enable_order_flow_analysis`

#### Stale Data Warnings
- Decrease `staleness_threshold_ms`
- Check network connectivity
- Verify WebSocket connection settings

#### Memory Usage
- Reduce `buffer_size`
- Enable `compression_enabled`
- Lower `depth_levels`

#### Data Quality Issues
- Increase `quality_threshold`
- Enable all validation features
- Check exchange API status

For additional support, refer to the main documentation or check the system logs for specific error messages.
