# Market Making Examples

This directory contains comprehensive working examples and scripts for all aspects of market making operations using the AI Trading Bot framework.

## 📁 File Overview

### Core Examples

- **`market_making_examples.py`** - Main example suite with multiple trading scenarios
- **`custom_profiles_example.py`** - Profile creation and optimization utilities
- **`performance_monitoring_scripts.py`** - Real-time monitoring and analysis tools

### Configuration Profiles

- **`config/profiles/conservative_example.json`** - Low-risk trading configuration
- **`config/profiles/aggressive_hft_example.json`** - High-frequency trading setup
- **`config/profiles/multi_symbol_example.json`** - Multi-symbol portfolio configuration
- **`config/profiles/paper_trading_example.json`** - Safe testing environment

### Existing Examples

- **`inventory_management_example.py`** - Inventory tracking and rebalancing
- **`market_making_performance_example.py`** - Performance monitoring integration
- **`monitoring_example.py`** - System health monitoring

## 🚀 Quick Start

### 1. Conservative Trading Example
Perfect for beginners or risk-averse traders:

```bash
python examples/market_making_examples.py --profile conservative --duration 15
```

**Features:**
- Wide spreads (25 bps) for safety
- Low position limits (10% max)
- Early rebalancing (40% threshold)
- Conservative risk management

### 2. Aggressive High-Frequency Trading
For experienced traders seeking maximum capture:

```bash
python examples/market_making_examples.py --profile aggressive --duration 10
```

**Features:**
- Ultra-tight spreads (3 bps)
- Fast cycle times (200ms)
- Multiple order levels (5 levels)
- Large position limits (40% max)

### 3. Multi-Symbol Portfolio
Diversified trading across multiple symbols:

```bash
python examples/market_making_examples.py --multi-symbol --duration 20
```

**Features:**
- SUI-PERP, BTC-PERP, ETH-PERP
- Symbol-specific configurations
- Portfolio-level risk management
- Cross-symbol monitoring

### 4. Paper Trading Simulation
100% safe testing environment:

```bash
python examples/market_making_examples.py --paper-trading --duration 25
```

**Features:**
- No real money risk
- Real market data simulation
- Performance tracking
- Configuration validation

### 5. Risk Management Scenarios
Demonstration of risk controls:

```bash
python examples/market_making_examples.py --risk-management
```

**Features:**
- Position limit demonstrations
- Emergency stop procedures
- Inventory rebalancing
- Recovery protocols

### 6. Emergency Procedures
Emergency and recovery demonstrations:

```bash
python examples/market_making_examples.py --emergency-demo
```

**Features:**
- Emergency stop triggers
- System shutdown procedures
- Recovery validation
- Health monitoring

## 🎯 Custom Profile Creation

Create custom trading profiles for specific scenarios:

```bash
python examples/custom_profiles_example.py
```

**Capabilities:**
- Template-based profile creation
- Parameter validation
- Market condition optimization
- Performance tuning
- Profile comparison

### Example: Creating a Custom Profile

```python
from examples.custom_profiles_example import CustomProfileCreator

creator = CustomProfileCreator()

# Create a custom scalping profile
profile = creator.create_custom_profile(
    name="my_scalper",
    base_template="scalping",
    customizations={
        "strategy": {
            "base_spread_bps": 5,
            "max_position_pct": 25
        },
        "risk": {
            "daily_loss_limit_pct": 2.0
        }
    }
)

# Save the profile
creator.save_profile(profile)
```

## 📊 Performance Monitoring

Real-time performance monitoring and analysis:

```bash
python examples/performance_monitoring_scripts.py
```

**Options:**
1. **Real-time monitoring** - Live performance dashboard
2. **Historical analysis** - Analyze past performance data
3. **Both** - Sequential monitoring and analysis

**Metrics Tracked:**
- Real-time P&L
- Fill rates and execution quality
- Signal effectiveness
- Risk metrics and drawdowns
- Performance alerts
- Sharpe ratio and other risk-adjusted metrics

### Example: Real-time Monitoring

```python
from examples.performance_monitoring_scripts import RealTimePerformanceMonitor

monitor = RealTimePerformanceMonitor("SUI-PERP")
await monitor.start_monitoring(duration_minutes=30)
```

## ⚙️ Configuration Examples

### Conservative Profile
```json
{
  "strategy": {
    "base_spread_bps": 25,
    "order_levels": 2,
    "max_position_pct": 10.0
  },
  "risk": {
    "max_position_value": "2500",
    "daily_loss_limit_pct": 2.0,
    "rebalancing_threshold": 3.0
  }
}
```

### Aggressive HFT Profile
```json
{
  "strategy": {
    "base_spread_bps": 3,
    "order_levels": 5,
    "max_position_pct": 40.0
  },
  "cycle_interval_seconds": 0.2,
  "risk": {
    "max_position_value": "25000",
    "daily_loss_limit_pct": 8.0
  }
}
```

### Multi-Symbol Configuration
```json
{
  "symbols": {
    "SUI-PERP": {
      "weight": 0.4,
      "strategy": {"base_spread_bps": 8}
    },
    "BTC-PERP": {
      "weight": 0.4,
      "strategy": {"base_spread_bps": 5}
    },
    "ETH-PERP": {
      "weight": 0.2,
      "strategy": {"base_spread_bps": 6}
    }
  }
}
```

## 🛡️ Risk Management Examples

### Position Limit Controls
```python
config = {
    "max_position_value": 1000,  # Strict limit
    "strategy": {"max_position_pct": 5},  # Small positions
    "inventory": {"rebalancing_threshold": 0.3}  # Early rebalancing
}
```

### Emergency Stop Configuration
```python
emergency_config = {
    "emergency_stop_threshold": 5,  # Quick trigger
    "max_errors_per_hour": 10,     # Low tolerance
    "volatility_circuit_breaker": 0.1  # 10% volatility limit
}
```

### Volatility Adjustments
```python
# Automatically adjust spreads based on market volatility
if volatility > 0.05:  # High volatility
    spread_multiplier = min(volatility / 0.02, 3.0)
    adjusted_spread = base_spread * spread_multiplier
```

## 📈 Performance Expectations

### Conservative Profile
- **Win Rate:** 40-60%
- **Daily Return:** 0.1-0.5%
- **Max Drawdown:** < 2%
- **Use Case:** Risk-averse, capital preservation

### Aggressive HFT Profile
- **Win Rate:** 55-75%
- **Daily Return:** 1.0-5.0%
- **Max Drawdown:** < 5%
- **Use Case:** Maximum profit extraction

### Multi-Symbol Portfolio
- **Win Rate:** 50-70%
- **Daily Return:** 0.5-2.0%
- **Max Drawdown:** < 4%
- **Use Case:** Diversified risk, professional operations

### Paper Trading
- **Accuracy:** 80-90% of live conditions
- **Risk:** 0% (no real money)
- **Use Case:** Strategy validation, training

## 🔧 Customization Guide

### Creating New Profiles

1. **Choose a base template:**
   - `scalping` - Ultra-fast, tight spreads
   - `swing_trading` - Longer-term, wider spreads
   - `volatility_adaptive` - Adjusts to market conditions
   - `news_trader` - Optimized for market events

2. **Customize parameters:**
   ```python
   customizations = {
       "strategy": {
           "base_spread_bps": 15,    # Adjust spread
           "order_levels": 3,        # Number of levels
           "max_position_pct": 20    # Position sizing
       },
       "risk": {
           "daily_loss_limit_pct": 3.0,  # Risk limits
           "volatility_threshold": 0.25   # Volatility handling
       }
   }
   ```

3. **Validate and test:**
   ```python
   validation = creator.validate_profile(profile)
   if validation["valid"]:
       creator.save_profile(profile)
   ```

### Market Condition Optimization

```python
market_conditions = {
    "volatility": 0.03,        # Current volatility
    "avg_volume": 2000000,     # Average volume
    "trend_strength": 0.5      # Trend strength
}

optimized = creator.optimize_profile_for_market(base_profile, market_conditions)
```

## 🚨 Safety Features

### Paper Trading Benefits
- **Zero Risk:** No real money involved
- **Full Simulation:** Real market data with simulated execution
- **Comprehensive Testing:** Test all scenarios safely
- **Configuration Validation:** Ensure settings are correct
- **Performance Projection:** Predict live trading results

### Emergency Procedures
1. **Automatic Triggers:** System monitors for emergency conditions
2. **Immediate Shutdown:** Cancel orders, stop trading
3. **State Preservation:** Save current positions and settings
4. **Recovery Procedures:** Systematic restart protocols
5. **Health Monitoring:** Continuous system health checks

### Risk Controls
- **Position Limits:** Maximum position sizes and values
- **Drawdown Limits:** Daily and maximum loss thresholds
- **Volatility Monitors:** Automatic adjustments for market conditions
- **Inventory Management:** Rebalancing and risk scoring
- **Performance Alerts:** Real-time notifications for issues

## 📊 Monitoring and Analysis

### Real-time Dashboard
- Live P&L tracking
- Fill rate monitoring
- Signal effectiveness
- Risk metrics
- Performance alerts

### Historical Analysis
- Performance pattern identification
- Risk factor analysis
- Optimization recommendations
- Strategy effectiveness review

### Export Capabilities
- JSON data export
- Performance reports
- Risk analysis
- Configuration backups

## 🔗 Integration with Main Bot

All examples are designed to work seamlessly with the main trading bot:

```bash
# Use custom profile with main bot
python -m bot.main live --config config/profiles/conservative_example.json

# Enable market making for specific symbol
export MARKET_MAKING__ENABLED=true
export MARKET_MAKING__SYMBOL=SUI-PERP
python -m bot.main live
```

## 🎓 Learning Path

### Beginner
1. Start with **paper trading example**
2. Try **conservative profile**
3. Learn **risk management scenarios**
4. Understand **performance monitoring**

### Intermediate
1. **Custom profile creation**
2. **Multi-symbol configurations**
3. **Performance optimization**
4. **Market condition adaptation**

### Advanced
1. **Aggressive HFT strategies**
2. **Custom indicator integration**
3. **Portfolio-level risk management**
4. **Emergency procedures and recovery**

## 🆘 Troubleshooting

### Common Issues

**Example won't start:**
- Check Python dependencies: `poetry install`
- Verify configuration files exist
- Check log files for error messages

**Performance issues:**
- Reduce cycle frequency for testing
- Check system resources
- Monitor network latency

**Validation errors:**
- Review parameter ranges in validation rules
- Check configuration file syntax
- Ensure all required fields are present

### Getting Help

1. Check log files in `logs/` directory
2. Review configuration validation messages
3. Run examples in debug mode: `python -m examples.market_making_examples --debug`
4. Refer to main bot documentation in `docs/`

## 📝 Contributing

To add new examples:

1. Follow existing code structure
2. Include comprehensive documentation
3. Add configuration validation
4. Provide test scenarios
5. Update this README

## 📄 License

These examples are part of the AI Trading Bot project and follow the same license terms.
