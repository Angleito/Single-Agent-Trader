# Unified Indicator Framework Documentation

## Overview

The Unified Indicator Framework provides a comprehensive, high-performance indicator management system that seamlessly supports both momentum (1-5 minute) and scalping (15 second - 1 minute) strategies with optimized calculations, unified interfaces, and intelligent caching.

## Key Features

- **Multi-timeframe Support**: Optimized configurations for scalping, momentum, swing, and position trading
- **Async Calculation Engine**: Parallel processing with dependency graph optimization
- **Intelligent Caching**: Thread-safe caching with configurable TTL
- **Incremental Updates**: Real-time updates for supported indicators
- **Unified Interface**: Consistent API across all indicators
- **Performance Optimization**: Built-in performance monitoring and optimization suggestions
- **Lazy Loading**: Efficient memory usage with on-demand indicator instantiation

## Architecture

```
UnifiedIndicatorFramework
├── IndicatorRegistry          # Manages indicator registration
├── TimeframeConfigManager     # Timeframe-specific configurations
├── MultiTimeframeCalculator   # Async calculation engine
├── IncrementalUpdater         # Real-time updates
├── IndicatorCache            # Thread-safe caching
└── PerformanceOptimizer      # Performance analysis
```

## Core Components

### TimeframeType Enum

```python
class TimeframeType(Enum):
    SCALPING = "scalping"    # 15s-1m: Ultra-fast signals
    MOMENTUM = "momentum"    # 1m-5m: Trend following
    SWING = "swing"         # 5m-15m: Position trading
    POSITION = "position"   # 15m-1h: Long-term trends
```

### IndicatorType Enum

```python
class IndicatorType(Enum):
    TREND = "trend"         # EMAs, MAs, trend analysis
    MOMENTUM = "momentum"   # RSI, MACD, Williams %R
    VOLUME = "volume"       # VWAP, OBV, Volume profile
    VOLATILITY = "volatility" # ATR, Bollinger Bands
    CUSTOM = "custom"       # VuManChu, proprietary indicators
```

## Usage Examples

### Basic Strategy Calculation

```python
from bot.indicators import calculate_indicators_for_strategy
import pandas as pd

# Prepare market data
market_data = {
    'scalping': pd.DataFrame({
        'open': [100, 101, 102],
        'high': [101, 102, 103], 
        'low': [99, 100, 101],
        'close': [101, 102, 101],
        'volume': [1000, 1100, 1200]
    })
}

# Calculate indicators for scalping strategy
results = await calculate_indicators_for_strategy(
    strategy_type='scalping',
    market_data=market_data
)

print(f"Calculated {results['performance_metrics']['indicator_count']} indicators")
print(f"Total time: {results['performance_metrics']['total_calculation_time_ms']:.2f}ms")

# Access individual indicators
vumanchu_data = results['indicators']['vumanchu_cipher_a']
ema_data = results['indicators']['fast_ema']

# Access combined signals
signals = results['combined_signals']
for signal in signals:
    print(f"Signal: {signal['type']} from {signal['indicator']} (strength: {signal['strength']})")
```

### Custom Indicator Selection

```python
# Calculate specific indicators only
results = await calculate_indicators_for_strategy(
    strategy_type='momentum',
    market_data=market_data,
    custom_indicators=['vumanchu_cipher_a', 'fast_ema']
)
```

### Real-time Incremental Updates

```python
from bot.indicators import unified_framework

# Setup incremental mode
setup_results = await unified_framework.setup_incremental_mode(
    strategy_type='scalping',
    initial_data=market_data
)

# Process new tick data
new_tick = {
    'timestamp': datetime.now(),
    'open': 102.0,
    'high': 102.5,
    'low': 101.8,
    'close': 102.3,
    'volume': 1500
}

# Update indicators incrementally
incremental_results = await unified_framework.update_incremental(
    strategy_type='scalping',
    new_tick=new_tick
)
```

### Performance Analysis

```python
from bot.indicators import get_framework_performance

# Get performance analysis
performance = get_framework_performance()

print("Performance Summary:")
print(f"  Total indicators: {performance['summary']['total_indicators']}")
print(f"  Average time: {performance['summary']['avg_time_ms']:.2f}ms")

# Check for slow indicators
for indicator in performance['slow_indicators']:
    print(f"Slow indicator: {indicator['name']} ({indicator['avg_time_ms']:.2f}ms)")

# Get optimization suggestions
for suggestion in performance['optimization_suggestions']:
    print(f"Suggestion: {suggestion['suggestion']} (Priority: {suggestion['priority']})")
```

## Available Indicators by Timeframe

### Scalping Strategy (15s-1m)
- **vumanchu_cipher_a**: VuManChu Cipher A with optimized 3/5/2 periods
- **fast_ema**: Ultra-fast EMA [3, 5, 8] with incremental updates
- **scalping_momentum**: Fast RSI (7), MACD (5/10/3), Williams %R (7)
- **scalping_volume**: VWAP (20), Volume Profile (50 bins)

### Momentum Strategy (1m-5m)
- **vumanchu_cipher_a**: VuManChu Cipher A with standard 9/13/3 periods
- **vumanchu_cipher_b**: VuManChu Cipher B for divergence detection
- **fast_ema**: EMA ribbon [12, 26, 50] for trend analysis
- **scalping_momentum**: Standard RSI (14), MACD (12/26/9), Williams %R (14)
- **scalping_volume**: VWAP (50), Volume analysis with daily reset

### Swing Strategy (5m-15m)
- **vumanchu_cipher_a**: VuManChu Cipher A with 10/21/4 periods
- **vumanchu_cipher_b**: Full divergence analysis
- **scalping_momentum**: Standard momentum indicators
- **scalping_volume**: Extended volume analysis

## Timeframe-Specific Optimizations

### Scalping Timeframe (15s-1m)
```python
{
    'vumanchu_cipher_a': {
        'wt_channel_length': 3,      # Faster response
        'wt_average_length': 5,      # Reduced smoothing
        'wt_ma_length': 2,           # Minimal lag
        'overbought_level': 45.0,    # Earlier signals
        'oversold_level': -45.0,
        'cache_duration': 15         # Short cache for real-time
    },
    'fast_ema': {
        'periods': [3, 5, 8],        # Ultra-fast periods
        'supports_incremental': True, # Real-time updates
        'cache_duration': 10
    }
}
```

### Momentum Timeframe (1m-5m)
```python
{
    'vumanchu_cipher_a': {
        'wt_channel_length': 9,      # Balanced response
        'wt_average_length': 13,     # Standard smoothing
        'wt_ma_length': 3,           # Moderate lag
        'overbought_level': 60.0,    # Standard levels
        'oversold_level': -60.0,
        'cache_duration': 30         # Longer cache OK
    },
    'ema_ribbon': {
        'periods': [12, 26, 50],     # Standard EMA periods
        'cache_duration': 45
    }
}
```

## Performance Characteristics

### Target Performance
- **< 50ms**: Complete indicator calculation for scalping
- **< 100ms**: Complete indicator calculation for momentum
- **> 80%**: Cache hit rate for repeated calculations
- **Thread-safe**: Concurrent calculation support

### Optimization Features
1. **Dependency Graph**: Optimal calculation order
2. **Parallel Execution**: Independent indicators calculated concurrently
3. **Intelligent Caching**: TTL-based with automatic cleanup
4. **Lazy Loading**: Indicators loaded only when needed
5. **Incremental Updates**: Real-time updates without full recalculation

## Error Handling

The framework includes comprehensive error handling:

```python
# Graceful degradation for missing indicators
try:
    results = await calculate_indicators_for_strategy('scalping', data)
except ImportError as e:
    logger.warning(f"Some indicators unavailable: {e}")
    # Framework continues with available indicators

# Invalid data handling
if not indicator.validate_data(data):
    return {'error': 'Invalid data format', 'indicators': {}}
```

## Integration with Existing Systems

### Strategy Manager Integration
```python
from bot.indicators import unified_framework

class AdaptiveStrategyManager:
    def __init__(self):
        self.framework = unified_framework
    
    async def get_market_signals(self, strategy_type, market_data):
        return await self.framework.calculate_for_strategy(
            strategy_type, market_data
        )
```

### Trading Agent Integration
```python
class LLMAgent:
    async def make_decision(self, market_data):
        # Get optimized indicators for current strategy
        indicators = await calculate_indicators_for_strategy(
            self.strategy_type, market_data
        )
        
        # Use indicators in decision making
        return self.process_with_indicators(indicators)
```

## Configuration Management

### Environment Variables
- `INDICATORS_CACHE_TTL`: Default cache duration (default: 60s)
- `INDICATORS_MAX_PARALLEL`: Max parallel calculations (default: 10)
- `INDICATORS_PERFORMANCE_TRACKING`: Enable performance tracking (default: true)

### Runtime Configuration
```python
# Adjust cache settings
unified_framework.calculator.cache.default_ttl = 30

# Modify calculation semaphore
unified_framework.calculator._calculation_semaphore = asyncio.Semaphore(5)

# Get framework status
status = unified_framework.get_framework_status()
```

## Best Practices

1. **Strategy-Specific Usage**: Use `calculate_indicators_for_strategy()` for automatic optimization
2. **Incremental Updates**: Setup incremental mode for real-time trading
3. **Performance Monitoring**: Regularly check `get_framework_performance()`
4. **Cache Management**: Monitor cache hit rates and adjust TTL accordingly
5. **Error Handling**: Always handle potential ImportError for missing indicators

## Troubleshooting

### Common Issues

**Import Errors**: Framework uses lazy loading - imports happen when indicators are first used
```python
# Check which indicators are available
available = get_available_indicators_for_timeframe('scalping')
```

**Performance Issues**: Monitor slow indicators
```python
performance = get_framework_performance()
slow_indicators = performance['slow_indicators']
```

**Cache Issues**: Clear cache if needed
```python
await unified_framework.calculator.cache.cleanup_expired()
```

## Extension Points

### Adding New Indicators
1. Create indicator class inheriting from `UnifiedIndicatorInterface`
2. Implement required methods (`calculate`, optional `calculate_async`)
3. Register with framework using `register_indicator()`

### Custom Adapters
```python
class CustomIndicatorAdapter(UnifiedIndicatorInterface):
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Custom calculation logic
        return {'custom_value': 42}

# Register the adapter
unified_framework.registry.register_indicator(
    IndicatorConfig(
        name='custom_indicator',
        type=IndicatorType.CUSTOM,
        timeframes=[TimeframeType.SCALPING],
        # ... other config
    ),
    CustomIndicatorAdapter
)
```

This framework provides a robust, scalable foundation for indicator management across all trading strategies while maintaining high performance and ease of use.