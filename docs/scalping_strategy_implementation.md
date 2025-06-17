# Scalping Strategy Implementation

## Overview

The ScalpingStrategy is a sophisticated high-frequency trading strategy optimized for low-volume and ranging market conditions. It captures small, frequent profits during consolidation phases with tight risk management.

## Key Features

### Signal Types
- **Mean Reversion**: RSI/Williams extremes near support/resistance levels
- **Micro Breakouts**: Small range breakouts with volume confirmation  
- **VWAP Bounces**: Price bounces off VWAP levels and bands
- **Support/Resistance**: Touch/bounce patterns at key levels
- **Momentum Spikes**: Short-term momentum bursts with volume
- **Volume Anomalies**: Unusual volume patterns indicating moves

### Architecture Components

#### 1. ScalpingMicrostructureAnalyzer
- Identifies support/resistance levels using swing highs/lows
- Analyzes current price position within ranges
- Detects breakout potential based on volume and momentum
- Optimized for 15-second timeframe analysis

#### 2. ScalpingMomentumAnalyzer  
- Fast RSI (7-period) with 25/75 oversold/overbought levels
- Williams %R (7-period) with -85/-15 sensitivity levels
- Ultra-fast EMAs (3, 5, 8 periods) for micro-trend detection
- Real-time reversal signal detection

#### 3. ScalpingVWAPAnalyzer
- Multiple VWAP periods (20, 50) for different timeframes
- VWAP bands using volume-weighted standard deviation
- Bounce signal detection at VWAP levels
- Volume-weighted momentum calculation

#### 4. ScalpingSignalGenerator
- Coordinates all analysis components
- Generates prioritized signals with timing urgency
- Filters signals based on quality and market conditions
- Target <10ms signal generation time

#### 5. ScalpingRiskManager
- Dynamic position sizing based on confidence and frequency
- Consecutive loss protection with cooldown periods
- Daily trade limits and risk level monitoring
- Real-time validation of signal execution

#### 6. ScalpingPerformanceTracker
- Comprehensive trade and signal tracking
- Win rate, holding time, and profit analysis
- Signal type performance breakdown
- Scalping efficiency scoring

### Configuration

```python
@dataclass
class ScalpingConfig:
    # Profit targets (basis points)
    min_profit_target_pct: float = 0.03  # 3 bps minimum
    max_profit_target_pct: float = 0.08  # 8 bps maximum
    stop_loss_pct: float = 0.02          # 2 bps stop loss
    
    # Timing controls
    max_holding_time: int = 300          # 5 minutes max
    quick_exit_threshold: int = 60       # 1 minute quick exit
    immediate_exit_time: int = 15        # 15 seconds immediate exit
    
    # Position sizing
    base_position_pct: float = 0.5       # 0.5% of account
    max_position_pct: float = 1.5        # 1.5% maximum
    
    # Risk management
    max_consecutive_losses: int = 3      # Loss limit
    max_daily_trades: int = 200          # Daily trade limit
    cooldown_after_loss: int = 30        # 30 second cooldown
```

## Usage

### Basic Integration

```python
from bot.strategy.scalping_strategy import create_scalping_strategy

# Create strategy with custom configuration
config_overrides = {
    'min_profit_target_pct': 0.02,
    'max_holding_time': 180,
    'max_daily_trades': 100
}

strategy = create_scalping_strategy(config_overrides)

# Process market data
market_data = {
    'ohlcv': ohlcv_dataframe,
    'current_price': current_price,
    'timestamp': time.time(),
    'account_balance': account_balance
}

result = await strategy.analyze_and_signal(market_data)
```

### Integration with Adaptive Strategy Manager

The scalping strategy is designed to integrate seamlessly with the adaptive strategy manager:

```python
# In adaptive_strategy_manager.py
from .scalping_strategy import ScalpingStrategy

class AdaptiveStrategyManager:
    def _initialize_strategies(self):
        self.strategies[TradingStrategy.SCALPING] = ScalpingStrategy(
            config=self._get_scalping_config()
        )
    
    async def _execute_scalping_strategy(self, market_data):
        strategy = self.strategies[TradingStrategy.SCALPING]
        return await strategy.analyze_and_signal(market_data)
```

## Signal Output Format

```python
{
    'strategy_type': 'scalping',
    'action': 'LONG',  # LONG, SHORT, HOLD
    'size_pct': 75,    # Position size percentage
    'take_profit_pct': 0.4,  # Take profit percentage
    'stop_loss_pct': 0.2,    # Stop loss percentage
    'rationale': 'mean_reversion: Momentum oversold, Near key level',
    'execution_speed': 'high_frequency',
    'signals': [...],  # Detailed signal information
    'market_analysis': {
        'analysis_time_ms': 8.5,
        'strategy_state': 'scanning',
        'signal_count': 3,
        'approved_signal_count': 1
    },
    'risk_assessment': {
        'risk_level': 'low',
        'daily_trades': 15,
        'consecutive_losses': 0
    }
}
```

## Performance Optimization

### Target Metrics
- **Analysis Time**: <10ms per cycle
- **Win Rate**: >60%
- **Average Profit**: >0.1% (10 basis points)
- **Max Drawdown**: <2%
- **Trades per Hour**: 5-20 depending on conditions

### Optimization Techniques
1. **Vectorized Calculations**: All indicators use numpy vectorization
2. **Memory Efficiency**: Bounded collections and cleanup
3. **Early Termination**: Skip analysis on insufficient data
4. **Signal Caching**: Avoid recalculation within timeout periods
5. **Thread Safety**: All operations designed for concurrent access

## Market Conditions

### Optimal Conditions
- **Low Volatility**: <1% daily price range
- **Ranging Markets**: Clear support/resistance levels
- **Consistent Volume**: Regular trading activity
- **Tight Spreads**: <2 basis points bid/ask spread

### Avoid During
- **High Impact News**: Economic releases, earnings
- **Market Opens/Closes**: Increased volatility
- **Low Liquidity**: Thin order books
- **Trending Markets**: Strong directional moves

## Risk Management

### Position Sizing
- Dynamic sizing based on signal confidence
- Frequency adjustment for high-frequency trading
- Consecutive loss protection with size reduction
- Maximum position limits per signal type

### Exit Conditions
- **Immediate**: Signal expires or risk factors increase
- **Quick**: Profit target hit within 1 minute  
- **Normal**: Standard time-based exit at max holding time
- **Emergency**: Stop loss triggered or margin concerns

### Daily Limits
- Maximum number of trades per day
- Maximum drawdown limits
- Cooling-off periods after losses
- Real-time risk monitoring

## Testing and Validation

### Unit Tests
Run specific scalping strategy tests:
```bash
poetry run pytest tests/unit/test_scalping_strategy.py -v
```

### Integration Tests
Test with market data:
```bash
poetry run pytest tests/integration/test_scalping_integration.py -v
```

### Performance Tests
Benchmark analysis speed:
```bash
poetry run python tests/performance/test_scalping_performance.py
```

## Monitoring and Alerts

### Key Metrics to Monitor
- Win rate trend over rolling periods
- Average holding time vs targets
- Signal generation frequency
- Risk level escalation
- Performance vs benchmark

### Alert Conditions
- Win rate drops below 50%
- Consecutive losses exceed limit
- Analysis time exceeds 50ms
- Daily trade limit approached
- Risk level elevated to 'high'

## Future Enhancements

### Planned Features
1. **Machine Learning Integration**: Signal confidence scoring
2. **Multi-Asset Support**: Cross-asset scalping opportunities
3. **Market Microstructure**: Order book analysis
4. **Execution Optimization**: Smart order routing
5. **Regime Detection**: Automatic parameter adjustment

### Advanced Configurations
- Asset-specific parameter sets
- Time-of-day adjustments
- Volatility-based scaling
- News-aware signal filtering
- Cross-timeframe confirmation