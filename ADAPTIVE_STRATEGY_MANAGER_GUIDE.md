# Adaptive Strategy Manager Guide

## Overview

The Adaptive Strategy Manager is a sophisticated trading system component that dynamically switches between different trading strategies based on real-time market regime analysis. It provides seamless transitions between momentum trading, scalping, breakout trading, and defensive strategies while maintaining optimal risk management.

## Key Features

### 🎯 Dynamic Strategy Selection
- **Market Regime Analysis**: Analyzes current market conditions using correlation analysis, sentiment indicators, and technical patterns
- **Intelligent Strategy Mapping**: Maps market regimes to optimal trading strategies
- **Confidence-Based Decisions**: Only switches strategies when confidence levels are sufficiently high

### 🔄 Smooth Strategy Transitions
- **Transition Types**: Supports immediate and gradual transitions based on strategy compatibility
- **Position Management**: Handles existing positions during transitions (close, modify, or keep)
- **Risk-Aware Switching**: Considers market volatility and position exposure when transitioning

### 📊 Performance Monitoring
- **Real-Time Tracking**: Monitors win rates, profit factors, and effectiveness scores for each strategy
- **Regime-Specific Performance**: Tracks how strategies perform under different market conditions
- **Adaptive Optimization**: Adjusts strategy parameters based on historical performance

### 🤖 LLM Integration
- **Context Provider**: Supplies comprehensive market context and strategy recommendations to LLM agents
- **Decision Support**: Provides reasoning and confidence scores for strategy recommendations
- **Performance Insights**: Delivers strategy comparison and adaptation assessments

## Strategy Types

### 1. Momentum Strategy (`TradingStrategy.MOMENTUM`)
**Best For**: Trending markets with clear directional movement

**Configuration**:
- **Timeframe**: 1 minute
- **Position Size**: 2-5% of capital
- **Risk Management**: 0.8% stop loss, 2.0% take profit
- **Holding Time**: Up to 30 minutes
- **Key Indicators**: EMA ribbon, MACD, ADX, volume analysis

**Entry Conditions**:
- Trend strength > 70%
- Volume confirmation required
- Risk/reward ratio > 2.0

### 2. Scalping Strategy (`TradingStrategy.SCALPING`)
**Best For**: Range-bound, low-volatility markets

**Configuration**:
- **Timeframe**: 15 seconds
- **Position Size**: 0.5-1.5% of capital
- **Risk Management**: 0.3% stop loss, 0.6% take profit
- **Holding Time**: Up to 5 minutes
- **Key Indicators**: Fast EMA, VWAP, volume profile, Williams %R

**Entry Conditions**:
- Signal strength > 60%
- Bid-ask spread < 2 basis points
- Quick profit potential confirmed

### 3. Breakout Strategy (`TradingStrategy.BREAKOUT`)
**Best For**: Consolidation periods with high breakout potential

**Configuration**:
- **Timeframe**: 5 minutes
- **Position Size**: 3-8% of capital
- **Risk Management**: 1.2% stop loss, 4.0% take profit
- **Holding Time**: Up to 60 minutes
- **Key Indicators**: Bollinger Bands, volume breakout, support/resistance

**Entry Conditions**:
- Breakout strength > 80%
- Volume surge > 150%
- Price acceleration confirmed

### 4. Defensive Strategy (`TradingStrategy.DEFENSIVE`)
**Best For**: Uncertain or high-risk market conditions

**Configuration**:
- **Timeframe**: 5 minutes
- **Position Size**: 0.5-2% of capital
- **Risk Management**: 0.5% stop loss, 1.0% take profit
- **Holding Time**: Up to 20 minutes
- **Key Indicators**: Risk metrics, correlation analysis, volatility

**Entry Conditions**:
- Risk-adjusted return > 30%
- Maximum correlation < 80%
- Defensive score > 70%

## Usage Examples

### Basic Implementation

```python
from bot.strategy.adaptive_strategy_manager import AdaptiveStrategyManager
from bot.analysis.market_context import MarketContextAnalyzer

# Initialize the system
analyzer = MarketContextAnalyzer()
strategy_manager = AdaptiveStrategyManager(analyzer)

# Execute trading cycle
async def trading_cycle(market_state):
    # Analyze market and execute optimal strategy
    result = await strategy_manager.analyze_and_execute(market_state)
    
    # Get current strategy info
    active_strategy = result['active_strategy']['name']
    confidence = result['strategy_decision']['confidence']
    
    print(f"Active Strategy: {active_strategy} (confidence: {confidence:.1%})")
    
    return result
```

### LLM Integration

```python
# Get context for LLM decision making
llm_context = strategy_manager.context_provider.get_llm_context(market_analysis)

# Context includes:
# - Current strategy status and performance
# - Strategy recommendations with reasoning
# - Transition readiness and risks
# - Performance comparisons and market adaptation

# Use context in LLM prompt
prompt = f"""
Current Strategy: {llm_context['current_strategy']['name']}
Recommended: {llm_context['strategy_recommendations']['primary']['strategy']}
Confidence: {llm_context['strategy_recommendations']['confidence']:.1%}

Market Analysis: {market_analysis}

Please provide trading decision based on current strategy context.
"""
```

### Performance Monitoring

```python
# Get comprehensive performance summary
performance = strategy_manager.get_strategy_performance()

# Analyze strategy effectiveness
for strategy_name, data in performance['strategies'].items():
    effectiveness = data['effectiveness']
    win_rate = data['metrics']['win_rate']
    total_trades = data['metrics']['total_trades']
    
    print(f"{strategy_name}: {effectiveness:.1%} effective, "
          f"{win_rate:.1%} win rate, {total_trades} trades")

# Get best performing strategy
best_strategy = performance.get('best_strategy', {})
print(f"Best Strategy: {best_strategy.get('name', 'None')} "
      f"({best_strategy.get('effectiveness', 0):.1%})")
```

### Manual Strategy Override

```python
# Force strategy change for testing or manual control
success = strategy_manager.force_strategy_change(TradingStrategy.SCALPING)

if success:
    print(f"Strategy changed to: {strategy_manager.current_strategy_name.value}")
else:
    print("Failed to change strategy")
```

## Integration with Existing Bot

### 1. Update Main Trading Loop

```python
# In bot/main.py or similar
from bot.strategy.adaptive_strategy_manager import AdaptiveStrategyManager

class TradingBot:
    def __init__(self):
        self.strategy_manager = AdaptiveStrategyManager()
        # ... other initialization
    
    async def trading_loop(self):
        while True:
            # Get market data
            market_state = await self.get_market_state()
            
            # Execute adaptive strategy
            strategy_result = await self.strategy_manager.analyze_and_execute(market_state)
            
            # Process results
            await self.process_strategy_result(strategy_result)
            
            # Wait for next cycle
            await asyncio.sleep(self.trading_interval)
```

### 2. Enhance LLM Agent

```python
# In bot/strategy/llm_agent.py
class LLMAgent:
    def __init__(self, strategy_manager=None):
        self.strategy_manager = strategy_manager
        # ... other initialization
    
    async def make_decision(self, market_state):
        # Get strategy context if available
        strategy_context = {}
        if self.strategy_manager:
            regime_analysis = await self.strategy_manager._perform_regime_analysis(market_state)
            strategy_context = self.strategy_manager.context_provider.get_llm_context(regime_analysis)
        
        # Enhanced prompt with strategy context
        enhanced_prompt = self.build_prompt(market_state, strategy_context)
        
        # ... continue with LLM processing
```

### 3. Risk Management Integration

```python
# In bot/risk.py
class RiskManager:
    def __init__(self, strategy_manager=None):
        self.strategy_manager = strategy_manager
        
    def get_dynamic_position_size(self, base_size, market_conditions):
        if self.strategy_manager:
            current_strategy = self.strategy_manager.current_strategy_name
            config = self.strategy_manager.strategy_configs.get(current_strategy)
            
            if config:
                # Adjust size based on strategy configuration
                strategy_multiplier = config.position_sizing.get('size_multiplier', 1.0)
                return base_size * strategy_multiplier
        
        return base_size
```

## Configuration Options

### Strategy Transition Rules

```python
# Customize transition behavior
transition_manager = TransitionManager()
transition_manager.transition_rules.update({
    (TradingStrategy.MOMENTUM, TradingStrategy.SCALPING): 'immediate',
    (TradingStrategy.SCALPING, TradingStrategy.DEFENSIVE): 'gradual'
})
```

### Performance Thresholds

```python
# Set minimum performance requirements
strategy_config = StrategyConfig(
    name=TradingStrategy.MOMENTUM,
    performance_thresholds={
        'min_win_rate': 0.65,      # Minimum 65% win rate
        'min_profit_factor': 1.8,  # Minimum 1.8 profit factor
        'max_drawdown': 0.05       # Maximum 5% drawdown
    }
)
```

### Risk Parameters by Strategy

```python
# Customize risk management per strategy
CUSTOM_MOMENTUM_CONFIG = {
    'risk_management': {
        'stop_loss_pct': 1.0,      # Wider stops for momentum
        'take_profit_pct': 3.0,    # Higher targets
        'max_holding_time': 3600,  # 1 hour max
        'trailing_stop': True      # Use trailing stops
    }
}
```

## Monitoring and Debugging

### Enable Detailed Logging

```python
import logging
logging.getLogger('bot.strategy.adaptive_strategy_manager').setLevel(logging.DEBUG)
```

### Performance Metrics

Monitor these key metrics:
- **Strategy Effectiveness**: Overall performance score (0-100%)
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss ratio
- **Transition Success Rate**: Percentage of successful strategy transitions
- **Regime Detection Accuracy**: How well regime analysis predicts optimal strategies

### Common Issues and Solutions

1. **Frequent Strategy Switching**
   - Increase confidence threshold for transitions
   - Extend minimum strategy duration
   - Smooth market regime analysis

2. **Poor Performance in Specific Regimes**
   - Adjust strategy-regime mappings
   - Fine-tune entry/exit criteria
   - Update performance thresholds

3. **Transition Conflicts**
   - Review position management during transitions
   - Adjust transition rules for strategy pairs
   - Implement position-aware transition logic

## Advanced Features

### Custom Strategy Implementation

```python
class CustomStrategy(StrategyExecutor):
    async def execute_strategy(self, market_data):
        # Implement custom strategy logic
        analysis = await self._custom_analysis(market_data)
        signals = await self._generate_custom_signals(analysis)
        
        return {
            'strategy': 'CUSTOM',
            'signals': signals,
            'execution': await self._execute_custom_logic(signals),
            'analysis': analysis
        }
```

### Dynamic Configuration Updates

```python
# Update strategy configuration at runtime
strategy_manager.strategy_configs[TradingStrategy.MOMENTUM].position_sizing['base_size_pct'] = 3.0
strategy_manager.strategy_configs[TradingStrategy.MOMENTUM].risk_parameters['stop_loss_pct'] = 0.9
```

### Performance-Based Strategy Selection

```python
# Override strategy selection based on historical performance
def performance_weighted_selection(regime_analysis, performance_tracker):
    base_strategy, confidence = selector.select_strategy(regime_analysis)
    
    # Adjust based on recent performance
    effectiveness = performance_tracker.get_strategy_effectiveness(base_strategy)
    
    if effectiveness < 0.4:  # Poor recent performance
        # Try alternative strategy
        alternatives = [TradingStrategy.DEFENSIVE, TradingStrategy.SCALPING]
        base_strategy = max(alternatives, 
                          key=lambda s: performance_tracker.get_strategy_effectiveness(s))
    
    return base_strategy, confidence * effectiveness
```

## Best Practices

1. **Gradual Rollout**: Start with conservative thresholds and gradually optimize
2. **Backtesting**: Test strategy configurations on historical data before live deployment
3. **Monitor Performance**: Regularly review strategy effectiveness and adjust parameters
4. **Risk Management**: Always prioritize risk control over profit maximization
5. **Market Adaptation**: Allow the system to learn and adapt to changing market conditions

## Future Enhancements

- **Machine Learning Integration**: Use ML models for strategy selection and optimization
- **Multi-Asset Support**: Extend to multiple trading pairs and asset classes
- **Real-Time Optimization**: Dynamic parameter adjustment based on live performance
- **Advanced Risk Models**: Integration with sophisticated risk management systems
- **Portfolio-Level Strategy**: Coordinate strategies across multiple positions

---

The Adaptive Strategy Manager represents a significant advancement in automated trading systems, providing the flexibility and intelligence needed to navigate diverse market conditions while maintaining robust risk management and performance optimization.