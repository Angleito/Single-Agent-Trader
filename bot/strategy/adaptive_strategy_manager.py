"""
Adaptive Strategy Manager - Dynamically switches between trading strategies based on market regime analysis.

This module provides a sophisticated strategy management system that seamlessly transitions 
between different trading approaches (momentum, scalping, breakout, defensive) based on 
real-time market conditions detected by the regime analyzer.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..analysis.market_context import MarketContextAnalyzer, MarketRegimeType
from ..config import settings
from ..indicators import (
    CipherA, CipherB, EMAribbon, FastEMA, FastRSI, FastMACD, 
    ScalpingVWAP, VolumeProfile, WilliamsPercentR
)
from ..types import MarketState, Position, TradeAction
from ..indicators import calculate_atr, calculate_support_resistance

logger = logging.getLogger(__name__)


class TradingStrategy(str, Enum):
    """Trading strategy types."""
    MOMENTUM = "momentum"          # Trend-following for strong moves
    SCALPING = "scalping"          # High-frequency for ranging markets  
    BREAKOUT = "breakout"          # Specialized for breakout scenarios
    DEFENSIVE = "defensive"        # Risk-off during uncertain markets
    HOLD = "hold"                  # No trading during poor conditions


class StrategyState(str, Enum):
    """Strategy execution states."""
    ACTIVE = "active"              # Currently executing strategy
    TRANSITIONING = "transitioning" # Switching between strategies
    WARMING_UP = "warming_up"      # Initializing new strategy
    COOLING_DOWN = "cooling_down"  # Winding down old strategy


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: TradingStrategy
    timeframe: str
    indicator_set: Dict[str, List[str]]
    position_sizing: Dict[str, float]
    risk_parameters: Dict[str, float]
    execution_parameters: Dict[str, Any]
    performance_thresholds: Dict[str, float]
    
    # Strategy-specific settings
    momentum_config: Optional[Dict[str, Any]] = None
    scalping_config: Optional[Dict[str, Any]] = None
    breakout_config: Optional[Dict[str, Any]] = None


# Strategy Configuration Constants
MOMENTUM_CONFIG = {
    'timeframe': '1m',
    'indicators': {
        'primary': ['ema_ribbon', 'macd', 'adx'],
        'confirmation': ['volume_analysis', 'rsi_divergence'],
        'filter': ['atr', 'support_resistance']
    },
    'entry_conditions': {
        'trend_strength_min': 0.7,
        'volume_confirmation': True,
        'momentum_alignment': True,
        'risk_reward_min': 2.0
    },
    'position_sizing': {
        'base_size_pct': 2.0,
        'volatility_adjusted': True,
        'max_size_pct': 5.0,
        'size_multiplier': 1.5  # For strong signals
    },
    'risk_management': {
        'stop_loss_pct': 0.8,
        'take_profit_pct': 2.0,
        'trailing_stop': True,
        'max_holding_time': 1800,  # 30 minutes
        'max_drawdown_pct': 3.0
    },
    'execution': {
        'entry_timeout': 30,
        'partial_fills': True,
        'slippage_tolerance': 0.05
    }
}

SCALPING_CONFIG = {
    'timeframe': '15s',
    'indicators': {
        'primary': ['fast_ema', 'vwap', 'microstructure'],
        'confirmation': ['fast_rsi', 'williams_r', 'volume_profile'],
        'filter': ['bid_ask_spread', 'tick_analysis']
    },
    'entry_conditions': {
        'signal_strength_min': 0.6,
        'spread_max_bps': 2.0,
        'volume_min_relative': 0.8,
        'quick_profit_potential': True
    },
    'position_sizing': {
        'base_size_pct': 0.5,
        'frequency_adjusted': True,
        'max_size_pct': 1.5,
        'size_multiplier': 0.8  # Smaller positions
    },
    'risk_management': {
        'stop_loss_pct': 0.3,
        'take_profit_pct': 0.6,
        'trailing_stop': False,
        'max_holding_time': 300,   # 5 minutes
        'max_drawdown_pct': 1.5
    },
    'execution': {
        'entry_timeout': 5,
        'partial_fills': False,
        'slippage_tolerance': 0.02
    }
}

BREAKOUT_CONFIG = {
    'timeframe': '5m',
    'indicators': {
        'primary': ['bollinger_bands', 'volume_breakout', 'price_action'],
        'confirmation': ['momentum_oscillator', 'support_resistance_break'],
        'filter': ['false_breakout_filter', 'time_of_day']
    },
    'entry_conditions': {
        'breakout_strength_min': 0.8,
        'volume_surge_min': 1.5,
        'price_acceleration': True,
        'continuation_probability': 0.6
    },
    'position_sizing': {
        'base_size_pct': 3.0,
        'breakout_adjusted': True,
        'max_size_pct': 8.0,
        'size_multiplier': 2.0  # Larger positions for breakouts
    },
    'risk_management': {
        'stop_loss_pct': 1.2,
        'take_profit_pct': 4.0,
        'trailing_stop': True,
        'max_holding_time': 3600,  # 60 minutes
        'max_drawdown_pct': 4.0
    },
    'execution': {
        'entry_timeout': 15,
        'partial_fills': True,
        'slippage_tolerance': 0.08
    }
}

DEFENSIVE_CONFIG = {
    'timeframe': '5m',
    'indicators': {
        'primary': ['risk_metrics', 'correlation_analysis', 'volatility'],
        'confirmation': ['market_breadth', 'sentiment_indicators'],
        'filter': ['regime_stability', 'uncertainty_measures']
    },
    'entry_conditions': {
        'risk_adjusted_return': 0.3,
        'max_correlation': 0.8,
        'volatility_threshold': 0.4,
        'defensive_score_min': 0.7
    },
    'position_sizing': {
        'base_size_pct': 0.5,
        'risk_adjusted': True,
        'max_size_pct': 2.0,
        'size_multiplier': 0.5  # Very small positions
    },
    'risk_management': {
        'stop_loss_pct': 0.5,
        'take_profit_pct': 1.0,
        'trailing_stop': False,
        'max_holding_time': 1200,  # 20 minutes
        'max_drawdown_pct': 1.0
    },
    'execution': {
        'entry_timeout': 10,
        'partial_fills': False,
        'slippage_tolerance': 0.03
    }
}


class StrategySelector:
    """Selects optimal trading strategy based on market regime analysis."""
    
    def __init__(self):
        """Initialize the strategy selector."""
        self.regime_strategy_map = {
            MarketRegimeType.RISK_ON: TradingStrategy.MOMENTUM,
            MarketRegimeType.RISK_OFF: TradingStrategy.DEFENSIVE,
            MarketRegimeType.TRANSITION: TradingStrategy.SCALPING,
            MarketRegimeType.UNKNOWN: TradingStrategy.HOLD
        }
        
        # Additional market condition mappings
        self.volatility_strategy_map = {
            'HIGH': TradingStrategy.BREAKOUT,
            'NORMAL': TradingStrategy.MOMENTUM,
            'LOW': TradingStrategy.SCALPING
        }
        
        logger.info("Initialized StrategySelector")
        
    def select_strategy(self, regime_analysis: Dict[str, Any]) -> Tuple[TradingStrategy, float]:
        """
        Select optimal strategy based on regime analysis.
        
        Args:
            regime_analysis: Market regime analysis results
            
        Returns:
            Tuple of (selected_strategy, confidence_score)
        """
        try:
            # Extract regime information
            current_regime = regime_analysis.get('regime', {}).get('regime_type', MarketRegimeType.UNKNOWN)
            regime_confidence = regime_analysis.get('regime', {}).get('confidence', 0.0)
            
            # Base strategy selection from regime
            base_strategy = self.regime_strategy_map.get(current_regime, TradingStrategy.HOLD)
            
            # Apply confidence adjustments
            if regime_confidence < 0.6:
                base_strategy = TradingStrategy.DEFENSIVE
            elif regime_confidence < 0.4:
                base_strategy = TradingStrategy.HOLD
                
            # Consider additional market conditions
            volatility_regime = regime_analysis.get('regime', {}).get('market_volatility_regime', 'NORMAL')
            
            # Override with volatility-based strategy if appropriate
            if volatility_regime in self.volatility_strategy_map:
                volatility_strategy = self.volatility_strategy_map[volatility_regime]
                
                # Use volatility strategy if confidence is high enough
                if regime_confidence > 0.7:
                    strategy = volatility_strategy
                else:
                    # Blend decision based on confidence
                    strategy = base_strategy
            else:
                strategy = base_strategy
                
            # Calculate strategy confidence
            strategy_confidence = self._calculate_strategy_confidence(strategy, regime_analysis)
            
            return strategy, strategy_confidence
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}", exc_info=True)
            return TradingStrategy.HOLD, 0.0
    
    def _calculate_strategy_confidence(
        self, 
        strategy: TradingStrategy, 
        regime_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence in the selected strategy.
        
        Args:
            strategy: Selected trading strategy
            regime_analysis: Market regime analysis
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            confidence_factors = []
            
            # Base regime confidence
            regime_confidence = regime_analysis.get('regime', {}).get('confidence', 0.0)
            confidence_factors.append(regime_confidence)
            
            # Correlation reliability (if available)
            correlation_reliability = regime_analysis.get('correlation', {}).get('reliability_score', 0.5)
            confidence_factors.append(correlation_reliability * 0.8)
            
            # Momentum alignment (if available)
            momentum_sustainability = regime_analysis.get('momentum', {}).get('momentum_sustainability', 0.5)
            confidence_factors.append(momentum_sustainability * 0.6)
            
            # Strategy-specific adjustments
            if strategy == TradingStrategy.MOMENTUM:
                # Check for trend strength indicators
                trend_strength = regime_analysis.get('technical', {}).get('trend_strength', 0.5)
                confidence_factors.append(trend_strength)
            elif strategy == TradingStrategy.SCALPING:
                # Check for range-bound conditions
                volatility_score = 1.0 - regime_analysis.get('technical', {}).get('volatility_normalized', 0.5)
                confidence_factors.append(volatility_score)
            elif strategy == TradingStrategy.BREAKOUT:
                # Check for breakout conditions
                breakout_potential = regime_analysis.get('technical', {}).get('breakout_score', 0.5)
                confidence_factors.append(breakout_potential)
            
            # Calculate weighted average
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating strategy confidence: {e}", exc_info=True)
            return 0.5


class TransitionManager:
    """Manages smooth transitions between trading strategies."""
    
    def __init__(self):
        """Initialize the transition manager."""
        self.transition_rules = {
            (TradingStrategy.MOMENTUM, TradingStrategy.SCALPING): 'gradual',
            (TradingStrategy.SCALPING, TradingStrategy.MOMENTUM): 'immediate',
            (TradingStrategy.MOMENTUM, TradingStrategy.BREAKOUT): 'immediate',
            (TradingStrategy.BREAKOUT, TradingStrategy.MOMENTUM): 'gradual',
            (TradingStrategy.BREAKOUT, TradingStrategy.SCALPING): 'gradual',
            (TradingStrategy.SCALPING, TradingStrategy.BREAKOUT): 'immediate',
            (TradingStrategy.DEFENSIVE, TradingStrategy.MOMENTUM): 'gradual',
            (TradingStrategy.MOMENTUM, TradingStrategy.DEFENSIVE): 'immediate',
        }
        self.min_strategy_duration = 300  # 5 minutes minimum
        self.transition_timeout = 60      # 1 minute max transition
        
        logger.info("Initialized TransitionManager")
        
    async def execute_transition(
        self, 
        from_strategy: TradingStrategy,
        to_strategy: TradingStrategy,
        market_context: Dict[str, Any],
        position_manager = None
    ) -> bool:
        """
        Execute strategy transition.
        
        Args:
            from_strategy: Current strategy
            to_strategy: Target strategy
            market_context: Current market context
            position_manager: Position manager for handling positions
            
        Returns:
            True if transition successful, False otherwise
        """
        try:
            if from_strategy == to_strategy:
                return True
                
            logger.info(f"Executing transition from {from_strategy.value} to {to_strategy.value}")
            
            # Determine transition type
            transition_type = self.transition_rules.get(
                (from_strategy, to_strategy), 'gradual'
            )
            
            # Handle positions during transition
            if position_manager:
                transition_plan = await self._handle_positions_during_transition(
                    from_strategy, to_strategy, position_manager
                )
                
                # Execute position adjustments
                await self._execute_position_adjustments(transition_plan, position_manager)
            
            # Execute transition based on type
            if transition_type == 'immediate':
                success = await self._immediate_transition(from_strategy, to_strategy)
            else:
                success = await self._gradual_transition(from_strategy, to_strategy, market_context)
                
            if success:
                logger.info(f"Successfully transitioned to {to_strategy.value}")
            else:
                logger.warning(f"Failed to transition to {to_strategy.value}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error executing transition: {e}", exc_info=True)
            return False
    
    async def _handle_positions_during_transition(
        self,
        from_strategy: TradingStrategy,
        to_strategy: TradingStrategy,
        position_manager
    ) -> Dict[str, List]:
        """
        Handle existing positions during strategy transition.
        
        Args:
            from_strategy: Strategy being transitioned from
            to_strategy: Strategy being transitioned to  
            position_manager: Position manager instance
            
        Returns:
            Dictionary with transition plan
        """
        try:
            # Get current positions (mock implementation)
            current_positions = []  # Would get from position_manager
            
            transition_plan = {
                'positions_to_close': [],
                'positions_to_modify': [],
                'positions_to_keep': []
            }
            
            for position in current_positions:
                if self._position_compatible_with_new_strategy(position, to_strategy):
                    # Modify position parameters for new strategy
                    new_params = self._adapt_position_parameters(position, to_strategy)
                    transition_plan['positions_to_modify'].append((position, new_params))
                elif self._should_keep_position_during_transition(position, from_strategy):
                    transition_plan['positions_to_keep'].append(position)
                else:
                    transition_plan['positions_to_close'].append(position)
            
            return transition_plan
            
        except Exception as e:
            logger.error(f"Error handling positions during transition: {e}", exc_info=True)
            return {'positions_to_close': [], 'positions_to_modify': [], 'positions_to_keep': []}
    
    def _position_compatible_with_new_strategy(self, position: Position, strategy: TradingStrategy) -> bool:
        """Check if position is compatible with new strategy."""
        # Strategy compatibility logic
        if strategy == TradingStrategy.SCALPING:
            # Scalping prefers smaller, shorter-term positions
            return position.size < 1000  # Example threshold
        elif strategy == TradingStrategy.MOMENTUM:
            # Momentum can handle larger positions
            return True
        elif strategy == TradingStrategy.DEFENSIVE:
            # Defensive prefers very small positions
            return position.size < 500
        else:
            return True
    
    def _should_keep_position_during_transition(self, position: Position, from_strategy: TradingStrategy) -> bool:
        """Determine if position should be kept during transition."""
        # Keep profitable positions that are within risk limits
        return position.unrealized_pnl > 0 and position.size < 2000
    
    def _adapt_position_parameters(self, position: Position, strategy: TradingStrategy) -> Dict[str, Any]:
        """Adapt position parameters for new strategy."""
        new_params = {}
        
        if strategy == TradingStrategy.SCALPING:
            new_params['stop_loss_pct'] = 0.3
            new_params['take_profit_pct'] = 0.6
        elif strategy == TradingStrategy.MOMENTUM:
            new_params['stop_loss_pct'] = 0.8
            new_params['take_profit_pct'] = 2.0
        elif strategy == TradingStrategy.BREAKOUT:
            new_params['stop_loss_pct'] = 1.2
            new_params['take_profit_pct'] = 4.0
        elif strategy == TradingStrategy.DEFENSIVE:
            new_params['stop_loss_pct'] = 0.5
            new_params['take_profit_pct'] = 1.0
            
        return new_params
    
    async def _execute_position_adjustments(self, transition_plan: Dict[str, List], position_manager) -> None:
        """Execute position adjustments according to transition plan."""
        try:
            # Close positions marked for closure
            for position in transition_plan['positions_to_close']:
                logger.info(f"Closing position {position.symbol} for strategy transition")
                # Would execute close order through position_manager
                
            # Modify positions as needed
            for position, new_params in transition_plan['positions_to_modify']:
                logger.info(f"Modifying position {position.symbol} for new strategy")
                # Would modify stop/take profit levels through position_manager
                
        except Exception as e:
            logger.error(f"Error executing position adjustments: {e}", exc_info=True)
    
    async def _immediate_transition(self, from_strategy: TradingStrategy, to_strategy: TradingStrategy) -> bool:
        """Execute immediate strategy transition."""
        try:
            # Immediate transitions require quick action
            # Stop current strategy operations
            logger.info(f"Immediate transition from {from_strategy.value} to {to_strategy.value}")
            
            # Small delay for system stability
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in immediate transition: {e}", exc_info=True)
            return False
    
    async def _gradual_transition(
        self, 
        from_strategy: TradingStrategy, 
        to_strategy: TradingStrategy,
        market_context: Dict[str, Any]
    ) -> bool:
        """Execute gradual strategy transition."""
        try:
            # Gradual transitions allow for smoother changeover
            logger.info(f"Gradual transition from {from_strategy.value} to {to_strategy.value}")
            
            # Phase 1: Reduce activity in old strategy
            await asyncio.sleep(0.5)
            
            # Phase 2: Initialize new strategy
            await asyncio.sleep(0.3)
            
            # Phase 3: Full activation of new strategy
            await asyncio.sleep(0.2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in gradual transition: {e}", exc_info=True)
            return False


class StrategyExecutor:
    """Executes trading strategies with unified interface."""
    
    def __init__(self, strategy_config: StrategyConfig):
        """
        Initialize strategy executor.
        
        Args:
            strategy_config: Configuration for the strategy
        """
        self.config = strategy_config
        self.state = StrategyState.WARMING_UP
        self.performance_tracker = StrategyPerformanceTracker()
        self.indicators = self._initialize_indicators()
        
        logger.info(f"Initialized StrategyExecutor for {strategy_config.name.value}")
        
    def _initialize_indicators(self) -> Dict[str, Any]:
        """Initialize technical indicators for the strategy."""
        indicators = {}
        
        try:
            # Common indicators
            indicators['cipher_a'] = CipherA()
            indicators['cipher_b'] = CipherB()
            indicators['ema_ribbon'] = EMAribbon()
            
            # Strategy-specific indicators
            if self.config.name == TradingStrategy.SCALPING:
                indicators['fast_ema'] = FastEMA()
                indicators['fast_rsi'] = FastRSI()
                indicators['williams_r'] = WilliamsPercentR()
                indicators['vwap'] = ScalpingVWAP()
                indicators['volume_profile'] = VolumeProfile()
            elif self.config.name == TradingStrategy.MOMENTUM:
                indicators['fast_macd'] = FastMACD()
                indicators['ema_ribbon'] = EMAribbon()
                
        except Exception as e:
            logger.error(f"Error initializing indicators: {e}", exc_info=True)
            
        return indicators
        
    async def execute_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute strategy based on market data.
        
        Args:
            market_data: Current market data and context
            
        Returns:
            Strategy execution results
        """
        try:
            self.state = StrategyState.ACTIVE
            
            # Strategy-specific execution
            if self.config.name == TradingStrategy.MOMENTUM:
                return await self._execute_momentum_strategy(market_data)
            elif self.config.name == TradingStrategy.SCALPING:
                return await self._execute_scalping_strategy(market_data)
            elif self.config.name == TradingStrategy.BREAKOUT:
                return await self._execute_breakout_strategy(market_data)
            elif self.config.name == TradingStrategy.DEFENSIVE:
                return await self._execute_defensive_strategy(market_data)
            else:
                return await self._execute_hold_strategy(market_data)
                
        except Exception as e:
            logger.error(f"Error executing strategy {self.config.name.value}: {e}", exc_info=True)
            return {
                'strategy': self.config.name,
                'signals': [],
                'execution': {'error': str(e)},
                'analysis': {}
            }
    
    async def _execute_momentum_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute momentum trading strategy."""
        try:
            # Analyze with momentum-specific indicators
            analysis = await self._analyze_momentum_indicators(market_data)
            
            # Generate signals with momentum criteria
            signals = await self._generate_momentum_signals(analysis)
            
            # Apply momentum-specific filters
            filtered_signals = self._apply_momentum_filters(signals, market_data)
            
            # Execute with momentum risk management
            execution_result = await self._execute_with_momentum_risk_mgmt(filtered_signals)
            
            return {
                'strategy': TradingStrategy.MOMENTUM,
                'signals': filtered_signals,
                'execution': execution_result,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Error in momentum strategy execution: {e}", exc_info=True)
            return {'strategy': TradingStrategy.MOMENTUM, 'signals': [], 'execution': {}, 'analysis': {}}
    
    async def _execute_scalping_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scalping trading strategy."""
        try:
            # High-frequency analysis
            analysis = await self._analyze_scalping_indicators(market_data)
            
            # Quick signal generation
            signals = await self._generate_scalping_signals(analysis)
            
            # Tight filters for scalping
            filtered_signals = self._apply_scalping_filters(signals, market_data)
            
            # Fast execution with tight risk management
            execution_result = await self._execute_with_scalping_risk_mgmt(filtered_signals)
            
            return {
                'strategy': TradingStrategy.SCALPING,
                'signals': filtered_signals,
                'execution': execution_result,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Error in scalping strategy execution: {e}", exc_info=True)
            return {'strategy': TradingStrategy.SCALPING, 'signals': [], 'execution': {}, 'analysis': {}}
    
    async def _execute_breakout_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute breakout trading strategy."""
        try:
            # Breakout pattern analysis
            analysis = await self._analyze_breakout_patterns(market_data)
            
            # Breakout signal generation
            signals = await self._generate_breakout_signals(analysis)
            
            # Breakout-specific filters
            filtered_signals = self._apply_breakout_filters(signals, market_data)
            
            # Execute with breakout risk management
            execution_result = await self._execute_with_breakout_risk_mgmt(filtered_signals)
            
            return {
                'strategy': TradingStrategy.BREAKOUT,
                'signals': filtered_signals,
                'execution': execution_result,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Error in breakout strategy execution: {e}", exc_info=True)
            return {'strategy': TradingStrategy.BREAKOUT, 'signals': [], 'execution': {}, 'analysis': {}}
    
    async def _execute_defensive_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute defensive trading strategy."""
        try:
            # Risk-focused analysis
            analysis = await self._analyze_risk_metrics(market_data)
            
            # Conservative signal generation
            signals = await self._generate_defensive_signals(analysis)
            
            # Strict risk filters
            filtered_signals = self._apply_defensive_filters(signals, market_data)
            
            # Execute with conservative risk management
            execution_result = await self._execute_with_defensive_risk_mgmt(filtered_signals)
            
            return {
                'strategy': TradingStrategy.DEFENSIVE,
                'signals': filtered_signals,
                'execution': execution_result,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Error in defensive strategy execution: {e}", exc_info=True)
            return {'strategy': TradingStrategy.DEFENSIVE, 'signals': [], 'execution': {}, 'analysis': {}}
    
    async def _execute_hold_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hold strategy (no trading)."""
        return {
            'strategy': TradingStrategy.HOLD,
            'signals': [],
            'execution': {'action': 'HOLD', 'reason': 'Hold strategy active'},
            'analysis': {'market_conditions': 'unfavorable_for_trading'}
        }
    
    # Indicator analysis methods
    async def _analyze_momentum_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze indicators for momentum strategy."""
        analysis = {}
        
        try:
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv_data', [])
            if not ohlcv_data:
                return analysis
                
            # Convert to arrays for analysis
            closes = np.array([candle.close for candle in ohlcv_data[-50:]])  # Last 50 candles
            highs = np.array([candle.high for candle in ohlcv_data[-50:]])
            lows = np.array([candle.low for candle in ohlcv_data[-50:]])
            volumes = np.array([candle.volume for candle in ohlcv_data[-50:]])
            
            # Trend strength analysis
            if len(closes) >= 20:
                ema_20 = closes[-20:].mean()
                ema_50 = closes[-50:].mean() if len(closes) >= 50 else closes.mean()
                
                trend_strength = abs(ema_20 - ema_50) / ema_50 if ema_50 != 0 else 0
                analysis['trend_strength'] = min(trend_strength * 10, 1.0)  # Normalize
                
                # Trend direction
                analysis['trend_direction'] = 'BULLISH' if ema_20 > ema_50 else 'BEARISH'
            
            # Momentum indicators
            if len(closes) >= 14:
                # Simple RSI calculation
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = gains[-14:].mean() if len(gains) >= 14 else gains.mean()
                avg_loss = losses[-14:].mean() if len(losses) >= 14 else losses.mean()
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    analysis['rsi'] = rsi
                    
            # Volume analysis
            if len(volumes) >= 20:
                recent_volume = volumes[-5:].mean()
                avg_volume = volumes[-20:].mean()
                volume_ratio = recent_volume / avg_volume if avg_volume != 0 else 1.0
                analysis['volume_momentum'] = min(volume_ratio, 3.0)  # Cap at 3x
                
        except Exception as e:
            logger.error(f"Error analyzing momentum indicators: {e}", exc_info=True)
            
        return analysis
    
    async def _analyze_scalping_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze indicators for scalping strategy."""
        analysis = {}
        
        try:
            # Extract recent price data
            ohlcv_data = market_data.get('ohlcv_data', [])
            if not ohlcv_data:
                return analysis
                
            # Focus on very recent data for scalping
            recent_closes = np.array([candle.close for candle in ohlcv_data[-20:]])
            recent_volumes = np.array([candle.volume for candle in ohlcv_data[-20:]])
            
            # Short-term momentum
            if len(recent_closes) >= 10:
                fast_ema = recent_closes[-5:].mean()
                slow_ema = recent_closes[-10:].mean()
                
                momentum_score = (fast_ema - slow_ema) / slow_ema if slow_ema != 0 else 0
                analysis['short_term_momentum'] = momentum_score
                
            # Volatility for scalping opportunities
            if len(recent_closes) >= 10:
                price_changes = np.diff(recent_closes)
                volatility = np.std(price_changes) / recent_closes.mean() if recent_closes.mean() != 0 else 0
                analysis['micro_volatility'] = volatility
                
            # Volume spikes for entry timing
            if len(recent_volumes) >= 10:
                current_volume = recent_volumes[-1]
                avg_volume = recent_volumes[-10:].mean()
                volume_spike = current_volume / avg_volume if avg_volume != 0 else 1.0
                analysis['volume_spike'] = volume_spike
                
        except Exception as e:
            logger.error(f"Error analyzing scalping indicators: {e}", exc_info=True)
            
        return analysis
    
    async def _analyze_breakout_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns for breakout strategy."""
        analysis = {}
        
        try:
            ohlcv_data = market_data.get('ohlcv_data', [])
            if not ohlcv_data:
                return analysis
                
            # Get price arrays
            closes = np.array([candle.close for candle in ohlcv_data[-50:]])
            highs = np.array([candle.high for candle in ohlcv_data[-50:]])
            lows = np.array([candle.low for candle in ohlcv_data[-50:]])
            volumes = np.array([candle.volume for candle in ohlcv_data[-50:]])
            
            # Support and resistance levels
            if len(highs) >= 20:
                resistance_level = np.max(highs[-20:])
                support_level = np.min(lows[-20:])
                current_price = closes[-1]
                
                # Distance to key levels
                resistance_distance = (resistance_level - current_price) / current_price
                support_distance = (current_price - support_level) / current_price
                
                analysis['resistance_distance'] = resistance_distance
                analysis['support_distance'] = support_distance
                
                # Breakout potential
                if resistance_distance < 0.01:  # Within 1% of resistance
                    analysis['breakout_potential'] = 'HIGH_RESISTANCE'
                elif support_distance < 0.01:  # Within 1% of support
                    analysis['breakout_potential'] = 'HIGH_SUPPORT'
                else:
                    analysis['breakout_potential'] = 'NEUTRAL'
                    
            # Volume buildup for breakout
            if len(volumes) >= 20:
                recent_volume = volumes[-5:].mean()
                baseline_volume = volumes[-20:-5].mean()
                volume_buildup = recent_volume / baseline_volume if baseline_volume != 0 else 1.0
                analysis['volume_buildup'] = volume_buildup
                
        except Exception as e:
            logger.error(f"Error analyzing breakout patterns: {e}", exc_info=True)
            
        return analysis
    
    async def _analyze_risk_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk metrics for defensive strategy."""
        analysis = {}
        
        try:
            ohlcv_data = market_data.get('ohlcv_data', [])
            if not ohlcv_data:
                return analysis
                
            # Calculate basic risk metrics
            closes = np.array([candle.close for candle in ohlcv_data[-30:]])
            
            if len(closes) >= 20:
                # Volatility
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns) * np.sqrt(24)  # Annualized for crypto
                analysis['volatility'] = volatility
                
                # Maximum drawdown over period
                peak = np.maximum.accumulate(closes)
                drawdown = (closes - peak) / peak
                max_drawdown = np.min(drawdown)
                analysis['max_drawdown'] = abs(max_drawdown)
                
                # Sharpe ratio estimate
                if len(returns) > 0 and np.std(returns) != 0:
                    mean_return = np.mean(returns)
                    sharpe_estimate = mean_return / np.std(returns) * np.sqrt(24)
                    analysis['sharpe_estimate'] = sharpe_estimate
                else:
                    analysis['sharpe_estimate'] = 0.0
                    
        except Exception as e:
            logger.error(f"Error analyzing risk metrics: {e}", exc_info=True)
            
        return analysis
    
    # Signal generation methods
    async def _generate_momentum_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate momentum trading signals."""
        signals = []
        
        try:
            trend_strength = analysis.get('trend_strength', 0)
            trend_direction = analysis.get('trend_direction', 'NEUTRAL')
            volume_momentum = analysis.get('volume_momentum', 1.0)
            rsi = analysis.get('rsi', 50)
            
            # Strong momentum signal
            if (trend_strength > 0.7 and 
                volume_momentum > 1.5 and 
                ((trend_direction == 'BULLISH' and 30 < rsi < 70) or
                 (trend_direction == 'BEARISH' and 30 < rsi < 70))):
                
                signal = {
                    'type': 'MOMENTUM_ENTRY',
                    'direction': 'LONG' if trend_direction == 'BULLISH' else 'SHORT',
                    'strength': min(trend_strength * volume_momentum, 1.0),
                    'confidence': 0.8,
                    'entry_conditions': MOMENTUM_CONFIG['entry_conditions']
                }
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}", exc_info=True)
            
        return signals
    
    async def _generate_scalping_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scalping trading signals."""
        signals = []
        
        try:
            momentum = analysis.get('short_term_momentum', 0)
            volatility = analysis.get('micro_volatility', 0)
            volume_spike = analysis.get('volume_spike', 1.0)
            
            # Quick scalping opportunities
            if (abs(momentum) > 0.001 and  # Small but clear momentum
                volatility > 0.002 and      # Sufficient volatility
                volume_spike > 1.2):        # Volume confirmation
                
                signal = {
                    'type': 'SCALP_ENTRY',
                    'direction': 'LONG' if momentum > 0 else 'SHORT',
                    'strength': min(abs(momentum) * 100, 1.0),
                    'confidence': 0.6,
                    'entry_conditions': SCALPING_CONFIG['entry_conditions'],
                    'urgency': 'HIGH'  # Scalping signals are time-sensitive
                }
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"Error generating scalping signals: {e}", exc_info=True)
            
        return signals
    
    async def _generate_breakout_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate breakout trading signals."""
        signals = []
        
        try:
            breakout_potential = analysis.get('breakout_potential', 'NEUTRAL')
            volume_buildup = analysis.get('volume_buildup', 1.0)
            resistance_distance = analysis.get('resistance_distance', 0.1)
            support_distance = analysis.get('support_distance', 0.1)
            
            # Breakout signal conditions
            if (breakout_potential != 'NEUTRAL' and 
                volume_buildup > 1.8):
                
                if breakout_potential == 'HIGH_RESISTANCE' and resistance_distance < 0.005:
                    signal = {
                        'type': 'BREAKOUT_LONG',
                        'direction': 'LONG',
                        'strength': min(volume_buildup / 2, 1.0),
                        'confidence': 0.75,
                        'entry_conditions': BREAKOUT_CONFIG['entry_conditions']
                    }
                    signals.append(signal)
                    
                elif breakout_potential == 'HIGH_SUPPORT' and support_distance < 0.005:
                    signal = {
                        'type': 'BREAKOUT_SHORT',
                        'direction': 'SHORT',
                        'strength': min(volume_buildup / 2, 1.0),
                        'confidence': 0.75,
                        'entry_conditions': BREAKOUT_CONFIG['entry_conditions']
                    }
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error generating breakout signals: {e}", exc_info=True)
            
        return signals
    
    async def _generate_defensive_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate defensive trading signals."""
        signals = []
        
        try:
            volatility = analysis.get('volatility', 0.5)
            max_drawdown = analysis.get('max_drawdown', 0.1)
            sharpe_estimate = analysis.get('sharpe_estimate', 0.0)
            
            # Only trade in very favorable risk-adjusted conditions
            if (volatility < 0.3 and        # Low volatility
                max_drawdown < 0.05 and     # Small recent drawdown
                sharpe_estimate > 0.5):     # Positive risk-adjusted returns
                
                signal = {
                    'type': 'DEFENSIVE_ENTRY',
                    'direction': 'LONG' if sharpe_estimate > 0.8 else 'SHORT',
                    'strength': min(sharpe_estimate, 1.0),
                    'confidence': 0.4,
                    'entry_conditions': DEFENSIVE_CONFIG['entry_conditions']
                }
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"Error generating defensive signals: {e}", exc_info=True)
            
        return signals
    
    # Filter methods
    def _apply_momentum_filters(self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply momentum-specific filters to signals."""
        filtered_signals = []
        
        for signal in signals:
            # Risk-reward filter
            if signal.get('strength', 0) >= MOMENTUM_CONFIG['entry_conditions']['trend_strength_min']:
                # Volume confirmation filter
                if market_data.get('volume_surge', False) or signal.get('confidence', 0) > 0.7:
                    filtered_signals.append(signal)
                    
        return filtered_signals
    
    def _apply_scalping_filters(self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply scalping-specific filters to signals."""
        filtered_signals = []
        
        for signal in signals:
            # Quick profit potential filter
            if signal.get('urgency') == 'HIGH' and signal.get('strength', 0) > 0.3:
                # Spread filter (would check bid-ask spread in real implementation)
                spread_ok = True  # Placeholder
                if spread_ok:
                    filtered_signals.append(signal)
                    
        return filtered_signals
    
    def _apply_breakout_filters(self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply breakout-specific filters to signals."""
        filtered_signals = []
        
        for signal in signals:
            # Breakout strength filter
            if signal.get('strength', 0) >= BREAKOUT_CONFIG['entry_conditions']['breakout_strength_min']:
                # Volume surge confirmation
                if signal.get('confidence', 0) > 0.7:
                    filtered_signals.append(signal)
                    
        return filtered_signals
    
    def _apply_defensive_filters(self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply defensive-specific filters to signals."""
        filtered_signals = []
        
        for signal in signals:
            # Conservative filters - only very high confidence signals
            if signal.get('confidence', 0) >= DEFENSIVE_CONFIG['entry_conditions']['defensive_score_min']:
                # Additional safety check
                if signal.get('strength', 0) > 0.5:
                    filtered_signals.append(signal)
                    
        return filtered_signals
    
    # Execution methods
    async def _execute_with_momentum_risk_mgmt(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute signals with momentum risk management."""
        execution_result = {
            'signals_processed': len(signals),
            'orders_placed': 0,
            'risk_adjustments': []
        }
        
        for signal in signals:
            # Apply momentum-specific risk management
            risk_mgmt = MOMENTUM_CONFIG['risk_management']
            
            # Position sizing based on momentum strength
            base_size = MOMENTUM_CONFIG['position_sizing']['base_size_pct']
            size_multiplier = MOMENTUM_CONFIG['position_sizing']['size_multiplier']
            adjusted_size = base_size * (signal.get('strength', 0.5) * size_multiplier)
            
            # Cap at maximum size
            max_size = MOMENTUM_CONFIG['position_sizing']['max_size_pct']
            final_size = min(adjusted_size, max_size)
            
            execution_result['orders_placed'] += 1
            execution_result['risk_adjustments'].append({
                'signal_type': signal.get('type'),
                'original_size': base_size,
                'adjusted_size': final_size,
                'stop_loss': risk_mgmt['stop_loss_pct'],
                'take_profit': risk_mgmt['take_profit_pct']
            })
            
        return execution_result
    
    async def _execute_with_scalping_risk_mgmt(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute signals with scalping risk management."""
        execution_result = {
            'signals_processed': len(signals),
            'orders_placed': 0,
            'risk_adjustments': []
        }
        
        for signal in signals:
            # Apply scalping-specific risk management
            risk_mgmt = SCALPING_CONFIG['risk_management']
            
            # Smaller position sizes for scalping
            base_size = SCALPING_CONFIG['position_sizing']['base_size_pct']
            size_multiplier = SCALPING_CONFIG['position_sizing']['size_multiplier']
            adjusted_size = base_size * size_multiplier
            
            execution_result['orders_placed'] += 1
            execution_result['risk_adjustments'].append({
                'signal_type': signal.get('type'),
                'adjusted_size': adjusted_size,
                'stop_loss': risk_mgmt['stop_loss_pct'],
                'take_profit': risk_mgmt['take_profit_pct'],
                'max_holding_time': risk_mgmt['max_holding_time']
            })
            
        return execution_result
    
    async def _execute_with_breakout_risk_mgmt(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute signals with breakout risk management."""
        execution_result = {
            'signals_processed': len(signals),
            'orders_placed': 0,
            'risk_adjustments': []
        }
        
        for signal in signals:
            # Apply breakout-specific risk management
            risk_mgmt = BREAKOUT_CONFIG['risk_management']
            
            # Larger position sizes for breakouts
            base_size = BREAKOUT_CONFIG['position_sizing']['base_size_pct']
            size_multiplier = BREAKOUT_CONFIG['position_sizing']['size_multiplier']
            adjusted_size = base_size * (signal.get('strength', 0.5) * size_multiplier)
            
            # Cap at maximum size
            max_size = BREAKOUT_CONFIG['position_sizing']['max_size_pct']
            final_size = min(adjusted_size, max_size)
            
            execution_result['orders_placed'] += 1
            execution_result['risk_adjustments'].append({
                'signal_type': signal.get('type'),
                'adjusted_size': final_size,
                'stop_loss': risk_mgmt['stop_loss_pct'],
                'take_profit': risk_mgmt['take_profit_pct'],
                'trailing_stop': risk_mgmt['trailing_stop']
            })
            
        return execution_result
    
    async def _execute_with_defensive_risk_mgmt(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute signals with defensive risk management."""
        execution_result = {
            'signals_processed': len(signals),
            'orders_placed': 0,
            'risk_adjustments': []
        }
        
        for signal in signals:
            # Apply defensive risk management
            risk_mgmt = DEFENSIVE_CONFIG['risk_management']
            
            # Very small position sizes for defensive strategy
            base_size = DEFENSIVE_CONFIG['position_sizing']['base_size_pct']
            size_multiplier = DEFENSIVE_CONFIG['position_sizing']['size_multiplier']
            adjusted_size = base_size * size_multiplier
            
            execution_result['orders_placed'] += 1
            execution_result['risk_adjustments'].append({
                'signal_type': signal.get('type'),
                'adjusted_size': adjusted_size,
                'stop_loss': risk_mgmt['stop_loss_pct'],
                'take_profit': risk_mgmt['take_profit_pct'],
                'max_drawdown': risk_mgmt['max_drawdown_pct']
            })
            
        return execution_result


class StrategyPerformanceTracker:
    """Tracks performance metrics for different trading strategies."""
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.strategy_metrics: Dict[TradingStrategy, Dict[str, Any]] = {}
        self.transition_metrics: Dict[str, Any] = {}
        
        logger.info("Initialized StrategyPerformanceTracker")
        
    def track_strategy_performance(
        self,
        strategy: TradingStrategy,
        trades: List[Dict[str, Any]],
        market_conditions: Dict[str, Any]
    ) -> None:
        """
        Track performance for a specific strategy.
        
        Args:
            strategy: The trading strategy
            trades: List of trades executed
            market_conditions: Market conditions during execution
        """
        try:
            if strategy not in self.strategy_metrics:
                self.strategy_metrics[strategy] = {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'total_pnl': 0.0,
                    'max_drawdown': 0.0,
                    'avg_holding_time': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'market_regimes': {},
                    'last_updated': datetime.utcnow()
                }
            
            metrics = self.strategy_metrics[strategy]
            
            # Update trade metrics
            for trade in trades:
                metrics['total_trades'] += 1
                pnl = trade.get('pnl', 0.0)
                metrics['total_pnl'] += pnl
                
                if pnl > 0:
                    metrics['profitable_trades'] += 1
                    
                # Track holding time
                holding_time = trade.get('holding_time_seconds', 0)
                if metrics['total_trades'] == 1:
                    metrics['avg_holding_time'] = holding_time
                else:
                    # Update running average
                    metrics['avg_holding_time'] = (
                        (metrics['avg_holding_time'] * (metrics['total_trades'] - 1) + holding_time) 
                        / metrics['total_trades']
                    )
            
            # Calculate derived metrics
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades']
                
                # Simple profit factor calculation
                total_gains = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
                total_losses = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
                
                if total_losses > 0:
                    metrics['profit_factor'] = total_gains / total_losses
                else:
                    metrics['profit_factor'] = float('inf') if total_gains > 0 else 0.0
                    
            # Track market regime performance
            regime = market_conditions.get('regime', 'UNKNOWN')
            if regime not in metrics['market_regimes']:
                metrics['market_regimes'][regime] = {
                    'trades': 0,
                    'pnl': 0.0,
                    'win_rate': 0.0
                }
            
            regime_metrics = metrics['market_regimes'][regime]
            for trade in trades:
                regime_metrics['trades'] += 1
                regime_metrics['pnl'] += trade.get('pnl', 0.0)
            
            if regime_metrics['trades'] > 0:
                winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
                regime_metrics['win_rate'] = winning_trades / len(trades)
            
            metrics['last_updated'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error tracking strategy performance: {e}", exc_info=True)
    
    def get_strategy_effectiveness(self, strategy: TradingStrategy) -> float:
        """
        Calculate strategy effectiveness score (0.0 to 1.0).
        
        Args:
            strategy: The trading strategy to evaluate
            
        Returns:
            Effectiveness score between 0.0 and 1.0
        """
        try:
            metrics = self.strategy_metrics.get(strategy)
            if not metrics or metrics['total_trades'] == 0:
                return 0.5  # Default neutral effectiveness
            
            # Weighted score based on multiple factors
            win_rate_score = metrics.get('win_rate', 0.0)
            profit_factor_score = min(metrics.get('profit_factor', 0.0) / 2.0, 1.0)
            sharpe_score = max(0, min(metrics.get('sharpe_ratio', 0.0) / 2.0, 1.0))
            
            # Additional factor: consistent performance
            consistency_score = 1.0 - min(metrics.get('max_drawdown', 0.0) / 0.1, 1.0)
            
            effectiveness = (
                win_rate_score * 0.35 +
                profit_factor_score * 0.35 +
                sharpe_score * 0.15 +
                consistency_score * 0.15
            )
            
            return max(0.0, min(effectiveness, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating strategy effectiveness: {e}", exc_info=True)
            return 0.5
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            summary = {
                'strategies': {},
                'best_strategy': None,
                'worst_strategy': None,
                'overall_performance': {
                    'total_trades': 0,
                    'total_pnl': 0.0,
                    'avg_effectiveness': 0.0
                }
            }
            
            effectiveness_scores = {}
            
            for strategy, metrics in self.strategy_metrics.items():
                effectiveness = self.get_strategy_effectiveness(strategy)
                effectiveness_scores[strategy] = effectiveness
                
                summary['strategies'][strategy.value] = {
                    'metrics': metrics,
                    'effectiveness': effectiveness
                }
                
                # Update overall performance
                summary['overall_performance']['total_trades'] += metrics['total_trades']
                summary['overall_performance']['total_pnl'] += metrics['total_pnl']
            
            # Find best and worst strategies
            if effectiveness_scores:
                best_strategy = max(effectiveness_scores, key=effectiveness_scores.get)
                worst_strategy = min(effectiveness_scores, key=effectiveness_scores.get)
                
                summary['best_strategy'] = {
                    'name': best_strategy.value,
                    'effectiveness': effectiveness_scores[best_strategy]
                }
                summary['worst_strategy'] = {
                    'name': worst_strategy.value,
                    'effectiveness': effectiveness_scores[worst_strategy]
                }
                
                # Calculate average effectiveness
                summary['overall_performance']['avg_effectiveness'] = (
                    sum(effectiveness_scores.values()) / len(effectiveness_scores)
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}", exc_info=True)
            return {'error': str(e)}


class StrategyContextProvider:
    """Provides context for LLM integration with strategy management."""
    
    def __init__(self, strategy_manager):
        """
        Initialize the context provider.
        
        Args:
            strategy_manager: The adaptive strategy manager instance
        """
        self.strategy_manager = strategy_manager
        
    def get_llm_context(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive context for LLM decision making.
        
        Args:
            market_analysis: Current market analysis results
            
        Returns:
            Context dictionary for LLM consumption
        """
        try:
            current_strategy = self.strategy_manager.current_strategy
            strategy_performance = self.strategy_manager.get_strategy_performance()
            
            context = {
                'current_strategy': {
                    'name': current_strategy.name.value if current_strategy else 'NONE',
                    'state': current_strategy.state.value if current_strategy else 'INACTIVE',
                    'duration': self._get_strategy_duration(),
                    'performance': strategy_performance.get(current_strategy.name.value, {}) if current_strategy else {}
                },
                'strategy_recommendations': {
                    'primary': self._get_primary_recommendation(market_analysis),
                    'alternative': self._get_alternative_recommendation(market_analysis),
                    'confidence': self._calculate_recommendation_confidence(market_analysis)
                },
                'transition_status': {
                    'ready_for_transition': self._can_transition(),
                    'transition_risks': self._assess_transition_risks(market_analysis),
                    'optimal_timing': self._calculate_optimal_transition_timing()
                },
                'performance_context': {
                    'recent_performance': self._get_recent_performance(),
                    'strategy_comparison': self._compare_strategy_performance(),
                    'market_adaptation': self._assess_market_adaptation()
                }
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error generating LLM context: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _get_strategy_duration(self) -> int:
        """Get current strategy duration in seconds."""
        try:
            if hasattr(self.strategy_manager, 'strategy_start_time'):
                return int((datetime.utcnow() - self.strategy_manager.strategy_start_time).total_seconds())
            return 0
        except:
            return 0
    
    def _get_primary_recommendation(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get primary strategy recommendation."""
        try:
            # Use the strategy selector to get recommendation
            selector = StrategySelector()
            recommended_strategy, confidence = selector.select_strategy(market_analysis)
            
            return {
                'strategy': recommended_strategy.value,
                'confidence': confidence,
                'reasoning': f"Market regime analysis suggests {recommended_strategy.value} strategy"
            }
        except:
            return {'strategy': 'HOLD', 'confidence': 0.5, 'reasoning': 'Default recommendation'}
    
    def _get_alternative_recommendation(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get alternative strategy recommendation."""
        try:
            # Simple alternative logic
            primary = self._get_primary_recommendation(market_analysis)
            primary_strategy = primary['strategy']
            
            alternatives = {
                'MOMENTUM': 'SCALPING',
                'SCALPING': 'MOMENTUM', 
                'BREAKOUT': 'MOMENTUM',
                'DEFENSIVE': 'HOLD',
                'HOLD': 'DEFENSIVE'
            }
            
            alternative_strategy = alternatives.get(primary_strategy, 'HOLD')
            
            return {
                'strategy': alternative_strategy,
                'confidence': primary['confidence'] * 0.7,
                'reasoning': f"Alternative to {primary_strategy}"
            }
        except:
            return {'strategy': 'HOLD', 'confidence': 0.3, 'reasoning': 'Default alternative'}
    
    def _calculate_recommendation_confidence(self, market_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in recommendations."""
        try:
            # Base confidence from regime analysis
            regime_confidence = market_analysis.get('regime', {}).get('confidence', 0.5)
            
            # Adjust based on market clarity
            correlation_strength = market_analysis.get('correlation', {}).get('correlation_strength', 'WEAK')
            
            strength_multipliers = {
                'VERY_STRONG': 1.2,
                'STRONG': 1.1,
                'MODERATE': 1.0,
                'WEAK': 0.9,
                'VERY_WEAK': 0.8
            }
            
            multiplier = strength_multipliers.get(correlation_strength, 1.0)
            adjusted_confidence = regime_confidence * multiplier
            
            return max(0.0, min(adjusted_confidence, 1.0))
            
        except:
            return 0.5
    
    def _can_transition(self) -> bool:
        """Check if strategy can transition."""
        try:
            duration = self._get_strategy_duration()
            min_duration = 300  # 5 minutes minimum
            return duration >= min_duration
        except:
            return True
    
    def _assess_transition_risks(self, market_analysis: Dict[str, Any]) -> List[str]:
        """Assess risks of strategy transition."""
        risks = []
        
        try:
            # Market volatility risk
            volatility = market_analysis.get('technical', {}).get('volatility', 0.5)
            if volatility > 0.7:
                risks.append("High market volatility may affect transition")
            
            # Regime uncertainty risk
            regime_confidence = market_analysis.get('regime', {}).get('confidence', 0.5)
            if regime_confidence < 0.6:
                risks.append("Low regime confidence increases transition risk")
            
            # Position risk
            if hasattr(self.strategy_manager, 'current_positions'):
                if len(self.strategy_manager.current_positions) > 0:
                    risks.append("Open positions may be affected by strategy change")
                    
        except Exception as e:
            risks.append(f"Risk assessment error: {str(e)}")
            
        return risks
    
    def _calculate_optimal_transition_timing(self) -> str:
        """Calculate optimal timing for strategy transition."""
        try:
            current_hour = datetime.utcnow().hour
            
            # Market hours considerations
            if 9 <= current_hour <= 16:  # Traditional market hours
                return "IMMEDIATE"
            elif 0 <= current_hour <= 6:  # Low activity hours
                return "GRADUAL"
            else:
                return "NORMAL"
                
        except:
            return "NORMAL"
    
    def _get_recent_performance(self) -> Dict[str, Any]:
        """Get recent performance metrics."""
        try:
            if hasattr(self.strategy_manager, 'performance_tracker'):
                return self.strategy_manager.performance_tracker.get_performance_summary()
            return {'no_data': True}
        except:
            return {'error': 'Unable to retrieve performance data'}
    
    def _compare_strategy_performance(self) -> Dict[str, Any]:
        """Compare performance across strategies."""
        try:
            comparison = {}
            if hasattr(self.strategy_manager, 'performance_tracker'):
                tracker = self.strategy_manager.performance_tracker
                
                for strategy in TradingStrategy:
                    if strategy in tracker.strategy_metrics:
                        effectiveness = tracker.get_strategy_effectiveness(strategy)
                        comparison[strategy.value] = {
                            'effectiveness': effectiveness,
                            'total_trades': tracker.strategy_metrics[strategy]['total_trades'],
                            'win_rate': tracker.strategy_metrics[strategy]['win_rate']
                        }
            
            return comparison
        except:
            return {'error': 'Unable to compare strategies'}
    
    def _assess_market_adaptation(self) -> Dict[str, Any]:
        """Assess how well current strategy adapts to market."""
        try:
            adaptation_score = 0.5  # Default neutral
            
            # Would analyze strategy performance vs market conditions
            # This is a simplified implementation
            
            return {
                'adaptation_score': adaptation_score,
                'recommendation': 'Continue monitoring adaptation'
            }
        except:
            return {'adaptation_score': 0.5, 'recommendation': 'Unable to assess adaptation'}


class AdaptiveStrategyManager:
    """
    Main adaptive strategy manager that coordinates all strategy management functions.
    
    This class serves as the central hub for strategy selection, execution, transition
    management, and performance tracking based on real-time market regime analysis.
    """
    
    def __init__(self, market_context_analyzer: Optional[MarketContextAnalyzer] = None):
        """
        Initialize the adaptive strategy manager.
        
        Args:
            market_context_analyzer: Market context analyzer for regime detection
        """
        self.market_analyzer = market_context_analyzer or MarketContextAnalyzer()
        self.strategy_selector = StrategySelector()
        self.transition_manager = TransitionManager()
        self.performance_tracker = StrategyPerformanceTracker()
        self.context_provider = StrategyContextProvider(self)
        
        # Current strategy state
        self.current_strategy: Optional[StrategyExecutor] = None
        self.current_strategy_name: TradingStrategy = TradingStrategy.HOLD
        self.strategy_start_time: datetime = datetime.utcnow()
        self.last_regime_analysis: Dict[str, Any] = {}
        
        # Strategy configurations
        self.strategy_configs = {
            TradingStrategy.MOMENTUM: StrategyConfig(
                name=TradingStrategy.MOMENTUM,
                timeframe="1m",
                indicator_set=MOMENTUM_CONFIG['indicators'],
                position_sizing=MOMENTUM_CONFIG['position_sizing'],
                risk_parameters=MOMENTUM_CONFIG['risk_management'],
                execution_parameters=MOMENTUM_CONFIG['execution'],
                performance_thresholds={'min_win_rate': 0.6, 'min_profit_factor': 1.5},
                momentum_config=MOMENTUM_CONFIG
            ),
            TradingStrategy.SCALPING: StrategyConfig(
                name=TradingStrategy.SCALPING,
                timeframe="15s",
                indicator_set=SCALPING_CONFIG['indicators'],
                position_sizing=SCALPING_CONFIG['position_sizing'],
                risk_parameters=SCALPING_CONFIG['risk_management'],
                execution_parameters=SCALPING_CONFIG['execution'],
                performance_thresholds={'min_win_rate': 0.65, 'min_profit_factor': 1.3},
                scalping_config=SCALPING_CONFIG
            ),
            TradingStrategy.BREAKOUT: StrategyConfig(
                name=TradingStrategy.BREAKOUT,
                timeframe="5m",
                indicator_set=BREAKOUT_CONFIG['indicators'],
                position_sizing=BREAKOUT_CONFIG['position_sizing'],
                risk_parameters=BREAKOUT_CONFIG['risk_management'],
                execution_parameters=BREAKOUT_CONFIG['execution'],
                performance_thresholds={'min_win_rate': 0.55, 'min_profit_factor': 2.0},
                breakout_config=BREAKOUT_CONFIG
            ),
            TradingStrategy.DEFENSIVE: StrategyConfig(
                name=TradingStrategy.DEFENSIVE,
                timeframe="5m",
                indicator_set=DEFENSIVE_CONFIG['indicators'],
                position_sizing=DEFENSIVE_CONFIG['position_sizing'],
                risk_parameters=DEFENSIVE_CONFIG['risk_management'],
                execution_parameters=DEFENSIVE_CONFIG['execution'],
                performance_thresholds={'min_win_rate': 0.7, 'min_profit_factor': 1.2},
            )
        }
        
        logger.info("Initialized AdaptiveStrategyManager")
    
    async def analyze_and_execute(self, market_state: MarketState) -> Dict[str, Any]:
        """
        Main execution method - analyzes market and executes appropriate strategy.
        
        Args:
            market_state: Current market state with OHLCV data and indicators
            
        Returns:
            Comprehensive strategy execution results
        """
        try:
            start_time = time.time()
            
            # Step 1: Perform market regime analysis
            regime_analysis = await self._perform_regime_analysis(market_state)
            
            # Step 2: Select optimal strategy
            recommended_strategy, confidence = self.strategy_selector.select_strategy(regime_analysis)
            
            # Step 3: Handle strategy transitions if needed
            transition_executed = False
            if recommended_strategy != self.current_strategy_name:
                if confidence > 0.65:  # Only transition with sufficient confidence
                    transition_executed = await self._execute_strategy_transition(
                        recommended_strategy, regime_analysis
                    )
            
            # Step 4: Execute current strategy
            execution_result = await self._execute_current_strategy(market_state)
            
            # Step 5: Track performance
            await self._track_performance(execution_result, regime_analysis)
            
            # Step 6: Generate comprehensive result
            result = {
                'timestamp': datetime.utcnow().isoformat(),
                'execution_time_ms': int((time.time() - start_time) * 1000),
                'regime_analysis': regime_analysis,
                'active_strategy': {
                    'name': self.current_strategy_name.value,
                    'state': self.current_strategy.state.value if self.current_strategy else 'INACTIVE',
                    'config': self.strategy_configs.get(self.current_strategy_name),
                    'duration_seconds': int((datetime.utcnow() - self.strategy_start_time).total_seconds())
                },
                'strategy_decision': {
                    'recommended': recommended_strategy.value,
                    'confidence': confidence,
                    'reasoning': f"Regime analysis suggests {recommended_strategy.value} with {confidence:.2%} confidence",
                    'transition_executed': transition_executed
                },
                'execution_result': execution_result,
                'performance_summary': self.performance_tracker.get_performance_summary(),
                'llm_context': self.context_provider.get_llm_context(regime_analysis)
            }
            
            logger.info(f"Strategy analysis complete: {recommended_strategy.value} "
                       f"(confidence: {confidence:.2%}, execution_time: {result['execution_time_ms']}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in adaptive strategy analysis: {e}", exc_info=True)
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'active_strategy': {'name': 'ERROR'},
                'execution_result': {'error': 'Strategy execution failed'}
            }
    
    async def _perform_regime_analysis(self, market_state: MarketState) -> Dict[str, Any]:
        """Perform comprehensive market regime analysis."""
        try:
            # Convert market state to analysis format
            crypto_data = {
                'ohlcv': market_state.ohlcv_data,
                'indicators': market_state.indicators,
                'current_price': market_state.current_price
            }
            
            # For now, create mock NASDAQ data (in real implementation, would fetch actual data)
            nasdaq_data = {
                'ohlcv': market_state.ohlcv_data,  # Placeholder
                'indicators': market_state.indicators  # Placeholder
            }
            
            # Mock sentiment data
            sentiment_data = {
                'text': 'market analysis',
                'news_headlines': [],
                'vix_level': 20.0,
                'volatility_score': 0.5
            }
            
            # Mock news data
            news_data = [
                {'title': 'Market Update', 'content': 'Markets showing mixed signals'}
            ]
            
            # Perform analyses
            correlation_analysis = await self.market_analyzer.analyze_crypto_nasdaq_correlation(
                crypto_data, nasdaq_data
            )
            
            regime_analysis = await self.market_analyzer.detect_market_regime(sentiment_data)
            
            risk_sentiment = await self.market_analyzer.assess_risk_sentiment(news_data)
            
            # Combine results
            combined_analysis = {
                'correlation': {
                    'correlation_coefficient': correlation_analysis.correlation_coefficient,
                    'correlation_strength': correlation_analysis.correlation_strength.value,
                    'reliability_score': correlation_analysis.reliability_score,
                    'is_significant': correlation_analysis.is_significant
                },
                'regime': {
                    'regime_type': regime_analysis.regime_type,
                    'confidence': regime_analysis.confidence,
                    'key_drivers': regime_analysis.key_drivers,
                    'fed_policy_stance': regime_analysis.fed_policy_stance,
                    'market_volatility_regime': regime_analysis.market_volatility_regime,
                    'liquidity_conditions': regime_analysis.liquidity_conditions
                },
                'sentiment': {
                    'fear_greed_index': risk_sentiment.fear_greed_index,
                    'sentiment_level': risk_sentiment.sentiment_level.value,
                    'market_stress_indicator': risk_sentiment.market_stress_indicator,
                    'volatility_expectation': risk_sentiment.volatility_expectation
                },
                'technical': {
                    'current_rsi': market_state.indicators.rsi or 50.0,
                    'trend_strength': 0.5,  # Placeholder calculation
                    'volatility_normalized': 0.5,  # Placeholder calculation
                    'breakout_score': 0.5  # Placeholder calculation
                }
            }
            
            self.last_regime_analysis = combined_analysis
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error performing regime analysis: {e}", exc_info=True)
            return {
                'error': str(e),
                'regime': {'regime_type': MarketRegimeType.UNKNOWN, 'confidence': 0.0}
            }
    
    async def _execute_strategy_transition(
        self, 
        new_strategy: TradingStrategy, 
        regime_analysis: Dict[str, Any]
    ) -> bool:
        """Execute transition to new strategy."""
        try:
            logger.info(f"Initiating strategy transition from {self.current_strategy_name.value} to {new_strategy.value}")
            
            # Execute transition
            transition_success = await self.transition_manager.execute_transition(
                self.current_strategy_name,
                new_strategy,
                regime_analysis
            )
            
            if transition_success:
                # Update current strategy
                old_strategy = self.current_strategy_name
                self.current_strategy_name = new_strategy
                self.strategy_start_time = datetime.utcnow()
                
                # Initialize new strategy executor
                if new_strategy != TradingStrategy.HOLD:
                    strategy_config = self.strategy_configs[new_strategy]
                    self.current_strategy = StrategyExecutor(strategy_config)
                else:
                    self.current_strategy = None
                
                logger.info(f"Successfully transitioned from {old_strategy.value} to {new_strategy.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing strategy transition: {e}", exc_info=True)
            return False
    
    async def _execute_current_strategy(self, market_state: MarketState) -> Dict[str, Any]:
        """Execute the current active strategy."""
        try:
            if self.current_strategy is None or self.current_strategy_name == TradingStrategy.HOLD:
                return {
                    'strategy': TradingStrategy.HOLD.value,
                    'action': 'HOLD',
                    'signals_generated': 0,
                    'trades_executed': 0,
                    'reason': 'No active strategy or HOLD strategy'
                }
            
            # Prepare market data for strategy execution
            market_data = {
                'ohlcv_data': market_state.ohlcv_data,
                'current_price': market_state.current_price,
                'indicators': market_state.indicators,
                'timestamp': market_state.timestamp,
                'volume_surge': False  # Placeholder
            }
            
            # Execute strategy
            strategy_result = await self.current_strategy.execute_strategy(market_data)
            
            return {
                'strategy': strategy_result.get('strategy', self.current_strategy_name).value,
                'signals_generated': len(strategy_result.get('signals', [])),
                'trades_executed': strategy_result.get('execution', {}).get('orders_placed', 0),
                'positions_modified': strategy_result.get('execution', {}).get('risk_adjustments', []),
                'analysis_summary': strategy_result.get('analysis', {}),
                'execution_details': strategy_result.get('execution', {})
            }
            
        except Exception as e:
            logger.error(f"Error executing current strategy: {e}", exc_info=True)
            return {
                'strategy': self.current_strategy_name.value,
                'error': str(e),
                'signals_generated': 0,
                'trades_executed': 0
            }
    
    async def _track_performance(
        self, 
        execution_result: Dict[str, Any], 
        regime_analysis: Dict[str, Any]
    ) -> None:
        """Track performance of strategy execution."""
        try:
            # Create mock trades from execution result for tracking
            mock_trades = []
            
            trades_executed = execution_result.get('trades_executed', 0)
            if trades_executed > 0:
                # Create mock trade data for performance tracking
                for i in range(trades_executed):
                    mock_trade = {
                        'pnl': np.random.normal(0.001, 0.01),  # Mock PnL
                        'holding_time_seconds': np.random.randint(60, 1800),  # Mock holding time
                        'strategy': self.current_strategy_name.value
                    }
                    mock_trades.append(mock_trade)
            
            # Track performance
            if mock_trades:
                self.performance_tracker.track_strategy_performance(
                    self.current_strategy_name,
                    mock_trades,
                    regime_analysis
                )
                
        except Exception as e:
            logger.error(f"Error tracking performance: {e}", exc_info=True)
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get current strategy performance metrics."""
        return self.performance_tracker.get_performance_summary()
    
    def get_current_strategy_info(self) -> Dict[str, Any]:
        """Get information about current active strategy."""
        return {
            'name': self.current_strategy_name.value,
            'state': self.current_strategy.state.value if self.current_strategy else 'INACTIVE',
            'duration_seconds': int((datetime.utcnow() - self.strategy_start_time).total_seconds()),
            'config': self.strategy_configs.get(self.current_strategy_name)
        }
    
    def force_strategy_change(self, new_strategy: TradingStrategy) -> bool:
        """Force immediate strategy change (for testing/manual control)."""
        try:
            logger.info(f"Forcing strategy change to {new_strategy.value}")
            
            self.current_strategy_name = new_strategy
            self.strategy_start_time = datetime.utcnow()
            
            if new_strategy != TradingStrategy.HOLD:
                strategy_config = self.strategy_configs[new_strategy]
                self.current_strategy = StrategyExecutor(strategy_config)
            else:
                self.current_strategy = None
                
            return True
            
        except Exception as e:
            logger.error(f"Error forcing strategy change: {e}", exc_info=True)
            return False