"""
High-frequency scalping strategy optimized for low-volume and ranging market conditions.

This module implements a sophisticated scalping strategy that captures small, frequent profits
during low-volume periods, consolidation phases, and ranging markets. It focuses on quick
entries/exits with tight risk management.

Key Features:
- Microstructure analysis for support/resistance scalping
- Fast momentum indicators for mean reversion
- VWAP-based bounce signals
- High-frequency execution with sub-10ms signal generation
- Thread-safe operations for continuous trading
- Adaptive position sizing based on market conditions
- Quick exit mechanisms for risk management

Signal Types:
- Mean reversion: RSI/Williams extremes near support/resistance
- Micro breakouts: Small range breakouts with volume confirmation
- VWAP bounces: Price bounces off VWAP levels
- Support/resistance: Touch/bounce patterns at key levels
- Momentum spikes: Short-term momentum bursts
- Volume anomalies: Unusual volume patterns

Architecture:
- ScalpingStrategy: Main strategy orchestrator
- ScalpingSignalGenerator: Signal identification and scoring
- ScalpingRiskManager: Position sizing and risk controls
- ScalpingExecutor: High-frequency trade execution
- ScalpingPerformanceTracker: Metrics and optimization
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..indicators import (
    FastEMA, ScalpingEMASignals,
    FastRSI, FastMACD, WilliamsPercentR, ScalpingMomentumSignals,
    ScalpingVWAP, OnBalanceVolume, VolumeMovingAverage, ScalpingVolumeSignals,
    calculate_atr, calculate_support_resistance
)
from ..types import TradeAction, Position

logger = logging.getLogger(__name__)


class ScalpingSignalType(Enum):
    """Types of scalping signals."""
    MEAN_REVERSION = "mean_reversion"
    MICRO_BREAKOUT = "micro_breakout"
    VWAP_BOUNCE = "vwap_bounce"
    SUPPORT_RESISTANCE = "support_resistance"
    MOMENTUM_SPIKE = "momentum_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    NONE = "none"


class ScalpingTiming(Enum):
    """Timing urgency for scalping signals."""
    IMMEDIATE = "immediate"      # Execute within 1-2 seconds
    QUICK = "quick"             # Execute within 5 seconds
    NORMAL = "normal"           # Execute within 15 seconds
    WAIT = "wait"               # Wait for better setup


class ScalpingState(Enum):
    """Current scalping operation state."""
    SCANNING = "scanning"
    ENTERING = "entering"
    MANAGING = "managing"
    EXITING = "exiting"
    COOLDOWN = "cooldown"


@dataclass
class ScalpingConfig:
    """Configuration for scalping strategy operations."""
    
    # Timeframe settings
    primary_timeframe: str = "15s"
    tick_analysis: bool = True
    min_data_points: int = 50
    
    # Quick indicators
    ema_ultra_fast: int = 3
    ema_fast: int = 5
    ema_medium: int = 8
    rsi_period: int = 7
    williams_r_period: int = 7
    vwap_periods: List[int] = field(default_factory=lambda: [20, 50])
    
    # Scalping thresholds
    min_profit_target_pct: float = 0.03  # 3 basis points minimum
    max_profit_target_pct: float = 0.08  # 8 basis points maximum
    stop_loss_pct: float = 0.02         # 2 basis points stop loss
    breakeven_threshold_pct: float = 0.01  # 1 basis point for breakeven
    
    # Volume requirements
    min_volume_relative: float = 0.8    # 80% of average volume
    max_spread_bps: float = 2.0         # Maximum 2 basis points spread
    require_volume_confirmation: bool = False  # Less strict for scalping
    
    # Entry criteria
    min_signal_strength: float = 0.6    # Lower threshold for quick entries
    max_holding_time: int = 300         # 5 minutes maximum
    quick_exit_threshold: int = 60      # 1 minute for quick exits
    immediate_exit_time: int = 15       # 15 seconds for immediate exits
    
    # Position sizing
    base_position_pct: float = 0.5      # Base 0.5% of account
    max_position_pct: float = 1.5       # Maximum 1.5% of account
    frequency_adjusted: bool = True     # Reduce size for high frequency
    
    # Risk management
    trailing_stop: bool = False         # Too fast for trailing stops
    immediate_exit_conditions: bool = True
    max_consecutive_losses: int = 3
    cooldown_after_loss: int = 30       # 30 seconds cooldown
    max_daily_trades: int = 200         # Maximum trades per day
    
    # Market microstructure
    support_resistance_lookback: int = 20
    recent_extremes_period: int = 10
    range_detection_period: int = 50
    breakout_volume_multiplier: float = 1.5
    
    # Performance thresholds
    max_analysis_time_ms: float = 10.0  # Maximum 10ms analysis time
    signal_timeout_seconds: int = 60    # Signal expires after 1 minute


@dataclass
class ScalpingSignal:
    """Individual scalping signal with execution metadata."""
    
    signal_type: ScalpingSignalType
    direction: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    timing: ScalpingTiming
    target_profit_pct: float
    stop_loss_pct: float
    max_holding_time: int
    
    # Market analysis context
    current_price: float
    entry_reasons: List[str]
    risk_factors: List[str]
    market_conditions: Dict[str, Any]
    
    # Execution metadata
    timestamp: float
    expires_at: float
    priority_score: float
    
    # Analysis results
    microstructure: Dict[str, Any] = field(default_factory=dict)
    momentum: Dict[str, Any] = field(default_factory=dict)
    vwap_analysis: Dict[str, Any] = field(default_factory=dict)


class ScalpingMicrostructureAnalyzer:
    """Analyzes market microstructure for scalping opportunities."""
    
    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.support_levels = deque(maxlen=10)
        self.resistance_levels = deque(maxlen=10)
        self.recent_ranges = deque(maxlen=5)
        
    def analyze_microstructure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive microstructure analysis.
        
        Args:
            data: OHLCV DataFrame with recent market data
            
        Returns:
            Dictionary containing microstructure analysis results
        """
        if len(data) < self.config.min_data_points:
            return self._get_default_microstructure()
        
        try:
            current_price = float(data['close'].iloc[-1])
            recent_data = data.tail(self.config.range_detection_period)
            
            # Calculate support and resistance levels
            support_levels = self._identify_support_levels(recent_data)
            resistance_levels = self._identify_resistance_levels(recent_data)
            
            # Analyze current price position
            price_position = self._analyze_price_position(
                current_price, support_levels, resistance_levels
            )
            
            # Detect current trading range
            range_info = self._detect_current_range(recent_data)
            
            # Assess breakout potential
            breakout_potential = self._assess_breakout_potential(
                current_price, support_levels, resistance_levels, 
                recent_data['volume'], recent_data
            )
            
            # Calculate distances to key levels
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else current_price
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else current_price
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'price_position': price_position,
                'range_info': range_info,
                'breakout_potential': breakout_potential,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance_pct': abs(current_price - nearest_support) / current_price * 100,
                'resistance_distance_pct': abs(current_price - nearest_resistance) / current_price * 100,
                'in_range': range_info.get('in_range', False),
                'range_position': range_info.get('position', 'unknown'),
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Microstructure analysis error: {e}")
            return self._get_default_microstructure()
    
    def _identify_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Identify support levels using swing lows and volume."""
        try:
            lows = data['low'].values
            volumes = data['volume'].values
            
            # Find local minima (swing lows)
            support_candidates = []
            
            for i in range(2, len(lows) - 2):
                if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and
                    lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                    
                    # Weight by volume
                    volume_weight = volumes[i] / np.mean(volumes) if np.mean(volumes) > 0 else 1.0
                    support_candidates.append((lows[i], volume_weight))
            
            # Sort by strength (price level + volume weight)
            support_candidates.sort(key=lambda x: -x[1])  # Higher volume weight first
            
            # Return unique support levels (remove duplicates within 0.1%)
            unique_supports = []
            for level, weight in support_candidates[:5]:  # Top 5 candidates
                if not any(abs(level - existing) / existing < 0.001 for existing in unique_supports):
                    unique_supports.append(level)
            
            return unique_supports
            
        except Exception as e:
            logger.error(f"Support level identification error: {e}")
            return []
    
    def _identify_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Identify resistance levels using swing highs and volume."""
        try:
            highs = data['high'].values
            volumes = data['volume'].values
            
            # Find local maxima (swing highs)
            resistance_candidates = []
            
            for i in range(2, len(highs) - 2):
                if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and
                    highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                    
                    # Weight by volume
                    volume_weight = volumes[i] / np.mean(volumes) if np.mean(volumes) > 0 else 1.0
                    resistance_candidates.append((highs[i], volume_weight))
            
            # Sort by strength (price level + volume weight)
            resistance_candidates.sort(key=lambda x: -x[1])  # Higher volume weight first
            
            # Return unique resistance levels (remove duplicates within 0.1%)
            unique_resistances = []
            for level, weight in resistance_candidates[:5]:  # Top 5 candidates
                if not any(abs(level - existing) / existing < 0.001 for existing in unique_resistances):
                    unique_resistances.append(level)
            
            return unique_resistances
            
        except Exception as e:
            logger.error(f"Resistance level identification error: {e}")
            return []
    
    def _analyze_price_position(self, current_price: float, 
                               support_levels: List[float], 
                               resistance_levels: List[float]) -> Dict[str, Any]:
        """Analyze current price position relative to support/resistance."""
        position_info = {
            'current': current_price,
            'relative_position': 'middle',
            'near_support': False,
            'near_resistance': False,
            'between_levels': True
        }
        
        if not support_levels or not resistance_levels:
            return position_info
        
        # Find immediate support and resistance
        supports_below = [s for s in support_levels if s < current_price]
        resistances_above = [r for r in resistance_levels if r > current_price]
        
        immediate_support = max(supports_below) if supports_below else min(support_levels)
        immediate_resistance = min(resistances_above) if resistances_above else max(resistance_levels)
        
        # Calculate position within range
        if immediate_support and immediate_resistance:
            range_size = immediate_resistance - immediate_support
            if range_size > 0:
                position_pct = (current_price - immediate_support) / range_size
                
                if position_pct < 0.2:
                    position_info['relative_position'] = 'bottom'
                elif position_pct > 0.8:
                    position_info['relative_position'] = 'top'
                else:
                    position_info['relative_position'] = 'middle'
        
        # Check proximity to levels (within 0.05%)
        position_info['near_support'] = any(
            abs(current_price - s) / current_price < 0.0005 for s in support_levels
        )
        position_info['near_resistance'] = any(
            abs(current_price - r) / current_price < 0.0005 for r in resistance_levels
        )
        
        return position_info
    
    def _detect_current_range(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect if market is currently in a ranging state."""
        try:
            recent_highs = data['high'].tail(self.config.recent_extremes_period)
            recent_lows = data['low'].tail(self.config.recent_extremes_period)
            
            range_high = recent_highs.max()
            range_low = recent_lows.min()
            range_size = range_high - range_low
            
            current_price = float(data['close'].iloc[-1])
            
            # Calculate range characteristics
            range_midpoint = (range_high + range_low) / 2
            range_position = (current_price - range_low) / range_size if range_size > 0 else 0.5
            
            # Determine if we're in a ranging market
            # Check if recent price action is contained within a narrow range
            price_volatility = data['close'].tail(20).std() / data['close'].tail(20).mean()
            in_range = price_volatility < 0.01  # Less than 1% volatility indicates ranging
            
            return {
                'in_range': in_range,
                'range_high': range_high,
                'range_low': range_low,
                'range_size': range_size,
                'range_size_pct': range_size / range_midpoint * 100,
                'range_midpoint': range_midpoint,
                'position': range_position,
                'volatility': price_volatility,
                'near_top': range_position > 0.8,
                'near_bottom': range_position < 0.2,
                'in_middle': 0.3 < range_position < 0.7
            }
            
        except Exception as e:
            logger.error(f"Range detection error: {e}")
            return {'in_range': False, 'position': 0.5}
    
    def _assess_breakout_potential(self, current_price: float,
                                  support_levels: List[float],
                                  resistance_levels: List[float],
                                  volume: pd.Series,
                                  data: pd.DataFrame) -> Dict[str, Any]:
        """Assess potential for micro-breakouts."""
        try:
            # Volume analysis for breakout confirmation
            recent_volume = volume.tail(5).mean()
            avg_volume = volume.tail(20).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum for breakout
            price_change = data['close'].pct_change().tail(3).sum()
            
            # Distance to nearest levels
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else current_price
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else current_price
            
            support_distance = abs(current_price - nearest_support) / current_price
            resistance_distance = abs(current_price - nearest_resistance) / current_price
            
            # Breakout conditions
            upside_breakout_potential = (
                current_price > nearest_resistance * 0.999 and  # Very close to resistance
                volume_ratio > self.config.breakout_volume_multiplier and
                price_change > 0
            )
            
            downside_breakout_potential = (
                current_price < nearest_support * 1.001 and  # Very close to support
                volume_ratio > self.config.breakout_volume_multiplier and
                price_change < 0
            )
            
            return {
                'upside_potential': upside_breakout_potential,
                'downside_potential': downside_breakout_potential,
                'volume_ratio': volume_ratio,
                'price_momentum': price_change,
                'support_distance_pct': support_distance * 100,
                'resistance_distance_pct': resistance_distance * 100,
                'breakout_imminent': upside_breakout_potential or downside_breakout_potential
            }
            
        except Exception as e:
            logger.error(f"Breakout assessment error: {e}")
            return {'upside_potential': False, 'downside_potential': False}
    
    def _get_default_microstructure(self) -> Dict[str, Any]:
        """Return default microstructure analysis for error cases."""
        return {
            'support_levels': [],
            'resistance_levels': [],
            'price_position': {'relative_position': 'unknown'},
            'range_info': {'in_range': False},
            'breakout_potential': {'upside_potential': False, 'downside_potential': False},
            'nearest_support': 0.0,
            'nearest_resistance': 0.0,
            'analysis_timestamp': time.time()
        }


class ScalpingMomentumAnalyzer:
    """Analyzes fast momentum indicators for scalping signals."""
    
    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.fast_rsi = FastRSI(period=config.rsi_period)
        self.williams_r = WilliamsPercentR(period=config.williams_r_period)
        self.fast_ema = FastEMA()
        
    def analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze momentum conditions for scalping opportunities.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary containing momentum analysis results
        """
        try:
            if len(data) < max(self.config.rsi_period, self.config.williams_r_period) + 5:
                return self._get_default_momentum()
            
            # Calculate momentum indicators
            rsi_result = self.fast_rsi.calculate(data)
            williams_result = self.williams_r.calculate(data)
            ema_result = self.fast_ema.calculate(data)
            
            current_rsi = rsi_result.get('rsi', 50)
            current_williams = williams_result.get('williams_r', -50)
            
            # EMA trend analysis
            emas = ema_result.get('emas', {})
            ema_3 = emas.get(self.config.ema_ultra_fast, 0)
            ema_5 = emas.get(self.config.ema_fast, 0)
            ema_8 = emas.get(self.config.ema_medium, 0)
            
            # Momentum state determination
            momentum_state = self._determine_momentum_state(
                current_rsi, current_williams, ema_3, ema_5, ema_8
            )
            
            # Quick reversal signal detection
            reversal_signals = self._detect_quick_reversals(
                rsi_result, williams_result, data['close']
            )
            
            # Momentum divergence analysis
            divergence_analysis = self._analyze_momentum_divergence(
                data, current_rsi, current_williams
            )
            
            return {
                'rsi': {
                    'value': current_rsi,
                    'oversold': current_rsi < 25,
                    'overbought': current_rsi > 75,
                    'extreme_oversold': current_rsi < 20,
                    'extreme_overbought': current_rsi > 80,
                    'momentum': 'bullish' if current_rsi > 50 else 'bearish',
                    'strength': abs(current_rsi - 50) / 50
                },
                'williams_r': {
                    'value': current_williams,
                    'oversold': current_williams < -85,
                    'overbought': current_williams > -15,
                    'extreme_oversold': current_williams < -90,
                    'extreme_overbought': current_williams > -10,
                    'momentum': 'bullish' if current_williams > -50 else 'bearish',
                    'strength': abs(current_williams + 50) / 50
                },
                'ema_trend': {
                    'ultra_fast': ema_3,
                    'fast': ema_5,
                    'medium': ema_8,
                    'direction': self._determine_ema_direction(ema_3, ema_5, ema_8),
                    'strength': self._calculate_ema_strength(ema_3, ema_5, ema_8),
                    'aligned': self._check_ema_alignment(ema_3, ema_5, ema_8)
                },
                'momentum_state': momentum_state,
                'reversal_signals': reversal_signals,
                'divergence': divergence_analysis,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
            return self._get_default_momentum()
    
    def _determine_momentum_state(self, rsi: float, williams: float,
                                 ema_3: float, ema_5: float, ema_8: float) -> str:
        """Determine overall momentum state."""
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI signals
        if rsi > 60:
            bullish_signals += 1
        elif rsi < 40:
            bearish_signals += 1
        
        # Williams %R signals
        if williams > -40:
            bullish_signals += 1
        elif williams < -60:
            bearish_signals += 1
        
        # EMA alignment
        if ema_3 > ema_5 > ema_8:
            bullish_signals += 2  # Strong weight for EMA alignment
        elif ema_3 < ema_5 < ema_8:
            bearish_signals += 2
        
        if bullish_signals > bearish_signals + 1:
            return 'strong_bullish' if bullish_signals >= 3 else 'bullish'
        elif bearish_signals > bullish_signals + 1:
            return 'strong_bearish' if bearish_signals >= 3 else 'bearish'
        else:
            return 'neutral'
    
    def _detect_quick_reversals(self, rsi_result: Dict, williams_result: Dict,
                               prices: pd.Series) -> List[Dict[str, Any]]:
        """Detect quick momentum reversal patterns."""
        reversal_signals = []
        
        try:
            # RSI reversal from extreme levels
            current_rsi = rsi_result.get('rsi', 50)
            if current_rsi < 20:  # Extreme oversold
                reversal_signals.append({
                    'type': 'rsi_oversold_reversal',
                    'strength': (20 - current_rsi) / 20,
                    'direction': 'bullish'
                })
            elif current_rsi > 80:  # Extreme overbought
                reversal_signals.append({
                    'type': 'rsi_overbought_reversal',
                    'strength': (current_rsi - 80) / 20,
                    'direction': 'bearish'
                })
            
            # Williams %R reversal
            current_williams = williams_result.get('williams_r', -50)
            if current_williams < -90:  # Extreme oversold
                reversal_signals.append({
                    'type': 'williams_oversold_reversal',
                    'strength': abs(current_williams + 90) / 10,
                    'direction': 'bullish'
                })
            elif current_williams > -10:  # Extreme overbought
                reversal_signals.append({
                    'type': 'williams_overbought_reversal',
                    'strength': abs(current_williams + 10) / 10,
                    'direction': 'bearish'
                })
            
            # Price action reversal (recent 3-bar pattern)
            if len(prices) >= 3:
                recent_prices = prices.tail(3).values
                if recent_prices[0] > recent_prices[1] > recent_prices[2]:  # Falling prices
                    # Look for bullish reversal
                    if current_rsi < 30 or current_williams < -80:
                        reversal_signals.append({
                            'type': 'price_action_bullish_reversal',
                            'strength': 0.7,
                            'direction': 'bullish'
                        })
                elif recent_prices[0] < recent_prices[1] < recent_prices[2]:  # Rising prices
                    # Look for bearish reversal
                    if current_rsi > 70 or current_williams > -20:
                        reversal_signals.append({
                            'type': 'price_action_bearish_reversal',
                            'strength': 0.7,
                            'direction': 'bearish'
                        })
            
        except Exception as e:
            logger.error(f"Reversal detection error: {e}")
        
        return reversal_signals
    
    def _analyze_momentum_divergence(self, data: pd.DataFrame, 
                                   current_rsi: float, current_williams: float) -> Dict[str, Any]:
        """Analyze momentum divergence patterns."""
        try:
            if len(data) < 10:
                return {'has_divergence': False}
            
            recent_data = data.tail(10)
            price_trend = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
            
            # Simple divergence check (could be enhanced)
            rsi_bullish_divergence = price_trend < 0 and current_rsi > 30
            rsi_bearish_divergence = price_trend > 0 and current_rsi < 70
            
            williams_bullish_divergence = price_trend < 0 and current_williams > -70
            williams_bearish_divergence = price_trend > 0 and current_williams < -30
            
            return {
                'has_divergence': any([
                    rsi_bullish_divergence, rsi_bearish_divergence,
                    williams_bullish_divergence, williams_bearish_divergence
                ]),
                'rsi_bullish_divergence': rsi_bullish_divergence,
                'rsi_bearish_divergence': rsi_bearish_divergence,
                'williams_bullish_divergence': williams_bullish_divergence,
                'williams_bearish_divergence': williams_bearish_divergence
            }
            
        except Exception as e:
            logger.error(f"Divergence analysis error: {e}")
            return {'has_divergence': False}
    
    def _determine_ema_direction(self, ema_3: float, ema_5: float, ema_8: float) -> str:
        """Determine EMA trend direction."""
        if ema_3 > ema_5 > ema_8:
            return 'strong_bullish'
        elif ema_3 > ema_5:
            return 'bullish'
        elif ema_3 < ema_5 < ema_8:
            return 'strong_bearish'
        elif ema_3 < ema_5:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_ema_strength(self, ema_3: float, ema_5: float, ema_8: float) -> float:
        """Calculate EMA trend strength."""
        if ema_8 == 0:
            return 0.0
        
        # Calculate relative distance between EMAs
        spread_3_5 = abs(ema_3 - ema_5) / ema_5 if ema_5 != 0 else 0
        spread_5_8 = abs(ema_5 - ema_8) / ema_8 if ema_8 != 0 else 0
        
        return min((spread_3_5 + spread_5_8) * 100, 1.0)  # Normalize to 0-1
    
    def _check_ema_alignment(self, ema_3: float, ema_5: float, ema_8: float) -> bool:
        """Check if EMAs are properly aligned for trend."""
        return (ema_3 > ema_5 > ema_8) or (ema_3 < ema_5 < ema_8)
    
    def _get_default_momentum(self) -> Dict[str, Any]:
        """Return default momentum analysis for error cases."""
        return {
            'rsi': {'value': 50, 'momentum': 'neutral'},
            'williams_r': {'value': -50, 'momentum': 'neutral'},
            'ema_trend': {'direction': 'neutral'},
            'momentum_state': 'neutral',
            'reversal_signals': [],
            'divergence': {'has_divergence': False},
            'analysis_timestamp': time.time()
        }


class ScalpingVWAPAnalyzer:
    """Analyzes VWAP levels for scalping bounce signals."""
    
    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.vwap_calculator = ScalpingVWAP()
        
    def analyze_vwap_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze VWAP-based scalping opportunities.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary containing VWAP analysis results
        """
        try:
            if len(data) < max(self.config.vwap_periods) + 10:
                return self._get_default_vwap()
            
            current_price = float(data['close'].iloc[-1])
            current_volume = float(data['volume'].iloc[-1])
            
            # Calculate VWAP for different periods
            vwap_results = {}
            for period in self.config.vwap_periods:
                vwap_data = self._calculate_vwap(data, period)
                vwap_results[f'vwap_{period}'] = vwap_data
            
            # VWAP position analysis
            vwap_position = self._analyze_vwap_position(current_price, vwap_results)
            
            # VWAP bounce signal detection
            bounce_signals = self._detect_vwap_bounce_signals(
                data, current_price, vwap_results
            )
            
            # Volume-weighted momentum
            volume_momentum = self._calculate_volume_momentum(data)
            
            # VWAP deviation analysis
            deviation_analysis = self._analyze_vwap_deviation(current_price, vwap_results)
            
            return {
                'vwap_levels': {k: v['current'] for k, v in vwap_results.items()},
                'position': vwap_position,
                'bounce_signals': bounce_signals,
                'volume_momentum': volume_momentum,
                'deviation': deviation_analysis,
                'current_price': current_price,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"VWAP analysis error: {e}")
            return self._get_default_vwap()
    
    def _calculate_vwap(self, data: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Calculate VWAP for a specific period."""
        try:
            recent_data = data.tail(period)
            typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
            
            cumulative_volume = recent_data['volume'].cumsum()
            cumulative_pv = (typical_price * recent_data['volume']).cumsum()
            
            vwap_series = cumulative_pv / cumulative_volume
            current_vwap = float(vwap_series.iloc[-1])
            
            # Calculate VWAP bands (standard deviation based)
            price_deviations = typical_price - vwap_series
            squared_deviations = price_deviations ** 2
            volume_weighted_variance = (squared_deviations * recent_data['volume']).cumsum() / cumulative_volume
            vwap_std = np.sqrt(volume_weighted_variance.iloc[-1])
            
            return {
                'current': current_vwap,
                'series': vwap_series,
                'std': vwap_std,
                'upper_band': current_vwap + vwap_std,
                'lower_band': current_vwap - vwap_std,
                'period': period
            }
            
        except Exception as e:
            logger.error(f"VWAP calculation error for period {period}: {e}")
            return {'current': 0.0, 'std': 0.0}
    
    def _analyze_vwap_position(self, current_price: float, 
                              vwap_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze current price position relative to VWAP levels."""
        position_info = {
            'above_vwap': {},
            'below_vwap': {},
            'near_vwap': {},
            'overall_position': 'neutral'
        }
        
        above_count = 0
        below_count = 0
        
        for vwap_name, vwap_data in vwap_results.items():
            vwap_level = vwap_data.get('current', 0)
            if vwap_level == 0:
                continue
                
            deviation_pct = (current_price - vwap_level) / vwap_level * 100
            
            if current_price > vwap_level:
                position_info['above_vwap'][vwap_name] = deviation_pct
                above_count += 1
            else:
                position_info['below_vwap'][vwap_name] = deviation_pct
                below_count += 1
            
            # Check if near VWAP (within 0.1%)
            if abs(deviation_pct) < 0.1:
                position_info['near_vwap'][vwap_name] = deviation_pct
        
        # Determine overall position
        if above_count > below_count:
            position_info['overall_position'] = 'above'
        elif below_count > above_count:
            position_info['overall_position'] = 'below'
        else:
            position_info['overall_position'] = 'neutral'
        
        return position_info
    
    def _detect_vwap_bounce_signals(self, data: pd.DataFrame, 
                                   current_price: float,
                                   vwap_results: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Detect VWAP bounce signals."""
        bounce_signals = []
        
        try:
            # Check recent price action relative to VWAP
            recent_prices = data['close'].tail(5).values
            recent_volumes = data['volume'].tail(5).values
            
            for vwap_name, vwap_data in vwap_results.items():
                vwap_level = vwap_data.get('current', 0)
                if vwap_level == 0:
                    continue
                
                vwap_std = vwap_data.get('std', 0)
                upper_band = vwap_data.get('upper_band', vwap_level)
                lower_band = vwap_data.get('lower_band', vwap_level)
                
                # Bullish bounce from VWAP or lower band
                if (current_price < vwap_level * 1.001 and  # Very close to VWAP
                    current_price > vwap_level * 0.999 and
                    len(recent_prices) >= 3 and
                    recent_prices[-1] > recent_prices[-2]):  # Price starting to rise
                    
                    bounce_signals.append({
                        'type': 'vwap_bullish_bounce',
                        'vwap_period': vwap_data.get('period', 0),
                        'strength': 0.7,
                        'direction': 'bullish',
                        'entry_level': vwap_level,
                        'target_level': upper_band,
                        'stop_level': lower_band
                    })
                
                # Bearish rejection from VWAP or upper band
                elif (current_price > vwap_level * 0.999 and  # Very close to VWAP
                      current_price < vwap_level * 1.001 and
                      len(recent_prices) >= 3 and
                      recent_prices[-1] < recent_prices[-2]):  # Price starting to fall
                    
                    bounce_signals.append({
                        'type': 'vwap_bearish_rejection',
                        'vwap_period': vwap_data.get('period', 0),
                        'strength': 0.7,
                        'direction': 'bearish',
                        'entry_level': vwap_level,
                        'target_level': lower_band,
                        'stop_level': upper_band
                    })
                
                # Band bounce signals
                if current_price <= lower_band * 1.001:  # At or below lower band
                    bounce_signals.append({
                        'type': 'vwap_lower_band_bounce',
                        'vwap_period': vwap_data.get('period', 0),
                        'strength': 0.8,
                        'direction': 'bullish',
                        'entry_level': current_price,
                        'target_level': vwap_level,
                        'stop_level': lower_band * 0.999
                    })
                
                elif current_price >= upper_band * 0.999:  # At or above upper band
                    bounce_signals.append({
                        'type': 'vwap_upper_band_rejection',
                        'vwap_period': vwap_data.get('period', 0),
                        'strength': 0.8,
                        'direction': 'bearish',
                        'entry_level': current_price,
                        'target_level': vwap_level,
                        'stop_level': upper_band * 1.001
                    })
                    
        except Exception as e:
            logger.error(f"VWAP bounce detection error: {e}")
        
        return bounce_signals
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-weighted momentum."""
        try:
            if len(data) < 10:
                return {'momentum': 'neutral', 'strength': 0.0}
            
            recent_data = data.tail(10)
            
            # Volume-weighted price change
            price_changes = recent_data['close'].pct_change().fillna(0)
            volumes = recent_data['volume']
            
            # Weight price changes by volume
            volume_weighted_momentum = (price_changes * volumes).sum() / volumes.sum()
            
            # Determine momentum direction and strength
            if volume_weighted_momentum > 0.001:  # 0.1% threshold
                momentum = 'bullish'
                strength = min(volume_weighted_momentum * 100, 1.0)
            elif volume_weighted_momentum < -0.001:
                momentum = 'bearish'
                strength = min(abs(volume_weighted_momentum) * 100, 1.0)
            else:
                momentum = 'neutral'
                strength = 0.0
            
            return {
                'momentum': momentum,
                'strength': strength,
                'raw_value': volume_weighted_momentum
            }
            
        except Exception as e:
            logger.error(f"Volume momentum calculation error: {e}")
            return {'momentum': 'neutral', 'strength': 0.0}
    
    def _analyze_vwap_deviation(self, current_price: float,
                               vwap_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze price deviation from VWAP levels."""
        deviations = {}
        max_deviation = 0.0
        max_deviation_period = None
        
        for vwap_name, vwap_data in vwap_results.items():
            vwap_level = vwap_data.get('current', 0)
            if vwap_level == 0:
                continue
            
            deviation_pct = (current_price - vwap_level) / vwap_level * 100
            deviations[vwap_name] = {
                'deviation_pct': deviation_pct,
                'is_extreme': abs(deviation_pct) > 0.2,  # 0.2% is extreme for scalping
                'direction': 'above' if deviation_pct > 0 else 'below'
            }
            
            if abs(deviation_pct) > abs(max_deviation):
                max_deviation = deviation_pct
                max_deviation_period = vwap_data.get('period', 0)
        
        return {
            'individual_deviations': deviations,
            'max_deviation_pct': max_deviation,
            'max_deviation_period': max_deviation_period,
            'mean_reversion_opportunity': abs(max_deviation) > 0.15
        }
    
    def _get_default_vwap(self) -> Dict[str, Any]:
        """Return default VWAP analysis for error cases."""
        return {
            'vwap_levels': {},
            'position': {'overall_position': 'neutral'},
            'bounce_signals': [],
            'volume_momentum': {'momentum': 'neutral'},
            'deviation': {'mean_reversion_opportunity': False},
            'analysis_timestamp': time.time()
        }


class ScalpingSignalGenerator:
    """Generates and prioritizes scalping signals from multiple analysis sources."""
    
    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.microstructure_analyzer = ScalpingMicrostructureAnalyzer(config)
        self.momentum_analyzer = ScalpingMomentumAnalyzer(config)
        self.vwap_analyzer = ScalpingVWAPAnalyzer(config)
        
        # Signal generation statistics
        self.signals_generated = 0
        self.signals_executed = 0
        
    def generate_signals(self, data: pd.DataFrame) -> List[ScalpingSignal]:
        """
        Generate prioritized scalping signals from market analysis.
        
        Args:
            data: OHLCV DataFrame with recent market data
            
        Returns:
            List of ScalpingSignal objects sorted by priority
        """
        start_time = time.perf_counter()
        
        try:
            if len(data) < self.config.min_data_points:
                logger.warning(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
                return []
            
            current_price = float(data['close'].iloc[-1])
            current_time = time.time()
            
            # Perform analysis
            microstructure = self.microstructure_analyzer.analyze_microstructure(data)
            momentum = self.momentum_analyzer.analyze_momentum(data)
            vwap_analysis = self.vwap_analyzer.analyze_vwap_signals(data)
            
            signals = []
            
            # Generate different types of scalping signals
            mean_reversion_signal = self._check_mean_reversion(
                microstructure, momentum, vwap_analysis, current_price, current_time
            )
            if mean_reversion_signal:
                signals.append(mean_reversion_signal)
            
            micro_breakout_signal = self._check_micro_breakout(
                microstructure, momentum, data, current_price, current_time
            )
            if micro_breakout_signal:
                signals.append(micro_breakout_signal)
            
            vwap_bounce_signal = self._check_vwap_bounce(
                vwap_analysis, momentum, current_price, current_time
            )
            if vwap_bounce_signal:
                signals.append(vwap_bounce_signal)
            
            support_resistance_signal = self._check_support_resistance_scalp(
                microstructure, momentum, current_price, current_time
            )
            if support_resistance_signal:
                signals.append(support_resistance_signal)
            
            momentum_spike_signal = self._check_momentum_spike(
                momentum, data, current_price, current_time
            )
            if momentum_spike_signal:
                signals.append(momentum_spike_signal)
            
            volume_anomaly_signal = self._check_volume_anomaly(
                data, momentum, current_price, current_time
            )
            if volume_anomaly_signal:
                signals.append(volume_anomaly_signal)
            
            # Filter and prioritize signals
            filtered_signals = self._filter_scalping_signals(signals, data)
            prioritized_signals = self._prioritize_by_timing(filtered_signals)
            
            # Performance check
            analysis_time = time.perf_counter() - start_time
            if analysis_time > self.config.max_analysis_time_ms / 1000:
                logger.warning(f"Slow signal generation: {analysis_time*1000:.2f}ms")
            
            self.signals_generated += len(prioritized_signals)
            return prioritized_signals
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return []
    
    def _check_mean_reversion(self, microstructure: Dict, momentum: Dict, 
                             vwap_analysis: Dict, current_price: float,
                             current_time: float) -> Optional[ScalpingSignal]:
        """Check for mean reversion scalping opportunities."""
        try:
            entry_reasons = []
            risk_factors = []
            confidence_factors = []
            
            # Mean reversion conditions
            conditions = {}
            
            # Price extended from VWAP
            vwap_deviation = vwap_analysis.get('deviation', {})
            max_deviation = abs(vwap_deviation.get('max_deviation_pct', 0))
            if max_deviation > 0.15:  # 15 basis points deviation
                conditions['price_extended'] = True
                entry_reasons.append(f"Price extended {max_deviation:.2f}% from VWAP")
                confidence_factors.append(min(max_deviation / 0.3, 1.0))  # Scale to 1.0 max
            else:
                conditions['price_extended'] = False
            
            # Momentum extremes
            rsi_data = momentum.get('rsi', {})
            williams_data = momentum.get('williams_r', {})
            
            momentum_extreme = (
                rsi_data.get('oversold', False) or rsi_data.get('overbought', False) or
                williams_data.get('oversold', False) or williams_data.get('overbought', False)
            )
            conditions['momentum_extreme'] = momentum_extreme
            
            if momentum_extreme:
                if rsi_data.get('oversold', False) or williams_data.get('oversold', False):
                    entry_reasons.append("Momentum oversold")
                    confidence_factors.append(0.8)
                elif rsi_data.get('overbought', False) or williams_data.get('overbought', False):
                    entry_reasons.append("Momentum overbought")
                    confidence_factors.append(0.8)
            
            # Near support/resistance
            support_distance = microstructure.get('support_distance_pct', 100)
            resistance_distance = microstructure.get('resistance_distance_pct', 100)
            near_level = min(support_distance, resistance_distance) < 0.05  # Within 5 basis points
            conditions['near_support_resistance'] = near_level
            
            if near_level:
                entry_reasons.append(f"Near key level ({min(support_distance, resistance_distance):.3f}%)")
                confidence_factors.append(0.7)
            
            # Reversal signals present
            reversal_signals = momentum.get('reversal_signals', [])
            conditions['reversal_signal'] = len(reversal_signals) > 0
            
            if reversal_signals:
                entry_reasons.append(f"{len(reversal_signals)} reversal signals")
                confidence_factors.append(0.6)
            
            # Calculate overall confidence
            confidence = np.mean(confidence_factors) if confidence_factors else 0.0
            condition_count = sum(conditions.values())
            
            # Require at least 2 conditions and minimum confidence
            if condition_count >= 2 and confidence >= self.config.min_signal_strength:
                
                # Determine direction
                direction = 'hold'
                if (rsi_data.get('oversold', False) or williams_data.get('oversold', False) or
                    vwap_deviation.get('max_deviation_pct', 0) < -0.15):
                    direction = 'buy'
                elif (rsi_data.get('overbought', False) or williams_data.get('overbought', False) or
                      vwap_deviation.get('max_deviation_pct', 0) > 0.15):
                    direction = 'sell'
                
                if direction != 'hold':
                    # Risk factors
                    if not vwap_analysis.get('volume_momentum', {}).get('momentum') in ['bullish', 'bearish']:
                        risk_factors.append("weak_volume_momentum")
                    
                    if microstructure.get('range_info', {}).get('in_range', False):
                        risk_factors.append("ranging_market")
                    
                    return ScalpingSignal(
                        signal_type=ScalpingSignalType.MEAN_REVERSION,
                        direction=direction,
                        confidence=confidence,
                        timing=ScalpingTiming.QUICK,
                        target_profit_pct=self._calculate_target_profit(confidence),
                        stop_loss_pct=self.config.stop_loss_pct,
                        max_holding_time=self.config.max_holding_time,
                        current_price=current_price,
                        entry_reasons=entry_reasons,
                        risk_factors=risk_factors,
                        market_conditions=conditions,
                        timestamp=current_time,
                        expires_at=current_time + self.config.signal_timeout_seconds,
                        priority_score=confidence * (condition_count / 4.0),
                        microstructure=microstructure,
                        momentum=momentum,
                        vwap_analysis=vwap_analysis
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Mean reversion check error: {e}")
            return None
    
    def _check_micro_breakout(self, microstructure: Dict, momentum: Dict,
                             data: pd.DataFrame, current_price: float,
                             current_time: float) -> Optional[ScalpingSignal]:
        """Check for micro-breakout opportunities."""
        try:
            breakout_potential = microstructure.get('breakout_potential', {})
            
            if not (breakout_potential.get('upside_potential') or breakout_potential.get('downside_potential')):
                return None
            
            entry_reasons = []
            risk_factors = []
            
            # Determine direction
            direction = 'hold'
            if breakout_potential.get('upside_potential'):
                direction = 'buy'
                entry_reasons.append("Upside breakout potential")
            elif breakout_potential.get('downside_potential'):
                direction = 'sell'
                entry_reasons.append("Downside breakout potential")
            
            if direction == 'hold':
                return None
            
            # Volume confirmation
            volume_ratio = breakout_potential.get('volume_ratio', 1.0)
            if volume_ratio >= self.config.breakout_volume_multiplier:
                entry_reasons.append(f"Volume confirmation ({volume_ratio:.1f}x)")
                volume_confidence = 0.8
            else:
                risk_factors.append("insufficient_volume")
                volume_confidence = 0.4
            
            # Momentum alignment
            momentum_state = momentum.get('momentum_state', 'neutral')
            momentum_aligned = (
                (direction == 'buy' and 'bullish' in momentum_state) or
                (direction == 'sell' and 'bearish' in momentum_state)
            )
            
            if momentum_aligned:
                entry_reasons.append("Momentum aligned")
                momentum_confidence = 0.7
            else:
                risk_factors.append("momentum_misalignment")
                momentum_confidence = 0.3
            
            # Calculate confidence
            base_confidence = 0.6  # Base confidence for breakouts
            confidence = base_confidence * volume_confidence * momentum_confidence
            
            if confidence >= self.config.min_signal_strength:
                return ScalpingSignal(
                    signal_type=ScalpingSignalType.MICRO_BREAKOUT,
                    direction=direction,
                    confidence=confidence,
                    timing=ScalpingTiming.IMMEDIATE,  # Breakouts need immediate action
                    target_profit_pct=self._calculate_target_profit(confidence, is_breakout=True),
                    stop_loss_pct=self.config.stop_loss_pct * 1.2,  # Wider stop for breakouts
                    max_holding_time=self.config.max_holding_time // 2,  # Shorter holding time
                    current_price=current_price,
                    entry_reasons=entry_reasons,
                    risk_factors=risk_factors,
                    market_conditions={'breakout_potential': breakout_potential},
                    timestamp=current_time,
                    expires_at=current_time + 30,  # Breakouts expire quickly
                    priority_score=confidence * 1.2,  # Higher priority for breakouts
                    microstructure=microstructure,
                    momentum=momentum
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Micro breakout check error: {e}")
            return None
    
    def _check_vwap_bounce(self, vwap_analysis: Dict, momentum: Dict,
                          current_price: float, current_time: float) -> Optional[ScalpingSignal]:
        """Check for VWAP bounce opportunities."""
        try:
            bounce_signals = vwap_analysis.get('bounce_signals', [])
            
            if not bounce_signals:
                return None
            
            # Take the strongest bounce signal
            best_signal = max(bounce_signals, key=lambda x: x.get('strength', 0))
            
            entry_reasons = [f"VWAP {best_signal.get('type', 'bounce')}"]
            risk_factors = []
            
            # Volume momentum confirmation
            volume_momentum = vwap_analysis.get('volume_momentum', {})
            volume_aligned = (
                (best_signal.get('direction') == 'bullish' and volume_momentum.get('momentum') == 'bullish') or
                (best_signal.get('direction') == 'bearish' and volume_momentum.get('momentum') == 'bearish')
            )
            
            if volume_aligned:
                entry_reasons.append("Volume momentum aligned")
                volume_factor = 1.0
            else:
                risk_factors.append("volume_momentum_misalignment")
                volume_factor = 0.7
            
            # Calculate confidence
            base_confidence = best_signal.get('strength', 0.7)
            confidence = base_confidence * volume_factor
            
            if confidence >= self.config.min_signal_strength:
                direction = 'buy' if best_signal.get('direction') == 'bullish' else 'sell'
                
                return ScalpingSignal(
                    signal_type=ScalpingSignalType.VWAP_BOUNCE,
                    direction=direction,
                    confidence=confidence,
                    timing=ScalpingTiming.QUICK,
                    target_profit_pct=self._calculate_target_profit(confidence),
                    stop_loss_pct=self.config.stop_loss_pct,
                    max_holding_time=self.config.max_holding_time,
                    current_price=current_price,
                    entry_reasons=entry_reasons,
                    risk_factors=risk_factors,
                    market_conditions={'vwap_signal': best_signal},
                    timestamp=current_time,
                    expires_at=current_time + self.config.signal_timeout_seconds,
                    priority_score=confidence * 0.9,
                    vwap_analysis=vwap_analysis,
                    momentum=momentum
                )
            
            return None
            
        except Exception as e:
            logger.error(f"VWAP bounce check error: {e}")
            return None
    
    def _check_support_resistance_scalp(self, microstructure: Dict, momentum: Dict,
                                       current_price: float, current_time: float) -> Optional[ScalpingSignal]:
        """Check for support/resistance scalping opportunities."""
        try:
            price_position = microstructure.get('price_position', {})
            
            if not (price_position.get('near_support') or price_position.get('near_resistance')):
                return None
            
            entry_reasons = []
            risk_factors = []
            direction = 'hold'
            
            # Support bounce opportunity
            if price_position.get('near_support'):
                direction = 'buy'
                entry_reasons.append("Near support level")
                
                # Check for bullish momentum
                momentum_state = momentum.get('momentum_state', 'neutral')
                if 'bullish' in momentum_state:
                    entry_reasons.append("Bullish momentum at support")
                elif 'bearish' in momentum_state:
                    risk_factors.append("bearish_momentum_at_support")
            
            # Resistance rejection opportunity
            elif price_position.get('near_resistance'):
                direction = 'sell'
                entry_reasons.append("Near resistance level")
                
                # Check for bearish momentum
                momentum_state = momentum.get('momentum_state', 'neutral')
                if 'bearish' in momentum_state:
                    entry_reasons.append("Bearish momentum at resistance")
                elif 'bullish' in momentum_state:
                    risk_factors.append("bullish_momentum_at_resistance")
            
            if direction == 'hold':
                return None
            
            # Calculate confidence based on level strength and momentum
            base_confidence = 0.65
            momentum_factor = 0.8 if len(risk_factors) == 0 else 0.5
            confidence = base_confidence * momentum_factor
            
            if confidence >= self.config.min_signal_strength:
                return ScalpingSignal(
                    signal_type=ScalpingSignalType.SUPPORT_RESISTANCE,
                    direction=direction,
                    confidence=confidence,
                    timing=ScalpingTiming.NORMAL,
                    target_profit_pct=self._calculate_target_profit(confidence),
                    stop_loss_pct=self.config.stop_loss_pct,
                    max_holding_time=self.config.max_holding_time,
                    current_price=current_price,
                    entry_reasons=entry_reasons,
                    risk_factors=risk_factors,
                    market_conditions={'price_position': price_position},
                    timestamp=current_time,
                    expires_at=current_time + self.config.signal_timeout_seconds,
                    priority_score=confidence * 0.8,
                    microstructure=microstructure,
                    momentum=momentum
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Support/resistance check error: {e}")
            return None
    
    def _check_momentum_spike(self, momentum: Dict, data: pd.DataFrame,
                             current_price: float, current_time: float) -> Optional[ScalpingSignal]:
        """Check for short-term momentum spike opportunities."""
        try:
            # Look for sudden momentum changes
            if len(data) < 5:
                return None
            
            recent_returns = data['close'].pct_change().tail(3)
            volume_recent = data['volume'].tail(3)
            volume_avg = data['volume'].tail(10).mean()
            
            # Check for momentum spike conditions
            price_spike = abs(recent_returns.iloc[-1]) > 0.002  # 20 basis points move
            volume_spike = volume_recent.iloc[-1] > volume_avg * 1.5
            
            if not (price_spike and volume_spike):
                return None
            
            entry_reasons = []
            risk_factors = []
            
            # Determine direction
            direction = 'buy' if recent_returns.iloc[-1] > 0 else 'sell'
            entry_reasons.append(f"Momentum spike: {recent_returns.iloc[-1]*100:.2f}%")
            entry_reasons.append(f"Volume spike: {volume_recent.iloc[-1]/volume_avg:.1f}x")
            
            # Check if momentum is sustainable
            momentum_state = momentum.get('momentum_state', 'neutral')
            if ((direction == 'buy' and 'bullish' in momentum_state) or
                (direction == 'sell' and 'bearish' in momentum_state)):
                entry_reasons.append("Momentum alignment")
                sustainability_factor = 0.8
            else:
                risk_factors.append("momentum_reversal_risk")
                sustainability_factor = 0.6
            
            # Calculate confidence
            spike_strength = min(abs(recent_returns.iloc[-1]) * 100, 1.0)  # Normalize
            volume_strength = min(volume_recent.iloc[-1] / volume_avg / 2, 1.0)
            confidence = (spike_strength + volume_strength) / 2 * sustainability_factor
            
            if confidence >= self.config.min_signal_strength:
                return ScalpingSignal(
                    signal_type=ScalpingSignalType.MOMENTUM_SPIKE,
                    direction=direction,
                    confidence=confidence,
                    timing=ScalpingTiming.IMMEDIATE,  # Momentum spikes need immediate action
                    target_profit_pct=self._calculate_target_profit(confidence, is_momentum=True),
                    stop_loss_pct=self.config.stop_loss_pct * 0.8,  # Tighter stop for momentum
                    max_holding_time=self.config.quick_exit_threshold,  # Quick exit
                    current_price=current_price,
                    entry_reasons=entry_reasons,
                    risk_factors=risk_factors,
                    market_conditions={'price_spike': recent_returns.iloc[-1], 'volume_spike': volume_recent.iloc[-1]/volume_avg},
                    timestamp=current_time,
                    expires_at=current_time + 20,  # Very short expiry
                    priority_score=confidence * 1.3,  # High priority
                    momentum=momentum
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Momentum spike check error: {e}")
            return None
    
    def _check_volume_anomaly(self, data: pd.DataFrame, momentum: Dict,
                             current_price: float, current_time: float) -> Optional[ScalpingSignal]:
        """Check for volume anomaly scalping opportunities."""
        try:
            if len(data) < 20:
                return None
            
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].tail(20).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Look for significant volume anomalies
            if volume_ratio < 2.0:  # Need at least 2x average volume
                return None
            
            entry_reasons = [f"Volume anomaly: {volume_ratio:.1f}x average"]
            risk_factors = []
            
            # Determine direction based on price action with high volume
            recent_price_change = data['close'].pct_change().iloc[-1]
            
            if abs(recent_price_change) < 0.001:  # No significant price movement
                risk_factors.append("high_volume_no_price_movement")
                return None
            
            direction = 'buy' if recent_price_change > 0 else 'sell'
            entry_reasons.append(f"Price movement: {recent_price_change*100:.2f}%")
            
            # Check momentum alignment
            momentum_state = momentum.get('momentum_state', 'neutral')
            if ((direction == 'buy' and 'bullish' in momentum_state) or
                (direction == 'sell' and 'bearish' in momentum_state)):
                entry_reasons.append("Momentum aligned with volume")
                alignment_factor = 1.0
            else:
                risk_factors.append("momentum_volume_divergence")
                alignment_factor = 0.7
            
            # Calculate confidence
            volume_strength = min(volume_ratio / 5, 1.0)  # Normalize to max 1.0
            price_strength = min(abs(recent_price_change) * 50, 1.0)
            confidence = (volume_strength + price_strength) / 2 * alignment_factor
            
            if confidence >= self.config.min_signal_strength:
                return ScalpingSignal(
                    signal_type=ScalpingSignalType.VOLUME_ANOMALY,
                    direction=direction,
                    confidence=confidence,
                    timing=ScalpingTiming.QUICK,
                    target_profit_pct=self._calculate_target_profit(confidence),
                    stop_loss_pct=self.config.stop_loss_pct,
                    max_holding_time=self.config.max_holding_time // 2,  # Shorter holding time
                    current_price=current_price,
                    entry_reasons=entry_reasons,
                    risk_factors=risk_factors,
                    market_conditions={'volume_ratio': volume_ratio, 'price_change': recent_price_change},
                    timestamp=current_time,
                    expires_at=current_time + self.config.signal_timeout_seconds,
                    priority_score=confidence * 1.1,
                    momentum=momentum
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Volume anomaly check error: {e}")
            return None
    
    def _calculate_target_profit(self, confidence: float, is_breakout: bool = False, 
                                is_momentum: bool = False) -> float:
        """Calculate target profit based on signal confidence and type."""
        base_target = self.config.min_profit_target_pct
        max_target = self.config.max_profit_target_pct
        
        # Scale target based on confidence
        confidence_multiplier = 1.0 + confidence  # 1.0 to 2.0 range
        target = base_target * confidence_multiplier
        
        # Adjust for signal type
        if is_breakout:
            target *= 1.5  # Breakouts can have larger targets
        elif is_momentum:
            target *= 1.2  # Momentum can have slightly larger targets
        
        return min(target, max_target)
    
    def _filter_scalping_signals(self, signals: List[ScalpingSignal], 
                                data: pd.DataFrame) -> List[ScalpingSignal]:
        """Filter signals based on quality and market conditions."""
        filtered = []
        
        for signal in signals:
            # Check signal expiry
            if signal.expires_at < time.time():
                continue
            
            # Check minimum confidence
            if signal.confidence < self.config.min_signal_strength:
                continue
            
            # Check for too many risk factors
            if len(signal.risk_factors) > 2:
                continue
            
            # Check spread conditions (if available)
            # This would require bid/ask data which may not be available
            
            filtered.append(signal)
        
        return filtered
    
    def _prioritize_by_timing(self, signals: List[ScalpingSignal]) -> List[ScalpingSignal]:
        """Prioritize signals by timing urgency and confidence."""
        # Define timing priority weights
        timing_weights = {
            ScalpingTiming.IMMEDIATE: 3.0,
            ScalpingTiming.QUICK: 2.0,
            ScalpingTiming.NORMAL: 1.0,
            ScalpingTiming.WAIT: 0.5
        }
        
        # Calculate final priority scores
        for signal in signals:
            timing_weight = timing_weights.get(signal.timing, 1.0)
            signal.priority_score = signal.confidence * timing_weight
        
        # Sort by priority score (highest first)
        return sorted(signals, key=lambda x: x.priority_score, reverse=True)


class ScalpingRiskManager:
    """Manages risk for scalping operations with tight controls."""
    
    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.consecutive_losses = 0
        self.recent_trades = deque(maxlen=50)
        self.daily_trades = 0
        self.last_trade_time = 0
        self.last_reset_date = time.strftime('%Y-%m-%d')
        
    def calculate_position_size(self, signal: ScalpingSignal, 
                               account_balance: float,
                               current_position: Optional[Position] = None) -> Dict[str, Any]:
        """
        Calculate appropriate position size for scalping signal.
        
        Args:
            signal: Scalping signal to size
            account_balance: Available account balance
            current_position: Current position if any
            
        Returns:
            Dictionary with position sizing information
        """
        try:
            self._reset_daily_counters_if_needed()
            
            # Check daily trade limit
            if self.daily_trades >= self.config.max_daily_trades:
                return self._get_no_position_response("daily_trade_limit_reached")
            
            # Check consecutive loss protection
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                return self._get_no_position_response("consecutive_loss_limit")
            
            # Base position size
            base_size_pct = self.config.base_position_pct
            
            # Frequency adjustment
            if self.config.frequency_adjusted:
                frequency_factor = self._calculate_frequency_factor()
                base_size_pct *= frequency_factor
            
            # Confidence-based sizing
            confidence_multiplier = signal.confidence * 1.5  # Scale to 0-1.5
            position_size_pct = base_size_pct * confidence_multiplier
            
            # Apply maximum limit
            max_size_pct = self.config.max_position_pct
            position_size_pct = min(position_size_pct, max_size_pct)
            
            # Consecutive loss protection
            if self.consecutive_losses > 0:
                loss_factor = max(0.5, 1.0 - (self.consecutive_losses * 0.2))
                position_size_pct *= loss_factor
            
            # Convert to dollar amount
            position_size = account_balance * (position_size_pct / 100)
            
            # Minimum position size check
            min_position_size = account_balance * 0.001  # 0.1% minimum
            if position_size < min_position_size:
                return self._get_no_position_response("position_too_small")
            
            return {
                'size_usd': position_size,
                'size_pct': position_size_pct,
                'stop_loss_pct': signal.stop_loss_pct,
                'target_profit_pct': signal.target_profit_pct,
                'max_holding_time': signal.max_holding_time,
                'quick_exit_time': self.config.quick_exit_threshold,
                'immediate_exit_time': self.config.immediate_exit_time,
                'confidence': signal.confidence,
                'risk_factors': signal.risk_factors,
                'consecutive_losses': self.consecutive_losses,
                'daily_trades': self.daily_trades,
                'approved': True
            }
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return self._get_no_position_response("calculation_error")
    
    def validate_signal_execution(self, signal: ScalpingSignal) -> Dict[str, Any]:
        """
        Validate if a signal should be executed based on risk parameters.
        
        Args:
            signal: Signal to validate
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'approved': True,
            'reasons': [],
            'warnings': []
        }
        
        try:
            # Check signal freshness
            signal_age = time.time() - signal.timestamp
            if signal_age > self.config.signal_timeout_seconds:
                validation_result['approved'] = False
                validation_result['reasons'].append("signal_expired")
            
            # Check signal expiry
            if signal.expires_at < time.time():
                validation_result['approved'] = False
                validation_result['reasons'].append("signal_past_expiry")
            
            # Check cooldown period after losses
            if self.consecutive_losses > 0:
                time_since_last_trade = time.time() - self.last_trade_time
                if time_since_last_trade < self.config.cooldown_after_loss:
                    validation_result['approved'] = False
                    validation_result['reasons'].append("cooldown_period_active")
            
            # Check confidence threshold
            if signal.confidence < self.config.min_signal_strength:
                validation_result['approved'] = False
                validation_result['reasons'].append("insufficient_confidence")
            
            # Check risk factors
            if len(signal.risk_factors) > 2:
                validation_result['warnings'].append("high_risk_factors")
                if len(signal.risk_factors) > 3:
                    validation_result['approved'] = False
                    validation_result['reasons'].append("too_many_risk_factors")
            
            # Check market conditions
            if signal.timing == ScalpingTiming.WAIT:
                validation_result['approved'] = False
                validation_result['reasons'].append("signal_suggests_waiting")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Signal validation error: {e}")
            return {
                'approved': False,
                'reasons': ['validation_error'],
                'warnings': []
            }
    
    def update_trade_result(self, trade_result: Dict[str, Any]):
        """
        Update risk manager with trade result.
        
        Args:
            trade_result: Dictionary containing trade outcome
        """
        try:
            self.recent_trades.append(trade_result)
            self.daily_trades += 1
            self.last_trade_time = time.time()
            
            # Update consecutive loss counter
            if trade_result.get('profit_loss', 0) < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0  # Reset on profit
            
            logger.info(f"Trade result updated: P&L={trade_result.get('profit_loss', 0):.4f}, "
                       f"Consecutive losses: {self.consecutive_losses}, Daily trades: {self.daily_trades}")
            
        except Exception as e:
            logger.error(f"Trade result update error: {e}")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        self._reset_daily_counters_if_needed()
        
        recent_trades_count = len(self.recent_trades)
        win_rate = 0.0
        avg_profit = 0.0
        
        if recent_trades_count > 0:
            wins = sum(1 for trade in self.recent_trades if trade.get('profit_loss', 0) > 0)
            win_rate = wins / recent_trades_count
            avg_profit = np.mean([trade.get('profit_loss', 0) for trade in self.recent_trades])
        
        return {
            'consecutive_losses': self.consecutive_losses,
            'daily_trades': self.daily_trades,
            'daily_trades_remaining': self.config.max_daily_trades - self.daily_trades,
            'recent_trades_count': recent_trades_count,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit,
            'risk_level': self._calculate_risk_level(),
            'cooldown_active': self.consecutive_losses > 0 and (time.time() - self.last_trade_time) < self.config.cooldown_after_loss
        }
    
    def _calculate_frequency_factor(self) -> float:
        """Calculate frequency adjustment factor."""
        # Reduce position size based on recent trade frequency
        if len(self.recent_trades) < 5:
            return 1.0
        
        recent_trade_times = [trade.get('timestamp', 0) for trade in list(self.recent_trades)[-5:]]
        if not recent_trade_times:
            return 1.0
        
        # Calculate average time between recent trades
        time_diffs = [recent_trade_times[i] - recent_trade_times[i-1] 
                     for i in range(1, len(recent_trade_times))]
        
        if not time_diffs:
            return 1.0
        
        avg_time_between_trades = np.mean(time_diffs)
        
        # If trading very frequently (< 5 minutes between trades), reduce size
        if avg_time_between_trades < 300:  # 5 minutes
            return max(0.5, avg_time_between_trades / 300)
        
        return 1.0
    
    def _calculate_risk_level(self) -> str:
        """Calculate current risk level."""
        risk_score = 0
        
        # Consecutive losses increase risk
        risk_score += self.consecutive_losses * 0.2
        
        # High daily trade count increases risk
        daily_trade_ratio = self.daily_trades / self.config.max_daily_trades
        risk_score += daily_trade_ratio * 0.3
        
        # Recent performance
        if len(self.recent_trades) >= 10:
            recent_profits = [trade.get('profit_loss', 0) for trade in list(self.recent_trades)[-10:]]
            if sum(recent_profits) < 0:
                risk_score += 0.3
        
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _reset_daily_counters_if_needed(self):
        """Reset daily counters if date has changed."""
        current_date = time.strftime('%Y-%m-%d')
        if current_date != self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = current_date
            logger.info("Daily counters reset for new trading day")
    
    def _get_no_position_response(self, reason: str) -> Dict[str, Any]:
        """Get standard response for rejected position sizing."""
        return {
            'size_usd': 0,
            'size_pct': 0,
            'approved': False,
            'rejection_reason': reason,
            'consecutive_losses': self.consecutive_losses,
            'daily_trades': self.daily_trades
        }


class ScalpingPerformanceTracker:
    """Tracks and analyzes scalping strategy performance."""
    
    def __init__(self):
        self.trades = deque(maxlen=1000)  # Keep last 1000 trades
        self.signals_generated = 0
        self.signals_executed = 0
        self.start_time = time.time()
        
    def track_signal(self, signal: ScalpingSignal, executed: bool = False):
        """Track signal generation and execution."""
        self.signals_generated += 1
        if executed:
            self.signals_executed += 1
    
    def track_trade(self, trade_result: Dict[str, Any]):
        """Track completed trade result."""
        trade_result['timestamp'] = time.time()
        self.trades.append(trade_result)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.trades:
            return self._get_default_metrics()
        
        trades_list = list(self.trades)
        total_trades = len(trades_list)
        
        # Basic metrics
        profits = [trade.get('profit_loss', 0) for trade in trades_list]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_profit = np.mean(profits) if profits else 0
        total_profit = sum(profits)
        
        # Holding time analysis
        holding_times = [trade.get('holding_time_seconds', 0) for trade in trades_list if trade.get('holding_time_seconds')]
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # Signal type performance
        signal_type_performance = {}
        for trade in trades_list:
            signal_type = trade.get('signal_type', 'unknown')
            if signal_type not in signal_type_performance:
                signal_type_performance[signal_type] = {'count': 0, 'total_profit': 0, 'wins': 0}
            
            signal_type_performance[signal_type]['count'] += 1
            signal_type_performance[signal_type]['total_profit'] += trade.get('profit_loss', 0)
            if trade.get('profit_loss', 0) > 0:
                signal_type_performance[signal_type]['wins'] += 1
        
        # Calculate win rates for each signal type
        for signal_type in signal_type_performance:
            perf = signal_type_performance[signal_type]
            perf['win_rate'] = perf['wins'] / perf['count'] if perf['count'] > 0 else 0
            perf['avg_profit'] = perf['total_profit'] / perf['count'] if perf['count'] > 0 else 0
        
        # Trading frequency
        uptime_hours = (time.time() - self.start_time) / 3600
        trades_per_hour = total_trades / uptime_hours if uptime_hours > 0 else 0
        
        # Recent performance (last 50 trades)
        recent_trades = trades_list[-50:] if len(trades_list) >= 50 else trades_list
        recent_profits = [trade.get('profit_loss', 0) for trade in recent_trades]
        recent_win_rate = len([p for p in recent_profits if p > 0]) / len(recent_profits) if recent_profits else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit,
            'total_profit': total_profit,
            'avg_holding_time_seconds': avg_holding_time,
            'trades_per_hour': trades_per_hour,
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'execution_rate': self.signals_executed / self.signals_generated if self.signals_generated > 0 else 0,
            'signal_type_performance': signal_type_performance,
            'recent_win_rate': recent_win_rate,
            'recent_avg_profit': np.mean(recent_profits) if recent_profits else 0,
            'max_consecutive_wins': self._calculate_max_consecutive_wins(),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(),
            'scalping_efficiency': self._calculate_scalping_efficiency()
        }
    
    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades."""
        if not self.trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade.get('profit_loss', 0) > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades."""
        if not self.trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade.get('profit_loss', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_scalping_efficiency(self) -> float:
        """Calculate scalping efficiency score."""
        if not self.trades:
            return 0.0
        
        trades_list = list(self.trades)
        
        # Factors for efficiency:
        # 1. Win rate (target: >60%)
        # 2. Average profit per trade (positive)
        # 3. Quick execution (short holding times)
        # 4. High trade frequency
        
        profits = [trade.get('profit_loss', 0) for trade in trades_list]
        win_rate = len([p for p in profits if p > 0]) / len(profits)
        avg_profit = np.mean(profits)
        
        holding_times = [trade.get('holding_time_seconds', 0) for trade in trades_list if trade.get('holding_time_seconds')]
        avg_holding_time = np.mean(holding_times) if holding_times else 300  # Default 5 minutes
        
        # Normalize metrics to 0-1 scale
        win_rate_score = min(win_rate / 0.6, 1.0)  # Target 60% win rate
        profit_score = max(min(avg_profit * 1000, 1.0), 0.0)  # Assume target 0.1% per trade
        time_score = max(1.0 - (avg_holding_time / 300), 0.0)  # Prefer under 5 minutes
        
        # Weight the scores
        efficiency = (win_rate_score * 0.4 + profit_score * 0.4 + time_score * 0.2)
        return min(efficiency, 1.0)
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics when no trades exist."""
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'total_profit': 0.0,
            'avg_holding_time_seconds': 0.0,
            'trades_per_hour': 0.0,
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'execution_rate': 0.0,
            'signal_type_performance': {},
            'scalping_efficiency': 0.0
        }


class ScalpingStrategy:
    """
    Main scalping strategy orchestrator that coordinates all components.
    
    This class manages the complete scalping workflow:
    1. Market analysis and signal generation
    2. Risk assessment and position sizing
    3. Trade execution coordination
    4. Performance tracking and optimization
    """
    
    def __init__(self, config: Optional[ScalpingConfig] = None):
        self.config = config or ScalpingConfig()
        
        # Initialize strategy components
        self.signal_generator = ScalpingSignalGenerator(self.config)
        self.risk_manager = ScalpingRiskManager(self.config)
        self.performance_tracker = ScalpingPerformanceTracker()
        
        # Strategy state
        self.state = ScalpingState.SCANNING
        self.active_positions = {}
        self.pending_signals = deque(maxlen=10)
        self.last_analysis_time = 0
        
        logger.info(f"ScalpingStrategy initialized with config: {self.config}")
    
    async def analyze_and_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main strategy entry point for market analysis and signal generation.
        
        Args:
            market_data: Dictionary containing OHLCV data, current price, timestamp, etc.
            
        Returns:
            Dictionary containing strategy analysis results and signals
        """
        start_time = time.perf_counter()
        
        try:
            # Extract data
            ohlcv_data = market_data.get('ohlcv')
            current_price = market_data.get('current_price', 0)
            timestamp = market_data.get('timestamp', time.time())
            
            if ohlcv_data is None or len(ohlcv_data) < self.config.min_data_points:
                return {
                    'strategy_type': 'scalping',
                    'action': 'HOLD',
                    'reason': 'insufficient_data',
                    'signals': [],
                    'market_analysis': {},
                    'performance_metrics': self.performance_tracker.get_performance_metrics()
                }
            
            # Update strategy state
            self._update_strategy_state()
            
            # Generate scalping signals
            signals = self.signal_generator.generate_signals(ohlcv_data)
            
            # Track signal generation
            for signal in signals:
                self.performance_tracker.track_signal(signal)
            
            # Filter signals through risk management
            approved_signals = []
            for signal in signals:
                validation = self.risk_manager.validate_signal_execution(signal)
                if validation['approved']:
                    approved_signals.append(signal)
                    self.performance_tracker.track_signal(signal, executed=True)
            
            # Select best signal for execution
            execution_signal = None
            if approved_signals:
                # Take highest priority signal
                execution_signal = approved_signals[0]
            
            # Generate strategy response
            response = self._generate_strategy_response(
                execution_signal, signals, market_data, start_time
            )
            
            self.last_analysis_time = time.time()
            return response
            
        except Exception as e:
            logger.error(f"Scalping strategy analysis error: {e}")
            return self._get_error_response(str(e))
    
    def update_position_status(self, position_update: Dict[str, Any]):
        """
        Update strategy with current position status.
        
        Args:
            position_update: Dictionary containing position information
        """
        try:
            symbol = position_update.get('symbol')
            if symbol:
                self.active_positions[symbol] = position_update
            
            # Update state based on positions
            if self.active_positions:
                if self.state == ScalpingState.SCANNING:
                    self.state = ScalpingState.MANAGING
            else:
                if self.state == ScalpingState.MANAGING:
                    self.state = ScalpingState.SCANNING
                    
        except Exception as e:
            logger.error(f"Position status update error: {e}")
    
    def update_trade_result(self, trade_result: Dict[str, Any]):
        """
        Update strategy with completed trade result.
        
        Args:
            trade_result: Dictionary containing trade outcome
        """
        try:
            # Update risk manager
            self.risk_manager.update_trade_result(trade_result)
            
            # Update performance tracker
            self.performance_tracker.track_trade(trade_result)
            
            # Clean up active positions
            symbol = trade_result.get('symbol')
            if symbol and symbol in self.active_positions:
                del self.active_positions[symbol]
            
            logger.info(f"Trade result updated: {trade_result.get('symbol')} "
                       f"P&L: {trade_result.get('profit_loss', 0):.4f}")
                       
        except Exception as e:
            logger.error(f"Trade result update error: {e}")
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and metrics."""
        return {
            'strategy_type': 'scalping',
            'state': self.state.value,
            'config': {
                'min_profit_target_pct': self.config.min_profit_target_pct,
                'max_profit_target_pct': self.config.max_profit_target_pct,
                'stop_loss_pct': self.config.stop_loss_pct,
                'max_holding_time': self.config.max_holding_time,
                'max_daily_trades': self.config.max_daily_trades
            },
            'active_positions': len(self.active_positions),
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'performance_metrics': self.performance_tracker.get_performance_metrics(),
            'last_analysis_time': self.last_analysis_time
        }
    
    def _update_strategy_state(self):
        """Update strategy operational state."""
        # Simple state management based on positions and risk
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        if risk_metrics.get('cooldown_active', False):
            self.state = ScalpingState.COOLDOWN
        elif len(self.active_positions) > 0:
            self.state = ScalpingState.MANAGING
        else:
            self.state = ScalpingState.SCANNING
    
    def _generate_strategy_response(self, execution_signal: Optional[ScalpingSignal],
                                   all_signals: List[ScalpingSignal],
                                   market_data: Dict[str, Any],
                                   start_time: float) -> Dict[str, Any]:
        """Generate comprehensive strategy response."""
        
        # Default to HOLD
        action = "HOLD"
        size_pct = 0
        take_profit_pct = 0.0
        stop_loss_pct = 0.0
        rationale = "No qualifying scalping signals"
        
        # If we have an execution signal, prepare trade action
        if execution_signal:
            # Get position sizing
            account_balance = market_data.get('account_balance', 10000)  # Default for testing
            sizing_info = self.risk_manager.calculate_position_size(execution_signal, account_balance)
            
            if sizing_info.get('approved', False):
                action = execution_signal.direction.upper()
                if action == 'BUY':
                    action = 'LONG'
                elif action == 'SELL':
                    action = 'SHORT'
                
                size_pct = min(int(sizing_info.get('size_pct', 0)), 100)
                take_profit_pct = execution_signal.target_profit_pct * 100  # Convert to percentage
                stop_loss_pct = execution_signal.stop_loss_pct * 100
                
                rationale = f"{execution_signal.signal_type.value}: {', '.join(execution_signal.entry_reasons[:2])}"
                rationale = rationale[:200]  # Truncate to fit TradeAction constraint
            else:
                rationale = f"Signal rejected: {sizing_info.get('rejection_reason', 'unknown')}"
        
        # Calculate analysis performance
        analysis_time = time.perf_counter() - start_time
        
        return {
            'strategy_type': 'scalping',
            'action': action,
            'size_pct': size_pct,
            'take_profit_pct': take_profit_pct,
            'stop_loss_pct': stop_loss_pct,
            'rationale': rationale,
            'execution_speed': 'high_frequency',
            'signals': [self._serialize_signal(s) for s in all_signals],
            'execution_signal': self._serialize_signal(execution_signal) if execution_signal else None,
            'market_analysis': {
                'signal_count': len(all_signals),
                'approved_signal_count': len([s for s in all_signals if self.risk_manager.validate_signal_execution(s)['approved']]),
                'analysis_time_ms': analysis_time * 1000,
                'strategy_state': self.state.value
            },
            'risk_assessment': self.risk_manager.get_risk_metrics(),
            'performance_metrics': self.performance_tracker.get_performance_metrics()
        }
    
    def _serialize_signal(self, signal: ScalpingSignal) -> Dict[str, Any]:
        """Convert ScalpingSignal to serializable dictionary."""
        if signal is None:
            return None
            
        return {
            'type': signal.signal_type.value,
            'direction': signal.direction,
            'confidence': signal.confidence,
            'timing': signal.timing.value,
            'target_profit_pct': signal.target_profit_pct,
            'stop_loss_pct': signal.stop_loss_pct,
            'max_holding_time': signal.max_holding_time,
            'current_price': signal.current_price,
            'entry_reasons': signal.entry_reasons,
            'risk_factors': signal.risk_factors,
            'priority_score': signal.priority_score,
            'timestamp': signal.timestamp,
            'expires_at': signal.expires_at
        }
    
    def _get_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response for strategy failures."""
        return {
            'strategy_type': 'scalping',
            'action': 'HOLD',
            'size_pct': 0,
            'take_profit_pct': 0.0,
            'stop_loss_pct': 0.0,
            'rationale': f"Strategy error: {error_message[:150]}",
            'signals': [],
            'error': error_message,
            'market_analysis': {},
            'performance_metrics': self.performance_tracker.get_performance_metrics()
        }


# Factory function for creating scalping strategy instances
def create_scalping_strategy(config_overrides: Optional[Dict[str, Any]] = None) -> ScalpingStrategy:
    """
    Factory function to create ScalpingStrategy with optional configuration overrides.
    
    Args:
        config_overrides: Optional dictionary of configuration overrides
        
    Returns:
        Configured ScalpingStrategy instance
    """
    config = ScalpingConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return ScalpingStrategy(config)


# Export main classes and functions
__all__ = [
    'ScalpingStrategy',
    'ScalpingConfig',
    'ScalpingSignalType',
    'ScalpingTiming',
    'ScalpingState',
    'ScalpingSignal',
    'create_scalping_strategy'
]