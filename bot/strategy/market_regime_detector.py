"""
Market Regime Detection System

Continuously analyzes market conditions to determine optimal trading strategy.
Identifies trending, ranging, breakout, and consolidation market conditions.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_BULLISH = "trending_bullish"      # Strong uptrend + volume
    TRENDING_BEARISH = "trending_bearish"      # Strong downtrend + volume  
    RANGING_HIGH_VOL = "ranging_high_vol"      # Sideways + high volume
    RANGING_LOW_VOL = "ranging_low_vol"        # Sideways + low volume
    BREAKOUT_BULLISH = "breakout_bullish"      # Breaking resistance + volume
    BREAKOUT_BEARISH = "breakout_bearish"      # Breaking support + volume
    CONSOLIDATION = "consolidation"            # Low volatility + low volume
    UNCERTAIN = "uncertain"                    # Mixed/conflicting signals


@dataclass
class RegimeAnalysis:
    """Container for regime analysis results"""
    regime: MarketRegime
    confidence: float
    stability: float
    duration_seconds: float
    previous_regime: Optional[MarketRegime]
    analysis_details: Dict[str, Any]
    strategy_recommendation: Dict[str, Any]
    risk_factors: List[str]


class TrendAnalyzer:
    """Analyzes trend direction and strength"""
    
    def __init__(self):
        self.ema_fast = 8    # Fast trend detection
        self.ema_slow = 21   # Slow trend confirmation
        self.adx_period = 14 # Trend strength
        
    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        try:
            # Calculate EMAs
            ema_fast = data['close'].ewm(span=self.ema_fast).mean()
            ema_slow = data['close'].ewm(span=self.ema_slow).mean()
            
            # Trend direction
            trend_direction = self._calculate_trend_direction(ema_fast, ema_slow)
            
            # ADX for trend strength
            adx_strength = self._calculate_adx(data)
            
            # Momentum calculation
            momentum = self._calculate_momentum(data)
            
            # Trend consistency
            consistency = self._calculate_trend_consistency(ema_fast, ema_slow)
            
            return {
                'direction': trend_direction,
                'strength': adx_strength,
                'momentum': momentum,
                'consistency': consistency,
                'ema_fast': ema_fast.iloc[-1] if not ema_fast.empty else 0,
                'ema_slow': ema_slow.iloc[-1] if not ema_slow.empty else 0
            }
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {
                'direction': 'sideways',
                'strength': 0.0,
                'momentum': 0.0,
                'consistency': 0.0,
                'ema_fast': 0,
                'ema_slow': 0
            }
    
    def _calculate_trend_direction(self, ema_fast: pd.Series, ema_slow: pd.Series) -> str:
        """Calculate trend direction based on EMA relationship"""
        if ema_fast.empty or ema_slow.empty:
            return 'sideways'
            
        fast_val = ema_fast.iloc[-1]
        slow_val = ema_slow.iloc[-1]
        
        if fast_val > slow_val * 1.002:  # 0.2% threshold
            return 'bullish'
        elif fast_val < slow_val * 0.998:
            return 'bearish'
        else:
            return 'sideways'
    
    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """Calculate Average Directional Index for trend strength"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                              np.maximum(high - high.shift(1), 0), 0)
            dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                               np.maximum(low.shift(1) - low, 0), 0)
            
            # Smooth the values
            tr_smooth = pd.Series(tr).rolling(window=self.adx_period).mean()
            dm_plus_smooth = pd.Series(dm_plus).rolling(window=self.adx_period).mean()
            dm_minus_smooth = pd.Series(dm_minus).rolling(window=self.adx_period).mean()
            
            # Calculate DI+ and DI-
            di_plus = 100 * dm_plus_smooth / tr_smooth
            di_minus = 100 * dm_minus_smooth / tr_smooth
            
            # Calculate ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=self.adx_period).mean()
            
            return min(max(adx.iloc[-1] / 100.0 if not adx.empty else 0.0, 0.0), 1.0)
        
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 0.0
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum"""
        try:
            close_prices = data['close']
            if len(close_prices) < 10:
                return 0.0
                
            # Rate of change over 10 periods
            roc = (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]
            return np.clip(roc * 10, -1.0, 1.0)  # Normalize to [-1, 1]
        
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def _calculate_trend_consistency(self, ema_fast: pd.Series, ema_slow: pd.Series) -> float:
        """Calculate how consistent the trend has been"""
        try:
            if len(ema_fast) < 10 or len(ema_slow) < 10:
                return 0.0
                
            # Check consistency over last 10 periods
            fast_vals = ema_fast.tail(10)
            slow_vals = ema_slow.tail(10)
            
            consistent_periods = 0
            for i in range(1, len(fast_vals)):
                if (fast_vals.iloc[i] > slow_vals.iloc[i]) == (fast_vals.iloc[i-1] > slow_vals.iloc[i-1]):
                    consistent_periods += 1
            
            return consistent_periods / 9.0  # 9 comparisons for 10 periods
        
        except Exception as e:
            logger.error(f"Error calculating trend consistency: {e}")
            return 0.0


class VolatilityAnalyzer:
    """Analyzes market volatility"""
    
    def __init__(self):
        self.atr_period = 14
        self.bb_period = 20
        self.bb_std = 2.0
        
    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility metrics"""
        try:
            # Average True Range
            atr = self._calculate_atr(data)
            
            # Bollinger Bands
            bb_metrics = self._calculate_bollinger_bands(data)
            
            # Price volatility (standard deviation)
            price_volatility = self._calculate_price_volatility(data)
            
            # Volatility trend
            volatility_trend = self._calculate_volatility_trend(data)
            
            return {
                'level': atr,
                'trend': volatility_trend,
                'percentile': bb_metrics['percentile'],
                'bb_width': bb_metrics['width'],
                'price_volatility': price_volatility
            }
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return {
                'level': 0.0,
                'trend': 0.0,
                'percentile': 0.5,
                'bb_width': 0.0,
                'price_volatility': 0.0
            }
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range normalized"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=self.atr_period).mean()
            
            # Normalize by current price
            current_price = close.iloc[-1]
            normalized_atr = atr.iloc[-1] / current_price if current_price > 0 else 0
            
            return min(normalized_atr * 100, 10.0) / 10.0  # Scale to [0, 1]
        
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Bollinger Bands metrics"""
        try:
            close = data['close']
            
            # Calculate bands
            sma = close.rolling(window=self.bb_period).mean()
            std = close.rolling(window=self.bb_period).std()
            
            upper_band = sma + (std * self.bb_std)
            lower_band = sma - (std * self.bb_std)
            
            # Current position within bands
            current_price = close.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            if current_upper != current_lower:
                percentile = (current_price - current_lower) / (current_upper - current_lower)
            else:
                percentile = 0.5
                
            # Band width (volatility measure)
            width = (current_upper - current_lower) / sma.iloc[-1] if sma.iloc[-1] > 0 else 0
            
            return {
                'percentile': np.clip(percentile, 0.0, 1.0),
                'width': min(width, 1.0)
            }
        
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {'percentile': 0.5, 'width': 0.0}
    
    def _calculate_price_volatility(self, data: pd.DataFrame) -> float:
        """Calculate price volatility over recent periods"""
        try:
            close_prices = data['close']
            if len(close_prices) < 20:
                return 0.0
                
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            
            # Standard deviation of returns
            volatility = returns.tail(20).std()
            
            return min(volatility * 100, 10.0) / 10.0  # Normalize to [0, 1]
        
        except Exception as e:
            logger.error(f"Error calculating price volatility: {e}")
            return 0.0
    
    def _calculate_volatility_trend(self, data: pd.DataFrame) -> float:
        """Calculate if volatility is increasing or decreasing"""
        try:
            close_prices = data['close']
            if len(close_prices) < 40:
                return 0.0
                
            # Calculate rolling volatility
            returns = close_prices.pct_change().dropna()
            vol_short = returns.tail(10).std()
            vol_long = returns.tail(30).std()
            
            if vol_long > 0:
                trend = (vol_short - vol_long) / vol_long
                return np.clip(trend, -1.0, 1.0)
            
            return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating volatility trend: {e}")
            return 0.0


class VolumeAnalyzer:
    """Analyzes trading volume patterns"""
    
    def __init__(self):
        self.volume_ma_periods = [10, 20, 50]
        self.volume_spike_threshold = 2.0
        
    def analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume metrics"""
        try:
            # Relative volume
            relative_volume = self._calculate_relative_volume(data)
            
            # Volume trend
            volume_trend = self._calculate_volume_trend(data)
            
            # Volume spikes
            spike_detected = self._detect_volume_spikes(data)
            
            # Volume profile
            volume_profile = self._calculate_volume_profile(data)
            
            return {
                'relative': relative_volume,
                'trend': volume_trend,
                'spike_detected': spike_detected,
                'profile': volume_profile
            }
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {
                'relative': 1.0,
                'trend': 0.0,
                'spike_detected': False,
                'profile': 'normal'
            }
    
    def _calculate_relative_volume(self, data: pd.DataFrame) -> float:
        """Calculate current volume relative to average"""
        try:
            if 'volume' not in data.columns or data['volume'].empty:
                return 1.0
                
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].tail(20).mean()
            
            if avg_volume > 0:
                return current_volume / avg_volume
            return 1.0
        
        except Exception as e:
            logger.error(f"Error calculating relative volume: {e}")
            return 1.0
    
    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """Calculate volume trend direction"""
        try:
            if 'volume' not in data.columns or len(data['volume']) < 20:
                return 0.0
                
            volume = data['volume']
            vol_ma_short = volume.tail(5).mean()
            vol_ma_long = volume.tail(15).mean()
            
            if vol_ma_long > 0:
                trend = (vol_ma_short - vol_ma_long) / vol_ma_long
                return np.clip(trend, -1.0, 1.0)
            
            return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return 0.0
    
    def _detect_volume_spikes(self, data: pd.DataFrame) -> bool:
        """Detect unusual volume spikes"""
        try:
            if 'volume' not in data.columns or len(data['volume']) < 20:
                return False
                
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].tail(20).mean()
            
            return current_volume > (avg_volume * self.volume_spike_threshold)
        
        except Exception as e:
            logger.error(f"Error detecting volume spikes: {e}")
            return False
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> str:
        """Categorize volume profile"""
        try:
            relative_vol = self._calculate_relative_volume(data)
            
            if relative_vol > 2.0:
                return 'very_high'
            elif relative_vol > 1.5:
                return 'high'
            elif relative_vol > 0.8:
                return 'normal'
            elif relative_vol > 0.5:
                return 'low'
            else:
                return 'very_low'
        
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return 'normal'


class MicrostructureAnalyzer:
    """Analyzes market microstructure"""
    
    def __init__(self):
        self.support_resistance_periods = [20, 50, 100]
        self.range_detection_threshold = 0.02
        
    def analyze_microstructure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support/resistance and range detection"""
        try:
            # Support and resistance levels
            levels = self._calculate_support_resistance(data)
            
            # Range-bound detection
            range_bound = self._detect_range_bound_market(data)
            
            # Breakout potential
            breakout_potential = self._calculate_breakout_potential(data, levels)
            
            return {
                'support': levels['support'],
                'resistance': levels['resistance'],
                'range_bound': range_bound,
                'breakout_potential': breakout_potential,
                'key_levels': levels['key_levels']
            }
        except Exception as e:
            logger.error(f"Error in microstructure analysis: {e}")
            return {
                'support': 0.0,
                'resistance': 0.0,
                'range_bound': 0.0,
                'breakout_potential': 0.0,
                'key_levels': []
            }
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate dynamic support and resistance levels"""
        try:
            high_prices = data['high']
            low_prices = data['low']
            
            # Calculate pivot points over different periods
            support_levels = []
            resistance_levels = []
            
            for period in self.support_resistance_periods:
                if len(data) >= period:
                    # Rolling max/min for resistance/support
                    resistance = high_prices.tail(period).max()
                    support = low_prices.tail(period).min()
                    
                    resistance_levels.append(resistance)
                    support_levels.append(support)
            
            # Use the most recent significant levels
            current_resistance = max(resistance_levels) if resistance_levels else high_prices.iloc[-1]
            current_support = min(support_levels) if support_levels else low_prices.iloc[-1]
            
            # Key levels (significant price points)
            key_levels = self._identify_key_levels(data)
            
            return {
                'support': current_support,
                'resistance': current_resistance,
                'key_levels': key_levels
            }
        
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {
                'support': 0.0,
                'resistance': 0.0,
                'key_levels': []
            }
    
    def _detect_range_bound_market(self, data: pd.DataFrame) -> float:
        """Detect if market is range-bound"""
        try:
            if len(data) < 50:
                return 0.0
                
            close_prices = data['close'].tail(50)
            high_prices = data['high'].tail(50)
            low_prices = data['low'].tail(50)
            
            # Calculate range metrics
            price_range = high_prices.max() - low_prices.min()
            avg_price = close_prices.mean()
            
            if avg_price > 0:
                range_ratio = price_range / avg_price
                
                # Check if price is staying within a tight range
                recent_range = close_prices.tail(20).max() - close_prices.tail(20).min()
                recent_range_ratio = recent_range / avg_price if avg_price > 0 else 0
                
                # Range-bound if recent range is small relative to overall range
                range_bound_score = 1.0 - min(recent_range_ratio / self.range_detection_threshold, 1.0)
                
                return max(0.0, range_bound_score)
            
            return 0.0
        
        except Exception as e:
            logger.error(f"Error detecting range-bound market: {e}")
            return 0.0
    
    def _calculate_breakout_potential(self, data: pd.DataFrame, levels: Dict[str, float]) -> float:
        """Calculate potential for breakout"""
        try:
            current_price = data['close'].iloc[-1]
            support = levels['support']
            resistance = levels['resistance']
            
            if resistance > support and resistance > 0 and support > 0:
                # Distance to key levels
                dist_to_resistance = abs(current_price - resistance) / resistance
                dist_to_support = abs(current_price - support) / support
                
                # Closer to levels = higher breakout potential
                min_distance = min(dist_to_resistance, dist_to_support)
                breakout_potential = max(0.0, 1.0 - (min_distance / 0.02))  # 2% threshold
                
                return min(breakout_potential, 1.0)
            
            return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating breakout potential: {e}")
            return 0.0
    
    def _identify_key_levels(self, data: pd.DataFrame) -> List[float]:
        """Identify key price levels"""
        try:
            # Simple approach: recent highs and lows
            recent_data = data.tail(100)
            
            key_levels = []
            
            # Add recent significant highs and lows
            if not recent_data.empty:
                key_levels.extend([
                    recent_data['high'].max(),
                    recent_data['low'].min(),
                    recent_data['close'].iloc[-1]  # Current price
                ])
            
            return sorted(list(set(key_levels)))  # Remove duplicates and sort
        
        except Exception as e:
            logger.error(f"Error identifying key levels: {e}")
            return []


class AdaptiveThresholds:
    """Dynamically adjusts thresholds based on market behavior"""
    
    def __init__(self):
        self.volatility_percentiles = deque(maxlen=1000)
        self.volume_percentiles = deque(maxlen=1000)
        self.trend_strengths = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    def update_thresholds(self, market_data: Dict[str, Any]):
        """Update adaptive thresholds with new market data"""
        try:
            with self._lock:
                if 'volatility' in market_data:
                    self.volatility_percentiles.append(market_data['volatility']['level'])
                
                if 'volume' in market_data:
                    self.volume_percentiles.append(market_data['volume']['relative'])
                
                if 'trend' in market_data:
                    self.trend_strengths.append(market_data['trend']['strength'])
                    
        except Exception as e:
            logger.error(f"Error updating adaptive thresholds: {e}")
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds"""
        try:
            with self._lock:
                thresholds = {
                    'high_volatility': 0.7,  # Default values
                    'low_volatility': 0.3,
                    'high_volume': 1.5,
                    'low_volume': 0.7,
                    'strong_trend': 0.6,
                    'weak_trend': 0.3
                }
                
                # Calculate percentile-based thresholds if we have enough data
                if len(self.volatility_percentiles) > 50:
                    vol_data = list(self.volatility_percentiles)
                    thresholds['high_volatility'] = np.percentile(vol_data, 75)
                    thresholds['low_volatility'] = np.percentile(vol_data, 25)
                
                if len(self.volume_percentiles) > 50:
                    vol_data = list(self.volume_percentiles)
                    thresholds['high_volume'] = np.percentile(vol_data, 70)
                    thresholds['low_volume'] = np.percentile(vol_data, 30)
                
                if len(self.trend_strengths) > 50:
                    trend_data = list(self.trend_strengths)
                    thresholds['strong_trend'] = np.percentile(trend_data, 65)
                    thresholds['weak_trend'] = np.percentile(trend_data, 35)
                
                return thresholds
                
        except Exception as e:
            logger.error(f"Error getting current thresholds: {e}")
            return {
                'high_volatility': 0.7,
                'low_volatility': 0.3,
                'high_volume': 1.5,
                'low_volume': 0.7,
                'strong_trend': 0.6,
                'weak_trend': 0.3
            }


class RegimePerformanceTracker:
    """Tracks regime detection accuracy and performance"""
    
    def __init__(self):
        self.regime_predictions = deque(maxlen=1000)
        self.actual_outcomes = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    def track_prediction(self, regime: MarketRegime, confidence: float):
        """Track a regime prediction"""
        try:
            with self._lock:
                self.regime_predictions.append({
                    'regime': regime,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })
        except Exception as e:
            logger.error(f"Error tracking prediction: {e}")
    
    def track_outcome(self, actual_regime: MarketRegime):
        """Track actual market outcome"""
        try:
            with self._lock:
                self.actual_outcomes.append({
                    'regime': actual_regime,
                    'timestamp': datetime.now()
                })
        except Exception as e:
            logger.error(f"Error tracking outcome: {e}")
    
    def calculate_accuracy(self) -> Dict[str, float]:
        """Calculate regime detection accuracy metrics"""
        try:
            with self._lock:
                if len(self.regime_predictions) < 10:
                    return {'overall_accuracy': 0.0}
                
                # Simple accuracy calculation
                correct_predictions = 0
                total_predictions = min(len(self.regime_predictions), len(self.actual_outcomes))
                
                for i in range(total_predictions):
                    if self.regime_predictions[i]['regime'] == self.actual_outcomes[i]['regime']:
                        correct_predictions += 1
                
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                
                return {
                    'overall_accuracy': accuracy,
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions
                }
                
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return {'overall_accuracy': 0.0}


class RealtimeRegimeMonitor:
    """Monitors market regime in real-time"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=100)
        self.regime_stability_threshold = 0.8
        self.min_regime_duration = 60  # seconds
        self.current_regime_start = datetime.now()
        self._lock = threading.Lock()
        
    async def monitor_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and analyze current market regime"""
        try:
            # Analyze current market state
            current_analysis = await self._analyze_current_state(market_data)
            
            # Detect regime
            regime, confidence = self._detect_regime(current_analysis)
            
            # Apply stability filter
            stable_regime = self._apply_stability_filter(regime, confidence)
            
            # Update history
            self._update_regime_history(stable_regime, confidence)
            
            # Calculate metrics
            stability = self._calculate_stability()
            duration = self._calculate_regime_duration()
            
            return {
                'current_regime': stable_regime,
                'confidence': confidence,
                'stability': stability,
                'duration': duration,
                'analysis': current_analysis
            }
            
        except Exception as e:
            logger.error(f"Error monitoring regime: {e}")
            return {
                'current_regime': MarketRegime.UNCERTAIN,
                'confidence': 0.0,
                'stability': 0.0,
                'duration': 0.0,
                'analysis': {}
            }
    
    async def _analyze_current_state(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market state using all analyzers"""
        # This would be called with the market data from your existing system
        # For now, return a basic analysis structure
        return {
            'trend': market_data.get('trend', {}),
            'volatility': market_data.get('volatility', {}),
            'volume': market_data.get('volume', {}),
            'microstructure': market_data.get('microstructure', {}),
            'price': market_data.get('price', {})
        }
    
    def _detect_regime(self, analysis: Dict[str, Any]) -> Tuple[MarketRegime, float]:
        """Detect current market regime"""
        try:
            # Get trend information
            trend_data = analysis.get('trend', {})
            trend_direction = trend_data.get('direction', 'sideways')
            trend_strength = trend_data.get('strength', 0.0)
            
            # Get volatility information
            volatility_data = analysis.get('volatility', {})
            volatility_level = volatility_data.get('level', 0.0)
            
            # Get volume information
            volume_data = analysis.get('volume', {})
            volume_relative = volume_data.get('relative', 1.0)
            volume_spike = volume_data.get('spike_detected', False)
            
            # Get microstructure information
            micro_data = analysis.get('microstructure', {})
            range_bound = micro_data.get('range_bound', 0.0)
            breakout_potential = micro_data.get('breakout_potential', 0.0)
            
            # Get price information
            price_data = analysis.get('price', {})
            current_price = price_data.get('current', 0.0)
            resistance = micro_data.get('resistance', 0.0)
            support = micro_data.get('support', 0.0)
            
            # Detection logic
            confidence = 0.5
            
            # Check for breakouts first
            if breakout_potential > 0.7 and volume_spike:
                if current_price > resistance and trend_direction == 'bullish':
                    return MarketRegime.BREAKOUT_BULLISH, 0.85
                elif current_price < support and trend_direction == 'bearish':
                    return MarketRegime.BREAKOUT_BEARISH, 0.85
            
            # Check for trending markets
            if trend_strength > 0.6 and volume_relative > 1.2:
                if trend_direction == 'bullish':
                    return MarketRegime.TRENDING_BULLISH, 0.8
                elif trend_direction == 'bearish':
                    return MarketRegime.TRENDING_BEARISH, 0.8
            
            # Check for ranging markets
            if range_bound > 0.7 and trend_strength < 0.4:
                if volume_relative > 1.0:
                    return MarketRegime.RANGING_HIGH_VOL, 0.75
                else:
                    return MarketRegime.RANGING_LOW_VOL, 0.75
            
            # Check for consolidation
            if volatility_level < 0.3 and volume_relative < 0.8:
                return MarketRegime.CONSOLIDATION, 0.7
            
            # Default to uncertain
            return MarketRegime.UNCERTAIN, 0.3
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return MarketRegime.UNCERTAIN, 0.0
    
    def _apply_stability_filter(self, regime: MarketRegime, confidence: float) -> MarketRegime:
        """Apply stability filter to prevent rapid regime changes"""
        try:
            with self._lock:
                if not self.regime_history:
                    return regime
                
                recent_regime = self.regime_history[-1]['regime']
                
                # If same regime, return it
                if regime == recent_regime:
                    return regime
                
                # Check if we should stick with current regime for stability
                regime_duration = (datetime.now() - self.current_regime_start).total_seconds()
                
                if regime_duration < self.min_regime_duration and confidence < self.regime_stability_threshold:
                    return recent_regime
                
                # New regime confirmed
                self.current_regime_start = datetime.now()
                return regime
                
        except Exception as e:
            logger.error(f"Error applying stability filter: {e}")
            return regime
    
    def _update_regime_history(self, regime: MarketRegime, confidence: float):
        """Update regime history"""
        try:
            with self._lock:
                self.regime_history.append({
                    'regime': regime,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })
        except Exception as e:
            logger.error(f"Error updating regime history: {e}")
    
    def _calculate_stability(self) -> float:
        """Calculate regime stability score"""
        try:
            with self._lock:
                if len(self.regime_history) < 10:
                    return 0.5
                
                recent_regimes = [entry['regime'] for entry in list(self.regime_history)[-10:]]
                most_common = max(set(recent_regimes), key=recent_regimes.count)
                stability = recent_regimes.count(most_common) / len(recent_regimes)
                
                return stability
                
        except Exception as e:
            logger.error(f"Error calculating stability: {e}")
            return 0.5
    
    def _calculate_regime_duration(self) -> float:
        """Calculate current regime duration in seconds"""
        try:
            return (datetime.now() - self.current_regime_start).total_seconds()
        except Exception as e:
            logger.error(f"Error calculating regime duration: {e}")
            return 0.0


class MarketRegimeDetector:
    """Main market regime detection system"""
    
    # Strategy mapping
    REGIME_STRATEGY_MAP = {
        MarketRegime.TRENDING_BULLISH: 'momentum',
        MarketRegime.TRENDING_BEARISH: 'momentum', 
        MarketRegime.RANGING_HIGH_VOL: 'momentum',    # Can capture range bounces
        MarketRegime.RANGING_LOW_VOL: 'scalping',     # Best for small moves
        MarketRegime.BREAKOUT_BULLISH: 'momentum',    # Capture big moves
        MarketRegime.BREAKOUT_BEARISH: 'momentum',    # Capture big moves
        MarketRegime.CONSOLIDATION: 'scalping',       # Small, frequent moves
        MarketRegime.UNCERTAIN: 'hold'                # Avoid trading
    }
    
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.adaptive_thresholds = AdaptiveThresholds()
        self.performance_tracker = RegimePerformanceTracker()
        self.realtime_monitor = RealtimeRegimeMonitor()
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def analyze_market_regime(self, data: pd.DataFrame) -> RegimeAnalysis:
        """Analyze current market regime with full analysis"""
        try:
            # Run all analyzers in parallel
            loop = asyncio.get_event_loop()
            
            trend_task = loop.run_in_executor(self.executor, self.trend_analyzer.analyze_trend, data)
            volatility_task = loop.run_in_executor(self.executor, self.volatility_analyzer.analyze_volatility, data)
            volume_task = loop.run_in_executor(self.executor, self.volume_analyzer.analyze_volume, data)
            microstructure_task = loop.run_in_executor(self.executor, self.microstructure_analyzer.analyze_microstructure, data)
            
            # Wait for all analyses to complete
            trend_analysis = await trend_task
            volatility_analysis = await volatility_task
            volume_analysis = await volume_task
            microstructure_analysis = await microstructure_task
            
            # Combine all analyses
            market_data = {
                'trend': trend_analysis,
                'volatility': volatility_analysis,
                'volume': volume_analysis,
                'microstructure': microstructure_analysis,
                'price': {
                    'current': data['close'].iloc[-1] if not data.empty else 0.0,
                    'high': data['high'].iloc[-1] if not data.empty else 0.0,
                    'low': data['low'].iloc[-1] if not data.empty else 0.0
                }
            }
            
            # Update adaptive thresholds
            self.adaptive_thresholds.update_thresholds(market_data)
            
            # Monitor regime
            regime_result = await self.realtime_monitor.monitor_regime(market_data)
            
            # Get strategy recommendation
            strategy_rec = self.get_strategy_recommendation(regime_result['current_regime'])
            
            # Assess risk factors
            risk_factors = self._assess_risk_factors(market_data, regime_result['current_regime'])
            
            # Track prediction
            self.performance_tracker.track_prediction(regime_result['current_regime'], regime_result['confidence'])
            
            return RegimeAnalysis(
                regime=regime_result['current_regime'],
                confidence=regime_result['confidence'],
                stability=regime_result['stability'],
                duration_seconds=regime_result['duration'],
                previous_regime=self._get_previous_regime(),
                analysis_details=market_data,
                strategy_recommendation=strategy_rec,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return RegimeAnalysis(
                regime=MarketRegime.UNCERTAIN,
                confidence=0.0,
                stability=0.0,
                duration_seconds=0.0,
                previous_regime=None,
                analysis_details={},
                strategy_recommendation={'strategy': 'hold', 'confidence': 0.0},
                risk_factors=['Analysis error occurred']
            )
    
    def get_strategy_recommendation(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get strategy recommendation for current regime"""
        try:
            strategy = self.REGIME_STRATEGY_MAP.get(regime, 'hold')
            parameters = self.get_strategy_parameters(regime)
            
            # Calculate confidence based on regime
            confidence_map = {
                MarketRegime.TRENDING_BULLISH: 0.9,
                MarketRegime.TRENDING_BEARISH: 0.9,
                MarketRegime.BREAKOUT_BULLISH: 0.85,
                MarketRegime.BREAKOUT_BEARISH: 0.85,
                MarketRegime.RANGING_HIGH_VOL: 0.75,
                MarketRegime.RANGING_LOW_VOL: 0.7,
                MarketRegime.CONSOLIDATION: 0.6,
                MarketRegime.UNCERTAIN: 0.2
            }
            
            confidence = confidence_map.get(regime, 0.5)
            
            # Generate reasoning
            reasoning = self._generate_strategy_reasoning(regime)
            
            return {
                'strategy': strategy,
                'confidence': confidence,
                'parameters': parameters,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy recommendation: {e}")
            return {
                'strategy': 'hold',
                'confidence': 0.0,
                'parameters': {},
                'reasoning': 'Error in strategy recommendation'
            }
    
    def get_strategy_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """Return optimal parameters for current regime"""
        try:
            if regime in [MarketRegime.TRENDING_BULLISH, MarketRegime.TRENDING_BEARISH]:
                return {
                    'timeframe': '1m',
                    'position_size_multiplier': 1.5,
                    'stop_loss_pct': 0.8,
                    'take_profit_pct': 2.0,
                    'max_holding_time': 1800,  # 30 minutes
                    'entry_confidence_threshold': 0.7
                }
            elif regime in [MarketRegime.BREAKOUT_BULLISH, MarketRegime.BREAKOUT_BEARISH]:
                return {
                    'timeframe': '1m',
                    'position_size_multiplier': 2.0,
                    'stop_loss_pct': 1.0,
                    'take_profit_pct': 3.0,
                    'max_holding_time': 2400,  # 40 minutes
                    'entry_confidence_threshold': 0.8
                }
            elif regime == MarketRegime.RANGING_HIGH_VOL:
                return {
                    'timeframe': '1m',
                    'position_size_multiplier': 1.2,
                    'stop_loss_pct': 0.5,
                    'take_profit_pct': 1.0,
                    'max_holding_time': 900,   # 15 minutes
                    'entry_confidence_threshold': 0.6
                }
            elif regime in [MarketRegime.RANGING_LOW_VOL, MarketRegime.CONSOLIDATION]:
                return {
                    'timeframe': '15s',
                    'position_size_multiplier': 0.8,
                    'stop_loss_pct': 0.3,
                    'take_profit_pct': 0.6,
                    'max_holding_time': 300,   # 5 minutes
                    'entry_confidence_threshold': 0.5
                }
            else:  # UNCERTAIN
                return {
                    'timeframe': '1m',
                    'position_size_multiplier': 0.5,
                    'stop_loss_pct': 0.4,
                    'take_profit_pct': 0.8,
                    'max_holding_time': 600,   # 10 minutes
                    'entry_confidence_threshold': 0.8
                }
                
        except Exception as e:
            logger.error(f"Error getting strategy parameters: {e}")
            return {
                'timeframe': '1m',
                'position_size_multiplier': 1.0,
                'stop_loss_pct': 0.5,
                'take_profit_pct': 1.0,
                'max_holding_time': 600,
                'entry_confidence_threshold': 0.7
            }
    
    def _generate_strategy_reasoning(self, regime: MarketRegime) -> str:
        """Generate reasoning for strategy recommendation"""
        reasoning_map = {
            MarketRegime.TRENDING_BULLISH: "Strong bullish trend with volume confirmation supports momentum strategy",
            MarketRegime.TRENDING_BEARISH: "Strong bearish trend with volume confirmation supports momentum strategy",
            MarketRegime.BREAKOUT_BULLISH: "Bullish breakout with volume spike indicates strong momentum opportunity",
            MarketRegime.BREAKOUT_BEARISH: "Bearish breakout with volume spike indicates strong momentum opportunity",
            MarketRegime.RANGING_HIGH_VOL: "High volume ranging market allows for momentum plays on range bounces",
            MarketRegime.RANGING_LOW_VOL: "Low volume ranging market favors scalping small moves",
            MarketRegime.CONSOLIDATION: "Low volatility consolidation best suited for scalping small profits",
            MarketRegime.UNCERTAIN: "Mixed signals suggest holding positions until regime clarifies"
        }
        
        return reasoning_map.get(regime, "No specific reasoning available")
    
    def _assess_risk_factors(self, market_data: Dict[str, Any], regime: MarketRegime) -> List[str]:
        """Assess current risk factors"""
        risk_factors = []
        
        try:
            # Check volatility risks
            volatility = market_data.get('volatility', {})
            if volatility.get('level', 0) > 0.8:
                risk_factors.append("High volatility environment")
            
            # Check volume risks
            volume = market_data.get('volume', {})
            if volume.get('relative', 1.0) < 0.5:
                risk_factors.append("Low volume may indicate weak price moves")
            
            # Check microstructure risks
            microstructure = market_data.get('microstructure', {})
            if microstructure.get('breakout_potential', 0) > 0.8:
                risk_factors.append("High breakout potential - price may move quickly")
            
            # Check trend risks
            trend = market_data.get('trend', {})
            if trend.get('consistency', 0) < 0.5:
                risk_factors.append("Inconsistent trend direction")
            
            # Regime-specific risks
            if regime == MarketRegime.UNCERTAIN:
                risk_factors.append("Market regime unclear - higher risk of false signals")
            elif regime in [MarketRegime.BREAKOUT_BULLISH, MarketRegime.BREAKOUT_BEARISH]:
                risk_factors.append("Breakout trades carry higher risk of false breakouts")
            
        except Exception as e:
            logger.error(f"Error assessing risk factors: {e}")
            risk_factors.append("Error in risk assessment")
        
        return risk_factors if risk_factors else ["No significant risk factors identified"]
    
    def _get_previous_regime(self) -> Optional[MarketRegime]:
        """Get the previous regime from history"""
        try:
            history = list(self.realtime_monitor.regime_history)
            if len(history) >= 2:
                return history[-2]['regime']
            return None
        except Exception as e:
            logger.error(f"Error getting previous regime: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the detector"""
        try:
            accuracy_metrics = self.performance_tracker.calculate_accuracy()
            thresholds = self.adaptive_thresholds.get_current_thresholds()
            
            return {
                'accuracy': accuracy_metrics,
                'current_thresholds': thresholds,
                'regime_history_length': len(self.realtime_monitor.regime_history)
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")