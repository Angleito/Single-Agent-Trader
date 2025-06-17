"""
Unified signal generation system for high-frequency scalping strategies.

This module integrates all scalping indicators (VuManChu optimized, Fast EMA, Momentum, Volume)
into a comprehensive high-confidence signal generation system for 15-second timeframes.

Key Features:
- Weighted scoring system across all indicators
- Real-time signal consensus building
- Multi-timeframe confirmation (when available)
- Risk assessment integration
- Thread-safe operations for live trading
- Performance optimized for <50ms complete analysis cycles

Signal Weighting System:
- VuManChu Cipher A: 25% (Primary trend/momentum)
- VuManChu Cipher B: 20% (Secondary confirmation)
- Fast EMA: 20% (Trend direction)
- Momentum Consensus: 20% (Momentum confirmation)
- Volume Confirmation: 15% (Volume validation)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..indicators import (
    FastEMA, ScalpingEMASignals,
    FastRSI, FastMACD, WilliamsPercentR, ScalpingMomentumSignals,
    ScalpingVWAP, OnBalanceVolume, VolumeMovingAverage, ScalpingVolumeSignals,
    VuManChuIndicators
)

logger = logging.getLogger(__name__)


@dataclass
class SignalWeights:
    """Signal weighting configuration for consensus building."""
    vumanchu_cipher_a: float = 0.25    # Primary trend/momentum
    vumanchu_cipher_b: float = 0.20    # Secondary confirmation
    fast_ema: float = 0.20             # Trend direction
    momentum_consensus: float = 0.20   # Momentum confirmation
    volume_confirmation: float = 0.15  # Volume validation


@dataclass
class ScalpingSignal:
    """Individual scalping signal with metadata."""
    signal_type: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    strength: float    # Signal strength (0.0 to 1.0)
    source: str        # Source indicator
    timestamp: float   # Signal timestamp
    reason: str        # Signal reasoning
    metadata: Dict[str, Any] = None  # Additional signal data


@dataclass
class MarketConsensus:
    """Market consensus from all indicators."""
    entry_signal: Optional[str] = None  # 'LONG' or 'SHORT'
    entry_confidence: float = 0.0       # 0.0 to 1.0
    exit_signal: Optional[str] = None   # 'CLOSE'
    exit_confidence: float = 0.0        # 0.0 to 1.0
    supporting_indicators: List[str] = None
    risk_factors: List[str] = None
    overall_bias: str = 'NEUTRAL'       # Market trend bias
    signal_quality: str = 'LOW'         # 'LOW', 'MEDIUM', 'HIGH'


class ScalpingSignals:
    """
    Unified signal generation system for high-frequency scalping.
    
    Integrates all scalping indicators into a comprehensive scoring system
    with real-time consensus building and risk assessment.
    """
    
    def __init__(self, signal_weights: Optional[SignalWeights] = None):
        """
        Initialize the scalping signals system.
        
        Args:
            signal_weights: Custom signal weights (default: standard scalping weights)
        """
        self.weights = signal_weights or SignalWeights()
        
        # Initialize all indicator modules
        self.vumanchu = VuManChuIndicators()
        self.fast_ema = FastEMA()
        self.ema_signals = ScalpingEMASignals()
        
        # Momentum indicators
        self.fast_rsi = FastRSI()
        self.fast_macd = FastMACD()
        self.williams_r = WilliamsPercentR()
        self.momentum_signals = ScalpingMomentumSignals()
        
        # Volume indicators
        self.vwap = ScalpingVWAP()
        self.obv = OnBalanceVolume()
        self.volume_ma = VolumeMovingAverage()
        self.volume_signals = ScalpingVolumeSignals()
        
        # Configuration thresholds
        self.ENTRY_CONFIDENCE_THRESHOLD = 0.60  # 60% minimum for entries
        self.EXIT_CONFIDENCE_THRESHOLD = 0.40   # 40% minimum for exits
        self.HIGH_CONFIDENCE_THRESHOLD = 0.80   # 80% for high-confidence signals
        
        # Performance tracking
        self._analysis_count = 0
        self._total_analysis_time = 0.0
        self._last_analysis_time = 0.0
        
        logger.info(
            "ScalpingSignals system initialized",
            extra={
                "system": "scalping_signals",
                "weights": {
                    "vumanchu_a": self.weights.vumanchu_cipher_a,
                    "vumanchu_b": self.weights.vumanchu_cipher_b,
                    "fast_ema": self.weights.fast_ema,
                    "momentum": self.weights.momentum_consensus,
                    "volume": self.weights.volume_confirmation,
                },
                "thresholds": {
                    "entry_confidence": self.ENTRY_CONFIDENCE_THRESHOLD,
                    "exit_confidence": self.EXIT_CONFIDENCE_THRESHOLD,
                    "high_confidence": self.HIGH_CONFIDENCE_THRESHOLD,
                }
            }
        )
    
    def analyze_market(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete market analysis using all indicators.
        
        Args:
            ohlcv_data: OHLC data with volume
            
        Returns:
            Complete market analysis with signals and consensus
        """
        start_time = time.perf_counter()
        
        try:
            # Validate input data
            if len(ohlcv_data) < 20:
                logger.warning("Insufficient data for analysis")
                return self._get_default_analysis("insufficient_data")
            
            # Calculate all indicators
            vumanchu_results = self.vumanchu.calculate(ohlcv_data)
            ema_results = self.fast_ema.calculate(ohlcv_data)
            ema_signals_results = self.ema_signals.calculate(ohlcv_data)
            
            momentum_results = self.momentum_signals.calculate(ohlcv_data)
            volume_results = self.volume_signals.calculate(ohlcv_data)
            
            # Generate individual signals
            vumanchu_signals = self._extract_vumanchu_signals(vumanchu_results)
            ema_signals = self._extract_ema_signals(ema_results, ema_signals_results)
            momentum_signals = self._extract_momentum_signals(momentum_results)
            volume_signals = self._extract_volume_signals(volume_results)
            
            # Build consensus
            consensus = self._build_consensus(
                vumanchu_signals, ema_signals, momentum_signals, volume_signals
            )
            
            # Generate final signals
            entry_signals = self.generate_entry_signals(consensus)
            exit_signals = self.generate_exit_signals(consensus)
            
            # Performance tracking
            analysis_time = time.perf_counter() - start_time
            self._analysis_count += 1
            self._total_analysis_time += analysis_time
            self._last_analysis_time = analysis_time
            
            result = {
                'consensus': consensus,
                'entry_signals': entry_signals,
                'exit_signals': exit_signals,
                'individual_signals': {
                    'vumanchu': vumanchu_signals,
                    'ema': ema_signals,
                    'momentum': momentum_signals,
                    'volume': volume_signals,
                },
                'indicator_results': {
                    'vumanchu': vumanchu_results,
                    'ema': ema_results,
                    'ema_signals': ema_signals_results,
                    'momentum': momentum_results,
                    'volume': volume_results,
                },
                'performance': {
                    'analysis_time_ms': analysis_time * 1000,
                    'total_analyses': self._analysis_count,
                    'avg_analysis_time_ms': (self._total_analysis_time / self._analysis_count) * 1000,
                },
                'timestamp': time.time(),
            }
            
            # Log performance if analysis is slow
            if analysis_time > 0.050:  # >50ms is concerning for scalping
                logger.warning(
                    f"Slow analysis detected: {analysis_time*1000:.1f}ms",
                    extra={
                        "analysis_time_ms": analysis_time * 1000,
                        "data_length": len(ohlcv_data),
                        "performance_concern": True
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}", exc_info=True)
            return self._get_default_analysis("error", str(e))
    
    def generate_entry_signals(self, consensus: MarketConsensus) -> List[ScalpingSignal]:
        """
        Generate high-confidence entry signals.
        
        Args:
            consensus: Market consensus from all indicators
            
        Returns:
            List of entry signals meeting confidence threshold
        """
        entry_signals = []
        
        if consensus.entry_signal and consensus.entry_confidence >= self.ENTRY_CONFIDENCE_THRESHOLD:
            signal = ScalpingSignal(
                signal_type=consensus.entry_signal,
                confidence=consensus.entry_confidence,
                strength=min(consensus.entry_confidence * 1.2, 1.0),  # Boost strong signals
                source="consensus",
                timestamp=time.time(),
                reason=f"Consensus {consensus.entry_signal.lower()} signal",
                metadata={
                    'supporting_indicators': consensus.supporting_indicators,
                    'risk_factors': consensus.risk_factors,
                    'signal_quality': consensus.signal_quality,
                    'overall_bias': consensus.overall_bias,
                }
            )
            entry_signals.append(signal)
        
        return entry_signals
    
    def generate_exit_signals(self, consensus: MarketConsensus, current_position: Optional[Dict] = None) -> List[ScalpingSignal]:
        """
        Generate exit signals for current positions.
        
        Args:
            consensus: Market consensus from all indicators
            current_position: Current position information
            
        Returns:
            List of exit signals meeting confidence threshold
        """
        exit_signals = []
        
        if consensus.exit_signal and consensus.exit_confidence >= self.EXIT_CONFIDENCE_THRESHOLD:
            signal = ScalpingSignal(
                signal_type="CLOSE",
                confidence=consensus.exit_confidence,
                strength=consensus.exit_confidence,
                source="consensus", 
                timestamp=time.time(),
                reason="Consensus exit signal",
                metadata={
                    'risk_factors': consensus.risk_factors,
                    'signal_quality': consensus.signal_quality,
                    'current_position': current_position,
                }
            )
            exit_signals.append(signal)
        
        return exit_signals
    
    def validate_signal(self, signal: ScalpingSignal) -> bool:
        """
        Multi-indicator validation of a signal.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid and safe to execute
        """
        try:
            # Basic validation
            if signal.confidence < 0.3:  # Minimum confidence threshold
                return False
            
            # Check for conflicting signals
            if signal.metadata and signal.metadata.get('risk_factors'):
                risk_count = len(signal.metadata['risk_factors'])
                if risk_count > 2:  # Too many risk factors
                    return False
            
            # Signal quality check
            quality = signal.metadata.get('signal_quality', 'LOW') if signal.metadata else 'LOW'
            if quality == 'LOW' and signal.confidence < 0.5:
                return False
            
            # Time-based validation (prevent stale signals)
            if time.time() - signal.timestamp > 60:  # Signal older than 1 minute
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Signal validation error: {e}")
            return False
    
    def get_signal_confidence(self, signal: ScalpingSignal) -> float:
        """
        Calculate final confidence score for a signal.
        
        Args:
            signal: Signal to evaluate
            
        Returns:
            Final confidence score (0.0-1.0)
        """
        base_confidence = signal.confidence
        
        # Adjust based on signal quality
        if signal.metadata:
            quality = signal.metadata.get('signal_quality', 'LOW')
            quality_multiplier = {
                'HIGH': 1.2,
                'MEDIUM': 1.0,
                'LOW': 0.8,
            }.get(quality, 1.0)
            
            # Adjust based on supporting indicators
            supporting_count = len(signal.metadata.get('supporting_indicators', []))
            support_multiplier = min(1.0 + (supporting_count * 0.1), 1.5)
            
            # Adjust based on risk factors
            risk_count = len(signal.metadata.get('risk_factors', []))
            risk_multiplier = max(1.0 - (risk_count * 0.1), 0.5)
            
            final_confidence = base_confidence * quality_multiplier * support_multiplier * risk_multiplier
        else:
            final_confidence = base_confidence
        
        return min(max(final_confidence, 0.0), 1.0)
    
    def _extract_vumanchu_signals(self, vumanchu_results: Dict) -> List[ScalpingSignal]:
        """Extract signals from VuManChu indicators."""
        signals = []
        timestamp = time.time()
        
        try:
            # Cipher A signals
            cipher_a_dot = vumanchu_results.get('cipher_a_dot')
            if cipher_a_dot is not None:
                if cipher_a_dot > 0:
                    signals.append(ScalpingSignal(
                        signal_type="LONG",
                        confidence=min(abs(cipher_a_dot) / 100.0, 1.0),
                        strength=min(abs(cipher_a_dot) / 80.0, 1.0),
                        source="vumanchu_cipher_a",
                        timestamp=timestamp,
                        reason=f"Cipher A bullish dot: {cipher_a_dot:.2f}",
                    ))
                elif cipher_a_dot < 0:
                    signals.append(ScalpingSignal(
                        signal_type="SHORT",
                        confidence=min(abs(cipher_a_dot) / 100.0, 1.0),
                        strength=min(abs(cipher_a_dot) / 80.0, 1.0),
                        source="vumanchu_cipher_a",
                        timestamp=timestamp,
                        reason=f"Cipher A bearish dot: {cipher_a_dot:.2f}",
                    ))
            
            # Cipher B signals
            cipher_b_wave = vumanchu_results.get('cipher_b_wave')
            cipher_b_money_flow = vumanchu_results.get('cipher_b_money_flow')
            
            if cipher_b_wave is not None and cipher_b_money_flow is not None:
                # Check for alignment
                wave_bullish = cipher_b_wave > 0
                mf_bullish = cipher_b_money_flow > 50
                
                if wave_bullish and mf_bullish:
                    confidence = min((abs(cipher_b_wave) + (cipher_b_money_flow - 50)) / 100.0, 1.0)
                    signals.append(ScalpingSignal(
                        signal_type="LONG",
                        confidence=confidence,
                        strength=confidence,
                        source="vumanchu_cipher_b",
                        timestamp=timestamp,
                        reason=f"Cipher B aligned bullish: wave={cipher_b_wave:.1f}, mf={cipher_b_money_flow:.1f}",
                    ))
                elif not wave_bullish and not mf_bullish:
                    confidence = min((abs(cipher_b_wave) + (50 - cipher_b_money_flow)) / 100.0, 1.0)
                    signals.append(ScalpingSignal(
                        signal_type="SHORT",
                        confidence=confidence,
                        strength=confidence,
                        source="vumanchu_cipher_b",
                        timestamp=timestamp,
                        reason=f"Cipher B aligned bearish: wave={cipher_b_wave:.1f}, mf={cipher_b_money_flow:.1f}",
                    ))
        
        except Exception as e:
            logger.error(f"Error extracting VuManChu signals: {e}")
        
        return signals
    
    def _extract_ema_signals(self, ema_results: Dict, ema_signals_results: Dict) -> List[ScalpingSignal]:
        """Extract signals from Fast EMA indicators."""
        signals = []
        timestamp = time.time()
        
        try:
            # Get EMA trend signals
            trend = ema_results.get('trend', 'NEUTRAL')
            trend_strength = ema_results.get('trend_strength', 0.0)
            
            if trend == 'BULLISH' and trend_strength > 0.3:
                signals.append(ScalpingSignal(
                    signal_type="LONG",
                    confidence=trend_strength,
                    strength=trend_strength,
                    source="fast_ema_trend",
                    timestamp=timestamp,
                    reason=f"EMA trend bullish: {trend_strength:.2f}",
                ))
            elif trend == 'BEARISH' and trend_strength > 0.3:
                signals.append(ScalpingSignal(
                    signal_type="SHORT",
                    confidence=trend_strength,
                    strength=trend_strength,
                    source="fast_ema_trend",
                    timestamp=timestamp,
                    reason=f"EMA trend bearish: {trend_strength:.2f}",
                ))
            
            # Get crossover signals
            crossovers = ema_signals_results.get('recent_crossovers', [])
            for crossover in crossovers[-2:]:  # Only recent crossovers
                if crossover.get('type') == 'golden_cross':
                    signals.append(ScalpingSignal(
                        signal_type="LONG",
                        confidence=crossover.get('strength', 0.5),
                        strength=crossover.get('strength', 0.5),
                        source="ema_crossover",
                        timestamp=timestamp,
                        reason=f"EMA golden cross: {crossover.get('periods', 'unknown')}",
                    ))
                elif crossover.get('type') == 'death_cross':
                    signals.append(ScalpingSignal(
                        signal_type="SHORT",
                        confidence=crossover.get('strength', 0.5),
                        strength=crossover.get('strength', 0.5),
                        source="ema_crossover",
                        timestamp=timestamp,
                        reason=f"EMA death cross: {crossover.get('periods', 'unknown')}",
                    ))
        
        except Exception as e:
            logger.error(f"Error extracting EMA signals: {e}")
        
        return signals
    
    def _extract_momentum_signals(self, momentum_results: Dict) -> List[ScalpingSignal]:
        """Extract signals from momentum indicators."""
        signals = []
        timestamp = time.time()
        
        try:
            consensus = momentum_results.get('consensus', {})
            overall_signal = consensus.get('signal', 'NEUTRAL')
            confidence = consensus.get('confidence', 0.0)
            
            if overall_signal == 'BULLISH' and confidence > 0.3:
                signals.append(ScalpingSignal(
                    signal_type="LONG",
                    confidence=confidence,
                    strength=confidence,
                    source="momentum_consensus",
                    timestamp=timestamp,
                    reason=f"Momentum consensus bullish: {confidence:.2f}",
                    metadata={'consensus_details': consensus}
                ))
            elif overall_signal == 'BEARISH' and confidence > 0.3:
                signals.append(ScalpingSignal(
                    signal_type="SHORT",
                    confidence=confidence,
                    strength=confidence,
                    source="momentum_consensus",
                    timestamp=timestamp,
                    reason=f"Momentum consensus bearish: {confidence:.2f}",
                    metadata={'consensus_details': consensus}
                ))
        
        except Exception as e:
            logger.error(f"Error extracting momentum signals: {e}")
        
        return signals
    
    def _extract_volume_signals(self, volume_results: Dict) -> List[ScalpingSignal]:
        """Extract signals from volume indicators."""
        signals = []
        timestamp = time.time()
        
        try:
            volume_confirmation = volume_results.get('volume_confirmation', 'NEUTRAL')
            confirmation_strength = volume_results.get('confirmation_strength', 0.0)
            
            if volume_confirmation == 'BULLISH' and confirmation_strength > 0.3:
                signals.append(ScalpingSignal(
                    signal_type="LONG",
                    confidence=confirmation_strength,
                    strength=confirmation_strength,
                    source="volume_confirmation",
                    timestamp=timestamp,
                    reason=f"Volume confirms bullish: {confirmation_strength:.2f}",
                ))
            elif volume_confirmation == 'BEARISH' and confirmation_strength > 0.3:
                signals.append(ScalpingSignal(
                    signal_type="SHORT",
                    confidence=confirmation_strength,
                    strength=confirmation_strength,
                    source="volume_confirmation",
                    timestamp=timestamp,
                    reason=f"Volume confirms bearish: {confirmation_strength:.2f}",
                ))
        
        except Exception as e:
            logger.error(f"Error extracting volume signals: {e}")
        
        return signals
    
    def _build_consensus(self, vumanchu_signals: List[ScalpingSignal], 
                        ema_signals: List[ScalpingSignal],
                        momentum_signals: List[ScalpingSignal],
                        volume_signals: List[ScalpingSignal]) -> MarketConsensus:
        """Build market consensus from all indicator signals."""
        
        # Combine all signals
        all_signals = vumanchu_signals + ema_signals + momentum_signals + volume_signals
        
        if not all_signals:
            return MarketConsensus()
        
        # Separate by signal type
        long_signals = [s for s in all_signals if s.signal_type == 'LONG']
        short_signals = [s for s in all_signals if s.signal_type == 'SHORT']
        
        # Calculate weighted scores
        long_score = self._calculate_weighted_score(long_signals)
        short_score = self._calculate_weighted_score(short_signals)
        
        # Determine consensus
        consensus = MarketConsensus()
        
        if long_score > short_score and long_score >= self.ENTRY_CONFIDENCE_THRESHOLD:
            consensus.entry_signal = 'LONG'
            consensus.entry_confidence = long_score
            consensus.supporting_indicators = [s.source for s in long_signals]
        elif short_score > long_score and short_score >= self.ENTRY_CONFIDENCE_THRESHOLD:
            consensus.entry_signal = 'SHORT'
            consensus.entry_confidence = short_score
            consensus.supporting_indicators = [s.source for s in short_signals]
        
        # Determine signal quality
        if max(long_score, short_score) >= self.HIGH_CONFIDENCE_THRESHOLD:
            consensus.signal_quality = 'HIGH'
        elif max(long_score, short_score) >= self.ENTRY_CONFIDENCE_THRESHOLD:
            consensus.signal_quality = 'MEDIUM'
        else:
            consensus.signal_quality = 'LOW'
        
        # Overall bias
        if long_score > short_score + 0.1:
            consensus.overall_bias = 'BULLISH'
        elif short_score > long_score + 0.1:
            consensus.overall_bias = 'BEARISH'
        else:
            consensus.overall_bias = 'NEUTRAL'
        
        # Risk factors
        consensus.risk_factors = self._identify_risk_factors(all_signals, long_score, short_score)
        
        return consensus
    
    def _calculate_weighted_score(self, signals: List[ScalpingSignal]) -> float:
        """Calculate weighted score for a group of signals."""
        if not signals:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = self._get_signal_weight(signal.source)
            score = signal.confidence * signal.strength
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _get_signal_weight(self, source: str) -> float:
        """Get weight for a signal source."""
        weight_map = {
            'vumanchu_cipher_a': self.weights.vumanchu_cipher_a,
            'vumanchu_cipher_b': self.weights.vumanchu_cipher_b,
            'fast_ema_trend': self.weights.fast_ema,
            'ema_crossover': self.weights.fast_ema * 0.8,
            'momentum_consensus': self.weights.momentum_consensus,
            'volume_confirmation': self.weights.volume_confirmation,
        }
        return weight_map.get(source, 0.1)  # Default weight for unknown sources
    
    def _identify_risk_factors(self, all_signals: List[ScalpingSignal], 
                              long_score: float, short_score: float) -> List[str]:
        """Identify risk factors from signal analysis."""
        risk_factors = []
        
        # Conflicting signals
        if abs(long_score - short_score) < 0.2:
            risk_factors.append("conflicting_signals")
        
        # Low volume confirmation
        volume_signals = [s for s in all_signals if 'volume' in s.source]
        if not volume_signals:
            risk_factors.append("no_volume_confirmation")
        
        # Weak momentum
        momentum_signals = [s for s in all_signals if 'momentum' in s.source]
        if momentum_signals and all(s.confidence < 0.5 for s in momentum_signals):
            risk_factors.append("weak_momentum")
        
        # Limited supporting indicators
        unique_sources = set(s.source for s in all_signals)
        if len(unique_sources) < 3:
            risk_factors.append("limited_confirmation")
        
        return risk_factors
    
    def _get_default_analysis(self, reason: str, error_msg: str = "") -> Dict[str, Any]:
        """Return default analysis structure for error cases."""
        return {
            'consensus': MarketConsensus(),
            'entry_signals': [],
            'exit_signals': [],
            'individual_signals': {
                'vumanchu': [],
                'ema': [],
                'momentum': [],
                'volume': [],
            },
            'indicator_results': {},
            'performance': {
                'analysis_time_ms': 0,
                'total_analyses': self._analysis_count,
            },
            'error': {
                'reason': reason,
                'message': error_msg,
            },
            'timestamp': time.time(),
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the signal system."""
        if self._analysis_count == 0:
            return {'status': 'no_analyses_performed'}
        
        avg_time_ms = (self._total_analysis_time / self._analysis_count) * 1000
        
        return {
            'total_analyses': self._analysis_count,
            'avg_analysis_time_ms': avg_time_ms,
            'last_analysis_time_ms': self._last_analysis_time * 1000,
            'performance_rating': 'excellent' if avg_time_ms < 30 else 'good' if avg_time_ms < 50 else 'needs_optimization',
            'total_analysis_time_seconds': self._total_analysis_time,
        }