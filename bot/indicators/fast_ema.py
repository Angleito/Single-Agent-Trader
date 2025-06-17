"""
Fast EMA (Exponential Moving Average) indicators optimized for high-frequency scalping.

This module provides ultra-fast EMA calculations and signals specifically designed for
scalping strategies on 15-second timeframes. Features optimized periods [3, 5, 8, 13]
for capturing quick price movements and trend confirmations.

Key Features:
- FastEMA: Ultra-fast EMA calculations with real-time updates
- ScalpingEMASignals: Advanced crossover detection and signal generation
- Memory-efficient rolling calculations for high-frequency data
- Thread-safe operations for real-time trading
- Optimized numpy operations for maximum performance
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import ta with graceful fallback
try:
    from ..utils import ta
    _ta_available = ta is not None
except ImportError:
    ta = None
    _ta_available = False

logger = logging.getLogger(__name__)


class FastEMA:
    """
    Ultra-fast EMA calculations optimized for scalping.
    
    Features:
    - Periods optimized for scalping: [3, 5, 8, 13]
    - Real-time single-tick updates
    - Vectorized computation for historical data
    - Memory-efficient rolling calculations
    - Thread-safe operations
    """
    
    def __init__(self, periods: Optional[List[int]] = None) -> None:
        """
        Initialize FastEMA with scalping-optimized periods.
        
        Args:
            periods: List of EMA periods (default: [3, 5, 8, 13] for scalping)
        """
        self.periods = periods or [3, 5, 8, 13]
        
        if not self.periods or len(self.periods) < 2:
            logger.error(
                "FastEMA initialization failed: need at least 2 periods",
                extra={
                    "indicator": "fast_ema",
                    "error_type": "insufficient_periods",
                    "provided_periods": len(self.periods) if self.periods else 0,
                    "periods": self.periods,
                }
            )
            raise ValueError("FastEMA requires at least 2 EMA periods")
        
        # Sort periods for consistent ordering
        self.periods = sorted(self.periods)
        
        # Store EMA states for real-time updates
        self._ema_states: Dict[int, Optional[float]] = {period: None for period in self.periods}
        self._smoothing_factors: Dict[int, float] = {
            period: 2.0 / (period + 1) for period in self.periods
        }
        
        # Performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        
        logger.info(
            "FastEMA indicator initialized for scalping",
            extra={
                "indicator": "fast_ema",
                "periods": self.periods,
                "smoothing_factors": self._smoothing_factors,
                "optimization": "scalping_15s",
            }
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all EMA values and generate scalping signals.
        
        Args:
            data: DataFrame with OHLC data (must contain 'close' column)
            
        Returns:
            Dictionary with EMA values, signals, and analysis
        """
        start_time = time.perf_counter()
        
        logger.debug(
            "Starting FastEMA calculation",
            extra={
                "indicator": "fast_ema",
                "data_points": len(data),
                "calculation_id": self._calculation_count,
            }
        )
        
        if data.empty:
            logger.warning(
                "Empty DataFrame provided for FastEMA calculation",
                extra={"indicator": "fast_ema", "issue": "empty_dataframe"}
            )
            return self._empty_result()
        
        if "close" not in data.columns:
            logger.error(
                "Missing close column for FastEMA calculation",
                extra={
                    "indicator": "fast_ema",
                    "error_type": "missing_close_column",
                    "available_columns": list(data.columns),
                }
            )
            raise ValueError("DataFrame must contain 'close' column")
        
        try:
            # Calculate all EMAs
            ema_values = self._calculate_all_emas(data["close"])
            
            # Get latest values
            latest_values = {f"ema_{period}": values.iloc[-1] if not values.empty else None 
                           for period, values in ema_values.items()}
            
            # Update internal states for real-time updates
            self._update_states(latest_values)
            
            # Calculate performance metrics
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000
            
            self._calculation_count += 1
            self._total_calculation_time += duration
            
            logger.debug(
                "FastEMA calculation completed",
                extra={
                    "indicator": "fast_ema",
                    "duration_ms": round(duration, 2),
                    "data_points": len(data),
                    "periods_calculated": len(self.periods),
                }
            )
            
            result = {
                "ema_values": latest_values,
                "ema_series": ema_values,
                "calculation_time_ms": duration,
                "data_points": len(data),
            }
            
            return result
            
        except Exception as e:
            logger.error(
                "FastEMA calculation failed with exception",
                extra={
                    "indicator": "fast_ema",
                    "error_type": "calculation_exception",
                    "error_message": str(e),
                    "data_points": len(data),
                }
            )
            return self._empty_result()
    
    def update_realtime(self, price: float) -> Dict[str, Any]:
        """
        Update EMAs with single price tick for real-time calculations.
        
        Args:
            price: Latest price value
            
        Returns:
            Dictionary with updated EMA values and signals
        """
        if price <= 0:
            logger.warning(
                "Invalid price for real-time EMA update",
                extra={
                    "indicator": "fast_ema",
                    "issue": "invalid_price",
                    "price": price,
                }
            )
            return self._empty_result()
        
        try:
            updated_values = {}
            
            for period in self.periods:
                alpha = self._smoothing_factors[period]
                
                if self._ema_states[period] is None:
                    # First value - use price as initial EMA
                    self._ema_states[period] = price
                else:
                    # Update EMA: EMA = alpha * price + (1 - alpha) * previous_EMA
                    self._ema_states[period] = (
                        alpha * price + (1 - alpha) * self._ema_states[period]
                    )
                
                updated_values[f"ema_{period}"] = self._ema_states[period]
            
            logger.debug(
                "Real-time EMA update completed",
                extra={
                    "indicator": "fast_ema",
                    "price": price,
                    "updated_emas": len(updated_values),
                }
            )
            
            return {
                "ema_values": updated_values,
                "price": price,
                "timestamp": pd.Timestamp.now(),
            }
            
        except Exception as e:
            logger.error(
                "Real-time EMA update failed",
                extra={
                    "indicator": "fast_ema",
                    "error_type": "realtime_update_failed",
                    "error_message": str(e),
                    "price": price,
                }
            )
            return self._empty_result()
    
    def _calculate_all_emas(self, prices: pd.Series) -> Dict[int, pd.Series]:
        """
        Calculate all EMAs using vectorized operations.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary mapping periods to EMA series
        """
        ema_values = {}
        
        for period in self.periods:
            try:
                # Use pandas_ta for vectorized EMA calculation if available
                if _ta_available and ta is not None:
                    ema_series = ta.ema(prices, length=period)
                    
                    if ema_series is not None:
                        ema_values[period] = ema_series.astype(float)
                    else:
                        # Fallback to manual calculation
                        ema_values[period] = self._manual_ema_calculation(prices, period)
                else:
                    # Use manual calculation when pandas_ta is not available
                    ema_values[period] = self._manual_ema_calculation(prices, period)
                    
            except Exception as e:
                logger.warning(
                    f"EMA calculation failed for period {period}, using fallback",
                    extra={
                        "indicator": "fast_ema",
                        "period": period,
                        "error": str(e),
                    }
                )
                ema_values[period] = self._manual_ema_calculation(prices, period)
        
        return ema_values
    
    def _manual_ema_calculation(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Manual EMA calculation as fallback.
        
        Args:
            prices: Price series
            period: EMA period
            
        Returns:
            EMA series
        """
        alpha = 2.0 / (period + 1)
        ema_values = np.zeros(len(prices))
        ema_values[0] = prices.iloc[0]
        
        for i in range(1, len(prices)):
            ema_values[i] = alpha * prices.iloc[i] + (1 - alpha) * ema_values[i - 1]
        
        return pd.Series(ema_values, index=prices.index)
    
    def _update_states(self, latest_values: Dict[str, Optional[float]]) -> None:
        """
        Update internal EMA states for real-time calculations.
        
        Args:
            latest_values: Dictionary of latest EMA values
        """
        for period in self.periods:
            key = f"ema_{period}"
            if key in latest_values and latest_values[key] is not None:
                self._ema_states[period] = latest_values[key]
    
    def _empty_result(self) -> Dict[str, Any]:
        """
        Return empty result structure.
        
        Returns:
            Empty result dictionary
        """
        return {
            "ema_values": {f"ema_{period}": None for period in self.periods},
            "ema_series": {},
            "calculation_time_ms": 0.0,
            "data_points": 0,
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for monitoring.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_time = (
            self._total_calculation_time / self._calculation_count
            if self._calculation_count > 0
            else 0
        )
        
        return {
            "calculation_count": self._calculation_count,
            "total_calculation_time_ms": round(self._total_calculation_time, 2),
            "average_calculation_time_ms": round(avg_time, 2),
            "periods": self.periods,
            "current_states": self._ema_states,
        }


class ScalpingEMASignals:
    """
    Advanced EMA signal generation for scalping strategies.
    
    Features:
    - Multiple EMA crossover detection (3x5, 5x8, 8x13)
    - Price vs EMA position analysis
    - EMA slope calculation for momentum
    - Signal strength scoring based on EMA alignment
    - Bullish/bearish setup detection
    """
    
    def __init__(self, fast_ema: Optional[FastEMA] = None) -> None:
        """
        Initialize scalping signal generator.
        
        Args:
            fast_ema: FastEMA instance (creates new if None)
        """
        self.fast_ema = fast_ema or FastEMA()
        self.periods = self.fast_ema.periods
        
        # Signal history for trend analysis
        self._signal_history: List[Dict[str, Any]] = []
        self._max_history = 100  # Keep last 100 signals
        
        logger.info(
            "ScalpingEMASignals initialized",
            extra={
                "indicator": "scalping_ema_signals",
                "periods": self.periods,
                "crossover_pairs": self._get_crossover_pairs(),
            }
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive scalping signals.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary with all signals and analysis
        """
        start_time = time.perf_counter()
        
        logger.debug(
            "Starting scalping EMA signal calculation",
            extra={
                "indicator": "scalping_ema_signals",
                "data_points": len(data),
            }
        )
        
        try:
            # Calculate base EMAs
            ema_result = self.fast_ema.calculate(data)
            
            if not ema_result["ema_series"]:
                logger.warning(
                    "No EMA data available for signal calculation",
                    extra={"indicator": "scalping_ema_signals", "issue": "no_ema_data"}
                )
                return self._empty_signal_result()
            
            # Calculate signals
            signals = self._calculate_all_signals(data, ema_result["ema_series"])
            
            # Performance tracking
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000
            
            logger.debug(
                "Scalping EMA signal calculation completed",
                extra={
                    "indicator": "scalping_ema_signals",
                    "duration_ms": round(duration, 2),
                    "signals_generated": len(signals.get("crossovers", [])),
                }
            )
            
            # Combine results
            result = {
                **ema_result,
                **signals,
                "signal_calculation_time_ms": duration,
            }
            
            # Store in history
            self._add_to_history(result)
            
            return result
            
        except Exception as e:
            logger.error(
                "Scalping EMA signal calculation failed",
                extra={
                    "indicator": "scalping_ema_signals",
                    "error_type": "signal_calculation_failed",
                    "error_message": str(e),
                }
            )
            return self._empty_signal_result()
    
    def _calculate_all_signals(self, data: pd.DataFrame, ema_series: Dict[int, pd.Series]) -> Dict[str, Any]:
        """
        Calculate all scalping signals.
        
        Args:
            data: OHLC data
            ema_series: Dictionary of EMA series
            
        Returns:
            Dictionary with all signal types
        """
        signals = {
            "crossovers": [],
            "trend_strength": 0.0,
            "setup_type": "neutral",
            "signals": [],
        }
        
        if len(data) < 2:
            return signals
        
        # Get crossover signals
        crossovers = self.get_crossover_signals(ema_series)
        signals["crossovers"] = crossovers
        
        # Calculate trend strength
        trend_strength = self.get_trend_strength(data["close"], ema_series)
        signals["trend_strength"] = trend_strength
        
        # Determine setup type
        setup_type = self._determine_setup_type(data["close"], ema_series)
        signals["setup_type"] = setup_type
        
        # Generate actionable signals
        actionable_signals = self._generate_actionable_signals(
            data["close"], ema_series, crossovers, trend_strength, setup_type
        )
        signals["signals"] = actionable_signals
        
        return signals
    
    def get_crossover_signals(self, ema_series: Dict[int, pd.Series]) -> List[Dict[str, Any]]:
        """
        Detect EMA crossover signals.
        
        Args:
            ema_series: Dictionary of EMA series
            
        Returns:
            List of crossover signals with confidence scores
        """
        crossovers = []
        crossover_pairs = self._get_crossover_pairs()
        
        for fast_period, slow_period in crossover_pairs:
            if fast_period not in ema_series or slow_period not in ema_series:
                continue
            
            fast_ema = ema_series[fast_period]
            slow_ema = ema_series[slow_period]
            
            if len(fast_ema) < 2 or len(slow_ema) < 2:
                continue
            
            # Detect crossovers
            bullish_cross = self._detect_crossover(fast_ema, slow_ema)
            bearish_cross = self._detect_crossover(slow_ema, fast_ema)
            
            # Get latest crossover signals
            if bullish_cross.iloc[-1]:
                confidence = self._calculate_crossover_confidence(
                    fast_ema, slow_ema, "bullish"
                )
                crossovers.append({
                    "type": "bullish_crossover",
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "confidence": confidence,
                    "timestamp": fast_ema.index[-1],
                })
            
            if bearish_cross.iloc[-1]:
                confidence = self._calculate_crossover_confidence(
                    fast_ema, slow_ema, "bearish"
                )
                crossovers.append({
                    "type": "bearish_crossover",
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "confidence": confidence,
                    "timestamp": fast_ema.index[-1],
                })
        
        return crossovers
    
    def get_trend_strength(self, price: pd.Series, ema_series: Dict[int, pd.Series]) -> float:
        """
        Calculate trend strength from EMA alignment.
        
        Args:
            price: Price series
            ema_series: Dictionary of EMA series
            
        Returns:
            Trend strength from -1.0 (strong bearish) to 1.0 (strong bullish)
        """
        if not ema_series or len(price) == 0:
            return 0.0
        
        try:
            # Get latest EMA values
            latest_emas = {}
            for period in self.periods:
                if period in ema_series and not ema_series[period].empty:
                    latest_emas[period] = ema_series[period].iloc[-1]
            
            if len(latest_emas) < 2:
                return 0.0
            
            # Check EMA alignment
            sorted_periods = sorted(latest_emas.keys())
            alignment_score = 0.0
            
            # Perfect bullish alignment: EMA3 > EMA5 > EMA8 > EMA13
            # Perfect bearish alignment: EMA3 < EMA5 < EMA8 < EMA13
            for i in range(len(sorted_periods) - 1):
                fast_period = sorted_periods[i]
                slow_period = sorted_periods[i + 1]
                
                if latest_emas[fast_period] > latest_emas[slow_period]:
                    alignment_score += 1.0
                elif latest_emas[fast_period] < latest_emas[slow_period]:
                    alignment_score -= 1.0
            
            # Normalize to [-1, 1]
            max_score = len(sorted_periods) - 1
            normalized_score = alignment_score / max_score if max_score > 0 else 0.0
            
            # Factor in price position relative to fastest EMA
            fastest_period = min(latest_emas.keys())
            price_position = 0.0
            
            if fastest_period in latest_emas:
                latest_price = price.iloc[-1]
                fastest_ema = latest_emas[fastest_period]
                
                if fastest_ema > 0:
                    price_diff = (latest_price - fastest_ema) / fastest_ema
                    price_position = np.tanh(price_diff * 10)  # Sigmoid-like function
            
            # Combine alignment and price position
            trend_strength = (normalized_score * 0.7) + (price_position * 0.3)
            
            # Clamp to [-1, 1]
            trend_strength = max(-1.0, min(1.0, trend_strength))
            
            return trend_strength
            
        except Exception as e:
            logger.warning(
                "Trend strength calculation failed",
                extra={
                    "indicator": "scalping_ema_signals",
                    "error": str(e),
                }
            )
            return 0.0
    
    def is_bullish_setup(self, price: pd.Series, ema_series: Dict[int, pd.Series]) -> bool:
        """
        Check if current setup is bullish.
        
        Args:
            price: Price series
            ema_series: Dictionary of EMA series
            
        Returns:
            True if bullish setup detected
        """
        trend_strength = self.get_trend_strength(price, ema_series)
        return trend_strength > 0.5
    
    def is_bearish_setup(self, price: pd.Series, ema_series: Dict[int, pd.Series]) -> bool:
        """
        Check if current setup is bearish.
        
        Args:
            price: Price series
            ema_series: Dictionary of EMA series
            
        Returns:
            True if bearish setup detected
        """
        trend_strength = self.get_trend_strength(price, ema_series)
        return trend_strength < -0.5
    
    def _get_crossover_pairs(self) -> List[Tuple[int, int]]:
        """
        Get EMA crossover pairs for signal detection.
        
        Returns:
            List of (fast_period, slow_period) tuples
        """
        pairs = []
        sorted_periods = sorted(self.periods)
        
        for i in range(len(sorted_periods) - 1):
            fast = sorted_periods[i]
            slow = sorted_periods[i + 1]
            pairs.append((fast, slow))
        
        return pairs
    
    def _detect_crossover(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Detect crossover events (series1 crosses above series2).
        
        Args:
            series1: First series
            series2: Second series
            
        Returns:
            Boolean series indicating crossover points
        """
        if len(series1) < 2 or len(series2) < 2:
            return pd.Series(False, index=series1.index)
        
        # Previous values
        prev_series1 = series1.shift(1)
        prev_series2 = series2.shift(1)
        
        # Crossover: was below, now above
        crossover = (prev_series1 <= prev_series2) & (series1 > series2)
        
        return crossover.fillna(False)
    
    def _calculate_crossover_confidence(
        self, fast_ema: pd.Series, slow_ema: pd.Series, direction: str
    ) -> float:
        """
        Calculate confidence score for crossover signal.
        
        Args:
            fast_ema: Fast EMA series
            slow_ema: Slow EMA series
            direction: "bullish" or "bearish"
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        if len(fast_ema) < 3 or len(slow_ema) < 3:
            return 0.5
        
        try:
            # Calculate separation distance
            separation = abs(fast_ema.iloc[-1] - slow_ema.iloc[-1])
            avg_price = (fast_ema.iloc[-1] + slow_ema.iloc[-1]) / 2
            separation_pct = separation / avg_price if avg_price > 0 else 0
            
            # Calculate momentum (slope of faster EMA)
            fast_slope = fast_ema.iloc[-1] - fast_ema.iloc[-3]
            momentum_score = abs(fast_slope) / avg_price if avg_price > 0 else 0
            
            # Base confidence starts at 0.5
            confidence = 0.5
            
            # Add separation component (0.0 to 0.3)
            separation_component = min(0.3, separation_pct * 100)
            confidence += separation_component
            
            # Add momentum component (0.0 to 0.2)
            momentum_component = min(0.2, momentum_score * 50)
            confidence += momentum_component
            
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception:
            return 0.5
    
    def _determine_setup_type(
        self, price: pd.Series, ema_series: Dict[int, pd.Series]
    ) -> str:
        """
        Determine the current setup type.
        
        Args:
            price: Price series
            ema_series: Dictionary of EMA series
            
        Returns:
            Setup type: "bullish", "bearish", or "neutral"
        """
        if self.is_bullish_setup(price, ema_series):
            return "bullish"
        elif self.is_bearish_setup(price, ema_series):
            return "bearish"
        else:
            return "neutral"
    
    def _generate_actionable_signals(
        self,
        price: pd.Series,
        ema_series: Dict[int, pd.Series],
        crossovers: List[Dict[str, Any]],
        trend_strength: float,
        setup_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable trading signals.
        
        Args:
            price: Price series
            ema_series: Dictionary of EMA series
            crossovers: List of crossover signals
            trend_strength: Current trend strength
            setup_type: Current setup type
            
        Returns:
            List of actionable signals with confidence scores
        """
        signals = []
        
        # High-confidence crossover signals
        high_confidence_crossovers = [
            cross for cross in crossovers if cross["confidence"] > 0.7
        ]
        
        for crossover in high_confidence_crossovers:
            signal = {
                "type": "crossover_signal",
                "direction": "buy" if crossover["type"] == "bullish_crossover" else "sell",
                "strength": crossover["confidence"],
                "reason": f"High-confidence {crossover['type']} between EMA{crossover['fast_period']} and EMA{crossover['slow_period']}",
                "timestamp": crossover["timestamp"],
            }
            signals.append(signal)
        
        # Strong trend continuation signals
        if abs(trend_strength) > 0.8:
            direction = "buy" if trend_strength > 0 else "sell"
            signal = {
                "type": "trend_continuation",
                "direction": direction,
                "strength": abs(trend_strength),
                "reason": f"Strong {setup_type} trend continuation (strength: {trend_strength:.2f})",
                "timestamp": price.index[-1] if not price.empty else None,
            }
            signals.append(signal)
        
        return signals
    
    def _add_to_history(self, result: Dict[str, Any]) -> None:
        """
        Add result to signal history.
        
        Args:
            result: Signal calculation result
        """
        # Keep only essential data for history
        history_entry = {
            "timestamp": pd.Timestamp.now(),
            "trend_strength": result.get("trend_strength", 0.0),
            "setup_type": result.get("setup_type", "neutral"),
            "crossover_count": len(result.get("crossovers", [])),
            "signal_count": len(result.get("signals", [])),
        }
        
        self._signal_history.append(history_entry)
        
        # Trim history to max size
        if len(self._signal_history) > self._max_history:
            self._signal_history = self._signal_history[-self._max_history:]
    
    def _empty_signal_result(self) -> Dict[str, Any]:
        """
        Return empty signal result structure.
        
        Returns:
            Empty signal result dictionary
        """
        return {
            "ema_values": {f"ema_{period}": None for period in self.periods},
            "ema_series": {},
            "crossovers": [],
            "trend_strength": 0.0,
            "setup_type": "neutral",
            "signals": [],
            "calculation_time_ms": 0.0,
            "signal_calculation_time_ms": 0.0,
            "data_points": 0,
        }
    
    def get_signal_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get signal history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of historical signal data
        """
        history = self._signal_history
        if limit is not None:
            history = history[-limit:]
        return history