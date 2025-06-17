"""
Volume-based indicators optimized for high-frequency scalping on 15-second timeframes.

This module provides comprehensive volume analysis to confirm price movements and 
validate breakouts for scalping strategies. Features real-time updates, volume 
profiles, and consensus-based signal generation.

Key Components:
- ScalpingVWAP: Intraday VWAP with bands and position analysis
- OnBalanceVolume: Volume accumulation with divergence detection
- VolumeMovingAverage: Multi-period volume analysis with spike detection
- VolumeProfile: Real-time volume distribution and POC identification
- ScalpingVolumeSignals: Consensus-based volume confirmation system

Performance Requirements:
- Optimized for 15-second real-time updates
- Maximum 3ms calculation time per update
- Memory-efficient rolling window calculations
- Thread-safe for concurrent operations
"""

import logging
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import ta with graceful fallback
try:
    from ..utils.ta_import import ta
    _ta_available = ta is not None
except ImportError:
    ta = None
    _ta_available = False

logger = logging.getLogger(__name__)


class ScalpingVWAP:
    """
    Intraday VWAP calculation optimized for scalping strategies.
    
    Features:
    - Intraday VWAP with market open reset
    - Rolling VWAP for last 50 periods (scalping-specific)
    - VWAP bands (±1 and ±2 standard deviations)
    - Price vs VWAP position analysis
    - Real-time tick-by-tick updates
    """
    
    def __init__(self, rolling_periods: int = 50, reset_at_open: bool = True) -> None:
        """
        Initialize VWAP calculator.
        
        Args:
            rolling_periods: Number of periods for rolling VWAP calculation
            reset_at_open: Whether to reset VWAP at market open
        """
        self.rolling_periods = rolling_periods
        self.reset_at_open = reset_at_open
        
        # Rolling data storage
        self._prices: deque = deque(maxlen=rolling_periods)
        self._volumes: deque = deque(maxlen=rolling_periods)
        self._pv_cumsum: float = 0.0  # Price * Volume cumulative sum
        self._volume_cumsum: float = 0.0  # Volume cumulative sum
        
        # Intraday VWAP calculation
        self._intraday_pv_sum: float = 0.0
        self._intraday_volume_sum: float = 0.0
        self._last_reset_date: Optional[datetime] = None
        
        # Real-time state
        self._current_vwap: Optional[float] = None
        self._current_bands: Dict[str, Optional[float]] = {
            'upper_1': None, 'lower_1': None,
            'upper_2': None, 'lower_2': None
        }
        
        # Performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        
        logger.info(
            "ScalpingVWAP initialized",
            extra={
                "indicator": "scalping_vwap",
                "rolling_periods": rolling_periods,
                "reset_at_open": reset_at_open,
            }
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate VWAP values and bands from historical data.
        
        Args:
            data: DataFrame with OHLCV data (must contain 'close', 'volume')
            
        Returns:
            Dictionary with VWAP values, bands, and analysis
        """
        start_time = time.perf_counter()
        
        logger.debug(
            "Starting VWAP calculation",
            extra={
                "indicator": "scalping_vwap",
                "data_points": len(data),
            }
        )
        
        if data.empty or 'close' not in data.columns or 'volume' not in data.columns:
            logger.warning(
                "Invalid data for VWAP calculation",
                extra={
                    "indicator": "scalping_vwap",
                    "issue": "missing_required_columns",
                    "columns": list(data.columns) if not data.empty else [],
                }
            )
            return self._empty_result()
        
        try:
            # Use typical price (HLC/3) if high/low available, otherwise close
            if 'high' in data.columns and 'low' in data.columns:
                typical_price = (data['high'] + data['low'] + data['close']) / 3
            else:
                typical_price = data['close']
            
            volume = data['volume']
            
            # Calculate rolling VWAP
            rolling_vwap = self._calculate_rolling_vwap(typical_price, volume)
            
            # Calculate intraday VWAP if timestamps available
            intraday_vwap = None
            if hasattr(data.index, 'date') or isinstance(data.index, pd.DatetimeIndex):
                intraday_vwap = self._calculate_intraday_vwap(typical_price, volume, data.index)
            
            # Use intraday VWAP if available, otherwise rolling VWAP
            vwap_series = intraday_vwap if intraday_vwap is not None else rolling_vwap
            
            # Calculate VWAP bands
            vwap_bands = self._calculate_vwap_bands(typical_price, vwap_series)
            
            # Current values
            current_vwap = vwap_series.iloc[-1] if not vwap_series.empty else None
            current_price = data['close'].iloc[-1] if not data.empty else None
            
            # Position analysis
            position = self._analyze_price_position(current_price, current_vwap, vwap_bands)
            
            # Generate signals
            signal = self._generate_vwap_signal(current_price, current_vwap, vwap_bands, volume.iloc[-1])
            
            # Performance tracking
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000
            
            self._calculation_count += 1
            self._total_calculation_time += duration
            
            # Update internal state
            self._current_vwap = current_vwap
            self._current_bands = vwap_bands
            
            logger.debug(
                "VWAP calculation completed",
                extra={
                    "indicator": "scalping_vwap",
                    "duration_ms": round(duration, 2),
                    "current_vwap": current_vwap,
                    "position": position,
                }
            )
            
            return {
                'vwap': {
                    'current': current_vwap,
                    'upper_band_1': vwap_bands.get('upper_1'),
                    'lower_band_1': vwap_bands.get('lower_1'),
                    'upper_band_2': vwap_bands.get('upper_2'),
                    'lower_band_2': vwap_bands.get('lower_2'),
                    'position': position,
                    'signal': signal,
                },
                'vwap_series': vwap_series,
                'bands_series': vwap_bands,
                'calculation_time_ms': duration,
                'data_points': len(data),
            }
            
        except Exception as e:
            logger.error(
                "VWAP calculation failed",
                extra={
                    "indicator": "scalping_vwap",
                    "error_type": "calculation_exception",
                    "error_message": str(e),
                }
            )
            return self._empty_result()
    
    def update_realtime(self, price: float, volume: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update VWAP with single price/volume tick.
        
        Args:
            price: Latest price value
            volume: Latest volume value
            timestamp: Optional timestamp for intraday reset
            
        Returns:
            Dictionary with updated VWAP values and analysis
        """
        if price <= 0 or volume < 0:
            logger.warning(
                "Invalid price/volume for VWAP update",
                extra={
                    "indicator": "scalping_vwap",
                    "price": price,
                    "volume": volume,
                }
            )
            return self._empty_result()
        
        try:
            # Check if we need to reset for new trading day
            if self.reset_at_open and timestamp:
                self._check_and_reset_intraday(timestamp)
            
            # Update rolling VWAP
            self._update_rolling_vwap(price, volume)
            
            # Update intraday VWAP
            self._update_intraday_vwap(price, volume)
            
            # Use intraday VWAP if available, otherwise rolling
            current_vwap = (self._intraday_pv_sum / self._intraday_volume_sum 
                          if self._intraday_volume_sum > 0 
                          else self._pv_cumsum / self._volume_cumsum if self._volume_cumsum > 0 else price)
            
            # Calculate bands using recent price data
            bands = self._calculate_realtime_bands(current_vwap)
            
            # Position analysis
            position = self._analyze_price_position(price, current_vwap, bands)
            
            # Generate signal
            signal = self._generate_vwap_signal(price, current_vwap, bands, volume)
            
            # Update internal state
            self._current_vwap = current_vwap
            self._current_bands = bands
            
            logger.debug(
                "Real-time VWAP update completed",
                extra={
                    "indicator": "scalping_vwap",
                    "price": price,
                    "volume": volume,
                    "vwap": current_vwap,
                    "position": position,
                }
            )
            
            return {
                'vwap': {
                    'current': current_vwap,
                    'upper_band_1': bands.get('upper_1'),
                    'lower_band_1': bands.get('lower_1'),
                    'upper_band_2': bands.get('upper_2'),
                    'lower_band_2': bands.get('lower_2'),
                    'position': position,
                    'signal': signal,
                },
                'price': price,
                'volume': volume,
                'timestamp': timestamp or datetime.now(UTC),
            }
            
        except Exception as e:
            logger.error(
                "Real-time VWAP update failed",
                extra={
                    "indicator": "scalping_vwap",
                    "error_message": str(e),
                    "price": price,
                    "volume": volume,
                }
            )
            return self._empty_result()
    
    def _calculate_rolling_vwap(self, prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Calculate rolling VWAP series."""
        pv = prices * volumes
        rolling_pv_sum = pv.rolling(window=self.rolling_periods, min_periods=1).sum()
        rolling_volume_sum = volumes.rolling(window=self.rolling_periods, min_periods=1).sum()
        
        # Avoid division by zero
        rolling_vwap = rolling_pv_sum / rolling_volume_sum.replace(0, np.nan)
        return rolling_vwap.fillna(method='ffill')
    
    def _calculate_intraday_vwap(self, prices: pd.Series, volumes: pd.Series, index: pd.Index) -> Optional[pd.Series]:
        """Calculate intraday VWAP series with daily reset."""
        if not isinstance(index, pd.DatetimeIndex):
            return None
        
        try:
            df = pd.DataFrame({'price': prices, 'volume': volumes}, index=index)
            df['pv'] = df['price'] * df['volume']
            
            # Group by date and calculate intraday cumulative sums
            df['date'] = df.index.date
            df['pv_cumsum'] = df.groupby('date')['pv'].cumsum()
            df['volume_cumsum'] = df.groupby('date')['volume'].cumsum()
            
            # Calculate intraday VWAP
            intraday_vwap = df['pv_cumsum'] / df['volume_cumsum'].replace(0, np.nan)
            return intraday_vwap.fillna(method='ffill')
            
        except Exception as e:
            logger.warning(
                "Intraday VWAP calculation failed, using rolling VWAP",
                extra={"error": str(e)}
            )
            return None
    
    def _calculate_vwap_bands(self, prices: pd.Series, vwap_series: pd.Series) -> Dict[str, pd.Series]:
        """Calculate VWAP bands using standard deviation."""
        # Calculate rolling standard deviation of price deviations from VWAP
        price_dev = prices - vwap_series
        rolling_std = price_dev.rolling(window=self.rolling_periods, min_periods=1).std()
        
        bands = {}
        bands['upper_1'] = vwap_series + rolling_std
        bands['lower_1'] = vwap_series - rolling_std
        bands['upper_2'] = vwap_series + 2 * rolling_std
        bands['lower_2'] = vwap_series - 2 * rolling_std
        
        return bands
    
    def _calculate_realtime_bands(self, current_vwap: float) -> Dict[str, Optional[float]]:
        """Calculate VWAP bands for real-time updates."""
        if len(self._prices) < 10:  # Need minimum data for std calculation
            return {'upper_1': None, 'lower_1': None, 'upper_2': None, 'lower_2': None}
        
        # Calculate standard deviation of recent price deviations
        prices_array = np.array(self._prices)
        price_dev = prices_array - current_vwap
        std_dev = np.std(price_dev)
        
        return {
            'upper_1': current_vwap + std_dev,
            'lower_1': current_vwap - std_dev,
            'upper_2': current_vwap + 2 * std_dev,
            'lower_2': current_vwap - 2 * std_dev,
        }
    
    def _analyze_price_position(self, price: Optional[float], vwap: Optional[float], 
                              bands: Dict[str, Optional[float]]) -> str:
        """Analyze price position relative to VWAP and bands."""
        if price is None or vwap is None:
            return 'unknown'
        
        if bands.get('upper_2') and price > bands['upper_2']:
            return 'extreme_above'
        elif bands.get('upper_1') and price > bands['upper_1']:
            return 'above_upper'
        elif price > vwap:
            return 'above'
        elif bands.get('lower_1') and price < bands['lower_1']:
            return 'below_lower'
        elif bands.get('lower_2') and price < bands['lower_2']:
            return 'extreme_below'
        else:
            return 'at_vwap'
    
    def _generate_vwap_signal(self, price: Optional[float], vwap: Optional[float], 
                            bands: Dict[str, Optional[float]], volume: float) -> str:
        """Generate VWAP-based trading signal."""
        if price is None or vwap is None:
            return 'neutral'
        
        position = self._analyze_price_position(price, vwap, bands)
        
        # Determine volume strength
        avg_volume = np.mean(list(self._volumes)) if self._volumes else volume
        volume_strength = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Generate signals based on position and volume
        if position == 'extreme_above' and volume_strength > 1.5:
            return 'strong_sell'  # Likely reversal
        elif position == 'above_upper' and volume_strength > 1.2:
            return 'sell'
        elif position == 'above' and volume_strength > 1.0:
            return 'weak_buy'  # Above VWAP with volume
        elif position == 'below_lower' and volume_strength > 1.2:
            return 'buy'
        elif position == 'extreme_below' and volume_strength > 1.5:
            return 'strong_buy'  # Likely reversal
        else:
            return 'neutral'
    
    def _update_rolling_vwap(self, price: float, volume: float) -> None:
        """Update rolling VWAP state."""
        # Remove oldest values if at capacity
        if len(self._prices) == self.rolling_periods:
            old_price = self._prices[0]
            old_volume = self._volumes[0]
            self._pv_cumsum -= old_price * old_volume
            self._volume_cumsum -= old_volume
        
        # Add new values
        self._prices.append(price)
        self._volumes.append(volume)
        self._pv_cumsum += price * volume
        self._volume_cumsum += volume
    
    def _update_intraday_vwap(self, price: float, volume: float) -> None:
        """Update intraday VWAP state."""
        self._intraday_pv_sum += price * volume
        self._intraday_volume_sum += volume
    
    def _check_and_reset_intraday(self, timestamp: datetime) -> None:
        """Check if we need to reset intraday VWAP for new trading day."""
        current_date = timestamp.date()
        
        if self._last_reset_date is None or current_date != self._last_reset_date:
            self._intraday_pv_sum = 0.0
            self._intraday_volume_sum = 0.0
            self._last_reset_date = current_date
            
            logger.debug(
                "Intraday VWAP reset for new trading day",
                extra={"date": current_date}
            )
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'vwap': {
                'current': None,
                'upper_band_1': None,
                'lower_band_1': None,
                'upper_band_2': None,
                'lower_band_2': None,
                'position': 'unknown',
                'signal': 'neutral',
            },
            'vwap_series': pd.Series(),
            'bands_series': {},
            'calculation_time_ms': 0.0,
            'data_points': 0,
        }


class OnBalanceVolume:
    """
    On-Balance Volume (OBV) indicator with divergence detection.
    
    Features:
    - Traditional OBV calculation with volume accumulation
    - OBV momentum analysis (rising/falling trends)
    - OBV divergence detection with price
    - Rate of change calculation for volume pressure
    - Real-time volume flow analysis
    """
    
    def __init__(self, momentum_periods: int = 10, divergence_periods: int = 20) -> None:
        """
        Initialize OBV calculator.
        
        Args:
            momentum_periods: Periods for momentum calculation
            divergence_periods: Periods for divergence detection
        """
        self.momentum_periods = momentum_periods
        self.divergence_periods = divergence_periods
        
        # Real-time state
        self._current_obv: float = 0.0
        self._obv_history: deque = deque(maxlen=max(momentum_periods, divergence_periods) + 5)
        self._price_history: deque = deque(maxlen=max(momentum_periods, divergence_periods) + 5)
        
        logger.info(
            "OnBalanceVolume initialized",
            extra={
                "indicator": "obv",
                "momentum_periods": momentum_periods,
                "divergence_periods": divergence_periods,
            }
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate OBV values and analysis from historical data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with OBV values and analysis
        """
        start_time = time.perf_counter()
        
        if data.empty or 'close' not in data.columns or 'volume' not in data.columns:
            return self._empty_result()
        
        try:
            # Calculate OBV series
            obv_series = self._calculate_obv_series(data['close'], data['volume'])
            
            # Calculate momentum
            momentum = self._calculate_momentum(obv_series)
            
            # Detect trend
            trend = self._detect_trend(obv_series)
            
            # Detect divergences
            divergence = self._detect_divergence(data['close'], obv_series)
            
            # Current values
            current_obv = obv_series.iloc[-1] if not obv_series.empty else 0.0
            
            # Update internal state
            self._current_obv = current_obv
            self._update_history(data['close'], obv_series)
            
            # Performance tracking
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000
            
            logger.debug(
                "OBV calculation completed",
                extra={
                    "indicator": "obv",
                    "duration_ms": round(duration, 2),
                    "current_obv": current_obv,
                    "trend": trend,
                }
            )
            
            return {
                'obv': {
                    'value': current_obv,
                    'trend': trend,
                    'momentum': momentum,
                    'divergence': divergence,
                },
                'obv_series': obv_series,
                'calculation_time_ms': duration,
                'data_points': len(data),
            }
            
        except Exception as e:
            logger.error(
                "OBV calculation failed",
                extra={
                    "indicator": "obv",
                    "error_message": str(e),
                }
            )
            return self._empty_result()
    
    def update_realtime(self, price: float, volume: float, prev_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Update OBV with single price/volume tick.
        
        Args:
            price: Current price
            volume: Current volume
            prev_price: Previous price (uses last price from history if None)
            
        Returns:
            Dictionary with updated OBV analysis
        """
        if price <= 0 or volume < 0:
            return self._empty_result()
        
        try:
            # Determine previous price
            if prev_price is None and self._price_history:
                prev_price = self._price_history[-1]
            
            if prev_price is None:
                prev_price = price  # First update
            
            # Update OBV
            if price > prev_price:
                self._current_obv += volume
            elif price < prev_price:
                self._current_obv -= volume
            # If price == prev_price, OBV remains unchanged
            
            # Update histories
            self._price_history.append(price)
            self._obv_history.append(self._current_obv)
            
            # Calculate momentum and trend
            momentum = self._calculate_realtime_momentum()
            trend = self._detect_realtime_trend()
            divergence = self._detect_realtime_divergence()
            
            logger.debug(
                "Real-time OBV update completed",
                extra={
                    "indicator": "obv",
                    "price": price,
                    "volume": volume,
                    "obv": self._current_obv,
                    "trend": trend,
                }
            )
            
            return {
                'obv': {
                    'value': self._current_obv,
                    'trend': trend,
                    'momentum': momentum,
                    'divergence': divergence,
                },
                'price': price,
                'volume': volume,
            }
            
        except Exception as e:
            logger.error(
                "Real-time OBV update failed",
                extra={
                    "indicator": "obv",
                    "error_message": str(e),
                }
            )
            return self._empty_result()
    
    def _calculate_obv_series(self, prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Calculate OBV series from price and volume data."""
        price_changes = prices.diff()
        
        # Initialize OBV series
        obv = np.zeros(len(prices))
        obv[0] = volumes.iloc[0]  # Start with first volume
        
        for i in range(1, len(prices)):
            if price_changes.iloc[i] > 0:
                obv[i] = obv[i-1] + volumes.iloc[i]
            elif price_changes.iloc[i] < 0:
                obv[i] = obv[i-1] - volumes.iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=prices.index)
    
    def _calculate_momentum(self, obv_series: pd.Series) -> float:
        """Calculate OBV momentum."""
        if len(obv_series) < self.momentum_periods + 1:
            return 0.0
        
        # Rate of change over momentum periods
        current = obv_series.iloc[-1]
        past = obv_series.iloc[-(self.momentum_periods + 1)]
        
        if past != 0:
            momentum = (current - past) / abs(past)
        else:
            momentum = 0.0
        
        return momentum
    
    def _detect_trend(self, obv_series: pd.Series) -> str:
        """Detect OBV trend direction."""
        if len(obv_series) < 5:
            return 'flat'
        
        # Compare recent values
        recent_avg = obv_series.iloc[-5:].mean()
        earlier_avg = obv_series.iloc[-10:-5].mean() if len(obv_series) >= 10 else recent_avg
        
        if recent_avg > earlier_avg * 1.01:  # 1% threshold
            return 'rising'
        elif recent_avg < earlier_avg * 0.99:
            return 'falling'
        else:
            return 'flat'
    
    def _detect_divergence(self, prices: pd.Series, obv_series: pd.Series) -> Optional[str]:
        """Detect divergence between price and OBV."""
        if len(prices) < self.divergence_periods or len(obv_series) < self.divergence_periods:
            return None
        
        # Get recent data
        recent_prices = prices.iloc[-self.divergence_periods:]
        recent_obv = obv_series.iloc[-self.divergence_periods:]
        
        # Calculate trends
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        obv_trend = (recent_obv.iloc[-1] - recent_obv.iloc[0]) / abs(recent_obv.iloc[0]) if recent_obv.iloc[0] != 0 else 0
        
        # Detect divergence (opposite trends)
        if price_trend > 0.02 and obv_trend < -0.02:  # Price up, OBV down
            return 'bearish'
        elif price_trend < -0.02 and obv_trend > 0.02:  # Price down, OBV up
            return 'bullish'
        else:
            return None
    
    def _calculate_realtime_momentum(self) -> float:
        """Calculate momentum from real-time history."""
        if len(self._obv_history) < self.momentum_periods + 1:
            return 0.0
        
        current = self._obv_history[-1]
        past = self._obv_history[-(self.momentum_periods + 1)]
        
        if past != 0:
            return (current - past) / abs(past)
        else:
            return 0.0
    
    def _detect_realtime_trend(self) -> str:
        """Detect trend from real-time history."""
        if len(self._obv_history) < 5:
            return 'flat'
        
        recent_values = list(self._obv_history)[-5:]
        earlier_values = list(self._obv_history)[-10:-5] if len(self._obv_history) >= 10 else recent_values
        
        recent_avg = np.mean(recent_values)
        earlier_avg = np.mean(earlier_values)
        
        if recent_avg > earlier_avg * 1.01:
            return 'rising'
        elif recent_avg < earlier_avg * 0.99:
            return 'falling'
        else:
            return 'flat'
    
    def _detect_realtime_divergence(self) -> Optional[str]:
        """Detect divergence from real-time history."""
        if len(self._price_history) < self.divergence_periods or len(self._obv_history) < self.divergence_periods:
            return None
        
        recent_prices = list(self._price_history)[-self.divergence_periods:]
        recent_obv = list(self._obv_history)[-self.divergence_periods:]
        
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        obv_trend = (recent_obv[-1] - recent_obv[0]) / abs(recent_obv[0]) if recent_obv[0] != 0 else 0
        
        if price_trend > 0.02 and obv_trend < -0.02:
            return 'bearish'
        elif price_trend < -0.02 and obv_trend > 0.02:
            return 'bullish'
        else:
            return None
    
    def _update_history(self, prices: pd.Series, obv_series: pd.Series) -> None:
        """Update internal history for real-time calculations."""
        if not prices.empty and not obv_series.empty:
            self._price_history.extend(prices.tolist())
            self._obv_history.extend(obv_series.tolist())
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'obv': {
                'value': 0.0,
                'trend': 'flat',
                'momentum': 0.0,
                'divergence': None,
            },
            'obv_series': pd.Series(),
            'calculation_time_ms': 0.0,
            'data_points': 0,
        }


class VolumeMovingAverage:
    """
    Volume Moving Average analysis with spike detection.
    
    Features:
    - Multiple volume MA periods: [5, 10, 20] for scalping
    - Volume spike detection (>200% of average)
    - Volume trend analysis (increasing/decreasing)
    - Relative volume strength vs historical patterns
    - Abnormal volume alerts
    """
    
    def __init__(self, periods: Optional[List[int]] = None, spike_threshold: float = 2.0) -> None:
        """
        Initialize Volume MA analyzer.
        
        Args:
            periods: List of MA periods (default: [5, 10, 20])
            spike_threshold: Multiplier for spike detection (default: 2.0 = 200%)
        """
        self.periods = periods or [5, 10, 20]
        self.spike_threshold = spike_threshold
        
        # Real-time state
        self._volume_history: deque = deque(maxlen=max(self.periods) + 10)
        self._current_mas: Dict[int, float] = {}
        
        logger.info(
            "VolumeMovingAverage initialized",
            extra={
                "indicator": "volume_ma",
                "periods": self.periods,
                "spike_threshold": spike_threshold,
            }
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate volume MA values and analysis.
        
        Args:
            data: DataFrame with volume data
            
        Returns:
            Dictionary with volume MA analysis
        """
        start_time = time.perf_counter()
        
        if data.empty or 'volume' not in data.columns:
            return self._empty_result()
        
        try:
            volume = data['volume']
            
            # Calculate all volume MAs
            volume_mas = {}
            for period in self.periods:
                volume_mas[period] = volume.rolling(window=period, min_periods=1).mean()
            
            # Current values
            current_volume = volume.iloc[-1]
            current_mas = {period: ma.iloc[-1] for period, ma in volume_mas.items()}
            
            # Relative strength analysis
            relative_strength = self._calculate_relative_strength(current_volume, current_mas)
            
            # Spike detection
            spike_detected = self._detect_volume_spike(current_volume, current_mas)
            
            # Trend analysis
            trend = self._analyze_volume_trend(volume_mas)
            
            # Update internal state
            self._current_mas = current_mas
            self._volume_history.extend(volume.tolist())
            
            # Performance tracking
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000
            
            logger.debug(
                "Volume MA calculation completed",
                extra={
                    "indicator": "volume_ma",
                    "duration_ms": round(duration, 2),
                    "current_volume": current_volume,
                    "spike_detected": spike_detected,
                }
            )
            
            return {
                'volume_ma': {
                    'current': current_volume,
                    **{f'ma_{period}': current_mas[period] for period in self.periods},
                    'relative_strength': relative_strength,
                    'spike_detected': spike_detected,
                    'trend': trend,
                },
                'volume_ma_series': volume_mas,
                'calculation_time_ms': duration,
                'data_points': len(data),
            }
            
        except Exception as e:
            logger.error(
                "Volume MA calculation failed",
                extra={
                    "indicator": "volume_ma",
                    "error_message": str(e),
                }
            )
            return self._empty_result()
    
    def update_realtime(self, volume: float) -> Dict[str, Any]:
        """
        Update volume analysis with single volume tick.
        
        Args:
            volume: Latest volume value
            
        Returns:
            Dictionary with updated volume analysis
        """
        if volume < 0:
            return self._empty_result()
        
        try:
            # Add to history
            self._volume_history.append(volume)
            
            # Calculate current MAs
            current_mas = {}
            for period in self.periods:
                if len(self._volume_history) >= period:
                    recent_volumes = list(self._volume_history)[-period:]
                    current_mas[period] = np.mean(recent_volumes)
                else:
                    current_mas[period] = np.mean(list(self._volume_history))
            
            # Update internal state
            self._current_mas = current_mas
            
            # Analysis
            relative_strength = self._calculate_relative_strength(volume, current_mas)
            spike_detected = self._detect_volume_spike(volume, current_mas)
            trend = self._analyze_realtime_trend()
            
            logger.debug(
                "Real-time volume MA update completed",
                extra={
                    "indicator": "volume_ma",
                    "volume": volume,
                    "spike_detected": spike_detected,
                    "relative_strength": relative_strength,
                }
            )
            
            return {
                'volume_ma': {
                    'current': volume,
                    **{f'ma_{period}': current_mas[period] for period in self.periods},
                    'relative_strength': relative_strength,
                    'spike_detected': spike_detected,
                    'trend': trend,
                },
                'volume': volume,
            }
            
        except Exception as e:
            logger.error(
                "Real-time volume MA update failed",
                extra={
                    "indicator": "volume_ma",
                    "error_message": str(e),
                }
            )
            return self._empty_result()
    
    def _calculate_relative_strength(self, current_volume: float, mas: Dict[int, float]) -> float:
        """Calculate relative volume strength vs 20-period average."""
        if not mas or 20 not in mas or mas[20] == 0:
            return 1.0
        
        return current_volume / mas[20]
    
    def _detect_volume_spike(self, current_volume: float, mas: Dict[int, float]) -> bool:
        """Detect volume spike vs 20-period average."""
        if not mas or 20 not in mas or mas[20] == 0:
            return False
        
        return current_volume > (mas[20] * self.spike_threshold)
    
    def _analyze_volume_trend(self, volume_mas: Dict[int, pd.Series]) -> str:
        """Analyze volume trend from MA series."""
        if not volume_mas or 10 not in volume_mas:
            return 'flat'
        
        ma_10 = volume_mas[10]
        if len(ma_10) < 5:
            return 'flat'
        
        # Compare recent vs earlier averages
        recent_avg = ma_10.iloc[-3:].mean()
        earlier_avg = ma_10.iloc[-6:-3].mean() if len(ma_10) >= 6 else recent_avg
        
        if recent_avg > earlier_avg * 1.1:  # 10% threshold
            return 'increasing'
        elif recent_avg < earlier_avg * 0.9:
            return 'decreasing'
        else:
            return 'flat'
    
    def _analyze_realtime_trend(self) -> str:
        """Analyze trend from real-time history."""
        if len(self._volume_history) < 10:
            return 'flat'
        
        volumes = list(self._volume_history)
        recent_avg = np.mean(volumes[-5:])
        earlier_avg = np.mean(volumes[-10:-5])
        
        if recent_avg > earlier_avg * 1.1:
            return 'increasing'
        elif recent_avg < earlier_avg * 0.9:
            return 'decreasing'
        else:
            return 'flat'
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'volume_ma': {
                'current': 0.0,
                **{f'ma_{period}': 0.0 for period in self.periods},
                'relative_strength': 1.0,
                'spike_detected': False,
                'trend': 'flat',
            },
            'volume_ma_series': {},
            'calculation_time_ms': 0.0,
            'data_points': 0,
        }


class VolumeProfile:
    """
    Real-time volume profile analysis.
    
    Features:
    - Real-time volume profile for last 100 periods
    - Point of Control (POC) identification
    - High/Low volume nodes detection
    - Volume concentration analysis
    - Support/resistance level validation
    """
    
    def __init__(self, profile_periods: int = 100, price_bins: int = 20, 
                 node_threshold: float = 1.5) -> None:
        """
        Initialize Volume Profile analyzer.
        
        Args:
            profile_periods: Number of periods to include in profile
            price_bins: Number of price bins for volume distribution
            node_threshold: Multiplier for high/low volume node detection
        """
        self.profile_periods = profile_periods
        self.price_bins = price_bins
        self.node_threshold = node_threshold
        
        # Real-time state
        self._price_volume_data: deque = deque(maxlen=profile_periods)
        self._current_profile: Dict[str, Any] = {}
        
        logger.info(
            "VolumeProfile initialized",
            extra={
                "indicator": "volume_profile",
                "profile_periods": profile_periods,
                "price_bins": price_bins,
                "node_threshold": node_threshold,
            }
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate volume profile analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume profile analysis
        """
        start_time = time.perf_counter()
        
        if data.empty or 'close' not in data.columns or 'volume' not in data.columns:
            return self._empty_result()
        
        try:
            # Use typical price if available
            if 'high' in data.columns and 'low' in data.columns:
                typical_price = (data['high'] + data['low'] + data['close']) / 3
            else:
                typical_price = data['close']
            
            volume = data['volume']
            
            # Get recent data for profile
            recent_data = data.tail(self.profile_periods)
            if len(recent_data) > 0:
                recent_prices = typical_price.tail(self.profile_periods)
                recent_volumes = volume.tail(self.profile_periods)
            else:
                recent_prices = typical_price
                recent_volumes = volume
            
            # Calculate volume profile
            profile = self._calculate_volume_profile(recent_prices, recent_volumes)
            
            # Update internal state
            self._current_profile = profile
            self._price_volume_data.extend(list(zip(recent_prices.tolist(), recent_volumes.tolist())))
            
            # Performance tracking
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000
            
            logger.debug(
                "Volume profile calculation completed",
                extra={
                    "indicator": "volume_profile",
                    "duration_ms": round(duration, 2),
                    "poc": profile.get('poc'),
                    "high_volume_nodes": len(profile.get('high_volume_nodes', [])),
                }
            )
            
            return {
                'volume_profile': profile,
                'calculation_time_ms': duration,
                'data_points': len(recent_data),
            }
            
        except Exception as e:
            logger.error(
                "Volume profile calculation failed",
                extra={
                    "indicator": "volume_profile",
                    "error_message": str(e),
                }
            )
            return self._empty_result()
    
    def update_realtime(self, price: float, volume: float) -> Dict[str, Any]:
        """
        Update volume profile with single price/volume tick.
        
        Args:
            price: Latest price
            volume: Latest volume
            
        Returns:
            Dictionary with updated volume profile
        """
        if price <= 0 or volume < 0:
            return self._empty_result()
        
        try:
            # Add to data
            self._price_volume_data.append((price, volume))
            
            # Recalculate profile
            if len(self._price_volume_data) > 5:  # Need minimum data
                prices = [pv[0] for pv in self._price_volume_data]
                volumes = [pv[1] for pv in self._price_volume_data]
                
                prices_series = pd.Series(prices)
                volumes_series = pd.Series(volumes)
                
                profile = self._calculate_volume_profile(prices_series, volumes_series)
                self._current_profile = profile
                
                logger.debug(
                    "Real-time volume profile update completed",
                    extra={
                        "indicator": "volume_profile",
                        "price": price,
                        "volume": volume,
                        "poc": profile.get('poc'),
                    }
                )
                
                return {
                    'volume_profile': profile,
                    'price': price,
                    'volume': volume,
                }
            
            return self._empty_result()
            
        except Exception as e:
            logger.error(
                "Real-time volume profile update failed",
                extra={
                    "indicator": "volume_profile",
                    "error_message": str(e),
                }
            )
            return self._empty_result()
    
    def _calculate_volume_profile(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, Any]:
        """Calculate volume profile from price and volume data."""
        if len(prices) == 0 or len(volumes) == 0:
            return {'poc': None, 'high_volume_nodes': [], 'low_volume_nodes': [], 'concentration': 0.0}
        
        # Create price bins
        price_min = prices.min()
        price_max = prices.max()
        
        if price_max == price_min:
            # Single price point
            return {
                'poc': price_min,
                'high_volume_nodes': [price_min],
                'low_volume_nodes': [],
                'concentration': 1.0,
            }
        
        # Create bins
        bin_edges = np.linspace(price_min, price_max, self.price_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Assign volumes to bins
        bin_volumes = np.zeros(self.price_bins)
        
        for price, volume in zip(prices, volumes):
            # Find appropriate bin
            bin_idx = np.searchsorted(bin_edges[1:], price)
            bin_idx = min(bin_idx, self.price_bins - 1)  # Ensure within bounds
            bin_volumes[bin_idx] += volume
        
        # Find Point of Control (highest volume bin)
        poc_idx = np.argmax(bin_volumes)
        poc = bin_centers[poc_idx]
        
        # Identify high and low volume nodes
        avg_volume = np.mean(bin_volumes)
        high_volume_threshold = avg_volume * self.node_threshold
        low_volume_threshold = avg_volume / self.node_threshold
        
        high_volume_nodes = []
        low_volume_nodes = []
        
        for i, volume in enumerate(bin_volumes):
            if volume > high_volume_threshold:
                high_volume_nodes.append(bin_centers[i])
            elif volume < low_volume_threshold and volume > 0:
                low_volume_nodes.append(bin_centers[i])
        
        # Calculate concentration (how concentrated volume is)
        total_volume = np.sum(bin_volumes)
        if total_volume > 0:
            volume_distribution = bin_volumes / total_volume
            # Calculate entropy-based concentration
            non_zero_dist = volume_distribution[volume_distribution > 0]
            entropy = -np.sum(non_zero_dist * np.log2(non_zero_dist))
            max_entropy = np.log2(len(non_zero_dist))
            concentration = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
        else:
            concentration = 0.0
        
        return {
            'poc': poc,
            'high_volume_nodes': high_volume_nodes,
            'low_volume_nodes': low_volume_nodes,
            'concentration': concentration,
            'bin_volumes': bin_volumes,
            'bin_centers': bin_centers,
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'volume_profile': {
                'poc': None,
                'high_volume_nodes': [],
                'low_volume_nodes': [],
                'concentration': 0.0,
            },
            'calculation_time_ms': 0.0,
            'data_points': 0,
        }


class ScalpingVolumeSignals:
    """
    Combined volume indicator signals for scalping strategies.
    
    Features:
    - Combine all volume indicators for consensus
    - Volume confirmation for price movements
    - Volume-based signal filtering
    - Volume momentum scoring
    """
    
    def __init__(self) -> None:
        """Initialize the combined volume signals analyzer."""
        self.vwap = ScalpingVWAP()
        self.obv = OnBalanceVolume()
        self.volume_ma = VolumeMovingAverage()
        self.volume_profile = VolumeProfile()
        
        logger.info(
            "ScalpingVolumeSignals initialized",
            extra={"indicator": "scalping_volume_signals"}
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive volume analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all volume analysis and consensus
        """
        start_time = time.perf_counter()
        
        try:
            # Calculate all volume indicators
            vwap_result = self.vwap.calculate(data)
            obv_result = self.obv.calculate(data)
            volume_ma_result = self.volume_ma.calculate(data)
            volume_profile_result = self.volume_profile.calculate(data)
            
            # Generate consensus analysis
            consensus = self._generate_consensus(
                vwap_result, obv_result, volume_ma_result, volume_profile_result
            )
            
            # Performance tracking
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000
            
            logger.debug(
                "Volume signals calculation completed",
                extra={
                    "indicator": "scalping_volume_signals",
                    "duration_ms": round(duration, 2),
                    "consensus_strength": consensus.get('strength', 0.0),
                }
            )
            
            return {
                **vwap_result,
                **obv_result,
                **volume_ma_result,
                **volume_profile_result,
                'consensus': consensus,
                'total_calculation_time_ms': duration,
            }
            
        except Exception as e:
            logger.error(
                "Volume signals calculation failed",
                extra={
                    "indicator": "scalping_volume_signals",
                    "error_message": str(e),
                }
            )
            return self._empty_consensus_result()
    
    def get_volume_confirmation(self, price_signal: str) -> Dict[str, Any]:
        """
        Get volume confirmation for a price signal.
        
        Args:
            price_signal: Price signal to confirm ('buy', 'sell', 'strong_buy', 'strong_sell')
            
        Returns:
            Dictionary with confirmation analysis
        """
        # This would use the latest calculated results
        # For now, return a basic structure
        return {
            'confirmed': False,
            'confidence': 0.5,
            'supporting_indicators': [],
            'conflicting_indicators': [],
        }
    
    def detect_volume_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect unusual volume patterns.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check for volume spikes
        if hasattr(self.volume_ma, '_current_mas') and self.volume_ma._current_mas:
            # Implementation would check various anomaly patterns
            pass
        
        return anomalies
    
    def get_volume_strength(self) -> float:
        """
        Get overall volume strength score.
        
        Returns:
            Volume strength from 0.0 to 1.0
        """
        # This would combine all indicators to produce a strength score
        return 0.5  # Placeholder
    
    def is_breakout_valid(self, price_breakout: Dict[str, Any]) -> bool:
        """
        Validate price breakout with volume confirmation.
        
        Args:
            price_breakout: Dictionary with breakout information
            
        Returns:
            True if breakout is volume-confirmed
        """
        # Implementation would check:
        # - Volume spike during breakout
        # - OBV confirmation
        # - VWAP position
        # - Volume profile support/resistance
        return False  # Placeholder
    
    def _generate_consensus(self, vwap_result: Dict[str, Any], obv_result: Dict[str, Any],
                          volume_ma_result: Dict[str, Any], volume_profile_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consensus analysis from all volume indicators."""
        supporting_indicators = []
        conflicting_indicators = []
        
        # Analyze VWAP signal
        vwap_signal = vwap_result.get('vwap', {}).get('signal', 'neutral')
        if vwap_signal in ['buy', 'strong_buy']:
            supporting_indicators.append('vwap')
        elif vwap_signal in ['sell', 'strong_sell']:
            conflicting_indicators.append('vwap')
        
        # Analyze OBV signal
        obv_trend = obv_result.get('obv', {}).get('trend', 'flat')
        if obv_trend == 'rising':
            supporting_indicators.append('obv')
        elif obv_trend == 'falling':
            conflicting_indicators.append('obv')
        
        # Analyze volume MA
        volume_spike = volume_ma_result.get('volume_ma', {}).get('spike_detected', False)
        if volume_spike:
            supporting_indicators.append('volume_ma')
        
        # Calculate overall strength
        total_indicators = len(supporting_indicators) + len(conflicting_indicators)
        if total_indicators > 0:
            strength = len(supporting_indicators) / total_indicators
        else:
            strength = 0.5
        
        # Determine confirmation level
        if strength >= 0.8:
            confirmation = 'strong'
        elif strength >= 0.6:
            confirmation = 'moderate'
        elif strength >= 0.4:
            confirmation = 'weak'
        else:
            confirmation = 'negative'
        
        return {
            'confirmation': confirmation,
            'strength': strength,
            'supporting_indicators': supporting_indicators,
            'conflicting_indicators': conflicting_indicators,
            'anomalies': [],
        }
    
    def _empty_consensus_result(self) -> Dict[str, Any]:
        """Return empty consensus result."""
        return {
            'vwap': {'current': None, 'signal': 'neutral'},
            'obv': {'value': 0.0, 'trend': 'flat'},
            'volume_ma': {'current': 0.0, 'spike_detected': False},
            'volume_profile': {'poc': None},
            'consensus': {
                'confirmation': 'none',
                'strength': 0.0,
                'supporting_indicators': [],
                'conflicting_indicators': [],
                'anomalies': [],
            },
            'total_calculation_time_ms': 0.0,
        }


# Export all classes
__all__ = [
    "ScalpingVWAP",
    "OnBalanceVolume", 
    "VolumeMovingAverage",
    "VolumeProfile",
    "ScalpingVolumeSignals",
]