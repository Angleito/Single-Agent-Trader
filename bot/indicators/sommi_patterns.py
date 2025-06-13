"""
Sommi Flag and Diamond Patterns implementation with Higher Timeframe Analysis.

This module provides advanced pattern recognition for the Sommi Flag and Diamond patterns
from the VuManChu Cipher B indicator. These patterns use higher timeframe analysis 
and Heikin Ashi candle direction detection for enhanced signal quality.

Based on Pine Script patterns:
- Sommi Flag: Uses HTF WaveTrend, RSI+MFI, and VWAP conditions
- Sommi Diamond: Uses HTF WaveTrend and dual timeframe Heikin Ashi confirmation
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd
import pandas_ta as ta
from numpy.typing import NDArray

from .wavetrend import WaveTrend
from .rsimfi import RSIMFIIndicator

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Pattern type enumeration."""
    SOMMI_FLAG_BULL = "sommi_flag_bull"
    SOMMI_FLAG_BEAR = "sommi_flag_bear"
    SOMMI_DIAMOND_BULL = "sommi_diamond_bull"
    SOMMI_DIAMOND_BEAR = "sommi_diamond_bear"


@dataclass
class SommiLevels:
    """Configuration for Sommi pattern levels."""
    # Sommi Flag levels
    rsimfi_bear_level: float = -40.0
    rsimfi_bull_level: float = 40.0
    flag_wt_bear_level: float = 30.0
    flag_wt_bull_level: float = -30.0
    vwap_bear_level: float = 0.0
    vwap_bull_level: float = 0.0
    
    # Sommi Diamond levels
    diamond_wt_bear_level: float = 45.0
    diamond_wt_bull_level: float = -45.0


@dataclass
class PatternSignal:
    """Represents a pattern detection signal."""
    pattern_type: PatternType
    confidence: float
    timestamp: pd.Timestamp
    wt1: Optional[float] = None
    wt2: Optional[float] = None
    rsimfi: Optional[float] = None
    htf_vwap: Optional[float] = None
    candle_alignment: Optional[bool] = None
    description: str = ""


class SommiPatterns:
    """
    Advanced Sommi Flag and Diamond pattern detector with Higher Timeframe Analysis.
    
    This class implements the sophisticated pattern recognition logic from VuManChu Cipher B:
    
    Sommi Flag Pattern:
    - Uses higher timeframe WaveTrend analysis
    - Requires RSI+MFI condition alignment
    - Includes VWAP higher timeframe confirmation
    - Cross pattern validation with WaveTrend signals
    
    Sommi Diamond Pattern:
    - Higher timeframe WaveTrend extreme levels
    - Dual timeframe Heikin Ashi candle direction alignment
    - Cross confirmation with WaveTrend momentum
    - Enhanced pattern validation
    
    Features:
    - Higher timeframe data simulation
    - Heikin Ashi candle processing for multiple timeframes
    - Pattern confidence scoring
    - Signal timing validation
    - Integration with existing WaveTrend and RSI+MFI indicators
    """

    def __init__(
        self,
        levels: Optional[SommiLevels] = None,
        htf_multiplier: int = 4,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        Initialize Sommi patterns detector.

        Args:
            levels: Custom pattern levels configuration
            htf_multiplier: Higher timeframe multiplier (4 = 4x base timeframe)
            confidence_threshold: Minimum confidence for pattern signals
        """
        self.levels = levels or SommiLevels()
        self.htf_multiplier = htf_multiplier
        self.confidence_threshold = confidence_threshold
        
        # Initialize component indicators
        self.wavetrend = WaveTrend()
        self.rsimfi = RSIMFIIndicator()
        
        # Cache for higher timeframe data
        self._htf_cache: Dict[str, pd.DataFrame] = {}
        
        # Performance tracking
        self._pattern_detection_count = 0
        self._flag_pattern_count = 0
        self._diamond_pattern_count = 0
        self._total_calculation_time = 0.0
        
        logger.info("Sommi patterns detector initialized", extra={
            "indicator": "sommi_patterns",
            "parameters": {
                "htf_multiplier": htf_multiplier,
                "confidence_threshold": confidence_threshold,
                "levels": {
                    "rsimfi_bear_level": self.levels.rsimfi_bear_level,
                    "rsimfi_bull_level": self.levels.rsimfi_bull_level,
                    "flag_wt_bear_level": self.levels.flag_wt_bear_level,
                    "flag_wt_bull_level": self.levels.flag_wt_bull_level,
                    "diamond_wt_bear_level": self.levels.diamond_wt_bear_level,
                    "diamond_wt_bull_level": self.levels.diamond_wt_bull_level
                }
            }
        })

    def calculate_higher_timeframe_wt(
        self, 
        price_data: pd.DataFrame, 
        timeframe_multiplier: Optional[int] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate higher timeframe WaveTrend with VWAP.
        
        Simulates higher timeframe analysis by resampling data and calculating
        WaveTrend on the higher timeframe, then forward-filling to match
        the original timeframe.
        
        Args:
            price_data: OHLCV DataFrame with datetime index
            timeframe_multiplier: HTF multiplier override
            
        Returns:
            Tuple of (htf_wt1, htf_wt2, htf_vwap) series aligned to original timeframe
        """
        multiplier = timeframe_multiplier or self.htf_multiplier
        
        try:
            # Validate input data
            if not isinstance(price_data.index, pd.DatetimeIndex):
                logger.warning("Price data index is not DatetimeIndex, attempting conversion")
                price_data = price_data.copy()
                price_data.index = pd.to_datetime(price_data.index)
            
            # Determine resampling rule based on original frequency
            freq = pd.infer_freq(price_data.index)
            if freq is None:
                # Default to 1-minute base with multiplier
                resample_rule = f"{multiplier}T"
                logger.info(f"Could not infer frequency, using {resample_rule}")
            else:
                # Extract numeric part and unit
                import re
                match = re.match(r'(\d*)([A-Z]+)', freq)
                if match:
                    num_part = int(match.group(1)) if match.group(1) else 1
                    unit_part = match.group(2)
                    new_num = num_part * multiplier
                    resample_rule = f"{new_num}{unit_part}"
                else:
                    resample_rule = f"{multiplier}T"
                    
            logger.debug(f"Resampling to higher timeframe: {resample_rule}")
            
            # Resample OHLCV data to higher timeframe
            htf_data = price_data.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(htf_data) < 50:  # Need minimum data for WaveTrend calculation
                logger.warning("Insufficient HTF data for WaveTrend calculation")
                return self._empty_htf_series(price_data.index)
            
            # Calculate WaveTrend on higher timeframe
            htf_wt1, htf_wt2 = self.wavetrend.calculate_wavetrend(htf_data['close'])
            
            # Calculate VWAP on higher timeframe
            htf_vwap = self._calculate_vwap(htf_data)
            
            # Forward fill to match original timeframe
            htf_wt1_aligned = htf_wt1.reindex(price_data.index, method='ffill')
            htf_wt2_aligned = htf_wt2.reindex(price_data.index, method='ffill')
            htf_vwap_aligned = htf_vwap.reindex(price_data.index, method='ffill')
            
            return htf_wt1_aligned, htf_wt2_aligned, htf_vwap_aligned
            
        except Exception as e:
            logger.error(f"Error calculating higher timeframe WaveTrend: {e}")
            return self._empty_htf_series(price_data.index)

    def _calculate_vwap(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            ohlcv_data: OHLCV DataFrame
            
        Returns:
            VWAP series
        """
        try:
            typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
            vwap = (typical_price * ohlcv_data['volume']).cumsum() / ohlcv_data['volume'].cumsum()
            return vwap.fillna(0)
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return pd.Series(0, index=ohlcv_data.index)

    def _empty_htf_series(self, index: pd.Index) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Return empty series tuple for error cases."""
        empty = pd.Series(dtype=float, index=index)
        return empty, empty, empty

    def calculate_heikin_ashi_candles(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin Ashi candles for candle direction analysis.
        
        Heikin Ashi formula:
        - HA_Close = (O + H + L + C) / 4
        - HA_Open = (previous HA_Open + previous HA_Close) / 2  
        - HA_High = max(H, HA_Open, HA_Close)
        - HA_Low = min(L, HA_Open, HA_Close)
        
        Args:
            ohlc_data: OHLC DataFrame
            
        Returns:
            DataFrame with Heikin Ashi OHLC and candle direction
        """
        try:
            if len(ohlc_data) < 2:
                logger.warning("Insufficient data for Heikin Ashi calculation")
                return pd.DataFrame(index=ohlc_data.index)
            
            ha_data = pd.DataFrame(index=ohlc_data.index)
            
            # Calculate Heikin Ashi Close
            ha_data['ha_close'] = (
                ohlc_data['open'] + ohlc_data['high'] + 
                ohlc_data['low'] + ohlc_data['close']
            ) / 4
            
            # Calculate Heikin Ashi Open
            ha_data['ha_open'] = np.nan
            ha_data.iloc[0, ha_data.columns.get_loc('ha_open')] = (
                ohlc_data.iloc[0]['open'] + ohlc_data.iloc[0]['close']
            ) / 2
            
            for i in range(1, len(ha_data)):
                ha_data.iloc[i, ha_data.columns.get_loc('ha_open')] = (
                    ha_data.iloc[i-1]['ha_open'] + ha_data.iloc[i-1]['ha_close']
                ) / 2
            
            # Calculate Heikin Ashi High and Low
            ha_data['ha_high'] = np.maximum.reduce([
                ohlc_data['high'], ha_data['ha_open'], ha_data['ha_close']
            ])
            ha_data['ha_low'] = np.minimum.reduce([
                ohlc_data['low'], ha_data['ha_open'], ha_data['ha_close']
            ])
            
            # Calculate candle direction (body direction)
            ha_data['candle_body_dir'] = ha_data['ha_close'] > ha_data['ha_open']
            ha_data['is_bullish'] = ha_data['candle_body_dir']
            ha_data['is_bearish'] = ~ha_data['candle_body_dir']
            
            # Calculate body size for confidence
            ha_data['body_size'] = abs(ha_data['ha_close'] - ha_data['ha_open'])
            ha_data['wick_size'] = (ha_data['ha_high'] - ha_data['ha_low']) - ha_data['body_size']
            ha_data['body_ratio'] = ha_data['body_size'] / (ha_data['ha_high'] - ha_data['ha_low'] + 1e-10)
            
            return ha_data
            
        except Exception as e:
            logger.error(f"Error calculating Heikin Ashi candles: {e}")
            return pd.DataFrame(index=ohlc_data.index)

    def calculate_sommi_flags(
        self,
        wt_data: Dict[str, pd.Series],
        rsimfi_data: pd.Series,
        htf_wt_data: Dict[str, pd.Series],
        levels: Optional[SommiLevels] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate Sommi Flag patterns.
        
        Pine Script logic:
        bearPattern = rsimfi < soomiRSIMFIBearLevel and wt2 > soomiFlagWTBearLevel 
                     and wtCross and wtCrossDown and hwtVwap < sommiVwapBearLevel
        bullPattern = rsimfi > soomiRSIMFIBullLevel and wt2 < soomiFlagWTBullLevel 
                     and wtCross and wtCrossUp and hwtVwap > sommiVwapBullLevel
        
        Args:
            wt_data: Dict with 'wt1', 'wt2', 'cross_up', 'cross_down' series
            rsimfi_data: RSI+MFI series
            htf_wt_data: Dict with HTF 'wt1', 'wt2', 'vwap' series
            levels: Pattern levels override
            
        Returns:
            Dict with 'bear_flag', 'bull_flag', 'confidence' series
        """
        config = levels or self.levels
        
        try:
            # Extract required data
            wt2 = wt_data.get('wt2', pd.Series(dtype=float))
            wt_cross_up = wt_data.get('cross_up', pd.Series(dtype=bool))
            wt_cross_down = wt_data.get('cross_down', pd.Series(dtype=bool))
            wt_cross = wt_cross_up | wt_cross_down
            
            htf_vwap = htf_wt_data.get('vwap', pd.Series(dtype=float))
            
            if any(s.empty for s in [wt2, rsimfi_data, htf_vwap]):
                logger.warning("Empty data provided for Sommi Flag calculation")
                return self._empty_flag_result(wt2.index if not wt2.empty else pd.Index([]))
            
            # Sommi Flag Bear Pattern
            bear_rsimfi_cond = rsimfi_data < config.rsimfi_bear_level
            bear_wt2_cond = wt2 > config.flag_wt_bear_level
            bear_cross_cond = wt_cross & wt_cross_down
            bear_vwap_cond = htf_vwap < config.vwap_bear_level
            
            bear_flag = bear_rsimfi_cond & bear_wt2_cond & bear_cross_cond & bear_vwap_cond
            
            # Sommi Flag Bull Pattern
            bull_rsimfi_cond = rsimfi_data > config.rsimfi_bull_level
            bull_wt2_cond = wt2 < config.flag_wt_bull_level
            bull_cross_cond = wt_cross & wt_cross_up
            bull_vwap_cond = htf_vwap > config.vwap_bull_level
            
            bull_flag = bull_rsimfi_cond & bull_wt2_cond & bull_cross_cond & bull_vwap_cond
            
            # Calculate confidence based on condition strength
            bear_confidence = self._calculate_flag_confidence(
                bear_rsimfi_cond, bear_wt2_cond, bear_cross_cond, bear_vwap_cond
            )
            bull_confidence = self._calculate_flag_confidence(
                bull_rsimfi_cond, bull_wt2_cond, bull_cross_cond, bull_vwap_cond
            )
            
            # Combined confidence
            confidence = pd.Series(0.0, index=wt2.index)
            confidence[bear_flag] = bear_confidence[bear_flag]
            confidence[bull_flag] = bull_confidence[bull_flag]
            
            return {
                'bear_flag': bear_flag,
                'bull_flag': bull_flag,
                'confidence': confidence,
                'bear_confidence': bear_confidence,
                'bull_confidence': bull_confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating Sommi flags: {e}")
            return self._empty_flag_result(pd.Index([]))

    def _calculate_flag_confidence(
        self, 
        cond1: pd.Series, 
        cond2: pd.Series, 
        cond3: pd.Series, 
        cond4: pd.Series
    ) -> pd.Series:
        """Calculate confidence score for flag patterns."""
        # Weight each condition
        weights = [0.3, 0.25, 0.25, 0.2]  # RSI+MFI, WT2, Cross, VWAP
        conditions = [cond1, cond2, cond3, cond4]
        
        confidence = pd.Series(0.0, index=cond1.index)
        for condition, weight in zip(conditions, weights):
            confidence += condition.astype(float) * weight
            
        return confidence

    def _empty_flag_result(self, index: pd.Index) -> Dict[str, pd.Series]:
        """Return empty flag result for error cases."""
        empty_bool = pd.Series(False, index=index)
        empty_float = pd.Series(0.0, index=index)
        return {
            'bear_flag': empty_bool,
            'bull_flag': empty_bool,
            'confidence': empty_float,
            'bear_confidence': empty_float,
            'bull_confidence': empty_float
        }

    def calculate_sommi_diamonds(
        self,
        wt_data: Dict[str, pd.Series],
        ha_candles_tf1: pd.DataFrame,
        ha_candles_tf2: pd.DataFrame,
        levels: Optional[SommiLevels] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate Sommi Diamond patterns.
        
        Pine Script logic:
        bearPattern = wt2 >= soomiDiamondWTBearLevel and wtCross and wtCrossDown 
                     and not candleBodyDir and not candleBodyDir2
        bullPattern = wt2 <= soomiDiamondWTBullLevel and wtCross and wtCrossUp 
                     and candleBodyDir and candleBodyDir2
        
        Args:
            wt_data: Dict with WaveTrend data
            ha_candles_tf1: Heikin Ashi candles for timeframe 1
            ha_candles_tf2: Heikin Ashi candles for timeframe 2 (higher)
            levels: Pattern levels override
            
        Returns:
            Dict with 'bear_diamond', 'bull_diamond', 'confidence' series
        """
        config = levels or self.levels
        
        try:
            # Extract WaveTrend data
            wt2 = wt_data.get('wt2', pd.Series(dtype=float))
            wt_cross_up = wt_data.get('cross_up', pd.Series(dtype=bool))
            wt_cross_down = wt_data.get('cross_down', pd.Series(dtype=bool))
            wt_cross = wt_cross_up | wt_cross_down
            
            if wt2.empty or ha_candles_tf1.empty or ha_candles_tf2.empty:
                logger.warning("Empty data provided for Sommi Diamond calculation")
                return self._empty_diamond_result(wt2.index if not wt2.empty else pd.Index([]))
            
            # Get candle body directions (aligned to main timeframe)
            candle_dir_tf1 = ha_candles_tf1.get('candle_body_dir', pd.Series(False, index=wt2.index))
            candle_dir_tf2 = ha_candles_tf2.get('candle_body_dir', pd.Series(False, index=wt2.index))
            
            # Align series to wt2 index
            candle_dir_tf1 = candle_dir_tf1.reindex(wt2.index, method='ffill').fillna(False)
            candle_dir_tf2 = candle_dir_tf2.reindex(wt2.index, method='ffill').fillna(False)
            
            # Sommi Diamond Bear Pattern
            bear_wt2_cond = wt2 >= config.diamond_wt_bear_level
            bear_cross_cond = wt_cross & wt_cross_down
            bear_candle_cond = (~candle_dir_tf1) & (~candle_dir_tf2)  # Both bearish
            
            bear_diamond = bear_wt2_cond & bear_cross_cond & bear_candle_cond
            
            # Sommi Diamond Bull Pattern
            bull_wt2_cond = wt2 <= config.diamond_wt_bull_level
            bull_cross_cond = wt_cross & wt_cross_up
            bull_candle_cond = candle_dir_tf1 & candle_dir_tf2  # Both bullish
            
            bull_diamond = bull_wt2_cond & bull_cross_cond & bull_candle_cond
            
            # Calculate confidence with candle quality
            bear_confidence = self._calculate_diamond_confidence(
                bear_wt2_cond, bear_cross_cond, bear_candle_cond, 
                ha_candles_tf1, ha_candles_tf2
            )
            bull_confidence = self._calculate_diamond_confidence(
                bull_wt2_cond, bull_cross_cond, bull_candle_cond,
                ha_candles_tf1, ha_candles_tf2
            )
            
            # Combined confidence
            confidence = pd.Series(0.0, index=wt2.index)
            confidence[bear_diamond] = bear_confidence[bear_diamond]
            confidence[bull_diamond] = bull_confidence[bull_diamond]
            
            return {
                'bear_diamond': bear_diamond,
                'bull_diamond': bull_diamond,
                'confidence': confidence,
                'bear_confidence': bear_confidence,
                'bull_confidence': bull_confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating Sommi diamonds: {e}")
            return self._empty_diamond_result(pd.Index([]))

    def _calculate_diamond_confidence(
        self,
        wt_cond: pd.Series,
        cross_cond: pd.Series,
        candle_cond: pd.Series,
        ha_tf1: pd.DataFrame,
        ha_tf2: pd.DataFrame
    ) -> pd.Series:
        """Calculate confidence score for diamond patterns with candle quality."""
        base_weights = [0.4, 0.3, 0.3]  # WT level, Cross, Candle alignment
        conditions = [wt_cond, cross_cond, candle_cond]
        
        confidence = pd.Series(0.0, index=wt_cond.index)
        for condition, weight in zip(conditions, base_weights):
            confidence += condition.astype(float) * weight
        
        # Enhance confidence with candle quality
        if 'body_ratio' in ha_tf1.columns and 'body_ratio' in ha_tf2.columns:
            body_ratio_tf1 = ha_tf1['body_ratio'].reindex(wt_cond.index, method='ffill').fillna(0)
            body_ratio_tf2 = ha_tf2['body_ratio'].reindex(wt_cond.index, method='ffill').fillna(0)
            
            # Higher body ratio = stronger candle = higher confidence
            candle_quality = (body_ratio_tf1 + body_ratio_tf2) / 2
            confidence *= (1 + candle_quality * 0.2)  # Up to 20% boost
        
        return confidence.clip(0, 1)

    def _empty_diamond_result(self, index: pd.Index) -> Dict[str, pd.Series]:
        """Return empty diamond result for error cases."""
        empty_bool = pd.Series(False, index=index)
        empty_float = pd.Series(0.0, index=index)
        return {
            'bear_diamond': empty_bool,
            'bull_diamond': empty_bool,
            'confidence': empty_float,
            'bear_confidence': empty_float,
            'bull_confidence': empty_float
        }

    def get_combined_sommi_analysis(
        self,
        price_data: pd.DataFrame,
        levels: Optional[SommiLevels] = None
    ) -> pd.DataFrame:
        """
        Perform complete Sommi pattern analysis.
        
        Args:
            price_data: OHLCV DataFrame with datetime index
            levels: Custom pattern levels
            
        Returns:
            DataFrame with all Sommi pattern indicators and analysis
        """
        try:
            if len(price_data) < 100:  # Need sufficient data
                logger.warning("Insufficient data for Sommi analysis")
                return price_data.copy()
            
            result = price_data.copy()
            config = levels or self.levels
            
            # Calculate base indicators
            logger.debug("Calculating WaveTrend indicators")
            wt_result = self.wavetrend.calculate(price_data)
            wt_data = {
                'wt1': wt_result['wt1'],
                'wt2': wt_result['wt2'],
                'cross_up': wt_result['wt_cross_up'],
                'cross_down': wt_result['wt_cross_down']
            }
            
            logger.debug("Calculating RSI+MFI indicator")
            rsimfi_values = self.rsimfi.calculate_rsimfi(price_data)
            
            # Calculate higher timeframe analysis
            logger.debug("Calculating higher timeframe WaveTrend")
            htf_wt1, htf_wt2, htf_vwap = self.calculate_higher_timeframe_wt(price_data)
            htf_wt_data = {
                'wt1': htf_wt1,
                'wt2': htf_wt2,
                'vwap': htf_vwap
            }
            
            # Calculate Heikin Ashi for multiple timeframes
            logger.debug("Calculating Heikin Ashi candles")
            ha_tf1 = self.calculate_heikin_ashi_candles(price_data)
            
            # Higher timeframe Heikin Ashi (simulate by resampling)
            htf_price_data = self._resample_for_htf(price_data)
            ha_tf2 = self.calculate_heikin_ashi_candles(htf_price_data) if not htf_price_data.empty else ha_tf1
            
            # Calculate Sommi Flag patterns
            logger.debug("Calculating Sommi Flag patterns")
            flag_results = self.calculate_sommi_flags(wt_data, rsimfi_values, htf_wt_data, config)
            
            # Calculate Sommi Diamond patterns
            logger.debug("Calculating Sommi Diamond patterns")
            diamond_results = self.calculate_sommi_diamonds(wt_data, ha_tf1, ha_tf2, config)
            
            # Add all results to DataFrame
            result['wt1'] = wt_data['wt1']
            result['wt2'] = wt_data['wt2']
            result['wt_cross_up'] = wt_data['cross_up']
            result['wt_cross_down'] = wt_data['cross_down']
            result['rsimfi'] = rsimfi_values
            
            # Higher timeframe data
            result['htf_wt1'] = htf_wt_data['wt1']
            result['htf_wt2'] = htf_wt_data['wt2']
            result['htf_vwap'] = htf_vwap
            
            # Heikin Ashi indicators
            if not ha_tf1.empty:
                result['ha_bullish'] = ha_tf1['is_bullish']
                result['ha_bearish'] = ha_tf1['is_bearish']
                result['ha_body_ratio'] = ha_tf1['body_ratio']
            
            # Sommi Flag results
            result['sommi_flag_bear'] = flag_results['bear_flag']
            result['sommi_flag_bull'] = flag_results['bull_flag']
            result['sommi_flag_confidence'] = flag_results['confidence']
            
            # Sommi Diamond results
            result['sommi_diamond_bear'] = diamond_results['bear_diamond']
            result['sommi_diamond_bull'] = diamond_results['bull_diamond']
            result['sommi_diamond_confidence'] = diamond_results['confidence']
            
            # Combined pattern signals
            result['sommi_any_bull'] = result['sommi_flag_bull'] | result['sommi_diamond_bull']
            result['sommi_any_bear'] = result['sommi_flag_bear'] | result['sommi_diamond_bear']
            result['sommi_signal_strength'] = np.maximum(
                result['sommi_flag_confidence'], 
                result['sommi_diamond_confidence']
            )
            
            logger.info("Sommi pattern analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in combined Sommi analysis: {e}")
            return price_data.copy()

    def _resample_for_htf(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Resample price data for higher timeframe analysis."""
        try:
            if not isinstance(price_data.index, pd.DatetimeIndex):
                logger.warning("Cannot resample non-datetime index")
                return pd.DataFrame()
            
            resample_rule = f"{self.htf_multiplier}T"
            
            htf_data = price_data.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return htf_data
            
        except Exception as e:
            logger.error(f"Error resampling for HTF: {e}")
            return pd.DataFrame()

    def get_latest_signals(self, analysis_df: pd.DataFrame) -> List[PatternSignal]:
        """
        Extract latest pattern signals from analysis DataFrame.
        
        Args:
            analysis_df: DataFrame from get_combined_sommi_analysis()
            
        Returns:
            List of active pattern signals
        """
        if analysis_df.empty:
            return []
        
        signals = []
        latest = analysis_df.iloc[-1]
        timestamp = latest.name if hasattr(latest, 'name') else pd.Timestamp.now()
        
        # Check Sommi Flag patterns
        if latest.get('sommi_flag_bull', False):
            confidence = latest.get('sommi_flag_confidence', 0)
            if confidence >= self.confidence_threshold:
                signals.append(PatternSignal(
                    pattern_type=PatternType.SOMMI_FLAG_BULL,
                    confidence=confidence,
                    timestamp=timestamp,
                    wt1=latest.get('wt1'),
                    wt2=latest.get('wt2'),
                    rsimfi=latest.get('rsimfi'),
                    htf_vwap=latest.get('htf_vwap'),
                    description=f"Sommi Flag Bull (confidence: {confidence:.2f})"
                ))
        
        if latest.get('sommi_flag_bear', False):
            confidence = latest.get('sommi_flag_confidence', 0)
            if confidence >= self.confidence_threshold:
                signals.append(PatternSignal(
                    pattern_type=PatternType.SOMMI_FLAG_BEAR,
                    confidence=confidence,
                    timestamp=timestamp,
                    wt1=latest.get('wt1'),
                    wt2=latest.get('wt2'),
                    rsimfi=latest.get('rsimfi'),
                    htf_vwap=latest.get('htf_vwap'),
                    description=f"Sommi Flag Bear (confidence: {confidence:.2f})"
                ))
        
        # Check Sommi Diamond patterns
        if latest.get('sommi_diamond_bull', False):
            confidence = latest.get('sommi_diamond_confidence', 0)
            if confidence >= self.confidence_threshold:
                signals.append(PatternSignal(
                    pattern_type=PatternType.SOMMI_DIAMOND_BULL,
                    confidence=confidence,
                    timestamp=timestamp,
                    wt1=latest.get('wt1'),
                    wt2=latest.get('wt2'),
                    candle_alignment=latest.get('ha_bullish'),
                    description=f"Sommi Diamond Bull (confidence: {confidence:.2f})"
                ))
        
        if latest.get('sommi_diamond_bear', False):
            confidence = latest.get('sommi_diamond_confidence', 0)
            if confidence >= self.confidence_threshold:
                signals.append(PatternSignal(
                    pattern_type=PatternType.SOMMI_DIAMOND_BEAR,
                    confidence=confidence,
                    timestamp=timestamp,
                    wt1=latest.get('wt1'),
                    wt2=latest.get('wt2'),
                    candle_alignment=latest.get('ha_bearish'),
                    description=f"Sommi Diamond Bear (confidence: {confidence:.2f})"
                ))
        
        return signals

    def validate_pattern_conditions(
        self,
        pattern_type: PatternType,
        current_data: Dict,
        lookback_periods: int = 5
    ) -> bool:
        """
        Validate pattern conditions with additional timing checks.
        
        Args:
            pattern_type: Type of pattern to validate
            current_data: Current market data
            lookback_periods: Periods to look back for validation
            
        Returns:
            True if pattern conditions are valid
        """
        try:
            # Basic validation based on pattern type
            if pattern_type in [PatternType.SOMMI_FLAG_BULL, PatternType.SOMMI_FLAG_BEAR]:
                return self._validate_flag_conditions(current_data, pattern_type)
            elif pattern_type in [PatternType.SOMMI_DIAMOND_BULL, PatternType.SOMMI_DIAMOND_BEAR]:
                return self._validate_diamond_conditions(current_data, pattern_type)
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating pattern conditions: {e}")
            return False

    def _validate_flag_conditions(self, data: Dict, pattern_type: PatternType) -> bool:
        """Validate Sommi Flag pattern conditions."""
        required_keys = ['rsimfi', 'wt2', 'htf_vwap', 'wt_cross']
        if not all(key in data for key in required_keys):
            return False
        
        if pattern_type == PatternType.SOMMI_FLAG_BULL:
            return (
                data['rsimfi'] > self.levels.rsimfi_bull_level and
                data['wt2'] < self.levels.flag_wt_bull_level and
                data['htf_vwap'] > self.levels.vwap_bull_level and
                data.get('wt_cross_up', False)
            )
        else:  # BEAR
            return (
                data['rsimfi'] < self.levels.rsimfi_bear_level and
                data['wt2'] > self.levels.flag_wt_bear_level and
                data['htf_vwap'] < self.levels.vwap_bear_level and
                data.get('wt_cross_down', False)
            )

    def _validate_diamond_conditions(self, data: Dict, pattern_type: PatternType) -> bool:
        """Validate Sommi Diamond pattern conditions."""
        required_keys = ['wt2', 'candle_alignment']
        if not all(key in data for key in required_keys):
            return False
        
        if pattern_type == PatternType.SOMMI_DIAMOND_BULL:
            return (
                data['wt2'] <= self.levels.diamond_wt_bull_level and
                data.get('candle_alignment', False) and
                data.get('wt_cross_up', False)
            )
        else:  # BEAR
            return (
                data['wt2'] >= self.levels.diamond_wt_bear_level and
                not data.get('candle_alignment', True) and
                data.get('wt_cross_down', False)
            )

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main calculate method for compatibility with the CipherB interface.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Sommi pattern indicators using standard column names
        """
        try:
            # Use the comprehensive analysis method
            result = self.get_combined_sommi_analysis(df)
            
            # Map to standard column names expected by CipherB
            result['sommi_flag_up'] = result.get('sommi_flag_bull', pd.Series(False, index=result.index))
            result['sommi_flag_down'] = result.get('sommi_flag_bear', pd.Series(False, index=result.index))
            result['sommi_diamond_up'] = result.get('sommi_diamond_bull', pd.Series(False, index=result.index))
            result['sommi_diamond_down'] = result.get('sommi_diamond_bear', pd.Series(False, index=result.index))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Sommi patterns calculate method: {e}")
            # Return DataFrame with empty pattern columns
            result = df.copy()
            empty_bool = pd.Series(False, index=result.index)
            result['sommi_flag_up'] = empty_bool
            result['sommi_flag_down'] = empty_bool
            result['sommi_diamond_up'] = empty_bool
            result['sommi_diamond_down'] = empty_bool
            return result