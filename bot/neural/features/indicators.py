"""
Technical Indicator Integration for Neural Networks

This module integrates the existing VuManChu Cipher indicators and other
technical analysis tools with the neural network feature engineering pipeline.
"""


import numpy as np
import pandas as pd

from ...indicators.vumanchu import VuManChuCipher


class TechnicalIndicatorFeatures:
    """
    Integration layer for technical indicators with neural network features.

    This class bridges the existing VuManChu Cipher indicators and other
    technical analysis tools with the neural network preprocessing pipeline.

    Key Features:
    - VuManChu Cipher A & B integration
    - RSI, MACD, Bollinger Bands
    - Volume indicators
    - Momentum oscillators
    - Trend indicators
    """

    def __init__(
        self,
        enable_vumanchu: bool = True,
        enable_volume_indicators: bool = True,
        enable_momentum_indicators: bool = True,
        enable_trend_indicators: bool = True,
    ):
        """
        Initialize the technical indicator feature extractor.

        Args:
            enable_vumanchu: Whether to include VuManChu Cipher indicators
            enable_volume_indicators: Whether to include volume-based indicators
            enable_momentum_indicators: Whether to include momentum indicators
            enable_trend_indicators: Whether to include trend indicators
        """
        self.enable_vumanchu = enable_vumanchu
        self.enable_volume_indicators = enable_volume_indicators
        self.enable_momentum_indicators = enable_momentum_indicators
        self.enable_trend_indicators = enable_trend_indicators

        # Initialize VuManChu Cipher if enabled
        self.vumanchu = VuManChuCipher() if enable_vumanchu else None

        # Store feature names for consistent ordering
        self.feature_names = []

    def extract_vumanchu_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract VuManChu Cipher A & B features.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with VuManChu features added
        """
        if not self.enable_vumanchu or self.vumanchu is None:
            return data

        try:
            # Calculate VuManChu indicators
            cipher_data = self.vumanchu.calculate_all(data)

            # Extract key features from VuManChu results
            vumanchu_features = {
                # Cipher A features
                "cipher_a_dot": cipher_data.get("cipher_a_dot", 0.0),
                "cipher_a_signal_line": cipher_data.get("cipher_a_signal_line", 0.0),
                "cipher_a_divergence": cipher_data.get("cipher_a_divergence", 0.0),
                # Cipher B features
                "cipher_b_wave": cipher_data.get("cipher_b_wave", 0.0),
                "cipher_b_money_flow": cipher_data.get("cipher_b_money_flow", 50.0),
                "cipher_b_signal_strength": cipher_data.get(
                    "cipher_b_signal_strength", 0.0
                ),
                # Combined signals
                "cipher_combined_signal": cipher_data.get("combined_signal", 0.0),
                "cipher_trend_strength": cipher_data.get("trend_strength", 0.0),
            }

            # Add features to DataFrame
            for feature_name, feature_value in vumanchu_features.items():
                if isinstance(feature_value, (list, np.ndarray, pd.Series)):
                    data[feature_name] = pd.Series(feature_value, index=data.index)
                else:
                    data[feature_name] = feature_value

            # Add feature names to tracking list
            self.feature_names.extend(list(vumanchu_features.keys()))

        except Exception as e:
            # If VuManChu calculation fails, add zeros for consistency
            print(f"Warning: VuManChu calculation failed: {e}")
            vumanchu_features = {
                "cipher_a_dot": 0.0,
                "cipher_a_signal_line": 0.0,
                "cipher_a_divergence": 0.0,
                "cipher_b_wave": 0.0,
                "cipher_b_money_flow": 50.0,
                "cipher_b_signal_strength": 0.0,
                "cipher_combined_signal": 0.0,
                "cipher_trend_strength": 0.0,
            }

            for feature_name, feature_value in vumanchu_features.items():
                data[feature_name] = feature_value

        return data

    def extract_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract momentum-based technical indicators.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with momentum indicators added
        """
        if not self.enable_momentum_indicators:
            return data

        # RSI (Relative Strength Index)
        for period in [14, 21]:
            data[f"rsi_{period}"] = self._calculate_rsi(data["close"], period)

        # Stochastic Oscillator
        data["stoch_k"], data["stoch_d"] = self._calculate_stochastic(
            data["high"], data["low"], data["close"]
        )

        # Williams %R
        data["williams_r"] = self._calculate_williams_r(
            data["high"], data["low"], data["close"]
        )

        # Rate of Change (ROC)
        for period in [10, 20]:
            data[f"roc_{period}"] = data["close"].pct_change(periods=period) * 100

        # Momentum
        for period in [10, 20]:
            data[f"momentum_{period}"] = data["close"] / data["close"].shift(period)

        # Add feature names
        momentum_features = [
            "rsi_14",
            "rsi_21",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "roc_10",
            "roc_20",
            "momentum_10",
            "momentum_20",
        ]
        self.feature_names.extend(momentum_features)

        return data

    def extract_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract trend-based technical indicators.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with trend indicators added
        """
        if not self.enable_trend_indicators:
            return data

        # Moving Averages
        for period in [10, 20, 50, 200]:
            data[f"sma_{period}"] = data["close"].rolling(window=period).mean()
            data[f"ema_{period}"] = data["close"].ewm(span=period).mean()

        # MACD
        data["macd"], data["macd_signal"], data["macd_histogram"] = (
            self._calculate_macd(data["close"])
        )

        # Bollinger Bands
        data["bb_upper"], data["bb_middle"], data["bb_lower"] = (
            self._calculate_bollinger_bands(data["close"])
        )
        data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]
        data["bb_position"] = (data["close"] - data["bb_lower"]) / (
            data["bb_upper"] - data["bb_lower"]
        )

        # Average True Range (ATR)
        data["atr"] = self._calculate_atr(data["high"], data["low"], data["close"])

        # Parabolic SAR
        data["psar"] = self._calculate_psar(data["high"], data["low"], data["close"])

        # Trend strength
        data["trend_strength"] = self._calculate_trend_strength(data["close"])

        # Add feature names
        trend_features = [
            "sma_10",
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_10",
            "ema_20",
            "ema_50",
            "ema_200",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "bb_position",
            "atr",
            "psar",
            "trend_strength",
        ]
        self.feature_names.extend(trend_features)

        return data

    def extract_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract volume-based technical indicators.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with volume indicators added
        """
        if not self.enable_volume_indicators or "volume" not in data.columns:
            return data

        # Volume Moving Averages
        for period in [10, 20, 50]:
            data[f"volume_sma_{period}"] = data["volume"].rolling(window=period).mean()

        # Volume Ratio
        data["volume_ratio"] = data["volume"] / data["volume_sma_20"]

        # On-Balance Volume (OBV)
        data["obv"] = self._calculate_obv(data["close"], data["volume"])

        # Volume Price Trend (VPT)
        data["vpt"] = self._calculate_vpt(data["close"], data["volume"])

        # Accumulation/Distribution Line
        data["ad_line"] = self._calculate_ad_line(
            data["high"], data["low"], data["close"], data["volume"]
        )

        # Money Flow Index (MFI)
        data["mfi"] = self._calculate_mfi(
            data["high"], data["low"], data["close"], data["volume"]
        )

        # Volume Weighted Average Price (VWAP)
        data["vwap"] = self._calculate_vwap(
            data["high"], data["low"], data["close"], data["volume"]
        )

        # Add feature names
        volume_features = [
            "volume_sma_10",
            "volume_sma_20",
            "volume_sma_50",
            "volume_ratio",
            "obv",
            "vpt",
            "ad_line",
            "mfi",
            "vwap",
        ]
        self.feature_names.extend(volume_features)

        return data

    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all enabled technical indicator features.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with all technical indicator features
        """
        # Reset feature names list
        self.feature_names = []

        # Extract features in order
        data = self.extract_vumanchu_features(data.copy())
        data = self.extract_momentum_indicators(data)
        data = self.extract_trend_indicators(data)
        data = self.extract_volume_indicators(data)

        # Handle any remaining NaN values
        data = data.fillna(method="ffill").fillna(method="bfill").fillna(0)

        return data

    def get_feature_names(self) -> list[str]:
        """Get list of extracted feature names."""
        return self.feature_names.copy()

    # Helper methods for indicator calculations
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    def _calculate_williams_r(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> tuple:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(window=period).mean()

    def _calculate_psar(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        af: float = 0.02,
        max_af: float = 0.2,
    ) -> pd.Series:
        """Calculate Parabolic SAR (simplified version)."""
        # Simplified PSAR calculation
        psar = close.copy()
        psar.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:  # Uptrend
                psar.iloc[i] = min(
                    psar.iloc[i - 1] + af * (high.iloc[i - 1] - psar.iloc[i - 1]),
                    low.iloc[i - 1],
                )
            else:  # Downtrend
                psar.iloc[i] = max(
                    psar.iloc[i - 1] + af * (low.iloc[i - 1] - psar.iloc[i - 1]),
                    high.iloc[i - 1],
                )

        return psar

    def _calculate_trend_strength(
        self, prices: pd.Series, period: int = 20
    ) -> pd.Series:
        """Calculate trend strength indicator."""
        returns = prices.pct_change()
        rolling_std = returns.rolling(window=period).std()
        rolling_mean = returns.rolling(window=period).mean()
        return rolling_mean / (rolling_std + 1e-8)

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = np.zeros(len(close))
        obv[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv[i] = obv[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv[i] = obv[i - 1] - volume.iloc[i]
            else:
                obv[i] = obv[i - 1]

        return pd.Series(obv, index=close.index)

    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Price Trend."""
        vpt = np.zeros(len(close))
        vpt[0] = volume.iloc[0]

        for i in range(1, len(close)):
            vpt[i] = (
                vpt[i - 1]
                + volume.iloc[i]
                * (close.iloc[i] - close.iloc[i - 1])
                / close.iloc[i - 1]
            )

        return pd.Series(vpt, index=close.index)

    def _calculate_ad_line(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_line = (clv * volume).cumsum()
        return ad_line

    def _calculate_mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = (
            money_flow.where(typical_price > typical_price.shift(), 0)
            .rolling(window=period)
            .sum()
        )
        negative_flow = (
            money_flow.where(typical_price < typical_price.shift(), 0)
            .rolling(window=period)
            .sum()
        )

        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))
        return mfi

    def _calculate_vwap(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
