"""
Financial Data Preprocessing for Neural Networks

This module provides comprehensive data preprocessing capabilities specifically
designed for financial time series data used in neural network training.
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class FinancialDataPreprocessor:
    """
    Comprehensive data preprocessor for financial time series data.

    This class handles all aspects of data preprocessing needed for neural
    network training, including normalization, sequence creation, and
    feature engineering integration.

    Key Features:
    - Multiple scaling methods (MinMax, Standard, Robust)
    - Handling of missing values
    - Sequence creation for time series models
    - Multi-horizon target creation
    - Integration with existing VuManChu indicators
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        handle_missing: str = "forward_fill",
        clip_outliers: bool = True,
        outlier_threshold: float = 3.0,
    ):
        """
        Initialize the preprocessor.

        Args:
            scaler_type: Type of scaler ('minmax', 'standard', 'robust')
            handle_missing: How to handle missing values ('forward_fill', 'drop', 'interpolate')
            clip_outliers: Whether to clip extreme outliers
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.clip_outliers = clip_outliers
        self.outlier_threshold = outlier_threshold

        # Initialize scalers
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []

        # Statistics for normalization
        self.feature_stats = {}
        self.target_stats = {}

        # Imputer for missing values
        self.imputer = SimpleImputer(strategy="median")

    def _get_scaler(self, scaler_type: str):
        """Get the appropriate scaler based on type."""
        if scaler_type == "minmax":
            return MinMaxScaler(feature_range=(-1, 1))
        if scaler_type == "standard":
            return StandardScaler()
        if scaler_type == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        if self.handle_missing == "forward_fill":
            # Forward fill followed by backward fill
            data = data.fillna(method="ffill").fillna(method="bfill")
        elif self.handle_missing == "drop":
            data = data.dropna()
        elif self.handle_missing == "interpolate":
            data = data.interpolate(method="linear")
        elif self.handle_missing == "impute":
            # Use trained imputer
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = self.imputer.fit_transform(data[numeric_columns])

        return data

    def _clip_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip extreme outliers using z-score method."""
        if not self.clip_outliers:
            return data

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            outlier_mask = z_scores > self.outlier_threshold

            if outlier_mask.any():
                # Clip to percentiles instead of removing
                lower_percentile = data[column].quantile(0.01)
                upper_percentile = data[column].quantile(0.99)
                data[column] = data[column].clip(
                    lower=lower_percentile, upper=upper_percentile
                )

        return data

    def _create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional price-based features."""
        # Price momentum features
        for period in [1, 5, 10, 20]:
            data[f"return_{period}"] = data["close"].pct_change(periods=period)
            data[f"price_momentum_{period}"] = (
                data["close"] / data["close"].shift(period) - 1
            )

        # Volatility features
        for period in [5, 10, 20]:
            data[f"volatility_{period}"] = data["close"].rolling(window=period).std()
            data[f"high_low_ratio_{period}"] = (data["high"] - data["low"]) / data[
                "close"
            ]

        # Volume features
        if "volume" in data.columns:
            data["volume_sma_10"] = data["volume"].rolling(window=10).mean()
            data["volume_ratio"] = data["volume"] / data["volume_sma_10"]
            data["price_volume"] = data["close"] * data["volume"]

        # OHLC relationships
        data["body_size"] = abs(data["close"] - data["open"]) / data["close"]
        data["upper_shadow"] = (
            data["high"] - np.maximum(data["open"], data["close"])
        ) / data["close"]
        data["lower_shadow"] = (
            np.minimum(data["open"], data["close"]) - data["low"]
        ) / data["close"]

        return data

    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return data

        # Cyclical time features
        data["hour_sin"] = np.sin(2 * np.pi * data.index.hour / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data.index.hour / 24)
        data["day_sin"] = np.sin(2 * np.pi * data.index.dayofweek / 7)
        data["day_cos"] = np.cos(2 * np.pi * data.index.dayofweek / 7)
        data["month_sin"] = np.sin(2 * np.pi * data.index.month / 12)
        data["month_cos"] = np.cos(2 * np.pi * data.index.month / 12)

        # Market session indicators
        data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)
        data["is_trading_hours"] = (
            (data.index.hour >= 9) & (data.index.hour <= 16)
        ).astype(int)

        return data

    def fit(
        self, data: pd.DataFrame, target_column: str = "close"
    ) -> "FinancialDataPreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            data: Training data DataFrame
            target_column: Name of the target column

        Returns:
            Self for method chaining
        """
        # Handle missing values
        data = self._handle_missing_values(data.copy())

        # Create additional features
        data = self._create_price_features(data)
        data = self._create_time_features(data)

        # Clip outliers
        data = self._clip_outliers(data)

        # Separate features and targets
        self.target_columns = [target_column]
        self.feature_columns = [
            col for col in data.columns if col not in self.target_columns
        ]

        # Fit scalers
        feature_data = data[self.feature_columns]
        target_data = data[self.target_columns]

        # Feature scaler
        self.scalers["features"] = self._get_scaler(self.scaler_type)
        self.scalers["features"].fit(feature_data)

        # Target scaler (always use robust scaling for targets)
        self.scalers["targets"] = self._get_scaler("robust")
        self.scalers["targets"].fit(target_data)

        # Store statistics
        self.feature_stats = {
            "mean": feature_data.mean().to_dict(),
            "std": feature_data.std().to_dict(),
            "min": feature_data.min().to_dict(),
            "max": feature_data.max().to_dict(),
        }

        self.target_stats = {
            "mean": target_data.mean().to_dict(),
            "std": target_data.std().to_dict(),
            "min": target_data.min().to_dict(),
            "max": target_data.max().to_dict(),
        }

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Args:
            data: Data to transform

        Returns:
            Transformed DataFrame
        """
        # Handle missing values
        data = self._handle_missing_values(data.copy())

        # Create additional features
        data = self._create_price_features(data)
        data = self._create_time_features(data)

        # Clip outliers
        data = self._clip_outliers(data)

        # Ensure we have the same columns as training
        missing_features = set(self.feature_columns) - set(data.columns)
        if missing_features:
            for feature in missing_features:
                data[feature] = 0.0  # Fill missing features with zeros

        # Select and order columns consistently
        feature_data = data[self.feature_columns]
        target_data = (
            data[self.target_columns]
            if all(col in data.columns for col in self.target_columns)
            else None
        )

        # Transform features
        transformed_features = self.scalers["features"].transform(feature_data)
        result = pd.DataFrame(
            transformed_features, index=data.index, columns=self.feature_columns
        )

        # Transform targets if available
        if target_data is not None:
            transformed_targets = self.scalers["targets"].transform(target_data)
            target_df = pd.DataFrame(
                transformed_targets, index=data.index, columns=self.target_columns
            )
            result = pd.concat([result, target_df], axis=1)

        return result

    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform target predictions back to original scale.

        Args:
            data: Transformed target data

        Returns:
            Data in original scale
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        return self.scalers["targets"].inverse_transform(data)

    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target_column: str = "close",
        sequence_length: int = 60,
        prediction_horizons: list[int] = [1, 5, 15, 30],
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series prediction.

        Args:
            data: Input DataFrame
            target_column: Column to predict
            sequence_length: Length of input sequences
            prediction_horizons: Future steps to predict
            stride: Step size between sequences

        Returns:
            Tuple of (features, targets) arrays
        """
        # Transform data
        transformed_data = self.transform(data)

        # Get feature and target arrays
        features = transformed_data[self.feature_columns].values
        targets = transformed_data[target_column].values

        # Create sequences
        X, y = [], []

        max_horizon = max(prediction_horizons)
        end_idx = len(features) - max_horizon

        for i in range(0, end_idx, stride):
            if i + sequence_length >= end_idx:
                break

            # Input sequence
            sequence = features[i : i + sequence_length]
            X.append(sequence)

            # Multi-horizon targets
            target_values = []
            for horizon in prediction_horizons:
                future_idx = i + sequence_length + horizon - 1
                if future_idx < len(targets):
                    # Calculate return instead of absolute price
                    current_price = targets[i + sequence_length - 1]
                    future_price = targets[future_idx]
                    return_value = (future_price - current_price) / (
                        current_price + 1e-8
                    )
                    target_values.append(return_value)
                else:
                    target_values.append(0.0)

            y.append(target_values)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def get_feature_names(self) -> list[str]:
        """Get the list of feature column names."""
        return self.feature_columns.copy()

    def get_preprocessing_info(self) -> dict:
        """Get information about the preprocessing configuration."""
        return {
            "scaler_type": self.scaler_type,
            "handle_missing": self.handle_missing,
            "clip_outliers": self.clip_outliers,
            "outlier_threshold": self.outlier_threshold,
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "feature_stats": self.feature_stats,
            "target_stats": self.target_stats,
        }
