"""
Financial Time Series Dataset for Neural Network Training

This module provides PyTorch dataset classes optimized for financial time series
data with support for walk-forward validation and multi-horizon prediction.
"""


import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..features.indicators import TechnicalIndicatorFeatures
from ..features.preprocessing import FinancialDataPreprocessor


class FinancialTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for financial time series data.

    This dataset handles the complexities of financial data including:
    - Time-based splitting for validation
    - Multi-horizon target creation
    - Technical indicator integration
    - Proper handling of look-ahead bias

    Key Features:
    - Integration with VuManChu indicators
    - Walk-forward validation support
    - Multi-horizon prediction targets
    - Efficient data loading and caching
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str = "close",
        sequence_length: int = 60,
        prediction_horizons: list[int] = [1, 5, 15, 30],
        stride: int = 1,
        include_technical_indicators: bool = True,
        preprocessor: FinancialDataPreprocessor | None = None,
        mode: str = "train",
    ):
        """
        Initialize the financial time series dataset.

        Args:
            data: DataFrame with OHLCV data and optional indicators
            target_column: Column name for prediction target
            sequence_length: Length of input sequences
            prediction_horizons: List of future steps to predict
            stride: Step size between sequences
            include_technical_indicators: Whether to include technical indicators
            preprocessor: Optional preprocessor (will create if None)
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.data = data.copy()
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.stride = stride
        self.mode = mode

        # Initialize preprocessor
        if preprocessor is None:
            self.preprocessor = FinancialDataPreprocessor()
        else:
            self.preprocessor = preprocessor

        # Initialize technical indicators
        if include_technical_indicators:
            self.indicator_extractor = TechnicalIndicatorFeatures()
            self.data = self.indicator_extractor.extract_all_features(self.data)
        else:
            self.indicator_extractor = None

        # Prepare sequences
        self._prepare_data()

    def _prepare_data(self):
        """Prepare the data for training/inference."""
        # Fit preprocessor only in training mode
        if self.mode == "train":
            self.preprocessor.fit(self.data, target_column=self.target_column)

        # Create sequences
        self.features, self.targets = self.preprocessor.prepare_sequences(
            self.data,
            target_column=self.target_column,
            sequence_length=self.sequence_length,
            prediction_horizons=self.prediction_horizons,
            stride=self.stride,
        )

        # Store additional metadata
        self.feature_names = self.preprocessor.get_feature_names()
        self.n_features = len(self.feature_names)

        print(
            f"Dataset prepared: {len(self.features)} sequences, "
            f"{self.n_features} features, {len(self.prediction_horizons)} horizons"
        )

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sequence and its targets.

        Args:
            idx: Index of the sequence

        Returns:
            Dictionary with features and targets
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {"features": features, "targets": targets, "index": idx}

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self.feature_names

    def get_preprocessor(self) -> FinancialDataPreprocessor:
        """Get the data preprocessor."""
        return self.preprocessor

    def get_data_info(self) -> dict:
        """Get information about the dataset."""
        return {
            "n_sequences": len(self.features),
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "prediction_horizons": self.prediction_horizons,
            "target_column": self.target_column,
            "feature_names": self.feature_names,
            "data_shape": {
                "features": self.features.shape,
                "targets": self.targets.shape,
            },
        }


class WalkForwardDataModule:
    """
    Data module for walk-forward validation on financial time series.

    This class implements proper time series validation by ensuring that
    training data always precedes validation data, preventing look-ahead bias.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str = "close",
        sequence_length: int = 60,
        prediction_horizons: list[int] = [1, 5, 15, 30],
        train_size: float = 0.7,
        val_size: float = 0.2,
        test_size: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        include_technical_indicators: bool = True,
    ):
        """
        Initialize the walk-forward data module.

        Args:
            data: DataFrame with financial time series data
            target_column: Column name for prediction target
            sequence_length: Length of input sequences
            prediction_horizons: List of future steps to predict
            train_size: Fraction of data for training
            val_size: Fraction of data for validation
            test_size: Fraction of data for testing
            batch_size: Batch size for data loading
            num_workers: Number of worker processes for data loading
            include_technical_indicators: Whether to include technical indicators
        """
        self.data = data.copy()
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.include_technical_indicators = include_technical_indicators

        # Validate split sizes
        assert (
            abs(train_size + val_size + test_size - 1.0) < 1e-6
        ), "Split sizes must sum to 1.0"

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # Split data chronologically
        self._split_data()

        # Create datasets
        self._create_datasets()

    def _split_data(self):
        """Split data chronologically into train/val/test sets."""
        n_samples = len(self.data)

        # Calculate split indices
        train_end = int(n_samples * self.train_size)
        val_end = int(n_samples * (self.train_size + self.val_size))

        # Ensure we have enough data for sequences
        min_samples = self.sequence_length + max(self.prediction_horizons)

        if train_end < min_samples:
            raise ValueError(
                f"Training set too small. Need at least {min_samples} samples."
            )

        # Split data chronologically
        self.train_data = self.data.iloc[:train_end].copy()
        self.val_data = self.data.iloc[train_end:val_end].copy()
        self.test_data = self.data.iloc[val_end:].copy()

        print(
            f"Data split: Train={len(self.train_data)}, "
            f"Val={len(self.val_data)}, Test={len(self.test_data)}"
        )

    def _create_datasets(self):
        """Create PyTorch datasets for each split."""
        # Create training dataset and fit preprocessor
        self.train_dataset = FinancialTimeSeriesDataset(
            data=self.train_data,
            target_column=self.target_column,
            sequence_length=self.sequence_length,
            prediction_horizons=self.prediction_horizons,
            include_technical_indicators=self.include_technical_indicators,
            mode="train",
        )

        # Create validation dataset using fitted preprocessor
        self.val_dataset = FinancialTimeSeriesDataset(
            data=self.val_data,
            target_column=self.target_column,
            sequence_length=self.sequence_length,
            prediction_horizons=self.prediction_horizons,
            include_technical_indicators=self.include_technical_indicators,
            preprocessor=self.train_dataset.get_preprocessor(),
            mode="val",
        )

        # Create test dataset using fitted preprocessor
        self.test_dataset = FinancialTimeSeriesDataset(
            data=self.test_data,
            target_column=self.target_column,
            sequence_length=self.sequence_length,
            prediction_horizons=self.prediction_horizons,
            include_technical_indicators=self.include_technical_indicators,
            preprocessor=self.train_dataset.get_preprocessor(),
            mode="test",
        )

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for better training
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle test
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self.train_dataset.get_feature_names()

    def get_preprocessor(self) -> FinancialDataPreprocessor:
        """Get the fitted preprocessor."""
        return self.train_dataset.get_preprocessor()

    def get_data_info(self) -> dict:
        """Get comprehensive information about the data."""
        return {
            "total_samples": len(self.data),
            "splits": {
                "train": len(self.train_data),
                "val": len(self.val_data),
                "test": len(self.test_data),
            },
            "train_dataset_info": self.train_dataset.get_data_info(),
            "val_dataset_info": self.val_dataset.get_data_info(),
            "test_dataset_info": self.test_dataset.get_data_info(),
            "feature_names": self.get_feature_names(),
            "sequence_length": self.sequence_length,
            "prediction_horizons": self.prediction_horizons,
            "batch_size": self.batch_size,
        }


class MultiTimeframeDataset(Dataset):
    """
    Dataset that combines multiple timeframes for enhanced prediction.

    This dataset creates features from multiple timeframes (e.g., 1m, 5m, 15m)
    to provide the model with both high-frequency and longer-term patterns.
    """

    def __init__(
        self,
        datasets: dict[str, pd.DataFrame],
        primary_timeframe: str,
        target_column: str = "close",
        sequence_length: int = 60,
        prediction_horizons: list[int] = [1, 5, 15, 30],
    ):
        """
        Initialize multi-timeframe dataset.

        Args:
            datasets: Dictionary mapping timeframe names to DataFrames
            primary_timeframe: Primary timeframe for sequencing
            target_column: Column name for prediction target
            sequence_length: Length of input sequences
            prediction_horizons: List of future steps to predict
        """
        self.datasets = datasets
        self.primary_timeframe = primary_timeframe
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons

        if primary_timeframe not in datasets:
            raise ValueError(f"Primary timeframe {primary_timeframe} not in datasets")

        # Prepare each timeframe
        self.timeframe_features = {}
        self.preprocessors = {}

        for timeframe, data in datasets.items():
            # Create preprocessor for this timeframe
            preprocessor = FinancialDataPreprocessor()
            indicator_extractor = TechnicalIndicatorFeatures()

            # Extract indicators and preprocess
            processed_data = indicator_extractor.extract_all_features(data)
            features, targets = preprocessor.prepare_sequences(
                processed_data,
                target_column=target_column,
                sequence_length=sequence_length,
                prediction_horizons=prediction_horizons,
            )

            self.timeframe_features[timeframe] = features
            self.preprocessors[timeframe] = preprocessor

        # Use primary timeframe for targets and sequencing
        primary_data = datasets[primary_timeframe]
        primary_preprocessor = FinancialDataPreprocessor()
        primary_indicator_extractor = TechnicalIndicatorFeatures()

        processed_primary = primary_indicator_extractor.extract_all_features(
            primary_data
        )
        _, self.targets = primary_preprocessor.prepare_sequences(
            processed_primary,
            target_column=target_column,
            sequence_length=sequence_length,
            prediction_horizons=prediction_horizons,
        )

        # Align all timeframes to the primary timeframe length
        min_length = min(len(features) for features in self.timeframe_features.values())
        min_length = min(min_length, len(self.targets))

        for timeframe in self.timeframe_features:
            self.timeframe_features[timeframe] = self.timeframe_features[timeframe][
                :min_length
            ]
        self.targets = self.targets[:min_length]

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get multi-timeframe features and targets.

        Args:
            idx: Index of the sequence

        Returns:
            Dictionary with features from all timeframes and targets
        """
        result = {"targets": torch.tensor(self.targets[idx], dtype=torch.float32)}

        # Add features from each timeframe
        for timeframe, features in self.timeframe_features.items():
            result[f"features_{timeframe}"] = torch.tensor(
                features[idx], dtype=torch.float32
            )

        return result

    def get_timeframes(self) -> list[str]:
        """Get list of available timeframes."""
        return list(self.datasets.keys())

    def get_feature_dimensions(self) -> dict[str, int]:
        """Get feature dimensions for each timeframe."""
        return {
            timeframe: features.shape[-1]
            for timeframe, features in self.timeframe_features.items()
        }
