"""
Feature Engineering Module for Neural Network Training

This module provides comprehensive feature engineering capabilities for
cryptocurrency trading data, including:

- Technical indicator integration (VuManChu Cipher A/B, RSI, MACD, etc.)
- OHLCV data preprocessing and normalization
- Order book and volume profile features
- Multi-timeframe feature aggregation
- Trade execution history features

Key Components:
- preprocessing.py: Data preprocessing and normalization
- engineering.py: Feature extraction and creation
- indicators.py: Technical indicator integration
"""

from .engineering import FeatureEngineer
from .indicators import TechnicalIndicatorFeatures
from .preprocessing import FinancialDataPreprocessor

__all__ = ["FeatureEngineer", "FinancialDataPreprocessor", "TechnicalIndicatorFeatures"]
