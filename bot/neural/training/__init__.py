"""
Training Pipeline for Neural Network Models

This module provides comprehensive training capabilities for cryptocurrency
prediction models with walk-forward validation and hyperparameter optimization.

Key Components:
- trainer.py: PyTorch Lightning training orchestration
- dataset.py: Financial time series dataset classes
- callbacks.py: Training callbacks and monitoring
- metrics.py: Financial performance metrics

Training Features:
- Walk-forward validation for time series
- Hyperparameter optimization with Optuna
- Early stopping and model checkpointing
- Financial metrics (Sharpe ratio, max drawdown)
- Real-time training monitoring
"""

from .callbacks import FinancialMetricsCallback
from .dataset import FinancialTimeSeriesDataset
from .metrics import FinancialMetrics
from .trainer import NeuralTrainer

__all__ = [
    "FinancialMetrics",
    "FinancialMetricsCallback",
    "FinancialTimeSeriesDataset",
    "NeuralTrainer",
]
