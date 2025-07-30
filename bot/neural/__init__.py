"""
Neural Network Module for Cryptocurrency Price Prediction

This module provides neural network capabilities for enhancing the trading bot's
prediction accuracy using state-of-the-art deep learning models.

Key Components:
- models/: Neural network architectures (LSTM, Transformer, CNN-LSTM, Ensemble)
- training/: Training pipeline with walk-forward validation
- features/: Feature engineering and preprocessing
- inference/: Real-time prediction server and streaming
- evaluation/: Performance metrics and backtesting

Architecture Overview:
The neural network module integrates seamlessly with the existing VuManChu
indicators and LLM-based strategy, providing quantitative price predictions
to enhance the qualitative reasoning from the LLM agent.

Key Features:
- Multi-horizon forecasting
- Real-time inference (<50ms latency)
- Walk-forward validation
- Hyperparameter optimization with Optuna
- Financial metrics (Sharpe ratio, max drawdown)
- Integration with existing risk management
"""

from .features import *
from .inference import *
from .models import *

__version__ = "0.1.0"
__author__ = "AI Trading Bot Neural Team"
