"""
Real-Time Neural Network Inference Module

This module provides real-time prediction capabilities for the neural network
models with FastAPI server and WebSocket streaming support.

Key Components:
- predictor.py: Real-time prediction engine
- api_server.py: FastAPI REST API for predictions
- streaming.py: WebSocket streaming for live predictions

Inference Features:
- Sub-50ms prediction latency
- Batch prediction support
- Model ensemble capabilities
- Real-time feature engineering
- WebSocket streaming for live updates
- REST API endpoints for integration
"""

from .api_server import PredictionAPIServer
from .predictor import NeuralPredictor
from .streaming import StreamingPredictor

__all__ = ["NeuralPredictor", "PredictionAPIServer", "StreamingPredictor"]
