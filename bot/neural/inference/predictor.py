"""
Real-Time Neural Network Predictor

This module provides a high-performance predictor for real-time cryptocurrency
price prediction with support for multiple models and ensemble predictions.
"""

import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..features.indicators import TechnicalIndicatorFeatures
from ..features.preprocessing import FinancialDataPreprocessor
from ..models.base_model import BaseNeuralModel
from ..models.lstm_model import LSTMModel
from ..models.transformer_model import TemporalFusionTransformer


class NeuralPredictor:
    """
    High-performance neural network predictor for real-time trading.

    This class provides optimized inference for cryptocurrency price prediction
    with support for model ensembles, uncertainty quantification, and
    real-time feature engineering.

    Key Features:
    - Sub-50ms prediction latency
    - Multi-model ensemble support
    - Real-time feature engineering
    - Uncertainty quantification
    - Batch prediction capabilities
    - GPU acceleration support
    """

    def __init__(
        self,
        models: dict[str, BaseNeuralModel] | None = None,
        model_weights: dict[str, float] | None = None,
        device: str = "auto",
        enable_uncertainty: bool = True,
        max_sequence_length: int = 60,
        prediction_horizons: list[int] = [1, 5, 15, 30],
    ):
        """
        Initialize the neural predictor.

        Args:
            models: Dictionary of model name -> model instance
            model_weights: Dictionary of model name -> ensemble weight
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            enable_uncertainty: Whether to enable uncertainty estimation
            max_sequence_length: Maximum sequence length to process
            prediction_horizons: List of prediction horizons
        """
        self.models = models or {}
        self.model_weights = model_weights or {}
        self.enable_uncertainty = enable_uncertainty
        self.max_sequence_length = max_sequence_length
        self.prediction_horizons = prediction_horizons

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Neural predictor initialized on device: {self.device}")

        # Move models to device and set to eval mode
        for model_name, model in self.models.items():
            model.to(self.device)
            model.eval()

        # Normalize model weights
        if self.model_weights:
            total_weight = sum(self.model_weights.values())
            self.model_weights = {
                name: weight / total_weight
                for name, weight in self.model_weights.items()
            }
        else:
            # Equal weights if not specified
            num_models = len(self.models)
            if num_models > 0:
                self.model_weights = {
                    name: 1.0 / num_models for name in self.models.keys()
                }

        # Initialize feature engineering components
        self.preprocessor = None
        self.indicator_extractor = TechnicalIndicatorFeatures()

        # Performance tracking
        self.prediction_times = []
        self.total_predictions = 0

    def add_model(self, name: str, model: BaseNeuralModel, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.

        Args:
            name: Name of the model
            model: Model instance
            weight: Ensemble weight for the model
        """
        model.to(self.device)
        model.eval()
        self.models[name] = model
        self.model_weights[name] = weight

        # Renormalize weights
        total_weight = sum(self.model_weights.values())
        self.model_weights = {
            name: weight / total_weight for name, weight in self.model_weights.items()
        }

        print(f"Added model '{name}' with weight {self.model_weights[name]:.3f}")

    def remove_model(self, name: str) -> None:
        """
        Remove a model from the ensemble.

        Args:
            name: Name of the model to remove
        """
        if name in self.models:
            del self.models[name]
            del self.model_weights[name]

            # Renormalize weights
            if self.model_weights:
                total_weight = sum(self.model_weights.values())
                self.model_weights = {
                    model_name: weight / total_weight
                    for model_name, weight in self.model_weights.items()
                }

            print(f"Removed model '{name}'")
        else:
            print(f"Model '{name}' not found in ensemble")

    def load_model(self, name: str, model_path: str, weight: float = 1.0) -> None:
        """
        Load a trained model from file.

        Args:
            name: Name for the loaded model
            model_path: Path to the saved model
            weight: Ensemble weight for the model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model based on the file extension or naming convention
        if "lstm" in model_path.name.lower():
            model = LSTMModel.load_model(str(model_path))
        elif (
            "transformer" in model_path.name.lower() or "tft" in model_path.name.lower()
        ):
            model = TemporalFusionTransformer.load_model(str(model_path))
        else:
            # Try to load as base model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            raise NotImplementedError("Auto-detection of model type not implemented")

        self.add_model(name, model, weight)

        # Load preprocessor from the first model
        if self.preprocessor is None and hasattr(model, "preprocessor"):
            self.preprocessor = model.preprocessor

    def preprocess_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Preprocess raw market data for prediction.

        Args:
            data: Raw OHLCV DataFrame

        Returns:
            Preprocessed tensor ready for model input
        """
        # Extract technical indicators
        processed_data = self.indicator_extractor.extract_all_features(data.copy())

        # Initialize preprocessor if needed
        if self.preprocessor is None:
            self.preprocessor = FinancialDataPreprocessor()
            self.preprocessor.fit(processed_data)

        # Transform data
        transformed_data = self.preprocessor.transform(processed_data)

        # Create sequences
        features = transformed_data[self.preprocessor.get_feature_names()].values

        # Take the last sequence_length points
        if len(features) >= self.max_sequence_length:
            sequence = features[-self.max_sequence_length :]
        else:
            # Pad with zeros if not enough data
            padding = np.zeros(
                (self.max_sequence_length - len(features), features.shape[1])
            )
            sequence = np.vstack([padding, features])

        # Convert to tensor and add batch dimension
        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def predict(
        self,
        data: pd.DataFrame | torch.Tensor,
        return_uncertainty: bool = None,
        return_individual_predictions: bool = False,
    ) -> dict[str, np.ndarray | dict]:
        """
        Make ensemble predictions on input data.

        Args:
            data: Input data (DataFrame or preprocessed tensor)
            return_uncertainty: Whether to return uncertainty estimates
            return_individual_predictions: Whether to return individual model predictions

        Returns:
            Dictionary with predictions and optional uncertainty/individual results
        """
        start_time = time.time()

        if return_uncertainty is None:
            return_uncertainty = self.enable_uncertainty

        # Preprocess data if needed
        if isinstance(data, pd.DataFrame):
            input_tensor = self.preprocess_data(data)
        else:
            input_tensor = data.to(self.device)

        # Collect predictions from all models
        model_predictions = {}
        model_uncertainties = {}

        with torch.no_grad():
            for model_name, model in self.models.items():
                try:
                    # Get basic predictions
                    pred = model(input_tensor)
                    model_predictions[model_name] = pred.cpu().numpy()

                    # Get uncertainty if supported and requested
                    if return_uncertainty and hasattr(
                        model, "forward_with_uncertainty"
                    ):
                        _, uncertainty = model.forward_with_uncertainty(input_tensor)
                        model_uncertainties[model_name] = uncertainty.cpu().numpy()

                except Exception as e:
                    print(f"Warning: Model {model_name} prediction failed: {e}")
                    continue

        if not model_predictions:
            raise RuntimeError("No models produced valid predictions")

        # Ensemble predictions using weighted average
        ensemble_pred = np.zeros_like(list(model_predictions.values())[0])
        total_weight = 0

        for model_name, pred in model_predictions.items():
            weight = self.model_weights.get(model_name, 0)
            ensemble_pred += pred * weight
            total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        # Calculate ensemble uncertainty
        ensemble_uncertainty = None
        if return_uncertainty and model_uncertainties:
            # Use weighted average of uncertainties plus model disagreement
            uncertainty_sum = np.zeros_like(list(model_uncertainties.values())[0])
            disagreement = np.zeros_like(ensemble_pred)

            for model_name, uncertainty in model_uncertainties.items():
                weight = self.model_weights.get(model_name, 0)
                uncertainty_sum += uncertainty * weight

                # Add model disagreement (variance across models)
                pred_diff = model_predictions[model_name] - ensemble_pred
                disagreement += (pred_diff**2) * weight

            ensemble_uncertainty = uncertainty_sum + np.sqrt(disagreement)

        # Track performance
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        self.total_predictions += 1

        # Keep only last 1000 timing measurements
        if len(self.prediction_times) > 1000:
            self.prediction_times = self.prediction_times[-1000:]

        # Prepare result
        result = {
            "ensemble_prediction": ensemble_pred,
            "prediction_time_ms": prediction_time * 1000,
            "num_models": len(model_predictions),
            "prediction_horizons": self.prediction_horizons,
        }

        if return_uncertainty and ensemble_uncertainty is not None:
            result["uncertainty"] = ensemble_uncertainty

        if return_individual_predictions:
            result["individual_predictions"] = model_predictions
            if model_uncertainties:
                result["individual_uncertainties"] = model_uncertainties

        return result

    async def predict_async(
        self,
        data: pd.DataFrame | torch.Tensor,
        return_uncertainty: bool = None,
        return_individual_predictions: bool = False,
    ) -> dict[str, np.ndarray | dict]:
        """
        Async version of predict for concurrent processing.

        Args:
            data: Input data
            return_uncertainty: Whether to return uncertainty estimates
            return_individual_predictions: Whether to return individual model predictions

        Returns:
            Prediction results dictionary
        """
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.predict, data, return_uncertainty, return_individual_predictions
        )

    def predict_batch(
        self,
        data_batch: list[pd.DataFrame | torch.Tensor],
        return_uncertainty: bool = None,
        return_individual_predictions: bool = False,
    ) -> list[dict[str, np.ndarray | dict]]:
        """
        Make predictions on a batch of inputs.

        Args:
            data_batch: List of input data
            return_uncertainty: Whether to return uncertainty estimates
            return_individual_predictions: Whether to return individual model predictions

        Returns:
            List of prediction results
        """
        results = []

        for data in data_batch:
            try:
                result = self.predict(
                    data,
                    return_uncertainty=return_uncertainty,
                    return_individual_predictions=return_individual_predictions,
                )
                results.append(result)
            except Exception as e:
                print(f"Warning: Batch prediction failed for one sample: {e}")
                # Add empty result to maintain batch size
                results.append({"ensemble_prediction": None, "error": str(e)})

        return results

    def get_performance_stats(self) -> dict[str, float]:
        """
        Get performance statistics for the predictor.

        Returns:
            Dictionary with performance metrics
        """
        if not self.prediction_times:
            return {"total_predictions": 0}

        times_ms = [t * 1000 for t in self.prediction_times]

        return {
            "total_predictions": self.total_predictions,
            "avg_prediction_time_ms": np.mean(times_ms),
            "median_prediction_time_ms": np.median(times_ms),
            "p95_prediction_time_ms": np.percentile(times_ms, 95),
            "p99_prediction_time_ms": np.percentile(times_ms, 99),
            "max_prediction_time_ms": np.max(times_ms),
            "min_prediction_time_ms": np.min(times_ms),
            "predictions_per_second": 1.0 / np.mean(self.prediction_times)
            if self.prediction_times
            else 0,
        }

    def optimize_for_inference(self) -> None:
        """
        Optimize models for faster inference.
        """
        for model_name, model in self.models.items():
            try:
                # Compile model for faster inference (PyTorch 2.0+)
                if hasattr(torch, "compile"):
                    model = torch.compile(model, mode="reduce-overhead")
                    self.models[model_name] = model
                    print(f"Compiled model '{model_name}' for faster inference")

                # Enable inference mode optimizations
                model.eval()

                # Set memory efficient attention if available
                if hasattr(model, "set_memory_efficient_attention"):
                    model.set_memory_efficient_attention(True)

            except Exception as e:
                print(f"Warning: Could not optimize model '{model_name}': {e}")

        print("Inference optimization completed")

    def warm_up(self, num_warmup: int = 10) -> None:
        """
        Warm up the predictor with dummy predictions to optimize performance.

        Args:
            num_warmup: Number of warmup predictions to make
        """
        print(f"Warming up predictor with {num_warmup} dummy predictions...")

        # Create dummy data
        dummy_features = torch.randn(
            1,
            self.max_sequence_length,
            len(self.preprocessor.get_feature_names()) if self.preprocessor else 50,
        ).to(self.device)

        start_time = time.time()

        for i in range(num_warmup):
            try:
                with torch.no_grad():
                    for model in self.models.values():
                        _ = model(dummy_features)
            except Exception as e:
                print(f"Warning: Warmup prediction {i} failed: {e}")

        warmup_time = time.time() - start_time
        avg_warmup_time = (warmup_time / num_warmup) * 1000

        print(
            f"Warmup completed in {warmup_time:.2f}s "
            f"(avg: {avg_warmup_time:.2f}ms per prediction)"
        )

    def get_model_info(self) -> dict[str, dict]:
        """
        Get information about loaded models.

        Returns:
            Dictionary with model information
        """
        model_info = {}

        for model_name, model in self.models.items():
            info = {
                "model_type": type(model).__name__,
                "ensemble_weight": self.model_weights.get(model_name, 0),
                "device": str(next(model.parameters()).device),
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "prediction_horizons": getattr(
                    model, "prediction_horizons", self.prediction_horizons
                ),
            }

            # Add model-specific information
            if hasattr(model, "get_feature_importance"):
                try:
                    info["feature_importance"] = model.get_feature_importance()
                except:
                    pass

            model_info[model_name] = info

        return model_info
