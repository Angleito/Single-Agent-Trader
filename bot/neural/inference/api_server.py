"""
FastAPI Server for Real-Time Neural Network Predictions

This module provides a high-performance REST API server for serving neural
network predictions with WebSocket streaming capabilities.
"""

import asyncio
import json
import os
import time
from datetime import UTC, datetime

import pandas as pd
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .predictor import NeuralPredictor


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    data: list[dict] = Field(..., description="List of OHLCV data points")
    return_uncertainty: bool = Field(
        default=True, description="Whether to return uncertainty estimates"
    )
    return_individual_predictions: bool = Field(
        default=False, description="Whether to return individual model predictions"
    )

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "timestamp": "2024-01-01T00:00:00Z",
                        "open": 42000.0,
                        "high": 42500.0,
                        "low": 41800.0,
                        "close": 42200.0,
                        "volume": 1234.56,
                    }
                ],
                "return_uncertainty": True,
                "return_individual_predictions": False,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    ensemble_prediction: list[float] = Field(
        ..., description="Ensemble model predictions"
    )
    prediction_horizons: list[int] = Field(
        ..., description="Prediction horizons in steps"
    )
    uncertainty: list[float] | None = Field(
        None, description="Uncertainty estimates"
    )
    individual_predictions: dict[str, list[float]] | None = Field(
        None, description="Individual model predictions"
    )
    prediction_time_ms: float = Field(
        ..., description="Prediction time in milliseconds"
    )
    num_models: int = Field(..., description="Number of models in ensemble")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    batch_data: list[list[dict]] = Field(
        ..., description="List of data sequences for batch prediction"
    )
    return_uncertainty: bool = Field(
        default=True, description="Whether to return uncertainty estimates"
    )
    return_individual_predictions: bool = Field(
        default=False, description="Whether to return individual model predictions"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: list[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    batch_size: int = Field(..., description="Batch size")
    total_time_ms: float = Field(..., description="Total batch processing time")


class ModelInfo(BaseModel):
    """Model information response."""

    model_type: str
    ensemble_weight: float
    device: str
    num_parameters: int
    prediction_horizons: list[int]
    feature_importance: dict[str, float] | None = None


class ServerStatus(BaseModel):
    """Server status response."""

    status: str = "healthy"
    uptime_seconds: float
    total_predictions: int
    avg_prediction_time_ms: float
    models: dict[str, ModelInfo]
    device: str
    memory_usage_mb: float | None = None


class PredictionAPIServer:
    """
    FastAPI server for neural network predictions.

    This server provides REST endpoints and WebSocket streaming for real-time
    cryptocurrency price predictions using trained neural network models.

    Key Features:
    - REST API for single and batch predictions
    - WebSocket streaming for real-time updates
    - Model management and health monitoring
    - Performance metrics and monitoring
    - CORS support for web applications
    """

    def __init__(
        self,
        predictor: NeuralPredictor,
        host: str | None = None,
        port: int | None = None,
        enable_cors: bool = True,
        cors_origins: list[str] = ["*"],
    ):
        """
        Initialize the prediction API server.

        Args:
            predictor: Neural predictor instance
            host: Server host address (defaults to 127.0.0.1 for security, configurable via NEURAL_API_HOST)
            port: Server port number (defaults to 8000, configurable via NEURAL_API_PORT)
            enable_cors: Whether to enable CORS
            cors_origins: List of allowed CORS origins
        """
        self.predictor = predictor
        
        # Configure host with security-first defaults
        # Use environment variable if available, otherwise parameter, otherwise secure default
        self.host = (
            os.getenv("NEURAL_API_HOST") or 
            host or 
            "127.0.0.1"  # Secure default: localhost only
        )
        
        # Configure port with environment variable support
        self.port = (
            int(os.getenv("NEURAL_API_PORT", "0")) or
            port or
            8000  # Default port
        )
        
        self.start_time = time.time()

        # Initialize FastAPI app
        self.app = FastAPI(
            title="Neural Network Prediction API",
            description="Real-time cryptocurrency price prediction using neural networks",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Enable CORS if requested
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # WebSocket connections
        self.websocket_connections: list[WebSocket] = []

        # Setup routes
        self._setup_routes()

        # Background tasks
        self.background_tasks = set()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/", response_model=dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "Neural Network Prediction API",
                "version": "1.0.0",
                "status": "healthy",
            }

        @self.app.get("/health", response_model=ServerStatus)
        async def health():
            """Health check endpoint."""
            uptime = time.time() - self.start_time
            perf_stats = self.predictor.get_performance_stats()
            model_info = self.predictor.get_model_info()

            # Convert model info to Pydantic models
            models = {}
            for name, info in model_info.items():
                models[name] = ModelInfo(**info)

            return ServerStatus(
                uptime_seconds=uptime,
                total_predictions=perf_stats.get("total_predictions", 0),
                avg_prediction_time_ms=perf_stats.get("avg_prediction_time_ms", 0),
                models=models,
                device=str(self.predictor.device),
            )

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Make a single prediction."""
            try:
                # Convert request data to DataFrame
                df = pd.DataFrame(request.data)

                # Ensure timestamp column is datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)

                # Make prediction
                result = await self.predictor.predict_async(
                    df,
                    return_uncertainty=request.return_uncertainty,
                    return_individual_predictions=request.return_individual_predictions,
                )

                # Convert numpy arrays to lists for JSON serialization
                response_data = {
                    "ensemble_prediction": result["ensemble_prediction"].tolist(),
                    "prediction_horizons": result["prediction_horizons"],
                    "prediction_time_ms": result["prediction_time_ms"],
                    "num_models": result["num_models"],
                    "timestamp": datetime.now(UTC).isoformat(),
                }

                if "uncertainty" in result:
                    response_data["uncertainty"] = result["uncertainty"].tolist()

                if "individual_predictions" in result:
                    response_data["individual_predictions"] = {
                        name: pred.tolist()
                        for name, pred in result["individual_predictions"].items()
                    }

                return PredictionResponse(**response_data)

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/predict/batch", response_model=BatchPredictionResponse)
        async def predict_batch(request: BatchPredictionRequest):
            """Make batch predictions."""
            try:
                start_time = time.time()

                # Convert batch data to DataFrames
                batch_dfs = []
                for data_seq in request.batch_data:
                    df = pd.DataFrame(data_seq)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df.set_index("timestamp", inplace=True)
                    batch_dfs.append(df)

                # Make batch predictions
                results = self.predictor.predict_batch(
                    batch_dfs,
                    return_uncertainty=request.return_uncertainty,
                    return_individual_predictions=request.return_individual_predictions,
                )

                # Convert results to response format
                predictions = []
                for result in results:
                    if result.get("ensemble_prediction") is not None:
                        response_data = {
                            "ensemble_prediction": result[
                                "ensemble_prediction"
                            ].tolist(),
                            "prediction_horizons": result["prediction_horizons"],
                            "prediction_time_ms": result["prediction_time_ms"],
                            "num_models": result["num_models"],
                            "timestamp": datetime.now(UTC).isoformat(),
                        }

                        if "uncertainty" in result:
                            response_data["uncertainty"] = result[
                                "uncertainty"
                            ].tolist()

                        if "individual_predictions" in result:
                            response_data["individual_predictions"] = {
                                name: pred.tolist()
                                for name, pred in result[
                                    "individual_predictions"
                                ].items()
                            }

                        predictions.append(PredictionResponse(**response_data))
                    else:
                        # Handle failed predictions
                        predictions.append(
                            PredictionResponse(
                                ensemble_prediction=[0.0]
                                * len(self.predictor.prediction_horizons),
                                prediction_horizons=self.predictor.prediction_horizons,
                                prediction_time_ms=0.0,
                                num_models=0,
                                timestamp=datetime.now(UTC).isoformat(),
                            )
                        )

                total_time = (time.time() - start_time) * 1000

                return BatchPredictionResponse(
                    predictions=predictions,
                    batch_size=len(predictions),
                    total_time_ms=total_time,
                )

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/models", response_model=dict[str, ModelInfo])
        async def get_models():
            """Get information about loaded models."""
            model_info = self.predictor.get_model_info()
            return {name: ModelInfo(**info) for name, info in model_info.items()}

        @self.app.get("/performance", response_model=dict[str, float])
        async def get_performance():
            """Get performance statistics."""
            return self.predictor.get_performance_stats()

        @self.app.websocket("/ws/predictions")
        async def websocket_predictions(websocket: WebSocket):
            """WebSocket endpoint for real-time predictions."""
            await websocket.accept()
            self.websocket_connections.append(websocket)

            try:
                while True:
                    # Wait for client message
                    data = await websocket.receive_text()

                    try:
                        # Parse request
                        request_data = json.loads(data)

                        # Convert to DataFrame
                        df = pd.DataFrame(request_data.get("data", []))
                        if "timestamp" in df.columns:
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                            df.set_index("timestamp", inplace=True)

                        # Make prediction
                        result = await self.predictor.predict_async(
                            df,
                            return_uncertainty=request_data.get(
                                "return_uncertainty", True
                            ),
                            return_individual_predictions=request_data.get(
                                "return_individual_predictions", False
                            ),
                        )

                        # Prepare response
                        response = {
                            "type": "prediction",
                            "ensemble_prediction": result[
                                "ensemble_prediction"
                            ].tolist(),
                            "prediction_horizons": result["prediction_horizons"],
                            "prediction_time_ms": result["prediction_time_ms"],
                            "num_models": result["num_models"],
                            "timestamp": datetime.now(UTC).isoformat(),
                        }

                        if "uncertainty" in result:
                            response["uncertainty"] = result["uncertainty"].tolist()

                        if "individual_predictions" in result:
                            response["individual_predictions"] = {
                                name: pred.tolist()
                                for name, pred in result[
                                    "individual_predictions"
                                ].items()
                            }

                        # Send response
                        await websocket.send_text(json.dumps(response))

                    except json.JSONDecodeError:
                        await websocket.send_text(
                            json.dumps(
                                {"type": "error", "message": "Invalid JSON format"}
                            )
                        )
                    except Exception as e:
                        await websocket.send_text(
                            json.dumps({"type": "error", "message": str(e)})
                        )

            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)

        @self.app.post("/optimize")
        async def optimize_models():
            """Optimize models for faster inference."""
            try:
                self.predictor.optimize_for_inference()
                return {"message": "Models optimized for inference"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/warmup")
        async def warmup_models(num_warmup: int = 10):
            """Warm up models with dummy predictions."""
            try:
                self.predictor.warm_up(num_warmup=num_warmup)
                return {"message": f"Models warmed up with {num_warmup} predictions"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def broadcast_prediction(self, prediction_data: dict):
        """
        Broadcast prediction to all connected WebSocket clients.

        Args:
            prediction_data: Prediction data to broadcast
        """
        if not self.websocket_connections:
            return

        message = json.dumps(
            {
                "type": "broadcast",
                **prediction_data,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Send to all connected clients
        disconnected_clients = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except Exception:
                # Client disconnected
                disconnected_clients.append(websocket)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_connections.remove(client)

    def start_prediction_stream(self, data_source, interval_ms: int = 1000):
        """
        Start a background task that streams predictions from a data source.

        Args:
            data_source: Source of market data
            interval_ms: Interval between predictions in milliseconds
        """

        async def prediction_loop():
            while True:
                try:
                    # Get latest data from source
                    latest_data = await data_source.get_latest_data()

                    # Make prediction
                    result = await self.predictor.predict_async(latest_data)

                    # Broadcast to WebSocket clients
                    await self.broadcast_prediction(
                        {
                            "ensemble_prediction": result[
                                "ensemble_prediction"
                            ].tolist(),
                            "prediction_horizons": result["prediction_horizons"],
                            "prediction_time_ms": result["prediction_time_ms"],
                            "num_models": result["num_models"],
                        }
                    )

                    # Wait for next interval
                    await asyncio.sleep(interval_ms / 1000.0)

                except Exception as e:
                    print(f"Error in prediction stream: {e}")
                    await asyncio.sleep(1.0)  # Wait before retrying

        # Start background task
        task = asyncio.create_task(prediction_loop())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        return task

    def run(self, **kwargs):
        """
        Run the FastAPI server.

        Args:
            **kwargs: Additional arguments for uvicorn.run()
        """
        # Default uvicorn configuration
        config = {
            "host": self.host,
            "port": self.port,
            "log_level": "info",
            "access_log": True,
            "reload": False,
        }
        config.update(kwargs)

        print(
            f"Starting Neural Network Prediction API server on {self.host}:{self.port}"
        )
        print(f"API documentation available at http://{self.host}:{self.port}/docs")
        print(f"WebSocket endpoint: ws://{self.host}:{self.port}/ws/predictions")

        # Warm up predictor
        try:
            self.predictor.warm_up(num_warmup=5)
            self.predictor.optimize_for_inference()
        except Exception as e:
            print(f"Warning: Could not optimize predictor: {e}")

        # Run server
        uvicorn.run(self.app, **config)

    async def shutdown(self):
        """Shutdown the server gracefully."""
        # Close all WebSocket connections
        for websocket in self.websocket_connections:
            try:
                await websocket.close()
            except:
                pass

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        print("Neural Network Prediction API server shutdown complete")
