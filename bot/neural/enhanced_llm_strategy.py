"""
Enhanced LLM Strategy with Neural Network Integration

This module creates a hybrid trading strategy that combines the quantitative
predictions from neural networks with the qualitative reasoning from LLM agents.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..risk.risk_manager import RiskManager
from ..strategy.llm_agent import LLMAgent
from ..trading_types import MarketData, TradeAction
from .inference.predictor import NeuralPredictor

logger = logging.getLogger(__name__)


class NeuralEnhancedLLMStrategy:
    """
    Hybrid trading strategy combining neural network predictions with LLM reasoning.

    This strategy uses neural networks to provide quantitative price predictions
    and uncertainty estimates, which are then combined with LLM-based qualitative
    analysis for final trading decisions.

    Key Features:
    - Neural network price predictions (1-step to 30-step ahead)
    - LLM reasoning and market context analysis
    - Uncertainty-aware position sizing
    - Risk management integration
    - Performance tracking and adaptation
    """

    def __init__(
        self,
        neural_predictor: NeuralPredictor,
        llm_agent: LLMAgent,
        risk_manager: RiskManager,
        neural_weight: float = 0.6,
        llm_weight: float = 0.4,
        confidence_threshold: float = 0.7,
        enable_uncertainty_scaling: bool = True,
        prediction_horizons: list[int] = [1, 5, 15, 30],
    ):
        """
        Initialize the enhanced LLM strategy.

        Args:
            neural_predictor: Neural network predictor instance
            llm_agent: LLM agent for reasoning
            risk_manager: Risk management system
            neural_weight: Weight for neural predictions (0-1)
            llm_weight: Weight for LLM analysis (0-1)
            confidence_threshold: Minimum confidence for trades
            enable_uncertainty_scaling: Whether to scale positions by uncertainty
            prediction_horizons: List of prediction horizons to use
        """
        self.neural_predictor = neural_predictor
        self.llm_agent = llm_agent
        self.risk_manager = risk_manager
        self.neural_weight = neural_weight
        self.llm_weight = llm_weight
        self.confidence_threshold = confidence_threshold
        self.enable_uncertainty_scaling = enable_uncertainty_scaling
        self.prediction_horizons = prediction_horizons

        # Normalize weights
        total_weight = neural_weight + llm_weight
        self.neural_weight = neural_weight / total_weight
        self.llm_weight = llm_weight / total_weight

        # Performance tracking
        self.trade_history = []
        self.prediction_accuracy = {"neural": [], "llm": [], "combined": []}
        self.total_trades = 0
        self.successful_trades = 0

        logger.info(
            f"Initialized NeuralEnhancedLLMStrategy with weights: "
            f"Neural={self.neural_weight:.2f}, LLM={self.llm_weight:.2f}"
        )

    async def analyze_market(self, market_data: MarketData) -> TradeAction:
        """
        Analyze market conditions and generate trading signals.

        Args:
            market_data: Current market data and indicators

        Returns:
            Enhanced trade action with combined neural and LLM analysis
        """
        try:
            # Step 1: Get neural network predictions
            neural_analysis = await self._get_neural_analysis(market_data)

            # Step 2: Get LLM analysis with neural context
            llm_analysis = await self._get_llm_analysis(market_data, neural_analysis)

            # Step 3: Combine analyses
            combined_action = self._combine_analyses(
                neural_analysis, llm_analysis, market_data
            )

            # Step 4: Apply risk management
            final_action = self.risk_manager.validate_trade_action(
                combined_action, market_data
            )

            # Step 5: Track performance
            self._track_decision(neural_analysis, llm_analysis, final_action)

            return final_action

        except Exception as e:
            logger.error(f"Error in enhanced market analysis: {e}")
            return self._get_safe_action("Analysis error occurred")

    async def _get_neural_analysis(self, market_data: MarketData) -> dict[str, Any]:
        """
        Get neural network analysis and predictions.

        Args:
            market_data: Current market data

        Returns:
            Dictionary with neural analysis results
        """
        try:
            # Convert market data to DataFrame for neural network
            df = self._market_data_to_dataframe(market_data)

            # Get predictions with uncertainty
            prediction_result = await self.neural_predictor.predict_async(
                df, return_uncertainty=True, return_individual_predictions=True
            )

            # Extract key information
            predictions = prediction_result["ensemble_prediction"]
            uncertainties = prediction_result.get(
                "uncertainty", np.zeros_like(predictions)
            )

            # Calculate confidence and signals
            avg_prediction = np.mean(predictions)
            avg_uncertainty = np.mean(uncertainties)
            confidence = max(
                0, 1 - avg_uncertainty
            )  # Convert uncertainty to confidence

            # Determine signal strength and direction
            if avg_prediction > 0.02:  # 2% positive return threshold
                signal = "LONG"
                signal_strength = min(
                    1.0, avg_prediction * 10
                )  # Scale prediction to strength
            elif avg_prediction < -0.02:  # 2% negative return threshold
                signal = "SHORT"
                signal_strength = min(1.0, abs(avg_prediction) * 10)
            else:
                signal = "HOLD"
                signal_strength = 0.0

            # Calculate position size based on confidence and uncertainty
            base_size = 0.2  # 20% base position
            if self.enable_uncertainty_scaling:
                uncertainty_factor = max(0.1, 1 - avg_uncertainty)
                suggested_size = base_size * confidence * uncertainty_factor
            else:
                suggested_size = base_size * confidence

            # Multi-horizon analysis
            horizon_analysis = {}
            for i, horizon in enumerate(self.prediction_horizons):
                if i < len(predictions):
                    horizon_analysis[f"{horizon}_step"] = {
                        "prediction": float(predictions[i]),
                        "uncertainty": float(uncertainties[i])
                        if i < len(uncertainties)
                        else 0.0,
                        "confidence": float(
                            max(0, 1 - uncertainties[i])
                            if i < len(uncertainties)
                            else 0.5
                        ),
                    }

            return {
                "signal": signal,
                "signal_strength": signal_strength,
                "confidence": confidence,
                "avg_prediction": avg_prediction,
                "avg_uncertainty": avg_uncertainty,
                "suggested_size": suggested_size,
                "horizon_analysis": horizon_analysis,
                "prediction_time_ms": prediction_result.get("prediction_time_ms", 0),
                "num_models": prediction_result.get("num_models", 0),
                "individual_predictions": prediction_result.get(
                    "individual_predictions", {}
                ),
                "raw_predictions": predictions.tolist(),
                "raw_uncertainties": uncertainties.tolist(),
            }

        except Exception as e:
            logger.error(f"Error in neural analysis: {e}")
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "confidence": 0.0,
                "avg_prediction": 0.0,
                "avg_uncertainty": 1.0,
                "suggested_size": 0.0,
                "horizon_analysis": {},
                "error": str(e),
            }

    async def _get_llm_analysis(
        self, market_data: MarketData, neural_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get LLM analysis enhanced with neural context.

        Args:
            market_data: Current market data
            neural_analysis: Neural network analysis results

        Returns:
            Dictionary with LLM analysis results
        """
        try:
            # Create enhanced prompt with neural context
            neural_context = self._format_neural_context(neural_analysis)

            # Get LLM analysis
            llm_action = await self.llm_agent.analyze_market_data(market_data)

            # Extract LLM confidence and reasoning
            llm_confidence = getattr(llm_action, "confidence", 0.7)
            llm_rationale = getattr(llm_action, "rationale", "No rationale provided")

            # Map LLM action to signal
            llm_signal = llm_action.action if hasattr(llm_action, "action") else "HOLD"

            # Calculate LLM signal strength based on size and confidence
            llm_size = getattr(llm_action, "size_pct", 0) / 100.0
            llm_signal_strength = llm_size * llm_confidence

            return {
                "signal": llm_signal,
                "signal_strength": llm_signal_strength,
                "confidence": llm_confidence,
                "rationale": llm_rationale,
                "suggested_size": llm_size,
                "take_profit_pct": getattr(llm_action, "take_profit_pct", 2.0),
                "stop_loss_pct": getattr(llm_action, "stop_loss_pct", 1.5),
                "neural_context_used": neural_context,
                "raw_action": llm_action,
            }

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                "signal": "HOLD",
                "signal_strength": 0.0,
                "confidence": 0.0,
                "rationale": f"LLM analysis error: {e}",
                "suggested_size": 0.0,
                "take_profit_pct": 2.0,
                "stop_loss_pct": 1.5,
                "error": str(e),
            }

    def _combine_analyses(
        self,
        neural_analysis: dict[str, Any],
        llm_analysis: dict[str, Any],
        market_data: MarketData,
    ) -> TradeAction:
        """
        Combine neural and LLM analyses into a final trading decision.

        Args:
            neural_analysis: Neural network analysis
            llm_analysis: LLM analysis
            market_data: Current market data

        Returns:
            Combined trade action
        """
        # Get signals and confidences
        neural_signal = neural_analysis.get("signal", "HOLD")
        neural_confidence = neural_analysis.get("confidence", 0.0)
        neural_strength = neural_analysis.get("signal_strength", 0.0)

        llm_signal = llm_analysis.get("signal", "HOLD")
        llm_confidence = llm_analysis.get("confidence", 0.0)
        llm_strength = llm_analysis.get("signal_strength", 0.0)

        # Combine signals using weighted voting
        signal_scores = {"LONG": 0.0, "SHORT": 0.0, "HOLD": 0.0, "CLOSE": 0.0}

        # Add neural vote
        if neural_signal in signal_scores:
            signal_scores[neural_signal] += (
                self.neural_weight * neural_strength * neural_confidence
            )

        # Add LLM vote
        if llm_signal in signal_scores:
            signal_scores[llm_signal] += self.llm_weight * llm_strength * llm_confidence

        # Add base HOLD preference for safety
        signal_scores["HOLD"] += 0.1

        # Select signal with highest score
        final_signal = max(signal_scores, key=signal_scores.get)
        final_strength = signal_scores[final_signal]

        # Calculate combined confidence
        combined_confidence = (
            self.neural_weight * neural_confidence + self.llm_weight * llm_confidence
        )

        # Apply confidence threshold
        if combined_confidence < self.confidence_threshold:
            final_signal = "HOLD"
            final_strength = 0.0

        # Calculate position size
        neural_size = neural_analysis.get("suggested_size", 0.0)
        llm_size = llm_analysis.get("suggested_size", 0.0)

        combined_size = (
            self.neural_weight * neural_size + self.llm_weight * llm_size
        ) * combined_confidence

        # Convert size to percentage (0-100)
        size_pct = max(0, min(50, combined_size * 100))  # Cap at 50%

        # Calculate take profit and stop loss
        take_profit_pct = llm_analysis.get("take_profit_pct", 2.0)
        stop_loss_pct = llm_analysis.get("stop_loss_pct", 1.5)

        # Adjust TP/SL based on neural uncertainty
        neural_uncertainty = neural_analysis.get("avg_uncertainty", 0.5)
        if neural_uncertainty > 0.7:  # High uncertainty
            take_profit_pct *= 0.8  # Tighter TP
            stop_loss_pct *= 1.2  # Wider SL
        elif neural_uncertainty < 0.3:  # Low uncertainty
            take_profit_pct *= 1.2  # Wider TP
            stop_loss_pct *= 0.8  # Tighter SL

        # Create comprehensive rationale
        rationale = self._create_combined_rationale(
            neural_analysis, llm_analysis, final_signal, combined_confidence
        )

        # Create trade action
        trade_action = TradeAction(
            action=final_signal,
            size_pct=int(size_pct),
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            rationale=rationale,
            confidence=combined_confidence,
            metadata={
                "neural_analysis": neural_analysis,
                "llm_analysis": llm_analysis,
                "signal_scores": signal_scores,
                "neural_weight": self.neural_weight,
                "llm_weight": self.llm_weight,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        return trade_action

    def _create_combined_rationale(
        self,
        neural_analysis: dict[str, Any],
        llm_analysis: dict[str, Any],
        final_signal: str,
        combined_confidence: float,
    ) -> str:
        """Create a comprehensive rationale combining both analyses."""

        neural_signal = neural_analysis.get("signal", "HOLD")
        neural_confidence = neural_analysis.get("confidence", 0.0)
        neural_prediction = neural_analysis.get("avg_prediction", 0.0)

        llm_signal = llm_analysis.get("signal", "HOLD")
        llm_confidence = llm_analysis.get("confidence", 0.0)
        llm_rationale = llm_analysis.get("rationale", "")

        # Start with signal agreement/disagreement
        if neural_signal == llm_signal:
            agreement = f"✅ Neural & LLM agree: {neural_signal}"
        else:
            agreement = f"⚠️ Mixed signals: Neural={neural_signal}, LLM={llm_signal}"

        # Add neural insights
        neural_insight = (
            f"🧠 Neural: {neural_prediction:.3f} return prediction "
            f"(confidence: {neural_confidence:.2f})"
        )

        # Add LLM reasoning (truncated)
        llm_insight = (
            f"🤖 LLM: {llm_rationale[:100]}{'...' if len(llm_rationale) > 100 else ''}"
        )

        # Final decision
        decision = f"📊 Combined Decision: {final_signal} (confidence: {combined_confidence:.2f})"

        return f"{agreement}\n{neural_insight}\n{llm_insight}\n{decision}"

    def _market_data_to_dataframe(self, market_data: MarketData) -> pd.DataFrame:
        """Convert MarketData to DataFrame for neural network input."""

        # Extract OHLCV data
        data = {
            "open": [market_data.open],
            "high": [market_data.high],
            "low": [market_data.low],
            "close": [market_data.close],
            "volume": [market_data.volume],
        }

        # Add indicators if available
        if hasattr(market_data, "indicators") and market_data.indicators:
            indicators = market_data.indicators
            for attr in dir(indicators):
                if not attr.startswith("_"):
                    value = getattr(indicators, attr)
                    if isinstance(value, (int, float)):
                        data[attr] = [value]

        # Create DataFrame with timestamp
        df = pd.DataFrame(data)
        df.index = [datetime.now(UTC)]

        return df

    def _format_neural_context(self, neural_analysis: dict[str, Any]) -> str:
        """Format neural analysis for LLM context."""

        signal = neural_analysis.get("signal", "HOLD")
        confidence = neural_analysis.get("confidence", 0.0)
        prediction = neural_analysis.get("avg_prediction", 0.0)
        uncertainty = neural_analysis.get("avg_uncertainty", 0.5)

        context = f"""
        Neural Network Analysis:
        - Signal: {signal}
        - Confidence: {confidence:.2f}
        - Return Prediction: {prediction:.3f} ({prediction*100:.1f}%)
        - Uncertainty: {uncertainty:.2f}
        
        Multi-horizon Predictions:
        """

        horizon_analysis = neural_analysis.get("horizon_analysis", {})
        for horizon, data in horizon_analysis.items():
            pred = data.get("prediction", 0.0)
            conf = data.get("confidence", 0.0)
            context += f"- {horizon}: {pred:.3f} (conf: {conf:.2f})\n"

        return context.strip()

    def _get_safe_action(self, reason: str) -> TradeAction:
        """Get a safe default action."""
        return TradeAction(
            action="HOLD",
            size_pct=0,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale=f"Enhanced strategy: {reason} - defaulting to HOLD for safety",
            confidence=0.0,
        )

    def _track_decision(
        self,
        neural_analysis: dict[str, Any],
        llm_analysis: dict[str, Any],
        final_action: TradeAction,
    ) -> None:
        """Track decision for performance analysis."""

        decision_record = {
            "timestamp": datetime.now(UTC),
            "neural_signal": neural_analysis.get("signal", "HOLD"),
            "neural_confidence": neural_analysis.get("confidence", 0.0),
            "llm_signal": llm_analysis.get("signal", "HOLD"),
            "llm_confidence": llm_analysis.get("confidence", 0.0),
            "final_action": final_action.action,
            "final_confidence": final_action.confidence,
            "size_pct": final_action.size_pct,
        }

        self.trade_history.append(decision_record)

        # Keep only last 1000 decisions
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

        self.total_trades += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the enhanced strategy."""

        if not self.trade_history:
            return {"total_decisions": 0}

        recent_decisions = self.trade_history[-100:]  # Last 100 decisions

        # Signal distribution
        signals = [d["final_action"] for d in recent_decisions]
        signal_counts = {
            signal: signals.count(signal)
            for signal in ["LONG", "SHORT", "HOLD", "CLOSE"]
        }

        # Confidence statistics
        confidences = [d["final_confidence"] for d in recent_decisions]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Agreement rate between neural and LLM
        agreements = [d["neural_signal"] == d["llm_signal"] for d in recent_decisions]
        agreement_rate = np.mean(agreements) if agreements else 0.0

        return {
            "total_decisions": len(self.trade_history),
            "recent_decisions": len(recent_decisions),
            "signal_distribution": signal_counts,
            "avg_confidence": avg_confidence,
            "neural_llm_agreement_rate": agreement_rate,
            "neural_weight": self.neural_weight,
            "llm_weight": self.llm_weight,
            "confidence_threshold": self.confidence_threshold,
        }

    def update_weights(self, neural_weight: float, llm_weight: float) -> None:
        """Update the ensemble weights dynamically."""

        total_weight = neural_weight + llm_weight
        self.neural_weight = neural_weight / total_weight
        self.llm_weight = llm_weight / total_weight

        logger.info(
            f"Updated weights: Neural={self.neural_weight:.2f}, LLM={self.llm_weight:.2f}"
        )

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold for trading."""

        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Updated confidence threshold: {self.confidence_threshold:.2f}")

    async def backtest_strategy(
        self, historical_data: pd.DataFrame, initial_balance: float = 10000.0
    ) -> dict[str, Any]:
        """
        Backtest the enhanced strategy on historical data.

        Args:
            historical_data: Historical OHLCV data
            initial_balance: Starting balance for simulation

        Returns:
            Backtest results dictionary
        """
        # This would implement a comprehensive backtesting framework
        # For now, return a placeholder
        return {
            "message": "Backtesting not yet implemented",
            "data_points": len(historical_data),
            "initial_balance": initial_balance,
        }
