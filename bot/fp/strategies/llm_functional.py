"""Functional LLM strategy integration for AI-powered trading decisions."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ...types import Signal
from ..indicators.vumanchu_functional import VuManchuState
from ..types import MarketState, TradingParams


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM integration."""

    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 500
    timeout: float = 30.0
    max_context_window: int = 4000
    system_prompt: str | None = None


@dataclass(frozen=True)
class LLMContext:
    """Context for LLM decision making."""

    market_state: MarketState
    indicators: dict[str, Any]
    recent_trades: list[dict[str, Any]]
    market_sentiment: dict[str, Any] | None = None
    news_events: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Parsed LLM response with trading decision."""

    signal: Signal
    confidence: float
    reasoning: str
    risk_assessment: dict[str, Any]
    suggested_params: dict[str, Any] | None = None


def create_market_context(
    market_state: MarketState,
    vumanchu_state: VuManchuState,
    recent_trades: list[dict[str, Any]] = None,
    lookback_periods: int = 20,
) -> LLMContext:
    """Create context for LLM from market state and indicators."""
    # Extract relevant indicator values
    indicators = {
        "vumanchu_a": {
            "current": vumanchu_state.cipher_a[-1] if vumanchu_state.cipher_a else 0,
            "trend": _calculate_trend(vumanchu_state.cipher_a, lookback_periods),
            "signal": vumanchu_state.buy_signal or vumanchu_state.sell_signal,
            "divergence": vumanchu_state.bullish_divergence
            or vumanchu_state.bearish_divergence,
        },
        "vumanchu_b": {
            "wave_trend": (
                vumanchu_state.wave_trend[-1] if vumanchu_state.wave_trend else 0
            ),
            "money_flow": (
                vumanchu_state.money_flow[-1] if vumanchu_state.money_flow else 0
            ),
            "trend_direction": _get_trend_direction(vumanchu_state.wave_trend),
        },
        "price_action": {
            "current_price": market_state.current_price,
            "price_change_24h": _calculate_price_change(market_state),
            "volatility": _calculate_volatility(market_state),
            "support_resistance": _identify_support_resistance(market_state),
        },
        "volume": {
            "current": market_state.volume,
            "average": _calculate_average_volume(market_state),
            "trend": _calculate_volume_trend(market_state),
        },
    }

    return LLMContext(
        market_state=market_state,
        indicators=indicators,
        recent_trades=recent_trades or [],
    )


def generate_trading_prompt(
    context: LLMContext, params: TradingParams, include_examples: bool = True
) -> str:
    """Generate prompt for LLM trading decision."""
    prompt_parts = []

    # Market overview
    prompt_parts.append(
        f"""
Current Market State for {context.market_state.symbol}:
- Price: ${context.market_state.current_price:.2f}
- 24h Change: {context.indicators["price_action"]["price_change_24h"]:.2%}
- Volume: {context.market_state.volume:.2f} (Trend: {context.indicators["volume"]["trend"]})
- Volatility: {context.indicators["price_action"]["volatility"]:.2%}
"""
    )

    # Technical indicators
    prompt_parts.append(
        f"""
Technical Indicators:
- VuManchu Cipher A: {context.indicators["vumanchu_a"]["current"]:.2f} ({context.indicators["vumanchu_a"]["trend"]})
- VuManchu Wave Trend: {context.indicators["vumanchu_b"]["wave_trend"]:.2f}
- Money Flow: {context.indicators["vumanchu_b"]["money_flow"]:.2f}
- Trend Direction: {context.indicators["vumanchu_b"]["trend_direction"]}
"""
    )

    # Signals
    if context.indicators["vumanchu_a"]["signal"]:
        signal_type = (
            "BUY" if context.indicators["vumanchu_a"]["signal"] > 0 else "SELL"
        )
        prompt_parts.append(f"- Active Signal: {signal_type}")

    if context.indicators["vumanchu_a"]["divergence"]:
        div_type = (
            "Bullish"
            if context.indicators["vumanchu_a"]["divergence"] > 0
            else "Bearish"
        )
        prompt_parts.append(f"- Divergence Detected: {div_type}")

    # Recent trades context
    if context.recent_trades:
        prompt_parts.append("\nRecent Trading Activity:")
        for trade in context.recent_trades[-3:]:  # Last 3 trades
            prompt_parts.append(
                f"- {trade['action']} at ${trade['price']:.2f} "
                f"(P&L: {trade['pnl']:.2%})"
            )

    # Trading parameters
    prompt_parts.append(
        f"""
Trading Parameters:
- Risk per trade: {params.risk_per_trade:.1%}
- Maximum leverage: {params.max_leverage}x
- Stop loss range: {params.stop_loss_pct:.1%}
- Take profit range: {params.take_profit_pct:.1%}
"""
    )

    # Decision request
    prompt_parts.append(
        """
Based on the above analysis, provide a trading decision in the following JSON format:
{
    "action": "LONG" | "SHORT" | "HOLD",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of the decision",
    "risk_assessment": {
        "risk_level": "LOW" | "MEDIUM" | "HIGH",
        "key_risks": ["risk1", "risk2"],
        "mitigation": "How to manage identified risks"
    },
    "suggested_params": {
        "position_size": 0.0 to 1.0,
        "stop_loss": price level,
        "take_profit": price level
    }
}
"""
    )

    if include_examples:
        prompt_parts.append(_get_example_responses())

    return "\n".join(prompt_parts)


def parse_llm_response(response_text: str, market_state: MarketState) -> LLMResponse:
    """Parse LLM response into structured format."""
    try:
        # Extract JSON from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            return _create_hold_response("No valid JSON found in response")

        json_str = response_text[json_start:json_end]
        data = json.loads(json_str)

        # Map action to signal
        action = data.get("action", "HOLD").upper()
        signal_map = {"LONG": Signal.LONG, "SHORT": Signal.SHORT, "HOLD": Signal.HOLD}
        signal = signal_map.get(action, Signal.HOLD)

        # Extract confidence
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # Extract other fields
        reasoning = data.get("reasoning", "No reasoning provided")
        risk_assessment = data.get(
            "risk_assessment",
            {
                "risk_level": "MEDIUM",
                "key_risks": ["Market volatility"],
                "mitigation": "Use appropriate stop loss",
            },
        )

        # Process suggested parameters
        suggested_params = None
        if "suggested_params" in data and signal != Signal.HOLD:
            params = data["suggested_params"]
            suggested_params = {
                "position_size": float(params.get("position_size", 0.25)),
                "stop_loss": float(
                    params.get(
                        "stop_loss",
                        (
                            market_state.current_price * 0.98
                            if signal == Signal.LONG
                            else market_state.current_price * 1.02
                        ),
                    )
                ),
                "take_profit": float(
                    params.get(
                        "take_profit",
                        (
                            market_state.current_price * 1.02
                            if signal == Signal.LONG
                            else market_state.current_price * 0.98
                        ),
                    )
                ),
            }

        return LLMResponse(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            risk_assessment=risk_assessment,
            suggested_params=suggested_params,
        )

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return _create_hold_response(f"Error parsing response: {e!s}")


def adjust_confidence_by_market_conditions(
    response: LLMResponse, market_state: MarketState, volatility_threshold: float = 0.02
) -> LLMResponse:
    """Adjust LLM confidence based on market conditions."""
    adjusted_confidence = response.confidence

    # Calculate current volatility
    volatility = _calculate_volatility(market_state)

    # Reduce confidence in high volatility
    if volatility > volatility_threshold:
        volatility_penalty = min(0.3, (volatility - volatility_threshold) * 10)
        adjusted_confidence *= 1 - volatility_penalty

    # Reduce confidence for counter-trend trades
    trend = _calculate_price_trend(market_state)
    if (response.signal == Signal.SHORT and trend > 0.01) or (
        response.signal == Signal.LONG and trend < -0.01
    ):
        adjusted_confidence *= 0.8

    # Don't trade with low confidence
    if adjusted_confidence < 0.4 and response.signal != Signal.HOLD:
        return _create_hold_response(
            f"Confidence too low after adjustments: {adjusted_confidence:.2f}"
        )

    return LLMResponse(
        signal=response.signal,
        confidence=adjusted_confidence,
        reasoning=response.reasoning
        + f" (Adjusted confidence from {response.confidence:.2f})",
        risk_assessment=response.risk_assessment,
        suggested_params=response.suggested_params,
    )


def create_context_window(
    history: list[MarketState], max_size: int = 100, compression_ratio: float = 0.5
) -> list[MarketState]:
    """Create optimized context window from market history."""
    if len(history) <= max_size:
        return history

    # Keep recent data at full resolution
    recent_size = int(max_size * (1 - compression_ratio))
    recent_data = history[-recent_size:]

    # Compress older data
    older_data = history[:-recent_size]
    compression_step = len(older_data) // int(max_size * compression_ratio)
    compressed_data = older_data[:: max(1, compression_step)]

    return compressed_data + recent_data


def validate_llm_decision(
    response: LLMResponse, market_state: MarketState, params: TradingParams
) -> tuple[bool, str | None]:
    """Validate LLM decision against safety rules."""
    # Check confidence threshold
    if response.confidence < 0.5:
        return False, "Confidence below minimum threshold"

    # Validate suggested parameters if present
    if response.suggested_params:
        # Check position size
        if response.suggested_params["position_size"] > params.max_position_size:
            return False, "Suggested position size exceeds maximum"

        # Check stop loss is reasonable
        if response.signal == Signal.LONG:
            if response.suggested_params["stop_loss"] >= market_state.current_price:
                return False, "Invalid stop loss for long position"
        elif response.signal == Signal.SHORT:
            if response.suggested_params["stop_loss"] <= market_state.current_price:
                return False, "Invalid stop loss for short position"

    # Check risk assessment
    if (
        response.risk_assessment.get("risk_level") == "HIGH"
        and response.confidence < 0.7
    ):
        return False, "High risk with insufficient confidence"

    return True, None


# Helper functions


def _calculate_trend(values: list[float], periods: int) -> str:
    """Calculate trend direction from values."""
    if not values or len(values) < 2:
        return "NEUTRAL"

    recent = values[-min(periods, len(values)) :]
    if len(recent) < 2:
        return "NEUTRAL"

    slope = (recent[-1] - recent[0]) / len(recent)

    if slope > 0.1:
        return "STRONG_UP"
    if slope > 0.01:
        return "UP"
    if slope < -0.1:
        return "STRONG_DOWN"
    if slope < -0.01:
        return "DOWN"
    return "NEUTRAL"


def _get_trend_direction(wave_trend: list[float]) -> str:
    """Get trend direction from wave trend values."""
    if not wave_trend or len(wave_trend) < 2:
        return "NEUTRAL"

    current = wave_trend[-1]
    previous = wave_trend[-2]

    if current > previous + 0.5:
        return "BULLISH"
    if current < previous - 0.5:
        return "BEARISH"
    return "NEUTRAL"


def _calculate_price_change(market_state: MarketState) -> float:
    """Calculate 24h price change percentage."""
    if market_state.open_24h > 0:
        return (
            market_state.current_price - market_state.open_24h
        ) / market_state.open_24h
    return 0.0


def _calculate_volatility(market_state: MarketState) -> float:
    """Calculate price volatility."""
    if market_state.low_24h > 0:
        return (market_state.high_24h - market_state.low_24h) / market_state.low_24h
    return 0.0


def _calculate_average_volume(market_state: MarketState) -> float:
    """Calculate average volume."""
    # Simplified - in real implementation would use historical data
    return market_state.volume


def _calculate_volume_trend(market_state: MarketState) -> str:
    """Determine volume trend."""
    # Simplified - in real implementation would compare to historical average
    return "NORMAL"


def _identify_support_resistance(market_state: MarketState) -> dict[str, float]:
    """Identify support and resistance levels."""
    return {"support": market_state.low_24h, "resistance": market_state.high_24h}


def _calculate_price_trend(market_state: MarketState) -> float:
    """Calculate overall price trend."""
    if market_state.open_24h > 0:
        return (
            market_state.current_price - market_state.open_24h
        ) / market_state.open_24h
    return 0.0


def _create_hold_response(reason: str) -> LLMResponse:
    """Create a HOLD response with given reason."""
    return LLMResponse(
        signal=Signal.HOLD,
        confidence=1.0,
        reasoning=f"Defaulting to HOLD: {reason}",
        risk_assessment={
            "risk_level": "LOW",
            "key_risks": ["Uncertain market conditions"],
            "mitigation": "Wait for clearer signals",
        },
    )


def _get_example_responses() -> str:
    """Get example LLM responses for few-shot learning."""
    return """
Example responses:

1. Strong bullish signal:
{
    "action": "LONG",
    "confidence": 0.85,
    "reasoning": "Strong bullish divergence on VuManchu with increasing volume and positive money flow",
    "risk_assessment": {
        "risk_level": "LOW",
        "key_risks": ["Potential resistance at $45,000"],
        "mitigation": "Set tight stop loss below support"
    },
    "suggested_params": {
        "position_size": 0.3,
        "stop_loss": 43500,
        "take_profit": 46500
    }
}

2. Neutral market:
{
    "action": "HOLD",
    "confidence": 0.9,
    "reasoning": "Mixed signals with low volume, waiting for clearer direction",
    "risk_assessment": {
        "risk_level": "MEDIUM",
        "key_risks": ["Choppy market conditions", "Low liquidity"],
        "mitigation": "Stay out until trend emerges"
    }
}
"""
