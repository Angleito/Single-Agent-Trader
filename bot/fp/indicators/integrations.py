"""
Functional integration layer for combining multiple technical indicators.

This module provides integration functions to combine VuManChu signals with
other technical indicators like RSI, MACD, Bollinger Bands, etc., while
maintaining functional purity and type safety.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from bot.fp.types.indicators import (
    BollingerBandsResult,
    CompositeSignal,
    MACDResult,
    RSIResult,
    VuManchuSignalSet,
)

from .momentum import calculate_all_momentum_indicators
from .moving_averages import sma
from .oscillators import calculate_rsi, create_rsi_result
from .vumanchu_functional import vumanchu_comprehensive_analysis


def create_bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
    timestamp: datetime | None = None,
) -> BollingerBandsResult:
    """Create Bollinger Bands from price data."""
    if len(prices) < period:
        # Return fallback values
        latest_price = prices[-1] if len(prices) > 0 else 0.0
        return BollingerBandsResult(
            timestamp=timestamp or datetime.now(),
            upper=latest_price * 1.02,
            middle=latest_price,
            lower=latest_price * 0.98,
            width=latest_price * 0.04,
        )

    # Calculate simple moving average
    sma_result = sma(prices, period, timestamp or datetime.now())
    middle = sma_result.value

    # Calculate standard deviation
    recent_prices = prices[-period:]
    std = float(np.std(recent_prices))

    # Calculate bands
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    width = upper - lower

    return BollingerBandsResult(
        timestamp=timestamp or datetime.now(),
        upper=upper,
        middle=middle,
        lower=lower,
        width=width,
    )


def integrate_vumanchu_with_indicators(
    ohlcv: np.ndarray,
    vumanchu_config: dict[str, Any] | None = None,
    indicator_config: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> VuManchuSignalSet:
    """
    Integrate VuManChu analysis with additional technical indicators.

    Args:
        ohlcv: OHLCV data array [open, high, low, close, volume]
        vumanchu_config: VuManChu configuration parameters
        indicator_config: Configuration for additional indicators
        timestamp: Timestamp for analysis

    Returns:
        Enhanced VuManchuSignalSet with additional indicators
    """
    timestamp = timestamp or datetime.now()
    vumanchu_config = vumanchu_config or {}
    indicator_config = indicator_config or {}

    # Extract price data
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]
    ohlcv[:, 4] if ohlcv.shape[1] > 4 else np.ones_like(close)

    # Get VuManChu analysis
    vumanchu_signal_set = vumanchu_comprehensive_analysis(
        ohlcv, timestamp=timestamp, **vumanchu_config
    )

    # Calculate additional indicators
    rsi_result = None
    if len(close) >= indicator_config.get("rsi_period", 14):
        rsi_value = calculate_rsi(
            close.tolist(), indicator_config.get("rsi_period", 14)
        )
        rsi_result = create_rsi_result(
            rsi_value,
            indicator_config.get("rsi_overbought", 70.0),
            indicator_config.get("rsi_oversold", 30.0),
            timestamp,
        )

    # Calculate MACD
    macd_result = None
    momentum_indicators = calculate_all_momentum_indicators(
        close,
        high,
        low,
        timestamp,
        rsi_period=indicator_config.get("rsi_period", 14),
        macd_fast=indicator_config.get("macd_fast", 12),
        macd_slow=indicator_config.get("macd_slow", 26),
        macd_signal=indicator_config.get("macd_signal", 9),
    )

    if momentum_indicators.get("macd"):
        macd_result = momentum_indicators["macd"]

    # Calculate Bollinger Bands
    bollinger_result = create_bollinger_bands(
        close,
        indicator_config.get("bb_period", 20),
        indicator_config.get("bb_std_dev", 2.0),
        timestamp,
    )

    # Create enhanced signal set with additional indicators
    enhanced_signal_set = VuManchuSignalSet(
        timestamp=timestamp,
        vumanchu_result=vumanchu_signal_set.vumanchu_result,
        diamond_patterns=vumanchu_signal_set.diamond_patterns,
        yellow_cross_signals=vumanchu_signal_set.yellow_cross_signals,
        candle_patterns=vumanchu_signal_set.candle_patterns,
        divergence_patterns=vumanchu_signal_set.divergence_patterns,
        rsi_result=rsi_result,
        macd_result=macd_result,
        bollinger_result=bollinger_result,
        composite_signal=vumanchu_signal_set.composite_signal,
    )

    # Update composite signal to include additional indicators
    return _enhance_composite_signal(enhanced_signal_set, timestamp)


def _enhance_composite_signal(
    signal_set: VuManchuSignalSet, timestamp: datetime
) -> VuManchuSignalSet:
    """Enhance composite signal with additional indicator inputs."""
    components: dict[str, Any] = {}

    # Add VuManChu components
    components["vumanchu"] = signal_set.vumanchu_result

    # Add diamond patterns
    for i, pattern in enumerate(signal_set.diamond_patterns):
        components[f"diamond_{i}"] = pattern

    # Add yellow cross signals
    for i, signal in enumerate(signal_set.yellow_cross_signals):
        components[f"yellow_{i}"] = signal

    # Add additional indicators
    if signal_set.rsi_result:
        components["rsi"] = signal_set.rsi_result

    if signal_set.macd_result:
        components["macd"] = signal_set.macd_result

    if signal_set.bollinger_result:
        components["bollinger"] = signal_set.bollinger_result

    # Calculate enhanced composite signal
    enhanced_composite = _calculate_enhanced_composite_signal(components, timestamp)

    # Return updated signal set
    return VuManchuSignalSet(
        timestamp=signal_set.timestamp,
        vumanchu_result=signal_set.vumanchu_result,
        diamond_patterns=signal_set.diamond_patterns,
        yellow_cross_signals=signal_set.yellow_cross_signals,
        candle_patterns=signal_set.candle_patterns,
        divergence_patterns=signal_set.divergence_patterns,
        rsi_result=signal_set.rsi_result,
        macd_result=signal_set.macd_result,
        bollinger_result=signal_set.bollinger_result,
        composite_signal=enhanced_composite,
    )


def _calculate_enhanced_composite_signal(
    components: dict[str, Any], timestamp: datetime
) -> CompositeSignal:
    """Calculate composite signal with weighted contributions from all indicators."""
    bullish_count = 0
    bearish_count = 0
    total_strength = 0.0
    total_confidence = 0.0
    weighted_strength = 0.0

    # Define weights for different indicator types
    weights = {
        "vumanchu": 3.0,
        "diamond": 2.5,
        "yellow": 3.0,
        "rsi": 1.5,
        "macd": 2.0,
        "bollinger": 1.0,
    }

    total_weight = 0.0

    for name, component in components.items():
        weight = 1.0  # Default weight

        # Determine weight based on component type
        for indicator_type, type_weight in weights.items():
            if indicator_type in name:
                weight = type_weight
                break

        # Check if component is bullish/bearish
        is_bullish = False
        is_bearish = False
        strength = 0.0
        confidence = 0.0

        if hasattr(component, "is_bullish") and component.is_bullish():
            is_bullish = True
            bullish_count += 1
        elif hasattr(component, "is_bearish") and component.is_bearish():
            is_bearish = True
            bearish_count += 1
        elif hasattr(component, "signal"):
            if component.signal == "LONG":
                is_bullish = True
                bullish_count += 1
            elif component.signal == "SHORT":
                is_bearish = True
                bearish_count += 1

        # Handle RSI specific logic
        if name == "rsi" and isinstance(component, RSIResult):
            if component.is_oversold():
                is_bullish = True
                bullish_count += 1
            elif component.is_overbought():
                is_bearish = True
                bearish_count += 1

        # Handle MACD specific logic
        if name == "macd" and isinstance(component, MACDResult):
            if component.is_bullish_crossover():
                is_bullish = True
                bullish_count += 1
            elif component.is_bearish_crossover():
                is_bearish = True
                bearish_count += 1

        # Extract strength and confidence
        if hasattr(component, "strength"):
            strength = component.strength
        elif hasattr(component, "confidence"):
            strength = component.confidence
        elif hasattr(component, "momentum_strength"):
            strength = component.momentum_strength() / 100.0

        if hasattr(component, "confidence"):
            confidence = component.confidence
        elif hasattr(component, "strength"):
            confidence = component.strength

        # Apply weights
        if is_bullish or is_bearish:
            weighted_strength += strength * weight
            total_strength += strength
            total_confidence += confidence
            total_weight += weight

    # Calculate metrics
    component_count = len(components)
    total_strength / max(component_count, 1)
    avg_confidence = total_confidence / max(component_count, 1)
    weighted_avg_strength = weighted_strength / max(total_weight, 1)

    # Determine signal direction
    if bullish_count > bearish_count:
        signal_direction = 1
        dominant = "bullish"
    elif bearish_count > bullish_count:
        signal_direction = -1
        dominant = "bearish"
    else:
        signal_direction = 0
        dominant = "neutral"

    # Calculate agreement score
    total_signals = bullish_count + bearish_count
    agreement_score = max(bullish_count, bearish_count) / max(total_signals, 1)

    # Enhance confidence based on agreement and strength
    enhanced_confidence = (
        avg_confidence + agreement_score + weighted_avg_strength
    ) / 3.0

    return CompositeSignal(
        timestamp=timestamp,
        signal_direction=signal_direction,
        components=components,
        confidence=enhanced_confidence,
        strength=weighted_avg_strength,
        agreement_score=agreement_score,
        dominant_component=dominant,
    )


def filter_high_quality_signals(signal_set: VuManchuSignalSet) -> VuManchuSignalSet:
    """Filter signal set to include only high-quality signals."""

    # Filter diamond patterns by strength
    filtered_diamonds = [
        p
        for p in signal_set.diamond_patterns
        if p.strength >= 0.6 and p.confluence_score() >= 0.7
    ]

    # Filter yellow cross signals by confidence
    filtered_yellow = [
        s
        for s in signal_set.yellow_cross_signals
        if s.confidence >= 0.8 and s.all_conditions_met()
    ]

    # Filter candle patterns by validity and strength
    filtered_candles = [
        p for p in signal_set.candle_patterns if p.is_valid() and p.strength >= 0.7
    ]

    # Filter divergence patterns by strength
    filtered_divergences = [
        d for d in signal_set.divergence_patterns if d.strength >= 0.6
    ]

    # Create filtered signal set
    return VuManchuSignalSet(
        timestamp=signal_set.timestamp,
        vumanchu_result=signal_set.vumanchu_result,
        diamond_patterns=filtered_diamonds,
        yellow_cross_signals=filtered_yellow,
        candle_patterns=filtered_candles,
        divergence_patterns=filtered_divergences,
        rsi_result=signal_set.rsi_result,
        macd_result=signal_set.macd_result,
        bollinger_result=signal_set.bollinger_result,
        composite_signal=signal_set.composite_signal,
    )


def calculate_signal_confluence(signal_set: VuManchuSignalSet) -> dict[str, float]:
    """Calculate confluence scores for different signal combinations."""
    confluence_scores = {
        "vumanchu_rsi": 0.0,
        "vumanchu_macd": 0.0,
        "diamond_bollinger": 0.0,
        "yellow_cross_momentum": 0.0,
        "overall_confluence": signal_set.signal_confluence_score(),
    }

    # VuManChu + RSI confluence
    if signal_set.rsi_result and signal_set.vumanchu_result:
        vumanchu_bullish = signal_set.vumanchu_result.signal == "LONG"
        rsi_bullish = signal_set.rsi_result.is_oversold()

        if (vumanchu_bullish and rsi_bullish) or (
            not vumanchu_bullish and not rsi_bullish
        ):
            confluence_scores["vumanchu_rsi"] = 0.8

    # VuManChu + MACD confluence
    if signal_set.macd_result and signal_set.vumanchu_result:
        vumanchu_bullish = signal_set.vumanchu_result.signal == "LONG"
        macd_bullish = signal_set.macd_result.is_bullish_crossover()

        if (vumanchu_bullish and macd_bullish) or (
            not vumanchu_bullish and not macd_bullish
        ):
            confluence_scores["vumanchu_macd"] = 0.9

    # Diamond + Bollinger confluence
    if signal_set.bollinger_result and signal_set.diamond_patterns:
        # Check if any diamond patterns align with Bollinger extremes
        current_price = signal_set.vumanchu_result.wave_a  # Proxy for current price
        in_bollinger_extremes = (
            current_price <= signal_set.bollinger_result.lower
            or current_price >= signal_set.bollinger_result.upper
        )

        if in_bollinger_extremes and signal_set.diamond_patterns:
            confluence_scores["diamond_bollinger"] = 0.7

    # Yellow cross + momentum confluence
    if signal_set.yellow_cross_signals and (
        signal_set.rsi_result or signal_set.macd_result
    ):
        yellow_signals = len(signal_set.yellow_cross_signals)
        momentum_signals = 0

        if signal_set.rsi_result and (
            signal_set.rsi_result.is_overbought() or signal_set.rsi_result.is_oversold()
        ):
            momentum_signals += 1

        if signal_set.macd_result and (
            signal_set.macd_result.is_bullish_crossover()
            or signal_set.macd_result.is_bearish_crossover()
        ):
            momentum_signals += 1

        if yellow_signals > 0 and momentum_signals > 0:
            confluence_scores["yellow_cross_momentum"] = min(
                0.9, (yellow_signals + momentum_signals) / 4.0
            )

    return confluence_scores


def create_indicator_summary(signal_set: VuManchuSignalSet) -> dict[str, Any]:
    """Create a comprehensive summary of all indicators and signals."""
    summary = {
        "timestamp": signal_set.timestamp.isoformat(),
        "overall_direction": signal_set.overall_direction(),
        "signal_confluence": signal_set.signal_confluence_score(),
        "active_patterns": signal_set.get_active_patterns(),
        "bullish_signals": len(signal_set.get_bullish_signals()),
        "bearish_signals": len(signal_set.get_bearish_signals()),
    }

    # VuManChu summary
    summary["vumanchu"] = {
        "signal": signal_set.vumanchu_result.signal,
        "wave_a": signal_set.vumanchu_result.wave_a,
        "wave_b": signal_set.vumanchu_result.wave_b,
        "momentum_strength": signal_set.vumanchu_result.momentum_strength(),
    }

    # Additional indicators summary
    if signal_set.rsi_result:
        summary["rsi"] = {
            "value": signal_set.rsi_result.value,
            "strength_level": signal_set.rsi_result.strength_level(),
            "is_overbought": signal_set.rsi_result.is_overbought(),
            "is_oversold": signal_set.rsi_result.is_oversold(),
        }

    if signal_set.macd_result:
        summary["macd"] = {
            "macd": signal_set.macd_result.macd,
            "signal": signal_set.macd_result.signal,
            "histogram": signal_set.macd_result.histogram,
            "momentum_direction": signal_set.macd_result.momentum_direction(),
        }

    if signal_set.bollinger_result:
        summary["bollinger"] = {
            "upper": signal_set.bollinger_result.upper,
            "middle": signal_set.bollinger_result.middle,
            "lower": signal_set.bollinger_result.lower,
            "is_squeeze": signal_set.bollinger_result.is_squeeze(),
        }

    # Composite signal summary
    if signal_set.composite_signal:
        summary["composite"] = {
            "direction": signal_set.composite_signal.signal_direction,
            "confidence": signal_set.composite_signal.confidence,
            "strength": signal_set.composite_signal.strength,
            "agreement_score": signal_set.composite_signal.agreement_score,
            "is_high_quality": signal_set.composite_signal.is_high_quality(),
        }

    # Confluence analysis
    summary["confluence"] = calculate_signal_confluence(signal_set)

    return summary


def create_trading_recommendation(signal_set: VuManchuSignalSet) -> dict[str, Any]:
    """Create actionable trading recommendation from signal analysis."""
    recommendation = {
        "action": "HOLD",  # Default to neutral
        "confidence": 0.0,
        "strength": 0.0,
        "risk_level": "MEDIUM",
        "reasons": [],
        "stop_loss_suggestion": None,
        "take_profit_suggestion": None,
    }

    if not signal_set.composite_signal:
        recommendation["reasons"].append("No composite signal available")
        return recommendation

    composite = signal_set.composite_signal
    confluence_scores = calculate_signal_confluence(signal_set)

    # Determine action based on composite signal
    if composite.signal_direction > 0:
        recommendation["action"] = "LONG"
    elif composite.signal_direction < 0:
        recommendation["action"] = "SHORT"

    # Set confidence and strength
    recommendation["confidence"] = composite.confidence
    recommendation["strength"] = composite.strength

    # Determine risk level
    if composite.is_high_quality() and confluence_scores["overall_confluence"] >= 0.8:
        recommendation["risk_level"] = "LOW"
    elif composite.confidence >= 0.6 and composite.agreement_score >= 0.7:
        recommendation["risk_level"] = "MEDIUM"
    else:
        recommendation["risk_level"] = "HIGH"

    # Add reasoning
    if composite.is_high_quality():
        recommendation["reasons"].append("High-quality signal with strong confluence")

    if confluence_scores["vumanchu_rsi"] >= 0.8:
        recommendation["reasons"].append("VuManChu and RSI signals align")

    if confluence_scores["vumanchu_macd"] >= 0.8:
        recommendation["reasons"].append("VuManChu and MACD signals align")

    if len(signal_set.diamond_patterns) > 0:
        recommendation["reasons"].append(
            f"{len(signal_set.diamond_patterns)} diamond pattern(s) detected"
        )

    if len(signal_set.yellow_cross_signals) > 0:
        recommendation["reasons"].append(
            f"{len(signal_set.yellow_cross_signals)} yellow cross signal(s) detected"
        )

    # Add risk management suggestions
    if recommendation["action"] != "HOLD":
        current_value = (
            signal_set.vumanchu_result.wave_a
        )  # Use as proxy for price level

        if recommendation["action"] == "LONG":
            recommendation["stop_loss_suggestion"] = (
                current_value * 0.98
            )  # 2% stop loss
            recommendation["take_profit_suggestion"] = (
                current_value * 1.04
            )  # 4% take profit
        else:  # SHORT
            recommendation["stop_loss_suggestion"] = (
                current_value * 1.02
            )  # 2% stop loss
            recommendation["take_profit_suggestion"] = (
                current_value * 0.96
            )  # 4% take profit

    return recommendation


def analyze_signal_quality(signal_set: VuManchuSignalSet) -> dict[str, Any]:
    """Analyze the quality of signals in the signal set."""
    quality_analysis = {
        "overall_quality": "POOR",
        "quality_score": 0.0,
        "strengths": [],
        "weaknesses": [],
        "improvement_suggestions": [],
    }

    quality_factors = []

    # Analyze VuManChu signal quality
    vumanchu_quality = 0.5  # Neutral baseline
    if signal_set.vumanchu_result.signal != "NEUTRAL":
        vumanchu_quality += 0.2
        quality_analysis["strengths"].append("Clear VuManChu directional signal")

    momentum_strength = signal_set.vumanchu_result.momentum_strength()
    if momentum_strength >= 70:
        vumanchu_quality += 0.3
        quality_analysis["strengths"].append("Strong momentum detected")
    elif momentum_strength < 30:
        quality_analysis["weaknesses"].append("Weak momentum signal")

    quality_factors.append(vumanchu_quality)

    # Analyze pattern quality
    pattern_quality = 0.5
    if signal_set.diamond_patterns:
        high_strength_diamonds = [
            p for p in signal_set.diamond_patterns if p.strength >= 0.7
        ]
        if high_strength_diamonds:
            pattern_quality += 0.3
            quality_analysis["strengths"].append(
                f"{len(high_strength_diamonds)} high-strength diamond patterns"
            )

    if signal_set.yellow_cross_signals:
        high_conf_yellows = [
            s for s in signal_set.yellow_cross_signals if s.confidence >= 0.8
        ]
        if high_conf_yellows:
            pattern_quality += 0.2
            quality_analysis["strengths"].append(
                f"{len(high_conf_yellows)} high-confidence yellow cross signals"
            )

    quality_factors.append(pattern_quality)

    # Analyze indicator confluence
    confluence_quality = 0.5
    confluence_scores = calculate_signal_confluence(signal_set)

    if confluence_scores["overall_confluence"] >= 0.8:
        confluence_quality += 0.4
        quality_analysis["strengths"].append("Strong signal confluence")
    elif confluence_scores["overall_confluence"] < 0.5:
        quality_analysis["weaknesses"].append("Poor signal confluence")

    quality_factors.append(confluence_quality)

    # Calculate overall quality score
    quality_analysis["quality_score"] = sum(quality_factors) / len(quality_factors)

    # Determine overall quality rating
    if quality_analysis["quality_score"] >= 0.8:
        quality_analysis["overall_quality"] = "EXCELLENT"
    elif quality_analysis["quality_score"] >= 0.7:
        quality_analysis["overall_quality"] = "GOOD"
    elif quality_analysis["quality_score"] >= 0.6:
        quality_analysis["overall_quality"] = "FAIR"
    elif quality_analysis["quality_score"] >= 0.4:
        quality_analysis["overall_quality"] = "POOR"
    else:
        quality_analysis["overall_quality"] = "VERY_POOR"

    # Add improvement suggestions
    if not signal_set.diamond_patterns:
        quality_analysis["improvement_suggestions"].append(
            "Wait for diamond pattern confirmation"
        )

    if confluence_scores["overall_confluence"] < 0.6:
        quality_analysis["improvement_suggestions"].append(
            "Wait for better signal confluence"
        )

    if not signal_set.composite_signal or signal_set.composite_signal.confidence < 0.7:
        quality_analysis["improvement_suggestions"].append(
            "Wait for higher confidence signals"
        )

    return quality_analysis


def create_market_context_analysis(signal_set: VuManchuSignalSet) -> dict[str, Any]:
    """Create market context analysis from signal set."""
    context = {
        "market_phase": "UNKNOWN",
        "volatility_regime": "MEDIUM",
        "trend_strength": 0.0,
        "momentum_direction": "NEUTRAL",
        "key_levels": {},
        "market_sentiment": "NEUTRAL",
    }

    # Analyze market phase using Bollinger Bands
    if signal_set.bollinger_result:
        bb = signal_set.bollinger_result
        current_price = signal_set.vumanchu_result.wave_a  # Proxy for price

        if bb.is_squeeze():
            context["market_phase"] = "CONSOLIDATION"
            context["volatility_regime"] = "LOW"
        elif bb.is_price_above_upper(current_price):
            context["market_phase"] = "BREAKOUT_UP"
            context["volatility_regime"] = "HIGH"
        elif bb.is_price_below_lower(current_price):
            context["market_phase"] = "BREAKOUT_DOWN"
            context["volatility_regime"] = "HIGH"
        else:
            context["market_phase"] = "RANGING"

        context["key_levels"] = {
            "resistance": bb.upper,
            "support": bb.lower,
            "pivot": bb.middle,
        }

    # Analyze trend strength using MACD
    if signal_set.macd_result:
        macd = signal_set.macd_result
        context["trend_strength"] = min(abs(macd.histogram) / 10.0, 1.0)
        context["momentum_direction"] = macd.momentum_direction()

    # Determine market sentiment
    bullish_signals = len(signal_set.get_bullish_signals())
    bearish_signals = len(signal_set.get_bearish_signals())

    if bullish_signals > bearish_signals * 1.5:
        context["market_sentiment"] = "BULLISH"
    elif bearish_signals > bullish_signals * 1.5:
        context["market_sentiment"] = "BEARISH"
    else:
        context["market_sentiment"] = "NEUTRAL"

    return context
