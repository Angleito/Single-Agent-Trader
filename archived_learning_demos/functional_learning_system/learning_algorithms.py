"""
Pure functional learning algorithms for trading bot.

This module provides purely functional algorithms for pattern analysis,
strategy optimization, and learning insights generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bot.fp.types.result import Err, Ok, Result

from .experience import ExperienceState, TradeExperience


@dataclass(frozen=True)
class LearningResult:
    """Immutable result from learning analysis."""

    insights: tuple[str, ...]
    confidence: float
    sample_size: int
    analysis_timestamp: datetime

    @classmethod
    def empty(cls) -> LearningResult:
        """Create empty learning result."""
        return cls(
            insights=(),
            confidence=0.0,
            sample_size=0,
            analysis_timestamp=datetime.now(UTC),
        )


@dataclass(frozen=True)
class PatternAnalysis:
    """Immutable analysis of trading patterns."""

    pattern_name: str
    occurrence_count: int
    success_rate: float
    avg_pnl: float
    confidence_score: float
    market_conditions: tuple[str, ...]
    correlations: dict[str, float]

    def is_reliable(self, min_samples: int = 5, min_confidence: float = 0.6) -> bool:
        """Check if pattern analysis is reliable."""
        return (
            self.occurrence_count >= min_samples
            and self.confidence_score >= min_confidence
        )


@dataclass(frozen=True)
class StrategyInsight:
    """Immutable strategy optimization insight."""

    insight_type: str  # "parameter", "timing", "market_condition", "risk"
    description: str
    recommended_action: str
    expected_improvement: float
    confidence: float
    supporting_evidence: tuple[str, ...]  # Experience IDs

    def is_actionable(self, min_confidence: float = 0.7) -> bool:
        """Check if insight is actionable."""
        return self.confidence >= min_confidence


@dataclass(frozen=True)
class MarketRegime:
    """Immutable representation of market regime."""

    regime_name: str
    indicators_range: dict[str, tuple[float, float]]
    dominance_range: dict[str, tuple[float, float]] | None
    performance_stats: dict[str, float]

    def matches_current_market(
        self,
        indicators: dict[str, float],
        dominance: dict[str, float] | None = None,
    ) -> bool:
        """Check if current market matches this regime."""
        # Check indicator ranges
        for indicator, (min_val, max_val) in self.indicators_range.items():
            current_val = indicators.get(indicator, 0.0)
            if not (min_val <= current_val <= max_val):
                return False

        # Check dominance ranges if available
        if self.dominance_range and dominance:
            for dom_indicator, (min_val, max_val) in self.dominance_range.items():
                current_val = dominance.get(dom_indicator, 0.0)
                if not (min_val <= current_val <= max_val):
                    return False

        return True


# Pure functional learning algorithms


def analyze_trading_patterns(
    state: ExperienceState,
    min_samples: int = 5,
) -> dict[str, PatternAnalysis]:
    """Analyze all trading patterns in the experience state."""
    pattern_analyses = {}

    for pattern_name, pattern_state in state.patterns.items():
        if pattern_state.occurrence_count >= min_samples:
            # Calculate correlations with other patterns
            correlations = _calculate_pattern_correlations(
                state, pattern_name, min_samples
            )

            # Identify market conditions for this pattern
            market_conditions = _identify_pattern_market_conditions(state, pattern_name)

            analysis = PatternAnalysis(
                pattern_name=pattern_name,
                occurrence_count=pattern_state.occurrence_count,
                success_rate=pattern_state.win_rate,
                avg_pnl=pattern_state.avg_pnl,
                confidence_score=pattern_state.confidence_score,
                market_conditions=tuple(market_conditions),
                correlations=correlations,
            )

            pattern_analyses[pattern_name] = analysis

    return pattern_analyses


def generate_strategy_insights(
    state: ExperienceState,
    recent_hours: int = 24,
) -> tuple[StrategyInsight, ...]:
    """Generate strategy optimization insights from recent experiences."""
    insights = []

    # Filter recent completed experiences
    cutoff_time = datetime.now(UTC) - timedelta(hours=recent_hours)
    recent_experiences = tuple(
        exp
        for exp in state.experiences
        if exp.is_completed() and exp.timestamp >= cutoff_time
    )

    if len(recent_experiences) < 3:
        return ()

    # Parameter optimization insights
    param_insights = _analyze_parameter_performance(recent_experiences)
    insights.extend(param_insights)

    # Timing pattern insights
    timing_insights = _analyze_timing_patterns(recent_experiences)
    insights.extend(timing_insights)

    # Risk management insights
    risk_insights = _analyze_risk_patterns(recent_experiences)
    insights.extend(risk_insights)

    # Market condition insights
    market_insights = _analyze_market_condition_performance(recent_experiences)
    insights.extend(market_insights)

    return tuple(insights)


def calculate_pattern_confidence(
    sample_size: int,
    win_rate: float,
    time_decay_factor: float = 0.95,
) -> float:
    """Calculate confidence score for a pattern."""
    # Base confidence from win rate
    base_confidence = win_rate

    # Sample size factor (sigmoid curve)
    sample_factor = 1 / (1 + math.exp(-(sample_size - 10) / 5))

    # Combine factors
    confidence = base_confidence * sample_factor * time_decay_factor

    return max(0.1, min(0.95, confidence))


def optimize_parameters(
    state: ExperienceState,
    parameter_name: str,
    parameter_values: list[float],
) -> Result[tuple[float, float], str]:
    """
    Optimize a parameter based on historical performance.

    Returns: (optimal_value, expected_improvement)
    """
    if len(state.experiences) < 10:
        return Err("Insufficient data for parameter optimization")

    # Group experiences by parameter value ranges
    value_performance = {}

    for experience in state.experiences:
        if not experience.is_completed():
            continue

        param_value = experience.decision.get(parameter_name)
        if param_value is None:
            continue

        # Find closest parameter value from provided values
        closest_value = min(parameter_values, key=lambda x: abs(x - param_value))

        if closest_value not in value_performance:
            value_performance[closest_value] = []

        value_performance[closest_value].append(experience)

    # Calculate performance metrics for each value
    performance_scores = {}

    for value, experiences in value_performance.items():
        if len(experiences) < 3:  # Need minimum samples
            continue

        # Calculate weighted score: win_rate * 0.6 + avg_pnl_ratio * 0.4
        wins = sum(1 for exp in experiences if exp.is_successful())
        win_rate = wins / len(experiences)

        avg_pnl = sum(exp.get_pnl() or 0.0 for exp in experiences) / len(experiences)
        pnl_ratio = min(avg_pnl / 100, 1.0)  # Normalize to 0-1

        score = win_rate * 0.6 + pnl_ratio * 0.4
        performance_scores[value] = score

    if not performance_scores:
        return Err(f"No usable data for parameter {parameter_name}")

    # Find optimal value
    optimal_value = max(performance_scores.keys(), key=lambda x: performance_scores[x])
    optimal_score = performance_scores[optimal_value]

    # Calculate current performance for comparison
    current_experiences = [
        exp
        for exp in state.experiences[-20:]  # Last 20 experiences
        if exp.is_completed()
    ]

    if current_experiences:
        current_wins = sum(1 for exp in current_experiences if exp.is_successful())
        current_win_rate = current_wins / len(current_experiences)
        current_avg_pnl = sum(
            exp.get_pnl() or 0.0 for exp in current_experiences
        ) / len(current_experiences)
        current_pnl_ratio = min(current_avg_pnl / 100, 1.0)
        current_score = current_win_rate * 0.6 + current_pnl_ratio * 0.4

        expected_improvement = optimal_score - current_score
    else:
        expected_improvement = 0.1  # Default improvement estimate

    return Ok((optimal_value, expected_improvement))


def identify_market_regimes(
    state: ExperienceState,
    min_samples_per_regime: int = 10,
) -> tuple[MarketRegime, ...]:
    """Identify distinct market regimes from trading history."""
    completed_experiences = tuple(
        exp for exp in state.experiences if exp.is_completed()
    )

    if len(completed_experiences) < min_samples_per_regime * 2:
        return ()

    # Simple clustering based on indicator ranges
    regimes = []

    # Define regime by RSI and trend conditions
    rsi_ranges = [(0, 30), (30, 70), (70, 100)]
    trend_conditions = ["uptrend", "downtrend", "sideways"]

    for rsi_min, rsi_max in rsi_ranges:
        for trend in trend_conditions:
            regime_experiences = []

            for exp in completed_experiences:
                rsi = exp.indicators.get("rsi", 50.0)
                has_trend = trend in exp.pattern_tags

                if rsi_min <= rsi <= rsi_max and has_trend:
                    regime_experiences.append(exp)

            if len(regime_experiences) >= min_samples_per_regime:
                regime_name = f"{trend}_rsi_{rsi_min}_{rsi_max}"

                # Calculate performance stats
                wins = sum(1 for exp in regime_experiences if exp.is_successful())
                win_rate = wins / len(regime_experiences)
                avg_pnl = sum(exp.get_pnl() or 0.0 for exp in regime_experiences) / len(
                    regime_experiences
                )

                regime = MarketRegime(
                    regime_name=regime_name,
                    indicators_range={"rsi": (rsi_min, rsi_max)},
                    dominance_range=None,
                    performance_stats={
                        "win_rate": win_rate,
                        "avg_pnl": avg_pnl,
                        "sample_size": len(regime_experiences),
                    },
                )

                regimes.append(regime)

    return tuple(regimes)


# Helper functions for analysis


def _calculate_pattern_correlations(
    state: ExperienceState,
    pattern_name: str,
    min_samples: int,
) -> dict[str, float]:
    """Calculate correlations between patterns."""
    correlations = {}

    # Get experiences with this pattern
    pattern_experiences = state.filter_experiences(
        lambda exp: pattern_name in exp.pattern_tags and exp.is_completed()
    )

    if len(pattern_experiences) < min_samples:
        return correlations

    pattern_win_rate = sum(
        1 for exp in pattern_experiences if exp.is_successful()
    ) / len(pattern_experiences)

    # Check correlation with other patterns
    for other_pattern in state.patterns.keys():
        if other_pattern == pattern_name:
            continue

        # Get experiences with both patterns
        combined_experiences = state.filter_experiences(
            lambda exp: (
                pattern_name in exp.pattern_tags
                and other_pattern in exp.pattern_tags
                and exp.is_completed()
            )
        )

        if len(combined_experiences) >= 3:
            combined_win_rate = sum(
                1 for exp in combined_experiences if exp.is_successful()
            ) / len(combined_experiences)
            correlation = combined_win_rate - pattern_win_rate
            correlations[other_pattern] = correlation

    return correlations


def _identify_pattern_market_conditions(
    state: ExperienceState,
    pattern_name: str,
) -> list[str]:
    """Identify common market conditions for a pattern."""
    pattern_experiences = state.filter_experiences(
        lambda exp: pattern_name in exp.pattern_tags
    )

    if not pattern_experiences:
        return []

    # Count condition occurrences
    condition_counts = {}
    total_experiences = len(pattern_experiences)

    for exp in pattern_experiences:
        for tag in exp.pattern_tags:
            if tag != pattern_name:
                condition_counts[tag] = condition_counts.get(tag, 0) + 1

    # Return conditions that appear in >50% of experiences
    common_conditions = [
        condition
        for condition, count in condition_counts.items()
        if count / total_experiences > 0.5
    ]

    return common_conditions


def _analyze_parameter_performance(
    experiences: tuple[TradeExperience, ...],
) -> list[StrategyInsight]:
    """Analyze parameter performance for insights."""
    insights = []

    # Analyze position sizing
    size_buckets = {"small": (0, 10), "medium": (10, 20), "large": (20, 100)}
    size_performance = {}

    for bucket_name, (min_size, max_size) in size_buckets.items():
        bucket_experiences = [
            exp
            for exp in experiences
            if min_size <= exp.decision.get("size_pct", 0) < max_size
        ]

        if len(bucket_experiences) >= 3:
            wins = sum(1 for exp in bucket_experiences if exp.is_successful())
            win_rate = wins / len(bucket_experiences)
            size_performance[bucket_name] = win_rate

    if size_performance:
        best_size = max(size_performance.keys(), key=lambda x: size_performance[x])
        best_win_rate = size_performance[best_size]

        if best_win_rate > 0.65:
            insights.append(
                StrategyInsight(
                    insight_type="parameter",
                    description=f"Position sizing: {best_size} positions show {best_win_rate:.1%} win rate",
                    recommended_action=f"Consider adjusting position sizing to {best_size} range",
                    expected_improvement=0.05,
                    confidence=0.7,
                    supporting_evidence=tuple(),  # Would include experience IDs
                )
            )

    return insights


def _analyze_timing_patterns(
    experiences: tuple[TradeExperience, ...],
) -> list[StrategyInsight]:
    """Analyze timing patterns for insights."""
    insights = []

    # Analyze hourly performance
    hourly_performance = {}

    for exp in experiences:
        hour = exp.timestamp.hour
        if hour not in hourly_performance:
            hourly_performance[hour] = []
        hourly_performance[hour].append(exp)

    # Find best performing hours
    best_hours = []
    for hour, hour_experiences in hourly_performance.items():
        if len(hour_experiences) >= 3:
            wins = sum(1 for exp in hour_experiences if exp.is_successful())
            win_rate = wins / len(hour_experiences)
            if win_rate > 0.7:
                best_hours.append((hour, win_rate))

    if best_hours:
        best_hours.sort(key=lambda x: x[1], reverse=True)
        top_hours = [f"{hour}:00 UTC" for hour, _ in best_hours[:3]]

        insights.append(
            StrategyInsight(
                insight_type="timing",
                description=f"Best trading hours: {', '.join(top_hours)}",
                recommended_action="Focus trading activity during optimal hours",
                expected_improvement=0.08,
                confidence=0.6,
                supporting_evidence=tuple(),
            )
        )

    return insights


def _analyze_risk_patterns(
    experiences: tuple[TradeExperience, ...],
) -> list[StrategyInsight]:
    """Analyze risk management patterns for insights."""
    insights = []

    losing_trades = [exp for exp in experiences if not exp.is_successful()]

    if len(losing_trades) >= 3:
        # Analyze common factors in losses
        high_leverage_losses = sum(
            1 for exp in losing_trades if exp.decision.get("leverage", 1) > 10
        )

        large_position_losses = sum(
            1 for exp in losing_trades if exp.decision.get("size_pct", 0) > 20
        )

        if high_leverage_losses / len(losing_trades) > 0.4:
            insights.append(
                StrategyInsight(
                    insight_type="risk",
                    description=f"High leverage present in {high_leverage_losses}/{len(losing_trades)} losing trades",
                    recommended_action="Consider reducing maximum leverage",
                    expected_improvement=0.06,
                    confidence=0.75,
                    supporting_evidence=tuple(),
                )
            )

    return insights


def _analyze_market_condition_performance(
    experiences: tuple[TradeExperience, ...],
) -> list[StrategyInsight]:
    """Analyze performance under different market conditions."""
    insights = []

    # Group by dominance conditions
    high_dominance_trades = [
        exp
        for exp in experiences
        if exp.dominance_data and exp.dominance_data.get("stablecoin_dominance", 0) > 10
    ]

    if len(high_dominance_trades) >= 3:
        wins = sum(1 for exp in high_dominance_trades if exp.is_successful())
        win_rate = wins / len(high_dominance_trades)

        if win_rate < 0.4:  # Poor performance in high dominance
            insights.append(
                StrategyInsight(
                    insight_type="market_condition",
                    description=f"Poor performance during high stablecoin dominance: {win_rate:.1%} win rate",
                    recommended_action="Reduce trading activity or position sizes during high dominance periods",
                    expected_improvement=0.07,
                    confidence=0.65,
                    supporting_evidence=tuple(),
                )
            )

    return insights
