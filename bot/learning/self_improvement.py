"""
Self-improvement module for AI trading bot.

Analyzes trading patterns, identifies successful strategies,
and evolves trading parameters based on performance.
"""

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, Field

from bot.config import settings
from bot.mcp.memory_server import MCPMemoryServer, TradingExperience

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PatternPerformance(BaseModel):
    """Performance metrics for a specific pattern."""

    pattern_name: str
    occurrence_count: int = 0
    success_count: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    avg_duration_minutes: float = 0.0
    confidence_score: float = 0.5
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StrategyAdjustment(BaseModel):
    """Suggested strategy adjustment based on learning."""

    parameter: str
    current_value: Any
    suggested_value: Any
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    expected_improvement: float  # Expected improvement in win rate or PnL


class LearningInsight(BaseModel):
    """A learned insight about market behavior."""

    insight_id: str
    insight_type: str  # "pattern", "correlation", "timing", "risk"
    description: str
    supporting_evidence: list[str]  # Experience IDs
    confidence: float = Field(ge=0.0, le=1.0)
    actionable: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SelfImprovementEngine:
    """
    Engine for analyzing trading performance and generating improvements.

    Uses historical trading data to identify patterns, correlations,
    and optimal parameters for different market conditions.
    """

    def __init__(self, memory_server: MCPMemoryServer):
        """Initialize the self-improvement engine."""
        self.memory_server = memory_server

        # Pattern performance tracking
        self.pattern_performance: dict[str, PatternPerformance] = {}

        # Learned insights
        self.insights: list[LearningInsight] = []

        # Strategy parameters and their performance
        self.parameter_performance: dict[str, list[tuple[Any, float]]] = defaultdict(
            list
        )

        # Market condition classifiers
        self.market_conditions: dict[str, Callable[[Any], bool]] = {
            "trending_up": lambda ind: ind.get("ema_fast", 0) > ind.get("ema_slow", 0),
            "trending_down": lambda ind: ind.get("ema_fast", 0)
            < ind.get("ema_slow", 0),
            "high_volatility": lambda ind: abs(ind.get("cipher_b_wave", 0)) > 60,
            "oversold": lambda ind: ind.get("rsi", 50) < 30,
            "overbought": lambda ind: ind.get("rsi", 50) > 70,
            "high_dominance": lambda dom: dom
            and dom.get("stablecoin_dominance", 0) > 10,
            "rising_dominance": lambda dom: dom
            and dom.get("dominance_24h_change", 0) > 1,
        }

        logger.info("Initialized self-improvement engine")

    async def analyze_recent_performance(self, hours: int = 24) -> dict[str, Any]:
        """
        Analyze recent trading performance and generate insights.

        Args:
            hours: Number of hours to analyze

        Returns:
            Performance analysis with insights and recommendations
        """
        # Get recent experiences from memory
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        recent_experiences = [
            exp
            for exp in self.memory_server.memory_cache.values()
            if exp.timestamp >= cutoff_time and exp.outcome is not None
        ]

        if not recent_experiences:
            return {
                "status": "insufficient_data",
                "message": "No completed trades in the specified timeframe",
            }

        # Analyze patterns
        pattern_analysis = await self._analyze_patterns(recent_experiences)

        # Analyze market conditions
        condition_analysis = self._analyze_market_conditions(recent_experiences)

        # Generate parameter adjustments
        adjustments = await self._generate_adjustments(recent_experiences)

        # Calculate overall metrics
        total_trades = len(recent_experiences)
        successful_trades = sum(
            1
            for exp in recent_experiences
            if exp.outcome and exp.outcome.get("success", False)
        )
        total_pnl = sum(
            exp.outcome.get("pnl", 0.0) for exp in recent_experiences if exp.outcome
        )

        analysis = {
            "period_hours": hours,
            "total_trades": total_trades,
            "success_rate": successful_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": total_pnl / total_trades if total_trades > 0 else 0,
            "pattern_analysis": pattern_analysis,
            "market_condition_performance": condition_analysis,
            "suggested_adjustments": adjustments,
            "new_insights": await self._generate_insights(recent_experiences),
        }

        logger.info("Performance analysis complete: %s trades, " "%s success rate, " "$%s total PnL" ), total_trades, analysis['success_rate']:.1%, total_pnl:.2f)

        return analysis

    async def _analyze_patterns(
        self, experiences: list[TradingExperience]
    ) -> dict[str, dict[str, Any]]:
        """Analyze pattern performance."""
        pattern_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "successes": 0, "total_pnl": 0.0, "durations": []}
        )

        for exp in experiences:
            if exp.outcome is None:
                continue
            for pattern in exp.pattern_tags:
                stats = pattern_stats[pattern]
                stats["count"] += 1

                if exp.outcome.get("success", False):
                    stats["successes"] += 1

                stats["total_pnl"] += exp.outcome.get("pnl", 0.0)

                if exp.trade_duration_minutes and isinstance(stats["durations"], list):
                    stats["durations"].append(exp.trade_duration_minutes)

        # Calculate aggregated metrics
        pattern_performance = {}
        for pattern, stats in pattern_stats.items():
            count = stats["count"]
            successes = stats["successes"]
            total_pnl = stats["total_pnl"]
            durations = stats["durations"]

            if count >= settings.mcp.min_samples_for_pattern:
                perf = PatternPerformance(
                    pattern_name=pattern,
                    occurrence_count=count,
                    success_count=successes,
                    total_pnl=total_pnl,
                    avg_pnl=total_pnl / count,
                    win_rate=successes / count,
                    avg_duration_minutes=(
                        float(np.mean(durations))
                        if isinstance(durations, list) and durations
                        else 0.0
                    ),
                    confidence_score=self._calculate_pattern_confidence(
                        count, successes / count
                    ),
                )

                pattern_performance[pattern] = perf.dict()

                # Update cached performance
                self.pattern_performance[pattern] = perf

        return pattern_performance

    def _analyze_market_conditions(
        self, experiences: list[TradingExperience]
    ) -> dict[str, dict[str, Any]]:
        """Analyze performance under different market conditions."""
        condition_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "successes": 0, "total_pnl": 0.0}
        )

        for exp in experiences:
            # Identify active conditions
            active_conditions = []

            for condition_name, condition_func in self.market_conditions.items():
                if "dominance" in condition_name:
                    if condition_func(exp.dominance_data):
                        active_conditions.append(condition_name)
                elif condition_func(exp.indicators):
                    active_conditions.append(condition_name)

            # Update stats for each active condition
            for condition in active_conditions:
                if exp.outcome is None:
                    continue
                stats = condition_stats[condition]
                stats["count"] += 1

                if exp.outcome.get("success", False):
                    stats["successes"] += 1

                stats["total_pnl"] += exp.outcome.get("pnl", 0.0)

        # Calculate performance metrics
        condition_performance = {}
        for condition, stats in condition_stats.items():
            count = stats["count"]
            successes = stats["successes"]
            total_pnl = stats["total_pnl"]

            if count > 0:
                condition_performance[condition] = {
                    "count": count,
                    "win_rate": successes / count,
                    "avg_pnl": total_pnl / count,
                    "total_pnl": total_pnl,
                }

        return condition_performance

    async def _generate_adjustments(
        self, experiences: list[TradingExperience]
    ) -> list[dict[str, Any]]:
        """Generate strategy parameter adjustments based on performance."""
        adjustments = []

        # Analyze position sizing performance
        size_performance = self._analyze_position_sizing(experiences)
        if size_performance:
            adjustments.append(size_performance.dict())

        # Analyze leverage usage
        leverage_adjustment = self._analyze_leverage_performance(experiences)
        if leverage_adjustment:
            adjustments.append(leverage_adjustment.dict())

        # Analyze stop loss effectiveness
        stop_loss_adjustment = self._analyze_stop_loss_performance(experiences)
        if stop_loss_adjustment:
            adjustments.append(stop_loss_adjustment.dict())

        # Analyze timing patterns
        timing_adjustment = self._analyze_timing_patterns(experiences)
        if timing_adjustment:
            adjustments.append(timing_adjustment.dict())

        return adjustments

    def _analyze_position_sizing(
        self, experiences: list[TradingExperience]
    ) -> StrategyAdjustment | None:
        """Analyze position sizing effectiveness."""
        # Group by position size ranges
        size_buckets = {"small": (0, 10), "medium": (10, 20), "large": (20, 100)}

        bucket_performance: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "successes": 0, "pnl": 0.0}
        )

        for exp in experiences:
            if exp.outcome is None:
                continue
            size_pct = exp.decision.get("size_pct", 0)

            for bucket_name, (min_size, max_size) in size_buckets.items():
                if min_size <= size_pct < max_size:
                    stats = bucket_performance[bucket_name]
                    stats["count"] += 1
                    if exp.outcome.get("success", False):
                        stats["successes"] += 1
                    stats["pnl"] += exp.outcome.get("pnl", 0.0)
                    break

        # Find best performing bucket
        best_bucket = None
        best_win_rate = 0.0

        for bucket_name, stats in bucket_performance.items():
            count = stats["count"]
            successes = stats["successes"]
            if count >= 3:  # Minimum sample size
                win_rate = successes / count
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_bucket = bucket_name

        # Generate adjustment if significant difference found
        if best_bucket and best_win_rate > 0.6:
            current_max_size = settings.trading.max_size_pct

            if best_bucket == "small" and current_max_size > 15:
                return StrategyAdjustment(
                    parameter="max_size_pct",
                    current_value=current_max_size,
                    suggested_value=10,
                    reason=f"Small positions showing {best_win_rate:.1%} win rate",
                    confidence=0.7,
                    expected_improvement=0.1,
                )
            elif best_bucket == "large" and current_max_size < 25:
                return StrategyAdjustment(
                    parameter="max_size_pct",
                    current_value=current_max_size,
                    suggested_value=25,
                    reason=f"Larger positions showing {best_win_rate:.1%} win rate",
                    confidence=0.6,
                    expected_improvement=0.15,
                )

        return None

    def _analyze_leverage_performance(
        self, experiences: list[TradingExperience]
    ) -> StrategyAdjustment | None:
        """Analyze leverage usage and performance."""
        leverage_stats: dict[int | float, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "successes": 0, "pnl": 0.0}
        )

        for exp in experiences:
            if exp.outcome is None:
                continue
            leverage = exp.decision.get("leverage", 1)
            stats = leverage_stats[leverage]
            stats["count"] += 1
            if exp.outcome.get("success", False):
                stats["successes"] += 1
            stats["pnl"] += exp.outcome.get("pnl", 0.0)

        # Find optimal leverage
        best_leverage = None
        best_score = 0.0  # Combination of win rate and avg PnL

        for leverage, stats in leverage_stats.items():
            count = stats["count"]
            successes = stats["successes"]
            pnl = stats["pnl"]
            if count >= 3:
                win_rate = successes / count
                avg_pnl = pnl / count

                # Score combines win rate and profitability
                score = win_rate * 0.6 + min(avg_pnl / 100, 0.4)

                if score > best_score:
                    best_score = score
                    best_leverage = leverage

        # Suggest adjustment if different from current
        current_leverage = settings.trading.leverage
        if best_leverage and best_leverage != current_leverage and best_score > 0.5:
            return StrategyAdjustment(
                parameter="leverage",
                current_value=current_leverage,
                suggested_value=best_leverage,
                reason=f"Leverage {best_leverage}x showing optimal risk/reward",
                confidence=0.65,
                expected_improvement=0.08,
            )

        return None

    def _analyze_stop_loss_performance(
        self, experiences: list[TradingExperience]
    ) -> StrategyAdjustment | None:
        """Analyze stop loss effectiveness."""
        # Track trades that likely hit stop loss
        stop_loss_hits = 0
        total_losses = 0
        large_losses = 0

        for exp in experiences:
            if exp.outcome is None:
                continue
            if (
                not exp.outcome.get("success", False)
                and exp.outcome.get("pnl", 0.0) < 0
            ):
                total_losses += 1

                # Check if loss exceeded stop loss threshold
                stop_loss_pct = exp.decision.get("stop_loss_pct", 2.0)
                actual_loss_pct = abs(exp.outcome.get("price_change_pct", 0.0))

                if actual_loss_pct >= stop_loss_pct * 0.9:  # Near or beyond stop
                    stop_loss_hits += 1

                if actual_loss_pct > stop_loss_pct * 1.5:  # Significantly beyond
                    large_losses += 1

        # Suggest tighter stops if many large losses
        if total_losses >= 5 and large_losses / total_losses > 0.3:
            current_stop = settings.risk.default_stop_loss_pct
            suggested_stop = max(current_stop * 0.75, 1.0)  # 25% tighter, min 1%

            return StrategyAdjustment(
                parameter="default_stop_loss_pct",
                current_value=current_stop,
                suggested_value=round(suggested_stop, 1),
                reason=f"{large_losses} trades exceeded stop loss significantly",
                confidence=0.75,
                expected_improvement=0.05,
            )

        return None

    def _analyze_timing_patterns(
        self, experiences: list[TradingExperience]
    ) -> StrategyAdjustment | None:
        """Analyze entry and exit timing patterns."""
        # Analyze hold duration vs success
        duration_buckets = {
            "scalp": (0, 60),  # < 1 hour
            "intraday": (60, 360),  # 1-6 hours
            "swing": (360, 1440),  # 6-24 hours
            "position": (1440, float("inf")),  # > 24 hours
        }

        bucket_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"count": 0, "successes": 0}
        )

        for exp in experiences:
            if exp.trade_duration_minutes:
                duration = exp.trade_duration_minutes

                for bucket_name, (min_dur, max_dur) in duration_buckets.items():
                    if min_dur <= duration < max_dur:
                        if exp.outcome is None:
                            continue
                        stats = bucket_stats[bucket_name]
                        stats["count"] += 1
                        if exp.outcome.get("success", False):
                            stats["successes"] += 1
                        break

        # Find optimal holding period
        best_duration = None
        best_win_rate = 0.0

        for bucket_name, stats in bucket_stats.items():
            if stats["count"] >= 3:
                win_rate = stats["successes"] / stats["count"]
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_duration = bucket_name

        # Suggest position hold time adjustment
        if best_duration == "scalp" and best_win_rate > 0.65:
            return StrategyAdjustment(
                parameter="suggested_trading_style",
                current_value="mixed",
                suggested_value="scalping",
                reason=f"Short-term trades showing {best_win_rate:.1%} success",
                confidence=0.6,
                expected_improvement=0.1,
            )

        return None

    async def _generate_insights(
        self, experiences: list[TradingExperience]
    ) -> list[dict[str, Any]]:
        """Generate new learning insights from recent experiences."""
        insights = []

        # Pattern correlation insights
        pattern_correlations = self._find_pattern_correlations(experiences)
        for correlation in pattern_correlations:
            insights.append(correlation.dict())

        # Market regime insights
        regime_insights = self._identify_market_regimes(experiences)
        for insight in regime_insights:
            insights.append(insight.dict())

        # Risk factor insights
        risk_insights = self._analyze_risk_factors(experiences)
        for insight in risk_insights:
            insights.append(insight.dict())

        # Store insights
        self.insights.extend(
            [LearningInsight(**insight_dict) for insight_dict in insights]
        )

        return insights

    def _find_pattern_correlations(
        self, experiences: list[TradingExperience]
    ) -> list[LearningInsight]:
        """Find correlations between patterns and outcomes."""
        insights = []

        # Look for pattern combinations that work well together
        pattern_combos: dict[str, dict[str, int]] = defaultdict(
            lambda: {"count": 0, "successes": 0}
        )

        for exp in experiences:
            if len(exp.pattern_tags) >= 2:
                # Check specific interesting combinations
                patterns = set(exp.pattern_tags)

                # Trend + momentum alignment
                if "uptrend" in patterns and "falling_dominance" in patterns:
                    if exp.outcome is None:
                        continue
                    combo = "uptrend_falling_dominance"
                    stats = pattern_combos[combo]
                    stats["count"] += 1
                    if exp.outcome.get("success", False):
                        stats["successes"] += 1

                # Oversold + dominance
                if "oversold" in patterns and "high_stablecoin_dominance" in patterns:
                    if exp.outcome is None:
                        continue
                    combo = "oversold_high_dominance"
                    stats = pattern_combos[combo]
                    stats["count"] += 1
                    if exp.outcome.get("success", False):
                        stats["successes"] += 1

        # Generate insights for strong correlations
        for combo, stats in pattern_combos.items():
            if stats["count"] >= 3:
                win_rate = stats["successes"] / stats["count"]

                if win_rate > 0.7:
                    insights.append(
                        LearningInsight(
                            insight_id=f"correlation_{combo}_{datetime.now(UTC).timestamp()}",
                            insight_type="correlation",
                            description=f"Pattern combination '{combo}' shows {win_rate:.1%} success rate",
                            supporting_evidence=[],  # Would add experience IDs
                            confidence=min(0.5 + stats["count"] * 0.05, 0.9),
                            actionable=True,
                        )
                    )

        return insights

    def _identify_market_regimes(
        self, experiences: list[TradingExperience]
    ) -> list[LearningInsight]:
        """Identify market regime patterns."""
        insights = []

        # Group experiences by time periods
        hourly_performance: dict[int, dict[str, int]] = defaultdict(
            lambda: {"count": 0, "successes": 0}
        )

        for exp in experiences:
            if exp.outcome is None:
                continue
            hour = exp.timestamp.hour
            stats = hourly_performance[hour]
            stats["count"] += 1
            if exp.outcome.get("success", False):
                stats["successes"] += 1

        # Find best performing hours
        best_hours = []
        for hour, stats in hourly_performance.items():
            if stats["count"] >= 2:
                win_rate = stats["successes"] / stats["count"]
                if win_rate > 0.7:
                    best_hours.append((hour, win_rate))

        if best_hours:
            best_hours.sort(key=lambda x: x[1], reverse=True)
            hours_str = ", ".join([f"{h}:00 UTC" for h, _ in best_hours[:3]])

            insights.append(
                LearningInsight(
                    insight_id=f"timing_pattern_{datetime.now(UTC).timestamp()}",
                    insight_type="timing",
                    description=f"Best trading performance during: {hours_str}",
                    supporting_evidence=[],
                    confidence=0.65,
                    actionable=True,
                )
            )

        return insights

    def _analyze_risk_factors(
        self, experiences: list[TradingExperience]
    ) -> list[LearningInsight]:
        """Analyze risk factors and generate insights."""
        insights = []

        # Analyze losing trades for common factors
        losing_trades = [
            exp
            for exp in experiences
            if exp.outcome and not exp.outcome.get("success", False)
        ]

        if len(losing_trades) >= 3:
            # Check for common patterns in losses
            loss_patterns: dict[str, int] = defaultdict(int)

            for exp in losing_trades:
                # High leverage losses
                if exp.decision.get("leverage", 1) > 10:
                    loss_patterns["high_leverage"] += 1

                # Large position losses
                if exp.decision.get("size_pct", 0) > 20:
                    loss_patterns["large_position"] += 1

                # Counter-trend losses
                if (
                    "downtrend" in exp.pattern_tags and exp.decision["action"] == "LONG"
                ) or (
                    "uptrend" in exp.pattern_tags and exp.decision["action"] == "SHORT"
                ):
                    loss_patterns["counter_trend"] += 1

            # Generate insights for common loss patterns
            total_losses = len(losing_trades)
            for pattern, count in loss_patterns.items():
                if count / total_losses > 0.4:  # 40% of losses
                    insights.append(
                        LearningInsight(
                            insight_id=f"risk_factor_{pattern}_{datetime.now(UTC).timestamp()}",
                            insight_type="risk",
                            description=f"Risk factor '{pattern}' present in {count}/{total_losses} losing trades",
                            supporting_evidence=[],
                            confidence=0.7,
                            actionable=True,
                        )
                    )

        return insights

    def _calculate_pattern_confidence(self, sample_size: int, win_rate: float) -> float:
        """Calculate confidence score for a pattern based on sample size and performance."""
        # Base confidence from win rate
        base_confidence = win_rate

        # Adjust for sample size (more samples = higher confidence)
        sample_factor = min(sample_size / 20, 1.0)  # Max confidence at 20 samples

        # Apply decay for older patterns
        # This would be implemented with actual timestamps
        time_decay = settings.mcp.confidence_decay_rate

        # Combined confidence
        confidence = base_confidence * sample_factor * time_decay

        return max(0.1, min(0.95, confidence))

    async def get_recommendations_for_market(
        self,
        current_indicators: dict[str, float],
        current_dominance: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Get specific recommendations for current market conditions.

        Args:
            current_indicators: Current technical indicators
            current_dominance: Current dominance data

        Returns:
            Recommendations based on learned patterns
        """
        recommendations = {
            "confidence": 0.5,
            "suggested_action": "HOLD",
            "suggested_size_pct": 10,
            "suggested_leverage": settings.trading.leverage,
            "reasoning": [],
            "warnings": [],
        }

        # Identify current market conditions
        active_conditions = []
        for condition_name, condition_func in self.market_conditions.items():
            if "dominance" in condition_name:
                if current_dominance and condition_func(current_dominance):
                    active_conditions.append(condition_name)
            elif condition_func(current_indicators):
                active_conditions.append(condition_name)

        # Look up historical performance for these conditions
        relevant_patterns = []
        for pattern_name, pattern_perf in self.pattern_performance.items():
            # Check if this pattern is relevant to current conditions
            for condition in active_conditions:
                if condition in pattern_name or pattern_name in condition:
                    relevant_patterns.append(pattern_perf)

        if not relevant_patterns:
            reasoning_list = recommendations.get("reasoning", [])
            if isinstance(reasoning_list, list):
                reasoning_list.append(
                    "No strong historical patterns match current conditions"
                )
            return recommendations

        # Find best performing relevant pattern
        best_pattern = max(
            relevant_patterns, key=lambda p: p.win_rate * p.confidence_score
        )

        if best_pattern.win_rate > 0.6 and best_pattern.confidence_score > 0.5:
            # Extract action from pattern name
            if "action_long" in best_pattern.pattern_name:
                recommendations["suggested_action"] = "LONG"
            elif "action_short" in best_pattern.pattern_name:
                recommendations["suggested_action"] = "SHORT"

            recommendations["confidence"] = best_pattern.confidence_score
            reasoning_list = recommendations.get("reasoning", [])
            if isinstance(reasoning_list, list):
                reasoning_list.append(
                    f"Pattern '{best_pattern.pattern_name}' shows {best_pattern.win_rate:.1%} "
                    f"success rate over {best_pattern.occurrence_count} trades"
                )

            # Adjust size based on confidence
            if best_pattern.confidence_score > 0.7:
                recommendations["suggested_size_pct"] = 15
            elif best_pattern.confidence_score > 0.8:
                recommendations["suggested_size_pct"] = 20

        # Add warnings for risk factors
        warnings_list = recommendations.get("warnings", [])
        if isinstance(warnings_list, list):
            if "high_stablecoin_dominance" in active_conditions:
                warnings_list.append(
                    "High stablecoin dominance indicates risk-off sentiment"
                )

            if current_indicators.get("rsi", 50) > 80:
                warnings_list.append(
                    "Extreme overbought conditions - consider reduced position size"
                )

        return recommendations

    def export_learning_report(self) -> dict[str, Any]:
        """Export a comprehensive learning report."""
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "total_patterns_learned": len(self.pattern_performance),
            "total_insights": len(self.insights),
            "top_performing_patterns": [],
            "recent_insights": [],
            "recommended_focus_areas": [],
        }

        # Top patterns by win rate
        sorted_patterns = sorted(
            self.pattern_performance.values(),
            key=lambda p: p.win_rate * p.confidence_score,
            reverse=True,
        )

        for pattern in sorted_patterns[:5]:
            top_patterns_list = report.get("top_performing_patterns", [])
            if isinstance(top_patterns_list, list):
                top_patterns_list.append(
                    {
                        "pattern": pattern.pattern_name,
                        "win_rate": pattern.win_rate,
                        "avg_pnl": pattern.avg_pnl,
                        "confidence": pattern.confidence_score,
                        "samples": pattern.occurrence_count,
                    }
                )

        # Recent insights
        recent_insights = sorted(
            self.insights, key=lambda i: i.created_at, reverse=True
        )[:10]

        for insight in recent_insights:
            recent_insights_list = report.get("recent_insights", [])
            if isinstance(recent_insights_list, list):
                recent_insights_list.append(
                    {
                        "type": insight.insight_type,
                        "description": insight.description,
                        "confidence": insight.confidence,
                        "actionable": insight.actionable,
                    }
                )

        # Recommended focus areas based on learning
        focus_areas_list = report.get("recommended_focus_areas", [])
        if isinstance(focus_areas_list, list):
            if len(self.pattern_performance) < 10:
                focus_areas_list.append(
                    "Gather more trading data across different market conditions"
                )

            low_confidence_patterns = [
                p for p in self.pattern_performance.values() if p.confidence_score < 0.5
            ]
            if len(low_confidence_patterns) > 5:
                focus_areas_list.append(
                    "Review and refine pattern identification logic"
                )

        return report
