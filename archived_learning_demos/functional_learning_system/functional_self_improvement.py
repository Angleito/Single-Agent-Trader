"""
Functional Self-Improvement Engine using pure functional algorithms.

This module provides a functional alternative to the imperative self-improvement
engine, using pure functions, immutable data structures, and functional composition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from bot.fp.types.result import Result, Ok, Err
from bot.mcp.memory_server import MCPMemoryServer

from .experience import ExperienceState, TradeExperience
from .learning_algorithms import (
    LearningResult,
    PatternAnalysis,
    StrategyInsight,
    MarketRegime,
    analyze_trading_patterns,
    generate_strategy_insights,
    optimize_parameters,
    identify_market_regimes,
    calculate_pattern_confidence,
)
from .combinators import (
    sequence_learning_operations,
    parallel_learning_analysis,
    validate_minimum_completed,
    build_analysis_pipeline,
    fold_experiences,
    filter_experiences,
    group_experiences_by,
    tap,
    memoize,
)
from .memory_effects import MemoryEffectInterpreter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImprovementRecommendation:
    """Immutable recommendation for strategy improvement."""
    
    category: str  # "parameter", "timing", "risk", "market_condition"
    priority: str  # "high", "medium", "low"
    description: str
    implementation_steps: Tuple[str, ...]
    expected_benefit: str
    confidence_level: float
    supporting_insights: Tuple[StrategyInsight, ...]
    
    def is_high_priority(self) -> bool:
        """Check if this is a high priority recommendation."""
        return self.priority == "high" and self.confidence_level > 0.7


@dataclass(frozen=True)
class PerformanceMetrics:
    """Immutable performance metrics snapshot."""
    
    total_trades: int
    successful_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    max_drawdown: float
    avg_trade_duration_hours: float
    sharpe_ratio: Optional[float]
    timestamp: datetime
    
    @classmethod
    def from_experiences(
        cls, 
        experiences: Tuple[TradeExperience, ...]
    ) -> PerformanceMetrics:
        """Calculate metrics from experiences (pure function)."""
        completed_experiences = tuple(exp for exp in experiences if exp.is_completed())
        
        if not completed_experiences:
            return cls.empty()
        
        total_trades = len(completed_experiences)
        successful_trades = sum(1 for exp in completed_experiences if exp.is_successful())
        win_rate = successful_trades / total_trades
        
        pnls = [exp.get_pnl() or 0.0 for exp in completed_experiences]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total_trades
        
        # Calculate max drawdown
        running_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0
        
        for pnl in pnls:
            running_pnl += pnl
            peak_pnl = max(peak_pnl, running_pnl)
            drawdown = peak_pnl - running_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate average duration
        durations = [
            exp.duration_minutes or 0.0 for exp in completed_experiences 
            if exp.duration_minutes
        ]
        avg_duration_hours = (sum(durations) / len(durations) / 60) if durations else 0.0
        
        # Calculate simple Sharpe ratio (if we have enough data)
        sharpe_ratio = None
        if len(pnls) >= 10:
            mean_return = sum(pnls) / len(pnls)
            variance = sum((pnl - mean_return) ** 2 for pnl in pnls) / len(pnls)
            std_dev = variance ** 0.5
            sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0.0
        
        return cls(
            total_trades=total_trades,
            successful_trades=successful_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl_per_trade=avg_pnl,
            max_drawdown=max_drawdown,
            avg_trade_duration_hours=avg_duration_hours,
            sharpe_ratio=sharpe_ratio,
            timestamp=datetime.now(UTC),
        )
    
    @classmethod
    def empty(cls) -> PerformanceMetrics:
        """Create empty performance metrics."""
        return cls(
            total_trades=0,
            successful_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_pnl_per_trade=0.0,
            max_drawdown=0.0,
            avg_trade_duration_hours=0.0,
            sharpe_ratio=None,
            timestamp=datetime.now(UTC),
        )


@dataclass(frozen=True)
class FunctionalSelfImprovementState:
    """Immutable state for the self-improvement engine."""
    
    experience_state: ExperienceState
    current_metrics: PerformanceMetrics
    pattern_analyses: Dict[str, PatternAnalysis]
    market_regimes: Tuple[MarketRegime, ...]
    active_insights: Tuple[StrategyInsight, ...]
    recommendations: Tuple[ImprovementRecommendation, ...]
    last_analysis_time: datetime
    
    @classmethod
    def from_experience_state(cls, experience_state: ExperienceState) -> FunctionalSelfImprovementState:
        """Create improvement state from experience state."""
        return cls(
            experience_state=experience_state,
            current_metrics=PerformanceMetrics.from_experiences(experience_state.experiences),
            pattern_analyses={},
            market_regimes=(),
            active_insights=(),
            recommendations=(),
            last_analysis_time=datetime.now(UTC),
        )
    
    def with_analysis_results(
        self,
        pattern_analyses: Dict[str, PatternAnalysis],
        market_regimes: Tuple[MarketRegime, ...],
        insights: Tuple[StrategyInsight, ...],
        recommendations: Tuple[ImprovementRecommendation, ...],
    ) -> FunctionalSelfImprovementState:
        """Return new state with analysis results."""
        from dataclasses import replace
        return replace(
            self,
            pattern_analyses=pattern_analyses,
            market_regimes=market_regimes,
            active_insights=insights,
            recommendations=recommendations,
            last_analysis_time=datetime.now(UTC),
        )


class FunctionalSelfImprovementEngine:
    """
    Functional self-improvement engine using pure algorithms and immutable state.
    
    Provides strategy optimization and learning insights through functional
    composition of analysis algorithms.
    """
    
    def __init__(self, memory_server: MCPMemoryServer):
        """Initialize the functional self-improvement engine."""
        self.memory_interpreter = MemoryEffectInterpreter(memory_server)
        
        # Memoized functions for performance
        self.memoized_pattern_analysis = memoize(analyze_trading_patterns)
        self.memoized_regime_identification = memoize(identify_market_regimes)
        
        logger.info("🚀 Functional Self-Improvement Engine: Initialized with pure algorithms")
    
    async def analyze_performance(
        self, 
        experience_state: ExperienceState,
        analysis_hours: int = 24,
    ) -> Result[FunctionalSelfImprovementState, str]:
        """
        Perform comprehensive performance analysis using functional algorithms.
        
        Uses pure functions and functional composition to analyze patterns,
        generate insights, and create improvement recommendations.
        """
        try:
            # Validation pipeline
            validation_pipeline = build_analysis_pipeline(
                validate_minimum_completed(5),
                tap(lambda state: logger.info(
                    "📊 Starting functional analysis on %d experiences",
                    state.total_experiences
                ))
            )
            
            validation_result = validation_pipeline(experience_state)
            if validation_result.is_failure():
                return Err(validation_result.failure())
            
            # Core analysis operations
            analysis_operations = [
                self._analyze_patterns_functional,
                self._generate_insights_functional,
                self._identify_regimes_functional,
            ]
            
            # Run analyses in parallel
            parallel_analysis = parallel_learning_analysis(analysis_operations)
            analysis_results = parallel_analysis(experience_state)
            
            if analysis_results.is_failure():
                return Err(analysis_results.failure())
            
            results = analysis_results.success()
            
            # Extract results
            pattern_analyses = results[0].insights[0] if results and results[0].insights else {}
            insights = tuple(results[1].insights) if len(results) > 1 else ()
            market_regimes = tuple(results[2].insights) if len(results) > 2 else ()
            
            # Generate recommendations
            recommendations = self._generate_recommendations_functional(
                experience_state, pattern_analyses, insights, market_regimes
            )
            
            # Create comprehensive improvement state
            improvement_state = FunctionalSelfImprovementState.from_experience_state(
                experience_state
            ).with_analysis_results(
                pattern_analyses=pattern_analyses,
                market_regimes=market_regimes, 
                insights=insights,
                recommendations=recommendations,
            )
            
            logger.info(
                "✅ Functional Analysis Complete: %d patterns, %d insights, %d recommendations",
                len(pattern_analyses),
                len(insights),
                len(recommendations),
            )
            
            return Ok(improvement_state)
            
        except Exception as e:
            logger.exception("Functional performance analysis failed")
            return Err(f"Analysis failed: {str(e)}")
    
    def get_optimization_suggestions(
        self,
        improvement_state: FunctionalSelfImprovementState,
        parameters: List[str],
    ) -> Result[Dict[str, Tuple[float, float]], str]:
        """
        Get parameter optimization suggestions using functional algorithms.
        
        Returns optimized parameter values and expected improvements.
        """
        try:
            optimization_results = {}
            
            for parameter in parameters:
                # Define parameter value ranges
                parameter_ranges = {
                    "max_size_pct": [5, 10, 15, 20, 25],
                    "leverage": [1, 2, 3, 5, 10],
                    "stop_loss_pct": [1.0, 1.5, 2.0, 2.5, 3.0],
                    "take_profit_pct": [2.0, 3.0, 4.0, 5.0, 6.0],
                }
                
                if parameter in parameter_ranges:
                    result = optimize_parameters(
                        improvement_state.experience_state,
                        parameter,
                        parameter_ranges[parameter],
                    )
                    
                    if result.is_success():
                        optimal_value, expected_improvement = result.success()
                        optimization_results[parameter] = (optimal_value, expected_improvement)
            
            return Ok(optimization_results)
            
        except Exception as e:
            return Err(f"Parameter optimization failed: {str(e)}")
    
    def get_market_condition_recommendations(
        self,
        improvement_state: FunctionalSelfImprovementState,
        current_indicators: Dict[str, float],
        current_dominance: Optional[Dict[str, float]] = None,
    ) -> Result[Dict[str, Any], str]:
        """
        Get recommendations for current market conditions using functional analysis.
        """
        try:
            recommendations = {
                "suggested_action": "HOLD",
                "confidence": 0.5,
                "suggested_size_pct": 10,
                "reasoning": [],
                "warnings": [],
            }
            
            # Find matching market regime
            matching_regime = None
            for regime in improvement_state.market_regimes:
                if regime.matches_current_market(current_indicators, current_dominance):
                    matching_regime = regime
                    break
            
            if matching_regime:
                performance = matching_regime.performance_stats
                win_rate = performance.get("win_rate", 0.5)
                
                recommendations["confidence"] = win_rate
                recommendations["reasoning"].append(
                    f"Market regime '{matching_regime.regime_name}' "
                    f"has {win_rate:.1%} historical success rate"
                )
                
                if win_rate > 0.6:
                    recommendations["suggested_size_pct"] = 15
                elif win_rate > 0.7:
                    recommendations["suggested_size_pct"] = 20
            
            # Check pattern-based recommendations
            for insight in improvement_state.active_insights:
                if insight.is_actionable():
                    if "reduce" in insight.recommended_action.lower():
                        recommendations["warnings"].append(insight.description)
                    else:
                        recommendations["reasoning"].append(insight.description)
            
            # Add risk warnings based on indicators
            if current_indicators.get("rsi", 50) > 80:
                recommendations["warnings"].append("Extreme overbought conditions")
                recommendations["suggested_size_pct"] = max(
                    recommendations["suggested_size_pct"] * 0.5, 5
                )
            
            return Ok(recommendations)
            
        except Exception as e:
            return Err(f"Market condition analysis failed: {str(e)}")
    
    def export_improvement_report(
        self, 
        improvement_state: FunctionalSelfImprovementState
    ) -> Dict[str, Any]:
        """Export comprehensive improvement report."""
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "analysis_summary": {
                "total_experiences": improvement_state.experience_state.total_experiences,
                "completed_trades": improvement_state.experience_state.total_completed,
                "success_rate": improvement_state.current_metrics.win_rate,
                "total_pnl": improvement_state.current_metrics.total_pnl,
                "sharpe_ratio": improvement_state.current_metrics.sharpe_ratio,
            },
            "pattern_performance": {
                name: {
                    "success_rate": analysis.success_rate,
                    "avg_pnl": analysis.avg_pnl,
                    "confidence": analysis.confidence_score,
                    "sample_size": analysis.occurrence_count,
                    "reliable": analysis.is_reliable(),
                }
                for name, analysis in improvement_state.pattern_analyses.items()
            },
            "market_regimes": [
                {
                    "name": regime.regime_name,
                    "performance": regime.performance_stats,
                    "indicators": regime.indicators_range,
                }
                for regime in improvement_state.market_regimes
            ],
            "active_insights": [
                {
                    "type": insight.insight_type,
                    "description": insight.description,
                    "action": insight.recommended_action,
                    "confidence": insight.confidence,
                    "actionable": insight.is_actionable(),
                }
                for insight in improvement_state.active_insights
            ],
            "recommendations": [
                {
                    "category": rec.category,
                    "priority": rec.priority,
                    "description": rec.description,
                    "implementation": list(rec.implementation_steps),
                    "expected_benefit": rec.expected_benefit,
                    "confidence": rec.confidence_level,
                    "high_priority": rec.is_high_priority(),
                }
                for rec in improvement_state.recommendations
            ],
            "performance_trends": self._calculate_performance_trends(
                improvement_state.experience_state
            ),
        }
        
        return report
    
    # Functional analysis methods
    
    def _analyze_patterns_functional(
        self, 
        experience_state: ExperienceState
    ) -> Result[LearningResult, str]:
        """Analyze patterns using functional algorithms."""
        try:
            pattern_analyses = self.memoized_pattern_analysis(experience_state)
            
            insights = [
                f"Pattern '{name}': {analysis.success_rate:.1%} success rate, "
                f"{analysis.occurrence_count} samples"
                for name, analysis in pattern_analyses.items()
                if analysis.is_reliable()
            ]
            
            avg_confidence = (
                sum(analysis.confidence_score for analysis in pattern_analyses.values()) 
                / len(pattern_analyses)
            ) if pattern_analyses else 0.0
            
            return Ok(LearningResult(
                insights=(pattern_analyses,),  # Pass the full analyses as first insight
                confidence=avg_confidence,
                sample_size=sum(analysis.occurrence_count for analysis in pattern_analyses.values()),
                analysis_timestamp=datetime.now(UTC),
            ))
            
        except Exception as e:
            return Err(f"Pattern analysis failed: {str(e)}")
    
    def _generate_insights_functional(
        self, 
        experience_state: ExperienceState
    ) -> Result[LearningResult, str]:
        """Generate strategy insights using functional algorithms."""
        try:
            insights = generate_strategy_insights(experience_state)
            
            insight_descriptions = [
                f"{insight.insight_type}: {insight.description}"
                for insight in insights
                if insight.is_actionable()
            ]
            
            avg_confidence = (
                sum(insight.confidence for insight in insights) / len(insights)
            ) if insights else 0.0
            
            return Ok(LearningResult(
                insights=tuple(insight_descriptions),
                confidence=avg_confidence,
                sample_size=len(insights),
                analysis_timestamp=datetime.now(UTC),
            ))
            
        except Exception as e:
            return Err(f"Insight generation failed: {str(e)}")
    
    def _identify_regimes_functional(
        self, 
        experience_state: ExperienceState
    ) -> Result[LearningResult, str]:
        """Identify market regimes using functional algorithms."""
        try:
            market_regimes = self.memoized_regime_identification(experience_state)
            
            regime_descriptions = [
                f"Regime '{regime.regime_name}': "
                f"{regime.performance_stats.get('win_rate', 0):.1%} success rate"
                for regime in market_regimes
            ]
            
            return Ok(LearningResult(
                insights=tuple(regime_descriptions),
                confidence=0.7,  # Fixed confidence for regime analysis
                sample_size=len(market_regimes),
                analysis_timestamp=datetime.now(UTC),
            ))
            
        except Exception as e:
            return Err(f"Regime identification failed: {str(e)}")
    
    def _generate_recommendations_functional(
        self,
        experience_state: ExperienceState,
        pattern_analyses: Dict[str, PatternAnalysis],
        insights: Tuple[StrategyInsight, ...],
        market_regimes: Tuple[MarketRegime, ...],
    ) -> Tuple[ImprovementRecommendation, ...]:
        """Generate improvement recommendations using functional composition."""
        recommendations = []
        
        # Parameter optimization recommendations
        high_confidence_patterns = [
            analysis for analysis in pattern_analyses.values()
            if analysis.is_reliable() and analysis.confidence_score > 0.7
        ]
        
        if high_confidence_patterns:
            best_pattern = max(high_confidence_patterns, key=lambda x: x.success_rate)
            
            if best_pattern.success_rate > 0.7:
                recommendations.append(ImprovementRecommendation(
                    category="parameter",
                    priority="high",
                    description=f"Focus on pattern '{best_pattern.pattern_name}' "
                               f"with {best_pattern.success_rate:.1%} success rate",
                    implementation_steps=(
                        "Identify market conditions for this pattern",
                        "Increase position sizing when pattern is detected",
                        "Monitor performance and adjust accordingly",
                    ),
                    expected_benefit=f"Potential {(best_pattern.success_rate - 0.5) * 100:.1f}% "
                                   "improvement in win rate",
                    confidence_level=best_pattern.confidence_score,
                    supporting_insights=tuple(
                        insight for insight in insights 
                        if best_pattern.pattern_name in insight.description
                    ),
                ))
        
        # Risk management recommendations
        risk_insights = [
            insight for insight in insights 
            if insight.insight_type == "risk" and insight.is_actionable()
        ]
        
        for risk_insight in risk_insights:
            recommendations.append(ImprovementRecommendation(
                category="risk",
                priority="high",
                description=risk_insight.description,
                implementation_steps=(
                    "Review risk management parameters",
                    "Implement suggested changes gradually",
                    "Monitor impact on performance",
                ),
                expected_benefit=f"Potential {risk_insight.expected_improvement:.1%} "
                               "improvement in risk-adjusted returns",
                confidence_level=risk_insight.confidence,
                supporting_insights=(risk_insight,),
            ))
        
        # Market regime recommendations
        profitable_regimes = [
            regime for regime in market_regimes
            if regime.performance_stats.get("win_rate", 0) > 0.6
        ]
        
        if profitable_regimes:
            recommendations.append(ImprovementRecommendation(
                category="market_condition",
                priority="medium",
                description=f"Focus trading on {len(profitable_regimes)} "
                           "profitable market regimes",
                implementation_steps=(
                    "Create market regime detection system",
                    "Adjust position sizing based on regime",
                    "Reduce trading in unfavorable regimes",
                ),
                expected_benefit="Better market timing and improved consistency",
                confidence_level=0.65,
                supporting_insights=(),
            ))
        
        return tuple(recommendations)
    
    def _calculate_performance_trends(
        self, 
        experience_state: ExperienceState
    ) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        completed_experiences = experience_state.filter_experiences(
            lambda exp: exp.is_completed()
        )
        
        if len(completed_experiences) < 10:
            return {"insufficient_data": True}
        
        # Group by week
        weekly_groups = group_experiences_by(
            lambda exp: exp.timestamp.strftime("%Y-W%U")
        )(experience_state)
        
        weekly_performance = []
        for week, experiences in weekly_groups.items():
            if experiences and len(experiences) >= 3:
                metrics = PerformanceMetrics.from_experiences(experiences)
                weekly_performance.append({
                    "week": week,
                    "win_rate": metrics.win_rate,
                    "total_pnl": metrics.total_pnl,
                    "trade_count": metrics.total_trades,
                })
        
        # Calculate trends
        if len(weekly_performance) >= 4:
            recent_weeks = weekly_performance[-4:]
            earlier_weeks = weekly_performance[-8:-4] if len(weekly_performance) >= 8 else weekly_performance[:-4]
            
            recent_avg_win_rate = sum(w["win_rate"] for w in recent_weeks) / len(recent_weeks)
            earlier_avg_win_rate = sum(w["win_rate"] for w in earlier_weeks) / len(earlier_weeks) if earlier_weeks else recent_avg_win_rate
            
            win_rate_trend = recent_avg_win_rate - earlier_avg_win_rate
            
            return {
                "weekly_performance": weekly_performance,
                "win_rate_trend": win_rate_trend,
                "trend_direction": "improving" if win_rate_trend > 0.05 else "declining" if win_rate_trend < -0.05 else "stable",
                "weeks_analyzed": len(weekly_performance),
            }
        
        return {
            "weekly_performance": weekly_performance,
            "insufficient_data_for_trends": True,
        }