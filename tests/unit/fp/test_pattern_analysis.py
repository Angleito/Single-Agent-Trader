"""
Functional programming tests for pattern analysis and statistics.

These tests validate pattern identification, statistical analysis,
and learning insights generation using immutable data structures
and pure functional patterns.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot.fp.types import (
    ExperienceId,
    LearningInsight,
    MarketSnapshot,
    Nothing,
    PatternStatistics,
    PatternTag,
    Some,
    Symbol,
    TradingExperienceFP,
    TradingOutcome,
)

# Pattern Recognition Tests


class TestPatternRecognition:
    """Test pattern recognition and tagging functionality."""

    def create_market_snapshot(
        self,
        symbol_str: str = "BTC-USD",
        price: Decimal = Decimal(50000),
        indicators: dict[str, float] = None,
        dominance_data: dict[str, float] = None,
        position_side: str = "FLAT",
    ) -> MarketSnapshot:
        """Helper to create market snapshots for testing."""
        symbol = Symbol.create(symbol_str).success()
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price=price,
            indicators=indicators or {},
            dominance_data=dominance_data,
            position_side=position_side,
            position_size=Decimal(0),
        )

    def test_oversold_pattern_recognition(self):
        """Test recognition of oversold market conditions."""
        oversold_snapshot = self.create_market_snapshot(
            indicators={"rsi": 25.0, "stoch_rsi": 20.0}
        )

        # This would be implemented in a pattern recognition service
        # For now, we test the pattern tag creation
        oversold_pattern = PatternTag.create("oversold_rsi").success()

        assert oversold_pattern.name == "oversold_rsi"
        assert str(oversold_pattern) == "oversold_rsi"

        # Test pattern matching logic
        rsi = oversold_snapshot.indicators.get("rsi", 50)
        is_oversold = rsi < 30
        assert is_oversold is True

    def test_overbought_pattern_recognition(self):
        """Test recognition of overbought market conditions."""
        overbought_snapshot = self.create_market_snapshot(
            indicators={"rsi": 80.0, "stoch_rsi": 85.0}
        )

        overbought_pattern = PatternTag.create("overbought_rsi").success()

        assert overbought_pattern.name == "overbought_rsi"

        # Test pattern matching logic
        rsi = overbought_snapshot.indicators.get("rsi", 50)
        is_overbought = rsi > 70
        assert is_overbought is True

    def test_trend_pattern_recognition(self):
        """Test recognition of trend patterns."""
        uptrend_snapshot = self.create_market_snapshot(
            indicators={
                "ema_fast": 50100.0,
                "ema_slow": 49900.0,
                "sma_20": 49800.0,
                "sma_50": 49500.0,
            }
        )

        uptrend_pattern = PatternTag.create("strong_uptrend").success()

        # Test trend logic
        ema_fast = uptrend_snapshot.indicators.get("ema_fast", 0)
        ema_slow = uptrend_snapshot.indicators.get("ema_slow", 0)
        is_uptrend = ema_fast > ema_slow
        assert is_uptrend is True

        # Test multiple timeframe alignment
        sma_20 = uptrend_snapshot.indicators.get("sma_20", 0)
        sma_50 = uptrend_snapshot.indicators.get("sma_50", 0)
        is_strong_uptrend = ema_fast > ema_slow > sma_20 > sma_50
        assert is_strong_uptrend is True

    def test_volume_pattern_recognition(self):
        """Test recognition of volume patterns."""
        high_volume_snapshot = self.create_market_snapshot(
            indicators={
                "volume": 150000.0,
                "volume_sma_20": 100000.0,
                "volume_ratio": 1.5,
            }
        )

        high_volume_pattern = PatternTag.create("high_volume_breakout").success()

        # Test volume logic
        volume = high_volume_snapshot.indicators.get("volume", 0)
        volume_sma = high_volume_snapshot.indicators.get("volume_sma_20", 0)
        is_high_volume = volume > volume_sma * 1.3
        assert is_high_volume is True

    def test_dominance_pattern_recognition(self):
        """Test recognition of dominance-based patterns."""
        high_dominance_snapshot = self.create_market_snapshot(
            dominance_data={
                "stablecoin_dominance": 12.5,
                "dominance_24h_change": 2.1,
                "dominance_rsi": 75.0,
            }
        )

        high_dominance_pattern = PatternTag.create(
            "high_stablecoin_dominance"
        ).success()

        # Test dominance logic
        dominance = high_dominance_snapshot.dominance_data.get(
            "stablecoin_dominance", 0
        )
        is_high_dominance = dominance > 10.0
        assert is_high_dominance is True

        # Test rising dominance
        dominance_change = high_dominance_snapshot.dominance_data.get(
            "dominance_24h_change", 0
        )
        is_rising_dominance = dominance_change > 1.0
        assert is_rising_dominance is True

    def test_cipher_pattern_recognition(self):
        """Test recognition of VuManChu Cipher patterns."""
        cipher_extreme_snapshot = self.create_market_snapshot(
            indicators={
                "cipher_a_dot": -25.0,
                "cipher_b_wave": -85.0,
                "cipher_b_money_flow": 20.0,
            }
        )

        cipher_pattern = PatternTag.create("cipher_b_extreme_low").success()

        # Test cipher logic
        cipher_b_wave = cipher_extreme_snapshot.indicators.get("cipher_b_wave", 0)
        is_extreme_low = cipher_b_wave < -80
        assert is_extreme_low is True

        # Test cipher A confirmation
        cipher_a_dot = cipher_extreme_snapshot.indicators.get("cipher_a_dot", 0)
        is_cipher_a_confirmation = cipher_a_dot < -20
        assert is_cipher_a_confirmation is True


# Pattern Statistics Tests


class TestPatternStatistics:
    """Test pattern statistics calculation and analysis."""

    def create_experience_with_outcome(
        self,
        experience_id: str,
        patterns: list[PatternTag],
        pnl: Decimal,
        entry_price: Decimal = Decimal(50000),
        duration_minutes: float = 30.0,
    ) -> TradingExperienceFP:
        """Helper to create trading experience with outcome."""
        exp_id = ExperienceId.create(experience_id).success()

        exit_price = entry_price + (pnl / Decimal(1))  # Simplified calculation
        outcome = TradingOutcome.create(
            pnl=pnl,
            exit_price=exit_price,
            entry_price=entry_price,
            duration_minutes=duration_minutes,
        ).success()

        experience = TradingExperienceFP(
            experience_id=exp_id,
            timestamp=datetime.utcnow(),
            market_snapshot=MagicMock(),
            trade_decision="LONG",
            decision_rationale="Test experience",
            pattern_tags=patterns,
            outcome=Some(outcome),
            learned_insights=Nothing(),
            confidence_score=Decimal("0.5"),
        )

        return experience

    def test_pattern_statistics_basic_calculation(self):
        """Test basic pattern statistics calculation."""
        pattern = PatternTag.create("oversold_rsi").success()

        # Create 10 experiences with 70% success rate
        experiences = []
        for i in range(10):
            pnl = Decimal(100) if i < 7 else Decimal(-50)
            exp = self.create_experience_with_outcome(
                experience_id=f"exp_{i}",
                patterns=[pattern],
                pnl=pnl,
            )
            experiences.append(exp)

        result = PatternStatistics.calculate(pattern, experiences)
        assert result.is_success()

        stats = result.success()
        assert stats.pattern == pattern
        assert stats.total_occurrences == 10
        assert stats.successful_trades == 7
        assert stats.success_rate == Decimal("0.7")
        assert stats.total_pnl == Decimal(550)  # 7*100 - 3*50
        assert stats.average_pnl == Decimal(55)
        assert stats.is_profitable()
        assert stats.is_reliable(min_occurrences=5)

    def test_pattern_statistics_filter_by_pattern(self):
        """Test filtering experiences by specific pattern."""
        target_pattern = PatternTag.create("oversold_rsi").success()
        other_pattern = PatternTag.create("overbought_rsi").success()

        # Create experiences with different patterns
        experiences = []

        # 5 experiences with target pattern (80% success)
        for i in range(5):
            pnl = Decimal(100) if i < 4 else Decimal(-50)
            exp = self.create_experience_with_outcome(
                experience_id=f"target_{i}",
                patterns=[target_pattern],
                pnl=pnl,
            )
            experiences.append(exp)

        # 3 experiences with other pattern (should be ignored)
        for i in range(3):
            exp = self.create_experience_with_outcome(
                experience_id=f"other_{i}",
                patterns=[other_pattern],
                pnl=Decimal(-100),  # All losses
            )
            experiences.append(exp)

        # Calculate stats for target pattern only
        result = PatternStatistics.calculate(target_pattern, experiences)
        assert result.is_success()

        stats = result.success()
        assert stats.total_occurrences == 5  # Only target pattern
        assert stats.successful_trades == 4
        assert stats.success_rate == Decimal("0.8")
        assert stats.total_pnl == Decimal(350)  # 4*100 - 1*50

    def test_pattern_statistics_multiple_patterns_per_experience(self):
        """Test statistics when experiences have multiple patterns."""
        pattern1 = PatternTag.create("oversold_rsi").success()
        pattern2 = PatternTag.create("high_volume").success()

        # Create experiences with both patterns
        experiences = []
        for i in range(6):
            pnl = Decimal(120) if i < 5 else Decimal(-80)
            exp = self.create_experience_with_outcome(
                experience_id=f"multi_{i}",
                patterns=[pattern1, pattern2],  # Both patterns
                pnl=pnl,
            )
            experiences.append(exp)

        # Calculate stats for each pattern
        stats1_result = PatternStatistics.calculate(pattern1, experiences)
        stats2_result = PatternStatistics.calculate(pattern2, experiences)

        assert stats1_result.is_success()
        assert stats2_result.is_success()

        stats1 = stats1_result.success()
        stats2 = stats2_result.success()

        # Both should have same statistics since all experiences have both patterns
        assert stats1.total_occurrences == stats2.total_occurrences == 6
        assert stats1.successful_trades == stats2.successful_trades == 5
        assert (
            stats1.success_rate
            == stats2.success_rate
            == Decimal("0.833333333333333333333333333")
        )

    def test_pattern_statistics_no_completed_experiences(self):
        """Test statistics calculation with no completed experiences."""
        pattern = PatternTag.create("incomplete_pattern").success()

        # Create experiences without outcomes
        experiences = []
        for i in range(3):
            exp_id = ExperienceId.create(f"incomplete_{i}").success()
            exp = TradingExperienceFP(
                experience_id=exp_id,
                timestamp=datetime.utcnow(),
                market_snapshot=MagicMock(),
                trade_decision="LONG",
                decision_rationale="Incomplete test",
                pattern_tags=[pattern],
                outcome=Nothing(),  # No outcome
                learned_insights=Nothing(),
                confidence_score=Decimal("0.5"),
            )
            experiences.append(exp)

        result = PatternStatistics.calculate(pattern, experiences)
        assert result.is_success()

        stats = result.success()
        assert stats.total_occurrences == 0  # No completed experiences
        assert stats.successful_trades == 0
        assert stats.total_pnl == Decimal(0)
        assert not stats.is_profitable()
        assert not stats.is_reliable()

    def test_pattern_statistics_edge_cases(self):
        """Test pattern statistics edge cases."""
        pattern = PatternTag.create("edge_case").success()

        # Empty experience list
        result = PatternStatistics.calculate(pattern, [])
        assert result.is_success()
        stats = result.success()
        assert stats.total_occurrences == 0

        # Single experience
        single_exp = self.create_experience_with_outcome(
            experience_id="single",
            patterns=[pattern],
            pnl=Decimal(100),
        )

        result = PatternStatistics.calculate(pattern, [single_exp])
        assert result.is_success()
        stats = result.success()
        assert stats.total_occurrences == 1
        assert stats.successful_trades == 1
        assert stats.success_rate == Decimal(1)
        assert stats.is_profitable()
        assert not stats.is_reliable(min_occurrences=5)  # Not enough data

    def test_pattern_reliability_thresholds(self):
        """Test pattern reliability with different thresholds."""
        pattern = PatternTag.create("reliability_test").success()

        # Create experiences
        experiences = []
        for i in range(8):
            pnl = Decimal(50)
            exp = self.create_experience_with_outcome(
                experience_id=f"rel_{i}",
                patterns=[pattern],
                pnl=pnl,
            )
            experiences.append(exp)

        result = PatternStatistics.calculate(pattern, experiences)
        assert result.is_success()
        stats = result.success()

        # Test different reliability thresholds
        assert stats.is_reliable(min_occurrences=5) is True  # 8 >= 5
        assert stats.is_reliable(min_occurrences=10) is False  # 8 < 10
        assert stats.is_reliable(min_occurrences=1) is True  # 8 >= 1


# Learning Insights Tests


class TestLearningInsights:
    """Test learning insights generation and analysis."""

    def test_performance_insight_generation(self):
        """Test generation of performance-based insights."""
        # Create high-performance pattern insight
        high_perf_result = LearningInsight.create(
            insight_type="pattern_performance",
            description="Oversold RSI pattern shows exceptional success rate (85%)",
            confidence=0.9,
            supporting_evidence=[
                "20 occurrences analyzed",
                "85% success rate",
                "Average profit: $120",
                "Consistent across different market conditions",
            ],
            related_patterns=[PatternTag.create("oversold_rsi").success()],
        )

        assert high_perf_result.is_success()
        insight = high_perf_result.success()

        assert insight.insight_type == "pattern_performance"
        assert "85%" in insight.description
        assert insight.confidence == Decimal("0.9")
        assert len(insight.supporting_evidence) == 4
        assert len(insight.related_patterns) == 1

    def test_timing_insight_generation(self):
        """Test generation of timing-based insights."""
        timing_result = LearningInsight.create(
            insight_type="timing_pattern",
            description="Quick exits (<30min) show 60% higher success rate than longer holds",
            confidence=0.7,
            supporting_evidence=[
                "Fast exits: 78% success rate",
                "Slow exits: 48% success rate",
                "Analyzed 50 trades",
            ],
        )

        assert timing_result.is_success()
        insight = timing_result.success()

        assert insight.insight_type == "timing_pattern"
        assert "30min" in insight.description
        assert insight.confidence == Decimal("0.7")

    def test_market_condition_insight_generation(self):
        """Test generation of market condition insights."""
        market_result = LearningInsight.create(
            insight_type="market_condition",
            description="High volume confirmation increases pattern reliability by 25%",
            confidence=0.8,
            supporting_evidence=[
                "High volume + pattern: 82% success",
                "Pattern only: 57% success",
                "Volume threshold: 1.5x average",
            ],
            related_patterns=[
                PatternTag.create("high_volume").success(),
                PatternTag.create("oversold_rsi").success(),
            ],
        )

        assert market_result.is_success()
        insight = market_result.success()

        assert insight.insight_type == "market_condition"
        assert "25%" in insight.description
        assert len(insight.related_patterns) == 2

    def test_risk_insight_generation(self):
        """Test generation of risk management insights."""
        risk_result = LearningInsight.create(
            insight_type="risk_management",
            description="Tight stop losses (1%) reduce drawdown but lower win rate",
            confidence=0.6,
            supporting_evidence=[
                "1% stops: 45% win rate, -2% max drawdown",
                "2% stops: 65% win rate, -5% max drawdown",
                "Trade-off between risk and reward",
            ],
        )

        assert risk_result.is_success()
        insight = risk_result.success()

        assert insight.insight_type == "risk_management"
        assert "stop losses" in insight.description
        assert insight.confidence == Decimal("0.6")

    def test_insight_confidence_levels(self):
        """Test different confidence levels for insights."""
        confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

        for confidence in confidence_levels:
            result = LearningInsight.create(
                insight_type="confidence_test",
                description=f"Test insight with {confidence} confidence",
                confidence=confidence,
            )

            assert result.is_success()
            insight = result.success()
            assert insight.confidence == Decimal(str(confidence))

    def test_insight_supporting_evidence_analysis(self):
        """Test analysis of supporting evidence quality."""
        strong_evidence = [
            "Large sample size: 100+ trades",
            "Multiple market conditions tested",
            "Statistical significance: p < 0.05",
            "Consistent results across timeframes",
            "Validated on out-of-sample data",
        ]

        weak_evidence = [
            "Small sample: 5 trades",
            "Single market condition",
        ]

        # Strong evidence should support higher confidence
        strong_result = LearningInsight.create(
            insight_type="evidence_test",
            description="Well-supported insight",
            confidence=0.9,
            supporting_evidence=strong_evidence,
        )

        weak_result = LearningInsight.create(
            insight_type="evidence_test",
            description="Weakly-supported insight",
            confidence=0.4,
            supporting_evidence=weak_evidence,
        )

        assert strong_result.is_success()
        assert weak_result.is_success()

        strong_insight = strong_result.success()
        weak_insight = weak_result.success()

        assert len(strong_insight.supporting_evidence) > len(
            weak_insight.supporting_evidence
        )
        assert strong_insight.confidence > weak_insight.confidence


# Pattern Combination Tests


class TestPatternCombinations:
    """Test analysis of pattern combinations and interactions."""

    def test_dual_pattern_combination(self):
        """Test analysis of two-pattern combinations."""
        pattern1 = PatternTag.create("oversold_rsi").success()
        pattern2 = PatternTag.create("high_volume").success()

        # Create experiences with different pattern combinations
        single_pattern_experiences = []
        dual_pattern_experiences = []

        # Single pattern: 60% success rate
        for i in range(10):
            pnl = Decimal(80) if i < 6 else Decimal(-40)
            exp_id = ExperienceId.create(f"single_{i}").success()
            exp = TradingExperienceFP(
                experience_id=exp_id,
                timestamp=datetime.utcnow(),
                market_snapshot=MagicMock(),
                trade_decision="LONG",
                decision_rationale="Single pattern test",
                pattern_tags=[pattern1],
                outcome=Some(
                    TradingOutcome.create(
                        pnl=pnl,
                        exit_price=Decimal(50000) + pnl,
                        entry_price=Decimal(50000),
                        duration_minutes=30.0,
                    ).success()
                ),
                learned_insights=Nothing(),
                confidence_score=Decimal("0.5"),
            )
            single_pattern_experiences.append(exp)

        # Dual pattern: 80% success rate (synergy effect)
        for i in range(10):
            pnl = Decimal(100) if i < 8 else Decimal(-50)
            exp_id = ExperienceId.create(f"dual_{i}").success()
            exp = TradingExperienceFP(
                experience_id=exp_id,
                timestamp=datetime.utcnow(),
                market_snapshot=MagicMock(),
                trade_decision="LONG",
                decision_rationale="Dual pattern test",
                pattern_tags=[pattern1, pattern2],
                outcome=Some(
                    TradingOutcome.create(
                        pnl=pnl,
                        exit_price=Decimal(50000) + pnl,
                        entry_price=Decimal(50000),
                        duration_minutes=30.0,
                    ).success()
                ),
                learned_insights=Nothing(),
                confidence_score=Decimal("0.5"),
            )
            dual_pattern_experiences.append(exp)

        # Calculate statistics for pattern1 in both scenarios
        single_stats = PatternStatistics.calculate(
            pattern1, single_pattern_experiences
        ).success()
        dual_stats = PatternStatistics.calculate(
            pattern1, dual_pattern_experiences
        ).success()

        # Dual pattern should show better performance
        assert single_stats.success_rate == Decimal("0.6")
        assert dual_stats.success_rate == Decimal("0.8")
        assert dual_stats.average_pnl > single_stats.average_pnl

    def test_pattern_conflict_detection(self):
        """Test detection of conflicting patterns."""
        # These patterns might conflict
        oversold_pattern = PatternTag.create("oversold_rsi").success()
        overbought_pattern = PatternTag.create("overbought_rsi").success()

        # Create experience with conflicting patterns (should be rare)
        conflicting_exp_id = ExperienceId.create("conflict_test").success()
        conflicting_exp = TradingExperienceFP(
            experience_id=conflicting_exp_id,
            timestamp=datetime.utcnow(),
            market_snapshot=MagicMock(),
            trade_decision="HOLD",  # Conflicting signals
            decision_rationale="Conflicting pattern signals",
            pattern_tags=[oversold_pattern, overbought_pattern],
            outcome=Some(
                TradingOutcome.create(
                    pnl=Decimal(-25),  # Poor performance expected
                    exit_price=Decimal(49975),
                    entry_price=Decimal(50000),
                    duration_minutes=10.0,  # Quick exit due to confusion
                ).success()
            ),
            learned_insights=Some("Conflicting signals led to poor decision"),
            confidence_score=Decimal("0.2"),  # Low confidence
        )

        # Analysis should detect the conflict
        patterns = conflicting_exp.pattern_tags
        has_oversold = any(p.name == "oversold_rsi" for p in patterns)
        has_overbought = any(p.name == "overbought_rsi" for p in patterns)
        is_conflicting = has_oversold and has_overbought

        assert is_conflicting is True
        assert conflicting_exp.confidence_score < Decimal("0.5")
        assert conflicting_exp.learned_insights.is_some()
        assert "conflicting" in conflicting_exp.learned_insights.value.lower()

    def test_pattern_hierarchy_analysis(self):
        """Test analysis of pattern importance hierarchy."""
        # Create patterns with different importance levels
        primary_patterns = [
            PatternTag.create("strong_uptrend").success(),
            PatternTag.create("oversold_rsi").success(),
        ]

        secondary_patterns = [
            PatternTag.create("high_volume").success(),
            PatternTag.create("bullish_divergence").success(),
        ]

        # Create experiences showing primary patterns are more important
        experiences = []

        # Experiences with primary patterns: high success
        for i in range(15):
            pnl = Decimal(120) if i < 12 else Decimal(-60)
            exp_id = ExperienceId.create(f"primary_{i}").success()
            exp = TradingExperienceFP(
                experience_id=exp_id,
                timestamp=datetime.utcnow(),
                market_snapshot=MagicMock(),
                trade_decision="LONG",
                decision_rationale="Primary pattern trade",
                pattern_tags=primary_patterns,
                outcome=Some(
                    TradingOutcome.create(
                        pnl=pnl,
                        exit_price=Decimal(50000) + pnl,
                        entry_price=Decimal(50000),
                        duration_minutes=30.0,
                    ).success()
                ),
                learned_insights=Nothing(),
                confidence_score=Decimal("0.8"),
            )
            experiences.append(exp)

        # Experiences with only secondary patterns: moderate success
        for i in range(10):
            pnl = Decimal(60) if i < 6 else Decimal(-40)
            exp_id = ExperienceId.create(f"secondary_{i}").success()
            exp = TradingExperienceFP(
                experience_id=exp_id,
                timestamp=datetime.utcnow(),
                market_snapshot=MagicMock(),
                trade_decision="LONG",
                decision_rationale="Secondary pattern trade",
                pattern_tags=secondary_patterns,
                outcome=Some(
                    TradingOutcome.create(
                        pnl=pnl,
                        exit_price=Decimal(50000) + pnl,
                        entry_price=Decimal(50000),
                        duration_minutes=30.0,
                    ).success()
                ),
                learned_insights=Nothing(),
                confidence_score=Decimal("0.6"),
            )
            experiences.append(exp)

        # Calculate statistics for different pattern types
        primary_stats = []
        for pattern in primary_patterns:
            stats = PatternStatistics.calculate(pattern, experiences).success()
            primary_stats.append(stats)

        secondary_stats = []
        for pattern in secondary_patterns:
            stats = PatternStatistics.calculate(pattern, experiences).success()
            secondary_stats.append(stats)

        # Primary patterns should show better performance
        avg_primary_success = sum(s.success_rate for s in primary_stats) / len(
            primary_stats
        )
        avg_secondary_success = sum(s.success_rate for s in secondary_stats) / len(
            secondary_stats
        )

        assert avg_primary_success > avg_secondary_success

        # Generate hierarchy insight
        hierarchy_insight = LearningInsight.create(
            insight_type="pattern_hierarchy",
            description=f"Primary patterns show {float(avg_primary_success):.1%} vs {float(avg_secondary_success):.1%} success rate",
            confidence=0.8,
            supporting_evidence=[
                f"Primary patterns: {len(primary_patterns)} analyzed",
                f"Secondary patterns: {len(secondary_patterns)} analyzed",
                f"Performance gap: {float(avg_primary_success - avg_secondary_success):.1%}",
            ],
            related_patterns=primary_patterns + secondary_patterns,
        )

        assert hierarchy_insight.is_success()
        insight = hierarchy_insight.success()
        assert "primary" in insight.description.lower()
        assert len(insight.related_patterns) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
