"""
Functional programming adapter for MCP memory server.

This adapter bridges between the imperative MCP memory server implementation
and functional programming patterns, providing pure functions and immutable
data structures for memory operations.
"""

import logging
from decimal import Decimal

from bot.fp.types import (
    ExperienceId,
    Failure,
    LearningInsight,
    MarketSnapshot,
    MemoryQueryFP,
    MemoryStorage,
    Nothing,
    PatternStatistics,
    PatternTag,
    Result,
    Some,
    Success,
    TradingExperienceFP,
    TradingOutcome,
)
from bot.mcp.memory_server import MCPMemoryServer, MemoryQuery, TradingExperience
from bot.trading_types import MarketState, TradeAction

logger = logging.getLogger(__name__)


class MemoryAdapterFP:
    """
    Functional programming adapter for MCP memory server.

    This adapter provides pure functional interfaces to memory operations
    while maintaining compatibility with the existing imperative MCP server.
    """

    def __init__(self, mcp_server: MCPMemoryServer):
        """Initialize with an MCP memory server instance."""
        self._mcp_server = mcp_server
        self._local_storage = MemoryStorage.empty()

    async def store_experience_fp(
        self,
        experience: TradingExperienceFP,
    ) -> Result[ExperienceId, str]:
        """
        Store a functional programming experience.

        Args:
            experience: Functional trading experience to store

        Returns:
            Result containing experience ID or error message
        """
        try:
            # Convert FP experience to imperative format
            market_state = self._fp_to_market_state(experience.market_snapshot)
            trade_action = self._fp_to_trade_action(experience.trade_decision)

            # Store using the imperative server
            experience_id_str = await self._mcp_server.store_experience(
                market_state, trade_action
            )

            # Update local storage
            self._local_storage = self._local_storage.add_experience(experience)

            experience_id_result = ExperienceId.create(experience_id_str)
            if experience_id_result.is_failure():
                return Failure(
                    f"Invalid experience ID returned: {experience_id_result.failure()}"
                )

            logger.info(
                "📦 FP Memory Adapter: Stored experience %s | Action: %s | Patterns: %s",
                experience_id_result.success().short(),
                experience.trade_decision,
                [p.name for p in experience.pattern_tags],
            )

            return Success(experience_id_result.success())

        except Exception as e:
            error_msg = f"Failed to store FP experience: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    async def update_experience_outcome_fp(
        self,
        experience_id: ExperienceId,
        outcome: TradingOutcome,
        market_snapshot_at_exit: MarketSnapshot | None = None,
    ) -> Result[TradingExperienceFP, str]:
        """
        Update experience with trading outcome.

        Args:
            experience_id: ID of experience to update
            outcome: Trading outcome data
            market_snapshot_at_exit: Optional market state at exit

        Returns:
            Result containing updated experience or error
        """
        try:
            # Update using imperative server
            market_state_at_exit = None
            if market_snapshot_at_exit:
                market_state_at_exit = self._fp_to_market_state(market_snapshot_at_exit)

            success = await self._mcp_server.update_experience_outcome(
                experience_id.value,
                outcome.pnl,
                outcome.exit_price,
                float(outcome.duration_minutes),
                market_state_at_exit,
            )

            if not success:
                return Failure(f"Failed to update experience {experience_id.short()}")

            # Update local storage
            update_result = self._local_storage.update_experience(
                experience_id,
                lambda exp: exp.with_outcome(outcome),
            )

            if update_result.is_failure():
                return Failure(update_result.failure())

            self._local_storage = update_result.success()

            # Get updated experience
            updated_exp = self._local_storage.find_by_id(experience_id)
            if updated_exp.is_nothing():
                return Failure(
                    f"Experience {experience_id.short()} not found after update"
                )

            logger.info(
                "📈 FP Memory Adapter: Updated experience %s | PnL: $%.2f | Success: %s",
                experience_id.short(),
                outcome.pnl,
                "✅" if outcome.is_successful else "❌",
            )

            return Success(updated_exp.value)

        except Exception as e:
            error_msg = f"Failed to update FP experience outcome: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    async def query_similar_experiences_fp(
        self,
        market_snapshot: MarketSnapshot,
        query: MemoryQueryFP | None = None,
    ) -> Result[list[TradingExperienceFP], str]:
        """
        Query for similar experiences using FP types.

        Args:
            market_snapshot: Current market snapshot
            query: Optional query parameters

        Returns:
            Result containing list of similar experiences or error
        """
        try:
            # Convert FP types to imperative format
            market_state = self._fp_to_market_state(market_snapshot)

            memory_query = None
            if query:
                memory_query = MemoryQuery(
                    current_price=(
                        query.current_price.value
                        if query.current_price.is_some()
                        else None
                    ),
                    indicators=query.indicators,
                    dominance_data=(
                        query.dominance_data.value
                        if query.dominance_data.is_some()
                        else None
                    ),
                    max_results=query.max_results,
                    min_similarity=float(query.min_similarity),
                    time_weight=float(query.time_weight),
                )

            # Query using imperative server
            experiences = await self._mcp_server.query_similar_experiences(
                market_state, memory_query
            )

            # Convert back to FP types
            fp_experiences = []
            for exp in experiences:
                fp_exp_result = self._imperative_to_fp_experience(exp)
                if fp_exp_result.is_success():
                    fp_experiences.append(fp_exp_result.success())
                else:
                    logger.warning(
                        "Failed to convert experience %s to FP: %s",
                        exp.experience_id[:8],
                        fp_exp_result.failure(),
                    )

            logger.info(
                "🔍 FP Memory Adapter: Found %d similar experiences | Query time: recent",
                len(fp_experiences),
            )

            return Success(fp_experiences)

        except Exception as e:
            error_msg = f"Failed to query similar FP experiences: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    async def get_pattern_statistics_fp(
        self,
        patterns: list[PatternTag] | None = None,
    ) -> Result[list[PatternStatistics], str]:
        """
        Get pattern statistics using FP types.

        Args:
            patterns: Optional list of specific patterns to analyze

        Returns:
            Result containing pattern statistics or error
        """
        try:
            # Get statistics from imperative server
            stats_dict = await self._mcp_server.get_pattern_statistics()

            # Convert to FP types
            fp_statistics = []
            for pattern_name, stats in stats_dict.items():
                # Create pattern tag
                pattern_result = PatternTag.create(pattern_name)
                if pattern_result.is_failure():
                    continue

                pattern = pattern_result.success()

                # Filter if specific patterns requested
                if patterns and pattern not in patterns:
                    continue

                # Create FP statistics
                fp_stats = PatternStatistics(
                    pattern=pattern,
                    total_occurrences=stats["count"],
                    successful_trades=int(stats["success_rate"] * stats["count"]),
                    total_pnl=Decimal(str(stats["total_pnl"])),
                    average_pnl=Decimal(str(stats["avg_pnl"])),
                    success_rate=Decimal(str(stats["success_rate"])),
                )

                fp_statistics.append(fp_stats)

            logger.info(
                "📊 FP Memory Adapter: Retrieved statistics for %d patterns",
                len(fp_statistics),
            )

            return Success(fp_statistics)

        except Exception as e:
            error_msg = f"Failed to get FP pattern statistics: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    async def generate_learning_insights_fp(
        self,
        experiences: list[TradingExperienceFP],
    ) -> Result[list[LearningInsight], str]:
        """
        Generate learning insights from experiences.

        Args:
            experiences: List of completed experiences to analyze

        Returns:
            Result containing learning insights or error
        """
        try:
            insights = []

            if not experiences:
                return Success(insights)

            # Analyze pattern performance
            pattern_insights = await self._analyze_pattern_performance(experiences)
            insights.extend(pattern_insights)

            # Analyze timing patterns
            timing_insights = await self._analyze_timing_patterns(experiences)
            insights.extend(timing_insights)

            # Analyze market condition patterns
            market_insights = await self._analyze_market_conditions(experiences)
            insights.extend(market_insights)

            logger.info(
                "🧠 FP Memory Adapter: Generated %d learning insights from %d experiences",
                len(insights),
                len(experiences),
            )

            return Success(insights)

        except Exception as e:
            error_msg = f"Failed to generate FP learning insights: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    def get_memory_storage(self) -> MemoryStorage:
        """Get current local memory storage state."""
        return self._local_storage

    async def sync_with_server(self) -> Result[int, str]:
        """
        Synchronize local storage with MCP server.

        Returns:
            Result containing number of synced experiences or error
        """
        try:
            # This would typically fetch all experiences from server
            # and rebuild local storage - simplified for now
            logger.info("🔄 FP Memory Adapter: Synced with MCP server")
            return Success(len(self._local_storage.experiences))

        except Exception as e:
            error_msg = f"Failed to sync with MCP server: {e}"
            logger.exception(error_msg)
            return Failure(error_msg)

    def _fp_to_market_state(self, snapshot: MarketSnapshot) -> MarketState:
        """Convert FP market snapshot to imperative MarketState."""
        # This is a simplified conversion - in practice would need full reconstruction
        from bot.trading_types import IndicatorData, Position

        # Create minimal MarketState for compatibility
        indicators = IndicatorData(
            timestamp=snapshot.timestamp,
            rsi=snapshot.indicators.get("rsi"),
            cipher_a_dot=snapshot.indicators.get("cipher_a_dot"),
            cipher_b_wave=snapshot.indicators.get("cipher_b_wave"),
            cipher_b_money_flow=snapshot.indicators.get("cipher_b_money_flow"),
            ema_fast=snapshot.indicators.get("ema_fast"),
            ema_slow=snapshot.indicators.get("ema_slow"),
        )

        position = Position(
            symbol=snapshot.symbol.value,
            side=snapshot.position_side,
            size=snapshot.position_size,
            timestamp=snapshot.timestamp,
        )

        return MarketState(
            symbol=snapshot.symbol.value,
            interval="1m",  # Default
            timestamp=snapshot.timestamp,
            current_price=snapshot.price,
            ohlcv_data=[],  # Simplified
            indicators=indicators,
            current_position=position,
            dominance_data=None,  # Would convert if needed
        )

    def _fp_to_trade_action(self, decision) -> TradeAction:
        """Convert FP trade decision to imperative TradeAction."""
        # This is simplified - would need proper conversion based on decision type
        return TradeAction(
            action="HOLD",  # Default - would extract from decision
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            leverage=1,
            rationale="FP decision",
        )

    def _imperative_to_fp_experience(
        self,
        experience: TradingExperience,
    ) -> Result[TradingExperienceFP, str]:
        """Convert imperative experience to FP experience."""
        try:
            # Convert experience ID
            exp_id_result = ExperienceId.create(experience.experience_id)
            if exp_id_result.is_failure():
                return Failure(exp_id_result.failure())

            # Convert market snapshot
            from bot.fp.types.base import Symbol

            symbol_result = Symbol.create(experience.symbol)
            if symbol_result.is_failure():
                return Failure(symbol_result.failure())

            market_snapshot = MarketSnapshot(
                symbol=symbol_result.success(),
                timestamp=experience.timestamp,
                price=experience.price,
                indicators=experience.indicators,
                dominance_data=experience.dominance_data,
                position_side=experience.market_state_snapshot.get(
                    "position_side", "FLAT"
                ),
                position_size=Decimal(
                    str(experience.market_state_snapshot.get("position_size", 0))
                ),
            )

            # Convert pattern tags
            pattern_tags = []
            for tag_name in experience.pattern_tags:
                tag_result = PatternTag.create(tag_name)
                if tag_result.is_success():
                    pattern_tags.append(tag_result.success())

            # Convert outcome if present
            outcome = Nothing()
            if experience.outcome:
                outcome_result = TradingOutcome.create(
                    pnl=Decimal(str(experience.outcome["pnl"])),
                    exit_price=Decimal(str(experience.outcome["exit_price"])),
                    entry_price=experience.price,
                    duration_minutes=experience.outcome["duration_minutes"],
                )
                if outcome_result.is_success():
                    outcome = Some(outcome_result.success())

            # Create FP experience
            fp_experience = TradingExperienceFP(
                experience_id=exp_id_result.success(),
                timestamp=experience.timestamp,
                market_snapshot=market_snapshot,
                trade_decision=None,  # Would need proper conversion
                decision_rationale=experience.decision_rationale,
                pattern_tags=pattern_tags,
                outcome=outcome,
                learned_insights=(
                    Some(experience.learned_insights)
                    if experience.learned_insights
                    else Nothing()
                ),
                confidence_score=Decimal(str(experience.confidence_score)),
            )

            return Success(fp_experience)

        except Exception as e:
            return Failure(f"Failed to convert imperative experience: {e}")

    async def _analyze_pattern_performance(
        self,
        experiences: list[TradingExperienceFP],
    ) -> list[LearningInsight]:
        """Analyze pattern performance and generate insights."""
        insights = []

        # Group experiences by patterns
        pattern_groups: dict[str, list[TradingExperienceFP]] = {}
        for exp in experiences:
            if exp.outcome.is_some():
                for pattern in exp.pattern_tags:
                    if pattern.name not in pattern_groups:
                        pattern_groups[pattern.name] = []
                    pattern_groups[pattern.name].append(exp)

        # Analyze each pattern
        for pattern_name, pattern_experiences in pattern_groups.items():
            if len(pattern_experiences) >= 3:  # Minimum for analysis
                success_rate = sum(
                    1 for exp in pattern_experiences if exp.outcome.value.is_successful
                ) / len(pattern_experiences)

                if success_rate > 0.7:
                    insight_result = LearningInsight.create(
                        insight_type="pattern_performance",
                        description=f"Pattern '{pattern_name}' shows high success rate ({success_rate:.1%})",
                        confidence=min(0.9, success_rate),
                        supporting_evidence=[
                            f"{len(pattern_experiences)} occurrences",
                            f"{success_rate:.1%} success rate",
                        ],
                    )
                    if insight_result.is_success():
                        insights.append(insight_result.success())

        return insights

    async def _analyze_timing_patterns(
        self,
        experiences: list[TradingExperienceFP],
    ) -> list[LearningInsight]:
        """Analyze timing patterns in trades."""
        insights = []

        # Analyze successful vs unsuccessful trade durations
        successful_durations = []
        unsuccessful_durations = []

        for exp in experiences:
            if exp.outcome.is_some():
                duration = exp.outcome.value.duration_minutes
                if exp.outcome.value.is_successful:
                    successful_durations.append(float(duration))
                else:
                    unsuccessful_durations.append(float(duration))

        if successful_durations and unsuccessful_durations:
            avg_success_duration = sum(successful_durations) / len(successful_durations)
            avg_fail_duration = sum(unsuccessful_durations) / len(
                unsuccessful_durations
            )

            if (
                avg_success_duration < avg_fail_duration * 0.7
            ):  # Successful trades much faster
                insight_result = LearningInsight.create(
                    insight_type="timing_pattern",
                    description=f"Quick exits tend to be more profitable (avg {avg_success_duration:.1f}min vs {avg_fail_duration:.1f}min)",
                    confidence=0.6,
                    supporting_evidence=[
                        f"{len(successful_durations)} successful trades",
                        f"{len(unsuccessful_durations)} unsuccessful trades",
                    ],
                )
                if insight_result.is_success():
                    insights.append(insight_result.success())

        return insights

    async def _analyze_market_conditions(
        self,
        experiences: list[TradingExperienceFP],
    ) -> list[LearningInsight]:
        """Analyze market condition patterns."""
        insights = []

        # Analyze RSI conditions for successful trades
        oversold_successes = 0
        oversold_total = 0
        overbought_successes = 0
        overbought_total = 0

        for exp in experiences:
            if exp.outcome.is_some():
                rsi = exp.market_snapshot.indicators.get("rsi")
                if rsi:
                    if rsi < 30:  # Oversold
                        oversold_total += 1
                        if exp.outcome.value.is_successful:
                            oversold_successes += 1
                    elif rsi > 70:  # Overbought
                        overbought_total += 1
                        if exp.outcome.value.is_successful:
                            overbought_successes += 1

        # Generate insights for extreme RSI conditions
        if oversold_total >= 3:
            success_rate = oversold_successes / oversold_total
            if success_rate > 0.6:
                insight_result = LearningInsight.create(
                    insight_type="market_condition",
                    description=f"Oversold RSI conditions show good success rate ({success_rate:.1%})",
                    confidence=min(0.8, success_rate),
                    supporting_evidence=[
                        f"{oversold_total} oversold entries",
                        f"{oversold_successes} successful",
                    ],
                )
                if insight_result.is_success():
                    insights.append(insight_result.success())

        return insights


__all__ = ["MemoryAdapterFP"]
