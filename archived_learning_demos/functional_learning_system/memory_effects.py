"""
Memory effects for functional MCP integration.

This module provides effects for interacting with the MCP memory server
in a purely functional way, using the existing effects system.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from bot.fp.effects.io import AsyncIO
from bot.fp.types.result import Err, Ok, Result
from bot.mcp.memory_server import MCPMemoryServer
from bot.mcp.memory_server import TradingExperience as LegacyExperience
from bot.trading_types import MarketState, TradeAction

from .experience import TradeExperience


@dataclass(frozen=True)
class MemoryEffect:
    """Base class for memory effects."""

    operation: str
    params: dict[str, Any]


@dataclass(frozen=True)
class StoreExperienceEffect(MemoryEffect):
    """Effect for storing a trading experience."""

    experience: TradeExperience

    def __init__(self, experience: TradeExperience):
        super().__init__(
            "store_experience", {"experience_id": experience.experience_id}
        )
        object.__setattr__(self, "experience", experience)


@dataclass(frozen=True)
class RetrieveExperienceEffect(MemoryEffect):
    """Effect for retrieving a specific experience."""

    experience_id: str

    def __init__(self, experience_id: str):
        super().__init__("retrieve_experience", {"experience_id": experience_id})
        object.__setattr__(self, "experience_id", experience_id)


@dataclass(frozen=True)
class QueryExperiencesEffect(MemoryEffect):
    """Effect for querying similar experiences."""

    indicators: dict[str, float]
    similarity_threshold: float
    max_results: int

    def __init__(
        self,
        indicators: dict[str, float],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
    ):
        super().__init__(
            "query_experiences",
            {
                "indicators": indicators,
                "similarity_threshold": similarity_threshold,
                "max_results": max_results,
            },
        )
        object.__setattr__(self, "indicators", indicators)
        object.__setattr__(self, "similarity_threshold", similarity_threshold)
        object.__setattr__(self, "max_results", max_results)


@dataclass(frozen=True)
class UpdateExperienceEffect(MemoryEffect):
    """Effect for updating an experience with outcome."""

    experience_id: str
    updated_experience: TradeExperience

    def __init__(self, experience_id: str, updated_experience: TradeExperience):
        super().__init__("update_experience", {"experience_id": experience_id})
        object.__setattr__(self, "experience_id", experience_id)
        object.__setattr__(self, "updated_experience", updated_experience)


@dataclass(frozen=True)
class CleanupOldExperiencesEffect(MemoryEffect):
    """Effect for cleaning up old experiences."""

    retention_days: int

    def __init__(self, retention_days: int):
        super().__init__("cleanup_old_experiences", {"retention_days": retention_days})
        object.__setattr__(self, "retention_days", retention_days)


# Effect constructors - these are pure functions that create effect descriptions


def store_experience(experience: TradeExperience) -> AsyncIO[Result[str, str]]:
    """Create an effect to store a trading experience."""

    async def store_operation():
        try:
            # Convert functional experience to legacy format for MCP compatibility
            legacy_experience = _convert_to_legacy_experience(experience)

            # This would be injected with the actual memory server
            # For now, we'll simulate the operation
            experience_id = experience.experience_id

            return Ok(experience_id)
        except Exception as e:
            return Err(f"Failed to store experience: {e!s}")

    return AsyncIO(lambda: asyncio.create_task(store_operation()))


def retrieve_experience(
    experience_id: str,
) -> AsyncIO[Result[TradeExperience | None, str]]:
    """Create an effect to retrieve a specific experience."""

    async def retrieve_operation():
        try:
            # This would interact with the actual memory server
            # For now, we'll return None to indicate not found
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to retrieve experience: {e!s}")

    return AsyncIO(lambda: asyncio.create_task(retrieve_operation()))


def query_experiences(
    indicators: dict[str, float],
    similarity_threshold: float = 0.7,
    max_results: int = 10,
) -> AsyncIO[Result[tuple[TradeExperience, ...], str]]:
    """Create an effect to query similar experiences."""

    async def query_operation():
        try:
            # This would interact with the actual memory server
            # For now, we'll return empty results
            return Ok(())
        except Exception as e:
            return Err(f"Failed to query experiences: {e!s}")

    return AsyncIO(lambda: asyncio.create_task(query_operation()))


def update_experience(
    experience_id: str, updated_experience: TradeExperience
) -> AsyncIO[Result[bool, str]]:
    """Create an effect to update an experience."""

    async def update_operation():
        try:
            # This would interact with the actual memory server
            # Convert to legacy format and update
            return Ok(True)
        except Exception as e:
            return Err(f"Failed to update experience: {e!s}")

    return AsyncIO(lambda: asyncio.create_task(update_operation()))


def cleanup_old_experiences(retention_days: int) -> AsyncIO[Result[int, str]]:
    """Create an effect to cleanup old experiences."""

    async def cleanup_operation():
        try:
            # This would interact with the actual memory server
            cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)
            # Count would be returned from actual cleanup
            removed_count = 0
            return Ok(removed_count)
        except Exception as e:
            return Err(f"Failed to cleanup experiences: {e!s}")

    return AsyncIO(lambda: asyncio.create_task(cleanup_operation()))


# Effect interpreter - this handles the actual MCP server interaction


class MemoryEffectInterpreter:
    """Interpreter for memory effects that handles MCP server interaction."""

    def __init__(self, memory_server: MCPMemoryServer):
        self.memory_server = memory_server

    async def interpret_store_experience(
        self, experience: TradeExperience
    ) -> Result[str, str]:
        """Interpret store experience effect."""
        try:
            # Convert functional experience to MCP format
            market_state = self._convert_to_market_state(experience)
            trade_action = self._convert_to_trade_action(experience)

            experience_id = await self.memory_server.store_experience(
                market_state, trade_action
            )
            return Ok(experience_id)
        except Exception as e:
            return Err(f"Failed to store experience: {e!s}")

    async def interpret_retrieve_experience(
        self, experience_id: str
    ) -> Result[TradeExperience | None, str]:
        """Interpret retrieve experience effect."""
        try:
            legacy_exp = self.memory_server.memory_cache.get(experience_id)
            if legacy_exp:
                functional_exp = self._convert_from_legacy_experience(legacy_exp)
                return Ok(functional_exp)
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to retrieve experience: {e!s}")

    async def interpret_query_experiences(
        self,
        indicators: dict[str, float],
        similarity_threshold: float,
        max_results: int,
    ) -> Result[tuple[TradeExperience, ...], str]:
        """Interpret query experiences effect."""
        try:
            # Create a dummy market state for querying
            from decimal import Decimal

            from bot.trading_types import (
                IndicatorData,
                MarketState,
                Position,
            )

            market_state = MarketState(
                symbol="BTC-USD",
                interval="1h",
                timestamp=datetime.now(UTC),
                current_price=Decimal(50000),
                ohlcv_data=[],
                indicators=IndicatorData(timestamp=datetime.now(UTC), **indicators),
                current_position=Position(
                    symbol="BTC-USD",
                    side="FLAT",
                    size=Decimal(0),
                    entry_price=None,
                    timestamp=datetime.now(UTC),
                ),
            )

            from bot.mcp.memory_server import MemoryQuery

            query_params = MemoryQuery(
                indicators=indicators,
                min_similarity=similarity_threshold,
                max_results=max_results,
            )

            legacy_experiences = await self.memory_server.query_similar_experiences(
                market_state, query_params
            )

            functional_experiences = tuple(
                self._convert_from_legacy_experience(exp) for exp in legacy_experiences
            )

            return Ok(functional_experiences)
        except Exception as e:
            return Err(f"Failed to query experiences: {e!s}")

    async def interpret_update_experience(
        self,
        experience_id: str,
        updated_experience: TradeExperience,
    ) -> Result[bool, str]:
        """Interpret update experience effect."""
        try:
            if not updated_experience.outcome:
                return Err("Cannot update experience without outcome data")

            outcome = updated_experience.outcome
            success = await self.memory_server.update_experience_outcome(
                experience_id=experience_id,
                pnl=Decimal(str(outcome["pnl"])),
                exit_price=Decimal(str(outcome["exit_price"])),
                duration_minutes=outcome["duration_minutes"],
                market_data_at_exit=None,  # Could be reconstructed if needed
            )

            return Ok(success)
        except Exception as e:
            return Err(f"Failed to update experience: {e!s}")

    async def interpret_cleanup_old_experiences(
        self, retention_days: int
    ) -> Result[int, str]:
        """Interpret cleanup experiences effect."""
        try:
            removed_count = await self.memory_server.cleanup_old_memories(
                retention_days
            )
            return Ok(removed_count)
        except Exception as e:
            return Err(f"Failed to cleanup experiences: {e!s}")

    def _convert_to_market_state(self, experience: TradeExperience) -> MarketState:
        """Convert functional experience to MarketState for MCP compatibility."""
        from decimal import Decimal

        from bot.trading_types import IndicatorData, MarketState, Position

        # Reconstruct market state from experience snapshot
        snapshot = experience.market_snapshot

        return MarketState(
            symbol=experience.symbol,
            interval=snapshot.get("interval", "1h"),
            timestamp=experience.timestamp,
            current_price=experience.price,
            ohlcv_data=[],  # Not stored in functional experience
            indicators=IndicatorData(
                timestamp=experience.timestamp,
                **experience.indicators,
            ),
            current_position=Position(
                symbol=experience.symbol,
                side=snapshot.get("position_side", "FLAT"),
                size=Decimal(str(snapshot.get("position_size", "0"))),
                entry_price=None,
                timestamp=experience.timestamp,
            ),
            dominance_data=experience.dominance_data,
        )

    def _convert_to_trade_action(self, experience: TradeExperience) -> TradeAction:
        """Convert functional experience to TradeAction for MCP compatibility."""
        from typing import Literal, cast

        from bot.trading_types import TradeAction

        decision = experience.decision
        action_value = decision.get("action", "HOLD")

        # Ensure action is valid
        if action_value not in ["LONG", "SHORT", "CLOSE", "HOLD"]:
            action_value = "HOLD"

        return TradeAction(
            action=cast("Literal['LONG', 'SHORT', 'CLOSE', 'HOLD']", action_value),
            size_pct=decision.get("size_pct", 0.0),
            take_profit_pct=decision.get("take_profit_pct", 0.0),
            stop_loss_pct=decision.get("stop_loss_pct", 0.0),
            leverage=decision.get("leverage", 1),
            reduce_only=decision.get("reduce_only", False),
            rationale=experience.rationale,
        )

    def _convert_from_legacy_experience(
        self, legacy_exp: LegacyExperience
    ) -> TradeExperience:
        """Convert legacy MCP experience to functional experience."""
        # Convert outcome if it exists
        outcome = None
        if legacy_exp.outcome:
            outcome = {
                "pnl": legacy_exp.outcome.get("pnl", 0.0),
                "exit_price": legacy_exp.outcome.get("exit_price", 0.0),
                "price_change_pct": legacy_exp.outcome.get("price_change_pct", 0.0),
                "success": legacy_exp.outcome.get("success", False),
                "duration_minutes": legacy_exp.outcome.get("duration_minutes", 0.0),
            }

        return TradeExperience(
            experience_id=legacy_exp.experience_id,
            timestamp=legacy_exp.timestamp,
            symbol=legacy_exp.symbol,
            price=legacy_exp.price,
            market_snapshot=legacy_exp.market_state_snapshot,
            indicators=legacy_exp.indicators,
            dominance_data=legacy_exp.dominance_data,
            decision=legacy_exp.decision,
            rationale=legacy_exp.decision_rationale,
            outcome=outcome,
            duration_minutes=legacy_exp.trade_duration_minutes,
            market_reaction=legacy_exp.market_reaction,
            pattern_tags=tuple(legacy_exp.pattern_tags),
            confidence_score=legacy_exp.confidence_score,
            learned_insights=legacy_exp.learned_insights,
        )


def _convert_to_legacy_experience(experience: TradeExperience) -> LegacyExperience:
    """Convert functional experience to legacy MCP format."""
    # This would be used when we need to interact with the legacy MCP server
