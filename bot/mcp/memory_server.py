"""
MCP Memory Server for AI Trading Bot.

Provides persistent memory storage and retrieval for trading experiences,
enabling the LLM agent to learn from past trades and market conditions.
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
import numpy as np
from pydantic import BaseModel, Field

from ..config import settings
from ..types import MarketState, TradeAction

logger = logging.getLogger(__name__)


class TradingExperience(BaseModel):
    """A single trading experience stored in memory."""

    experience_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Market context at decision time
    symbol: str
    price: Decimal
    market_state_snapshot: Dict[str, Any]  # Serialized MarketState
    indicators: Dict[str, float]
    dominance_data: Optional[Dict[str, float]] = None

    # Trading decision
    decision: Dict[str, Any]  # Serialized TradeAction
    decision_rationale: str

    # Outcome tracking
    outcome: Optional[Dict[str, Any]] = None  # Filled after trade completion
    trade_duration_minutes: Optional[float] = None
    market_reaction: Optional[Dict[str, float]] = None

    # Learning insights
    learned_insights: Optional[str] = None
    pattern_tags: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)

    class Config:
        json_encoders = {Decimal: str, datetime: lambda v: v.isoformat()}


class MemoryQuery(BaseModel):
    """Query parameters for retrieving memories."""

    current_price: Optional[Decimal] = None
    indicators: Optional[Dict[str, float]] = None
    dominance_data: Optional[Dict[str, float]] = None
    market_sentiment: Optional[str] = None
    pattern_tags: Optional[List[str]] = None

    max_results: int = Field(default=10, ge=1, le=50)
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0)
    time_weight: float = Field(default=0.2, ge=0.0, le=1.0)  # Weight for recency


class MCPMemoryServer:
    """
    MCP-based memory server for trading bot learning.

    Stores trading experiences and provides intelligent retrieval
    based on market similarity and performance outcomes.
    """

    def __init__(self, server_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the MCP memory server connection."""
        self.server_url = server_url or settings.mcp.server_url
        self.api_key = api_key or (
            settings.mcp.memory_api_key.get_secret_value()
            if settings.mcp.memory_api_key
            else None
        )

        # Local cache for fast access
        self.memory_cache: Dict[str, TradingExperience] = {}
        self.pattern_index: Dict[str, List[str]] = {}  # pattern -> experience_ids

        # Session for async HTTP requests
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False

        # Local persistence
        self.local_storage_path = Path("data/mcp_memory")
        self.local_storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized MCP memory server client for {self.server_url}")

    async def connect(self) -> bool:
        """Connect to the MCP memory server."""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()

            # Test connection
            headers = (
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            async with self._session.get(
                f"{self.server_url}/health",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    self._connected = True
                    logger.info("Connected to MCP memory server")

                    # Load any cached memories
                    await self._load_local_cache()
                    return True
                else:
                    logger.warning(f"MCP server returned status {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to connect to MCP memory server: {e}")
            # Fall back to local-only mode
            await self._load_local_cache()
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP memory server."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

        # Save cache locally
        await self._save_local_cache()
        logger.info("Disconnected from MCP memory server")

    async def store_experience(
        self,
        market_state: MarketState,
        trade_action: TradeAction,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a new trading experience in memory.

        Args:
            market_state: Current market state
            trade_action: Trading decision made
            additional_context: Any additional context to store

        Returns:
            Experience ID for later updates
        """
        # Create experience object
        experience = TradingExperience(
            symbol=market_state.symbol,
            price=market_state.current_price,
            market_state_snapshot=self._serialize_market_state(market_state),
            indicators=self._extract_indicators(market_state),
            dominance_data=self._extract_dominance_data(market_state),
            decision=self._serialize_trade_action(trade_action),
            decision_rationale=trade_action.rationale,
        )

        # Add pattern tags based on market conditions
        experience.pattern_tags = self._identify_patterns(market_state, trade_action)

        # Store locally
        self.memory_cache[experience.experience_id] = experience

        # Update pattern index
        for pattern in experience.pattern_tags:
            if pattern not in self.pattern_index:
                self.pattern_index[pattern] = []
            self.pattern_index[pattern].append(experience.experience_id)

        # Store remotely if connected
        if self._connected and self._session:
            try:
                await self._store_remote(experience)
            except Exception as e:
                logger.error(f"Failed to store experience remotely: {e}")

        # Persist locally
        await self._save_experience_local(experience)

        logger.info(
            f"Stored trading experience {experience.experience_id} "
            f"for {trade_action.action} decision at ${market_state.current_price}"
        )

        return experience.experience_id

    async def update_experience_outcome(
        self,
        experience_id: str,
        pnl: Decimal,
        exit_price: Decimal,
        duration_minutes: float,
        market_data_at_exit: Optional[MarketState] = None,
    ) -> bool:
        """
        Update an experience with its outcome after trade completion.

        Args:
            experience_id: ID of the experience to update
            pnl: Profit/loss from the trade
            exit_price: Price at trade exit
            duration_minutes: How long the position was held
            market_data_at_exit: Market state at exit time

        Returns:
            True if update successful
        """
        if experience_id not in self.memory_cache:
            logger.warning(f"Experience {experience_id} not found in cache")
            return False

        experience = self.memory_cache[experience_id]

        # Calculate outcome metrics
        entry_price = experience.price
        price_change_pct = float((exit_price - entry_price) / entry_price * 100)

        experience.outcome = {
            "pnl": float(pnl),
            "exit_price": float(exit_price),
            "price_change_pct": price_change_pct,
            "success": pnl > 0,
            "duration_minutes": duration_minutes,
        }

        experience.trade_duration_minutes = duration_minutes

        # Calculate market reaction if exit data provided
        if market_data_at_exit:
            experience.market_reaction = {
                "price_change": float(exit_price - entry_price),
                "rsi_change": (
                    market_data_at_exit.indicators.rsi
                    - experience.indicators.get("rsi", 50)
                    if market_data_at_exit.indicators.rsi
                    else 0
                ),
                "volume_ratio": 1.0,  # Placeholder - would calculate from OHLCV data
            }

        # Generate learning insights
        experience.learned_insights = await self._generate_insights(experience)

        # Update confidence score based on outcome
        experience.confidence_score = self._calculate_confidence(experience)

        # Update remotely if connected
        if self._connected:
            await self._update_remote(experience)

        # Persist locally
        await self._save_experience_local(experience)

        logger.info(
            f"Updated experience {experience_id} with outcome: "
            f"PnL=${pnl}, Success={experience.outcome['success']}"
        )

        return True

    async def query_similar_experiences(
        self, market_state: MarketState, query_params: Optional[MemoryQuery] = None
    ) -> List[TradingExperience]:
        """
        Query for similar past trading experiences.

        Args:
            market_state: Current market state to compare against
            query_params: Additional query parameters

        Returns:
            List of similar experiences sorted by relevance
        """
        if not query_params:
            query_params = MemoryQuery()

        # Extract current features
        current_features = self._extract_features(market_state)

        # Score all experiences
        scored_experiences = []
        for exp_id, experience in self.memory_cache.items():
            # Skip experiences without outcomes
            if not experience.outcome:
                continue

            # Calculate similarity score
            similarity = self._calculate_similarity(
                current_features,
                self._extract_features_from_experience(experience),
                query_params.time_weight,
            )

            if similarity >= query_params.min_similarity:
                scored_experiences.append((similarity, experience))

        # Sort by similarity (descending)
        scored_experiences.sort(key=lambda x: x[0], reverse=True)

        # Return top results
        results = [exp for _, exp in scored_experiences[: query_params.max_results]]

        logger.info(
            f"Found {len(results)} similar experiences "
            f"(out of {len(self.memory_cache)} total)"
        )

        return results

    async def get_pattern_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about identified patterns and their success rates.

        Returns:
            Dictionary of pattern statistics
        """
        pattern_stats = {}

        for pattern, experience_ids in self.pattern_index.items():
            successes = 0
            total_pnl = 0.0
            count = 0

            for exp_id in experience_ids:
                if exp_id in self.memory_cache:
                    exp = self.memory_cache[exp_id]
                    if exp.outcome:
                        count += 1
                        if exp.outcome["success"]:
                            successes += 1
                        total_pnl += exp.outcome["pnl"]

            if count > 0:
                pattern_stats[pattern] = {
                    "count": count,
                    "success_rate": successes / count,
                    "avg_pnl": total_pnl / count,
                    "total_pnl": total_pnl,
                }

        return pattern_stats

    def _serialize_market_state(self, market_state: MarketState) -> Dict[str, Any]:
        """Serialize MarketState to dictionary."""
        return {
            "symbol": market_state.symbol,
            "interval": market_state.interval,
            "timestamp": market_state.timestamp.isoformat(),
            "current_price": float(market_state.current_price),
            "ohlcv_count": len(market_state.ohlcv_data),
            "position_side": market_state.current_position.side,
            "position_size": float(market_state.current_position.size),
        }

    def _serialize_trade_action(self, trade_action: TradeAction) -> Dict[str, Any]:
        """Serialize TradeAction to dictionary."""
        return {
            "action": trade_action.action,
            "size_pct": trade_action.size_pct,
            "take_profit_pct": trade_action.take_profit_pct,
            "stop_loss_pct": trade_action.stop_loss_pct,
            "leverage": trade_action.leverage,
            "reduce_only": trade_action.reduce_only,
            "rationale": trade_action.rationale,
        }

    def _extract_indicators(self, market_state: MarketState) -> Dict[str, float]:
        """Extract indicator values from market state."""
        indicators = {}

        if market_state.indicators:
            ind = market_state.indicators
            indicators.update(
                {
                    "rsi": float(ind.rsi) if ind.rsi else 50.0,
                    "cipher_a_dot": (
                        float(ind.cipher_a_dot) if ind.cipher_a_dot else 0.0
                    ),
                    "cipher_b_wave": (
                        float(ind.cipher_b_wave) if ind.cipher_b_wave else 0.0
                    ),
                    "cipher_b_money_flow": (
                        float(ind.cipher_b_money_flow)
                        if ind.cipher_b_money_flow
                        else 50.0
                    ),
                    "ema_fast": float(ind.ema_fast) if ind.ema_fast else 0.0,
                    "ema_slow": float(ind.ema_slow) if ind.ema_slow else 0.0,
                }
            )

        return indicators

    def _extract_dominance_data(
        self, market_state: MarketState
    ) -> Optional[Dict[str, float]]:
        """Extract dominance data from market state."""
        if not market_state.dominance_data:
            return None

        dom = market_state.dominance_data
        return {
            "stablecoin_dominance": float(dom.stablecoin_dominance),
            "dominance_24h_change": float(dom.dominance_24h_change),
            "dominance_rsi": float(dom.dominance_rsi) if dom.dominance_rsi else 50.0,
        }

    def _identify_patterns(
        self, market_state: MarketState, trade_action: TradeAction
    ) -> List[str]:
        """Identify market patterns for tagging."""
        patterns = []

        # Price action patterns
        if market_state.indicators:
            # Trend patterns
            if market_state.indicators.ema_fast and market_state.indicators.ema_slow:
                if market_state.indicators.ema_fast > market_state.indicators.ema_slow:
                    patterns.append("uptrend")
                else:
                    patterns.append("downtrend")

            # RSI patterns
            if market_state.indicators.rsi:
                if market_state.indicators.rsi > 70:
                    patterns.append("overbought")
                elif market_state.indicators.rsi < 30:
                    patterns.append("oversold")

            # Cipher patterns
            if market_state.indicators.cipher_b_wave:
                if market_state.indicators.cipher_b_wave > 60:
                    patterns.append("cipher_b_extreme_high")
                elif market_state.indicators.cipher_b_wave < -60:
                    patterns.append("cipher_b_extreme_low")

        # Dominance patterns
        if market_state.dominance_data:
            if market_state.dominance_data.stablecoin_dominance > 10:
                patterns.append("high_stablecoin_dominance")
            elif market_state.dominance_data.stablecoin_dominance < 5:
                patterns.append("low_stablecoin_dominance")

            if market_state.dominance_data.dominance_24h_change > 1:
                patterns.append("rising_dominance")
            elif market_state.dominance_data.dominance_24h_change < -1:
                patterns.append("falling_dominance")

        # Action patterns
        patterns.append(f"action_{trade_action.action.lower()}")

        return patterns

    def _extract_features(self, market_state: MarketState) -> np.ndarray:
        """Extract feature vector from market state for similarity comparison."""
        features = []

        # Price (normalized by log)
        features.append(np.log(float(market_state.current_price)))

        # Indicators (normalized to 0-1 range where applicable)
        if market_state.indicators:
            features.append(
                market_state.indicators.rsi / 100.0
                if market_state.indicators.rsi
                else 0.5
            )
            features.append(
                market_state.indicators.cipher_b_wave / 100.0
                if market_state.indicators.cipher_b_wave
                else 0.0
            )
            features.append(
                market_state.indicators.cipher_b_money_flow / 100.0
                if market_state.indicators.cipher_b_money_flow
                else 0.5
            )
            features.append(
                1.0
                if market_state.indicators.cipher_a_dot
                and market_state.indicators.cipher_a_dot > 0
                else 0.0
            )
        else:
            features.extend([0.5, 0.0, 0.5, 0.0])  # Default values

        # Dominance data
        if market_state.dominance_data:
            features.append(
                market_state.dominance_data.stablecoin_dominance / 20.0
            )  # Normalize to ~0-1
            features.append(
                (market_state.dominance_data.dominance_24h_change + 5) / 10.0
            )  # Normalize ±5% to 0-1
        else:
            features.extend([0.5, 0.5])

        # Position state
        position_encoding = {"FLAT": 0.5, "LONG": 1.0, "SHORT": 0.0}
        features.append(position_encoding.get(market_state.current_position.side, 0.5))

        return np.array(features)

    def _extract_features_from_experience(
        self, experience: TradingExperience
    ) -> np.ndarray:
        """Extract feature vector from stored experience."""
        features = []

        # Price
        features.append(np.log(float(experience.price)))

        # Indicators
        features.append(experience.indicators.get("rsi", 50.0) / 100.0)
        features.append(experience.indicators.get("cipher_b_wave", 0.0) / 100.0)
        features.append(experience.indicators.get("cipher_b_money_flow", 50.0) / 100.0)
        features.append(
            1.0 if experience.indicators.get("cipher_a_dot", 0.0) > 0 else 0.0
        )

        # Dominance
        if experience.dominance_data:
            features.append(
                experience.dominance_data.get("stablecoin_dominance", 10.0) / 20.0
            )
            features.append(
                (experience.dominance_data.get("dominance_24h_change", 0.0) + 5) / 10.0
            )
        else:
            features.extend([0.5, 0.5])

        # Position
        position_side = experience.market_state_snapshot.get("position_side", "FLAT")
        position_encoding = {"FLAT": 0.5, "LONG": 1.0, "SHORT": 0.0}
        features.append(position_encoding.get(position_side, 0.5))

        return np.array(features)

    def _calculate_similarity(
        self, features1: np.ndarray, features2: np.ndarray, time_weight: float = 0.2
    ) -> float:
        """Calculate similarity score between two feature vectors."""
        # Cosine similarity for features
        dot_product = np.dot(features1, features2)
        norm_product = np.linalg.norm(features1) * np.linalg.norm(features2)

        if norm_product == 0:
            feature_similarity = 0.0
        else:
            feature_similarity = dot_product / norm_product

        # Normalize to 0-1 range
        feature_similarity = (feature_similarity + 1) / 2

        # Time decay factor (more recent experiences get higher scores)
        # This would be implemented with actual timestamps
        time_similarity = 0.8  # Placeholder

        # Weighted combination
        total_similarity = (
            1 - time_weight
        ) * feature_similarity + time_weight * time_similarity

        return total_similarity

    async def _generate_insights(self, experience: TradingExperience) -> str:
        """Generate learning insights from completed trade."""
        insights = []

        if not experience.outcome:
            return "No outcome data available"

        # Analyze what worked or didn't
        if experience.outcome["success"]:
            insights.append(f"Successful {experience.decision['action']} trade")

            # What indicators were favorable?
            if (
                experience.indicators.get("rsi", 50) < 30
                and experience.decision["action"] == "LONG"
            ):
                insights.append("Oversold RSI entry worked well")
            elif (
                experience.indicators.get("rsi", 50) > 70
                and experience.decision["action"] == "SHORT"
            ):
                insights.append("Overbought RSI entry worked well")

            # Dominance correlation
            if experience.dominance_data:
                if (
                    experience.dominance_data.get("dominance_24h_change", 0) < 0
                    and experience.decision["action"] == "LONG"
                ):
                    insights.append(
                        "Falling dominance correctly predicted crypto strength"
                    )
        else:
            insights.append(f"Failed {experience.decision['action']} trade")

            # What went wrong?
            if abs(experience.outcome["price_change_pct"]) > 5:
                insights.append("Large adverse price movement - consider tighter stops")

            if (
                experience.trade_duration_minutes
                and experience.trade_duration_minutes < 30
            ):
                insights.append("Quick reversal - may have entered too early")

        # Pattern observations
        if (
            "cipher_b_extreme_high" in experience.pattern_tags
            and experience.decision["action"] == "SHORT"
        ):
            if experience.outcome["success"]:
                insights.append("Cipher B extreme high successfully predicted reversal")
            else:
                insights.append("Cipher B extreme high gave false reversal signal")

        return "; ".join(insights) if insights else "Standard trade outcome"

    def _calculate_confidence(self, experience: TradingExperience) -> float:
        """Calculate confidence score for the experience."""
        if not experience.outcome:
            return 0.5

        base_confidence = 0.6 if experience.outcome["success"] else 0.4

        # Adjust based on profit magnitude
        if experience.outcome["success"]:
            # Higher profit = higher confidence
            profit_factor = min(experience.outcome["pnl"] / 100, 0.3)  # Cap at 0.3
            base_confidence += profit_factor
        else:
            # Larger loss = lower confidence
            loss_factor = min(abs(experience.outcome["pnl"]) / 100, 0.2)
            base_confidence -= loss_factor

        # Ensure within bounds
        return max(0.1, min(0.9, base_confidence))

    async def _store_remote(self, experience: TradingExperience) -> None:
        """Store experience on remote MCP server."""
        if not self._session:
            return

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers["Content-Type"] = "application/json"

        try:
            async with self._session.post(
                f"{self.server_url}/memories",
                headers=headers,
                json=json.loads(experience.json()),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status != 201:
                    logger.warning(f"Failed to store remotely: {response.status}")
        except Exception as e:
            logger.error(f"Remote storage error: {e}")

    async def _update_remote(self, experience: TradingExperience) -> None:
        """Update experience on remote MCP server."""
        if not self._session:
            return

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers["Content-Type"] = "application/json"

        try:
            async with self._session.put(
                f"{self.server_url}/memories/{experience.experience_id}",
                headers=headers,
                json=json.loads(experience.json()),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status not in [200, 204]:
                    logger.warning(f"Failed to update remotely: {response.status}")
        except Exception as e:
            logger.error(f"Remote update error: {e}")

    async def _save_experience_local(self, experience: TradingExperience) -> None:
        """Save experience to local storage."""
        file_path = self.local_storage_path / f"{experience.experience_id}.json"

        try:
            with open(file_path, "w") as f:
                f.write(experience.json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save experience locally: {e}")

    async def _load_local_cache(self) -> None:
        """Load experiences from local storage into cache."""
        try:
            for file_path in self.local_storage_path.glob("*.json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    experience = TradingExperience(**data)
                    self.memory_cache[experience.experience_id] = experience

                    # Rebuild pattern index
                    for pattern in experience.pattern_tags:
                        if pattern not in self.pattern_index:
                            self.pattern_index[pattern] = []
                        self.pattern_index[pattern].append(experience.experience_id)

            logger.info(f"Loaded {len(self.memory_cache)} experiences from local cache")

        except Exception as e:
            logger.error(f"Failed to load local cache: {e}")

    async def _save_local_cache(self) -> None:
        """Save current cache to local storage."""
        for experience in self.memory_cache.values():
            await self._save_experience_local(experience)

    async def cleanup_old_memories(self, days: Optional[int] = None) -> int:
        """
        Clean up old memories based on retention policy.

        Args:
            days: Days to retain (overrides config)

        Returns:
            Number of memories removed
        """
        retention_days = days or settings.mcp.memory_retention_days
        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)

        removed_count = 0
        experiences_to_remove = []

        for exp_id, experience in self.memory_cache.items():
            if experience.timestamp < cutoff_date:
                experiences_to_remove.append(exp_id)

        for exp_id in experiences_to_remove:
            # Remove from cache
            del self.memory_cache[exp_id]

            # Remove from pattern index
            exp = self.memory_cache.get(exp_id)
            if exp:
                for pattern in exp.pattern_tags:
                    if pattern in self.pattern_index:
                        self.pattern_index[pattern].remove(exp_id)

            # Remove local file
            file_path = self.local_storage_path / f"{exp_id}.json"
            if file_path.exists():
                file_path.unlink()

            removed_count += 1

        logger.info(f"Cleaned up {removed_count} old memories")
        return removed_count


async def main():
    """Main entry point for running the MCP memory server."""
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    # Create FastAPI app
    app = FastAPI(
        title="MCP Memory Server",
        description="Memory and learning server for AI trading bot",
        version="1.0.0",
    )

    # Initialize memory server
    memory_server = MCPMemoryServer()
    await memory_server.connect()

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "connected": memory_server._connected,
            "memory_count": len(memory_server.memory_cache),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @app.post("/experience")
    async def store_experience(
        market_state: dict,
        trade_action: dict,
        additional_context: Optional[dict] = None,
    ):
        """Store a new trading experience."""
        try:
            # Convert dicts to proper objects
            # This would need proper validation in production
            experience_id = await memory_server.store_experience(
                market_state,  # Would need conversion
                trade_action,  # Would need conversion
                additional_context,
            )
            return {"experience_id": experience_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/experience/{experience_id}")
    async def get_experience(experience_id: str):
        """Retrieve a specific experience."""
        experience = memory_server.memory_cache.get(experience_id)
        if not experience:
            raise HTTPException(status_code=404, detail="Experience not found")
        return experience.dict()

    @app.post("/query")
    async def query_experiences(
        market_state: dict, query_params: Optional[dict] = None
    ):
        """Query similar experiences."""
        try:
            # Convert and query
            experiences = await memory_server.query_similar_experiences(
                market_state,  # Would need conversion
                query_params,  # Would need conversion
            )
            return {
                "count": len(experiences),
                "experiences": [exp.dict() for exp in experiences],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        await memory_server.disconnect()

    # Get port from environment
    port = int(os.getenv("MCP_SERVER_PORT", "8765"))

    # Run server
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
