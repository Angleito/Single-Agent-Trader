"""
Memory-enhanced LLM agent with MCP integration.

This module extends the base LLM agent with memory capabilities,
allowing it to learn from past experiences and improve over time.
"""

import logging
from typing import Any, List, Optional

from ..config import settings
from ..mcp.memory_server import MCPMemoryServer, MemoryQuery, TradingExperience
from ..types import MarketState, TradeAction
from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class MemoryEnhancedLLMAgent(LLMAgent):
    """
    Enhanced LLM agent that uses past trading experiences to improve decisions.

    Extends the base LLM agent with:
    - Memory retrieval from similar market conditions
    - Learning from past successes and failures
    - Context-aware decision making based on historical performance
    """

    def __init__(
        self,
        model_provider: str = None,
        model_name: str = None,
        memory_server: Optional[MCPMemoryServer] = None,
    ):
        """
        Initialize the memory-enhanced LLM agent.

        Args:
            model_provider: LLM provider ('openai', 'ollama')
            model_name: Specific model to use
            memory_server: MCP memory server instance
        """
        # Initialize base LLM agent
        super().__init__(model_provider, model_name)

        # Memory components
        self.memory_server = memory_server
        self._memory_available = memory_server is not None and settings.mcp.enabled

        # Enhanced prompt template for memory context
        self._memory_prompt_addon = """

PAST TRADING EXPERIENCES:
Based on similar market conditions, here are relevant past trades:

{memory_context}

LEARNED PATTERNS:
{pattern_insights}

IMPORTANT: Consider these past experiences when making your decision, but adapt to current unique conditions.
"""

        logger.info(
            f"Initialized memory-enhanced LLM agent "
            f"(memory={'enabled' if self._memory_available else 'disabled'})"
        )

    async def analyze_market(self, market_state: MarketState) -> TradeAction:
        """
        Analyze market state with memory-enhanced context.

        Args:
            market_state: Complete market state including OHLCV and indicators

        Returns:
            TradeAction with decision and parameters
        """
        # If memory not available, fall back to base implementation
        if not self._memory_available:
            return await super().analyze_market(market_state)

        try:
            # Retrieve relevant past experiences
            similar_experiences = await self._retrieve_relevant_memories(market_state)

            # Generate memory context for the prompt
            memory_context = self._format_memory_context(similar_experiences)

            # Get pattern insights if available
            pattern_insights = await self._get_pattern_insights()

            # Enhance the prompt with memory context
            enhanced_market_state = self._enhance_with_memory(
                market_state, memory_context, pattern_insights
            )

            # Get decision using enhanced context
            result = await super().analyze_market(enhanced_market_state)

            # Log memory-enhanced decision
            logger.info(
                f"Memory-enhanced decision: {result.action} "
                f"(considered {len(similar_experiences)} past experiences)"
            )

            return result

        except Exception as e:
            logger.error(f"Error in memory-enhanced analysis: {e}")
            # Fall back to base implementation
            return await super().analyze_market(market_state)

    async def _retrieve_relevant_memories(
        self, market_state: MarketState
    ) -> List[TradingExperience]:
        """
        Retrieve relevant past trading experiences.

        Args:
            market_state: Current market state

        Returns:
            List of similar past experiences
        """
        if not self.memory_server:
            return []

        try:
            # Create query based on current conditions
            query = MemoryQuery(
                current_price=market_state.current_price,
                indicators=self._extract_indicator_dict(market_state),
                dominance_data=self._extract_dominance_dict(market_state),
                market_sentiment=market_state.indicators.market_sentiment,
                max_results=settings.mcp.max_memories_per_query,
                min_similarity=settings.mcp.similarity_threshold,
            )

            # Query memory server
            experiences = await self.memory_server.query_similar_experiences(
                market_state, query
            )

            return experiences

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def _format_memory_context(self, experiences: List[TradingExperience]) -> str:
        """
        Format past experiences into context for the LLM.

        Args:
            experiences: List of relevant past experiences

        Returns:
            Formatted string for prompt inclusion
        """
        if not experiences:
            return "No similar past experiences found."

        context_lines = []

        for i, exp in enumerate(experiences[:5]):  # Limit to top 5
            # Format basic trade info
            action = exp.decision.get("action", "UNKNOWN")
            size = exp.decision.get("size_pct", 0)
            leverage = exp.decision.get("leverage", 1)

            context_lines.append(f"\n{i+1}. Past {action} trade:")
            context_lines.append(f"   Market conditions: ${exp.price}")

            # Add indicator snapshot
            indicators = []
            if exp.indicators.get("rsi"):
                indicators.append(f"RSI={exp.indicators['rsi']:.1f}")
            if exp.indicators.get("cipher_b_wave"):
                indicators.append(f"Wave={exp.indicators['cipher_b_wave']:.1f}")
            if indicators:
                context_lines.append(f"   Indicators: {', '.join(indicators)}")

            # Add outcome if available
            if exp.outcome:
                success = "SUCCESS" if exp.outcome["success"] else "FAILURE"
                pnl = exp.outcome["pnl"]
                duration = exp.trade_duration_minutes or 0

                context_lines.append(
                    f"   Outcome: {success} (PnL=${pnl:.2f}, "
                    f"Duration={duration:.0f}min)"
                )

                # Add learned insights
                if exp.learned_insights:
                    context_lines.append(f"   Insight: {exp.learned_insights}")
            else:
                context_lines.append("   Outcome: Trade still active")

            # Add confidence score
            context_lines.append(f"   Relevance: {exp.confidence_score:.1%}")

        return "\n".join(context_lines)

    async def _get_pattern_insights(self) -> str:
        """
        Get insights about successful patterns.

        Returns:
            Formatted string with pattern insights
        """
        if not self.memory_server:
            return "No pattern analysis available."

        try:
            pattern_stats = await self.memory_server.get_pattern_statistics()

            if not pattern_stats:
                return "Insufficient data for pattern analysis."

            insights = []

            # Find best performing patterns
            sorted_patterns = sorted(
                pattern_stats.items(),
                key=lambda x: x[1]["success_rate"] * x[1]["count"],
                reverse=True,
            )

            for pattern, stats in sorted_patterns[:3]:
                if stats["count"] >= settings.mcp.min_samples_for_pattern:
                    insights.append(
                        f"• Pattern '{pattern}': "
                        f"{stats['success_rate']:.1%} win rate "
                        f"({stats['count']} trades, "
                        f"avg PnL=${stats['avg_pnl']:.2f})"
                    )

            return (
                "\n".join(insights)
                if insights
                else "No significant patterns identified yet."
            )

        except Exception as e:
            logger.error(f"Failed to get pattern insights: {e}")
            return "Pattern analysis temporarily unavailable."

    def _enhance_with_memory(
        self, market_state: MarketState, memory_context: str, pattern_insights: str
    ) -> MarketState:
        """
        Enhance market state with memory context.

        This is a workaround to inject memory context into the prompt
        without modifying the base prompt template.

        Args:
            market_state: Original market state
            memory_context: Formatted memory context
            pattern_insights: Pattern analysis insights

        Returns:
            Enhanced market state
        """
        # Create a copy of the market state
        # In practice, we'd modify the prompt template directly
        # For now, we'll append memory context to the rationale

        # Store memory context for use in prompt preparation
        self._temp_memory_context = {
            "memory_context": memory_context,
            "pattern_insights": pattern_insights,
        }

        return market_state

    def _prepare_llm_input(self, market_state: MarketState) -> dict[str, Any]:
        """
        Prepare LLM input with memory context.

        Overrides base method to include memory information.
        """
        # Get base input from parent
        llm_input = super()._prepare_llm_input(market_state)

        # Add memory context if available
        if hasattr(self, "_temp_memory_context"):
            # Append memory context to the OHLCV tail
            # This is a workaround - ideally we'd have a dedicated field
            memory_section = (
                f"\n\n=== MEMORY CONTEXT ===\n"
                f"{self._temp_memory_context['memory_context']}\n\n"
                f"=== PATTERN INSIGHTS ===\n"
                f"{self._temp_memory_context['pattern_insights']}"
            )

            llm_input["ohlcv_tail"] += memory_section

            # Clean up temporary storage
            delattr(self, "_temp_memory_context")

        return llm_input

    def _extract_indicator_dict(self, market_state: MarketState) -> dict[str, float]:
        """Extract indicators as a dictionary for memory queries."""
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
                }
            )

        return indicators

    def _extract_dominance_dict(
        self, market_state: MarketState
    ) -> Optional[dict[str, float]]:
        """Extract dominance data as a dictionary for memory queries."""
        if not market_state.dominance_data:
            return None

        dom = market_state.dominance_data
        return {
            "stablecoin_dominance": float(dom.stablecoin_dominance),
            "dominance_24h_change": float(dom.dominance_24h_change),
            "dominance_rsi": float(dom.dominance_rsi) if dom.dominance_rsi else 50.0,
        }

    def get_status(self) -> dict[str, Any]:
        """Get status including memory availability."""
        status = super().get_status()

        # Add memory-specific status
        status.update(
            {
                "memory_enabled": self._memory_available,
                "memory_server_connected": (
                    self.memory_server._connected if self.memory_server else False
                ),
            }
        )

        return status
