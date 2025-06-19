"""
Memory-enhanced LLM agent with MCP integration.

This module extends the base LLM agent with memory capabilities,
allowing it to learn from past experiences and improve over time.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from ..config import settings
from ..logging.trade_logger import TradeLogger
from ..mcp.memory_server import MCPMemoryServer, MemoryQuery, TradingExperience
from ..mcp.omnisearch_client import (
    OmniSearchClient,
)
from ..trading_types import MarketState, TradeAction
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
        model_provider: str | None = None,
        model_name: str | None = None,
        memory_server: MCPMemoryServer | None = None,
        omnisearch_client: OmniSearchClient | None = None,
    ):
        """
        Initialize the memory-enhanced LLM agent.

        Args:
            model_provider: LLM provider ('openai', 'ollama')
            model_name: Specific model to use
            memory_server: MCP memory server instance
            omnisearch_client: Optional OmniSearch client for market intelligence
        """
        # Initialize base LLM agent with OmniSearch client
        super().__init__(model_provider, model_name, omnisearch_client)

        # Memory components
        self.memory_server = memory_server
        self._memory_available = memory_server is not None and settings.mcp.enabled

        # Initialize trade logger
        self.trade_logger = TradeLogger()

        # Performance optimization settings
        self._enable_api_parallelization = settings.llm.enable_api_call_parallelization
        self._use_optimized_prompts = settings.llm.use_optimized_prompts

        # Enhanced prompt template for memory and sentiment context
        self._memory_prompt_addon = """

PAST TRADING EXPERIENCES:
Based on similar market conditions, here are relevant past trades:

{memory_context}

LEARNED PATTERNS:
{pattern_insights}

SENTIMENT-ENHANCED CONTEXT:
{sentiment_enhanced_context}

IMPORTANT: Consider these past experiences and sentiment correlations when making your decision, but adapt to current unique conditions.
"""

        logger.info(
            f"🧠 Memory-Enhanced LLM Agent: Initialized "
            f"(memory={'✅ enabled' if self._memory_available else '❌ disabled'}, "
            f"optimized_prompts={'✅ enabled' if self._use_optimized_prompts else '❌ disabled'}, "
            f"api_parallelization={'✅ enabled' if self._enable_api_parallelization else '❌ disabled'})"
        )

    async def analyze_market(self, market_state: MarketState) -> TradeAction:
        """
        Analyze market state with memory-enhanced and sentiment-enhanced context.

        Args:
            market_state: Complete market state including OHLCV and indicators

        Returns:
            TradeAction with decision and parameters
        """
        # If memory not available, fall back to base implementation
        if not self._memory_available:
            return await super().analyze_market(market_state)

        try:
            # Use cache if enabled, but with memory-enhanced computation
            if self._cache_enabled and self._cache and self._chain is not None:
                result = await self._cache.get_or_compute(
                    market_state, self._get_memory_enhanced_decision, market_state
                )
            else:
                # Original memory-enhanced path without caching
                result = await self._get_memory_enhanced_decision(market_state)

            # Store sentiment data with the decision for future learning
            await self._store_sentiment_data_for_learning(market_state, result)

            return result

        except Exception as e:
            logger.error(f"Error in memory-enhanced analysis: {e}")
            # Fall back to base implementation
            return await super().analyze_market(market_state)

    async def _get_memory_enhanced_decision(
        self, market_state: MarketState
    ) -> TradeAction:
        """
        Get memory-enhanced decision (used by cache system).

        Args:
            market_state: Current market state

        Returns:
            TradeAction with memory and sentiment enhancement
        """
        overall_start_time = time.time()
        # Retrieve relevant past experiences
        similar_experiences = await self._retrieve_relevant_memories(market_state)

        # Generate memory context for the prompt
        memory_context = self._format_memory_context(similar_experiences)

        # PERFORMANCE OPTIMIZATION: Parallel processing of pattern insights and sentiment context
        if self._enable_api_parallelization:
            # These are independent operations that can be executed concurrently
            pattern_insights, sentiment_enhanced_context = await asyncio.gather(
                self._get_pattern_insights(),
                self._get_sentiment_enhanced_context(market_state),
                return_exceptions=True,
            )

            # Handle exceptions in parallel operations
            if isinstance(pattern_insights, Exception):
                logger.debug(f"Pattern insights failed: {pattern_insights}")
                pattern_insights = "No pattern insights available"

            if isinstance(sentiment_enhanced_context, Exception):
                logger.debug(f"Sentiment context failed: {sentiment_enhanced_context}")
                sentiment_enhanced_context = ""
        else:
            # Sequential execution if parallelization is disabled
            pattern_insights = await self._get_pattern_insights()
            sentiment_enhanced_context = await self._get_sentiment_enhanced_context(
                market_state
            )

        # Ensure pattern_insights and sentiment_enhanced_context are strings
        pattern_insights_str = (
            pattern_insights
            if isinstance(pattern_insights, str)
            else "No pattern insights available"
        )
        sentiment_enhanced_context_str = (
            sentiment_enhanced_context
            if isinstance(sentiment_enhanced_context, str)
            else ""
        )

        # Enhance the prompt with memory and sentiment context
        enhanced_market_state = self._enhance_with_memory(
            market_state,
            memory_context,
            pattern_insights_str,
            sentiment_enhanced_context_str,
        )

        # Get decision using enhanced context (bypass cache for this call)
        if self._chain is not None:
            # Prepare input data for the LLM
            llm_input = await self._prepare_llm_input(enhanced_market_state)
            result = await self._get_llm_decision(llm_input)
        else:
            result = self._get_fallback_decision(market_state)

        # Log memory-enhanced decision
        logger.info(
            f"🤖 Memory+Sentiment-Enhanced Decision: {result.action} | "
            f"Similar experiences: {len(similar_experiences)} | "
            f"Memory context: {'✅ Applied' if memory_context != 'No similar past experiences found.' else '❌ None'} | "
            f"Sentiment context: {'✅ Applied' if 'sentiment' in sentiment_enhanced_context_str.lower() else '❌ None'}"
        )

        # Log detailed memory context used
        if similar_experiences:
            logger.debug(
                f"Top 3 similar experiences: "
                f"{[exp.experience_id for exp in similar_experiences[:3]]}"
            )

            # Log success rates of similar trades
            successful_similar = sum(
                1
                for exp in similar_experiences
                if exp.outcome and exp.outcome.get("success", False)
            )
            logger.debug(
                f"Similar trade success rate: {successful_similar}/{len(similar_experiences)} "
                f"({successful_similar/len(similar_experiences)*100:.1f}%)"
            )

        # Store memory context for external logging access
        self._last_memory_context = {
            "experiences": [
                {
                    "experience_id": exp.experience_id,
                    "action": exp.decision.get("action", "UNKNOWN"),
                    "outcome": exp.outcome,
                    "patterns": exp.pattern_tags,
                    "timestamp": exp.timestamp.isoformat() if exp.timestamp else None,
                }
                for exp in similar_experiences
            ],
            "pattern_insights": pattern_insights_str,
            "sentiment_context": sentiment_enhanced_context_str,
            "memory_context_formatted": memory_context,
        }

        # Log memory context for structured trade logging
        if similar_experiences:
            self.trade_logger.log_memory_query(
                query_params={
                    "symbol": market_state.symbol,
                    "current_price": float(market_state.current_price),
                    "indicators": (
                        {
                            "rsi": getattr(market_state.indicators, "rsi", None),
                            "cipher_a_dot": getattr(
                                market_state.indicators, "cipher_a_dot", None
                            ),
                            "cipher_b_wave": getattr(
                                market_state.indicators, "cipher_b_wave", None
                            ),
                        }
                        if market_state.indicators
                        else {}
                    ),
                },
                results=[
                    {
                        "experience_id": exp.experience_id,
                        "action": exp.decision.get("action", "UNKNOWN"),
                        "similarity": 0.8,  # Placeholder, actual similarity would come from memory server
                        "success": (
                            exp.outcome.get("success", False) if exp.outcome else False
                        ),
                    }
                    for exp in similar_experiences
                ],
                execution_time_ms=0.0,  # Would be measured in actual implementation
            )

        # Log overall performance metrics
        total_execution_time = time.time() - overall_start_time
        optimization_mode = []
        if self._use_optimized_prompts:
            optimization_mode.append("optimized_prompts")
        if self._enable_api_parallelization:
            optimization_mode.append("parallel_apis")

        logger.info(
            f"🚀 Memory-Enhanced Decision Complete: {total_execution_time:.3f}s | "
            f"Action: {result.action} | "
            f"Optimizations: {'+'.join(optimization_mode) if optimization_mode else 'none'} | "
            f"Experiences: {len(similar_experiences)}"
        )

        return result

    async def _retrieve_relevant_memories(
        self, market_state: MarketState
    ) -> list[TradingExperience]:
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
                market_sentiment=(
                    market_state.indicators.market_sentiment
                    if market_state.indicators
                    else None
                ),
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

    def _format_memory_context(self, experiences: list[TradingExperience]) -> str:
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

            context_lines.append(f"\n{i+1}. Past {action} trade:")
            context_lines.append(f"   Market conditions: ${float(exp.price)}")

            # Add indicator snapshot
            indicators = []
            if exp.indicators and exp.indicators.get("rsi"):
                indicators.append(f"RSI={exp.indicators['rsi']:.1f}")
            if exp.indicators and exp.indicators.get("cipher_b_wave"):
                indicators.append(f"Wave={exp.indicators['cipher_b_wave']:.1f}")
            if indicators:
                context_lines.append(f"   Indicators: {', '.join(indicators)}")

            # Add outcome if available
            if exp.outcome:
                success = "SUCCESS" if exp.outcome.get("success", False) else "FAILURE"
                pnl = exp.outcome.get("pnl", 0.0)
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
        self,
        market_state: MarketState,
        memory_context: str,
        pattern_insights: str,
        sentiment_enhanced_context: str = "",
    ) -> MarketState:
        """
        Enhance market state with memory and sentiment context.

        This is a workaround to inject memory context into the prompt
        without modifying the base prompt template.

        Args:
            market_state: Original market state
            memory_context: Formatted memory context
            pattern_insights: Pattern analysis insights
            sentiment_enhanced_context: Sentiment and correlation analysis

        Returns:
            Enhanced market state
        """
        # Create a copy of the market state
        # In practice, we'd modify the prompt template directly
        # For now, we'll append memory context to the rationale

        # Store memory and sentiment context for use in prompt preparation
        self._temp_memory_context = {
            "memory_context": memory_context,
            "pattern_insights": pattern_insights,
            "sentiment_enhanced_context": sentiment_enhanced_context,
        }

        return market_state

    async def _prepare_llm_input(self, market_state: MarketState) -> dict[str, Any]:
        """
        Prepare LLM input with memory and sentiment context.

        Overrides base method to include memory and sentiment information.
        """
        # Get base input from parent - await since it might be async
        llm_input = await super()._prepare_llm_input(market_state)

        # Add memory and sentiment context if available
        if hasattr(self, "_temp_memory_context"):
            # Append memory and sentiment context to the OHLCV tail
            # This is a workaround - ideally we'd have a dedicated field
            memory_section = (
                f"\n\n=== MEMORY CONTEXT ===\n"
                f"{self._temp_memory_context['memory_context']}\n\n"
                f"=== PATTERN INSIGHTS ===\n"
                f"{self._temp_memory_context['pattern_insights']}\n\n"
                f"=== SENTIMENT-ENHANCED CONTEXT ===\n"
                f"{self._temp_memory_context['sentiment_enhanced_context']}"
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
    ) -> dict[str, float] | None:
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
        """Get status including memory and OmniSearch availability."""
        status = super().get_status()

        # Add memory-specific status
        status.update(
            {
                "memory_enabled": self._memory_available,
                "memory_server_connected": (
                    self.memory_server._connected if self.memory_server else False
                ),
                "sentiment_analysis_enabled": hasattr(self, "_omnisearch_client")
                and self._omnisearch_client is not None,
                "sentiment_learning_active": self._memory_available
                and hasattr(self, "_omnisearch_client")
                and self._omnisearch_client is not None,
            }
        )

        return status

    async def _get_sentiment_enhanced_context(self, market_state: MarketState) -> str:
        """
        Combine memory retrieval with financial sentiment data.

        Args:
            market_state: Current market state

        Returns:
            Formatted string with combined memory and sentiment context
        """
        context_sections = []

        try:
            # Get financial sentiment if OmniSearch is available
            if hasattr(self, "_omnisearch_client") and self._omnisearch_client:
                sentiment_context = await self._get_financial_sentiment_context(
                    market_state
                )
                if sentiment_context:
                    context_sections.append(sentiment_context)

            # Get correlation analysis between historical patterns and sentiment
            if self.memory_server:
                correlation_analysis = (
                    await self._analyze_sentiment_pattern_correlation(market_state)
                )
                if correlation_analysis:
                    context_sections.append(correlation_analysis)

            # Get sentiment trend analysis
            sentiment_trends = await self._analyze_sentiment_trends(market_state)
            if sentiment_trends:
                context_sections.append(sentiment_trends)

            # Combine all sections
            if context_sections:
                return "\n\n".join(context_sections)
            else:
                return "No sentiment-enhanced context available"

        except Exception as e:
            logger.error(f"Error generating sentiment-enhanced context: {e}")
            return f"Error generating sentiment context: {str(e)}"

    async def _get_financial_sentiment_context(self, market_state: MarketState) -> str:
        """
        Get financial sentiment context from OmniSearch.

        Args:
            market_state: Current market state

        Returns:
            Formatted sentiment context string
        """
        if not hasattr(self, "_omnisearch_client") or not self._omnisearch_client:
            return ""

        try:
            start_time = time.time()
            base_symbol = market_state.symbol.split("-")[0]
            context_lines = []

            # PERFORMANCE OPTIMIZATION: Parallel API calls using asyncio.gather()
            # Check if parallelization is enabled in configuration
            if self._enable_api_parallelization:
                # Define async functions for parallel execution
                async def get_crypto_sentiment():
                    try:
                        return await self._omnisearch_client.search_crypto_sentiment(
                            base_symbol
                        )
                    except Exception as e:
                        logger.debug(f"Crypto sentiment search failed: {e}")
                        return None

                async def get_market_correlation():
                    try:
                        return await self._omnisearch_client.search_market_correlation(
                            base_symbol, "QQQ"
                        )
                    except Exception as e:
                        logger.debug(f"Market correlation search failed: {e}")
                        return None

                # Execute both API calls in parallel - This is the key performance improvement
                crypto_sentiment, correlation = await asyncio.gather(
                    get_crypto_sentiment(),
                    get_market_correlation(),
                    return_exceptions=True,
                )
            else:
                # Fallback to sequential API calls if parallelization is disabled
                try:
                    crypto_sentiment = (
                        await self._omnisearch_client.search_crypto_sentiment(
                            base_symbol
                        )
                    )
                except Exception as e:
                    logger.debug(f"Crypto sentiment search failed: {e}")
                    crypto_sentiment = None

                try:
                    correlation = (
                        await self._omnisearch_client.search_market_correlation(
                            base_symbol, "QQQ"
                        )
                    )
                except Exception as e:
                    logger.debug(f"Market correlation search failed: {e}")
                    correlation = None

            # Process crypto sentiment results
            if crypto_sentiment and not isinstance(crypto_sentiment, Exception):
                context_lines.append(
                    f"=== {base_symbol} Financial Sentiment Context ==="
                )
                context_lines.append(
                    f"Current Sentiment: {crypto_sentiment.overall_sentiment.upper()} ({crypto_sentiment.sentiment_score:+.2f})"
                )
                context_lines.append(f"Confidence: {crypto_sentiment.confidence:.1%}")

                if crypto_sentiment.key_drivers:
                    context_lines.append(
                        f"Key Drivers: {', '.join(crypto_sentiment.key_drivers[:3])}"
                    )
                if crypto_sentiment.risk_factors:
                    context_lines.append(
                        f"Risk Factors: {', '.join(crypto_sentiment.risk_factors[:3])}"
                    )

            # Process correlation results
            if correlation and not isinstance(correlation, Exception):
                context_lines.append(
                    f"Market Correlation: {correlation.direction.upper()} {correlation.strength.upper()} ({correlation.correlation_coefficient:+.3f})"
                )

                if abs(correlation.correlation_coefficient) > 0.5:
                    correlation_impact = (
                        "High correlation - expect macro market influence"
                    )
                else:
                    correlation_impact = "Low correlation - crypto moving independently"
                context_lines.append(f"Correlation Impact: {correlation_impact}")

            # Log performance metrics
            execution_time = time.time() - start_time
            method_type = (
                "parallel" if self._enable_api_parallelization else "sequential"
            )
            logger.debug(
                f"⚡ Sentiment context ({method_type}): {execution_time:.3f}s | "
                f"Sentiment: {'✅' if crypto_sentiment and not isinstance(crypto_sentiment, Exception) else '❌'} | "
                f"Correlation: {'✅' if correlation and not isinstance(correlation, Exception) else '❌'}"
            )

            return "\n".join(context_lines) if context_lines else ""

        except Exception as e:
            logger.warning(f"Failed to get financial sentiment context: {e}")
            return ""

    async def _analyze_sentiment_pattern_correlation(
        self, market_state: MarketState
    ) -> str:
        """
        Analyze correlation between historical patterns and sentiment data.

        Args:
            market_state: Current market state

        Returns:
            Formatted correlation analysis string
        """
        if not self.memory_server:
            return ""

        try:
            # Get recent experiences with sentiment data
            recent_experiences = await self._get_recent_experiences_with_sentiment(
                market_state, limit=20
            )

            if len(recent_experiences) < 5:
                return "=== Sentiment-Pattern Correlation ===\nInsufficient data for correlation analysis"

            # Analyze patterns
            correlation_lines = ["=== Sentiment-Pattern Correlation ==="]

            # Group by sentiment and analyze success rates
            sentiment_groups = {}
            for exp in recent_experiences:
                sentiment_data = (exp.market_state_snapshot or {}).get(
                    "sentiment_data", {}
                )
                if sentiment_data:
                    sentiment = sentiment_data.get("overall_sentiment", "neutral")
                    if sentiment not in sentiment_groups:
                        sentiment_groups[sentiment] = {"total": 0, "successful": 0}

                    sentiment_groups[sentiment]["total"] += 1
                    if exp.outcome and exp.outcome.get("success", False):
                        sentiment_groups[sentiment]["successful"] += 1

            # Calculate success rates by sentiment
            for sentiment, data in sentiment_groups.items():
                if data["total"] >= 3:  # Only show if we have enough samples
                    success_rate = data["successful"] / data["total"] * 100
                    correlation_lines.append(
                        f"{sentiment.upper()} sentiment: {success_rate:.1f}% success rate ({data['successful']}/{data['total']} trades)"
                    )

            # Find best performing sentiment conditions
            if sentiment_groups:
                best_sentiment = max(
                    sentiment_groups.items(),
                    key=lambda x: x[1]["successful"] / max(x[1]["total"], 1),
                )
                correlation_lines.append(
                    f"Best performing sentiment: {best_sentiment[0].upper()}"
                )

            return "\n".join(correlation_lines)

        except Exception as e:
            logger.warning(f"Failed to analyze sentiment-pattern correlation: {e}")
            return ""

    async def _analyze_sentiment_trends(self, market_state: MarketState) -> str:
        """
        Analyze sentiment trends from recent experiences.

        Args:
            market_state: Current market state

        Returns:
            Formatted sentiment trends analysis
        """
        if not self.memory_server:
            return ""

        try:
            # Get recent experiences with sentiment data
            recent_experiences = await self._get_recent_experiences_with_sentiment(
                market_state, limit=10
            )

            if len(recent_experiences) < 3:
                return "=== Sentiment Trends ===\nInsufficient data for trend analysis"

            trend_lines = ["=== Sentiment Trends ==="]

            # Extract sentiment scores over time
            sentiment_scores = []
            for exp in recent_experiences:
                sentiment_data = (exp.market_state_snapshot or {}).get(
                    "sentiment_data", {}
                )
                if sentiment_data and "sentiment_score" in sentiment_data:
                    sentiment_scores.append(sentiment_data["sentiment_score"])

            if len(sentiment_scores) >= 3:
                # Calculate trend direction
                if len(sentiment_scores) >= 2:
                    recent_change = sentiment_scores[-1] - sentiment_scores[-2]
                    if recent_change > 0.1:
                        trend_direction = "📈 IMPROVING"
                    elif recent_change < -0.1:
                        trend_direction = "📉 DETERIORATING"
                    else:
                        trend_direction = "➡️ STABLE"

                    trend_lines.append(f"Recent Sentiment Trend: {trend_direction}")
                    trend_lines.append(f"Latest Score: {sentiment_scores[-1]:+.2f}")
                    trend_lines.append(f"Previous Score: {sentiment_scores[-2]:+.2f}")
                    trend_lines.append(f"Change: {recent_change:+.2f}")

            return "\n".join(trend_lines)

        except Exception as e:
            logger.warning(f"Failed to analyze sentiment trends: {e}")
            return ""

    async def _get_recent_experiences_with_sentiment(
        self, market_state: MarketState, limit: int = 20
    ) -> list[TradingExperience]:
        """
        Get recent trading experiences that include sentiment data.

        Args:
            market_state: Current market state
            limit: Maximum number of experiences to retrieve

        Returns:
            List of recent experiences with sentiment data
        """
        if not self.memory_server:
            return []

        try:
            # Use query_similar_experiences with relaxed similarity threshold to get recent experiences
            query = MemoryQuery(
                current_price=market_state.current_price,
                indicators=self._extract_indicator_dict(market_state),
                dominance_data=self._extract_dominance_dict(market_state),
                market_sentiment=(
                    market_state.indicators.market_sentiment
                    if market_state.indicators
                    else None
                ),
                max_results=limit * 2,  # Get more to filter for sentiment data
                min_similarity=0.1,  # Very low threshold to get recent experiences
                time_weight=0.8,  # High time weight to prioritize recent experiences
            )

            # Query for similar experiences
            all_experiences = await self.memory_server.query_similar_experiences(
                market_state, query
            )

            # Filter for experiences with sentiment data
            sentiment_experiences = []
            for exp in all_experiences:
                if (
                    hasattr(exp, "market_state_snapshot")
                    and exp.market_state_snapshot
                    and exp.market_state_snapshot.get("sentiment_data")
                ):
                    sentiment_experiences.append(exp)
                    if len(sentiment_experiences) >= limit:
                        break

            return sentiment_experiences

        except Exception as e:
            logger.warning(f"Failed to get recent experiences with sentiment: {e}")
            return []

    async def _store_sentiment_data_for_learning(
        self, market_state: MarketState, trade_action: TradeAction
    ) -> None:
        """
        Store sentiment analysis results with trading experiences for future learning.

        Args:
            market_state: Current market state
            trade_action: Trading decision made
        """
        if not hasattr(self, "_omnisearch_client") or not self._omnisearch_client:
            return

        try:
            base_symbol = market_state.symbol.split("-")[0]
            sentiment_data = {}

            # PERFORMANCE OPTIMIZATION: Parallel API calls for sentiment data storage
            if self._enable_api_parallelization:
                # Run sentiment and correlation calls in parallel to reduce latency
                async def get_sentiment_for_storage():
                    try:
                        return await self._omnisearch_client.search_crypto_sentiment(
                            base_symbol
                        )
                    except Exception as e:
                        logger.debug(f"Crypto sentiment for storage failed: {e}")
                        return None

                async def get_correlation_for_storage():
                    try:
                        return await self._omnisearch_client.search_market_correlation(
                            base_symbol, "QQQ"
                        )
                    except Exception as e:
                        logger.debug(f"Market correlation for storage failed: {e}")
                        return None

                # Execute both storage API calls in parallel - Key performance improvement
                crypto_sentiment, correlation = await asyncio.gather(
                    get_sentiment_for_storage(),
                    get_correlation_for_storage(),
                    return_exceptions=True,
                )
            else:
                # Fallback to sequential calls
                try:
                    crypto_sentiment = (
                        await self._omnisearch_client.search_crypto_sentiment(
                            base_symbol
                        )
                    )
                except Exception as e:
                    logger.debug(f"Crypto sentiment for storage failed: {e}")
                    crypto_sentiment = None

                try:
                    correlation = (
                        await self._omnisearch_client.search_market_correlation(
                            base_symbol, "QQQ"
                        )
                    )
                except Exception as e:
                    logger.debug(f"Market correlation for storage failed: {e}")
                    correlation = None

            # Process sentiment data
            if crypto_sentiment and not isinstance(crypto_sentiment, Exception):
                sentiment_data = {
                    "overall_sentiment": crypto_sentiment.overall_sentiment,
                    "sentiment_score": crypto_sentiment.sentiment_score,
                    "confidence": crypto_sentiment.confidence,
                    "key_drivers": crypto_sentiment.key_drivers[:3],
                    "risk_factors": crypto_sentiment.risk_factors[:3],
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            # Process correlation data
            if correlation and not isinstance(correlation, Exception):
                sentiment_data["market_correlation"] = {
                    "coefficient": correlation.correlation_coefficient,
                    "strength": correlation.strength,
                    "direction": correlation.direction,
                }

            # Store in market state snapshot for memory server
            if sentiment_data and self.memory_server:
                # Create a temporary market state snapshot with sentiment data
                {
                    "symbol": market_state.symbol,
                    "price": float(market_state.current_price),
                    "sentiment_data": sentiment_data,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

                # Store this for the next memory creation
                # This will be picked up when the trading experience is created
                self._pending_sentiment_data = sentiment_data

                logger.debug(
                    f"Stored sentiment data for learning: {sentiment_data.get('overall_sentiment', 'unknown')} sentiment"
                )

        except Exception as e:
            logger.warning(f"Failed to store sentiment data for learning: {e}")
