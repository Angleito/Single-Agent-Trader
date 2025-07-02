"""
Strategy adapters for migrating from imperative to functional implementations.

This module provides adapter classes that maintain the existing imperative APIs
while using functional strategy implementations internally.
"""

import asyncio
import logging
import time
from typing import Any, Union, cast

from bot.config import settings
from bot.fp.strategies.llm_functional import (
    LLMConfig,
    LLMContext,
    LLMProvider,
    LLMResponse,
    adjust_confidence_by_market_conditions,
    create_market_context,
    generate_trading_prompt,
    parse_llm_response,
    validate_llm_decision,
)
from bot.trading_types import MarketState, TradeAction

# Import TradingParams and TradeSignal types
try:
    from bot.fp.types import Hold, Long, Short, TradeSignal, TradingParams

except ImportError:
    # Define minimal types if they don't exist
    from dataclasses import dataclass

    @dataclass
    class Long:
        confidence: float
        size: float
        reason: str

    @dataclass
    class Short:
        confidence: float
        size: float
        reason: str

    @dataclass
    class Hold:
        reason: str

    TradeSignal = Union[Long, Short, Hold]

    @dataclass
    class TradingParams:
        risk_per_trade: float = 0.1
        max_leverage: int = 5
        stop_loss_pct: float = 1.0
        take_profit_pct: float = 2.0
        max_position_size: float = 0.25


# Add Signal type alias for compatibility
Signal = TradeSignal


logger = logging.getLogger(__name__)


class TypeConverter:
    """Convert between functional LLM responses and imperative TradeAction."""

    @staticmethod
    def llm_response_to_trade_action(
        llm_response: LLMResponse, market_state: MarketState
    ) -> TradeAction:
        """Convert LLMResponse to TradeAction."""

        # Map functional signal to action string
        if isinstance(llm_response.signal, Long):
            action = "LONG"
            # Extract size from the Long signal
            size_pct = llm_response.signal.size * 100  # Convert to percentage
        elif isinstance(llm_response.signal, Short):
            action = "SHORT"
            # Extract size from the Short signal
            size_pct = llm_response.signal.size * 100  # Convert to percentage
        else:  # Hold
            action = "HOLD"
            size_pct = 0.0

        # Override with suggested params if available
        if (
            llm_response.suggested_params
            and "position_size" in llm_response.suggested_params
        ):
            size_pct = llm_response.suggested_params["position_size"] * 100

        # Extract stop loss and take profit
        if action in ["LONG", "SHORT"]:
            if llm_response.suggested_params:
                # Try to convert price levels to percentages
                current_price = float(market_state.current_price)

                # Get stop loss percentage
                if "stop_loss" in llm_response.suggested_params:
                    stop_price = llm_response.suggested_params["stop_loss"]
                    if stop_price > 0:
                        stop_loss_pct = (
                            abs(stop_price - current_price) / current_price * 100
                        )
                    else:
                        stop_loss_pct = 1.0
                else:
                    stop_loss_pct = 1.0

                # Get take profit percentage
                if "take_profit" in llm_response.suggested_params:
                    tp_price = llm_response.suggested_params["take_profit"]
                    if tp_price > 0:
                        take_profit_pct = (
                            abs(tp_price - current_price) / current_price * 100
                        )
                    else:
                        take_profit_pct = 2.0
                else:
                    take_profit_pct = 2.0
            else:
                # Use defaults
                take_profit_pct = 2.0
                stop_loss_pct = 1.0
        else:
            take_profit_pct = 1.0
            stop_loss_pct = 1.0

        # Create TradeAction
        return TradeAction(
            action=cast("Any", action),
            size_pct=min(size_pct, settings.trading.max_size_pct),  # Respect max size
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            rationale=llm_response.reasoning,
            leverage=settings.trading.leverage,
            reduce_only=False,
        )

    @staticmethod
    def create_trading_params() -> TradingParams:
        """Create TradingParams from settings."""
        return TradingParams(
            risk_per_trade=settings.trading.max_size_pct / 100,
            max_leverage=settings.trading.max_futures_leverage,
            stop_loss_pct=1.0,
            take_profit_pct=2.0,
            max_position_size=settings.trading.max_size_pct / 100,
        )


class FunctionalLLMStrategy:
    """Functional strategy implementation using LLM."""

    def __init__(
        self,
        model_provider: str | None = None,
        model_name: str | None = None,
        omnisearch_client: Any | None = None,
    ):
        self.config = LLMConfig(
            provider=LLMProvider(model_provider or settings.llm.provider),
            model=model_name or settings.llm.model_name,
            api_key=settings.llm.openai_api_key.get_secret_value(),
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
        self.omnisearch_client = omnisearch_client

    async def analyze_market_functional(self, market_state: MarketState) -> LLMResponse:
        """Execute the functional strategy on a market state."""
        try:
            # Fetch omnisearch data if client is available
            omnisearch_data = None
            if self.omnisearch_client:
                omnisearch_data = await self._fetch_omnisearch_data(market_state)
            
            # Create market context for LLM using the functional approach
            context = create_market_context(
                market_state=market_state,
                vumanchu_state=self._extract_vumanchu_state(market_state),
                recent_trades=[],  # Would be populated from trade history
            )
            
            # Add omnisearch data to context if available
            if omnisearch_data:
                context = self._enhance_context_with_omnisearch(context, omnisearch_data)

            # Generate trading prompt
            params = TypeConverter.create_trading_params()
            prompt = generate_trading_prompt(context, params)

            # Get LLM response (this would call actual LLM)
            response_text = await self._call_llm(prompt)

            # Parse LLM response
            llm_response = parse_llm_response(response_text, market_state)

            # Validate decision
            is_valid, validation_error = validate_llm_decision(
                llm_response, market_state, params
            )

            if not is_valid:
                logger.warning(f"Invalid LLM decision: {validation_error}")
                # Create a HOLD response
                llm_response = LLMResponse(
                    signal=Hold(reason=f"Invalid decision: {validation_error}"),
                    confidence=1.0,
                    reasoning=f"Invalid decision: {validation_error}",
                    risk_assessment={
                        "risk_level": "LOW",
                        "key_risks": ["Invalid decision"],
                        "mitigation": "Hold until valid signal",
                    },
                )

            # Adjust confidence based on market conditions
            return adjust_confidence_by_market_conditions(llm_response, market_state)

        except Exception as e:
            logger.exception(f"Error in functional LLM strategy: {e}")
            # Return safe HOLD response
            return LLMResponse(
                signal=Hold(reason=f"Strategy error: {e!s}"),
                confidence=1.0,
                reasoning=f"Strategy error: {e!s}",
                risk_assessment={
                    "risk_level": "HIGH",
                    "key_risks": ["Strategy execution error"],
                    "mitigation": "Hold position until error resolved",
                },
            )

    def _extract_vumanchu_state(self, market_state: MarketState):
        """Extract VuManChu state from market state indicators."""
        # This would extract the VuManChu state from the indicators
        # For now, return None to work with the create_market_context function
        return None

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        try:
            # Import OpenAI client
            from openai import AsyncOpenAI
            
            # Initialize OpenAI client with API key from config
            client = AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=30.0,
                max_retries=2,
            )
            
            # Prepare messages for chat completion
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert cryptocurrency futures trader. Respond only in valid JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call OpenAI API
            logger.debug(f"Calling {self.config.provider}/{self.config.model} with prompt length: {len(prompt)}")
            
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "text"},  # We parse JSON from text response
            )
            
            # Extract the response content
            llm_response = response.choices[0].message.content
            
            logger.debug(f"LLM response received, length: {len(llm_response) if llm_response else 0}")
            return llm_response or self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            # Return a safe fallback response on error
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Get a safe fallback response when LLM call fails."""
        return """
        MOMENTUM ANALYSIS:
        Unable to complete full market analysis due to technical issues.
        Defaulting to safe HOLD position to protect capital.

        JSON_DECISION:
        {
            "action": "HOLD",
            "confidence": 0.5,
            "reasoning": "Technical analysis unavailable, holding for safety",
            "risk_assessment": {
                "risk_level": "LOW",
                "key_risks": ["Analysis system temporarily unavailable"],
                "mitigation": "Wait for system recovery before trading"
            },
            "suggested_params": {
                "position_size": 0.0,
                "stop_loss": 0,
                "take_profit": 0
            }
        }
        """
    
    async def _fetch_omnisearch_data(self, market_state: MarketState) -> dict[str, Any] | None:
        """Fetch relevant data from omnisearch for market analysis."""
        try:
            symbol = market_state.symbol
            base_symbol = symbol.split("-")[0]  # Extract base symbol (e.g., BTC from BTC-USD)
            
            # Fetch multiple data points concurrently
            tasks = []
            
            # Crypto sentiment
            tasks.append(self.omnisearch_client.search_crypto_sentiment(symbol))
            
            # Financial news
            tasks.append(self.omnisearch_client.search_financial_news(
                f"{base_symbol} cryptocurrency news",
                limit=5,
                timeframe="24h"
            ))
            
            # NASDAQ sentiment (for correlation)
            tasks.append(self.omnisearch_client.search_nasdaq_sentiment())
            
            # Market correlation
            tasks.append(self.omnisearch_client.search_market_correlation(
                base_symbol, "QQQ", "30d"
            ))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            omnisearch_data = {
                "crypto_sentiment": results[0] if not isinstance(results[0], Exception) else None,
                "financial_news": results[1] if not isinstance(results[1], Exception) else [],
                "nasdaq_sentiment": results[2] if not isinstance(results[2], Exception) else None,
                "market_correlation": results[3] if not isinstance(results[3], Exception) else None,
            }
            
            logger.info(
                f"OmniSearch data fetched - Sentiment: {omnisearch_data['crypto_sentiment'].overall_sentiment if omnisearch_data['crypto_sentiment'] else 'N/A'}, "
                f"News items: {len(omnisearch_data['financial_news'])}, "
                f"Correlation: {omnisearch_data['market_correlation'].correlation_coefficient if omnisearch_data['market_correlation'] else 'N/A'}"
            )
            
            return omnisearch_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch omnisearch data: {e}")
            return None
    
    def _enhance_context_with_omnisearch(self, context: LLMContext, omnisearch_data: dict[str, Any]) -> LLMContext:
        """Enhance the LLM context with omnisearch data."""
        # Create a new context with additional omnisearch data
        enhanced_context = LLMContext(
            market_state=context.market_state,
            indicators=context.indicators,
            recent_trades=context.recent_trades,
            market_sentiment=self._extract_market_sentiment(omnisearch_data),
            news_events=self._extract_news_events(omnisearch_data),
        )
        
        # Add omnisearch-specific indicators
        enhanced_context.indicators["omnisearch"] = {
            "crypto_sentiment_score": omnisearch_data["crypto_sentiment"].sentiment_score if omnisearch_data.get("crypto_sentiment") else 0.0,
            "crypto_sentiment_label": omnisearch_data["crypto_sentiment"].overall_sentiment if omnisearch_data.get("crypto_sentiment") else "neutral",
            "nasdaq_sentiment_score": omnisearch_data["nasdaq_sentiment"].sentiment_score if omnisearch_data.get("nasdaq_sentiment") else 0.0,
            "market_correlation": omnisearch_data["market_correlation"].correlation_coefficient if omnisearch_data.get("market_correlation") else 0.0,
            "correlation_strength": omnisearch_data["market_correlation"].strength if omnisearch_data.get("market_correlation") else "weak",
        }
        
        return enhanced_context
    
    def _extract_market_sentiment(self, omnisearch_data: dict[str, Any]) -> dict[str, Any]:
        """Extract market sentiment from omnisearch data."""
        sentiment = {
            "overall": "neutral",
            "confidence": 0.5,
            "sources": [],
        }
        
        if omnisearch_data.get("crypto_sentiment"):
            crypto_sent = omnisearch_data["crypto_sentiment"]
            sentiment["overall"] = crypto_sent.overall_sentiment
            sentiment["confidence"] = crypto_sent.confidence
            sentiment["score"] = crypto_sent.sentiment_score
            sentiment["key_drivers"] = crypto_sent.key_drivers
            sentiment["risk_factors"] = crypto_sent.risk_factors
            
        return sentiment
    
    def _extract_news_events(self, omnisearch_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract relevant news events from omnisearch data."""
        news_events = []
        
        if omnisearch_data.get("financial_news"):
            for news_item in omnisearch_data["financial_news"][:3]:  # Top 3 news items
                event = {
                    "title": news_item.base_result.title,
                    "sentiment": news_item.sentiment,
                    "impact_level": news_item.impact_level,
                    "category": news_item.news_category,
                    "url": news_item.base_result.url,
                    "snippet": news_item.base_result.snippet,
                }
                news_events.append(event)
                
        return news_events


class LLMAgentAdapter:
    """
    Adapter that maintains the LLMAgent API while using functional strategies.

    This class provides exact compatibility with the original LLMAgent while
    using functional implementations internally.
    """

    def __init__(
        self,
        model_provider: str | None = None,
        model_name: str | None = None,
        omnisearch_client: Any | None = None,
    ):
        """
        Initialize the LLM agent adapter.

        Args:
            model_provider: LLM provider ('openai', 'ollama')
            model_name: Specific model to use
            omnisearch_client: Optional OmniSearch client for market intelligence
        """
        self.model_provider = model_provider or settings.llm.provider
        self.model_name = model_name or settings.llm.model_name
        self._omnisearch_client = omnisearch_client

        # Initialize functional strategy
        self._strategy = FunctionalLLMStrategy(model_provider, model_name)

        # Initialize type converter
        self._converter = TypeConverter()

        # Tracking for compatibility
        self._completion_count = 0

        logger.info(
            f"🔄 LLMAgentAdapter: Initialized functional strategy with {self.model_provider}:{self.model_name}"
        )

    async def analyze_market(self, market_state: MarketState) -> TradeAction:
        """
        Analyze market state and generate trading decision.

        This method maintains exact compatibility with the original LLMAgent
        while using functional strategies internally.

        Args:
            market_state: Complete market state including OHLCV and indicators

        Returns:
            TradeAction with decision and parameters
        """
        start_time = time.time()
        self._completion_count += 1

        try:
            # Execute functional strategy directly with MarketState
            llm_response = await self._strategy.analyze_market_functional(market_state)

            # Convert LLMResponse to TradeAction
            trade_action = self._converter.llm_response_to_trade_action(
                llm_response, market_state
            )

            # Log decision (maintaining compatibility)
            execution_time = time.time() - start_time
            logger.info(
                f"🤖 Functional LLM Strategy Decision: {trade_action.action} | "
                f"Execution time: {execution_time:.3f}s | "
                f"Confidence: {llm_response.confidence:.2f} | "
                f"Rationale: {trade_action.rationale}"
            )

            return trade_action

        except Exception as e:
            logger.exception(f"Error in functional LLM strategy adapter: {e}")
            # Return safe default action
            return TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=2.0,
                stop_loss_pct=1.0,
                leverage=1,
                rationale=f"Adapter error: {e!s}",
            )

    def is_available(self) -> bool:
        """Check if the LLM agent is available."""
        return True  # Functional strategies are always available

    def get_status(self) -> dict[str, Any]:
        """Get status information about the LLM agent."""
        return {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "strategy_type": "functional",
            "completion_count": self._completion_count,
            "omnisearch_enabled": self._omnisearch_client is not None,
            "available": True,
        }


class MemoryEnhancedLLMAgentAdapter:
    """
    Adapter that maintains the MemoryEnhancedLLMAgent API while using functional strategies.

    This class provides exact compatibility with the original MemoryEnhancedLLMAgent
    while using functional implementations internally.
    """

    def __init__(
        self,
        model_provider: str | None = None,
        model_name: str | None = None,
        memory_server: Any | None = None,
        omnisearch_client: Any | None = None,
    ):
        """
        Initialize the memory-enhanced LLM agent adapter.

        Args:
            model_provider: LLM provider ('openai', 'ollama')
            model_name: Specific model to use
            memory_server: MCP memory server instance
            omnisearch_client: Optional OmniSearch client for market intelligence
        """
        self.model_provider = model_provider or settings.llm.provider
        self.model_name = model_name or settings.llm.model_name
        self.memory_server = memory_server
        self._omnisearch_client = omnisearch_client

        # Memory availability check
        self._memory_available = memory_server is not None and settings.mcp.enabled

        # Initialize base functional strategy
        self._strategy = FunctionalLLMStrategy(
            model_provider, model_name, omnisearch_client
        )

        # Initialize type converter
        self._converter = TypeConverter()

        # Tracking for compatibility
        self._completion_count = 0

        # Memory context tracking (for compatibility with logging systems)
        self._last_memory_context: dict[str, Any] | None = None

        logger.info(
            f"🧠 MemoryEnhancedLLMAgentAdapter: Initialized functional strategy "
            f"with {self.model_provider}:{self.model_name} | "
            f"Memory: {'✅ enabled' if self._memory_available else '❌ disabled'}"
        )

    async def analyze_market(self, market_state: MarketState) -> TradeAction:
        """
        Analyze market state with memory-enhanced context.

        This method maintains exact compatibility with the original MemoryEnhancedLLMAgent
        while using functional strategies internally.

        Args:
            market_state: Complete market state including OHLCV and indicators

        Returns:
            TradeAction with decision and parameters
        """
        start_time = time.time()
        self._completion_count += 1

        try:
            # If memory not available, fall back to base implementation
            if not self._memory_available:
                return await self._analyze_without_memory(market_state)

            # Retrieve relevant memories (simplified for now)
            similar_experiences = await self._retrieve_relevant_memories(market_state)

            # Execute functional strategy with memory context
            llm_response = await self._strategy.analyze_market_functional(market_state)

            # Enhance response with memory insights (if available)
            if similar_experiences:
                enhanced_response = self._enhance_response_with_memory(
                    llm_response, similar_experiences
                )
            else:
                enhanced_response = llm_response

            # Convert to TradeAction
            trade_action = self._converter.llm_response_to_trade_action(
                enhanced_response, market_state
            )

            # Store memory context for external access
            self._last_memory_context = {
                "experiences": [
                    {
                        "experience_id": f"exp_{i}",
                        "action": "UNKNOWN",  # Would be extracted from actual experience
                        "outcome": None,
                        "patterns": [],
                        "timestamp": None,
                    }
                    for i, _ in enumerate(similar_experiences)
                ],
                "pattern_insights": "No pattern insights available yet",
                "sentiment_context": "No sentiment context available yet",
                "memory_context_formatted": f"Found {len(similar_experiences)} similar experiences",
            }

            # Log decision with memory info
            execution_time = time.time() - start_time
            logger.info(
                f"🧠 Memory-Enhanced Functional Strategy Decision: {trade_action.action} | "
                f"Execution time: {execution_time:.3f}s | "
                f"Confidence: {enhanced_response.confidence:.2f} | "
                f"Similar experiences: {len(similar_experiences)} | "
                f"Rationale: {trade_action.rationale}"
            )

            return trade_action

        except Exception as e:
            logger.exception(
                f"Error in memory-enhanced functional strategy adapter: {e}"
            )
            # Return safe default action
            return TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=2.0,
                stop_loss_pct=1.0,
                leverage=1,
                rationale=f"Memory-enhanced adapter error: {e!s}",
            )

    async def _analyze_without_memory(self, market_state: MarketState) -> TradeAction:
        """Fallback to standard functional strategy when memory is not available."""
        llm_response = await self._strategy.analyze_market_functional(market_state)
        return self._converter.llm_response_to_trade_action(llm_response, market_state)

    async def _retrieve_relevant_memories(self, market_state: MarketState) -> list[Any]:
        """Retrieve relevant past trading experiences."""
        if not self.memory_server:
            return []

        try:
            # Simplified memory retrieval - in a full implementation this would
            # use the actual memory server query functionality
            logger.debug(
                f"Querying memory server for similar experiences to {market_state.symbol}"
            )

            # For now, return empty list - would be implemented with actual memory queries
            return []

        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
            return []

    def _enhance_response_with_memory(
        self, response: LLMResponse, experiences: list[Any]
    ) -> LLMResponse:
        """Enhance LLM response with insights from past experiences."""

        if not experiences:
            return response

        # Simple enhancement - adjust confidence based on historical success
        # In a full implementation, this would analyze the experiences and
        # modify the response accordingly
        enhanced_reasoning = (
            f"{response.reasoning} "
            f"[Enhanced with insights from {len(experiences)} similar past experiences]"
        )

        return LLMResponse(
            signal=response.signal,
            confidence=response.confidence,
            reasoning=enhanced_reasoning,
            risk_assessment=response.risk_assessment,
            suggested_params=response.suggested_params,
        )

    def is_available(self) -> bool:
        """Check if the LLM agent is available."""
        return True  # Functional strategies are always available

    def get_status(self) -> dict[str, Any]:
        """Get status including memory availability."""
        return {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "strategy_type": "functional_memory_enhanced",
            "completion_count": self._completion_count,
            "omnisearch_enabled": self._omnisearch_client is not None,
            "memory_enabled": self._memory_available,
            "memory_server_connected": (
                self.memory_server is not None if self.memory_server else False
            ),
            "available": True,
        }
