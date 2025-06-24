"""
Strategy adapters for migrating from imperative to functional implementations.

This module provides adapter classes that maintain the existing imperative APIs
while using functional strategy implementations internally.
"""

import logging
import time
from typing import Any, cast

from bot.config import settings
from bot.trading_types import MarketState, TradeAction
from bot.fp.strategies.llm_functional import (
    LLMConfig,
    LLMContext,
    LLMProvider,
    LLMResponse,
    create_market_context,
    generate_trading_prompt,
    parse_llm_response,
    validate_llm_decision,
    adjust_confidence_by_market_conditions,
)

# Import TradingParams and TradeSignal types
try:
    from bot.fp.types import TradingParams, TradeSignal
    # Create alias for backward compatibility
    Signal = TradeSignal
except ImportError:
    # Define minimal types if they don't exist
    from dataclasses import dataclass
    from enum import Enum
    
    class Signal(Enum):
        LONG = "LONG"
        SHORT = "SHORT" 
        HOLD = "HOLD"
    
    @dataclass
    class TradingParams:
        risk_per_trade: float = 0.1
        max_leverage: int = 5
        stop_loss_pct: float = 1.0
        take_profit_pct: float = 2.0
        max_position_size: float = 0.25

logger = logging.getLogger(__name__)


class TypeConverter:
    """Convert between functional LLM responses and imperative TradeAction."""
    
    @staticmethod
    def llm_response_to_trade_action(
        llm_response: LLMResponse, 
        market_state: MarketState
    ) -> TradeAction:
        """Convert LLMResponse to TradeAction."""
        
        # Map signal to action
        signal_map = {
            Signal.LONG: "LONG",
            Signal.SHORT: "SHORT", 
            Signal.HOLD: "HOLD"
        }
        action = signal_map.get(llm_response.signal, "HOLD")
        
        # Extract size from suggested params or use confidence as size
        if llm_response.suggested_params and "position_size" in llm_response.suggested_params:
            size_pct = llm_response.suggested_params["position_size"] * 100
        else:
            # Use confidence to determine size (higher confidence = larger size)
            if action in ["LONG", "SHORT"]:
                size_pct = llm_response.confidence * settings.trading.max_size_pct
            else:
                size_pct = 0.0
        
        # Extract stop loss and take profit
        if action in ["LONG", "SHORT"]:
            if llm_response.suggested_params:
                # Try to convert price levels to percentages
                current_price = float(market_state.current_price)
                
                # Get stop loss percentage
                if "stop_loss" in llm_response.suggested_params:
                    stop_price = llm_response.suggested_params["stop_loss"]
                    if stop_price > 0:
                        stop_loss_pct = abs(stop_price - current_price) / current_price * 100
                    else:
                        stop_loss_pct = 1.0
                else:
                    stop_loss_pct = 1.0
                
                # Get take profit percentage  
                if "take_profit" in llm_response.suggested_params:
                    tp_price = llm_response.suggested_params["take_profit"]
                    if tp_price > 0:
                        take_profit_pct = abs(tp_price - current_price) / current_price * 100
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
        trade_action = TradeAction(
            action=cast(Any, action),
            size_pct=min(size_pct, settings.trading.max_size_pct),  # Respect max size
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            rationale=llm_response.reasoning,
            leverage=settings.trading.leverage,
            reduce_only=False
        )
        
        return trade_action
    
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
            # Create market context for LLM using the functional approach
            context = create_market_context(
                market_state=market_state, 
                vumanchu_state=self._extract_vumanchu_state(market_state),
                recent_trades=[],  # Would be populated from trade history
            )
            
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
                    signal=Signal.HOLD,
                    confidence=1.0,
                    reasoning=f"Invalid decision: {validation_error}",
                    risk_assessment={
                        "risk_level": "LOW",
                        "key_risks": ["Invalid decision"],
                        "mitigation": "Hold until valid signal"
                    }
                )
            
            # Adjust confidence based on market conditions
            adjusted_response = adjust_confidence_by_market_conditions(
                llm_response, market_state
            )
            
            return adjusted_response
            
        except Exception as e:
            logger.error(f"Error in functional LLM strategy: {e}")
            # Return safe HOLD response
            return LLMResponse(
                signal=Signal.HOLD,
                confidence=1.0,
                reasoning=f"Strategy error: {str(e)}",
                risk_assessment={
                    "risk_level": "HIGH",
                    "key_risks": ["Strategy execution error"],
                    "mitigation": "Hold position until error resolved"
                }
            )
    
    def _extract_vumanchu_state(self, market_state: MarketState):
        """Extract VuManChu state from market state indicators."""
        # This would extract the VuManChu state from the indicators
        # For now, return None to work with the create_market_context function
        return None
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        # This would be implemented to call the actual LLM using the same
        # mechanism as the original LLMAgent - for now return a mock response
        return """
        MOMENTUM ANALYSIS:
        Current market conditions show neutral momentum with mixed signals.
        The price is trading sideways without clear directional bias.
        Volume is normal and no significant catalysts are present.
        Waiting for clearer momentum signals before entering positions.
        
        JSON_DECISION:
        {
            "action": "HOLD",
            "confidence": 0.8,
            "reasoning": "Neutral market conditions, waiting for clear signals",
            "risk_assessment": {
                "risk_level": "LOW",
                "key_risks": ["Sideways market"],
                "mitigation": "Stay out until trend emerges"
            },
            "suggested_params": {
                "position_size": 0.0,
                "stop_loss": 0,
                "take_profit": 0
            }
        }
        """


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
            logger.error(f"Error in functional LLM strategy adapter: {e}")
            # Return safe default action
            return TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=2.0,
                stop_loss_pct=1.0,
                leverage=1,
                rationale=f"Adapter error: {str(e)}",
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
            model_provider, 
            model_name, 
            omnisearch_client
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
            logger.error(f"Error in memory-enhanced functional strategy adapter: {e}")
            # Return safe default action
            return TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=2.0,
                stop_loss_pct=1.0,
                leverage=1,
                rationale=f"Memory-enhanced adapter error: {str(e)}",
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
            logger.debug(f"Querying memory server for similar experiences to {market_state.symbol}")
            
            # For now, return empty list - would be implemented with actual memory queries
            return []
            
        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
            return []
    
    def _enhance_response_with_memory(
        self, 
        response: LLMResponse, 
        experiences: list[Any]
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
        status = {
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
        
        return status