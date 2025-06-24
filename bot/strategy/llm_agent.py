"""
Functional replacement for LLMAgent using adapter pattern.

This module provides a drop-in replacement for the original LLMAgent
that uses functional strategies internally while maintaining exact API compatibility.
"""

# Import the functional adapter that maintains exact compatibility
from bot.fp.adapters.strategy_adapter import LLMAgentAdapter

# Export LLMAgent as an alias to the functional adapter
LLMAgent = LLMAgentAdapter

# Maintain compatibility with any direct imports
__all__ = ["LLMAgent"]

# This replacement maintains:
# 1. Exact same constructor signature: LLMAgent(model_provider, model_name, omnisearch_client)
# 2. Exact same method signature: async def analyze_market(market_state: MarketState) -> TradeAction
# 3. Exact same is_available() and get_status() methods
# 4. All original functionality through functional implementation

# Usage remains identical:
# agent = LLMAgent(model_provider="openai", model_name="gpt-4")
# action = await agent.analyze_market(market_state)