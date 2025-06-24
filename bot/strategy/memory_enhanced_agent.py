"""
Functional replacement for MemoryEnhancedLLMAgent using adapter pattern.

This module provides a drop-in replacement for the original MemoryEnhancedLLMAgent
that uses functional strategies internally while maintaining exact API compatibility.
"""

# Import the functional adapter that maintains exact compatibility
from bot.fp.adapters.strategy_adapter import MemoryEnhancedLLMAgentAdapter

# Export MemoryEnhancedLLMAgent as an alias to the functional adapter
MemoryEnhancedLLMAgent = MemoryEnhancedLLMAgentAdapter

# Maintain compatibility with any direct imports
__all__ = ["MemoryEnhancedLLMAgent"]

# This replacement maintains:
# 1. Exact same constructor signature: MemoryEnhancedLLMAgent(model_provider, model_name, memory_server, omnisearch_client)
# 2. Exact same method signature: async def analyze_market(market_state: MarketState) -> TradeAction
# 3. Exact same is_available() and get_status() methods
# 4. All memory enhancement functionality through functional implementation
# 5. Compatibility with _last_memory_context attribute access

# Usage remains identical:
# agent = MemoryEnhancedLLMAgent(
#     model_provider="openai",
#     model_name="gpt-4",
#     memory_server=memory_server,
#     omnisearch_client=omnisearch_client
# )
# action = await agent.analyze_market(market_state)
# memory_context = agent._last_memory_context  # Still works
