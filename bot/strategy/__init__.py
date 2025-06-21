"""Trading strategy and decision-making modules."""

from .core import CoreStrategy
from .llm_agent import LLMAgent
from .market_making_strategy import MarketMakingStrategy

# Optional imports for memory-enhanced features
try:
    from .memory_enhanced_agent import MemoryEnhancedLLMAgent
except ImportError:
    MemoryEnhancedLLMAgent = None  # type: ignore[assignment]

__all__ = [
    "CoreStrategy",
    "LLMAgent",
    "MarketMakingStrategy",
]

# Add MemoryEnhancedLLMAgent to __all__ if available
if MemoryEnhancedLLMAgent is not None:
    __all__.append("MemoryEnhancedLLMAgent")
