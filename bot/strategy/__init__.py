"""Trading strategy and decision-making modules."""

from .core import CoreStrategy
from .llm_agent import LLMAgent

# Optional imports for memory-enhanced features
try:
    from .memory_enhanced_agent import MemoryEnhancedLLMAgent
except ImportError:
    MemoryEnhancedLLMAgent = None

__all__ = [
    "LLMAgent",
    "CoreStrategy",
]

# Add MemoryEnhancedLLMAgent to __all__ if available
if MemoryEnhancedLLMAgent is not None:
    __all__.append("MemoryEnhancedLLMAgent")
