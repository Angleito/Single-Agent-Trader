"""
Functional Programming Trading Strategies

This module provides pure functional implementations of trading strategies
using IOEither monads and functional composition.
"""

# Base strategy components
from .base import (
    BaseStrategy,
    StrategyComposition,
    StrategyConfig,
    StrategyResult,
)

# Execution strategies
from .execution import (
    ExecutionConfig,
    ExecutionResult,
    FunctionalExecutionEngine,
)

# Strategy filters
# from .filters import (
#     FilterConfig,
#     FilterResult,
#     StrategyFilter,
# )
# LLM strategies
from .llm_functional import (
    LLMConfig,
    LLMProvider,
    LLMResponse,
    adjust_confidence_by_market_conditions,
    create_market_context,
    generate_trading_prompt,
    parse_llm_response,
    validate_llm_decision,
)


# Lazy loading for FunctionalLLMStrategy to avoid circular imports
def get_functional_llm_strategy():
    """Get FunctionalLLMStrategy class (lazy loaded to avoid circular imports)"""
    try:
        from bot.fp.adapters.strategy_adapter import FunctionalLLMStrategy

        return FunctionalLLMStrategy
    except ImportError:
        # Return placeholder if import fails
        return None


# Runtime fallback for compatibility
FunctionalLLMStrategy = None

# Market making strategies
from .market_making import (
    FunctionalMarketMakingStrategy,
    MarketMakingConfig,
)

# Risk management
from .risk_management import (
    FunctionalRiskManager,
    RiskConfig,
    RiskResult,
)

__all__ = [
    "BaseStrategy",
    # Execution
    "ExecutionConfig",
    "ExecutionResult",
    # Filters
    "FilterConfig",
    "FilterResult",
    "FunctionalExecutionEngine",
    "FunctionalMarketMakingStrategy",
    "FunctionalRiskManager",
    # LLM strategies
    "LLMConfig",
    # Market making
    "MarketMakingConfig",
    # Risk management
    "RiskConfig",
    "RiskResult",
    "StrategyComposition",
    # Base components
    "StrategyConfig",
    "StrategyFilter",
    "StrategyResult",
    "get_functional_llm_strategy",  # Lazy loaded to avoid circular imports
]
