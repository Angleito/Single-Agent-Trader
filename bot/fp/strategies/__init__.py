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

# Import FunctionalLLMStrategy from strategy_adapter
try:
    from ..adapters.strategy_adapter import FunctionalLLMStrategy
except ImportError:
    # Define a placeholder if import fails
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
    # Base components
    "StrategyConfig",
    "StrategyResult",
    "BaseStrategy",
    "StrategyComposition",
    # LLM strategies
    "LLMConfig",
    "FunctionalLLMStrategy",
    # Market making
    "MarketMakingConfig",
    "FunctionalMarketMakingStrategy",
    # Risk management
    "RiskConfig",
    "RiskResult",
    "FunctionalRiskManager",
    # Execution
    "ExecutionConfig",
    "ExecutionResult",
    "FunctionalExecutionEngine",
    # Filters
    "FilterConfig",
    "FilterResult",
    "StrategyFilter",
]
