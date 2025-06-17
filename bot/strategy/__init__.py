"""Trading strategy and decision-making modules."""

from .core import CoreStrategy
from .llm_agent import LLMAgent
from .adaptive_strategy_manager import (
    AdaptiveStrategyManager,
    TradingStrategy,
    StrategyState,
    StrategyConfig,
    StrategySelector,
    TransitionManager,
    StrategyExecutor,
    StrategyPerformanceTracker,
    StrategyContextProvider
)
from .momentum_strategy import (
    MomentumStrategyExecutor,
    MomentumConfig,
    MomentumSignalType,
    MomentumSignalStrength,
    create_momentum_strategy
)
from .scalping_strategy import (
    ScalpingStrategy,
    ScalpingConfig,
    ScalpingSignalType,
    ScalpingTiming,
    ScalpingState,
    ScalpingSignal,
    create_scalping_strategy
)
try:
    from .market_regime_detector import (
        MarketRegimeDetector,
        MarketRegime,
        TrendAnalyzer,
        VolatilityAnalyzer,
        VolumeAnalyzer,
        MicrostructureAnalyzer
    )
except ImportError:
    pass

try:
    from .llm_context_provider import (
        LLMContextProvider,
        MarketContextType,
        StrategyRecommendation,
        LLMTradingContext
    )
except ImportError:
    pass

__all__ = [
    "LLMAgent", 
    "CoreStrategy",
    "AdaptiveStrategyManager",
    "TradingStrategy",
    "StrategyState", 
    "StrategyConfig",
    "StrategySelector",
    "TransitionManager",
    "StrategyExecutor",
    "StrategyPerformanceTracker",
    "StrategyContextProvider",
    "MomentumStrategyExecutor",
    "MomentumConfig",
    "MomentumSignalType",
    "MomentumSignalStrength",
    "create_momentum_strategy",
    "ScalpingStrategy",
    "ScalpingConfig",
    "ScalpingSignalType",
    "ScalpingTiming",
    "ScalpingState",
    "ScalpingSignal",
    "create_scalping_strategy",
    "MarketRegimeDetector",
    "MarketRegime",
    "TrendAnalyzer",
    "VolatilityAnalyzer",
    "VolumeAnalyzer",
    "MicrostructureAnalyzer",
    "LLMContextProvider",
    "MarketContextType",
    "StrategyRecommendation",
    "LLMTradingContext"
]
