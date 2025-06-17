"""LLM Context Provider for adaptive trading strategies.

This is a placeholder implementation to prevent import errors.
The full implementation will be added later.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any


class MarketContextType(Enum):
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    BREAKOUT_POTENTIAL = "breakout_potential"
    CONSOLIDATION = "consolidation"
    UNCERTAIN = "uncertain"


class StrategyRecommendation(Enum):
    MOMENTUM_AGGRESSIVE = "momentum_aggressive"
    MOMENTUM_CONSERVATIVE = "momentum_conservative"
    SCALPING_ACTIVE = "scalping_active"
    SCALPING_SELECTIVE = "scalping_selective"
    DEFENSIVE = "defensive"
    HOLD = "hold"


@dataclass
class LLMTradingContext:
    market_regime: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    volume_analysis: Dict[str, Any]
    microstructure_analysis: Dict[str, Any]
    current_strategy: Dict[str, Any]
    strategy_performance: Dict[str, Any]
    strategy_recommendation: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    position_context: Dict[str, Any]
    recent_performance: Dict[str, Any]
    market_adaptation: Dict[str, Any]
    momentum_signals: Dict[str, Any]
    scalping_signals: Dict[str, Any]
    combined_signals: Dict[str, Any]
    market_narrative: str
    strategy_rationale: str
    risk_considerations: str
    opportunity_assessment: str


class LLMContextProvider:
    """Placeholder LLM context provider."""
    
    def __init__(self, regime_detector=None, strategy_manager=None, 
                 momentum_strategy=None, scalping_strategy=None):
        self.regime_detector = regime_detector
        self.strategy_manager = strategy_manager
        self.momentum_strategy = momentum_strategy
        self.scalping_strategy = scalping_strategy
        
    async def provide_llm_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide basic LLM context."""
        return {
            'llm_prompt': "Market analysis context not yet fully implemented.",
            'context_data': {},
            'decision_framework': {},
            'confidence_metrics': {}
        }