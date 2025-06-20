"""
AI Trading Bot - A LangChain-powered crypto futures trading bot.

This package provides a complete trading bot implementation with:
- Real-time market data ingestion from Coinbase
- VuManChu Cipher A & B technical indicators
- LLM-powered trading decisions via LangChain
- Comprehensive risk management and validation
- Backtesting and performance analysis capabilities
"""

__version__ = "0.1.0"
__author__ = "Architect Roo"
__description__ = "AI-powered crypto futures trading bot"

# Core components
from .config import settings

# Conditional imports for optional components
try:
    # Import backtest components when pandas is available
    from .backtest.engine import BacktestEngine, BacktestResults, BacktestTrade
except ImportError:
    # Create dummy classes if pandas not available
    class _BacktestEngine:
        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "Backtesting requires pandas. Install with: pip install pandas"
            )

    class _BacktestResults:
        pass

    class _BacktestTrade:
        pass

    # Assign dummy classes to expected names
    BacktestEngine = _BacktestEngine  # type: ignore[assignment]
    BacktestResults = _BacktestResults  # type: ignore[assignment]
    BacktestTrade = _BacktestTrade  # type: ignore[assignment]

    _BACKTEST_AVAILABLE = False

# Data and indicators
from .data.dominance import DominanceData, DominanceDataProvider
from .data.market import MarketDataProvider

# Exchange integration
from .exchange.coinbase import CoinbaseClient
from .indicators.vumanchu import CipherA, CipherB, VuManChuIndicators
from .risk import RiskManager
from .strategy.core import CoreStrategy

# Strategy components
from .strategy.llm_agent import LLMAgent
from .trading_types import MarketData, MarketState, Position, TradeAction

# Training and RAG
from .train.reader import RAGReader
from .validator import TradeValidator

__all__ = [
    "BacktestEngine",
    "BacktestResults",
    "BacktestTrade",
    "CipherA",
    "CipherB",
    "CoinbaseClient",
    "CoreStrategy",
    "DominanceData",
    "DominanceDataProvider",
    "LLMAgent",
    "MarketData",
    "MarketDataProvider",
    "MarketState",
    "Position",
    "RAGReader",
    "RiskManager",
    "TradeAction",
    "TradeValidator",
    "VuManChuIndicators",
    "settings",
]
