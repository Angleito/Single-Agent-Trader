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
    # Temporarily disable backtest import to debug async issue
    # from .backtest.engine import BacktestEngine, BacktestResults, BacktestTrade
    raise ImportError("Temporarily disabled for debugging")

    _BACKTEST_AVAILABLE = True
except ImportError:
    # Create dummy classes if pandas not available
    class _BacktestEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Backtesting requires pandas. Install with: pip install pandas"
            )

    class _BacktestResults:
        pass

    class _BacktestTrade:
        pass

    # Assign dummy classes to expected names
    BacktestEngine = _BacktestEngine
    BacktestResults = _BacktestResults
    BacktestTrade = _BacktestTrade

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
    "settings",
    "TradeAction",
    "Position",
    "MarketData",
    "MarketState",
    "TradeValidator",
    "RiskManager",
    "LLMAgent",
    "CoreStrategy",
    "MarketDataProvider",
    "DominanceData",
    "DominanceDataProvider",
    "VuManChuIndicators",
    "CipherA",
    "CipherB",
    "CoinbaseClient",
    "BacktestEngine",
    "BacktestResults",
    "BacktestTrade",
    "RAGReader",
]
