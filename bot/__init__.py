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
# Backtesting
from .backtest.engine import BacktestEngine
from .config import settings

# Data and indicators
from .data.market import MarketDataProvider

# Exchange integration
from .exchange.coinbase import CoinbaseClient
from .indicators.vumanchu import CipherA, CipherB, VuManChuIndicators
from .risk import RiskManager
from .strategy.core import CoreStrategy

# Strategy components
from .strategy.llm_agent import LLMAgent

# Training and RAG
from .train.reader import RAGReader
from .types import MarketData, MarketState, Position, TradeAction
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
    "VuManChuIndicators",
    "CipherA",
    "CipherB",
    "CoinbaseClient",
    "BacktestEngine",
    "RAGReader",
]
