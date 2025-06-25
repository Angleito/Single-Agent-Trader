"""
AI Trading Bot - A LangChain-powered crypto futures trading bot.

This package provides a complete trading bot implementation with:
- Real-time market data ingestion from Coinbase and Bluefin
- VuManChu Cipher A & B technical indicators
- LLM-powered trading decisions via LangChain
- Comprehensive risk management and validation
- Backtesting and performance analysis capabilities
"""

import logging
import warnings
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__author__ = "Architect Roo"
__description__ = "AI-powered crypto futures trading bot"

# Set up logger for initialization warnings
logger = logging.getLogger(__name__)

# Suppress warnings during import to prevent startup noise
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Core components - these are required and will raise if missing
    try:
        from .config import settings
    except ImportError:
        logger.exception("Failed to import core configuration")
        logger.exception("This is a critical dependency - bot cannot start without it")
        raise

# Feature availability flags
_BACKTEST_AVAILABLE = False
_EXCHANGE_COMPONENTS_AVAILABLE = False
_INDICATOR_COMPONENTS_AVAILABLE = False
_LLM_COMPONENTS_AVAILABLE = False
_DATA_COMPONENTS_AVAILABLE = False


# Conditional imports for optional components with detailed error handling
def _create_import_fallback(component_name: str, missing_deps: list[str]) -> type:
    """Create a fallback class for missing optional dependencies."""

    class _MissingComponent:
        def __init__(self, *_args, **_kwargs):
            deps_str = ", ".join(missing_deps)
            raise ImportError(
                f"{component_name} requires missing dependencies: {deps_str}\n"
                f"Install with: pip install {' '.join(missing_deps)}"
            )

    return _MissingComponent


# Backtest components (requires pandas, numpy for heavy analysis)
try:
    from .backtest.engine import BacktestEngine, BacktestResults, BacktestTrade

    _BACKTEST_AVAILABLE = True
    logger.debug("Backtest components loaded successfully")
except ImportError as e:
    logger.warning(
        "Backtest components unavailable - backtesting will be disabled: %s", e
    )
    # Create informative dummy classes
    BacktestEngine = _create_import_fallback("BacktestEngine", ["pandas", "numpy"])  # type: ignore[assignment]
    BacktestResults = _create_import_fallback("BacktestResults", ["pandas", "numpy"])  # type: ignore[assignment]
    BacktestTrade = _create_import_fallback("BacktestTrade", ["pandas", "numpy"])  # type: ignore[assignment]

# Data and indicators with fallback handling
try:
    from .data.dominance import DominanceData, DominanceDataProvider
    from .data.market import MarketDataProvider

    _DATA_COMPONENTS_AVAILABLE = True
    logger.debug("Data components loaded successfully")
except ImportError as e:
    logger.warning("Data components partially unavailable: %s", e)
    # Try to import what we can
    try:
        from .data.market import MarketDataProvider

        DominanceData = _create_import_fallback("DominanceData", ["requests"])  # type: ignore[assignment]
        DominanceDataProvider = _create_import_fallback(
            "DominanceDataProvider", ["requests"]
        )  # type: ignore[assignment]
    except ImportError:
        MarketDataProvider = _create_import_fallback(
            "MarketDataProvider", ["websockets", "aiohttp"]
        )  # type: ignore[assignment]
        DominanceData = _create_import_fallback("DominanceData", ["requests"])  # type: ignore[assignment]
        DominanceDataProvider = _create_import_fallback(
            "DominanceDataProvider", ["requests"]
        )  # type: ignore[assignment]

# Exchange integration with graceful degradation
try:
    from .exchange.coinbase import CoinbaseClient

    _EXCHANGE_COMPONENTS_AVAILABLE = True
    logger.debug("Exchange components loaded successfully")
except ImportError as e:
    logger.warning("Exchange components unavailable: %s", e)
    CoinbaseClient = _create_import_fallback("CoinbaseClient", ["coinbase-advanced-py"])  # type: ignore[assignment]

# Technical indicators
try:
    from .indicators.vumanchu import CipherA, CipherB, VuManChuIndicators

    _INDICATOR_COMPONENTS_AVAILABLE = True
    logger.debug("Indicator components loaded successfully")
except ImportError as e:
    logger.warning("Indicator components unavailable: %s", e)
    CipherA = _create_import_fallback("CipherA", ["pandas", "numpy", "ta"])  # type: ignore[assignment]
    CipherB = _create_import_fallback("CipherB", ["pandas", "numpy", "ta"])  # type: ignore[assignment]
    VuManChuIndicators = _create_import_fallback(
        "VuManChuIndicators", ["pandas", "numpy", "ta"]
    )  # type: ignore[assignment]

# Risk management and core strategy
try:
    from .risk import RiskManager
    from .strategy.core import CoreStrategy
except ImportError:
    logger.exception("Failed to import critical trading components")
    RiskManager = _create_import_fallback("RiskManager", ["decimal"])  # type: ignore[assignment]
    CoreStrategy = _create_import_fallback("CoreStrategy", ["abc"])  # type: ignore[assignment]

# LLM Strategy components
try:
    from .strategy.llm_agent import LLMAgent

    _LLM_COMPONENTS_AVAILABLE = True
    logger.debug("LLM components loaded successfully")
except ImportError as e:
    logger.warning("LLM components unavailable - AI trading will be disabled: %s", e)
    LLMAgent = _create_import_fallback("LLMAgent", ["langchain", "openai"])  # type: ignore[assignment]

# Trading types (core data structures)
try:
    from .trading_types import MarketData, MarketState, Position, TradeAction
except ImportError:
    logger.exception("Failed to import core trading types")
    raise  # These are critical

# Training and RAG (optional for enhanced features)
try:
    from .train.reader import RAGReader
except ImportError as e:
    logger.info("RAG reader unavailable - enhanced training features disabled: %s", e)
    RAGReader = _create_import_fallback("RAGReader", ["langchain", "chromadb"])  # type: ignore[assignment]

# Validator (critical for safe trading)
try:
    from .validator import TradeValidator
except ImportError:
    logger.exception("Failed to import trade validator - this is critical for safety")
    raise


# Startup diagnostics function
def get_startup_diagnostics() -> dict[str, Any]:
    """Get comprehensive startup diagnostics for troubleshooting."""
    return {
        "version": __version__,
        "components": {
            "backtest_available": _BACKTEST_AVAILABLE,
            "exchange_components_available": _EXCHANGE_COMPONENTS_AVAILABLE,
            "indicator_components_available": _INDICATOR_COMPONENTS_AVAILABLE,
            "llm_components_available": _LLM_COMPONENTS_AVAILABLE,
            "data_components_available": _DATA_COMPONENTS_AVAILABLE,
        },
        "core_imports_successful": True,  # If we get here, core imports worked
    }


def check_startup_health() -> tuple[bool, list[str]]:
    """Check startup health and return issues."""
    issues = []
    healthy = True

    if not _LLM_COMPONENTS_AVAILABLE:
        issues.append("LLM components unavailable - AI trading disabled")
        healthy = False

    if not _EXCHANGE_COMPONENTS_AVAILABLE:
        issues.append("Exchange components unavailable - trading disabled")
        healthy = False

    if not _INDICATOR_COMPONENTS_AVAILABLE:
        issues.append("Technical indicators unavailable - analysis limited")

    if not _BACKTEST_AVAILABLE:
        issues.append("Backtesting unavailable - install pandas/numpy for backtesting")

    return healthy, issues


__all__ = [
    # Component classes (may be fallback classes if dependencies missing)
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
    "check_startup_health",
    "get_startup_diagnostics",
    # Core exports
    "settings",
]
