"""Functional programming types for the trading bot.

This module exports functional types used throughout the trading system,
providing immutable data structures and algebraic data types for type-safe
trading operations.
"""

# IMPORTANT: Ensure Python stdlib types module is accessible first
# Prevent circular import conflict with Python's types module
import sys

if "types" not in sys.modules:
    import types as _stdlib_types

# Base types and utilities
# Balance validation types
from .balance_validation import (
    BalanceRange,
    BalanceValidationConfig,
    BalanceValidationError,
    BalanceValidationResult,
    BalanceValidationType,
    ComprehensiveBalanceValidation,
    TradeAffordabilityCheck,
)
from .base import (
    Maybe,
    Money,
    Nothing,
    Percentage,
    Some,
    Symbol,
    TimeInterval,
    TimeIntervalUnit,
    Timestamp,
    TradingMode,
)

# Configuration types
from .config import (
    AccountType,
    APIKey,
    BacktestConfig,
    BinanceExchangeConfig,
    BluefinExchangeConfig,
    CoinbaseExchangeConfig,
    Config,
    Environment,
    ExchangeConfig,
    ExchangeType,
    FeatureFlags,
    FeeStructure,
    LLMStrategyConfig,
    LogLevel,
    MeanReversionStrategyConfig,
    MomentumStrategyConfig,
    PrivateKey,
    RateLimits,
    StrategyConfig,
    SystemConfig,
)

# Effect system types
from .effects import (
    IO,
    CancelResult,
    DbQuery,
    Effect,
    HttpRequest,
    Log,
    PositionUpdate,
    RateLimit,
    ReadFile,
    RetryPolicy,
    WebSocketConnection,
    WriteFile,
)

# Event sourcing types
from .events import (
    EVENT_REGISTRY,
    AlertLevel,
    AlertTriggered,
    AuditTrailEntry,
    ConfigurationChanged,
    ErrorOccurred,
    EventMetadata,
    MarketDataReceived,
    NotificationChannel,
    NotificationSent,
    OrderCancelled,
    OrderFilled,
    OrderPlaced,
    PositionClosed,
    PositionOpened,
    StrategySignal,
    SystemComponent,
    SystemPerformanceMetric,
    TradingEvent,
)

# Indicator types
from .indicators import (
    BollingerBandsResult,
    CandlePattern,
    CompositeSignal,
    DiamondPattern,
    DivergencePattern,
    IndicatorConfig,
    IndicatorHistory,
    IndicatorResult,
    IndicatorSet,
    MACDResult,
    MarketStructure,
    MovingAverageResult,
    ROCResult,
    RSIResult,
    SignalHistory,
    SignalSet,
    StochasticResult,
    TimeSeries,
    VolumeProfile,
    VuManchuResult,
    VuManchuSet,
    VuManchuSignalSet,
    VuManchuState,  # Alias for backward compatibility
    YellowCrossSignal,
)

# Learning and memory types
from .learning import (
    ExperienceId,
    LearningInsight,
    MarketSnapshot,
    MemoryQueryFP,
    MemoryStorage,
    PatternStatistics,
    PatternTag,
    TradingExperienceFP,
    TradingOutcome,
)

# Market data types
from .market import (
    OHLCV,
    AggregatedData,
    # OHLCV and market data types
    Candle,
    ConnectionState,
    ConnectionStatus,
    DataQuality,
    # Type aliases
    MarketDataMessage,
    MarketDataStream,
    MarketSnapshot,
    OrderBook,
    OrderBookMessage,
    PriceData,
    RealtimeUpdate,
    StreamingData,
    # WebSocket and streaming types
    Subscription,
    Ticker,
    TickerMessage,
    Trade,
    TradeMessage,
    WebSocketMessage,
)

# Paper trading types
from .paper_trading import (
    AccountStateTransition,
    AccountUpdate,
    PaperPosition,
    PaperTrade,
    PaperTradeState,
    PaperTradingAccountState,
    TradeExecution,
    TradeStateTransition,
    TradingFees,
)
from .portfolio import (
    AccountSnapshot,
    AssetAllocation,
    AssetBalance,
    BalanceType,
    PerformanceSnapshot,
    PortfolioAllocation,
)

# Enhanced portfolio types (avoiding conflicts)
from .portfolio import (
    AccountType as PortfolioAccountType,  # Rename to avoid conflict
)

# Position management types
from .positions import (
    FLAT,
    LONG,
    SHORT,
    FunctionalLot,
    FunctionalPosition,
    LotSale,
    PositionSide,
    PositionSnapshot,
    create_empty_position,
    create_empty_snapshot,
    create_position_from_lot,
)

# Result monad
from .result import Failure, Result, Success

# Risk management types
from .risk import (
    AdvancedRiskAlert,
    AdvancedRiskAlertType,
    AllRiskAlerts,
    APIProtectionState,
    CircuitBreakerState,
    ComprehensiveRiskState,
    CorrelationMatrix,
    DailyLossLimit,
    DailyPnL,
    DrawdownAnalysis,
    EmergencyStopReason,
    EmergencyStopState,
    FailureRecord,
    LeverageAnalysis,
    MarginCall,
    PortfolioExposure,
    PositionLimitExceeded,
    PositionValidationResult,
    RiskAlert,
    RiskAlertType,
    RiskLevelAssessment,
    RiskMetricsSnapshot,
    RiskParameters,
    RiskValidationResult,
)

# Core trading types
from .trading import (
    CBI_ACCOUNT,
    CFM_ACCOUNT,
    # Account and position types
    AccountBalance,
    AccountType,
    CashTransferRequest,
    # Market data types
    FunctionalMarketData,
    FunctionalMarketState,
    FuturesAccountBalance,
    FuturesLimitOrder,
    FuturesMarketData,
    FuturesMarketOrder,
    FuturesStopOrder,
    Hold,
    # Order types
    LimitOrder,
    # Trade signals
    Long,
    MarginHealthStatus,
    # Margin and risk types
    MarginInfo,
    MarketData,  # Alias
    MarketMake,
    MarketOrder,
    MarketState,  # Alias
    Order,
    OrderResult,
    OrderStatus,
    Position,
    RiskLimits,
    RiskMetrics,
    Short,
    StopOrder,
    TradeDecision,
    TradeSignal,
    TradingIndicators,
    TradingParams,
    create_limit_orders_from_market_make,
    create_market_order_from_signal,
    get_signal_confidence,
    get_signal_size,
    # Utility functions
    is_directional_signal,
    is_pending_order,
    signal_to_side,
)

# Re-export commonly used types for backward compatibility
TradeAction = TradeSignal  # Alias for backward compatibility
FPCandle = Candle  # Alias for functional programming candle type

__all__ = [
    "CBI_ACCOUNT",
    "CFM_ACCOUNT",
    "EVENT_REGISTRY",
    "FLAT",
    # Effect system types
    "IO",
    "LONG",
    "OHLCV",
    "SHORT",
    # Configuration types
    "APIKey",
    "APIProtectionState",
    "AccountBalance",
    "AccountSnapshot",
    "AccountStateTransition",
    "AccountType",
    "AccountType",
    "AccountUpdate",
    "AdvancedRiskAlert",
    "AdvancedRiskAlertType",
    "AggregatedData",
    "AlertLevel",
    "AlertTriggered",
    "AllRiskAlerts",
    "AssetAllocation",
    "AssetBalance",
    "AuditTrailEntry",
    "BacktestConfig",
    "BalanceRange",
    "BalanceType",
    "BalanceValidationConfig",
    "BalanceValidationError",
    "BalanceValidationResult",
    # Balance validation types
    "BalanceValidationType",
    "BinanceExchangeConfig",
    "BluefinExchangeConfig",
    "BollingerBandsResult",
    "CancelResult",
    # Market data types
    "Candle",
    "CandlePattern",
    "CashTransferRequest",
    "CircuitBreakerState",
    "CoinbaseExchangeConfig",
    "CompositeSignal",
    "ComprehensiveBalanceValidation",
    "ComprehensiveRiskState",
    "Config",
    "ConfigurationChanged",
    "ConnectionState",
    "ConnectionStatus",
    "CorrelationMatrix",
    "DailyLossLimit",
    "DailyPnL",
    "DataQuality",
    "DbQuery",
    "DiamondPattern",
    "DivergencePattern",
    "DrawdownAnalysis",
    "Effect",
    "EmergencyStopReason",
    "EmergencyStopState",
    "Environment",
    "ErrorOccurred",
    # Event sourcing types
    "EventMetadata",
    "ExchangeConfig",
    "ExchangeType",
    # Learning and memory
    "ExperienceId",
    "FPCandle",  # Alias
    "Failure",
    "FailureRecord",
    "FeatureFlags",
    "FeeStructure",
    "FunctionalLot",
    # Market data types
    "FunctionalMarketData",
    "FunctionalMarketState",
    "FunctionalPosition",
    "FuturesAccountBalance",
    # Additional trading types
    "FuturesLimitOrder",
    "FuturesMarketData",
    "FuturesMarketOrder",
    "FuturesStopOrder",
    "Hold",
    "HttpRequest",
    "IndicatorConfig",
    "IndicatorHistory",
    # Indicator types
    "IndicatorResult",
    "IndicatorSet",
    "LLMStrategyConfig",
    "LearningInsight",
    "LeverageAnalysis",
    "LimitOrder",
    "Log",
    "LogLevel",
    # Trading types
    "Long",
    "LotSale",
    "MACDResult",
    "MarginCall",
    "MarginHealthStatus",
    # Margin and risk types
    "MarginInfo",
    "MarketData",
    "MarketDataMessage",
    "MarketDataReceived",
    "MarketDataStream",
    "MarketMake",
    "MarketOrder",
    "MarketSnapshot",
    "MarketSnapshot",
    "MarketState",
    "MarketStructure",
    "Maybe",
    "MeanReversionStrategyConfig",
    "MemoryQueryFP",
    "MemoryStorage",
    "MomentumStrategyConfig",
    # Base types
    "Money",
    "MovingAverageResult",
    "Nothing",
    "NotificationChannel",
    "NotificationSent",
    "Order",
    "OrderBook",
    "OrderBookMessage",
    "OrderCancelled",
    "OrderFilled",
    "OrderPlaced",
    "OrderResult",
    "OrderStatus",
    "PaperPosition",
    # Paper trading types
    "PaperTrade",
    "PaperTradeState",
    "PaperTradingAccountState",
    "PatternStatistics",
    "PatternTag",
    "Percentage",
    "PerformanceSnapshot",
    # Enhanced portfolio types
    "PortfolioAccountType",
    "PortfolioAllocation",
    "PortfolioExposure",
    "Position",
    "PositionClosed",
    "PositionLimitExceeded",
    "PositionOpened",
    # Positions
    "PositionSide",
    "PositionSnapshot",
    "PositionUpdate",
    "PositionValidationResult",
    "PriceData",
    "PrivateKey",
    "ROCResult",
    "RSIResult",
    "RateLimit",
    "RateLimits",
    "ReadFile",
    "RealtimeUpdate",
    # Result monad
    "Result",
    "RetryPolicy",
    "RiskAlert",
    "RiskAlertType",
    "RiskLevelAssessment",
    "RiskLimits",
    "RiskMetrics",
    "RiskMetricsSnapshot",
    # Risk management types
    "RiskParameters",
    "RiskValidationResult",
    "Short",
    "SignalHistory",
    "SignalSet",
    "Some",
    "StochasticResult",
    "StopOrder",
    "StrategyConfig",
    "StrategySignal",
    "StreamingData",
    "Subscription",
    "Success",
    "Symbol",
    "SystemComponent",
    "SystemConfig",
    "SystemPerformanceMetric",
    "Ticker",
    "TickerMessage",
    "TimeInterval",
    "TimeIntervalUnit",
    "TimeSeries",
    "Timestamp",
    "Trade",
    "TradeAction",  # Alias
    "TradeAffordabilityCheck",
    "TradeDecision",
    "TradeExecution",
    "TradeMessage",
    "TradeSignal",
    "TradeStateTransition",
    "TradingEvent",
    "TradingExperienceFP",
    "TradingFees",
    "TradingIndicators",
    "TradingMode",
    "TradingOutcome",
    "TradingParams",
    "VolumeProfile",
    "VuManchuResult",
    "VuManchuSet",
    "VuManchuSignalSet",
    "VuManchuState",
    "WebSocketConnection",
    "WebSocketMessage",
    "WriteFile",
    "YellowCrossSignal",
    "create_empty_position",
    "create_empty_snapshot",
    "create_limit_orders_from_market_make",
    "create_market_order_from_signal",
    "create_position_from_lot",
    "get_signal_confidence",
    "get_signal_size",
    # Utility functions
    "is_directional_signal",
    "is_pending_order",
    "signal_to_side",
]
