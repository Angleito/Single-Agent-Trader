"""Functional programming types for the trading bot.

This module exports functional types used throughout the trading system,
providing immutable data structures and algebraic data types for type-safe
trading operations.
"""

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
    # Base types
    "Money",
    "Percentage",
    "Symbol",
    "TimeInterval",
    "TimeIntervalUnit",
    "Timestamp",
    "TradingMode",
    "Maybe",
    "Some",
    "Nothing",
    # Result monad
    "Result",
    "Success",
    "Failure",
    # Market data types
    "Candle",
    "FPCandle",  # Alias
    "OHLCV",
    "Ticker",
    "OrderBook",
    "Trade",
    "MarketSnapshot",
    "Subscription",
    "ConnectionStatus",
    "ConnectionState",
    "DataQuality",
    "WebSocketMessage",
    "TickerMessage",
    "TradeMessage",
    "OrderBookMessage",
    "RealtimeUpdate",
    "AggregatedData",
    "MarketDataStream",
    "MarketDataMessage",
    "PriceData",
    "StreamingData",
    # Trading types
    "Long",
    "Short",
    "Hold",
    "MarketMake",
    "TradeSignal",
    "LimitOrder",
    "MarketOrder",
    "StopOrder",
    "Order",
    "AccountBalance",
    "AccountType",
    "CFM_ACCOUNT",
    "CBI_ACCOUNT",
    "OrderStatus",
    "OrderResult",
    "Position",
    "TradeAction",  # Alias
    # Positions
    "PositionSide",
    "FunctionalLot",
    "LotSale",
    "FunctionalPosition",
    "PositionSnapshot",
    "LONG",
    "SHORT",
    "FLAT",
    # Indicator types
    "IndicatorResult",
    "MovingAverageResult",
    "RSIResult",
    "MACDResult",
    "BollingerBandsResult",
    "VuManchuResult",
    "VuManchuSignalSet",
    "VuManchuState",
    "DiamondPattern",
    "YellowCrossSignal",
    "CandlePattern",
    "DivergencePattern",
    "CompositeSignal",
    "VolumeProfile",
    "MarketStructure",
    "StochasticResult",
    "ROCResult",
    "TimeSeries",
    "IndicatorConfig",
    "IndicatorSet",
    "VuManchuSet",
    "SignalSet",
    "IndicatorHistory",
    "SignalHistory",
    # Configuration types
    "APIKey",
    "AccountType",
    "BacktestConfig",
    "BinanceExchangeConfig",
    "BluefinExchangeConfig",
    "CoinbaseExchangeConfig",
    "Config",
    "Environment",
    "ExchangeConfig",
    "ExchangeType",
    "FeatureFlags",
    "FeeStructure",
    "LLMStrategyConfig",
    "LogLevel",
    "MeanReversionStrategyConfig",
    "MomentumStrategyConfig",
    "PrivateKey",
    "RateLimits",
    "StrategyConfig",
    "SystemConfig",
    # Learning and memory
    "ExperienceId",
    "TradingOutcome",
    "PatternTag",
    "MarketSnapshot",
    "TradingExperienceFP",
    "MemoryQueryFP",
    "PatternStatistics",
    "LearningInsight",
    "MemoryStorage",
    # Utility functions
    "is_directional_signal",
    "is_pending_order",
    "get_signal_confidence",
    "get_signal_size",
    "signal_to_side",
    "create_market_order_from_signal",
    "create_limit_orders_from_market_make",
    "create_empty_position",
    "create_position_from_lot",
    "create_empty_snapshot",
    # Market data types
    "FunctionalMarketData",
    "FuturesMarketData",
    "FunctionalMarketState",
    "TradingIndicators",
    "MarketData",
    "MarketState",
    "TradeDecision",
    # Margin and risk types
    "MarginInfo",
    "MarginHealthStatus",
    "FuturesAccountBalance",
    "RiskLimits",
    "RiskMetrics",
    "TradingParams",
    
    # Additional trading types
    "FuturesLimitOrder",
    "FuturesMarketOrder",
    "FuturesStopOrder",
    "CashTransferRequest",
    
    # Balance validation types
    "BalanceValidationType",
    "BalanceRange",
    "BalanceValidationError",
    "BalanceValidationResult",
    "TradeAffordabilityCheck",
    "BalanceValidationConfig",
    "ComprehensiveBalanceValidation",
    
    # Effect system types
    "IO",
    "ReadFile",
    "WriteFile",
    "HttpRequest",
    "DbQuery",
    "Log",
    "WebSocketConnection",
    "RateLimit",
    "RetryPolicy",
    "CancelResult",
    "PositionUpdate",
    "Effect",
    
    # Event sourcing types
    "EventMetadata",
    "TradingEvent",
    "MarketDataReceived",
    "OrderPlaced",
    "OrderFilled",
    "OrderCancelled",
    "PositionOpened",
    "PositionClosed",
    "StrategySignal",
    "AlertLevel",
    "NotificationChannel",
    "SystemComponent",
    "AlertTriggered",
    "NotificationSent",
    "ConfigurationChanged",
    "SystemPerformanceMetric",
    "ErrorOccurred",
    "AuditTrailEntry",
    "EVENT_REGISTRY",
    
    # Paper trading types
    "PaperTrade",
    "PaperPosition",
    "PaperTradeState",
    "PaperTradingAccountState",
    "TradingFees",
    "TradeExecution",
    "AccountUpdate",
    "TradeStateTransition",
    "AccountStateTransition",
    
    # Enhanced portfolio types
    "PortfolioAccountType",
    "BalanceType",
    "AssetBalance",
    "AssetAllocation",
    "AccountSnapshot",
    "PortfolioAllocation",
    "PerformanceSnapshot",
    
    # Risk management types
    "RiskParameters",
    "RiskAlertType",
    "PositionLimitExceeded",
    "MarginCall",
    "DailyLossLimit",
    "RiskAlert",
    "FailureRecord",
    "CircuitBreakerState",
    "EmergencyStopReason",
    "EmergencyStopState",
    "APIProtectionState",
    "DailyPnL",
    "RiskValidationResult",
    "PositionValidationResult",
    "RiskLevelAssessment",
    "PortfolioExposure",
    "LeverageAnalysis",
    "CorrelationMatrix",
    "DrawdownAnalysis",
    "RiskMetricsSnapshot",
    "ComprehensiveRiskState",
    "AdvancedRiskAlertType",
    "AdvancedRiskAlert",
    "AllRiskAlerts",
]
