"""
Type system foundation for the trading bot.

This module provides centralized type definitions, exception hierarchy,
and type guards for improved type safety across the codebase.
"""

from .base_types import (
    AccountId,
    IndicatorDict,
    MarketDataDict,
    OrderId,
    Percentage,
    Price,
    Quantity,
    Symbol,
    Timestamp,
    ValidationResult,
)
from .exceptions import (
    BalanceValidationError,
    ExchangeAuthError,
    ExchangeConnectionError,
    ExchangeError,
    IndicatorError,
    LLMError,
    OrderExecutionError,
    PositionValidationError,
    StrategyError,
    TradeValidationError,
    TradingBotError,
    ValidationError,
)

# Import order-related types from FP module
try:
    from bot.fp.orders import OrderSide, OrderStatus, OrderType
except ImportError:
    # Fallback if FP module is not available
    from enum import Enum

    class OrderSide(Enum):
        BUY = "BUY"
        SELL = "SELL"

    class OrderStatus(Enum):
        PENDING = "PENDING"
        OPEN = "OPEN"
        FILLED = "FILLED"
        CANCELLED = "CANCELLED"

    class OrderType(Enum):
        MARKET = "MARKET"
        LIMIT = "LIMIT"


from .guards import (
    ensure_decimal,
    ensure_positive_decimal,
    is_valid_percentage,
    is_valid_price,
    is_valid_quantity,
    is_valid_symbol,
)
from .market_data import (
    AggregatedMarketData,
    CandleData,
    ConnectionState,
    MarketDataQuality,
    MarketDataStatus,
    MarketDepth,
    OrderBook,
    OrderBookLevel,
    OrderType,
    PriceLevel,
    Spread,
    TickerData,
    TradeExecution,
    TradeId,
    TradeSide,
    aggregate_candles,
    is_valid_timestamp,
    is_valid_volume,
)
from .market_data import (
    is_valid_price as is_valid_market_price,
)
from .services import (
    AsyncHealthCheck,
    AsyncServiceValidator,
    ConnectionInfo,
    ConnectionState,
    DiscoveredService,
    DiscoveryMethod,
    DockerService,
    HealthCheckable,
    RetryConfig,
    ServiceCallback,
    ServiceConfig,
    ServiceConnectionError,
    ServiceDependencyError,
    ServiceEndpoint,
    ServiceError,
    ServiceHealth,
    ServiceHealthCheckError,
    ServiceManager,
    ServiceNotFoundError,
    ServiceRegistration,
    ServiceRegistry,
    ServiceStartupError,
    ServiceStatus,
    ServiceTimeoutError,
    create_endpoint,
    create_health_status,
    is_docker_service,
    is_healthy_service,
    is_valid_endpoint,
    validate_service_config,
)

__all__ = [
    "AccountId",
    # Market data types
    "AggregatedMarketData",
    # Service type aliases
    "AsyncHealthCheck",
    "AsyncServiceValidator",
    "BalanceValidationError",
    "CandleData",
    "ConnectionInfo",
    "ConnectionState",
    "ConnectionState",
    "DiscoveredService",
    "DiscoveryMethod",
    # Service protocols
    "DockerService",
    "ExchangeAuthError",
    "ExchangeConnectionError",
    "ExchangeError",
    "HealthCheckable",
    "IndicatorDict",
    "IndicatorError",
    "LLMError",
    "MarketDataDict",
    "MarketDataQuality",
    "MarketDataStatus",
    "MarketDepth",
    "OrderBook",
    "OrderBookLevel",
    "OrderExecutionError",
    "OrderId",
    "OrderType",
    "Percentage",
    "PositionValidationError",
    # Base types
    "Price",
    "PriceLevel",
    "Quantity",
    "RetryConfig",
    "ServiceCallback",
    "ServiceConfig",
    "ServiceConnectionError",
    "ServiceDependencyError",
    "ServiceEndpoint",
    # Service errors
    "ServiceError",
    # Service types
    "ServiceHealth",
    "ServiceHealthCheckError",
    "ServiceManager",
    "ServiceNotFoundError",
    "ServiceRegistration",
    "ServiceRegistry",
    "ServiceStartupError",
    "ServiceStatus",
    "ServiceTimeoutError",
    "Spread",
    "StrategyError",
    "Symbol",
    "TickerData",
    "Timestamp",
    "TradeExecution",
    "TradeId",
    "TradeSide",
    "TradeValidationError",
    # Exceptions
    "TradingBotError",
    "ValidationError",
    "ValidationResult",
    "aggregate_candles",
    "create_endpoint",
    # Service utilities
    "create_health_status",
    "ensure_decimal",
    "ensure_positive_decimal",
    "is_docker_service",
    "is_healthy_service",
    # Service type guards
    "is_valid_endpoint",
    # Market data guards and helpers
    "is_valid_market_price",
    "is_valid_percentage",
    # Guards
    "is_valid_price",
    "is_valid_quantity",
    "is_valid_symbol",
    "is_valid_timestamp",
    "is_valid_volume",
    "validate_service_config",
]
