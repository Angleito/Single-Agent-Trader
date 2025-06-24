"""
Effects module for functional trading bot.

This module provides all the effect types and combinators needed for
functional programming in the trading bot.
"""

# Core IO types
# Configuration effects
from .config import (
    Config,
    ConfigSource,
    Secret,
    ValidatedConfig,
    load_config,
    load_secret,
    merge_configs,
    validate_config,
)

# Error handling effects
from .error import CircuitConfig, RetryPolicy, RetryStrategy, fallback, recover, retry

# Exchange effects
from .exchange import (
    cancel_order,
    get_balance,
    get_positions,
    modify_position,
    place_order,
    stream_order_updates,
)
from .io import (
    IO,
    AsyncIO,
    Either,
    IOEither,
    Left,
    Right,
    from_future,
    from_option,
    from_try,
    parallel,
    race,
    sequence,
    traverse,
    unless,
    void,
    when,
)

# Logging effects
from .logging import (
    LogConfig,
    LogContext,
    LogLevel,
    debug,
    error,
    info,
    log,
    log_performance,
    log_trade_event,
    warn,
    with_context,
)

# Market data effects
from .market_data import (
    APIConfig,
    ConnectionConfig,
    connect_websocket,
    fetch_candles,
    fetch_orderbook,
    fetch_recent_trades,
    rate_limit,
    stream_market_data,
    subscribe_to_symbol,
    with_rate_limit,
)

# Monitoring effects
from .monitoring import (
    AlertLevel,
    HealthStatus,
    Span,
    alert,
    create_alert,
    send_alert,
    health_check,
    increment_counter,
    record_gauge,
    record_histogram,
    start_span,
)

# Persistence effects
from .persistence import (
    Event,
    EventFilter,
    State,
    append_to_file,
    cache,
    load_events,
    load_state,
    save_event,
    save_state,
    transaction,
)

# Time effects
from .time import delay, measure_time, now, timeout

__all__ = [
    # Core IO
    "IO",
    "AsyncIO",
    "IOEither",
    "Either",
    "Left",
    "Right",
    "sequence",
    "parallel",
    "traverse",
    "race",
    "from_try",
    "from_option",
    "from_future",
    "void",
    "when",
    "unless",
    # Market data
    "ConnectionConfig",
    "APIConfig",
    "connect_websocket",
    "subscribe_to_symbol",
    "stream_market_data",
    "fetch_candles",
    "fetch_orderbook",
    "fetch_recent_trades",
    "rate_limit",
    "with_rate_limit",
    # Exchange
    "place_order",
    "cancel_order",
    "get_positions",
    "get_balance",
    "modify_position",
    "stream_order_updates",
    # Logging
    "LogLevel",
    "LogConfig",
    "LogContext",
    "log",
    "debug",
    "info",
    "warn",
    "error",
    "with_context",
    "log_performance",
    "log_trade_event",
    # Persistence
    "Event",
    "EventFilter",
    "State",
    "save_event",
    "load_events",
    "save_state",
    "load_state",
    "transaction",
    "cache",
    "append_to_file",
    # Time
    "now",
    "delay",
    "timeout",
    "measure_time",
    # Error handling
    "RetryStrategy",
    "RetryPolicy",
    "CircuitConfig",
    "retry",
    "fallback",
    "recover",
    # Monitoring
    "AlertLevel",
    "HealthStatus",
    "Span",
    "increment_counter",
    "record_gauge",
    "record_histogram",
    "health_check",
    "start_span",
    "alert",
    "create_alert",
    "send_alert",
    # Configuration
    "ConfigSource",
    "Config",
    "ValidatedConfig",
    "Secret",
    "load_config",
    "validate_config",
    "load_secret",
    "merge_configs",
]
