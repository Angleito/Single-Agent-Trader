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
    health_check,
    increment_counter,
    record_gauge,
    record_histogram,
    send_alert,
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
    "APIConfig",
    # Monitoring
    "AlertLevel",
    "AsyncIO",
    "CircuitConfig",
    "Config",
    # Configuration
    "ConfigSource",
    # Market data
    "ConnectionConfig",
    "Either",
    # Persistence
    "Event",
    "EventFilter",
    "HealthStatus",
    "IOEither",
    "Left",
    "LogConfig",
    "LogContext",
    # Logging
    "LogLevel",
    "RetryPolicy",
    # Error handling
    "RetryStrategy",
    "Right",
    "Secret",
    "Span",
    "State",
    "ValidatedConfig",
    "alert",
    "append_to_file",
    "cache",
    "cancel_order",
    "connect_websocket",
    "create_alert",
    "debug",
    "delay",
    "error",
    "fallback",
    "fetch_candles",
    "fetch_orderbook",
    "fetch_recent_trades",
    "from_future",
    "from_option",
    "from_try",
    "get_balance",
    "get_positions",
    "health_check",
    "increment_counter",
    "info",
    "load_config",
    "load_events",
    "load_secret",
    "load_state",
    "log",
    "log_performance",
    "log_trade_event",
    "measure_time",
    "merge_configs",
    "modify_position",
    # Time
    "now",
    "parallel",
    # Exchange
    "place_order",
    "race",
    "rate_limit",
    "record_gauge",
    "record_histogram",
    "recover",
    "retry",
    "save_event",
    "save_state",
    "send_alert",
    "sequence",
    "start_span",
    "stream_market_data",
    "stream_order_updates",
    "subscribe_to_symbol",
    "timeout",
    "transaction",
    "traverse",
    "unless",
    "validate_config",
    "void",
    "warn",
    "when",
    "with_context",
    "with_rate_limit",
]
