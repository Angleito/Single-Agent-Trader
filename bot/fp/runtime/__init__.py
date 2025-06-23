"""
Runtime module for functional trading bot.

This module provides all the runtime components needed for executing
the functional trading bot including interpreters, schedulers, and monitoring.
"""

# Core runtime components
from .backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
    run_backtest,
)
from .cli import CLIConfig, FunctionalCLI, main
from .interpreter import (
    EffectInterpreter,
    RuntimeConfig,
    RuntimeContext,
    get_interpreter,
    run,
    run_async,
    run_either,
)
from .monitoring import (
    MonitoringConfig,
    MonitoringRuntime,
    SystemMetrics,
    get_monitoring,
)
from .scheduler import (
    ScheduledTask,
    SchedulerConfig,
    TradingScheduler,
    create_default_tasks,
    get_scheduler,
)
from .state import StateManager, STMRef, create_state, get_state, get_state_manager
from .websocket import WebSocketManager, get_websocket_manager

__all__ = [
    # Interpreter
    "EffectInterpreter",
    "RuntimeConfig",
    "RuntimeContext",
    "get_interpreter",
    "run",
    "run_async",
    "run_either",
    # Scheduler
    "TradingScheduler",
    "SchedulerConfig",
    "ScheduledTask",
    "get_scheduler",
    "create_default_tasks",
    # State management
    "STMRef",
    "StateManager",
    "get_state_manager",
    "create_state",
    "get_state",
    # CLI
    "FunctionalCLI",
    "CLIConfig",
    "main",
    # Monitoring
    "MonitoringRuntime",
    "MonitoringConfig",
    "SystemMetrics",
    "get_monitoring",
    # WebSocket
    "WebSocketManager",
    "get_websocket_manager",
    # Backtesting
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "run_backtest",
]
