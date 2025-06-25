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
    "BacktestConfig",
    # Backtesting
    "BacktestEngine",
    "BacktestResult",
    "CLIConfig",
    # Interpreter
    "EffectInterpreter",
    # CLI
    "FunctionalCLI",
    "MonitoringConfig",
    # Monitoring
    "MonitoringRuntime",
    "RuntimeConfig",
    "RuntimeContext",
    # State management
    "STMRef",
    "ScheduledTask",
    "SchedulerConfig",
    "StateManager",
    "SystemMetrics",
    "Trade",
    # Scheduler
    "TradingScheduler",
    # WebSocket
    "WebSocketManager",
    "create_default_tasks",
    "create_state",
    "get_interpreter",
    "get_monitoring",
    "get_scheduler",
    "get_state",
    "get_state_manager",
    "get_websocket_manager",
    "main",
    "run",
    "run_async",
    "run_backtest",
    "run_either",
]
