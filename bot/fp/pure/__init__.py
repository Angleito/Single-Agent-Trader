"""Pure functions module for functional trading bot."""

from .paper_trading_calculations import (
    apply_slippage,
    calculate_account_metrics,
    calculate_average_trade_duration,
    calculate_drawdown_series,
    calculate_largest_win_loss,
    calculate_portfolio_performance,
    calculate_position_size,
    calculate_profit_factor,
    calculate_required_margin,
    calculate_unrealized_pnl,
    calculate_win_rate,
    normalize_decimal_precision,
    simulate_position_close,
    simulate_trade_execution,
    validate_account_state,
)

__all__ = [
    "apply_slippage",
    "calculate_account_metrics",
    "calculate_average_trade_duration",
    "calculate_drawdown_series",
    "calculate_largest_win_loss",
    "calculate_portfolio_performance",
    "calculate_position_size",
    "calculate_profit_factor",
    "calculate_required_margin",
    "calculate_unrealized_pnl",
    "calculate_win_rate",
    "normalize_decimal_precision",
    "simulate_position_close",
    "simulate_trade_execution",
    "validate_account_state",
]
