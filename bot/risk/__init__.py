"""
Risk management module for the trading bot.

This module provides risk management functionality including circuit breakers,
API protection, emergency stops, and core risk management logic.
"""

from .api_protection import APIFailureProtection
from .circuit_breaker import TradingCircuitBreaker
from .emergency_stop import EmergencyStopManager
from .risk_manager import RiskManager
from .types import DailyPnL, FailureRecord

__all__ = [
    "APIFailureProtection",
    "DailyPnL",
    "EmergencyStopManager",
    "FailureRecord",
    "RiskManager",
    "TradingCircuitBreaker",
]
