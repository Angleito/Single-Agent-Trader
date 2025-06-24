"""
Functional Monitoring Integration

This module provides integration between functional monitoring enhancements
and existing imperative monitoring systems, preserving all existing APIs
while adding functional programming capabilities.
"""

from .monitoring_integration import (
    IntegrationConfig,
    MonitoringState,
    UnifiedMonitoringSystem,
    create_unified_monitoring_system,
    get_unified_monitoring,
    initialize_monitoring_integration,
    integrate_with_existing_monitoring,
)

__all__ = [
    "IntegrationConfig",
    "MonitoringState",
    "UnifiedMonitoringSystem",
    "create_unified_monitoring_system",
    "get_unified_monitoring",
    "initialize_monitoring_integration",
    "integrate_with_existing_monitoring",
]
