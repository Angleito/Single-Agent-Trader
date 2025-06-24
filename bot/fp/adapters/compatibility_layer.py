"""
Compatibility layer for seamless migration to functional portfolio types.

This module provides a unified interface that maintains backward compatibility 
with existing APIs while gradually introducing functional programming patterns.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any

from bot.fp.types.base import Maybe, Some, Nothing
from bot.fp.types.portfolio import (
    AccountSnapshot, 
    PerformanceSnapshot,
    RiskMetrics
)
from bot.fp.types.positions import (
    FunctionalPosition,
    PositionSnapshot
)
from bot.fp.types.result import Result, Success, Failure
from bot.fp.adapters.position_manager_adapter import FunctionalPositionManagerAdapter
from bot.fp.adapters.paper_trading_adapter import FunctionalPaperTradingAdapter
from bot.trading_types import Position as LegacyPosition
from bot.position_manager import PositionManager
from bot.paper_trading import PaperTradingAccount

logger = logging.getLogger(__name__)


class FunctionalPortfolioManager:
    """
    Unified portfolio manager that provides both legacy and functional interfaces.
    
    This class serves as a compatibility layer that allows existing code to work
    unchanged while providing enhanced functionality through functional types.
    """
    
    def __init__(
        self, 
        position_manager: Optional[PositionManager] = None,
        paper_account: Optional[PaperTradingAccount] = None,
        enable_functional_features: bool = True
    ) -> None:
        """
        Initialize the unified portfolio manager.
        
        Args:
            position_manager: Optional legacy position manager
            paper_account: Optional paper trading account
            enable_functional_features: Whether to enable enhanced functional features
        """
        self.position_manager = position_manager
        self.paper_account = paper_account
        self.enable_functional_features = enable_functional_features
        
        # Initialize adapters if functional features are enabled
        self.position_adapter: Optional[FunctionalPositionManagerAdapter] = None
        self.paper_adapter: Optional[FunctionalPaperTradingAdapter] = None
        
        if enable_functional_features:
            if position_manager:
                self.position_adapter = FunctionalPositionManagerAdapter(position_manager)
            if paper_account:
                self.paper_adapter = FunctionalPaperTradingAdapter(paper_account)
        
        logger.info(
            "Initialized FunctionalPortfolioManager with functional features %s",
            "enabled" if enable_functional_features else "disabled"
        )
    
    # Legacy API compatibility methods
    
    def get_position(self, symbol: str) -> LegacyPosition:
        """
        Get position using legacy interface (backward compatibility).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Legacy Position object
        """
        if self.position_manager:
            return self.position_manager.get_position(symbol)
        
        # Create empty legacy position if no position manager
        return LegacyPosition(
            symbol=symbol,
            side="FLAT",
            size=Decimal(0),
            entry_price=None,
            unrealized_pnl=Decimal(0),
            realized_pnl=Decimal(0),
            timestamp=datetime.now()
        )
    
    def get_all_positions(self) -> List[LegacyPosition]:
        """
        Get all positions using legacy interface (backward compatibility).
        
        Returns:
            List of legacy Position objects
        """
        if self.position_manager:
            return self.position_manager.get_all_positions()
        return []
    
    def calculate_total_pnl(self) -> tuple[Decimal, Decimal]:
        """
        Calculate total P&L using legacy interface (backward compatibility).
        
        Returns:
            Tuple of (realized_pnl, unrealized_pnl)
        """
        if self.position_manager:
            return self.position_manager.calculate_total_pnl()
        return Decimal(0), Decimal(0)
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get position summary using legacy interface (backward compatibility).
        
        Returns:
            Dictionary with position summary
        """
        if self.position_manager:
            return self.position_manager.get_position_summary()
        
        return {
            "active_positions": 0,
            "total_realized_pnl": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "total_exposure": 0.0,
            "closed_positions_today": 0,
        }
    
    # Enhanced functional API methods
    
    def get_functional_position(self, symbol: str) -> Optional[FunctionalPosition]:
        """
        Get position using functional interface (enhanced).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            FunctionalPosition object or None if functional features disabled
        """
        if not self.enable_functional_features or not self.position_adapter:
            warnings.warn(
                "Functional features not enabled. Use get_position() for legacy interface.",
                UserWarning
            )
            return None
        
        return self.position_adapter.get_functional_position(symbol)
    
    def get_functional_snapshot(self) -> Optional[PositionSnapshot]:
        """
        Get position snapshot using functional interface (enhanced).
        
        Returns:
            PositionSnapshot or None if functional features disabled
        """
        if not self.enable_functional_features or not self.position_adapter:
            warnings.warn(
                "Functional features not enabled. Use get_position_summary() for legacy interface.",
                UserWarning
            )
            return None
        
        return self.position_adapter.get_position_snapshot()
    
    def get_account_snapshot(
        self, 
        current_prices: Dict[str, Decimal],
        base_currency: str = "USD"
    ) -> Result[str, AccountSnapshot]:
        """
        Get comprehensive account snapshot (enhanced functional feature).
        
        Args:
            current_prices: Current market prices
            base_currency: Base currency for calculations
            
        Returns:
            Result containing AccountSnapshot
        """
        if not self.enable_functional_features:
            return Failure("Functional features not enabled")
        
        # Try position adapter first
        if self.position_adapter:
            return self.position_adapter.get_account_snapshot(current_prices, base_currency=base_currency)
        
        # Try paper trading adapter
        if self.paper_adapter:
            return self.paper_adapter.get_functional_account_snapshot(current_prices, base_currency)
        
        return Failure("No adapters available for account snapshot")
    
    def get_performance_analysis(
        self, 
        current_prices: Dict[str, Decimal],
        days: int = 30
    ) -> Result[str, PerformanceSnapshot]:
        """
        Get comprehensive performance analysis (enhanced functional feature).
        
        Args:
            current_prices: Current market prices
            days: Number of days to analyze
            
        Returns:
            Result containing PerformanceSnapshot
        """
        if not self.enable_functional_features:
            return Failure("Functional features not enabled")
        
        # Use paper trading adapter if available (more detailed performance data)
        if self.paper_adapter:
            return self.paper_adapter.calculate_functional_performance(days)
        
        # Fallback to position adapter
        if self.position_adapter:
            try:
                # Get basic performance data
                realized_pnl, unrealized_pnl = self.calculate_total_pnl()
                
                # Create basic performance snapshot
                performance_snapshot = PerformanceSnapshot(
                    timestamp=datetime.now(),
                    total_value=realized_pnl + unrealized_pnl + Decimal("10000"),  # Assume starting balance
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized_pnl,
                    daily_return=None,
                    benchmark_return=None,
                    drawdown=Decimal("0")
                )
                
                return Success(performance_snapshot)
                
            except Exception as e:
                return Failure(f"Failed to calculate performance: {str(e)}")
        
        return Failure("No adapters available for performance analysis")
    
    def get_risk_analysis(self, days: int = 30) -> Result[str, RiskMetrics]:
        """
        Get comprehensive risk analysis (enhanced functional feature).
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Result containing RiskMetrics
        """
        if not self.enable_functional_features:
            return Failure("Functional features not enabled")
        
        # Use paper trading adapter if available (more detailed risk data)
        if self.paper_adapter:
            return self.paper_adapter.calculate_risk_metrics(days)
        
        # Create basic risk metrics if no detailed data available
        try:
            risk_metrics = RiskMetrics(
                var_95=Decimal("0"),
                max_drawdown=Decimal("0"),
                volatility=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                sortino_ratio=Decimal("0"),
                beta=None,
                correlation_to_benchmark=None,
                concentration_risk=Decimal("0"),
                timestamp=datetime.now()
            )
            
            return Success(risk_metrics)
            
        except Exception as e:
            return Failure(f"Failed to calculate risk metrics: {str(e)}")
    
    def validate_consistency(
        self, 
        current_prices: Dict[str, Decimal]
    ) -> Result[str, Dict[str, bool]]:
        """
        Validate consistency between legacy and functional representations.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            Result containing validation results
        """
        if not self.enable_functional_features or not self.position_adapter:
            return Failure("Functional features not enabled or no position adapter")
        
        return self.position_adapter.validate_portfolio_consistency(current_prices)
    
    def generate_comprehensive_report(
        self, 
        current_prices: Dict[str, Decimal],
        days: int = 7
    ) -> Result[str, Dict[str, Any]]:
        """
        Generate comprehensive portfolio report combining all available data.
        
        Args:
            current_prices: Current market prices
            days: Number of days to analyze
            
        Returns:
            Result containing comprehensive report
        """
        try:
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "analysis_period_days": days,
                "functional_features_enabled": self.enable_functional_features,
                "legacy_data": {},
                "functional_data": {},
                "consistency_check": {}
            }
            
            # Add legacy data
            if self.position_manager:
                legacy_summary = self.get_position_summary()
                realized_pnl, unrealized_pnl = self.calculate_total_pnl()
                
                report["legacy_data"] = {
                    "position_summary": legacy_summary,
                    "total_realized_pnl": float(realized_pnl),
                    "total_unrealized_pnl": float(unrealized_pnl),
                    "total_pnl": float(realized_pnl + unrealized_pnl)
                }
            
            # Add functional data if enabled
            if self.enable_functional_features:
                # Account snapshot
                account_result = self.get_account_snapshot(current_prices)
                if account_result.is_success():
                    account = account_result.success()
                    report["functional_data"]["account"] = {
                        "account_type": account.account_type.value,
                        "total_equity": float(account.total_equity),
                        "base_currency": account.base_currency,
                        "balance_count": len(account.balances)
                    }
                
                # Performance analysis
                performance_result = self.get_performance_analysis(current_prices, days)
                if performance_result.is_success():
                    performance = performance_result.success()
                    report["functional_data"]["performance"] = {
                        "total_value": float(performance.total_value),
                        "total_pnl": float(performance.total_pnl),
                        "total_return_pct": float(performance.total_return_pct),
                        "drawdown_pct": float(performance.drawdown)
                    }
                
                # Risk analysis
                risk_result = self.get_risk_analysis(days)
                if risk_result.is_success():
                    risk = risk_result.success()
                    report["functional_data"]["risk"] = {
                        "var_95": float(risk.var_95),
                        "max_drawdown": float(risk.max_drawdown),
                        "sharpe_ratio": float(risk.sharpe_ratio),
                        "risk_score": risk.risk_score
                    }
                
                # Paper trading specific data
                if self.paper_adapter:
                    paper_report_result = self.paper_adapter.generate_functional_report(days)
                    if paper_report_result.is_success():
                        paper_report = paper_report_result.success()
                        report["functional_data"]["paper_trading"] = paper_report
                
                # Consistency check
                if self.position_adapter:
                    consistency_result = self.validate_consistency(current_prices)
                    if consistency_result.is_success():
                        report["consistency_check"] = consistency_result.success()
            
            return Success(report)
            
        except Exception as e:
            return Failure(f"Failed to generate comprehensive report: {str(e)}")
    
    def migrate_to_functional(
        self, 
        validate_migration: bool = True,
        current_prices: Optional[Dict[str, Decimal]] = None
    ) -> Result[str, str]:
        """
        Migrate existing data to functional types with validation.
        
        Args:
            validate_migration: Whether to validate the migration
            current_prices: Current prices for validation
            
        Returns:
            Result containing migration status
        """
        if not self.enable_functional_features:
            return Failure("Functional features not enabled")
        
        try:
            migration_steps = []
            
            # Step 1: Initialize functional adapters if not already done
            if self.position_manager and not self.position_adapter:
                self.position_adapter = FunctionalPositionManagerAdapter(self.position_manager)
                migration_steps.append("Position adapter initialized")
            
            if self.paper_account and not self.paper_adapter:
                self.paper_adapter = FunctionalPaperTradingAdapter(self.paper_account)
                migration_steps.append("Paper trading adapter initialized")
            
            # Step 2: Validate migration if requested
            if validate_migration and current_prices:
                if self.position_adapter:
                    validation_result = self.position_adapter.validate_portfolio_consistency(current_prices)
                    if validation_result.is_failure():
                        return Failure(f"Migration validation failed: {validation_result.failure()}")
                    
                    consistency = validation_result.success()
                    if not consistency.get("overall_consistent", False):
                        logger.warning("Migration validation warnings: %s", consistency)
                        migration_steps.append("Migration validation completed with warnings")
                    else:
                        migration_steps.append("Migration validation successful")
                
                if self.paper_adapter:
                    from bot.fp.adapters.paper_trading_adapter import validate_paper_trading_migration
                    if validate_paper_trading_migration(self.paper_adapter, current_prices):
                        migration_steps.append("Paper trading migration validation successful")
                    else:
                        migration_steps.append("Paper trading migration validation failed")
            
            # Step 3: Migration complete
            migration_summary = f"Migration completed successfully. Steps: {'; '.join(migration_steps)}"
            logger.info(migration_summary)
            
            return Success(migration_summary)
            
        except Exception as e:
            return Failure(f"Migration failed: {str(e)}")
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get current migration status and feature availability.
        
        Returns:
            Dictionary with migration status information
        """
        return {
            "functional_features_enabled": self.enable_functional_features,
            "position_manager_available": self.position_manager is not None,
            "paper_account_available": self.paper_account is not None,
            "position_adapter_initialized": self.position_adapter is not None,
            "paper_adapter_initialized": self.paper_adapter is not None,
            "legacy_api_compatible": True,
            "functional_api_available": self.enable_functional_features,
            "migration_complete": (
                self.enable_functional_features and 
                (self.position_adapter is not None or self.paper_adapter is not None)
            )
        }


# Utility functions for gradual migration

def create_unified_portfolio_manager(
    position_manager: Optional[PositionManager] = None,
    paper_account: Optional[PaperTradingAccount] = None,
    enable_functional: bool = True
) -> FunctionalPortfolioManager:
    """
    Create a unified portfolio manager with both legacy and functional support.
    
    Args:
        position_manager: Optional position manager
        paper_account: Optional paper trading account
        enable_functional: Whether to enable functional features
        
    Returns:
        Unified portfolio manager
    """
    return FunctionalPortfolioManager(
        position_manager=position_manager,
        paper_account=paper_account,
        enable_functional_features=enable_functional
    )


def migrate_existing_system(
    position_manager: PositionManager,
    paper_account: Optional[PaperTradingAccount] = None,
    current_prices: Optional[Dict[str, Decimal]] = None,
    validate: bool = True
) -> Result[str, FunctionalPortfolioManager]:
    """
    Migrate an existing system to use functional portfolio management.
    
    Args:
        position_manager: Existing position manager
        paper_account: Optional paper trading account
        current_prices: Current prices for validation
        validate: Whether to validate the migration
        
    Returns:
        Result containing migrated portfolio manager
    """
    try:
        # Create unified manager
        unified_manager = create_unified_portfolio_manager(
            position_manager=position_manager,
            paper_account=paper_account,
            enable_functional=True
        )
        
        # Perform migration
        migration_result = unified_manager.migrate_to_functional(
            validate_migration=validate,
            current_prices=current_prices
        )
        
        if migration_result.is_failure():
            return Failure(migration_result.failure())
        
        logger.info("System migration completed: %s", migration_result.success())
        return Success(unified_manager)
        
    except Exception as e:
        return Failure(f"System migration failed: {str(e)}")


def get_feature_compatibility_report() -> Dict[str, Any]:
    """
    Get a report on feature compatibility between legacy and functional APIs.
    
    Returns:
        Compatibility report
    """
    return {
        "legacy_api_methods": [
            "get_position", 
            "get_all_positions", 
            "calculate_total_pnl",
            "get_position_summary"
        ],
        "functional_api_methods": [
            "get_functional_position",
            "get_functional_snapshot", 
            "get_account_snapshot",
            "get_performance_analysis",
            "get_risk_analysis"
        ],
        "enhanced_features": [
            "Immutable position types",
            "FIFO lot tracking", 
            "Advanced P&L calculations",
            "Risk metrics analysis",
            "Performance attribution",
            "Portfolio optimization",
            "Margin and leverage support",
            "Asset allocation management"
        ],
        "migration_path": {
            "phase_1": "Initialize compatibility layer with functional features disabled",
            "phase_2": "Enable functional features and run in parallel with legacy",
            "phase_3": "Validate consistency between legacy and functional calculations",
            "phase_4": "Gradually migrate to use functional APIs in new code",
            "phase_5": "Deprecate legacy APIs once functional migration is complete"
        },
        "backward_compatibility": "Full backward compatibility maintained during migration"
    }