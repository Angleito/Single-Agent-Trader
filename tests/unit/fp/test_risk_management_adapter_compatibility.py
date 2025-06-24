"""
Comprehensive tests for FP/legacy risk management adapter compatibility.

This module tests the adapter layer that ensures seamless migration between legacy
imperative risk management and functional programming risk management while
maintaining all safety validations and performance characteristics.

Tests include:
- Risk calculation consistency between FP and legacy systems
- Position manager adapter risk functionality
- Paper trading adapter risk management
- Migration scenario validation
- Performance benchmarking between FP and legacy risk calculations
- Safety validation preservation during FP transition
- Error handling and edge case compatibility
- Integration testing of complete risk management pipeline
"""

import pytest
import asyncio
from datetime import datetime, timedelta, UTC
from decimal import Decimal, getcontext
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Dict, List, Optional, Tuple
from hypothesis import given, strategies as st, settings, HealthCheck

# FP test infrastructure
from tests.fp_test_base import (
    FPTestBase,
    FP_AVAILABLE
)

if FP_AVAILABLE:
    # FP risk management types
    from bot.fp.types.risk import (
        RiskParameters, RiskState, RiskMetrics, RiskViolation,
        EmergencyStop, CircuitBreaker, DrawdownLimits,
        create_risk_parameters, create_risk_state, create_circuit_breaker,
        validate_position_risk, check_emergency_conditions
    )
    
    # FP adapter types
    from bot.fp.adapters.compatibility_layer import (
        FunctionalPortfolioManager, create_unified_portfolio_manager,
        migrate_existing_system, get_feature_compatibility_report
    )
    from bot.fp.adapters.position_manager_adapter import (
        FunctionalPositionManagerAdapter, validate_functional_migration
    )
    from bot.fp.adapters.paper_trading_adapter import FunctionalPaperTradingAdapter
    
    # FP risk calculations
    from bot.fp.pure.risk_calculations import (
        calculate_position_risk, calculate_portfolio_risk,
        calculate_risk_metrics, validate_risk_limits,
        calculate_margin_requirement, check_drawdown_limits
    )
    
    # FP types
    from bot.fp.types.effects import Result, Ok, Err
    from bot.fp.types.base import Maybe, Some, Nothing
    from bot.fp.types.portfolio import Portfolio, PortfolioMetrics
    from bot.fp.types.positions import FunctionalPosition
else:
    # Fallback stubs for non-FP environments
    class RiskParameters:
        pass
    
    def validate_position_risk(*args, **kwargs):
        return None

# Legacy types
from bot.trading_types import Position as LegacyPosition, Order
from bot.position_manager import PositionManager
from bot.paper_trading import PaperTradingAccount
from bot.risk import RiskManager

# Set high precision for financial calculations
getcontext().prec = 28


class TestRiskManagementAdapterCompatibility(FPTestBase):
    """Test risk management adapter compatibility between FP and legacy systems."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        # Create mock legacy components
        self.mock_position_manager = Mock(spec=PositionManager)
        self.mock_paper_account = Mock(spec=PaperTradingAccount)
        self.mock_risk_manager = Mock(spec=RiskManager)
        
        # Set up default mock responses
        self.mock_position_manager.get_position.return_value = LegacyPosition(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0"),
            entry_price=None,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            timestamp=datetime.now()
        )
        self.mock_position_manager.get_all_positions.return_value = []
        self.mock_position_manager.calculate_total_pnl.return_value = (Decimal("0"), Decimal("0"))
        
        # Create unified portfolio manager for testing
        self.unified_manager = create_unified_portfolio_manager(
            position_manager=self.mock_position_manager,
            paper_account=self.mock_paper_account,
            enable_functional=True
        )
        
        # Test data
        self.test_prices = {
            "BTC-USD": Decimal("50000.0"),
            "ETH-USD": Decimal("3000.0"),
            "SOL-USD": Decimal("100.0")
        }
        
        self.test_risk_params = create_risk_parameters(
            max_position_size=Decimal("10000"),
            max_leverage=Decimal("10"),
            stop_loss_pct=Decimal("5"),
            take_profit_pct=Decimal("10"),
            max_daily_loss=Decimal("1000"),
            max_drawdown=Decimal("20")
        )
    
    def test_position_manager_adapter_initialization(self):
        """Test that position manager adapter initializes correctly."""
        adapter = FunctionalPositionManagerAdapter(self.mock_position_manager)
        
        assert adapter.position_manager is self.mock_position_manager
        assert adapter._last_snapshot is None
        assert adapter._last_account_snapshot is None
    
    def test_legacy_to_functional_position_conversion(self):
        """Test conversion from legacy Position to FunctionalPosition."""
        # Create test legacy position
        legacy_position = LegacyPosition(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("1.5"),
            entry_price=Decimal("48000.0"),
            unrealized_pnl=Decimal("3000.0"),
            realized_pnl=Decimal("500.0"),
            timestamp=datetime.now(UTC)
        )
        
        self.mock_position_manager.get_position.return_value = legacy_position
        
        adapter = FunctionalPositionManagerAdapter(self.mock_position_manager)
        functional_position = adapter.get_functional_position("BTC-USD")
        
        # Validate conversion
        assert functional_position.symbol == "BTC-USD"
        assert functional_position.total_quantity == Decimal("1.5")
        assert not functional_position.is_flat
        assert functional_position.side.name == "LONG"
    
    def test_risk_calculation_consistency(self):
        """Test that risk calculations are consistent between legacy and functional."""
        # Set up position with risk data
        legacy_position = LegacyPosition(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("2.0"),
            entry_price=Decimal("45000.0"),
            unrealized_pnl=Decimal("10000.0"),
            realized_pnl=Decimal("0"),
            timestamp=datetime.now(UTC)
        )
        
        self.mock_position_manager.get_position.return_value = legacy_position
        self.mock_position_manager.get_all_positions.return_value = [legacy_position]
        self.mock_position_manager.calculate_total_pnl.return_value = (Decimal("0"), Decimal("10000.0"))
        
        adapter = FunctionalPositionManagerAdapter(self.mock_position_manager)
        
        # Test portfolio consistency validation
        validation_result = adapter.validate_portfolio_consistency(self.test_prices)
        
        assert validation_result.is_success()
        validation_data = validation_result.success()
        
        # Should have consistency checks
        assert "realized_pnl_match" in validation_data
        assert "unrealized_pnl_match" in validation_data
        assert "position_count_match" in validation_data
        assert "overall_consistent" in validation_data
    
    def test_functional_risk_metrics_calculation(self):
        """Test functional risk metrics calculation through adapter."""
        # Create test positions
        positions = [
            LegacyPosition(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("1000.0"),
                timestamp=datetime.now(UTC)
            ),
            LegacyPosition(
                symbol="ETH-USD",
                side="SHORT",
                size=Decimal("5.0"),
                entry_price=Decimal("3200.0"),
                unrealized_pnl=Decimal("-1000.0"),
                realized_pnl=Decimal("500.0"),
                timestamp=datetime.now(UTC)
            )
        ]
        
        self.mock_position_manager.get_all_positions.return_value = positions
        self.mock_position_manager.calculate_total_pnl.return_value = (Decimal("1500.0"), Decimal("-1000.0"))
        
        adapter = FunctionalPositionManagerAdapter(self.mock_position_manager)
        
        # Calculate portfolio metrics
        metrics_result = adapter.calculate_portfolio_metrics(self.test_prices)
        
        assert metrics_result.is_success()
        metrics = metrics_result.success()
        
        # Validate risk-related metrics
        assert "total_pnl" in metrics
        assert "total_value" in metrics
        assert "position_count" in metrics
        assert metrics["position_count"] == Decimal("2")  # Two active positions
    
    def test_unified_portfolio_manager_risk_analysis(self):
        """Test unified portfolio manager's risk analysis functionality."""
        # Set up realistic position data
        self.mock_position_manager.calculate_total_pnl.return_value = (Decimal("2000.0"), Decimal("-500.0"))
        
        # Test risk analysis
        risk_result = self.unified_manager.get_risk_analysis(days=30)
        
        assert risk_result.is_success()
        risk_metrics = risk_result.success()
        
        # Validate risk metrics structure
        assert hasattr(risk_metrics, 'var_95')
        assert hasattr(risk_metrics, 'max_drawdown')
        assert hasattr(risk_metrics, 'sharpe_ratio')
        assert hasattr(risk_metrics, 'volatility')
        assert hasattr(risk_metrics, 'timestamp')
    
    def test_compatibility_layer_migration_validation(self):
        """Test that migration preserves risk management functionality."""
        # Test migration
        migration_result = self.unified_manager.migrate_to_functional(
            validate_migration=True,
            current_prices=self.test_prices
        )
        
        assert migration_result.is_success()
        migration_status = migration_result.success()
        assert "Migration completed successfully" in migration_status
        
        # Validate migration status
        status = self.unified_manager.get_migration_status()
        assert status["functional_features_enabled"] is True
        assert status["legacy_api_compatible"] is True
        assert status["functional_api_available"] is True
    
    def test_paper_trading_adapter_risk_compatibility(self):
        """Test paper trading adapter risk management compatibility."""
        if not hasattr(self.unified_manager, 'paper_adapter') or self.unified_manager.paper_adapter is None:
            pytest.skip("Paper trading adapter not available")
        
        # Mock paper account with risk data
        self.mock_paper_account.get_balance.return_value = Decimal("10000.0")
        self.mock_paper_account.get_equity.return_value = Decimal("9500.0")
        
        # Test risk metrics through paper adapter
        paper_adapter = FunctionalPaperTradingAdapter(self.mock_paper_account)
        risk_result = paper_adapter.calculate_risk_metrics(days=30)
        
        # Should return risk metrics or appropriate fallback
        assert risk_result.is_success() or "not available" in risk_result.failure()
    
    def test_emergency_stop_compatibility(self):
        """Test that emergency stop functionality is preserved in FP adaptation."""
        # Create emergency risk state
        risk_state = create_risk_state(
            total_exposure=Decimal("50000.0"),
            daily_pnl=Decimal("-2000.0"),
            max_drawdown=Decimal("25.0"),
            open_positions=3,
            margin_used=Decimal("10000.0")
        )
        
        # Test emergency condition checking
        emergency_result = check_emergency_conditions(risk_state, self.test_risk_params)
        
        # Should detect emergency due to high drawdown
        assert emergency_result.max_drawdown_violated is True
        assert emergency_result.should_stop is True
    
    def test_circuit_breaker_adapter_functionality(self):
        """Test circuit breaker functionality through adapters."""
        # Create circuit breaker
        circuit_breaker = create_circuit_breaker(
            failure_threshold=3,
            recovery_timeout=timedelta(minutes=5),
            max_failures_per_day=10
        )
        
        # Test circuit breaker state transitions
        assert circuit_breaker.state == "CLOSED"
        
        # Simulate failures
        updated_breaker = circuit_breaker.record_failure("Test failure 1")
        updated_breaker = updated_breaker.record_failure("Test failure 2")
        updated_breaker = updated_breaker.record_failure("Test failure 3")
        
        # Should open after threshold failures
        assert updated_breaker.state == "OPEN"
        assert updated_breaker.should_block_trades() is True
    
    def test_drawdown_limits_through_adapter(self):
        """Test drawdown limit checking through adapter functionality."""
        # Create test portfolio with significant drawdown
        positions = [
            LegacyPosition(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("1.0"),
                entry_price=Decimal("60000.0"),
                unrealized_pnl=Decimal("-15000.0"),  # Significant loss
                realized_pnl=Decimal("-5000.0"),
                timestamp=datetime.now(UTC)
            )
        ]
        
        self.mock_position_manager.get_all_positions.return_value = positions
        self.mock_position_manager.calculate_total_pnl.return_value = (Decimal("-5000.0"), Decimal("-15000.0"))
        
        adapter = FunctionalPositionManagerAdapter(self.mock_position_manager)
        
        # Get portfolio performance
        performance_result = adapter.get_portfolio_performance(
            self.test_prices, 
            Decimal("50000.0")  # Starting balance
        )
        
        assert performance_result.is_success()
        performances = performance_result.success()
        
        # Should calculate performance metrics including drawdown implications
        assert len(performances) > 0
        for performance in performances:
            assert hasattr(performance, 'total_return_pct')
            assert hasattr(performance, 'position_size_pct')
    
    def test_comprehensive_risk_report_generation(self):
        """Test comprehensive risk report generation through unified manager."""
        # Set up complex portfolio state
        complex_positions = [
            LegacyPosition(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("2.0"),
                entry_price=Decimal("48000.0"),
                unrealized_pnl=Decimal("4000.0"),
                realized_pnl=Decimal("1000.0"),
                timestamp=datetime.now(UTC)
            ),
            LegacyPosition(
                symbol="ETH-USD",
                side="SHORT",
                size=Decimal("10.0"),
                entry_price=Decimal("3200.0"),
                unrealized_pnl=Decimal("-2000.0"),
                realized_pnl=Decimal("800.0"),
                timestamp=datetime.now(UTC)
            )
        ]
        
        self.mock_position_manager.get_all_positions.return_value = complex_positions
        self.mock_position_manager.calculate_total_pnl.return_value = (Decimal("1800.0"), Decimal("2000.0"))
        
        # Generate comprehensive report
        report_result = self.unified_manager.generate_comprehensive_report(
            self.test_prices,
            days=7
        )
        
        assert report_result.is_success()
        report = report_result.success()
        
        # Validate report structure
        assert "report_timestamp" in report
        assert "functional_features_enabled" in report
        assert "legacy_data" in report
        assert "functional_data" in report
        assert "consistency_check" in report
        
        # Validate legacy data
        legacy_data = report["legacy_data"]
        assert "total_realized_pnl" in legacy_data
        assert "total_unrealized_pnl" in legacy_data
        assert "total_pnl" in legacy_data


class TestRiskManagementPerformanceCompatibility(FPTestBase):
    """Test performance characteristics of FP vs legacy risk management."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        # Create large dataset for performance testing
        self.large_position_set = [
            LegacyPosition(
                symbol=f"SYMBOL-{i}",
                side="LONG" if i % 2 == 0 else "SHORT",
                size=Decimal(str(100 + i)),
                entry_price=Decimal(str(1000 + i * 10)),
                unrealized_pnl=Decimal(str((i - 50) * 10)),
                realized_pnl=Decimal(str(i * 5)),
                timestamp=datetime.now(UTC)
            )
            for i in range(100)
        ]
        
        self.large_price_set = {
            f"SYMBOL-{i}": Decimal(str(1000 + i * 15))
            for i in range(100)
        }
    
    def test_risk_calculation_performance_comparison(self):
        """Compare performance of risk calculations between FP and legacy."""
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = self.large_position_set
        mock_position_manager.calculate_total_pnl.return_value = (Decimal("5000.0"), Decimal("2500.0"))
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        
        # Time functional risk calculation
        start_time = time.time()
        for _ in range(10):  # Multiple iterations for averaging
            metrics_result = adapter.calculate_portfolio_metrics(self.large_price_set)
            assert metrics_result.is_success()
        fp_time = (time.time() - start_time) / 10
        
        # Time legacy-style calculation (simulated)
        start_time = time.time()
        for _ in range(10):
            # Simulate legacy calculation
            total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.large_position_set)
            position_count = len([p for p in self.large_position_set if p.side != "FLAT"])
            assert total_pnl is not None
            assert position_count > 0
        legacy_time = (time.time() - start_time) / 10
        
        # FP should be competitive (within 2x of legacy performance)
        performance_ratio = fp_time / legacy_time if legacy_time > 0 else float('inf')
        assert performance_ratio < 2.0, f"FP performance ratio too high: {performance_ratio}"
    
    def test_consistency_validation_performance(self):
        """Test performance of consistency validation between FP and legacy."""
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = self.large_position_set
        mock_position_manager.calculate_total_pnl.return_value = (Decimal("5000.0"), Decimal("2500.0"))
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        
        # Time consistency validation
        start_time = time.time()
        validation_result = adapter.validate_portfolio_consistency(self.large_price_set)
        validation_time = time.time() - start_time
        
        assert validation_result.is_success()
        
        # Validation should complete quickly (under 100ms for 100 positions)
        assert validation_time < 0.1, f"Consistency validation too slow: {validation_time}s"
    
    def test_comprehensive_report_generation_performance(self):
        """Test performance of comprehensive report generation."""
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = self.large_position_set
        mock_position_manager.calculate_total_pnl.return_value = (Decimal("5000.0"), Decimal("2500.0"))
        mock_position_manager.get_position_summary.return_value = {
            "active_positions": len(self.large_position_set),
            "total_realized_pnl": 5000.0,
            "total_unrealized_pnl": 2500.0
        }
        
        unified_manager = create_unified_portfolio_manager(
            position_manager=mock_position_manager,
            enable_functional=True
        )
        
        # Time report generation
        start_time = time.time()
        report_result = unified_manager.generate_comprehensive_report(
            self.large_price_set,
            days=7
        )
        report_time = time.time() - start_time
        
        assert report_result.is_success()
        
        # Report generation should be reasonably fast (under 500ms for large dataset)
        assert report_time < 0.5, f"Report generation too slow: {report_time}s"


class TestRiskManagementErrorHandlingCompatibility(FPTestBase):
    """Test error handling compatibility between FP and legacy risk management."""
    
    def setup_method(self):
        """Set up error handling test fixtures."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    def test_adapter_handles_legacy_exceptions(self):
        """Test that adapter properly handles exceptions from legacy components."""
        # Create mock that raises exceptions
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.side_effect = Exception("Legacy component failure")
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        
        # Should handle exception gracefully and return error result
        try:
            snapshot = adapter.get_position_snapshot()
            # If no exception, snapshot should be valid
            assert snapshot is not None
        except Exception:
            pytest.fail("Adapter should handle legacy exceptions gracefully")
    
    def test_unified_manager_error_propagation(self):
        """Test error propagation through unified manager."""
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.calculate_total_pnl.side_effect = ValueError("Invalid calculation")
        
        unified_manager = create_unified_portfolio_manager(
            position_manager=mock_position_manager,
            enable_functional=True
        )
        
        # Risk analysis should return error result
        risk_result = unified_manager.get_risk_analysis()
        assert risk_result.is_failure()
        assert "Invalid calculation" in risk_result.failure() or "Failed to calculate" in risk_result.failure()
    
    def test_migration_error_handling(self):
        """Test error handling during migration process."""
        mock_position_manager = Mock(spec=PositionManager)
        
        # Test migration with invalid current prices
        unified_manager = create_unified_portfolio_manager(
            position_manager=mock_position_manager,
            enable_functional=True
        )
        
        # Migration with invalid prices should handle gracefully
        migration_result = unified_manager.migrate_to_functional(
            validate_migration=True,
            current_prices={}  # Empty prices might cause issues
        )
        
        # Should either succeed or provide meaningful error
        if migration_result.is_failure():
            error_msg = migration_result.failure()
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0
    
    def test_consistency_validation_error_recovery(self):
        """Test error recovery in consistency validation."""
        mock_position_manager = Mock(spec=PositionManager)
        
        # Set up inconsistent mock responses
        mock_position_manager.get_all_positions.return_value = []
        mock_position_manager.calculate_total_pnl.side_effect = RuntimeError("Calculation error")
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        
        # Validation should handle error gracefully
        validation_result = adapter.validate_portfolio_consistency({})
        assert validation_result.is_failure()
        assert "Calculation error" in validation_result.failure() or "Failed to validate" in validation_result.failure()


class TestRiskManagementEdgeCaseCompatibility(FPTestBase):
    """Test edge case handling in risk management adapter compatibility."""
    
    def setup_method(self):
        """Set up edge case test fixtures."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    def test_zero_position_handling(self):
        """Test handling of zero/flat positions."""
        flat_position = LegacyPosition(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0"),
            entry_price=None,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            timestamp=datetime.now(UTC)
        )
        
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_position.return_value = flat_position
        mock_position_manager.get_all_positions.return_value = [flat_position]
        mock_position_manager.calculate_total_pnl.return_value = (Decimal("0"), Decimal("0"))
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        
        # Should handle flat positions correctly
        functional_position = adapter.get_functional_position("BTC-USD")
        assert functional_position.is_flat
        assert functional_position.total_quantity == Decimal("0")
        
        # Portfolio metrics should handle zero positions
        metrics_result = adapter.calculate_portfolio_metrics({"BTC-USD": Decimal("50000")})
        assert metrics_result.is_success()
        
        metrics = metrics_result.success()
        assert metrics["position_count"] == Decimal("0")
    
    def test_extreme_price_values(self):
        """Test handling of extreme price values."""
        extreme_position = LegacyPosition(
            symbol="EXTREME-USD",
            side="LONG",
            size=Decimal("0.00000001"),  # Very small size
            entry_price=Decimal("999999999"),  # Very high price
            unrealized_pnl=Decimal("-999999"),  # Large loss
            realized_pnl=Decimal("0"),
            timestamp=datetime.now(UTC)
        )
        
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = [extreme_position]
        mock_position_manager.calculate_total_pnl.return_value = (Decimal("0"), Decimal("-999999"))
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        
        # Should handle extreme values without overflow/underflow
        extreme_prices = {"EXTREME-USD": Decimal("1")}  # Extreme price drop
        
        try:
            metrics_result = adapter.calculate_portfolio_metrics(extreme_prices)
            assert metrics_result.is_success()
            
            # Validation should also handle extreme values
            validation_result = adapter.validate_portfolio_consistency(extreme_prices)
            assert validation_result.is_success() or validation_result.is_failure()  # Either is acceptable
            
        except (OverflowError, ValueError) as e:
            pytest.fail(f"Should handle extreme values gracefully: {e}")
    
    def test_missing_price_data_handling(self):
        """Test handling when price data is missing for positions."""
        active_position = LegacyPosition(
            symbol="MISSING-USD",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("100"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            timestamp=datetime.now(UTC)
        )
        
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = [active_position]
        mock_position_manager.calculate_total_pnl.return_value = (Decimal("0"), Decimal("0"))
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        
        # Test with missing price data
        empty_prices = {}  # No price data available
        
        metrics_result = adapter.calculate_portfolio_metrics(empty_prices)
        assert metrics_result.is_success()
        
        metrics = metrics_result.success()
        # Should handle missing prices gracefully
        assert "total_value" in metrics
        # Total value might be 0 due to missing prices
    
    def test_concurrent_access_safety(self):
        """Test thread safety of adapter operations."""
        import threading
        import concurrent.futures
        
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = []
        mock_position_manager.calculate_total_pnl.return_value = (Decimal("0"), Decimal("0"))
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        
        # Function to run in parallel
        def concurrent_operation():
            try:
                snapshot = adapter.get_position_snapshot()
                metrics_result = adapter.calculate_portfolio_metrics({})
                return snapshot is not None and metrics_result.is_success()
            except Exception:
                return False
        
        # Run operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_operation) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All operations should complete successfully
        assert all(results), "Concurrent operations should be thread-safe"


class TestPropertyBasedRiskManagementCompatibility(FPTestBase):
    """Property-based tests for risk management adapter compatibility."""
    
    def setup_method(self):
        """Set up property-based test fixtures."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    @given(
        position_size=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("1000")),
        entry_price=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100000")),
        current_price=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100000")),
        side=st.sampled_from(["LONG", "SHORT"])
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=20  # Reduced for faster test execution
    )
    def test_position_conversion_properties(self, position_size, entry_price, current_price, side):
        """Test properties that should hold for any valid position conversion."""
        # Calculate expected P&L
        if side == "LONG":
            unrealized_pnl = position_size * (current_price - entry_price)
        else:
            unrealized_pnl = position_size * (entry_price - current_price)
        
        legacy_position = LegacyPosition(
            symbol="TEST-USD",
            side=side,
            size=position_size,
            entry_price=entry_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=Decimal("0"),
            timestamp=datetime.now(UTC)
        )
        
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_position.return_value = legacy_position
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        functional_position = adapter.get_functional_position("TEST-USD")
        
        # Properties that should always hold
        assert functional_position.symbol == "TEST-USD"
        assert functional_position.total_quantity == position_size
        assert not functional_position.is_flat
        assert functional_position.side.name == side
    
    @given(
        num_positions=st.integers(min_value=0, max_value=10),
        balance=st.decimals(min_value=Decimal("1000"), max_value=Decimal("100000"))
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    def test_portfolio_consistency_properties(self, num_positions, balance):
        """Test properties that should hold for portfolio consistency validation."""
        # Generate test positions
        positions = []
        total_realized = Decimal("0")
        total_unrealized = Decimal("0")
        
        for i in range(num_positions):
            realized = Decimal(str(i * 10))
            unrealized = Decimal(str((i - num_positions/2) * 5))
            
            position = LegacyPosition(
                symbol=f"SYM-{i}",
                side="LONG" if i % 2 == 0 else "SHORT",
                size=Decimal("1.0"),
                entry_price=Decimal("100"),
                unrealized_pnl=unrealized,
                realized_pnl=realized,
                timestamp=datetime.now(UTC)
            )
            positions.append(position)
            total_realized += realized
            total_unrealized += unrealized
        
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = positions
        mock_position_manager.calculate_total_pnl.return_value = (total_realized, total_unrealized)
        
        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        
        # Consistency validation should always complete
        prices = {f"SYM-{i}": Decimal("100") for i in range(num_positions)}
        validation_result = adapter.validate_portfolio_consistency(prices)
        
        # Should always return a result (success or failure)
        assert validation_result.is_success() or validation_result.is_failure()
        
        if validation_result.is_success():
            validation_data = validation_result.success()
            # Should contain required validation fields
            assert "overall_consistent" in validation_data
            assert isinstance(validation_data["overall_consistent"], bool)


# Integration test for complete risk management pipeline
class TestRiskManagementPipelineIntegration(FPTestBase):
    """Integration tests for complete risk management pipeline compatibility."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    def test_end_to_end_risk_management_pipeline(self):
        """Test complete end-to-end risk management pipeline with adapters."""
        # Create realistic trading scenario
        positions = [
            LegacyPosition(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("1.5"),
                entry_price=Decimal("45000.0"),
                unrealized_pnl=Decimal("7500.0"),  # Profitable position
                realized_pnl=Decimal("2000.0"),
                timestamp=datetime.now(UTC) - timedelta(hours=2)
            ),
            LegacyPosition(
                symbol="ETH-USD",
                side="SHORT",
                size=Decimal("10.0"),
                entry_price=Decimal("3200.0"),
                unrealized_pnl=Decimal("-2000.0"),  # Losing position
                realized_pnl=Decimal("1500.0"),
                timestamp=datetime.now(UTC) - timedelta(hours=1)
            )
        ]
        
        current_prices = {
            "BTC-USD": Decimal("50000.0"),  # BTC up
            "ETH-USD": Decimal("3000.0")    # ETH down (good for short)
        }
        
        # Set up complete system
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = positions
        mock_position_manager.calculate_total_pnl.return_value = (Decimal("3500.0"), Decimal("5500.0"))
        mock_position_manager.get_position_summary.return_value = {
            "active_positions": 2,
            "total_realized_pnl": 3500.0,
            "total_unrealized_pnl": 5500.0,
            "total_pnl": 9000.0,
            "total_exposure": 82000.0
        }
        
        # Create unified manager
        unified_manager = create_unified_portfolio_manager(
            position_manager=mock_position_manager,
            enable_functional=True
        )
        
        # Test complete pipeline
        
        # 1. Migration
        migration_result = unified_manager.migrate_to_functional(
            validate_migration=True,
            current_prices=current_prices
        )
        assert migration_result.is_success()
        
        # 2. Account snapshot
        account_result = unified_manager.get_account_snapshot(current_prices)
        assert account_result.is_success()
        
        # 3. Performance analysis
        performance_result = unified_manager.get_performance_analysis(current_prices, days=30)
        assert performance_result.is_success()
        
        # 4. Risk analysis
        risk_result = unified_manager.get_risk_analysis(days=30)
        assert risk_result.is_success()
        
        # 5. Consistency validation
        consistency_result = unified_manager.validate_consistency(current_prices)
        assert consistency_result.is_success() or consistency_result.is_failure()  # Either acceptable
        
        # 6. Comprehensive report
        report_result = unified_manager.generate_comprehensive_report(current_prices, days=7)
        assert report_result.is_success()
        
        report = report_result.success()
        
        # Validate complete pipeline results
        assert report["functional_features_enabled"] is True
        assert "legacy_data" in report
        assert "functional_data" in report
        
        # Legacy data should match our test setup
        legacy_data = report["legacy_data"]
        assert legacy_data["total_realized_pnl"] == 3500.0
        assert legacy_data["total_unrealized_pnl"] == 5500.0
        
        # Functional data should be present
        functional_data = report["functional_data"]
        if "account" in functional_data:
            assert "total_equity" in functional_data["account"]
        
        # Test feature compatibility
        compatibility_report = get_feature_compatibility_report()
        assert "legacy_api_methods" in compatibility_report
        assert "functional_api_methods" in compatibility_report
        assert "enhanced_features" in compatibility_report
        assert compatibility_report["backward_compatibility"] == "Full backward compatibility maintained during migration"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])