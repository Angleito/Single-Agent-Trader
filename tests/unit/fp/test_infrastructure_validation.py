"""
FP Test Infrastructure Validation

This module validates that the FP test infrastructure is working correctly
and provides examples of how to use the various components.
"""

from decimal import Decimal

import pytest

from tests.fp_test_base import (
    FP_AVAILABLE,
    FPExchangeTestBase,
    FPRiskTestBase,
    FPStrategyTestBase,
    FPTestBase,
)

if FP_AVAILABLE:
    from bot.fp.types.effects import IO, Err, Nothing, Ok, Some
    from bot.fp.types.trading import Hold, Long, Short


class TestFPTestInfrastructure(FPTestBase):
    """Test the FP test infrastructure itself."""

    def test_fp_assertions_work(self):
        """Test that FP assertions work correctly."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test Result assertions
        ok_result = Ok(42)
        self.assert_result_ok(ok_result, 42)

        err_result = Err("test error")
        self.assert_result_err(err_result, "test error")

        # Test Maybe assertions
        some_value = Some(100)
        self.assert_maybe_some(some_value, 100)

        nothing_value = Nothing()
        self.assert_maybe_nothing(nothing_value)

        # Test IO assertions
        io_computation = IO.pure(200)
        self.assert_io_result(io_computation, 200)

    def test_fp_utilities_work(self):
        """Test that FP utilities work correctly."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test IO execution
        io_comp = IO.pure(300)
        result = self.run_io(io_comp)
        assert result == 300

        # Test safe IO execution
        def failing_computation():
            raise ValueError("Test error")

        failing_io = IO(failing_computation)
        result = self.run_io_safe(failing_io, default="fallback")
        assert result == "fallback"

    def test_mock_creation_utilities(self):
        """Test mock creation utilities."""
        # Test Result mock
        mock = self.create_mock_with_fp_result("get_value", 42, is_error=False)
        result = mock.get_value()
        self.assert_result_ok(result, 42)

        # Test Maybe mock
        mock = self.create_mock_with_fp_maybe("get_optional", 100)
        maybe = mock.get_optional()
        self.assert_maybe_some(maybe, 100)

        # Test IO mock
        mock = self.create_mock_with_fp_io("compute", 200)
        io_result = mock.compute()
        self.assert_io_result(io_result, 200)


class TestFPExchangeTestInfrastructure(FPExchangeTestBase):
    """Test the exchange-specific FP test infrastructure."""

    def test_exchange_mock_creation(self):
        """Test exchange mock creation."""
        adapter = self.create_mock_exchange_adapter(balance=Decimal("5000.00"))

        # Test balance retrieval
        balance_io = adapter.get_balance()
        balance_result = self.run_io(balance_io)
        self.assert_result_ok(balance_result, Decimal("5000.00"))

        # Test order placement
        order_io = adapter.place_order({})
        order_result = self.run_io(order_io)
        self.assert_result_ok(order_result, {"order_id": "test-123"})

    def test_market_snapshot_creation(self):
        """Test market snapshot creation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        snapshot = self.create_test_market_snapshot(
            price=Decimal("60000.00"), symbol="ETH-USD"
        )

        assert snapshot.symbol == "ETH-USD"
        assert snapshot.price == Decimal("60000.00")
        assert snapshot.bid <= snapshot.price <= snapshot.ask

    def test_order_assertion(self):
        """Test order placement assertion."""
        adapter = self.create_mock_exchange_adapter()

        # Simulate order placement
        adapter.place_order({"symbol": "BTC-USD"})

        # Test assertion
        self.assert_order_placed(adapter, "BTC-USD")


class TestFPStrategyTestInfrastructure(FPStrategyTestBase):
    """Test the strategy-specific FP test infrastructure."""

    def test_strategy_mock_creation(self):
        """Test strategy mock creation."""
        strategy = self.create_mock_strategy()

        # Test signal generation
        signal_io = strategy.generate_signal()
        signal_result = self.run_io(signal_io)

        if FP_AVAILABLE:
            self.assert_result_ok(signal_result)
            signal = signal_result.unwrap()
            assert isinstance(signal, Hold)

    def test_signal_creation_and_assertions(self):
        """Test signal creation and assertions."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test Long signal
        long_signal = self.create_test_long_signal(confidence=0.9, size=0.3)
        self.assert_signal_type(long_signal, Long)
        self.assert_signal_confidence(
            long_signal, min_confidence=0.8, max_confidence=1.0
        )

        # Test Short signal
        short_signal = self.create_test_short_signal(confidence=0.8, size=0.2)
        self.assert_signal_type(short_signal, Short)
        self.assert_signal_confidence(
            short_signal, min_confidence=0.7, max_confidence=0.9
        )

        # Test Hold signal
        hold_signal = self.create_test_hold_signal("Testing hold")
        self.assert_signal_type(hold_signal, Hold)


class TestFPRiskTestInfrastructure(FPRiskTestBase):
    """Test the risk management FP test infrastructure."""

    def test_risk_manager_mock_creation(self):
        """Test risk manager mock creation."""
        risk_manager = self.create_mock_risk_manager()

        # Test trade validation
        validation_io = risk_manager.validate_trade({})
        validation_result = self.run_io(validation_io)
        self.assert_result_ok(validation_result, True)

        # Test position size calculation
        size_io = risk_manager.calculate_position_size({})
        size_result = self.run_io(size_io)
        self.assert_result_ok(size_result, 0.1)

    def test_position_creation_and_assertions(self):
        """Test position creation and risk assertions."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        position = self.create_test_position(side="LONG", size=Decimal("0.2"))

        assert position.symbol == "BTC-USD"
        assert position.side == "LONG"
        assert position.size == Decimal("0.2")

        # Test risk assessment
        risk_metrics = {"risk_score": 0.3, "max_risk": 1.0}
        self.assert_risk_within_limits(risk_metrics, max_risk=1.0)


def test_fp_fixtures_available(
    fp_result_ok,
    fp_result_err,
    fp_maybe_some,
    fp_maybe_nothing,
    fp_io_pure,
    fp_market_snapshot,
    fp_position,
    fp_test_utils,
):
    """Test that FP fixtures are available and working."""
    if not FP_AVAILABLE:
        pytest.skip("FP types not available")

    # Test Result fixtures
    assert fp_result_ok.is_ok()
    assert fp_result_ok.unwrap() == 42

    assert fp_result_err.is_err()
    assert fp_result_err.error == "Test error"

    # Test Maybe fixtures
    assert fp_maybe_some.is_some()
    assert fp_maybe_some.unwrap() == 42

    assert fp_maybe_nothing.is_nothing()

    # Test IO fixture
    assert fp_io_pure.run() == 42

    # Test domain fixtures
    assert fp_market_snapshot.symbol == "BTC-USD"
    assert fp_position.symbol == "BTC-USD"

    # Test utilities
    fp_test_utils.assert_result_ok(fp_result_ok, 42)
    fp_test_utils.assert_maybe_some(fp_maybe_some, 42)


def test_fp_mock_fixtures(
    fp_mock_exchange_adapter, fp_mock_strategy, fp_mock_risk_manager
):
    """Test that FP mock fixtures work correctly."""
    if not FP_AVAILABLE:
        pytest.skip("FP types not available")

    # Test exchange adapter mock
    balance_io = fp_mock_exchange_adapter.get_balance()
    balance_result = balance_io.run()
    assert balance_result.is_ok()
    assert balance_result.unwrap() == Decimal("10000.00")

    # Test strategy mock
    signal_io = fp_mock_strategy.generate_signal()
    signal_result = signal_io.run()
    assert signal_result.is_ok()

    # Test risk manager mock
    validation_io = fp_mock_risk_manager.validate_trade()
    validation_result = validation_io.run()
    assert validation_result.is_ok()
    assert validation_result.unwrap() == True


def test_fp_property_strategies(fp_property_strategies):
    """Test that property-based testing strategies work."""
    if not FP_AVAILABLE or not fp_property_strategies:
        pytest.skip("FP types or strategies not available")

    # Test that strategies are available
    assert "decimal" in fp_property_strategies
    assert "timestamp" in fp_property_strategies
    assert "symbol" in fp_property_strategies

    # Test strategy usage (basic validation)
    decimal_strategy = fp_property_strategies["decimal"]
    assert decimal_strategy is not None


def test_migration_scenario_fixture(fp_migration_scenario):
    """Test migration scenario fixture."""
    # This fixture should provide either FP or imperative scenario
    assert "type" in fp_migration_scenario
    assert "market_data" in fp_migration_scenario
    assert "is_fp" in fp_migration_scenario

    scenario_type = fp_migration_scenario["type"]
    assert scenario_type in ["functional", "imperative"]


class TestInfrastructureCompatibility:
    """Test infrastructure compatibility with missing FP types."""

    def test_graceful_degradation_without_fp(self):
        """Test that infrastructure works even without FP types."""
        # This should not raise even if FP types are missing
        base = FPTestBase()

        # Methods should exist even with fallback implementations
        assert hasattr(base, "assert_result_ok")
        assert hasattr(base, "assert_maybe_some")
        assert hasattr(base, "run_io")

    def test_fp_availability_flag(self):
        """Test FP availability flag."""
        # Flag should be a boolean
        assert isinstance(FP_AVAILABLE, bool)

        # If True, we should be able to import at least some FP types
        if FP_AVAILABLE:
            try:
                from bot.fp.types.effects import Err, Ok

                assert Ok is not None
                assert Err is not None
            except ImportError:
                pytest.fail("FP_AVAILABLE is True but FP types cannot be imported")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
