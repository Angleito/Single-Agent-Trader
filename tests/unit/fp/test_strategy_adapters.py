"""
Comprehensive tests for strategy adapter classes that bridge FP and legacy systems.

This module tests the adapter layer that enables seamless migration between legacy
imperative trading code and functional programming trading architecture while
maintaining all functionality and backward compatibility.

Tests include:
- TradingTypeAdapter for type conversions between FP and legacy formats
- OrderExecutionAdapter for order preparation and execution
- FunctionalTradingIntegration for unified adapter operations
- FunctionalPositionManagerAdapter for position management
- FunctionalPortfolioManager compatibility layer
- Error handling and edge cases in all adapters
- Performance consistency between FP and legacy systems
- Migration validation and consistency checks
"""

from datetime import UTC, datetime
from decimal import Decimal, getcontext
from unittest.mock import Mock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# FP test infrastructure
from tests.fp_test_base import FP_AVAILABLE, FPTestBase

if FP_AVAILABLE:
    # Trading type adapter
    # Compatibility layer
    from bot.fp.adapters.compatibility_layer import (
        FunctionalPortfolioManager,
        create_unified_portfolio_manager,
        get_feature_compatibility_report,
        migrate_existing_system,
    )

    # Position manager adapter
    from bot.fp.adapters.position_manager_adapter import (
        FunctionalPositionManagerAdapter,
        migrate_legacy_positions_to_functional,
        validate_functional_migration,
    )
    from bot.fp.adapters.trading_type_adapter import (
        FunctionalTradingIntegration,
        OrderExecutionAdapter,
        RiskAdapterMixin,
        TradingTypeAdapter,
    )
    from bot.fp.types.portfolio import AccountSnapshot, AccountType
    from bot.fp.types.positions import FunctionalPosition, PositionSnapshot

    # FP types
    from bot.fp.types.trading import (
        FunctionalMarketData,
        FunctionalMarketState,
        LimitOrder,
        MarketOrder,
        Position,
        RiskLimits,
        RiskMetrics,
        StopOrder,
        TradingIndicators,
    )
else:
    # Fallback stubs for non-FP environments
    class TradingTypeAdapter:
        pass

    def create_unified_portfolio_manager(*args, **kwargs):
        return None


# Legacy types
from bot.paper_trading import PaperTradingAccount
from bot.position_manager import PositionManager
from bot.trading_types import Order
from bot.trading_types import Position as LegacyPosition

# Set high precision for financial calculations
getcontext().prec = 28


class TestTradingTypeAdapter(FPTestBase):
    """Test TradingTypeAdapter for type conversions between FP and legacy formats."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        self.adapter = TradingTypeAdapter()

        # Sample legacy market data
        self.legacy_market_data = {
            "symbol": "BTC-USD",
            "timestamp": datetime.now(UTC),
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0,
        }

        # Sample legacy position
        self.legacy_position = {
            "symbol": "BTC-USD",
            "side": "LONG",
            "size": 0.1,
            "entry_price": 50000.0,
            "unrealized_pnl": 50.0,
            "realized_pnl": 0.0,
            "timestamp": datetime.now(UTC),
        }

        # Sample legacy indicators
        self.legacy_indicators = {
            "timestamp": datetime.now(UTC),
            "rsi": 65.5,
            "cipher_a_dot": 0.8,
            "cipher_b_wave": -0.2,
            "cipher_b_money_flow": 0.5,
            "stablecoin_dominance": 0.05,
            "dominance_trend": "BULLISH",
        }

    def test_adapt_market_data_from_dict(self):
        """Test adapting market data from dictionary format."""
        functional_data = self.adapter.adapt_market_data_to_functional(
            self.legacy_market_data
        )

        assert isinstance(functional_data, FunctionalMarketData)
        assert functional_data.symbol == "BTC-USD"
        assert functional_data.open == Decimal(50000)
        assert functional_data.high == Decimal(51000)
        assert functional_data.low == Decimal(49000)
        assert functional_data.close == Decimal(50500)
        assert functional_data.volume == Decimal(100)

    def test_adapt_market_data_from_pydantic(self):
        """Test adapting market data from Pydantic model."""
        # Create mock Pydantic model
        mock_pydantic = Mock()
        mock_pydantic.dict.return_value = self.legacy_market_data

        with patch(
            "bot.fp.types.trading.convert_pydantic_to_functional_market_data"
        ) as mock_convert:
            mock_convert.return_value = FunctionalMarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(UTC),
                open=Decimal(50000),
                high=Decimal(51000),
                low=Decimal(49000),
                close=Decimal(50500),
                volume=Decimal(100),
            )

            functional_data = self.adapter.adapt_market_data_to_functional(
                mock_pydantic
            )

            assert isinstance(functional_data, FunctionalMarketData)
            mock_convert.assert_called_once_with(mock_pydantic)

    def test_adapt_market_data_invalid_format(self):
        """Test error handling for invalid market data format."""
        with pytest.raises(ValueError, match="Unsupported market data format"):
            self.adapter.adapt_market_data_to_functional("invalid_data")

    def test_adapt_position_from_dict(self):
        """Test adapting position from dictionary format."""
        functional_position = self.adapter.adapt_position_to_functional(
            self.legacy_position
        )

        assert isinstance(functional_position, Position)
        assert functional_position.symbol == "BTC-USD"
        assert functional_position.side == "LONG"
        assert functional_position.size == Decimal("0.1")
        assert functional_position.entry_price == Decimal(50000)

    def test_adapt_position_from_pydantic(self):
        """Test adapting position from Pydantic model."""
        mock_pydantic = Mock()
        mock_pydantic.dict.return_value = self.legacy_position

        with patch(
            "bot.fp.types.trading.convert_pydantic_to_functional_position"
        ) as mock_convert:
            mock_convert.return_value = Position(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("0.1"),
                entry_price=Decimal(50000),
                unrealized_pnl=Decimal(50),
                realized_pnl=Decimal(0),
                timestamp=datetime.now(UTC),
            )

            functional_position = self.adapter.adapt_position_to_functional(
                mock_pydantic
            )

            assert isinstance(functional_position, Position)
            mock_convert.assert_called_once_with(mock_pydantic)

    def test_adapt_indicators_from_dict(self):
        """Test adapting indicators from dictionary format."""
        functional_indicators = self.adapter.adapt_indicators_to_functional(
            self.legacy_indicators
        )

        assert isinstance(functional_indicators, TradingIndicators)
        assert functional_indicators.rsi == 65.5
        assert functional_indicators.cipher_a_dot == 0.8
        assert functional_indicators.cipher_b_wave == -0.2
        assert functional_indicators.stablecoin_dominance == 0.05
        assert functional_indicators.dominance_trend == "BULLISH"

    def test_adapt_indicators_from_pydantic(self):
        """Test adapting indicators from Pydantic model."""
        mock_pydantic = Mock()
        mock_pydantic.dict.return_value = self.legacy_indicators

        functional_indicators = self.adapter.adapt_indicators_to_functional(
            mock_pydantic
        )

        assert isinstance(functional_indicators, TradingIndicators)
        mock_pydantic.dict.assert_called_once()

    def test_create_functional_market_state(self):
        """Test creating functional market state from legacy components."""
        functional_state = self.adapter.adapt_legacy_to_functional_state(
            symbol="BTC-USD",
            legacy_market_data=self.legacy_market_data,
            legacy_indicators=self.legacy_indicators,
            legacy_position=self.legacy_position,
        )

        assert isinstance(functional_state, FunctionalMarketState)
        assert functional_state.symbol == "BTC-USD"
        assert isinstance(functional_state.market_data, FunctionalMarketData)
        assert isinstance(functional_state.indicators, TradingIndicators)
        assert isinstance(functional_state.position, Position)

    def test_convert_functional_state_to_legacy(self):
        """Test converting functional market state back to legacy format."""
        functional_state = self.adapter.adapt_legacy_to_functional_state(
            symbol="BTC-USD",
            legacy_market_data=self.legacy_market_data,
            legacy_indicators=self.legacy_indicators,
            legacy_position=self.legacy_position,
        )

        with patch(
            "bot.fp.types.trading.convert_functional_to_pydantic_position"
        ) as mock_convert:
            mock_convert.return_value = self.legacy_position

            legacy_data = self.adapter.convert_functional_state_to_legacy(
                functional_state
            )

            assert isinstance(legacy_data, dict)
            assert legacy_data["symbol"] == "BTC-USD"
            assert "market_data" in legacy_data
            assert "indicators" in legacy_data
            assert "position" in legacy_data

    @given(
        st.decimals(min_value=1, max_value=100),
        st.decimals(min_value=1000, max_value=100000),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_based_market_data_conversion(self, size, price):
        """Test market data conversion with property-based testing."""
        test_data = {
            "symbol": "TEST-USD",
            "timestamp": datetime.now(UTC),
            "open": float(price),
            "high": float(price * Decimal("1.01")),
            "low": float(price * Decimal("0.99")),
            "close": float(price * Decimal("1.005")),
            "volume": float(size),
        }

        functional_data = self.adapter.adapt_market_data_to_functional(test_data)

        assert functional_data.symbol == "TEST-USD"
        assert functional_data.open == Decimal(str(test_data["open"]))
        assert functional_data.volume == Decimal(str(test_data["volume"]))


class TestOrderExecutionAdapter(FPTestBase):
    """Test OrderExecutionAdapter for order preparation and execution."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        self.adapter = OrderExecutionAdapter()

        # Sample functional orders
        self.limit_order = LimitOrder(
            symbol="BTC-USD",
            side="buy",
            size=0.1,
            price=50000.0,
            post_only=False,
            time_in_force="GTC",
            order_id="limit_123",
        )

        self.market_order = MarketOrder(
            symbol="BTC-USD", side="sell", size=0.05, order_id="market_456"
        )

        self.stop_order = StopOrder(
            symbol="BTC-USD",
            side="sell",
            size=0.08,
            stop_price=49000.0,
            time_in_force="GTC",
            order_id="stop_789",
        )

    def test_prepare_limit_order_for_execution(self):
        """Test preparing limit order for execution."""
        prepared_order = self.adapter.prepare_order_for_execution(self.limit_order)

        assert prepared_order["symbol"] == "BTC-USD"
        assert prepared_order["side"] == "BUY"
        assert prepared_order["size"] == "0.1"
        assert prepared_order["type"] == "LIMIT"
        assert prepared_order["price"] == "50000.0"
        assert prepared_order["order_id"] == "limit_123"

    def test_prepare_market_order_for_execution(self):
        """Test preparing market order for execution."""
        prepared_order = self.adapter.prepare_order_for_execution(self.market_order)

        assert prepared_order["symbol"] == "BTC-USD"
        assert prepared_order["side"] == "SELL"
        assert prepared_order["size"] == "0.05"
        assert prepared_order["type"] == "MARKET"
        assert prepared_order["order_id"] == "market_456"

    def test_prepare_stop_order_for_execution(self):
        """Test preparing stop order for execution."""
        prepared_order = self.adapter.prepare_order_for_execution(self.stop_order)

        assert prepared_order["symbol"] == "BTC-USD"
        assert prepared_order["side"] == "SELL"
        assert prepared_order["size"] == "0.08"
        assert prepared_order["type"] == "STOP"
        assert prepared_order["stop_price"] == "49000.0"
        assert prepared_order["order_id"] == "stop_789"

    def test_format_for_coinbase(self):
        """Test Coinbase-specific order formatting."""
        prepared_order = self.adapter.prepare_order_for_execution(
            self.limit_order, exchange_format="coinbase"
        )

        assert prepared_order["product_id"] == "BTC-USD"
        assert prepared_order["client_order_id"] == "limit_123"
        assert prepared_order["limit_price"] == "50000.0"
        assert "symbol" not in prepared_order
        assert "order_id" not in prepared_order

    def test_format_for_bluefin(self):
        """Test Bluefin-specific order formatting."""
        prepared_order = self.adapter.prepare_order_for_execution(
            self.limit_order, exchange_format="bluefin"
        )

        assert prepared_order["symbol"] == "BTC-USD"
        assert prepared_order["quantity"] == int(0.1 * 1e18)  # Wei conversion
        assert prepared_order["price"] == int(50000.0 * 1e18)
        assert "size" not in prepared_order

    def test_order_preparation_error_handling(self):
        """Test error handling in order preparation."""
        invalid_order = Mock()
        invalid_order.symbol = None  # Invalid symbol

        with pytest.raises(ValueError, match="Order preparation failed"):
            self.adapter.prepare_order_for_execution(invalid_order)


class TestRiskAdapterMixin(FPTestBase):
    """Test RiskAdapterMixin for risk management adaptation."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        self.risk_adapter = RiskAdapterMixin()

        # Sample legacy risk data
        self.legacy_risk_data = {
            "account_balance": 10000.0,
            "available_margin": 8000.0,
            "used_margin": 2000.0,
            "daily_pnl": 150.0,
            "total_exposure": 5000.0,
            "current_positions": 3,
            "max_daily_loss_reached": False,
            "value_at_risk_95": 500.0,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "max_drawdown": 0.05,
        }

        # Sample risk limits
        self.risk_limits = RiskLimits(
            max_position_size=Decimal(1000),
            max_leverage=5,
            max_daily_loss=Decimal(500),
            max_drawdown=Decimal("0.1"),
            concentration_limit=Decimal("0.3"),
            var_limit=Decimal(1000),
        )

    def test_adapt_risk_metrics_from_dict(self):
        """Test adapting risk metrics from dictionary format."""
        risk_metrics = self.risk_adapter.adapt_risk_metrics_to_functional(
            self.legacy_risk_data
        )

        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.account_balance == Decimal(10000)
        assert risk_metrics.available_margin == Decimal(8000)
        assert risk_metrics.daily_pnl == Decimal(150)
        assert risk_metrics.sharpe_ratio == 1.2
        assert risk_metrics.max_drawdown == 0.05

    def test_adapt_risk_metrics_from_pydantic(self):
        """Test adapting risk metrics from Pydantic model."""
        mock_pydantic = Mock()
        mock_pydantic.dict.return_value = self.legacy_risk_data

        risk_metrics = self.risk_adapter.adapt_risk_metrics_to_functional(mock_pydantic)

        assert isinstance(risk_metrics, RiskMetrics)
        mock_pydantic.dict.assert_called_once()

    def test_validate_functional_risk_compliance_success(self):
        """Test successful risk compliance validation."""
        # Create functional market state
        functional_state = Mock(spec=FunctionalMarketState)
        functional_state.symbol = "BTC-USD"
        functional_state.has_position = True
        functional_state.position_value = Decimal(500)  # Within limits
        functional_state.risk_metrics = None
        functional_state.account_balance = None

        validation = self.risk_adapter.validate_functional_risk_compliance(
            functional_state, self.risk_limits
        )

        assert validation["compliant"] is True
        assert len(validation["violations"]) == 0
        assert validation["symbol"] == "BTC-USD"

    def test_validate_functional_risk_compliance_violation(self):
        """Test risk compliance validation with violations."""
        # Create functional market state with position exceeding limits
        functional_state = Mock(spec=FunctionalMarketState)
        functional_state.symbol = "BTC-USD"
        functional_state.has_position = True
        functional_state.position_value = Decimal(2000)  # Exceeds limit
        functional_state.risk_metrics = None
        functional_state.account_balance = None

        validation = self.risk_adapter.validate_functional_risk_compliance(
            functional_state, self.risk_limits
        )

        assert validation["compliant"] is False
        assert len(validation["violations"]) > 0
        assert "exceeds limit" in validation["violations"][0]


class TestFunctionalTradingIntegration(FPTestBase):
    """Test FunctionalTradingIntegration for unified adapter operations."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        self.integration = FunctionalTradingIntegration()

        # Sample legacy data batch
        self.legacy_batch = [
            {
                "symbol": "BTC-USD",
                "market_data": {
                    "symbol": "BTC-USD",
                    "open": 50000.0,
                    "high": 51000.0,
                    "low": 49000.0,
                    "close": 50500.0,
                    "volume": 100.0,
                    "timestamp": datetime.now(UTC),
                },
                "indicators": {
                    "rsi": 65.5,
                    "cipher_a_dot": 0.8,
                    "timestamp": datetime.now(UTC),
                },
                "position": {
                    "symbol": "BTC-USD",
                    "side": "LONG",
                    "size": 0.1,
                    "entry_price": 50000.0,
                    "timestamp": datetime.now(UTC),
                },
            },
            {
                "symbol": "ETH-USD",
                "market_data": {
                    "symbol": "ETH-USD",
                    "open": 3000.0,
                    "high": 3100.0,
                    "low": 2900.0,
                    "close": 3050.0,
                    "volume": 500.0,
                    "timestamp": datetime.now(UTC),
                },
                "indicators": {
                    "rsi": 45.2,
                    "cipher_a_dot": -0.3,
                    "timestamp": datetime.now(UTC),
                },
                "position": {
                    "symbol": "ETH-USD",
                    "side": "FLAT",
                    "size": 0.0,
                    "timestamp": datetime.now(UTC),
                },
            },
        ]

    def test_batch_convert_legacy_data_success(self):
        """Test successful batch conversion of legacy data."""
        functional_states = self.integration.batch_convert_legacy_data(
            self.legacy_batch
        )

        assert len(functional_states) == 2
        assert all(
            isinstance(state, FunctionalMarketState) for state in functional_states
        )
        assert functional_states[0].symbol == "BTC-USD"
        assert functional_states[1].symbol == "ETH-USD"

        # Check conversion statistics
        stats = self.integration.get_conversion_statistics()
        assert stats["market_data_conversions"] == 2
        assert stats["failures"] == 0

    def test_batch_convert_with_errors(self):
        """Test batch conversion with some errors."""
        # Add invalid data item
        invalid_batch = self.legacy_batch + [
            {
                "symbol": "INVALID",
                "market_data": None,  # Invalid market data
                "indicators": {},
                "position": {},
            }
        ]

        functional_states = self.integration.batch_convert_legacy_data(invalid_batch)

        # Should have 2 successful conversions, 1 failure
        assert len(functional_states) == 2

        stats = self.integration.get_conversion_statistics()
        assert stats["market_data_conversions"] == 2
        assert stats["failures"] == 1

    def test_conversion_statistics_tracking(self):
        """Test conversion statistics tracking and reset."""
        # Initial state
        stats = self.integration.get_conversion_statistics()
        initial_conversions = stats["market_data_conversions"]

        # Perform conversion
        self.integration.batch_convert_legacy_data(self.legacy_batch)

        # Check updated stats
        stats = self.integration.get_conversion_statistics()
        assert stats["market_data_conversions"] == initial_conversions + 2

        # Reset statistics
        self.integration.reset_statistics()
        stats = self.integration.get_conversion_statistics()
        assert stats["market_data_conversions"] == 0
        assert stats["failures"] == 0


class TestFunctionalPositionManagerAdapter(FPTestBase):
    """Test FunctionalPositionManagerAdapter for position management bridging."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Create mock legacy position manager
        self.mock_position_manager = Mock(spec=PositionManager)
        self.adapter = FunctionalPositionManagerAdapter(self.mock_position_manager)

        # Sample legacy position
        self.legacy_position = LegacyPosition(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal(50000),
            unrealized_pnl=Decimal(50),
            realized_pnl=Decimal(0),
            timestamp=datetime.now(UTC),
        )

        # Mock position manager responses
        self.mock_position_manager.get_position.return_value = self.legacy_position
        self.mock_position_manager.get_all_positions.return_value = [
            self.legacy_position
        ]
        self.mock_position_manager.calculate_total_pnl.return_value = (
            Decimal(100),
            Decimal(50),
        )

    def test_get_functional_position(self):
        """Test getting functional position from legacy position manager."""
        functional_position = self.adapter.get_functional_position("BTC-USD")

        assert isinstance(functional_position, FunctionalPosition)
        assert functional_position.symbol == "BTC-USD"
        self.mock_position_manager.get_position.assert_called_once_with("BTC-USD")

    def test_get_all_functional_positions(self):
        """Test getting all functional positions."""
        functional_positions = self.adapter.get_all_functional_positions()

        assert len(functional_positions) == 1
        assert isinstance(functional_positions[0], FunctionalPosition)
        self.mock_position_manager.get_all_positions.assert_called_once()

    def test_get_position_snapshot(self):
        """Test getting position snapshot in functional form."""
        snapshot = self.adapter.get_position_snapshot()

        assert isinstance(snapshot, PositionSnapshot)
        assert len(snapshot.positions) == 1
        assert snapshot.total_unrealized_pnl >= Decimal(0)
        assert snapshot.total_realized_pnl >= Decimal(0)

    def test_get_account_snapshot_spot(self):
        """Test getting spot account snapshot."""
        current_prices = {"BTC-USD": Decimal(51000)}

        result = self.adapter.get_account_snapshot(
            current_prices, AccountType.SPOT, "USD"
        )

        assert result.is_success()
        account_snapshot = result.success()
        assert isinstance(account_snapshot, AccountSnapshot)
        assert account_snapshot.account_type == AccountType.SPOT
        assert account_snapshot.base_currency == "USD"

    def test_get_account_snapshot_futures(self):
        """Test getting futures account snapshot."""
        current_prices = {"BTC-USD": Decimal(51000)}

        result = self.adapter.get_account_snapshot(
            current_prices, AccountType.FUTURES, "USD"
        )

        assert result.is_success()
        account_snapshot = result.success()
        assert isinstance(account_snapshot, AccountSnapshot)
        assert account_snapshot.account_type == AccountType.FUTURES

    def test_get_portfolio_performance(self):
        """Test getting portfolio performance analysis."""
        current_prices = {"BTC-USD": Decimal(51000)}
        account_balance = Decimal(10000)

        result = self.adapter.get_portfolio_performance(current_prices, account_balance)

        assert result.is_success()
        performances = result.success()
        assert isinstance(performances, list)

    def test_update_position_from_order_functional(self):
        """Test updating position from order using functional types."""
        mock_order = Mock(spec=Order)
        mock_order.symbol = "BTC-USD"
        mock_order.side = "BUY"
        mock_order.quantity = Decimal("0.05")

        fill_price = Decimal(50500)

        # Mock the legacy position manager update
        updated_position = LegacyPosition(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.15"),  # Increased size
            entry_price=Decimal(50200),  # Updated entry price
            unrealized_pnl=Decimal(75),
            realized_pnl=Decimal(0),
            timestamp=datetime.now(UTC),
        )
        self.mock_position_manager.update_position_from_order.return_value = (
            updated_position
        )

        result = self.adapter.update_position_from_order_functional(
            mock_order, fill_price
        )

        assert result.is_success()
        functional_position = result.success()
        assert isinstance(functional_position, FunctionalPosition)
        assert functional_position.total_quantity == Decimal("0.15")

    def test_calculate_portfolio_metrics(self):
        """Test calculating comprehensive portfolio metrics."""
        current_prices = {"BTC-USD": Decimal(51000)}

        result = self.adapter.calculate_portfolio_metrics(current_prices)

        assert result.is_success()
        metrics = result.success()
        assert isinstance(metrics, dict)
        assert "total_pnl" in metrics
        assert "realized_pnl" in metrics
        assert "unrealized_pnl" in metrics
        assert "position_count" in metrics

    def test_validate_portfolio_consistency(self):
        """Test validating consistency between legacy and functional representations."""
        current_prices = {"BTC-USD": Decimal(51000)}

        result = self.adapter.validate_portfolio_consistency(current_prices)

        assert result.is_success()
        validation = result.success()
        assert isinstance(validation, dict)
        assert "overall_consistent" in validation
        assert "realized_pnl_match" in validation
        assert "position_count_match" in validation

    def test_get_functional_summary(self):
        """Test getting comprehensive functional portfolio summary."""
        summary = self.adapter.get_functional_summary()

        assert isinstance(summary, dict)
        assert "timestamp" in summary
        assert "active_positions" in summary
        assert "total_pnl" in summary
        assert summary["functional_types_enabled"] is True
        assert summary["adapter_version"] == "1.0.0"


class TestFunctionalPortfolioManager(FPTestBase):
    """Test FunctionalPortfolioManager compatibility layer."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Create mock legacy components
        self.mock_position_manager = Mock(spec=PositionManager)
        self.mock_paper_account = Mock(spec=PaperTradingAccount)

        # Sample legacy position
        self.legacy_position = LegacyPosition(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal(50000),
            unrealized_pnl=Decimal(50),
            realized_pnl=Decimal(0),
            timestamp=datetime.now(UTC),
        )

        # Mock responses
        self.mock_position_manager.get_position.return_value = self.legacy_position
        self.mock_position_manager.get_all_positions.return_value = [
            self.legacy_position
        ]
        self.mock_position_manager.calculate_total_pnl.return_value = (
            Decimal(100),
            Decimal(50),
        )
        self.mock_position_manager.get_position_summary.return_value = {
            "active_positions": 1,
            "total_pnl": 150.0,
        }

        # Create unified portfolio manager
        self.portfolio_manager = FunctionalPortfolioManager(
            position_manager=self.mock_position_manager,
            paper_account=self.mock_paper_account,
            enable_functional_features=True,
        )

    def test_legacy_api_compatibility_get_position(self):
        """Test legacy API compatibility for get_position."""
        position = self.portfolio_manager.get_position("BTC-USD")

        assert isinstance(position, LegacyPosition)
        assert position.symbol == "BTC-USD"
        self.mock_position_manager.get_position.assert_called_once_with("BTC-USD")

    def test_legacy_api_compatibility_get_all_positions(self):
        """Test legacy API compatibility for get_all_positions."""
        positions = self.portfolio_manager.get_all_positions()

        assert len(positions) == 1
        assert isinstance(positions[0], LegacyPosition)
        self.mock_position_manager.get_all_positions.assert_called_once()

    def test_legacy_api_compatibility_calculate_total_pnl(self):
        """Test legacy API compatibility for calculate_total_pnl."""
        realized, unrealized = self.portfolio_manager.calculate_total_pnl()

        assert realized == Decimal(100)
        assert unrealized == Decimal(50)
        self.mock_position_manager.calculate_total_pnl.assert_called_once()

    def test_legacy_api_compatibility_get_position_summary(self):
        """Test legacy API compatibility for get_position_summary."""
        summary = self.portfolio_manager.get_position_summary()

        assert isinstance(summary, dict)
        assert summary["active_positions"] == 1
        self.mock_position_manager.get_position_summary.assert_called_once()

    def test_functional_api_get_functional_position(self):
        """Test enhanced functional API for getting positions."""
        functional_position = self.portfolio_manager.get_functional_position("BTC-USD")

        assert isinstance(functional_position, FunctionalPosition)
        assert functional_position.symbol == "BTC-USD"

    def test_functional_api_get_functional_snapshot(self):
        """Test enhanced functional API for getting position snapshot."""
        snapshot = self.portfolio_manager.get_functional_snapshot()

        assert isinstance(snapshot, PositionSnapshot)
        assert len(snapshot.positions) >= 0

    def test_get_account_snapshot_functional(self):
        """Test getting account snapshot through functional API."""
        current_prices = {"BTC-USD": Decimal(51000)}

        result = self.portfolio_manager.get_account_snapshot(current_prices)

        assert result.is_success()
        account_snapshot = result.success()
        assert isinstance(account_snapshot, AccountSnapshot)

    def test_get_performance_analysis(self):
        """Test getting comprehensive performance analysis."""
        current_prices = {"BTC-USD": Decimal(51000)}

        result = self.portfolio_manager.get_performance_analysis(current_prices, days=7)

        assert result.is_success()
        # Performance analysis should be available through adapters

    def test_validate_consistency(self):
        """Test validating consistency between legacy and functional."""
        current_prices = {"BTC-USD": Decimal(51000)}

        result = self.portfolio_manager.validate_consistency(current_prices)

        assert result.is_success()
        validation = result.success()
        assert isinstance(validation, dict)
        assert "overall_consistent" in validation

    def test_generate_comprehensive_report(self):
        """Test generating comprehensive portfolio report."""
        current_prices = {"BTC-USD": Decimal(51000)}

        result = self.portfolio_manager.generate_comprehensive_report(
            current_prices, days=7
        )

        assert result.is_success()
        report = result.success()
        assert isinstance(report, dict)
        assert "legacy_data" in report
        assert "functional_data" in report
        assert "consistency_check" in report
        assert report["functional_features_enabled"] is True

    def test_migrate_to_functional(self):
        """Test migration to functional types."""
        current_prices = {"BTC-USD": Decimal(51000)}

        result = self.portfolio_manager.migrate_to_functional(
            validate_migration=True, current_prices=current_prices
        )

        assert result.is_success()
        migration_message = result.success()
        assert "Migration completed successfully" in migration_message

    def test_get_migration_status(self):
        """Test getting migration status information."""
        status = self.portfolio_manager.get_migration_status()

        assert isinstance(status, dict)
        assert status["functional_features_enabled"] is True
        assert status["position_manager_available"] is True
        assert status["legacy_api_compatible"] is True
        assert status["functional_api_available"] is True

    def test_functional_features_disabled(self):
        """Test behavior when functional features are disabled."""
        disabled_manager = FunctionalPortfolioManager(
            position_manager=self.mock_position_manager,
            enable_functional_features=False,
        )

        # Legacy API should still work
        position = disabled_manager.get_position("BTC-USD")
        assert isinstance(position, LegacyPosition)

        # Functional API should return None or warnings
        with pytest.warns(UserWarning):
            functional_position = disabled_manager.get_functional_position("BTC-USD")
            assert functional_position is None


class TestUtilityFunctions(FPTestBase):
    """Test utility functions for adapter classes."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

    def test_create_unified_portfolio_manager(self):
        """Test creating unified portfolio manager utility."""
        mock_position_manager = Mock(spec=PositionManager)
        mock_paper_account = Mock(spec=PaperTradingAccount)

        manager = create_unified_portfolio_manager(
            position_manager=mock_position_manager,
            paper_account=mock_paper_account,
            enable_functional=True,
        )

        assert isinstance(manager, FunctionalPortfolioManager)
        assert manager.position_manager is mock_position_manager
        assert manager.paper_account is mock_paper_account
        assert manager.enable_functional_features is True

    def test_migrate_existing_system(self):
        """Test migrating existing system utility."""
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = []
        mock_position_manager.calculate_total_pnl.return_value = (
            Decimal(0),
            Decimal(0),
        )

        current_prices = {"BTC-USD": Decimal(50000)}

        result = migrate_existing_system(
            position_manager=mock_position_manager,
            current_prices=current_prices,
            validate=True,
        )

        assert result.is_success()
        migrated_manager = result.success()
        assert isinstance(migrated_manager, FunctionalPortfolioManager)

    def test_get_feature_compatibility_report(self):
        """Test getting feature compatibility report."""
        report = get_feature_compatibility_report()

        assert isinstance(report, dict)
        assert "legacy_api_methods" in report
        assert "functional_api_methods" in report
        assert "enhanced_features" in report
        assert "migration_path" in report
        assert "backward_compatibility" in report

        # Check specific content
        assert "get_position" in report["legacy_api_methods"]
        assert "get_functional_position" in report["functional_api_methods"]
        assert "Immutable position types" in report["enhanced_features"]

    def test_migrate_legacy_positions_to_functional(self):
        """Test migrating legacy positions to functional utility."""
        legacy_positions = [
            LegacyPosition(
                symbol="BTC-USD",
                side="LONG",
                size=Decimal("0.1"),
                entry_price=Decimal(50000),
                unrealized_pnl=Decimal(50),
                realized_pnl=Decimal(0),
                timestamp=datetime.now(UTC),
            ),
            LegacyPosition(
                symbol="ETH-USD",
                side="SHORT",
                size=Decimal("1.0"),
                entry_price=Decimal(3000),
                unrealized_pnl=Decimal(-30),
                realized_pnl=Decimal(100),
                timestamp=datetime.now(UTC),
            ),
        ]

        functional_positions = migrate_legacy_positions_to_functional(legacy_positions)

        assert len(functional_positions) == 2
        assert all(isinstance(pos, FunctionalPosition) for pos in functional_positions)
        assert functional_positions[0].symbol == "BTC-USD"
        assert functional_positions[1].symbol == "ETH-USD"

    def test_validate_functional_migration(self):
        """Test validating functional migration utility."""
        mock_position_manager = Mock(spec=PositionManager)
        mock_position_manager.get_all_positions.return_value = []
        mock_position_manager.calculate_total_pnl.return_value = (
            Decimal(0),
            Decimal(0),
        )

        adapter = FunctionalPositionManagerAdapter(mock_position_manager)
        current_prices = {"BTC-USD": Decimal(50000)}

        is_valid = validate_functional_migration(adapter, current_prices)

        # Should be valid for empty portfolio manager
        assert isinstance(is_valid, bool)


class TestAdapterErrorHandling(FPTestBase):
    """Test error handling across all adapter classes."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

    def test_trading_type_adapter_error_handling(self):
        """Test error handling in TradingTypeAdapter."""
        adapter = TradingTypeAdapter()

        # Test with None input
        with pytest.raises(ValueError):
            adapter.adapt_market_data_to_functional(None)

        # Test with invalid data structure
        with pytest.raises(ValueError):
            adapter.adapt_position_to_functional("invalid")

    def test_order_execution_adapter_error_handling(self):
        """Test error handling in OrderExecutionAdapter."""
        adapter = OrderExecutionAdapter()

        # Test with invalid order object
        invalid_order = Mock()
        invalid_order.symbol = None

        with pytest.raises(ValueError):
            adapter.prepare_order_for_execution(invalid_order)

    def test_position_manager_adapter_error_handling(self):
        """Test error handling in FunctionalPositionManagerAdapter."""
        # Create adapter with None position manager (should handle gracefully)
        adapter = FunctionalPositionManagerAdapter(None)

        # Operations should fail gracefully
        with pytest.raises(AttributeError):
            adapter.get_functional_position("BTC-USD")

    def test_portfolio_manager_error_handling(self):
        """Test error handling in FunctionalPortfolioManager."""
        # Create manager without any components
        manager = FunctionalPortfolioManager(
            position_manager=None, paper_account=None, enable_functional_features=True
        )

        # Should return empty/default values rather than errors
        position = manager.get_position("BTC-USD")
        assert position.side == "FLAT"
        assert position.size == Decimal(0)

        positions = manager.get_all_positions()
        assert len(positions) == 0

        realized, unrealized = manager.calculate_total_pnl()
        assert realized == Decimal(0)
        assert unrealized == Decimal(0)


if __name__ == "__main__":
    pytest.main([__file__])
