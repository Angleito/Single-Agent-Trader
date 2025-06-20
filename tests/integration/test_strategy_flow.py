"""Integration tests for strategy decision flow."""

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from bot.indicators.vumanchu import VuManChuIndicators
from bot.risk import RiskManager
from bot.strategy.core import CoreStrategy
from bot.trading_types import IndicatorData, MarketState, Position
from bot.validator import TradeValidator


class TestStrategyFlow:
    """Integration tests for the complete strategy decision flow."""

    @pytest.fixture()
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        return pd.DataFrame(
            {
                "open": np.random.uniform(45000, 55000, 100),
                "high": np.random.uniform(46000, 56000, 100),
                "low": np.random.uniform(44000, 54000, 100),
                "close": np.random.uniform(45000, 55000, 100),
                "volume": np.random.uniform(10, 100, 100),
            },
            index=dates,
        )

    @pytest.fixture()
    def strategy_components(self):
        """Create strategy components for testing."""
        return {
            "indicator_calc": VuManChuIndicators(),
            "core_strategy": CoreStrategy(),
            "validator": TradeValidator(),
            "risk_manager": RiskManager(),
        }

    def test_complete_decision_flow(self, sample_market_data, strategy_components):
        """Test the complete decision-making flow."""
        # Calculate indicators
        data_with_indicators = strategy_components["indicator_calc"].calculate_all(
            sample_market_data
        )

        # Create market state
        latest_data = data_with_indicators.iloc[-1]
        market_state = MarketState(
            symbol="BTC-USD",
            interval="1h",
            timestamp=datetime.now(UTC),
            current_price=Decimal(str(latest_data["close"])),
            ohlcv_data=[],
            indicators=IndicatorData(
                timestamp=datetime.now(UTC),
                cipher_a_dot=latest_data.get("trend_dot"),
                cipher_b_wave=latest_data.get("wave"),
                cipher_b_money_flow=latest_data.get("money_flow"),
                rsi=latest_data.get("rsi"),
                ema_fast=latest_data.get("ema_fast"),
                ema_slow=latest_data.get("ema_slow"),
            ),
            current_position=Position(
                symbol="BTC-USD",
                side="FLAT",
                size=Decimal(0),
                timestamp=datetime.now(UTC),
            ),
        )

        # Get strategy decision
        trade_action = strategy_components["core_strategy"].analyze_market(market_state)

        # Validate decision
        validated_action = strategy_components["validator"].validate(trade_action)

        # Apply risk management
        approved, final_action, reason = strategy_components[
            "risk_manager"
        ].evaluate_risk(
            validated_action, market_state.current_position, market_state.current_price
        )

        # Verify flow completed without errors
        assert trade_action is not None
        assert validated_action is not None
        assert final_action is not None
        assert reason is not None
        assert isinstance(approved, bool)

    def test_indicator_calculation_flow(self, sample_market_data):
        """Test indicator calculation produces expected results."""
        calc = VuManChuIndicators()
        result = calc.calculate_all(sample_market_data)

        # Check all expected indicators are calculated
        expected_indicators = [
            "ema_fast",
            "ema_slow",
            "rsi",
            "trend_dot",
            "vwap",
            "money_flow",
            "wave",
            "ema_200",
            "atr",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns
            # Check that we have some non-null values
            assert result[indicator].notna().any()

    def test_risk_management_integration(self, strategy_components):
        """Test risk management integration with strategy decisions."""
        risk_manager = strategy_components["risk_manager"]

        # Test with a high-risk trade action
        from bot.trading_types import TradeAction

        high_risk_action = TradeAction(
            action="LONG",
            size_pct=50,  # Very high size
            take_profit_pct=1.0,
            stop_loss_pct=5.0,  # Poor risk-reward
            rationale="High risk test",
        )

        current_position = Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal(0),
            timestamp=datetime.now(UTC),
        )

        approved, final_action, reason = risk_manager.evaluate_risk(
            high_risk_action, current_position, Decimal(50000)
        )

        # Risk manager should modify or reject the action
        assert final_action.size_pct <= 20  # Should be capped

    def test_market_data_to_decision_flow(self, sample_market_data):
        """Test complete flow from market data to trading decision."""
        # Initialize components
        indicator_calc = VuManChuIndicators()
        core_strategy = CoreStrategy()
        validator = TradeValidator()

        # Calculate indicators
        data_with_indicators = indicator_calc.calculate_all(sample_market_data)

        # Get latest state
        latest_state = indicator_calc.get_latest_state(data_with_indicators)

        # Create simplified market state
        market_state = MarketState(
            symbol="BTC-USD",
            interval="1h",
            timestamp=datetime.now(UTC),
            current_price=Decimal(50000),
            ohlcv_data=[],
            indicators=IndicatorData(
                timestamp=datetime.now(UTC),
                **{
                    k: v
                    for k, v in latest_state.items()
                    if k
                    in [
                        "cipher_a_dot",
                        "cipher_b_wave",
                        "cipher_b_money_flow",
                        "rsi",
                        "ema_fast",
                        "ema_slow",
                    ]
                },
            ),
            current_position=Position(
                symbol="BTC-USD",
                side="FLAT",
                size=Decimal(0),
                timestamp=datetime.now(UTC),
            ),
        )

        # Generate decision
        decision = core_strategy.analyze_market(market_state)
        validated_decision = validator.validate(decision)

        # Verify decision is valid
        assert validated_decision.action in ["LONG", "SHORT", "CLOSE", "HOLD"]
        assert 0 <= validated_decision.size_pct <= 100
        assert validated_decision.take_profit_pct > 0
        assert validated_decision.stop_loss_pct > 0

    def test_error_handling_in_flow(self, strategy_components):
        """Test error handling throughout the decision flow."""
        # Test with invalid/empty market state
        empty_market_state = MarketState(
            symbol="BTC-USD",
            interval="1h",
            timestamp=datetime.now(UTC),
            current_price=Decimal(50000),
            ohlcv_data=[],
            indicators=IndicatorData(timestamp=datetime.now(UTC)),
            current_position=Position(
                symbol="BTC-USD",
                side="FLAT",
                size=Decimal(0),
                timestamp=datetime.now(UTC),
            ),
        )

        # Strategy should handle empty indicators gracefully
        decision = strategy_components["core_strategy"].analyze_market(
            empty_market_state
        )

        # Should return a valid decision (likely HOLD)
        assert decision is not None
        assert decision.action in ["LONG", "SHORT", "CLOSE", "HOLD"]

    def test_validation_edge_cases(self, strategy_components):
        """Test validator handles edge cases properly."""
        validator = strategy_components["validator"]

        # Test various invalid inputs
        test_cases = [
            "",  # Empty string
            "not json",  # Invalid JSON
            {"invalid": "schema"},  # Wrong schema
            {
                "action": "INVALID",
                "size_pct": 10,
                "take_profit_pct": 2,
                "stop_loss_pct": 1,
                "rationale": "test",
            },  # Invalid action
        ]

        for test_case in test_cases:
            result = validator.validate(test_case)

            # All should default to HOLD
            assert result.action == "HOLD"
            assert result.size_pct == 0
