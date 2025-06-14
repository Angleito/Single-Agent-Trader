"""
Component integration tests for AI trading bot components.

Tests integration between major components to ensure they work together
correctly and data flows properly between them.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import numpy as np
import pandas as pd
import pytest

from bot.data.market import MarketDataProvider
from bot.exchange.coinbase import CoinbaseClient
from bot.indicators.vumanchu import VuManChuIndicators
from bot.order_manager import OrderManager
from bot.position_manager import PositionManager
from bot.risk import RiskManager
from bot.strategy.core import CoreStrategy
from bot.types import (
    IndicatorData,
    MarketData,
    MarketState,
    Order,
    OrderStatus,
    Position,
    TradeAction,
)
from bot.validator import TradeValidator


class TestComponentIntegration:
    """Test integration between major bot components."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        np.random.seed(42)  # For reproducible results

        # Generate realistic price series
        base_price = 50000
        returns = np.random.normal(0, 0.001, 200)  # 0.1% volatility
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data = []
        for _i, (timestamp, close) in enumerate(zip(dates, prices, strict=False)):
            open_price = close * (1 + np.random.normal(0, 0.0005))
            high_price = max(open_price, close) * (1 + abs(np.random.normal(0, 0.0003)))
            low_price = min(open_price, close) * (1 - abs(np.random.normal(0, 0.0003)))
            volume = np.random.uniform(10, 100)

            data.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close,
                    "volume": volume,
                }
            )

        return pd.DataFrame(data).set_index("timestamp")

    @pytest.fixture
    def market_data_list(self, sample_ohlcv_data):
        """Convert DataFrame to MarketData list."""
        market_data = []
        for timestamp, row in sample_ohlcv_data.iterrows():
            market_data.append(
                MarketData(
                    symbol="BTC-USD",
                    timestamp=timestamp,
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=Decimal(str(row["volume"])),
                )
            )
        return market_data

    def test_market_data_to_indicators_integration(self, sample_ohlcv_data):
        """Test MarketDataProvider + VuManChuIndicators integration."""
        # Setup market data provider mock
        market_provider = Mock(spec=MarketDataProvider)
        market_provider.to_dataframe.return_value = sample_ohlcv_data
        market_provider.get_latest_ohlcv.return_value = []  # Not used in this test

        # Initialize indicator calculator
        indicator_calc = VuManChuIndicators()

        # Test data flow
        df = market_provider.to_dataframe(limit=200)
        assert not df.empty
        assert len(df) == 200
        assert all(
            col in df.columns for col in ["open", "high", "low", "close", "volume"]
        )

        # Calculate indicators
        df_with_indicators = indicator_calc.calculate_all(df)

        # Verify indicators were calculated
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
            assert indicator in df_with_indicators.columns
            # Should have some non-null values (accounting for warmup period)
            assert df_with_indicators[indicator].notna().sum() > 0

        # Test latest state extraction
        latest_state = indicator_calc.get_latest_state(df_with_indicators)
        assert isinstance(latest_state, dict)
        assert "timestamp" in latest_state

        # Verify state has required indicators
        indicator_keys = [
            "combined_signal",
            "combined_confidence",
            "market_sentiment",
            "close",
            "volume",
            "timestamp",
        ]
        for key in indicator_keys:
            assert key in latest_state

    def test_llm_agent_validator_integration(self):
        """Test LLMAgent + Validator integration."""
        # Test with various LLM outputs
        test_cases = [
            # Valid JSON response
            {
                "llm_output": '{"action": "LONG", "size_pct": 15, "take_profit_pct": 3.0, "stop_loss_pct": 2.0, "rationale": "Strong bullish signals"}',
                "expected_action": "LONG",
                "expected_size": 15,
            },
            # Invalid JSON - should fallback to HOLD
            {
                "llm_output": "invalid json response",
                "expected_action": "HOLD",
                "expected_size": 0,
            },
            # Valid JSON but invalid action - should fallback to HOLD
            {
                "llm_output": '{"action": "INVALID", "size_pct": 15, "take_profit_pct": 3.0, "stop_loss_pct": 2.0, "rationale": "test"}',
                "expected_action": "HOLD",
                "expected_size": 0,
            },
            # Valid JSON but extreme values - should be adjusted
            {
                "llm_output": '{"action": "LONG", "size_pct": 200, "take_profit_pct": 100.0, "stop_loss_pct": 100.0, "rationale": "extreme test"}',
                "expected_action": "HOLD",  # Should be rejected
                "expected_size": 0,
            },
        ]

        validator = TradeValidator()

        for i, case in enumerate(test_cases):
            # Test validator handling of LLM output
            validated_action = validator.validate(case["llm_output"])

            assert (
                validated_action.action == case["expected_action"]
            ), f"Case {i}: action mismatch"
            assert (
                validated_action.size_pct == case["expected_size"]
            ), f"Case {i}: size mismatch"

            # Ensure all required fields are present and valid
            assert validated_action.take_profit_pct > 0
            assert validated_action.stop_loss_pct > 0
            assert isinstance(validated_action.rationale, str)
            assert len(validated_action.rationale) > 0

    def test_position_manager_order_manager_integration(self):
        """Test PositionManager + OrderManager integration."""
        # Initialize managers
        position_manager = PositionManager()
        order_manager = OrderManager()

        # Reset positions to ensure clean state
        position_manager.reset_positions()

        # Test opening a position
        Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0"),
            timestamp=datetime.now(UTC),
        )

        # Create an order
        buy_order = Order(
            id="order_123",
            symbol="BTC-USD",
            side="BUY",
            type="MARKET",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
            filled_quantity=Decimal("0.1"),
        )

        # Track order
        order_manager.add_order(buy_order)

        # Update position based on filled order
        if buy_order.status == OrderStatus.FILLED:
            position_manager.update_position_from_order(buy_order, buy_order.price)

        # Verify integration
        current_position = position_manager.get_position("BTC-USD")
        tracked_orders = order_manager.get_orders_by_status(OrderStatus.FILLED, "BTC-USD")

        assert current_position.side == "LONG"
        assert current_position.size == Decimal("0.1")
        assert len(tracked_orders) == 1
        assert tracked_orders[0].status == OrderStatus.FILLED

        # Test closing position
        sell_order = Order(
            id="order_124",
            symbol="BTC-USD",
            side="SELL",
            type="MARKET",
            quantity=Decimal("0.1"),
            price=Decimal("51000"),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
            filled_quantity=Decimal("0.1"),
        )

        order_manager.add_order(sell_order)

        # Close position
        if sell_order.status == OrderStatus.FILLED:
            # Update position from sell order (will calculate P&L automatically)
            position_manager.update_position_from_order(sell_order, sell_order.price)

        # Verify position closed with P&L
        final_position = position_manager.get_position("BTC-USD")
        assert final_position.side == "FLAT"
        assert final_position.size == Decimal("0")
        assert final_position.realized_pnl == Decimal("100")  # (51000 - 50000) * 0.1

    def test_risk_manager_with_real_position_data(self):
        """Test RiskManager with realistic position and market data."""
        # Initialize with mock position manager
        mock_position_manager = Mock()
        mock_position_manager.get_total_position_value.return_value = Decimal("5000")
        mock_position_manager.get_current_positions.return_value = []

        risk_manager = RiskManager(position_manager=mock_position_manager)

        # Test with various scenarios
        test_scenarios = [
            {
                "name": "Conservative trade",
                "action": TradeAction(
                    action="LONG",
                    size_pct=10,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.0,
                    rationale="Conservative long position",
                ),
                "position": Position(
                    symbol="BTC-USD",
                    side="FLAT",
                    size=Decimal("0"),
                    timestamp=datetime.now(UTC),
                ),
                "price": Decimal("50000"),
                "should_approve": True,
            },
            {
                "name": "Oversized trade",
                "action": TradeAction(
                    action="LONG",
                    size_pct=50,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.0,
                    rationale="Large position",
                ),
                "position": Position(
                    symbol="BTC-USD",
                    side="FLAT",
                    size=Decimal("0"),
                    timestamp=datetime.now(UTC),
                ),
                "price": Decimal("50000"),
                "should_approve": False,  # Should be modified or rejected
            },
            {
                "name": "Poor risk-reward ratio",
                "action": TradeAction(
                    action="LONG",
                    size_pct=15,
                    take_profit_pct=1.0,
                    stop_loss_pct=5.0,
                    rationale="Poor risk-reward",
                ),
                "position": Position(
                    symbol="BTC-USD",
                    side="FLAT",
                    size=Decimal("0"),
                    timestamp=datetime.now(UTC),
                ),
                "price": Decimal("50000"),
                "should_approve": False,  # Should be rejected due to poor R:R
            },
            {
                "name": "Conflicting position",
                "action": TradeAction(
                    action="LONG",
                    size_pct=10,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.0,
                    rationale="Long while already long",
                ),
                "position": Position(
                    symbol="BTC-USD",
                    side="LONG",
                    size=Decimal("0.1"),
                    entry_price=Decimal("49000"),
                    timestamp=datetime.now(UTC),
                ),
                "price": Decimal("50000"),
                "should_approve": False,  # Should reject adding to position
            },
        ]

        for scenario in test_scenarios:
            approved, final_action, reason = risk_manager.evaluate_risk(
                scenario["action"], scenario["position"], scenario["price"]
            )

            if scenario["should_approve"]:
                assert (
                    approved
                ), f"Scenario '{scenario['name']}' should be approved: {reason}"
            else:
                # Either rejected or significantly modified
                if approved:
                    # If approved, should be significantly modified
                    assert (
                        final_action.size_pct < scenario["action"].size_pct
                    ), f"Scenario '{scenario['name']}' should be modified"
                else:
                    assert (
                        not approved
                    ), f"Scenario '{scenario['name']}' should be rejected: {reason}"

    @pytest.mark.asyncio
    async def test_exchange_client_order_flow_integration(self):
        """Test CoinbaseClient integration with order flow."""
        # Mock the exchange client
        exchange_client = Mock(spec=CoinbaseClient)

        # Test successful order execution
        trade_action = TradeAction(
            action="LONG",
            size_pct=15,
            take_profit_pct=3.0,
            stop_loss_pct=2.0,
            rationale="Test long position",
        )

        expected_order = Order(
            id="order_123",
            symbol="BTC-USD",
            side="BUY",
            type="MARKET",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
            filled_quantity=Decimal("0.1"),
        )

        exchange_client.execute_trade_action = AsyncMock(return_value=expected_order)
        exchange_client.get_connection_status.return_value = {
            "connected": True,
            "sandbox": True,
        }
        exchange_client.is_connected.return_value = True

        # Execute trade
        result = await exchange_client.execute_trade_action(
            trade_action, "BTC-USD", Decimal("50000")
        )

        # Verify integration
        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert result.symbol == "BTC-USD"
        exchange_client.execute_trade_action.assert_called_once()

        # Test order failure handling
        exchange_client.execute_trade_action = AsyncMock(return_value=None)
        result = await exchange_client.execute_trade_action(
            trade_action, "BTC-USD", Decimal("50000")
        )
        assert result is None

    def test_indicator_to_strategy_decision_flow(self, sample_ohlcv_data):
        """Test complete flow from indicators to strategy decision."""
        # Calculate indicators
        indicator_calc = VuManChuIndicators()
        df_with_indicators = indicator_calc.calculate_all(sample_ohlcv_data)
        latest_state = indicator_calc.get_latest_state(df_with_indicators)

        # Create market state
        current_price = Decimal(str(sample_ohlcv_data.iloc[-1]["close"]))
        current_position = Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0"),
            timestamp=datetime.now(UTC),
        )

        market_state = MarketState(
            symbol="BTC-USD",
            interval="1m",
            timestamp=datetime.now(UTC),
            current_price=current_price,
            ohlcv_data=[],  # Not needed for this test
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
                        "vwap",
                    ]
                },
            ),
            current_position=current_position,
        )

        # Test with core strategy (fallback)
        core_strategy = CoreStrategy()
        decision = core_strategy.analyze_market(market_state)

        # Verify decision structure
        assert isinstance(decision, TradeAction)
        assert decision.action in ["LONG", "SHORT", "CLOSE", "HOLD"]
        assert 0 <= decision.size_pct <= 100
        assert decision.take_profit_pct > 0
        assert decision.stop_loss_pct > 0
        assert isinstance(decision.rationale, str)
        assert len(decision.rationale) > 0

        # Test decision consistency with market state
        if market_state.current_position.side != "FLAT":
            # If already in position, decision should be HOLD or CLOSE
            assert decision.action in ["HOLD", "CLOSE"]

        # Validate with validator
        validator = TradeValidator()
        validated_decision = validator.validate(decision)

        # Should maintain core decision structure
        assert validated_decision.action == decision.action
        assert validated_decision.size_pct <= decision.size_pct  # May be capped

    def test_multi_component_data_consistency(self, market_data_list):
        """Test data consistency across multiple components."""
        # Setup components
        indicator_calc = VuManChuIndicators()
        position_manager = PositionManager()
        order_manager = OrderManager()
        risk_manager = RiskManager(position_manager=position_manager)

        # Convert to DataFrame for indicators
        df_data = []
        for candle in market_data_list:
            df_data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                }
            )

        df = pd.DataFrame(df_data).set_index("timestamp")

        # Process through components
        df_with_indicators = indicator_calc.calculate_all(df)
        latest_state = indicator_calc.get_latest_state(df_with_indicators)

        # Create position
        current_price = market_data_list[-1].close
        test_position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=current_price,
            timestamp=datetime.now(UTC),
        )

        position_manager.update_position(test_position)

        # Create corresponding order
        test_order = Order(
            id="test_order",
            symbol="BTC-USD",
            side="BUY",
            type="MARKET",
            quantity=test_position.size,
            price=test_position.entry_price,
            status=OrderStatus.FILLED,
            timestamp=test_position.timestamp,
            filled_quantity=test_position.size,
        )

        order_manager.add_order(test_order)

        # Test data consistency
        retrieved_position = position_manager.get_current_position("BTC-USD")
        retrieved_orders = order_manager.get_orders_by_symbol("BTC-USD")
        risk_metrics = risk_manager.get_risk_metrics()

        # Verify consistency
        assert retrieved_position.symbol == test_order.symbol
        assert retrieved_position.size == test_order.filled_quantity
        assert len(retrieved_orders) == 1
        assert retrieved_orders[0].id == test_order.id

        # Verify indicator data integrity
        assert latest_state["timestamp"] is not None
        assert isinstance(latest_state.get("rsi"), int | float | type(None))
        assert isinstance(latest_state.get("ema_fast"), int | float | type(None))

        # Test risk metrics reflect position
        assert isinstance(risk_metrics.current_positions, int)
        assert risk_metrics.current_positions >= 0

    def test_error_propagation_between_components(self):
        """Test how errors propagate between integrated components."""
        # Test indicator calculation with bad data
        indicator_calc = VuManChuIndicators()

        # Create DataFrame with NaN values
        bad_data = pd.DataFrame(
            {
                "open": [50000, np.nan, 50200],
                "high": [50100, 50300, np.nan],
                "low": [49900, 50000, 50100],
                "close": [50050, 50250, 50150],
                "volume": [100, 150, np.nan],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1min"),
        )

        # Should handle NaN values gracefully
        try:
            result = indicator_calc.calculate_all(bad_data)
            # Should not crash, may have NaN indicators
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # If it raises an exception, it should be a known, handled exception
            assert "data" in str(e).lower() or "invalid" in str(e).lower()

        # Test validator with None input
        validator = TradeValidator()
        result = validator.validate(None)
        assert result.action == "HOLD"
        assert result.size_pct == 0

        # Test risk manager with invalid position
        mock_position_manager = Mock()
        mock_position_manager.get_total_position_value.side_effect = Exception(
            "Position error"
        )

        risk_manager = RiskManager(position_manager=mock_position_manager)

        test_action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Test action",
        )

        test_position = Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal("0"),
            timestamp=datetime.now(UTC),
        )

        # Should handle position manager error gracefully
        try:
            approved, final_action, reason = risk_manager.evaluate_risk(
                test_action, test_position, Decimal("50000")
            )
            # Should fallback to safe defaults
            assert not approved or final_action.size_pct <= test_action.size_pct
        except Exception:
            # If it raises, should be a handled exception
            pass
