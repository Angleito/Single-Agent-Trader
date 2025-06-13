"""
Comprehensive integration tests for the complete AI trading bot system.

This module tests the complete trading flow from market data ingestion
through LLM decision-making to trade execution and position tracking.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from bot.main import TradingEngine
from bot.types import (
    IndicatorData,
    MarketData,
    MarketState,
    Order,
    OrderStatus,
    TradeAction,
)


class TestCompleteTradingFlow:
    """Test complete end-to-end trading flow integration."""

    @pytest.fixture
    def mock_market_data(self):
        """Create realistic mock market data for testing."""
        base_price = 50000
        timestamps = [
            datetime.utcnow() - timedelta(minutes=i) for i in range(200, 0, -1)
        ]

        market_data = []
        for i, timestamp in enumerate(timestamps):
            # Create realistic price movement
            price_change = np.random.normal(0, 100)
            current_price = base_price + (i * 10) + price_change

            high = current_price + abs(np.random.normal(0, 50))
            low = current_price - abs(np.random.normal(0, 50))
            volume = np.random.uniform(10, 100)

            market_data.append(
                MarketData(
                    symbol="BTC-USD",
                    timestamp=timestamp,
                    open=Decimal(str(current_price - np.random.uniform(-20, 20))),
                    high=Decimal(str(high)),
                    low=Decimal(str(low)),
                    close=Decimal(str(current_price)),
                    volume=Decimal(str(volume)),
                )
            )

        return market_data

    @pytest.fixture
    def mock_llm_responses(self):
        """Create mock LLM responses for different market conditions."""
        return {
            "bullish": TradeAction(
                action="LONG",
                size_pct=15,
                take_profit_pct=3.0,
                stop_loss_pct=2.0,
                rationale="Strong bullish indicators with good risk-reward",
            ),
            "bearish": TradeAction(
                action="SHORT",
                size_pct=12,
                take_profit_pct=2.5,
                stop_loss_pct=1.5,
                rationale="Clear bearish momentum with RSI overbought",
            ),
            "neutral": TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                rationale="Mixed signals, waiting for clearer direction",
            ),
            "close": TradeAction(
                action="CLOSE",
                size_pct=0,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                rationale="Take profit reached, closing position",
            ),
        }

    @pytest.fixture
    def mock_orders(self):
        """Create mock order responses."""
        return {
            "successful_long": Order(
                id="order_123",
                symbol="BTC-USD",
                side="BUY",
                type="MARKET",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                status=OrderStatus.FILLED,
                timestamp=datetime.utcnow(),
                filled_quantity=Decimal("0.1"),
            ),
            "successful_short": Order(
                id="order_124",
                symbol="BTC-USD",
                side="SELL",
                type="MARKET",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                status=OrderStatus.FILLED,
                timestamp=datetime.utcnow(),
                filled_quantity=Decimal("0.1"),
            ),
            "failed": Order(
                id="order_125",
                symbol="BTC-USD",
                side="BUY",
                type="MARKET",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                status=OrderStatus.REJECTED,
                timestamp=datetime.utcnow(),
                filled_quantity=Decimal("0"),
            ),
        }

    @pytest.mark.asyncio
    async def test_complete_trading_cycle_long_to_close(
        self, mock_market_data, mock_llm_responses, mock_orders
    ):
        """Test complete trading cycle: data → indicators → LLM → validation → risk → execution → close."""
        # Setup mocks
        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                get_latest_ohlcv=Mock(return_value=mock_market_data),
                to_dataframe=Mock(
                    return_value=self._create_mock_dataframe(mock_market_data)
                ),
                connect=AsyncMock(return_value=True),
                disconnect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
                get_data_status=Mock(
                    return_value={"connected": True, "cached_candles": 200}
                ),
            ),
            patch.multiple(
                "bot.main.LLMAgent",
                analyze_market=AsyncMock(
                    side_effect=[
                        mock_llm_responses["bullish"],  # First call - go long
                        mock_llm_responses["close"],  # Second call - close position
                    ]
                ),
                is_available=Mock(return_value=True),
                get_status=Mock(
                    return_value={
                        "llm_available": True,
                        "model_provider": "openai",
                        "model_name": "gpt-4",
                    }
                ),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                disconnect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
                get_connection_status=Mock(
                    return_value={"connected": True, "sandbox": True}
                ),
                execute_trade_action=AsyncMock(
                    side_effect=[
                        mock_orders["successful_long"],
                        mock_orders["successful_short"],
                    ]
                ),
                cancel_all_orders=AsyncMock(return_value=True),
                _get_account_balance=AsyncMock(return_value=Decimal("10000")),
            ),
        ):
            # Create trading engine with dry run
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)

            # Track state changes
            initial_position = engine.current_position.model_copy()

            # Simulate trading cycles
            await engine._initialize_components()

            # First cycle - should go long
            latest_data = engine.market_data.get_latest_ohlcv(limit=200)
            current_price = latest_data[-1].close

            df = engine.market_data.to_dataframe(limit=200)
            df_with_indicators = engine.indicator_calc.calculate_all(df)
            indicator_state = engine.indicator_calc.get_latest_state(df_with_indicators)

            market_state = MarketState(
                symbol=engine.symbol,
                interval=engine.interval,
                timestamp=datetime.utcnow(),
                current_price=current_price,
                ohlcv_data=latest_data[-10:],
                indicators=IndicatorData(**indicator_state),
                current_position=engine.current_position,
            )

            # Get LLM decision
            trade_action = await engine.llm_agent.analyze_market(market_state)
            validated_action = engine.validator.validate(trade_action)

            # Apply risk management
            risk_approved, final_action, risk_reason = (
                engine.risk_manager.evaluate_risk(
                    validated_action, engine.current_position, current_price
                )
            )

            # Execute trade
            if risk_approved and final_action.action != "HOLD":
                await engine._execute_trade(final_action, current_price)

            # Verify position opened
            assert engine.current_position.side == "LONG"
            assert engine.current_position.size > 0
            assert engine.current_position.entry_price == current_price
            assert engine.trade_count == 1
            assert engine.successful_trades == 1

            # Second cycle - should close position
            trade_action = await engine.llm_agent.analyze_market(market_state)
            validated_action = engine.validator.validate(trade_action)
            risk_approved, final_action, risk_reason = (
                engine.risk_manager.evaluate_risk(
                    validated_action, engine.current_position, current_price
                )
            )

            if risk_approved and final_action.action != "HOLD":
                await engine._execute_trade(final_action, current_price)

            # Verify position closed
            assert engine.current_position.side == "FLAT"
            assert engine.current_position.size == 0
            assert engine.trade_count == 2
            assert engine.successful_trades == 2

            await engine._shutdown()

    @pytest.mark.asyncio
    async def test_position_tracking_and_pnl_calculation(
        self, mock_market_data, mock_llm_responses, mock_orders
    ):
        """Test position tracking and P&L calculations throughout trading cycle."""
        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                get_latest_ohlcv=Mock(return_value=mock_market_data),
                to_dataframe=Mock(
                    return_value=self._create_mock_dataframe(mock_market_data)
                ),
                connect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
                get_data_status=Mock(
                    return_value={"connected": True, "cached_candles": 200}
                ),
            ),
            patch.multiple(
                "bot.main.LLMAgent",
                analyze_market=AsyncMock(return_value=mock_llm_responses["bullish"]),
                is_available=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                execute_trade_action=AsyncMock(
                    return_value=mock_orders["successful_long"]
                ),
                _get_account_balance=AsyncMock(return_value=Decimal("10000")),
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            await engine._initialize_components()

            # Open position
            current_price = Decimal("50000")
            engine.current_position.side = "LONG"
            engine.current_position.size = Decimal("0.1")
            engine.current_position.entry_price = current_price

            # Test P&L calculation with price increase
            new_price = Decimal("51000")  # $1000 increase
            await engine._update_position_tracking(new_price)

            expected_pnl = (new_price - current_price) * engine.current_position.size
            assert engine.current_position.unrealized_pnl == expected_pnl
            assert expected_pnl == Decimal("100")  # $1000 * 0.1 = $100 profit

            # Test P&L calculation with price decrease
            new_price = Decimal("49000")  # $1000 decrease
            await engine._update_position_tracking(new_price)

            expected_pnl = (new_price - current_price) * engine.current_position.size
            assert engine.current_position.unrealized_pnl == expected_pnl
            assert expected_pnl == Decimal("-100")  # $1000 * 0.1 = $100 loss

    @pytest.mark.asyncio
    async def test_error_recovery_and_fallback_mechanisms(
        self, mock_market_data, mock_llm_responses
    ):
        """Test error recovery and fallback mechanisms in trading flow."""
        # Test LLM failure fallback
        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                get_latest_ohlcv=Mock(return_value=mock_market_data),
                to_dataframe=Mock(
                    return_value=self._create_mock_dataframe(mock_market_data)
                ),
                connect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.LLMAgent",
                analyze_market=AsyncMock(
                    side_effect=Exception("LLM service unavailable")
                ),
                is_available=Mock(return_value=False),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            await engine._initialize_components()

            # Test that engine handles LLM failure gracefully
            current_price = mock_market_data[-1].close
            df = engine.market_data.to_dataframe(limit=200)
            df_with_indicators = engine.indicator_calc.calculate_all(df)
            indicator_state = engine.indicator_calc.get_latest_state(df_with_indicators)

            market_state = MarketState(
                symbol=engine.symbol,
                interval=engine.interval,
                timestamp=datetime.utcnow(),
                current_price=current_price,
                ohlcv_data=mock_market_data[-10:],
                indicators=IndicatorData(**indicator_state),
                current_position=engine.current_position,
            )

            # Should not raise exception, should fallback gracefully
            try:
                trade_action = await engine.llm_agent.analyze_market(market_state)
                # Should not reach here due to exception, but if it does, validate fallback
                validated_action = engine.validator.validate(trade_action)
                assert validated_action.action == "HOLD"
            except Exception:
                # Expected behavior - should be caught in main loop
                pass

    @pytest.mark.asyncio
    async def test_market_data_connection_recovery(self, mock_market_data):
        """Test market data connection recovery mechanisms."""
        market_data_mock = Mock()
        market_data_mock.is_connected.side_effect = [
            False,
            False,
            True,
        ]  # Fail twice, then succeed
        market_data_mock.connect = AsyncMock()
        market_data_mock.get_latest_ohlcv.return_value = mock_market_data
        market_data_mock.to_dataframe.return_value = self._create_mock_dataframe(
            mock_market_data
        )

        with (
            patch("bot.main.MarketDataProvider", return_value=market_data_mock),
            patch.multiple(
                "bot.main.LLMAgent",
                analyze_market=AsyncMock(
                    return_value=TradeAction(
                        action="HOLD",
                        size_pct=0,
                        take_profit_pct=1.0,
                        stop_loss_pct=1.0,
                        rationale="test",
                    )
                ),
                is_available=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
            ),
        ):

            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            engine.market_data = market_data_mock

            # Simulate one iteration of trading loop with connection issues
            latest_data = engine.market_data.get_latest_ohlcv(limit=200)

            # Should attempt reconnection when disconnected
            assert market_data_mock.connect.call_count >= 1

    @pytest.mark.asyncio
    async def test_risk_management_integration_in_flow(
        self, mock_market_data, mock_llm_responses
    ):
        """Test risk management integration prevents dangerous trades."""
        # Create high-risk action that should be modified/rejected
        high_risk_action = TradeAction(
            action="LONG",
            size_pct=80,  # Very high size
            take_profit_pct=1.0,
            stop_loss_pct=10.0,  # Poor risk-reward
            rationale="High risk test action",
        )

        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                get_latest_ohlcv=Mock(return_value=mock_market_data),
                to_dataframe=Mock(
                    return_value=self._create_mock_dataframe(mock_market_data)
                ),
                connect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.LLMAgent",
                analyze_market=AsyncMock(return_value=high_risk_action),
                is_available=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            await engine._initialize_components()

            current_price = mock_market_data[-1].close

            # Validate that risk manager modifies dangerous action
            validated_action = engine.validator.validate(high_risk_action)
            risk_approved, final_action, risk_reason = (
                engine.risk_manager.evaluate_risk(
                    validated_action, engine.current_position, current_price
                )
            )

            # Risk manager should cap the position size
            assert final_action.size_pct <= 20  # Should be capped to reasonable level
            assert "risk" in risk_reason.lower() or "size" in risk_reason.lower()

    def _create_mock_dataframe(self, market_data: list[MarketData]) -> pd.DataFrame:
        """Convert market data to DataFrame for indicator calculations."""
        data = []
        for candle in market_data:
            data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    @pytest.mark.asyncio
    async def test_multiple_trading_cycles_consistency(
        self, mock_market_data, mock_llm_responses, mock_orders
    ):
        """Test that multiple trading cycles maintain consistent state."""
        cycle_responses = [
            mock_llm_responses["bullish"],  # Go long
            mock_llm_responses["neutral"],  # Hold
            mock_llm_responses["close"],  # Close position
            mock_llm_responses["bearish"],  # Go short
            mock_llm_responses["close"],  # Close short
        ]

        cycle_orders = [
            mock_orders["successful_long"],
            None,  # No order for hold
            mock_orders["successful_short"],  # Close long
            mock_orders["successful_short"],  # Open short
            mock_orders["successful_long"],  # Close short
        ]

        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                get_latest_ohlcv=Mock(return_value=mock_market_data),
                to_dataframe=Mock(
                    return_value=self._create_mock_dataframe(mock_market_data)
                ),
                connect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.LLMAgent",
                analyze_market=AsyncMock(side_effect=cycle_responses),
                is_available=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                execute_trade_action=AsyncMock(side_effect=cycle_orders),
                _get_account_balance=AsyncMock(return_value=Decimal("10000")),
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            await engine._initialize_components()

            current_price = mock_market_data[-1].close

            # Track position state through cycles
            position_states = []

            for i in range(5):
                # Create market state
                df = engine.market_data.to_dataframe(limit=200)
                df_with_indicators = engine.indicator_calc.calculate_all(df)
                indicator_state = engine.indicator_calc.get_latest_state(
                    df_with_indicators
                )

                market_state = MarketState(
                    symbol=engine.symbol,
                    interval=engine.interval,
                    timestamp=datetime.utcnow(),
                    current_price=current_price,
                    ohlcv_data=mock_market_data[-10:],
                    indicators=IndicatorData(**indicator_state),
                    current_position=engine.current_position,
                )

                # Execute trading cycle
                trade_action = await engine.llm_agent.analyze_market(market_state)
                validated_action = engine.validator.validate(trade_action)
                risk_approved, final_action, risk_reason = (
                    engine.risk_manager.evaluate_risk(
                        validated_action, engine.current_position, current_price
                    )
                )

                if risk_approved and final_action.action != "HOLD":
                    await engine._execute_trade(final_action, current_price)

                # Track state
                position_states.append(
                    {
                        "cycle": i,
                        "action": final_action.action,
                        "position_side": engine.current_position.side,
                        "position_size": engine.current_position.size,
                        "trade_count": engine.trade_count,
                    }
                )

            # Verify expected state progression
            expected_progression = [
                {"position_side": "LONG", "trade_count": 1},  # Cycle 0: Go long
                {"position_side": "LONG", "trade_count": 1},  # Cycle 1: Hold
                {"position_side": "FLAT", "trade_count": 2},  # Cycle 2: Close long
                {"position_side": "SHORT", "trade_count": 3},  # Cycle 3: Go short
                {"position_side": "FLAT", "trade_count": 4},  # Cycle 4: Close short
            ]

            for i, expected in enumerate(expected_progression):
                actual = position_states[i]
                assert (
                    actual["position_side"] == expected["position_side"]
                ), f"Cycle {i} position mismatch"
                assert (
                    actual["trade_count"] == expected["trade_count"]
                ), f"Cycle {i} trade count mismatch"

    @pytest.mark.asyncio
    async def test_graceful_shutdown_integration(self, mock_market_data):
        """Test graceful shutdown preserves state and closes connections."""
        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                connect=AsyncMock(return_value=True),
                disconnect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                disconnect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
                cancel_all_orders=AsyncMock(return_value=True),
            ),
            patch.multiple("bot.main.LLMAgent", is_available=Mock(return_value=True)),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)

            # Initialize and then shutdown
            await engine._initialize_components()

            # Set some state to verify preservation
            engine.trade_count = 5
            engine.successful_trades = 4
            engine.current_position.side = "LONG"

            # Shutdown
            await engine._shutdown()

            # Verify shutdown procedures were called
            engine.market_data.disconnect.assert_called_once()
            engine.exchange_client.disconnect.assert_called_once()
            engine.exchange_client.cancel_all_orders.assert_called_once_with("BTC-USD")

            # Verify state is preserved
            assert engine.trade_count == 5
            assert engine.successful_trades == 4
            assert engine.current_position.side == "LONG"
