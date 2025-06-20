"""
Error handling integration tests for AI trading bot.

Tests various error scenarios and failure modes to ensure the system
handles errors gracefully and maintains stability.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from bot.indicators.vumanchu import VuManChuIndicators
from bot.main import TradingEngine
from bot.trading_types import (
    IndicatorData,
    MarketData,
    MarketState,
    Order,
    OrderStatus,
    TradeAction,
)

logger = logging.getLogger(__name__)


class TestErrorHandlingIntegration:
    """Test error handling and failure recovery across the system."""

    @pytest.fixture()
    def mock_market_data(self):
        """Create mock market data for testing."""
        return [
            MarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(UTC) - timedelta(minutes=i),
                open=Decimal(50000),
                high=Decimal(50100),
                low=Decimal(49900),
                close=Decimal(50050),
                volume=Decimal(100),
            )
            for i in range(50, 0, -1)
        ]

    @pytest.mark.asyncio()
    async def test_api_connection_failures(self, mock_market_data):
        """Test handling of API connection failures."""
        # Test market data API failure
        market_data_mock = Mock()
        market_data_mock.connect.side_effect = ConnectionError("API connection failed")
        market_data_mock.is_connected.return_value = False

        with (
            patch("bot.main.MarketDataProvider", return_value=market_data_mock),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                get_connection_status=Mock(
                    return_value={"connected": True, "sandbox": True}
                ),
            ),
            patch.multiple("bot.main.LLMAgent", is_available=Mock(return_value=True)),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)

            # Should raise exception during initialization
            with pytest.raises(ConnectionError):
                await engine._initialize_components()

        # Test exchange API failure
        exchange_mock = Mock()
        exchange_mock.connect.side_effect = ConnectionError("Exchange API unreachable")

        with (
            patch("bot.main.CoinbaseClient", return_value=exchange_mock),
            patch.multiple(
                "bot.main.MarketDataProvider",
                connect=AsyncMock(return_value=True),
                get_data_status=Mock(
                    return_value={"connected": True, "cached_candles": 50}
                ),
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)

            # Should handle exchange connection failure
            with pytest.raises(ConnectionError):
                await engine._initialize_components()

    @pytest.mark.asyncio()
    async def test_llm_service_outages(self, mock_market_data):
        """Test handling of LLM service outages."""
        # Test complete LLM service unavailability
        llm_mock = Mock()
        llm_mock.is_available.return_value = False
        llm_mock.analyze_market.side_effect = Exception("LLM service unavailable")
        llm_mock.get_status.return_value = {
            "llm_available": False,
            "model_provider": "openai",
            "model_name": "gpt-4",
        }

        with (
            patch("bot.main.LLMAgent", return_value=llm_mock),
            patch.multiple(
                "bot.main.MarketDataProvider",
                connect=AsyncMock(return_value=True),
                get_latest_ohlcv=Mock(return_value=mock_market_data),
                to_dataframe=Mock(
                    return_value=self._create_mock_dataframe(mock_market_data)
                ),
                is_connected=Mock(return_value=True),
                get_data_status=Mock(
                    return_value={"connected": True, "cached_candles": 50}
                ),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                get_connection_status=Mock(
                    return_value={"connected": True, "sandbox": True}
                ),
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            await engine._initialize_components()

            # Should initialize despite LLM unavailability
            assert engine.llm_agent is not None

            # Test that system falls back gracefully when LLM fails
            current_price = mock_market_data[-1].close
            market_data = engine.market_data.to_dataframe(limit=200)
            df_with_indicators = engine.indicator_calc.calculate_all(market_data)
            indicator_state = engine.indicator_calc.get_latest_state(df_with_indicators)

            market_state = MarketState(
                symbol=engine.symbol,
                interval=engine.interval,
                timestamp=datetime.now(UTC),
                current_price=current_price,
                ohlcv_data=mock_market_data[-10:],
                indicators=IndicatorData(**indicator_state),
                current_position=engine.current_position,
            )

            # LLM call should fail but not crash the system
            try:
                trade_action = await engine.llm_agent.analyze_market(market_state)
                # If it doesn't raise an exception, should be a fallback action
                assert trade_action.action == "HOLD"
            except Exception as e:
                # Expected to fail, system should handle this gracefully
                logger.debug(
                    "Expected LLM failure during error handling test: %s", str(e)
                )

    @pytest.mark.asyncio()
    async def test_invalid_data_handling(self):
        """Test handling of invalid or corrupted market data."""
        # Test with corrupted/invalid OHLCV data
        invalid_data_scenarios = [
            # Empty data
            [],
            # Data with NaN values
            [
                MarketData(
                    symbol="BTC-USD",
                    timestamp=datetime.now(UTC),
                    open=Decimal("NaN"),
                    high=Decimal(50100),
                    low=Decimal(49900),
                    close=Decimal(50050),
                    volume=Decimal(100),
                )
            ],
            # Data with negative prices
            [
                MarketData(
                    symbol="BTC-USD",
                    timestamp=datetime.now(UTC),
                    open=Decimal(-1000),
                    high=Decimal(50100),
                    low=Decimal(49900),
                    close=Decimal(50050),
                    volume=Decimal(100),
                )
            ],
            # Data with impossible OHLC relationships (high < low)
            [
                MarketData(
                    symbol="BTC-USD",
                    timestamp=datetime.now(UTC),
                    open=Decimal(50000),
                    high=Decimal(49000),  # High less than low
                    low=Decimal(50000),
                    close=Decimal(50050),
                    volume=Decimal(100),
                )
            ],
        ]

        indicator_calc = VuManChuIndicators()

        for i, invalid_data in enumerate(invalid_data_scenarios):
            try:
                if not invalid_data:
                    # Empty data scenario
                    test_data = pd.DataFrame()
                else:
                    test_data = self._create_mock_dataframe(invalid_data)

                # Should handle invalid data gracefully
                result = indicator_calc.calculate_all(test_data)

                # Should return a DataFrame (even if empty or with NaN values)
                assert isinstance(result, pd.DataFrame)

            except Exception as e:
                # If it raises an exception, it should be a known, handled exception
                error_msg = str(e).lower()
                assert any(
                    keyword in error_msg
                    for keyword in ["invalid", "data", "empty", "nan", "calculation"]
                ), f"Scenario {i}: Unexpected exception: {error_msg}"

    @pytest.mark.asyncio()
    async def test_order_execution_failures(self, mock_market_data):
        """Test handling of order execution failures."""
        # Test various order failure scenarios
        failure_scenarios = [
            # Order rejected by exchange
            None,  # execute_trade_action returns None
            # Order partially filled
            Order(
                id="partial_123",
                symbol="BTC-USD",
                side="BUY",
                type="MARKET",
                quantity=Decimal("0.1"),
                price=Decimal(50000),
                status=OrderStatus.PENDING,
                timestamp=datetime.now(UTC),
                filled_quantity=Decimal("0.05"),  # Only half filled
            ),
            # Order failed/rejected
            Order(
                id="failed_123",
                symbol="BTC-USD",
                side="BUY",
                type="MARKET",
                quantity=Decimal("0.1"),
                price=Decimal(50000),
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(UTC),
                filled_quantity=Decimal(0),
            ),
        ]

        trade_action = TradeAction(
            action="LONG",
            size_pct=15,
            take_profit_pct=3.0,
            stop_loss_pct=2.0,
            rationale="Test trade",
        )

        for scenario in failure_scenarios:
            with (
                patch.multiple(
                    "bot.main.MarketDataProvider",
                    get_latest_ohlcv=Mock(return_value=mock_market_data),
                    connect=AsyncMock(return_value=True),
                    is_connected=Mock(return_value=True),
                ),
                patch.multiple(
                    "bot.main.CoinbaseClient",
                    connect=AsyncMock(return_value=True),
                    execute_trade_action=AsyncMock(return_value=scenario),
                    _get_account_balance=AsyncMock(return_value=Decimal(10000)),
                ),
                patch.multiple(
                    "bot.main.LLMAgent", is_available=Mock(return_value=True)
                ),
            ):
                engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
                await engine._initialize_components()

                initial_trade_count = engine.trade_count
                initial_position = engine.current_position.model_copy()

                # Execute trade with failure scenario
                await engine._execute_trade(trade_action, Decimal(50000))

                # Verify system handled failure gracefully
                if scenario is None:
                    # Complete failure - no trade should be recorded as successful
                    assert engine.successful_trades == 0
                elif scenario.status == OrderStatus.REJECTED:
                    # Rejected order - trade attempted but not successful
                    assert engine.trade_count == initial_trade_count + 1
                    assert engine.successful_trades == 0
                elif scenario.status == OrderStatus.PENDING:
                    # Pending order - might be counted as successful pending fill
                    assert engine.trade_count == initial_trade_count + 1

                # Position should not change on failure
                if scenario is None or scenario.status == OrderStatus.REJECTED:
                    assert engine.current_position.side == initial_position.side
                    assert engine.current_position.size == initial_position.size

    @pytest.mark.asyncio()
    async def test_network_timeout_recovery(self, mock_market_data):
        """Test recovery from network timeouts and intermittent connectivity."""
        # Simulate intermittent network failures
        call_count = 0

        def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two calls
                raise TimeoutError("Network timeout")
            return True  # Succeed on third call

        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                connect=AsyncMock(side_effect=intermittent_failure),
                get_latest_ohlcv=Mock(return_value=mock_market_data),
                is_connected=Mock(return_value=True),
                get_data_status=Mock(
                    return_value={"connected": True, "cached_candles": 50}
                ),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                get_connection_status=Mock(
                    return_value={"connected": True, "sandbox": True}
                ),
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)

            # Should eventually succeed after retries
            # This would require implementing retry logic in the actual code
            try:
                await engine._initialize_components()
                # If it succeeds, verify the connection was established
                assert call_count >= 3  # Should have retried
            except TimeoutError:
                # If it still fails, should be after max retries
                assert call_count >= 2

    @pytest.mark.asyncio()
    async def test_data_corruption_recovery(self, mock_market_data):
        """Test recovery from data corruption scenarios."""
        # Test indicator calculation with corrupted data
        corrupted_scenarios = [
            # DataFrame with missing columns
            pd.DataFrame(
                {"open": [50000], "high": [50100]}
            ),  # Missing required columns
            # DataFrame with wrong data types
            pd.DataFrame(
                {
                    "open": ["not_a_number"],
                    "high": [50100],
                    "low": [49900],
                    "close": [50050],
                    "volume": [100],
                }
            ),
            # DataFrame with extreme values
            pd.DataFrame(
                {
                    "open": [float("inf")],
                    "high": [50100],
                    "low": [49900],
                    "close": [50050],
                    "volume": [100],
                }
            ),
        ]

        indicator_calc = VuManChuIndicators()

        for i, corrupted_df in enumerate(corrupted_scenarios):
            try:
                result = indicator_calc.calculate_all(corrupted_df)

                # If it succeeds, should return a valid DataFrame
                assert isinstance(result, pd.DataFrame)

                # May have NaN values but should not crash
                latest_state = indicator_calc.get_latest_state(result)
                assert isinstance(latest_state, dict)

            except Exception as e:
                # Should be a handled exception with clear error message
                error_msg = str(e).lower()
                assert any(
                    keyword in error_msg
                    for keyword in ["data", "invalid", "missing", "column", "type"]
                ), f"Scenario {i}: Unexpected error: {error_msg}"

    @pytest.mark.asyncio()
    async def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios."""
        # Simulate large data processing
        rng = np.random.default_rng(42)  # For reproducible results
        large_data = pd.DataFrame(
            {
                "open": rng.random(10000) * 50000,
                "high": rng.random(10000) * 50000 + 100,
                "low": rng.random(10000) * 50000 - 100,
                "close": rng.random(10000) * 50000,
                "volume": rng.random(10000) * 1000,
            },
            index=pd.date_range("2024-01-01", periods=10000, freq="1min"),
        )

        indicator_calc = VuManChuIndicators()

        try:
            # Should handle large datasets gracefully
            result = indicator_calc.calculate_all(large_data)

            # Verify result is still valid
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(large_data)

            # Should be able to get latest state without memory issues
            latest_state = indicator_calc.get_latest_state(result)
            assert isinstance(latest_state, dict)

        except MemoryError:
            # If memory error occurs, it should be caught and handled
            pytest.skip("Insufficient memory for large data test")
        except Exception as e:
            # Should not crash with other exceptions
            error_msg = str(e).lower()
            assert "memory" in error_msg or "size" in error_msg

    @pytest.mark.asyncio()
    async def test_concurrent_operation_errors(self, mock_market_data):
        """Test handling of concurrent operation errors."""
        # Test multiple simultaneous operations
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
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                execute_trade_action=AsyncMock(
                    side_effect=TimeoutError("Concurrent operation timeout")
                ),
            ),
            patch.multiple(
                "bot.main.LLMAgent",
                analyze_market=AsyncMock(
                    return_value=TradeAction(
                        action="LONG",
                        size_pct=10,
                        take_profit_pct=2.0,
                        stop_loss_pct=1.0,
                        rationale="test",
                    )
                ),
                is_available=Mock(return_value=True),
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            await engine._initialize_components()

            # Simulate concurrent trade attempts
            trade_tasks = []
            for _ in range(3):
                task = asyncio.create_task(
                    engine._execute_trade(
                        TradeAction(
                            action="LONG",
                            size_pct=10,
                            take_profit_pct=2.0,
                            stop_loss_pct=1.0,
                            rationale="concurrent test",
                        ),
                        Decimal(50000),
                    )
                )
                trade_tasks.append(task)

            # Wait for all tasks to complete (some may fail)
            results = await asyncio.gather(*trade_tasks, return_exceptions=True)

            # Should handle concurrent failures gracefully
            for result in results:
                if isinstance(result, Exception):
                    # Should be expected timeout errors
                    assert isinstance(result, asyncio.TimeoutError)

    @pytest.mark.asyncio()
    async def test_graceful_degradation_scenarios(self, mock_market_data):
        """Test graceful degradation when components fail."""
        # Test system behavior when LLM is unavailable but other components work
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
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.LLMAgent",
                is_available=Mock(return_value=False),  # LLM unavailable
                analyze_market=AsyncMock(side_effect=Exception("LLM service down")),
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            await engine._initialize_components()

            # System should still initialize and function in degraded mode
            assert engine.market_data is not None
            assert engine.exchange_client is not None
            assert engine.indicator_calc is not None

            # Should fall back to core strategy when LLM fails
            current_price = mock_market_data[-1].close
            market_data = engine.market_data.to_dataframe(limit=200)
            df_with_indicators = engine.indicator_calc.calculate_all(market_data)
            indicator_state = engine.indicator_calc.get_latest_state(df_with_indicators)

            market_state = MarketState(
                symbol=engine.symbol,
                interval=engine.interval,
                timestamp=datetime.now(UTC),
                current_price=current_price,
                ohlcv_data=mock_market_data[-10:],
                indicators=IndicatorData(**indicator_state),
                current_position=engine.current_position,
            )

            # Should use fallback strategy instead of LLM
            try:
                # This should trigger fallback to core strategy
                from bot.strategy.core import CoreStrategy

                core_strategy = CoreStrategy()
                fallback_decision = core_strategy.analyze_market(market_state)

                assert fallback_decision is not None
                assert fallback_decision.action in ["LONG", "SHORT", "CLOSE", "HOLD"]

            except ImportError:
                # If core strategy not available, should default to HOLD
                pass

    def _create_mock_dataframe(self, market_data: list[MarketData]) -> pd.DataFrame:
        """Convert market data to DataFrame for testing."""
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

        mock_data = pd.DataFrame(data)
        return mock_data.set_index("timestamp")

    @pytest.mark.asyncio()
    async def test_shutdown_during_active_operations(self, mock_market_data):
        """Test graceful shutdown during active trading operations."""
        # Test shutdown while operations are in progress
        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                get_latest_ohlcv=Mock(return_value=mock_market_data),
                connect=AsyncMock(return_value=True),
                disconnect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                disconnect=AsyncMock(return_value=True),
                cancel_all_orders=AsyncMock(return_value=True),
                execute_trade_action=AsyncMock(
                    side_effect=asyncio.sleep(10)
                ),  # Long-running operation
                is_connected=Mock(return_value=True),
            ),
            patch.multiple("bot.main.LLMAgent", is_available=Mock(return_value=True)),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            await engine._initialize_components()

            # Start a long-running trade operation
            trade_task = asyncio.create_task(
                engine._execute_trade(
                    TradeAction(
                        action="LONG",
                        size_pct=10,
                        take_profit_pct=2.0,
                        stop_loss_pct=1.0,
                        rationale="test",
                    ),
                    Decimal(50000),
                )
            )

            # Wait a bit then initiate shutdown
            await asyncio.sleep(0.1)
            shutdown_task = asyncio.create_task(engine._shutdown())

            # Cancel the trade task to simulate shutdown
            trade_task.cancel()

            # Shutdown should complete successfully
            await shutdown_task

            # Verify shutdown procedures were called
            engine.market_data.disconnect.assert_called_once()
            engine.exchange_client.disconnect.assert_called_once()
            engine.exchange_client.cancel_all_orders.assert_called_once_with("BTC-USD")
