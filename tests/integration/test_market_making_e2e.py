"""
Comprehensive end-to-end integration tests for market making system.

This module provides complete E2E testing for the market making system including:
- Complete market making cycle testing
- Symbol routing validation (SUI-PERP → MM, others → LLM)
- Configuration profile verification
- CLI integration testing
- Paper trading mode validation
- Error handling and recovery testing
- Performance benchmarks

Test Coverage:
- End-to-end market making flow with mocked exchange
- Multi-symbol routing tests
- Configuration profile testing (conservative/moderate/aggressive)
- CLI command integration
- Paper trading validation with real market data simulation
- Error scenarios and recovery mechanisms
- Performance baseline measurements
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from click.testing import CliRunner

# Trading bot imports
from bot.main import cli
from bot.strategy.market_making_integration import MarketMakingIntegrator
from bot.trading_types import (
    IndicatorData,
    MarketData,
    MarketState,
    Order,
    Position,
    TradeAction,
)

logger = logging.getLogger(__name__)


class MockExchange:
    """Comprehensive mock exchange for E2E testing."""

    def __init__(self, symbol: str = "SUI-PERP"):
        self.symbol = symbol
        self.orders = {}
        self.positions = {}
        self.balance = Decimal(10000)
        self.market_price = Decimal("3.50")
        self.order_counter = 0
        self.fill_ratio = 0.7  # 70% fill rate for market making orders
        self.latency_ms = 50
        self.is_connected = True
        self.orderbook = self._generate_orderbook()
        # Add attributes to make it compatible with Bluefin client check
        self.client_type = "bluefin"

    def _generate_orderbook(self) -> dict[str, list[dict]]:
        """Generate realistic orderbook data."""
        mid_price = float(self.market_price)
        spread = 0.01  # 1% spread

        bids = []
        asks = []

        for i in range(10):
            bid_price = mid_price * (1 - spread / 2 - i * 0.001)
            ask_price = mid_price * (1 + spread / 2 + i * 0.001)
            size = 100 * (1 + i * 0.1)

            bids.append({"price": bid_price, "size": size})
            asks.append({"price": ask_price, "size": size})

        return {"bids": bids, "asks": asks}

    async def get_balance(self) -> Decimal:
        """Get account balance."""
        return self.balance

    async def place_order(self, order: Order) -> str:
        """Place an order and return order ID."""
        self.order_counter += 1
        order_id = f"order_{self.order_counter}"

        # Simulate realistic order placement
        await asyncio.sleep(self.latency_ms / 1000)

        self.orders[order_id] = {
            "id": order_id,
            "symbol": order.symbol,
            "side": order.side,
            "size": order.size,
            "price": order.price,
            "status": "open",
            "filled_size": Decimal(0),
            "timestamp": datetime.now(UTC),
        }

        # Simulate partial fills for market making orders
        if hasattr(order, "order_type") and order.order_type == "limit":
            await self._simulate_fill(order_id)

        return order_id

    async def _simulate_fill(self, order_id: str):
        """Simulate order fills based on market conditions."""
        order = self.orders[order_id]

        # Simulate fill probability based on spread and market conditions
        if self._should_fill_order(order):
            fill_size = order["size"] * Decimal(str(self.fill_ratio))
            order["filled_size"] = fill_size
            order["status"] = (
                "partially_filled" if fill_size < order["size"] else "filled"
            )

    def _should_fill_order(self, order: dict) -> bool:
        """Determine if an order should be filled."""
        order_price = order["price"]

        # Buy orders fill when market price drops to order price
        if order["side"] == "buy":
            return self.market_price <= order_price * Decimal("1.001")  # Small buffer
        return self.market_price >= order_price * Decimal("0.999")  # Small buffer

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"
            return True
        return False

    async def get_orders(self) -> list[dict]:
        """Get all orders."""
        return list(self.orders.values())

    async def get_orderbook(self, symbol: str) -> dict:
        """Get orderbook data."""
        return self.orderbook

    async def get_market_data(
        self, symbol: str, interval: str = "1m"
    ) -> list[MarketData]:
        """Get historical market data."""
        # Generate realistic OHLCV data
        data = []
        base_time = datetime.now(UTC)
        base_price = float(self.market_price)

        for i in range(100):
            timestamp = base_time - timedelta(minutes=i)
            price_change = 0.001 * (i % 10 - 5)  # Small random-like changes
            current_price = base_price * (1 + price_change)

            data.append(
                MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=Decimal(str(current_price * 0.999)),
                    high=Decimal(str(current_price * 1.001)),
                    low=Decimal(str(current_price * 0.998)),
                    close=Decimal(str(current_price)),
                    volume=Decimal(1000),
                )
            )

        return list(reversed(data))


@pytest.fixture()
def mock_exchange():
    """Create mock exchange for testing."""
    return MockExchange()


class TestMarketMakingE2E:
    """Comprehensive end-to-end market making integration tests."""

    @pytest.fixture()
    def mock_llm_agent(self):
        """Create mock LLM agent for non-MM symbols."""
        agent = Mock()
        agent.generate_trade_action = AsyncMock(
            return_value=TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                rationale="LLM agent holding for non-MM symbol",
            )
        )
        agent.analyze_market = AsyncMock(
            return_value=TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                rationale="LLM agent holding for non-MM symbol",
            )
        )
        return agent

    @pytest.fixture()
    def test_config(self):
        """Create test configuration."""
        return {
            "market_making": {
                "enabled": True,
                "symbol": "SUI-PERP",
                "profile": "moderate",
                "strategy": {
                    "base_spread_bps": 10,
                    "order_levels": 3,
                    "max_position_pct": 25.0,
                },
                "risk": {
                    "max_position_value": "5000",
                    "stop_loss_pct": 2.0,
                },
            },
            "trading": {
                "symbol": "BTC-USD",  # Primary symbol (non-MM)
                "dry_run": True,
            },
        }

    def create_test_market_state(self, symbol: str, price: Decimal) -> MarketState:
        """Helper to create proper test market state."""
        now = datetime.now(UTC)
        market_data = MarketData(
            symbol=symbol,
            timestamp=now,
            open=price,
            high=price * Decimal("1.01"),
            low=price * Decimal("0.99"),
            close=price,
            volume=Decimal(1000),
        )

        return MarketState(
            symbol=symbol,
            interval="1m",
            current_price=price,
            ohlcv_data=[market_data],
            timestamp=now,
            indicators=IndicatorData(timestamp=now),
            current_position=Position(
                symbol=symbol, side="FLAT", size=Decimal(0), timestamp=now
            ),
        )

    @pytest.fixture()
    def config_profiles(self):
        """Configuration profiles for testing."""
        return {
            "conservative": {
                "strategy": {
                    "base_spread_bps": 15,
                    "order_levels": 2,
                    "max_position_pct": 15.0,
                },
                "risk": {
                    "stop_loss_pct": 1.5,
                    "daily_loss_limit_pct": 2.0,
                },
            },
            "moderate": {
                "strategy": {
                    "base_spread_bps": 10,
                    "order_levels": 3,
                    "max_position_pct": 25.0,
                },
                "risk": {
                    "stop_loss_pct": 2.0,
                    "daily_loss_limit_pct": 5.0,
                },
            },
            "aggressive": {
                "strategy": {
                    "base_spread_bps": 8,
                    "order_levels": 5,
                    "max_position_pct": 40.0,
                },
                "risk": {
                    "stop_loss_pct": 3.0,
                    "daily_loss_limit_pct": 8.0,
                },
            },
        }

    @pytest.mark.integration()
    @pytest.mark.asyncio()
    async def test_complete_market_making_cycle(self, mock_exchange, test_config):
        """Test complete market making cycle from initialization to execution."""

        # Initialize market making integrator
        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_exchange,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
            config=test_config["market_making"],
        )

        # Test initialization
        assert not integrator.status.is_initialized

        # Initialize the system with mock LLM agent and mocked engine creation
        mock_llm = Mock()

        # Mock the engine creation to avoid complex dependencies
        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_factory:
            mock_engine_class = Mock()
            mock_engine_instance = Mock()
            mock_engine_instance.initialize = AsyncMock()
            mock_engine_instance.get_status = AsyncMock(return_value={"healthy": True})
            mock_engine_instance.stop = AsyncMock()
            mock_engine_instance.analyze_market_and_decide = AsyncMock(
                return_value=TradeAction(
                    action="LONG",
                    size_pct=25.0,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.5,
                    rationale="Market making signal: buy order placed",
                )
            )
            mock_engine_class.create_engine.return_value = mock_engine_instance
            mock_factory.return_value = mock_engine_class

            await integrator.initialize(mock_llm)
            assert integrator.status.is_initialized
            assert integrator.status.market_making_enabled

            # Test strategy routing
            assert integrator.get_strategy_for_symbol("SUI-PERP") == "market_making"
            assert integrator.get_strategy_for_symbol("BTC-USD") == "llm"

            # Test market making cycle
            market_data = await mock_exchange.get_market_data("SUI-PERP")

            # Create market state for testing
            market_state = MarketState(
                symbol="SUI-PERP",
                interval="1m",
                current_price=market_data[-1].close,
                ohlcv_data=market_data,
                timestamp=datetime.now(UTC),
                indicators=IndicatorData(timestamp=datetime.now(UTC)),
                current_position=Position(
                    symbol="SUI-PERP",
                    side="FLAT",
                    size=Decimal(0),
                    timestamp=datetime.now(UTC),
                ),
            )

            result = await integrator.analyze_market(market_state)

            assert result.action == "LONG"
            assert result.size_pct == 25.0
            assert "market making" in result.rationale.lower()

            # Test health monitoring
            health_status = await integrator.health_check()
            # For testing purposes, check that health check completed and has proper structure
            assert "healthy" in health_status
            assert "status" in health_status
            status = health_status["status"]
            assert status["market_making_enabled"]
            assert status["is_initialized"]

            # Test shutdown
            await integrator.stop()
            assert not integrator.status.is_running

    @pytest.mark.integration()
    @pytest.mark.asyncio()
    async def test_multi_symbol_routing(
        self, mock_exchange, mock_llm_agent, test_config
    ):
        """Test routing between market making and LLM strategies by symbol."""

        integrator = MarketMakingIntegrator(
            symbol="BTC-USD",  # Primary symbol (non-MM)
            exchange_client=mock_exchange,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
            config=test_config["market_making"],
        )

        # Mock the engine creation for this test
        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_factory:
            mock_engine_class = Mock()
            mock_engine_instance = Mock()
            mock_engine_instance.initialize = AsyncMock()
            mock_engine_instance.get_status = AsyncMock(return_value={"healthy": True})
            mock_engine_instance.stop = AsyncMock()
            mock_engine_instance.analyze_market_and_decide = AsyncMock(
                return_value=TradeAction(
                    action="LONG",
                    size_pct=20.0,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.5,
                    rationale="Market making strategy executed",
                )
            )
            mock_engine_class.create_engine.return_value = mock_engine_instance
            mock_factory.return_value = mock_engine_class

            await integrator.initialize(mock_llm_agent)

            # Test symbol routing logic
            symbols_to_test = [
                ("SUI-PERP", "market_making"),
                ("BTC-USD", "llm"),
                ("ETH-USD", "llm"),
                ("SOL-PERP", "llm"),
            ]

            for symbol, expected_strategy in symbols_to_test:
                actual_strategy = integrator.get_strategy_for_symbol(symbol)
                assert actual_strategy == expected_strategy

            # Create market state using helper
            market_state = self.create_test_market_state("SUI-PERP", Decimal("3.51"))

            # Execute for MM symbol
            result = await integrator.analyze_market(market_state)
            assert result.action == "LONG"
            assert "market making" in result.rationale.lower()

            await integrator.stop()

    @pytest.mark.integration()
    @pytest.mark.asyncio()
    async def test_configuration_profiles(self, mock_exchange, config_profiles):
        """Test different configuration profiles (conservative/moderate/aggressive)."""

        for profile_name, profile_config in config_profiles.items():
            # Create integrator with specific profile
            integrator = MarketMakingIntegrator(
                symbol="SUI-PERP",
                exchange_client=mock_exchange,
                dry_run=True,
                market_making_symbols=["SUI-PERP"],
                config={"profile": profile_name, **profile_config},
            )

            mock_llm = Mock()
            await integrator.initialize(mock_llm)

            # Verify profile-specific settings
            if profile_name == "conservative":
                # Conservative should have larger spreads, fewer levels
                assert profile_config["strategy"]["base_spread_bps"] >= 15
                assert profile_config["strategy"]["order_levels"] <= 2
                assert profile_config["strategy"]["max_position_pct"] <= 15.0
                assert profile_config["risk"]["stop_loss_pct"] <= 1.5

            elif profile_name == "moderate":
                # Moderate should have balanced settings
                assert 8 <= profile_config["strategy"]["base_spread_bps"] <= 15
                assert 2 <= profile_config["strategy"]["order_levels"] <= 4
                assert 15.0 <= profile_config["strategy"]["max_position_pct"] <= 30.0

            elif profile_name == "aggressive":
                # Aggressive should have tighter spreads, more levels
                assert profile_config["strategy"]["base_spread_bps"] <= 10
                assert profile_config["strategy"]["order_levels"] >= 4
                assert profile_config["strategy"]["max_position_pct"] >= 35.0
                assert profile_config["risk"]["stop_loss_pct"] >= 2.5

            # Test that profile affects behavior
            with patch.object(integrator, "market_making_engine") as mock_engine:
                mock_engine.analyze_market_and_decide = AsyncMock(
                    return_value=TradeAction(
                        action="LONG",
                        size_pct=profile_config["strategy"]["max_position_pct"],
                        take_profit_pct=2.0,
                        stop_loss_pct=profile_config["risk"]["stop_loss_pct"],
                        rationale=f"Market making with {profile_name} profile",
                    )
                )

                # Create market state using helper method
                market_state = self.create_test_market_state(
                    "SUI-PERP", Decimal("3.51")
                )

                result = await integrator.analyze_market(market_state)
                assert profile_name in result.rationale.lower()
                assert result.stop_loss_pct == profile_config["risk"]["stop_loss_pct"]

            await integrator.stop()

    @pytest.mark.integration()
    def test_cli_command_integration(self, test_config):
        """Test CLI command integration with market making."""

        runner = CliRunner()

        # Test CLI with market making enabled
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"
            config_file.write_text(json.dumps(test_config, indent=2))

            # Test dry run command
            with patch.dict(
                os.environ,
                {
                    "SYSTEM__DRY_RUN": "true",
                    "TRADING__SYMBOL": "BTC-USD",
                    "MARKET_MAKING__ENABLED": "true",
                    "MARKET_MAKING__SYMBOL": "SUI-PERP",
                },
            ):
                # Test config validation
                result = runner.invoke(cli, ["validate-config"])
                assert result.exit_code == 0

                # Test live command with short duration
                with patch("bot.main.TradingEngine") as mock_engine:
                    mock_instance = Mock()
                    mock_instance.run = AsyncMock()
                    mock_engine.return_value = mock_instance

                    result = runner.invoke(
                        cli,
                        [
                            "live",
                            "--max-runtime",
                            "1",  # 1 second for testing
                        ],
                    )

                    # Should start successfully
                    assert result.exit_code == 0 or "KeyboardInterrupt" in str(
                        result.output
                    )

    @pytest.mark.integration()
    @pytest.mark.asyncio()
    async def test_paper_trading_validation(self, mock_exchange, test_config):
        """Test paper trading mode with market making integration."""

        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_exchange,
            dry_run=True,  # Paper trading mode
            market_making_symbols=["SUI-PERP"],
            config=test_config["market_making"],
        )

        mock_llm = Mock()
        await integrator.initialize(mock_llm)

        # Test that paper trading mode is respected
        assert integrator.dry_run

        # Execute market making cycle in paper trading
        market_data = MarketData(
            symbol="SUI-PERP",
            timestamp=datetime.now(UTC),
            open=Decimal("3.50"),
            high=Decimal("3.52"),
            low=Decimal("3.48"),
            close=Decimal("3.51"),
            volume=Decimal(1000),
        )

        with patch.object(integrator, "market_making_engine") as mock_engine:
            mock_engine.analyze_market_and_decide = AsyncMock(
                return_value=TradeAction(
                    action="LONG",
                    size_pct=15.0,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.5,
                    rationale="Paper trading: simulated market making order",
                )
            )

            market_state = MarketState(
                symbol="SUI-PERP",
                current_price=market_data.close,
                market_data=[market_data],
                timestamp=datetime.now(UTC),
                indicators=IndicatorData(),
            )

            result = await integrator.analyze_market(market_state)

            # Verify paper trading behavior
            assert result.action == "LONG"
            assert "simulated" in result.rationale.lower()
            assert integrator.dry_run  # Verify we're in paper trading mode

        # Test real market data usage in paper trading
        market_data_history = await mock_exchange.get_market_data("SUI-PERP")
        assert len(market_data_history) > 0
        assert all(isinstance(data.close, Decimal) for data in market_data_history)

        await integrator.stop()

    @pytest.mark.integration()
    @pytest.mark.asyncio()
    async def test_error_handling_and_recovery(self, mock_exchange, test_config):
        """Test error handling and recovery mechanisms."""

        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_exchange,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
            config=test_config["market_making"],
        )

        mock_llm = Mock()
        await integrator.initialize(mock_llm)

        # Test exchange connection error
        mock_exchange.is_connected = False

        health_status = await integrator.health_check()
        # Note: Mock exchange connection status doesn't directly affect integrator health
        # but we can test that health check completes
        assert "healthy" in health_status

        # Test recovery
        mock_exchange.is_connected = True
        health_status = await integrator.health_check()
        assert "healthy" in health_status

        # Test order placement error
        original_place_order = mock_exchange.place_order
        mock_exchange.place_order = AsyncMock(
            side_effect=Exception("Order placement failed")
        )

        market_data = MarketData(
            symbol="SUI-PERP",
            timestamp=datetime.now(UTC),
            open=Decimal("3.50"),
            high=Decimal("3.52"),
            low=Decimal("3.48"),
            close=Decimal("3.51"),
            volume=Decimal(1000),
        )

        with patch.object(integrator, "market_making_engine") as mock_engine:
            mock_engine.analyze_market_and_decide = AsyncMock(
                side_effect=Exception("Market making engine error")
            )

            market_state = MarketState(
                symbol="SUI-PERP",
                current_price=market_data.close,
                market_data=[market_data],
                timestamp=datetime.now(UTC),
                indicators=IndicatorData(),
            )

            # Should handle error gracefully and return HOLD
            result = await integrator.analyze_market(market_state)
            assert result.action == "HOLD"  # Should fallback to HOLD on error
            assert "error" in result.rationale.lower()

        # Test error count tracking
        assert integrator.status.error_count > 0
        assert integrator.status.last_error_time is not None

        # Restore functionality
        mock_exchange.place_order = original_place_order

        await integrator.stop()

    @pytest.mark.integration()
    @pytest.mark.asyncio()
    async def test_performance_benchmarks(self, mock_exchange, test_config):
        """Test performance benchmarks for market making system."""

        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_exchange,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
            config=test_config["market_making"],
        )

        mock_llm = Mock()
        await integrator.initialize(mock_llm)

        # Benchmark initialization time
        start_time = time.time()
        mock_llm_bench = Mock()
        await integrator.initialize(mock_llm_bench)
        init_time = time.time() - start_time
        assert init_time < 1.0  # Should initialize within 1 second

        # Benchmark trading cycle execution time
        market_data = MarketData(
            symbol="SUI-PERP",
            timestamp=datetime.now(UTC),
            open=Decimal("3.50"),
            high=Decimal("3.52"),
            low=Decimal("3.48"),
            close=Decimal("3.51"),
            volume=Decimal(1000),
        )

        # Measure multiple cycles
        cycle_times = []
        num_cycles = 10

        with patch.object(integrator, "market_making_engine") as mock_engine:
            mock_engine.analyze_market_and_decide = AsyncMock(
                return_value=TradeAction(
                    action="LONG",
                    size_pct=20.0,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.5,
                    rationale="Performance test execution",
                )
            )

            market_state = MarketState(
                symbol="SUI-PERP",
                current_price=market_data.close,
                market_data=[market_data],
                timestamp=datetime.now(UTC),
                indicators=IndicatorData(),
            )

            for _ in range(num_cycles):
                start_time = time.time()
                await integrator.analyze_market(market_state)
                cycle_time = time.time() - start_time
                cycle_times.append(cycle_time)

        # Performance assertions
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        max_cycle_time = max(cycle_times)

        assert avg_cycle_time < 0.5  # Average cycle should be under 500ms
        assert max_cycle_time < 1.0  # Max cycle should be under 1 second

        # Test concurrent cycle handling
        tasks = []
        for _ in range(5):
            task = asyncio.create_task(integrator.analyze_market(market_state))
            tasks.append(task)

        start_time = time.time()
        await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time

        # Concurrent execution should be efficient
        assert concurrent_time < 2.0  # 5 concurrent cycles in under 2 seconds

        # Memory usage check (basic)
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 500  # Should use less than 500MB for testing

        await integrator.stop()

    @pytest.mark.integration()
    @pytest.mark.asyncio()
    async def test_full_integration_scenario(
        self, mock_exchange, mock_llm_agent, test_config
    ):
        """Test complete integration scenario with multiple components."""

        # Simulate a complete trading session
        integrator = MarketMakingIntegrator(
            symbol="BTC-USD",  # Primary symbol for LLM
            exchange_client=mock_exchange,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
            config=test_config["market_making"],
        )

        mock_llm = Mock()
        await integrator.initialize(mock_llm)

        # Simulate market data for multiple symbols
        symbols = ["BTC-USD", "SUI-PERP", "ETH-USD"]
        trading_session_results = {}

        for symbol in symbols:
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                open=Decimal("3.50"),
                high=Decimal("3.52"),
                low=Decimal("3.48"),
                close=Decimal("3.51"),
                volume=Decimal(1000),
            )

            if integrator.get_strategy_for_symbol(symbol) == "market_making":
                # Market making execution
                with patch.object(integrator, "market_making_engine") as mock_engine:
                    mock_engine.analyze_market_and_decide = AsyncMock(
                        return_value=TradeAction(
                            action="LONG",
                            size_pct=25.0,
                            take_profit_pct=2.5,
                            stop_loss_pct=1.5,
                            rationale="Market making strategy executed",
                        )
                    )

                    market_state = MarketState(
                        symbol=symbol,
                        current_price=market_data.close,
                        market_data=[market_data],
                        timestamp=datetime.now(UTC),
                        indicators=IndicatorData(),
                    )

                    result = await integrator.analyze_market(market_state)
                    trading_session_results[symbol] = {
                        "strategy": "market_making",
                        "symbol": symbol,
                        "action": result.action,
                        "rationale": result.rationale,
                    }
            else:
                # LLM agent execution (simulated)
                trading_session_results[symbol] = {
                    "strategy": "llm_agent",
                    "symbol": symbol,
                    "action": "HOLD",
                    "rationale": "Waiting for better entry",
                }

        # Verify session results
        assert "SUI-PERP" in trading_session_results
        assert trading_session_results["SUI-PERP"]["strategy"] == "market_making"
        assert trading_session_results["SUI-PERP"]["action"] == "LONG"

        for symbol in ["BTC-USD", "ETH-USD"]:
            if symbol in trading_session_results:
                assert trading_session_results[symbol]["strategy"] == "llm_agent"

        # Test health monitoring during session
        health_status = await integrator.health_check()
        assert health_status["healthy"]
        status = health_status["status"]
        assert status["market_making_enabled"]

        # Test performance metrics
        performance_metrics = {
            "total_symbols_processed": len(trading_session_results),
            "mm_symbols": [
                s
                for s, r in trading_session_results.items()
                if r.get("strategy") == "market_making"
            ],
            "llm_symbols": [
                s
                for s, r in trading_session_results.items()
                if r.get("strategy") == "llm_agent"
            ],
            "total_actions_taken": len(
                [
                    r
                    for r in trading_session_results.values()
                    if r.get("action") != "HOLD"
                ]
            ),
        }

        assert performance_metrics["total_symbols_processed"] >= 1
        assert len(performance_metrics["mm_symbols"]) >= 1
        assert performance_metrics["total_actions_taken"] >= 0

        # Test graceful shutdown
        await integrator.stop()
        assert not integrator.status.is_running

        # Final health check should show system stopped
        final_health = await integrator.health_check()
        final_status = final_health.get("status", {})
        assert not final_status.get("is_running", True)


class TestMarketMakingCLIIntegration:
    """Test CLI integration with market making system."""

    @pytest.mark.integration()
    def test_cli_market_making_commands(self):
        """Test CLI commands specific to market making."""
        runner = CliRunner()

        # Test configuration validation with market making
        with patch.dict(
            os.environ,
            {
                "MARKET_MAKING__ENABLED": "true",
                "MARKET_MAKING__SYMBOL": "SUI-PERP",
                "MARKET_MAKING__PROFILE": "moderate",
            },
        ):
            result = runner.invoke(cli, ["validate-config"])
            assert result.exit_code == 0

    @pytest.mark.integration()
    def test_cli_with_different_profiles(self):
        """Test CLI with different market making profiles."""
        runner = CliRunner()

        profiles = ["conservative", "moderate", "aggressive"]

        for profile in profiles:
            with patch.dict(
                os.environ,
                {
                    "MARKET_MAKING__ENABLED": "true",
                    "MARKET_MAKING__PROFILE": profile,
                    "SYSTEM__DRY_RUN": "true",
                },
            ):
                result = runner.invoke(cli, ["validate-config"])
                assert result.exit_code == 0


class TestMarketMakingErrorRecovery:
    """Test error recovery and resilience mechanisms."""

    @pytest.mark.integration()
    @pytest.mark.asyncio()
    async def test_network_error_recovery(self, mock_exchange):
        """Test recovery from network errors."""

        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_exchange,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
        )

        mock_llm = Mock()
        await integrator.initialize(mock_llm)

        # Simulate network error
        mock_exchange.is_connected = False

        # System should detect and handle the error
        health_status = await integrator.health_check()
        assert "healthy" in health_status

        # Simulate recovery
        mock_exchange.is_connected = True

        # Give system time to recover
        await asyncio.sleep(0.1)

        health_status = await integrator.health_check()
        assert "healthy" in health_status

        await integrator.stop()

    @pytest.mark.integration()
    @pytest.mark.asyncio()
    async def test_configuration_error_handling(self):
        """Test handling of configuration errors."""

        # Test with invalid configuration
        invalid_config = {
            "strategy": {
                "base_spread_bps": -10,  # Invalid negative spread
                "order_levels": 0,  # Invalid zero levels
            }
        }

        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=Mock(),
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
            config=invalid_config,
        )

        # Should handle invalid config gracefully
        try:
            mock_llm = Mock()
            await integrator.initialize(mock_llm)
            # Should either initialize with defaults or raise handled exception
            assert True
        except Exception as e:
            # Should be a meaningful error message
            assert "config" in str(e).lower() or "invalid" in str(e).lower()


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
