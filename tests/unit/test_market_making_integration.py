"""
Comprehensive tests for the Market Making Integration module.

This test suite covers all aspects of the market making integration including:
- Initialization and configuration
- Symbol-specific strategy routing
- Health monitoring and error recovery
- Paper trading mode compatibility
- Lifecycle management
- Error handling and fallback behavior
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import the classes we're testing
try:
    from bot.strategy.market_making_integration import (
        MarketMakingIntegrationStatus,
        MarketMakingIntegrator,
        MarketMakingIntegratorFactory,
    )
    from bot.trading_types import MarketState, TradeAction
except ImportError as e:
    # Handle import errors gracefully for testing
    pytest.skip(f"Market making integration not available: {e}")


class MockExchangeClient:
    """Mock exchange client for testing."""

    def __init__(self, client_type: str = "bluefin"):
        self.client_type = client_type
        self.is_connected = True

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False


class MockBluefinClient(MockExchangeClient):
    """Mock Bluefin client specifically."""

    def __init__(self):
        super().__init__("bluefin")


class MockLLMAgent:
    """Mock LLM agent for testing."""

    def __init__(self):
        self.is_available_value = True
        self.analyze_market_calls = []

    async def analyze_market(self, market_state: MarketState) -> TradeAction:
        self.analyze_market_calls.append(market_state)
        return TradeAction(
            action="HOLD",
            size_pct=0.0,
            take_profit_pct=0.0,
            stop_loss_pct=0.0,
            rationale="Mock LLM decision",
        )

    def is_available(self) -> bool:
        return self.is_available_value


class MockMarketMakingEngine:
    """Mock market making engine for testing."""

    def __init__(self):
        self.is_initialized = False
        self.is_running = False
        self.analyze_calls = []
        self.emergency_stopped = False

    async def initialize(self):
        self.is_initialized = True

    async def start(self):
        self.is_running = True

    async def stop(self):
        self.is_running = False

    async def analyze_market_and_decide(self, market_state: MarketState) -> TradeAction:
        self.analyze_calls.append(market_state)
        return TradeAction(
            action="LONG",
            size_pct=25.0,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale="Mock market making decision",
        )

    async def get_status(self) -> dict[str, Any]:
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "emergency_stop": self.emergency_stopped,
            "cycle_count": 10,
            "error_count": 0,
        }

    async def emergency_stop(self):
        self.emergency_stopped = True
        self.is_running = False


def create_mock_market_state(symbol: str = "SUI-PERP") -> MarketState:
    """Create a mock market state for testing."""
    from bot.trading_types import IndicatorData, MarketData, Position

    timestamp = datetime.now(UTC)

    # Create mock OHLCV data
    ohlcv_data = [
        MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=Decimal("1.45"),
            high=Decimal("1.48"),
            low=Decimal("1.44"),
            close=Decimal("1.46"),
            volume=Decimal(1000),
        ),
        MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=Decimal("1.46"),
            high=Decimal("1.49"),
            low=Decimal("1.45"),
            close=Decimal("1.47"),
            volume=Decimal(1100),
        ),
        MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=Decimal("1.47"),
            high=Decimal("1.50"),
            low=Decimal("1.46"),
            close=Decimal("1.48"),
            volume=Decimal(1200),
        ),
    ]

    # Create mock indicators
    indicators = IndicatorData(
        timestamp=timestamp,
        cipher_a_dot=0.1,
        cipher_b_wave=-0.05,
        cipher_b_money_flow=0.15,
        rsi=55.0,
        ema_fast=1.47,
        ema_slow=1.45,
        vwap=1.46,
    )

    # Create mock position
    current_position = Position(
        symbol=symbol,
        side="FLAT",
        size=Decimal(0),
        entry_price=None,
        unrealized_pnl=Decimal(0),
        realized_pnl=Decimal(0),
        timestamp=timestamp,
    )

    return MarketState(
        symbol=symbol,
        interval="1m",
        timestamp=timestamp,
        current_price=Decimal("1.50"),
        ohlcv_data=ohlcv_data,
        indicators=indicators,
        current_position=current_position,
    )


class TestMarketMakingIntegrationStatus:
    """Test the MarketMakingIntegrationStatus class."""

    def test_initial_status(self):
        """Test initial status values."""
        status = MarketMakingIntegrationStatus()

        assert not status.is_initialized
        assert not status.is_running
        assert not status.market_making_enabled
        assert status.symbol_strategy_map == {}
        assert status.initialization_time is None
        assert status.error_count == 0
        assert status.last_error_time is None
        assert status.last_error_message is None
        assert status.engine_status is None


class TestMarketMakingIntegrator:
    """Test the MarketMakingIntegrator class."""

    @pytest.fixture()
    def mock_exchange_client(self):
        """Create a mock Bluefin exchange client."""
        return MockBluefinClient()

    @pytest.fixture()
    def mock_llm_agent(self):
        """Create a mock LLM agent."""
        return MockLLMAgent()

    @pytest.fixture()
    def integrator(self, mock_exchange_client):
        """Create a MarketMakingIntegrator instance for testing."""
        return MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_exchange_client,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
        )

    def test_initialization(self, integrator):
        """Test integrator initialization."""
        assert integrator.symbol == "SUI-PERP"
        assert integrator.dry_run is True
        assert integrator.market_making_symbols == ["SUI-PERP"]
        assert not integrator.status.is_initialized
        assert not integrator.status.is_running
        assert integrator.market_making_engine is None
        assert integrator.llm_agent is None

    @pytest.mark.asyncio()
    async def test_initialize_with_market_making(self, integrator, mock_llm_agent):
        """Test initialization with market making enabled."""
        # Mock the factory to return our mock engine
        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm_agent)

            assert integrator.status.is_initialized
            assert integrator.status.market_making_enabled
            assert integrator.llm_agent is mock_llm_agent
            assert integrator.market_making_engine is mock_engine
            assert mock_engine.is_initialized
            assert integrator.status.symbol_strategy_map["SUI-PERP"] == "market_making"

    @pytest.mark.asyncio()
    async def test_initialize_without_market_making(self, mock_llm_agent):
        """Test initialization without market making for non-SUI-PERP symbols."""
        integrator = MarketMakingIntegrator(
            symbol="BTC-USD",
            exchange_client=None,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
        )

        await integrator.initialize(mock_llm_agent)

        assert integrator.status.is_initialized
        assert not integrator.status.market_making_enabled
        assert integrator.llm_agent is mock_llm_agent
        assert integrator.market_making_engine is None

    @pytest.mark.asyncio()
    async def test_analyze_market_with_market_making(self, integrator, mock_llm_agent):
        """Test market analysis using market making engine."""
        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm_agent)

            market_state = create_mock_market_state("SUI-PERP")
            action = await integrator.analyze_market(market_state)

            assert action.action == "LONG"
            assert action.size_pct == 25.0
            assert action.rationale == "Mock market making decision"
            assert len(mock_engine.analyze_calls) == 1
            assert len(mock_llm_agent.analyze_market_calls) == 0

    @pytest.mark.asyncio()
    async def test_analyze_market_with_llm_agent(self, mock_llm_agent):
        """Test market analysis using LLM agent."""
        integrator = MarketMakingIntegrator(
            symbol="BTC-USD",
            exchange_client=None,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
        )

        await integrator.initialize(mock_llm_agent)

        market_state = create_mock_market_state("BTC-USD")
        action = await integrator.analyze_market(market_state)

        assert action.action == "HOLD"
        assert action.size_pct == 0.0
        assert action.rationale == "Mock LLM decision"
        assert len(mock_llm_agent.analyze_market_calls) == 1

    @pytest.mark.asyncio()
    async def test_analyze_market_error_handling(self, integrator, mock_llm_agent):
        """Test error handling during market analysis."""
        mock_engine = MockMarketMakingEngine()
        mock_engine.analyze_market_and_decide = AsyncMock(
            side_effect=Exception("Analysis failed")
        )

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm_agent)

            market_state = create_mock_market_state("SUI-PERP")
            action = await integrator.analyze_market(market_state)

            # Should return safe HOLD action on error
            assert action.action == "HOLD"
            assert action.size_pct == 0.0
            assert "Error in analysis" in action.rationale
            assert integrator.status.error_count == 1
            assert integrator.status.last_error_message == "Analysis failed"

    @pytest.mark.asyncio()
    async def test_start_and_stop(self, integrator, mock_llm_agent):
        """Test starting and stopping the integrator."""
        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm_agent)

            # Test start
            await integrator.start()
            assert integrator.status.is_running
            assert mock_engine.is_running

            # Test stop
            await integrator.stop()
            assert not integrator.status.is_running
            assert not mock_engine.is_running

    @pytest.mark.asyncio()
    async def test_get_status(self, integrator, mock_llm_agent):
        """Test status reporting."""
        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm_agent)
            await integrator.start()

            status = await integrator.get_status()

            assert status["is_initialized"]
            assert status["is_running"]
            assert status["market_making_enabled"]
            assert status["symbol"] == "SUI-PERP"
            assert status["market_making_symbols"] == ["SUI-PERP"]
            assert status["symbol_strategy_map"]["SUI-PERP"] == "market_making"
            assert status["dry_run"] is True
            assert status["error_count"] == 0
            assert status["engine_status"] is not None

    @pytest.mark.asyncio()
    async def test_health_check(self, integrator, mock_llm_agent):
        """Test health check functionality."""
        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm_agent)
            await integrator.start()

            health = await integrator.health_check()

            assert health["healthy"]
            assert health["checks"]["initialized"]
            assert health["checks"]["running"]
            assert health["checks"]["no_errors"]
            assert health["checks"]["engine_healthy"]

    @pytest.mark.asyncio()
    async def test_health_check_unhealthy(self, integrator, mock_llm_agent):
        """Test health check with unhealthy state."""
        mock_engine = MockMarketMakingEngine()
        mock_engine.emergency_stopped = True

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm_agent)

            # Introduce an error
            integrator.status.error_count = 1

            health = await integrator.health_check()

            assert not health["healthy"]
            assert health["checks"]["initialized"]
            assert not health["checks"]["no_errors"]

    def test_is_available(self, integrator, mock_llm_agent):
        """Test availability check."""
        # Not available before initialization
        assert not integrator.is_available()

        # Mock initialization
        integrator.status.is_initialized = True
        integrator.llm_agent = mock_llm_agent

        # Available after initialization
        assert integrator.is_available()

    def test_get_strategy_for_symbol(self, integrator):
        """Test strategy determination for symbols."""
        assert integrator.get_strategy_for_symbol("SUI-PERP") == "market_making"
        assert integrator.get_strategy_for_symbol("BTC-USD") == "llm"
        assert integrator.get_strategy_for_symbol("ETH-USD") == "llm"

    @pytest.mark.asyncio()
    async def test_managed_lifecycle(self, integrator, mock_llm_agent):
        """Test managed lifecycle context manager."""
        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm_agent)

            async with integrator.managed_lifecycle():
                assert integrator.status.is_running
                assert mock_engine.is_running

            # Should be stopped after context exit
            assert not integrator.status.is_running
            assert not mock_engine.is_running

    @pytest.mark.asyncio()
    async def test_emergency_stop(self, integrator, mock_llm_agent):
        """Test emergency stop functionality."""
        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm_agent)
            await integrator.start()

            assert integrator.status.is_running
            assert mock_engine.is_running

            await integrator.emergency_stop()

            assert not integrator.status.is_running
            assert mock_engine.emergency_stopped

    def test_repr(self, integrator):
        """Test string representation."""
        repr_str = repr(integrator)
        assert "MarketMakingIntegrator" in repr_str
        assert "symbol=SUI-PERP" in repr_str
        assert "market_making_enabled=False" in repr_str
        assert "initialized=False" in repr_str
        assert "running=False" in repr_str


class TestMarketMakingIntegratorFactory:
    """Test the MarketMakingIntegratorFactory class."""

    def test_create_integrator(self):
        """Test creating integrator with factory."""
        mock_client = MockBluefinClient()

        integrator = MarketMakingIntegratorFactory.create_integrator(
            symbol="SUI-PERP",
            exchange_client=mock_client,
            dry_run=True,
            config={"market_making_symbols": ["SUI-PERP", "ETH-PERP"]},
        )

        assert integrator.symbol == "SUI-PERP"
        assert integrator.exchange_client is mock_client
        assert integrator.dry_run is True
        assert integrator.market_making_symbols == ["SUI-PERP", "ETH-PERP"]

    def test_create_integrator_defaults(self):
        """Test creating integrator with default configuration."""
        integrator = MarketMakingIntegratorFactory.create_integrator(
            symbol="BTC-USD", dry_run=False
        )

        assert integrator.symbol == "BTC-USD"
        assert integrator.dry_run is False
        assert integrator.market_making_symbols == ["SUI-PERP"]  # Default

    @patch("bot.strategy.market_making_integration.settings")
    def test_create_from_settings(self, mock_settings):
        """Test creating integrator from global settings."""
        mock_settings.system.dry_run = True

        integrator = MarketMakingIntegratorFactory.create_from_settings(
            symbol="SUI-PERP"
        )

        assert integrator.symbol == "SUI-PERP"
        assert integrator.dry_run is True


class TestIntegrationScenarios:
    """Test complex integration scenarios."""

    @pytest.mark.asyncio()
    async def test_multiple_symbol_routing(self):
        """Test routing decisions for multiple symbols."""
        mock_client = MockBluefinClient()
        mock_llm = MockLLMAgent()

        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_client,
            dry_run=True,
            market_making_symbols=["SUI-PERP", "ETH-PERP"],
        )

        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm)

            # Test SUI-PERP (should use market making)
            sui_state = create_mock_market_state("SUI-PERP")
            sui_action = await integrator.analyze_market(sui_state)
            assert sui_action.action == "LONG"  # From market making

            # Test ETH-PERP (should use market making)
            eth_state = create_mock_market_state("ETH-PERP")
            eth_action = await integrator.analyze_market(eth_state)
            assert eth_action.action == "LONG"  # From market making

            # Test BTC-USD (should use LLM)
            btc_state = create_mock_market_state("BTC-USD")
            btc_action = await integrator.analyze_market(btc_state)
            assert btc_action.action == "HOLD"  # From LLM

            # Verify call counts
            assert len(mock_engine.analyze_calls) == 2  # SUI-PERP and ETH-PERP
            assert len(mock_llm.analyze_market_calls) == 1  # BTC-USD

    @pytest.mark.asyncio()
    async def test_fallback_to_llm_on_engine_failure(self):
        """Test fallback to LLM when market making engine fails."""
        mock_client = MockBluefinClient()
        mock_llm = MockLLMAgent()

        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_client,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
        )

        # Mock engine initialization failure
        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.side_effect = Exception("Engine creation failed")
            mock_get_factory.return_value = mock_factory

            # Should still initialize successfully (graceful degradation)
            try:
                await integrator.initialize(mock_llm)
                # Should fail during initialization
                assert False, "Expected initialization to fail"
            except Exception as e:
                assert "Engine creation failed" in str(e)
                assert integrator.status.error_count > 0

    @pytest.mark.asyncio()
    async def test_concurrent_operations(self):
        """Test concurrent operations on the integrator."""
        mock_client = MockBluefinClient()
        mock_llm = MockLLMAgent()

        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_client,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
        )

        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm)
            await integrator.start()

            # Simulate concurrent market analysis calls
            market_state = create_mock_market_state("SUI-PERP")

            tasks = [
                integrator.analyze_market(market_state),
                integrator.analyze_market(market_state),
                integrator.analyze_market(market_state),
                integrator.get_status(),
                integrator.health_check(),
            ]

            results = await asyncio.gather(*tasks)

            # All market analysis should succeed
            for i in range(3):
                assert results[i].action == "LONG"

            # Status and health check should succeed
            assert results[3]["is_running"]
            assert results[4]["healthy"]

    @pytest.mark.asyncio()
    async def test_paper_trading_mode_compatibility(self):
        """Test compatibility with paper trading mode."""
        mock_client = MockBluefinClient()
        mock_llm = MockLLMAgent()

        # Test dry_run=True
        integrator = MarketMakingIntegrator(
            symbol="SUI-PERP",
            exchange_client=mock_client,
            dry_run=True,
            market_making_symbols=["SUI-PERP"],
        )

        mock_engine = MockMarketMakingEngine()

        with patch(
            "bot.strategy.market_making_integration._get_market_making_engine_factory"
        ) as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_engine.return_value = mock_engine
            mock_get_factory.return_value = mock_factory

            await integrator.initialize(mock_llm)

            status = await integrator.get_status()
            assert status["dry_run"] is True

            # Should still work in paper trading mode
            market_state = create_mock_market_state("SUI-PERP")
            action = await integrator.analyze_market(market_state)
            assert action.action == "LONG"


@pytest.mark.asyncio()
async def test_full_integration_lifecycle():
    """Test the complete integration lifecycle from creation to shutdown."""
    # Create all components
    mock_client = MockBluefinClient()
    mock_llm = MockLLMAgent()
    mock_engine = MockMarketMakingEngine()

    # Create integrator
    integrator = MarketMakingIntegratorFactory.create_integrator(
        symbol="SUI-PERP",
        exchange_client=mock_client,
        dry_run=True,
    )

    with patch(
        "bot.strategy.market_making_integration._get_market_making_engine_factory"
    ) as mock_get_factory:
        mock_factory = Mock()
        mock_factory.create_engine.return_value = mock_engine
        mock_get_factory.return_value = mock_factory

        # Full lifecycle test
        async with integrator.managed_lifecycle():
            # Initialize
            await integrator.initialize(mock_llm)
            assert integrator.is_available()

            # Verify health
            health = await integrator.health_check()
            assert health["healthy"]

            # Test trading decisions
            market_state = create_mock_market_state("SUI-PERP")
            action = await integrator.analyze_market(market_state)
            assert action.action == "LONG"

            # Test status reporting
            status = await integrator.get_status()
            assert status["is_running"]
            assert status["market_making_enabled"]

        # After context exit, should be stopped
        final_status = await integrator.get_status()
        assert not final_status["is_running"]
