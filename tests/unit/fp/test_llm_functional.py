"""Comprehensive tests for functional LLM strategy integration.

This module tests:
1. Functional LLM strategy integration
2. LLM prompt generation and context creation
3. LLM response parsing and validation
4. Strategy adapter functionality
5. Type conversions between FP and legacy types
6. Error handling and fallback mechanisms
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

# Import adapters
from bot.fp.adapters.strategy_adapter import (
    FunctionalLLMStrategy,
    LLMAgentAdapter,
    MemoryEnhancedLLMAgentAdapter,
    TypeConverter,
)
from bot.fp.strategies.llm_functional import (
    LLMConfig,
    LLMContext,
    LLMProvider,
    LLMResponse,
    adjust_confidence_by_market_conditions,
    create_context_window,
    create_market_context,
    generate_trading_prompt,
    parse_llm_response,
    validate_llm_decision,
)
from bot.fp.types.market import MarketSnapshot

# Import FP types and functions
from bot.fp.types.trading import Hold, Long, Short

# Import legacy types for compatibility testing
from bot.trading_types import TradeAction


class TestLLMConfig:
    """Test LLM configuration types."""

    def test_llm_config_creation(self):
        """Test creating LLM configuration."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            temperature=0.3,
            max_tokens=500,
        )

        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.temperature == 0.3
        assert config.max_tokens == 500

    def test_llm_config_defaults(self):
        """Test LLM configuration with defaults."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
        )

        assert config.temperature == 0.3
        assert config.max_tokens == 500
        assert config.timeout == 30.0


class TestLLMContext:
    """Test LLM context creation and handling."""

    def test_create_market_context_basic(self):
        """Test creating basic market context."""
        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        # Create context with minimal VuManChu state (None for now)
        context = create_market_context(
            market_state=market_state,
            vumanchu_state=None,
            recent_trades=[],
            lookback_periods=20,
        )

        assert isinstance(context, LLMContext)
        assert context.market_state == market_state
        assert context.recent_trades == []
        assert "price_action" in context.indicators
        assert "volume" in context.indicators

    def test_create_market_context_with_trades(self):
        """Test creating context with recent trades."""
        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        recent_trades = [
            {"action": "LONG", "price": 44000, "pnl": 0.02},
            {"action": "SHORT", "price": 46000, "pnl": -0.01},
        ]

        context = create_market_context(
            market_state=market_state, vumanchu_state=None, recent_trades=recent_trades
        )

        assert len(context.recent_trades) == 2
        assert context.recent_trades[0]["action"] == "LONG"


class TestPromptGeneration:
    """Test LLM prompt generation."""

    def test_generate_trading_prompt_basic(self):
        """Test basic prompt generation."""
        context = LLMContext(
            market_state=MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                price=Decimal(45000),
                volume=Decimal(1000),
                bid=Decimal(44990),
                ask=Decimal(45010),
            ),
            indicators={
                "price_action": {
                    "current_price": 45000,
                    "price_change_24h": 0.02,
                    "volatility": 0.03,
                },
                "volume": {
                    "current": 1000,
                    "trend": "NORMAL",
                },
            },
            recent_trades=[],
        )

        # Mock TradingParams
        class MockTradingParams:
            risk_per_trade = 0.1
            max_leverage = 5
            stop_loss_pct = 1.0
            take_profit_pct = 2.0

        params = MockTradingParams()
        prompt = generate_trading_prompt(context, params, include_examples=True)

        assert "BTC-USD" in prompt
        assert "$45000.00" in prompt
        assert "Volume: 1000.00" in prompt
        assert "JSON format" in prompt
        assert "action" in prompt
        assert "confidence" in prompt

    def test_generate_prompt_with_recent_trades(self):
        """Test prompt generation with recent trading history."""
        context = LLMContext(
            market_state=MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                price=Decimal(45000),
                volume=Decimal(1000),
                bid=Decimal(44990),
                ask=Decimal(45010),
            ),
            indicators={
                "price_action": {"current_price": 45000},
                "volume": {"current": 1000, "trend": "NORMAL"},
            },
            recent_trades=[
                {"action": "LONG", "price": 44000, "pnl": 0.02},
                {"action": "SHORT", "price": 46000, "pnl": -0.01},
            ],
        )

        class MockTradingParams:
            risk_per_trade = 0.1
            max_leverage = 5
            stop_loss_pct = 1.0
            take_profit_pct = 2.0

        params = MockTradingParams()
        prompt = generate_trading_prompt(context, params)

        assert "Recent Trading Activity" in prompt
        assert "LONG at $44000.00" in prompt
        assert "SHORT at $46000.00" in prompt


class TestLLMResponseParsing:
    """Test parsing LLM responses."""

    def test_parse_valid_llm_response(self):
        """Test parsing valid JSON response."""
        response_text = """
        Based on market analysis, I recommend:

        {
            "action": "LONG",
            "confidence": 0.75,
            "reasoning": "Strong bullish momentum with high volume",
            "risk_assessment": {
                "risk_level": "MEDIUM",
                "key_risks": ["Potential resistance at $46000"],
                "mitigation": "Set tight stop loss"
            },
            "suggested_params": {
                "position_size": 0.3,
                "stop_loss": 44000,
                "take_profit": 47000
            }
        }
        """

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        response = parse_llm_response(response_text, market_state)

        assert isinstance(response, LLMResponse)
        assert response.signal == Long  # Should be mapped to Long signal type
        assert response.confidence == 0.75
        assert "bullish momentum" in response.reasoning
        assert response.risk_assessment["risk_level"] == "MEDIUM"
        assert response.suggested_params["position_size"] == 0.3

    def test_parse_invalid_json_response(self):
        """Test parsing invalid JSON falls back to HOLD."""
        response_text = "This is not valid JSON"

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        response = parse_llm_response(response_text, market_state)

        assert isinstance(response, LLMResponse)
        assert response.signal == Hold
        assert "No valid JSON" in response.reasoning

    def test_parse_response_clamps_confidence(self):
        """Test that confidence is clamped to valid range."""
        response_text = """
        {
            "action": "LONG",
            "confidence": 1.5,
            "reasoning": "Over-confident response"
        }
        """

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        response = parse_llm_response(response_text, market_state)

        assert 0.0 <= response.confidence <= 1.0

    def test_parse_short_signal(self):
        """Test parsing SHORT signal response."""
        response_text = """
        {
            "action": "SHORT",
            "confidence": 0.8,
            "reasoning": "Bearish divergence detected"
        }
        """

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        response = parse_llm_response(response_text, market_state)

        assert response.signal == Short
        assert response.confidence == 0.8


class TestConfidenceAdjustment:
    """Test confidence adjustment based on market conditions."""

    def test_adjust_confidence_high_volatility(self):
        """Test confidence reduction in high volatility."""
        original_response = LLMResponse(
            signal=Long,
            confidence=0.8,
            reasoning="Original reasoning",
            risk_assessment={"risk_level": "LOW"},
        )

        # Create high volatility market state
        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44500),  # Wide spread = high volatility
            ask=Decimal(45500),
        )

        adjusted = adjust_confidence_by_market_conditions(
            original_response, market_state, volatility_threshold=0.01
        )

        # Confidence should be reduced due to high volatility
        assert adjusted.confidence < original_response.confidence
        assert "Adjusted confidence" in adjusted.reasoning

    def test_adjust_confidence_counter_trend(self):
        """Test confidence reduction for counter-trend trades."""
        original_response = LLMResponse(
            signal=Short,  # SHORT signal
            confidence=0.8,
            reasoning="Original reasoning",
            risk_assessment={"risk_level": "LOW"},
        )

        # Mock a market state that would indicate uptrend
        # This would require implementing _calculate_price_trend
        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        # For now, test basic functionality
        adjusted = adjust_confidence_by_market_conditions(
            original_response, market_state
        )

        assert isinstance(adjusted, LLMResponse)
        assert adjusted.signal == Short

    def test_low_confidence_becomes_hold(self):
        """Test that very low confidence results in HOLD."""
        original_response = LLMResponse(
            signal=Long,
            confidence=0.3,  # Low confidence
            reasoning="Weak signal",
            risk_assessment={"risk_level": "HIGH"},
        )

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44000),  # Very high volatility
            ask=Decimal(46000),
        )

        adjusted = adjust_confidence_by_market_conditions(
            original_response, market_state, volatility_threshold=0.01
        )

        # Should become HOLD due to low adjusted confidence
        assert adjusted.signal == Hold


class TestLLMDecisionValidation:
    """Test LLM decision validation."""

    def test_validate_valid_decision(self):
        """Test validation of valid LLM decision."""
        response = LLMResponse(
            signal=Long,
            confidence=0.8,
            reasoning="Strong bullish signal",
            risk_assessment={"risk_level": "MEDIUM"},
            suggested_params={
                "position_size": 0.2,
                "stop_loss": 44000,
                "take_profit": 47000,
            },
        )

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        class MockTradingParams:
            max_position_size = 0.5

        params = MockTradingParams()
        is_valid, error = validate_llm_decision(response, market_state, params)

        assert is_valid
        assert error is None

    def test_validate_low_confidence(self):
        """Test validation rejects low confidence."""
        response = LLMResponse(
            signal=Long,
            confidence=0.3,  # Below threshold
            reasoning="Weak signal",
            risk_assessment={"risk_level": "LOW"},
        )

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        class MockTradingParams:
            max_position_size = 0.5

        params = MockTradingParams()
        is_valid, error = validate_llm_decision(response, market_state, params)

        assert not is_valid
        assert "Confidence below minimum" in error

    def test_validate_invalid_stop_loss(self):
        """Test validation catches invalid stop loss."""
        response = LLMResponse(
            signal=Long,
            confidence=0.8,
            reasoning="Good signal",
            risk_assessment={"risk_level": "LOW"},
            suggested_params={
                "position_size": 0.2,
                "stop_loss": 46000,  # Invalid: above current price for LONG
                "take_profit": 47000,
            },
        )

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        class MockTradingParams:
            max_position_size = 0.5

        params = MockTradingParams()
        is_valid, error = validate_llm_decision(response, market_state, params)

        assert not is_valid
        assert "Invalid stop loss" in error


class TestTypeConverter:
    """Test type conversion between FP and legacy types."""

    def test_llm_response_to_trade_action_long(self):
        """Test converting Long signal to TradeAction."""
        llm_response = LLMResponse(
            signal=Long,
            confidence=0.8,
            reasoning="Strong bullish signal",
            risk_assessment={"risk_level": "MEDIUM"},
            suggested_params={
                "position_size": 0.25,
                "stop_loss": 44000,
                "take_profit": 47000,
            },
        )

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        trade_action = TypeConverter.llm_response_to_trade_action(
            llm_response, market_state
        )

        assert isinstance(trade_action, TradeAction)
        assert trade_action.action == "LONG"
        assert trade_action.size_pct == 25.0  # 0.25 * 100
        assert trade_action.rationale == "Strong bullish signal"

    def test_llm_response_to_trade_action_short(self):
        """Test converting Short signal to TradeAction."""
        llm_response = LLMResponse(
            signal=Short,
            confidence=0.7,
            reasoning="Bearish divergence",
            risk_assessment={"risk_level": "HIGH"},
        )

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        trade_action = TypeConverter.llm_response_to_trade_action(
            llm_response, market_state
        )

        assert trade_action.action == "SHORT"
        assert trade_action.rationale == "Bearish divergence"

    def test_llm_response_to_trade_action_hold(self):
        """Test converting Hold signal to TradeAction."""
        llm_response = LLMResponse(
            signal=Hold,
            confidence=1.0,
            reasoning="Unclear market direction",
            risk_assessment={"risk_level": "LOW"},
        )

        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        trade_action = TypeConverter.llm_response_to_trade_action(
            llm_response, market_state
        )

        assert trade_action.action == "HOLD"
        assert trade_action.size_pct == 0.0

    def test_create_trading_params(self):
        """Test creating TradingParams from settings."""
        with patch("bot.fp.adapters.strategy_adapter.settings") as mock_settings:
            mock_settings.trading.max_size_pct = 20
            mock_settings.trading.max_futures_leverage = 10

            params = TypeConverter.create_trading_params()

            assert params.risk_per_trade == 0.2  # 20 / 100
            assert params.max_leverage == 10


class TestFunctionalLLMStrategy:
    """Test the functional LLM strategy implementation."""

    @pytest.fixture
    def strategy(self):
        """Create a FunctionalLLMStrategy instance."""
        with patch("bot.fp.adapters.strategy_adapter.settings") as mock_settings:
            mock_settings.llm.provider = "openai"
            mock_settings.llm.model_name = "gpt-4"
            mock_settings.llm.openai_api_key.get_secret_value.return_value = "test-key"
            mock_settings.llm.temperature = 0.3
            mock_settings.llm.max_tokens = 500

            return FunctionalLLMStrategy()

    @pytest.mark.asyncio
    async def test_analyze_market_functional(self, strategy):
        """Test functional market analysis."""
        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        # Mock the LLM call
        with patch.object(strategy, "_call_llm") as mock_llm:
            mock_llm.return_value = """
            {
                "action": "LONG",
                "confidence": 0.8,
                "reasoning": "Bullish momentum",
                "risk_assessment": {
                    "risk_level": "MEDIUM",
                    "key_risks": ["Resistance at $46k"],
                    "mitigation": "Tight stops"
                }
            }
            """

            response = await strategy.analyze_market_functional(market_state)

            assert isinstance(response, LLMResponse)
            assert response.signal == Long
            assert response.confidence == 0.8

    @pytest.mark.asyncio
    async def test_analyze_market_with_error(self, strategy):
        """Test error handling in market analysis."""
        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        # Mock an error in LLM call
        with patch.object(strategy, "_call_llm") as mock_llm:
            mock_llm.side_effect = Exception("LLM API error")

            response = await strategy.analyze_market_functional(market_state)

            assert isinstance(response, LLMResponse)
            assert response.signal == Hold
            assert "Strategy error" in response.reasoning


class TestLLMAgentAdapter:
    """Test the LLM agent adapter for backward compatibility."""

    @pytest.fixture
    def adapter(self):
        """Create an LLMAgentAdapter instance."""
        with patch("bot.fp.adapters.strategy_adapter.settings") as mock_settings:
            mock_settings.llm.provider = "openai"
            mock_settings.llm.model_name = "gpt-4"

            return LLMAgentAdapter()

    @pytest.mark.asyncio
    async def test_analyze_market_compatibility(self, adapter):
        """Test that adapter maintains API compatibility."""
        # Create a legacy-style MarketState for compatibility
        market_state = Mock()
        market_state.symbol = "BTC-USD"
        market_state.current_price = Decimal(45000)

        # Mock the functional strategy
        with patch.object(
            adapter._strategy, "analyze_market_functional"
        ) as mock_analyze:
            mock_response = LLMResponse(
                signal=Long,
                confidence=0.8,
                reasoning="Test signal",
                risk_assessment={"risk_level": "MEDIUM"},
            )
            mock_analyze.return_value = mock_response

            # Mock the market snapshot conversion
            with patch(
                "bot.fp.adapters.strategy_adapter.MarketSnapshot"
            ) as mock_snapshot:
                mock_snapshot.return_value = MarketSnapshot(
                    timestamp=datetime.now(),
                    symbol="BTC-USD",
                    price=Decimal(45000),
                    volume=Decimal(1000),
                    bid=Decimal(44990),
                    ask=Decimal(45010),
                )

                trade_action = await adapter.analyze_market(market_state)

                assert isinstance(trade_action, TradeAction)
                assert trade_action.action == "LONG"
                assert trade_action.rationale == "Test signal"

    def test_is_available(self, adapter):
        """Test availability check."""
        assert adapter.is_available()

    def test_get_status(self, adapter):
        """Test status reporting."""
        status = adapter.get_status()

        assert isinstance(status, dict)
        assert status["strategy_type"] == "functional"
        assert status["available"] is True
        assert "completion_count" in status


class TestMemoryEnhancedLLMAgentAdapter:
    """Test the memory-enhanced LLM agent adapter."""

    @pytest.fixture
    def memory_adapter(self):
        """Create a MemoryEnhancedLLMAgentAdapter instance."""
        mock_memory_server = Mock()

        with patch("bot.fp.adapters.strategy_adapter.settings") as mock_settings:
            mock_settings.llm.provider = "openai"
            mock_settings.llm.model_name = "gpt-4"
            mock_settings.mcp.enabled = True

            return MemoryEnhancedLLMAgentAdapter(memory_server=mock_memory_server)

    @pytest.mark.asyncio
    async def test_analyze_market_with_memory(self, memory_adapter):
        """Test memory-enhanced market analysis."""
        market_state = Mock()
        market_state.symbol = "BTC-USD"
        market_state.current_price = Decimal(45000)

        # Mock memory retrieval
        with patch.object(memory_adapter, "_retrieve_relevant_memories") as mock_memory:
            mock_memory.return_value = [
                {"experience_id": "test", "outcome": "profitable"}
            ]

            # Mock the functional strategy
            with patch.object(
                memory_adapter._strategy, "analyze_market_functional"
            ) as mock_analyze:
                mock_response = LLMResponse(
                    signal=Long,
                    confidence=0.8,
                    reasoning="Memory-enhanced signal",
                    risk_assessment={"risk_level": "MEDIUM"},
                )
                mock_analyze.return_value = mock_response

                # Mock market snapshot creation
                with patch("bot.fp.adapters.strategy_adapter.MarketSnapshot"):
                    trade_action = await memory_adapter.analyze_market(market_state)

                    assert isinstance(trade_action, TradeAction)
                    assert "Memory-enhanced" in trade_action.rationale

    @pytest.mark.asyncio
    async def test_fallback_without_memory(self, memory_adapter):
        """Test fallback when memory is not available."""
        # Disable memory
        memory_adapter._memory_available = False

        market_state = Mock()
        market_state.symbol = "BTC-USD"
        market_state.current_price = Decimal(45000)

        with patch.object(memory_adapter, "_analyze_without_memory") as mock_fallback:
            mock_trade_action = TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=2.0,
                stop_loss_pct=1.0,
                rationale="No memory available",
            )
            mock_fallback.return_value = mock_trade_action

            trade_action = await memory_adapter.analyze_market(market_state)

            assert trade_action.action == "HOLD"
            mock_fallback.assert_called_once()

    def test_get_status_with_memory(self, memory_adapter):
        """Test status reporting with memory information."""
        status = memory_adapter.get_status()

        assert status["strategy_type"] == "functional_memory_enhanced"
        assert status["memory_enabled"] is True
        assert "memory_server_connected" in status


class TestContextWindow:
    """Test context window optimization."""

    def test_create_context_window_no_compression(self):
        """Test context window without compression."""
        history = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                price=Decimal(f"{40000 + i * 100}"),
                volume=Decimal(1000),
                bid=Decimal(f"{40000 + i * 100 - 10}"),
                ask=Decimal(f"{40000 + i * 100 + 10}"),
            )
            for i in range(50)
        ]

        result = create_context_window(history, max_size=100)

        assert len(result) == 50  # No compression needed
        assert result == history

    def test_create_context_window_with_compression(self):
        """Test context window with compression."""
        history = [
            MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                price=Decimal(f"{40000 + i * 100}"),
                volume=Decimal(1000),
                bid=Decimal(f"{40000 + i * 100 - 10}"),
                ask=Decimal(f"{40000 + i * 100 + 10}"),
            )
            for i in range(200)
        ]

        result = create_context_window(history, max_size=100, compression_ratio=0.5)

        assert len(result) <= 100
        # Recent data should be preserved at full resolution
        assert result[-1] == history[-1]  # Most recent data preserved


class TestIntegration:
    """Integration tests for LLM functional components."""

    @pytest.mark.asyncio
    async def test_end_to_end_llm_decision(self):
        """Test complete end-to-end LLM decision flow."""
        # Create market snapshot
        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        # Mock LLM response
        llm_response_text = """
        {
            "action": "LONG",
            "confidence": 0.8,
            "reasoning": "Strong bullish momentum with good volume",
            "risk_assessment": {
                "risk_level": "MEDIUM",
                "key_risks": ["Potential resistance"],
                "mitigation": "Use stop loss"
            },
            "suggested_params": {
                "position_size": 0.25,
                "stop_loss": 44000,
                "take_profit": 47000
            }
        }
        """

        # Parse LLM response
        llm_response = parse_llm_response(llm_response_text, market_state)

        # Convert to TradeAction
        trade_action = TypeConverter.llm_response_to_trade_action(
            llm_response, market_state
        )

        # Verify end-to-end conversion
        assert trade_action.action == "LONG"
        assert trade_action.size_pct == 25.0
        assert "bullish momentum" in trade_action.rationale
        assert trade_action.stop_loss_pct > 0
        assert trade_action.take_profit_pct > 0

    def test_error_handling_chain(self):
        """Test error handling throughout the decision chain."""
        market_state = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            price=Decimal(45000),
            volume=Decimal(1000),
            bid=Decimal(44990),
            ask=Decimal(45010),
        )

        # Test with invalid JSON
        invalid_response = "This is not JSON"
        llm_response = parse_llm_response(invalid_response, market_state)

        # Should fall back to HOLD
        assert llm_response.signal == Hold

        # Convert to TradeAction
        trade_action = TypeConverter.llm_response_to_trade_action(
            llm_response, market_state
        )

        # Should result in safe HOLD action
        assert trade_action.action == "HOLD"
        assert trade_action.size_pct == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
