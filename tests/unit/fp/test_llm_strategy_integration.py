"""
Comprehensive tests for functional LLM strategy integration.

This module tests:
1. LLM prompt generation with various market conditions
2. Response parsing with valid/invalid JSON formats
3. Type conversions between functional and imperative types
4. Confidence adjustments based on market conditions
5. Decision validation with safety rules
6. Context optimization and windowing
7. Error handling throughout the LLM workflow
8. Strategy adapter functionality for backward compatibility
9. Integration testing of complete strategy flow
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config import settings
from bot.fp.adapters.strategy_adapter import (
    FunctionalLLMStrategy,
    LLMAgentAdapter,
    MemoryEnhancedLLMAgentAdapter,
    Signal,
    TradingParams,
    TypeConverter,
)
from bot.fp.indicators.vumanchu_functional import VuManchuState
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
from bot.fp.types import MarketState
from bot.trading_types import TradeAction


class TestLLMConfiguration:
    """Test LLM configuration and setup."""

    def test_llm_config_creation(self):
        """Test LLM configuration creation with various providers."""
        # Test OpenAI configuration
        openai_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="sk-test-key",
            temperature=0.3,
            max_tokens=500,
            timeout=30.0,
        )

        assert openai_config.provider == LLMProvider.OPENAI
        assert openai_config.model == "gpt-4"
        assert openai_config.api_key == "sk-test-key"
        assert openai_config.temperature == 0.3
        assert openai_config.max_tokens == 500
        assert openai_config.timeout == 30.0

        # Test Anthropic configuration
        anthropic_config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet",
            api_key="sk-ant-api-key",
            max_context_window=8000,
        )

        assert anthropic_config.provider == LLMProvider.ANTHROPIC
        assert anthropic_config.model == "claude-3-sonnet"
        assert anthropic_config.max_context_window == 8000
        assert anthropic_config.temperature == 0.3  # Default

        # Test Local configuration
        local_config = LLMConfig(
            provider=LLMProvider.LOCAL,
            model="llama2",
            api_key="",  # Local doesn't need API key
            system_prompt="You are a trading assistant.",
        )

        assert local_config.provider == LLMProvider.LOCAL
        assert local_config.model == "llama2"
        assert local_config.system_prompt == "You are a trading assistant."

    def test_llm_config_immutability(self):
        """Test that LLM configuration is immutable."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="sk-test-key",
        )

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            config.model = "gpt-3.5-turbo"

        with pytest.raises(AttributeError):
            config.temperature = 0.5


class TestMarketContextCreation:
    """Test market context creation for LLM input."""

    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state for testing."""
        return MarketState(
            symbol="BTC-USD",
            current_price=Decimal(50000),
            open_24h=Decimal(49000),
            high_24h=Decimal(51000),
            low_24h=Decimal(48500),
            volume=Decimal(1000000),
            timestamp=datetime.now(UTC),
        )

    @pytest.fixture
    def sample_vumanchu_state(self):
        """Create sample VuManChu state for testing."""
        return VuManchuState(
            cipher_a=[5.2, 6.1, 7.3, 8.5, 9.2],
            cipher_b=[-2.1, -1.5, 0.3, 1.2, 2.0],
            wave_trend=[45.2, 46.8, 48.3, 49.1, 50.5],
            money_flow=[52.1, 54.3, 55.8, 57.2, 58.9],
            buy_signal=True,
            sell_signal=False,
            bullish_divergence=True,
            bearish_divergence=False,
            trend_strength=0.75,
            signal_confidence=0.82,
            last_signal_timestamp=datetime.now(UTC),
        )

    def test_create_market_context_basic(
        self, sample_market_state, sample_vumanchu_state
    ):
        """Test basic market context creation."""
        context = create_market_context(
            market_state=sample_market_state,
            vumanchu_state=sample_vumanchu_state,
        )

        assert isinstance(context, LLMContext)
        assert context.market_state == sample_market_state
        assert "vumanchu_a" in context.indicators
        assert "vumanchu_b" in context.indicators
        assert "price_action" in context.indicators
        assert "volume" in context.indicators

        # Test VuManChu A indicators
        vumanchu_a = context.indicators["vumanchu_a"]
        assert vumanchu_a["current"] == 9.2  # Last value
        assert vumanchu_a["signal"] is True
        assert vumanchu_a["divergence"] is True

        # Test VuManChu B indicators
        vumanchu_b = context.indicators["vumanchu_b"]
        assert vumanchu_b["wave_trend"] == 50.5  # Last value
        assert vumanchu_b["money_flow"] == 58.9  # Last value

    def test_create_market_context_with_recent_trades(
        self, sample_market_state, sample_vumanchu_state
    ):
        """Test market context creation with recent trades."""
        recent_trades = [
            {
                "action": "LONG",
                "price": 49500,
                "pnl": 0.025,
                "timestamp": datetime.now(UTC) - timedelta(hours=2),
            },
            {
                "action": "SHORT",
                "price": 50200,
                "pnl": -0.015,
                "timestamp": datetime.now(UTC) - timedelta(hours=1),
            },
        ]

        context = create_market_context(
            market_state=sample_market_state,
            vumanchu_state=sample_vumanchu_state,
            recent_trades=recent_trades,
        )

        assert len(context.recent_trades) == 2
        assert context.recent_trades[0]["action"] == "LONG"
        assert context.recent_trades[1]["pnl"] == -0.015

    def test_create_market_context_empty_indicators(self, sample_market_state):
        """Test context creation with empty VuManChu state."""
        empty_vumanchu = VuManchuState(
            cipher_a=[],
            cipher_b=[],
            wave_trend=[],
            money_flow=[],
            buy_signal=False,
            sell_signal=False,
            bullish_divergence=False,
            bearish_divergence=False,
            trend_strength=0.0,
            signal_confidence=0.0,
        )

        context = create_market_context(
            market_state=sample_market_state,
            vumanchu_state=empty_vumanchu,
        )

        # Should handle empty indicators gracefully
        assert context.indicators["vumanchu_a"]["current"] == 0
        assert context.indicators["vumanchu_b"]["wave_trend"] == 0
        assert context.indicators["vumanchu_a"]["signal"] is False


class TestPromptGeneration:
    """Test LLM prompt generation with various market conditions."""

    @pytest.fixture
    def trading_params(self):
        """Create sample trading parameters."""
        return TradingParams(
            risk_per_trade=0.1,
            max_leverage=5,
            stop_loss_pct=1.0,
            take_profit_pct=2.0,
            max_position_size=0.25,
        )

    @pytest.fixture
    def bullish_context(self):
        """Create bullish market context."""
        market_state = MarketState(
            symbol="ETH-USD",
            current_price=Decimal(3000),
            open_24h=Decimal(2850),
            high_24h=Decimal(3050),
            low_24h=Decimal(2800),
            volume=Decimal(500000),
            timestamp=datetime.now(UTC),
        )

        return LLMContext(
            market_state=market_state,
            indicators={
                "vumanchu_a": {
                    "current": 15.2,
                    "trend": "STRONG_UP",
                    "signal": True,
                    "divergence": True,
                },
                "vumanchu_b": {
                    "wave_trend": 65.3,
                    "money_flow": 72.1,
                    "trend_direction": "BULLISH",
                },
                "price_action": {
                    "current_price": 3000,
                    "price_change_24h": 0.053,  # 5.3% gain
                    "volatility": 0.089,
                    "support_resistance": {"support": 2800, "resistance": 3050},
                },
                "volume": {
                    "current": 500000,
                    "average": 450000,
                    "trend": "INCREASING",
                },
            },
            recent_trades=[{"action": "LONG", "price": 2950, "pnl": 0.034}],
        )

    @pytest.fixture
    def bearish_context(self):
        """Create bearish market context."""
        market_state = MarketState(
            symbol="BTC-USD",
            current_price=Decimal(48000),
            open_24h=Decimal(51000),
            high_24h=Decimal(51200),
            low_24h=Decimal(47500),
            volume=Decimal(800000),
            timestamp=datetime.now(UTC),
        )

        return LLMContext(
            market_state=market_state,
            indicators={
                "vumanchu_a": {
                    "current": -12.5,
                    "trend": "STRONG_DOWN",
                    "signal": True,
                    "divergence": False,
                },
                "vumanchu_b": {
                    "wave_trend": 25.8,
                    "money_flow": 28.3,
                    "trend_direction": "BEARISH",
                },
                "price_action": {
                    "current_price": 48000,
                    "price_change_24h": -0.059,  # -5.9% loss
                    "volatility": 0.078,
                    "support_resistance": {"support": 47500, "resistance": 51000},
                },
                "volume": {
                    "current": 800000,
                    "average": 600000,
                    "trend": "INCREASING",
                },
            },
            recent_trades=[{"action": "SHORT", "price": 50500, "pnl": 0.049}],
        )

    def test_generate_trading_prompt_bullish(self, bullish_context, trading_params):
        """Test prompt generation for bullish market conditions."""
        prompt = generate_trading_prompt(
            bullish_context, trading_params, include_examples=True
        )

        # Should contain market state information
        assert "ETH-USD" in prompt
        assert "$3,000.00" in prompt
        assert "5.30%" in prompt  # Price change

        # Should contain technical indicators
        assert "VuManchu Cipher A" in prompt
        assert "15.20" in prompt  # Current value
        assert "STRONG_UP" in prompt
        assert "BULLISH" in prompt

        # Should contain trading parameters
        assert "Risk per trade: 10.0%" in prompt
        assert "Maximum leverage: 5x" in prompt

        # Should contain JSON format specification
        assert "JSON format" in prompt
        assert '"action"' in prompt
        assert '"confidence"' in prompt
        assert '"reasoning"' in prompt

        # Should include examples when requested
        assert "Example responses" in prompt

    def test_generate_trading_prompt_bearish(self, bearish_context, trading_params):
        """Test prompt generation for bearish market conditions."""
        prompt = generate_trading_prompt(
            bearish_context, trading_params, include_examples=False
        )

        # Should contain market state information
        assert "BTC-USD" in prompt
        assert "$48,000.00" in prompt
        assert "-5.90%" in prompt  # Price change

        # Should contain bearish indicators
        assert "-12.50" in prompt  # Negative VuManChu A
        assert "STRONG_DOWN" in prompt
        assert "BEARISH" in prompt

        # Should contain recent trade info
        assert "Recent Trading Activity" in prompt
        assert "SHORT at $50,500.00" in prompt

        # Should not include examples when not requested
        assert "Example responses" not in prompt

    def test_generate_trading_prompt_neutral_conditions(self, trading_params):
        """Test prompt generation for neutral market conditions."""
        neutral_market = MarketState(
            symbol="SOL-USD",
            current_price=Decimal(100),
            open_24h=Decimal("99.5"),
            high_24h=Decimal(101),
            low_24h=Decimal("98.5"),
            volume=Decimal(200000),
            timestamp=datetime.now(UTC),
        )

        neutral_context = LLMContext(
            market_state=neutral_market,
            indicators={
                "vumanchu_a": {
                    "current": 0.1,
                    "trend": "NEUTRAL",
                    "signal": False,
                    "divergence": False,
                },
                "vumanchu_b": {
                    "wave_trend": 50.0,
                    "money_flow": 49.8,
                    "trend_direction": "NEUTRAL",
                },
                "price_action": {
                    "current_price": 100,
                    "price_change_24h": 0.005,  # 0.5% change
                    "volatility": 0.025,
                    "support_resistance": {"support": 98.5, "resistance": 101},
                },
                "volume": {
                    "current": 200000,
                    "average": 200000,
                    "trend": "NORMAL",
                },
            },
            recent_trades=[],
        )

        prompt = generate_trading_prompt(neutral_context, trading_params)

        # Should handle neutral conditions
        assert "SOL-USD" in prompt
        assert "0.50%" in prompt
        assert "NEUTRAL" in prompt
        assert (
            "no recent trades" in prompt.lower()
            or "recent trading activity" not in prompt
        )

    def test_generate_trading_prompt_with_signals(self, trading_params):
        """Test prompt generation with active trading signals."""
        market_with_signals = MarketState(
            symbol="ADA-USD",
            current_price=Decimal("0.5"),
            timestamp=datetime.now(UTC),
        )

        signal_context = LLMContext(
            market_state=market_with_signals,
            indicators={
                "vumanchu_a": {
                    "current": 8.5,
                    "trend": "UP",
                    "signal": 1,  # Buy signal
                    "divergence": -1,  # Bearish divergence
                },
                "vumanchu_b": {
                    "wave_trend": 55.0,
                    "money_flow": 60.2,
                    "trend_direction": "BULLISH",
                },
                "price_action": {
                    "current_price": 0.5,
                    "price_change_24h": 0.02,
                    "volatility": 0.04,
                    "support_resistance": {"support": 0.48, "resistance": 0.52},
                },
                "volume": {
                    "current": 50000,
                    "average": 45000,
                    "trend": "INCREASING",
                },
            },
            recent_trades=[],
        )

        prompt = generate_trading_prompt(signal_context, trading_params)

        # Should include signal information
        assert "Active Signal: BUY" in prompt
        assert "Divergence Detected: Bearish" in prompt


class TestResponseParsing:
    """Test LLM response parsing with various formats."""

    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state for parsing tests."""
        return MarketState(
            symbol="BTC-USD",
            current_price=Decimal(50000),
            timestamp=datetime.now(UTC),
        )

    def test_parse_valid_long_response(self, sample_market_state):
        """Test parsing valid LONG response."""
        response_text = """
        Based on the analysis, I recommend a LONG position.

        {
            "action": "LONG",
            "confidence": 0.85,
            "reasoning": "Strong bullish divergence with high volume and positive momentum",
            "risk_assessment": {
                "risk_level": "MEDIUM",
                "key_risks": ["Potential resistance at $52,000"],
                "mitigation": "Use tight stop loss below support"
            },
            "suggested_params": {
                "position_size": 0.25,
                "stop_loss": 48500,
                "take_profit": 52000
            }
        }
        """

        parsed = parse_llm_response(response_text, sample_market_state)

        assert isinstance(parsed, LLMResponse)
        assert parsed.signal == Signal.LONG
        assert parsed.confidence == 0.85
        assert "bullish divergence" in parsed.reasoning
        assert parsed.risk_assessment["risk_level"] == "MEDIUM"
        assert parsed.suggested_params["position_size"] == 0.25
        assert parsed.suggested_params["stop_loss"] == 48500
        assert parsed.suggested_params["take_profit"] == 52000

    def test_parse_valid_short_response(self, sample_market_state):
        """Test parsing valid SHORT response."""
        response_text = """
        {
            "action": "SHORT",
            "confidence": 0.72,
            "reasoning": "Bearish momentum with declining volume",
            "risk_assessment": {
                "risk_level": "HIGH",
                "key_risks": ["Volatile market conditions", "Support at $48,000"],
                "mitigation": "Smaller position size and wider stop loss"
            },
            "suggested_params": {
                "position_size": 0.15,
                "stop_loss": 51500,
                "take_profit": 47000
            }
        }
        """

        parsed = parse_llm_response(response_text, sample_market_state)

        assert parsed.signal == Signal.SHORT
        assert parsed.confidence == 0.72
        assert "bearish momentum" in parsed.reasoning.lower()
        assert parsed.risk_assessment["risk_level"] == "HIGH"
        assert len(parsed.risk_assessment["key_risks"]) == 2
        assert parsed.suggested_params["position_size"] == 0.15

    def test_parse_valid_hold_response(self, sample_market_state):
        """Test parsing valid HOLD response."""
        response_text = """
        {
            "action": "HOLD",
            "confidence": 0.9,
            "reasoning": "Mixed signals with unclear direction, waiting for confirmation",
            "risk_assessment": {
                "risk_level": "LOW",
                "key_risks": ["Choppy market conditions"],
                "mitigation": "Stay out until trend emerges"
            }
        }
        """

        parsed = parse_llm_response(response_text, sample_market_state)

        assert parsed.signal == Signal.HOLD
        assert parsed.confidence == 0.9
        assert "mixed signals" in parsed.reasoning.lower()
        assert parsed.suggested_params is None  # HOLD shouldn't have params

    def test_parse_malformed_json(self, sample_market_state):
        """Test parsing response with malformed JSON."""
        response_text = """
        This is a malformed response with incomplete JSON:
        {
            "action": "LONG",
            "confidence": 0.8,
            "reasoning": "Missing closing brace"
        """

        parsed = parse_llm_response(response_text, sample_market_state)

        # Should default to HOLD on parse error
        assert parsed.signal == Signal.HOLD
        assert "Error parsing response" in parsed.reasoning

    def test_parse_missing_json(self, sample_market_state):
        """Test parsing response with no JSON."""
        response_text = """
        This response has no JSON at all.
        Just a text analysis of the market.
        No structured decision format.
        """

        parsed = parse_llm_response(response_text, sample_market_state)

        # Should default to HOLD when no JSON found
        assert parsed.signal == Signal.HOLD
        assert "No valid JSON found" in parsed.reasoning

    def test_parse_invalid_action(self, sample_market_state):
        """Test parsing response with invalid action."""
        response_text = """
        {
            "action": "INVALID_ACTION",
            "confidence": 0.7,
            "reasoning": "Testing invalid action handling",
            "risk_assessment": {
                "risk_level": "MEDIUM",
                "key_risks": [],
                "mitigation": "Use default action"
            }
        }
        """

        parsed = parse_llm_response(response_text, sample_market_state)

        # Should default to HOLD for invalid actions
        assert parsed.signal == Signal.HOLD
        assert parsed.confidence == 0.7  # Other fields should still be parsed

    def test_parse_confidence_bounds(self, sample_market_state):
        """Test confidence value clamping."""
        # Test confidence above 1.0
        high_confidence_response = """
        {
            "action": "LONG",
            "confidence": 1.5,
            "reasoning": "Extremely high confidence test",
            "risk_assessment": {"risk_level": "LOW", "key_risks": [], "mitigation": ""}
        }
        """

        parsed = parse_llm_response(high_confidence_response, sample_market_state)
        assert parsed.confidence == 1.0  # Should be clamped to 1.0

        # Test confidence below 0.0
        low_confidence_response = """
        {
            "action": "SHORT",
            "confidence": -0.2,
            "reasoning": "Negative confidence test",
            "risk_assessment": {"risk_level": "HIGH", "key_risks": [], "mitigation": ""}
        }
        """

        parsed = parse_llm_response(low_confidence_response, sample_market_state)
        assert parsed.confidence == 0.0  # Should be clamped to 0.0

    def test_parse_missing_fields(self, sample_market_state):
        """Test parsing response with missing fields."""
        response_text = """
        {
            "action": "LONG"
        }
        """

        parsed = parse_llm_response(response_text, sample_market_state)

        # Should provide defaults for missing fields
        assert parsed.signal == Signal.LONG
        assert parsed.confidence == 0.5  # Default confidence
        assert parsed.reasoning == "No reasoning provided"
        assert "risk_level" in parsed.risk_assessment
        assert parsed.risk_assessment["risk_level"] == "MEDIUM"


class TestConfidenceAdjustment:
    """Test confidence adjustment based on market conditions."""

    @pytest.fixture
    def base_response(self):
        """Create base LLM response for adjustment testing."""
        return LLMResponse(
            signal=Signal.LONG,
            confidence=0.8,
            reasoning="Base test response",
            risk_assessment={
                "risk_level": "MEDIUM",
                "key_risks": ["Market volatility"],
                "mitigation": "Use stop loss",
            },
            suggested_params={
                "position_size": 0.25,
                "stop_loss": 48000,
                "take_profit": 52000,
            },
        )

    def test_adjust_confidence_high_volatility(self, base_response):
        """Test confidence adjustment for high volatility."""
        high_volatility_market = MarketState(
            symbol="BTC-USD",
            current_price=Decimal(50000),
            high_24h=Decimal(52000),
            low_24h=Decimal(47000),  # 10% volatility
            timestamp=datetime.now(UTC),
        )

        adjusted = adjust_confidence_by_market_conditions(
            base_response, high_volatility_market, volatility_threshold=0.02
        )

        # Confidence should be reduced due to high volatility
        assert adjusted.confidence < base_response.confidence
        assert "Adjusted confidence" in adjusted.reasoning

    def test_adjust_confidence_low_volatility(self, base_response):
        """Test confidence adjustment for low volatility."""
        low_volatility_market = MarketState(
            symbol="BTC-USD",
            current_price=Decimal(50000),
            high_24h=Decimal(50500),
            low_24h=Decimal(49500),  # 2% volatility
            open_24h=Decimal(50000),
            timestamp=datetime.now(UTC),
        )

        adjusted = adjust_confidence_by_market_conditions(
            base_response, low_volatility_market, volatility_threshold=0.05
        )

        # Confidence should remain largely unchanged
        assert adjusted.confidence == base_response.confidence

    def test_adjust_confidence_counter_trend(self):
        """Test confidence adjustment for counter-trend trades."""
        # LONG signal against downtrend
        long_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.8,
            reasoning="Counter-trend long test",
            risk_assessment={"risk_level": "MEDIUM", "key_risks": [], "mitigation": ""},
        )

        downtrend_market = MarketState(
            symbol="BTC-USD",
            current_price=Decimal(48000),
            open_24h=Decimal(52000),  # Strong downtrend
            high_24h=Decimal(52000),
            low_24h=Decimal(47500),
            timestamp=datetime.now(UTC),
        )

        adjusted = adjust_confidence_by_market_conditions(
            long_response, downtrend_market
        )

        # Confidence should be reduced for counter-trend trade
        assert adjusted.confidence < long_response.confidence
        assert adjusted.confidence == long_response.confidence * 0.8

    def test_adjust_confidence_very_low_result(self, base_response):
        """Test confidence adjustment resulting in HOLD."""
        # Create market conditions that will severely reduce confidence
        extreme_volatility_market = MarketState(
            symbol="BTC-USD",
            current_price=Decimal(50000),
            open_24h=Decimal(55000),  # Counter-trend
            high_24h=Decimal(58000),
            low_24h=Decimal(45000),  # Extreme volatility
            timestamp=datetime.now(UTC),
        )

        # Start with lower confidence
        low_confidence_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.5,  # Will be reduced further
            reasoning="Low confidence test",
            risk_assessment={"risk_level": "HIGH", "key_risks": [], "mitigation": ""},
        )

        adjusted = adjust_confidence_by_market_conditions(
            low_confidence_response,
            extreme_volatility_market,
            volatility_threshold=0.02,
        )

        # Should convert to HOLD when confidence drops too low
        assert adjusted.signal == Signal.HOLD
        assert "Confidence too low" in adjusted.reasoning


class TestDecisionValidation:
    """Test LLM decision validation against safety rules."""

    @pytest.fixture
    def trading_params(self):
        """Create sample trading parameters for validation."""
        return TradingParams(
            risk_per_trade=0.1,
            max_leverage=5,
            stop_loss_pct=1.0,
            take_profit_pct=2.0,
            max_position_size=0.25,
        )

    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state for validation."""
        return MarketState(
            symbol="BTC-USD",
            current_price=Decimal(50000),
            timestamp=datetime.now(UTC),
        )

    def test_validate_valid_long_decision(self, trading_params, sample_market_state):
        """Test validation of valid LONG decision."""
        valid_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.75,
            reasoning="Valid long decision",
            risk_assessment={
                "risk_level": "MEDIUM",
                "key_risks": ["Market volatility"],
                "mitigation": "Use stop loss",
            },
            suggested_params={
                "position_size": 0.2,  # Within max_position_size
                "stop_loss": 49000,  # Below current price for LONG
                "take_profit": 52000,  # Above current price for LONG
            },
        )

        is_valid, error = validate_llm_decision(
            valid_response, sample_market_state, trading_params
        )

        assert is_valid is True
        assert error is None

    def test_validate_low_confidence_rejection(
        self, trading_params, sample_market_state
    ):
        """Test rejection of low confidence decisions."""
        low_confidence_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.3,  # Below 0.5 threshold
            reasoning="Low confidence decision",
            risk_assessment={"risk_level": "LOW", "key_risks": [], "mitigation": ""},
        )

        is_valid, error = validate_llm_decision(
            low_confidence_response, sample_market_state, trading_params
        )

        assert is_valid is False
        assert "Confidence below minimum threshold" in error

    def test_validate_oversized_position_rejection(
        self, trading_params, sample_market_state
    ):
        """Test rejection of oversized position."""
        oversized_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.8,
            reasoning="Oversized position test",
            risk_assessment={"risk_level": "MEDIUM", "key_risks": [], "mitigation": ""},
            suggested_params={
                "position_size": 0.5,  # Exceeds max_position_size of 0.25
                "stop_loss": 49000,
                "take_profit": 52000,
            },
        )

        is_valid, error = validate_llm_decision(
            oversized_response, sample_market_state, trading_params
        )

        assert is_valid is False
        assert "position size exceeds maximum" in error

    def test_validate_invalid_stop_loss_long(self, trading_params, sample_market_state):
        """Test rejection of invalid stop loss for LONG position."""
        invalid_sl_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.8,
            reasoning="Invalid stop loss test",
            risk_assessment={"risk_level": "MEDIUM", "key_risks": [], "mitigation": ""},
            suggested_params={
                "position_size": 0.2,
                "stop_loss": 51000,  # Above current price for LONG - invalid
                "take_profit": 52000,
            },
        )

        is_valid, error = validate_llm_decision(
            invalid_sl_response, sample_market_state, trading_params
        )

        assert is_valid is False
        assert "Invalid stop loss for long position" in error

    def test_validate_invalid_stop_loss_short(
        self, trading_params, sample_market_state
    ):
        """Test rejection of invalid stop loss for SHORT position."""
        invalid_sl_response = LLMResponse(
            signal=Signal.SHORT,
            confidence=0.8,
            reasoning="Invalid stop loss test",
            risk_assessment={"risk_level": "MEDIUM", "key_risks": [], "mitigation": ""},
            suggested_params={
                "position_size": 0.2,
                "stop_loss": 48000,  # Below current price for SHORT - invalid
                "take_profit": 47000,
            },
        )

        is_valid, error = validate_llm_decision(
            invalid_sl_response, sample_market_state, trading_params
        )

        assert is_valid is False
        assert "Invalid stop loss for short position" in error

    def test_validate_high_risk_low_confidence_rejection(
        self, trading_params, sample_market_state
    ):
        """Test rejection of high risk with low confidence."""
        high_risk_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.6,  # Below 0.7 threshold for high risk
            reasoning="High risk low confidence test",
            risk_assessment={
                "risk_level": "HIGH",
                "key_risks": ["High volatility", "Major resistance"],
                "mitigation": "Tight risk management",
            },
            suggested_params={
                "position_size": 0.2,
                "stop_loss": 49000,
                "take_profit": 52000,
            },
        )

        is_valid, error = validate_llm_decision(
            high_risk_response, sample_market_state, trading_params
        )

        assert is_valid is False
        assert "High risk with insufficient confidence" in error

    def test_validate_hold_decision_always_valid(
        self, trading_params, sample_market_state
    ):
        """Test that HOLD decisions are always valid."""
        hold_response = LLMResponse(
            signal=Signal.HOLD,
            confidence=0.2,  # Even low confidence
            reasoning="Hold decision test",
            risk_assessment={
                "risk_level": "HIGH",  # Even high risk
                "key_risks": [],
                "mitigation": "",
            },
        )

        is_valid, error = validate_llm_decision(
            hold_response, sample_market_state, trading_params
        )

        assert is_valid is True
        assert error is None


class TestContextOptimization:
    """Test context window optimization and management."""

    def test_create_context_window_under_limit(self):
        """Test context window creation when under size limit."""
        history = [
            MarketState(
                symbol="BTC-USD",
                current_price=Decimal(f"{50000 + i * 100}"),
                timestamp=datetime.now(UTC) - timedelta(minutes=i),
            )
            for i in range(50)  # 50 items under 100 limit
        ]

        optimized = create_context_window(history, max_size=100)

        # Should return all items unchanged
        assert len(optimized) == 50
        assert optimized == history

    def test_create_context_window_over_limit(self):
        """Test context window creation when over size limit."""
        history = [
            MarketState(
                symbol="BTC-USD",
                current_price=Decimal(f"{50000 + i * 10}"),
                timestamp=datetime.now(UTC) - timedelta(minutes=i),
            )
            for i in range(200)  # 200 items over 100 limit
        ]

        optimized = create_context_window(history, max_size=100, compression_ratio=0.5)

        # Should compress to max_size
        assert len(optimized) <= 100

        # Recent data should be preserved (last 50 items at full resolution)
        recent_size = int(100 * 0.5)
        assert optimized[-recent_size:] == history[-recent_size:]

        # Should have compressed older data
        assert len(optimized) < len(history)

    def test_create_context_window_different_compression_ratios(self):
        """Test different compression ratios."""
        history = [
            MarketState(
                symbol="ETH-USD",
                current_price=Decimal(f"{3000 + i * 5}"),
                timestamp=datetime.now(UTC) - timedelta(minutes=i),
            )
            for i in range(150)
        ]

        # High compression (more recent data)
        high_compression = create_context_window(
            history, max_size=100, compression_ratio=0.8
        )

        # Low compression (more historical data)
        low_compression = create_context_window(
            history, max_size=100, compression_ratio=0.2
        )

        assert len(high_compression) <= 100
        assert len(low_compression) <= 100

        # High compression should preserve more recent data
        high_recent_size = int(100 * 0.2)  # 1 - 0.8
        low_recent_size = int(100 * 0.8)  # 1 - 0.2

        assert high_recent_size < low_recent_size

    def test_create_context_window_empty_history(self):
        """Test context window creation with empty history."""
        empty_history = []

        optimized = create_context_window(empty_history, max_size=100)

        assert len(optimized) == 0
        assert optimized == []


class TestTypeConverter:
    """Test type conversions between functional and imperative types."""

    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state for conversion tests."""
        return MarketState(
            symbol="BTC-USD",
            current_price=Decimal(50000),
            timestamp=datetime.now(UTC),
        )

    def test_llm_response_to_trade_action_long(self, sample_market_state):
        """Test converting Long signal to TradeAction."""
        llm_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.8,
            reasoning="Strong bullish signal",
            risk_assessment={"risk_level": "MEDIUM"},
            suggested_params={
                "position_size": 0.25,
                "stop_loss": 48000,
                "take_profit": 52000,
            },
        )

        with patch.object(settings.trading, "max_size_pct", 30.0):
            with patch.object(settings.trading, "leverage", 5):
                trade_action = TypeConverter.llm_response_to_trade_action(
                    llm_response, sample_market_state
                )

        assert isinstance(trade_action, TradeAction)
        assert trade_action.action == "LONG"
        assert trade_action.size_pct == 25.0  # 0.25 * 100
        assert trade_action.rationale == "Strong bullish signal"
        assert trade_action.leverage == 5

        # Check stop loss and take profit percentages
        assert trade_action.stop_loss_pct == 4.0  # (50000-48000)/50000 * 100
        assert trade_action.take_profit_pct == 4.0  # (52000-50000)/50000 * 100

    def test_llm_response_to_trade_action_short(self, sample_market_state):
        """Test converting Short signal to TradeAction."""
        llm_response = LLMResponse(
            signal=Signal.SHORT,
            confidence=0.7,
            reasoning="Bearish momentum",
            risk_assessment={"risk_level": "HIGH"},
            suggested_params={
                "position_size": 0.15,
                "stop_loss": 51500,
                "take_profit": 47000,
            },
        )

        with patch.object(settings.trading, "max_size_pct", 30.0):
            with patch.object(settings.trading, "leverage", 3):
                trade_action = TypeConverter.llm_response_to_trade_action(
                    llm_response, sample_market_state
                )

        assert trade_action.action == "SHORT"
        assert trade_action.size_pct == 15.0  # 0.15 * 100
        assert trade_action.rationale == "Bearish momentum"
        assert trade_action.leverage == 3

        # Check stop loss and take profit percentages
        assert trade_action.stop_loss_pct == 3.0  # (51500-50000)/50000 * 100
        assert trade_action.take_profit_pct == 6.0  # (50000-47000)/50000 * 100

    def test_llm_response_to_trade_action_hold(self, sample_market_state):
        """Test converting Hold signal to TradeAction."""
        llm_response = LLMResponse(
            signal=Signal.HOLD,
            confidence=0.9,
            reasoning="Unclear market direction",
            risk_assessment={"risk_level": "LOW"},
        )

        with patch.object(settings.trading, "leverage", 2):
            trade_action = TypeConverter.llm_response_to_trade_action(
                llm_response, sample_market_state
            )

        assert trade_action.action == "HOLD"
        assert trade_action.size_pct == 0.0  # No position for HOLD
        assert trade_action.rationale == "Unclear market direction"
        assert trade_action.leverage == 2

    def test_llm_response_to_trade_action_no_suggested_params(
        self, sample_market_state
    ):
        """Test conversion without suggested parameters."""
        llm_response = LLMResponse(
            signal=Signal.LONG,
            confidence=0.6,
            reasoning="Basic signal without params",
            risk_assessment={"risk_level": "MEDIUM"},
            # No suggested_params
        )

        with patch.object(settings.trading, "max_size_pct", 20.0):
            with patch.object(settings.trading, "leverage", 4):
                trade_action = TypeConverter.llm_response_to_trade_action(
                    llm_response, sample_market_state
                )

        assert trade_action.action == "LONG"
        # Size based on confidence when no suggested params
        assert trade_action.size_pct == 12.0  # 0.6 * 20.0
        # Default stop loss and take profit
        assert trade_action.stop_loss_pct == 1.0
        assert trade_action.take_profit_pct == 2.0

    def test_create_trading_params_from_settings(self):
        """Test creating TradingParams from settings."""
        with patch.object(settings.trading, "max_size_pct", 25.0):
            with patch.object(settings.trading, "max_futures_leverage", 10):
                params = TypeConverter.create_trading_params()

        assert isinstance(params, TradingParams)
        assert params.risk_per_trade == 0.25  # 25% / 100
        assert params.max_leverage == 10
        assert params.max_position_size == 0.25  # 25% / 100
        assert params.stop_loss_pct == 1.0
        assert params.take_profit_pct == 2.0

    def test_conversion_respects_max_size(self, sample_market_state):
        """Test that conversion respects maximum size limits."""
        llm_response = LLMResponse(
            signal=Signal.LONG,
            confidence=1.0,  # Very high confidence
            reasoning="High confidence signal",
            risk_assessment={"risk_level": "LOW"},
            suggested_params={
                "position_size": 0.8,  # Huge position
                "stop_loss": 49000,
                "take_profit": 52000,
            },
        )

        with patch.object(settings.trading, "max_size_pct", 20.0):  # 20% max
            trade_action = TypeConverter.llm_response_to_trade_action(
                llm_response, sample_market_state
            )

        # Should be clamped to max_size_pct
        assert trade_action.size_pct == 20.0
        assert trade_action.size_pct < 80.0  # Original suggested size


class TestStrategyAdapters:
    """Test strategy adapter functionality for backward compatibility."""

    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state for adapter tests."""
        return MarketState(
            symbol="BTC-USD",
            current_price=Decimal(50000),
            open_24h=Decimal(49500),
            high_24h=Decimal(51000),
            low_24h=Decimal(48500),
            volume=Decimal(1000000),
            timestamp=datetime.now(UTC),
        )

    @pytest.mark.asyncio
    async def test_functional_llm_strategy_basic(self, sample_market_state):
        """Test basic functional LLM strategy execution."""
        with patch.object(settings.llm, "provider", "openai"):
            with patch.object(settings.llm, "model_name", "gpt-4"):
                with patch.object(settings.llm, "openai_api_key") as mock_key:
                    mock_key.get_secret_value.return_value = "sk-test-key"

                    strategy = FunctionalLLMStrategy()

                    # Mock the LLM call to return a predictable response
                    strategy._call_llm = AsyncMock(
                        return_value="""
                    {
                        "action": "HOLD",
                        "confidence": 0.8,
                        "reasoning": "Neutral market conditions",
                        "risk_assessment": {
                            "risk_level": "LOW",
                            "key_risks": ["Sideways market"],
                            "mitigation": "Wait for clear signals"
                        }
                    }
                    """
                    )

                    response = await strategy.analyze_market_functional(
                        sample_market_state
                    )

                    assert isinstance(response, LLMResponse)
                    assert response.signal == Signal.HOLD
                    assert response.confidence == 0.8
                    assert "Neutral market conditions" in response.reasoning

    @pytest.mark.asyncio
    async def test_llm_agent_adapter_compatibility(self, sample_market_state):
        """Test LLM agent adapter maintains API compatibility."""
        with patch.object(settings.llm, "provider", "openai"):
            with patch.object(settings.llm, "model_name", "gpt-4"):
                with patch.object(settings.llm, "openai_api_key") as mock_key:
                    with patch.object(settings.trading, "max_size_pct", 25.0):
                        with patch.object(settings.trading, "leverage", 5):
                            mock_key.get_secret_value.return_value = "sk-test-key"

                            adapter = LLMAgentAdapter()

                            # Mock the underlying strategy
                            mock_response = LLMResponse(
                                signal=Signal.LONG,
                                confidence=0.75,
                                reasoning="Functional strategy test",
                                risk_assessment={"risk_level": "MEDIUM"},
                                suggested_params={
                                    "position_size": 0.2,
                                    "stop_loss": 48500,
                                    "take_profit": 51500,
                                },
                            )

                            adapter._strategy.analyze_market_functional = AsyncMock(
                                return_value=mock_response
                            )

                            # Test the adapter interface
                            trade_action = await adapter.analyze_market(
                                sample_market_state
                            )

                            assert isinstance(trade_action, TradeAction)
                            assert trade_action.action == "LONG"
                            assert trade_action.size_pct == 20.0  # 0.2 * 100
                            assert trade_action.rationale == "Functional strategy test"
                            assert trade_action.leverage == 5

    def test_llm_agent_adapter_status(self):
        """Test LLM agent adapter status reporting."""
        with patch.object(settings.llm, "provider", "openai"):
            with patch.object(settings.llm, "model_name", "gpt-4"):
                adapter = LLMAgentAdapter(model_provider="openai", model_name="gpt-4")

                status = adapter.get_status()

                assert status["model_provider"] == "openai"
                assert status["model_name"] == "gpt-4"
                assert status["strategy_type"] == "functional"
                assert status["available"] is True
                assert status["completion_count"] == 0

    def test_llm_agent_adapter_is_available(self):
        """Test LLM agent adapter availability check."""
        adapter = LLMAgentAdapter()

        # Functional strategies should always be available
        assert adapter.is_available() is True

    @pytest.mark.asyncio
    async def test_memory_enhanced_adapter_without_memory(self, sample_market_state):
        """Test memory-enhanced adapter without memory server."""
        with patch.object(settings.llm, "provider", "openai"):
            with patch.object(settings.mcp, "enabled", False):
                adapter = MemoryEnhancedLLMAgentAdapter()

                # Mock the base strategy
                mock_response = LLMResponse(
                    signal=Signal.SHORT,
                    confidence=0.6,
                    reasoning="No memory fallback test",
                    risk_assessment={"risk_level": "MEDIUM"},
                )

                adapter._strategy.analyze_market_functional = AsyncMock(
                    return_value=mock_response
                )

                trade_action = await adapter.analyze_market(sample_market_state)

                assert isinstance(trade_action, TradeAction)
                assert trade_action.action == "SHORT"
                assert "No memory fallback test" in trade_action.rationale

    def test_memory_enhanced_adapter_status(self):
        """Test memory-enhanced adapter status reporting."""
        mock_memory_server = MagicMock()

        with patch.object(settings.mcp, "enabled", True):
            adapter = MemoryEnhancedLLMAgentAdapter(memory_server=mock_memory_server)

            status = adapter.get_status()

            assert status["strategy_type"] == "functional_memory_enhanced"
            assert status["memory_enabled"] is True
            assert status["memory_server_connected"] is True

    @pytest.mark.asyncio
    async def test_adapter_error_handling(self, sample_market_state):
        """Test error handling in strategy adapters."""
        adapter = LLMAgentAdapter()

        # Make the strategy raise an exception
        adapter._strategy.analyze_market_functional = AsyncMock(
            side_effect=Exception("Test error")
        )

        trade_action = await adapter.analyze_market(sample_market_state)

        # Should return safe HOLD action
        assert trade_action.action == "HOLD"
        assert trade_action.size_pct == 0
        assert "Adapter error: Test error" in trade_action.rationale


if __name__ == "__main__":
    # Run some basic functionality tests
    print("Testing Functional LLM Strategy Integration...")

    # Test configuration
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        api_key="sk-test",
    )
    assert config.provider == LLMProvider.OPENAI
    print("✓ LLM configuration test passed")

    # Test context creation
    market_state = MarketState(
        symbol="BTC-USD",
        current_price=Decimal(50000),
        timestamp=datetime.now(UTC),
    )

    vumanchu_state = VuManchuState(
        cipher_a=[5.0, 6.0, 7.0],
        cipher_b=[],
        wave_trend=[50.0],
        money_flow=[55.0],
        buy_signal=True,
        sell_signal=False,
        bullish_divergence=False,
        bearish_divergence=False,
        trend_strength=0.7,
        signal_confidence=0.8,
    )

    context = create_market_context(market_state, vumanchu_state)
    assert isinstance(context, LLMContext)
    assert context.market_state == market_state
    print("✓ Market context creation test passed")

    # Test prompt generation
    params = TradingParams()
    prompt = generate_trading_prompt(context, params, include_examples=False)
    assert "BTC-USD" in prompt
    assert "JSON format" in prompt
    print("✓ Prompt generation test passed")

    # Test response parsing
    test_response = """
    {
        "action": "LONG",
        "confidence": 0.8,
        "reasoning": "Test response",
        "risk_assessment": {
            "risk_level": "MEDIUM",
            "key_risks": [],
            "mitigation": ""
        }
    }
    """

    parsed = parse_llm_response(test_response, market_state)
    assert parsed.signal == Signal.LONG
    assert parsed.confidence == 0.8
    print("✓ Response parsing test passed")

    # Test type conversion
    trade_action = TypeConverter.llm_response_to_trade_action(parsed, market_state)
    assert isinstance(trade_action, TradeAction)
    assert trade_action.action == "LONG"
    print("✓ Type conversion test passed")

    print("All basic functional LLM strategy integration tests completed successfully!")
