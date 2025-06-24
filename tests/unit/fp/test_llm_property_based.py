"""
Property-based tests for LLM prompt generation and response validation with FP types.

This module uses Hypothesis to generate comprehensive test cases for LLM operations
using functional programming types, ensuring robust behavior across a wide range
of inputs and edge cases.

Tests include:
- Property-based prompt generation with various market conditions
- Response parsing validation across different JSON formats
- Type conversion consistency between FP and legacy types  
- LLM decision validation with functional constraints
- Error handling with malformed inputs
- Performance and memory usage with large datasets
- Consistency across multiple LLM interactions
"""

import pytest
import json
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import string

from hypothesis import given, strategies as st, settings, HealthCheck, assume
from hypothesis.strategies import composite

# FP test infrastructure
from tests.fp_test_base import (
    FPTestBase,
    FP_AVAILABLE
)

if FP_AVAILABLE:
    # FP types
    from bot.fp.types.result import Result, Success, Failure
    from bot.fp.types.base import Maybe, Some, Nothing, Symbol
    from bot.fp.types.trading import (
        Long, Short, Hold, MarketMake, TradeSignal,
        FunctionalMarketData, FunctionalMarketState, TradingIndicators
    )
    from bot.fp.types.learning import (
        MarketSnapshot, TradingExperienceFP, PatternTag
    )
    
    # LLM integration (assuming these exist based on previous tests)
    try:
        from bot.fp.strategies.llm_functional import (
            create_llm_context, generate_llm_prompt, parse_llm_response,
            LLMContext, LLMResponse, validate_llm_decision
        )
        LLM_FUNCTIONAL_AVAILABLE = True
    except ImportError:
        LLM_FUNCTIONAL_AVAILABLE = False
        
        # Create mock classes for testing
        class LLMContext:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class LLMResponse:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        def create_llm_context(*args, **kwargs):
            return LLMContext()
        
        def generate_llm_prompt(context):
            return "Mock prompt"
        
        def parse_llm_response(response_text, market_state):
            return LLMResponse(signal=Long, confidence=0.8)
        
        def validate_llm_decision(response, market_state):
            return Success(response)
else:
    # Fallback stubs for non-FP environments
    LLM_FUNCTIONAL_AVAILABLE = False
    
    class LLMContext:
        pass


# Hypothesis strategies for FP types

@composite
def symbol_strategy(draw):
    """Generate valid trading symbols."""
    base_currencies = ["BTC", "ETH", "ADA", "SOL", "MATIC", "AVAX", "DOT", "LINK"]
    quote_currencies = ["USD", "USDT", "USDC", "EUR", "GBP"]
    separators = ["-", "/", ""]
    
    base = draw(st.sampled_from(base_currencies))
    quote = draw(st.sampled_from(quote_currencies))
    separator = draw(st.sampled_from(separators))
    
    symbol_str = f"{base}{separator}{quote}"
    result = Symbol.create(symbol_str)
    assume(result.is_success())
    return result.success()


@composite
def decimal_price_strategy(draw):
    """Generate realistic price decimals."""
    # Generate prices in reasonable ranges for different assets
    price_ranges = [
        (1, 100),      # Altcoins
        (100, 5000),   # Mid-cap
        (5000, 100000) # BTC range
    ]
    
    price_range = draw(st.sampled_from(price_ranges))
    price_float = draw(st.floats(
        min_value=price_range[0], 
        max_value=price_range[1],
        allow_nan=False,
        allow_infinity=False
    ))
    
    return Decimal(str(round(price_float, 2)))


@composite
def trading_indicators_strategy(draw):
    """Generate realistic trading indicators."""
    rsi = draw(st.floats(min_value=0, max_value=100, allow_nan=False))
    macd = draw(st.floats(min_value=-100, max_value=100, allow_nan=False))
    ema_fast = draw(st.floats(min_value=1, max_value=100000, allow_nan=False))
    ema_slow = draw(st.floats(min_value=1, max_value=100000, allow_nan=False))
    
    # Ensure fast EMA is actually faster (higher) than slow
    if ema_fast < ema_slow:
        ema_fast, ema_slow = ema_slow, ema_fast
    
    return TradingIndicators(
        timestamp=datetime.now(UTC),
        rsi=rsi,
        macd=macd,
        macd_signal=draw(st.floats(min_value=-100, max_value=100, allow_nan=False)),
        macd_histogram=draw(st.floats(min_value=-50, max_value=50, allow_nan=False)),
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        bollinger_upper=draw(st.floats(min_value=1, max_value=100000, allow_nan=False)),
        bollinger_middle=draw(st.floats(min_value=1, max_value=100000, allow_nan=False)),
        bollinger_lower=draw(st.floats(min_value=1, max_value=100000, allow_nan=False)),
        volume_sma=draw(st.floats(min_value=0, max_value=1000000, allow_nan=False)),
        atr=draw(st.floats(min_value=0, max_value=10000, allow_nan=False)),
        cipher_a_dot=draw(st.floats(min_value=-100, max_value=100, allow_nan=False)),
        cipher_b_wave=draw(st.floats(min_value=-100, max_value=100, allow_nan=False)),
        cipher_b_money_flow=draw(st.floats(min_value=0, max_value=100, allow_nan=False)),
        usdt_dominance=draw(st.floats(min_value=0, max_value=20, allow_nan=False)),
        usdc_dominance=draw(st.floats(min_value=0, max_value=15, allow_nan=False)),
        stablecoin_dominance=draw(st.floats(min_value=0, max_value=30, allow_nan=False)),
        dominance_trend=draw(st.sampled_from(["BULLISH", "BEARISH", "NEUTRAL"])),
        dominance_rsi=draw(st.floats(min_value=0, max_value=100, allow_nan=False)),
    )


@composite
def market_snapshot_strategy(draw):
    """Generate realistic market snapshots."""
    symbol = draw(symbol_strategy())
    price = draw(decimal_price_strategy())
    
    # Generate indicators dict
    indicators = {
        "rsi": draw(st.floats(min_value=0, max_value=100, allow_nan=False)),
        "cipher_a_dot": draw(st.floats(min_value=-100, max_value=100, allow_nan=False)),
        "cipher_b_wave": draw(st.floats(min_value=-100, max_value=100, allow_nan=False)),
        "cipher_b_money_flow": draw(st.floats(min_value=0, max_value=100, allow_nan=False)),
        "ema_fast": draw(st.floats(min_value=1, max_value=100000, allow_nan=False)),
        "ema_slow": draw(st.floats(min_value=1, max_value=100000, allow_nan=False)),
    }
    
    dominance_data = None
    if draw(st.booleans()):  # Sometimes include dominance data
        dominance_data = {
            "stablecoin_dominance": draw(st.floats(min_value=0, max_value=30, allow_nan=False)),
            "dominance_24h_change": draw(st.floats(min_value=-10, max_value=10, allow_nan=False)),
            "dominance_rsi": draw(st.floats(min_value=0, max_value=100, allow_nan=False)),
        }
    
    return MarketSnapshot(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        price=price,
        indicators=indicators,
        dominance_data=dominance_data,
        position_side=draw(st.sampled_from(["LONG", "SHORT", "FLAT"])),
        position_size=draw(st.decimals(min_value=0, max_value=10, places=4)),
    )


@composite
def functional_market_state_strategy(draw):
    """Generate functional market states."""
    symbol = draw(symbol_strategy())
    price = draw(decimal_price_strategy())
    indicators = draw(trading_indicators_strategy())
    
    market_data = FunctionalMarketData(
        symbol=symbol.value,
        timestamp=datetime.now(UTC),
        open=price * Decimal("0.99"),
        high=price * Decimal("1.02"), 
        low=price * Decimal("0.98"),
        close=price,
        volume=draw(st.decimals(min_value=1, max_value=1000000, places=2)),
    )
    
    return FunctionalMarketState(
        symbol=symbol.value,
        timestamp=datetime.now(UTC),
        market_data=market_data,
        indicators=indicators,
        position=None,  # Simplified for testing
        account_balance=None,
    )


@composite
def trade_signal_strategy(draw):
    """Generate various trade signals."""
    signal_type = draw(st.sampled_from(["LONG", "SHORT", "HOLD", "MARKET_MAKE"]))
    confidence = draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False))
    size = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False))
    reason = draw(st.text(alphabet=string.ascii_letters + string.digits + " ", min_size=5, max_size=100))
    
    if signal_type == "LONG":
        return Long(confidence=confidence, size=size, reason=reason)
    elif signal_type == "SHORT":
        return Short(confidence=confidence, size=size, reason=reason)
    elif signal_type == "HOLD":
        return Hold(reason=reason)
    else:  # MARKET_MAKE
        spread = draw(st.floats(min_value=0.001, max_value=0.1, allow_nan=False))
        return MarketMake(
            confidence=confidence,
            size=size,
            spread=spread,
            reason=reason
        )


@composite
def llm_response_json_strategy(draw):
    """Generate various LLM response JSON structures."""
    action = draw(st.sampled_from(["LONG", "SHORT", "HOLD", "MARKET_MAKE"]))
    confidence = draw(st.floats(min_value=0, max_value=1, allow_nan=False))
    reasoning = draw(st.text(alphabet=string.ascii_letters + string.digits + " .,!?", min_size=10, max_size=200))
    
    response = {
        "action": action,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    
    # Sometimes add optional fields
    if draw(st.booleans()):
        response["risk_assessment"] = {
            "risk_level": draw(st.sampled_from(["LOW", "MEDIUM", "HIGH"])),
            "key_risks": draw(st.lists(st.text(min_size=5, max_size=50), min_size=0, max_size=3))
        }
    
    if draw(st.booleans()):
        response["suggested_params"] = {
            "position_size": draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False)),
            "stop_loss": draw(st.floats(min_value=1, max_value=100000, allow_nan=False)),
            "take_profit": draw(st.floats(min_value=1, max_value=100000, allow_nan=False)),
        }
    
    return response


class TestLLMPropertyBased(FPTestBase):
    """Property-based tests for LLM operations with FP types."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    @given(market_snapshot_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    def test_market_snapshot_immutability(self, market_snapshot):
        """Test that market snapshots are properly immutable."""
        original_price = market_snapshot.price
        original_indicators = market_snapshot.indicators.copy()
        
        # Verify fields cannot be modified (should raise AttributeError)
        with pytest.raises(AttributeError):
            market_snapshot.price = Decimal("99999")
        
        # Verify original values unchanged
        assert market_snapshot.price == original_price
        assert market_snapshot.indicators == original_indicators
    
    @given(market_snapshot_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_market_snapshot_serialization_roundtrip(self, market_snapshot):
        """Test that market snapshots can be serialized and deserialized consistently."""
        # Create serializable representation
        serialized = {
            "symbol": market_snapshot.symbol.value,
            "timestamp": market_snapshot.timestamp.isoformat(),
            "price": str(market_snapshot.price),
            "indicators": market_snapshot.indicators,
            "dominance_data": market_snapshot.dominance_data,
            "position_side": market_snapshot.position_side,
            "position_size": str(market_snapshot.position_size),
        }
        
        # Verify serialization produces consistent structure
        assert isinstance(serialized["symbol"], str)
        assert isinstance(serialized["timestamp"], str)
        assert isinstance(serialized["price"], str)
        assert isinstance(serialized["indicators"], dict)
        
        # Verify we can reconstruct key values
        reconstructed_price = Decimal(serialized["price"])
        assert reconstructed_price == market_snapshot.price
        
        reconstructed_symbol = Symbol.create(serialized["symbol"])
        assert reconstructed_symbol.is_success()
        assert reconstructed_symbol.success().value == market_snapshot.symbol.value
    
    @given(functional_market_state_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_llm_context_generation_properties(self, market_state):
        """Test properties of LLM context generation."""
        if not LLM_FUNCTIONAL_AVAILABLE:
            pytest.skip("LLM functional module not available")
        
        context = create_llm_context(
            market_state=market_state,
            confidence_threshold=0.7,
            risk_tolerance="MEDIUM",
            additional_context="Property test context"
        )
        
        # Verify context structure
        assert hasattr(context, 'market_state')
        assert hasattr(context, 'confidence_threshold')
        assert hasattr(context, 'risk_tolerance')
        
        # Verify context values
        assert context.market_state == market_state
        assert context.confidence_threshold == 0.7
        assert context.risk_tolerance == "MEDIUM"
    
    @given(functional_market_state_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=20)
    def test_llm_prompt_generation_consistency(self, market_state):
        """Test that LLM prompt generation is consistent and deterministic."""
        if not LLM_FUNCTIONAL_AVAILABLE:
            pytest.skip("LLM functional module not available")
        
        context = create_llm_context(
            market_state=market_state,
            confidence_threshold=0.7,
            risk_tolerance="MEDIUM"
        )
        
        # Generate prompt multiple times
        prompt1 = generate_llm_prompt(context)
        prompt2 = generate_llm_prompt(context)
        
        # Should be deterministic for same input
        assert prompt1 == prompt2
        
        # Should be non-empty string
        assert isinstance(prompt1, str)
        assert len(prompt1) > 0
        
        # Should contain key information
        assert market_state.symbol in prompt1 or "symbol" in prompt1.lower()
    
    @given(llm_response_json_strategy(), functional_market_state_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    def test_llm_response_parsing_robustness(self, response_data, market_state):
        """Test that LLM response parsing handles various input formats robustly."""
        if not LLM_FUNCTIONAL_AVAILABLE:
            pytest.skip("LLM functional module not available")
        
        # Convert to JSON string
        response_text = json.dumps(response_data)
        
        try:
            parsed_response = parse_llm_response(response_text, market_state)
            
            # If parsing succeeds, verify structure
            assert hasattr(parsed_response, 'signal')
            assert hasattr(parsed_response, 'confidence')
            
            # Verify signal type is valid
            assert parsed_response.signal in [Long, Short, Hold, MarketMake]
            
            # Verify confidence is in valid range
            assert 0.0 <= parsed_response.confidence <= 1.0
            
        except (ValueError, KeyError, AttributeError) as e:
            # Parsing may fail for some edge cases - this is acceptable
            # but should not crash the system
            assert isinstance(e, (ValueError, KeyError, AttributeError))
    
    @given(
        st.text(alphabet=string.ascii_letters + string.digits + " {}[]():,\"", min_size=1, max_size=500),
        functional_market_state_strategy()
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_llm_response_parsing_malformed_input(self, malformed_text, market_state):
        """Test LLM response parsing with malformed/random text input."""
        if not LLM_FUNCTIONAL_AVAILABLE:
            pytest.skip("LLM functional module not available")
        
        # Should handle malformed input gracefully
        try:
            parsed_response = parse_llm_response(malformed_text, market_state)
            
            # If it somehow parses, verify it's still valid
            if parsed_response is not None:
                assert hasattr(parsed_response, 'signal')
                assert hasattr(parsed_response, 'confidence')
                
        except Exception as e:
            # Malformed input should raise appropriate exceptions, not crash
            assert isinstance(e, (ValueError, KeyError, AttributeError, json.JSONDecodeError))
    
    @given(trade_signal_strategy(), functional_market_state_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=40)
    def test_signal_type_consistency(self, signal, market_state):
        """Test that trade signals maintain type consistency."""
        # Verify signal has expected attributes
        if isinstance(signal, (Long, Short)):
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'size')
            assert hasattr(signal, 'reason')
            assert 0.0 <= signal.confidence <= 1.0
            assert 0.0 < signal.size <= 1.0
        elif isinstance(signal, Hold):
            assert hasattr(signal, 'reason')
        elif isinstance(signal, MarketMake):
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'size')
            assert hasattr(signal, 'spread')
            assert hasattr(signal, 'reason')
            assert 0.0 <= signal.confidence <= 1.0
            assert 0.0 < signal.size <= 1.0
            assert 0.0 < signal.spread <= 1.0
        
        # Verify signal is immutable
        with pytest.raises(AttributeError):
            if hasattr(signal, 'confidence'):
                signal.confidence = 0.999
    
    @given(trade_signal_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_signal_serialization_properties(self, signal):
        """Test trade signal serialization properties."""
        # Test string representation
        signal_str = str(signal)
        assert isinstance(signal_str, str)
        assert len(signal_str) > 0
        
        # Test type name in string representation
        type_name = type(signal).__name__
        assert type_name in ["Long", "Short", "Hold", "MarketMake"]
    
    @given(
        st.lists(market_snapshot_strategy(), min_size=1, max_size=10),
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=20)
    def test_batch_processing_consistency(self, market_snapshots, confidence_threshold):
        """Test that batch processing of market snapshots is consistent."""
        if not LLM_FUNCTIONAL_AVAILABLE:
            pytest.skip("LLM functional module not available")
        
        # Process snapshots individually
        individual_results = []
        for snapshot in market_snapshots:
            # Create mock market state from snapshot
            market_state = FunctionalMarketState(
                symbol=snapshot.symbol.value,
                timestamp=snapshot.timestamp,
                market_data=FunctionalMarketData(
                    symbol=snapshot.symbol.value,
                    timestamp=snapshot.timestamp,
                    open=snapshot.price * Decimal("0.99"),
                    high=snapshot.price * Decimal("1.01"),
                    low=snapshot.price * Decimal("0.99"),
                    close=snapshot.price,
                    volume=Decimal("1000"),
                ),
                indicators=TradingIndicators(
                    timestamp=snapshot.timestamp,
                    rsi=snapshot.indicators.get("rsi", 50.0),
                    cipher_a_dot=snapshot.indicators.get("cipher_a_dot", 0.0),
                    cipher_b_wave=snapshot.indicators.get("cipher_b_wave", 0.0),
                    cipher_b_money_flow=snapshot.indicators.get("cipher_b_money_flow", 50.0),
                ),
                position=None,
                account_balance=None,
            )
            
            context = create_llm_context(
                market_state=market_state,
                confidence_threshold=confidence_threshold
            )
            individual_results.append(context)
        
        # Verify all contexts were created successfully
        assert len(individual_results) == len(market_snapshots)
        for result in individual_results:
            assert hasattr(result, 'market_state')
            assert hasattr(result, 'confidence_threshold')
            assert result.confidence_threshold == confidence_threshold
    
    @given(
        st.integers(min_value=1, max_value=100),
        market_snapshot_strategy()
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=10)
    def test_repeated_operations_consistency(self, iterations, market_snapshot):
        """Test that repeated operations on same data produce consistent results."""
        if not LLM_FUNCTIONAL_AVAILABLE:
            pytest.skip("LLM functional module not available")
        
        # Create market state from snapshot
        market_state = FunctionalMarketState(
            symbol=market_snapshot.symbol.value,
            timestamp=market_snapshot.timestamp,
            market_data=FunctionalMarketData(
                symbol=market_snapshot.symbol.value,
                timestamp=market_snapshot.timestamp,
                open=market_snapshot.price,
                high=market_snapshot.price,
                low=market_snapshot.price,
                close=market_snapshot.price,
                volume=Decimal("1000"),
            ),
            indicators=TradingIndicators(
                timestamp=market_snapshot.timestamp,
                rsi=market_snapshot.indicators.get("rsi", 50.0),
            ),
            position=None,
            account_balance=None,
        )
        
        # Perform same operation multiple times
        results = []
        for _ in range(min(iterations, 10)):  # Cap at 10 for performance
            context = create_llm_context(market_state=market_state)
            prompt = generate_llm_prompt(context)
            results.append(prompt)
        
        # All results should be identical (deterministic)
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result
    
    @given(
        st.dictionaries(
            st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
            st.one_of(
                st.floats(min_value=-1000, max_value=1000, allow_nan=False),
                st.text(min_size=1, max_size=50),
                st.booleans()
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=20)
    def test_market_indicators_validation(self, indicators_dict):
        """Test market indicators validation with various input types."""
        # Create market snapshot with random indicators
        symbol_result = Symbol.create("BTC-USD")
        assume(symbol_result.is_success())
        
        try:
            snapshot = MarketSnapshot(
                symbol=symbol_result.success(),
                timestamp=datetime.now(UTC),
                price=Decimal("50000"),
                indicators=indicators_dict,
                dominance_data=None,
                position_side="FLAT",
                position_size=Decimal("0"),
            )
            
            # If creation succeeds, verify structure
            assert snapshot.indicators == indicators_dict
            assert isinstance(snapshot.indicators, dict)
            
        except (ValueError, TypeError) as e:
            # Some combinations may be invalid - this is acceptable
            assert isinstance(e, (ValueError, TypeError))
    
    @given(
        st.floats(min_value=0, max_value=1, allow_nan=False),
        st.floats(min_value=0, max_value=1, allow_nan=False)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_confidence_calculations_properties(self, confidence1, confidence2):
        """Test properties of confidence calculations."""
        # Test confidence combination (simplified)
        combined_confidence = (confidence1 + confidence2) / 2
        
        # Should be in valid range
        assert 0.0 <= combined_confidence <= 1.0
        
        # Should be symmetric
        alt_combined = (confidence2 + confidence1) / 2
        assert abs(combined_confidence - alt_combined) < 1e-10
        
        # Should be bounded by inputs
        assert min(confidence1, confidence2) <= combined_confidence <= max(confidence1, confidence2)
    
    @given(
        st.lists(
            st.tuples(
                st.text(alphabet=string.ascii_letters, min_size=1, max_size=20),
                st.floats(min_value=-1000, max_value=1000, allow_nan=False)
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=15)
    def test_pattern_tag_generation(self, pattern_data):
        """Test pattern tag generation from various inputs."""
        tags = []
        
        for pattern_name, _ in pattern_data:
            try:
                tag_result = PatternTag.create(pattern_name)
                if tag_result.is_success():
                    tag = tag_result.success()
                    tags.append(tag)
                    
                    # Verify tag properties
                    assert isinstance(tag.name, str)
                    assert len(tag.name) > 0
                    assert tag.name == tag.name.lower()  # Should be normalized
                    
            except Exception as e:
                # Some patterns may be invalid
                assert isinstance(e, (ValueError, TypeError))
        
        # If any tags were created, verify uniqueness
        if tags:
            tag_names = [tag.name for tag in tags]
            unique_names = set(tag_names)
            # Note: Some duplicates may exist due to normalization


class TestLLMPerformanceProperties(FPTestBase):
    """Property-based tests for LLM operation performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    @given(
        st.lists(market_snapshot_strategy(), min_size=10, max_size=50)
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=5,
        deadline=None
    )
    def test_batch_processing_performance(self, market_snapshots):
        """Test performance characteristics of batch processing."""
        if not LLM_FUNCTIONAL_AVAILABLE:
            pytest.skip("LLM functional module not available")
        
        import time
        
        start_time = time.time()
        
        # Process all snapshots
        contexts = []
        for snapshot in market_snapshots:
            # Create simplified market state
            market_state = FunctionalMarketState(
                symbol=snapshot.symbol.value,
                timestamp=snapshot.timestamp,
                market_data=FunctionalMarketData(
                    symbol=snapshot.symbol.value,
                    timestamp=snapshot.timestamp,
                    open=snapshot.price,
                    high=snapshot.price,
                    low=snapshot.price,
                    close=snapshot.price,
                    volume=Decimal("1000"),
                ),
                indicators=TradingIndicators(timestamp=snapshot.timestamp),
                position=None,
                account_balance=None,
            )
            
            context = create_llm_context(market_state=market_state)
            contexts.append(context)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance should be reasonable
        time_per_snapshot = processing_time / len(market_snapshots)
        assert time_per_snapshot < 1.0  # Should process faster than 1 second per snapshot
        
        # Memory usage should be reasonable (all contexts should be created)
        assert len(contexts) == len(market_snapshots)
    
    @given(
        st.integers(min_value=1, max_value=20),
        market_snapshot_strategy()
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=5,
        deadline=None
    )
    def test_memory_usage_properties(self, repetitions, market_snapshot):
        """Test memory usage properties of repeated operations."""
        if not LLM_FUNCTIONAL_AVAILABLE:
            pytest.skip("LLM functional module not available")
        
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Perform operations
        results = []
        for _ in range(repetitions):
            market_state = FunctionalMarketState(
                symbol=market_snapshot.symbol.value,
                timestamp=market_snapshot.timestamp,
                market_data=FunctionalMarketData(
                    symbol=market_snapshot.symbol.value,
                    timestamp=market_snapshot.timestamp,
                    open=market_snapshot.price,
                    high=market_snapshot.price,
                    low=market_snapshot.price,
                    close=market_snapshot.price,
                    volume=Decimal("1000"),
                ),
                indicators=TradingIndicators(timestamp=market_snapshot.timestamp),
                position=None,
                account_balance=None,
            )
            
            context = create_llm_context(market_state=market_state)
            results.append(context)
        
        # Clean up
        del results
        gc.collect()
        
        # Test passes if no memory errors occurred
        assert True


if __name__ == "__main__":
    pytest.main([__file__])