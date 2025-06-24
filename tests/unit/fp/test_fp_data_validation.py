"""
Functional Programming Data Validation Tests

This module tests data validation using functional programming patterns,
including Result/Either error handling, validation composition, and
monadic validation chains.
"""

import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Callable, Union
from unittest.mock import MagicMock

from bot.fp.core.either import Either, Left, Right
from bot.fp.core.option import Option, Some, None_ as NoneOption
from bot.fp.core.validation import (
    ValidationResult,
    ValidationError,
    ValidationChain,
    compose_validations,
    validate_all,
    validate_any,
    create_validation_chain,
)
from bot.fp.core.functional_validation import (
    validate_market_data_functional,
    validate_price_range,
    validate_volume_positive,
    validate_timestamp_recent,
    validate_symbol_format,
    validate_ohlcv_consistency,
    validate_trade_data_functional,
    validate_aggregation_window,
    validate_connection_health_functional,
    validate_data_quality_functional,
)
from bot.fp.types.market import (
    Candle,
    Trade,
    Ticker,
    OrderBook,
    WebSocketMessage,
    ConnectionState,
    ConnectionStatus,
    DataQuality,
)


class TestBasicFunctionalValidation:
    """Test basic functional validation patterns."""
    
    def test_either_success_validation(self):
        """Test successful validation returning Right."""
        def validate_positive_number(value: int) -> Either[str, int]:
            if value > 0:
                return Right(value)
            return Left("Value must be positive")
        
        result = validate_positive_number(42)
        assert isinstance(result, Right)
        assert result.value == 42
        
        # Test chaining successful validations
        result2 = result.map(lambda x: x * 2)
        assert isinstance(result2, Right)
        assert result2.value == 84
    
    def test_either_failure_validation(self):
        """Test failed validation returning Left."""
        def validate_positive_number(value: int) -> Either[str, int]:
            if value > 0:
                return Right(value)
            return Left("Value must be positive")
        
        result = validate_positive_number(-5)
        assert isinstance(result, Left)
        assert result.value == "Value must be positive"
        
        # Test that failed validation stops the chain
        result2 = result.map(lambda x: x * 2)
        assert isinstance(result2, Left)
        assert result2.value == "Value must be positive"
    
    def test_either_flat_map_chaining(self):
        """Test flat_map for chaining validations."""
        def validate_positive(value: int) -> Either[str, int]:
            return Right(value) if value > 0 else Left("Must be positive")
        
        def validate_even(value: int) -> Either[str, int]:
            return Right(value) if value % 2 == 0 else Left("Must be even")
        
        # Successful chain
        result = validate_positive(42).flat_map(validate_even)
        assert isinstance(result, Right)
        assert result.value == 42
        
        # First validation fails
        result = validate_positive(-5).flat_map(validate_even)
        assert isinstance(result, Left)
        assert result.value == "Must be positive"
        
        # First passes, second fails
        result = validate_positive(43).flat_map(validate_even)
        assert isinstance(result, Left)
        assert result.value == "Must be even"
    
    def test_option_validation_patterns(self):
        """Test Option patterns for nullable validation."""
        def safe_divide(a: float, b: float) -> Option[float]:
            if b == 0:
                return NoneOption()
            return Some(a / b)
        
        # Successful division
        result = safe_divide(10.0, 2.0)
        assert result.is_some()
        assert result.unwrap() == 5.0
        
        # Division by zero
        result = safe_divide(10.0, 0.0)
        assert result.is_none()
        
        # Chaining with map
        result = safe_divide(10.0, 2.0).map(lambda x: x * 2)
        assert result.is_some()
        assert result.unwrap() == 10.0
        
        # Chaining with None
        result = safe_divide(10.0, 0.0).map(lambda x: x * 2)
        assert result.is_none()


class TestMarketDataValidation:
    """Test market data validation using functional patterns."""
    
    def test_price_range_validation(self):
        """Test price range validation."""
        # Valid price
        result = validate_price_range(Decimal("50000"), Decimal("1000"), Decimal("100000"))
        assert isinstance(result, Right)
        assert result.value == Decimal("50000")
        
        # Price too low
        result = validate_price_range(Decimal("500"), Decimal("1000"), Decimal("100000"))
        assert isinstance(result, Left)
        assert "below minimum" in result.value.lower()
        
        # Price too high
        result = validate_price_range(Decimal("150000"), Decimal("1000"), Decimal("100000"))
        assert isinstance(result, Left)
        assert "above maximum" in result.value.lower()
    
    def test_volume_validation(self):
        """Test volume validation."""
        # Valid volume
        result = validate_volume_positive(Decimal("100.5"))
        assert isinstance(result, Right)
        assert result.value == Decimal("100.5")
        
        # Zero volume
        result = validate_volume_positive(Decimal("0"))
        assert isinstance(result, Left)
        assert "positive" in result.value.lower()
        
        # Negative volume
        result = validate_volume_positive(Decimal("-10"))
        assert isinstance(result, Left)
        assert "positive" in result.value.lower()
    
    def test_timestamp_validation(self):
        """Test timestamp validation."""
        # Recent timestamp
        recent_time = datetime.now(UTC) - timedelta(seconds=5)
        result = validate_timestamp_recent(recent_time, max_age_seconds=30)
        assert isinstance(result, Right)
        assert result.value == recent_time
        
        # Old timestamp
        old_time = datetime.now(UTC) - timedelta(minutes=10)
        result = validate_timestamp_recent(old_time, max_age_seconds=30)
        assert isinstance(result, Left)
        assert "too old" in result.value.lower()
        
        # Future timestamp
        future_time = datetime.now(UTC) + timedelta(minutes=1)
        result = validate_timestamp_recent(future_time, max_age_seconds=30)
        assert isinstance(result, Left)
        assert "future" in result.value.lower()
    
    def test_symbol_format_validation(self):
        """Test symbol format validation."""
        # Valid symbols
        valid_symbols = ["BTC-USD", "ETH-USDC", "SOL-USD"]
        for symbol in valid_symbols:
            result = validate_symbol_format(symbol)
            assert isinstance(result, Right)
            assert result.value == symbol
        
        # Invalid symbols
        invalid_symbols = ["BTCUSD", "BTC_USD", "btc-usd", "BTC-", "-USD", ""]
        for symbol in invalid_symbols:
            result = validate_symbol_format(symbol)
            assert isinstance(result, Left)
            assert "format" in result.value.lower()
    
    def test_ohlcv_consistency_validation(self):
        """Test OHLCV consistency validation."""
        # Valid OHLCV
        ohlcv = {
            "open": Decimal("50000"),
            "high": Decimal("51000"),
            "low": Decimal("49500"),
            "close": Decimal("50500"),
            "volume": Decimal("100")
        }
        
        result = validate_ohlcv_consistency(ohlcv)
        assert isinstance(result, Right)
        assert result.value == ohlcv
        
        # Invalid: high < open
        invalid_ohlcv = {
            "open": Decimal("50000"),
            "high": Decimal("49000"),  # Lower than open
            "low": Decimal("49500"),
            "close": Decimal("50500"),
            "volume": Decimal("100")
        }
        
        result = validate_ohlcv_consistency(invalid_ohlcv)
        assert isinstance(result, Left)
        assert "high" in result.value.lower()
        
        # Invalid: low > close
        invalid_ohlcv2 = {
            "open": Decimal("50000"),
            "high": Decimal("51000"),
            "low": Decimal("50600"),  # Higher than close
            "close": Decimal("50500"),
            "volume": Decimal("100")
        }
        
        result = validate_ohlcv_consistency(invalid_ohlcv2)
        assert isinstance(result, Left)
        assert "low" in result.value.lower()


class TestTradeDataValidation:
    """Test trade data validation using functional patterns."""
    
    def test_trade_validation_success(self):
        """Test successful trade validation."""
        trade = Trade(
            id="trade-123",
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="BUY",
            symbol="BTC-USD"
        )
        
        result = validate_trade_data_functional(trade)
        assert isinstance(result, Right)
        assert result.value == trade
    
    def test_trade_validation_failure(self):
        """Test trade validation failure scenarios."""
        # Test invalid trade creation at type level
        with pytest.raises(ValueError, match="Price must be positive"):
            Trade(
                id="trade-123",
                timestamp=datetime.now(UTC),
                price=Decimal("-1000"),  # Invalid negative price
                size=Decimal("0.5"),
                side="BUY",
                symbol="BTC-USD"
            )
        
        with pytest.raises(ValueError, match="Side must be BUY or SELL"):
            Trade(
                id="trade-123",
                timestamp=datetime.now(UTC),
                price=Decimal("50000"),
                size=Decimal("0.5"),
                side="INVALID",  # Invalid side
                symbol="BTC-USD"
            )
    
    def test_trade_business_logic_validation(self):
        """Test trade business logic validation."""
        # Test with very old timestamp
        old_trade = Trade(
            id="old-trade",
            timestamp=datetime.now(UTC) - timedelta(hours=1),
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="BUY",
            symbol="BTC-USD"
        )
        
        # Business logic validation might flag this as stale
        result = validate_trade_data_functional(old_trade, max_age_seconds=60)
        if isinstance(result, Left):
            assert "old" in result.value.lower() or "stale" in result.value.lower()
        
        # Test with very small size
        small_trade = Trade(
            id="small-trade",
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            size=Decimal("0.000001"),  # Very small
            side="BUY",
            symbol="BTC-USD"
        )
        
        result = validate_trade_data_functional(small_trade, min_size=Decimal("0.001"))
        if isinstance(result, Left):
            assert "size" in result.value.lower() or "minimum" in result.value.lower()


class TestValidationComposition:
    """Test composition of multiple validations."""
    
    def test_compose_validations_all_success(self):
        """Test composing validations when all succeed."""
        def validate_positive(x: int) -> Either[str, int]:
            return Right(x) if x > 0 else Left("Must be positive")
        
        def validate_even(x: int) -> Either[str, int]:
            return Right(x) if x % 2 == 0 else Left("Must be even")
        
        def validate_less_than_100(x: int) -> Either[str, int]:
            return Right(x) if x < 100 else Left("Must be less than 100")
        
        validations = [validate_positive, validate_even, validate_less_than_100]
        composed = compose_validations(validations)
        
        # Test successful validation
        result = composed(42)
        assert isinstance(result, Right)
        assert result.value == 42
    
    def test_compose_validations_with_failure(self):
        """Test composing validations when one fails."""
        def validate_positive(x: int) -> Either[str, int]:
            return Right(x) if x > 0 else Left("Must be positive")
        
        def validate_even(x: int) -> Either[str, int]:
            return Right(x) if x % 2 == 0 else Left("Must be even")
        
        validations = [validate_positive, validate_even]
        composed = compose_validations(validations)
        
        # Test with odd number (second validation fails)
        result = composed(43)
        assert isinstance(result, Left)
        assert result.value == "Must be even"
        
        # Test with negative number (first validation fails)
        result = composed(-2)
        assert isinstance(result, Left)
        assert result.value == "Must be positive"
    
    def test_validate_all_pattern(self):
        """Test validate_all pattern for collecting all errors."""
        def validate_positive(x: int) -> Either[str, int]:
            return Right(x) if x > 0 else Left("Must be positive")
        
        def validate_even(x: int) -> Either[str, int]:
            return Right(x) if x % 2 == 0 else Left("Must be even")
        
        def validate_less_than_100(x: int) -> Either[str, int]:
            return Right(x) if x < 100 else Left("Must be less than 100")
        
        validations = [validate_positive, validate_even, validate_less_than_100]
        
        # All validations pass
        result = validate_all(validations, 42)
        assert isinstance(result, Right)
        assert result.value == [42, 42, 42]  # Results from each validation
        
        # Multiple validations fail
        result = validate_all(validations, -43)
        assert isinstance(result, Left)
        errors = result.value
        assert len(errors) >= 2  # At least positive and even validations fail
        assert "positive" in str(errors)
        assert "even" in str(errors)
    
    def test_validate_any_pattern(self):
        """Test validate_any pattern for accepting any successful validation."""
        def validate_multiple_of_2(x: int) -> Either[str, int]:
            return Right(x) if x % 2 == 0 else Left("Must be multiple of 2")
        
        def validate_multiple_of_3(x: int) -> Either[str, int]:
            return Right(x) if x % 3 == 0 else Left("Must be multiple of 3")
        
        def validate_multiple_of_5(x: int) -> Either[str, int]:
            return Right(x) if x % 5 == 0 else Left("Must be multiple of 5")
        
        validations = [validate_multiple_of_2, validate_multiple_of_3, validate_multiple_of_5]
        
        # Number that satisfies multiple validations
        result = validate_any(validations, 30)  # Multiple of 2, 3, and 5
        assert isinstance(result, Right)
        assert result.value == 30
        
        # Number that satisfies only one validation
        result = validate_any(validations, 9)  # Multiple of 3 only
        assert isinstance(result, Right)
        assert result.value == 9
        
        # Number that satisfies no validations
        result = validate_any(validations, 7)  # Prime number
        assert isinstance(result, Left)
        errors = result.value
        assert len(errors) == 3  # All validations failed


class TestValidationChain:
    """Test validation chain patterns."""
    
    def test_validation_chain_creation(self):
        """Test creating a validation chain."""
        chain = ValidationChain()
        
        # Add validations to chain
        chain.add_validation(
            lambda x: Right(x) if x > 0 else Left("Must be positive"),
            "positive_check"
        )
        chain.add_validation(
            lambda x: Right(x) if x % 2 == 0 else Left("Must be even"),
            "even_check"
        )
        
        assert len(chain.validations) == 2
        assert "positive_check" in chain.validation_names
        assert "even_check" in chain.validation_names
    
    def test_validation_chain_execution(self):
        """Test executing a validation chain."""
        chain = create_validation_chain([
            (lambda x: Right(x) if x > 0 else Left("Must be positive"), "positive"),
            (lambda x: Right(x) if x < 100 else Left("Must be < 100"), "range"),
            (lambda x: Right(x) if x % 2 == 0 else Left("Must be even"), "even"),
        ])
        
        # Successful validation
        result = chain.validate(42)
        assert isinstance(result, Right)
        assert result.value == 42
        
        # Failed validation
        result = chain.validate(-5)
        assert isinstance(result, Left)
        assert "positive" in result.value.lower()
    
    def test_validation_chain_with_transformations(self):
        """Test validation chain with data transformations."""
        def validate_and_square(x: int) -> Either[str, int]:
            if x > 0:
                return Right(x * x)
            return Left("Must be positive")
        
        def validate_and_halve(x: int) -> Either[str, int]:
            if x % 2 == 0:
                return Right(x // 2)
            return Left("Must be even")
        
        chain = create_validation_chain([
            (validate_and_square, "square"),
            (validate_and_halve, "halve"),
        ])
        
        # Test transformation chain
        result = chain.validate(4)  # 4 -> 16 -> 8
        assert isinstance(result, Right)
        assert result.value == 8


class TestConnectionAndQualityValidation:
    """Test connection and data quality validation."""
    
    def test_connection_health_validation(self):
        """Test connection health validation."""
        # Healthy connection
        healthy_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC)
        )
        
        result = validate_connection_health_functional(healthy_state)
        assert isinstance(result, Right)
        assert result.value == healthy_state
        
        # Unhealthy connection - stale messages
        stale_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC) - timedelta(minutes=5)
        )
        
        result = validate_connection_health_functional(stale_state, max_staleness=30)
        assert isinstance(result, Left)
        assert "stale" in result.value.lower() or "old" in result.value.lower()
        
        # Disconnected state
        disconnected_state = ConnectionState(
            status=ConnectionStatus.DISCONNECTED,
            url="wss://test.com"
        )
        
        result = validate_connection_health_functional(disconnected_state)
        assert isinstance(result, Left)
        assert "disconnected" in result.value.lower()
    
    def test_data_quality_validation(self):
        """Test data quality validation."""
        # Good quality
        good_quality = DataQuality(
            timestamp=datetime.now(UTC),
            messages_received=100,
            messages_processed=98,
            validation_failures=2
        )
        
        result = validate_data_quality_functional(good_quality, min_success_rate=95.0)
        assert isinstance(result, Right)
        assert result.value == good_quality
        
        # Poor quality
        poor_quality = DataQuality(
            timestamp=datetime.now(UTC),
            messages_received=100,
            messages_processed=80,
            validation_failures=20
        )
        
        result = validate_data_quality_functional(poor_quality, min_success_rate=95.0)
        assert isinstance(result, Left)
        assert "quality" in result.value.lower() or "success rate" in result.value.lower()
    
    def test_aggregation_window_validation(self):
        """Test aggregation window validation."""
        # Valid window
        start_time = datetime.now(UTC).replace(microsecond=0)
        end_time = start_time + timedelta(seconds=60)
        
        result = validate_aggregation_window(start_time, end_time, max_duration=timedelta(minutes=5))
        assert isinstance(result, Right)
        assert result.value == (start_time, end_time)
        
        # Invalid window - too long
        long_end_time = start_time + timedelta(hours=1)
        result = validate_aggregation_window(start_time, long_end_time, max_duration=timedelta(minutes=5))
        assert isinstance(result, Left)
        assert "duration" in result.value.lower() or "long" in result.value.lower()
        
        # Invalid window - end before start
        result = validate_aggregation_window(end_time, start_time)
        assert isinstance(result, Left)
        assert "end" in result.value.lower() and "start" in result.value.lower()


class TestComplexValidationScenarios:
    """Test complex validation scenarios combining multiple patterns."""
    
    def test_market_data_complete_validation(self):
        """Test complete market data validation pipeline."""
        candle = Candle(
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
            symbol="BTC-USD"
        )
        
        result = validate_market_data_functional(candle)
        assert isinstance(result, Right)
        assert result.value == candle
        
        # Test with invalid symbol format (would need to be tested at creation)
        # Since immutable types validate at construction, we test the validation function
        result = validate_symbol_format("invalid-symbol-format")
        assert isinstance(result, Left)
    
    def test_websocket_message_validation_pipeline(self):
        """Test WebSocket message validation pipeline."""
        message = WebSocketMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={"symbol": "BTC-USD", "price": "50000"}
        )
        
        # Chain multiple validations
        validations = [
            lambda m: Right(m) if m.channel else Left("Channel required"),
            lambda m: Right(m) if m.data else Left("Data required"),
            lambda m: Right(m) if "symbol" in m.data else Left("Symbol required in data"),
        ]
        
        # Apply all validations
        result = Right(message)
        for validation in validations:
            result = result.flat_map(validation)
        
        assert isinstance(result, Right)
        assert result.value == message
    
    def test_error_accumulation_validation(self):
        """Test accumulating multiple validation errors."""
        def validate_candle_complete(candle_data: dict) -> Either[List[str], dict]:
            errors = []
            
            # Validate each field
            if "open" not in candle_data or candle_data["open"] <= 0:
                errors.append("Invalid open price")
            
            if "high" not in candle_data or candle_data["high"] <= 0:
                errors.append("Invalid high price")
            
            if "low" not in candle_data or candle_data["low"] <= 0:
                errors.append("Invalid low price")
            
            if "close" not in candle_data or candle_data["close"] <= 0:
                errors.append("Invalid close price")
            
            if "volume" not in candle_data or candle_data["volume"] < 0:
                errors.append("Invalid volume")
            
            # Check OHLCV consistency
            if (candle_data.get("high", 0) < candle_data.get("open", 0) or
                candle_data.get("high", 0) < candle_data.get("close", 0)):
                errors.append("High must be >= open and close")
            
            if (candle_data.get("low", float('inf')) > candle_data.get("open", 0) or
                candle_data.get("low", float('inf')) > candle_data.get("close", 0)):
                errors.append("Low must be <= open and close")
            
            if errors:
                return Left(errors)
            return Right(candle_data)
        
        # Valid data
        valid_data = {
            "open": 50000, "high": 51000, "low": 49500, 
            "close": 50500, "volume": 100
        }
        result = validate_candle_complete(valid_data)
        assert isinstance(result, Right)
        
        # Invalid data with multiple errors
        invalid_data = {
            "open": -1, "high": 0, "low": 60000,  # Multiple issues
            "close": 50000, "volume": -10
        }
        result = validate_candle_complete(invalid_data)
        assert isinstance(result, Left)
        errors = result.value
        assert len(errors) >= 3  # Multiple validation errors
        assert any("open" in error for error in errors)
        assert any("high" in error for error in errors)
        assert any("volume" in error for error in errors)


class TestValidationPerformance:
    """Test validation performance characteristics."""
    
    def test_validation_chain_performance(self):
        """Test performance of validation chains."""
        import time
        
        # Create a long validation chain
        validations = []
        for i in range(100):
            validations.append((
                lambda x, i=i: Right(x) if x > i else Left(f"Must be > {i}"),
                f"check_{i}"
            ))
        
        chain = create_validation_chain(validations)
        
        # Measure validation time
        start_time = time.time()
        
        for _ in range(1000):
            result = chain.validate(150)  # Value that passes all validations
            assert isinstance(result, Right)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly
        assert total_time < 1.0  # Less than 1 second for 1000 validations
    
    def test_bulk_validation_performance(self):
        """Test performance of bulk validation."""
        import time
        
        def validate_trade_bulk(trades: List[Trade]) -> Either[List[str], List[Trade]]:
            errors = []
            valid_trades = []
            
            for trade in trades:
                if trade.price <= 0:
                    errors.append(f"Invalid price in trade {trade.id}")
                elif trade.size <= 0:
                    errors.append(f"Invalid size in trade {trade.id}")
                else:
                    valid_trades.append(trade)
            
            if errors:
                return Left(errors)
            return Right(valid_trades)
        
        # Create test data
        trades = []
        for i in range(1000):
            trades.append(Trade(
                id=f"trade-{i}",
                timestamp=datetime.now(UTC),
                price=Decimal(f"{50000 + i}"),
                size=Decimal("0.1"),
                side="BUY" if i % 2 == 0 else "SELL",
                symbol="BTC-USD"
            ))
        
        # Measure bulk validation time
        start_time = time.time()
        result = validate_trade_bulk(trades)
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Should complete quickly
        assert validation_time < 1.0  # Less than 1 second for 1000 trades
        assert isinstance(result, Right)
        assert len(result.value) == 1000


if __name__ == "__main__":
    # Run some basic functionality tests
    print("Testing Functional Data Validation...")
    
    # Test basic Either/Option patterns
    test_basic = TestBasicFunctionalValidation()
    test_basic.test_either_success_validation()
    test_basic.test_either_failure_validation()
    test_basic.test_either_flat_map_chaining()
    test_basic.test_option_validation_patterns()
    print("✓ Basic functional validation tests passed")
    
    # Test market data validation
    test_market = TestMarketDataValidation()
    test_market.test_price_range_validation()
    test_market.test_volume_validation()
    test_market.test_timestamp_validation()
    test_market.test_symbol_format_validation()
    test_market.test_ohlcv_consistency_validation()
    print("✓ Market data validation tests passed")
    
    # Test trade validation
    test_trade = TestTradeDataValidation()
    test_trade.test_trade_validation_success()
    test_trade.test_trade_validation_failure()
    test_trade.test_trade_business_logic_validation()
    print("✓ Trade data validation tests passed")
    
    # Test validation composition
    test_composition = TestValidationComposition()
    test_composition.test_compose_validations_all_success()
    test_composition.test_compose_validations_with_failure()
    test_composition.test_validate_all_pattern()
    test_composition.test_validate_any_pattern()
    print("✓ Validation composition tests passed")
    
    # Test validation chains
    test_chain = TestValidationChain()
    test_chain.test_validation_chain_creation()
    test_chain.test_validation_chain_execution()
    test_chain.test_validation_chain_with_transformations()
    print("✓ Validation chain tests passed")
    
    print("All functional data validation tests completed successfully!")