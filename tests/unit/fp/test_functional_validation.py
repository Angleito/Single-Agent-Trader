"""Comprehensive tests for functional validation with Result/Either types.

This module tests:
1. JSON schema validation using functional programming patterns
2. Result/Either monadic error handling
3. Validation combinators and pipelines
4. Trade action validation with FP types
5. Error composition and reporting
6. Conversion between legacy and FP types
"""

import json
from datetime import datetime
from decimal import Decimal

import pytest

# Import FP validation modules
from bot.fp.core.functional_validation import (
    ConversionError,
    FieldError,
    SchemaError,
    ValidationChainError,
    ValidationPipeline,
    any_validator,
    chain_validators,
    compose_validators,
    conditional_validator,
    convert_decimal,
    convert_money,
    convert_percentage,
    optional_field,
    validate_all,
    validate_batch_functional,
    validate_cross_field_constraints,
    validate_data_integrity,
    validate_enum,
    validate_pattern,
    validate_position_functional,
    validate_positive,
    validate_range,
    validate_string_length,
    validate_trade_action_functional,
)

# Import FP Result types
from bot.fp.types.result import Failure, Success

# Import trading types for conversion tests
from bot.trading_types import TradeAction


class TestBasicValidators:
    """Test basic validation primitives."""

    def test_validate_positive_success(self):
        """Test successful positive validation."""
        result = validate_positive(5.0, "test_field")
        assert result.is_success()
        assert result.success() == 5.0

    def test_validate_positive_failure(self):
        """Test positive validation failure."""
        result = validate_positive(-5.0, "test_field")
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, FieldError)
        assert error.field == "test_field"
        assert "Must be positive" in error.message

    def test_validate_range_success(self):
        """Test successful range validation."""
        validator = validate_range(0.0, 100.0)
        result = validator(50.0, "test_field")
        assert result.is_success()
        assert result.success() == 50.0

    def test_validate_range_failure(self):
        """Test range validation failure."""
        validator = validate_range(0.0, 100.0)
        result = validator(150.0, "test_field")
        assert result.is_failure()
        error = result.failure()
        assert "Must be between 0.0 and 100.0" in error.message
        assert error.context["min"] == 0.0
        assert error.context["max"] == 100.0

    def test_validate_string_length_success(self):
        """Test successful string length validation."""
        validator = validate_string_length(1, 10)
        result = validator("hello", "test_field")
        assert result.is_success()
        assert result.success() == "hello"

    def test_validate_string_length_failure(self):
        """Test string length validation failure."""
        validator = validate_string_length(1, 5)
        result = validator("too_long_string", "test_field")
        assert result.is_failure()
        error = result.failure()
        assert "Length must be between" in error.message

    def test_validate_pattern_success(self):
        """Test successful pattern validation."""
        validator = validate_pattern(r"^[A-Z]+-[A-Z]+$", "trading pair")
        result = validator("BTC-USD", "symbol")
        assert result.is_success()
        assert result.success() == "BTC-USD"

    def test_validate_pattern_failure(self):
        """Test pattern validation failure."""
        validator = validate_pattern(r"^[A-Z]+-[A-Z]+$", "trading pair")
        result = validator("invalid_symbol", "symbol")
        assert result.is_failure()
        error = result.failure()
        assert "Must match trading pair pattern" in error.message

    def test_validate_enum_success(self):
        """Test successful enum validation."""
        validator = validate_enum({"LONG", "SHORT", "HOLD"}, case_sensitive=False)
        result = validator("long", "action")
        assert result.is_success()
        assert result.success() == "long"

    def test_validate_enum_failure(self):
        """Test enum validation failure."""
        validator = validate_enum({"LONG", "SHORT", "HOLD"})
        result = validator("INVALID", "action")
        assert result.is_failure()
        error = result.failure()
        assert "Must be one of" in error.message


class TestValidationCombinators:
    """Test validation combinator functions."""

    def test_chain_validators_success(self):
        """Test successful validator chaining."""
        validator = chain_validators(validate_positive, validate_range(0, 100))
        result = validator(50.0)
        assert result.is_success()
        assert result.success() == 50.0

    def test_chain_validators_failure(self):
        """Test validator chaining with failures."""
        validator = chain_validators(validate_positive, validate_range(0, 10))
        result = validator(50.0)  # Passes positive but fails range
        assert result.is_failure()
        errors = result.failure()
        assert isinstance(errors, list)
        assert len(errors) == 1
        assert "Must be between 0 and 10" in errors[0].message

    def test_validate_all_success(self):
        """Test successful validation of all fields."""
        validators = {
            "action": validate_enum({"LONG", "SHORT", "HOLD"}),
            "size_pct": chain_validators(validate_positive, validate_range(0, 100)),
            "rationale": validate_string_length(1, 100),
        }

        schema_validator = validate_all(validators)
        data = {"action": "LONG", "size_pct": 25.0, "rationale": "Test trade"}

        result = schema_validator(data)
        assert result.is_success()
        validated_data = result.success()
        assert validated_data["action"] == "LONG"
        assert validated_data["size_pct"] == 25.0

    def test_validate_all_missing_field(self):
        """Test validation with missing required field."""
        validators = {
            "action": validate_enum({"LONG", "SHORT", "HOLD"}),
            "size_pct": validate_positive,
        }

        schema_validator = validate_all(validators)
        data = {"action": "LONG"}  # Missing size_pct

        result = schema_validator(data)
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, SchemaError)
        assert any(
            e.field == "size_pct" and "Required field missing" in e.message
            for e in error.errors
        )

    def test_optional_field_with_value(self):
        """Test optional field validator with value."""
        validator = optional_field(validate_positive, default=0.0)
        result = validator(5.0)
        assert result.is_success()
        assert result.success() == 5.0

    def test_optional_field_with_none(self):
        """Test optional field validator with None."""
        validator = optional_field(validate_positive, default=10.0)
        result = validator(None)
        assert result.is_success()
        assert result.success() == 10.0

    def test_compose_validators_success(self):
        """Test composed validators success."""
        validator = compose_validators(validate_positive, validate_range(0, 100))
        result = validator(50.0)
        assert result.is_success()
        assert result.success() == 50.0

    def test_any_validator_success(self):
        """Test any validator with at least one success."""
        validator = any_validator(
            validate_range(0, 10), validate_range(90, 100), validate_range(45, 55)
        )
        result = validator(50.0)  # Should pass third validator
        assert result.is_success()
        assert result.success() == 50.0

    def test_any_validator_all_fail(self):
        """Test any validator when all fail."""
        validator = any_validator(validate_range(0, 10), validate_range(90, 100))
        result = validator(50.0)  # Fails both ranges
        assert result.is_failure()
        errors = result.failure()
        assert isinstance(errors, list)
        assert len(errors) == 2

    def test_conditional_validator_condition_met(self):
        """Test conditional validator when condition is met."""
        validator = conditional_validator(
            condition=lambda x: x > 0, validator=validate_range(0, 100)
        )
        result = validator(50.0)
        assert result.is_success()
        assert result.success() == 50.0

    def test_conditional_validator_condition_not_met(self):
        """Test conditional validator when condition is not met."""
        validator = conditional_validator(
            condition=lambda x: x > 0, validator=validate_range(0, 100)
        )
        result = validator(-5.0)  # Condition not met, should pass through
        assert result.is_success()
        assert result.success() == -5.0


class TestDataConverters:
    """Test functional data converters."""

    def test_convert_decimal_from_string(self):
        """Test converting string to Decimal."""
        result = convert_decimal("123.45")
        assert result.is_success()
        assert result.success() == Decimal("123.45")

    def test_convert_decimal_from_float(self):
        """Test converting float to Decimal."""
        result = convert_decimal(123.45)
        assert result.is_success()
        assert result.success() == Decimal("123.45")

    def test_convert_decimal_from_decimal(self):
        """Test converting Decimal to Decimal (passthrough)."""
        original = Decimal("123.45")
        result = convert_decimal(original)
        assert result.is_success()
        assert result.success() == original

    def test_convert_decimal_failure(self):
        """Test Decimal conversion failure."""
        result = convert_decimal("not_a_number")
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, ConversionError)
        assert error.source_type == "str"
        assert error.target_type == "Decimal"

    def test_convert_percentage_from_decimal(self):
        """Test converting decimal percentage."""
        result = convert_percentage(0.25)
        assert result.is_success()
        # The exact behavior depends on Percentage implementation

    def test_convert_percentage_from_percent(self):
        """Test converting percentage > 1 (assumes 0-100 format)."""
        result = convert_percentage(25.0)
        assert result.is_success()
        # Should convert 25% to 0.25

    def test_convert_money_success(self):
        """Test successful money conversion."""
        result = convert_money(100.0, "USD")
        assert result.is_success()
        # The exact behavior depends on Money implementation

    def test_convert_money_failure(self):
        """Test money conversion failure."""
        result = convert_money("invalid", "USD")
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, ConversionError)


class TestValidationPipeline:
    """Test validation pipeline functionality."""

    def test_pipeline_success(self):
        """Test successful pipeline execution."""
        pipeline = ValidationPipeline()
        pipeline.add_step("positive", lambda x: validate_positive(x, "value"))
        pipeline.add_step("range", lambda x: validate_range(0, 100)(x, "value"))

        result = pipeline.validate(50.0)
        assert result.is_success()
        assert result.success() == 50.0

    def test_pipeline_failure(self):
        """Test pipeline failure at specific step."""
        pipeline = ValidationPipeline()
        pipeline.add_step("positive", lambda x: validate_positive(x, "value"))
        pipeline.add_step("range", lambda x: validate_range(0, 10)(x, "value"))

        result = pipeline.validate(50.0)  # Passes positive, fails range
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, ValidationChainError)
        assert error.chain_step == "range"

    def test_pipeline_collect_errors(self):
        """Test pipeline collecting all errors."""
        pipeline = ValidationPipeline()
        pipeline.add_step(
            "step1", lambda x: Failure(FieldError(field="test", message="Error 1"))
        )
        pipeline.add_step(
            "step2", lambda x: Failure(FieldError(field="test", message="Error 2"))
        )

        result = pipeline.validate_collect_errors("test")
        assert result.is_failure()
        errors = result.failure()
        assert isinstance(errors, list)
        assert len(errors) == 2


class TestTradeActionValidation:
    """Test trade action validation with functional patterns."""

    def test_validate_trade_action_success(self):
        """Test successful trade action validation."""
        action_data = {
            "action": "LONG",
            "size_pct": 25.0,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "leverage": 5,
            "rationale": "Strong bullish signal",
        }

        result = validate_trade_action_functional(action_data)
        assert result.is_success()
        trade_action = result.success()
        assert isinstance(trade_action, TradeAction)
        assert trade_action.action == "LONG"
        assert trade_action.size_pct == 25.0

    def test_validate_trade_action_invalid_action(self):
        """Test trade action validation with invalid action."""
        action_data = {
            "action": "INVALID",
            "size_pct": 25.0,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "rationale": "Test",
        }

        result = validate_trade_action_functional(action_data)
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, SchemaError)
        assert any(e.field == "action" for e in error.errors)

    def test_validate_trade_action_negative_size(self):
        """Test trade action validation with negative size."""
        action_data = {
            "action": "LONG",
            "size_pct": -25.0,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "rationale": "Test",
        }

        result = validate_trade_action_functional(action_data)
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, SchemaError)
        assert any(
            e.field == "size_pct" and "positive" in e.message.lower()
            for e in error.errors
        )

    def test_validate_trade_action_oversized_position(self):
        """Test trade action validation with oversized position."""
        action_data = {
            "action": "LONG",
            "size_pct": 150.0,  # Over 100%
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "rationale": "Test",
        }

        result = validate_trade_action_functional(action_data)
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, SchemaError)
        assert any(
            e.field == "size_pct" and "between 0 and 100" in e.message
            for e in error.errors
        )

    def test_validate_trade_action_missing_field(self):
        """Test trade action validation with missing required field."""
        action_data = {
            "action": "LONG",
            "size_pct": 25.0,
            # Missing required fields
        }

        result = validate_trade_action_functional(action_data)
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, SchemaError)
        assert len(error.errors) > 0

    def test_validate_trade_action_case_insensitive(self):
        """Test trade action validation is case insensitive."""
        action_data = {
            "action": "long",  # lowercase
            "size_pct": 25.0,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "leverage": 5,
            "rationale": "Test",
        }

        result = validate_trade_action_functional(action_data)
        assert result.is_success()
        trade_action = result.success()
        assert trade_action.action == "LONG"  # Should be converted to uppercase


class TestPositionValidation:
    """Test position validation with functional patterns."""

    def test_validate_position_success(self):
        """Test successful position validation."""
        position_data = {
            "symbol": "BTC-USD",
            "side": "LONG",
            "size": 1.5,
            "entry_price": 45000.0,
            "unrealized_pnl": 150.0,
            "realized_pnl": 0.0,
        }

        result = validate_position_functional(position_data)
        assert result.is_success()
        validated_data = result.success()
        assert validated_data["symbol"] == "BTC-USD"
        assert validated_data["side"] == "LONG"

    def test_validate_position_invalid_side(self):
        """Test position validation with invalid side."""
        position_data = {"symbol": "BTC-USD", "side": "INVALID", "size": 1.5}

        result = validate_position_functional(position_data)
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, SchemaError)
        assert any(e.field == "side" for e in error.errors)

    def test_validate_position_optional_fields(self):
        """Test position validation with optional fields."""
        position_data = {
            "symbol": "BTC-USD",
            "side": "FLAT",
            "size": 0.0,
            # Optional fields omitted
        }

        result = validate_position_functional(position_data)
        assert result.is_success()


class TestBatchValidation:
    """Test batch validation functionality."""

    def test_validate_batch_all_success(self):
        """Test batch validation with all items succeeding."""
        items = [10.0, 20.0, 30.0]
        validator = lambda x: validate_range(0, 100)(x, "value")

        result = validate_batch_functional(items, validator)
        assert result.is_success()
        validated_items = result.success()
        assert validated_items == items

    def test_validate_batch_some_failures(self):
        """Test batch validation with some failures."""
        items = [10.0, 150.0, 30.0]  # 150.0 should fail range validation
        validator = lambda x: validate_range(0, 100)(x, "value")

        result = validate_batch_functional(items, validator)
        assert result.is_failure()
        errors = result.failure()
        assert isinstance(errors, list)
        assert len(errors) == 1
        assert errors[0][0] == 1  # Index of failed item
        assert isinstance(errors[0][1], FieldError)

    def test_validate_batch_empty_list(self):
        """Test batch validation with empty list."""
        items = []
        validator = lambda x: validate_positive(x, "value")

        result = validate_batch_functional(items, validator)
        assert result.is_success()
        assert result.success() == []


class TestDataIntegrity:
    """Test data integrity validation."""

    def test_validate_data_integrity_success(self):
        """Test successful data integrity validation."""
        data = {
            "price": 45000.0,
            "volume": 1000.0,
            "timestamp": datetime.now().timestamp(),
        }

        integrity_rules = {
            "price": lambda x: x > 0,
            "volume": lambda x: x >= 0,
            "timestamp": lambda x: x > 0,
        }

        result = validate_data_integrity(data, integrity_rules)
        assert result.is_success()
        assert result.success() == data

    def test_validate_data_integrity_failure(self):
        """Test data integrity validation failure."""
        data = {
            "price": -45000.0,  # Should fail positive check
            "volume": 1000.0,
        }

        integrity_rules = {"price": lambda x: x > 0, "volume": lambda x: x >= 0}

        result = validate_data_integrity(data, integrity_rules)
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, SchemaError)
        assert any(e.field == "price" for e in error.errors)

    def test_validate_cross_field_constraints_success(self):
        """Test successful cross-field constraint validation."""
        data = {"stop_loss_pct": 1.0, "take_profit_pct": 2.0}

        constraints = [
            lambda d: d["take_profit_pct"] > d["stop_loss_pct"]  # TP should be > SL
        ]

        result = validate_cross_field_constraints(data, constraints)
        assert result.is_success()
        assert result.success() == data

    def test_validate_cross_field_constraints_failure(self):
        """Test cross-field constraint validation failure."""
        data = {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 1.0,  # TP < SL, should fail
        }

        constraints = [lambda d: d["take_profit_pct"] > d["stop_loss_pct"]]

        result = validate_cross_field_constraints(data, constraints)
        assert result.is_failure()
        error = result.failure()
        assert isinstance(error, SchemaError)
        assert any(e.field == "cross_field" for e in error.errors)


class TestJSONSchemaIntegration:
    """Test integration with JSON schema validation patterns."""

    def test_validate_json_trade_action(self):
        """Test validating JSON trade action input."""
        json_data = """
        {
            "action": "LONG",
            "size_pct": 25.0,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "leverage": 5,
            "rationale": "Strong bullish momentum"
        }
        """

        # Parse JSON and validate
        try:
            data = json.loads(json_data)
            result = validate_trade_action_functional(data)
            assert result.is_success()
            trade_action = result.success()
            assert trade_action.action == "LONG"
        except json.JSONDecodeError:
            pytest.fail("Valid JSON should parse successfully")

    def test_validate_malformed_json(self):
        """Test handling malformed JSON."""
        malformed_json = """
        {
            "action": "LONG",
            "size_pct": 25.0,
            "rationale": "Missing closing brace"
        """

        # Should fail at JSON parsing stage
        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_json)

    def test_validate_json_with_extra_fields(self):
        """Test validation with extra fields in JSON."""
        json_data = """
        {
            "action": "LONG",
            "size_pct": 25.0,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "leverage": 5,
            "rationale": "Test",
            "extra_field": "should_be_ignored"
        }
        """

        data = json.loads(json_data)
        result = validate_trade_action_functional(data)
        assert result.is_success()
        # Extra fields should be handled gracefully

    def test_complete_validation_pipeline_with_json(self):
        """Test complete validation pipeline with JSON input."""
        json_data = """
        {
            "action": "SHORT",
            "size_pct": 15.0,
            "take_profit_pct": 1.5,
            "stop_loss_pct": 0.8,
            "leverage": 3,
            "rationale": "Bearish divergence detected"
        }
        """

        # Create complete pipeline
        pipeline = ValidationPipeline()
        pipeline.add_step("parse_json", lambda x: Success(json.loads(x)))
        pipeline.add_step("validate_schema", validate_trade_action_functional)

        result = pipeline.validate(json_data)
        assert result.is_success()
        trade_action = result.success()
        assert isinstance(trade_action, TradeAction)
        assert trade_action.action == "SHORT"


class TestErrorHandling:
    """Test comprehensive error handling and reporting."""

    def test_field_error_representation(self):
        """Test FieldError string representation."""
        error = FieldError(
            field="test_field",
            message="Test error message",
            value="test_value",
            context={"key": "value"},
        )

        error_str = str(error)
        assert "test_field" in error_str
        assert "Test error message" in error_str
        assert "context:" in error_str

    def test_schema_error_representation(self):
        """Test SchemaError string representation."""
        field_errors = [
            FieldError(field="field1", message="Error 1"),
            FieldError(field="field2", message="Error 2"),
        ]

        schema_error = SchemaError(
            schema="test_schema", errors=field_errors, path=["root", "nested"]
        )

        error_str = str(schema_error)
        assert "test_schema" in error_str
        assert "root.nested" in error_str

    def test_conversion_error_representation(self):
        """Test ConversionError string representation."""
        error = ConversionError(
            source_type="str",
            target_type="Decimal",
            message="Invalid number format",
            conversion_path=["parse", "validate", "convert"],
        )

        error_str = str(error)
        assert "str -> Decimal" in error_str
        assert "parse -> validate -> convert" in error_str

    def test_validation_chain_error_representation(self):
        """Test ValidationChainError string representation."""
        field_error = FieldError(field="test", message="Test error")
        chain_error = ValidationChainError(chain_step="step1", errors=[field_error])

        error_str = str(chain_error)
        assert "step1" in error_str
        assert "1 error(s)" in error_str


class TestBackwardCompatibility:
    """Test backward compatibility with existing validation systems."""

    def test_legacy_trade_action_validation(self):
        """Test validation of legacy TradeAction objects."""
        # Create a legacy-style dict as might come from existing code
        legacy_data = {
            "action": "LONG",
            "size_pct": 20,
            "take_profit_pct": 2.5,
            "stop_loss_pct": 1.5,
            "rationale": "Legacy validation test",
        }

        result = validate_trade_action_functional(legacy_data)
        assert result.is_success()
        trade_action = result.success()
        assert isinstance(trade_action, TradeAction)

    def test_migration_from_imperative_validation(self):
        """Test migration patterns from imperative to functional validation."""
        # Simulate data that might come from the old validator
        old_style_result = {
            "action": "SHORT",
            "size_pct": 15.0,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "leverage": 5,
            "rationale": "Migration test",
        }

        # Should be able to validate using new functional approach
        result = validate_trade_action_functional(old_style_result)
        assert result.is_success()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
