"""Unit tests for trade action validator."""

import json

from bot.types import TradeAction
from bot.validator import TradeValidator


class TestTradeValidator:
    """Test cases for the trade validator."""

    def test_validator_initialization(self) -> None:
        """Test validator initialization."""
        validator = TradeValidator()

        assert validator.max_size_pct == 20  # From settings
        assert validator.max_tp_pct == 20.0
        assert validator.max_sl_pct == 10.0
        assert validator.valid_actions == {"LONG", "SHORT", "CLOSE", "HOLD"}

    def test_validate_valid_trade_action(self) -> None:
        """Test validation of a valid trade action."""
        validator = TradeValidator()

        valid_action = TradeAction(
            action="LONG",
            size_pct=15,
            take_profit_pct=2.5,
            stop_loss_pct=1.5,
            rationale="Valid test trade",
        )

        result = validator.validate(valid_action)

        assert result.action == "LONG"
        assert result.size_pct == 15
        assert result.take_profit_pct == 2.5
        assert result.stop_loss_pct == 1.5

    def test_validate_oversized_position(self) -> None:
        """Test validation caps oversized positions."""
        validator = TradeValidator()

        oversized_action = TradeAction(
            action="LONG",
            size_pct=50,  # Over the 20% limit
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Oversized position test",
        )

        result = validator.validate(oversized_action)

        assert result.size_pct == 20  # Should be capped

    def test_validate_json_string(self) -> None:
        """Test validation of JSON string input."""
        validator = TradeValidator()

        json_input = json.dumps(
            {
                "action": "SHORT",
                "size_pct": 10,
                "take_profit_pct": 3.0,
                "stop_loss_pct": 2.0,
                "rationale": "JSON input test",
            }
        )

        result = validator.validate(json_input)

        assert result.action == "SHORT"
        assert result.size_pct == 10

    def test_validate_invalid_json(self) -> None:
        """Test validation with invalid JSON returns HOLD."""
        validator = TradeValidator()

        invalid_json = "invalid json string"

        result = validator.validate(invalid_json)

        assert result.action == "HOLD"
        assert result.size_pct == 0
        assert "Validation error" in result.rationale

    def test_validate_invalid_action(self) -> None:
        """Test validation with invalid action returns HOLD."""
        validator = TradeValidator()

        invalid_dict = {
            "action": "INVALID",
            "size_pct": 10,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "rationale": "Invalid action test",
        }

        result = validator.validate(invalid_dict)

        assert result.action == "HOLD"
        assert result.size_pct == 0

    def test_validate_negative_values(self) -> None:
        """Test validation with negative values returns HOLD."""
        validator = TradeValidator()

        negative_dict = {
            "action": "LONG",
            "size_pct": -10,
            "take_profit_pct": -2.0,
            "stop_loss_pct": 1.0,
            "rationale": "Negative values test",
        }

        result = validator.validate(negative_dict)

        assert result.action == "HOLD"
        assert result.size_pct == 0

    def test_sanitize_llm_output(self) -> None:
        """Test LLM output sanitization."""
        validator = TradeValidator()

        # Test with markdown code blocks
        markdown_output = """```json
        {
            "action": "LONG",
            "size_pct": 15,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.5,
            "rationale": "Test output"
        }
        ```"""

        sanitized = validator.sanitize_llm_output(markdown_output)

        # Should be valid JSON without markdown
        parsed = json.loads(sanitized)
        assert parsed["action"] == "LONG"

    def test_validate_hold_action_zero_size(self) -> None:
        """Test that HOLD and CLOSE actions have zero size."""
        validator = TradeValidator()

        hold_action = TradeAction(
            action="HOLD",
            size_pct=10,  # Should be reset to 0
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Hold action test",
        )

        result = validator.validate(hold_action)

        assert result.action == "HOLD"
        assert result.size_pct == 0

    def test_validate_risk_reward_ratio(self) -> None:
        """Test risk-reward ratio validation."""
        validator = TradeValidator()

        poor_rr_action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=0.5,  # Poor risk-reward
            stop_loss_pct=2.0,
            rationale="Poor risk-reward test",
        )

        result = validator.validate(poor_rr_action)

        # Should adjust take profit to improve ratio
        assert result.take_profit_pct >= 2.0

    def test_validate_excessive_take_profit(self) -> None:
        """Test validation caps excessive take profit."""
        validator = TradeValidator()

        excessive_tp = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=50.0,  # Excessive
            stop_loss_pct=1.0,
            rationale="Excessive TP test",
        )

        result = validator.validate(excessive_tp)

        assert result.take_profit_pct <= 20.0  # Should be capped

    def test_get_validation_summary(self) -> None:
        """Test validation summary generation."""
        validator = TradeValidator()

        original_dict = {
            "action": "LONG",
            "size_pct": 30,  # Will be capped
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "rationale": "Summary test",
        }

        validated = validator.validate(original_dict)
        summary = validator.get_validation_summary(original_dict, validated)

        assert summary["validation_successful"] is True
        assert len(summary["modifications"]) > 0
        assert summary["modifications"][0]["field"] == "size_pct"
