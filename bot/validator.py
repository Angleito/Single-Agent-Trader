"""
Trade action validation and schema enforcement.

This module validates LLM outputs and ensures all trade actions
conform to required schemas and business rules.
"""

import json
import logging
from typing import Any

from pydantic import ValidationError

from .config import settings
from .types import TradeAction

logger = logging.getLogger(__name__)


class TradeValidator:
    """
    Validates trade actions from LLM output and enforces business rules.

    Ensures all trade actions are valid, safe, and conform to risk parameters
    before execution. Defaults to HOLD on any validation failures.
    """

    def __init__(self):
        """Initialize the trade validator."""
        self.max_size_pct = settings.trading.max_size_pct
        self.max_tp_pct = 20.0  # Maximum take profit percentage
        self.max_sl_pct = 10.0  # Maximum stop loss percentage
        self.valid_actions = {"LONG", "SHORT", "CLOSE", "HOLD"}

        logger.info("Initialized TradeValidator")

    def validate(self, llm_output: str | dict[str, Any] | TradeAction) -> TradeAction:
        """
        Validate LLM output and return a valid TradeAction.

        Args:
            llm_output: Raw LLM output (JSON string, dict, or TradeAction)

        Returns:
            Valid TradeAction object (defaults to HOLD on errors)
        """
        try:
            # Convert to TradeAction if needed
            trade_action = self._parse_llm_output(llm_output)

            # Validate the trade action
            validated_action = self._validate_trade_action(trade_action)

            logger.info(f"Validated trade action: {validated_action.action}")
            return validated_action

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self._get_default_hold_action(f"Validation error: {str(e)}")

    def _parse_llm_output(
        self, llm_output: str | dict[str, Any] | TradeAction
    ) -> TradeAction:
        """
        Parse LLM output into a TradeAction object.

        Args:
            llm_output: Raw LLM output in various formats

        Returns:
            TradeAction object

        Raises:
            ValidationError: If parsing fails
        """
        if isinstance(llm_output, TradeAction):
            return llm_output

        elif isinstance(llm_output, dict):
            return TradeAction(**llm_output)

        elif isinstance(llm_output, str):
            # Try to parse as JSON
            try:
                parsed_json = json.loads(llm_output.strip())
                return TradeAction(**parsed_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}") from e

        else:
            raise ValueError(f"Unsupported LLM output type: {type(llm_output)}")

    def _validate_trade_action(self, action: TradeAction) -> TradeAction:
        """
        Validate and potentially modify a trade action.

        Args:
            action: Trade action to validate

        Returns:
            Validated trade action

        Raises:
            ValidationError: If action cannot be validated
        """
        validated = action.copy()

        # Validate action type
        if validated.action not in self.valid_actions:
            raise ValueError(f"Invalid action: {validated.action}")

        # Validate size percentage
        if validated.size_pct < 0 or validated.size_pct > 100:
            raise ValueError(f"Size percentage out of range: {validated.size_pct}")

        # Set default size for LONG/SHORT actions if not provided
        if validated.action in ["LONG", "SHORT"] and validated.size_pct == 0:
            validated.size_pct = 5  # Default 5% position size
            logger.info(
                f"Set default position size: {validated.size_pct}% for {validated.action}"
            )

        if validated.size_pct > self.max_size_pct:
            logger.warning(
                f"Size capped from {validated.size_pct}% to {self.max_size_pct}%"
            )
            validated.size_pct = self.max_size_pct

        # Validate take profit (allow 0 for HOLD/CLOSE actions)
        if validated.action not in ["HOLD", "CLOSE"] and validated.take_profit_pct <= 0:
            raise ValueError(
                f"Take profit must be positive for {validated.action}: {validated.take_profit_pct}"
            )

        if validated.take_profit_pct > self.max_tp_pct:
            logger.warning(
                f"Take profit capped from {validated.take_profit_pct}% to {self.max_tp_pct}%"
            )
            validated.take_profit_pct = self.max_tp_pct

        # Validate stop loss (allow 0 for HOLD/CLOSE actions)
        if validated.action not in ["HOLD", "CLOSE"] and validated.stop_loss_pct <= 0:
            raise ValueError(
                f"Stop loss must be positive for {validated.action}: {validated.stop_loss_pct}"
            )

        if validated.stop_loss_pct > self.max_sl_pct:
            logger.warning(
                f"Stop loss capped from {validated.stop_loss_pct}% to {self.max_sl_pct}%"
            )
            validated.stop_loss_pct = self.max_sl_pct

        # Validate rationale
        if not validated.rationale or len(validated.rationale.strip()) == 0:
            validated.rationale = "No rationale provided"

        if len(validated.rationale) > 200:
            validated.rationale = validated.rationale[:197] + "..."
            logger.warning("Rationale truncated to 200 characters")

        # Business rule validations
        validated = self._apply_business_rules(validated)

        return validated

    def _apply_business_rules(self, action: TradeAction) -> TradeAction:
        """
        Apply business rules and risk management constraints.

        Args:
            action: Trade action to validate

        Returns:
            Modified trade action after applying business rules
        """
        validated = action.copy()

        # If action is HOLD or CLOSE, size should be 0
        if validated.action in ["HOLD", "CLOSE"]:
            if validated.size_pct > 0:
                logger.warning(f"Size reset to 0 for {validated.action} action")
                validated.size_pct = 0

        # Ensure reasonable risk-reward ratios (skip for HOLD/CLOSE actions)
        if validated.action not in ["HOLD", "CLOSE"] and validated.stop_loss_pct > 0:
            risk_reward_ratio = validated.take_profit_pct / validated.stop_loss_pct

            if risk_reward_ratio < 0.5:
                logger.warning("Poor risk-reward ratio detected, adjusting levels")
                validated.take_profit_pct = max(validated.stop_loss_pct * 1.0, 1.0)

            if risk_reward_ratio > 10.0:
                logger.warning("Excessive risk-reward ratio detected, adjusting levels")
                validated.stop_loss_pct = max(validated.take_profit_pct / 5.0, 0.5)

        return validated

    def _get_default_hold_action(
        self, reason: str = "Validation failed"
    ) -> TradeAction:
        """
        Get a safe default HOLD action.

        Args:
            reason: Reason for defaulting to HOLD

        Returns:
            Safe TradeAction with HOLD
        """
        # Truncate reason to fit within rationale max_length (200 chars)
        # Account for "Validator: " prefix (11 chars)
        max_reason_length = 200 - 11
        truncated_reason = (
            reason[:max_reason_length] if len(reason) > max_reason_length else reason
        )

        return TradeAction(
            action="HOLD",
            size_pct=0,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale=f"Validator: {truncated_reason}",
        )

    def validate_json_schema(self, json_data: dict[str, Any]) -> bool:
        """
        Validate JSON data against TradeAction schema.

        Args:
            json_data: JSON data to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            TradeAction(**json_data)
            return True
        except ValidationError:
            return False

    def sanitize_llm_output(self, raw_output: str) -> str:
        """
        Clean and sanitize raw LLM output.

        Args:
            raw_output: Raw text from LLM

        Returns:
            Cleaned JSON string
        """
        # Remove common LLM prefixes/suffixes
        cleaned = raw_output.strip()

        # Remove markdown code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        # Remove any text before the first {
        brace_index = cleaned.find("{")
        if brace_index > 0:
            cleaned = cleaned[brace_index:]

        # Remove any text after the last }
        last_brace_index = cleaned.rfind("}")
        if last_brace_index > 0:
            cleaned = cleaned[: last_brace_index + 1]

        return cleaned.strip()

    def get_validation_summary(
        self, original: str | dict, validated: TradeAction
    ) -> dict[str, Any]:
        """
        Get a summary of validation changes.

        Args:
            original: Original LLM output
            validated: Validated trade action

        Returns:
            Dictionary with validation summary
        """
        summary = {
            "original_type": type(original).__name__,
            "validated_action": validated.dict(),
            "modifications": [],
            "validation_successful": True,
        }

        # Try to compare with original if possible
        try:
            if isinstance(original, str):
                original_parsed = json.loads(self.sanitize_llm_output(original))
            elif isinstance(original, dict):
                original_parsed = original
            else:
                original_parsed = {}

            # Check for modifications
            for key in ["action", "size_pct", "take_profit_pct", "stop_loss_pct"]:
                if key in original_parsed:
                    original_value = original_parsed[key]
                    validated_value = getattr(validated, key)
                    if original_value != validated_value:
                        summary["modifications"].append(
                            {
                                "field": key,
                                "original": original_value,
                                "validated": validated_value,
                            }
                        )

        except Exception as e:
            summary["comparison_error"] = str(e)

        return summary
