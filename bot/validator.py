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
from .trading_types import Position, TradeAction

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

    def validate(
        self,
        llm_output: str | dict[str, Any] | TradeAction,
        current_position: Position | None = None,
    ) -> TradeAction:
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
            validated_action = self._validate_trade_action(
                trade_action, current_position
            )

        except Exception as e:
            logger.exception("Validation failed")
            return self._get_default_hold_action(f"Validation error: {e!s}")
        else:
            logger.info("Validated trade action: %s", validated_action.action)
            return validated_action

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

        if isinstance(llm_output, dict):
            # Normalize action field before creating TradeAction
            normalized_dict = self._normalize_action_dict(llm_output)
            return TradeAction(**normalized_dict)

        if isinstance(llm_output, str):
            # Try to parse as JSON
            try:
                parsed_json = json.loads(llm_output.strip())
                # Normalize action field before creating TradeAction
                normalized_dict = self._normalize_action_dict(parsed_json)
                return TradeAction(**normalized_dict)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}") from e

        raise TypeError(f"Unsupported LLM output type: {type(llm_output)}")

    def _normalize_action_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize action field in dictionary to handle common variations.

        Args:
            data: Dictionary containing action field

        Returns:
            Dictionary with normalized action field
        """
        normalized = data.copy()

        if "action" in normalized:
            action = str(normalized["action"]).upper().strip()

            # Map common variations to standard actions
            action_mappings = {
                "BUY": "LONG",
                "SELL": "SHORT",
                "EXIT": "CLOSE",
                "CLOSE_POSITION": "CLOSE",
                "STAY": "HOLD",
                "WAIT": "HOLD",
                "NO_ACTION": "HOLD",
            }

            if action in action_mappings:
                original_action = normalized["action"]
                normalized["action"] = action_mappings[action]
                logger.info(
                    "Normalized action '%s' to '%s'",
                    original_action,
                    normalized["action"],
                )
            else:
                normalized["action"] = action

        return normalized

    def _validate_trade_action(
        self, action: TradeAction, current_position: Position | None = None
    ) -> TradeAction:
        """
        Validate and potentially modify a trade action.

        Args:
            action: Trade action to validate
            current_position: Current position (if any)

        Returns:
            Validated trade action

        Raises:
            ValidationError: If action cannot be validated
        """
        validated = action.model_copy()

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
                "Set default position size: %s%% for %s",
                validated.size_pct,
                validated.action,
            )

        if validated.size_pct > self.max_size_pct:
            logger.warning(
                "Size capped from %s%% to %s%%", validated.size_pct, self.max_size_pct
            )
            validated.size_pct = self.max_size_pct

        # Validate take profit (allow 0 for HOLD/CLOSE actions)
        if validated.action not in ["HOLD", "CLOSE"] and validated.take_profit_pct <= 0:
            raise ValueError(
                f"Take profit must be positive for {validated.action}: {validated.take_profit_pct}"
            )

        if validated.take_profit_pct > self.max_tp_pct:
            logger.warning(
                "Take profit capped from %s%% to %s%%",
                validated.take_profit_pct,
                self.max_tp_pct,
            )
            validated.take_profit_pct = self.max_tp_pct

        # Validate stop loss (allow 0 for HOLD/CLOSE actions)
        if validated.action not in ["HOLD", "CLOSE"] and validated.stop_loss_pct <= 0:
            raise ValueError(
                f"Stop loss must be positive for {validated.action}: {validated.stop_loss_pct}"
            )

        if validated.stop_loss_pct > self.max_sl_pct:
            logger.warning(
                "Stop loss capped from %s%% to %s%%",
                validated.stop_loss_pct,
                self.max_sl_pct,
            )
            validated.stop_loss_pct = self.max_sl_pct

        # Validate leverage for futures trading
        if validated.leverage < 1:
            logger.warning(
                "Invalid leverage %s, setting to default 1", validated.leverage
            )
            validated.leverage = 1
        elif validated.leverage > 100:
            logger.warning("Excessive leverage %s, capping to 100", validated.leverage)
            validated.leverage = 100

        # Validate rationale
        if not validated.rationale or len(validated.rationale.strip()) == 0:
            validated.rationale = "No rationale provided"

        if len(validated.rationale) > 500:
            validated.rationale = validated.rationale[:497] + "..."
            logger.warning("Rationale truncated to 500 characters")

        # Business rule validations
        return self._apply_business_rules(validated, current_position)

    def _apply_business_rules(
        self, action: TradeAction, current_position: Position | None = None
    ) -> TradeAction:
        """
        Apply business rules and risk management constraints.

        Args:
            action: Trade action to validate
            current_position: Current position (if any)

        Returns:
            Modified trade action after applying business rules
        """
        validated = action.model_copy()

        # ENFORCE SINGLE POSITION RULE
        if (
            current_position
            and current_position.side != "FLAT"
            and validated.action in ["LONG", "SHORT"]
        ):
            logger.warning(
                "Cannot open new %s position - existing %s position. Changing to HOLD.",
                validated.action,
                current_position.side,
            )
            return self._get_default_hold_action(
                f"Position exists ({current_position.side}) - only CLOSE or HOLD allowed"
            )

        # If action is HOLD or CLOSE, size should be 0
        if validated.action in ["HOLD", "CLOSE"] and validated.size_pct > 0:
            logger.warning("Size reset to 0 for %s action", validated.action)
            validated.size_pct = 0

        # Ensure reasonable risk-reward ratios (skip for HOLD/CLOSE actions)
        if validated.action not in ["HOLD", "CLOSE"] and validated.stop_loss_pct > 0:
            risk_reward_ratio = validated.take_profit_pct / validated.stop_loss_pct

            if risk_reward_ratio < 0.5:
                logger.warning("Poor risk-reward ratio detected, adjusting levels")
                validated.take_profit_pct = max(validated.stop_loss_pct * 1.0, 1.0)
            elif risk_reward_ratio > 10.0:
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
        # Truncate reason to fit within rationale max_length (500 chars)
        # Account for "Validator: " prefix (11 chars)
        max_reason_length = 500 - 11
        truncated_reason = (
            reason[:max_reason_length] if len(reason) > max_reason_length else reason
        )

        return TradeAction(
            action="HOLD",
            size_pct=0,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            leverage=1,
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
        except ValidationError:
            return False
        else:
            return True

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
        cleaned = cleaned.removeprefix("```json")
        cleaned = cleaned.removeprefix("```")
        cleaned = cleaned.removesuffix("```")

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
        self, original: str | dict[str, Any], validated: TradeAction
    ) -> dict[str, Any]:
        """
        Get a summary of validation changes.

        Args:
            original: Original LLM output
            validated: Validated trade action

        Returns:
            Dictionary with validation summary
        """
        modifications: list[dict[str, Any]] = []
        summary = {
            "original_type": type(original).__name__,
            "validated_action": validated.model_dump(),
            "modifications": modifications,
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
                        modifications.append(
                            {
                                "field": key,
                                "original": original_value,
                                "validated": validated_value,
                            }
                        )

        except Exception as e:
            summary["comparison_error"] = str(e)

        return summary
