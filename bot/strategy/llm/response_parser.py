"""
Response parser for extracting and validating JSON trading decisions from LLM responses.

This module handles parsing of trading decisions from verbose LLM responses that may contain
both analysis text and structured JSON. It provides robust error handling and fallback
mechanisms to ensure the trading bot can always derive a valid action.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from bot.trading_types import TradeAction

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parser for extracting and validating trading decisions from LLM responses."""

    def __init__(self) -> None:
        """Initialize the response parser."""
        self.json_decision_pattern = re.compile(
            r"JSON_DECISION:\s*(\{.*?\})", re.DOTALL
        )
        self.fallback_json_pattern = re.compile(r'\{[^{}]*"action"[^{}]*\}', re.DOTALL)

        # Individual field patterns for manual parsing
        self.field_patterns = {
            "action": re.compile(r'"action":\s*"([^"]+)"'),
            "size_pct": re.compile(r'"size_pct":\s*(\d+(?:\.\d+)?)'),
            "take_profit_pct": re.compile(r'"take_profit_pct":\s*(\d+(?:\.\d+)?)'),
            "stop_loss_pct": re.compile(r'"stop_loss_pct":\s*(\d+(?:\.\d+)?)'),
            "leverage": re.compile(r'"leverage":\s*(\d+)'),
            "reduce_only": re.compile(r'"reduce_only":\s*(true|false)'),
            "rationale": re.compile(r'"rationale":\s*"([^"]+)"'),
        }

    def extract_json_from_response(self, response_content: str) -> dict[str, Any]:
        """
        Extract JSON from a verbose response that contains both analysis and JSON.

        This method attempts multiple strategies to extract JSON:
        1. Look for explicit JSON_DECISION: marker
        2. Search for any JSON-like structure with "action" field
        3. Parse individual fields and construct JSON
        4. Return safe HOLD default if all parsing fails

        Args:
            response_content: Full response content from LLM

        Returns:
            Dictionary containing the JSON decision with required trading fields
        """
        try:
            # Strategy 1: Look for JSON_DECISION: marker
            match = self.json_decision_pattern.search(response_content)
            if match:
                json_str = match.group(1).strip()
                logger.debug("Found JSON with JSON_DECISION marker")
                return json.loads(json_str)

            # Strategy 2: Look for any JSON-like structure with "action" field
            match = self.fallback_json_pattern.search(response_content)
            if match:
                json_str = match.group(0)
                logger.debug("Found JSON-like structure in response")
                return json.loads(json_str)

            # Strategy 3: Manual field extraction
            logger.warning(
                "Could not find JSON in response, attempting to parse manually"
            )
            return self._extract_fields_manually(response_content)

        except json.JSONDecodeError as e:
            logger.exception(f"JSON decode error: {e}")
            return self._get_safe_default_action("JSON parsing error")
        except Exception as e:
            logger.exception(f"Unexpected error parsing response: {e}")
            return self._get_safe_default_action("Unexpected parsing error")

    def _extract_fields_manually(self, response_content: str) -> dict[str, Any]:
        """
        Extract individual fields from response and construct JSON.

        Args:
            response_content: Full response content from LLM

        Returns:
            Dictionary with extracted fields or safe default
        """
        extracted_fields = {}

        # Extract each field using regex patterns
        for field_name, pattern in self.field_patterns.items():
            match = pattern.search(response_content)
            if match:
                value = match.group(1)

                # Convert to appropriate type
                if field_name == "action":
                    extracted_fields[field_name] = value
                elif field_name in ["size_pct", "take_profit_pct", "stop_loss_pct"]:
                    extracted_fields[field_name] = float(value)
                elif field_name == "leverage":
                    extracted_fields[field_name] = int(value)
                elif field_name == "reduce_only":
                    extracted_fields[field_name] = value.lower() == "true"
                elif field_name == "rationale":
                    extracted_fields[field_name] = value

        # Check if we at least have an action
        if "action" in extracted_fields:
            logger.info(
                f"Successfully extracted fields: {list(extracted_fields.keys())}"
            )

            # Fill in missing fields with defaults
            return {
                "action": extracted_fields.get("action"),
                "size_pct": extracted_fields.get("size_pct", 0),
                "take_profit_pct": extracted_fields.get("take_profit_pct", 2.0),
                "stop_loss_pct": extracted_fields.get("stop_loss_pct", 1.0),
                "leverage": extracted_fields.get("leverage", 1),
                "reduce_only": extracted_fields.get("reduce_only", False),
                "rationale": extracted_fields.get(
                    "rationale", "Parsed from verbose response"
                ),
            }

        # No action found, return safe default
        logger.error("Could not parse any decision from response, defaulting to HOLD")
        return self._get_safe_default_action("Failed to parse response")

    def parse_trade_action(self, json_data: dict[str, Any]) -> TradeAction | None:
        """
        Parse and validate JSON data into a TradeAction model.

        Args:
            json_data: Dictionary containing trading decision fields

        Returns:
            TradeAction instance if valid, None if validation fails
        """
        try:
            # Ensure all required fields are present
            required_fields = [
                "action",
                "size_pct",
                "take_profit_pct",
                "stop_loss_pct",
                "rationale",
            ]

            for field in required_fields:
                if field not in json_data:
                    logger.error(f"Missing required field: {field}")
                    return None

            # Create and validate TradeAction
            trade_action = TradeAction(
                action=json_data["action"],
                size_pct=json_data["size_pct"],
                take_profit_pct=json_data["take_profit_pct"],
                stop_loss_pct=json_data["stop_loss_pct"],
                rationale=json_data["rationale"],
                leverage=json_data.get("leverage", 1),
                reduce_only=json_data.get("reduce_only", False),
            )

            logger.info(f"Successfully parsed trade action: {trade_action.action}")
            return trade_action

        except ValueError as e:
            logger.exception(f"Validation error creating TradeAction: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error creating TradeAction: {e}")
            return None

    def handle_parsing_errors(
        self, error: Exception, response_content: str | None = None
    ) -> dict[str, Any]:
        """
        Handle parsing errors and return a safe default action.

        Args:
            error: The exception that occurred during parsing
            response_content: Optional original response for additional context

        Returns:
            Safe default HOLD action as dictionary
        """
        error_type = type(error).__name__
        error_msg = str(error)

        logger.error(
            f"Parsing error ({error_type}): {error_msg}",
            extra={
                "response_preview": (
                    response_content[:200] if response_content else "N/A"
                )
            },
        )

        # Log specific error handling
        if isinstance(error, json.JSONDecodeError):
            logger.error(f"JSON syntax error at position {error.pos}")
        elif isinstance(error, KeyError):
            logger.error(f"Missing key in response: {error_msg}")
        elif isinstance(error, ValueError):
            logger.error(f"Invalid value in response: {error_msg}")

        # Return safe default
        return self._get_safe_default_action(f"Parsing error: {error_type}")

    def _get_safe_default_action(self, reason: str) -> dict[str, Any]:
        """
        Get a safe default HOLD action.

        Args:
            reason: Reason for returning default action

        Returns:
            Dictionary with safe HOLD action
        """
        return {
            "action": "HOLD",
            "size_pct": 0,
            "take_profit_pct": 1.0,
            "stop_loss_pct": 1.0,
            "leverage": 1,
            "reduce_only": False,
            "rationale": reason,
        }
