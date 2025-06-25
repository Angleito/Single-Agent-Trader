"""Simplified LLM Agent orchestrating market analysis, prompt formatting, and response parsing."""

import logging
from pathlib import Path
from typing import Any

from bot.config import settings
from bot.trading_types import MarketState, TradeAction

from .market_analyzer import MarketAnalyzer
from .prompt_manager import PromptManager
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)


class LLMAgent:
    """Orchestrates LLM-based trading decisions using modular components."""

    def __init__(
        self,
        model_provider: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        prompt_file: Path | None = None,
    ):
        """
        Initialize the LLM Agent with model settings and components.

        Args:
            model_provider: LLM provider (e.g., 'openai', 'ollama')
            model_name: Specific model to use (e.g., 'gpt-4')
            temperature: Model temperature for response variability
            prompt_file: Optional custom prompt template file
        """
        # Model configuration
        self.model_provider = model_provider or settings.llm.provider
        self.model_name = model_name or settings.llm.model_name
        self.temperature = temperature or settings.llm.temperature

        # Handle special model requirements
        if self.model_name and self.model_name.startswith("o3"):
            self.temperature = 1.0  # o3 models require temperature=1.0

        # Initialize components
        self.prompt_manager = PromptManager(prompt_file_path=prompt_file)
        self.response_parser = ResponseParser()
        self.market_analyzer = MarketAnalyzer()

        # Track agent stats
        self._decision_count = 0
        self._error_count = 0

        logger.info(
            f"Initialized LLMAgent with {self.model_provider}/{self.model_name}"
        )

    async def analyze_market(
        self,
        market_state: MarketState,
        omnisearch_client: Any | None = None,
        omnisearch_enabled: bool = False,
    ) -> TradeAction:
        """
        Analyze market conditions and generate a trading decision.

        Args:
            market_state: Current market data and indicators
            omnisearch_client: Optional client for external market intelligence
            omnisearch_enabled: Whether to use external data sources

        Returns:
            TradeAction with the trading decision
        """
        try:
            self._decision_count += 1
            logger.info(f"Starting market analysis #{self._decision_count}")

            # Step 1: Prepare market data for LLM
            market_data = await self.market_analyzer.prepare_llm_input(
                market_state=market_state,
                omnisearch_client=omnisearch_client,
                omnisearch_enabled=omnisearch_enabled,
            )

            # Step 2: Format the prompt with market data
            prompt = self.prompt_manager.format_prompt(**market_data)

            # Step 3: Call LLM (simplified structure - actual implementation would use LangChain)
            llm_response = await self._call_llm(prompt)

            # Step 4: Parse and validate the response
            json_decision = self.response_parser.extract_json_from_response(
                llm_response
            )
            trade_action = self.response_parser.parse_trade_action(json_decision)

            if trade_action is None:
                # Fallback to safe HOLD action
                logger.warning("Failed to parse LLM response, defaulting to HOLD")
                trade_action = TradeAction(
                    action="HOLD",
                    size_pct=0,
                    take_profit_pct=1.0,
                    stop_loss_pct=1.0,
                    rationale="Failed to parse LLM response",
                )

            logger.info(
                f"Decision #{self._decision_count}: {trade_action.action} "
                f"(size: {trade_action.size_pct}%)"
            )
            return trade_action

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in market analysis: {e}", exc_info=True)

            # Return safe default on error
            return TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                rationale=f"Error during analysis: {e!s}",
            )

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with formatted prompt. Placeholder for LangChain integration."""
        # Placeholder for LLM call structure
        logger.debug(f"Calling {self.model_provider}/{self.model_name}")
        logger.debug(f"Prompt length: {len(prompt)} characters")

        # Simplified mock - actual implementation would call LangChain
        return 'MOMENTUM ANALYSIS:\nMock analysis\n\nJSON_DECISION:\n{"action": "HOLD", "size_pct": 0, "take_profit_pct": 1.0, "stop_loss_pct": 1.0, "rationale": "Mock response"}'

    def get_statistics(self) -> dict[str, Any]:
        """Get agent performance statistics."""
        error_rate = (
            (self._error_count / self._decision_count * 100)
            if self._decision_count > 0
            else 0
        )

        return {
            "total_decisions": self._decision_count,
            "error_count": self._error_count,
            "error_rate_pct": round(error_rate, 2),
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
        }
