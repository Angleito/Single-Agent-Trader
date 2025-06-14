"""
Enhanced logging infrastructure for LLM chat completions and trading decisions.

This module provides structured logging capabilities specifically designed for
monitoring LLM interactions, performance metrics, and decision-making processes
in the AI trading bot.
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Graceful degradation when LangChain is not available
    BaseCallbackHandler = object
    AIMessage = object
    BaseMessage = object
    HumanMessage = object
    LLMResult = object

    LANGCHAIN_AVAILABLE = False

from .config import settings


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal objects."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ChatCompletionLogger:
    """
    Enhanced logger for LLM chat completions with structured logging.

    Tracks:
    - Request/response data and timing
    - Token usage and cost estimation
    - Performance metrics
    - Error tracking and recovery
    - Decision rationale and market context
    """

    def __init__(self, log_file: str | None = None, log_level: str = "INFO"):
        """
        Initialize the chat completion logger.

        Args:
            log_file: Path to log file (defaults to logs/llm_completions.log)
            log_level: Logging level for this logger
        """
        self.log_file = log_file or "logs/llm_completions.log"
        self.logger = self._setup_logger(log_level)

        # Performance tracking
        self._completion_times: list[float] = []
        self._token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Cost tracking (OpenAI pricing as of 2024)
        self._pricing = {
            "o3": {"input": 0.015, "output": 0.06},  # per 1K tokens
            "o3-mini": {"input": 0.0015, "output": 0.006},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        }

        # Session tracking
        self._session_id = str(uuid.uuid4())[:8]
        self._completion_count = 0

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup dedicated logger for LLM completions."""
        logger = logging.getLogger("llm_completions")
        logger.setLevel(getattr(logging, log_level.upper()))

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # Create logs directory if it doesn't exist
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler with rotation
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=50 * 1024 * 1024, backupCount=5  # 50MB
        )

        # JSON formatter for structured logging
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler for development
        if settings.system.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def log_completion_request(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
        max_tokens: int,
        market_context: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> str:
        """
        Log LLM completion request with full context.

        Args:
            prompt: The prompt sent to the LLM
            model: Model name (e.g., 'o3', 'gpt-4')
            temperature: Model temperature setting
            max_tokens: Maximum tokens requested
            market_context: Trading context (symbol, price, indicators, etc.)
            request_id: Unique request identifier

        Returns:
            Generated request ID for tracking
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        self._completion_count += 1

        log_entry = {
            "event_type": "completion_request",
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "request_id": request_id,
            "completion_number": self._completion_count,
            "model": model,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "market_context": market_context or {},
        }

        # Only include temperature if it's not None (o3 models don't support it)
        if temperature is not None:
            log_entry["temperature"] = temperature

        self.logger.info(f"LLM_REQUEST: {json.dumps(log_entry, cls=DecimalEncoder)}")
        return request_id

    def log_completion_response(
        self,
        request_id: str,
        response: Any,
        response_time: float,
        token_usage: dict[str, int] | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """
        Log LLM completion response with metrics.

        Args:
            request_id: Request ID from log_completion_request
            response: The LLM response (parsed or raw)
            response_time: Time taken for completion in seconds
            token_usage: Token usage statistics
            success: Whether the request was successful
            error: Error message if request failed
        """
        self._completion_times.append(response_time)

        if token_usage:
            for key, value in token_usage.items():
                self._token_usage[key] += value

        # Calculate cost estimate
        model = (
            getattr(response, "model", "unknown")
            if hasattr(response, "model")
            else "unknown"
        )
        cost_estimate = self._calculate_cost(model, token_usage) if token_usage else 0.0

        log_entry = {
            "event_type": "completion_response",
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "request_id": request_id,
            "success": success,
            "response_time_ms": round(response_time * 1000, 2),
            "token_usage": token_usage or {},
            "cost_estimate_usd": round(cost_estimate, 6),
            "error": error,
            "response_preview": str(response)[:1000] if response else None,
        }

        level = "INFO" if success else "ERROR"
        self.logger.log(
            getattr(logging, level),
            f"LLM_RESPONSE: {json.dumps(log_entry, cls=DecimalEncoder)}",
        )

    def log_trading_decision(
        self,
        request_id: str,
        trade_action: Any,
        market_state: Any,
        validation_result: str | None = None,
        risk_assessment: str | None = None,
    ) -> None:
        """
        Log the final trading decision with full context.

        Args:
            request_id: Original request ID
            trade_action: The final trading action decided
            market_state: Current market state
            validation_result: Result of trade validation
            risk_assessment: Risk manager assessment
        """
        log_entry = {
            "event_type": "trading_decision",
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "request_id": request_id,
            "action": getattr(trade_action, "action", str(trade_action)),
            "size_pct": getattr(trade_action, "size_pct", 0),
            "rationale": getattr(trade_action, "rationale", ""),
            "symbol": getattr(market_state, "symbol", "unknown"),
            "current_price": getattr(market_state, "current_price", 0),
            "validation_result": validation_result,
            "risk_assessment": risk_assessment,
            "indicators": {
                "cipher_a_dot": (
                    getattr(market_state.indicators, "cipher_a_dot", None)
                    if hasattr(market_state, "indicators")
                    else None
                ),
                "cipher_b_wave": (
                    getattr(market_state.indicators, "cipher_b_wave", None)
                    if hasattr(market_state, "indicators")
                    else None
                ),
                "rsi": (
                    getattr(market_state.indicators, "rsi", None)
                    if hasattr(market_state, "indicators")
                    else None
                ),
            },
        }

        self.logger.info(
            f"TRADING_DECISION: {json.dumps(log_entry, cls=DecimalEncoder)}"
        )

    def log_performance_metrics(self) -> dict[str, Any]:
        """
        Log current performance metrics and return summary.

        Returns:
            Dictionary with performance statistics
        """
        metrics = {
            "session_id": self._session_id,
            "total_completions": self._completion_count,
            "avg_response_time_ms": (
                round(
                    sum(self._completion_times) / len(self._completion_times) * 1000, 2
                )
                if self._completion_times
                else 0
            ),
            "min_response_time_ms": (
                round(min(self._completion_times) * 1000, 2)
                if self._completion_times
                else 0
            ),
            "max_response_time_ms": (
                round(max(self._completion_times) * 1000, 2)
                if self._completion_times
                else 0
            ),
            "total_tokens": self._token_usage["total_tokens"],
            "total_cost_estimate_usd": self._calculate_total_cost(),
            "tokens_per_second": self._calculate_tokens_per_second(),
        }

        log_entry = {
            "event_type": "performance_metrics",
            "timestamp": datetime.now(UTC).isoformat(),
            **metrics,
        }

        self.logger.info(f"PERFORMANCE: {json.dumps(log_entry, cls=DecimalEncoder)}")
        return metrics

    def _calculate_cost(self, model: str, token_usage: dict[str, int]) -> float:
        """Calculate estimated cost for a completion."""
        if model not in self._pricing or not token_usage:
            return 0.0

        pricing = self._pricing[model]
        prompt_cost = (token_usage.get("prompt_tokens", 0) / 1000) * pricing["input"]
        completion_cost = (token_usage.get("completion_tokens", 0) / 1000) * pricing[
            "output"
        ]

        return prompt_cost + completion_cost

    def _calculate_total_cost(self) -> float:
        """Calculate total estimated cost for the session."""
        # This is a simplified calculation - in practice, you'd track per-model usage
        total_prompt = self._token_usage["prompt_tokens"]
        total_completion = self._token_usage["completion_tokens"]

        # Default to o3 pricing if we don't have model breakdown
        pricing = self._pricing.get(settings.llm.model_name, self._pricing["o3"])
        prompt_cost = (total_prompt / 1000) * pricing["input"]
        completion_cost = (total_completion / 1000) * pricing["output"]

        return prompt_cost + completion_cost

    def _calculate_tokens_per_second(self) -> float:
        """Calculate average tokens per second."""
        if not self._completion_times or self._token_usage["total_tokens"] == 0:
            return 0.0

        total_time = sum(self._completion_times)
        return round(self._token_usage["total_tokens"] / total_time, 2)


class LangChainCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for detailed chain execution logging.

    Integrates with the ChatCompletionLogger to provide comprehensive
    tracking of LangChain operations.

    NOTE: This callback handler is currently disabled for o3 models due to
    compatibility issues with older LangChain versions (0.1.x) that don't
    properly handle o3 model token usage response format.

    TODO: Upgrade to LangChain 0.2+ which properly supports newer OpenAI models
    """

    def __init__(self, completion_logger: ChatCompletionLogger):
        """Initialize with a completion logger instance."""
        super().__init__()
        self.completion_logger = completion_logger
        self.logger = logging.getLogger("langchain_callbacks")

        # Track chain execution
        self._chain_starts: dict[str, float] = {}
        self._current_request_id: str | None = None

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Called when a chain starts running."""
        run_id = kwargs.get("run_id", "unknown")
        self._chain_starts[str(run_id)] = time.time()

        self.logger.debug(
            f"Chain started: {serialized.get('name', 'unknown')} - {run_id}"
        )

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Called when a chain finishes running."""
        run_id = str(kwargs.get("run_id", "unknown"))
        start_time = self._chain_starts.pop(run_id, time.time())
        duration = time.time() - start_time

        self.logger.debug(f"Chain completed in {duration:.3f}s - {run_id}")

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Called when an LLM starts generating."""
        if prompts and self.completion_logger:
            # Extract model info from serialized data
            model = serialized.get("kwargs", {}).get("model_name", "unknown")
            temperature = serialized.get("kwargs", {}).get("temperature")
            # Handle both max_tokens and max_completion_tokens parameters
            max_tokens = serialized.get("kwargs", {}).get("max_tokens", 0)
            if max_tokens == 0:
                # Try model_kwargs for o3 models which use max_completion_tokens
                model_kwargs = serialized.get("kwargs", {}).get("model_kwargs", {})
                max_tokens = model_kwargs.get("max_completion_tokens", 0)

            # o3 models don't support temperature - set to None for logging
            if model.startswith("o3") and temperature is not None:
                temperature = None

            # Extract market context from prompt if available
            market_context = self._extract_market_context_from_prompt(prompts[0])

            # Log the request
            self._current_request_id = self.completion_logger.log_completion_request(
                prompt=prompts[0],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                market_context=market_context,
            )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when an LLM finishes generating."""
        if self._current_request_id and self.completion_logger:
            # Calculate response time (approximation)
            response_time = 1.0  # Default fallback

            # Extract token usage if available - safely handle missing fields
            token_usage = None
            try:
                if hasattr(response, "llm_output") and response.llm_output:
                    raw_usage = response.llm_output.get("token_usage", {})
                    if raw_usage:
                        # Safely extract token counts, handling missing completion_tokens_details
                        token_usage = {
                            "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                            "completion_tokens": raw_usage.get("completion_tokens", 0),
                            "total_tokens": raw_usage.get("total_tokens", 0),
                        }

                        # Handle o3-style response format if available
                        if "completion_tokens_details" in raw_usage:
                            token_usage["completion_tokens_details"] = raw_usage[
                                "completion_tokens_details"
                            ]
            except Exception as e:
                self.logger.warning(f"Error extracting token usage: {e}")
                token_usage = None

            # Log the response
            self.completion_logger.log_completion_response(
                request_id=self._current_request_id,
                response=(
                    response.generations[0][0].text if response.generations else None
                ),
                response_time=response_time,
                token_usage=token_usage,
                success=True,
            )

            self._current_request_id = None

    def on_llm_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        """Called when an LLM encounters an error."""
        if self._current_request_id and self.completion_logger:
            self.completion_logger.log_completion_response(
                request_id=self._current_request_id,
                response=None,
                response_time=0.0,
                success=False,
                error=str(error),
            )

            self._current_request_id = None

        self.logger.error(f"LLM error: {error}")

    def _extract_market_context_from_prompt(self, prompt: str) -> dict[str, Any] | None:
        """
        Extract market context information from the prompt text.

        This method parses the formatted prompt to extract key market data
        for logging purposes, avoiding the need for manual logging in the LLM agent.

        Args:
            prompt: The formatted prompt text

        Returns:
            Dictionary with market context data or None if extraction fails
        """
        import re

        try:
            context = {}

            # Extract basic market info
            symbol_match = re.search(r"Symbol:\s*(\S+)", prompt)
            if symbol_match:
                context["symbol"] = symbol_match.group(1)

            price_match = re.search(r"Current Price:\s*\$?([\d,.]+)", prompt)
            if price_match:
                try:
                    context["current_price"] = float(
                        price_match.group(1).replace(",", "")
                    )
                except ValueError:
                    pass

            position_match = re.search(r"Current Position:\s*([^\n]+)", prompt)
            if position_match:
                context["current_position"] = position_match.group(1).strip()

            # Extract technical indicators
            rsi_match = re.search(r"RSI:\s*([\d.]+)", prompt)
            if rsi_match:
                try:
                    context["rsi"] = float(rsi_match.group(1))
                except ValueError:
                    pass

            cipher_a_match = re.search(r"Cipher A Trend Dot:\s*([\d.-]+)", prompt)
            if cipher_a_match:
                try:
                    context["cipher_a_dot"] = float(cipher_a_match.group(1))
                except ValueError:
                    pass

            cipher_b_wave_match = re.search(r"Cipher B Wave:\s*([\d.-]+)", prompt)
            if cipher_b_wave_match:
                try:
                    context["cipher_b_wave"] = float(cipher_b_wave_match.group(1))
                except ValueError:
                    pass

            # Extract dominance data
            dominance_match = re.search(
                r"Total Stablecoin Dominance:\s*([\d.]+)%", prompt
            )
            if dominance_match:
                try:
                    context["stablecoin_dominance"] = float(dominance_match.group(1))
                except ValueError:
                    pass

            sentiment_match = re.search(r"Market Sentiment:\s*(\w+)", prompt)
            if sentiment_match:
                context["market_sentiment"] = sentiment_match.group(1)

            # Only return context if we extracted at least some data
            return context if context else None

        except Exception as e:
            self.logger.warning(f"Failed to extract market context from prompt: {e}")
            return None


def create_llm_logger(
    log_level: str | None = None, log_file: str | None = None
) -> ChatCompletionLogger:
    """
    Factory function to create a configured ChatCompletionLogger.

    Args:
        log_level: Logging level (defaults to system setting)
        log_file: Log file path (defaults to logs/llm_completions.log)

    Returns:
        Configured ChatCompletionLogger instance
    """
    return ChatCompletionLogger(
        log_level=log_level or settings.system.log_level, log_file=log_file
    )


def create_langchain_callback(
    completion_logger: ChatCompletionLogger,
) -> LangChainCallbackHandler | None:
    """
    Factory function to create LangChain callback handler.

    Args:
        completion_logger: ChatCompletionLogger instance

    Returns:
        LangChainCallbackHandler or None if LangChain not available
    """
    if not LANGCHAIN_AVAILABLE:
        return None

    return LangChainCallbackHandler(completion_logger)
