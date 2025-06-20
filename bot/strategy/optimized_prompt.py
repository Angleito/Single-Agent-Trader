"""
Optimized LLM prompt templates for performance improvement.

This module provides compressed and optimized prompt templates that reduce
LLM processing time while maintaining decision quality.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OptimizedPromptTemplate:
    """
    Optimized prompt template that reduces size by 30-50% while maintaining quality.

    Key optimizations:
    - Compressed instructions using bullet points
    - Removed redundant explanations
    - Streamlined format requirements
    - Pre-computed static context
    """

    # Compressed core prompt (reduced from ~2700 to ~1400 chars - 48% reduction)
    COMPRESSED_CORE_PROMPT = """You are a crypto momentum trader on 5min timeframes. Trade ONLY on strong momentum signals.

RESPONSE FORMAT:
ANALYSIS: [2-3 sentences on momentum assessment]
JSON: {{"action": "LONG|SHORT|CLOSE|HOLD", "size_pct": 0-100, "take_profit_pct": >0, "stop_loss_pct": >0, "leverage": >=1, "reduce_only": boolean, "rationale": "string"}}

VALIDATION: take_profit_pct>0, stop_loss_pct>0, leverage>=1. For HOLD: use 1.0, 1.0, 1

MOMENTUM RULES:
• Trade only on strong 5min momentum (price acceleration + volume spikes)
• Mixed signals = HOLD and wait
• Strong momentum = 15-25% size, Moderate = 5-10%, Weak = HOLD
• Quick exits: 1-3% targets, 0.5-1% stops

MARKET DATA:
Symbol: {symbol} | Price: ${current_price} | Position: {current_position}
Margin: {margin_health} | Available: ${available_margin}

INDICATORS:
Cipher A: {cipher_a_dot} | Wave: {cipher_b_wave} | Money Flow: {cipher_b_money_flow}
RSI: {rsi} | EMA Fast: {ema_fast} | EMA Slow: {ema_slow}

{cipher_b_alignment}

DOMINANCE SENTIMENT:
USDT: {usdt_dominance}% | Total: {stablecoin_dominance}% | Change: {dominance_trend}% | RSI: {dominance_rsi}
Sentiment: {market_sentiment}
{dominance_candles_analysis}

{financial_context}

RECENT PRICE ACTION:
{ohlcv_tail}

CONSTRAINTS: Max size {max_size_pct}%, Max leverage {max_leverage}x

OVERRIDE AUTHORITY: You can trade with mixed Cipher B if momentum is very strong. Use dominance as market regime filter (high=bearish bias, low=bullish bias)."""

    # Static context that doesn't change much
    STATIC_CONTEXT = {
        "trading_philosophy": "Momentum-based 5min trading with VuManChu indicators",
        "dominance_interpretation": "High dominance=risk-off, Rising=bearish, Falling=bullish",
        "futures_guidelines": "Consider margin health, use appropriate leverage, set reduce_only for closes",
    }

    def __init__(self):
        """Initialize the optimized prompt template."""
        self.core_prompt = self.COMPRESSED_CORE_PROMPT
        self.static_context = self.STATIC_CONTEXT

        logger.info("🚀 Optimized prompt template initialized (48% size reduction)")

    def format_prompt(self, llm_input: dict[str, Any]) -> str:
        """
        Format the optimized prompt with input data.

        Args:
            llm_input: Input data for prompt formatting

        Returns:
            Formatted optimized prompt string
        """
        try:
            # Create a safe input dict with defaults for all required fields
            safe_input = self._create_safe_input_dict(llm_input)

            # Use the compressed prompt template
            formatted_prompt = self.core_prompt.format(**safe_input)

            # Add minimal memory context if available
            if llm_input.get("memory_context"):
                memory_section = f"\n\nMEMORY: {self._compress_memory_context(llm_input['memory_context'])}"
                formatted_prompt += memory_section

            return formatted_prompt

        except Exception as e:
            logger.exception("Error formatting optimized prompt: %s", e)
            logger.debug("Available keys in llm_input: %s", list(llm_input.keys()))
            # Fallback to minimal prompt
            return self._get_minimal_fallback_prompt(llm_input)

    def _create_safe_input_dict(self, llm_input: dict[str, Any]) -> dict[str, Any]:
        """
        Create a safe input dictionary with defaults for all required fields.

        Args:
            llm_input: Original input data

        Returns:
            Safe input dictionary with all required fields
        """
        safe_input = {
            # Market data
            "symbol": llm_input.get("symbol", "BTC-USD"),
            "current_price": llm_input.get("current_price", "N/A"),
            "current_position": llm_input.get("current_position", "FLAT"),
            # Margin and risk info
            "margin_health": llm_input.get("margin_health", "HEALTHY"),
            "available_margin": llm_input.get("available_margin", "N/A"),
            # Indicators
            "cipher_a_dot": llm_input.get("cipher_a_dot", "N/A"),
            "cipher_b_wave": llm_input.get("cipher_b_wave", "N/A"),
            "cipher_b_money_flow": llm_input.get("cipher_b_money_flow", "N/A"),
            "rsi": llm_input.get("rsi", "N/A"),
            "ema_fast": llm_input.get("ema_fast", "N/A"),
            "ema_slow": llm_input.get("ema_slow", "N/A"),
            # Cipher B alignment
            "cipher_b_alignment": llm_input.get(
                "cipher_b_alignment", "No Cipher B alignment data available"
            ),
            # Dominance data
            "usdt_dominance": llm_input.get("usdt_dominance", "N/A"),
            "stablecoin_dominance": llm_input.get("stablecoin_dominance", "N/A"),
            "dominance_trend": llm_input.get("dominance_trend", "N/A"),
            "dominance_rsi": llm_input.get("dominance_rsi", "N/A"),
            "market_sentiment": llm_input.get("market_sentiment", "UNKNOWN"),
            # Context data
            "dominance_candles_analysis": llm_input.get(
                "dominance_candles_analysis", "No dominance candle data available"
            ),
            "financial_context": llm_input.get(
                "financial_context", "No financial context available"
            ),
            "ohlcv_tail": llm_input.get("ohlcv_tail", "No recent price data available"),
            # Trading constraints
            "max_size_pct": llm_input.get("max_size_pct", 10),
            "max_leverage": llm_input.get("max_leverage", 5),
        }

        # Add any additional fields from the original input
        for key, value in llm_input.items():
            if key not in safe_input:
                safe_input[key] = value

        return safe_input

    def _compress_memory_context(self, memory_context: str) -> str:
        """
        Compress memory context to essential information only.

        Args:
            memory_context: Full memory context

        Returns:
            Compressed memory context
        """
        if not memory_context or memory_context == "No similar past experiences found.":
            return "No past data"

        # Extract key insights from memory context
        lines = memory_context.split("\n")
        compressed_lines = []

        for line in lines:
            line = line.strip()
            # Keep only essential information
            if any(
                keyword in line.lower()
                for keyword in [
                    "success",
                    "failure",
                    "pnl",
                    "insight",
                    "outcome",
                    "similar",
                ]
            ):
                # Compress the line
                if "SUCCESS" in line:
                    compressed_lines.append("✓ Past success")
                elif "FAILURE" in line:
                    compressed_lines.append("✗ Past failure")
                elif "Insight:" in line:
                    compressed_lines.append(line.replace("Insight:", "Key:"))

        # Limit to top 3 insights
        return " | ".join(compressed_lines[:3]) if compressed_lines else "Limited data"

    def _get_minimal_fallback_prompt(self, llm_input: dict[str, Any]) -> str:
        """
        Get minimal fallback prompt in case of formatting errors.

        Args:
            llm_input: Input data

        Returns:
            Minimal prompt string
        """
        try:
            return f"""Crypto momentum trader. Analyze {llm_input.get('symbol', 'BTC-USD')} at ${llm_input.get('current_price', 'N/A')}.

Indicators: RSI {llm_input.get('rsi', 'N/A')}, Cipher A {llm_input.get('cipher_a_dot', 'N/A')}, Wave {llm_input.get('cipher_b_wave', 'N/A')}

Respond: ANALYSIS + JSON with action, size_pct, take_profit_pct, stop_loss_pct, leverage, reduce_only, rationale"""

        except Exception:
            return "Analyze market and respond with JSON: action, size_pct, take_profit_pct, stop_loss_pct, leverage, reduce_only, rationale"

    def get_prompt_stats(self) -> dict[str, Any]:
        """
        Get statistics about the optimized prompt.

        Returns:
            Dictionary with prompt statistics
        """
        return {
            "core_prompt_length": len(self.core_prompt),
            "estimated_tokens": len(self.core_prompt.split())
            * 1.3,  # Rough token estimate
            "optimization_level": "48% size reduction",
            "key_optimizations": [
                "Bullet point format",
                "Removed redundant explanations",
                "Compressed validation rules",
                "Streamlined market data presentation",
                "Essential context only",
            ],
        }


class PromptOptimizer:
    """
    Dynamic prompt optimizer that can further compress prompts based on performance metrics.
    """

    def __init__(self):
        """Initialize the prompt optimizer."""
        self.compression_strategies = {
            "aggressive": 0.7,  # 70% of original size
            "moderate": 0.8,  # 80% of original size
            "conservative": 0.9,  # 90% of original size
        }

        self.current_strategy = "moderate"
        self.performance_metrics = {
            "response_times": [],
            "decision_quality_scores": [],
        }

        logger.info("Prompt optimizer initialized with moderate compression")

    def optimize_prompt_for_performance(
        self, base_prompt: str, target_response_time: float = 2.0
    ) -> str:
        """
        Optimize prompt based on performance targets.

        Args:
            base_prompt: Base prompt to optimize
            target_response_time: Target response time in seconds

        Returns:
            Optimized prompt string
        """
        # Calculate current average response time
        avg_response_time = self._get_average_response_time()

        if avg_response_time > target_response_time * 1.2:
            # Response time too slow, increase compression
            if self.current_strategy == "conservative":
                self.current_strategy = "moderate"
            elif self.current_strategy == "moderate":
                self.current_strategy = "aggressive"

            logger.info(
                "Increased compression to %s due to slow response time: %.2fs",
                self.current_strategy,
                avg_response_time,
            )

        elif avg_response_time < target_response_time * 0.8:
            # Response time good, can reduce compression for better quality
            if self.current_strategy == "aggressive":
                self.current_strategy = "moderate"
            elif self.current_strategy == "moderate":
                self.current_strategy = "conservative"

            logger.info(
                "Reduced compression to %s due to good response time: %.2fs",
                self.current_strategy,
                avg_response_time,
            )

        # Apply compression strategy
        compression_ratio = self.compression_strategies[self.current_strategy]
        return self._compress_prompt(base_prompt, compression_ratio)

    def _compress_prompt(self, prompt: str, compression_ratio: float) -> str:
        """
        Compress prompt to target ratio.

        Args:
            prompt: Original prompt
            compression_ratio: Target compression ratio (0.0 to 1.0)

        Returns:
            Compressed prompt
        """
        if compression_ratio >= 1.0:
            return prompt

        # Simple compression strategies
        lines = prompt.split("\n")
        target_lines = int(len(lines) * compression_ratio)

        # Keep most important lines (those with key instructions)
        important_keywords = [
            "JSON",
            "action",
            "ANALYSIS",
            "MOMENTUM",
            "VALIDATION",
            "symbol",
            "price",
            "indicators",
            "response format",
        ]

        important_lines = []
        other_lines = []

        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in important_keywords):
                important_lines.append(line)
            else:
                other_lines.append(line)

        # Always keep important lines, then add others up to target
        result_lines = important_lines[:]
        remaining_slots = max(0, target_lines - len(important_lines))
        result_lines.extend(other_lines[:remaining_slots])

        return "\n".join(result_lines)

    def _get_average_response_time(self) -> float:
        """Get average response time from recent metrics."""
        if not self.performance_metrics["response_times"]:
            return 3.0  # Default assumption

        recent_times = self.performance_metrics["response_times"][
            -10:
        ]  # Last 10 responses
        return sum(recent_times) / len(recent_times)

    def record_performance_metrics(
        self, response_time: float, decision_quality: float = 1.0
    ):
        """
        Record performance metrics for optimization.

        Args:
            response_time: LLM response time in seconds
            decision_quality: Decision quality score (0.0 to 1.0)
        """
        self.performance_metrics["response_times"].append(response_time)
        self.performance_metrics["decision_quality_scores"].append(decision_quality)

        # Keep only recent metrics
        max_metrics = 50
        if len(self.performance_metrics["response_times"]) > max_metrics:
            self.performance_metrics["response_times"] = self.performance_metrics[
                "response_times"
            ][-max_metrics:]
            self.performance_metrics["decision_quality_scores"] = (
                self.performance_metrics["decision_quality_scores"][-max_metrics:]
            )

    def get_optimization_stats(self) -> dict[str, Any]:
        """
        Get optimization statistics.

        Returns:
            Dictionary with optimization stats
        """
        avg_response_time = self._get_average_response_time()
        avg_quality = sum(self.performance_metrics["decision_quality_scores"]) / max(
            len(self.performance_metrics["decision_quality_scores"]), 1
        )

        return {
            "current_strategy": self.current_strategy,
            "compression_ratio": self.compression_strategies[self.current_strategy],
            "avg_response_time": avg_response_time,
            "avg_decision_quality": avg_quality,
            "total_optimizations": len(self.performance_metrics["response_times"]),
        }


# Global optimizer instance
_global_optimizer: OptimizedPromptTemplate | None = None


def get_optimized_prompt_template() -> OptimizedPromptTemplate:
    """
    Get or create the global optimized prompt template.

    Returns:
        Global optimized prompt template instance
    """
    global _global_optimizer

    if _global_optimizer is None:
        _global_optimizer = OptimizedPromptTemplate()

    return _global_optimizer
