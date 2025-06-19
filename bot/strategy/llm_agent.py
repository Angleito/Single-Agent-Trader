# flake8: noqa
# pyright: reportMissingImports=false, reportUnknownVariableType=false

"""
LangChain-based LLM agent for trading decisions.

This module implements the AI decision-making core using LangChain and LLMs
to analyze market state and generate trading actions.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

try:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI
except ImportError:
    # Graceful degradation if LangChain not installed
    ChatOpenAI = None  # type: ignore
    PromptTemplate = None  # type: ignore
    JsonOutputParser = None  # type: ignore
    RunnablePassthrough = None  # type: ignore

from ..config import settings
from ..llm_logging import create_llm_logger, create_langchain_callback
from ..trading_types import MarketState, TradeAction
from ..mcp.omnisearch_client import OmniSearchClient
from .llm_cache import get_llm_cache
from .optimized_prompt import get_optimized_prompt_template
from .performance_monitor import get_performance_monitor, record_llm_performance

# Removed imports for deleted scalping indicators

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LangChain-powered LLM agent for trading decisions.

    Analyzes market state including OHLCV data, technical indicators,
    and current position to generate structured trading actions.
    """

    def __init__(
        self,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        omnisearch_client: Optional[OmniSearchClient] = None,
    ):
        """
        Initialize the LLM agent.

        Args:
            model_provider: LLM provider ('openai', 'ollama')
            model_name: Specific model to use
            omnisearch_client: Optional OmniSearch client for market intelligence
        """
        self.model_provider = model_provider or settings.llm.provider
        self.model_name = model_name or settings.llm.model_name

        # o3 models only support temperature=1.0
        if self.model_name.startswith("o3"):
            self.temperature = 1.0  # o3 models only support temperature=1.0
        else:
            self.temperature = settings.llm.temperature

        # LangChain components
        self._model: Optional[Any] = None
        self._prompt_template: Optional[Any] = None
        self._chain: Optional[Any] = None

        # Enhanced logging components
        self._completion_logger: Optional[Any] = None
        self._langchain_callback: Optional[Any] = None
        self._completion_count = 0

        # OmniSearch integration
        self._omnisearch_client = omnisearch_client
        self._omnisearch_enabled = (
            omnisearch_client is not None
            and settings.omnisearch.enabled
            and settings.omnisearch.api_key is not None
        )

        # Performance optimization: LLM response caching
        self._cache_enabled = getattr(settings, "llm_cache_enabled", True)
        self._cache = get_llm_cache() if self._cache_enabled else None

        # Performance optimization: Optimized prompt templates
        self._use_optimized_prompts = settings.llm.use_optimized_prompts
        self._optimized_prompt_template = (
            get_optimized_prompt_template() if self._use_optimized_prompts else None
        )

        # Performance monitoring
        self._performance_monitor = get_performance_monitor()
        self._enable_performance_tracking = getattr(
            settings, "llm_enable_performance_tracking", True
        )

        # Scalping signals system disabled - modules were removed
        self._scalping_signals = None
        self._scalping_enabled = False

        # Initialize completion logging if enabled
        if settings.llm.enable_completion_logging:
            self._completion_logger = create_llm_logger(
                log_level=settings.llm.completion_log_level,
                log_file=settings.llm.completion_log_file,
            )

            # Create LangChain callback handler if enabled
            # Temporarily disabled due to o3 model compatibility issues with older LangChain versions
            # TODO: Re-enable after upgrading LangChain to version that supports o3 models
            if (
                settings.llm.enable_langchain_callbacks
                and self._completion_logger
                and not self.model_name.startswith("o3")
            ):
                self._langchain_callback = create_langchain_callback(
                    self._completion_logger
                )

        # Load prompt template
        self._load_prompt_template()

        # Initialize model if dependencies available
        if ChatOpenAI is not None:
            self._initialize_model()
        else:
            logger.warning("LangChain not available - using fallback decision logic")

        # Log OmniSearch status
        if self._omnisearch_enabled:
            logger.info(
                "🔍 OmniSearch integration enabled for enhanced market intelligence"
            )
        else:
            logger.info(
                "OmniSearch integration disabled - using standard analysis only"
            )

        # Scalping signals system removed during cleanup
        logger.info(
            "ScalpingSignals system disabled - using basic indicator analysis only"
        )

    def _load_prompt_template(self) -> None:
        """Load the prompt template for trading decisions."""
        # Default prompt template
        default_prompt = """You are an expert cryptocurrency momentum trader operating on 5-minute timeframes for leveraged futures positions.

TRADING PHILOSOPHY: You are a momentum-based trader. You trade ONLY when you detect strong momentum signals that develop within 5-minute candles. If no clear momentum is present, you WAIT for the next 5-minute candle to complete.

Respond with BOTH a detailed analysis AND valid JSON. Format your response as:

MOMENTUM ANALYSIS:
[Provide a detailed 3-4 paragraph analysis explaining your momentum assessment, what signals you're seeing, why you're making this decision, and how you evaluated the 5-minute timeframe data]

JSON_DECISION:
{{
  "action": "LONG|SHORT|CLOSE|HOLD",
  "size_pct": 0-100,
  "take_profit_pct": positive_number_greater_than_0,
  "stop_loss_pct": positive_number_greater_than_0,
  "leverage": positive_integer_1_or_greater,
  "reduce_only": boolean,
  "rationale": "string"
}}

IMPORTANT VALIDATION RULES:
- take_profit_pct must be > 0 (e.g., 2.5 for 2.5%)
- stop_loss_pct must be > 0 (e.g., 1.0 for 1.0%)
- leverage must be >= 1 (e.g., 1, 2, 5, 10)
- For HOLD actions, use: take_profit_pct=1.0, stop_loss_pct=1.0, leverage=1

MOMENTUM TRADING RULES:
1. Only trade when you see STRONG momentum building within the current 5-minute period
2. Look for price acceleration, volume spikes, and indicator alignment
3. If momentum is weak or unclear, always choose HOLD and wait for next candle
4. Momentum trades should be quick and decisive - capture the move and exit
5. Use higher position sizes (15-25%) when momentum is very strong
6. Use lower position sizes (5-10%) when momentum is moderate

Market Analysis:
- Symbol: {symbol}
- Interval: {interval}
- Current Price: ${current_price}
- Current Position: {current_position}
- Margin Health: {margin_health}
- Available Margin: ${available_margin}

Technical Indicators:
- Cipher A Trend Dot: {cipher_a_dot}
- Cipher B Wave: {cipher_b_wave}
- Cipher B Money Flow: {cipher_b_money_flow}
- RSI: {rsi}
- EMA Fast: {ema_fast}
- EMA Slow: {ema_slow}

Cipher B Signal Alignment:
{cipher_b_alignment}

IMPORTANT: You have the FINAL decision authority. The Cipher B alignment above is informational only.
While traditional Cipher B requires both signals to align, you can override this if you see:
- Strong momentum in other indicators
- Clear price action patterns
- Favorable market sentiment from dominance
- Risk/reward opportunities that justify the trade

Market Sentiment (Stablecoin Dominance):
- USDT Dominance: {usdt_dominance}%
- USDC Dominance: {usdc_dominance}%
- Total Stablecoin Dominance: {stablecoin_dominance}%
- 24h Dominance Change: {dominance_trend}%
- Dominance RSI: {dominance_rsi}
- Market Sentiment: {market_sentiment}

Dominance Candlestick Analysis (3-minute candles):
{dominance_candles_analysis}

VuManChu Dominance Technical Analysis:
{dominance_vumanchu_analysis}

Financial Market Intelligence (Web Search Results):
{financial_context}

Historical Price Data (24h context):
{ohlcv_tail}

Trading Constraints:
- Maximum position size: {max_size_pct}% of equity
- Maximum leverage: {max_leverage}x
- Futures trading enabled: {futures_enabled}
- Auto cash transfer: {auto_cash_transfer}

Futures Trading Guidelines:
1. Consider margin health before opening positions
2. Use appropriate leverage based on market volatility
3. Set reduce_only=true for position-closing trades
4. Factor in funding costs for overnight positions
5. Respect liquidation risk thresholds

Stablecoin Dominance Analysis:
- High dominance (>10%) = Risk-off sentiment, bearish bias
- Rising dominance = Money flowing to safety, bearish signal
- Falling dominance = Risk-on sentiment, bullish signal
- Dominance RSI >70 = Potential reversal (bullish)
- Dominance RSI <30 = Potential reversal (bearish)

Dominance Candlestick Patterns (TradingView-style analysis):
- Green dominance candles = Increasing stablecoin inflows (bearish for crypto)
- Red dominance candles = Decreasing stablecoin dominance (bullish for crypto)
- Dominance breakouts above resistance = Strong bearish signal
- Dominance breakdowns below support = Strong bullish signal
- Divergences between dominance and price = Potential reversal signals
- Compare dominance candlesticks with VuManChu indicators for confluence

MOMENTUM ANALYSIS INSTRUCTIONS:
1. **PRIMARY**: Analyze the last 5-10 candles for momentum acceleration patterns
2. **VOLUME**: Look for volume spikes accompanying price moves (strong momentum signal)
3. **PRICE ACTION**: Identify breakouts, trend accelerations, or momentum shifts
4. **INDICATOR MOMENTUM**: Check if RSI is trending up/down rapidly, EMA crossovers, momentum in VuManChu
5. **DOMINANCE CONTEXT**: Use stablecoin dominance as market regime filter (high dominance = risk-off)
6. **DECISION TIMEFRAME**: You are evaluating THIS 5-minute candle - is momentum building RIGHT NOW?
7. **HOLD PREFERENCE**: When in doubt, HOLD. Only trade when momentum is unmistakable
8. **POSITION SIZING**: Scale size based on momentum strength:
   - Explosive momentum (volume spike + price acceleration): 20-25%
   - Strong momentum (clear direction + indicator alignment): 15-20%
   - Moderate momentum (some signals but not all): 5-10%
   - Weak/unclear momentum: HOLD (0%)
9. **EXIT STRATEGY**: Momentum trades are quick - target 1-3% moves, stop at 0.5-1% loss
10. **CIPHER B OVERRIDE**: You can trade even with mixed Cipher B if momentum is very strong
11. **TIME AWARENESS**: Remember, you trade on 5-minute candle closes - wait for clear signals
12. **RATIONALE**: Explain momentum reasoning in under 120 characters for the JSON, but provide full detail in the analysis section
13. Always provide BOTH the detailed momentum analysis AND the JSON decision

FINANCIAL INTELLIGENCE INTEGRATION:
- Use the Financial Market Intelligence section to validate or challenge your technical analysis
- Strong sentiment divergence from technicals may indicate potential reversals or false signals
- Consider market correlations - high positive correlation means crypto may follow stock market moves
- Factor in recent news impact levels and sentiment when assessing momentum strength
- NASDAQ sentiment provides broader market context for risk-on/risk-off dynamics"""

        self.prompt_text = default_prompt

        # Try to load from file if available
        prompt_file = Path("prompts/trade_action.txt")
        if prompt_file.exists():
            try:
                self.prompt_text = prompt_file.read_text()
                logger.info("Loaded prompt template from file")
            except Exception as e:
                logger.warning(f"Failed to load prompt file: {e}")

        if PromptTemplate is not None:
            self._prompt_template = PromptTemplate.from_template(self.prompt_text)

    def _initialize_model(self) -> None:
        """Initialize the LLM model and chain."""
        try:
            if self.model_provider == "openai":
                if not settings.llm.openai_api_key:
                    raise ValueError("OpenAI API key not configured")

                # Base kwargs required for any OpenAI chat completion
                base_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "api_key": settings.llm.openai_api_key.get_secret_value(),
                }

                # o3 family models don't support most parameters, only pass essentials
                if self.model_name.startswith("o3"):
                    # o3 models only support basic parameters - temperature must be 1.0
                    # For o3 models, pass max_completion_tokens directly in model_kwargs to avoid LangChain warning
                    base_kwargs["temperature"] = (
                        1.0  # o3 models only support temperature=1.0
                    )

                    # Ensure max_tokens is valid for o3 models
                    max_completion_tokens = max(
                        100, settings.llm.max_tokens
                    )  # Prevent 0 or negative values
                    base_kwargs["model_kwargs"] = {
                        "max_completion_tokens": max_completion_tokens
                    }
                    logger.info(
                        f"Initializing OpenAI o3 model with temperature=1.0, max_completion_tokens={max_completion_tokens}"
                    )
                else:
                    # Non-o3 models support full parameter set
                    base_kwargs["temperature"] = self.temperature
                    base_kwargs["max_tokens"] = settings.llm.max_tokens
                    base_kwargs["top_p"] = settings.llm.top_p
                    base_kwargs["frequency_penalty"] = settings.llm.frequency_penalty
                    base_kwargs["presence_penalty"] = settings.llm.presence_penalty

                self._model = ChatOpenAI(**base_kwargs)

            elif self.model_provider == "ollama":
                # TODO: Implement Ollama support
                raise NotImplementedError("Ollama support not yet implemented")

            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")

            # Create the chain
            if self._model and self._prompt_template:
                parser = JsonOutputParser(pydantic_object=TradeAction)
                self._chain = self._prompt_template | self._model | parser

                logger.info(
                    f"Initialized LLM agent with {self.model_provider}:{self.model_name}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            self._model = None
            self._chain = None

    async def analyze_market(self, market_state: MarketState) -> TradeAction:
        """
        Analyze market state and generate trading decision with enhanced logging and caching.

        Args:
            market_state: Complete market state including OHLCV and indicators

        Returns:
            TradeAction with decision and parameters
        """
        request_id = None

        try:
            # Track cache performance
            cache_hit = False
            decision_start_time = time.time()

            # Use cache if enabled and LLM is available
            if self._cache_enabled and self._cache and self._chain is not None:
                # Check cache first to determine if it will be a hit
                cache_key = self._cache.hasher.get_cache_key(market_state)
                cached_entry = self._cache._get_cached_entry(cache_key)
                cache_hit = cached_entry is not None

                result = await self._cache.get_or_compute(
                    market_state, self._get_cached_llm_decision, market_state
                )
                # Get request_id from callback handler if available
                request_id = (
                    getattr(self._langchain_callback, "_current_request_id", None)
                    if self._langchain_callback
                    else "llm_completion"
                )
            else:
                # Original path without caching
                if self._chain is not None:
                    # Prepare input data for the LLM
                    llm_input = await self._prepare_llm_input(market_state)
                    result = await self._get_llm_decision(llm_input)
                    # Get request_id from callback handler if available
                    request_id = (
                        getattr(self._langchain_callback, "_current_request_id", None)
                        if self._langchain_callback
                        else "llm_completion"
                    )
                else:
                    result = self._get_fallback_decision(market_state)

            # Record performance metrics for cached/non-cached requests
            if self._enable_performance_tracking:
                decision_time = time.time() - decision_start_time
                prompt_size = 0  # Will be updated if we have access to prompt size
                optimization_level = (
                    "optimized" if self._use_optimized_prompts else "standard"
                )

                record_llm_performance(
                    response_time_ms=decision_time * 1000,
                    prompt_size_chars=prompt_size,
                    cache_hit=cache_hit,
                    decision_action=result.action,
                    optimization_level=optimization_level,
                    error_occurred=False,
                )

            logger.info(f"Generated trade action: {result.action} - {result.rationale}")

            # Log the trading decision if completion logging is enabled
            if self._completion_logger and settings.llm.enable_completion_logging:
                self._completion_logger.log_trading_decision(
                    request_id=request_id or "fallback",
                    trade_action=result,
                    market_state=market_state,
                )

            return result

        except Exception as e:
            logger.error(f"Error in market analysis: {e}")

            # Log the error decision
            if self._completion_logger and settings.llm.enable_completion_logging:
                error_action = TradeAction(
                    action="HOLD",
                    size_pct=0,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.0,
                    leverage=1,
                    rationale="Error in analysis - holding position",
                )

                self._completion_logger.log_trading_decision(
                    request_id=request_id or "error",
                    trade_action=error_action,
                    market_state=market_state,
                    validation_result=f"Error: {str(e)}",
                )

            # Return safe default action
            return TradeAction(
                action="HOLD",
                size_pct=0,
                take_profit_pct=2.0,
                stop_loss_pct=1.0,
                leverage=1,
                rationale="Error in analysis - holding position",
            )

    async def _prepare_llm_input(self, market_state: MarketState) -> Dict[str, Any]:
        """
        Prepare market state data for LLM input.

        Args:
            market_state: Market state object

        Returns:
            Dictionary formatted for prompt template
        """
        # Get recent OHLCV data (show summary of 24h data)
        all_candles = market_state.ohlcv_data

        # Show detailed last 10 candles
        recent_candles = all_candles[-10:] if len(all_candles) >= 10 else all_candles

        ohlcv_lines = []
        ohlcv_lines.append(f"=== Last 10 candles (of {len(all_candles)} total) ===")
        for candle in recent_candles:
            ohlcv_lines.append(
                f"{candle.timestamp.strftime('%H:%M')}: "
                f"O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} V:{candle.volume}"
            )

        # Add 24h summary statistics
        if len(all_candles) > 10:
            ohlcv_lines.append("\n=== 24h Summary ===")
            high_24h = max(c.high for c in all_candles)
            low_24h = min(c.low for c in all_candles)
            volume_24h = sum(c.volume for c in all_candles)
            open_24h = all_candles[0].open
            close_24h = all_candles[-1].close
            change_24h = ((close_24h - open_24h) / open_24h) * 100

            ohlcv_lines.append(f"24h High: {high_24h}, Low: {low_24h}")
            ohlcv_lines.append(f"24h Change: {change_24h:+.2f}%")
            ohlcv_lines.append(f"24h Volume: {volume_24h}")
            ohlcv_lines.append(f"Total candles in context: {len(all_candles)}")

        ohlcv_tail = "\\n".join(ohlcv_lines)

        # Format current position
        if market_state.current_position.side == "FLAT":
            position_str = "No position (flat)"
        else:
            position_str = f"{market_state.current_position.side} {market_state.current_position.size} @ {market_state.current_position.entry_price}"

        # Get futures-specific information if available
        margin_health = "N/A"
        available_margin: Union[str, float] = "N/A"

        if hasattr(market_state, "futures_account") and market_state.futures_account:
            margin_health = market_state.futures_account.margin_info.health_status.value
            available_margin = float(
                market_state.futures_account.margin_info.available_margin
            )
        elif (
            hasattr(market_state, "margin_requirements")
            and market_state.margin_requirements
        ):
            margin_health = market_state.margin_requirements.health_status.value
            available_margin = float(market_state.margin_requirements.available_margin)

        # Extract dominance data if available
        dominance_data = {
            "usdt_dominance": market_state.indicators.usdt_dominance or "N/A",
            "usdc_dominance": market_state.indicators.usdc_dominance or "N/A",
            "stablecoin_dominance": market_state.indicators.stablecoin_dominance
            or "N/A",
            "dominance_trend": market_state.indicators.dominance_trend or "N/A",
            "dominance_rsi": market_state.indicators.dominance_rsi or "N/A",
            "market_sentiment": market_state.indicators.market_sentiment or "UNKNOWN",
        }

        # Format dominance candlestick data for analysis
        dominance_candles_analysis = self._format_dominance_candles(market_state)

        # Format VuManChu dominance technical analysis
        dominance_vumanchu_analysis = self._format_dominance_vumanchu_analysis(
            market_state
        )

        # Calculate Cipher B signal alignment
        cipher_b_alignment = self._calculate_cipher_b_alignment(market_state.indicators)

        # Get financial context from OmniSearch if enabled
        financial_context = await self._get_financial_context(market_state)

        # Get scalping signals analysis if enabled
        scalping_analysis = await self._get_scalping_analysis(market_state)

        return {
            "symbol": market_state.symbol,
            "interval": market_state.interval,
            "current_price": float(market_state.current_price),
            "current_position": position_str,
            "margin_health": margin_health,
            "available_margin": available_margin,
            "cipher_a_dot": market_state.indicators.cipher_a_dot,
            "cipher_b_wave": market_state.indicators.cipher_b_wave,
            "cipher_b_money_flow": market_state.indicators.cipher_b_money_flow,
            "rsi": market_state.indicators.rsi,
            "ema_fast": market_state.indicators.ema_fast,
            "ema_slow": market_state.indicators.ema_slow,
            "ohlcv_tail": ohlcv_tail,
            "dominance_candles_analysis": dominance_candles_analysis,
            "dominance_vumanchu_analysis": dominance_vumanchu_analysis,
            "cipher_b_alignment": cipher_b_alignment,
            "max_size_pct": settings.trading.max_size_pct,
            "leverage": settings.trading.leverage,
            "max_leverage": settings.trading.max_futures_leverage,
            "futures_enabled": settings.trading.enable_futures,
            "auto_cash_transfer": settings.trading.auto_cash_transfer,
            "financial_context": financial_context,  # Add OmniSearch financial context
            **dominance_data,  # Add dominance data to the context
            **scalping_analysis,  # Add scalping signals analysis
        }

    def _format_dominance_candles(self, market_state: MarketState) -> str:
        """
        Format dominance candlestick data for LLM analysis.

        Args:
            market_state: Market state containing dominance candles

        Returns:
            Formatted string with dominance candlestick analysis
        """
        try:
            # Try to get dominance candles from MarketState
            dominance_candles = getattr(market_state, "dominance_candles", None)

            if not dominance_candles:
                return "No dominance candle data available for analysis"

            # Take last 5 candles for analysis (like OHLCV)
            recent_candles = (
                dominance_candles[-5:]
                if len(dominance_candles) >= 5
                else dominance_candles
            )

            if not recent_candles:
                return "Insufficient dominance candle data for analysis"

            # Format candlestick data
            candle_lines = []
            for i, candle in enumerate(recent_candles):
                # Determine candle color/direction
                direction = (
                    "🟢"
                    if candle.close > candle.open
                    else "🔴"
                    if candle.close < candle.open
                    else "⚪"
                )
                change_pct = (
                    ((candle.close - candle.open) / candle.open * 100)
                    if candle.open > 0
                    else 0
                )

                # Format time
                time_str = (
                    candle.timestamp.strftime("%H:%M")
                    if hasattr(candle, "timestamp")
                    else f"T-{len(recent_candles)-i}"
                )

                # Create candle summary
                candle_line = (
                    f"{time_str}: {direction} O:{candle.open:.2f}% H:{candle.high:.2f}% "
                    f"L:{candle.low:.2f}% C:{candle.close:.2f}% ({change_pct:+.2f}%)"
                )

                # Add technical indicators if available
                indicators = []
                if hasattr(candle, "rsi") and candle.rsi is not None:
                    indicators.append(f"RSI:{candle.rsi:.1f}")
                if hasattr(candle, "trend_signal") and candle.trend_signal:
                    indicators.append(f"Signal:{candle.trend_signal}")

                if indicators:
                    candle_line += f" [{', '.join(indicators)}]"

                candle_lines.append(candle_line)

            # Calculate overall trend
            if len(recent_candles) >= 2:
                first_close = recent_candles[0].close
                last_close = recent_candles[-1].close
                overall_trend = (
                    ((last_close - first_close) / first_close * 100)
                    if first_close > 0
                    else 0
                )

                trend_direction = (
                    "RISING"
                    if overall_trend > 0.1
                    else "FALLING"
                    if overall_trend < -0.1
                    else "SIDEWAYS"
                )
                trend_line = f"Overall Trend: {trend_direction} ({overall_trend:+.2f}% over {len(recent_candles)} candles)"
            else:
                trend_line = "Overall Trend: Insufficient data"

            # Create analysis summary
            analysis_lines = [
                f"Last {len(recent_candles)} Dominance Candles (3-minute intervals):",
                *candle_lines,
                trend_line,
                f"Latest Dominance: {recent_candles[-1].close:.2f}% ({'increasing stablecoin inflows' if recent_candles[-1].close > recent_candles[-1].open else 'decreasing dominance'})",
            ]

            return "\n".join(analysis_lines)

        except Exception as e:
            logger.warning(f"Error formatting dominance candles: {e}")
            return f"Error formatting dominance candles: {str(e)}"

    def _format_dominance_vumanchu_analysis(self, market_state: MarketState) -> str:
        """
        Format VuManChu dominance technical analysis for LLM consumption.

        Args:
            market_state: Market state containing dominance VuManChu indicators

        Returns:
            Formatted string with VuManChu dominance analysis
        """
        try:
            indicators = market_state.indicators

            # Check if we have VuManChu dominance analysis
            if not hasattr(indicators, "dominance_cipher_a_signal"):
                return "VuManChu dominance analysis not available"

            analysis_lines = []

            # Cipher A Dominance Analysis
            cipher_a_signal = getattr(indicators, "dominance_cipher_a_signal", 0)
            cipher_a_confidence = getattr(
                indicators, "dominance_cipher_a_confidence", 0.0
            )

            if cipher_a_signal != 0:
                signal_text = (
                    "🔴 BEARISH for crypto"
                    if cipher_a_signal > 0
                    else "🟢 BULLISH for crypto"
                )
                analysis_lines.append(
                    f"Dominance Cipher A: {signal_text} (confidence: {cipher_a_confidence:.1f}%)"
                )
            else:
                analysis_lines.append("Dominance Cipher A: ⚪ NEUTRAL")

            # Cipher B Dominance Analysis
            cipher_b_signal = getattr(indicators, "dominance_cipher_b_signal", 0)
            cipher_b_confidence = getattr(
                indicators, "dominance_cipher_b_confidence", 0.0
            )

            if cipher_b_signal != 0:
                signal_text = (
                    "🔴 BEARISH for crypto"
                    if cipher_b_signal > 0
                    else "🟢 BULLISH for crypto"
                )
                analysis_lines.append(
                    f"Dominance Cipher B: {signal_text} (confidence: {cipher_b_confidence:.1f}%)"
                )
            else:
                analysis_lines.append("Dominance Cipher B: ⚪ NEUTRAL")

            # WaveTrend on Dominance
            wt1 = getattr(indicators, "dominance_wt1", None)
            wt2 = getattr(indicators, "dominance_wt2", None)
            if wt1 is not None and wt2 is not None:
                wt_condition = ""
                if wt2 > 60:
                    wt_condition = "📈 Overbought dominance (bearish for crypto)"
                elif wt2 < -60:
                    wt_condition = "📉 Oversold dominance (bullish for crypto)"
                else:
                    wt_condition = "📊 Neutral zone"
                analysis_lines.append(
                    f"Dominance WaveTrend: WT1={wt1:.1f}, WT2={wt2:.1f} - {wt_condition}"
                )

            # Price vs Dominance Divergence
            price_divergence = getattr(indicators, "dominance_price_divergence", "NONE")
            if price_divergence != "NONE":
                divergence_emoji = {
                    "BULLISH": "🚀",
                    "BEARISH": "🔻",
                    "HIDDEN_BULLISH": "💎",
                    "HIDDEN_BEARISH": "⚠️",
                }.get(price_divergence, "❓")
                analysis_lines.append(
                    f"Price-Dominance Divergence: {divergence_emoji} {price_divergence}"
                )
            else:
                analysis_lines.append(
                    "Price-Dominance Divergence: ➡️ No divergence detected"
                )

            # Overall Dominance Sentiment
            dominance_sentiment = getattr(indicators, "dominance_sentiment", "NEUTRAL")
            sentiment_emoji = {
                "STRONG_BULLISH": "🚀🚀",
                "BULLISH": "🚀",
                "NEUTRAL": "➡️",
                "BEARISH": "🔻",
                "STRONG_BEARISH": "🔻🔻",
            }.get(dominance_sentiment, "❓")
            analysis_lines.append(
                f"VuManChu Dominance Sentiment: {sentiment_emoji} {dominance_sentiment}"
            )

            # Key Insight
            analysis_lines.append("")
            analysis_lines.append(
                "📝 Key Insight: Dominance signals are INVERTED for crypto:"
            )
            analysis_lines.append(
                "   • Rising dominance = Money flowing to stables = Bearish for crypto"
            )
            analysis_lines.append(
                "   • Falling dominance = Money leaving stables = Bullish for crypto"
            )
            analysis_lines.append(
                "   • Cipher signals on dominance predict stablecoin flow direction"
            )

            return "\n".join(analysis_lines)

        except Exception as e:
            logger.warning(f"Error formatting VuManChu dominance analysis: {e}")
            return f"Error formatting VuManChu dominance analysis: {str(e)}"

    async def _get_financial_context(self, market_state: MarketState) -> str:
        """
        Get financial context from OmniSearch for enhanced market intelligence.

        Args:
            market_state: Current market state containing symbol and price data

        Returns:
            Formatted string with financial context and sentiment analysis
        """
        if not self._omnisearch_enabled or not self._omnisearch_client:
            return "OmniSearch disabled - no external market intelligence available"

        try:
            # Extract base symbol for searches
            base_symbol = market_state.symbol.split("-")[0]  # "BTC-USD" -> "BTC"

            context_sections = []

            # 1. Get crypto sentiment for current symbol
            if settings.omnisearch.enable_crypto_sentiment:
                try:
                    crypto_sentiment = (
                        await self._omnisearch_client.search_crypto_sentiment(
                            base_symbol
                        )
                    )
                    sentiment_section = [
                        f"=== {base_symbol} Sentiment Analysis ===",
                        f"Overall Sentiment: {crypto_sentiment.overall_sentiment.upper()} ({crypto_sentiment.sentiment_score:+.2f})",
                        f"Confidence: {crypto_sentiment.confidence:.1%}",
                        f"Sources: {crypto_sentiment.source_count}",
                    ]

                    if crypto_sentiment.news_sentiment is not None:
                        sentiment_section.append(
                            f"News Sentiment: {crypto_sentiment.news_sentiment:+.2f}"
                        )
                    if crypto_sentiment.social_sentiment is not None:
                        sentiment_section.append(
                            f"Social Sentiment: {crypto_sentiment.social_sentiment:+.2f}"
                        )
                    if crypto_sentiment.technical_sentiment is not None:
                        sentiment_section.append(
                            f"Technical Sentiment: {crypto_sentiment.technical_sentiment:+.2f}"
                        )

                    if crypto_sentiment.key_drivers:
                        sentiment_section.append(
                            f"Key Drivers: {', '.join(crypto_sentiment.key_drivers[:3])}"
                        )
                    if crypto_sentiment.risk_factors:
                        sentiment_section.append(
                            f"Risk Factors: {', '.join(crypto_sentiment.risk_factors[:3])}"
                        )

                    context_sections.append("\n".join(sentiment_section))

                except Exception as e:
                    logger.warning(
                        f"Failed to get crypto sentiment for {base_symbol}: {e}"
                    )
                    context_sections.append(
                        f"=== {base_symbol} Sentiment Analysis ===\nUnavailable - API error"
                    )

            # 2. Get NASDAQ sentiment for market context
            if settings.omnisearch.enable_nasdaq_sentiment:
                try:
                    nasdaq_sentiment = (
                        await self._omnisearch_client.search_nasdaq_sentiment()
                    )
                    nasdaq_section = [
                        "=== NASDAQ Market Sentiment ===",
                        f"Overall Sentiment: {nasdaq_sentiment.overall_sentiment.upper()} ({nasdaq_sentiment.sentiment_score:+.2f})",
                        f"Confidence: {nasdaq_sentiment.confidence:.1%}",
                    ]

                    if nasdaq_sentiment.key_drivers:
                        nasdaq_section.append(
                            f"Key Market Drivers: {', '.join(nasdaq_sentiment.key_drivers[:2])}"
                        )
                    if nasdaq_sentiment.risk_factors:
                        nasdaq_section.append(
                            f"Market Risks: {', '.join(nasdaq_sentiment.risk_factors[:2])}"
                        )

                    context_sections.append("\n".join(nasdaq_section))

                except Exception as e:
                    logger.warning(f"Failed to get NASDAQ sentiment: {e}")
                    context_sections.append(
                        "=== NASDAQ Market Sentiment ===\nUnavailable - API error"
                    )

            # 3. Get correlation analysis between crypto and traditional markets
            if settings.omnisearch.enable_correlation_analysis:
                try:
                    correlation = (
                        await self._omnisearch_client.search_market_correlation(
                            base_symbol, "QQQ"
                        )
                    )
                    correlation_section = [
                        f"=== {base_symbol}-NASDAQ Correlation ===",
                        f"Correlation: {correlation.direction.upper()} {correlation.strength.upper()} ({correlation.correlation_coefficient:+.3f})",
                        f"Timeframe: {correlation.timeframe}",
                    ]

                    if correlation.beta is not None:
                        correlation_section.append(f"Beta: {correlation.beta:.2f}")

                    # Interpret correlation for trading context
                    if abs(correlation.correlation_coefficient) > 0.5:
                        if correlation.correlation_coefficient > 0:
                            correlation_section.append(
                                "⚠️ Strong positive correlation - crypto may follow stock market moves"
                            )
                        else:
                            correlation_section.append(
                                "📈 Strong negative correlation - crypto may move opposite to stocks"
                            )
                    else:
                        correlation_section.append(
                            "➡️ Weak correlation - crypto moving independently from stocks"
                        )

                    context_sections.append("\n".join(correlation_section))

                except Exception as e:
                    logger.warning(
                        f"Failed to get market correlation for {base_symbol}: {e}"
                    )
                    context_sections.append(
                        f"=== {base_symbol}-NASDAQ Correlation ===\nUnavailable - API error"
                    )

            # 4. Get recent financial news
            try:
                # Search for crypto-specific news
                crypto_news = await self._omnisearch_client.search_financial_news(
                    f"{base_symbol} cryptocurrency", limit=3, timeframe="24h"
                )

                if crypto_news:
                    news_section = [f"=== Recent {base_symbol} News (24h) ==="]
                    for news in crypto_news[:3]:
                        sentiment_emoji = {
                            "positive": "🟢",
                            "negative": "🔴",
                            "neutral": "⚪",
                        }.get(news.sentiment or "neutral", "❓")

                        news_line = (
                            f"{sentiment_emoji} {news.base_result.title[:80]}..."
                        )
                        if news.impact_level:
                            news_line += f" [{news.impact_level.upper()} IMPACT]"
                        news_section.append(news_line)

                    context_sections.append("\n".join(news_section))
                else:
                    context_sections.append(
                        f"=== Recent {base_symbol} News (24h) ===\nNo recent news found"
                    )

            except Exception as e:
                logger.warning(f"Failed to get financial news for {base_symbol}: {e}")
                context_sections.append(
                    f"=== Recent {base_symbol} News (24h) ===\nUnavailable - API error"
                )

            # Combine all sections
            if context_sections:
                full_context = "\n\n".join(context_sections)

                # Add interpretation guidance
                full_context += "\n\n=== Market Intelligence Summary ===\n"
                full_context += "Use this external intelligence to validate or challenge your technical analysis.\n"
                full_context += "Strong sentiment divergence from technicals may indicate potential reversals.\n"
                full_context += (
                    "High market correlations suggest macro risk-on/risk-off dynamics."
                )

                logger.info(
                    f"🔍 OmniSearch: Retrieved financial context for {base_symbol}"
                )
                return full_context
            else:
                return "No financial context available from OmniSearch"

        except Exception as e:
            logger.error(f"Error getting financial context: {e}")
            return f"Error retrieving financial context: {str(e)}"

    async def _get_scalping_analysis(self, market_state: MarketState) -> Dict[str, Any]:
        """
        Get scalping analysis - currently disabled after module removal.

        Args:
            market_state: Current market state containing OHLCV data

        Returns:
            Dictionary indicating scalping is disabled
        """
        # Method kept for compatibility but always returns disabled status
        return {
            "scalping_enabled": False,
            "ema_trend": "N/A",
            "ema_alignment": "N/A",
            "ema_crossovers": "N/A",
            "ema_strength": "N/A",
            "fast_rsi_signal": "N/A",
            "rsi_strength": "N/A",
            "fast_macd_signal": "N/A",
            "macd_histogram": "N/A",
            "williams_signal": "N/A",
            "williams_level": "N/A",
            "momentum_consensus": "N/A",
            "vwap_position": "N/A",
            "volume_relative": "N/A",
            "obv_trend": "N/A",
            "volume_confirmation": "N/A",
            "entry_confidence": 0,
            "supporting_indicators": [],
            "risk_factors": [],
            "scalping_consensus": "DISABLED",
        }

    async def _get_llm_decision(self, llm_input: Dict[str, Any]) -> TradeAction:
        """
        Get decision from LLM using LangChain with enhanced logging and optimized prompts.

        Args:
            llm_input: Formatted input for the LLM

        Returns:
            TradeAction from LLM analysis
        """
        request_id = None
        start_time = time.time()

        try:
            self._completion_count += 1

            # LangChain callback handler will handle the logging automatically

            # Invoke the chain with callback if available
            chain_kwargs = (
                {"config": {"callbacks": [self._langchain_callback]}}
                if self._langchain_callback
                else {}
            )

            # For o3 models, we need to handle the response differently due to token usage format issues
            if self.model_name.startswith("o3"):
                # Use custom response handling for o3 models - bypass the chain and parse manually
                from langchain_core.messages import HumanMessage

                # Format the prompt manually - use optimized prompt if available
                if self._use_optimized_prompts and self._optimized_prompt_template:
                    formatted_prompt = self._optimized_prompt_template.format_prompt(
                        llm_input
                    )
                    logger.debug(
                        f"Using optimized prompt (size: {len(formatted_prompt)} chars)"
                    )
                else:
                    if self._prompt_template is None or self._model is None:
                        raise RuntimeError("LLM components not properly initialized")
                    formatted_prompt = self._prompt_template.format(**llm_input)
                    logger.debug(
                        f"Using standard prompt (size: {len(formatted_prompt)} chars)"
                    )

                message = HumanMessage(content=formatted_prompt)

                # Get raw response from model
                if self._model is None:
                    raise RuntimeError("Model not initialized")
                raw_response = await self._model.ainvoke([message])
                response_content = raw_response.content

                # Log the full response for debugging
                logger.info(f"Full LLM Response:\n{response_content}")

                # Extract JSON from the response
                result = self._extract_json_from_response(response_content)
            else:
                if self._chain is None:
                    raise RuntimeError("LLM chain not properly initialized")

                # Use optimized prompt if available
                if self._use_optimized_prompts and self._optimized_prompt_template:
                    # Get formatted prompt directly from optimized template
                    optimized_prompt_text = (
                        self._optimized_prompt_template.format_prompt(llm_input)
                    )

                    # For optimized prompts, use the same approach as o3 models
                    # since the prompt is already formatted
                    from langchain_core.messages import HumanMessage

                    message = HumanMessage(content=optimized_prompt_text)

                    # Get raw response from model
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    raw_response = await self._model.ainvoke([message])
                    response_content = raw_response.content

                    # Extract JSON from the response
                    result = self._extract_json_from_response(response_content)
                    logger.debug(
                        f"Used optimized prompt (size: {len(optimized_prompt_text)} chars)"
                    )
                else:
                    result = await self._chain.ainvoke(llm_input, **chain_kwargs)

            response_time = time.time() - start_time

            # Record performance metrics for prompt optimization
            if self._use_optimized_prompts and self._optimized_prompt_template:
                # Estimate decision quality (simplified for now)
                decision_quality = 1.0 if result.get("action") != "HOLD" else 0.8
                # Note: In production, you might want to track actual trade outcomes

                # Update prompt optimizer with performance data
                # This would be implemented in the optimizer class

            # Ensure result is a TradeAction
            if isinstance(result, dict):
                # Validate action field to ensure it's a valid literal
                if "action" in result and result["action"] not in [
                    "LONG",
                    "SHORT",
                    "CLOSE",
                    "HOLD",
                ]:
                    logger.warning(
                        f"Invalid action '{result['action']}', defaulting to HOLD"
                    )
                    result["action"] = "HOLD"
                trade_action = TradeAction(**result)
            elif isinstance(result, TradeAction):
                trade_action = result
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")

            # Performance metrics logging (if enabled)
            if (
                self._completion_logger
                and self._completion_count % settings.llm.performance_log_interval == 0
            ):
                self._completion_logger.log_performance_metrics()

            # Record performance metrics
            if self._enable_performance_tracking:
                prompt_size = (
                    len(formatted_prompt)
                    if "formatted_prompt" in locals()
                    else len(str(llm_input))
                )
                optimization_level = (
                    "optimized" if self._use_optimized_prompts else "standard"
                )

                record_llm_performance(
                    response_time_ms=response_time * 1000,
                    prompt_size_chars=prompt_size,
                    cache_hit=False,  # This path is always cache miss
                    decision_action=trade_action.action,
                    optimization_level=optimization_level,
                    error_occurred=False,
                )

            # Log performance improvement
            if response_time < 2.0:
                logger.info(
                    f"⚡ Fast LLM response: {response_time:.2f}s (target: <2.0s)"
                )
            elif response_time > 3.0:
                logger.warning(
                    f"🐌 Slow LLM response: {response_time:.2f}s (target: <2.0s)"
                )

            return trade_action

        except Exception as e:
            logger.error(f"LLM decision error: {e}")

            # Record error in performance metrics
            if self._enable_performance_tracking:
                response_time = time.time() - start_time
                prompt_size = len(str(llm_input))

                record_llm_performance(
                    response_time_ms=response_time * 1000,
                    prompt_size_chars=prompt_size,
                    cache_hit=False,
                    decision_action="ERROR",
                    optimization_level="standard",
                    error_occurred=True,
                )

            raise

    async def _get_cached_llm_decision(self, market_state: MarketState) -> TradeAction:
        """
        Get LLM decision for caching system.

        This method is called by the cache when a cache miss occurs.

        Args:
            market_state: Market state for analysis

        Returns:
            TradeAction from LLM analysis
        """
        # Prepare input data for the LLM
        llm_input = await self._prepare_llm_input(market_state)

        # Get decision from LLM
        return await self._get_llm_decision(llm_input)

    def _calculate_cipher_b_alignment(self, indicators: Any) -> str:
        """
        Calculate and describe Cipher B signal alignment.

        Args:
            indicators: IndicatorData object

        Returns:
            String describing Cipher B signal alignment
        """
        if indicators.cipher_b_wave is None or indicators.cipher_b_money_flow is None:
            return "Cipher B indicators not available"

        wave = indicators.cipher_b_wave
        money_flow = indicators.cipher_b_money_flow

        # Determine signal states
        wave_bullish = wave > 0.0
        wave_bearish = wave < 0.0
        money_flow_bullish = money_flow > 50.0
        money_flow_bearish = money_flow < 50.0

        # Check alignments
        bullish_aligned = wave_bullish and money_flow_bullish
        bearish_aligned = wave_bearish and money_flow_bearish

        # Build alignment description
        lines = []
        lines.append(f"Wave: {wave:.2f} ({'bullish' if wave_bullish else 'bearish'})")
        lines.append(
            f"Money Flow: {money_flow:.2f} ({'bullish' if money_flow_bullish else 'bearish'})"
        )

        if bullish_aligned:
            lines.append("✓ BULLISH ALIGNMENT - Both signals confirm upward momentum")
            lines.append("Traditional Cipher B would trigger LONG here")
        elif bearish_aligned:
            lines.append("✓ BEARISH ALIGNMENT - Both signals confirm downward momentum")
            lines.append("Traditional Cipher B would trigger SHORT here")
        else:
            lines.append("⚠ MIXED SIGNALS - Wave and Money Flow disagree")
            lines.append("Traditional Cipher B would wait for alignment")

        # Add signal strength
        wave_strength = abs(wave)
        if wave_strength > 60:
            lines.append(f"Wave strength: STRONG ({wave_strength:.1f})")
        elif wave_strength > 30:
            lines.append(f"Wave strength: MODERATE ({wave_strength:.1f})")
        else:
            lines.append(f"Wave strength: WEAK ({wave_strength:.1f})")

        return "\n".join(lines)

    def _get_fallback_decision(self, market_state: MarketState) -> TradeAction:
        """
        Fallback decision logic when LLM is not available.

        Args:
            market_state: Market state for analysis

        Returns:
            TradeAction based on simple technical rules
        """
        # Simple fallback logic based on indicators
        indicators = market_state.indicators
        current_pos = market_state.current_position

        # Default to hold
        action = "HOLD"
        size_pct = 0

        # Check dominance for market sentiment
        dominance_bias = 0  # -1 = bearish, 0 = neutral, 1 = bullish
        if indicators.stablecoin_dominance is not None:
            if indicators.stablecoin_dominance > 10:  # High dominance = bearish
                dominance_bias = -1
            elif indicators.stablecoin_dominance < 5:  # Low dominance = bullish
                dominance_bias = 1

            # Check dominance trend
            if indicators.dominance_trend is not None:
                if indicators.dominance_trend > 0.5:  # Rising dominance = bearish
                    dominance_bias = min(dominance_bias - 1, -1)
                elif indicators.dominance_trend < -0.5:  # Falling dominance = bullish
                    dominance_bias = max(dominance_bias + 1, 1)

        # Simple trend-following logic with dominance adjustment
        if indicators.cipher_a_dot and indicators.cipher_b_wave:
            if indicators.cipher_a_dot > 0 and indicators.cipher_b_wave > 0:
                if current_pos.side == "FLAT" and dominance_bias >= 0:
                    action = "LONG"
                    size_pct = (
                        10 if dominance_bias == 0 else 15
                    )  # Larger size if bullish dominance
            elif indicators.cipher_a_dot < 0 and indicators.cipher_b_wave < 0:
                if current_pos.side == "FLAT" and dominance_bias <= 0:
                    action = "SHORT"
                    size_pct = (
                        10 if dominance_bias == 0 else 15
                    )  # Larger size if bearish dominance
                elif current_pos.side == "LONG":
                    action = "CLOSE"

        # Determine appropriate leverage for futures
        leverage = None
        reduce_only = False

        if settings.trading.enable_futures:
            leverage = min(
                settings.trading.leverage, settings.trading.max_futures_leverage
            )
            # Set reduce_only for closing trades
            if action == "CLOSE" or (
                action in ["LONG", "SHORT"] and current_pos.side != "FLAT"
            ):
                reduce_only = True

        omnisearch_status = (
            " (OmniSearch enabled)"
            if self._omnisearch_enabled
            else " (OmniSearch disabled)"
        )

        # Cast action to proper literal type
        from typing import cast, Literal

        valid_action = cast(Literal["LONG", "SHORT", "CLOSE", "HOLD"], action)

        return TradeAction(
            action=valid_action,
            size_pct=size_pct,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            leverage=leverage if leverage is not None else 1,
            reduce_only=reduce_only,
            rationale=f"Fallback logic - simple trend following{omnisearch_status}",
        )

    def is_available(self) -> bool:
        """
        Check if the LLM agent is properly initialized and available.

        Returns:
            True if agent can make decisions
        """
        return self._chain is not None or True  # Fallback always available

    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about the LLM agent including logging and caching status.

        Returns:
            Dictionary with agent status and logging information
        """
        status = {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "temperature": (
                self.temperature if self.temperature is not None else "N/A (o3 model)"
            ),
            "llm_available": self._chain is not None,
            "prompt_loaded": self._prompt_template is not None,
            "completion_logging_enabled": settings.llm.enable_completion_logging,
            "completion_count": self._completion_count,
            "omnisearch_enabled": self._omnisearch_enabled,
            "omnisearch_client_available": self._omnisearch_client is not None,
            "scalping_signals_enabled": self._scalping_enabled,
            "scalping_signals_available": self._scalping_signals is not None,
            "cache_enabled": self._cache_enabled,
            "optimized_prompts_enabled": self._use_optimized_prompts,
            "performance_monitoring_enabled": self._enable_performance_tracking,
        }

        # Add cache statistics if available
        if self._cache:
            try:
                cache_stats = self._cache.get_cache_stats()
                status.update(
                    {
                        "cache_stats": cache_stats,
                        "cache_hit_rate": cache_stats.get("hit_rate", 0),
                        "cache_size": cache_stats.get("cache_size", 0),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not retrieve cache statistics: {e}")

        # Add performance optimization metrics
        if self._enable_performance_tracking and self._performance_monitor:
            try:
                perf_stats = self._performance_monitor.get_current_stats()
                status.update(
                    {
                        "performance_optimization": {
                            "avg_response_time_ms": perf_stats.avg_response_time_ms,
                            "target_achieved": perf_stats.target_achieved,
                            "performance_improvement_pct": perf_stats.performance_improvement_pct,
                            "cache_hit_rate": perf_stats.cache_hit_rate,
                            "error_rate": perf_stats.error_rate,
                            "total_requests": perf_stats.total_requests,
                        }
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Could not retrieve performance optimization metrics: {e}"
                )

        # Add performance metrics if completion logger is available
        if self._completion_logger and settings.llm.enable_performance_tracking:
            try:
                performance_metrics = self._completion_logger.log_performance_metrics()
                status.update(
                    {
                        "performance_metrics": performance_metrics,
                        "avg_response_time_ms": performance_metrics.get(
                            "avg_response_time_ms", 0
                        ),
                        "total_cost_estimate_usd": performance_metrics.get(
                            "total_cost_estimate_usd", 0
                        ),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not retrieve performance metrics: {e}")

        return status

    def _extract_json_from_response(self, response_content: str) -> Dict[str, Any]:
        """
        Extract JSON from a verbose response that contains both analysis and JSON.

        Args:
            response_content: Full response content from LLM

        Returns:
            Dictionary containing the JSON decision
        """
        import json
        import re

        try:
            # Look for JSON_DECISION: followed by JSON
            json_pattern = r"JSON_DECISION:\s*(\{.*?\})"
            match = re.search(json_pattern, response_content, re.DOTALL)

            if match:
                json_str = match.group(1)
                # Clean up the JSON string
                json_str = json_str.strip()
                return json.loads(json_str)

            # Fallback: look for any JSON-like structure
            json_pattern = r'\{[^{}]*"action"[^{}]*\}'
            match = re.search(json_pattern, response_content, re.DOTALL)

            if match:
                json_str = match.group(0)
                return json.loads(json_str)

            # Last resort: try to find just the values and construct JSON
            logger.warning(
                "Could not find JSON in response, attempting to parse manually"
            )

            # Extract individual fields
            action_match = re.search(r'"action":\s*"([^"]+)"', response_content)
            size_match = re.search(r'"size_pct":\s*(\d+(?:\.\d+)?)', response_content)
            tp_match = re.search(
                r'"take_profit_pct":\s*(\d+(?:\.\d+)?)', response_content
            )
            sl_match = re.search(
                r'"stop_loss_pct":\s*(\d+(?:\.\d+)?)', response_content
            )
            leverage_match = re.search(r'"leverage":\s*(\d+)', response_content)
            reduce_match = re.search(r'"reduce_only":\s*(true|false)', response_content)
            rationale_match = re.search(r'"rationale":\s*"([^"]+)"', response_content)

            if action_match:
                return {
                    "action": action_match.group(1),
                    "size_pct": float(size_match.group(1)) if size_match else 0,
                    "take_profit_pct": float(tp_match.group(1)) if tp_match else 2.0,
                    "stop_loss_pct": float(sl_match.group(1)) if sl_match else 1.0,
                    "leverage": int(leverage_match.group(1)) if leverage_match else 1,
                    "reduce_only": (
                        reduce_match.group(1) == "true" if reduce_match else False
                    ),
                    "rationale": (
                        rationale_match.group(1)
                        if rationale_match
                        else "Parsed from verbose response"
                    ),
                }

            # Final fallback - return HOLD
            logger.error(
                "Could not parse any decision from response, defaulting to HOLD"
            )
            return {
                "action": "HOLD",
                "size_pct": 0,
                "take_profit_pct": 1.0,
                "stop_loss_pct": 1.0,
                "leverage": 1,
                "reduce_only": False,
                "rationale": "Failed to parse response",
            }

        except Exception as e:
            logger.error(f"Error parsing JSON from response: {e}")
            # Return safe default
            return {
                "action": "HOLD",
                "size_pct": 0,
                "take_profit_pct": 1.0,
                "stop_loss_pct": 1.0,
                "leverage": 1,
                "reduce_only": False,
                "rationale": "JSON parsing error",
            }

    def log_decision_with_validation(
        self,
        request_id: str,
        trade_action: TradeAction,
        market_state: MarketState,
        validation_result: Optional[str] = None,
        risk_assessment: Optional[str] = None,
    ) -> None:
        """
        Log a trading decision with validation and risk assessment results.

        This method can be called from the main trading loop to include
        post-LLM processing information in the logs.

        Args:
            request_id: Request ID from the original LLM completion
            trade_action: Final trade action after validation/risk management
            market_state: Market state at time of decision
            validation_result: Result of trade validation
            risk_assessment: Risk manager assessment
        """
        if self._completion_logger and settings.llm.enable_completion_logging:
            self._completion_logger.log_trading_decision(
                request_id=request_id,
                trade_action=trade_action,
                market_state=market_state,
                validation_result=validation_result,
                risk_assessment=risk_assessment,
            )
