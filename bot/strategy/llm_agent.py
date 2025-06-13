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
from typing import Any, Optional

try:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI
except ImportError:
    # Graceful degradation if LangChain not installed
    ChatOpenAI = None
    PromptTemplate = None
    JsonOutputParser = None
    RunnablePassthrough = None

from ..config import settings
from ..logging import create_llm_logger, create_langchain_callback
from ..types import MarketState, TradeAction

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LangChain-powered LLM agent for trading decisions.

    Analyzes market state including OHLCV data, technical indicators,
    and current position to generate structured trading actions.
    """

    def __init__(self, model_provider: str = None, model_name: str = None):
        """
        Initialize the LLM agent.

        Args:
            model_provider: LLM provider ('openai', 'ollama')
            model_name: Specific model to use
        """
        self.model_provider = model_provider or settings.llm.provider
        self.model_name = model_name or settings.llm.model_name
        self.temperature = settings.llm.temperature

        # LangChain components
        self._model = None
        self._prompt_template = None
        self._chain = None

        # Enhanced logging components
        self._completion_logger: Optional[Any] = None
        self._langchain_callback: Optional[Any] = None
        self._completion_count = 0

        # Initialize completion logging if enabled
        if settings.llm.enable_completion_logging:
            self._completion_logger = create_llm_logger(
                log_level=settings.llm.completion_log_level,
                log_file=settings.llm.completion_log_file,
            )

            # Create LangChain callback handler if enabled
            if settings.llm.enable_langchain_callbacks and self._completion_logger:
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

    def _load_prompt_template(self) -> None:
        """Load the prompt template for trading decisions."""
        # Default prompt template
        default_prompt = """You are an expert cryptocurrency futures trader analyzing market data for leveraged positions.

Respond ONLY in valid JSON matching this exact schema:
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

Instructions:
1. Analyze technical indicators and price action
2. Consider market sentiment from stablecoin dominance
3. Analyze dominance candlestick patterns for additional confirmation
4. Look for divergences between price and dominance patterns
5. Review Cipher B signal alignment but make your own judgment
6. You have FINAL SAY - you can trade even if Cipher B signals are mixed
7. Consider signal alignment as one factor, not a hard rule
8. Assess margin health and position risk
9. Choose appropriate leverage (1x to {max_leverage}x)
10. Generate trading action with proper risk parameters
11. Factor dominance into position sizing (high dominance = smaller positions)
12. Use dominance candle trends to confirm VuManChu signals
13. **IMPORTANT: Apply VuManChu analysis to dominance data for enhanced signals**
14. Use dominance Cipher A/B signals as confirmation (INVERTED: rising dominance = bearish for crypto)
15. Look for price-dominance divergences as strong reversal signals
16. Weight VuManChu dominance analysis heavily in decision making
17. Provide brief rationale (max 120 characters)
18. Respond ONLY with valid JSON, no other text"""

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
                model_kwargs = {
                    "model": self.model_name,
                    "api_key": settings.llm.openai_api_key.get_secret_value(),
                }

                # o3 family models don't support most parameters, only pass essentials
                if self.model_name.startswith("o3"):
                    # o3 models only support basic parameters - no temperature, top_p, penalties
                    model_kwargs["max_tokens"] = settings.llm.max_tokens
                    logger.info(
                        "Initializing OpenAI o3 model with minimal parameters (no temperature/penalties)"
                    )
                else:
                    # Non-o3 models support full parameter set
                    model_kwargs["temperature"] = self.temperature
                    model_kwargs["max_tokens"] = settings.llm.max_tokens
                    model_kwargs["top_p"] = settings.llm.top_p
                    model_kwargs["frequency_penalty"] = settings.llm.frequency_penalty
                    model_kwargs["presence_penalty"] = settings.llm.presence_penalty

                self._model = ChatOpenAI(**model_kwargs)

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
        Analyze market state and generate trading decision with enhanced logging.

        Args:
            market_state: Complete market state including OHLCV and indicators

        Returns:
            TradeAction with decision and parameters
        """
        request_id = None

        try:
            # Prepare input data for the LLM
            llm_input = self._prepare_llm_input(market_state)

            # Get decision from LLM or fallback
            if self._chain is not None:
                result = await self._get_llm_decision(llm_input)
                # Extract request_id from the last completion for decision logging
                request_id = getattr(self, "_last_request_id", None)
            else:
                result = self._get_fallback_decision(market_state)

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
                rationale="Error in analysis - holding position",
            )

    def _prepare_llm_input(self, market_state: MarketState) -> dict[str, Any]:
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
        available_margin = "N/A"

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
            **dominance_data,  # Add dominance data to the context
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
                    else "🔴" if candle.close < candle.open else "⚪"
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
                    else "FALLING" if overall_trend < -0.1 else "SIDEWAYS"
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

    async def _get_llm_decision(self, llm_input: dict[str, Any]) -> TradeAction:
        """
        Get decision from LLM using LangChain with enhanced logging.

        Args:
            llm_input: Formatted input for the LLM

        Returns:
            TradeAction from LLM analysis
        """
        request_id = None
        start_time = time.time()

        try:
            self._completion_count += 1

            # Log the request if completion logging is enabled
            if self._completion_logger and settings.llm.enable_completion_logging:
                # Create prompt for logging (simulate what will be sent)
                formatted_prompt = (
                    self._prompt_template.format(**llm_input)
                    if self._prompt_template
                    else str(llm_input)
                )

                # Extract market context for logging
                market_context = (
                    {
                        "symbol": llm_input.get("symbol"),
                        "current_price": llm_input.get("current_price"),
                        "current_position": llm_input.get("current_position"),
                        "cipher_a_dot": llm_input.get("cipher_a_dot"),
                        "cipher_b_wave": llm_input.get("cipher_b_wave"),
                        "rsi": llm_input.get("rsi"),
                        "stablecoin_dominance": llm_input.get("stablecoin_dominance"),
                        "market_sentiment": llm_input.get("market_sentiment"),
                    }
                    if settings.llm.log_market_context
                    else None
                )

                request_id = self._completion_logger.log_completion_request(
                    prompt=formatted_prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=settings.llm.max_tokens,
                    market_context=market_context,
                )

                # Store request_id for decision logging
                self._last_request_id = request_id

            # Invoke the chain with callback if available
            chain_kwargs = (
                {"config": {"callbacks": [self._langchain_callback]}}
                if self._langchain_callback
                else {}
            )
            result = await self._chain.ainvoke(llm_input, **chain_kwargs)

            response_time = time.time() - start_time

            # Ensure result is a TradeAction
            if isinstance(result, dict):
                trade_action = TradeAction(**result)
            elif isinstance(result, TradeAction):
                trade_action = result
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")

            # Log successful response
            if self._completion_logger and request_id:
                # Note: Token usage would need to be extracted from the response
                # This is a limitation of the current LangChain integration
                self._completion_logger.log_completion_response(
                    request_id=request_id,
                    response=trade_action,
                    response_time=response_time,
                    token_usage=None,  # Would need special handling to extract from OpenAI response
                    success=True,
                )

                # Log performance metrics periodically
                if (
                    self._completion_count % settings.llm.performance_log_interval
                ) == 0:
                    self._completion_logger.log_performance_metrics()

            return trade_action

        except Exception as e:
            response_time = time.time() - start_time

            # Log failed response
            if self._completion_logger and request_id:
                self._completion_logger.log_completion_response(
                    request_id=request_id,
                    response=None,
                    response_time=response_time,
                    success=False,
                    error=str(e),
                )

            logger.error(f"LLM decision error: {e}")
            raise

    def _calculate_cipher_b_alignment(self, indicators) -> str:
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

        return TradeAction(
            action=action,
            size_pct=size_pct,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            leverage=leverage,
            reduce_only=reduce_only,
            rationale="Fallback logic - simple trend following",
        )

    def is_available(self) -> bool:
        """
        Check if the LLM agent is properly initialized and available.

        Returns:
            True if agent can make decisions
        """
        return self._chain is not None or True  # Fallback always available

    def get_status(self) -> dict[str, Any]:
        """
        Get status information about the LLM agent including logging status.

        Returns:
            Dictionary with agent status and logging information
        """
        status = {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "llm_available": self._chain is not None,
            "prompt_loaded": self._prompt_template is not None,
            "completion_logging_enabled": settings.llm.enable_completion_logging,
            "completion_count": self._completion_count,
        }

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
