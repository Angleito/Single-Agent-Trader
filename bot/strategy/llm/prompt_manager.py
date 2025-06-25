"""Prompt template management for LLM trading agents.

This module provides a centralized system for managing prompt templates used by
the LLM trading agents. It supports loading prompts from files, using default
prompts, and formatting prompts with dynamic variables.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt templates for LLM trading agents."""

    def __init__(self, prompt_file_path: Path | None = None):
        """Initialize the PromptManager.

        Args:
            prompt_file_path: Optional path to a custom prompt file.
                             If not provided, defaults to prompts/trade_action.txt
        """
        self.prompt_file_path = prompt_file_path
        self._prompt_template: str | None = None
        self._load_prompt_template()

    def _load_prompt_template(self) -> None:
        """Load the prompt template from file or use default."""
        # Try to load from file first
        if self.prompt_file_path is None:
            try:
                from bot.utils.path_utils import get_project_root

                self.prompt_file_path = (
                    get_project_root() / "prompts" / "trade_action.txt"
                )
            except Exception as e:
                logger.warning(f"Failed to determine project root: {e}")
                self.prompt_file_path = None

        if self.prompt_file_path and self.prompt_file_path.exists():
            try:
                self._prompt_template = self.prompt_file_path.read_text()
                logger.info(
                    f"Loaded prompt template from file: {self.prompt_file_path}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load prompt file: {e}")

        # Fall back to default prompt
        self._prompt_template = self.get_default_prompt()
        logger.info("Using default prompt template")

    def load_template(self, file_path: Path) -> str:
        """Load a prompt template from a specific file.

        Args:
            file_path: Path to the prompt template file

        Returns:
            The loaded prompt template text

        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        try:
            template = file_path.read_text()
            logger.info(f"Loaded prompt template from: {file_path}")
            return template
        except Exception as e:
            logger.exception(f"Error reading prompt file: {e}")
            raise OSError(f"Failed to read prompt file: {e}")

    def get_default_prompt(self) -> str:
        """Get the default trading prompt template.

        Returns:
            The default prompt template string
        """
        return """You are an expert cryptocurrency momentum trader operating on 5-minute timeframes for leveraged futures positions.

TRADING PHILOSOPHY: You are a momentum-based trader. You trade ONLY when you detect strong momentum signals that develop within 5-minute candles. If no clear momentum is present, you WAIT for the next 5-minute candle to complete.

Respond with BOTH a detailed analysis AND valid JSON. Format your response as:

MOMENTUM ANALYSIS:
[Provide a detailed 3-4 paragraph analysis explaining your momentum assessment, what signals you're seeing, why you're making this decision, and how you evaluated the 5-minute timeframe data]

JSON_DECISION:
{
  "action": "LONG|SHORT|CLOSE|HOLD",
  "size_pct": 0-100,
  "take_profit_pct": positive_number_greater_than_0,
  "stop_loss_pct": positive_number_greater_than_0,
  "leverage": positive_integer_1_or_greater,
  "reduce_only": boolean,
  "rationale": "string"
}

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

    def format_prompt(self, **kwargs: Any) -> str:
        """Format the prompt template with provided variables.

        Args:
            **kwargs: Variables to substitute in the prompt template

        Returns:
            The formatted prompt string

        Raises:
            KeyError: If a required template variable is missing
            ValueError: If the prompt template is not loaded
        """
        if self._prompt_template is None:
            raise ValueError("Prompt template not loaded")

        try:
            return self._prompt_template.format(**kwargs)
        except KeyError as e:
            logger.exception(f"Missing required template variable: {e}")
            raise KeyError(f"Missing required template variable: {e}")
        except Exception as e:
            logger.exception(f"Error formatting prompt: {e}")
            raise ValueError(f"Failed to format prompt: {e}")

    def get_template_variables(self) -> set:
        """Extract all template variables from the current prompt.

        Returns:
            Set of variable names found in the template
        """
        if self._prompt_template is None:
            return set()

        import re

        # Match {variable_name} patterns
        pattern = r"\{([^}]+)\}"
        matches = re.findall(pattern, self._prompt_template)
        return set(matches)

    def validate_variables(self, variables: dict[str, Any]) -> bool:
        """Validate that all required template variables are provided.

        Args:
            variables: Dictionary of variables to check

        Returns:
            True if all required variables are present, False otherwise
        """
        required_vars = self.get_template_variables()
        provided_vars = set(variables.keys())
        missing_vars = required_vars - provided_vars

        if missing_vars:
            logger.warning(f"Missing template variables: {missing_vars}")
            return False

        return True

    def reload_template(self) -> None:
        """Reload the prompt template from file or default."""
        self._load_prompt_template()

    @property
    def prompt_template(self) -> str:
        """Get the current prompt template.

        Returns:
            The current prompt template string

        Raises:
            ValueError: If no prompt template is loaded
        """
        if self._prompt_template is None:
            raise ValueError("No prompt template loaded")
        return self._prompt_template
