"""
Financial context provider for OmniSearch integration.

This module extracts financial context gathering logic from the LLM agent
into a dedicated provider class for better modularity and reusability.
"""

import logging
from typing import TYPE_CHECKING

from bot.trading_types import MarketState

if TYPE_CHECKING:
    # Legacy imports (maintained for compatibility)
    from bot.config import Settings
    from bot.mcp.omnisearch_client import OmniSearchClient

    # Functional imports (added for migration to functional programming patterns)
    try:
        from bot.fp.types.config import Config, SystemConfig
        from bot.fp.types.market import MarketSnapshot

        FUNCTIONAL_AVAILABLE = True
    except ImportError:
        # Functional implementations not available, continue with legacy
        FUNCTIONAL_AVAILABLE = False

logger = logging.getLogger(__name__)


class FinancialContextProvider:
    """
    Provides financial context from OmniSearch for enhanced market intelligence.

    This class encapsulates all logic for gathering external market data,
    sentiment analysis, and financial news to enhance trading decisions.
    """

    def __init__(self, omnisearch_client: "OmniSearchClient", settings: "Settings"):
        """
        Initialize the financial context provider.

        Args:
            omnisearch_client: Connected OmniSearch client instance
            settings: Application settings containing OmniSearch configuration
        """
        self._omnisearch_client = omnisearch_client
        self._settings = settings
        self._omnisearch_enabled = (
            settings.omnisearch.enabled and omnisearch_client is not None
        )

    async def get_financial_context(self, market_state: MarketState) -> str:
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
            if self._settings.omnisearch.enable_crypto_sentiment:
                crypto_section = await self._get_crypto_sentiment(base_symbol)
                context_sections.append(crypto_section)

            # 2. Get NASDAQ sentiment for market context
            if self._settings.omnisearch.enable_nasdaq_sentiment:
                nasdaq_section = await self._get_nasdaq_sentiment()
                context_sections.append(nasdaq_section)

            # 3. Get correlation analysis between crypto and traditional markets
            if self._settings.omnisearch.enable_correlation_analysis:
                correlation_section = await self._get_market_correlation(base_symbol)
                context_sections.append(correlation_section)

            # 4. Get recent financial news
            news_section = await self._get_financial_news(base_symbol)
            context_sections.append(news_section)

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
            return "No financial context available from OmniSearch"

        except Exception as e:
            logger.exception(f"Error getting financial context: {e}")
            return f"Error retrieving financial context: {e!s}"

    async def _get_crypto_sentiment(self, base_symbol: str) -> str:
        """
        Get cryptocurrency sentiment analysis.

        Args:
            base_symbol: Base cryptocurrency symbol (e.g., "BTC", "ETH")

        Returns:
            Formatted sentiment section string
        """
        try:
            crypto_sentiment = await self._omnisearch_client.search_crypto_sentiment(
                base_symbol
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

            return "\n".join(sentiment_section)

        except Exception as e:
            logger.warning(f"Failed to get crypto sentiment for {base_symbol}: {e}")
            return f"=== {base_symbol} Sentiment Analysis ===\nUnavailable - API error"

    async def _get_nasdaq_sentiment(self) -> str:
        """
        Get NASDAQ/traditional market sentiment analysis.

        Returns:
            Formatted NASDAQ sentiment section string
        """
        try:
            nasdaq_sentiment = await self._omnisearch_client.search_nasdaq_sentiment()

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

            return "\n".join(nasdaq_section)

        except Exception as e:
            logger.warning(f"Failed to get NASDAQ sentiment: {e}")
            return "=== NASDAQ Market Sentiment ===\nUnavailable - API error"

    async def _get_market_correlation(self, base_symbol: str) -> str:
        """
        Get correlation analysis between crypto and traditional markets.

        Args:
            base_symbol: Base cryptocurrency symbol

        Returns:
            Formatted correlation section string
        """
        try:
            correlation = await self._omnisearch_client.search_market_correlation(
                base_symbol, "QQQ"
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

            return "\n".join(correlation_section)

        except Exception as e:
            logger.warning(f"Failed to get market correlation for {base_symbol}: {e}")
            return f"=== {base_symbol}-NASDAQ Correlation ===\nUnavailable - API error"

    async def _get_financial_news(self, base_symbol: str) -> str:
        """
        Get recent financial news for the cryptocurrency.

        Args:
            base_symbol: Base cryptocurrency symbol

        Returns:
            Formatted news section string
        """
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

                    news_line = f"{sentiment_emoji} {news.base_result.title[:80]}..."
                    if news.impact_level:
                        news_line += f" [{news.impact_level.upper()} IMPACT]"
                    news_section.append(news_line)

                return "\n".join(news_section)
            return f"=== Recent {base_symbol} News (24h) ===\nNo recent news found"

        except Exception as e:
            logger.warning(f"Failed to get financial news for {base_symbol}: {e}")
            return f"=== Recent {base_symbol} News (24h) ===\nUnavailable - API error"
