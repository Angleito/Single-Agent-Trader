"""Web search results formatter for optimal LLM consumption.

This module provides comprehensive formatting of web search results, sentiment analysis,
and market context data for AI trading bot decision-making. The formatter optimizes
content for LLM processing while maintaining readability and trading relevance.
"""

import asyncio
import hashlib
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Import actual classes from their modules
from bot.analysis.market_context import (
    CorrelationAnalysis,
    MarketRegime,
    MomentumAlignment,
    RiskSentiment,
)
from bot.services.financial_sentiment import SentimentResult

logger = logging.getLogger(__name__)


class ContentPriority(BaseModel):
    """Content priority scoring for web search results."""

    model_config = ConfigDict(frozen=True)

    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Content relevance to trading decisions"
    )
    freshness_score: float = Field(
        ge=0.0, le=1.0, description="Content freshness/recency score"
    )
    authority_score: float = Field(
        ge=0.0, le=1.0, description="Source authority/credibility score"
    )
    trading_impact_score: float = Field(
        ge=0.0, le=1.0, description="Potential trading impact score"
    )
    final_priority: float = Field(
        ge=0.0, le=1.0, description="Final weighted priority score"
    )


class FormattedContent(BaseModel):
    """Formatted content container for LLM consumption."""

    model_config = ConfigDict(frozen=True)

    summary: str = Field(description="Concise content summary")
    key_insights: list[str] = Field(
        default_factory=list, description="Key insights extracted from content"
    )
    trading_signals: list[str] = Field(
        default_factory=list, description="Trading-relevant signals"
    )
    market_sentiment: str = Field(
        default="NEUTRAL", description="Overall market sentiment from content"
    )
    confidence_level: float = Field(
        ge=0.0, le=1.0, description="Confidence in analysis"
    )
    token_count: int = Field(
        ge=0, description="Estimated token count for LLM processing"
    )
    priority: ContentPriority = Field(description="Content priority scoring")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Processing timestamp"
    )


class WebSearchFormatter:
    """
    Advanced web search results formatter for optimal LLM consumption.

    This class provides comprehensive formatting capabilities for web search results,
    including intelligent text summarization, key insight extraction, relevance scoring,
    content deduplication, and token-efficient formatting optimized for trading decisions.
    """

    def __init__(self, max_tokens_per_section: int = 500, max_total_tokens: int = 2000):
        """
        Initialize the web search formatter.

        Args:
            max_tokens_per_section: Maximum tokens per content section
            max_total_tokens: Maximum total tokens for formatted output
        """
        self.max_tokens_per_section = max_tokens_per_section
        self.max_total_tokens = max_total_tokens

        # Initialize trading-relevant keywords
        self._trading_keywords = {
            "price_action": [
                "breakout",
                "support",
                "resistance",
                "trend",
                "momentum",
                "reversal",
                "consolidation",
                "bull",
                "bear",
                "rally",
                "correction",
                "volatility",
            ],
            "technical_analysis": [
                "rsi",
                "macd",
                "ema",
                "sma",
                "bollinger",
                "fibonacci",
                "volume",
                "oversold",
                "overbought",
                "divergence",
                "confluence",
                "pattern",
            ],
            "market_structure": [
                "liquidity",
                "orderbook",
                "whale",
                "institutional",
                "retail",
                "accumulation",
                "distribution",
                "manipulation",
                "squeeze",
            ],
            "sentiment_indicators": [
                "fear",
                "greed",
                "fomo",
                "capitulation",
                "euphoria",
                "panic",
                "sentiment",
                "positioning",
                "funding",
                "derivatives",
            ],
            "macro_factors": [
                "fed",
                "inflation",
                "rates",
                "policy",
                "economic",
                "gdp",
                "employment",
                "regulation",
                "adoption",
                "institutional",
            ],
        }

        # High-authority financial sources
        self._authority_sources = {
            "bloomberg.com": 0.95,
            "reuters.com": 0.95,
            "wsj.com": 0.9,
            "ft.com": 0.9,
            "coindesk.com": 0.85,
            "cointelegraph.com": 0.8,
            "yahoo.com/finance": 0.8,
            "marketwatch.com": 0.75,
            "cnbc.com": 0.75,
            "investing.com": 0.7,
        }

        # Content deduplication tracking
        self._content_hashes: set[str] = set()

        logger.info(
            "WebSearchFormatter initialized with %s tokens per section",
            max_tokens_per_section,
        )

    async def format_news_results(self, news_items: list[dict]) -> str:
        """
        Format news results for optimal LLM consumption.

        Args:
            news_items: List of news item dictionaries

        Returns:
            Formatted news content string optimized for LLM processing
        """
        try:
            if not news_items:
                return "📰 **NEWS ANALYSIS**\n\nNo recent news data available for analysis.\n"

            logger.info("Formatting %s news items", len(news_items))

            # Process and prioritize news items
            processed_items = await self._process_news_items(news_items)

            # Remove duplicates and low-priority content
            filtered_items = self._filter_and_deduplicate_content(processed_items)

            # Sort by priority
            sorted_items = sorted(
                filtered_items, key=lambda x: x.priority.final_priority, reverse=True
            )

            # Format for LLM consumption
            return self._format_news_for_llm(sorted_items[:10])  # Top 10 items

        except Exception as e:
            logger.error(f"Error formatting news results: {e}", exc_info=True)
            return f"📰 **NEWS ANALYSIS**\n\n❌ Error processing news data: {e!s}\n"

    async def format_sentiment_data(self, sentiment: SentimentResult) -> str:
        """
        Format sentiment analysis data for LLM consumption.

        Args:
            sentiment: SentimentResult object containing sentiment analysis

        Returns:
            Formatted sentiment data string
        """
        try:
            sentiment_label = self._get_sentiment_emoji_label(sentiment.sentiment_score)
            confidence_emoji = self._get_confidence_emoji(sentiment.confidence)

            output_lines = [
                "📊 **MARKET SENTIMENT ANALYSIS**",
                "",
                f"🎯 **Overall Sentiment**: {sentiment_label}",
                f"📈 **Sentiment Score**: {sentiment.sentiment_score:.2f} (Range: -1 to +1)",
                f"{confidence_emoji} **Confidence Level**: {sentiment.confidence:.2f}",
                "",
            ]

            # Key themes with emojis
            if sentiment.key_themes:
                output_lines.extend(
                    [
                        "🔍 **Key Market Themes**:",
                        *[f"  • {theme}" for theme in sentiment.key_themes[:5]],
                        "",
                    ]
                )

            # Bullish indicators
            if sentiment.bullish_indicators:
                output_lines.extend(
                    [
                        "🟢 **Bullish Signals**:",
                        *[
                            f"  ↗️ {indicator}"
                            for indicator in sentiment.bullish_indicators[:3]
                        ],
                        "",
                    ]
                )

            # Bearish indicators
            if sentiment.bearish_indicators:
                output_lines.extend(
                    [
                        "🔴 **Bearish Signals**:",
                        *[
                            f"  ↘️ {indicator}"
                            for indicator in sentiment.bearish_indicators[:3]
                        ],
                        "",
                    ]
                )

            # Volatility signals
            if sentiment.volatility_signals:
                output_lines.extend(
                    [
                        "⚡ **Volatility Indicators**:",
                        *[
                            f"  🌊 {signal}"
                            for signal in sentiment.volatility_signals[:2]
                        ],
                        "",
                    ]
                )

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"Error formatting sentiment data: {e}", exc_info=True)
            return f"📊 **MARKET SENTIMENT ANALYSIS**\n\n❌ Error processing sentiment data: {e!s}\n"

    async def format_correlation_analysis(
        self, correlation: CorrelationAnalysis
    ) -> str:
        """
        Format correlation analysis data for LLM consumption.

        Args:
            correlation: CorrelationAnalysis object containing correlation data

        Returns:
            Formatted correlation analysis string
        """
        try:
            correlation_emoji = self._get_correlation_emoji(
                correlation.correlation_coefficient
            )
            strength_emoji = self._get_strength_emoji(
                correlation.correlation_strength.value
            )
            significance_emoji = "✅" if correlation.is_significant else "❌"

            output_lines = [
                "🔗 **CRYPTO-NASDAQ CORRELATION ANALYSIS**",
                "",
                f"{correlation_emoji} **Correlation**: {correlation.correlation_coefficient:.3f} ({correlation.correlation_strength.value})",
                f"📍 **Direction**: {correlation.direction}",
                f"{significance_emoji} **Statistical Significance**: {'Yes' if correlation.is_significant else 'No'} (p={correlation.p_value:.3f})",
                f"📊 **Sample Size**: {correlation.sample_size:,} data points",
                f"{strength_emoji} **Reliability Score**: {correlation.reliability_score:.2f}",
                "",
            ]

            # Rolling correlations if available
            if correlation.rolling_correlation_24h is not None:
                output_lines.extend(
                    [
                        "⏰ **Rolling Correlations**:",
                        f"  📅 24-Hour: {correlation.rolling_correlation_24h:.3f}",
                        f"  📅 7-Day: {correlation.rolling_correlation_7d:.3f}",
                        f"  🎯 Stability: {correlation.correlation_stability:.2f}",
                        "",
                    ]
                )

            # Regime-dependent correlations
            if correlation.regime_dependent_correlation:
                output_lines.extend(
                    [
                        "🌐 **Regime-Dependent Correlations**:",
                        *[
                            f"  📈 {regime}: {corr:.3f}"
                            for regime, corr in correlation.regime_dependent_correlation.items()
                        ],
                        "",
                    ]
                )

            # Trading implications
            implications = self._generate_correlation_implications(correlation)
            if implications:
                output_lines.extend(
                    ["💡 **Trading Implications**:", f"  {implications}", ""]
                )

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"Error formatting correlation analysis: {e}", exc_info=True)
            return f"🔗 **CRYPTO-NASDAQ CORRELATION ANALYSIS**\n\n❌ Error processing correlation data: {e!s}\n"

    async def format_market_context(self, context: dict[str, Any]) -> str:
        """
        Format comprehensive market context analysis for LLM consumption.

        Args:
            context: Dictionary containing all market context data

        Returns:
            Formatted market context string
        """
        try:
            output_sections = []

            # News analysis
            if "news_results" in context:
                news_formatted = await self.format_news_results(context["news_results"])
                output_sections.append(news_formatted)

            # Sentiment analysis
            if "sentiment_result" in context:
                sentiment_formatted = await self.format_sentiment_data(
                    context["sentiment_result"]
                )
                output_sections.append(sentiment_formatted)

            # Correlation analysis
            if "correlation_analysis" in context:
                correlation_formatted = await self.format_correlation_analysis(
                    context["correlation_analysis"]
                )
                output_sections.append(correlation_formatted)

            # Market regime analysis
            if "market_regime" in context:
                regime_formatted = await self._format_market_regime(
                    context["market_regime"]
                )
                output_sections.append(regime_formatted)

            # Risk sentiment
            if "risk_sentiment" in context:
                risk_formatted = await self._format_risk_sentiment(
                    context["risk_sentiment"]
                )
                output_sections.append(risk_formatted)

            # Momentum alignment
            if "momentum_alignment" in context:
                momentum_formatted = await self._format_momentum_alignment(
                    context["momentum_alignment"]
                )
                output_sections.append(momentum_formatted)

            # Combine all sections
            full_content = "\n".join(output_sections)

            # Apply token limits and optimization
            optimized_content = self._optimize_content_for_tokens(full_content)

            # Add summary header
            summary_header = self._generate_context_summary_header(context)

            return f"{summary_header}\n\n{optimized_content}"

        except Exception as e:
            logger.error(f"Error formatting market context: {e}", exc_info=True)
            return f"🌐 **MARKET CONTEXT ANALYSIS**\n\n❌ Error processing market context: {e!s}\n"

    def truncate_content(self, text: str, max_length: int) -> str:
        """
        Smart content truncation that preserves meaning and structure.

        Args:
            text: Text to truncate
            max_length: Maximum character length

        Returns:
            Truncated text with preserved structure
        """
        try:
            if len(text) <= max_length:
                return text

            # Try to find a good breaking point
            break_points = [". ", "! ", "? ", "\n\n", "\n", "; ", ", "]

            for break_point in break_points:
                # Find the last occurrence of break_point before max_length
                last_break = text.rfind(
                    break_point, 0, max_length - 10
                )  # Leave room for ellipsis
                if last_break > max_length * 0.7:  # Don't truncate too aggressively
                    return text[: last_break + len(break_point)].rstrip() + "..."

            # If no good break point found, truncate at word boundary
            words = text[: max_length - 3].split()
            if words:
                return " ".join(words[:-1]) + "..."

            return text[: max_length - 3] + "..."

        except Exception as e:
            logger.error(f"Error truncating content: {e}", exc_info=True)
            return text[: max_length - 3] + "..." if len(text) > max_length else text

    async def extract_key_insights(self, search_results: dict) -> list[str]:
        """
        Extract key insights from search results for trading decisions.

        Args:
            search_results: Dictionary containing search results

        Returns:
            List of key insights relevant to trading
        """
        try:
            insights = []

            # Process different types of content
            if "news_items" in search_results:
                news_insights = await self._extract_news_insights(
                    search_results["news_items"]
                )
                insights.extend(news_insights)

            if "sentiment_data" in search_results:
                sentiment_insights = self._extract_sentiment_insights(
                    search_results["sentiment_data"]
                )
                insights.extend(sentiment_insights)

            if "price_data" in search_results:
                price_insights = self._extract_price_insights(
                    search_results["price_data"]
                )
                insights.extend(price_insights)

            if "technical_analysis" in search_results:
                technical_insights = self._extract_technical_insights(
                    search_results["technical_analysis"]
                )
                insights.extend(technical_insights)

            # Deduplicate and prioritize insights
            unique_insights = self._deduplicate_insights(insights)
            prioritized_insights = self._prioritize_insights(unique_insights)

            return prioritized_insights[:10]  # Return top 10 insights

        except Exception as e:
            logger.error(f"Error extracting key insights: {e}", exc_info=True)
            return [f"Error extracting insights: {e!s}"]

    # Private helper methods

    async def _process_news_items(
        self, news_items: list[dict]
    ) -> list[FormattedContent]:
        """Process and analyze news items for formatting."""
        processed_items = []

        tasks = [self._process_single_news_item(item) for item in news_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, FormattedContent):
                processed_items.append(result)
            elif isinstance(result, Exception):
                logger.warning("Error processing news item: %s", result)

        return processed_items

    async def _process_single_news_item(self, item: dict) -> FormattedContent:
        """Process a single news item."""
        try:
            title = item.get("title", "")
            content = item.get("content", item.get("description", ""))
            url = item.get("url", "")
            published_time = item.get("published_time", datetime.utcnow())

            # Calculate priority scores
            relevance_score = self._calculate_relevance_score(title, content)
            freshness_score = self._calculate_freshness_score(published_time)
            authority_score = self._calculate_authority_score(url)
            trading_impact_score = self._calculate_trading_impact_score(title, content)

            priority = ContentPriority(
                relevance_score=relevance_score,
                freshness_score=freshness_score,
                authority_score=authority_score,
                trading_impact_score=trading_impact_score,
                final_priority=(
                    relevance_score * 0.3
                    + freshness_score * 0.2
                    + authority_score * 0.25
                    + trading_impact_score * 0.25
                ),
            )

            # Extract key insights and trading signals
            key_insights = self._extract_item_insights(title, content)
            trading_signals = self._extract_trading_signals(title, content)

            # Generate summary
            summary = self._generate_content_summary(title, content)

            # Determine market sentiment
            market_sentiment = self._determine_content_sentiment(title, content)

            # Calculate confidence
            confidence_level = min(priority.final_priority + 0.2, 1.0)

            # Estimate token count
            token_count = self._estimate_token_count(summary + " ".join(key_insights))

            return FormattedContent(
                summary=summary,
                key_insights=key_insights,
                trading_signals=trading_signals,
                market_sentiment=market_sentiment,
                confidence_level=confidence_level,
                token_count=token_count,
                priority=priority,
            )

        except Exception as e:
            logger.error(f"Error processing single news item: {e}", exc_info=True)
            # Return minimal formatted content for failed items
            return FormattedContent(
                summary=f"Error processing item: {e!s}",
                key_insights=[],
                trading_signals=[],
                market_sentiment="NEUTRAL",
                confidence_level=0.0,
                token_count=10,
                priority=ContentPriority(
                    relevance_score=0.0,
                    freshness_score=0.0,
                    authority_score=0.0,
                    trading_impact_score=0.0,
                    final_priority=0.0,
                ),
            )

    def _filter_and_deduplicate_content(
        self, items: list[FormattedContent]
    ) -> list[FormattedContent]:
        """Filter and deduplicate content based on similarity."""
        filtered_items = []
        content_hashes = set()

        for item in items:
            # Create content hash for deduplication
            content_hash = hashlib.sha256(
                (item.summary + " ".join(item.key_insights)).encode(),
                usedforsecurity=False,
            ).hexdigest()

            if (
                content_hash not in content_hashes
                and item.priority.final_priority > 0.1
            ):
                filtered_items.append(item)
                content_hashes.add(content_hash)

        return filtered_items

    def _format_news_for_llm(self, items: list[FormattedContent]) -> str:
        """Format processed news items for LLM consumption."""
        if not items:
            return "📰 **NEWS ANALYSIS**\n\nNo significant news items found.\n"

        output_lines = ["📰 **NEWS ANALYSIS**", ""]

        # Overall sentiment summary
        sentiment_counts = Counter(item.market_sentiment for item in items)
        dominant_sentiment = (
            sentiment_counts.most_common(1)[0][0] if sentiment_counts else "NEUTRAL"
        )
        sentiment_emoji = self._get_sentiment_emoji_label(
            self._sentiment_to_score(dominant_sentiment)
        )

        output_lines.extend(
            [
                f"🎯 **Overall News Sentiment**: {sentiment_emoji}",
                f"📊 **Articles Analyzed**: {len(items)}",
                "",
            ]
        )

        # Top insights across all articles
        all_insights = []
        for item in items:
            all_insights.extend(item.key_insights)

        if all_insights:
            top_insights = Counter(all_insights).most_common(5)
            output_lines.extend(
                [
                    "🔍 **Key Market Insights**:",
                    *[f"  • {insight}" for insight, _ in top_insights],
                    "",
                ]
            )

        # Top trading signals
        all_signals = []
        for item in items:
            all_signals.extend(item.trading_signals)

        if all_signals:
            top_signals = Counter(all_signals).most_common(3)
            output_lines.extend(
                [
                    "📈 **Trading Signals**:",
                    *[f"  🎯 {signal}" for signal, _ in top_signals],
                    "",
                ]
            )

        # Recent high-priority articles
        output_lines.extend(["📑 **Recent High-Priority Articles**:", ""])

        for i, item in enumerate(items[:5], 1):
            priority_stars = "⭐" * min(int(item.priority.final_priority * 5), 5)
            confidence_emoji = self._get_confidence_emoji(item.confidence_level)

            output_lines.extend(
                [
                    f"**{i}. {priority_stars}** {confidence_emoji}",
                    f"   📄 **Summary**: {self.truncate_content(item.summary, 150)}",
                ]
            )

            if item.key_insights:
                output_lines.append(f"   💡 **Key Insight**: {item.key_insights[0]}")

            output_lines.append("")

        return "\n".join(output_lines)

    def _calculate_relevance_score(self, title: str, content: str) -> float:
        """Calculate content relevance score for trading decisions."""
        try:
            text = (title + " " + content).lower()

            total_score = 0.0
            max_category_score = 0.0

            for category, keywords in self._trading_keywords.items():
                category_score = sum(1 for keyword in keywords if keyword in text)
                category_weight = {
                    "price_action": 0.3,
                    "technical_analysis": 0.25,
                    "market_structure": 0.2,
                    "sentiment_indicators": 0.15,
                    "macro_factors": 0.1,
                }.get(category, 0.1)

                weighted_score = min(category_score * category_weight, category_weight)
                total_score += weighted_score
                max_category_score = max(max_category_score, weighted_score)

            # Bonus for crypto-specific terms
            crypto_terms = [
                "bitcoin",
                "btc",
                "ethereum",
                "eth",
                "crypto",
                "cryptocurrency",
            ]
            crypto_bonus = 0.1 if any(term in text for term in crypto_terms) else 0.0

            return min(total_score + crypto_bonus + max_category_score * 0.5, 1.0)

        except Exception:
            return 0.1

    def _calculate_freshness_score(self, published_time: datetime) -> float:
        """Calculate content freshness score."""
        try:
            if isinstance(published_time, str):
                # Try to parse string datetime
                from dateutil import parser

                published_time = parser.parse(published_time)

            time_diff = datetime.utcnow() - published_time
            hours_old = time_diff.total_seconds() / 3600

            if hours_old < 1:
                return 1.0
            if hours_old < 6:
                return 0.9
            if hours_old < 24:
                return 0.8
            if hours_old < 72:
                return 0.6
            if hours_old < 168:  # 1 week
                return 0.4
            return 0.2

        except Exception:
            return 0.5  # Default for unparseable dates

    def _calculate_authority_score(self, url: str) -> float:
        """Calculate source authority score."""
        try:
            for domain, score in self._authority_sources.items():
                if domain in url.lower():
                    return score
            return 0.3  # Default for unknown sources
        except Exception:
            return 0.3

    def _calculate_trading_impact_score(self, title: str, content: str) -> float:
        """Calculate potential trading impact score."""
        try:
            text = (title + " " + content).lower()

            # High impact indicators
            high_impact_terms = [
                "breaking",
                "urgent",
                "alert",
                "major",
                "significant",
                "announcement",
                "decision",
                "ruling",
                "approval",
                "rejection",
                "hack",
                "exploit",
                "crash",
                "surge",
                "rally",
                "dump",
            ]

            # Price movement indicators
            price_terms = [
                "price",
                "target",
                "forecast",
                "prediction",
                "analysis",
                "technical",
                "resistance",
                "support",
                "breakout",
            ]

            # Market moving events
            market_events = [
                "fed",
                "federal reserve",
                "interest rate",
                "inflation",
                "regulation",
                "sec",
                "etf",
                "institutional",
                "whale",
            ]

            high_impact_score = sum(0.2 for term in high_impact_terms if term in text)
            price_score = sum(0.15 for term in price_terms if term in text)
            event_score = sum(0.25 for term in market_events if term in text)

            # Title impact bonus
            title_bonus = (
                0.1 if any(term in title.lower() for term in high_impact_terms) else 0.0
            )

            return min(high_impact_score + price_score + event_score + title_bonus, 1.0)

        except Exception:
            return 0.2

    def _extract_item_insights(self, title: str, content: str) -> list[str]:
        """Extract key insights from individual content item."""
        insights = []
        text = (title + " " + content).lower()

        # Price movement insights
        if any(
            term in text
            for term in ["breakout", "break out", "breaks above", "breaks below"]
        ):
            insights.append("Technical breakout pattern identified")

        if any(
            term in text for term in ["support level", "resistance level", "key level"]
        ):
            insights.append("Critical price levels mentioned")

        # Institutional activity
        if any(term in text for term in ["institutional", "whale", "large holder"]):
            insights.append("Institutional/whale activity detected")

        # Regulatory developments
        if any(term in text for term in ["regulation", "regulatory", "sec", "cftc"]):
            insights.append("Regulatory developments affecting crypto")

        # Adoption news
        if any(
            term in text
            for term in ["adoption", "accepts", "integration", "partnership"]
        ):
            insights.append("Crypto adoption/integration progress")

        # Market sentiment shifts
        if any(
            term in text for term in ["sentiment", "fear", "greed", "panic", "euphoria"]
        ):
            insights.append("Market sentiment indicators present")

        return insights[:3]  # Limit to top 3 insights per item

    def _extract_trading_signals(self, title: str, content: str) -> list[str]:
        """Extract trading signals from content."""
        signals = []
        text = (title + " " + content).lower()

        # Bullish signals
        if any(term in text for term in ["bullish", "buy signal", "long", "uptrend"]):
            signals.append("Bullish trading signal identified")

        # Bearish signals
        if any(
            term in text for term in ["bearish", "sell signal", "short", "downtrend"]
        ):
            signals.append("Bearish trading signal identified")

        # Volatility signals
        if any(
            term in text for term in ["volatile", "volatility spike", "high volatility"]
        ):
            signals.append("High volatility expected")

        # Volume signals
        if any(
            term in text for term in ["volume spike", "high volume", "unusual volume"]
        ):
            signals.append("Unusual volume activity detected")

        return signals[:2]  # Limit to top 2 signals per item

    def _generate_content_summary(self, title: str, content: str) -> str:
        """Generate concise content summary."""
        try:
            # Start with title as base
            if not content or len(content) < 50:
                return self.truncate_content(title, 200)

            # Extract first meaningful sentence from content
            sentences = re.split(r"[.!?]+", content)
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            if meaningful_sentences:
                summary = f"{title}. {meaningful_sentences[0]}"
            else:
                summary = title

            return self.truncate_content(summary, 250)

        except Exception:
            return self.truncate_content(title, 200)

    def _determine_content_sentiment(self, title: str, content: str) -> str:
        """Determine sentiment from content."""
        text = (title + " " + content).lower()

        bullish_words = [
            "bullish",
            "positive",
            "surge",
            "rally",
            "moon",
            "pump",
            "breakout",
        ]
        bearish_words = [
            "bearish",
            "negative",
            "crash",
            "dump",
            "correction",
            "sell-off",
        ]

        bullish_count = sum(1 for word in bullish_words if word in text)
        bearish_count = sum(1 for word in bearish_words if word in text)

        if bullish_count > bearish_count:
            return "BULLISH"
        if bearish_count > bullish_count:
            return "BEARISH"
        return "NEUTRAL"

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def _optimize_content_for_tokens(self, content: str) -> str:
        """Optimize content to fit within token limits."""
        estimated_tokens = self._estimate_token_count(content)

        if estimated_tokens <= self.max_total_tokens:
            return content

        # Calculate compression ratio needed
        compression_ratio = self.max_total_tokens / estimated_tokens
        target_length = int(len(content) * compression_ratio * 0.9)  # 90% to be safe

        return self.truncate_content(content, target_length)

    async def _extract_news_insights(self, news_items: list[dict]) -> list[str]:
        """Extract insights from news items."""
        insights = []

        for item in news_items:
            title = item.get("title", "")
            content = item.get("content", item.get("description", ""))
            item_insights = self._extract_item_insights(title, content)
            insights.extend(item_insights)

        return insights

    def _extract_sentiment_insights(self, sentiment_data: dict) -> list[str]:
        """Extract insights from sentiment data."""
        insights = []

        if isinstance(sentiment_data, dict):
            sentiment_score = sentiment_data.get("sentiment_score", 0)
            confidence = sentiment_data.get("confidence", 0)

            if abs(sentiment_score) > 0.5 and confidence > 0.7:
                direction = "bullish" if sentiment_score > 0 else "bearish"
                insights.append(f"Strong {direction} sentiment with high confidence")

            if sentiment_data.get("volatility_signals"):
                insights.append("Elevated volatility expected based on sentiment")

        return insights

    def _extract_price_insights(self, price_data: dict) -> list[str]:
        """Extract insights from price data."""
        insights = []

        if isinstance(price_data, dict):
            price_change = price_data.get("price_change_24h", 0)
            volume_change = price_data.get("volume_change_24h", 0)

            if abs(price_change) > 0.05:  # 5% change
                direction = "upward" if price_change > 0 else "downward"
                insights.append(
                    f"Significant {direction} price movement ({price_change:.1%})"
                )

            if abs(volume_change) > 0.2:  # 20% volume change
                insights.append(f"Unusual volume activity ({volume_change:+.1%})")

        return insights

    def _extract_technical_insights(self, technical_data: dict) -> list[str]:
        """Extract insights from technical analysis data."""
        insights = []

        if isinstance(technical_data, dict):
            rsi = technical_data.get("rsi")
            if rsi:
                if rsi > 70:
                    insights.append("RSI indicates overbought conditions")
                elif rsi < 30:
                    insights.append("RSI indicates oversold conditions")

            if technical_data.get("trend_direction") == "BULLISH":
                insights.append("Technical indicators show bullish trend")
            elif technical_data.get("trend_direction") == "BEARISH":
                insights.append("Technical indicators show bearish trend")

        return insights

    def _deduplicate_insights(self, insights: list[str]) -> list[str]:
        """Remove duplicate insights while preserving order."""
        seen = set()
        unique_insights = []

        for insight in insights:
            normalized = insight.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_insights.append(insight)

        return unique_insights

    def _prioritize_insights(self, insights: list[str]) -> list[str]:
        """Prioritize insights based on trading relevance."""
        priority_keywords = {
            "breakout": 10,
            "institutional": 9,
            "regulatory": 8,
            "sentiment": 7,
            "volume": 6,
            "technical": 5,
            "adoption": 4,
        }

        def get_priority(insight: str) -> int:
            insight_lower = insight.lower()
            for keyword, priority in priority_keywords.items():
                if keyword in insight_lower:
                    return priority
            return 1

        return sorted(insights, key=get_priority, reverse=True)

    async def _format_market_regime(self, regime: MarketRegime) -> str:
        """Format market regime analysis."""
        regime_emoji = {
            "RISK_ON": "🟢",
            "RISK_OFF": "🔴",
            "TRANSITION": "🟡",
            "UNKNOWN": "⚪",
        }.get(regime.regime_type.value, "⚪")

        confidence_emoji = self._get_confidence_emoji(regime.confidence)

        output_lines = [
            "🌐 **MARKET REGIME ANALYSIS**",
            "",
            f"{regime_emoji} **Current Regime**: {regime.regime_type.value}",
            f"{confidence_emoji} **Confidence**: {regime.confidence:.2f}",
            "",
        ]

        if regime.key_drivers:
            output_lines.extend(
                [
                    "🔑 **Key Drivers**:",
                    *[f"  • {driver}" for driver in regime.key_drivers[:3]],
                    "",
                ]
            )

        # Key regime characteristics
        output_lines.extend(
            [
                "📊 **Regime Characteristics**:",
                f"  🏛️ Fed Policy: {regime.fed_policy_stance}",
                f"  💰 Inflation: {regime.inflation_environment}",
                f"  📈 Rate Trend: {regime.interest_rate_trend}",
                f"  🌍 Geopolitical Risk: {regime.geopolitical_risk_level}",
                f"  ⚡ Volatility: {regime.market_volatility_regime}",
                "",
            ]
        )

        return "\n".join(output_lines)

    async def _format_risk_sentiment(self, sentiment: RiskSentiment) -> str:
        """Format risk sentiment analysis."""
        sentiment_emoji = {
            "EXTREME_FEAR": "😰",
            "FEAR": "😨",
            "NEUTRAL": "😐",
            "GREED": "😀",
            "EXTREME_GREED": "🤑",
        }.get(sentiment.sentiment_level.value, "😐")

        output_lines = [
            "😰 **RISK SENTIMENT ANALYSIS**",
            "",
            f"{sentiment_emoji} **Sentiment Level**: {sentiment.sentiment_level.value}",
            f"📊 **Fear & Greed Index**: {sentiment.fear_greed_index:.0f}/100",
            f"⚡ **Expected Volatility**: {sentiment.volatility_expectation:.1f}%",
            f"🌡️ **Market Stress**: {sentiment.market_stress_indicator:.2f}",
            "",
        ]

        # Additional sentiment indicators if available
        if sentiment.vix_level:
            output_lines.append(f"📈 **VIX Level**: {sentiment.vix_level:.1f}")

        if sentiment.crypto_fear_greed:
            output_lines.append(
                f"₿ **Crypto Fear & Greed**: {sentiment.crypto_fear_greed:.0f}/100"
            )

        if sentiment.news_sentiment_score:
            news_sentiment_label = self._get_sentiment_emoji_label(
                sentiment.news_sentiment_score
            )
            output_lines.append(f"📰 **News Sentiment**: {news_sentiment_label}")

        output_lines.append("")
        return "\n".join(output_lines)

    async def _format_momentum_alignment(self, momentum: MomentumAlignment) -> str:
        """Format momentum alignment analysis."""
        alignment_emoji = (
            "🔄"
            if abs(momentum.directional_alignment) < 0.3
            else ("📈" if momentum.directional_alignment > 0 else "📉")
        )

        output_lines = [
            "🔄 **MOMENTUM ALIGNMENT ANALYSIS**",
            "",
            f"{alignment_emoji} **Directional Alignment**: {momentum.directional_alignment:+.2f}",
            f"💪 **Strength Alignment**: {momentum.strength_alignment:.2f}",
            f"₿ **Crypto Momentum**: {momentum.crypto_momentum_score:+.2f}",
            f"📊 **NASDAQ Momentum**: {momentum.nasdaq_momentum_score:+.2f}",
            "",
        ]

        if momentum.momentum_divergences:
            output_lines.extend(
                [
                    "⚠️ **Momentum Divergences**:",
                    *[f"  • {div}" for div in momentum.momentum_divergences[:2]],
                    "",
                ]
            )

        output_lines.extend(
            [
                f"🎯 **Momentum Regime**: {momentum.momentum_regime}",
                f"🌊 **Cross-Asset Flow**: {momentum.cross_asset_momentum_flow}",
                "",
            ]
        )

        return "\n".join(output_lines)

    def _generate_context_summary_header(self, context: dict[str, Any]) -> str:
        """Generate summary header for market context."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Count available data sources
        data_sources = []
        if "news_results" in context:
            data_sources.append("📰 News")
        if "sentiment_result" in context:
            data_sources.append("📊 Sentiment")
        if "correlation_analysis" in context:
            data_sources.append("🔗 Correlation")
        if "market_regime" in context:
            data_sources.append("🌐 Regime")

        return f"""🌐 **COMPREHENSIVE MARKET CONTEXT ANALYSIS**

⏰ **Analysis Time**: {timestamp}
📊 **Data Sources**: {" | ".join(data_sources)}
🎯 **Optimized for**: AI Trading Decision Making"""

    def _get_sentiment_emoji_label(self, score: float) -> str:
        """Get emoji label for sentiment score."""
        if score > 0.5:
            return "🚀 STRONGLY BULLISH"
        if score > 0.2:
            return "📈 BULLISH"
        if score > -0.2:
            return "😐 NEUTRAL"
        if score > -0.5:
            return "📉 BEARISH"
        return "💥 STRONGLY BEARISH"

    def _get_confidence_emoji(self, confidence: float) -> str:
        """Get emoji for confidence level."""
        if confidence > 0.8:
            return "🎯"
        if confidence > 0.6:
            return "✅"
        if confidence > 0.4:
            return "⚠️"
        return "❓"

    def _get_correlation_emoji(self, correlation: float) -> str:
        """Get emoji for correlation coefficient."""
        if correlation > 0.5:
            return "📈"
        if correlation > 0.2:
            return "↗️"
        if correlation > -0.2:
            return "↔️"
        if correlation > -0.5:
            return "↘️"
        return "📉"

    def _get_strength_emoji(self, strength: str) -> str:
        """Get emoji for correlation strength."""
        strength_emojis = {
            "VERY_STRONG": "💪",
            "STRONG": "💪",
            "MODERATE": "👍",
            "WEAK": "👌",
            "VERY_WEAK": "🤏",
        }
        return strength_emojis.get(strength, "❓")

    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment string to numeric score."""
        sentiment_scores = {
            "STRONGLY_BULLISH": 0.8,
            "BULLISH": 0.4,
            "NEUTRAL": 0.0,
            "BEARISH": -0.4,
            "STRONGLY_BEARISH": -0.8,
        }
        return sentiment_scores.get(sentiment, 0.0)

    def _generate_correlation_implications(
        self, correlation: CorrelationAnalysis
    ) -> str:
        """Generate trading implications from correlation analysis."""
        implications = []

        if correlation.correlation_strength.value in ["STRONG", "VERY_STRONG"]:
            if correlation.direction == "POSITIVE":
                implications.append("High systematic risk - diversification limited")
            else:
                implications.append("Potential hedging opportunities available")

        if correlation.is_significant:
            implications.append("Statistically reliable correlation")
        else:
            implications.append("Correlation not statistically significant")

        if correlation.correlation_stability < 0.5:
            implications.append("Unstable correlation - use with caution")

        return " | ".join(implications[:2])  # Limit to 2 key implications
