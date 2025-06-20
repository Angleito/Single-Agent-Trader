"""Financial sentiment analysis service for processing market news and data."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SentimentResult(BaseModel):
    """Sentiment analysis result for financial news and data."""

    model_config = ConfigDict(frozen=True)

    sentiment_score: float = Field(
        ge=-1.0, le=1.0, description="Sentiment score from -1 (bearish) to 1 (bullish)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence level of the sentiment analysis"
    )
    key_themes: list[str] = Field(
        default_factory=list, description="Key themes identified in the content"
    )
    bullish_indicators: list[str] = Field(
        default_factory=list, description="Bullish market indicators found"
    )
    bearish_indicators: list[str] = Field(
        default_factory=list, description="Bearish market indicators found"
    )
    volatility_signals: list[str] = Field(
        default_factory=list, description="Volatility-related signals"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Analysis timestamp"
    )


class CryptoIndicators(BaseModel):
    """Crypto-specific market indicators extracted from text."""

    model_config = ConfigDict(frozen=True)

    # Price and trend indicators
    price_mentions: list[dict[str, Any]] = Field(
        default_factory=list, description="Price mentions and targets"
    )
    trend_direction: str | None = Field(
        default=None, description="Overall trend direction (BULLISH/BEARISH/NEUTRAL)"
    )
    support_levels: list[float] = Field(
        default_factory=list, description="Support levels mentioned"
    )
    resistance_levels: list[float] = Field(
        default_factory=list, description="Resistance levels mentioned"
    )

    # Volume and momentum
    volume_indicators: list[str] = Field(
        default_factory=list, description="Volume-related indicators"
    )
    momentum_signals: list[str] = Field(
        default_factory=list, description="Momentum signals identified"
    )

    # Market structure
    adoption_signals: list[str] = Field(
        default_factory=list, description="Adoption and institutional signals"
    )
    regulatory_mentions: list[str] = Field(
        default_factory=list, description="Regulatory developments"
    )
    whale_activity: list[str] = Field(
        default_factory=list, description="Large holder activity mentions"
    )

    # Technical analysis
    technical_patterns: list[str] = Field(
        default_factory=list, description="Technical chart patterns mentioned"
    )
    indicator_signals: dict[str, str] = Field(
        default_factory=dict,
        description="Technical indicator signals (RSI, MACD, etc.)",
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Analysis timestamp"
    )


class NasdaqIndicators(BaseModel):
    """NASDAQ and traditional market indicators extracted from text."""

    model_config = ConfigDict(frozen=True)

    # Market indices
    nasdaq_trend: str | None = Field(default=None, description="NASDAQ trend direction")
    sp500_mentions: list[str] = Field(
        default_factory=list, description="S&P 500 related mentions"
    )
    dow_mentions: list[str] = Field(
        default_factory=list, description="Dow Jones related mentions"
    )

    # Economic indicators
    fed_policy_signals: list[str] = Field(
        default_factory=list, description="Federal Reserve policy signals"
    )
    interest_rate_mentions: list[str] = Field(
        default_factory=list, description="Interest rate related content"
    )
    inflation_indicators: list[str] = Field(
        default_factory=list, description="Inflation-related indicators"
    )

    # Market sentiment
    risk_on_signals: list[str] = Field(
        default_factory=list, description="Risk-on sentiment indicators"
    )
    risk_off_signals: list[str] = Field(
        default_factory=list, description="Risk-off sentiment indicators"
    )
    vix_mentions: list[str] = Field(
        default_factory=list, description="VIX volatility index mentions"
    )

    # Sector analysis
    tech_sector_signals: list[str] = Field(
        default_factory=list, description="Technology sector indicators"
    )
    financial_sector_signals: list[str] = Field(
        default_factory=list, description="Financial sector indicators"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Analysis timestamp"
    )


class FinancialSentimentService:
    """
    Service for analyzing financial sentiment from news and market data.

    Processes financial news, extracts market indicators, and provides
    sentiment analysis specifically tailored for crypto trading decisions.
    """

    def __init__(self):
        """Initialize the financial sentiment service."""
        self._bullish_keywords = {
            "strong",
            "bullish",
            "rally",
            "surge",
            "moon",
            "pump",
            "breakout",
            "bounce",
            "recovery",
            "support",
            "institutional",
            "adoption",
            "upgrade",
            "positive",
            "optimistic",
            "breakthrough",
            "milestone",
            "growth",
            "expansion",
            "accumulation",
            "buying",
            "long",
            "hodl",
        }

        self._bearish_keywords = {
            "weak",
            "bearish",
            "crash",
            "dump",
            "correction",
            "breakdown",
            "rejection",
            "resistance",
            "sell-off",
            "liquidation",
            "fear",
            "panic",
            "regulatory",
            "ban",
            "restriction",
            "concern",
            "risk",
            "decline",
            "drop",
            "fall",
            "short",
            "capitulation",
        }

        self._volatility_keywords = {
            "volatile",
            "volatility",
            "swing",
            "whipsaw",
            "choppy",
            "range",
            "sideways",
            "consolidation",
            "uncertain",
            "mixed",
            "divergence",
        }

        self._crypto_price_pattern = re.compile(
            r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|USDT|BTC|ETH)?",
            re.IGNORECASE,
        )

        self._technical_indicators = {
            "rsi",
            "macd",
            "sma",
            "ema",
            "bollinger",
            "stochastic",
            "fibonacci",
            "golden cross",
            "death cross",
            "moving average",
        }

        logger.info("Financial sentiment service initialized")

    async def analyze_news_sentiment(self, news_items: list[dict]) -> SentimentResult:
        """
        Analyze sentiment from a list of news items.

        Args:
            news_items: List of news dictionaries with 'title', 'content', etc.

        Returns:
            SentimentResult with aggregated sentiment analysis
        """
        if not news_items:
            logger.warning("No news items provided for sentiment analysis")
            return SentimentResult(
                sentiment_score=0.0,
                confidence=0.0,
                key_themes=["No news data available"],
            )

        try:
            # Process all news items concurrently
            sentiment_tasks = [self._analyze_single_item(item) for item in news_items]
            individual_results = await asyncio.gather(*sentiment_tasks)

            # Aggregate results
            return self._aggregate_sentiment_results(individual_results, news_items)

        except Exception:
            logger.exception("Error analyzing news sentiment")
            return SentimentResult(
                sentiment_score=0.0,
                confidence=0.0,
                key_themes=["Analysis error occurred"],
            )

    async def _analyze_single_item(self, item: dict) -> dict[str, Any]:
        """Analyze sentiment for a single news item."""
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()

        # Count keyword occurrences
        bullish_count = sum(1 for word in self._bullish_keywords if word in text)
        bearish_count = sum(1 for word in self._bearish_keywords if word in text)
        volatility_count = sum(1 for word in self._volatility_keywords if word in text)

        # Calculate sentiment score
        total_sentiment_words = bullish_count + bearish_count
        if total_sentiment_words > 0:
            sentiment_score = (bullish_count - bearish_count) / total_sentiment_words
        else:
            sentiment_score = 0.0

        # Calculate confidence based on word count and volatility
        word_count = len(text.split())
        confidence = min(total_sentiment_words / max(word_count * 0.05, 1), 1.0)

        # Adjust confidence based on volatility indicators
        if volatility_count > 0:
            confidence *= 0.8  # Reduce confidence when volatility is high

        return {
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "volatility_count": volatility_count,
            "text": text,
        }

    def _aggregate_sentiment_results(
        self, individual_results: list[dict], news_items: list[dict]
    ) -> SentimentResult:
        """Aggregate individual sentiment results into final result."""
        if not individual_results:
            return SentimentResult(sentiment_score=0.0, confidence=0.0)

        # Weight by confidence
        weighted_sentiment = 0.0
        total_weight = 0.0

        bullish_indicators = []
        bearish_indicators = []
        volatility_signals = []
        key_themes = set()

        for i, result in enumerate(individual_results):
            weight = result["confidence"]
            weighted_sentiment += result["sentiment_score"] * weight
            total_weight += weight

            # Extract indicators
            if result["bullish_count"] > 0:
                title = news_items[i].get("title", "News item")
                bullish_indicators.append(f"Bullish signals in: {title[:50]}...")

            if result["bearish_count"] > 0:
                title = news_items[i].get("title", "News item")
                bearish_indicators.append(f"Bearish signals in: {title[:50]}...")

            if result["volatility_count"] > 0:
                volatility_signals.append("Volatility indicators present")

            # Extract key themes from titles
            title = news_items[i].get("title", "")
            themes = self._extract_themes(title)
            key_themes.update(themes)

        # Calculate final sentiment
        final_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        final_confidence = (
            total_weight / len(individual_results) if individual_results else 0.0
        )

        return SentimentResult(
            sentiment_score=final_sentiment,
            confidence=min(final_confidence, 1.0),
            key_themes=list(key_themes)[:10],  # Limit to top 10 themes
            bullish_indicators=bullish_indicators[:5],
            bearish_indicators=bearish_indicators[:5],
            volatility_signals=volatility_signals[:3],
        )

    def extract_crypto_indicators(self, text: str) -> CryptoIndicators:
        """
        Extract crypto-specific indicators from text.

        Args:
            text: Text content to analyze

        Returns:
            CryptoIndicators with extracted crypto market data
        """
        try:
            text_lower = text.lower()

            # Extract price mentions
            price_mentions = []
            for match in self._crypto_price_pattern.finditer(text):
                price_mentions.append(
                    {
                        "price": match.group(1),
                        "context": text[max(0, match.start() - 20) : match.end() + 20],
                    }
                )

            # Determine trend direction
            trend_direction = self._determine_trend_direction(text_lower)

            # Extract support/resistance levels
            support_levels = self._extract_price_levels(
                text, ["support", "floor", "bottom"]
            )
            resistance_levels = self._extract_price_levels(
                text, ["resistance", "ceiling", "top"]
            )

            # Extract various indicators
            volume_indicators = self._extract_volume_indicators(text_lower)
            momentum_signals = self._extract_momentum_signals(text_lower)
            adoption_signals = self._extract_adoption_signals(text_lower)
            regulatory_mentions = self._extract_regulatory_mentions(text_lower)
            whale_activity = self._extract_whale_activity(text_lower)
            technical_patterns = self._extract_technical_patterns(text_lower)
            indicator_signals = self._extract_indicator_signals(text_lower)

            return CryptoIndicators(
                price_mentions=price_mentions,
                trend_direction=trend_direction,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                volume_indicators=volume_indicators,
                momentum_signals=momentum_signals,
                adoption_signals=adoption_signals,
                regulatory_mentions=regulatory_mentions,
                whale_activity=whale_activity,
                technical_patterns=technical_patterns,
                indicator_signals=indicator_signals,
            )

        except Exception:
            logger.exception("Error extracting crypto indicators")
            return CryptoIndicators()

    def extract_nasdaq_indicators(self, text: str) -> NasdaqIndicators:
        """
        Extract NASDAQ and traditional market indicators from text.

        Args:
            text: Text content to analyze

        Returns:
            NasdaqIndicators with extracted traditional market data
        """
        try:
            text_lower = text.lower()

            # Extract market index trends
            nasdaq_trend = self._extract_nasdaq_trend(text_lower)
            sp500_mentions = self._extract_index_mentions(
                text_lower, ["s&p 500", "sp500", "spx"]
            )
            dow_mentions = self._extract_index_mentions(
                text_lower, ["dow jones", "djia", "dow"]
            )

            # Extract economic indicators
            fed_policy_signals = self._extract_fed_signals(text_lower)
            interest_rate_mentions = self._extract_interest_rate_mentions(text_lower)
            inflation_indicators = self._extract_inflation_indicators(text_lower)

            # Extract sentiment indicators
            risk_on_signals = self._extract_risk_sentiment(text_lower, "on")
            risk_off_signals = self._extract_risk_sentiment(text_lower, "off")
            vix_mentions = self._extract_vix_mentions(text_lower)

            # Extract sector analysis
            tech_sector_signals = self._extract_sector_signals(text_lower, "tech")
            financial_sector_signals = self._extract_sector_signals(
                text_lower, "financial"
            )

            return NasdaqIndicators(
                nasdaq_trend=nasdaq_trend,
                sp500_mentions=sp500_mentions,
                dow_mentions=dow_mentions,
                fed_policy_signals=fed_policy_signals,
                interest_rate_mentions=interest_rate_mentions,
                inflation_indicators=inflation_indicators,
                risk_on_signals=risk_on_signals,
                risk_off_signals=risk_off_signals,
                vix_mentions=vix_mentions,
                tech_sector_signals=tech_sector_signals,
                financial_sector_signals=financial_sector_signals,
            )

        except Exception:
            logger.exception("Error extracting NASDAQ indicators")
            return NasdaqIndicators()

    def calculate_correlation_score(
        self, crypto_data: dict, nasdaq_data: dict
    ) -> float:
        """
        Calculate correlation score between crypto and NASDAQ indicators.

        Args:
            crypto_data: Crypto indicator data
            nasdaq_data: NASDAQ indicator data

        Returns:
            Correlation score between -1 and 1
        """
        try:
            correlation_factors = []

            # Trend direction correlation
            crypto_trend = crypto_data.get("trend_direction", "NEUTRAL")
            nasdaq_trend = nasdaq_data.get("nasdaq_trend", "NEUTRAL")

            if crypto_trend and nasdaq_trend:
                if crypto_trend == nasdaq_trend:
                    correlation_factors.append(0.8)
                elif "NEUTRAL" in [crypto_trend, nasdaq_trend]:
                    correlation_factors.append(0.0)
                else:
                    correlation_factors.append(-0.8)

            # Risk sentiment correlation
            crypto_risk_signals = len(crypto_data.get("volatility_signals", []))
            nasdaq_risk_off = len(nasdaq_data.get("risk_off_signals", []))

            if crypto_risk_signals > 0 and nasdaq_risk_off > 0:
                correlation_factors.append(0.6)
            elif crypto_risk_signals == 0 and nasdaq_risk_off == 0:
                correlation_factors.append(0.3)

            # Volume and momentum correlation
            crypto_momentum = len(crypto_data.get("momentum_signals", []))
            nasdaq_momentum_proxy = len(nasdaq_data.get("tech_sector_signals", []))

            if crypto_momentum > 0 and nasdaq_momentum_proxy > 0:
                correlation_factors.append(0.4)

            # Calculate weighted average
            if correlation_factors:
                return sum(correlation_factors) / len(correlation_factors)
        except Exception:
            logger.exception("Error calculating correlation score")
            return 0.0
        else:
            return 0.0

    def format_sentiment_for_llm(self, sentiment_data: dict) -> str:
        """
        Format sentiment analysis data for LLM consumption.

        Args:
            sentiment_data: Dictionary containing sentiment analysis results

        Returns:
            Formatted string for LLM prompt inclusion
        """
        try:
            sentiment_result = sentiment_data.get("sentiment_result")
            crypto_indicators = sentiment_data.get("crypto_indicators")
            nasdaq_indicators = sentiment_data.get("nasdaq_indicators")
            correlation_score = sentiment_data.get("correlation_score", 0.0)

            output_lines = ["=== FINANCIAL SENTIMENT ANALYSIS ===", ""]

            # Overall sentiment
            if sentiment_result:
                sentiment_label = self._get_sentiment_label(
                    sentiment_result.sentiment_score
                )
                output_lines.extend(
                    [
                        f"Overall Market Sentiment: {sentiment_label}",
                        f"Sentiment Score: {sentiment_result.sentiment_score:.2f} (Range: -1 to +1)",
                        f"Confidence Level: {sentiment_result.confidence:.2f}",
                        "",
                    ]
                )

                if sentiment_result.key_themes:
                    output_lines.extend(
                        [
                            "Key Market Themes:",
                            *[
                                f"  • {theme}"
                                for theme in sentiment_result.key_themes[:5]
                            ],
                            "",
                        ]
                    )

            # Crypto-specific indicators
            if crypto_indicators:
                output_lines.extend(
                    [
                        "=== CRYPTO MARKET INDICATORS ===",
                        f"Trend Direction: {crypto_indicators.trend_direction or 'NEUTRAL'}",
                        "",
                    ]
                )

                if crypto_indicators.momentum_signals:
                    output_lines.extend(
                        [
                            "Momentum Signals:",
                            *[
                                f"  • {signal}"
                                for signal in crypto_indicators.momentum_signals[:3]
                            ],
                            "",
                        ]
                    )

                if crypto_indicators.technical_patterns:
                    output_lines.extend(
                        [
                            "Technical Patterns:",
                            *[
                                f"  • {pattern}"
                                for pattern in crypto_indicators.technical_patterns[:3]
                            ],
                            "",
                        ]
                    )

            # Traditional market indicators
            if nasdaq_indicators:
                output_lines.extend(
                    [
                        "=== TRADITIONAL MARKET INDICATORS ===",
                        f"NASDAQ Trend: {nasdaq_indicators.nasdaq_trend or 'NEUTRAL'}",
                        "",
                    ]
                )

                if nasdaq_indicators.fed_policy_signals:
                    output_lines.extend(
                        [
                            "Fed Policy Signals:",
                            *[
                                f"  • {signal}"
                                for signal in nasdaq_indicators.fed_policy_signals[:2]
                            ],
                            "",
                        ]
                    )

                risk_sentiment = (
                    "RISK-ON"
                    if len(nasdaq_indicators.risk_on_signals)
                    > len(nasdaq_indicators.risk_off_signals)
                    else "RISK-OFF"
                )
                if (
                    nasdaq_indicators.risk_on_signals
                    or nasdaq_indicators.risk_off_signals
                ):
                    output_lines.append(f"Risk Sentiment: {risk_sentiment}")
                    output_lines.append("")

            # Correlation analysis
            correlation_label = self._get_correlation_label(correlation_score)
            output_lines.extend(
                [
                    "=== MARKET CORRELATION ===",
                    f"Crypto-Traditional Correlation: {correlation_label}",
                    f"Correlation Score: {correlation_score:.2f}",
                    "",
                ]
            )

            # Trading implications
            output_lines.extend(
                [
                    "=== TRADING IMPLICATIONS ===",
                    self._generate_trading_implications(sentiment_data),
                    "",
                ]
            )

            return "\n".join(output_lines)

        except Exception:
            logger.exception("Error formatting sentiment for LLM")
            return "Error: Could not format sentiment analysis data"

    # Helper methods for text analysis

    def _extract_themes(self, text: str) -> list[str]:
        """Extract key themes from text."""
        themes = []
        keywords_to_themes = {
            "bitcoin": "Bitcoin",
            "btc": "Bitcoin",
            "ethereum": "Ethereum",
            "eth": "Ethereum",
            "defi": "DeFi",
            "nft": "NFTs",
            "regulation": "Regulation",
            "adoption": "Adoption",
            "institutional": "Institutional Investment",
            "etf": "ETF",
            "fed": "Federal Reserve",
            "inflation": "Inflation",
            "rate": "Interest Rates",
        }

        text_lower = text.lower()
        for keyword, theme in keywords_to_themes.items():
            if keyword in text_lower:
                themes.append(theme)

        return themes

    def _determine_trend_direction(self, text: str) -> str | None:
        """Determine overall trend direction from text."""
        bullish_words = sum(1 for word in self._bullish_keywords if word in text)
        bearish_words = sum(1 for word in self._bearish_keywords if word in text)

        if bullish_words > bearish_words * 1.5:
            return "BULLISH"
        if bearish_words > bullish_words * 1.5:
            return "BEARISH"
        return "NEUTRAL"

    def _extract_price_levels(self, text: str, level_words: list[str]) -> list[float]:
        """Extract price levels associated with specific words."""
        levels = []
        for word in level_words:
            # Look for price patterns near level words
            pattern = rf"{word}.*?\$?(\d+(?:,\d{{3}})*(?:\.\d{{2}})?)"
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    price = float(match.group(1).replace(",", ""))
                    levels.append(price)
                except ValueError:
                    continue

        return sorted(set(levels))  # Remove duplicates and sort

    def _extract_volume_indicators(self, text: str) -> list[str]:
        """Extract volume-related indicators."""
        volume_keywords = [
            "volume",
            "trading volume",
            "high volume",
            "low volume",
            "volume spike",
        ]
        return [kw for kw in volume_keywords if kw in text]

    def _extract_momentum_signals(self, text: str) -> list[str]:
        """Extract momentum signals."""
        momentum_keywords = [
            "momentum",
            "accelerating",
            "building steam",
            "gaining traction",
            "losing steam",
        ]
        return [kw for kw in momentum_keywords if kw in text]

    def _extract_adoption_signals(self, text: str) -> list[str]:
        """Extract adoption signals."""
        adoption_keywords = [
            "adoption",
            "institutional",
            "mainstream",
            "corporate",
            "enterprise",
        ]
        return [kw for kw in adoption_keywords if kw in text]

    def _extract_regulatory_mentions(self, text: str) -> list[str]:
        """Extract regulatory mentions."""
        regulatory_keywords = [
            "regulation",
            "regulatory",
            "sec",
            "cftc",
            "compliance",
            "legal",
        ]
        return [kw for kw in regulatory_keywords if kw in text]

    def _extract_whale_activity(self, text: str) -> list[str]:
        """Extract whale activity mentions."""
        whale_keywords = [
            "whale",
            "large holder",
            "institutional buying",
            "massive transfer",
        ]
        return [kw for kw in whale_keywords if kw in text]

    def _extract_technical_patterns(self, text: str) -> list[str]:
        """Extract technical chart patterns."""
        pattern_keywords = [
            "triangle",
            "wedge",
            "flag",
            "pennant",
            "head and shoulders",
            "double top",
            "double bottom",
            "cup and handle",
            "ascending triangle",
        ]
        return [kw for kw in pattern_keywords if kw in text]

    def _extract_indicator_signals(self, text: str) -> dict[str, str]:
        """Extract technical indicator signals."""
        signals = {}
        for indicator in self._technical_indicators:
            if indicator in text:
                # Simple pattern to detect signal direction
                if any(bullish in text for bullish in ["bullish", "positive", "buy"]):
                    signals[indicator] = "BULLISH"
                elif any(
                    bearish in text for bearish in ["bearish", "negative", "sell"]
                ):
                    signals[indicator] = "BEARISH"
                else:
                    signals[indicator] = "NEUTRAL"

        return signals

    def _extract_nasdaq_trend(self, text: str) -> str | None:
        """Extract NASDAQ trend direction."""
        nasdaq_keywords = ["nasdaq", "qqq", "tech stocks"]
        for keyword in nasdaq_keywords:
            if keyword in text:
                return self._determine_trend_direction(text)
        return None

    def _extract_index_mentions(self, text: str, index_terms: list[str]) -> list[str]:
        """Extract mentions of market indices."""
        mentions = []
        for term in index_terms:
            if term in text:
                # Extract context around the mention
                idx = text.find(term)
                context = text[max(0, idx - 30) : idx + len(term) + 30]
                mentions.append(context.strip())
        return mentions

    def _extract_fed_signals(self, text: str) -> list[str]:
        """Extract Federal Reserve policy signals."""
        fed_keywords = [
            "fed",
            "federal reserve",
            "powell",
            "monetary policy",
            "interest rate",
        ]
        signals = []
        for keyword in fed_keywords:
            if keyword in text:
                signals.append(f"Fed policy mention: {keyword}")
        return signals

    def _extract_interest_rate_mentions(self, text: str) -> list[str]:
        """Extract interest rate mentions."""
        rate_keywords = [
            "interest rate",
            "fed rate",
            "rate hike",
            "rate cut",
            "basis points",
        ]
        return [kw for kw in rate_keywords if kw in text]

    def _extract_inflation_indicators(self, text: str) -> list[str]:
        """Extract inflation indicators."""
        inflation_keywords = ["inflation", "cpi", "pce", "deflation", "price pressure"]
        return [kw for kw in inflation_keywords if kw in text]

    def _extract_risk_sentiment(self, text: str, risk_type: str) -> list[str]:
        """Extract risk sentiment indicators."""
        if risk_type == "on":
            keywords = ["risk-on", "risk appetite", "buying the dip", "optimism"]
        else:
            keywords = [
                "risk-off",
                "flight to safety",
                "selling pressure",
                "uncertainty",
            ]

        return [kw for kw in keywords if kw in text]

    def _extract_vix_mentions(self, text: str) -> list[str]:
        """Extract VIX volatility mentions."""
        vix_keywords = ["vix", "volatility index", "fear index", "market volatility"]
        return [kw for kw in vix_keywords if kw in text]

    def _extract_sector_signals(self, text: str, sector: str) -> list[str]:
        """Extract sector-specific signals."""
        if sector == "tech":
            keywords = [
                "technology",
                "tech stocks",
                "faang",
                "semiconductors",
                "software",
            ]
        elif sector == "financial":
            keywords = ["banks", "financial", "credit", "lending", "fintech"]
        else:
            keywords = []

        return [kw for kw in keywords if kw in text]

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to human-readable label."""
        if score > 0.3:
            return "STRONGLY BULLISH"
        if score > 0.1:
            return "BULLISH"
        if score > -0.1:
            return "NEUTRAL"
        if score > -0.3:
            return "BEARISH"
        return "STRONGLY BEARISH"

    def _get_correlation_label(self, score: float) -> str:
        """Convert correlation score to human-readable label."""
        if score > 0.5:
            return "STRONG POSITIVE"
        if score > 0.2:
            return "MODERATE POSITIVE"
        if score > -0.2:
            return "WEAK/UNCORRELATED"
        if score > -0.5:
            return "MODERATE NEGATIVE"
        return "STRONG NEGATIVE"

    def _generate_trading_implications(self, sentiment_data: dict) -> str:
        """Generate trading implications from sentiment data."""
        try:
            sentiment_result = sentiment_data.get("sentiment_result")
            correlation_score = sentiment_data.get("correlation_score", 0.0)

            if not sentiment_result:
                return "Insufficient data for trading implications"

            implications = []

            # Sentiment-based implications
            if sentiment_result.sentiment_score > 0.3:
                implications.append("Strong bullish sentiment supports long positions")
            elif sentiment_result.sentiment_score < -0.3:
                implications.append(
                    "Strong bearish sentiment suggests caution or short opportunities"
                )
            else:
                implications.append(
                    "Neutral sentiment suggests range-bound or consolidation phase"
                )

            # Confidence-based implications
            if sentiment_result.confidence < 0.3:
                implications.append(
                    "Low confidence suggests waiting for clearer signals"
                )
            elif sentiment_result.confidence > 0.7:
                implications.append(
                    "High confidence supports acting on sentiment signals"
                )

            # Correlation-based implications
            if abs(correlation_score) > 0.5:
                implications.append(
                    "Strong market correlation increases systematic risk"
                )
            else:
                implications.append(
                    "Weak correlation suggests crypto-specific factors dominate"
                )

            return " | ".join(implications)

        except Exception:
            logger.exception("Error generating trading implications")
            return "Error generating trading implications"
