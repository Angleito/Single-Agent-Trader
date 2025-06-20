"""Unit tests for web search formatter module."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from bot.utils.web_search_formatter import (
    ContentPriority,
    FormattedContent,
    WebSearchFormatter,
)

# Import or mock the dependencies based on availability
try:
    from bot.analysis.market_context import CorrelationAnalysis
    from bot.services.financial_sentiment import SentimentResult
except ImportError:
    # Create mock classes for testing if imports fail
    class SentimentResult:
        def __init__(
            self,
            sentiment_score=0.0,
            confidence=0.0,
            key_themes=None,
            bullish_indicators=None,
            bearish_indicators=None,
            volatility_signals=None,
        ):
            self.sentiment_score = sentiment_score
            self.confidence = confidence
            self.key_themes = key_themes or []
            self.bullish_indicators = bullish_indicators or []
            self.bearish_indicators = bearish_indicators or []
            self.volatility_signals = volatility_signals or []
            self.timestamp = datetime.now(UTC)

    class CorrelationAnalysis:
        def __init__(self, **kwargs):
            self.correlation_coefficient = kwargs.get("correlation_coefficient", 0.0)
            self.correlation_strength = type("MockStrength", (), {"value": "WEAK"})()
            self.direction = kwargs.get("direction", "NEUTRAL")
            self.p_value = kwargs.get("p_value", 1.0)
            self.is_significant = kwargs.get("is_significant", False)
            self.sample_size = kwargs.get("sample_size", 0)
            self.rolling_correlation_24h = kwargs.get("rolling_correlation_24h")
            self.rolling_correlation_7d = kwargs.get("rolling_correlation_7d")
            self.correlation_stability = kwargs.get("correlation_stability", 0.0)
            self.regime_dependent_correlation = kwargs.get(
                "regime_dependent_correlation", {}
            )
            self.reliability_score = kwargs.get("reliability_score", 0.0)


class TestContentPriority:
    """Test cases for ContentPriority model."""

    def test_content_priority_creation(self):
        """Test creating ContentPriority with valid scores."""
        priority = ContentPriority(
            relevance_score=0.8,
            freshness_score=0.9,
            authority_score=0.7,
            trading_impact_score=0.85,
            final_priority=0.82,
        )

        assert priority.relevance_score == 0.8
        assert priority.freshness_score == 0.9
        assert priority.authority_score == 0.7
        assert priority.trading_impact_score == 0.85
        assert priority.final_priority == 0.82

    def test_content_priority_validation(self):
        """Test ContentPriority field validation."""
        with pytest.raises(ValueError):
            # Relevance score out of range
            ContentPriority(
                relevance_score=1.5,
                freshness_score=0.9,
                authority_score=0.7,
                trading_impact_score=0.85,
                final_priority=0.82,
            )

        with pytest.raises(ValueError):
            # Negative authority score
            ContentPriority(
                relevance_score=0.8,
                freshness_score=0.9,
                authority_score=-0.1,
                trading_impact_score=0.85,
                final_priority=0.82,
            )


class TestFormattedContent:
    """Test cases for FormattedContent model."""

    def test_formatted_content_creation(self):
        """Test creating FormattedContent with basic data."""
        priority = ContentPriority(
            relevance_score=0.8,
            freshness_score=0.9,
            authority_score=0.7,
            trading_impact_score=0.85,
            final_priority=0.82,
        )

        content = FormattedContent(
            summary="Bitcoin reaches new highs amid institutional demand",
            key_insights=[
                "Institutional adoption accelerating",
                "Technical breakout confirmed",
            ],
            trading_signals=["Bullish momentum building"],
            market_sentiment="BULLISH",
            confidence_level=0.85,
            token_count=150,
            priority=priority,
        )

        assert content.summary == "Bitcoin reaches new highs amid institutional demand"
        assert len(content.key_insights) == 2
        assert len(content.trading_signals) == 1
        assert content.market_sentiment == "BULLISH"
        assert content.confidence_level == 0.85

    def test_formatted_content_defaults(self):
        """Test FormattedContent with default values."""
        priority = ContentPriority(
            relevance_score=0.5,
            freshness_score=0.5,
            authority_score=0.5,
            trading_impact_score=0.5,
            final_priority=0.5,
        )

        content = FormattedContent(
            summary="Test summary",
            confidence_level=0.7,
            token_count=100,
            priority=priority,
        )

        assert content.key_insights == []  # Default empty list
        assert content.trading_signals == []  # Default empty list
        assert content.market_sentiment == "NEUTRAL"  # Default neutral
        assert isinstance(content.timestamp, datetime)

    def test_formatted_content_validation(self):
        """Test FormattedContent field validation."""
        priority = ContentPriority(
            relevance_score=0.5,
            freshness_score=0.5,
            authority_score=0.5,
            trading_impact_score=0.5,
            final_priority=0.5,
        )

        with pytest.raises(ValueError):
            # Confidence level out of range
            FormattedContent(
                summary="Test", confidence_level=1.5, token_count=100, priority=priority
            )

        with pytest.raises(ValueError):
            # Negative token count
            FormattedContent(
                summary="Test", confidence_level=0.7, token_count=-10, priority=priority
            )


class TestWebSearchFormatter:
    """Test cases for WebSearchFormatter."""

    def test_formatter_initialization_defaults(self):
        """Test formatter initialization with default values."""
        formatter = WebSearchFormatter()

        assert formatter.max_tokens_per_section == 500
        assert formatter.max_total_tokens == 2000
        assert len(formatter._trading_keywords) > 0
        assert len(formatter._authority_sources) > 0
        assert len(formatter._content_hashes) == 0

    def test_formatter_initialization_custom(self):
        """Test formatter initialization with custom values."""
        formatter = WebSearchFormatter(
            max_tokens_per_section=300, max_total_tokens=1500
        )

        assert formatter.max_tokens_per_section == 300
        assert formatter.max_total_tokens == 1500

    @pytest.mark.asyncio()
    async def test_format_news_results_empty(self):
        """Test formatting with empty news list."""
        formatter = WebSearchFormatter()

        result = await formatter.format_news_results([])

        assert "NEWS ANALYSIS" in result
        assert "No recent news data available" in result

    @pytest.mark.asyncio()
    async def test_format_news_results_basic(self):
        """Test basic news formatting."""
        formatter = WebSearchFormatter()

        news_items = [
            {
                "title": "Bitcoin Surges Past $60K on ETF Hopes",
                "content": "Bitcoin price rallies with strong bullish momentum as ETF approval hopes grow",
                "url": "https://bloomberg.com/bitcoin-surge",
                "published_time": datetime.now(UTC),
            },
            {
                "title": "Ethereum DeFi TVL Reaches Record High",
                "content": "Ethereum ecosystem shows bullish growth with institutional adoption",
                "url": "https://coindesk.com/ethereum-defi",
                "published_time": datetime.now(UTC) - timedelta(hours=2),
            },
        ]

        result = await formatter.format_news_results(news_items)

        assert "NEWS ANALYSIS" in result
        assert "Bitcoin" in result
        assert "Ethereum" in result
        assert "BULLISH" in result or "bullish" in result
        assert "Articles Analyzed: 2" in result

    @pytest.mark.asyncio()
    async def test_format_news_results_high_authority(self):
        """Test news formatting with high authority sources."""
        formatter = WebSearchFormatter()

        news_items = [
            {
                "title": "Fed Policy Impacts Crypto Markets",
                "content": "Federal Reserve policy changes affect cryptocurrency trading",
                "url": "https://reuters.com/fed-crypto",
                "published_time": datetime.now(UTC),
            }
        ]

        result = await formatter.format_news_results(news_items)

        # Should recognize Reuters as high-authority source
        assert "NEWS ANALYSIS" in result
        assert "Fed" in result or "Federal" in result

    @pytest.mark.asyncio()
    async def test_format_sentiment_data_basic(self):
        """Test basic sentiment data formatting."""
        formatter = WebSearchFormatter()

        sentiment = SentimentResult(
            sentiment_score=0.6,
            confidence=0.8,
            key_themes=["Bitcoin", "ETF", "Institutional"],
            bullish_indicators=["Strong demand", "Technical breakout"],
            bearish_indicators=["Regulatory concerns"],
            volatility_signals=["High options volume"],
        )

        result = await formatter.format_sentiment_data(sentiment)

        assert "MARKET SENTIMENT ANALYSIS" in result
        assert "0.60" in result  # Sentiment score
        assert "0.80" in result  # Confidence
        assert "Bitcoin" in result
        assert "Strong demand" in result
        assert "Regulatory concerns" in result

    @pytest.mark.asyncio()
    async def test_format_sentiment_data_bullish(self):
        """Test formatting strongly bullish sentiment."""
        formatter = WebSearchFormatter()

        sentiment = SentimentResult(
            sentiment_score=0.85,
            confidence=0.9,
            key_themes=["Institutional adoption", "Bull market"],
            bullish_indicators=["Massive buying pressure", "New ATH"],
            bearish_indicators=[],
            volatility_signals=[],
        )

        result = await formatter.format_sentiment_data(sentiment)

        assert "STRONGLY BULLISH" in result
        assert "🚀" in result  # Should include bullish emoji
        assert "🎯" in result  # High confidence emoji
        assert "Massive buying pressure" in result

    @pytest.mark.asyncio()
    async def test_format_sentiment_data_bearish(self):
        """Test formatting strongly bearish sentiment."""
        formatter = WebSearchFormatter()

        sentiment = SentimentResult(
            sentiment_score=-0.7,
            confidence=0.75,
            key_themes=["Market crash", "Liquidations"],
            bullish_indicators=[],
            bearish_indicators=["Panic selling", "Support breakdown"],
            volatility_signals=["VIX spike"],
        )

        result = await formatter.format_sentiment_data(sentiment)

        assert "BEARISH" in result
        assert "💥" in result  # Bearish emoji
        assert "Panic selling" in result
        assert "VIX spike" in result

    @pytest.mark.asyncio()
    async def test_format_correlation_analysis_basic(self):
        """Test basic correlation analysis formatting."""
        formatter = WebSearchFormatter()

        correlation = CorrelationAnalysis(
            correlation_coefficient=0.65,
            correlation_strength=type("MockStrength", (), {"value": "MODERATE"})(),
            direction="POSITIVE",
            p_value=0.02,
            is_significant=True,
            sample_size=150,
            correlation_stability=0.7,
            reliability_score=0.8,
        )

        result = await formatter.format_correlation_analysis(correlation)

        assert "CRYPTO-NASDAQ CORRELATION ANALYSIS" in result
        assert "0.650" in result  # Correlation coefficient
        assert "MODERATE" in result
        assert "POSITIVE" in result
        assert "150" in result  # Sample size
        assert "✅" in result  # Significance indicator

    @pytest.mark.asyncio()
    async def test_format_correlation_analysis_with_rolling(self):
        """Test correlation formatting with rolling data."""
        formatter = WebSearchFormatter()

        correlation = CorrelationAnalysis(
            correlation_coefficient=0.45,
            correlation_strength=type("MockStrength", (), {"value": "MODERATE"})(),
            direction="POSITIVE",
            p_value=0.01,
            is_significant=True,
            sample_size=200,
            rolling_correlation_24h=0.52,
            rolling_correlation_7d=0.38,
            correlation_stability=0.65,
            regime_dependent_correlation={
                "HIGH_VOLATILITY": 0.7,
                "LOW_VOLATILITY": 0.3,
            },
            reliability_score=0.85,
        )

        result = await formatter.format_correlation_analysis(correlation)

        assert "24-Hour: 0.520" in result
        assert "7-Day: 0.380" in result
        assert "HIGH_VOLATILITY: 0.700" in result
        assert "LOW_VOLATILITY: 0.300" in result

    @pytest.mark.asyncio()
    async def test_format_market_context_comprehensive(self):
        """Test comprehensive market context formatting."""
        formatter = WebSearchFormatter()

        context = {
            "news_results": [
                {
                    "title": "Bitcoin Bull Run Continues",
                    "content": "Strong institutional demand drives prices higher",
                    "url": "https://bloomberg.com/bitcoin",
                    "published_time": datetime.now(UTC),
                }
            ],
            "sentiment_result": SentimentResult(
                sentiment_score=0.7,
                confidence=0.85,
                key_themes=["Bitcoin", "Institutional"],
            ),
            "correlation_analysis": CorrelationAnalysis(
                correlation_coefficient=0.6,
                correlation_strength=type("MockStrength", (), {"value": "MODERATE"})(),
                direction="POSITIVE",
                p_value=0.03,
                is_significant=True,
                sample_size=100,
                correlation_stability=0.6,
                reliability_score=0.75,
            ),
        }

        result = await formatter.format_market_context(context)

        assert "COMPREHENSIVE MARKET CONTEXT ANALYSIS" in result
        assert "NEWS ANALYSIS" in result
        assert "MARKET SENTIMENT ANALYSIS" in result
        assert "CORRELATION ANALYSIS" in result
        assert "Bitcoin" in result
        assert "Institutional" in result

    def test_truncate_content_no_truncation_needed(self):
        """Test content truncation when no truncation is needed."""
        formatter = WebSearchFormatter()

        text = "Short text that doesn't need truncation"
        result = formatter.truncate_content(text, 100)

        assert result == text

    def test_truncate_content_sentence_boundary(self):
        """Test content truncation at sentence boundary."""
        formatter = WebSearchFormatter()

        text = "First sentence. Second sentence. Third sentence that will be cut off."
        result = formatter.truncate_content(text, 35)

        assert result.endswith("...")
        assert "First sentence." in result
        assert "Third sentence" not in result

    def test_truncate_content_word_boundary(self):
        """Test content truncation at word boundary."""
        formatter = WebSearchFormatter()

        text = "This is a long text without proper sentence endings"
        result = formatter.truncate_content(text, 25)

        assert result.endswith("...")
        assert len(result) <= 28  # Original length + "..."

    def test_truncate_content_error_handling(self):
        """Test content truncation error handling."""
        formatter = WebSearchFormatter()

        # Test with edge cases
        result1 = formatter.truncate_content("", 10)
        assert result1 == ""

        result2 = formatter.truncate_content("short", 3)
        assert result2 == "..."

    @pytest.mark.asyncio()
    async def test_extract_key_insights_comprehensive(self):
        """Test comprehensive key insights extraction."""
        formatter = WebSearchFormatter()

        search_results = {
            "news_items": [
                {
                    "title": "Bitcoin Breakout Above $60K Resistance",
                    "content": "Technical analysis shows strong bullish momentum with institutional buying",
                }
            ],
            "sentiment_data": {"sentiment_score": 0.7, "confidence": 0.8},
            "price_data": {
                "price_change_24h": 0.08,  # 8% increase
                "volume_change_24h": 0.3,  # 30% volume increase
            },
            "technical_analysis": {"rsi": 75, "trend_direction": "BULLISH"},
        }

        insights = await formatter.extract_key_insights(search_results)

        assert len(insights) > 0
        assert any("breakout" in insight.lower() for insight in insights)
        assert any("bullish" in insight.lower() for insight in insights)

    @pytest.mark.asyncio()
    async def test_extract_key_insights_empty(self):
        """Test key insights extraction with empty data."""
        formatter = WebSearchFormatter()

        insights = await formatter.extract_key_insights({})

        assert isinstance(insights, list)
        # Should handle empty data gracefully

    @pytest.mark.asyncio()
    async def test_extract_key_insights_error_handling(self):
        """Test key insights extraction error handling."""
        formatter = WebSearchFormatter()

        # Test with invalid data
        search_results = {
            "news_items": [{"invalid": "data"}],
            "sentiment_data": "invalid_data",
        }

        insights = await formatter.extract_key_insights(search_results)

        assert isinstance(insights, list)
        # Should handle errors gracefully

    def test_calculate_relevance_score_high(self):
        """Test relevance score calculation with highly relevant content."""
        formatter = WebSearchFormatter()

        title = "Bitcoin Technical Breakout Above Key Resistance"
        content = """
        Bitcoin shows strong bullish momentum with RSI indicating overbought conditions.
        Technical analysis reveals a clear breakout pattern above $50,000 resistance.
        Volume surge confirms the bullish trend with institutional accumulation evident.
        """

        score = formatter._calculate_relevance_score(title, content)

        assert score > 0.5  # Should be high relevance
        assert score <= 1.0

    def test_calculate_relevance_score_low(self):
        """Test relevance score calculation with low relevant content."""
        formatter = WebSearchFormatter()

        title = "Weather Report for New York"
        content = "Today will be sunny with temperatures reaching 75 degrees"

        score = formatter._calculate_relevance_score(title, content)

        assert score < 0.3  # Should be low relevance

    def test_calculate_relevance_score_crypto_bonus(self):
        """Test relevance score with crypto-specific terms."""
        formatter = WebSearchFormatter()

        title = "Bitcoin and Ethereum Show Strong Performance"
        content = "Cryptocurrency markets rally with BTC and ETH leading gains"

        score = formatter._calculate_relevance_score(title, content)

        # Should get bonus for crypto terms
        assert score > 0.1

    def test_calculate_freshness_score_recent(self):
        """Test freshness score with recent content."""
        formatter = WebSearchFormatter()

        recent_time = datetime.now(UTC) - timedelta(minutes=30)
        score = formatter._calculate_freshness_score(recent_time)

        assert score > 0.9  # Very fresh content

    def test_calculate_freshness_score_old(self):
        """Test freshness score with old content."""
        formatter = WebSearchFormatter()

        old_time = datetime.now(UTC) - timedelta(days=10)
        score = formatter._calculate_freshness_score(old_time)

        assert score < 0.5  # Old content

    def test_calculate_freshness_score_string_parsing(self):
        """Test freshness score with string datetime."""
        formatter = WebSearchFormatter()

        with patch("dateutil.parser.parse") as mock_parse:
            mock_parse.return_value = datetime.now(UTC) - timedelta(hours=2)

            score = formatter._calculate_freshness_score("2024-01-01T12:00:00Z")

            assert 0.0 <= score <= 1.0
            mock_parse.assert_called_once()

    def test_calculate_authority_score_high_authority(self):
        """Test authority score with high-authority source."""
        formatter = WebSearchFormatter()

        score = formatter._calculate_authority_score(
            "https://bloomberg.com/bitcoin-news"
        )

        assert score == 0.95  # Bloomberg is high authority

    def test_calculate_authority_score_unknown_source(self):
        """Test authority score with unknown source."""
        formatter = WebSearchFormatter()

        score = formatter._calculate_authority_score("https://unknown-blog.com/crypto")

        assert score == 0.3  # Default for unknown sources

    def test_calculate_trading_impact_score_high_impact(self):
        """Test trading impact score with high-impact content."""
        formatter = WebSearchFormatter()

        title = "BREAKING: SEC Approves Bitcoin ETF"
        content = """
        Major announcement from SEC on Bitcoin ETF approval. This significant
        decision will have massive impact on institutional adoption and price targets.
        """

        score = formatter._calculate_trading_impact_score(title, content)

        assert score > 0.5  # High impact

    def test_calculate_trading_impact_score_low_impact(self):
        """Test trading impact score with low-impact content."""
        formatter = WebSearchFormatter()

        title = "General Market Commentary"
        content = "Some general thoughts on market conditions"

        score = formatter._calculate_trading_impact_score(title, content)

        assert score < 0.5  # Low impact

    def test_extract_item_insights_breakout(self):
        """Test insight extraction for breakout patterns."""
        formatter = WebSearchFormatter()

        title = "Bitcoin Breaks Above $60K Resistance"
        content = "Clear breakout pattern with strong support levels holding at $55K"

        insights = formatter._extract_item_insights(title, content)

        assert any("breakout" in insight.lower() for insight in insights)
        assert any("level" in insight.lower() for insight in insights)

    def test_extract_item_insights_institutional(self):
        """Test insight extraction for institutional activity."""
        formatter = WebSearchFormatter()

        title = "Whale Moves $100M Bitcoin to Exchange"
        content = "Large institutional holder transfers significant Bitcoin position"

        insights = formatter._extract_item_insights(title, content)

        assert any(
            "institutional" in insight.lower() or "whale" in insight.lower()
            for insight in insights
        )

    def test_extract_trading_signals_bullish(self):
        """Test trading signal extraction for bullish signals."""
        formatter = WebSearchFormatter()

        title = "Strong Buy Signal for Bitcoin"
        content = "Technical analysis shows bullish momentum with long opportunities"

        signals = formatter._extract_trading_signals(title, content)

        assert any("bullish" in signal.lower() for signal in signals)

    def test_extract_trading_signals_bearish(self):
        """Test trading signal extraction for bearish signals."""
        formatter = WebSearchFormatter()

        title = "Short Signal Triggered for Crypto"
        content = "Bearish indicators suggest sell signals with downtrend confirmed"

        signals = formatter._extract_trading_signals(title, content)

        assert any("bearish" in signal.lower() for signal in signals)

    def test_generate_content_summary_basic(self):
        """Test basic content summary generation."""
        formatter = WebSearchFormatter()

        title = "Bitcoin Reaches New High"
        content = "Bitcoin price surged to a new all-time high today. The rally was driven by institutional demand."

        summary = formatter._generate_content_summary(title, content)

        assert "Bitcoin Reaches New High" in summary
        assert len(summary) <= 250

    def test_generate_content_summary_title_only(self):
        """Test summary generation with title only."""
        formatter = WebSearchFormatter()

        title = "Ethereum Network Upgrade Complete"
        content = ""

        summary = formatter._generate_content_summary(title, content)

        assert summary == "Ethereum Network Upgrade Complete"

    def test_determine_content_sentiment_bullish(self):
        """Test content sentiment determination - bullish."""
        formatter = WebSearchFormatter()

        title = "Bitcoin Surges on Bullish News"
        content = "Positive momentum drives rally with breakout confirmation"

        sentiment = formatter._determine_content_sentiment(title, content)

        assert sentiment == "BULLISH"

    def test_determine_content_sentiment_bearish(self):
        """Test content sentiment determination - bearish."""
        formatter = WebSearchFormatter()

        title = "Crypto Crash Continues"
        content = "Bearish sentiment dominates with negative price action and sell-off"

        sentiment = formatter._determine_content_sentiment(title, content)

        assert sentiment == "BEARISH"

    def test_determine_content_sentiment_neutral(self):
        """Test content sentiment determination - neutral."""
        formatter = WebSearchFormatter()

        title = "Market Analysis Update"
        content = "Mixed signals in the market with balanced conditions"

        sentiment = formatter._determine_content_sentiment(title, content)

        assert sentiment == "NEUTRAL"

    def test_estimate_token_count(self):
        """Test token count estimation."""
        formatter = WebSearchFormatter()

        text = "This is a test text for token estimation"
        tokens = formatter._estimate_token_count(text)

        # Rough approximation: 1 token ≈ 4 characters
        expected = len(text) // 4
        assert tokens == expected

    def test_optimize_content_for_tokens_no_optimization(self):
        """Test content optimization when no optimization needed."""
        formatter = WebSearchFormatter()

        short_content = "Short content that fits within limits"
        result = formatter._optimize_content_for_tokens(short_content)

        assert result == short_content

    def test_optimize_content_for_tokens_compression(self):
        """Test content optimization with compression."""
        formatter = WebSearchFormatter(max_total_tokens=10)  # Very small limit

        long_content = "This is a very long content that definitely exceeds the token limit and needs to be compressed significantly"
        result = formatter._optimize_content_for_tokens(long_content)

        assert len(result) < len(long_content)
        assert result.endswith("...")

    @pytest.mark.asyncio()
    async def test_extract_news_insights(self):
        """Test news insights extraction."""
        formatter = WebSearchFormatter()

        news_items = [
            {
                "title": "Bitcoin Breakout Above Resistance",
                "content": "Technical breakout confirms bullish trend with institutional support",
            },
            {
                "title": "Regulatory Clarity Expected",
                "content": "SEC expected to provide regulatory framework for crypto markets",
            },
        ]

        insights = await formatter._extract_news_insights(news_items)

        assert len(insights) > 0
        assert any("breakout" in insight.lower() for insight in insights)
        assert any("regulatory" in insight.lower() for insight in insights)

    def test_extract_sentiment_insights(self):
        """Test sentiment insights extraction."""
        formatter = WebSearchFormatter()

        sentiment_data = {
            "sentiment_score": 0.7,
            "confidence": 0.8,
            "volatility_signals": ["High VIX", "Options volatility"],
        }

        insights = formatter._extract_sentiment_insights(sentiment_data)

        assert len(insights) > 0
        assert any("bullish" in insight.lower() for insight in insights)
        assert any("volatility" in insight.lower() for insight in insights)

    def test_extract_price_insights(self):
        """Test price insights extraction."""
        formatter = WebSearchFormatter()

        price_data = {
            "price_change_24h": 0.08,  # 8% change
            "volume_change_24h": 0.25,  # 25% volume change
        }

        insights = formatter._extract_price_insights(price_data)

        assert len(insights) > 0
        assert any("upward" in insight.lower() for insight in insights)
        assert any("volume" in insight.lower() for insight in insights)

    def test_extract_technical_insights(self):
        """Test technical insights extraction."""
        formatter = WebSearchFormatter()

        technical_data = {"rsi": 75, "trend_direction": "BULLISH"}  # Overbought

        insights = formatter._extract_technical_insights(technical_data)

        assert len(insights) > 0
        assert any("overbought" in insight.lower() for insight in insights)
        assert any("bullish" in insight.lower() for insight in insights)

    def test_deduplicate_insights(self):
        """Test insight deduplication."""
        formatter = WebSearchFormatter()

        insights = [
            "Bitcoin shows bullish momentum",
            "Technical indicators are positive",
            "Bitcoin shows bullish momentum",  # Duplicate
            "Volume confirms the trend",
        ]

        unique_insights = formatter._deduplicate_insights(insights)

        assert len(unique_insights) == 3  # One duplicate removed
        assert "Bitcoin shows bullish momentum" in unique_insights
        assert "Technical indicators are positive" in unique_insights
        assert "Volume confirms the trend" in unique_insights

    def test_prioritize_insights(self):
        """Test insight prioritization."""
        formatter = WebSearchFormatter()

        insights = [
            "General market update",
            "Breakout pattern confirmed",
            "Institutional buying detected",
            "Technical analysis shows trend",
        ]

        prioritized = formatter._prioritize_insights(insights)

        # Breakout should be high priority
        assert prioritized[0] == "Breakout pattern confirmed"
        # Institutional should be second
        assert prioritized[1] == "Institutional buying detected"

    def test_get_sentiment_emoji_label(self):
        """Test sentiment emoji label generation."""
        formatter = WebSearchFormatter()

        assert "STRONGLY BULLISH" in formatter._get_sentiment_emoji_label(0.8)
        assert "🚀" in formatter._get_sentiment_emoji_label(0.8)

        assert "BULLISH" in formatter._get_sentiment_emoji_label(0.3)
        assert "📈" in formatter._get_sentiment_emoji_label(0.3)

        assert "NEUTRAL" in formatter._get_sentiment_emoji_label(0.0)
        assert "😐" in formatter._get_sentiment_emoji_label(0.0)

        assert "BEARISH" in formatter._get_sentiment_emoji_label(-0.3)
        assert "📉" in formatter._get_sentiment_emoji_label(-0.3)

        assert "STRONGLY BEARISH" in formatter._get_sentiment_emoji_label(-0.8)
        assert "💥" in formatter._get_sentiment_emoji_label(-0.8)

    def test_get_confidence_emoji(self):
        """Test confidence emoji generation."""
        formatter = WebSearchFormatter()

        assert formatter._get_confidence_emoji(0.9) == "🎯"  # High confidence
        assert formatter._get_confidence_emoji(0.7) == "✅"  # Good confidence
        assert formatter._get_confidence_emoji(0.5) == "⚠️"  # Medium confidence
        assert formatter._get_confidence_emoji(0.2) == "❓"  # Low confidence

    def test_get_correlation_emoji(self):
        """Test correlation emoji generation."""
        formatter = WebSearchFormatter()

        assert formatter._get_correlation_emoji(0.7) == "📈"  # Strong positive
        assert formatter._get_correlation_emoji(0.3) == "↗️"  # Moderate positive
        assert formatter._get_correlation_emoji(0.0) == "↔️"  # Uncorrelated
        assert formatter._get_correlation_emoji(-0.3) == "↘️"  # Moderate negative
        assert formatter._get_correlation_emoji(-0.7) == "📉"  # Strong negative

    def test_sentiment_to_score_conversion(self):
        """Test sentiment string to score conversion."""
        formatter = WebSearchFormatter()

        assert formatter._sentiment_to_score("STRONGLY_BULLISH") == 0.8
        assert formatter._sentiment_to_score("BULLISH") == 0.4
        assert formatter._sentiment_to_score("NEUTRAL") == 0.0
        assert formatter._sentiment_to_score("BEARISH") == -0.4
        assert formatter._sentiment_to_score("STRONGLY_BEARISH") == -0.8
        assert formatter._sentiment_to_score("UNKNOWN") == 0.0  # Default

    def test_generate_correlation_implications(self):
        """Test correlation implications generation."""
        formatter = WebSearchFormatter()

        # Strong positive correlation
        correlation = CorrelationAnalysis(
            correlation_coefficient=0.8,
            correlation_strength=type("MockStrength", (), {"value": "STRONG"})(),
            direction="POSITIVE",
            p_value=0.01,
            is_significant=True,
            sample_size=100,
            correlation_stability=0.8,
            reliability_score=0.9,
        )

        implications = formatter._generate_correlation_implications(correlation)

        assert (
            "systematic risk" in implications.lower()
            or "diversification" in implications.lower()
        )
        assert "significant" in implications.lower()

    @pytest.mark.asyncio()
    async def test_format_market_regime(self):
        """Test market regime formatting."""
        formatter = WebSearchFormatter()

        # Mock market regime
        regime = type(
            "MockRegime",
            (),
            {
                "regime_type": type("MockType", (), {"value": "RISK_ON"})(),
                "confidence": 0.8,
                "key_drivers": ["Dovish Fed policy", "Strong earnings"],
                "fed_policy_stance": "DOVISH",
                "inflation_environment": "STABLE",
                "interest_rate_trend": "FALLING",
                "geopolitical_risk_level": "LOW",
                "market_volatility_regime": "NORMAL",
            },
        )()

        result = await formatter._format_market_regime(regime)

        assert "MARKET REGIME ANALYSIS" in result
        assert "RISK_ON" in result
        assert "0.80" in result
        assert "Dovish Fed policy" in result
        assert "DOVISH" in result

    @pytest.mark.asyncio()
    async def test_format_risk_sentiment(self):
        """Test risk sentiment formatting."""
        formatter = WebSearchFormatter()

        # Mock risk sentiment
        sentiment = type(
            "MockSentiment",
            (),
            {
                "sentiment_level": type("MockLevel", (), {"value": "GREED"})(),
                "fear_greed_index": 75.0,
                "volatility_expectation": 22.5,
                "market_stress_indicator": 0.3,
                "vix_level": 18.5,
                "crypto_fear_greed": 80.0,
                "news_sentiment_score": 0.4,
            },
        )()

        result = await formatter._format_risk_sentiment(sentiment)

        assert "RISK SENTIMENT ANALYSIS" in result
        assert "GREED" in result
        assert "75" in result  # Fear greed index
        assert "22.5" in result  # Volatility expectation
        assert "18.5" in result  # VIX level

    @pytest.mark.asyncio()
    async def test_format_momentum_alignment(self):
        """Test momentum alignment formatting."""
        formatter = WebSearchFormatter()

        # Mock momentum alignment
        alignment = type(
            "MockAlignment",
            (),
            {
                "directional_alignment": 0.6,
                "strength_alignment": 0.7,
                "crypto_momentum_score": 0.5,
                "nasdaq_momentum_score": 0.4,
                "momentum_divergences": ["Crypto outperforming NASDAQ"],
                "momentum_regime": "ACCELERATION",
                "cross_asset_momentum_flow": "CRYPTO_OUTPERFORMING",
            },
        )()

        result = await formatter._format_momentum_alignment(alignment)

        assert "MOMENTUM ALIGNMENT ANALYSIS" in result
        assert "+0.60" in result  # Directional alignment
        assert "0.70" in result  # Strength alignment
        assert "Crypto outperforming NASDAQ" in result
        assert "ACCELERATION" in result

    def test_generate_context_summary_header(self):
        """Test context summary header generation."""
        formatter = WebSearchFormatter()

        context = {
            "news_results": [],
            "sentiment_result": {},
            "correlation_analysis": {},
            "market_regime": {},
        }

        header = formatter._generate_context_summary_header(context)

        assert "COMPREHENSIVE MARKET CONTEXT ANALYSIS" in header
        assert "Analysis Time" in header
        assert "Data Sources" in header
        assert "📰 News" in header
        assert "📊 Sentiment" in header
        assert "🔗 Correlation" in header
        assert "🌐 Regime" in header


# Fixtures for testing
@pytest.fixture()
def sample_news_items():
    """Sample news items for testing."""
    return [
        {
            "title": "Bitcoin Breaks $60K Resistance Level",
            "content": """
            Bitcoin has successfully broken above the critical $60,000 resistance level
            with strong bullish momentum. Technical indicators show RSI entering
            overbought territory while volume surge confirms the breakout.
            Institutional buying pressure continues to support the rally.
            """,
            "url": "https://bloomberg.com/bitcoin-breakout",
            "published_time": datetime.now(UTC),
        },
        {
            "title": "Ethereum DeFi TVL Reaches New High",
            "content": """
            Ethereum's DeFi ecosystem has reached a new total value locked milestone,
            showing strong adoption momentum. Smart contract activity increases
            while gas fees remain manageable. Institutional DeFi adoption accelerates.
            """,
            "url": "https://coindesk.com/ethereum-defi",
            "published_time": datetime.now(UTC) - timedelta(hours=2),
        },
        {
            "title": "Regulatory Clarity Expected Soon",
            "content": """
            Market expects regulatory clarity on cryptocurrency classification.
            SEC signals potential framework announcement. Industry awaits
            guidance on compliance requirements.
            """,
            "url": "https://reuters.com/crypto-regulation",
            "published_time": datetime.now(UTC) - timedelta(hours=4),
        },
    ]


@pytest.fixture()
def sample_sentiment_result():
    """Sample sentiment result for testing."""
    return SentimentResult(
        sentiment_score=0.65,
        confidence=0.82,
        key_themes=["Bitcoin", "Institutional", "Breakout", "DeFi"],
        bullish_indicators=[
            "Strong institutional buying pressure",
            "Technical breakout confirmed",
            "Volume surge supports rally",
        ],
        bearish_indicators=["Regulatory uncertainty persists"],
        volatility_signals=[
            "RSI entering overbought territory",
            "Options volatility elevated",
        ],
    )


@pytest.fixture()
def sample_correlation_analysis():
    """Sample correlation analysis for testing."""
    return CorrelationAnalysis(
        correlation_coefficient=0.73,
        correlation_strength=type("MockStrength", (), {"value": "STRONG"})(),
        direction="POSITIVE",
        p_value=0.005,
        is_significant=True,
        sample_size=250,
        rolling_correlation_24h=0.68,
        rolling_correlation_7d=0.75,
        correlation_stability=0.82,
        regime_dependent_correlation={
            "HIGH_VOLATILITY": 0.85,
            "LOW_VOLATILITY": 0.62,
            "RISK_ON": 0.78,
            "RISK_OFF": 0.45,
        },
        reliability_score=0.88,
    )


class TestWebSearchFormatterIntegration:
    """Integration tests for WebSearchFormatter."""

    @pytest.mark.asyncio()
    async def test_complete_news_formatting_workflow(self, sample_news_items):
        """Test complete news formatting workflow."""
        formatter = WebSearchFormatter()

        result = await formatter.format_news_results(sample_news_items)

        # Validate structure
        assert "📰 **NEWS ANALYSIS**" in result
        assert "🎯 **Overall News Sentiment**" in result
        assert "📊 **Articles Analyzed**: 3" in result
        assert "🔍 **Key Market Insights**" in result
        assert "📈 **Trading Signals**" in result
        assert "📑 **Recent High-Priority Articles**" in result

        # Validate content
        assert "Bitcoin" in result
        assert "Ethereum" in result
        assert "regulatory" in result.lower() or "Regulatory" in result
        assert "breakout" in result.lower() or "Breakout" in result

    @pytest.mark.asyncio()
    async def test_complete_sentiment_formatting_workflow(
        self, sample_sentiment_result
    ):
        """Test complete sentiment formatting workflow."""
        formatter = WebSearchFormatter()

        result = await formatter.format_sentiment_data(sample_sentiment_result)

        # Validate structure
        assert "📊 **MARKET SENTIMENT ANALYSIS**" in result
        assert "🎯 **Overall Sentiment**" in result
        assert "📈 **Sentiment Score**" in result
        assert "🔍 **Key Market Themes**" in result
        assert "🟢 **Bullish Signals**" in result
        assert "🔴 **Bearish Signals**" in result
        assert "⚡ **Volatility Indicators**" in result

        # Validate content
        assert "0.65" in result  # Sentiment score
        assert "0.82" in result  # Confidence
        assert "BULLISH" in result
        assert "Bitcoin" in result
        assert "Institutional" in result

    @pytest.mark.asyncio()
    async def test_complete_correlation_formatting_workflow(
        self, sample_correlation_analysis
    ):
        """Test complete correlation formatting workflow."""
        formatter = WebSearchFormatter()

        result = await formatter.format_correlation_analysis(
            sample_correlation_analysis
        )

        # Validate structure
        assert "🔗 **CRYPTO-NASDAQ CORRELATION ANALYSIS**" in result
        assert "**Correlation**" in result
        assert "**Direction**" in result
        assert "**Statistical Significance**" in result
        assert "**Sample Size**" in result
        assert "**Reliability Score**" in result
        assert "⏰ **Rolling Correlations**" in result
        assert "🌐 **Regime-Dependent Correlations**" in result
        assert "💡 **Trading Implications**" in result

        # Validate content
        assert "0.730" in result  # Correlation coefficient
        assert "STRONG" in result
        assert "POSITIVE" in result
        assert "250" in result  # Sample size
        assert "HIGH_VOLATILITY" in result
        assert "LOW_VOLATILITY" in result

    @pytest.mark.asyncio()
    async def test_comprehensive_market_context_workflow(
        self, sample_news_items, sample_sentiment_result, sample_correlation_analysis
    ):
        """Test comprehensive market context formatting workflow."""
        formatter = WebSearchFormatter()

        comprehensive_context = {
            "news_results": sample_news_items,
            "sentiment_result": sample_sentiment_result,
            "correlation_analysis": sample_correlation_analysis,
        }

        result = await formatter.format_market_context(comprehensive_context)

        # Validate overall structure
        assert "🌐 **COMPREHENSIVE MARKET CONTEXT ANALYSIS**" in result
        assert "⏰ **Analysis Time**" in result
        assert "📊 **Data Sources**" in result
        assert "🎯 **Optimized for**: AI Trading Decision Making" in result

        # Validate all sections are included
        assert "📰 **NEWS ANALYSIS**" in result
        assert "📊 **MARKET SENTIMENT ANALYSIS**" in result
        assert "🔗 **CRYPTO-NASDAQ CORRELATION ANALYSIS**" in result

        # Validate content integration
        assert "Bitcoin" in result
        assert "0.65" in result  # Sentiment score
        assert "0.730" in result  # Correlation coefficient
        assert len(result) > 1000  # Should be comprehensive

    @pytest.mark.asyncio()
    async def test_key_insights_extraction_workflow(self):
        """Test complete key insights extraction workflow."""
        formatter = WebSearchFormatter()

        comprehensive_search_results = {
            "news_items": [
                {
                    "title": "Bitcoin Institutional Breakout Above $60K",
                    "content": "Technical breakout confirmed with massive institutional buying volume",
                },
                {
                    "title": "Ethereum Smart Contract Adoption Surges",
                    "content": "DeFi protocols show strong growth with developer activity increasing",
                },
            ],
            "sentiment_data": {
                "sentiment_score": 0.8,
                "confidence": 0.9,
                "volatility_signals": ["High options activity", "VIX spike"],
            },
            "price_data": {
                "price_change_24h": 0.12,  # 12% gain
                "volume_change_24h": 0.45,  # 45% volume increase
            },
            "technical_analysis": {
                "rsi": 78,  # Overbought
                "trend_direction": "BULLISH",
                "support_levels": [55000, 58000],
                "resistance_levels": [65000, 68000],
            },
        }

        insights = await formatter.extract_key_insights(comprehensive_search_results)

        # Validate insights quality
        assert len(insights) > 0
        assert len(insights) <= 10  # Should be limited to top insights

        # Validate insight content
        insight_text = " ".join(insights).lower()
        assert "breakout" in insight_text or "institutional" in insight_text
        assert "bullish" in insight_text or "strong" in insight_text

        # Validate prioritization (first insight should be high priority)
        assert any(
            keyword in insights[0].lower()
            for keyword in ["breakout", "institutional", "surge", "significant"]
        )

    @pytest.mark.asyncio()
    async def test_token_optimization_workflow(self):
        """Test token optimization workflow with large content."""
        formatter = WebSearchFormatter(max_total_tokens=100)  # Small limit for testing

        # Generate large context
        large_news_items = [
            {
                "title": f"Large News Article {i}",
                "content": f"This is a very long news article with extensive content that goes on and on about market conditions and trading implications. Article number {i} contains detailed analysis and comprehensive market insights that would normally be very valuable for trading decisions."
                * 5,
                "url": f"https://example.com/article-{i}",
                "published_time": datetime.now(UTC),
            }
            for i in range(10)
        ]

        result = await formatter.format_news_results(large_news_items)

        # Should be optimized for token limits
        estimated_tokens = formatter._estimate_token_count(result)
        assert (
            estimated_tokens <= formatter.max_total_tokens * 1.1
        )  # Allow small margin

        # Should still contain essential information
        assert "NEWS ANALYSIS" in result
        assert "Articles Analyzed: 10" in result

    @pytest.mark.asyncio()
    async def test_error_resilience_workflow(self):
        """Test error resilience across formatting workflows."""
        formatter = WebSearchFormatter()

        # Test with various invalid inputs
        invalid_news = [
            {"title": None, "content": None},
            {"invalid_key": "invalid_value"},
            {},
        ]

        invalid_sentiment = type(
            "InvalidSentiment",
            (),
            {"sentiment_score": "invalid", "confidence": None, "key_themes": None},
        )()

        invalid_correlation = type(
            "InvalidCorrelation",
            (),
            {
                "correlation_coefficient": "invalid",
                "correlation_strength": None,
                "direction": None,
            },
        )()

        # All methods should handle errors gracefully
        news_result = await formatter.format_news_results(invalid_news)
        assert isinstance(news_result, str)
        assert "NEWS ANALYSIS" in news_result

        sentiment_result = await formatter.format_sentiment_data(invalid_sentiment)
        assert isinstance(sentiment_result, str)
        assert "Error processing sentiment data" in sentiment_result

        correlation_result = await formatter.format_correlation_analysis(
            invalid_correlation
        )
        assert isinstance(correlation_result, str)
        assert "Error processing correlation data" in correlation_result

    @pytest.mark.asyncio()
    async def test_performance_benchmark(self):
        """Test performance with realistic data volumes."""
        formatter = WebSearchFormatter()

        # Generate realistic volume of news items
        large_news_dataset = [
            {
                "title": f"Market Update {i}: Bitcoin and Crypto Analysis",
                "content": f"""
                Comprehensive market analysis for day {i} shows various indicators
                suggesting bullish/bearish momentum. Technical analysis reveals
                support at {45000 + i * 100} with resistance near {55000 + i * 150}.
                Volume indicators show {'increasing' if i % 2 == 0 else 'decreasing'} activity
                with institutional sentiment remaining {'positive' if i % 3 == 0 else 'neutral'}.
                """,
                "url": f"https://tradinganalysis.com/update-{i}",
                "published_time": datetime.now(UTC) - timedelta(hours=i),
            }
            for i in range(50)  # 50 news items
        ]

        import time

        start_time = time.time()

        result = await formatter.format_news_results(large_news_dataset)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should process reasonable volume efficiently (under 5 seconds)
        assert processing_time < 5.0

        # Should produce valid output
        assert isinstance(result, str)
        assert "NEWS ANALYSIS" in result
        assert len(result) > 500  # Should have substantial content

        # Should respect token limits
        estimated_tokens = formatter._estimate_token_count(result)
        assert estimated_tokens <= formatter.max_total_tokens * 1.2  # Allow some margin
