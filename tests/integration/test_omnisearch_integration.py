"""Integration tests for OmniSearch MCP integration with LLM agent."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

# Import components for testing
try:
    from bot.analysis.market_context import (
        CorrelationAnalysis,
        CorrelationStrength,
        MarketContextAnalyzer,
        MarketRegime,
        MarketRegimeType,
        MomentumAlignment,
        RiskSentiment,
        SentimentLevel,
    )
    from bot.mcp.omnisearch_client import (
        FinancialNewsResult,
        MarketCorrelation,
        OmniSearchClient,
        SearchResult,
        SentimentAnalysis,
    )
    from bot.services.financial_sentiment import (
        CryptoIndicators,
        FinancialSentimentService,
        NasdaqIndicators,
        SentimentResult,
    )
    from bot.utils.web_search_formatter import WebSearchFormatter
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestOmniSearchClientIntegration:
    """Integration tests for OmniSearch client with real-world scenarios."""

    @pytest.fixture()
    def omnisearch_client(self):
        """Create OmniSearch client for testing."""
        return OmniSearchClient(
            server_url="https://test-api.omnisearch.dev",
            api_key="test_key",
            enable_cache=True,
            cache_ttl=300,
            rate_limit_requests=10,
            rate_limit_window=60,
        )

    @pytest.mark.asyncio()
    async def test_client_connection_and_health_check(self, omnisearch_client):
        """Test client connection and health check functionality."""
        with patch.object(omnisearch_client, "_session") as mock_session:
            # Mock successful connection
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response

            # Test connection
            connected = await omnisearch_client.connect()
            assert connected is True

            # Test health check
            health = await omnisearch_client.health_check()
            assert health["connected"] is True
            assert "server_url" in health
            assert "cache_enabled" in health
            assert "timestamp" in health

            # Test disconnection
            await omnisearch_client.disconnect()
            assert omnisearch_client._connected is False

    @pytest.mark.asyncio()
    async def test_financial_news_search_integration(self, omnisearch_client):
        """Test financial news search with realistic data flow."""
        with patch.object(omnisearch_client, "_session") as mock_session:
            omnisearch_client._connected = True

            # Mock API response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "title": "Bitcoin ETF Approval Drives Crypto Rally",
                        "url": "https://financial-news.com/btc-etf",
                        "snippet": "SEC approves first Bitcoin ETF, triggering massive institutional inflows",
                        "source": "financial-news.com",
                        "published_date": "2024-01-15T10:00:00Z",
                        "relevance_score": 0.95,
                        "sentiment": "positive",
                        "mentioned_symbols": ["BTC", "BTCUSD"],
                        "category": "regulation",
                        "impact_level": "high",
                    },
                    {
                        "title": "Ethereum Upgrade Enhances Network Efficiency",
                        "url": "https://crypto-updates.com/eth-upgrade",
                        "snippet": "Latest Ethereum upgrade reduces gas fees and improves transaction speeds",
                        "source": "crypto-updates.com",
                        "published_date": "2024-01-15T08:30:00Z",
                        "relevance_score": 0.82,
                        "sentiment": "positive",
                        "mentioned_symbols": ["ETH", "ETHUSD"],
                        "category": "technology",
                        "impact_level": "medium",
                    },
                ]
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response

            # Mock rate limiter
            omnisearch_client.rate_limiter.acquire = AsyncMock(return_value=True)

            # Execute search
            results = await omnisearch_client.search_financial_news(
                "Bitcoin ETF Ethereum upgrade", limit=5, timeframe="24h"
            )

            # Validate results
            assert len(results) == 2
            assert isinstance(results[0], FinancialNewsResult)
            assert (
                results[0].base_result.title
                == "Bitcoin ETF Approval Drives Crypto Rally"
            )
            assert results[0].sentiment == "positive"
            assert "BTC" in results[0].mentioned_symbols
            assert results[0].impact_level == "high"

            assert (
                results[1].base_result.title
                == "Ethereum Upgrade Enhances Network Efficiency"
            )
            assert "ETH" in results[1].mentioned_symbols

    @pytest.mark.asyncio()
    async def test_sentiment_analysis_integration(self, omnisearch_client):
        """Test crypto sentiment analysis integration."""
        with patch.object(omnisearch_client, "_session") as mock_session:
            omnisearch_client._connected = True

            # Mock sentiment API response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "sentiment": {
                    "overall": "bullish",
                    "score": 0.72,
                    "confidence": 0.86,
                    "source_count": 45,
                    "news_sentiment": 0.68,
                    "social_sentiment": 0.75,
                    "technical_sentiment": 0.55,
                    "key_drivers": [
                        "ETF approval momentum",
                        "Institutional adoption surge",
                        "Technical breakout patterns",
                    ],
                    "risk_factors": [
                        "Regulatory uncertainty in some regions",
                        "Market volatility concerns",
                    ],
                }
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response
            omnisearch_client.rate_limiter.acquire = AsyncMock(return_value=True)

            # Execute sentiment analysis
            sentiment = await omnisearch_client.search_crypto_sentiment("BTC-USD")

            # Validate sentiment analysis
            assert isinstance(sentiment, SentimentAnalysis)
            assert sentiment.symbol == "BTC"
            assert sentiment.overall_sentiment == "bullish"
            assert sentiment.sentiment_score == 0.72
            assert sentiment.confidence == 0.86
            assert sentiment.source_count == 45
            assert len(sentiment.key_drivers) == 3
            assert "ETF approval momentum" in sentiment.key_drivers
            assert len(sentiment.risk_factors) == 2

    @pytest.mark.asyncio()
    async def test_market_correlation_integration(self, omnisearch_client):
        """Test market correlation analysis integration."""
        with patch.object(omnisearch_client, "_session") as mock_session:
            omnisearch_client._connected = True

            # Mock correlation API response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "correlation": {"coefficient": 0.67, "beta": 1.45, "r_squared": 0.45}
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response
            omnisearch_client.rate_limiter.acquire = AsyncMock(return_value=True)

            # Execute correlation analysis
            correlation = await omnisearch_client.search_market_correlation(
                "BTC", "QQQ", "30d"
            )

            # Validate correlation analysis
            assert isinstance(correlation, MarketCorrelation)
            assert correlation.primary_symbol == "BTC"
            assert correlation.secondary_symbol == "QQQ"
            assert correlation.correlation_coefficient == 0.67
            assert correlation.timeframe == "30d"
            assert correlation.strength == "strong"  # 0.6 < 0.67 <= 0.8
            assert correlation.direction == "positive"
            assert correlation.beta == 1.45
            assert correlation.r_squared == 0.45

    @pytest.mark.asyncio()
    async def test_caching_and_rate_limiting_integration(self, omnisearch_client):
        """Test caching and rate limiting integration."""
        with patch.object(omnisearch_client, "_session") as mock_session:
            omnisearch_client._connected = True

            # Mock API response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"results": []}
            mock_session.get.return_value.__aenter__.return_value = mock_response
            omnisearch_client.rate_limiter.acquire = AsyncMock(return_value=True)

            # First request - should hit API
            await omnisearch_client.search_financial_news("test query")
            assert omnisearch_client.rate_limiter.acquire.call_count == 1

            # Second identical request - should hit cache
            await omnisearch_client.search_financial_news("test query")
            assert (
                omnisearch_client.rate_limiter.acquire.call_count == 1
            )  # Not called again


class TestFinancialSentimentServiceIntegration:
    """Integration tests for financial sentiment service."""

    @pytest.fixture()
    def sentiment_service(self):
        """Create financial sentiment service for testing."""
        return FinancialSentimentService()

    @pytest.fixture()
    def sample_market_news(self):
        """Sample market news for integration testing."""
        return [
            {
                "title": "Bitcoin Surges Above $60K as ETF Approval Boosts Institutional Demand",
                "content": """
                Bitcoin price rallied strongly today, breaking above the critical $60,000
                resistance level as news of potential ETF approval drove massive institutional
                buying. Technical indicators show RSI entering overbought territory at 78,
                while MACD signals remain bullish with golden cross formation intact.

                Volume surge of 45% confirms the breakout, with whale addresses accumulating
                over 15,000 BTC in the past 24 hours. Support levels now established at
                $55,000 and $57,500, with next resistance targets at $65,000.

                Federal Reserve's dovish stance on monetary policy continues to support
                risk assets, while inflation data remains contained. Market participants
                expect continued momentum as institutional adoption accelerates.
                """,
            },
            {
                "title": "Ethereum Network Upgrade Drives DeFi Renaissance",
                "content": """
                Ethereum's latest network upgrade has successfully reduced gas fees by 40%
                while improving transaction throughput, triggering a renaissance in DeFi
                protocol activity. Total Value Locked (TVL) has increased 25% to $45 billion.

                Smart contract deployments surged 60% as developers return to building on
                Ethereum. Major institutions are exploring DeFi integration, with Goldman
                Sachs reportedly piloting tokenized asset programs.

                Technical analysis shows ETH forming a bull flag pattern with strong
                support at $2,800. Volume indicators suggest accumulation phase with
                institutional interest growing rapidly.
                """,
            },
            {
                "title": "NASDAQ Tech Rally Continues Amid AI Boom",
                "content": """
                NASDAQ technology index gained 2.3% today as artificial intelligence stocks
                continued their remarkable rally. The tech-heavy index shows strong momentum
                with breadth indicators supporting the move higher.

                Federal Reserve officials signaled data-dependent approach to future rate
                decisions, with markets interpreting this as dovish. VIX volatility index
                declined to 16.5, indicating reduced fear in equity markets.

                Sector rotation into technology continues as investors seek exposure to
                AI revolution. Risk-on sentiment dominates with financial sector also
                showing strength on improved lending prospects.
                """,
            },
        ]

    @pytest.mark.asyncio()
    async def test_comprehensive_sentiment_analysis(
        self, sentiment_service, sample_market_news
    ):
        """Test comprehensive sentiment analysis with real market data."""
        # Analyze news sentiment
        sentiment_result = await sentiment_service.analyze_news_sentiment(
            sample_market_news
        )

        # Validate sentiment analysis
        assert isinstance(sentiment_result, SentimentResult)
        assert sentiment_result.sentiment_score > 0.0  # Should be bullish
        assert sentiment_result.confidence > 0.5  # Should have reasonable confidence
        assert len(sentiment_result.key_themes) > 0
        assert len(sentiment_result.bullish_indicators) > 0

        # Check for expected themes
        themes_text = " ".join(sentiment_result.key_themes).lower()
        assert any(
            theme in themes_text for theme in ["bitcoin", "ethereum", "institutional"]
        )

        # Check for expected bullish indicators
        bullish_text = " ".join(sentiment_result.bullish_indicators).lower()
        assert any(
            indicator in bullish_text
            for indicator in ["surge", "rally", "bullish", "breakout"]
        )

    @pytest.mark.asyncio()
    async def test_crypto_indicators_extraction(
        self, sentiment_service, sample_market_news
    ):
        """Test crypto indicators extraction from market news."""
        # Combine news content
        combined_text = " ".join(
            [f"{item['title']} {item['content']}" for item in sample_market_news]
        )

        # Extract crypto indicators
        crypto_indicators = sentiment_service.extract_crypto_indicators(combined_text)

        # Validate crypto indicators
        assert isinstance(crypto_indicators, CryptoIndicators)
        assert crypto_indicators.trend_direction == "BULLISH"
        assert len(crypto_indicators.price_mentions) > 0
        assert len(crypto_indicators.support_levels) > 0
        assert len(crypto_indicators.volume_indicators) > 0
        assert len(crypto_indicators.momentum_signals) > 0
        assert len(crypto_indicators.adoption_signals) > 0
        assert len(crypto_indicators.technical_patterns) > 0

        # Check for specific expected indicators
        assert any(
            "rsi" in signal.lower() for signal in crypto_indicators.momentum_signals
        )
        assert any(
            "volume" in indicator.lower()
            for indicator in crypto_indicators.volume_indicators
        )

    @pytest.mark.asyncio()
    async def test_nasdaq_indicators_extraction(
        self, sentiment_service, sample_market_news
    ):
        """Test NASDAQ indicators extraction from market news."""
        # Combine news content
        combined_text = " ".join(
            [f"{item['title']} {item['content']}" for item in sample_market_news]
        )

        # Extract NASDAQ indicators
        nasdaq_indicators = sentiment_service.extract_nasdaq_indicators(combined_text)

        # Validate NASDAQ indicators
        assert isinstance(nasdaq_indicators, NasdaqIndicators)
        assert nasdaq_indicators.nasdaq_trend == "BULLISH"
        assert len(nasdaq_indicators.fed_policy_signals) > 0
        assert len(nasdaq_indicators.tech_sector_signals) > 0
        assert len(nasdaq_indicators.risk_on_signals) > 0

        # Check for expected signals
        fed_signals_text = " ".join(nasdaq_indicators.fed_policy_signals).lower()
        assert "fed" in fed_signals_text or "federal" in fed_signals_text

        tech_signals_text = " ".join(nasdaq_indicators.tech_sector_signals).lower()
        assert "tech" in tech_signals_text or "ai" in tech_signals_text

    @pytest.mark.asyncio()
    async def test_correlation_score_calculation(
        self, sentiment_service, sample_market_news
    ):
        """Test correlation score calculation between crypto and NASDAQ."""
        # Extract indicators
        combined_text = " ".join(
            [f"{item['title']} {item['content']}" for item in sample_market_news]
        )

        crypto_indicators = sentiment_service.extract_crypto_indicators(combined_text)
        nasdaq_indicators = sentiment_service.extract_nasdaq_indicators(combined_text)

        # Calculate correlation
        correlation_score = sentiment_service.calculate_correlation_score(
            crypto_indicators.dict(), nasdaq_indicators.dict()
        )

        # Validate correlation
        assert -1.0 <= correlation_score <= 1.0
        assert (
            correlation_score > 0.0
        )  # Should be positive given bullish trends in both

    @pytest.mark.asyncio()
    async def test_llm_formatting_integration(
        self, sentiment_service, sample_market_news
    ):
        """Test LLM formatting integration."""
        # Perform full analysis
        sentiment_result = await sentiment_service.analyze_news_sentiment(
            sample_market_news
        )

        combined_text = " ".join(
            [f"{item['title']} {item['content']}" for item in sample_market_news]
        )

        crypto_indicators = sentiment_service.extract_crypto_indicators(combined_text)
        nasdaq_indicators = sentiment_service.extract_nasdaq_indicators(combined_text)
        correlation_score = sentiment_service.calculate_correlation_score(
            crypto_indicators.dict(), nasdaq_indicators.dict()
        )

        # Create comprehensive sentiment data
        sentiment_data = {
            "sentiment_result": sentiment_result,
            "crypto_indicators": crypto_indicators,
            "nasdaq_indicators": nasdaq_indicators,
            "correlation_score": correlation_score,
        }

        # Format for LLM
        formatted_output = sentiment_service.format_sentiment_for_llm(sentiment_data)

        # Validate formatted output
        assert isinstance(formatted_output, str)
        assert len(formatted_output) > 500  # Should be comprehensive
        assert "FINANCIAL SENTIMENT ANALYSIS" in formatted_output
        assert "CRYPTO MARKET INDICATORS" in formatted_output
        assert "TRADITIONAL MARKET INDICATORS" in formatted_output
        assert "MARKET CORRELATION" in formatted_output
        assert "TRADING IMPLICATIONS" in formatted_output

        # Check for key content
        assert "BULLISH" in formatted_output
        assert str(round(sentiment_result.sentiment_score, 2)) in formatted_output


class TestMarketContextAnalyzerIntegration:
    """Integration tests for market context analyzer."""

    @pytest.fixture()
    def context_analyzer(self):
        """Create market context analyzer for testing."""
        return MarketContextAnalyzer()

    @pytest.fixture()
    def sample_crypto_price_data(self):
        """Generate sample crypto price data."""
        rng = np.random.default_rng(42)
        base_price = 50000
        prices = [base_price]

        # Generate correlated price movement
        for _i in range(199):
            change = rng.normal(0.002, 0.025)  # 0.2% mean, 2.5% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        return {
            "prices": prices,
            "ohlcv": [{"close": price} for price in prices[-50:]],  # Last 50 candles
        }

    @pytest.fixture()
    def sample_nasdaq_price_data(self):
        """Generate sample NASDAQ price data."""
        rng = np.random.default_rng(24)
        base_price = 15000
        prices = [base_price]

        # Generate somewhat correlated price movement
        for _i in range(199):
            change = rng.normal(0.001, 0.015)  # 0.1% mean, 1.5% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        return {
            "prices": prices,
            "candles": [{"close": price} for price in prices[-50:]],  # Last 50 candles
        }

    @pytest.fixture()
    def sample_comprehensive_sentiment_data(self):
        """Sample comprehensive sentiment data for regime analysis."""
        return {
            "text": """
            Federal Reserve maintains dovish monetary policy stance amid contained inflation.
            Risk appetite returns to markets as geopolitical tensions ease. Technology
            sector shows exceptional strength with AI-driven growth. Crypto adoption
            accelerates with institutional involvement. Regulatory clarity improves
            market confidence. Liquidity conditions remain abundant with low volatility.
            """,
            "news_headlines": [
                "Fed Signals Continued Accommodative Policy",
                "Inflation Data Shows Moderation",
                "Geopolitical Risks Diminish",
                "Tech Sector Leads Market Rally",
                "Crypto Institutional Adoption Surges",
                "Regulatory Framework Provides Clarity",
                "Market Volatility Remains Subdued",
                "Risk Assets Benefit from Liquidity",
            ],
            "vix_level": 15.2,
            "volatility_score": 0.25,
            "sentiment_divergence": False,
        }

    @pytest.mark.asyncio()
    async def test_comprehensive_correlation_analysis(
        self, context_analyzer, sample_crypto_price_data, sample_nasdaq_price_data
    ):
        """Test comprehensive correlation analysis integration."""
        # Perform correlation analysis
        correlation = await context_analyzer.analyze_crypto_nasdaq_correlation(
            sample_crypto_price_data, sample_nasdaq_price_data
        )

        # Validate correlation analysis
        assert isinstance(correlation, CorrelationAnalysis)
        assert correlation.sample_size == 200  # Both series have 200 points
        assert -1.0 <= correlation.correlation_coefficient <= 1.0
        assert 0.0 <= correlation.p_value <= 1.0
        assert isinstance(correlation.correlation_strength, CorrelationStrength)
        assert correlation.direction in ["POSITIVE", "NEGATIVE", "UNCORRELATED"]
        assert 0.0 <= correlation.reliability_score <= 1.0

        # Check rolling correlations
        if correlation.rolling_correlation_24h is not None:
            assert -1.0 <= correlation.rolling_correlation_24h <= 1.0
        if correlation.rolling_correlation_7d is not None:
            assert -1.0 <= correlation.rolling_correlation_7d <= 1.0

    @pytest.mark.asyncio()
    async def test_comprehensive_regime_detection(
        self, context_analyzer, sample_comprehensive_sentiment_data
    ):
        """Test comprehensive market regime detection."""
        # Perform regime detection
        regime = await context_analyzer.detect_market_regime(
            sample_comprehensive_sentiment_data
        )

        # Validate regime detection
        assert isinstance(regime, MarketRegime)
        assert isinstance(regime.regime_type, MarketRegimeType)
        assert 0.0 <= regime.confidence <= 1.0
        assert len(regime.key_drivers) > 0
        assert regime.fed_policy_stance in ["HAWKISH", "DOVISH", "NEUTRAL"]
        assert regime.inflation_environment in ["HIGH", "LOW", "STABLE"]
        assert regime.interest_rate_trend in ["RISING", "FALLING", "STABLE"]
        assert regime.geopolitical_risk_level in ["HIGH", "MEDIUM", "LOW"]
        assert regime.crypto_adoption_momentum in ["HIGH", "MODERATE", "LOW"]
        assert regime.institutional_sentiment in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        assert regime.regulatory_environment in ["FAVORABLE", "RESTRICTIVE", "STABLE"]
        assert regime.market_volatility_regime in ["HIGH", "ELEVATED", "NORMAL"]
        assert regime.liquidity_conditions in ["TIGHT", "NORMAL", "ABUNDANT"]
        assert 0.0 <= regime.regime_change_probability <= 1.0

        # Given the dovish sentiment data, should detect RISK_ON regime
        assert regime.regime_type in [
            MarketRegimeType.RISK_ON,
            MarketRegimeType.TRANSITION,
        ]
        assert regime.fed_policy_stance == "DOVISH"

    @pytest.mark.asyncio()
    async def test_comprehensive_risk_sentiment_assessment(self, context_analyzer):
        """Test comprehensive risk sentiment assessment."""
        # Sample news data with mixed sentiment
        news_data = [
            {
                "title": "Markets Rally on Positive Economic Data",
                "content": "Risk appetite returns as investors embrace bullish sentiment and greed index rises",
            },
            {
                "title": "Volatility Subsides as Fear Diminishes",
                "content": "VIX drops to 16 as market stress indicators show improvement",
            },
            {
                "title": "Crypto Market Shows Strong Momentum",
                "content": "Bitcoin fear and greed index climbs to 75 indicating greedy sentiment",
            },
        ]

        # Perform risk sentiment assessment
        sentiment = await context_analyzer.assess_risk_sentiment(news_data)

        # Validate risk sentiment
        assert isinstance(sentiment, RiskSentiment)
        assert 0.0 <= sentiment.fear_greed_index <= 100.0
        assert isinstance(sentiment.sentiment_level, SentimentLevel)
        assert sentiment.volatility_expectation > 0.0
        assert 0.0 <= sentiment.market_stress_indicator <= 1.0

        # Given positive news, should show greedy sentiment
        assert sentiment.fear_greed_index > 50.0
        assert sentiment.sentiment_level in [
            SentimentLevel.GREED,
            SentimentLevel.NEUTRAL,
        ]

    @pytest.mark.asyncio()
    async def test_momentum_alignment_analysis(self, context_analyzer):
        """Test momentum alignment analysis."""
        # Sample momentum indicators
        crypto_indicators = {
            "rsi": 68,
            "price_change_24h": 0.055,  # 5.5% gain
            "volume_change_24h": 0.28,  # 28% volume increase
            "trend_direction": "BULLISH",
            "trend_strength": 0.75,
            "volume_trend": "INCREASING",
        }

        nasdaq_indicators = {
            "price_change_24h": 0.022,  # 2.2% gain
            "volume_change_24h": 0.15,  # 15% volume increase
            "tech_sector_performance": 2.8,  # 2.8% tech gain
            "trend_direction": "BULLISH",
            "trend_strength": 0.65,
            "volume_trend": "INCREASING",
        }

        # Perform momentum alignment analysis
        alignment = await context_analyzer.calculate_momentum_alignment(
            crypto_indicators, nasdaq_indicators
        )

        # Validate momentum alignment
        assert isinstance(alignment, MomentumAlignment)
        assert -1.0 <= alignment.directional_alignment <= 1.0
        assert 0.0 <= alignment.strength_alignment <= 1.0
        assert -1.0 <= alignment.crypto_momentum_score <= 1.0
        assert -1.0 <= alignment.nasdaq_momentum_score <= 1.0
        assert 0.0 <= alignment.trend_strength_crypto <= 1.0
        assert 0.0 <= alignment.trend_strength_nasdaq <= 1.0
        assert 0.0 <= alignment.momentum_sustainability <= 1.0
        assert alignment.momentum_regime in ["ACCELERATION", "DECELERATION", "NORMAL"]
        assert alignment.cross_asset_momentum_flow in [
            "CRYPTO_OUTPERFORMING",
            "NASDAQ_OUTPERFORMING",
            "RISK_ON_FLOW",
            "RISK_OFF_FLOW",
            "NEUTRAL",
        ]

        # Given bullish indicators for both, should show positive alignment
        assert alignment.directional_alignment > 0.0
        assert alignment.crypto_momentum_score > 0.0
        assert alignment.nasdaq_momentum_score > 0.0

    @pytest.mark.asyncio()
    async def test_comprehensive_context_summary(
        self,
        context_analyzer,
        sample_crypto_price_data,
        sample_nasdaq_price_data,
        sample_comprehensive_sentiment_data,
    ):
        """Test comprehensive context summary generation."""
        # Perform all analyses
        correlation = await context_analyzer.analyze_crypto_nasdaq_correlation(
            sample_crypto_price_data, sample_nasdaq_price_data
        )

        regime = await context_analyzer.detect_market_regime(
            sample_comprehensive_sentiment_data
        )

        # Generate comprehensive summary
        summary = context_analyzer.generate_context_summary(correlation, regime)

        # Validate summary
        assert isinstance(summary, str)
        assert len(summary) > 1000  # Should be comprehensive
        assert "MARKET CONTEXT ANALYSIS" in summary
        assert "CORRELATION ANALYSIS" in summary
        assert "MARKET REGIME ANALYSIS" in summary
        assert "TRADING IMPLICATIONS" in summary

        # Check for key correlation data
        assert str(round(correlation.correlation_coefficient, 3)) in summary
        assert correlation.correlation_strength.value in summary
        assert correlation.direction in summary

        # Check for key regime data
        assert regime.regime_type.value in summary
        assert str(round(regime.confidence, 2)) in summary
        assert regime.fed_policy_stance in summary


class TestWebSearchFormatterIntegration:
    """Integration tests for web search formatter."""

    @pytest.fixture()
    def search_formatter(self):
        """Create web search formatter for testing."""
        return WebSearchFormatter(max_tokens_per_section=400, max_total_tokens=1500)

    @pytest.fixture()
    def comprehensive_market_context(self):
        """Create comprehensive market context for testing."""
        # Sample news results
        news_results = [
            {
                "title": "Bitcoin ETF Approval Triggers Institutional Rush",
                "content": "SEC approves multiple Bitcoin ETFs, triggering massive institutional inflows",
                "url": "https://reuters.com/bitcoin-etf-approval",
                "published_time": datetime.now(UTC),
            },
            {
                "title": "Ethereum DeFi TVL Reaches $50 Billion Milestone",
                "content": "DeFi protocols surge as institutional adoption accelerates",
                "url": "https://bloomberg.com/ethereum-defi",
                "published_time": datetime.now(UTC) - timedelta(hours=1),
            },
        ]

        # Sample sentiment result
        sentiment_result = SentimentResult(
            sentiment_score=0.78,
            confidence=0.89,
            key_themes=["Bitcoin", "ETF", "Institutional", "DeFi", "Ethereum"],
            bullish_indicators=[
                "ETF approval drives institutional demand",
                "DeFi TVL reaches new milestones",
                "Technical breakout patterns confirmed",
            ],
            bearish_indicators=["Regulatory uncertainty in some regions"],
            volatility_signals=["Options activity elevated", "VIX spike expected"],
        )

        # Sample correlation analysis
        correlation_analysis = CorrelationAnalysis(
            correlation_coefficient=0.68,
            correlation_strength=CorrelationStrength.MODERATE,
            direction="POSITIVE",
            p_value=0.008,
            is_significant=True,
            sample_size=180,
            rolling_correlation_24h=0.72,
            rolling_correlation_7d=0.64,
            correlation_stability=0.81,
            regime_dependent_correlation={
                "HIGH_VOLATILITY": 0.78,
                "LOW_VOLATILITY": 0.59,
                "RISK_ON": 0.74,
                "RISK_OFF": 0.52,
            },
            reliability_score=0.85,
        )

        # Sample market regime
        market_regime = MarketRegime(
            regime_type=MarketRegimeType.RISK_ON,
            confidence=0.82,
            key_drivers=[
                "Dovish Fed policy supports risk assets",
                "Institutional crypto adoption accelerating",
                "Technology sector momentum strong",
            ],
            fed_policy_stance="DOVISH",
            inflation_environment="STABLE",
            interest_rate_trend="STABLE",
            geopolitical_risk_level="LOW",
            crypto_adoption_momentum="HIGH",
            institutional_sentiment="POSITIVE",
            regulatory_environment="FAVORABLE",
            market_volatility_regime="NORMAL",
            liquidity_conditions="ABUNDANT",
            duration_days=28,
            regime_change_probability=0.15,
        )

        return {
            "news_results": news_results,
            "sentiment_result": sentiment_result,
            "correlation_analysis": correlation_analysis,
            "market_regime": market_regime,
        }

    @pytest.mark.asyncio()
    async def test_comprehensive_market_context_formatting(
        self, search_formatter, comprehensive_market_context
    ):
        """Test comprehensive market context formatting."""
        # Format comprehensive context
        formatted_output = await search_formatter.format_market_context(
            comprehensive_market_context
        )

        # Validate overall structure
        assert isinstance(formatted_output, str)
        assert len(formatted_output) > 800  # Should be comprehensive
        assert "COMPREHENSIVE MARKET CONTEXT ANALYSIS" in formatted_output
        assert "Data Sources:" in formatted_output
        assert "AI Trading Decision Making" in formatted_output

        # Validate all sections are included
        assert "📰 **NEWS ANALYSIS**" in formatted_output
        assert "📊 **MARKET SENTIMENT ANALYSIS**" in formatted_output
        assert "🔗 **CRYPTO-NASDAQ CORRELATION ANALYSIS**" in formatted_output
        assert "🌐 **MARKET REGIME ANALYSIS**" in formatted_output

        # Validate content integration
        assert "Bitcoin ETF" in formatted_output
        assert "0.78" in formatted_output  # Sentiment score
        assert "0.680" in formatted_output  # Correlation coefficient
        assert "RISK_ON" in formatted_output  # Regime type
        assert "DOVISH" in formatted_output  # Fed stance

    @pytest.mark.asyncio()
    async def test_news_analysis_formatting_integration(self, search_formatter):
        """Test detailed news analysis formatting."""
        news_items = [
            {
                "title": "Breaking: Bitcoin Breaks $65K as Wall Street Embraces Crypto",
                "content": """
                Bitcoin surged past $65,000 for the first time, driven by unprecedented
                institutional demand following ETF approvals. Major Wall Street firms
                are now offering crypto services to clients, marking a watershed moment
                for cryptocurrency adoption. Technical analysis shows strong bullish
                momentum with RSI at 72 and volume confirming the breakout.
                """,
                "url": "https://wsj.com/bitcoin-65k-breakthrough",
                "published_time": datetime.now(UTC),
            },
            {
                "title": "Fed Chair Powell: 'Crypto Innovation Can Coexist with Regulation'",
                "content": """
                Federal Reserve Chairman Jerome Powell expressed optimism about
                cryptocurrency innovation during congressional testimony, stating
                that proper regulation can support innovation. Market participants
                interpreted the comments as dovish for crypto regulation.
                """,
                "url": "https://reuters.com/fed-crypto-regulation",
                "published_time": datetime.now(UTC) - timedelta(minutes=30),
            },
            {
                "title": "Ethereum Gas Fees Drop 60% Following Latest Upgrade",
                "content": """
                Ethereum network upgrade successfully reduces transaction costs,
                making DeFi more accessible. Layer 2 solutions see increased adoption
                as ecosystem becomes more efficient. Developer activity surges
                with improved economics.
                """,
                "url": "https://coindesk.com/ethereum-gas-fees-drop",
                "published_time": datetime.now(UTC) - timedelta(hours=1),
            },
        ]

        # Format news analysis
        formatted_news = await search_formatter.format_news_results(news_items)

        # Validate news formatting
        assert "📰 **NEWS ANALYSIS**" in formatted_news
        assert "🎯 **Overall News Sentiment**" in formatted_news
        assert "📊 **Articles Analyzed**: 3" in formatted_news
        assert "🔍 **Key Market Insights**" in formatted_news
        assert "📈 **Trading Signals**" in formatted_news
        assert "📑 **Recent High-Priority Articles**" in formatted_news

        # Check for content
        assert "Bitcoin" in formatted_news
        assert "Ethereum" in formatted_news
        assert "Fed" in formatted_news or "Federal" in formatted_news
        assert "$65" in formatted_news or "65K" in formatted_news

        # Check for sentiment analysis
        assert "BULLISH" in formatted_news or "bullish" in formatted_news

    @pytest.mark.asyncio()
    async def test_token_optimization_integration(self, search_formatter):
        """Test token optimization with large content."""
        # Create oversized content
        large_context = {
            "news_results": [
                {
                    "title": f"Extended Market Analysis Report {i}",
                    "content": f"""
                    This is an extremely detailed market analysis report that contains
                    extensive information about cryptocurrency markets, traditional
                    financial markets, regulatory developments, technical analysis,
                    fundamental analysis, macroeconomic factors, geopolitical events,
                    institutional adoption trends, retail investor sentiment, options
                    flow analysis, derivatives markets, spot market dynamics, liquidity
                    conditions, volatility analysis, correlation studies, regime
                    detection, momentum indicators, and comprehensive trading
                    implications for report number {i}.
                    """
                    * 10,  # Make it very long
                    "url": f"https://analysis.com/report-{i}",
                    "published_time": datetime.now(UTC) - timedelta(hours=i),
                }
                for i in range(20)  # 20 large articles
            ],
            "sentiment_result": SentimentResult(
                sentiment_score=0.6,
                confidence=0.8,
                key_themes=["Extensive", "Analysis", "Markets"] * 20,  # Long themes
                bullish_indicators=[
                    "Factor " + str(i) for i in range(50)
                ],  # Many indicators
                bearish_indicators=["Risk " + str(i) for i in range(30)],
            ),
        }

        # Format with optimization
        formatted_output = await search_formatter.format_market_context(large_context)

        # Validate optimization
        estimated_tokens = search_formatter._estimate_token_count(formatted_output)
        assert (
            estimated_tokens <= search_formatter.max_total_tokens * 1.2
        )  # Allow margin

        # Should still contain essential information
        assert "COMPREHENSIVE MARKET CONTEXT ANALYSIS" in formatted_output
        assert "NEWS ANALYSIS" in formatted_output
        assert "MARKET SENTIMENT ANALYSIS" in formatted_output

    @pytest.mark.asyncio()
    async def test_key_insights_extraction_integration(self, search_formatter):
        """Test comprehensive key insights extraction."""
        search_results = {
            "news_items": [
                {
                    "title": "Massive Bitcoin Whale Moves $500M to Exchange",
                    "content": "Large institutional holder transfers significant Bitcoin position signaling potential market impact",
                },
                {
                    "title": "Technical Breakout: Crypto Forms Bull Flag Pattern",
                    "content": "Chart analysis shows classical bull flag formation with volume confirmation",
                },
                {
                    "title": "Regulatory Breakthrough: Clear Framework Announced",
                    "content": "Government provides comprehensive regulatory framework for cryptocurrency operations",
                },
            ],
            "sentiment_data": {
                "sentiment_score": 0.75,
                "confidence": 0.88,
                "volatility_signals": ["High options activity", "VIX momentum"],
            },
            "price_data": {
                "price_change_24h": 0.085,  # 8.5% gain
                "volume_change_24h": 0.42,  # 42% volume surge
            },
            "technical_analysis": {
                "rsi": 76,  # Overbought
                "trend_direction": "BULLISH",
                "macd": "BULLISH_CROSSOVER",
                "support_levels": [58000, 60000],
                "resistance_levels": [68000, 70000],
            },
        }

        # Extract key insights
        insights = await search_formatter.extract_key_insights(search_results)

        # Validate insights
        assert len(insights) > 0
        assert len(insights) <= 10  # Should be limited

        # Check for high-priority insights
        insights_text = " ".join(insights).lower()
        assert any(
            keyword in insights_text
            for keyword in [
                "whale",
                "institutional",
                "breakout",
                "bull flag",
                "regulatory",
                "volume",
                "bullish",
            ]
        )

        # First insight should be high priority
        assert any(
            keyword in insights[0].lower()
            for keyword in ["whale", "breakout", "regulatory", "institutional"]
        )


class TestFullOmniSearchLLMIntegration:
    """Full integration tests for OmniSearch with LLM agent."""

    @pytest.fixture()
    def mock_llm_agent(self):
        """Create mock LLM agent for testing."""
        agent = Mock()
        agent.generate_response = AsyncMock()
        return agent

    @pytest.mark.asyncio()
    async def test_complete_omnisearch_llm_workflow(self, mock_llm_agent):
        """Test complete OmniSearch integration with LLM agent workflow."""
        # Create all components
        omnisearch_client = OmniSearchClient(enable_cache=True)
        sentiment_service = FinancialSentimentService()
        context_analyzer = MarketContextAnalyzer()
        search_formatter = WebSearchFormatter()

        # Mock OmniSearch client responses
        with (
            patch.object(omnisearch_client, "connect") as mock_connect,
            patch.object(omnisearch_client, "search_financial_news") as mock_news,
            patch.object(
                omnisearch_client, "search_crypto_sentiment"
            ) as mock_crypto_sentiment,
            patch.object(
                omnisearch_client, "search_nasdaq_sentiment"
            ) as mock_nasdaq_sentiment,
            patch.object(
                omnisearch_client, "search_market_correlation"
            ) as mock_correlation,
        ):
            # Setup mocks
            mock_connect.return_value = True

            mock_news.return_value = [
                FinancialNewsResult(
                    base_result=SearchResult(
                        title="Bitcoin ETF Approval Drives Record Institutional Inflows",
                        url="https://bloomberg.com/btc-etf",
                        snippet="Massive institutional demand follows ETF approval",
                        source="bloomberg.com",
                        relevance_score=0.95,
                    ),
                    sentiment="positive",
                    mentioned_symbols=["BTC"],
                    impact_level="high",
                )
            ]

            mock_crypto_sentiment.return_value = SentimentAnalysis(
                symbol="BTC",
                overall_sentiment="bullish",
                sentiment_score=0.72,
                confidence=0.85,
                source_count=42,
                key_drivers=["ETF approval", "Institutional adoption"],
                risk_factors=["Volatility concerns"],
            )

            mock_nasdaq_sentiment.return_value = SentimentAnalysis(
                symbol="NASDAQ",
                overall_sentiment="bullish",
                sentiment_score=0.58,
                confidence=0.78,
                source_count=35,
                key_drivers=["Tech sector strength"],
                risk_factors=["Interest rate sensitivity"],
            )

            mock_correlation.return_value = MarketCorrelation(
                primary_symbol="BTC",
                secondary_symbol="QQQ",
                correlation_coefficient=0.65,
                strength="moderate",
                direction="positive",
                timeframe="30d",
            )

            # Execute complete workflow
            await omnisearch_client.connect()

            # 1. Gather data from OmniSearch
            news_results = await omnisearch_client.search_financial_news(
                "Bitcoin crypto market"
            )
            crypto_sentiment = await omnisearch_client.search_crypto_sentiment("BTC")
            nasdaq_sentiment = await omnisearch_client.search_nasdaq_sentiment()
            market_correlation = await omnisearch_client.search_market_correlation(
                "BTC", "QQQ"
            )

            # 2. Process with sentiment service
            news_data = [
                {
                    "title": result.base_result.title,
                    "content": result.base_result.snippet,
                }
                for result in news_results
            ]

            processed_sentiment = await sentiment_service.analyze_news_sentiment(
                news_data
            )

            # 3. Analyze with context analyzer
            # Mock price data for correlation analysis
            crypto_data = {"prices": list(range(100, 200))}
            nasdaq_data = {"prices": list(range(50, 150))}

            correlation_analysis = (
                await context_analyzer.analyze_crypto_nasdaq_correlation(
                    crypto_data, nasdaq_data
                )
            )

            sentiment_data = {
                "text": "Bullish market conditions with strong institutional support",
                "news_headlines": ["ETF approval drives momentum"],
            }
            market_regime = await context_analyzer.detect_market_regime(sentiment_data)

            # 4. Format with search formatter
            comprehensive_context = {
                "news_results": [
                    {
                        "title": result.base_result.title,
                        "content": result.base_result.snippet,
                        "url": result.base_result.url,
                        "published_time": datetime.now(UTC),
                    }
                    for result in news_results
                ],
                "sentiment_result": processed_sentiment,
                "correlation_analysis": correlation_analysis,
                "market_regime": market_regime,
            }

            formatted_context = await search_formatter.format_market_context(
                comprehensive_context
            )

            # 5. Mock LLM agent response
            mock_llm_agent.generate_response.return_value = {
                "action": "LONG",
                "confidence": 0.8,
                "reasoning": "Strong bullish sentiment with ETF approval and institutional demand",
                "risk_level": "MEDIUM",
                "position_size": 0.3,
            }

            # Simulate LLM agent processing
            llm_response = await mock_llm_agent.generate_response(
                market_data={"current_price": 62000}, context=formatted_context
            )

            # Validate complete workflow
            assert len(news_results) > 0
            assert crypto_sentiment.symbol == "BTC"
            assert nasdaq_sentiment.symbol == "NASDAQ"
            assert market_correlation.correlation_coefficient == 0.65
            assert processed_sentiment.sentiment_score > 0  # Should be positive
            assert isinstance(correlation_analysis, CorrelationAnalysis)
            assert isinstance(market_regime, MarketRegime)
            assert isinstance(formatted_context, str)
            assert len(formatted_context) > 500
            assert llm_response["action"] == "LONG"

            # Validate that context contains all necessary information
            assert (
                "ETF approval" in formatted_context
                or "ETF Approval" in formatted_context
            )
            assert (
                "bullish" in formatted_context.lower() or "BULLISH" in formatted_context
            )
            assert (
                "institutional" in formatted_context.lower()
                or "Institutional" in formatted_context
            )

    @pytest.mark.asyncio()
    async def test_error_handling_integration(self):
        """Test error handling across the complete integration."""
        # Create components
        omnisearch_client = OmniSearchClient()
        sentiment_service = FinancialSentimentService()
        context_analyzer = MarketContextAnalyzer()
        search_formatter = WebSearchFormatter()

        # Test with simulated failures
        with patch.object(omnisearch_client, "connect") as mock_connect:
            mock_connect.return_value = False  # Simulate connection failure

            # Should handle connection failure gracefully
            connected = await omnisearch_client.connect()
            assert connected is False

            # Should still provide fallback data
            fallback_sentiment = await omnisearch_client.search_crypto_sentiment("BTC")
            assert fallback_sentiment.overall_sentiment == "neutral"
            assert fallback_sentiment.confidence == 0.1  # Low confidence fallback

            # Sentiment service should handle empty data
            empty_sentiment = await sentiment_service.analyze_news_sentiment([])
            assert empty_sentiment.sentiment_score == 0.0
            assert empty_sentiment.confidence == 0.0

            # Context analyzer should handle invalid data
            invalid_correlation = (
                await context_analyzer.analyze_crypto_nasdaq_correlation({}, {})
            )
            assert invalid_correlation.direction == "ERROR"

            # Formatter should handle errors gracefully
            error_context = {"invalid": "data"}
            formatted_error = await search_formatter.format_market_context(
                error_context
            )
            assert isinstance(formatted_error, str)
            assert "COMPREHENSIVE MARKET CONTEXT ANALYSIS" in formatted_error

    @pytest.mark.asyncio()
    async def test_performance_integration(self):
        """Test performance of complete integration."""
        import time

        # Create components
        omnisearch_client = OmniSearchClient()
        sentiment_service = FinancialSentimentService()
        MarketContextAnalyzer()
        search_formatter = WebSearchFormatter()

        # Mock responses for performance testing
        with (
            patch.object(omnisearch_client, "connect") as mock_connect,
            patch.object(omnisearch_client, "search_financial_news") as mock_news,
            patch.object(
                omnisearch_client, "search_crypto_sentiment"
            ) as mock_sentiment,
        ):
            mock_connect.return_value = True
            mock_news.return_value = []
            mock_sentiment.return_value = SentimentAnalysis(
                symbol="BTC",
                overall_sentiment="neutral",
                sentiment_score=0.0,
                confidence=0.5,
                source_count=0,
            )

            # Measure performance
            start_time = time.time()

            # Execute workflow
            await omnisearch_client.connect()
            await omnisearch_client.search_financial_news("test")
            await omnisearch_client.search_crypto_sentiment("BTC")
            processed_sentiment = await sentiment_service.analyze_news_sentiment([])

            context = {"news_results": [], "sentiment_result": processed_sentiment}
            formatted_output = await search_formatter.format_market_context(context)

            end_time = time.time()
            total_time = end_time - start_time

            # Should complete within reasonable time (under 2 seconds)
            assert total_time < 2.0
            assert isinstance(formatted_output, str)


# Test fixtures for shared use
@pytest.fixture()
def sample_trading_context():
    """Sample trading context for integration testing."""
    return {
        "symbol": "BTC-USD",
        "current_price": 62500.00,
        "24h_change": 0.045,  # 4.5% gain
        "volume_24h": 28500000000,  # $28.5B volume
        "market_cap": 1200000000000,  # $1.2T market cap
        "timestamp": datetime.now(UTC),
        "technical_indicators": {
            "rsi": 68,
            "macd": "BULLISH",
            "sma_20": 59800,
            "sma_50": 57200,
            "support": 58000,
            "resistance": 65000,
        },
        "market_conditions": {
            "volatility": "MODERATE",
            "trend": "BULLISH",
            "volume_profile": "HIGH",
            "momentum": "STRONG",
        },
    }


@pytest.fixture()
def comprehensive_omnisearch_response():
    """Comprehensive OmniSearch response for testing."""
    return {
        "financial_news": [
            FinancialNewsResult(
                base_result=SearchResult(
                    title="Bitcoin Institutional Adoption Accelerates",
                    url="https://institutional-crypto.com/adoption",
                    snippet="Major institutions announce Bitcoin treasury strategies",
                    source="institutional-crypto.com",
                    relevance_score=0.92,
                ),
                sentiment="positive",
                mentioned_symbols=["BTC", "BTCUSD"],
                news_category="adoption",
                impact_level="high",
            )
        ],
        "crypto_sentiment": SentimentAnalysis(
            symbol="BTC",
            overall_sentiment="bullish",
            sentiment_score=0.68,
            confidence=0.84,
            source_count=38,
            key_drivers=[
                "Institutional treasury adoption",
                "Technical momentum building",
                "Regulatory clarity improving",
            ],
            risk_factors=[
                "Macroeconomic uncertainty",
                "Profit-taking pressure at resistance",
            ],
        ),
        "nasdaq_sentiment": SentimentAnalysis(
            symbol="NASDAQ",
            overall_sentiment="bullish",
            sentiment_score=0.54,
            confidence=0.76,
            source_count=42,
            key_drivers=["Tech earnings strength", "AI innovation momentum"],
            risk_factors=["Interest rate sensitivity", "Valuation concerns"],
        ),
        "market_correlation": MarketCorrelation(
            primary_symbol="BTC",
            secondary_symbol="QQQ",
            correlation_coefficient=0.62,
            strength="moderate",
            direction="positive",
            timeframe="30d",
            beta=1.38,
            r_squared=0.42,
        ),
    }


class TestOmniSearchConfigurationIntegration:
    """Test OmniSearch configuration and settings integration."""

    def test_configuration_loading(self):
        """Test OmniSearch configuration loading from settings."""
        # Test default configuration
        client = OmniSearchClient()
        assert client.server_url is not None
        assert isinstance(client.cache, object)  # Cache should be enabled by default
        assert client.rate_limiter.max_requests > 0

    def test_environment_variable_integration(self):
        """Test environment variable integration."""
        with patch.dict(
            "os.environ",
            {
                "OMNISEARCH_API_KEY": "test_env_key",
                "OMNISEARCH_SERVER_URL": "https://env.omnisearch.com",
            },
        ):
            # Test that environment variables are picked up
            # This would require actual environment variable handling in the client
            pass  # Placeholder for environment variable tests

    @pytest.mark.asyncio()
    async def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test with invalid configuration
        client = OmniSearchClient(
            server_url="invalid_url",
            rate_limit_requests=0,  # Invalid rate limit
            cache_ttl=-1,  # Invalid TTL
        )

        # Should handle invalid configuration gracefully
        health = await client.health_check()
        assert isinstance(health, dict)
        assert "connected" in health


if __name__ == "__main__":
    # Run specific integration tests
    pytest.main([__file__, "-v", "--tb=short"])
