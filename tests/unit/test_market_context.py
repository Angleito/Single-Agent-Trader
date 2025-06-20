"""Unit tests for market context analysis module."""

import numpy as np
import pytest

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


class TestCorrelationAnalysis:
    """Test cases for CorrelationAnalysis model."""

    def test_correlation_analysis_creation(self):
        """Test creating CorrelationAnalysis with basic data."""
        analysis = CorrelationAnalysis(
            correlation_coefficient=0.65,
            correlation_strength=CorrelationStrength.MODERATE,
            direction="POSITIVE",
            p_value=0.02,
            is_significant=True,
            sample_size=100,
            correlation_stability=0.7,
            reliability_score=0.8,
        )

        assert analysis.correlation_coefficient == 0.65
        assert analysis.correlation_strength == CorrelationStrength.MODERATE
        assert analysis.direction == "POSITIVE"
        assert analysis.is_significant is True
        assert analysis.sample_size == 100

    def test_correlation_analysis_with_rolling_data(self):
        """Test CorrelationAnalysis with rolling correlation data."""
        analysis = CorrelationAnalysis(
            correlation_coefficient=0.45,
            correlation_strength=CorrelationStrength.MODERATE,
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

        assert analysis.rolling_correlation_24h == 0.52
        assert analysis.rolling_correlation_7d == 0.38
        assert len(analysis.regime_dependent_correlation) == 2
        assert analysis.regime_dependent_correlation["HIGH_VOLATILITY"] == 0.7

    def test_correlation_analysis_validation(self):
        """Test CorrelationAnalysis field validation."""
        with pytest.raises(ValueError):
            # Correlation coefficient out of range
            CorrelationAnalysis(
                correlation_coefficient=1.5,
                correlation_strength=CorrelationStrength.STRONG,
                direction="POSITIVE",
                p_value=0.05,
                is_significant=True,
                sample_size=100,
                correlation_stability=0.5,
                reliability_score=0.8,
            )

        with pytest.raises(ValueError):
            # Reliability score out of range
            CorrelationAnalysis(
                correlation_coefficient=0.5,
                correlation_strength=CorrelationStrength.MODERATE,
                direction="POSITIVE",
                p_value=0.05,
                is_significant=True,
                sample_size=100,
                correlation_stability=0.5,
                reliability_score=1.5,
            )


class TestMarketRegime:
    """Test cases for MarketRegime model."""

    def test_market_regime_creation(self):
        """Test creating MarketRegime with basic data."""
        regime = MarketRegime(
            regime_type=MarketRegimeType.RISK_ON,
            confidence=0.8,
            key_drivers=["Dovish Fed policy", "Strong earnings"],
            fed_policy_stance="DOVISH",
            inflation_environment="STABLE",
        )

        assert regime.regime_type == MarketRegimeType.RISK_ON
        assert regime.confidence == 0.8
        assert len(regime.key_drivers) == 2
        assert regime.fed_policy_stance == "DOVISH"

    def test_market_regime_full_data(self):
        """Test MarketRegime with complete data."""
        regime = MarketRegime(
            regime_type=MarketRegimeType.RISK_OFF,
            confidence=0.75,
            key_drivers=["Hawkish Fed", "Geopolitical tensions", "Inflation concerns"],
            fed_policy_stance="HAWKISH",
            inflation_environment="HIGH",
            interest_rate_trend="RISING",
            geopolitical_risk_level="HIGH",
            crypto_adoption_momentum="LOW",
            institutional_sentiment="NEGATIVE",
            regulatory_environment="RESTRICTIVE",
            market_volatility_regime="HIGH",
            liquidity_conditions="TIGHT",
            duration_days=45,
            regime_change_probability=0.3,
        )

        assert regime.regime_type == MarketRegimeType.RISK_OFF
        assert regime.geopolitical_risk_level == "HIGH"
        assert regime.duration_days == 45
        assert regime.regime_change_probability == 0.3

    def test_market_regime_validation(self):
        """Test MarketRegime field validation."""
        with pytest.raises(ValueError):
            # Confidence out of range
            MarketRegime(
                regime_type=MarketRegimeType.RISK_ON, confidence=1.5, key_drivers=[]
            )


class TestRiskSentiment:
    """Test cases for RiskSentiment model."""

    def test_risk_sentiment_creation(self):
        """Test creating RiskSentiment with basic data."""
        sentiment = RiskSentiment(
            fear_greed_index=75.0,
            sentiment_level=SentimentLevel.GREED,
            volatility_expectation=25.0,
            market_stress_indicator=0.3,
        )

        assert sentiment.fear_greed_index == 75.0
        assert sentiment.sentiment_level == SentimentLevel.GREED
        assert sentiment.volatility_expectation == 25.0
        assert sentiment.market_stress_indicator == 0.3

    def test_risk_sentiment_with_details(self):
        """Test RiskSentiment with detailed indicators."""
        sentiment = RiskSentiment(
            fear_greed_index=25.0,
            sentiment_level=SentimentLevel.FEAR,
            volatility_expectation=35.0,
            market_stress_indicator=0.8,
            vix_level=28.5,
            crypto_fear_greed=20.0,
            social_sentiment_score=-0.4,
            news_sentiment_score=-0.3,
            funding_rates_sentiment=-0.2,
            options_flow_sentiment="BEARISH",
            insider_activity="BEARISH",
            retail_sentiment="NEUTRAL",
            sentiment_divergence=True,
        )

        assert sentiment.vix_level == 28.5
        assert sentiment.crypto_fear_greed == 20.0
        assert sentiment.social_sentiment_score == -0.4
        assert sentiment.sentiment_divergence is True

    def test_risk_sentiment_validation(self):
        """Test RiskSentiment field validation."""
        with pytest.raises(ValueError):
            # Fear greed index out of range
            RiskSentiment(
                fear_greed_index=150.0,
                sentiment_level=SentimentLevel.NEUTRAL,
                volatility_expectation=20.0,
                market_stress_indicator=0.5,
            )


class TestMomentumAlignment:
    """Test cases for MomentumAlignment model."""

    def test_momentum_alignment_creation(self):
        """Test creating MomentumAlignment with basic data."""
        alignment = MomentumAlignment(
            directional_alignment=0.6,
            strength_alignment=0.7,
            crypto_momentum_score=0.4,
            nasdaq_momentum_score=0.5,
            trend_strength_crypto=0.8,
            trend_strength_nasdaq=0.6,
        )

        assert alignment.directional_alignment == 0.6
        assert alignment.strength_alignment == 0.7
        assert alignment.crypto_momentum_score == 0.4
        assert alignment.nasdaq_momentum_score == 0.5

    def test_momentum_alignment_with_details(self):
        """Test MomentumAlignment with detailed data."""
        alignment = MomentumAlignment(
            directional_alignment=-0.3,
            strength_alignment=0.4,
            crypto_momentum_score=0.6,
            nasdaq_momentum_score=-0.2,
            momentum_divergences=["Crypto bullish while NASDAQ bearish"],
            trend_strength_crypto=0.7,
            trend_strength_nasdaq=0.5,
            volume_momentum_alignment=0.2,
            momentum_sustainability=0.6,
            momentum_regime="ACCELERATION",
            cross_asset_momentum_flow="CRYPTO_OUTPERFORMING",
        )

        assert alignment.directional_alignment == -0.3
        assert len(alignment.momentum_divergences) == 1
        assert alignment.momentum_regime == "ACCELERATION"
        assert alignment.cross_asset_momentum_flow == "CRYPTO_OUTPERFORMING"

    def test_momentum_alignment_validation(self):
        """Test MomentumAlignment field validation."""
        with pytest.raises(ValueError):
            # Directional alignment out of range
            MomentumAlignment(
                directional_alignment=1.5,
                strength_alignment=0.5,
                crypto_momentum_score=0.3,
                nasdaq_momentum_score=0.2,
                trend_strength_crypto=0.6,
                trend_strength_nasdaq=0.4,
            )


class TestMarketContextAnalyzer:
    """Test cases for MarketContextAnalyzer."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = MarketContextAnalyzer()

        assert len(analyzer._fed_policy_keywords) > 0
        assert len(analyzer._inflation_keywords) > 0
        assert len(analyzer._geopolitical_keywords) > 0
        assert len(analyzer._crypto_adoption_keywords) > 0

    @pytest.mark.asyncio
    async def test_analyze_crypto_nasdaq_correlation_insufficient_data(self):
        """Test correlation analysis with insufficient data."""
        analyzer = MarketContextAnalyzer()

        crypto_data = {"prices": [50000, 51000, 52000]}  # Too few data points
        nasdaq_data = {"prices": [100, 101, 102]}

        correlation = await analyzer.analyze_crypto_nasdaq_correlation(
            crypto_data, nasdaq_data
        )

        assert correlation.correlation_coefficient == 0.0
        assert correlation.correlation_strength == CorrelationStrength.VERY_WEAK
        assert correlation.direction == "INSUFFICIENT_DATA"
        assert correlation.is_significant is False

    @pytest.mark.asyncio
    async def test_analyze_crypto_nasdaq_correlation_success(self):
        """Test successful correlation analysis."""
        analyzer = MarketContextAnalyzer()

        # Generate correlated price series
        rng = np.random.default_rng(42)
        base_crypto = rng.normal(0, 0.01, 100)
        base_nasdaq = rng.normal(0, 0.005, 100)

        # Add some correlation
        correlated_nasdaq = 0.7 * base_crypto + 0.3 * base_nasdaq

        crypto_prices = [50000]
        nasdaq_prices = [100]

        for i in range(99):
            crypto_prices.append(crypto_prices[-1] * (1 + base_crypto[i]))
            nasdaq_prices.append(nasdaq_prices[-1] * (1 + correlated_nasdaq[i]))

        crypto_data = {"prices": crypto_prices}
        nasdaq_data = {"prices": nasdaq_prices}

        correlation = await analyzer.analyze_crypto_nasdaq_correlation(
            crypto_data, nasdaq_data
        )

        assert abs(correlation.correlation_coefficient) > 0.0
        assert correlation.sample_size == 100
        assert correlation.reliability_score > 0.0

    @pytest.mark.asyncio
    async def test_analyze_crypto_nasdaq_correlation_negative(self):
        """Test correlation analysis with negative correlation."""
        analyzer = MarketContextAnalyzer()

        # Generate negatively correlated data
        rng = np.random.default_rng(42)
        crypto_returns = rng.normal(0.001, 0.02, 50)
        nasdaq_returns = -0.8 * crypto_returns + rng.normal(0, 0.01, 50)

        crypto_prices = [50000]
        nasdaq_prices = [100]

        for i in range(49):
            crypto_prices.append(crypto_prices[-1] * (1 + crypto_returns[i]))
            nasdaq_prices.append(nasdaq_prices[-1] * (1 + nasdaq_returns[i]))

        crypto_data = {"prices": crypto_prices}
        nasdaq_data = {"prices": nasdaq_prices}

        correlation = await analyzer.analyze_crypto_nasdaq_correlation(
            crypto_data, nasdaq_data
        )

        assert correlation.correlation_coefficient < 0.0
        assert correlation.direction == "NEGATIVE"

    @pytest.mark.asyncio
    async def test_analyze_crypto_nasdaq_correlation_error_handling(self):
        """Test correlation analysis error handling."""
        analyzer = MarketContextAnalyzer()

        # Test with invalid data
        crypto_data = {"invalid_key": "invalid_data"}
        nasdaq_data = {"prices": []}

        correlation = await analyzer.analyze_crypto_nasdaq_correlation(
            crypto_data, nasdaq_data
        )

        assert correlation.correlation_coefficient == 0.0
        assert correlation.direction == "ERROR"
        assert correlation.reliability_score == 0.0

    @pytest.mark.asyncio
    async def test_detect_market_regime_risk_on(self):
        """Test market regime detection - risk on."""
        analyzer = MarketContextAnalyzer()

        sentiment_data = {
            "text": "Federal Reserve dovish policy stance supports risk assets",
            "news_headlines": [
                "Fed cuts rates to support growth",
                "Risk appetite returns to markets",
                "Technology stocks rally on optimism",
            ],
        }

        regime = await analyzer.detect_market_regime(sentiment_data)

        assert regime.regime_type in [
            MarketRegimeType.RISK_ON,
            MarketRegimeType.TRANSITION,
        ]
        assert regime.confidence > 0.0
        assert len(regime.key_drivers) > 0

    @pytest.mark.asyncio
    async def test_detect_market_regime_risk_off(self):
        """Test market regime detection - risk off."""
        analyzer = MarketContextAnalyzer()

        sentiment_data = {
            "text": "Federal Reserve hawkish policy with rate hikes amid geopolitical tensions",
            "news_headlines": [
                "Fed raises rates aggressively",
                "Geopolitical conflict escalates",
                "Inflation concerns mount",
                "Flight to safety in markets",
            ],
        }

        regime = await analyzer.detect_market_regime(sentiment_data)

        assert regime.regime_type in [
            MarketRegimeType.RISK_OFF,
            MarketRegimeType.TRANSITION,
        ]
        assert regime.confidence > 0.0
        assert (
            "hawkish" in regime.fed_policy_stance.lower()
            or regime.fed_policy_stance == "HAWKISH"
        )

    @pytest.mark.asyncio
    async def test_detect_market_regime_transition(self):
        """Test market regime detection - transition."""
        analyzer = MarketContextAnalyzer()

        sentiment_data = {
            "text": "Mixed signals from Federal Reserve with uncertain policy outlook",
            "news_headlines": [
                "Mixed economic data creates uncertainty",
                "Policy makers divided on rate decisions",
            ],
        }

        regime = await analyzer.detect_market_regime(sentiment_data)

        # Could be any regime type based on mixed signals
        assert isinstance(regime.regime_type, MarketRegimeType)
        assert regime.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_assess_risk_sentiment_extreme_fear(self):
        """Test risk sentiment assessment - extreme fear."""
        analyzer = MarketContextAnalyzer()

        news_data = [
            {
                "title": "Market Crash: Panic Selling Dominates",
                "content": "Fear grips markets as panic selling accelerates liquidation events",
            },
            {
                "title": "Crypto Capitulation: Bears Take Control",
                "content": "Bearish sentiment dominates with fear index spiking to extreme levels",
            },
        ]

        sentiment = await analyzer.assess_risk_sentiment(news_data)

        assert sentiment.fear_greed_index < 50.0  # Should be fearful
        assert sentiment.sentiment_level in [
            SentimentLevel.FEAR,
            SentimentLevel.EXTREME_FEAR,
        ]
        assert sentiment.volatility_expectation > 20.0

    @pytest.mark.asyncio
    async def test_assess_risk_sentiment_extreme_greed(self):
        """Test risk sentiment assessment - extreme greed."""
        analyzer = MarketContextAnalyzer()

        news_data = [
            {
                "title": "Market Euphoria: Bulls Dominate Rally",
                "content": "Greed drives markets higher as bullish sentiment reaches extreme levels",
            },
            {
                "title": "FOMO Buying Surge: New Highs Everywhere",
                "content": "Greedy investors chase rallies with extreme bullish positioning",
            },
        ]

        sentiment = await analyzer.assess_risk_sentiment(news_data)

        assert sentiment.fear_greed_index > 50.0  # Should be greedy
        assert sentiment.sentiment_level in [
            SentimentLevel.GREED,
            SentimentLevel.EXTREME_GREED,
        ]

    @pytest.mark.asyncio
    async def test_assess_risk_sentiment_neutral(self):
        """Test risk sentiment assessment - neutral."""
        analyzer = MarketContextAnalyzer()

        news_data = [
            {
                "title": "Markets Show Mixed Signals",
                "content": "Balanced market conditions with neutral investor sentiment",
            }
        ]

        sentiment = await analyzer.assess_risk_sentiment(news_data)

        assert 30.0 <= sentiment.fear_greed_index <= 70.0  # Should be neutral range
        assert sentiment.sentiment_level == SentimentLevel.NEUTRAL

    @pytest.mark.asyncio
    async def test_assess_risk_sentiment_error_handling(self):
        """Test risk sentiment assessment error handling."""
        analyzer = MarketContextAnalyzer()

        # Test with invalid news data
        news_data = [{"invalid": "data"}]

        sentiment = await analyzer.assess_risk_sentiment(news_data)

        # Should return default neutral sentiment
        assert sentiment.fear_greed_index == 50.0
        assert sentiment.sentiment_level == SentimentLevel.NEUTRAL

    @pytest.mark.asyncio
    async def test_calculate_momentum_alignment_positive(self):
        """Test momentum alignment calculation - positive alignment."""
        analyzer = MarketContextAnalyzer()

        crypto_indicators = {
            "rsi": 65,
            "price_change_24h": 0.05,  # 5% up
            "volume_change_24h": 0.3,  # 30% volume increase
            "trend_strength": 0.8,
        }

        nasdaq_indicators = {
            "price_change_24h": 0.02,  # 2% up
            "volume_change_24h": 0.2,  # 20% volume increase
            "tech_sector_performance": 3.0,  # 3% tech sector gain
            "trend_strength": 0.7,
        }

        alignment = await analyzer.calculate_momentum_alignment(
            crypto_indicators, nasdaq_indicators
        )

        assert alignment.directional_alignment > 0.0  # Both moving up
        assert alignment.crypto_momentum_score > 0.0
        assert alignment.nasdaq_momentum_score > 0.0
        assert alignment.strength_alignment > 0.0

    @pytest.mark.asyncio
    async def test_calculate_momentum_alignment_negative(self):
        """Test momentum alignment calculation - negative alignment."""
        analyzer = MarketContextAnalyzer()

        crypto_indicators = {
            "rsi": 35,
            "price_change_24h": 0.08,  # 8% up (crypto bullish)
            "volume_change_24h": 0.4,
            "trend_strength": 0.6,
            "trend_direction": "BULLISH",
        }

        nasdaq_indicators = {
            "price_change_24h": -0.03,  # 3% down (nasdaq bearish)
            "volume_change_24h": 0.1,
            "tech_sector_performance": -2.0,
            "trend_strength": 0.5,
            "trend_direction": "BEARISH",
        }

        alignment = await analyzer.calculate_momentum_alignment(
            crypto_indicators, nasdaq_indicators
        )

        assert alignment.directional_alignment < 0.0  # Opposite directions
        assert len(alignment.momentum_divergences) > 0
        assert (
            "Crypto bullish while NASDAQ bearish" in alignment.momentum_divergences[0]
        )

    @pytest.mark.asyncio
    async def test_calculate_momentum_alignment_error_handling(self):
        """Test momentum alignment calculation error handling."""
        analyzer = MarketContextAnalyzer()

        # Test with invalid indicators
        crypto_indicators = {}
        nasdaq_indicators = {}

        alignment = await analyzer.calculate_momentum_alignment(
            crypto_indicators, nasdaq_indicators
        )

        assert alignment.directional_alignment == 0.0
        assert alignment.strength_alignment == 0.0
        assert "Analysis error" in alignment.momentum_divergences[0]

    def test_generate_context_summary(self):
        """Test context summary generation."""
        analyzer = MarketContextAnalyzer()

        correlation = CorrelationAnalysis(
            correlation_coefficient=0.65,
            correlation_strength=CorrelationStrength.MODERATE,
            direction="POSITIVE",
            p_value=0.02,
            is_significant=True,
            sample_size=100,
            correlation_stability=0.7,
            reliability_score=0.8,
        )

        regime = MarketRegime(
            regime_type=MarketRegimeType.RISK_ON,
            confidence=0.75,
            key_drivers=["Dovish Fed policy", "Strong earnings"],
            fed_policy_stance="DOVISH",
            inflation_environment="STABLE",
        )

        summary = analyzer.generate_context_summary(correlation, regime)

        assert "MARKET CONTEXT ANALYSIS" in summary
        assert "CORRELATION ANALYSIS" in summary
        assert "MARKET REGIME ANALYSIS" in summary
        assert "TRADING IMPLICATIONS" in summary
        assert "0.650" in summary  # Correlation coefficient
        assert "RISK_ON" in summary  # Regime type
        assert "DOVISH" in summary  # Fed policy

    def test_extract_price_series_valid_data(self):
        """Test price series extraction with valid data."""
        analyzer = MarketContextAnalyzer()

        # Test with 'prices' key
        data1 = {"prices": [100, 101, 102, 103]}
        prices1 = analyzer._extract_price_series(data1)
        assert prices1 == [100.0, 101.0, 102.0, 103.0]

        # Test with 'ohlcv' key
        data2 = {"ohlcv": [{"close": 100}, {"close": 101}, {"close": 102}]}
        prices2 = analyzer._extract_price_series(data2)
        assert prices2 == [100.0, 101.0, 102.0]

        # Test with 'candles' key
        data3 = {"candles": [{"close": 100}, {"close": 101}]}
        prices3 = analyzer._extract_price_series(data3)
        assert prices3 == [100.0, 101.0]

    def test_extract_price_series_invalid_data(self):
        """Test price series extraction with invalid data."""
        analyzer = MarketContextAnalyzer()

        # Test with missing keys
        data1 = {"invalid_key": [100, 101]}
        prices1 = analyzer._extract_price_series(data1)
        assert prices1 == []

        # Test with invalid price data
        data2 = {"prices": ["invalid", "price", "data"]}
        prices2 = analyzer._extract_price_series(data2)
        assert prices2 == []

    def test_align_time_series(self):
        """Test time series alignment."""
        analyzer = MarketContextAnalyzer()

        crypto_prices = [1, 2, 3, 4, 5, 6]
        nasdaq_prices = [10, 20, 30, 40]

        crypto_aligned, nasdaq_aligned = analyzer._align_time_series(
            crypto_prices, nasdaq_prices
        )

        assert len(crypto_aligned) == len(nasdaq_aligned) == 4
        assert crypto_aligned == [3, 4, 5, 6]  # Last 4 elements
        assert nasdaq_aligned == [10, 20, 30, 40]

    def test_calculate_rolling_correlation(self):
        """Test rolling correlation calculation."""
        analyzer = MarketContextAnalyzer()

        # Perfect positive correlation
        series1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        series2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

        correlation = analyzer._calculate_rolling_correlation(series1, series2, 5)
        assert correlation is not None
        assert abs(correlation - 1.0) < 0.01  # Should be close to 1

    def test_calculate_rolling_correlation_insufficient_data(self):
        """Test rolling correlation with insufficient data."""
        analyzer = MarketContextAnalyzer()

        series1 = [1, 2, 3]
        series2 = [2, 4, 6]

        correlation = analyzer._calculate_rolling_correlation(series1, series2, 10)
        assert correlation is None

    def test_classify_correlation_strength(self):
        """Test correlation strength classification."""
        analyzer = MarketContextAnalyzer()

        assert (
            analyzer._classify_correlation_strength(0.9)
            == CorrelationStrength.VERY_STRONG
        )
        assert (
            analyzer._classify_correlation_strength(0.7) == CorrelationStrength.STRONG
        )
        assert (
            analyzer._classify_correlation_strength(0.5) == CorrelationStrength.MODERATE
        )
        assert analyzer._classify_correlation_strength(0.3) == CorrelationStrength.WEAK
        assert (
            analyzer._classify_correlation_strength(0.1)
            == CorrelationStrength.VERY_WEAK
        )

    def test_determine_correlation_direction(self):
        """Test correlation direction determination."""
        analyzer = MarketContextAnalyzer()

        assert analyzer._determine_correlation_direction(0.5) == "POSITIVE"
        assert analyzer._determine_correlation_direction(-0.5) == "NEGATIVE"
        assert analyzer._determine_correlation_direction(0.05) == "UNCORRELATED"

    def test_calculate_correlation_reliability(self):
        """Test correlation reliability calculation."""
        analyzer = MarketContextAnalyzer()

        # High reliability case
        reliability1 = analyzer._calculate_correlation_reliability(0.8, 0.001, 200, 0.9)
        assert reliability1 > 0.8

        # Low reliability case
        reliability2 = analyzer._calculate_correlation_reliability(0.1, 0.5, 20, 0.2)
        assert reliability2 < 0.5

    def test_analyze_fed_policy_stance(self):
        """Test Fed policy stance analysis."""
        analyzer = MarketContextAnalyzer()

        # Hawkish data
        hawkish_data = {
            "text": "Federal Reserve rate hike expected with tightening monetary policy",
            "news_headlines": ["Fed signals aggressive policy to fight inflation"],
        }
        stance1 = analyzer._analyze_fed_policy_stance(hawkish_data)
        assert stance1 == "HAWKISH"

        # Dovish data
        dovish_data = {
            "text": "Federal Reserve rate cut expected with accommodative policy",
            "news_headlines": ["Fed provides stimulus with easing measures"],
        }
        stance2 = analyzer._analyze_fed_policy_stance(dovish_data)
        assert stance2 == "DOVISH"

        # Neutral data
        neutral_data = {
            "text": "Federal Reserve remains data dependent",
            "news_headlines": ["Fed waits for more economic data"],
        }
        stance3 = analyzer._analyze_fed_policy_stance(neutral_data)
        assert stance3 == "NEUTRAL"

    def test_analyze_inflation_environment(self):
        """Test inflation environment analysis."""
        analyzer = MarketContextAnalyzer()

        # High inflation
        high_inflation_data = {
            "text": "Inflation surge creates price pressures with CPI spike",
            "news_headlines": [],
        }
        env1 = analyzer._analyze_inflation_environment(high_inflation_data)
        assert env1 == "HIGH"

        # Low inflation
        low_inflation_data = {
            "text": "Disinflation trend with low inflation and price stability",
            "news_headlines": [],
        }
        env2 = analyzer._analyze_inflation_environment(low_inflation_data)
        assert env2 == "LOW"

    def test_assess_geopolitical_risk(self):
        """Test geopolitical risk assessment."""
        analyzer = MarketContextAnalyzer()

        # High risk
        high_risk_data = {
            "text": "War escalates with sanctions and trade war tensions",
            "news_headlines": ["Geopolitical conflict intensifies"],
        }
        risk1 = analyzer._assess_geopolitical_risk(high_risk_data)
        assert risk1 == "HIGH"

        # Low risk
        low_risk_data = {
            "text": "Peace negotiations show stability and cooperation",
            "news_headlines": ["Diplomatic resolution achieved"],
        }
        risk2 = analyzer._assess_geopolitical_risk(low_risk_data)
        assert risk2 == "LOW"

    def test_analyze_crypto_adoption_momentum(self):
        """Test crypto adoption momentum analysis."""
        analyzer = MarketContextAnalyzer()

        # High adoption
        high_adoption_data = {
            "text": "Institutional adoption surges with mainstream acceptance",
            "news_headlines": ["Major ETF approval drives adoption"],
        }
        momentum1 = analyzer._analyze_crypto_adoption_momentum(high_adoption_data)
        assert momentum1 == "HIGH"

        # Low adoption
        low_adoption_data = {
            "text": "Skepticism grows with regulatory barriers and resistance",
            "news_headlines": ["Concerns over crypto adoption persist"],
        }
        momentum2 = analyzer._analyze_crypto_adoption_momentum(low_adoption_data)
        assert momentum2 == "LOW"

    def test_calculate_crypto_momentum_score(self):
        """Test crypto momentum score calculation."""
        analyzer = MarketContextAnalyzer()

        # Positive momentum
        positive_indicators = {
            "rsi": 75,  # Overbought but strong
            "price_change_24h": 0.05,  # 5% gain
            "volume_change_24h": 0.3,  # 30% volume increase
        }
        score1 = analyzer._calculate_crypto_momentum_score(positive_indicators)
        assert score1 > 0.0

        # Negative momentum
        negative_indicators = {
            "rsi": 25,  # Oversold
            "price_change_24h": -0.08,  # 8% loss
            "volume_change_24h": -0.1,  # 10% volume decrease
        }
        score2 = analyzer._calculate_crypto_momentum_score(negative_indicators)
        assert score2 < 0.0

    def test_calculate_nasdaq_momentum_score(self):
        """Test NASDAQ momentum score calculation."""
        analyzer = MarketContextAnalyzer()

        # Positive momentum
        positive_indicators = {
            "price_change_24h": 0.02,  # 2% gain
            "volume_change_24h": 0.2,  # 20% volume increase
            "tech_sector_performance": 3.0,  # 3% tech sector gain
        }
        score1 = analyzer._calculate_nasdaq_momentum_score(positive_indicators)
        assert score1 > 0.0

        # Negative momentum
        negative_indicators = {
            "price_change_24h": -0.03,  # 3% loss
            "volume_change_24h": -0.2,  # 20% volume decrease
            "tech_sector_performance": -2.0,  # 2% tech sector loss
        }
        score2 = analyzer._calculate_nasdaq_momentum_score(negative_indicators)
        assert score2 < 0.0

    def test_calculate_directional_alignment(self):
        """Test directional alignment calculation."""
        analyzer = MarketContextAnalyzer()

        # Positive alignment (both positive)
        alignment1 = analyzer._calculate_directional_alignment(0.5, 0.3)
        assert alignment1 > 0.0

        # Negative alignment (both negative)
        alignment2 = analyzer._calculate_directional_alignment(-0.4, -0.6)
        assert alignment2 > 0.0

        # Divergent alignment (opposite signs)
        alignment3 = analyzer._calculate_directional_alignment(0.5, -0.3)
        assert alignment3 < 0.0

    def test_identify_momentum_divergences(self):
        """Test momentum divergence identification."""
        analyzer = MarketContextAnalyzer()

        crypto_indicators = {"trend_direction": "BULLISH", "volume_trend": "INCREASING"}

        nasdaq_indicators = {"trend_direction": "BEARISH", "volume_trend": "DECREASING"}

        divergences = analyzer._identify_momentum_divergences(
            crypto_indicators, nasdaq_indicators
        )

        assert len(divergences) > 0
        assert any(
            "crypto bullish while nasdaq bearish" in div.lower() for div in divergences
        )

    def test_calculate_trend_strength(self):
        """Test trend strength calculation."""
        analyzer = MarketContextAnalyzer()

        # Strong trend indicators
        strong_indicators = {
            "rsi": 75,  # Strong momentum
            "price_change_24h": 0.08,  # 8% move
            "volume_change_24h": 0.3,  # High volume
        }
        strength1 = analyzer._calculate_trend_strength(strong_indicators)
        assert strength1 > 0.5

        # Weak trend indicators
        weak_indicators = {
            "rsi": 50,  # Neutral
            "price_change_24h": 0.005,  # 0.5% move
            "volume_change_24h": 0.05,  # Low volume
        }
        strength2 = analyzer._calculate_trend_strength(weak_indicators)
        assert strength2 < 0.5

    def test_determine_momentum_regime(self):
        """Test momentum regime determination."""
        analyzer = MarketContextAnalyzer()

        # High momentum (acceleration)
        high_momentum_crypto = {"rsi": 80, "price_change_24h": 0.1}
        high_momentum_nasdaq = {"price_change_24h": 0.05, "tech_sector_performance": 5}

        regime1 = analyzer._determine_momentum_regime(
            high_momentum_crypto, high_momentum_nasdaq
        )
        assert regime1 == "ACCELERATION"

        # Low momentum (deceleration)
        low_momentum_crypto = {"rsi": 45, "price_change_24h": 0.001}
        low_momentum_nasdaq = {"price_change_24h": 0.0, "tech_sector_performance": 0}

        regime2 = analyzer._determine_momentum_regime(
            low_momentum_crypto, low_momentum_nasdaq
        )
        assert regime2 == "DECELERATION"

    def test_analyze_cross_asset_momentum_flow(self):
        """Test cross-asset momentum flow analysis."""
        analyzer = MarketContextAnalyzer()

        # Crypto outperforming
        crypto_strong = {"rsi": 75, "price_change_24h": 0.08}
        nasdaq_weak = {"price_change_24h": 0.01, "tech_sector_performance": 1}

        flow1 = analyzer._analyze_cross_asset_momentum_flow(crypto_strong, nasdaq_weak)
        assert flow1 == "CRYPTO_OUTPERFORMING"

        # Risk-on flow (both strong)
        crypto_strong2 = {"rsi": 70, "price_change_24h": 0.06}
        nasdaq_strong = {"price_change_24h": 0.04, "tech_sector_performance": 4}

        flow2 = analyzer._analyze_cross_asset_momentum_flow(
            crypto_strong2, nasdaq_strong
        )
        assert flow2 == "RISK_ON_FLOW"


# Fixtures for integration tests
@pytest.fixture
def sample_crypto_data():
    """Sample crypto market data."""
    rng = np.random.default_rng(42)
    prices = [50000]
    for _i in range(99):
        change = rng.normal(0.001, 0.02)
        prices.append(prices[-1] * (1 + change))

    return {
        "prices": prices,
        "ohlcv": [{"close": price} for price in prices[-20:]],  # Last 20 candles
    }


@pytest.fixture
def sample_nasdaq_data():
    """Sample NASDAQ market data."""
    rng = np.random.default_rng(24)
    prices = [100]
    for _i in range(99):
        change = rng.normal(0.0005, 0.01)
        prices.append(prices[-1] * (1 + change))

    return {
        "prices": prices,
        "candles": [{"close": price} for price in prices[-20:]],  # Last 20 candles
    }


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment data for regime detection."""
    return {
        "text": "Federal Reserve dovish policy supports risk assets with low inflation",
        "news_headlines": [
            "Fed maintains accommodative policy",
            "Inflation remains contained",
            "Risk appetite returns to markets",
            "Technology sector shows strength",
            "Crypto adoption accelerates",
        ],
        "vix_level": 18.5,
        "volatility_score": 0.3,
        "sentiment_divergence": False,
    }


class TestMarketContextAnalyzerIntegration:
    """Integration tests for MarketContextAnalyzer."""

    @pytest.mark.asyncio
    async def test_full_correlation_analysis_workflow(
        self, sample_crypto_data, sample_nasdaq_data
    ):
        """Test complete correlation analysis workflow."""
        analyzer = MarketContextAnalyzer()

        correlation = await analyzer.analyze_crypto_nasdaq_correlation(
            sample_crypto_data, sample_nasdaq_data
        )

        assert isinstance(correlation, CorrelationAnalysis)
        assert correlation.sample_size > 50
        assert correlation.reliability_score > 0.0
        assert -1.0 <= correlation.correlation_coefficient <= 1.0
        assert correlation.p_value >= 0.0

    @pytest.mark.asyncio
    async def test_full_regime_detection_workflow(self, sample_sentiment_data):
        """Test complete regime detection workflow."""
        analyzer = MarketContextAnalyzer()

        regime = await analyzer.detect_market_regime(sample_sentiment_data)

        assert isinstance(regime, MarketRegime)
        assert regime.confidence > 0.0
        assert len(regime.key_drivers) > 0
        assert regime.fed_policy_stance in ["HAWKISH", "DOVISH", "NEUTRAL"]
        assert regime.inflation_environment in ["HIGH", "LOW", "STABLE"]

    @pytest.mark.asyncio
    async def test_full_risk_sentiment_workflow(self):
        """Test complete risk sentiment analysis workflow."""
        analyzer = MarketContextAnalyzer()

        news_data = [
            {
                "title": "Market Rally Continues with Strong Momentum",
                "content": "Bullish sentiment drives markets higher with greed index rising",
            },
            {
                "title": "Fear Subsides as Volatility Declines",
                "content": "Risk appetite returns as fear indicators show improvement",
            },
        ]

        sentiment = await analyzer.assess_risk_sentiment(news_data)

        assert isinstance(sentiment, RiskSentiment)
        assert 0.0 <= sentiment.fear_greed_index <= 100.0
        assert isinstance(sentiment.sentiment_level, SentimentLevel)
        assert sentiment.volatility_expectation > 0.0

    @pytest.mark.asyncio
    async def test_full_momentum_alignment_workflow(self):
        """Test complete momentum alignment analysis workflow."""
        analyzer = MarketContextAnalyzer()

        crypto_indicators = {
            "rsi": 65,
            "price_change_24h": 0.04,
            "volume_change_24h": 0.25,
            "trend_direction": "BULLISH",
            "trend_strength": 0.7,
            "volume_trend": "INCREASING",
        }

        nasdaq_indicators = {
            "price_change_24h": 0.02,
            "volume_change_24h": 0.15,
            "tech_sector_performance": 2.5,
            "trend_direction": "BULLISH",
            "trend_strength": 0.6,
            "volume_trend": "INCREASING",
        }

        alignment = await analyzer.calculate_momentum_alignment(
            crypto_indicators, nasdaq_indicators
        )

        assert isinstance(alignment, MomentumAlignment)
        assert -1.0 <= alignment.directional_alignment <= 1.0
        assert 0.0 <= alignment.strength_alignment <= 1.0
        assert alignment.momentum_regime in ["ACCELERATION", "DECELERATION", "NORMAL"]

    @pytest.mark.asyncio
    async def test_comprehensive_market_analysis(
        self, sample_crypto_data, sample_nasdaq_data, sample_sentiment_data
    ):
        """Test comprehensive market analysis combining all components."""
        analyzer = MarketContextAnalyzer()

        # Run all analyses
        correlation = await analyzer.analyze_crypto_nasdaq_correlation(
            sample_crypto_data, sample_nasdaq_data
        )

        regime = await analyzer.detect_market_regime(sample_sentiment_data)

        # Generate comprehensive summary
        summary = analyzer.generate_context_summary(correlation, regime)

        # Validate comprehensive results
        assert isinstance(correlation, CorrelationAnalysis)
        assert isinstance(regime, MarketRegime)
        assert isinstance(summary, str)
        assert len(summary) > 500  # Should be comprehensive

        # Check that all major sections are included
        assert "CORRELATION ANALYSIS" in summary
        assert "MARKET REGIME ANALYSIS" in summary
        assert "TRADING IMPLICATIONS" in summary

    @pytest.mark.asyncio
    async def test_error_resilience_comprehensive(self):
        """Test error resilience across all analyzer methods."""
        analyzer = MarketContextAnalyzer()

        # Test with various invalid inputs
        invalid_data = {"invalid": "data"}
        empty_data = {}

        # All methods should handle errors gracefully
        correlation = await analyzer.analyze_crypto_nasdaq_correlation(
            invalid_data, empty_data
        )
        assert correlation.direction == "ERROR"

        regime = await analyzer.detect_market_regime(empty_data)
        assert regime.regime_type == MarketRegimeType.UNKNOWN

        sentiment = await analyzer.assess_risk_sentiment([])
        assert sentiment.sentiment_level == SentimentLevel.NEUTRAL

        alignment = await analyzer.calculate_momentum_alignment(
            empty_data, invalid_data
        )
        assert "error" in alignment.momentum_divergences[0].lower()

    def test_performance_large_dataset(self):
        """Test performance with large datasets."""
        analyzer = MarketContextAnalyzer()

        # Generate large price series
        large_crypto_prices = list(range(1000))
        rng = np.random.default_rng(42)
        large_nasdaq_prices = [
            p * 0.1 + rng.normal(0, 0.01) for p in large_crypto_prices
        ]

        crypto_data = {"prices": large_crypto_prices}
        nasdaq_data = {"prices": large_nasdaq_prices}

        # Should handle large datasets efficiently
        crypto_series = analyzer._extract_price_series(crypto_data)
        nasdaq_series = analyzer._extract_price_series(nasdaq_data)

        assert len(crypto_series) == 1000
        assert len(nasdaq_series) == 1000

        # Test alignment with large series
        aligned_crypto, aligned_nasdaq = analyzer._align_time_series(
            crypto_series, nasdaq_series
        )
        assert len(aligned_crypto) == len(aligned_nasdaq) == 1000
