"""Unit tests for financial sentiment analysis service."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot.services.financial_sentiment import (
    CryptoIndicators,
    FinancialSentimentService,
    NasdaqIndicators,
    SentimentResult,
)


class TestSentimentResult:
    """Test cases for SentimentResult model."""

    def test_sentiment_result_creation(self):
        """Test creating SentimentResult with basic data."""
        result = SentimentResult(
            sentiment_score=0.75,
            confidence=0.85,
            key_themes=["Bitcoin", "ETF", "Institutional"]
        )
        
        assert result.sentiment_score == 0.75
        assert result.confidence == 0.85
        assert len(result.key_themes) == 3
        assert result.bullish_indicators == []  # Default empty list
        assert result.bearish_indicators == []  # Default empty list
        assert isinstance(result.timestamp, datetime)

    def test_sentiment_result_with_indicators(self):
        """Test SentimentResult with bullish and bearish indicators."""
        result = SentimentResult(
            sentiment_score=0.3,
            confidence=0.7,
            bullish_indicators=["Strong institutional demand", "Technical breakout"],
            bearish_indicators=["Regulatory concerns"],
            volatility_signals=["High VIX", "Options volatility spike"]
        )
        
        assert len(result.bullish_indicators) == 2
        assert len(result.bearish_indicators) == 1
        assert len(result.volatility_signals) == 2

    def test_sentiment_result_validation(self):
        """Test SentimentResult field validation."""
        with pytest.raises(ValueError):
            # Sentiment score out of range
            SentimentResult(sentiment_score=1.5, confidence=0.8)
        
        with pytest.raises(ValueError):
            # Confidence out of range
            SentimentResult(sentiment_score=0.5, confidence=1.2)


class TestCryptoIndicators:
    """Test cases for CryptoIndicators model."""

    def test_crypto_indicators_creation(self):
        """Test creating CryptoIndicators with basic data."""
        indicators = CryptoIndicators(
            trend_direction="BULLISH",
            support_levels=[45000.0, 47000.0],
            resistance_levels=[52000.0, 55000.0]
        )
        
        assert indicators.trend_direction == "BULLISH"
        assert len(indicators.support_levels) == 2
        assert len(indicators.resistance_levels) == 2
        assert indicators.price_mentions == []  # Default empty list

    def test_crypto_indicators_with_details(self):
        """Test CryptoIndicators with detailed data."""
        price_mentions = [
            {"price": "50000", "context": "Bitcoin targets $50,000"},
            {"price": "60000", "context": "Next resistance at $60,000"}
        ]
        
        indicators = CryptoIndicators(
            price_mentions=price_mentions,
            volume_indicators=["Volume spike", "Unusual activity"],
            momentum_signals=["RSI oversold bounce"],
            adoption_signals=["Corporate treasury adoption"],
            regulatory_mentions=["SEC clarity expected"],
            whale_activity=["Large holder accumulation"],
            technical_patterns=["Bull flag formation"],
            indicator_signals={"RSI": "BULLISH", "MACD": "NEUTRAL"}
        )
        
        assert len(indicators.price_mentions) == 2
        assert len(indicators.volume_indicators) == 2
        assert len(indicators.momentum_signals) == 1
        assert indicators.indicator_signals["RSI"] == "BULLISH"

    def test_crypto_indicators_frozen(self):
        """Test that CryptoIndicators is immutable."""
        indicators = CryptoIndicators()
        
        with pytest.raises(ValueError):
            # Should not be able to modify frozen model
            indicators.trend_direction = "BEARISH"


class TestNasdaqIndicators:
    """Test cases for NasdaqIndicators model."""

    def test_nasdaq_indicators_creation(self):
        """Test creating NasdaqIndicators with basic data."""
        indicators = NasdaqIndicators(
            nasdaq_trend="BULLISH",
            sp500_mentions=["S&P 500 reaches new high"],
            dow_mentions=["Dow Jones industrial strength"]
        )
        
        assert indicators.nasdaq_trend == "BULLISH"
        assert len(indicators.sp500_mentions) == 1
        assert len(indicators.dow_mentions) == 1

    def test_nasdaq_indicators_with_economic_data(self):
        """Test NasdaqIndicators with economic indicators."""
        indicators = NasdaqIndicators(
            fed_policy_signals=["Hawkish Fed stance", "Rate hike expected"],
            interest_rate_mentions=["Federal funds rate", "10-year yield"],
            inflation_indicators=["CPI data", "Core inflation"],
            risk_on_signals=["Risk appetite strong"],
            risk_off_signals=["Flight to safety"],
            vix_mentions=["VIX volatility spike"],
            tech_sector_signals=["Tech rotation", "AI stocks rally"],
            financial_sector_signals=["Bank earnings strong"]
        )
        
        assert len(indicators.fed_policy_signals) == 2
        assert len(indicators.interest_rate_mentions) == 2
        assert len(indicators.tech_sector_signals) == 2

    def test_nasdaq_indicators_frozen(self):
        """Test that NasdaqIndicators is immutable."""
        indicators = NasdaqIndicators()
        
        with pytest.raises(ValueError):
            # Should not be able to modify frozen model
            indicators.nasdaq_trend = "BEARISH"


class TestFinancialSentimentService:
    """Test cases for FinancialSentimentService."""

    def test_service_initialization(self):
        """Test service initialization."""
        service = FinancialSentimentService()
        
        assert len(service._bullish_keywords) > 0
        assert len(service._bearish_keywords) > 0
        assert len(service._volatility_keywords) > 0
        assert service._crypto_price_pattern is not None
        assert len(service._technical_indicators) > 0

    @pytest.mark.asyncio
    async def test_analyze_news_sentiment_empty(self):
        """Test sentiment analysis with empty news list."""
        service = FinancialSentimentService()
        
        result = await service.analyze_news_sentiment([])
        
        assert result.sentiment_score == 0.0
        assert result.confidence == 0.0
        assert "No news data available" in result.key_themes

    @pytest.mark.asyncio
    async def test_analyze_news_sentiment_bullish(self):
        """Test sentiment analysis with bullish news."""
        service = FinancialSentimentService()
        
        news_items = [
            {
                "title": "Bitcoin Surges to New High on Strong Institutional Adoption",
                "content": "Bitcoin rally continues with bullish momentum and positive sentiment"
            },
            {
                "title": "Ethereum Breakout Signals Bullish Trend",
                "content": "Strong support levels hold as bulls accumulate"
            }
        ]
        
        result = await service.analyze_news_sentiment(news_items)
        
        assert result.sentiment_score > 0.0  # Should be positive
        assert result.confidence > 0.0
        assert len(result.key_themes) > 0

    @pytest.mark.asyncio
    async def test_analyze_news_sentiment_bearish(self):
        """Test sentiment analysis with bearish news."""
        service = FinancialSentimentService()
        
        news_items = [
            {
                "title": "Bitcoin Crashes Amid Regulatory Concerns",
                "content": "Bearish sentiment dominates as weak support levels break down"
            },
            {
                "title": "Crypto Sell-off Continues with Panic Selling",
                "content": "Fear grips market as bearish indicators multiply"
            }
        ]
        
        result = await service.analyze_news_sentiment(news_items)
        
        assert result.sentiment_score < 0.0  # Should be negative
        assert result.confidence > 0.0
        assert len(result.bullish_indicators) < len(result.bearish_indicators)

    @pytest.mark.asyncio
    async def test_analyze_news_sentiment_mixed(self):
        """Test sentiment analysis with mixed news."""
        service = FinancialSentimentService()
        
        news_items = [
            {
                "title": "Bitcoin Shows Mixed Signals",
                "content": "Bullish technical patterns but bearish sentiment in derivatives"
            },
            {
                "title": "Crypto Market Consolidation",
                "content": "Neutral price action with sideways trading range"
            }
        ]
        
        result = await service.analyze_news_sentiment(news_items)
        
        assert abs(result.sentiment_score) < 0.5  # Should be relatively neutral
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_analyze_news_sentiment_volatility(self):
        """Test sentiment analysis with volatility keywords."""
        service = FinancialSentimentService()
        
        news_items = [
            {
                "title": "Bitcoin Volatility Spikes Amid Uncertain Market",
                "content": "Volatile trading and choppy price action creates uncertainty"
            }
        ]
        
        result = await service.analyze_news_sentiment(news_items)
        
        assert len(result.volatility_signals) > 0
        # Volatility should reduce confidence
        assert result.confidence < 1.0

    @pytest.mark.asyncio
    async def test_analyze_news_sentiment_exception_handling(self):
        """Test sentiment analysis error handling."""
        service = FinancialSentimentService()
        
        # Test with malformed news items
        news_items = [
            {"title": None, "content": None},  # Invalid data
            {"invalid_key": "invalid_value"}  # Missing expected keys
        ]
        
        result = await service.analyze_news_sentiment(news_items)
        
        # Should handle errors gracefully
        assert isinstance(result, SentimentResult)
        assert result.sentiment_score == 0.0

    def test_extract_crypto_indicators_basic(self):
        """Test basic crypto indicators extraction."""
        service = FinancialSentimentService()
        
        text = """
        Bitcoin breaks out above $50,000 resistance with strong bullish momentum.
        Support levels at $45,000 and $47,000 remain intact. High volume confirms
        the breakout pattern. Institutional adoption continues growing.
        """
        
        indicators = service.extract_crypto_indicators(text)
        
        assert indicators.trend_direction == "BULLISH"
        assert len(indicators.support_levels) > 0
        assert len(indicators.volume_indicators) > 0
        assert len(indicators.adoption_signals) > 0

    def test_extract_crypto_indicators_price_mentions(self):
        """Test price mention extraction."""
        service = FinancialSentimentService()
        
        text = "Bitcoin targets $60,000 with support at $55,000 and resistance near $65,000"
        
        indicators = service.extract_crypto_indicators(text)
        
        assert len(indicators.price_mentions) > 0
        # Check that prices are extracted
        extracted_prices = [mention['price'] for mention in indicators.price_mentions]
        assert any('60,000' in price or '60000' in price for price in extracted_prices)

    def test_extract_crypto_indicators_technical_patterns(self):
        """Test technical pattern extraction."""
        service = FinancialSentimentService()
        
        text = """
        Bitcoin forms ascending triangle pattern with RSI showing bullish divergence.
        MACD golden cross confirms the bullish technical setup.
        """
        
        indicators = service.extract_crypto_indicators(text)
        
        assert len(indicators.technical_patterns) > 0
        assert len(indicators.indicator_signals) > 0
        assert "rsi" in indicators.indicator_signals or "macd" in indicators.indicator_signals

    def test_extract_crypto_indicators_whale_activity(self):
        """Test whale activity extraction."""
        service = FinancialSentimentService()
        
        text = "Large whale transfers $100M Bitcoin to exchange, institutional accumulation continues"
        
        indicators = service.extract_crypto_indicators(text)
        
        assert len(indicators.whale_activity) > 0

    def test_extract_crypto_indicators_regulatory(self):
        """Test regulatory mention extraction."""
        service = FinancialSentimentService()
        
        text = "SEC provides regulatory clarity on Bitcoin ETF approval process"
        
        indicators = service.extract_crypto_indicators(text)
        
        assert len(indicators.regulatory_mentions) > 0

    def test_extract_nasdaq_indicators_basic(self):
        """Test basic NASDAQ indicators extraction."""
        service = FinancialSentimentService()
        
        text = """
        NASDAQ shows bullish momentum as tech stocks rally. S&P 500 reaches new highs
        while Dow Jones continues strong performance. Fed signals dovish policy stance.
        """
        
        indicators = service.extract_nasdaq_indicators(text)
        
        assert indicators.nasdaq_trend == "BULLISH"
        assert len(indicators.sp500_mentions) > 0
        assert len(indicators.dow_mentions) > 0
        assert len(indicators.fed_policy_signals) > 0

    def test_extract_nasdaq_indicators_fed_policy(self):
        """Test Fed policy signal extraction."""
        service = FinancialSentimentService()
        
        text = """
        Federal Reserve Chairman Powell signals potential rate hikes amid inflation concerns.
        Monetary policy outlook remains data dependent with interest rate decisions pending.
        """
        
        indicators = service.extract_nasdaq_indicators(text)
        
        assert len(indicators.fed_policy_signals) > 0
        assert len(indicators.interest_rate_mentions) > 0

    def test_extract_nasdaq_indicators_risk_sentiment(self):
        """Test risk sentiment extraction."""
        service = FinancialSentimentService()
        
        text = """
        Risk-on sentiment dominates markets with strong risk appetite.
        VIX volatility index spikes as uncertainty grows.
        """
        
        indicators = service.extract_nasdaq_indicators(text)
        
        assert len(indicators.risk_on_signals) > 0
        assert len(indicators.vix_mentions) > 0

    def test_extract_nasdaq_indicators_sectors(self):
        """Test sector signal extraction."""
        service = FinancialSentimentService()
        
        text = """
        Technology sector leads gains with software and semiconductor strength.
        Financial sector shows resilience with strong bank earnings.
        """
        
        indicators = service.extract_nasdaq_indicators(text)
        
        assert len(indicators.tech_sector_signals) > 0
        assert len(indicators.financial_sector_signals) > 0

    def test_calculate_correlation_score_positive(self):
        """Test correlation score calculation with positive correlation."""
        service = FinancialSentimentService()
        
        crypto_data = {
            'trend_direction': 'BULLISH',
            'momentum_signals': ['Strong momentum', 'Buying pressure'],
            'volatility_signals': []
        }
        
        nasdaq_data = {
            'nasdaq_trend': 'BULLISH',
            'risk_off_signals': [],
            'tech_sector_signals': ['Tech rally', 'AI stocks surge']
        }
        
        correlation = service.calculate_correlation_score(crypto_data, nasdaq_data)
        
        assert correlation > 0.0  # Should be positive correlation

    def test_calculate_correlation_score_negative(self):
        """Test correlation score calculation with negative correlation."""
        service = FinancialSentimentService()
        
        crypto_data = {
            'trend_direction': 'BULLISH',
            'momentum_signals': ['Strong rally'],
            'volatility_signals': ['High volatility']
        }
        
        nasdaq_data = {
            'nasdaq_trend': 'BEARISH',
            'risk_off_signals': ['Flight to safety', 'Risk aversion'],
            'tech_sector_signals': []
        }
        
        correlation = service.calculate_correlation_score(crypto_data, nasdaq_data)
        
        assert correlation < 0.0  # Should be negative correlation

    def test_calculate_correlation_score_neutral(self):
        """Test correlation score calculation with neutral correlation."""
        service = FinancialSentimentService()
        
        crypto_data = {
            'trend_direction': 'NEUTRAL',
            'momentum_signals': [],
            'volatility_signals': []
        }
        
        nasdaq_data = {
            'nasdaq_trend': 'NEUTRAL',
            'risk_off_signals': [],
            'tech_sector_signals': []
        }
        
        correlation = service.calculate_correlation_score(crypto_data, nasdaq_data)
        
        assert abs(correlation) < 0.5  # Should be relatively neutral

    def test_format_sentiment_for_llm_complete(self):
        """Test complete sentiment formatting for LLM."""
        service = FinancialSentimentService()
        
        sentiment_result = SentimentResult(
            sentiment_score=0.6,
            confidence=0.8,
            key_themes=["Bitcoin", "ETF", "Institutional"],
            bullish_indicators=["Strong demand", "Technical breakout"],
            bearish_indicators=["Regulatory concerns"],
            volatility_signals=["High options volume"]
        )
        
        crypto_indicators = CryptoIndicators(
            trend_direction="BULLISH",
            momentum_signals=["RSI momentum"],
            technical_patterns=["Bull flag"]
        )
        
        nasdaq_indicators = NasdaqIndicators(
            nasdaq_trend="BULLISH",
            fed_policy_signals=["Dovish stance"],
            risk_on_signals=["Risk appetite"]
        )
        
        sentiment_data = {
            'sentiment_result': sentiment_result,
            'crypto_indicators': crypto_indicators,
            'nasdaq_indicators': nasdaq_indicators,
            'correlation_score': 0.45
        }
        
        formatted = service.format_sentiment_for_llm(sentiment_data)
        
        assert "FINANCIAL SENTIMENT ANALYSIS" in formatted
        assert "CRYPTO MARKET INDICATORS" in formatted
        assert "TRADITIONAL MARKET INDICATORS" in formatted
        assert "MARKET CORRELATION" in formatted
        assert "TRADING IMPLICATIONS" in formatted
        assert "0.60" in formatted  # Sentiment score
        assert "BULLISH" in formatted  # Trend direction

    def test_format_sentiment_for_llm_minimal(self):
        """Test sentiment formatting with minimal data."""
        service = FinancialSentimentService()
        
        sentiment_data = {}
        
        formatted = service.format_sentiment_for_llm(sentiment_data)
        
        # Should handle empty data gracefully
        assert "FINANCIAL SENTIMENT ANALYSIS" in formatted
        assert "Error: Could not format" not in formatted

    def test_format_sentiment_for_llm_error_handling(self):
        """Test error handling in sentiment formatting."""
        service = FinancialSentimentService()
        
        # Test with invalid data that might cause errors
        sentiment_data = {
            'sentiment_result': "invalid_data",  # Should be SentimentResult object
            'crypto_indicators': None,
            'correlation_score': "invalid"
        }
        
        formatted = service.format_sentiment_for_llm(sentiment_data)
        
        # Should handle errors gracefully
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_determine_trend_direction_bullish(self):
        """Test trend direction determination - bullish."""
        service = FinancialSentimentService()
        
        text = "strong bullish rally with positive momentum and growth prospects"
        direction = service._determine_trend_direction(text)
        
        assert direction == "BULLISH"

    def test_determine_trend_direction_bearish(self):
        """Test trend direction determination - bearish."""
        service = FinancialSentimentService()
        
        text = "bearish sentiment dominates with weak support and fear in markets"
        direction = service._determine_trend_direction(text)
        
        assert direction == "BEARISH"

    def test_determine_trend_direction_neutral(self):
        """Test trend direction determination - neutral."""
        service = FinancialSentimentService()
        
        text = "market shows mixed signals with sideways price action"
        direction = service._determine_trend_direction(text)
        
        assert direction == "NEUTRAL"

    def test_extract_themes_crypto_terms(self):
        """Test theme extraction with crypto terms."""
        service = FinancialSentimentService()
        
        text = "Bitcoin ETF approval drives institutional adoption of Ethereum DeFi protocols"
        themes = service._extract_themes(text)
        
        assert "Bitcoin" in themes
        assert "Ethereum" in themes
        assert "ETF" in themes
        assert "DeFi" in themes

    def test_extract_themes_financial_terms(self):
        """Test theme extraction with financial terms."""
        service = FinancialSentimentService()
        
        text = "Federal Reserve raises rates amid inflation concerns affecting adoption trends"
        themes = service._extract_themes(text)
        
        assert "Federal Reserve" in themes
        assert "Interest Rates" in themes
        assert "Inflation" in themes
        assert "Adoption" in themes

    def test_extract_price_levels_support_resistance(self):
        """Test price level extraction."""
        service = FinancialSentimentService()
        
        text = "Bitcoin finds support at $45,000 and faces resistance near $52,000"
        support_levels = service._extract_price_levels(text, ['support', 'floor'])
        resistance_levels = service._extract_price_levels(text, ['resistance', 'ceiling'])
        
        assert 45000.0 in support_levels
        assert 52000.0 in resistance_levels

    def test_extract_price_levels_with_commas(self):
        """Test price level extraction with comma formatting."""
        service = FinancialSentimentService()
        
        text = "Strong support exists at $1,250,000 level"
        levels = service._extract_price_levels(text, ['support'])
        
        assert 1250000.0 in levels

    def test_sentiment_labels_conversion(self):
        """Test sentiment label conversion methods."""
        service = FinancialSentimentService()
        
        # Test strongly bullish
        label = service._get_sentiment_label(0.8)
        assert label == "STRONGLY BULLISH"
        
        # Test bullish
        label = service._get_sentiment_label(0.2)
        assert label == "BULLISH"
        
        # Test neutral
        label = service._get_sentiment_label(0.0)
        assert label == "NEUTRAL"
        
        # Test bearish
        label = service._get_sentiment_label(-0.2)
        assert label == "BEARISH"
        
        # Test strongly bearish
        label = service._get_sentiment_label(-0.8)
        assert label == "STRONGLY BEARISH"

    def test_correlation_labels_conversion(self):
        """Test correlation label conversion."""
        service = FinancialSentimentService()
        
        # Test strong positive
        label = service._get_correlation_label(0.8)
        assert label == "STRONG POSITIVE"
        
        # Test moderate positive
        label = service._get_correlation_label(0.3)
        assert label == "MODERATE POSITIVE"
        
        # Test weak/uncorrelated
        label = service._get_correlation_label(0.1)
        assert label == "WEAK/UNCORRELATED"
        
        # Test moderate negative
        label = service._get_correlation_label(-0.3)
        assert label == "MODERATE NEGATIVE"
        
        # Test strong negative
        label = service._get_correlation_label(-0.8)
        assert label == "STRONG NEGATIVE"

    def test_generate_trading_implications_bullish(self):
        """Test trading implications generation for bullish sentiment."""
        service = FinancialSentimentService()
        
        sentiment_result = SentimentResult(
            sentiment_score=0.7,
            confidence=0.9
        )
        
        sentiment_data = {
            'sentiment_result': sentiment_result,
            'correlation_score': 0.3
        }
        
        implications = service._generate_trading_implications(sentiment_data)
        
        assert "bullish" in implications.lower()
        assert "long" in implications.lower()
        assert "high confidence" in implications.lower()

    def test_generate_trading_implications_bearish(self):
        """Test trading implications generation for bearish sentiment."""
        service = FinancialSentimentService()
        
        sentiment_result = SentimentResult(
            sentiment_score=-0.6,
            confidence=0.8
        )
        
        sentiment_data = {
            'sentiment_result': sentiment_result,
            'correlation_score': 0.8
        }
        
        implications = service._generate_trading_implications(sentiment_data)
        
        assert "bearish" in implications.lower()
        assert "caution" in implications.lower() or "short" in implications.lower()
        assert "correlation" in implications.lower()

    def test_generate_trading_implications_low_confidence(self):
        """Test trading implications with low confidence."""
        service = FinancialSentimentService()
        
        sentiment_result = SentimentResult(
            sentiment_score=0.5,
            confidence=0.2
        )
        
        sentiment_data = {
            'sentiment_result': sentiment_result,
            'correlation_score': 0.1
        }
        
        implications = service._generate_trading_implications(sentiment_data)
        
        assert "low confidence" in implications.lower()
        assert "waiting" in implications.lower() or "clearer" in implications.lower()

    @pytest.mark.asyncio
    async def test_concurrent_sentiment_analysis(self):
        """Test concurrent processing of sentiment analysis."""
        service = FinancialSentimentService()
        
        news_items = [
            {"title": f"News item {i}", "content": f"Content {i}"} 
            for i in range(10)
        ]
        
        # This should process items concurrently
        result = await service.analyze_news_sentiment(news_items)
        
        assert isinstance(result, SentimentResult)
        assert result.confidence >= 0.0

    def test_aggregation_weighted_sentiment(self):
        """Test weighted sentiment aggregation."""
        service = FinancialSentimentService()
        
        # Mock individual results with different confidence levels
        individual_results = [
            {
                'sentiment_score': 0.8,
                'confidence': 0.9,
                'bullish_count': 3,
                'bearish_count': 0,
                'volatility_count': 0,
                'text': 'bullish news item'
            },
            {
                'sentiment_score': -0.2,
                'confidence': 0.3,
                'bullish_count': 0,
                'bearish_count': 1,
                'volatility_count': 0,
                'text': 'slightly bearish news'
            }
        ]
        
        news_items = [
            {"title": "Bullish News"},
            {"title": "Bearish News"}
        ]
        
        result = service._aggregate_sentiment_results(individual_results, news_items)
        
        # High confidence bullish should dominate
        assert result.sentiment_score > 0.0
        assert result.confidence > 0.0
        assert len(result.bullish_indicators) > 0

    def test_empty_aggregation(self):
        """Test aggregation with empty results."""
        service = FinancialSentimentService()
        
        result = service._aggregate_sentiment_results([], [])
        
        assert result.sentiment_score == 0.0
        assert result.confidence == 0.0


# Fixtures for testing
@pytest.fixture
def sample_bullish_news():
    """Sample bullish news items for testing."""
    return [
        {
            "title": "Bitcoin Surges Past $60,000 on Institutional Demand",
            "content": """
            Bitcoin reached a new milestone today, breaking through the $60,000 resistance level
            with strong bullish momentum. Institutional adoption continues to drive positive
            sentiment as major corporations add Bitcoin to their treasury reserves.
            Technical indicators show a clear bullish trend with RSI momentum building.
            """
        },
        {
            "title": "Ethereum DeFi Protocols See Massive Growth",
            "content": """
            Ethereum's decentralized finance ecosystem is experiencing unprecedented growth
            with bullish fundamentals supporting the rally. Smart contract adoption
            accelerates as institutional interest in DeFi protocols surges.
            """
        }
    ]


@pytest.fixture
def sample_bearish_news():
    """Sample bearish news items for testing."""
    return [
        {
            "title": "Crypto Markets Crash on Regulatory Fears",
            "content": """
            Cryptocurrency markets experienced a sharp sell-off today as regulatory
            concerns weighed heavily on sentiment. Bitcoin broke below key support
            levels amid bearish momentum and panic selling from retail investors.
            Fear dominates the market as uncertainty grows.
            """
        },
        {
            "title": "DeFi Hack Triggers Market-Wide Liquidations",
            "content": """
            A major DeFi protocol exploit led to cascading liquidations across
            crypto markets. Bearish sentiment accelerated as weak hands capitulated
            and risk-off sentiment spread to traditional markets.
            """
        }
    ]


@pytest.fixture
def sample_mixed_news():
    """Sample mixed sentiment news items for testing."""
    return [
        {
            "title": "Bitcoin Shows Mixed Signals Amid Volatile Trading",
            "content": """
            Bitcoin price action remains choppy with mixed signals from technical
            indicators. While some bullish patterns emerge, bearish divergences
            in momentum suggest caution. Market volatility remains elevated.
            """
        },
        {
            "title": "Crypto Regulation: Positive and Negative Developments",
            "content": """
            Mixed regulatory developments create uncertainty in crypto markets.
            While some jurisdictions provide clarity and adoption support,
            others impose restrictions causing bearish sentiment.
            """
        }
    ]


class TestFinancialSentimentServiceIntegration:
    """Integration tests for FinancialSentimentService with sample data."""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow_bullish(self, sample_bullish_news):
        """Test complete analysis workflow with bullish news."""
        service = FinancialSentimentService()
        
        # Analyze sentiment
        sentiment = await service.analyze_news_sentiment(sample_bullish_news)
        
        # Extract indicators
        combined_text = " ".join([
            item["title"] + " " + item["content"] 
            for item in sample_bullish_news
        ])
        
        crypto_indicators = service.extract_crypto_indicators(combined_text)
        nasdaq_indicators = service.extract_nasdaq_indicators(combined_text)
        
        # Calculate correlation
        correlation = service.calculate_correlation_score(
            crypto_indicators.dict(), 
            nasdaq_indicators.dict()
        )
        
        # Format for LLM
        sentiment_data = {
            'sentiment_result': sentiment,
            'crypto_indicators': crypto_indicators,
            'nasdaq_indicators': nasdaq_indicators,
            'correlation_score': correlation
        }
        
        formatted = service.format_sentiment_for_llm(sentiment_data)
        
        # Validate results
        assert sentiment.sentiment_score > 0.0  # Should be bullish
        assert sentiment.confidence > 0.0
        assert len(sentiment.bullish_indicators) > 0
        assert crypto_indicators.trend_direction == "BULLISH"
        assert "BULLISH" in formatted
        assert "institutional" in formatted.lower()

    @pytest.mark.asyncio
    async def test_full_analysis_workflow_bearish(self, sample_bearish_news):
        """Test complete analysis workflow with bearish news."""
        service = FinancialSentimentService()
        
        sentiment = await service.analyze_news_sentiment(sample_bearish_news)
        
        combined_text = " ".join([
            item["title"] + " " + item["content"] 
            for item in sample_bearish_news
        ])
        
        crypto_indicators = service.extract_crypto_indicators(combined_text)
        nasdaq_indicators = service.extract_nasdaq_indicators(combined_text)
        
        correlation = service.calculate_correlation_score(
            crypto_indicators.dict(), 
            nasdaq_indicators.dict()
        )
        
        sentiment_data = {
            'sentiment_result': sentiment,
            'crypto_indicators': crypto_indicators,
            'nasdaq_indicators': nasdaq_indicators,
            'correlation_score': correlation
        }
        
        formatted = service.format_sentiment_for_llm(sentiment_data)
        
        # Validate bearish results
        assert sentiment.sentiment_score < 0.0  # Should be bearish
        assert len(sentiment.bearish_indicators) > 0
        assert crypto_indicators.trend_direction == "BEARISH"
        assert "BEARISH" in formatted
        assert "regulatory" in formatted.lower() or "fear" in formatted.lower()

    @pytest.mark.asyncio
    async def test_full_analysis_workflow_mixed(self, sample_mixed_news):
        """Test complete analysis workflow with mixed sentiment news."""
        service = FinancialSentimentService()
        
        sentiment = await service.analyze_news_sentiment(sample_mixed_news)
        
        combined_text = " ".join([
            item["title"] + " " + item["content"] 
            for item in sample_mixed_news
        ])
        
        crypto_indicators = service.extract_crypto_indicators(combined_text)
        
        # Validate mixed/neutral results
        assert abs(sentiment.sentiment_score) < 0.5  # Should be relatively neutral
        assert len(sentiment.volatility_signals) > 0  # Should detect volatility
        assert crypto_indicators.trend_direction == "NEUTRAL"

    def test_performance_large_dataset(self):
        """Test performance with larger dataset."""
        service = FinancialSentimentService()
        
        # Create larger text sample
        large_text = """
        Bitcoin cryptocurrency market analysis shows mixed technical indicators
        with bullish momentum building despite bearish sentiment from regulatory
        concerns. Ethereum DeFi protocols continue adoption growth while NASDAQ
        technology sector shows strength. Federal Reserve policy remains data
        dependent with inflation concerns affecting interest rate decisions.
        Institutional adoption accelerates as volatility expectations remain
        elevated amid geopolitical uncertainty.
        """ * 50  # Repeat to create larger text
        
        # Test extraction performance
        crypto_indicators = service.extract_crypto_indicators(large_text)
        nasdaq_indicators = service.extract_nasdaq_indicators(large_text)
        
        # Should handle large text efficiently
        assert isinstance(crypto_indicators, CryptoIndicators)
        assert isinstance(nasdaq_indicators, NasdaqIndicators)
        
        # Should extract meaningful data
        assert len(crypto_indicators.momentum_signals) > 0
        assert len(nasdaq_indicators.fed_policy_signals) > 0