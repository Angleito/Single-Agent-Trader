"""Market context analysis for crypto/NASDAQ correlation and market regime detection."""

import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class MarketRegimeType(str, Enum):
    """Market regime classification."""

    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class CorrelationStrength(str, Enum):
    """Correlation strength classification."""

    VERY_STRONG = "VERY_STRONG"     # |r| > 0.8
    STRONG = "STRONG"               # 0.6 < |r| <= 0.8
    MODERATE = "MODERATE"           # 0.4 < |r| <= 0.6
    WEAK = "WEAK"                   # 0.2 < |r| <= 0.4
    VERY_WEAK = "VERY_WEAK"         # |r| <= 0.2


class SentimentLevel(str, Enum):
    """Risk sentiment levels."""

    EXTREME_FEAR = "EXTREME_FEAR"
    FEAR = "FEAR"
    NEUTRAL = "NEUTRAL"
    GREED = "GREED"
    EXTREME_GREED = "EXTREME_GREED"


class CorrelationAnalysis(BaseModel):
    """Crypto/NASDAQ correlation analysis results."""

    model_config = ConfigDict(frozen=True)

    correlation_coefficient: float = Field(
        ge=-1.0, le=1.0, description="Pearson correlation coefficient"
    )
    correlation_strength: CorrelationStrength = Field(
        description="Categorical strength of correlation"
    )
    direction: str = Field(
        description="Positive, negative, or uncorrelated"
    )
    p_value: float = Field(
        ge=0.0, le=1.0, description="Statistical significance p-value"
    )
    is_significant: bool = Field(
        description="Whether correlation is statistically significant (p < 0.05)"
    )
    sample_size: int = Field(
        ge=0, description="Number of data points used in analysis"
    )
    rolling_correlation_24h: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="24-hour rolling correlation"
    )
    rolling_correlation_7d: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="7-day rolling correlation"
    )
    correlation_stability: float = Field(
        ge=0.0, le=1.0, description="Stability of correlation over time"
    )
    regime_dependent_correlation: dict[str, float] = Field(
        default_factory=dict, description="Correlation by market regime"
    )
    reliability_score: float = Field(
        ge=0.0, le=1.0, description="Overall reliability of correlation analysis"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Analysis timestamp"
    )


class MarketRegime(BaseModel):
    """Market regime detection results."""

    model_config = ConfigDict(frozen=True)

    regime_type: MarketRegimeType = Field(
        description="Current market regime classification"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in regime classification"
    )
    key_drivers: list[str] = Field(
        default_factory=list, description="Primary factors driving current regime"
    )
    fed_policy_stance: str = Field(
        default="NEUTRAL", description="Federal Reserve policy stance"
    )
    inflation_environment: str = Field(
        default="STABLE", description="Current inflation environment"
    )
    interest_rate_trend: str = Field(
        default="STABLE", description="Interest rate trend direction"
    )
    geopolitical_risk_level: str = Field(
        default="LOW", description="Geopolitical risk assessment"
    )
    crypto_adoption_momentum: str = Field(
        default="MODERATE", description="Crypto adoption trend"
    )
    institutional_sentiment: str = Field(
        default="NEUTRAL", description="Institutional investor sentiment"
    )
    regulatory_environment: str = Field(
        default="STABLE", description="Regulatory environment assessment"
    )
    market_volatility_regime: str = Field(
        default="NORMAL", description="Volatility regime classification"
    )
    liquidity_conditions: str = Field(
        default="NORMAL", description="Market liquidity conditions"
    )
    duration_days: int | None = Field(
        default=None, ge=0, description="Days in current regime"
    )
    regime_change_probability: float = Field(
        ge=0.0, le=1.0, description="Probability of regime change in next 30 days"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Analysis timestamp"
    )


class RiskSentiment(BaseModel):
    """Risk sentiment analysis results."""

    model_config = ConfigDict(frozen=True)

    fear_greed_index: float = Field(
        ge=0.0, le=100.0, description="Fear & Greed index (0=extreme fear, 100=extreme greed)"
    )
    sentiment_level: SentimentLevel = Field(
        description="Categorical sentiment level"
    )
    volatility_expectation: float = Field(
        ge=0.0, description="Expected volatility percentage"
    )
    market_stress_indicator: float = Field(
        ge=0.0, le=1.0, description="Market stress level (0=calm, 1=extreme stress)"
    )
    vix_level: float | None = Field(
        default=None, ge=0.0, description="VIX volatility index level"
    )
    crypto_fear_greed: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Crypto-specific fear & greed index"
    )
    social_sentiment_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Social media sentiment score"
    )
    news_sentiment_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="News sentiment score"
    )
    funding_rates_sentiment: float | None = Field(
        default=None, description="Funding rates sentiment indicator"
    )
    options_flow_sentiment: str | None = Field(
        default=None, description="Options flow sentiment (BULLISH/BEARISH/NEUTRAL)"
    )
    insider_activity: str | None = Field(
        default=None, description="Insider/institutional activity sentiment"
    )
    retail_sentiment: str | None = Field(
        default=None, description="Retail investor sentiment"
    )
    sentiment_divergence: bool = Field(
        default=False, description="Whether sentiment diverges from price action"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Analysis timestamp"
    )


class MomentumAlignment(BaseModel):
    """Momentum alignment analysis between crypto and traditional markets."""

    model_config = ConfigDict(frozen=True)

    directional_alignment: float = Field(
        ge=-1.0, le=1.0, description="Directional momentum alignment (-1=opposite, 1=aligned)"
    )
    strength_alignment: float = Field(
        ge=0.0, le=1.0, description="Momentum strength alignment"
    )
    crypto_momentum_score: float = Field(
        ge=-1.0, le=1.0, description="Crypto momentum score"
    )
    nasdaq_momentum_score: float = Field(
        ge=-1.0, le=1.0, description="NASDAQ momentum score"
    )
    momentum_divergences: list[str] = Field(
        default_factory=list, description="Identified momentum divergences"
    )
    trend_strength_crypto: float = Field(
        ge=0.0, le=1.0, description="Crypto trend strength"
    )
    trend_strength_nasdaq: float = Field(
        ge=0.0, le=1.0, description="NASDAQ trend strength"
    )
    volume_momentum_alignment: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Volume momentum alignment"
    )
    momentum_sustainability: float = Field(
        ge=0.0, le=1.0, description="Momentum sustainability score"
    )
    momentum_regime: str = Field(
        default="NORMAL", description="Momentum regime (ACCELERATION/DECELERATION/NORMAL)"
    )
    cross_asset_momentum_flow: str = Field(
        default="NEUTRAL", description="Cross-asset momentum flow direction"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Analysis timestamp"
    )


class MarketContextAnalyzer:
    """
    Comprehensive market context analyzer for crypto/NASDAQ correlation and regime detection.
    
    Provides analysis of market correlations, regime detection, risk sentiment assessment,
    and momentum alignment to enhance trading bot decision-making with broader market context.
    """

    def __init__(self):
        """Initialize the market context analyzer."""
        self._fed_policy_keywords = {
            'hawkish': ['rate hike', 'tightening', 'inflation fight', 'aggressive policy'],
            'dovish': ['rate cut', 'easing', 'accommodative', 'stimulus'],
            'neutral': ['data dependent', 'wait and see', 'monitoring', 'balanced']
        }

        self._inflation_keywords = {
            'high': ['inflation surge', 'price pressures', 'cpi spike', 'inflation concern'],
            'low': ['disinflation', 'price stability', 'low inflation', 'deflationary'],
            'stable': ['inflation target', 'price stability', 'contained inflation']
        }

        self._geopolitical_keywords = {
            'high': ['war', 'conflict', 'sanctions', 'trade war', 'geopolitical tension'],
            'medium': ['uncertainty', 'diplomatic', 'negotiation', 'tension'],
            'low': ['peace', 'stability', 'cooperation', 'resolution']
        }

        self._crypto_adoption_keywords = {
            'high': ['institutional adoption', 'mainstream acceptance', 'etf approval'],
            'medium': ['growing adoption', 'increased interest', 'pilot programs'],
            'low': ['skepticism', 'barriers', 'resistance', 'regulatory concerns']
        }

        logger.info("Market context analyzer initialized")

    async def analyze_crypto_nasdaq_correlation(
        self,
        crypto_data: dict[str, Any],
        nasdaq_data: dict[str, Any]
    ) -> CorrelationAnalysis:
        """
        Analyze correlation between crypto and NASDAQ data.
        
        Args:
            crypto_data: Dictionary containing crypto price and indicator data
            nasdaq_data: Dictionary containing NASDAQ price and indicator data
            
        Returns:
            CorrelationAnalysis with comprehensive correlation metrics
        """
        try:
            # Extract price series
            crypto_prices = self._extract_price_series(crypto_data)
            nasdaq_prices = self._extract_price_series(nasdaq_data)

            if len(crypto_prices) < 30 or len(nasdaq_prices) < 30:
                logger.warning("Insufficient data for correlation analysis")
                return CorrelationAnalysis(
                    correlation_coefficient=0.0,
                    correlation_strength=CorrelationStrength.VERY_WEAK,
                    direction="INSUFFICIENT_DATA",
                    p_value=1.0,
                    is_significant=False,
                    sample_size=min(len(crypto_prices), len(nasdaq_prices)),
                    correlation_stability=0.0,
                    reliability_score=0.0
                )

            # Align data by timestamp
            crypto_aligned, nasdaq_aligned = self._align_time_series(crypto_prices, nasdaq_prices)

            if len(crypto_aligned) < 20:
                logger.warning("Insufficient aligned data for correlation analysis")
                return CorrelationAnalysis(
                    correlation_coefficient=0.0,
                    correlation_strength=CorrelationStrength.VERY_WEAK,
                    direction="INSUFFICIENT_ALIGNED_DATA",
                    p_value=1.0,
                    is_significant=False,
                    sample_size=len(crypto_aligned),
                    correlation_stability=0.0,
                    reliability_score=0.0
                )

            # Calculate correlation and statistical significance
            correlation_coef, p_value = pearsonr(crypto_aligned, nasdaq_aligned)

            # Handle NaN values
            if np.isnan(correlation_coef) or np.isnan(p_value):
                correlation_coef = 0.0
                p_value = 1.0

            # Calculate rolling correlations
            rolling_24h = self._calculate_rolling_correlation(crypto_aligned, nasdaq_aligned, 24)
            rolling_7d = self._calculate_rolling_correlation(crypto_aligned, nasdaq_aligned, 168)  # 7 days * 24 hours

            # Calculate correlation stability
            stability = self._calculate_correlation_stability(crypto_aligned, nasdaq_aligned)

            # Regime-dependent correlation
            regime_correlation = await self._calculate_regime_dependent_correlation(
                crypto_data, nasdaq_data, crypto_aligned, nasdaq_aligned
            )

            # Determine correlation strength and direction
            strength = self._classify_correlation_strength(abs(correlation_coef))
            direction = self._determine_correlation_direction(correlation_coef)

            # Calculate reliability score
            reliability = self._calculate_correlation_reliability(
                correlation_coef, p_value, len(crypto_aligned), stability
            )

            return CorrelationAnalysis(
                correlation_coefficient=float(correlation_coef),
                correlation_strength=strength,
                direction=direction,
                p_value=float(p_value),
                is_significant=p_value < 0.05,
                sample_size=len(crypto_aligned),
                rolling_correlation_24h=rolling_24h,
                rolling_correlation_7d=rolling_7d,
                correlation_stability=stability,
                regime_dependent_correlation=regime_correlation,
                reliability_score=reliability
            )

        except Exception as e:
            logger.error(f"Error analyzing crypto/NASDAQ correlation: {e}", exc_info=True)
            return CorrelationAnalysis(
                correlation_coefficient=0.0,
                correlation_strength=CorrelationStrength.VERY_WEAK,
                direction="ERROR",
                p_value=1.0,
                is_significant=False,
                sample_size=0,
                correlation_stability=0.0,
                reliability_score=0.0
            )

    async def detect_market_regime(self, sentiment_data: dict[str, Any]) -> MarketRegime:
        """
        Detect current market regime based on sentiment and economic data.
        
        Args:
            sentiment_data: Dictionary containing sentiment and economic indicators
            
        Returns:
            MarketRegime with comprehensive regime analysis
        """
        try:
            # Analyze Fed policy stance
            fed_stance = self._analyze_fed_policy_stance(sentiment_data)

            # Analyze inflation environment
            inflation_env = self._analyze_inflation_environment(sentiment_data)

            # Analyze interest rate trend
            rate_trend = self._analyze_interest_rate_trend(sentiment_data)

            # Assess geopolitical risk
            geo_risk = self._assess_geopolitical_risk(sentiment_data)

            # Analyze crypto adoption momentum
            crypto_adoption = self._analyze_crypto_adoption_momentum(sentiment_data)

            # Assess institutional sentiment
            institutional_sentiment = self._assess_institutional_sentiment(sentiment_data)

            # Analyze regulatory environment
            regulatory_env = self._analyze_regulatory_environment(sentiment_data)

            # Assess market volatility regime
            volatility_regime = self._assess_volatility_regime(sentiment_data)

            # Assess liquidity conditions
            liquidity_conditions = self._assess_liquidity_conditions(sentiment_data)

            # Determine overall market regime
            regime_type, confidence, key_drivers = self._determine_market_regime(
                fed_stance, inflation_env, rate_trend, geo_risk, crypto_adoption,
                institutional_sentiment, regulatory_env, volatility_regime, liquidity_conditions
            )

            # Calculate regime change probability
            regime_change_prob = self._calculate_regime_change_probability(sentiment_data)

            # Estimate regime duration
            regime_duration = self._estimate_regime_duration(sentiment_data)

            return MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                key_drivers=key_drivers,
                fed_policy_stance=fed_stance,
                inflation_environment=inflation_env,
                interest_rate_trend=rate_trend,
                geopolitical_risk_level=geo_risk,
                crypto_adoption_momentum=crypto_adoption,
                institutional_sentiment=institutional_sentiment,
                regulatory_environment=regulatory_env,
                market_volatility_regime=volatility_regime,
                liquidity_conditions=liquidity_conditions,
                duration_days=regime_duration,
                regime_change_probability=regime_change_prob
            )

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}", exc_info=True)
            return MarketRegime(
                regime_type=MarketRegimeType.UNKNOWN,
                confidence=0.0,
                key_drivers=[f"Analysis error: {str(e)}"]
            )

    async def assess_risk_sentiment(self, news_data: list[dict[str, Any]]) -> RiskSentiment:
        """
        Assess risk sentiment from news and market data.
        
        Args:
            news_data: List of news articles and market data
            
        Returns:
            RiskSentiment with comprehensive sentiment analysis
        """
        try:
            # Calculate fear & greed index
            fear_greed_index = self._calculate_fear_greed_index(news_data)

            # Determine sentiment level
            sentiment_level = self._classify_sentiment_level(fear_greed_index)

            # Calculate volatility expectation
            volatility_expectation = self._calculate_volatility_expectation(news_data)

            # Calculate market stress indicator
            market_stress = self._calculate_market_stress_indicator(news_data)

            # Extract additional sentiment metrics
            vix_level = self._extract_vix_level(news_data)
            crypto_fear_greed = self._extract_crypto_fear_greed(news_data)
            social_sentiment = self._calculate_social_sentiment(news_data)
            news_sentiment = self._calculate_news_sentiment(news_data)
            funding_rates_sentiment = self._calculate_funding_rates_sentiment(news_data)
            options_flow_sentiment = self._analyze_options_flow_sentiment(news_data)
            insider_activity = self._analyze_insider_activity(news_data)
            retail_sentiment = self._analyze_retail_sentiment(news_data)

            # Detect sentiment divergence
            sentiment_divergence = self._detect_sentiment_divergence(news_data)

            return RiskSentiment(
                fear_greed_index=fear_greed_index,
                sentiment_level=sentiment_level,
                volatility_expectation=volatility_expectation,
                market_stress_indicator=market_stress,
                vix_level=vix_level,
                crypto_fear_greed=crypto_fear_greed,
                social_sentiment_score=social_sentiment,
                news_sentiment_score=news_sentiment,
                funding_rates_sentiment=funding_rates_sentiment,
                options_flow_sentiment=options_flow_sentiment,
                insider_activity=insider_activity,
                retail_sentiment=retail_sentiment,
                sentiment_divergence=sentiment_divergence
            )

        except Exception as e:
            logger.error(f"Error assessing risk sentiment: {e}", exc_info=True)
            return RiskSentiment(
                fear_greed_index=50.0,  # Neutral
                sentiment_level=SentimentLevel.NEUTRAL,
                volatility_expectation=20.0,  # Default volatility
                market_stress_indicator=0.5   # Moderate stress
            )

    async def calculate_momentum_alignment(
        self,
        crypto_indicators: dict[str, Any],
        nasdaq_indicators: dict[str, Any]
    ) -> MomentumAlignment:
        """
        Calculate momentum alignment between crypto and NASDAQ markets.
        
        Args:
            crypto_indicators: Dictionary containing crypto momentum indicators
            nasdaq_indicators: Dictionary containing NASDAQ momentum indicators
            
        Returns:
            MomentumAlignment with comprehensive momentum analysis
        """
        try:
            # Calculate individual momentum scores
            crypto_momentum = self._calculate_crypto_momentum_score(crypto_indicators)
            nasdaq_momentum = self._calculate_nasdaq_momentum_score(nasdaq_indicators)

            # Calculate directional alignment
            directional_alignment = self._calculate_directional_alignment(
                crypto_momentum, nasdaq_momentum
            )

            # Calculate strength alignment
            strength_alignment = self._calculate_strength_alignment(
                crypto_indicators, nasdaq_indicators
            )

            # Identify momentum divergences
            divergences = self._identify_momentum_divergences(
                crypto_indicators, nasdaq_indicators
            )

            # Calculate trend strengths
            crypto_trend_strength = self._calculate_trend_strength(crypto_indicators)
            nasdaq_trend_strength = self._calculate_trend_strength(nasdaq_indicators)

            # Calculate volume momentum alignment
            volume_alignment = self._calculate_volume_momentum_alignment(
                crypto_indicators, nasdaq_indicators
            )

            # Assess momentum sustainability
            momentum_sustainability = self._assess_momentum_sustainability(
                crypto_indicators, nasdaq_indicators
            )

            # Determine momentum regime
            momentum_regime = self._determine_momentum_regime(
                crypto_indicators, nasdaq_indicators
            )

            # Analyze cross-asset momentum flow
            momentum_flow = self._analyze_cross_asset_momentum_flow(
                crypto_indicators, nasdaq_indicators
            )

            return MomentumAlignment(
                directional_alignment=directional_alignment,
                strength_alignment=strength_alignment,
                crypto_momentum_score=crypto_momentum,
                nasdaq_momentum_score=nasdaq_momentum,
                momentum_divergences=divergences,
                trend_strength_crypto=crypto_trend_strength,
                trend_strength_nasdaq=nasdaq_trend_strength,
                volume_momentum_alignment=volume_alignment,
                momentum_sustainability=momentum_sustainability,
                momentum_regime=momentum_regime,
                cross_asset_momentum_flow=momentum_flow
            )

        except Exception as e:
            logger.error(f"Error calculating momentum alignment: {e}", exc_info=True)
            return MomentumAlignment(
                directional_alignment=0.0,
                strength_alignment=0.0,
                crypto_momentum_score=0.0,
                nasdaq_momentum_score=0.0,
                momentum_divergences=[f"Analysis error: {str(e)}"],
                trend_strength_crypto=0.0,
                trend_strength_nasdaq=0.0,
                momentum_sustainability=0.0
            )

    def generate_context_summary(
        self,
        correlation: CorrelationAnalysis,
        regime: MarketRegime
    ) -> str:
        """
        Generate a comprehensive context summary for LLM consumption.
        
        Args:
            correlation: Correlation analysis results
            regime: Market regime detection results
            
        Returns:
            Formatted string summary of market context
        """
        try:
            summary_lines = [
                "=== MARKET CONTEXT ANALYSIS ===",
                "",
                "=== CORRELATION ANALYSIS ===",
                f"Crypto-NASDAQ Correlation: {correlation.correlation_coefficient:.3f} ({correlation.correlation_strength.value})",
                f"Direction: {correlation.direction}",
                f"Statistical Significance: {'Yes' if correlation.is_significant else 'No'} (p={correlation.p_value:.3f})",
                f"Sample Size: {correlation.sample_size} data points",
                f"Reliability Score: {correlation.reliability_score:.2f}",
                ""
            ]

            if correlation.rolling_correlation_24h is not None:
                summary_lines.extend([
                    f"24-Hour Rolling Correlation: {correlation.rolling_correlation_24h:.3f}",
                    f"7-Day Rolling Correlation: {correlation.rolling_correlation_7d:.3f}",
                    f"Correlation Stability: {correlation.correlation_stability:.2f}",
                    ""
                ])

            if correlation.regime_dependent_correlation:
                summary_lines.append("Regime-Dependent Correlations:")
                for regime_name, corr_value in correlation.regime_dependent_correlation.items():
                    summary_lines.append(f"  • {regime_name}: {corr_value:.3f}")
                summary_lines.append("")

            summary_lines.extend([
                "=== MARKET REGIME ANALYSIS ===",
                f"Current Regime: {regime.regime_type.value}",
                f"Confidence: {regime.confidence:.2f}",
                f"Fed Policy Stance: {regime.fed_policy_stance}",
                f"Inflation Environment: {regime.inflation_environment}",
                f"Interest Rate Trend: {regime.interest_rate_trend}",
                f"Geopolitical Risk: {regime.geopolitical_risk_level}",
                f"Crypto Adoption Momentum: {regime.crypto_adoption_momentum}",
                f"Institutional Sentiment: {regime.institutional_sentiment}",
                f"Regulatory Environment: {regime.regulatory_environment}",
                f"Volatility Regime: {regime.market_volatility_regime}",
                f"Liquidity Conditions: {regime.liquidity_conditions}",
                ""
            ])

            if regime.duration_days is not None:
                summary_lines.append(f"Regime Duration: {regime.duration_days} days")

            summary_lines.extend([
                f"Regime Change Probability (30d): {regime.regime_change_probability:.2%}",
                ""
            ])

            if regime.key_drivers:
                summary_lines.append("Key Market Drivers:")
                for driver in regime.key_drivers[:5]:
                    summary_lines.append(f"  • {driver}")
                summary_lines.append("")

            summary_lines.extend([
                "=== TRADING IMPLICATIONS ===",
                self._generate_trading_implications(correlation, regime),
                ""
            ])

            return "\n".join(summary_lines)

        except Exception as e:
            logger.error(f"Error generating context summary: {e}", exc_info=True)
            return "Error: Could not generate market context summary"

    # Helper methods

    def _extract_price_series(self, data: dict[str, Any]) -> list[float]:
        """Extract price series from market data."""
        try:
            if 'prices' in data:
                return [float(p) for p in data['prices']]
            elif 'ohlcv' in data:
                return [float(candle['close']) for candle in data['ohlcv']]
            elif 'candles' in data:
                return [float(candle['close']) for candle in data['candles']]
            else:
                return []
        except (ValueError, KeyError, TypeError):
            return []

    def _align_time_series(
        self,
        crypto_prices: list[float],
        nasdaq_prices: list[float]
    ) -> tuple[list[float], list[float]]:
        """Align two time series by length."""
        min_length = min(len(crypto_prices), len(nasdaq_prices))
        return crypto_prices[-min_length:], nasdaq_prices[-min_length:]

    def _calculate_rolling_correlation(
        self,
        series1: list[float],
        series2: list[float],
        window: int
    ) -> float | None:
        """Calculate rolling correlation over specified window."""
        try:
            if len(series1) < window or len(series2) < window:
                return None

            recent_series1 = series1[-window:]
            recent_series2 = series2[-window:]

            corr, _ = pearsonr(recent_series1, recent_series2)
            return float(corr) if not np.isnan(corr) else None
        except Exception:
            return None

    def _calculate_correlation_stability(
        self,
        series1: list[float],
        series2: list[float]
    ) -> float:
        """Calculate correlation stability over time."""
        try:
            if len(series1) < 60:  # Need at least 60 data points
                return 0.0

            # Calculate correlation for different windows
            correlations = []
            for i in range(30, len(series1), 10):
                window_series1 = series1[i-30:i]
                window_series2 = series2[i-30:i]

                if len(window_series1) >= 30:
                    corr, _ = pearsonr(window_series1, window_series2)
                    if not np.isnan(corr):
                        correlations.append(corr)

            if len(correlations) < 2:
                return 0.0

            # Stability is inverse of standard deviation
            stability = 1.0 - min(np.std(correlations), 1.0)
            return max(0.0, stability)

        except Exception:
            return 0.0

    async def _calculate_regime_dependent_correlation(
        self,
        crypto_data: dict[str, Any],
        nasdaq_data: dict[str, Any],
        crypto_aligned: list[float],
        nasdaq_aligned: list[float]
    ) -> dict[str, float]:
        """Calculate correlation for different market regimes."""
        try:
            regime_correlations = {}

            # For now, return a simple breakdown
            # In a full implementation, this would identify regime periods
            # and calculate correlations for each regime

            if len(crypto_aligned) >= 60:
                # High volatility periods (top 25% of price changes)
                returns_crypto = [
                    (crypto_aligned[i] - crypto_aligned[i-1]) / crypto_aligned[i-1]
                    for i in range(1, len(crypto_aligned))
                ]
                returns_nasdaq = [
                    (nasdaq_aligned[i] - nasdaq_aligned[i-1]) / nasdaq_aligned[i-1]
                    for i in range(1, len(nasdaq_aligned))
                ]

                # Calculate volatility threshold
                volatilities = [abs(r) for r in returns_crypto]
                volatility_threshold = np.percentile(volatilities, 75)

                # High volatility correlation
                high_vol_crypto = []
                high_vol_nasdaq = []
                low_vol_crypto = []
                low_vol_nasdaq = []

                for i, vol in enumerate(volatilities):
                    if vol > volatility_threshold:
                        high_vol_crypto.append(returns_crypto[i])
                        high_vol_nasdaq.append(returns_nasdaq[i])
                    else:
                        low_vol_crypto.append(returns_crypto[i])
                        low_vol_nasdaq.append(returns_nasdaq[i])

                if len(high_vol_crypto) >= 10:
                    corr, _ = pearsonr(high_vol_crypto, high_vol_nasdaq)
                    if not np.isnan(corr):
                        regime_correlations["HIGH_VOLATILITY"] = float(corr)

                if len(low_vol_crypto) >= 10:
                    corr, _ = pearsonr(low_vol_crypto, low_vol_nasdaq)
                    if not np.isnan(corr):
                        regime_correlations["LOW_VOLATILITY"] = float(corr)

            return regime_correlations

        except Exception:
            return {}

    def _classify_correlation_strength(self, abs_correlation: float) -> CorrelationStrength:
        """Classify correlation strength based on absolute value."""
        if abs_correlation > 0.8:
            return CorrelationStrength.VERY_STRONG
        elif abs_correlation > 0.6:
            return CorrelationStrength.STRONG
        elif abs_correlation > 0.4:
            return CorrelationStrength.MODERATE
        elif abs_correlation > 0.2:
            return CorrelationStrength.WEAK
        else:
            return CorrelationStrength.VERY_WEAK

    def _determine_correlation_direction(self, correlation: float) -> str:
        """Determine correlation direction."""
        if correlation > 0.1:
            return "POSITIVE"
        elif correlation < -0.1:
            return "NEGATIVE"
        else:
            return "UNCORRELATED"

    def _calculate_correlation_reliability(
        self,
        correlation: float,
        p_value: float,
        sample_size: int,
        stability: float
    ) -> float:
        """Calculate overall reliability of correlation analysis."""
        try:
            reliability_factors = []

            # Statistical significance factor
            if p_value < 0.01:
                reliability_factors.append(1.0)
            elif p_value < 0.05:
                reliability_factors.append(0.8)
            elif p_value < 0.1:
                reliability_factors.append(0.6)
            else:
                reliability_factors.append(0.3)

            # Sample size factor
            if sample_size > 100:
                reliability_factors.append(1.0)
            elif sample_size > 50:
                reliability_factors.append(0.8)
            elif sample_size > 30:
                reliability_factors.append(0.6)
            else:
                reliability_factors.append(0.4)

            # Correlation magnitude factor
            reliability_factors.append(min(abs(correlation) * 2, 1.0))

            # Stability factor
            reliability_factors.append(stability)

            return sum(reliability_factors) / len(reliability_factors)

        except Exception:
            return 0.0

    def _analyze_fed_policy_stance(self, sentiment_data: dict[str, Any]) -> str:
        """Analyze Federal Reserve policy stance from sentiment data."""
        try:
            text = str(sentiment_data.get('text', '') + ' ' +
                     ' '.join(sentiment_data.get('news_headlines', []))).lower()

            hawkish_score = sum(1 for keyword in self._fed_policy_keywords['hawkish']
                              for phrase in keyword.split() if phrase in text)
            dovish_score = sum(1 for keyword in self._fed_policy_keywords['dovish']
                             for phrase in keyword.split() if phrase in text)

            if hawkish_score > dovish_score * 1.5:
                return "HAWKISH"
            elif dovish_score > hawkish_score * 1.5:
                return "DOVISH"
            else:
                return "NEUTRAL"

        except Exception:
            return "NEUTRAL"

    def _analyze_inflation_environment(self, sentiment_data: dict[str, Any]) -> str:
        """Analyze inflation environment from sentiment data."""
        try:
            text = str(sentiment_data.get('text', '') + ' ' +
                     ' '.join(sentiment_data.get('news_headlines', []))).lower()

            high_inflation_indicators = sum(1 for keyword in self._inflation_keywords['high']
                                          if keyword in text)
            low_inflation_indicators = sum(1 for keyword in self._inflation_keywords['low']
                                         if keyword in text)

            if high_inflation_indicators > low_inflation_indicators:
                return "HIGH"
            elif low_inflation_indicators > high_inflation_indicators:
                return "LOW"
            else:
                return "STABLE"

        except Exception:
            return "STABLE"

    def _analyze_interest_rate_trend(self, sentiment_data: dict[str, Any]) -> str:
        """Analyze interest rate trend from sentiment data."""
        try:
            text = str(sentiment_data.get('text', '') + ' ' +
                     ' '.join(sentiment_data.get('news_headlines', []))).lower()

            if any(phrase in text for phrase in ['rate hike', 'raising rates', 'higher rates']):
                return "RISING"
            elif any(phrase in text for phrase in ['rate cut', 'lowering rates', 'rate reduction']):
                return "FALLING"
            else:
                return "STABLE"

        except Exception:
            return "STABLE"

    def _assess_geopolitical_risk(self, sentiment_data: dict[str, Any]) -> str:
        """Assess geopolitical risk level from sentiment data."""
        try:
            text = str(sentiment_data.get('text', '') + ' ' +
                     ' '.join(sentiment_data.get('news_headlines', []))).lower()

            high_risk_indicators = sum(1 for keyword in self._geopolitical_keywords['high']
                                     if keyword in text)
            low_risk_indicators = sum(1 for keyword in self._geopolitical_keywords['low']
                                    if keyword in text)

            if high_risk_indicators > 2:
                return "HIGH"
            elif low_risk_indicators > high_risk_indicators:
                return "LOW"
            else:
                return "MEDIUM"

        except Exception:
            return "MEDIUM"

    def _analyze_crypto_adoption_momentum(self, sentiment_data: dict[str, Any]) -> str:
        """Analyze crypto adoption momentum from sentiment data."""
        try:
            text = str(sentiment_data.get('text', '') + ' ' +
                     ' '.join(sentiment_data.get('news_headlines', []))).lower()

            high_adoption_indicators = sum(1 for keyword in self._crypto_adoption_keywords['high']
                                         if keyword in text)
            low_adoption_indicators = sum(1 for keyword in self._crypto_adoption_keywords['low']
                                        if keyword in text)

            if high_adoption_indicators > low_adoption_indicators:
                return "HIGH"
            elif low_adoption_indicators > high_adoption_indicators:
                return "LOW"
            else:
                return "MODERATE"

        except Exception:
            return "MODERATE"

    def _assess_institutional_sentiment(self, sentiment_data: dict[str, Any]) -> str:
        """Assess institutional sentiment from sentiment data."""
        try:
            text = str(sentiment_data.get('text', '') + ' ' +
                     ' '.join(sentiment_data.get('news_headlines', []))).lower()

            positive_indicators = ['institutional buying', 'hedge fund interest', 'corporate adoption']
            negative_indicators = ['institutional selling', 'hedge fund exodus', 'corporate concerns']

            positive_score = sum(1 for indicator in positive_indicators if indicator in text)
            negative_score = sum(1 for indicator in negative_indicators if indicator in text)

            if positive_score > negative_score:
                return "POSITIVE"
            elif negative_score > positive_score:
                return "NEGATIVE"
            else:
                return "NEUTRAL"

        except Exception:
            return "NEUTRAL"

    def _analyze_regulatory_environment(self, sentiment_data: dict[str, Any]) -> str:
        """Analyze regulatory environment from sentiment data."""
        try:
            text = str(sentiment_data.get('text', '') + ' ' +
                     ' '.join(sentiment_data.get('news_headlines', []))).lower()

            if any(phrase in text for phrase in ['regulatory clarity', 'favorable regulation', 'etf approval']):
                return "FAVORABLE"
            elif any(phrase in text for phrase in ['regulatory crackdown', 'ban', 'restriction']):
                return "RESTRICTIVE"
            else:
                return "STABLE"

        except Exception:
            return "STABLE"

    def _assess_volatility_regime(self, sentiment_data: dict[str, Any]) -> str:
        """Assess market volatility regime from sentiment data."""
        try:
            vix_level = sentiment_data.get('vix_level', 20.0)

            if vix_level > 30:
                return "HIGH"
            elif vix_level > 20:
                return "ELEVATED"
            else:
                return "NORMAL"

        except Exception:
            return "NORMAL"

    def _assess_liquidity_conditions(self, sentiment_data: dict[str, Any]) -> str:
        """Assess market liquidity conditions from sentiment data."""
        try:
            text = str(sentiment_data.get('text', '') + ' ' +
                     ' '.join(sentiment_data.get('news_headlines', []))).lower()

            if any(phrase in text for phrase in ['liquidity crisis', 'illiquid', 'tight liquidity']):
                return "TIGHT"
            elif any(phrase in text for phrase in ['ample liquidity', 'liquid markets', 'easy liquidity']):
                return "ABUNDANT"
            else:
                return "NORMAL"

        except Exception:
            return "NORMAL"

    def _determine_market_regime(
        self,
        fed_stance: str,
        inflation_env: str,
        rate_trend: str,
        geo_risk: str,
        crypto_adoption: str,
        institutional_sentiment: str,
        regulatory_env: str,
        volatility_regime: str,
        liquidity_conditions: str
    ) -> tuple[MarketRegimeType, float, list[str]]:
        """Determine overall market regime from component analyses."""
        try:
            risk_on_factors = []
            risk_off_factors = []
            key_drivers = []

            # Fed policy
            if fed_stance == "DOVISH":
                risk_on_factors.append(1.0)
                key_drivers.append("Dovish Fed policy supports risk assets")
            elif fed_stance == "HAWKISH":
                risk_off_factors.append(1.0)
                key_drivers.append("Hawkish Fed policy pressures risk assets")

            # Inflation
            if inflation_env == "HIGH":
                risk_off_factors.append(0.8)
                key_drivers.append("High inflation environment")
            elif inflation_env == "LOW":
                risk_on_factors.append(0.6)

            # Interest rates
            if rate_trend == "RISING":
                risk_off_factors.append(0.8)
                key_drivers.append("Rising interest rate environment")
            elif rate_trend == "FALLING":
                risk_on_factors.append(0.8)
                key_drivers.append("Falling interest rate environment")

            # Geopolitical risk
            if geo_risk == "HIGH":
                risk_off_factors.append(1.0)
                key_drivers.append("Elevated geopolitical risk")
            elif geo_risk == "LOW":
                risk_on_factors.append(0.5)

            # Crypto adoption
            if crypto_adoption == "HIGH":
                risk_on_factors.append(0.6)
                key_drivers.append("Strong crypto adoption momentum")
            elif crypto_adoption == "LOW":
                risk_off_factors.append(0.4)

            # Institutional sentiment
            if institutional_sentiment == "POSITIVE":
                risk_on_factors.append(0.7)
                key_drivers.append("Positive institutional sentiment")
            elif institutional_sentiment == "NEGATIVE":
                risk_off_factors.append(0.7)
                key_drivers.append("Negative institutional sentiment")

            # Regulatory environment
            if regulatory_env == "FAVORABLE":
                risk_on_factors.append(0.5)
                key_drivers.append("Favorable regulatory environment")
            elif regulatory_env == "RESTRICTIVE":
                risk_off_factors.append(0.8)
                key_drivers.append("Restrictive regulatory environment")

            # Volatility regime
            if volatility_regime == "HIGH":
                risk_off_factors.append(0.6)
                key_drivers.append("High volatility environment")

            # Liquidity conditions
            if liquidity_conditions == "TIGHT":
                risk_off_factors.append(0.7)
                key_drivers.append("Tight liquidity conditions")
            elif liquidity_conditions == "ABUNDANT":
                risk_on_factors.append(0.5)

            # Calculate scores
            risk_on_score = sum(risk_on_factors)
            risk_off_score = sum(risk_off_factors)

            # Determine regime
            if risk_on_score > risk_off_score * 1.3:
                regime = MarketRegimeType.RISK_ON
                confidence = min(risk_on_score / (risk_on_score + risk_off_score + 1), 0.9)
            elif risk_off_score > risk_on_score * 1.3:
                regime = MarketRegimeType.RISK_OFF
                confidence = min(risk_off_score / (risk_on_score + risk_off_score + 1), 0.9)
            else:
                regime = MarketRegimeType.TRANSITION
                confidence = 0.5
                key_drivers.append("Mixed signals suggest transitional regime")

            return regime, confidence, key_drivers[:5]

        except Exception:
            return MarketRegimeType.UNKNOWN, 0.0, ["Error in regime analysis"]

    def _calculate_regime_change_probability(self, sentiment_data: dict[str, Any]) -> float:
        """Calculate probability of regime change in next 30 days."""
        try:
            # Simple heuristic based on volatility and sentiment divergence
            volatility_score = sentiment_data.get('volatility_score', 0.5)
            sentiment_divergence = sentiment_data.get('sentiment_divergence', False)

            base_probability = 0.1  # 10% base probability

            if volatility_score > 0.7:
                base_probability += 0.3
            elif volatility_score > 0.5:
                base_probability += 0.1

            if sentiment_divergence:
                base_probability += 0.2

            return min(base_probability, 0.8)

        except Exception:
            return 0.1

    def _estimate_regime_duration(self, sentiment_data: dict[str, Any]) -> int | None:
        """Estimate how long current regime has been in place."""
        try:
            # In a full implementation, this would track regime changes over time
            # For now, return None to indicate unknown duration
            return None
        except Exception:
            return None

    def _calculate_fear_greed_index(self, news_data: list[dict[str, Any]]) -> float:
        """Calculate fear & greed index from news data."""
        try:
            if not news_data:
                return 50.0  # Neutral

            fear_keywords = ['fear', 'panic', 'crash', 'sell-off', 'liquidation', 'bearish']
            greed_keywords = ['greed', 'rally', 'surge', 'bullish', 'fomo', 'moon']

            fear_score = 0
            greed_score = 0

            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

                fear_score += sum(1 for keyword in fear_keywords if keyword in text)
                greed_score += sum(1 for keyword in greed_keywords if keyword in text)

            total_score = fear_score + greed_score
            if total_score == 0:
                return 50.0  # Neutral

            # Convert to 0-100 scale (0 = extreme fear, 100 = extreme greed)
            greed_ratio = greed_score / total_score
            return greed_ratio * 100

        except Exception:
            return 50.0

    def _classify_sentiment_level(self, fear_greed_index: float) -> SentimentLevel:
        """Classify sentiment level based on fear & greed index."""
        if fear_greed_index >= 80:
            return SentimentLevel.EXTREME_GREED
        elif fear_greed_index >= 60:
            return SentimentLevel.GREED
        elif fear_greed_index >= 40:
            return SentimentLevel.NEUTRAL
        elif fear_greed_index >= 20:
            return SentimentLevel.FEAR
        else:
            return SentimentLevel.EXTREME_FEAR

    def _calculate_volatility_expectation(self, news_data: list[dict[str, Any]]) -> float:
        """Calculate expected volatility from news data."""
        try:
            volatility_keywords = ['volatile', 'volatility', 'swing', 'choppy', 'uncertain']

            volatility_mentions = 0
            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()
                volatility_mentions += sum(1 for keyword in volatility_keywords if keyword in text)

            # Base volatility expectation
            base_volatility = 20.0

            # Adjust based on mentions
            if volatility_mentions > 5:
                return min(base_volatility * 2, 60.0)
            elif volatility_mentions > 2:
                return min(base_volatility * 1.5, 40.0)
            else:
                return base_volatility

        except Exception:
            return 20.0

    def _calculate_market_stress_indicator(self, news_data: list[dict[str, Any]]) -> float:
        """Calculate market stress indicator from news data."""
        try:
            stress_keywords = ['stress', 'pressure', 'concern', 'worry', 'uncertainty', 'crisis']

            stress_score = 0
            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()
                stress_score += sum(1 for keyword in stress_keywords if keyword in text)

            # Normalize to 0-1 scale
            return min(stress_score / 10.0, 1.0)

        except Exception:
            return 0.5

    def _extract_vix_level(self, news_data: list[dict[str, Any]]) -> float | None:
        """Extract VIX level from news data."""
        try:
            vix_pattern = re.compile(r'vix.*?(\d+\.?\d*)', re.IGNORECASE)

            for item in news_data:
                text = item.get('title', '') + ' ' + item.get('content', '')
                match = vix_pattern.search(text)
                if match:
                    return float(match.group(1))

            return None
        except Exception:
            return None

    def _extract_crypto_fear_greed(self, news_data: list[dict[str, Any]]) -> float | None:
        """Extract crypto-specific fear & greed index from news data."""
        try:
            # Look for crypto fear greed mentions
            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()
                if 'crypto fear' in text or 'bitcoin fear' in text:
                    # Extract number if present
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        return float(numbers[0])

            return None
        except Exception:
            return None

    def _calculate_social_sentiment(self, news_data: list[dict[str, Any]]) -> float | None:
        """Calculate social media sentiment score."""
        try:
            # Simple sentiment based on social media mentions
            positive_social = ['viral', 'trending', 'popular', 'hype', 'community']
            negative_social = ['fud', 'criticism', 'backlash', 'negativity']

            positive_score = 0
            negative_score = 0

            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

                if 'social' in text or 'twitter' in text or 'reddit' in text:
                    positive_score += sum(1 for keyword in positive_social if keyword in text)
                    negative_score += sum(1 for keyword in negative_social if keyword in text)

            total_score = positive_score + negative_score
            if total_score == 0:
                return None

            return (positive_score - negative_score) / total_score

        except Exception:
            return None

    def _calculate_news_sentiment(self, news_data: list[dict[str, Any]]) -> float | None:
        """Calculate news sentiment score."""
        try:
            if not news_data:
                return None

            positive_words = ['positive', 'bullish', 'optimistic', 'strong', 'growth']
            negative_words = ['negative', 'bearish', 'pessimistic', 'weak', 'decline']

            positive_score = 0
            negative_score = 0

            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

                positive_score += sum(1 for word in positive_words if word in text)
                negative_score += sum(1 for word in negative_words if word in text)

            total_score = positive_score + negative_score
            if total_score == 0:
                return 0.0

            return (positive_score - negative_score) / total_score

        except Exception:
            return None

    def _calculate_funding_rates_sentiment(self, news_data: list[dict[str, Any]]) -> float | None:
        """Calculate funding rates sentiment indicator."""
        try:
            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

                if 'funding rate' in text:
                    if 'high' in text or 'elevated' in text:
                        return 0.8  # High funding rates suggest bullish sentiment
                    elif 'low' in text or 'negative' in text:
                        return -0.5  # Low/negative funding rates suggest bearish sentiment
                    else:
                        return 0.0

            return None
        except Exception:
            return None

    def _analyze_options_flow_sentiment(self, news_data: list[dict[str, Any]]) -> str | None:
        """Analyze options flow sentiment."""
        try:
            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

                if 'options' in text:
                    if any(word in text for word in ['calls', 'bullish options', 'call buying']):
                        return "BULLISH"
                    elif any(word in text for word in ['puts', 'bearish options', 'put buying']):
                        return "BEARISH"
                    else:
                        return "NEUTRAL"

            return None
        except Exception:
            return None

    def _analyze_insider_activity(self, news_data: list[dict[str, Any]]) -> str | None:
        """Analyze insider/institutional activity sentiment."""
        try:
            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

                if any(word in text for word in ['insider', 'institutional', 'whale']):
                    if any(word in text for word in ['buying', 'accumulating', 'long']):
                        return "BULLISH"
                    elif any(word in text for word in ['selling', 'dumping', 'short']):
                        return "BEARISH"
                    else:
                        return "NEUTRAL"

            return None
        except Exception:
            return None

    def _analyze_retail_sentiment(self, news_data: list[dict[str, Any]]) -> str | None:
        """Analyze retail investor sentiment."""
        try:
            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

                if any(word in text for word in ['retail', 'individual investor', 'amateur']):
                    if any(word in text for word in ['buying', 'optimistic', 'bullish']):
                        return "BULLISH"
                    elif any(word in text for word in ['selling', 'pessimistic', 'bearish']):
                        return "BEARISH"
                    else:
                        return "NEUTRAL"

            return None
        except Exception:
            return None

    def _detect_sentiment_divergence(self, news_data: list[dict[str, Any]]) -> bool:
        """Detect sentiment divergence from price action."""
        try:
            # Look for explicit mentions of divergence
            for item in news_data:
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

                if any(word in text for word in ['divergence', 'disconnect', 'contrary']):
                    return True

            return False
        except Exception:
            return False

    def _calculate_crypto_momentum_score(self, crypto_indicators: dict[str, Any]) -> float:
        """Calculate crypto momentum score from indicators."""
        try:
            momentum_score = 0.0

            # RSI momentum
            rsi = crypto_indicators.get('rsi', 50)
            if rsi > 70:
                momentum_score += 0.5
            elif rsi > 50:
                momentum_score += 0.25
            elif rsi < 30:
                momentum_score -= 0.5
            elif rsi < 50:
                momentum_score -= 0.25

            # Price momentum (simplified)
            price_change = crypto_indicators.get('price_change_24h', 0)
            momentum_score += min(max(price_change / 10.0, -0.5), 0.5)

            # Volume momentum
            volume_change = crypto_indicators.get('volume_change_24h', 0)
            if volume_change > 0.2:  # 20% volume increase
                momentum_score += 0.2
            elif volume_change < -0.2:
                momentum_score -= 0.1

            return max(min(momentum_score, 1.0), -1.0)

        except Exception:
            return 0.0

    def _calculate_nasdaq_momentum_score(self, nasdaq_indicators: dict[str, Any]) -> float:
        """Calculate NASDAQ momentum score from indicators."""
        try:
            momentum_score = 0.0

            # Price momentum
            price_change = nasdaq_indicators.get('price_change_24h', 0)
            momentum_score += min(max(price_change / 5.0, -0.5), 0.5)

            # Volume momentum
            volume_change = nasdaq_indicators.get('volume_change_24h', 0)
            if volume_change > 0.15:  # 15% volume increase
                momentum_score += 0.2
            elif volume_change < -0.15:
                momentum_score -= 0.1

            # Sector rotation momentum
            tech_performance = nasdaq_indicators.get('tech_sector_performance', 0)
            momentum_score += min(max(tech_performance / 10.0, -0.3), 0.3)

            return max(min(momentum_score, 1.0), -1.0)

        except Exception:
            return 0.0

    def _calculate_directional_alignment(self, crypto_momentum: float, nasdaq_momentum: float) -> float:
        """Calculate directional alignment between momentum scores."""
        try:
            if crypto_momentum == 0 and nasdaq_momentum == 0:
                return 0.0

            # Both positive or both negative = aligned
            if (crypto_momentum > 0 and nasdaq_momentum > 0) or (crypto_momentum < 0 and nasdaq_momentum < 0):
                return min(abs(crypto_momentum), abs(nasdaq_momentum))
            else:
                return -min(abs(crypto_momentum), abs(nasdaq_momentum))

        except Exception:
            return 0.0

    def _calculate_strength_alignment(
        self,
        crypto_indicators: dict[str, Any],
        nasdaq_indicators: dict[str, Any]
    ) -> float:
        """Calculate strength alignment between momentum indicators."""
        try:
            crypto_strength = abs(crypto_indicators.get('trend_strength', 0))
            nasdaq_strength = abs(nasdaq_indicators.get('trend_strength', 0))

            if crypto_strength == 0 and nasdaq_strength == 0:
                return 0.0

            # Alignment is higher when strengths are similar
            strength_diff = abs(crypto_strength - nasdaq_strength)
            max_strength = max(crypto_strength, nasdaq_strength)

            if max_strength > 0:
                return 1.0 - (strength_diff / max_strength)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _identify_momentum_divergences(
        self,
        crypto_indicators: dict[str, Any],
        nasdaq_indicators: dict[str, Any]
    ) -> list[str]:
        """Identify momentum divergences between markets."""
        try:
            divergences = []

            crypto_trend = crypto_indicators.get('trend_direction', 'NEUTRAL')
            nasdaq_trend = nasdaq_indicators.get('trend_direction', 'NEUTRAL')

            if crypto_trend == 'BULLISH' and nasdaq_trend == 'BEARISH':
                divergences.append("Crypto bullish while NASDAQ bearish")
            elif crypto_trend == 'BEARISH' and nasdaq_trend == 'BULLISH':
                divergences.append("Crypto bearish while NASDAQ bullish")

            # Volume divergence
            crypto_volume_trend = crypto_indicators.get('volume_trend', 'NEUTRAL')
            nasdaq_volume_trend = nasdaq_indicators.get('volume_trend', 'NEUTRAL')

            if crypto_volume_trend != nasdaq_volume_trend and 'NEUTRAL' not in [crypto_volume_trend, nasdaq_volume_trend]:
                divergences.append(f"Volume divergence: Crypto {crypto_volume_trend}, NASDAQ {nasdaq_volume_trend}")

            return divergences

        except Exception:
            return []

    def _calculate_trend_strength(self, indicators: dict[str, Any]) -> float:
        """Calculate trend strength from indicators."""
        try:
            # Simplified trend strength calculation
            rsi = indicators.get('rsi', 50)
            price_change = indicators.get('price_change_24h', 0)
            volume_change = indicators.get('volume_change_24h', 0)

            strength = 0.0

            # RSI contribution
            if rsi > 70 or rsi < 30:
                strength += 0.4
            elif rsi > 60 or rsi < 40:
                strength += 0.2

            # Price change contribution
            strength += min(abs(price_change) / 10.0, 0.4)

            # Volume confirmation
            if abs(volume_change) > 0.2:
                strength += 0.2

            return min(strength, 1.0)

        except Exception:
            return 0.0

    def _calculate_volume_momentum_alignment(
        self,
        crypto_indicators: dict[str, Any],
        nasdaq_indicators: dict[str, Any]
    ) -> float | None:
        """Calculate volume momentum alignment."""
        try:
            crypto_volume_change = crypto_indicators.get('volume_change_24h')
            nasdaq_volume_change = nasdaq_indicators.get('volume_change_24h')

            if crypto_volume_change is None or nasdaq_volume_change is None:
                return None

            # Both increasing or both decreasing = aligned
            if (crypto_volume_change > 0 and nasdaq_volume_change > 0) or \
               (crypto_volume_change < 0 and nasdaq_volume_change < 0):
                return min(abs(crypto_volume_change), abs(nasdaq_volume_change))
            else:
                return -min(abs(crypto_volume_change), abs(nasdaq_volume_change))

        except Exception:
            return None

    def _assess_momentum_sustainability(
        self,
        crypto_indicators: dict[str, Any],
        nasdaq_indicators: dict[str, Any]
    ) -> float:
        """Assess momentum sustainability."""
        try:
            sustainability_score = 0.5  # Base sustainability

            # Volume confirmation increases sustainability
            crypto_volume_change = crypto_indicators.get('volume_change_24h', 0)
            nasdaq_volume_change = nasdaq_indicators.get('volume_change_24h', 0)

            if crypto_volume_change > 0.2 and nasdaq_volume_change > 0.2:
                sustainability_score += 0.3
            elif crypto_volume_change > 0.1 or nasdaq_volume_change > 0.1:
                sustainability_score += 0.1

            # Breadth indicators (if available)
            crypto_breadth = crypto_indicators.get('market_breadth', 0)
            if abs(crypto_breadth) > 0.5:
                sustainability_score += 0.2

            return min(sustainability_score, 1.0)

        except Exception:
            return 0.5

    def _determine_momentum_regime(
        self,
        crypto_indicators: dict[str, Any],
        nasdaq_indicators: dict[str, Any]
    ) -> str:
        """Determine momentum regime."""
        try:
            crypto_momentum = self._calculate_crypto_momentum_score(crypto_indicators)
            nasdaq_momentum = self._calculate_nasdaq_momentum_score(nasdaq_indicators)

            avg_momentum = (abs(crypto_momentum) + abs(nasdaq_momentum)) / 2

            if avg_momentum > 0.6:
                return "ACCELERATION"
            elif avg_momentum < 0.2:
                return "DECELERATION"
            else:
                return "NORMAL"

        except Exception:
            return "NORMAL"

    def _analyze_cross_asset_momentum_flow(
        self,
        crypto_indicators: dict[str, Any],
        nasdaq_indicators: dict[str, Any]
    ) -> str:
        """Analyze cross-asset momentum flow."""
        try:
            crypto_momentum = self._calculate_crypto_momentum_score(crypto_indicators)
            nasdaq_momentum = self._calculate_nasdaq_momentum_score(nasdaq_indicators)

            if crypto_momentum > nasdaq_momentum + 0.2:
                return "CRYPTO_OUTPERFORMING"
            elif nasdaq_momentum > crypto_momentum + 0.2:
                return "NASDAQ_OUTPERFORMING"
            elif crypto_momentum > 0.3 and nasdaq_momentum > 0.3:
                return "RISK_ON_FLOW"
            elif crypto_momentum < -0.3 and nasdaq_momentum < -0.3:
                return "RISK_OFF_FLOW"
            else:
                return "NEUTRAL"

        except Exception:
            return "NEUTRAL"

    def _generate_trading_implications(
        self,
        correlation: CorrelationAnalysis,
        regime: MarketRegime
    ) -> str:
        """Generate trading implications from context analysis."""
        try:
            implications = []

            # Correlation implications
            if correlation.correlation_strength in [CorrelationStrength.STRONG, CorrelationStrength.VERY_STRONG]:
                if correlation.direction == "POSITIVE":
                    implications.append("Strong positive correlation increases systematic risk")
                else:
                    implications.append("Strong negative correlation may provide hedging opportunities")
            elif correlation.correlation_strength == CorrelationStrength.VERY_WEAK:
                implications.append("Weak correlation suggests crypto-specific factors dominate")

            # Regime implications
            if regime.regime_type == MarketRegimeType.RISK_ON:
                implications.append("Risk-on regime favors crypto longs")
            elif regime.regime_type == MarketRegimeType.RISK_OFF:
                implications.append("Risk-off regime suggests defensive positioning")
            elif regime.regime_type == MarketRegimeType.TRANSITION:
                implications.append("Transitional regime calls for cautious approach")

            # Fed policy implications
            if regime.fed_policy_stance == "HAWKISH":
                implications.append("Hawkish Fed policy headwind for risk assets")
            elif regime.fed_policy_stance == "DOVISH":
                implications.append("Dovish Fed policy supportive for crypto")

            # Confidence implications
            if correlation.reliability_score < 0.5:
                implications.append("Low correlation reliability suggests careful position sizing")

            if regime.confidence < 0.5:
                implications.append("Uncertain regime calls for flexible strategy")

            return " | ".join(implications[:4])  # Limit to 4 key implications

        except Exception:
            return "Error generating trading implications"
