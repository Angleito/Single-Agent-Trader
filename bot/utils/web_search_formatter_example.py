#!/usr/bin/env python3
"""
Example usage of WebSearchFormatter for AI trading bot.

This example demonstrates how to use the WebSearchFormatter to process
web search results, sentiment analysis, and market context data for
optimal LLM consumption in trading decisions.
"""

import asyncio
from datetime import datetime, timedelta

from .web_search_formatter import WebSearchFormatter


async def example_news_formatting():
    """Example of formatting news results for LLM consumption."""

    # Create formatter instance
    formatter = WebSearchFormatter(max_tokens_per_section=500, max_total_tokens=2000)

    # Sample news data
    sample_news = [
        {
            "title": "Bitcoin Surges Past $50,000 as Institutional Demand Grows",
            "content": "Bitcoin has broken through the $50,000 resistance level with strong volume, "
                      "driven by increasing institutional adoption and positive regulatory developments. "
                      "Technical indicators suggest further upside potential with RSI entering bullish territory.",
            "url": "https://coindesk.com/markets/2024/01/15/bitcoin-institutional-demand",
            "published_time": datetime.now() - timedelta(hours=2)
        },
        {
            "title": "Federal Reserve Hints at More Dovish Policy Stance",
            "content": "Fed Chairman Powell indicated a potential shift toward more accommodative monetary policy, "
                      "citing concerns about economic growth. This development is generally positive for "
                      "risk assets including cryptocurrencies and growth stocks.",
            "url": "https://reuters.com/business/fed-policy-2024",
            "published_time": datetime.now() - timedelta(hours=4)
        },
        {
            "title": "Crypto Regulation Clarity Expected This Quarter",
            "content": "Regulatory authorities are expected to provide clearer guidelines for cryptocurrency "
                      "operations, potentially reducing uncertainty in the market. Industry leaders remain "
                      "optimistic about the regulatory framework's impact on adoption.",
            "url": "https://bloomberg.com/crypto-regulation-2024",
            "published_time": datetime.now() - timedelta(hours=6)
        }
    ]

    # Format news for LLM
    formatted_news = await formatter.format_news_results(sample_news)

    print("=== FORMATTED NEWS RESULTS ===")
    print(formatted_news)
    print("\n" + "="*50 + "\n")

    return formatted_news


async def example_sentiment_formatting():
    """Example of formatting sentiment analysis data."""

    formatter = WebSearchFormatter()

    # Mock sentiment result (in real usage, this comes from FinancialSentimentService)
    class MockSentimentResult:
        def __init__(self):
            self.sentiment_score = 0.65  # Bullish
            self.confidence = 0.82
            self.key_themes = [
                "Bitcoin institutional adoption",
                "Federal Reserve dovish policy",
                "Regulatory clarity expectations",
                "Technical breakout patterns",
                "Market momentum shift"
            ]
            self.bullish_indicators = [
                "Strong institutional buying pressure",
                "Technical breakout above resistance",
                "Positive regulatory developments"
            ]
            self.bearish_indicators = [
                "Profit-taking at resistance levels"
            ]
            self.volatility_signals = [
                "Increased trading volume",
                "Options market activity spike"
            ]

    mock_sentiment = MockSentimentResult()

    # Format sentiment data
    formatted_sentiment = await formatter.format_sentiment_data(mock_sentiment)

    print("=== FORMATTED SENTIMENT ANALYSIS ===")
    print(formatted_sentiment)
    print("\n" + "="*50 + "\n")

    return formatted_sentiment


async def example_correlation_formatting():
    """Example of formatting correlation analysis data."""

    formatter = WebSearchFormatter()

    # Mock correlation analysis (in real usage, this comes from MarketContextAnalyzer)
    class MockCorrelationAnalysis:
        def __init__(self):
            self.correlation_coefficient = 0.73
            self.correlation_strength = type('MockStrength', (), {'value': 'STRONG'})()
            self.direction = "POSITIVE"
            self.p_value = 0.003
            self.is_significant = True
            self.sample_size = 150
            self.rolling_correlation_24h = 0.78
            self.rolling_correlation_7d = 0.69
            self.correlation_stability = 0.85
            self.regime_dependent_correlation = {
                "HIGH_VOLATILITY": 0.82,
                "LOW_VOLATILITY": 0.64
            }
            self.reliability_score = 0.88

    mock_correlation = MockCorrelationAnalysis()

    # Format correlation data
    formatted_correlation = await formatter.format_correlation_analysis(mock_correlation)

    print("=== FORMATTED CORRELATION ANALYSIS ===")
    print(formatted_correlation)
    print("\n" + "="*50 + "\n")

    return formatted_correlation


async def example_comprehensive_market_context():
    """Example of formatting comprehensive market context."""

    formatter = WebSearchFormatter()

    # Create comprehensive market context data
    market_context = {
        "news_results": [
            {
                "title": "Bitcoin ETF Sees Record Inflows",
                "content": "Bitcoin ETFs recorded their highest daily inflows this year, "
                          "signaling strong institutional interest in cryptocurrency exposure.",
                "url": "https://coindesk.com/etf-inflows-2024",
                "published_time": datetime.now() - timedelta(hours=1)
            },
            {
                "title": "Tech Stocks Rally on AI Optimism",
                "content": "Technology stocks surged as investors show renewed optimism "
                          "about artificial intelligence developments and earnings prospects.",
                "url": "https://wsj.com/tech-rally-2024",
                "published_time": datetime.now() - timedelta(hours=3)
            }
        ],
        "sentiment_result": type('MockSentiment', (), {
            'sentiment_score': 0.58,
            'confidence': 0.79,
            'key_themes': ['AI developments', 'ETF inflows', 'Tech rally'],
            'bullish_indicators': ['Record ETF inflows', 'Tech sector strength'],
            'bearish_indicators': ['Profit-taking concerns'],
            'volatility_signals': ['Options activity increase']
        })(),
        "correlation_analysis": type('MockCorrelation', (), {
            'correlation_coefficient': 0.71,
            'correlation_strength': type('MockStrength', (), {'value': 'STRONG'})(),
            'direction': 'POSITIVE',
            'p_value': 0.002,
            'is_significant': True,
            'sample_size': 120,
            'rolling_correlation_24h': 0.75,
            'rolling_correlation_7d': 0.67,
            'correlation_stability': 0.83,
            'regime_dependent_correlation': {'HIGH_VOLATILITY': 0.80},
            'reliability_score': 0.86
        })()
    }

    # Format comprehensive context
    formatted_context = await formatter.format_market_context(market_context)

    print("=== COMPREHENSIVE MARKET CONTEXT ===")
    print(formatted_context)
    print("\n" + "="*50 + "\n")

    return formatted_context


async def example_key_insights_extraction():
    """Example of extracting key insights from search results."""

    formatter = WebSearchFormatter()

    # Sample search results
    search_results = {
        "news_items": [
            {
                "title": "Bitcoin Technical Analysis: Breakout Confirmed",
                "content": "Bitcoin has confirmed a technical breakout above the $49,500 resistance level "
                          "with strong volume support. RSI indicates bullish momentum continuation.",
                "published_time": datetime.now() - timedelta(hours=2)
            },
            {
                "title": "Institutional Whale Activity Detected",
                "content": "Large Bitcoin transfers to institutional wallets suggest continued "
                          "accumulation by major players in the cryptocurrency space.",
                "published_time": datetime.now() - timedelta(hours=4)
            }
        ],
        "sentiment_data": {
            "sentiment_score": 0.72,
            "confidence": 0.85,
            "volatility_signals": ["High volume breakout"]
        },
        "price_data": {
            "price_change_24h": 0.067,  # 6.7% increase
            "volume_change_24h": 0.34   # 34% volume increase
        },
        "technical_analysis": {
            "rsi": 68,
            "trend_direction": "BULLISH",
            "volume_trend": "INCREASING"
        }
    }

    # Extract key insights
    insights = await formatter.extract_key_insights(search_results)

    print("=== KEY INSIGHTS EXTRACTION ===")
    print(f"Extracted {len(insights)} key insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    print("\n" + "="*50 + "\n")

    return insights


async def example_content_optimization():
    """Example of content optimization for token limits."""

    formatter = WebSearchFormatter(max_tokens_per_section=200, max_total_tokens=800)

    # Long content that needs optimization
    long_content = """
    Bitcoin has been experiencing significant price movement over the past 24 hours, 
    breaking through multiple resistance levels with strong volume support. Technical 
    analysis indicates that the cryptocurrency is entering a new bullish phase, 
    supported by increasing institutional adoption and positive regulatory developments.
    
    The Federal Reserve's recent policy signals have created a more favorable 
    environment for risk assets, including cryptocurrencies. Market participants 
    are increasingly optimistic about the potential for continued price appreciation 
    in the digital asset space.
    
    Trading volume has increased substantially, with options markets showing elevated 
    activity levels. This suggests that both retail and institutional investors are 
    positioning for continued volatility and potential upside movement.
    
    Key resistance levels to watch include $52,000 and $55,000, while support 
    levels are established at $48,000 and $45,000. The overall market structure 
    remains bullish as long as Bitcoin maintains above the $47,000 level.
    
    Correlation with traditional markets has strengthened, particularly with 
    technology stocks, suggesting that macro factors continue to play an important 
    role in cryptocurrency price movements. This relationship should be monitored 
    closely for any changes that might indicate shifting market dynamics.
    """ * 3  # Make it longer to test optimization

    # Optimize content
    optimized_content = formatter._optimize_content_for_tokens(long_content)

    print("=== CONTENT OPTIMIZATION ===")
    print(f"Original length: {len(long_content)} characters")
    print(f"Optimized length: {len(optimized_content)} characters")
    print(f"Estimated original tokens: {formatter._estimate_token_count(long_content)}")
    print(f"Estimated optimized tokens: {formatter._estimate_token_count(optimized_content)}")
    print("\nOptimized content preview:")
    print(optimized_content[:500] + "..." if len(optimized_content) > 500 else optimized_content)
    print("\n" + "="*50 + "\n")

    return optimized_content


async def main():
    """Run all examples to demonstrate WebSearchFormatter capabilities."""

    print("🌐 WebSearchFormatter Example Usage")
    print("=" * 60)
    print()

    try:
        # Run all examples
        await example_news_formatting()
        await example_sentiment_formatting()
        await example_correlation_formatting()
        await example_comprehensive_market_context()
        await example_key_insights_extraction()
        await example_content_optimization()

        print("🎉 All examples completed successfully!")
        print("\nThe WebSearchFormatter is ready to be integrated into your AI trading bot")
        print("for optimal processing of web search results and market context data.")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
