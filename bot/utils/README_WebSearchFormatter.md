# WebSearchFormatter

A comprehensive web search results formatter optimized for AI trading bot LLM consumption.

## Overview

The `WebSearchFormatter` class provides advanced formatting capabilities for web search results, sentiment analysis, and market context data. It optimizes content for Large Language Model (LLM) processing while maintaining readability and trading relevance.

## Features

### Core Functionality
- **Intelligent Content Summarization**: Smart text summarization that preserves key trading insights
- **Key Insight Extraction**: Automated extraction of trading-relevant insights from web content
- **Relevance Scoring**: Multi-factor scoring system to prioritize content by trading relevance
- **Content Deduplication**: Advanced deduplication to avoid redundant information
- **Token-Efficient Formatting**: Optimized output that respects LLM token limits

### Content Optimization
- **Priority-Based Filtering**: Content ranking based on freshness, authority, and trading impact
- **Smart Truncation**: Intelligent text truncation that preserves meaning and structure
- **Visual Formatting**: Rich emoji and formatting for enhanced readability
- **Context-Appropriate Length Limits**: Dynamic content sizing based on importance

### Async Processing
- **Concurrent Processing**: Async/await support for efficient processing of multiple items
- **Batch Operations**: Optimized handling of large datasets
- **Error Handling**: Comprehensive error handling with graceful degradation

## Classes

### WebSearchFormatter

The main formatter class with the following key methods:

#### Core Formatting Methods
- `format_news_results(news_items)` - Format news articles for LLM consumption
- `format_sentiment_data(sentiment)` - Format sentiment analysis results
- `format_correlation_analysis(correlation)` - Format market correlation data
- `format_market_context(context)` - Format comprehensive market context

#### Utility Methods
- `truncate_content(text, max_length)` - Smart content truncation
- `extract_key_insights(search_results)` - Extract key trading insights

### Supporting Classes

#### ContentPriority
Pydantic model for content priority scoring:
- `relevance_score` - Content relevance to trading decisions (0.0-1.0)
- `freshness_score` - Content freshness/recency score (0.0-1.0)
- `authority_score` - Source authority/credibility score (0.0-1.0)
- `trading_impact_score` - Potential trading impact score (0.0-1.0)
- `final_priority` - Final weighted priority score (0.0-1.0)

#### FormattedContent
Pydantic model for formatted content:
- `summary` - Concise content summary
- `key_insights` - List of key insights
- `trading_signals` - List of trading-relevant signals
- `market_sentiment` - Overall market sentiment (BULLISH/BEARISH/NEUTRAL)
- `confidence_level` - Confidence in analysis (0.0-1.0)
- `token_count` - Estimated token count for LLM processing
- `priority` - ContentPriority object

## Usage Examples

### Basic News Formatting

```python
from bot.utils.web_search_formatter import WebSearchFormatter

# Initialize formatter
formatter = WebSearchFormatter(max_tokens_per_section=500, max_total_tokens=2000)

# Sample news data
news_items = [
    {
        "title": "Bitcoin Breaks $50K Resistance",
        "content": "Bitcoin surged past $50,000 with strong volume...",
        "url": "https://coindesk.com/...",
        "published_time": datetime.now()
    }
]

# Format for LLM
formatted_news = await formatter.format_news_results(news_items)
```

### Sentiment Analysis Formatting

```python
# Format sentiment analysis results
formatted_sentiment = await formatter.format_sentiment_data(sentiment_result)
```

### Comprehensive Market Context

```python
# Format complete market context
market_context = {
    "news_results": news_items,
    "sentiment_result": sentiment_data,
    "correlation_analysis": correlation_data,
    "market_regime": regime_data
}

formatted_context = await formatter.format_market_context(market_context)
```

### Key Insights Extraction

```python
# Extract key insights from search results
search_results = {
    "news_items": news_data,
    "sentiment_data": sentiment_info,
    "price_data": price_info,
    "technical_analysis": technical_data
}

insights = await formatter.extract_key_insights(search_results)
```

## Configuration

### Constructor Parameters
- `max_tokens_per_section` (int): Maximum tokens per content section (default: 500)
- `max_total_tokens` (int): Maximum total tokens for formatted output (default: 2000)

### Trading Keywords
The formatter includes pre-configured keyword sets for:
- **Price Action**: breakout, support, resistance, trend, momentum, etc.
- **Technical Analysis**: RSI, MACD, EMA, SMA, Bollinger bands, etc.
- **Market Structure**: liquidity, whale activity, institutional flows, etc.
- **Sentiment Indicators**: fear, greed, FOMO, capitulation, etc.
- **Macro Factors**: Fed policy, inflation, rates, regulation, etc.

### Authority Sources
Pre-configured authority scoring for major financial sources:
- Bloomberg (0.95), Reuters (0.95), WSJ (0.9), FT (0.9)
- CoinDesk (0.85), CoinTelegraph (0.8), Yahoo Finance (0.8)
- MarketWatch (0.75), CNBC (0.75), Investing.com (0.7)

## Error Handling

The formatter includes comprehensive error handling:
- Graceful degradation for missing data
- Fallback formatting for invalid inputs
- Detailed logging for debugging
- Mock classes for testing without dependencies

## Performance Considerations

- **Async Processing**: Use async methods for optimal performance
- **Batch Processing**: Process multiple items concurrently when possible
- **Token Optimization**: Content is automatically optimized for token efficiency
- **Memory Management**: Efficient handling of large datasets with streaming

## Integration with AI Trading Bot

The WebSearchFormatter is designed to integrate seamlessly with:
- Financial sentiment analysis services
- Market context analyzers
- LLM-based trading decision engines
- Risk management systems

For complete examples, see `web_search_formatter_example.py` in the same directory.

## Dependencies

- **Required**: `pydantic`, `asyncio`, `datetime`, `typing`
- **Optional**: `pandas` (for advanced data processing)
- **Internal**: `bot.services.financial_sentiment`, `bot.analysis.market_context`

The formatter includes fallback mock classes for standalone usage without internal dependencies.
