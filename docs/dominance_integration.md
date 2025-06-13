# USDT/USDC Dominance Integration Guide

## Overview

The AI Trading Bot now includes stablecoin dominance analysis to gauge market sentiment and improve trading decisions. This feature tracks the combined market share of USDT and USDC relative to the total crypto market cap, providing valuable insights into risk-on/risk-off market behavior.

## Key Concepts

### What is Stablecoin Dominance?

Stablecoin dominance represents the percentage of the total cryptocurrency market cap held in stablecoins (primarily USDT and USDC). It's calculated as:

```
Dominance % = (USDT Market Cap + USDC Market Cap) / Total Crypto Market Cap × 100
```

### Why It Matters

- **High Dominance (>10%)**: Indicates risk-off sentiment - traders are moving to the safety of stablecoins
- **Rising Dominance**: Money flowing from crypto to stables - bearish signal
- **Falling Dominance**: Money flowing from stables to crypto - bullish signal
- **Low Dominance (<5%)**: Risk-on sentiment - confidence in crypto markets

## Configuration

### Enable Dominance Data

In your `.env` file or configuration:

```bash
# Dominance Settings
DOMINANCE__ENABLE_DOMINANCE_DATA=true
DOMINANCE__DATA_SOURCE=coingecko  # or coinmarketcap
DOMINANCE__UPDATE_INTERVAL=300     # 5 minutes
DOMINANCE__DOMINANCE_WEIGHT_IN_DECISIONS=0.2  # 20% weight
```

### Configuration Options

```json
{
  "dominance": {
    "enable_dominance_data": true,
    "data_source": "coingecko",
    "update_interval": 300,
    "dominance_weight_in_decisions": 0.2,
    "dominance_alert_threshold": 12.0,
    "dominance_change_alert_threshold": 1.0
  }
}
```

## How It Works

### 1. Data Collection

The `DominanceDataProvider` fetches real-time data from:
- **CoinGecko API** (free, no API key required)
- **CoinMarketCap API** (requires API key for higher limits)

### 2. Metrics Calculated

- **USDT/USDC Individual Dominance**: Market share of each stablecoin
- **Combined Dominance**: Total stablecoin market share
- **24h Change**: Rate of change in dominance
- **Dominance RSI**: Technical indicator for dominance trends
- **Stablecoin Velocity**: Trading volume / market cap ratio

### 3. Market Sentiment Analysis

The bot analyzes dominance levels to determine market sentiment:

```python
# Sentiment Scoring Logic
- Dominance > 10%: Strong bearish bias (-2 score)
- Dominance 7-10%: Moderate bearish bias (-1 score)
- Dominance < 5%: Bullish bias (+1 score)
- Rising dominance: Additional bearish signal
- Falling dominance: Additional bullish signal
```

### 4. Integration with Trading Decisions

#### LLM Agent Integration

The dominance data is included in the LLM prompt:

```
Market Sentiment (Stablecoin Dominance):
- USDT Dominance: 4.5%
- USDC Dominance: 2.8%
- Total Stablecoin Dominance: 7.3%
- 24h Dominance Change: -0.5%
- Market Sentiment: BULLISH
```

#### Position Sizing Adjustments

- **High Dominance**: Reduce position sizes by 20-30%
- **Rising Dominance**: Use tighter stop losses
- **Low/Falling Dominance**: Allow normal or larger positions

#### Fallback Logic

When the LLM is unavailable, the bot's fallback logic considers dominance:

```python
# High dominance = reduce risk
if dominance > 10:
    position_size *= 0.7  # 30% reduction

# Rising dominance = bearish bias
if dominance_change > 0.5:
    prefer_short_positions = True
```

## Usage Examples

### Running with Dominance Data

```bash
# Run with dominance enabled (default)
poetry run ai-trading-bot live --symbol BTC-USD

# Run the example script
python examples/dominance_integration_example.py
```

### Monitoring Dominance

The bot displays dominance metrics in its status updates:

```
Trading Status - Loop 100
┌─────────────────────┬──────────────────────┐
│ Metric              │ Value                │
├─────────────────────┼──────────────────────┤
│ Current Price       │ $45,234.00           │
│ Stablecoin Dominance│ 8.45% (+0.32%)       │
│ Market Sentiment    │ NEUTRAL              │
└─────────────────────┴──────────────────────┘
```

## API Data Sources

### CoinGecko (Default)

- **Pros**: Free, no API key required
- **Cons**: Rate limited (10-50 calls/minute)
- **Endpoints Used**:
  - `/api/v3/global` - Total market cap
  - `/api/v3/coins/markets` - USDT/USDC data

### CoinMarketCap

- **Pros**: Higher rate limits, more reliable
- **Cons**: Requires API key
- **Setup**:
  ```bash
  DOMINANCE__DATA_SOURCE=coinmarketcap
  DOMINANCE__API_KEY=your-coinmarketcap-api-key
  ```

### Custom Data Source

You can implement custom data sources by extending the `DominanceDataProvider` class.

## Trading Strategies with Dominance

### 1. Trend Confirmation

Use dominance to confirm market trends:
- Technical indicators bullish + dominance falling = Strong buy signal
- Technical indicators bearish + dominance rising = Strong sell signal

### 2. Divergence Trading

Look for divergences between price and dominance:
- Price rising but dominance also rising = Potential top
- Price falling but dominance falling = Potential bottom

### 3. Risk Management

Adjust risk based on dominance levels:
- Dominance > 12%: Maximum 50% of normal position size
- Dominance 8-12%: 70-100% of normal position size
- Dominance < 8%: 100-120% of normal position size

## Backtesting with Dominance

The dominance data is included in backtesting:

```bash
# Backtest with dominance analysis
poetry run ai-trading-bot backtest \
  --from 2024-01-01 \
  --to 2024-12-31 \
  --dominance-weight 0.3
```

## Troubleshooting

### No Dominance Data

If dominance data is unavailable:
1. Check internet connection
2. Verify API rate limits haven't been exceeded
3. Check logs for API errors
4. The bot will continue trading without dominance data

### API Rate Limits

- CoinGecko: Wait 1-2 minutes between requests
- Use longer update intervals: `DOMINANCE__UPDATE_INTERVAL=600`

### Data Accuracy

- Dominance calculations may vary slightly between providers
- Use 5-minute or longer update intervals for stability
- Monitor the `dominance_24h_change` for significant shifts

## Performance Impact

- Minimal CPU/memory usage
- One API call every 5 minutes (default)
- Cached data reduces API calls
- Asynchronous updates don't block trading

## Future Enhancements

1. **Additional Stablecoins**: Include DAI, BUSD dominance
2. **Custom Indicators**: Dominance momentum, divergence indicators
3. **Machine Learning**: Train models on dominance patterns
4. **Alert System**: Telegram/Discord notifications for dominance shifts
5. **TradingView Integration**: Webhook support for custom dominance data

## Conclusion

Stablecoin dominance provides valuable market sentiment data that enhances the bot's trading decisions. By tracking where money is flowing (into or out of stablecoins), the bot can better gauge market conditions and adjust its trading strategy accordingly.