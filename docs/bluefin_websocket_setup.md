# Bluefin WebSocket Real-Time Data Setup

This guide explains how to enable and use real-time WebSocket data from Bluefin exchange instead of mock data.

## Overview

The Bluefin market data provider now supports real-time WebSocket connections to receive:
- Live price updates (tick data)
- Market data updates
- Order book updates
- Trade execution data

The system automatically builds OHLCV candles from the incoming tick data and calculates VuManChu Cipher indicators in real-time.

## Configuration

### 1. Enable Real Data Mode

To use real WebSocket data instead of mock data, you have two options:

#### Option A: Environment Variable (Recommended for Testing)
```bash
export BLUEFIN_USE_REAL_DATA=true
```

#### Option B: Disable Dry Run Mode
In your `.env` file:
```bash
SYSTEM__DRY_RUN=false
```

### 2. Select Bluefin Exchange

Ensure Bluefin is selected as your exchange in `.env`:
```bash
EXCHANGE__EXCHANGE_TYPE=bluefin
```

## WebSocket Features

### Real-Time Data Processing
- **Tick Aggregation**: Incoming trades are buffered and aggregated into candles
- **Candle Building**: Candles are built at your specified interval (1m, 5m, 15s, etc.)
- **Indicator Calculation**: VuManChu Cipher indicators are calculated on the real data
- **Auto-Reconnect**: Automatic reconnection with exponential backoff on disconnection

### Supported Market Events
1. **MarketDataUpdate**: Real-time price and volume updates
2. **RecentTrades**: Individual trade executions
3. **OrderbookUpdate**: Order book depth changes
4. **MarketHealth**: Exchange health status

## Testing the Connection

Run the test script to verify WebSocket connectivity:
```bash
python test_bluefin_websocket.py
```

This will:
- Connect to Bluefin's WebSocket API
- Subscribe to ETH-PERP market data
- Display real-time tick counts and candle updates
- Show connection status and latency

## Running the Bot with Real Data

### Paper Trading with Real Market Data
```bash
# Uses real market data but doesn't execute real trades
BLUEFIN_USE_REAL_DATA=true python -m bot.main live --dry-run
```

### Live Trading (Caution!)
```bash
# Real money, real trades
SYSTEM__DRY_RUN=false python -m bot.main live
```

## Monitoring WebSocket Health

The bot logs detailed WebSocket status:
```
INFO - Successfully connected to Bluefin WebSocket
INFO - Subscribed to market data for ETH-PERP
DEBUG - Market data update: ETH-PERP @ 2500.50
INFO - Built new candle from 15 ticks
```

Check connection status:
```python
# In the bot's status display
Market Data: ✓ Connected (WebSocket: ✓ Receiving data)
```

## Troubleshooting

### No Data Received
1. Check network connectivity to `wss://notifications.api.sui-prod.bluefin.io/`
2. Verify the symbol exists (e.g., ETH-PERP, BTC-PERP)
3. Check logs for subscription confirmation

### Connection Drops
- The bot automatically reconnects with exponential backoff
- Max 10 reconnection attempts before giving up
- Check logs for reconnection status

### Data Quality Issues
- Minimum 100 candles recommended for reliable indicators
- The bot will warn if insufficient data is available
- Historical data is fetched on startup to supplement real-time data

## Performance Considerations

### High-Frequency Trading
For intervals < 1 minute:
- Tick buffer holds up to 10,000 ticks
- Candles are built every second
- Ensure stable network connection for best results

### Resource Usage
- WebSocket maintains persistent connection
- Tick buffering uses minimal memory (~1MB for 10k ticks)
- CPU usage increases during indicator calculation

## API Endpoints

The Bluefin WebSocket implementation uses:
- **WebSocket URL**: `wss://notifications.api.sui-prod.bluefin.io/`
- **REST API**: `https://dapi.api.sui-prod.bluefin.io` (for historical data)

## Security

- No authentication required for public market data
- Private data (positions, orders) requires authenticated connection
- All connections use TLS encryption

## Example Log Output

```
2024-01-15 10:30:00 - bot.data.bluefin_market - INFO - Using real Bluefin market data via WebSocket
2024-01-15 10:30:01 - bot.data.bluefin_market - INFO - Successfully connected to Bluefin WebSocket
2024-01-15 10:30:01 - bot.data.bluefin_market - INFO - Subscribed to market data for ETH-PERP
2024-01-15 10:30:02 - bot.data.bluefin_market - DEBUG - Market data update: ETH-PERP @ 2505.25
2024-01-15 10:30:15 - bot.data.bluefin_market - INFO - Built new 15s candle: O:2505.00 H:2505.50 L:2504.75 C:2505.25
2024-01-15 10:30:15 - bot.main - INFO - Calculating VuManChu indicators on 200 candles
```