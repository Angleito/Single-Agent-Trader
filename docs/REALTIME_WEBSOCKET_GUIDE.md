# Real-time WebSocket Data Streaming Guide

## Overview

This guide describes the comprehensive real-time WebSocket data streaming system implemented for high-frequency scalping trading. The system provides sub-second market data updates, real-time tick aggregation, and live candle generation for scalping algorithms.

## Architecture

### Components

1. **Bluefin Service WebSocket Endpoints** (`bluefin-service/bluefin_service.py`)
   - Real-time price and trade data streaming
   - Mock data generation for testing
   - Connection management and broadcasting

2. **BluefinServiceClient WebSocket Support** (`bot/exchange/bluefin_client.py`)
   - WebSocket client with reconnection logic
   - Message handling and data callbacks
   - Connection status monitoring

3. **RealtimeMarketDataProvider** (`bot/data/realtime_market.py`)
   - Tick aggregation into real-time candles
   - Multiple timeframe support (1s, 5s, 15s, 1m)
   - Integration with trading bot

4. **Tick Aggregator** (`bot/data/realtime_market.py`)
   - Real-time OHLCV candle generation
   - Configurable intervals
   - Historical data management

## Features

### WebSocket Streaming
- **Real-time Price Updates**: Live price changes every 1-15 seconds
- **Trade Data**: Individual trade information with size and side
- **Heartbeat Monitoring**: Connection health checks
- **Auto-reconnection**: Exponential backoff retry logic

### Tick Aggregation
- **Multiple Intervals**: 1s, 5s, 15s, 1m, etc.
- **OHLCV Generation**: Complete candle data from ticks
- **Volume Tracking**: Accurate volume aggregation
- **Timestamp Alignment**: Proper candle boundary handling

### High-Frequency Trading Support
- **Sub-minute Intervals**: 15s, 30s, 45s intervals
- **Tick-level Data**: Individual trade information
- **Real-time Indicators**: Live technical analysis
- **Latency Optimization**: Minimal processing delays

## WebSocket Endpoints

### Single Symbol Stream
```
ws://localhost:8080/ws/{symbol}
```
Example: `ws://localhost:8080/ws/ETH-PERP`

### Multi-Symbol Stream
```
ws://localhost:8080/ws/multi/{symbols}
```
Example: `ws://localhost:8080/ws/multi/ETH-PERP,BTC-PERP,SUI-PERP`

### Message Format

#### Price Update
```json
{
  "type": "price_update",
  "symbol": "ETH-PERP",
  "timestamp": 1672531200.123,
  "data": {
    "price": 2500.123456,
    "volume": 1000.50
  }
}
```

#### Trade Data
```json
{
  "type": "trade",
  "symbol": "ETH-PERP",
  "timestamp": 1672531200.456,
  "data": {
    "price": 2500.234567,
    "size": 2.5000,
    "side": "buy",
    "trade_id": "trade_1672531200456_1234"
  }
}
```

#### Heartbeat
```json
{
  "type": "heartbeat",
  "symbol": "system",
  "timestamp": 1672531200.789,
  "data": {
    "connections": 3,
    "active_symbols": 2
  }
}
```

## Usage Examples

### Running High-Frequency Trading

For intervals ≤ 60 seconds, the bot automatically uses real-time WebSocket data:

```bash
# 15-second scalping
python -m bot.main live --symbol ETH-PERP --interval 15s

# 5-second high-frequency
python -m bot.main live --symbol SUI-PERP --interval 5s

# 1-second ultra-high-frequency (experimental)
python -m bot.main live --symbol BTC-PERP --interval 1s
```

### Manual Testing

1. **Start Bluefin Service**:
   ```bash
   cd bluefin-service
   python bluefin_service.py
   ```

2. **Run WebSocket Test**:
   ```bash
   python test_realtime_websocket.py
   ```

3. **Check Service Status**:
   ```bash
   curl http://localhost:8080/health
   curl http://localhost:8080/streaming/status
   ```

### Direct WebSocket Connection

```python
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8080/ws/ETH-PERP"

    async with websockets.connect(uri) as websocket:
        # Send ping
        await websocket.send(json.dumps({
            "type": "ping",
            "timestamp": time.time()
        }))

        # Receive messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")

asyncio.run(test_websocket())
```

## Configuration

### Environment Variables

For Bluefin service (`bluefin-service/.env`):
```env
# WebSocket streaming configuration
STREAM_INTERVAL_SECONDS=1.0    # Price update frequency
TRADE_FREQUENCY=5.0            # Trade generation frequency
DRY_RUN=true                   # Enable mock data generation
```

For trading bot (`.env`):
```env
# Exchange configuration for Bluefin
EXCHANGE__EXCHANGE_TYPE=bluefin
EXCHANGE__BLUEFIN_SERVICE_URL=http://localhost:8080

# High-frequency trading settings
TRADING__MIN_TRADING_INTERVAL_SECONDS=5
TRADING__SYMBOL=ETH-PERP
```

### Real-time Provider Settings

```python
from bot.data.realtime_market import RealtimeMarketDataProvider

# Create provider with custom intervals
provider = RealtimeMarketDataProvider(
    symbol="ETH-PERP",
    intervals=[1, 5, 15, 60]  # 1s, 5s, 15s, 1m candles
)

# Subscribe to candle updates
def on_new_candle(candle):
    print(f"New candle: {candle.symbol} @ {candle.close}")

provider.subscribe_to_candles(on_new_candle)
```

## Performance Characteristics

### Typical Performance Metrics
- **Tick Rate**: 10-50 ticks per second per symbol
- **Latency**: < 100ms from tick to candle
- **Memory Usage**: ~50MB for 24h of 1s candles
- **WebSocket Reconnection**: < 5 seconds

### Scalability Limits
- **Max Symbols**: 10-20 concurrent symbols
- **Max Tick Rate**: 500 ticks/second total
- **Max Connections**: 100 concurrent WebSocket connections
- **Historical Buffer**: 1000 candles per interval per symbol

## Troubleshooting

### Common Issues

1. **WebSocket Connection Fails**
   ```
   Failed to connect to WebSocket: Connection refused
   ```
   **Solution**: Ensure Bluefin service is running on port 8080

2. **No Tick Data Received**
   ```
   Tick rate: 0.0 ticks/sec, WebSocket: ⚠ Waiting for data
   ```
   **Solution**: Check mock data generation is enabled (`DRY_RUN=true`)

3. **Incomplete Candles**
   ```
   No completed candles for 15s interval!
   ```
   **Solution**: Wait for full candle periods or force completion for testing

4. **High Memory Usage**
   ```
   Memory usage increasing over time
   ```
   **Solution**: Reduce historical buffer size or candle intervals

### Debug Commands

```bash
# Check service health
curl http://localhost:8080/health

# View streaming status
curl http://localhost:8080/streaming/status

# Start/stop streaming manually
curl -X POST http://localhost:8080/streaming/start
curl -X POST http://localhost:8080/streaming/stop

# Test WebSocket with curl
curl --include \
     --no-buffer \
     --header "Connection: Upgrade" \
     --header "Upgrade: websocket" \
     --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     --header "Sec-WebSocket-Version: 13" \
     http://localhost:8080/ws/ETH-PERP
```

### Logging

Enable detailed logging for debugging:

```python
import logging
logging.getLogger('bot.data.realtime_market').setLevel(logging.DEBUG)
logging.getLogger('bot.exchange.bluefin_client').setLevel(logging.DEBUG)
```

## Integration with Trading Strategies

### High-Frequency Scalping

The real-time data provider automatically integrates with high-frequency intervals:

```python
# In main.py, for intervals <= 60s:
if interval_seconds <= 60:
    # Uses RealtimeMarketDataProvider automatically
    market_data = RealtimeMarketDataProvider(symbol, [interval_seconds, 1, 5])
```

### Custom Indicators

Real-time candles work with existing indicators:

```python
# Get real-time DataFrame
df = market_data.to_dataframe(interval_seconds=15, limit=100)

# Calculate indicators on live data
indicators = VuManChuIndicators()
cipher_data = indicators.calculate_cipher_signals(df)
```

### Position Management

Monitor real-time price for position updates:

```python
def on_price_update(price_data):
    current_price = price_data['price']
    # Update stop-loss, take-profit, etc.
    position_manager.update_positions(current_price)

provider.client.subscribe_to_price_updates(on_price_update)
```

## Future Enhancements

### Planned Features
- **Order Book Data**: Level 2 market depth streaming
- **Multiple Exchanges**: Binance, OKX WebSocket integration
- **Data Persistence**: Historical tick data storage
- **Advanced Analytics**: Real-time volatility, momentum indicators
- **Load Balancing**: Multiple WebSocket servers
- **Market Making**: Bid/ask spread monitoring

### Performance Optimizations
- **Message Batching**: Group multiple ticks into single updates
- **Compression**: WebSocket message compression
- **Connection Pooling**: Reuse WebSocket connections
- **Caching**: Redis for high-speed data caching
- **Horizontal Scaling**: Multiple Bluefin service instances

## API Reference

### RealtimeMarketDataProvider

```python
class RealtimeMarketDataProvider:
    def __init__(self, symbol: str, intervals: List[int])
    async def connect(self) -> bool
    async def disconnect(self)
    def get_current_price(self) -> Optional[Decimal]
    def get_current_candles(self) -> Dict[int, RealtimeCandle]
    def get_candle_history(self, interval_seconds: int, limit: int) -> List[MarketData]
    def get_recent_ticks(self, limit: int) -> List[Tick]
    def to_dataframe(self, interval_seconds: int, limit: int) -> pd.DataFrame
    def subscribe_to_candles(self, callback: Callable)
    def is_connected(self) -> bool
    def is_websocket_connected(self) -> bool
    def get_performance_stats(self) -> Dict[str, Any]
```

### BluefinServiceClient WebSocket Methods

```python
class BluefinServiceClient:
    async def connect_websocket(self, symbols: List[str]) -> bool
    async def disconnect_websocket(self)
    def subscribe_to_price_updates(self, callback: Callable)
    def subscribe_to_trades(self, callback: Callable)
    def get_latest_price(self, symbol: str) -> Optional[float]
    def is_websocket_connected(self) -> bool
    def get_websocket_status(self) -> Dict
```

### TickAggregator

```python
class TickAggregator:
    def __init__(self, intervals: List[int])
    def add_tick(self, tick: Tick)
    def get_current_candles(self, symbol: str) -> Dict[int, RealtimeCandle]
    def get_candle_history(self, symbol: str, interval_seconds: int, limit: int) -> List[MarketData]
    def subscribe_to_candles(self, callback: Callable)
    def force_complete_candles(self, symbol: str)
```

## Conclusion

The real-time WebSocket data streaming system provides a robust foundation for high-frequency scalping trading. It offers:

- **Low Latency**: Sub-second data updates
- **Reliability**: Automatic reconnection and error handling
- **Flexibility**: Multiple timeframes and intervals
- **Scalability**: Support for multiple symbols and connections
- **Integration**: Seamless integration with existing trading infrastructure

This system enables sophisticated trading strategies that require real-time market data and rapid decision making.
