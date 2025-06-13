# TradingView Data Feed Integration

This document describes the TradingView Charting Library integration for the AI Trading Bot Dashboard.

## Overview

The TradingView integration provides professional trading charts with real-time price data, AI decision markers, and technical indicators. It follows the TradingView Universal Data Feed (UDF) specification for compatibility with the TradingView Charting Library.

## Files Created

### 1. `tradingview_feed.py`
Main TradingView data feed formatter module that:
- Converts bot data to TradingView-compatible JSON format
- Manages OHLCV price data with proper Unix timestamps
- Creates AI decision markers as chart annotations
- Formats technical indicators (RSI, EMA, VuManChu Cipher)
- Provides real-time data feed interface for live updates
- Supports multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)

### 2. Updated `main.py`
Extended FastAPI server with TradingView UDF API endpoints:
- `/udf/config` - TradingView configuration
- `/udf/symbols` - Symbol information
- `/udf/search` - Symbol search
- `/udf/history` - Historical OHLCV data
- `/udf/marks` - AI decision markers
- `/udf/timescale_marks` - Timescale events
- `/udf/time` - Server time
- `/tradingview/*` - Additional endpoints for data management

## Key Classes and Components

### TradingViewDataFeed
Main class that manages all TradingView data formatting:

```python
# Initialize feed
from tradingview_feed import tradingview_feed

# Add price data
bars = [OHLCVBar(timestamp, open, high, low, close, volume), ...]
tradingview_feed.add_price_data("BTC-USD", "1", bars)

# Add AI decision
decision = AIDecisionMarker(
    timestamp=int(time.time()),
    decision=DecisionType.BUY,
    price=65000.0,
    confidence=0.85,
    reasoning="Strong bullish signals detected"
)
tradingview_feed.add_ai_decision("BTC-USD", decision)

# Get historical data for TradingView
history = tradingview_feed.get_history("BTC-USD", "1", from_timestamp, to_timestamp)
```

### Data Classes

#### OHLCVBar
```python
@dataclass
class OHLCVBar:
    timestamp: int  # Unix timestamp in seconds
    open: float
    high: float
    low: float
    close: float
    volume: float
```

#### AIDecisionMarker
```python
@dataclass
class AIDecisionMarker:
    timestamp: int
    decision: DecisionType  # BUY, SELL, HOLD, CLOSE_LONG, CLOSE_SHORT
    price: float
    confidence: float
    reasoning: str
    indicator_values: Dict[str, float]
```

#### TechnicalIndicator
```python
@dataclass
class TechnicalIndicator:
    timestamp: int
    name: str
    value: Union[float, Dict[str, float]]
    parameters: Dict[str, Any]
```

## API Endpoints

### UDF (Universal Data Feed) Endpoints

#### Configuration
- **GET** `/udf/config`
- Returns TradingView configuration including supported resolutions and features

#### Symbol Information
- **GET** `/udf/symbols?symbol={symbol}`
- Returns detailed symbol information (name, type, price scale, etc.)

#### Symbol Search
- **GET** `/udf/search?query={query}&limit={limit}`
- Search for symbols by name or description

#### Historical Data
- **GET** `/udf/history?symbol={symbol}&resolution={resolution}&from={timestamp}&to={timestamp}`
- Returns OHLCV data in TradingView format:
```json
{
    "s": "ok",
    "t": [1640995200, 1640995260],
    "o": [0.18100, 0.18120],
    "h": [0.18150, 0.18140],
    "l": [0.18090, 0.18100],
    "c": [0.18120, 0.18110],
    "v": [1000, 1200]
}
```

#### AI Decision Marks
- **GET** `/udf/marks?symbol={symbol}&from={timestamp}&to={timestamp}&resolution={resolution}`
- Returns AI trading decision markers for chart display

#### Server Time
- **GET** `/udf/time`
- Returns current server time as Unix timestamp

### Additional TradingView Endpoints

#### All Symbols
- **GET** `/tradingview/symbols`
- Get complete list of available symbols

#### Technical Indicators
- **GET** `/tradingview/indicators/{symbol}?indicator={name}&from={timestamp}&to={timestamp}`
- Get technical indicator values for specified time range

#### Real-time Data
- **GET** `/tradingview/realtime/{symbol}?resolution={resolution}`
- Get most recent bar for real-time updates

#### Data Updates
- **POST** `/tradingview/update/{symbol}`
- Update trading data from bot (for real-time integration)

Request body examples:
```json
{
    "price_data": {
        "resolution": "1",
        "new_bar": {
            "timestamp": 1640995200,
            "open": 65000.0,
            "high": 65100.0,
            "low": 64900.0,
            "close": 65050.0,
            "volume": 150.5
        }
    }
}

{
    "ai_decision": {
        "action": "buy",
        "price": 65000.0,
        "confidence": 0.85,
        "reasoning": "Strong bullish signals detected",
        "timestamp": 1640995200
    }
}

{
    "indicator": {
        "name": "RSI",
        "value": 67.5,
        "parameters": {"period": 14},
        "timestamp": 1640995200
    }
}
```

#### Data Summary
- **GET** `/tradingview/summary`
- Get summary of all available data for debugging

## Integration with Trading Bot

### Real-time Data Flow

1. **Market Data Updates**: Bot receives new price data from Coinbase
   ```python
   # In bot's market data handler
   bar_data = {
       "timestamp": int(time.time()),
       "open": 65000.0,
       "high": 65100.0,
       "low": 64900.0,
       "close": 65050.0,
       "volume": 150.5
   }
   
   # Send to dashboard
   requests.post("http://dashboard:8000/tradingview/update/BTC-USD", json={
       "price_data": {"resolution": "1", "new_bar": bar_data}
   })
   ```

2. **AI Decision Updates**: Bot makes trading decisions
   ```python
   # In bot's strategy handler
   decision_data = {
       "action": "buy",
       "price": current_price,
       "confidence": 0.85,
       "reasoning": "Strong bullish signals from VuManChu Cipher",
       "timestamp": int(time.time()),
       "indicators": {"rsi": 67.5, "ema": 65200.0}
   }
   
   # Send to dashboard
   requests.post("http://dashboard:8000/tradingview/update/BTC-USD", json={
       "ai_decision": decision_data
   })
   ```

3. **Technical Indicator Updates**: Bot calculates indicators
   ```python
   # In bot's indicator calculation
   indicator_data = {
       "name": "VuManChu_Cipher_A",
       "value": {"long": 1.5, "short": -0.8},
       "parameters": {"period": 9, "source": "hlc3"},
       "timestamp": int(time.time())
   }
   
   # Send to dashboard
   requests.post("http://dashboard:8000/tradingview/update/BTC-USD", json={
       "indicator": indicator_data
   })
   ```

## Frontend Integration

### TradingView Charting Library Setup

```javascript
// Initialize TradingView widget
const widget = new TradingView.widget({
    container_id: 'tradingview_chart',
    library_path: '/charting_library/',
    datafeed: new Datafeeds.UDFCompatibleDatafeed(
        'http://localhost:8000/udf',
        undefined,
        {
            maxResponseLength: 1000,
            expectedOrder: 'latestFirst'
        }
    ),
    symbol: 'BTC-USD',
    interval: '1',
    theme: 'dark',
    autosize: true,
    studies_overrides: {},
    overrides: {
        "paneProperties.background": "#1e1e1e",
        "paneProperties.vertGridProperties.color": "#363636",
        "paneProperties.horzGridProperties.color": "#363636"
    }
});
```

### WebSocket Integration for Real-time Updates

```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'tradingview_update') {
        // Handle real-time data updates
        if (data.data.price_data) {
            // Update chart with new price data
            widget.chart().executeActionById('ResetChart');
        }
        
        if (data.data.ai_decision) {
            // Show notification for new AI decision
            console.log('New AI Decision:', data.data.ai_decision.action);
        }
    }
};
```

## Testing

### Sample Data Generation
The module includes `generate_sample_data()` function that creates:
- 100 1-minute OHLCV bars with realistic price movements
- 10 AI trading decisions with different types (BUY/SELL/HOLD)
- 50 technical indicator values (RSI and EMA)

### Running Tests
```bash
# Test the data feed module
python3 test_tradingview.py

# Test FastAPI endpoints (requires fastapi installation)
python3 test_server.py

# Start the dashboard server
python3 main.py
```

### Manual Testing
```bash
# Test UDF config
curl http://localhost:8000/udf/config

# Test historical data
curl "http://localhost:8000/udf/history?symbol=BTC-USD&resolution=1&from=1640995200&to=1640999999"

# Test AI decision marks
curl "http://localhost:8000/udf/marks?symbol=BTC-USD&from=1640995200&to=1640999999&resolution=1"

# Update data
curl -X POST http://localhost:8000/tradingview/update/BTC-USD \
  -H "Content-Type: application/json" \
  -d '{"ai_decision": {"action": "buy", "price": 65000, "confidence": 0.85, "reasoning": "Test decision", "timestamp": 1640995200}}'
```

## Configuration

### Symbol Configuration
Add new trading pairs by creating `SymbolInfo` objects:

```python
new_symbol = SymbolInfo(
    name="ETH-USD",
    ticker="ETH-USD", 
    description="Ethereum vs US Dollar",
    pricescale=100  # 2 decimal places
)
tradingview_feed.add_symbol(new_symbol)
```

### Resolution Mapping
```python
# TradingView resolution to seconds
RESOLUTION_MAP = {
    "1": 60,        # 1 minute
    "5": 300,       # 5 minutes  
    "15": 900,      # 15 minutes
    "60": 3600,     # 1 hour
    "240": 14400,   # 4 hours
    "1D": 86400,    # 1 day
    "1W": 604800,   # 1 week
    "1M": 2592000   # 1 month
}
```

## Error Handling

The implementation includes comprehensive error handling:
- Invalid symbols return `404 Not Found`
- No data available returns `{"s": "no_data"}`
- Server errors return `{"s": "error", "errmsg": "description"}`
- Data validation prevents corruption
- Automatic fallbacks for missing data

## Performance Considerations

- **Memory Management**: Automatic cleanup keeps last 1000-5000 data points
- **Caching**: Data is cached in memory for fast access
- **Real-time Updates**: Efficient bar updates without full reloads
- **Batch Operations**: Support for bulk data updates
- **Timezone Handling**: All timestamps in UTC for consistency

## Security

- **Input Validation**: All inputs validated and sanitized
- **Rate Limiting**: Consider implementing rate limiting for production
- **CORS**: Configured for development (update for production)
- **Authentication**: Add authentication for sensitive endpoints in production

## Future Enhancements

1. **Database Integration**: Store data in persistent database instead of memory
2. **WebSocket Real-time Feed**: Direct WebSocket feed for TradingView
3. **Advanced Indicators**: Support for custom indicator overlays
4. **Multiple Exchanges**: Support for multiple exchange data sources
5. **Historical Data Import**: Bulk import of historical data
6. **Data Compression**: Compress large datasets for better performance
7. **Alerting**: Integration with TradingView alerting system

## Troubleshooting

### Common Issues

1. **No Data Returned**: Check symbol exists and has data for requested timeframe
2. **Timestamp Issues**: Ensure all timestamps are Unix seconds (not milliseconds)
3. **CORS Errors**: Update CORS settings for production domain
4. **Memory Issues**: Implement data cleanup for long-running instances

### Debug Endpoints

- `/tradingview/summary` - View all available data
- WebSocket messages show real-time updates
- Server logs include detailed operation information

## Dependencies

- **FastAPI**: Web framework for API endpoints
- **Python 3.12+**: Core language requirements
- **Pydantic**: Data validation and serialization
- **TradingView Charting Library**: Frontend charting (separate installation)

This integration provides a complete solution for displaying professional trading charts with AI trading bot data in the TradingView format.