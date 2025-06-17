# MCP Memory Logging Enhancements

## Overview

Enhanced the MCP (Model Context Protocol) memory server integration to provide comprehensive logging of all memory operations, making paper trades and learning activities visible in Docker logs.

## What Was Enhanced

### 1. **Memory Server Logging** (`bot/mcp/memory_server.py`)
- Added detailed logging when storing trading experiences
- Logs include experience ID, action, price, symbol, and patterns
- Added logging for memory queries showing results and execution time
- Enhanced connection status logging with clear indicators

### 2. **Experience Manager Logging** (`bot/learning/experience_manager.py`)
- Enhanced logging for trade lifecycle tracking
- Clear indicators when experiences are linked to orders
- Detailed logging when trades are started and completed
- Shows PnL, duration, and win/loss status

### 3. **Paper Trading Logging** (`bot/paper_trading.py`)
- Added enhanced logging for paper trade execution
- Shows trade ID, symbol, price, and fees
- Clear win/loss indicators for closed positions
- Percentage gains/losses displayed

### 4. **Memory-Enhanced Agent Logging** (`bot/strategy/memory_enhanced_agent.py`)
- Logs when memory context is applied to decisions
- Shows number of similar experiences considered
- Debug logging for success rates of similar trades

### 5. **Main Trading Loop Enhancements** (`bot/main.py`)
- Added periodic pattern performance logging (every 100 loops)
- Shows which agent type is being used (Memory-Enhanced vs Standard)
- Enhanced trade execution logging with experience IDs
- MCP integration status clearly logged

## Configuration Added

Added MCP configuration to `.env` file:
```env
# MCP MEMORY AND LEARNING CONFIGURATION
MCP_ENABLED=true
MCP_SERVER_URL=http://mcp-memory:8765
MCP_MEMORY_RETENTION_DAYS=90
MCP_ENABLE_PATTERN_LEARNING=true
MCP_LEARNING_RATE=0.1
MCP_MIN_SAMPLES_FOR_PATTERN=5
MCP_MAX_MEMORIES_PER_QUERY=10
MCP_SIMILARITY_THRESHOLD=0.7
MCP_TRACK_TRADE_LIFECYCLE=true
MCP_REFLECTION_DELAY_MINUTES=5
MEM0_API_KEY=
```

## Log Format Examples

### Memory Storage
```
💾 MCP Memory: Stored experience abc12345... | Action: LONG | Price: $2540.59 | Symbol: ETH-USD | Patterns: bullish_trend, high_rsi
```

### Trade Tracking
```
🚀 Experience Manager: Started tracking LONG trade | Trade ID: trade_1234 | Price: $2541.86 | Size: 2.95207 | Experience: abc12345...
```

### Trade Completion
```
🏁 Experience Manager: Trade completed | ID: trade_1234 | PnL: $45.23 (+1.78%) | Duration: 23.5min | ✅ WIN
```

### Memory Query
```
🔍 MCP Memory: Query completed | Found 5 similar experiences (from 127 total) | Time: 12.3ms
```

### Pattern Performance
```
📊 === MCP Pattern Performance Update ===
  📈 bullish_trend: 75.0% win rate | 8 trades | Avg PnL: $23.45
  📈 high_rsi_reversal: 66.7% win rate | 6 trades | Avg PnL: $15.23
```

## Viewing the Logs

To see memory operations in Docker:
```bash
# View MCP memory server logs
docker-compose logs -f mcp-memory | grep "MCP"

# View trading bot memory operations
docker-compose logs -f ai-trading-bot | grep -E "MCP|Memory|Experience"

# View all paper trading operations
docker-compose logs -f ai-trading-bot | grep "Paper Trading"

# Follow all logs with memory highlights
docker-compose logs -f | grep -E "💾|🧠|📊|🔍|🚀|🏁|📈"
```

## Testing

Created `test_memory_logging.py` to verify memory logging is working:
```bash
python test_memory_logging.py
```

## Benefits

1. **Complete Visibility**: Every memory operation is now logged with clear indicators
2. **Easy Debugging**: Emoji indicators make it easy to spot different types of operations
3. **Performance Tracking**: See query times and pattern performance
4. **Learning Insights**: Track how the bot learns from past experiences
5. **Paper Trade Tracking**: All paper trades are logged with their memory associations

## Next Steps

With MCP enabled and proper logging in place, the bot will now:
1. Store every trading decision in memory
2. Query past experiences before making new decisions
3. Track trade outcomes and learn from them
4. Build pattern recognition over time
5. Log all these operations clearly in Docker logs
