# MCP Memory Server Logging Guide

## Overview

The MCP (Memory-Context Protocol) memory server now includes comprehensive structured logging to track trading decisions, memory queries, and trade outcomes. This enables detailed analysis of the bot's learning process and performance over time.

## Log Structure

All logs are written in JSON Lines (JSONL) format with daily rotation. Each line is a complete JSON object that can be parsed independently.

### Log Files

1. **decisions_YYYYMMDD.jsonl** - Trading decisions with full market context
2. **outcomes_YYYYMMDD.jsonl** - Trade results and performance metrics
3. **memory_YYYYMMDD.jsonl** - Memory queries and pattern statistics

### Log Location

- Default: `logs/trades/`
- Docker: `/app/logs/trades/`
- Configurable via `TradeLogger(log_dir=Path("custom/path"))`

## Log Schemas

### Trade Decision Log

```json
{
  "timestamp": "ISO 8601 timestamp",
  "experience_id": "unique identifier",
  "symbol": "trading pair",
  "price": current market price,
  "position": {
    "side": "FLAT|LONG|SHORT",
    "size": position size,
    "unrealized_pnl": current P&L
  },
  "decision": {
    "action": "LONG|SHORT|CLOSE|HOLD",
    "size_pct": percentage of capital,
    "rationale": "LLM explanation",
    "leverage": leverage used
  },
  "indicators": {
    "rsi": RSI value,
    "cipher_b_wave": wave indicator,
    "cipher_a_dot": momentum dot,
    "ema_trend": "UP|DOWN"
  },
  "dominance": {
    "stablecoin_dominance": percentage,
    "dominance_24h_change": change percentage
  },
  "memory_used": boolean,
  "similar_experiences": count of relevant memories
}
```

### Trade Outcome Log

```json
{
  "timestamp": "ISO 8601 timestamp",
  "experience_id": "unique identifier",
  "entry_price": entry price,
  "exit_price": exit price,
  "pnl": profit/loss amount,
  "pnl_pct": profit/loss percentage,
  "duration_minutes": trade duration,
  "success": boolean,
  "insights": "learned insights or reflection"
}
```

### Memory Query Log

```json
{
  "timestamp": "ISO 8601 timestamp",
  "type": "query",
  "query": {
    "current_price": price at query time,
    "indicators": current indicator values,
    "max_results": requested results,
    "min_similarity": similarity threshold
  },
  "results_count": number of matches,
  "execution_time_ms": query performance,
  "top_similarities": [array of similarity scores]
}
```

### Pattern Statistics Log

```json
{
  "timestamp": "ISO 8601 timestamp",
  "type": "pattern_stats",
  "patterns": {
    "pattern_name": {
      "count": number of occurrences,
      "success_rate": win rate (0-1),
      "avg_pnl": average profit/loss,
      "total_pnl": cumulative P&L
    }
  },
  "total_patterns": count
}
```

## Usage Examples

### Reading Logs for Analysis

```python
import json
from pathlib import Path

# Read all trade decisions
decisions = []
with open("logs/trades/decisions_20250614.jsonl") as f:
    for line in f:
        decisions.append(json.loads(line))

# Find all LONG trades
long_trades = [d for d in decisions if d["decision"]["action"] == "LONG"]

# Calculate average RSI for winning trades
outcomes = []
with open("logs/trades/outcomes_20250614.jsonl") as f:
    for line in f:
        outcomes.append(json.loads(line))

winning_trades = [o for o in outcomes if o["success"]]
avg_pnl = sum(o["pnl"] for o in winning_trades) / len(winning_trades)
```

### Monitoring Memory Performance

```python
# Analyze memory query performance
queries = []
with open("logs/trades/memory_20250614.jsonl") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "query":
            queries.append(data)

avg_query_time = sum(q["execution_time_ms"] for q in queries) / len(queries)
print(f"Average query time: {avg_query_time:.1f}ms")
```

## Key Logging Points

1. **Trade Entry**: Full market state, indicators, and memory context
2. **Position Updates**: Periodic snapshots with unrealized P&L
3. **Trade Exit**: Final outcome, duration, and insights
4. **Memory Operations**: Query performance and similarity scores
5. **Pattern Analysis**: Success rates updated periodically

## Configuration

### Environment Variables

- `LLM__ENABLE_COMPLETION_LOGGING`: Enable LLM decision logging
- `LOG_LEVEL`: Set to DEBUG for verbose memory operations

### Docker Compose

```yaml
volumes:
  - ./logs:/app/logs  # Persist logs outside container
```

## Benefits

1. **Performance Analysis**: Track win rates, average P&L, and pattern success
2. **Memory Debugging**: Understand which past experiences influence decisions
3. **Strategy Optimization**: Identify successful patterns and market conditions
4. **Audit Trail**: Complete record of all trading decisions and rationale
5. **Learning Insights**: Track how the bot improves over time

## Log Rotation

- Files rotate daily at midnight UTC
- Recommended: Archive logs older than 30 days
- Use log aggregation tools for long-term analysis

## Privacy & Security

- No API keys or sensitive credentials are logged
- Trade sizes are logged as percentages, not absolute values
- Consider encrypting logs if storing sensitive strategies