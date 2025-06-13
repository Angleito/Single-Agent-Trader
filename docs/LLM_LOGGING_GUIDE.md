# LLM Completion Logging Guide

This guide covers the enhanced LLM logging system implemented in the AI Trading Bot, which provides comprehensive tracking and monitoring of LLM chat completions, performance metrics, and trading decisions.

## Overview

The enhanced logging system provides:

- **Structured logging** of LLM requests and responses
- **Performance tracking** including response times and token usage
- **Cost estimation** for OpenAI API usage
- **Trading decision correlation** with market context
- **LangChain integration** for detailed chain execution tracing
- **Dashboard integration** for real-time monitoring

## Architecture

### Core Components

1. **`ChatCompletionLogger`** - Main logging class for LLM interactions
2. **`LangChainCallbackHandler`** - Callback handler for LangChain operations
3. **Enhanced LLMAgent** - Updated agent with integrated logging
4. **Configuration System** - Settings for logging preferences

### Data Flow

```
Market State → LLM Agent → Enhanced Logging → Log Files
                    ↓           ↓
               LangChain → Callback Handler → Structured Logs
                    ↓
            Trading Decision → Decision Logging → Analytics
```

## Configuration

### LLM Logging Settings

Configure logging through the `LLMSettings` in your configuration:

```json
{
  "llm": {
    "enable_completion_logging": true,
    "completion_log_level": "INFO",
    "completion_log_file": "logs/llm_completions.log",
    "log_prompt_preview_length": 500,
    "log_response_preview_length": 1000,
    "enable_performance_tracking": true,
    "enable_langchain_callbacks": true,
    "log_market_context": true,
    "enable_token_usage_tracking": true,
    "performance_log_interval": 10
  }
}
```

### Environment Variables (Docker)

```bash
# Enable/disable completion logging
LLM__ENABLE_COMPLETION_LOGGING=true

# Set log level for completions
LLM__COMPLETION_LOG_LEVEL=INFO

# Specify log file path
LLM__COMPLETION_LOG_FILE=logs/llm_completions.log

# Enable performance tracking
LLM__ENABLE_PERFORMANCE_TRACKING=true

# Enable LangChain callbacks
LLM__ENABLE_LANGCHAIN_CALLBACKS=true

# Include market context in logs
LLM__LOG_MARKET_CONTEXT=true
```

## Log Structure

### Completion Request Log

```json
{
  "event_type": "completion_request",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "session_id": "abc12345",
  "request_id": "req_67890def",
  "completion_number": 1,
  "model": "o3-mini",
  "temperature": 0.1,
  "max_tokens": 30000,
  "prompt_length": 1250,
  "prompt_preview": "You are an expert cryptocurrency futures trader...",
  "market_context": {
    "symbol": "BTC-USD",
    "current_price": 45230.50,
    "cipher_a_dot": 0.75,
    "cipher_b_wave": 0.32,
    "rsi": 62.5,
    "stablecoin_dominance": 8.2,
    "market_sentiment": "BULLISH"
  }
}
```

### Completion Response Log

```json
{
  "event_type": "completion_response",
  "timestamp": "2024-01-15T10:30:46.890Z",
  "session_id": "abc12345",
  "request_id": "req_67890def",
  "success": true,
  "response_time_ms": 1767,
  "token_usage": {
    "prompt_tokens": 150,
    "completion_tokens": 75,
    "total_tokens": 225
  },
  "cost_estimate_usd": 0.000675,
  "response_preview": "{\"action\": \"LONG\", \"size_pct\": 15.0, \"rationale\": \"Bullish momentum...\"}"
}
```

### Trading Decision Log

```json
{
  "event_type": "trading_decision",
  "timestamp": "2024-01-15T10:30:47.123Z",
  "session_id": "abc12345",
  "request_id": "req_67890def",
  "action": "LONG",
  "size_pct": 15.0,
  "rationale": "Bullish momentum with healthy RSI",
  "symbol": "BTC-USD",
  "current_price": 45230.50,
  "validation_result": "PASSED",
  "risk_assessment": "APPROVED - Within risk parameters",
  "indicators": {
    "cipher_a_dot": 0.75,
    "cipher_b_wave": 0.32,
    "rsi": 62.5
  }
}
```

### Performance Metrics Log

```json
{
  "event_type": "performance_metrics",
  "timestamp": "2024-01-15T10:45:30.456Z",
  "session_id": "abc12345",
  "total_completions": 10,
  "avg_response_time_ms": 1520,
  "min_response_time_ms": 890,
  "max_response_time_ms": 2340,
  "total_tokens": 2250,
  "total_cost_estimate_usd": 0.006750,
  "tokens_per_second": 1.48
}
```

## Usage Examples

### Basic Usage

```python
from bot.logging import create_llm_logger

# Create logger
completion_logger = create_llm_logger(
    log_level="INFO",
    log_file="logs/llm_completions.log"
)

# Log a request
request_id = completion_logger.log_completion_request(
    prompt="Should I buy BTC?",
    model="o3-mini",
    temperature=0.1,
    max_tokens=1000,
    market_context={"symbol": "BTC-USD", "price": 45000}
)

# Log response
completion_logger.log_completion_response(
    request_id=request_id,
    response={"action": "LONG", "rationale": "Bullish signals"},
    response_time=1.5,
    token_usage={"total_tokens": 200},
    success=True
)
```

### Integration with LLM Agent

The enhanced logging is automatically integrated when using the `LLMAgent`:

```python
from bot.strategy.llm_agent import LLMAgent

# Create agent (logging configured via settings)
agent = LLMAgent()

# Analyze market (automatically logs completion)
trade_action = await agent.analyze_market(market_state)

# Get status including logging metrics
status = agent.get_status()
print(f"Completions: {status['completion_count']}")
print(f"Avg Response Time: {status['avg_response_time_ms']}ms")
```

### LangChain Callback Integration

```python
from bot.logging import create_llm_logger, create_langchain_callback

# Create logger and callback
completion_logger = create_llm_logger()
callback_handler = create_langchain_callback(completion_logger)

# Use with LangChain chain
result = await chain.ainvoke(
    inputs,
    config={"callbacks": [callback_handler]}
)
```

## Docker Integration

### Log Volume Mounts

The Docker setup includes dedicated volume mounts for LLM logs:

```yaml
volumes:
  - ./logs:/app/logs
  - ./logs/llm_completions:/app/logs/llm_completions
  - ./logs/trading_decisions:/app/logs/trading_decisions
```

### Dashboard Access

The dashboard backend has read-only access to LLM logs:

```yaml
volumes:
  - ./logs/llm_completions:/app/llm-logs:ro
  - ./logs/trading_decisions:/app/decision-logs:ro
```

## Monitoring and Analytics

### Performance Metrics

Track key metrics across your trading sessions:

- **Response Time**: Average, min, max completion times
- **Token Usage**: Prompt, completion, and total tokens
- **Cost Tracking**: Estimated API costs by model
- **Success Rate**: Completion success vs. error rate
- **Throughput**: Tokens per second, completions per hour

### Cost Analysis

Monitor API costs with automatic estimation:

```python
# Get current session metrics
metrics = completion_logger.log_performance_metrics()

print(f"Total Cost: ${metrics['total_cost_estimate_usd']:.6f}")
print(f"Cost per Completion: ${metrics['total_cost_estimate_usd'] / metrics['total_completions']:.6f}")
```

### Log Rotation

Logs are automatically rotated to prevent disk space issues:

- **Max File Size**: 50MB per log file
- **Backup Count**: 5 rotated files kept
- **Format**: JSON structured logging for easy parsing

## Troubleshooting

### Common Issues

1. **Log Files Not Created**
   - Check file permissions in logs directory
   - Verify log directory exists (`mkdir -p logs`)
   - Check `enable_completion_logging` setting

2. **Missing Token Usage Data**
   - Token usage extraction requires special handling
   - Currently limited by LangChain integration
   - Consider direct OpenAI client for detailed metrics

3. **High Log Volume**
   - Adjust `log_prompt_preview_length` and `log_response_preview_length`
   - Increase `performance_log_interval` to reduce frequency
   - Consider log level adjustment (INFO vs DEBUG)

4. **LangChain Callbacks Not Working**
   - Verify LangChain is installed and available
   - Check `enable_langchain_callbacks` setting
   - Ensure callbacks are passed to chain invocation

### Debug Mode

Enable debug logging for detailed information:

```python
completion_logger = create_llm_logger(log_level="DEBUG")
```

### Log File Analysis

Use standard tools to analyze log files:

```bash
# View latest completions
tail -f logs/llm_completions.log

# Count completions by type
grep "completion_request" logs/llm_completions.log | wc -l

# Extract performance metrics
grep "performance_metrics" logs/llm_completions.log | jq .

# Analyze error rates
grep "\"success\": false" logs/llm_completions.log | wc -l
```

## Best Practices

1. **Structured Logging**: Always use JSON format for easy parsing
2. **Context Preservation**: Include relevant market context in logs
3. **Performance Monitoring**: Regular metrics logging for optimization
4. **Cost Awareness**: Monitor token usage and API costs
5. **Error Tracking**: Log failures for debugging and improvement
6. **Privacy**: Be cautious about logging sensitive data in prompts
7. **Storage Management**: Implement log rotation and cleanup policies

## Future Enhancements

Planned improvements to the logging system:

- [ ] Real-time dashboard visualization
- [ ] Advanced cost analytics and budgeting
- [ ] Integration with external monitoring tools
- [ ] Enhanced token usage tracking
- [ ] Automated performance alerts
- [ ] Machine learning on completion patterns
- [ ] Export capabilities for analysis tools

## See Also

- [Configuration Guide](./Environment_Setup_Guide.md)
- [Docker Setup](../DOCKER.md)
- [Performance Optimization](./Performance_Optimization.md)
- [Dashboard Documentation](../dashboard/README.md)