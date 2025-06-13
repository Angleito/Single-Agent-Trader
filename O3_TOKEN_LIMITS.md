# OpenAI o3 Model Token Configuration

## Token Limits Updated ✅

The AI Trading Bot has been configured to use OpenAI's o3 model with proper token limits:

### **Configuration Changes Made:**
- **Max Tokens**: Updated from 1,000 to **30,000** tokens per request
- **Model Default**: Set to `o3` across all configuration files
- **Rate Limit Awareness**: Configured for o3's token constraints

### **Files Updated:**
1. **`bot/config.py`**: 
   - Default max_tokens: 30000
   - Field validation: up to 50000 (with note about o3 limit)
   - Template config updated

2. **`.env.example`**: 
   - OPENAI_MAX_TOKENS=30000 with explanatory comment
   - OPENAI_MODEL=o3 as default

3. **`config/development.json`**: 
   - max_tokens: 30000
   - model_name: "o3"

### **Token Limit Understanding:**

**OpenAI o3 Constraints:**
- **30,000 tokens maximum** per request
- This includes both input (prompt) and output (response) tokens
- Rate limits may apply (tokens per minute)

**How the Bot Uses Tokens:**
- **Market Data**: ~500-1000 tokens (OHLCV data, indicators)
- **Context**: ~1000-2000 tokens (current position, risk status)
- **Instructions**: ~500-1000 tokens (trading guidelines, risk rules)
- **AI Response**: ~500-2000 tokens (decision + rationale)
- **Total Typical Usage**: ~3000-6000 tokens per trading decision

**Safety Buffer:**
With 30,000 token limit, the bot has plenty of headroom for:
- Complex market analysis
- Detailed reasoning and rationale
- Extended context for better decisions
- Error handling and retries

### **Benefits of 30K Token Limit:**
1. **Rich Context**: More market history and indicators
2. **Detailed Analysis**: o3 can provide comprehensive reasoning
3. **Better Decisions**: More context leads to better trading choices
4. **Robust Responses**: Detailed rationales for transparency

### **Monitoring Token Usage:**
The bot will automatically:
- Track token usage per request
- Log high token usage for optimization
- Handle token limit errors gracefully
- Fall back to simpler prompts if needed

### **Usage Example:**
```bash
# Start with o3 and 30K token limit
docker-compose up

# Monitor token usage in logs
docker-compose logs -f ai-trading-bot | grep "tokens"
```

The configuration ensures optimal use of o3's capabilities while respecting OpenAI's token constraints. 🚀