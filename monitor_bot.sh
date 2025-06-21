#!/bin/bash
# Monitor bot trading activity

echo "Starting bot with SUI-PERP trading..."
echo "Monitoring for trading decisions (this may take up to 2 minutes for first decision)..."
echo "============================================"

# Run bot in background
poetry run python -m bot.main live --symbol SUI-PERP > bot_output.log 2>&1 &
BOT_PID=$!

# Monitor for 2 minutes
END=$(($(date +%s) + 120))

while [ $(date +%s) -lt $END ]; do
    # Check if bot is still running
    if ! ps -p $BOT_PID > /dev/null; then
        echo "Bot stopped unexpectedly!"
        cat bot_output.log | tail -50
        exit 1
    fi

    # Look for trading activity
    if grep -q "Scalping analysis" bot_output.log; then
        echo "✅ Trading analysis started!"
        grep -A 5 "Scalping analysis" bot_output.log | tail -10
    fi

    if grep -q "Making trading decision" bot_output.log; then
        echo "🤖 LLM making decision!"
        grep -A 10 "Making trading decision" bot_output.log | tail -20
    fi

    if grep -q "PAPER TRADING DECISION" bot_output.log; then
        echo "🎯 TRADE DECISION MADE!"
        grep -A 20 "PAPER TRADING DECISION" bot_output.log | tail -30
        echo "============================================"
    fi

    sleep 5
done

# Stop bot
kill $BOT_PID 2>/dev/null

echo "Monitoring complete. Final output:"
tail -50 bot_output.log | grep -E "(Trade|DECISION|Scalping|action)"
