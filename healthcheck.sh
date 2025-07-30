#!/bin/sh
# Health check script for AI Trading Bot

# Check if the main process is running
if pgrep -f "python -m bot.main" > /dev/null; then
    echo "Trading bot process is running"
    exit 0
else
    echo "Trading bot process is not running"
    exit 1
fi