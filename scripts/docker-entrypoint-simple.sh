#!/bin/sh
# Simplified entrypoint for system Python (no virtual environment)
echo "Starting AI Trading Bot..."
exec python -m bot.main "$@"
