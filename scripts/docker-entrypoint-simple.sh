#!/bin/sh
# Simplified entrypoint for legacy Docker compatibility
echo "Starting AI Trading Bot..."
exec /app/.venv/bin/python -m bot.main "$@"
