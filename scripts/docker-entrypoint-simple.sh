#!/bin/sh
# Simplified entrypoint for legacy Docker compatibility
echo "Starting AI Trading Bot..."
exec python -m bot.main "$@"
