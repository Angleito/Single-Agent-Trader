#!/bin/bash
set -e

# Lightweight health check for AI trading bot
cd /app

# Check if Python can start and PYTHONPATH is correct
timeout 8 python -c "
import sys
import os
print('Python OK')

# Check if we can find the bot module without importing it
import importlib.util
spec = importlib.util.find_spec('bot')
if spec is None:
    print('ERROR: bot module not found')
    exit(1)
print('Bot module path OK')

# Quick test of minimal imports without heavy dependencies
try:
    import logging
    import json
    print('Basic imports OK')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
" || exit 1

echo "Health check passed"
