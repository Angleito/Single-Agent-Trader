#!/bin/sh
# Simplified Bluefin entrypoint for legacy Docker compatibility
echo "Starting Bluefin SDK Service..."
exec python -u -O bluefin_sdk_service.py
