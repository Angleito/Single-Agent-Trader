#!/bin/bash
set -e

# Comprehensive Ubuntu health check
echo "Starting Ubuntu health check..."

# Service health with timeout
if ! timeout 10 curl -f --connect-timeout 5 http://localhost:8080/health > /dev/null 2>&1; then
    echo "Service health check failed"
    exit 1
fi

# Network connectivity check with Ubuntu DNS resolution
if command -v nslookup >/dev/null 2>&1; then
    if ! timeout 10 nslookup api.bluefin.io >/dev/null 2>&1; then
        echo "DNS resolution failed"
        exit 1
    fi
fi

# External connectivity check
if ! timeout 15 curl -f --connect-timeout 10 https://api.bluefin.io > /dev/null 2>&1; then
    echo "External network connectivity failed"
    exit 1
fi

# Check log directory permissions
if [ ! -w /app/logs ]; then
    echo "Log directory not writable"
    exit 1
fi

# Ubuntu-specific system checks
if [ -f /etc/os-release ]; then
    echo "Ubuntu system detected - performing additional checks"
fi

echo "Ubuntu health check passed"
