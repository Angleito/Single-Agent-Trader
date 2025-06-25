#!/bin/bash
set -e

# Basic health check
curl -f --connect-timeout 10 --max-time 30 http://localhost:8080/health || exit 1

# Ubuntu-specific resource checks
if [ "$UBUNTU_DEPLOYMENT" = "true" ]; then
    # Check disk space with Ubuntu-compatible commands
    if command -v df >/dev/null 2>&1; then
        DISK_USAGE=$(df /app 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//' || echo "0")
        if [ "$DISK_USAGE" -gt 90 ]; then
            echo "Disk usage critical: ${DISK_USAGE}%"
            exit 1
        fi
    fi

    # Check memory usage with Ubuntu-compatible commands
    if command -v free >/dev/null 2>&1; then
        MEM_USAGE=$(free 2>/dev/null | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}' || echo "0")
        if [ "$MEM_USAGE" -gt 95 ]; then
            echo "Memory usage critical: ${MEM_USAGE}%"
            exit 1
        fi
    fi
fi

echo "Health check passed"
