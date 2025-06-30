#!/bin/bash

# AI Trading Bot - Optimized Startup Script
# This script starts the bot with resource-optimized concurrency settings

set -e

echo "🚀 Starting AI Trading Bot with Optimized Concurrency Settings..."

# Set default environment if not provided
export ENVIRONMENT=${ENVIRONMENT:-production}

# Apply optimized concurrency settings
echo "📊 Applying optimized concurrency settings..."

# System-level optimizations
export SYSTEM__MAX_CONCURRENT_TASKS=${SYSTEM__MAX_CONCURRENT_TASKS:-4}
export SYSTEM__THREAD_POOL_SIZE=${SYSTEM__THREAD_POOL_SIZE:-2}
export SYSTEM__ASYNC_TIMEOUT=${SYSTEM__ASYNC_TIMEOUT:-15.0}
export SYSTEM__TASK_BATCH_SIZE=${SYSTEM__TASK_BATCH_SIZE:-2}
export SYSTEM__UPDATE_FREQUENCY_SECONDS=${SYSTEM__UPDATE_FREQUENCY_SECONDS:-45.0}

# WebSocket optimizations
export SYSTEM__WEBSOCKET_QUEUE_SIZE=${SYSTEM__WEBSOCKET_QUEUE_SIZE:-200}
export SYSTEM__WEBSOCKET_MAX_RETRIES=${SYSTEM__WEBSOCKET_MAX_RETRIES:-10}
export SYSTEM__WEBSOCKET_PING_INTERVAL=${SYSTEM__WEBSOCKET_PING_INTERVAL:-30}
export SYSTEM__WEBSOCKET_PING_TIMEOUT=${SYSTEM__WEBSOCKET_PING_TIMEOUT:-15}
export SYSTEM__WEBSOCKET_HEALTH_CHECK_INTERVAL=${SYSTEM__WEBSOCKET_HEALTH_CHECK_INTERVAL:-60}

# Functional Programming Runtime optimizations
export FP_RUNTIME_ENABLED=${FP_RUNTIME_ENABLED:-true}
export FP_RUNTIME_MODE=${FP_RUNTIME_MODE:-hybrid}
export FP_MAX_CONCURRENT_EFFECTS=${FP_MAX_CONCURRENT_EFFECTS:-25}
export FP_EFFECT_TIMEOUT=${FP_EFFECT_TIMEOUT:-20.0}
export FP_ERROR_RECOVERY=${FP_ERROR_RECOVERY:-true}
export FP_METRICS_ENABLED=${FP_METRICS_ENABLED:-true}
export FP_SCHEDULER_ENABLED=${FP_SCHEDULER_ENABLED:-true}
export FP_ASYNC_RUNTIME=${FP_ASYNC_RUNTIME:-true}
export FP_DEBUG_MODE=${FP_DEBUG_MODE:-false}

# Exchange optimizations
export EXCHANGE__RATE_LIMIT_REQUESTS=${EXCHANGE__RATE_LIMIT_REQUESTS:-8}
export EXCHANGE__RATE_LIMIT_WINDOW_SECONDS=${EXCHANGE__RATE_LIMIT_WINDOW_SECONDS:-60}
export EXCHANGE__API_TIMEOUT=${EXCHANGE__API_TIMEOUT:-15}

# Python optimizations
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore::UserWarning,ignore::DeprecationWarning,ignore::SyntaxWarning

# Memory and GC optimizations
export PYTHONHASHSEED=random
export PYTHONASYNCIODEBUG=0

echo "✅ Configuration applied:"
echo "  - Max concurrent tasks: $SYSTEM__MAX_CONCURRENT_TASKS"
echo "  - Thread pool size: $SYSTEM__THREAD_POOL_SIZE"
echo "  - Async timeout: $SYSTEM__ASYNC_TIMEOUT"
echo "  - Task batch size: $SYSTEM__TASK_BATCH_SIZE"
echo "  - Update frequency: $SYSTEM__UPDATE_FREQUENCY_SECONDS"
echo "  - WebSocket queue size: $SYSTEM__WEBSOCKET_QUEUE_SIZE"
echo "  - FP max concurrent effects: $FP_MAX_CONCURRENT_EFFECTS"
echo "  - FP effect timeout: $FP_EFFECT_TIMEOUT"

# Validate configuration
if [ -f "scripts/validate_config.py" ]; then
    echo "🔍 Validating configuration..."
    python scripts/validate_config.py --concurrency-check
fi

# Check system resources
echo "📊 System Resource Check:"
echo "  - Available CPU cores: $(nproc)"
echo "  - Available memory: $(free -h | awk '/^Mem:/ {print $7}')"
echo "  - Process limit: $(ulimit -u)"

# Check if we're in a container
if [ -f /.dockerenv ]; then
    echo "🐳 Running in Docker container"
    echo "  - Container memory limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null | awk '{print int($1/1024/1024) " MB"}' || echo 'Not available')"
    echo "  - Container CPU limit: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo 'Not limited')"
fi

# Create necessary directories
mkdir -p logs data tmp

# Set file permissions
chmod 755 logs data tmp

# Check if configuration file exists
if [ -f "config/concurrency_optimized.json" ]; then
    echo "📋 Using optimized configuration: config/concurrency_optimized.json"
    export CONFIG_FILE="config/concurrency_optimized.json"
fi

# Load environment variables
if [ -f ".env" ]; then
    echo "🔧 Loading environment variables from .env"
    set -a
    source .env
    set +a
else
    echo "⚠️  No .env file found, using environment defaults"
fi

# Start the bot
echo "🎯 Starting AI Trading Bot..."
if [ "$1" = "dev" ] || [ "$ENVIRONMENT" = "development" ]; then
    echo "🛠️  Development mode"
    python -m bot.main live --force --config="$CONFIG_FILE"
elif [ "$1" = "test" ]; then
    echo "🧪 Test mode with dry run"
    export SYSTEM__DRY_RUN=true
    python -m bot.main live --force --config="$CONFIG_FILE"
else
    echo "🚀 Production mode"
    python -m bot.main live --force --config="$CONFIG_FILE"
fi

echo "✅ AI Trading Bot startup complete"
