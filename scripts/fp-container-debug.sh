#!/bin/bash

# Functional Programming Container Debug and Monitoring Tool
# Comprehensive debugging capabilities for FP runtime in containers

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER_NAME="${1:-ai-trading-bot}"
LOG_DIR="$PROJECT_ROOT/logs/fp"
DEBUG_OUTPUT_DIR="$PROJECT_ROOT/debug_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG:${NC} $1"
}

info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

header() {
    echo -e "${PURPLE}======================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}======================================${NC}"
}

# Utility functions
check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
}

check_container() {
    if ! docker ps --format "table {{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
        error "Container '$CONTAINER_NAME' is not running"
        echo "Available containers:"
        docker ps --format "table {{.Names}}\t{{.Status}}"
        exit 1
    fi
}

create_debug_dir() {
    mkdir -p "$DEBUG_OUTPUT_DIR"
    log "Debug output directory: $DEBUG_OUTPUT_DIR"
}

# FP Runtime diagnostic functions
check_fp_environment() {
    header "FP Environment Variables"
    
    docker exec "$CONTAINER_NAME" bash -c '
        echo "=== FP Runtime Configuration ==="
        env | grep -E "^FP_" | sort || echo "No FP environment variables found"
        echo ""
        echo "=== Python Path Configuration ==="
        python -c "import sys; [print(f\"  {p}\") for p in sys.path if \"/app\" in p]"
        echo ""
        echo "=== FP Module Availability ==="
        python -c "
try:
    import bot.fp
    print(\"✓ bot.fp module available\")
    
    import bot.fp.runtime.interpreter
    print(\"✓ FP interpreter available\")
    
    import bot.fp.runtime.scheduler
    print(\"✓ FP scheduler available\")
    
    import bot.fp.adapters
    print(\"✓ FP adapters available\")
except ImportError as e:
    print(f\"✗ FP import error: {e}\")
        "
    ' 2>/dev/null || error "Failed to check FP environment in container"
}

check_fp_runtime_stats() {
    header "FP Runtime Statistics"
    
    docker exec "$CONTAINER_NAME" python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/bot/fp')

try:
    from bot.fp.runtime.interpreter import get_interpreter
    from bot.fp.runtime.scheduler import get_scheduler
    
    print('=== Effect Interpreter Stats ===')
    interpreter = get_interpreter()
    stats = interpreter.get_runtime_stats()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f'{key}:')
            for k, v in value.items():
                print(f'  {k}: {v}')
        else:
            print(f'{key}: {value}')
    
    print('')
    print('=== Scheduler Status ===')
    scheduler = get_scheduler()
    status = scheduler.get_status()
    
    for key, value in status.items():
        if isinstance(value, dict):
            print(f'{key}:')
            for k, v in value.items():
                print(f'  {k}: {v}')
        else:
            print(f'{key}: {value}')
            
except Exception as e:
    print(f'Error getting FP runtime stats: {e}')
    import traceback
    traceback.print_exc()
" 2>/dev/null || error "Failed to get FP runtime stats"
}

check_fp_adapters() {
    header "FP Adapter Status"
    
    docker exec "$CONTAINER_NAME" python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/bot/fp')

try:
    print('=== Testing FP Adapters ===')
    
    # Test compatibility layer
    try:
        from bot.fp.adapters.compatibility_layer import CompatibilityLayer
        comp_layer = CompatibilityLayer()
        print('✓ CompatibilityLayer: Available')
    except Exception as e:
        print(f'✗ CompatibilityLayer: {e}')
    
    # Test exchange adapter
    try:
        from bot.fp.adapters.exchange_adapter import ExchangeAdapter
        print('✓ ExchangeAdapter: Available')
    except Exception as e:
        print(f'✗ ExchangeAdapter: {e}')
    
    # Test strategy adapter
    try:
        from bot.fp.adapters.strategy_adapter import StrategyAdapter
        print('✓ StrategyAdapter: Available')
    except Exception as e:
        print(f'✗ StrategyAdapter: {e}')
        
    # Test other key adapters
    adapters = [
        'market_data_adapter',
        'paper_trading_adapter',
        'position_manager_adapter',
        'trading_type_adapter'
    ]
    
    for adapter in adapters:
        try:
            module = __import__(f'bot.fp.adapters.{adapter}', fromlist=[adapter])
            print(f'✓ {adapter}: Available')
        except Exception as e:
            print(f'✗ {adapter}: {e}')
            
except Exception as e:
    print(f'Error checking FP adapters: {e}')
    import traceback
    traceback.print_exc()
" 2>/dev/null || error "Failed to check FP adapters"
}

check_fp_directories() {
    header "FP Directory Structure"
    
    docker exec "$CONTAINER_NAME" bash -c '
        echo "=== FP Runtime Directories ==="
        ls -la /app/data/fp_runtime/ 2>/dev/null || echo "FP runtime data directory not found"
        echo ""
        
        echo "=== FP Log Directories ==="
        ls -la /app/logs/fp/ 2>/dev/null || echo "FP log directory not found"
        echo ""
        
        echo "=== FP Directory Permissions ==="
        for dir in "/app/data/fp_runtime" "/app/logs/fp"; do
            if [ -d "$dir" ]; then
                echo "Directory: $dir"
                ls -ld "$dir"
                echo "Writable: $([ -w "$dir" ] && echo "Yes" || echo "No")"
                echo ""
            fi
        done
    ' 2>/dev/null || error "Failed to check FP directories"
}

collect_fp_logs() {
    header "Collecting FP Logs"
    
    local log_output="$DEBUG_OUTPUT_DIR/fp_logs_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$log_output"
    
    # Copy FP logs from container
    if docker exec "$CONTAINER_NAME" test -d /app/logs/fp 2>/dev/null; then
        docker cp "$CONTAINER_NAME:/app/logs/fp" "$log_output/" 2>/dev/null || warn "Failed to copy some FP logs"
        info "FP logs copied to: $log_output/fp/"
    else
        warn "No FP logs directory found in container"
    fi
    
    # Get recent FP-related container logs
    docker logs --tail=500 "$CONTAINER_NAME" 2>&1 | grep -i "fp\|effect\|interpreter\|scheduler" > "$log_output/container_fp_logs.txt" || true
    
    # Get FP runtime state
    docker exec "$CONTAINER_NAME" bash -c '
        if [ -d "/app/data/fp_runtime" ]; then
            find /app/data/fp_runtime -type f -name "*.json" -o -name "*.txt" -o -name "*.log" 2>/dev/null
        fi
    ' > "$log_output/fp_runtime_files.txt" 2>/dev/null || true
    
    info "Log collection completed: $log_output"
}

test_fp_effects() {
    header "Testing FP Effects"
    
    docker exec "$CONTAINER_NAME" python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/bot/fp')

try:
    from bot.fp.effects.io import IO, AsyncIO
    from bot.fp.runtime.interpreter import get_interpreter
    
    print('=== Testing Basic IO Effect ===')
    
    # Test simple IO effect
    def simple_effect():
        return 'Hello from FP effect!'
    
    io_effect = IO(simple_effect)
    interpreter = get_interpreter()
    
    result = interpreter.run_effect(io_effect)
    print(f'Simple effect result: {result}')
    
    print('')
    print('=== Testing Effect with Error Handling ===')
    
    def error_effect():
        raise ValueError('Test error for debugging')
    
    error_io = IO(error_effect)
    
    try:
        interpreter.run_effect(error_io)
    except ValueError as e:
        print(f'Error effect handled correctly: {e}')
    
    print('')
    print('=== Effect Interpreter Stats After Test ===')
    stats = interpreter.get_runtime_stats()
    print(f'Active effects: {stats.get(\"active_effects\", \"unknown\")}')
    
except Exception as e:
    print(f'Error testing FP effects: {e}')
    import traceback
    traceback.print_exc()
" 2>/dev/null || error "Failed to test FP effects"
}

monitor_fp_performance() {
    header "FP Performance Monitoring"
    
    local monitor_duration="${2:-30}"
    info "Monitoring FP performance for $monitor_duration seconds..."
    
    # Create monitoring script
    local monitor_script="$DEBUG_OUTPUT_DIR/fp_monitor_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "FP Performance Monitoring - $(date)"
        echo "Monitoring Duration: $monitor_duration seconds"
        echo "Container: $CONTAINER_NAME"
        echo "==============================================="
        echo ""
    } > "$monitor_script"
    
    # Monitor in background
    (
        for i in $(seq 1 "$monitor_duration"); do
            {
                echo "=== Sample $i ($(date)) ==="
                docker exec "$CONTAINER_NAME" python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/bot/fp')

try:
    from bot.fp.runtime.interpreter import get_interpreter
    interpreter = get_interpreter()
    stats = interpreter.get_runtime_stats()
    
    print(f'Active Effects: {stats.get(\"active_effects\", 0)}')
    
    # Get system resource usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    
    print(f'Memory Usage: {memory_mb:.1f} MB')
    print(f'CPU Usage: {cpu_percent:.1f}%')
    
except Exception as e:
    print(f'Monitoring error: {e}')
                " 2>/dev/null || echo "Failed to get stats for sample $i"
                echo ""
            } >> "$monitor_script"
            sleep 1
        done
    ) &
    
    local monitor_pid=$!
    
    # Wait for monitoring to complete
    wait $monitor_pid
    
    info "Performance monitoring completed: $monitor_script"
}

# Interactive debugging functions
interactive_fp_shell() {
    header "Interactive FP Debugging Shell"
    
    info "Starting interactive Python shell with FP runtime loaded..."
    info "Available objects: interpreter, scheduler, CompatibilityLayer"
    info "Type 'exit()' to quit"
    
    docker exec -it "$CONTAINER_NAME" python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/bot/fp')

print('Loading FP runtime...')

try:
    from bot.fp.runtime.interpreter import get_interpreter
    from bot.fp.runtime.scheduler import get_scheduler
    from bot.fp.adapters.compatibility_layer import CompatibilityLayer
    
    interpreter = get_interpreter()
    scheduler = get_scheduler()
    comp_layer = CompatibilityLayer()
    
    print('FP runtime loaded successfully!')
    print('Available objects:')
    print('  - interpreter: Effect interpreter')
    print('  - scheduler: Task scheduler')
    print('  - comp_layer: Compatibility layer')
    print('')
    
    # Start interactive session
    import code
    code.interact(local=locals())
    
except Exception as e:
    print(f'Error loading FP runtime: {e}')
    import traceback
    traceback.print_exc()
"
}

# Main functions
run_full_diagnosis() {
    header "Full FP Container Diagnosis"
    
    create_debug_dir
    
    check_fp_environment
    check_fp_directories
    check_fp_adapters
    check_fp_runtime_stats
    test_fp_effects
    collect_fp_logs
    
    info "Full diagnosis completed. Check $DEBUG_OUTPUT_DIR for detailed logs."
}

show_usage() {
    echo "FP Container Debug Tool"
    echo ""
    echo "Usage: $0 [container_name] [command]"
    echo ""
    echo "Commands:"
    echo "  full              - Run full FP diagnosis (default)"
    echo "  env               - Check FP environment variables"
    echo "  stats             - Show FP runtime statistics"
    echo "  adapters          - Check FP adapter status"
    echo "  dirs              - Check FP directory structure"
    echo "  logs              - Collect FP logs"
    echo "  test              - Test FP effects"
    echo "  monitor [seconds] - Monitor FP performance"
    echo "  shell             - Interactive FP debugging shell"
    echo "  help              - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 ai-trading-bot full"
    echo "  $0 ai-trading-bot-fp-dev stats"
    echo "  $0 ai-trading-bot monitor 60"
    echo "  $0 ai-trading-bot shell"
}

# Main script execution
main() {
    local command="${2:-full}"
    
    check_docker
    
    if [ "$command" = "help" ]; then
        show_usage
        exit 0
    fi
    
    check_container
    
    info "Running FP debug command: $command"
    info "Target container: $CONTAINER_NAME"
    
    case "$command" in
        "full")
            run_full_diagnosis
            ;;
        "env")
            check_fp_environment
            ;;
        "stats")
            check_fp_runtime_stats
            ;;
        "adapters")
            check_fp_adapters
            ;;
        "dirs")
            check_fp_directories
            ;;
        "logs")
            create_debug_dir
            collect_fp_logs
            ;;
        "test")
            test_fp_effects
            ;;
        "monitor")
            create_debug_dir
            monitor_fp_performance "$@"
            ;;
        "shell")
            interactive_fp_shell
            ;;
        *)
            error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"