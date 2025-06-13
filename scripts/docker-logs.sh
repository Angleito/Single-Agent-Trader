#!/bin/bash

# AI Trading Bot - Docker Logs Script
# Monitors and manages container logs for validation and debugging

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_CONTAINER="ai-trading-bot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
AI Trading Bot - Docker Logs Script

Usage: $0 [OPTIONS] [CONTAINER]

OPTIONS:
    -f, --follow            Follow log output (tail -f behavior)
    -t, --tail LINES        Show last N lines (default: 100)
    -s, --since TIME        Show logs since timestamp (e.g., 2h, 30m, 2023-01-01T10:00:00)
    -u, --until TIME        Show logs until timestamp
    --timestamps            Show timestamps
    --details               Show extra details (labels, env vars)
    --filter FILTER         Filter logs (e.g., level=error)
    --grep PATTERN          Grep for specific pattern in logs
    --errors                Show only error logs
    --warnings              Show only warning logs
    --health               Show health check logs
    --export FILE           Export logs to file
    --clean                 Clean old log files
    --stats                 Show log statistics
    -h, --help              Show this help message

CONTAINER:
    Container name (default: ai-trading-bot)
    Use 'all' to show logs from all trading bot containers

EXAMPLES:
    $0                              # Show last 100 lines
    $0 -f                          # Follow logs in real-time
    $0 -t 50                       # Show last 50 lines
    $0 --since 1h                  # Show logs from last hour
    $0 --errors                    # Show only error logs
    $0 --grep "trade"              # Filter logs containing "trade"
    $0 --export trading.log        # Export logs to file
    $0 ai-trading-bot-dev          # Show logs from dev container
    $0 all                         # Show logs from all containers

FILTERING:
    --filter level=error           # Docker log level filtering
    --filter label=com.example.app # Filter by container labels
    --grep "ERROR\\|WARN"           # Grep for multiple patterns
    --since "2023-01-01T10:00:00"  # ISO 8601 timestamp

MONITORING:
    Use -f for real-time monitoring
    Use --errors for quick error checking
    Use --stats to analyze log patterns

EOF
}

# Default values
CONTAINER="$DEFAULT_CONTAINER"
FOLLOW=false
TAIL_LINES=100
SINCE=""
UNTIL=""
TIMESTAMPS=false
DETAILS=false
FILTER=""
GREP_PATTERN=""
EXPORT_FILE=""
CLEAN=false
STATS=false
ERRORS_ONLY=false
WARNINGS_ONLY=false
HEALTH_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -t|--tail)
            TAIL_LINES="$2"
            shift 2
            ;;
        -s|--since)
            SINCE="$2"
            shift 2
            ;;
        -u|--until)
            UNTIL="$2"
            shift 2
            ;;
        --timestamps)
            TIMESTAMPS=true
            shift
            ;;
        --details)
            DETAILS=true
            shift
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --grep)
            GREP_PATTERN="$2"
            shift 2
            ;;
        --errors)
            ERRORS_ONLY=true
            shift
            ;;
        --warnings)
            WARNINGS_ONLY=true
            shift
            ;;
        --health)
            HEALTH_ONLY=true
            shift
            ;;
        --export)
            EXPORT_FILE="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --stats)
            STATS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ ! "$1" =~ ^- ]]; then
                CONTAINER="$1"
                shift
            else
                log_error "Unknown option: $1"
                show_help
                exit 1
            fi
            ;;
    esac
done

# Validate Docker installation
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running"
    exit 1
fi

# Clean old log files
clean_logs() {
    log_info "Cleaning old log files..."
    cd "$PROJECT_DIR"
    
    # Clean logs directory
    if [[ -d "logs" ]]; then
        find logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
        find logs -name "*.log.*" -mtime +3 -delete 2>/dev/null || true
        log_success "Cleaned old log files"
    fi
    
    # Clean Docker logs (requires root or docker group)
    log_info "Note: Docker container logs are managed by Docker daemon"
    log_info "Use 'docker system prune' to clean Docker system logs"
}

# Show log statistics
show_stats() {
    log_info "Generating log statistics for $CONTAINER..."
    
    if ! docker container inspect "$CONTAINER" &> /dev/null; then
        log_error "Container $CONTAINER not found"
        return 1
    fi
    
    # Get container info
    echo -e "\n${CYAN}Container Information:${NC}"
    docker inspect "$CONTAINER" --format "table {{.Name}}\t{{.State.Status}}\t{{.State.StartedAt}}\t{{.RestartCount}}"
    
    # Get log statistics
    echo -e "\n${CYAN}Log Statistics:${NC}"
    LOGS=$(docker logs "$CONTAINER" 2>&1)
    
    if [[ -n "$LOGS" ]]; then
        TOTAL_LINES=$(echo "$LOGS" | wc -l)
        ERROR_LINES=$(echo "$LOGS" | grep -i "error\|exception\|fail" | wc -l)
        WARNING_LINES=$(echo "$LOGS" | grep -i "warn\|warning" | wc -l)
        INFO_LINES=$(echo "$LOGS" | grep -i "info" | wc -l)
        
        echo "Total log lines: $TOTAL_LINES"
        echo "Error lines: $ERROR_LINES"
        echo "Warning lines: $WARNING_LINES"
        echo "Info lines: $INFO_LINES"
        
        # Show recent activity
        echo -e "\n${CYAN}Recent Activity (last 10 lines):${NC}"
        echo "$LOGS" | tail -10
    else
        echo "No logs available"
    fi
    
    # Container resource usage
    echo -e "\n${CYAN}Resource Usage:${NC}"
    docker stats "$CONTAINER" --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Get container list for 'all' option
get_containers() {
    if [[ "$CONTAINER" == "all" ]]; then
        docker ps -a --filter "name=ai-trading-bot" --format "{{.Names}}" | grep "ai-trading-bot"
    else
        echo "$CONTAINER"
    fi
}

# Main function
main() {
    # Handle special operations
    if [[ "$CLEAN" == true ]]; then
        clean_logs
        exit 0
    fi
    
    if [[ "$STATS" == true ]]; then
        if [[ "$CONTAINER" == "all" ]]; then
            for container in $(get_containers); do
                show_stats "$container"
                echo
            done
        else
            show_stats
        fi
        exit 0
    fi
    
    # Check if container exists
    CONTAINERS=$(get_containers)
    if [[ -z "$CONTAINERS" ]]; then
        log_error "No AI Trading Bot containers found"
        log_info "Available containers:"
        docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
        exit 1
    fi
    
    # Build Docker logs arguments
    LOGS_ARGS=()
    
    if [[ "$FOLLOW" == true ]]; then
        LOGS_ARGS+=(--follow)
    fi
    
    if [[ -n "$TAIL_LINES" ]]; then
        LOGS_ARGS+=(--tail "$TAIL_LINES")
    fi
    
    if [[ -n "$SINCE" ]]; then
        LOGS_ARGS+=(--since "$SINCE")
    fi
    
    if [[ -n "$UNTIL" ]]; then
        LOGS_ARGS+=(--until "$UNTIL")
    fi
    
    if [[ "$TIMESTAMPS" == true ]]; then
        LOGS_ARGS+=(--timestamps)
    fi
    
    if [[ "$DETAILS" == true ]]; then
        LOGS_ARGS+=(--details)
    fi
    
    # Process each container
    for container in $CONTAINERS; do
        if [[ ! $(echo "$CONTAINERS" | wc -w) -eq 1 ]]; then
            echo -e "\n${CYAN}=== Logs for $container ===${NC}"
        fi
        
        # Check if container exists
        if ! docker container inspect "$container" &> /dev/null; then
            log_error "Container $container not found"
            continue
        fi
        
        # Get logs
        if [[ -n "$EXPORT_FILE" ]]; then
            log_info "Exporting logs to $EXPORT_FILE..."
            docker logs "${LOGS_ARGS[@]}" "$container" > "$EXPORT_FILE" 2>&1
            log_success "Logs exported to $EXPORT_FILE"
        else
            # Apply filtering and display
            if [[ "$ERRORS_ONLY" == true ]] || [[ "$WARNINGS_ONLY" == true ]] || [[ "$HEALTH_ONLY" == true ]] || [[ -n "$GREP_PATTERN" ]]; then
                # Use grep filtering
                GREP_PATTERNS=()
                
                if [[ "$ERRORS_ONLY" == true ]]; then
                    GREP_PATTERNS+=("ERROR\\|Exception\\|CRITICAL\\|FATAL")
                fi
                
                if [[ "$WARNINGS_ONLY" == true ]]; then
                    GREP_PATTERNS+=("WARN\\|WARNING")
                fi
                
                if [[ "$HEALTH_ONLY" == true ]]; then
                    GREP_PATTERNS+=("health\\|Health\\|HEALTH")
                fi
                
                if [[ -n "$GREP_PATTERN" ]]; then
                    GREP_PATTERNS+=("$GREP_PATTERN")
                fi
                
                # Combine patterns
                COMBINED_PATTERN=$(IFS='\\|'; echo "${GREP_PATTERNS[*]}")
                
                log_info "Filtering logs with pattern: $COMBINED_PATTERN"
                docker logs "${LOGS_ARGS[@]}" "$container" 2>&1 | grep -E "$COMBINED_PATTERN" --color=always || {
                    log_warning "No logs matching the filter pattern"
                }
            else
                # Show all logs
                docker logs "${LOGS_ARGS[@]}" "$container"
            fi
        fi
    done
}

# Handle Ctrl+C gracefully
trap 'log_info "\nLog monitoring stopped by user"; exit 0' INT

# Show header
log_info "AI Trading Bot - Log Monitor"

if [[ "$FOLLOW" == true ]]; then
    log_info "Following logs for $CONTAINER (Press Ctrl+C to stop)..."
else
    log_info "Showing logs for $CONTAINER..."
fi

# Run main function
main

exit 0