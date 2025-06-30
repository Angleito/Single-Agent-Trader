#!/bin/bash
"""
Log Validation Suite Startup Script
Starts all log validation and monitoring components in the correct order.
"""

set -e

# Configuration
LOGS_DIR="${LOGS_DIR:-./logs}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
ALERT_CONFIG="${ALERT_CONFIG:-./scripts/alerts_config.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE} Log Validation and Monitoring Suite${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running or not accessible"
        exit 1
    fi
    
    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if required Python packages are available
    for package in docker flask requests pyyaml psutil; do
        if ! python3 -c "import $package" >/dev/null 2>&1; then
            print_warning "Python package '$package' not found. Install with: pip install $package"
        fi
    done
    
    # Create logs directory if it doesn't exist
    mkdir -p "$LOGS_DIR"
    
    print_status "Prerequisites check completed"
}

# Function to create alert configuration if it doesn't exist
setup_alert_config() {
    if [ ! -f "$ALERT_CONFIG" ]; then
        print_status "Creating default alert configuration..."
        python3 ./scripts/alert_manager.py --create-config "$ALERT_CONFIG"
        print_status "Alert configuration created at: $ALERT_CONFIG"
    else
        print_status "Using existing alert configuration: $ALERT_CONFIG"
    fi
}

# Function to validate current logs
validate_existing_logs() {
    print_status "Validating existing logs..."
    
    if [ -d "$LOGS_DIR" ] && [ "$(ls -A $LOGS_DIR 2>/dev/null)" ]; then
        print_status "Found existing logs, running validation..."
        python3 ./scripts/log_analysis_validator.py \
            --logs-dir "$LOGS_DIR" \
            --format text \
            --output "$LOGS_DIR/initial_validation_report.md"
        
        if [ $? -eq 0 ]; then
            print_status "Initial log validation completed. Report saved to: $LOGS_DIR/initial_validation_report.md"
        else
            print_warning "Log validation completed with warnings"
        fi
    else
        print_status "No existing logs found. Starting fresh monitoring..."
    fi
}

# Function to start the log monitor
start_log_monitor() {
    print_status "Starting Docker log monitor..."
    
    # Start log monitor in background
    python3 ./scripts/docker_log_monitor.py \
        --logs-dir "$LOGS_DIR" \
        --verbose \
        > "$LOGS_DIR/log_monitor.log" 2>&1 &
    
    LOG_MONITOR_PID=$!
    echo $LOG_MONITOR_PID > "$LOGS_DIR/log_monitor.pid"
    
    print_status "Log monitor started (PID: $LOG_MONITOR_PID)"
    
    # Wait a moment for monitor to initialize
    sleep 3
    
    # Check if monitor is still running
    if ! kill -0 $LOG_MONITOR_PID 2>/dev/null; then
        print_error "Log monitor failed to start. Check $LOGS_DIR/log_monitor.log for details"
        return 1
    fi
}

# Function to start the alert manager
start_alert_manager() {
    print_status "Starting alert manager..."
    
    # Start alert manager in background
    python3 ./scripts/alert_manager.py \
        --config "$ALERT_CONFIG" \
        --logs-dir "$LOGS_DIR" \
        --verbose \
        > "$LOGS_DIR/alert_manager.log" 2>&1 &
    
    ALERT_MANAGER_PID=$!
    echo $ALERT_MANAGER_PID > "$LOGS_DIR/alert_manager.pid"
    
    print_status "Alert manager started (PID: $ALERT_MANAGER_PID)"
    
    # Wait a moment for alert manager to initialize
    sleep 2
    
    # Check if alert manager is still running
    if ! kill -0 $ALERT_MANAGER_PID 2>/dev/null; then
        print_error "Alert manager failed to start. Check $LOGS_DIR/alert_manager.log for details"
        return 1
    fi
}

# Function to start the dashboard
start_dashboard() {
    print_status "Starting test result dashboard..."
    
    # Start dashboard in background
    python3 ./scripts/test_result_dashboard.py \
        --port "$DASHBOARD_PORT" \
        --host "0.0.0.0" \
        --logs-dir "$LOGS_DIR" \
        --load-data \
        > "$LOGS_DIR/dashboard.log" 2>&1 &
    
    DASHBOARD_PID=$!
    echo $DASHBOARD_PID > "$LOGS_DIR/dashboard.pid"
    
    print_status "Dashboard started (PID: $DASHBOARD_PID)"
    
    # Wait for dashboard to start
    sleep 5
    
    # Check if dashboard is accessible
    if curl -f "http://localhost:$DASHBOARD_PORT" >/dev/null 2>&1; then
        print_status "Dashboard is accessible at: http://localhost:$DASHBOARD_PORT"
    else
        print_warning "Dashboard may not be fully initialized yet. Check $LOGS_DIR/dashboard.log for details"
    fi
}

# Function to display status
show_status() {
    print_status "Log Validation Suite Status:"
    echo ""
    
    # Check log monitor
    if [ -f "$LOGS_DIR/log_monitor.pid" ]; then
        LOG_MONITOR_PID=$(cat "$LOGS_DIR/log_monitor.pid")
        if kill -0 $LOG_MONITOR_PID 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Log Monitor (PID: $LOG_MONITOR_PID)"
        else
            echo -e "  ${RED}✗${NC} Log Monitor (stopped)"
        fi
    else
        echo -e "  ${RED}✗${NC} Log Monitor (not started)"
    fi
    
    # Check alert manager
    if [ -f "$LOGS_DIR/alert_manager.pid" ]; then
        ALERT_MANAGER_PID=$(cat "$LOGS_DIR/alert_manager.pid")
        if kill -0 $ALERT_MANAGER_PID 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Alert Manager (PID: $ALERT_MANAGER_PID)"
        else
            echo -e "  ${RED}✗${NC} Alert Manager (stopped)"
        fi
    else
        echo -e "  ${RED}✗${NC} Alert Manager (not started)"
    fi
    
    # Check dashboard
    if [ -f "$LOGS_DIR/dashboard.pid" ]; then
        DASHBOARD_PID=$(cat "$LOGS_DIR/dashboard.pid")
        if kill -0 $DASHBOARD_PID 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Dashboard (PID: $DASHBOARD_PID) - http://localhost:$DASHBOARD_PORT"
        else
            echo -e "  ${RED}✗${NC} Dashboard (stopped)"
        fi
    else
        echo -e "  ${RED}✗${NC} Dashboard (not started)"
    fi
    
    echo ""
    
    # Show recent container status
    print_status "Docker Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || print_warning "Could not retrieve container status"
    
    echo ""
    
    # Show log file status
    if [ -d "$LOGS_DIR" ]; then
        LOG_COUNT=$(find "$LOGS_DIR" -name "*.log" -o -name "*.jsonl" | wc -l)
        print_status "Log Files: $LOG_COUNT files in $LOGS_DIR"
        
        if [ $LOG_COUNT -gt 0 ]; then
            echo "Recent log files:"
            find "$LOGS_DIR" -name "*.log" -o -name "*.jsonl" | head -5 | while read file; do
                echo "  - $(basename "$file") ($(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown") bytes)"
            done
        fi
    fi
}

# Function to stop all services
stop_services() {
    print_status "Stopping log validation suite..."
    
    # Stop dashboard
    if [ -f "$LOGS_DIR/dashboard.pid" ]; then
        DASHBOARD_PID=$(cat "$LOGS_DIR/dashboard.pid")
        if kill -0 $DASHBOARD_PID 2>/dev/null; then
            kill $DASHBOARD_PID
            print_status "Dashboard stopped"
        fi
        rm -f "$LOGS_DIR/dashboard.pid"
    fi
    
    # Stop alert manager
    if [ -f "$LOGS_DIR/alert_manager.pid" ]; then
        ALERT_MANAGER_PID=$(cat "$LOGS_DIR/alert_manager.pid")
        if kill -0 $ALERT_MANAGER_PID 2>/dev/null; then
            kill $ALERT_MANAGER_PID
            print_status "Alert manager stopped"
        fi
        rm -f "$LOGS_DIR/alert_manager.pid"
    fi
    
    # Stop log monitor
    if [ -f "$LOGS_DIR/log_monitor.pid" ]; then
        LOG_MONITOR_PID=$(cat "$LOGS_DIR/log_monitor.pid")
        if kill -0 $LOG_MONITOR_PID 2>/dev/null; then
            kill $LOG_MONITOR_PID
            print_status "Log monitor stopped"
        fi
        rm -f "$LOGS_DIR/log_monitor.pid"
    fi
    
    print_status "All services stopped"
}

# Function to run a quick log analysis
run_analysis() {
    print_status "Running comprehensive log analysis..."
    
    # Run analysis on current logs
    python3 ./scripts/log_analysis_validator.py \
        --logs-dir "$LOGS_DIR" \
        --format text \
        --output "$LOGS_DIR/analysis_report_$(date +%Y%m%d_%H%M%S).md"
    
    if [ $? -eq 0 ]; then
        print_status "Analysis completed. Report saved to logs directory."
    else
        print_warning "Analysis completed with warnings"
    fi
    
    # Show alert status
    python3 ./scripts/alert_manager.py --config "$ALERT_CONFIG" --status
}

# Function to display help
show_help() {
    echo "Log Validation Suite Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start all log validation services"
    echo "  stop      Stop all running services"
    echo "  restart   Restart all services"
    echo "  status    Show status of all services"
    echo "  analyze   Run log analysis on current data"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  LOGS_DIR         Directory for log storage (default: ./logs)"
    echo "  DASHBOARD_PORT   Port for web dashboard (default: 8080)"
    echo "  ALERT_CONFIG     Path to alert configuration (default: ./scripts/alerts_config.yaml)"
    echo "  LOG_LEVEL        Logging level (default: INFO)"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start all services with defaults"
    echo "  DASHBOARD_PORT=3000 $0 start  # Start with custom dashboard port"
    echo "  $0 status                   # Check service status"
    echo "  $0 analyze                  # Run log analysis"
}

# Function to handle cleanup on exit
cleanup() {
    if [ "$1" != "stop" ]; then
        print_warning "Received interrupt signal. Stopping services..."
        stop_services
    fi
}

# Set up signal handlers
trap 'cleanup' INT TERM

# Main execution
main() {
    local command="${1:-start}"
    
    case "$command" in
        "start")
            print_header
            check_prerequisites
            setup_alert_config
            validate_existing_logs
            start_log_monitor
            start_alert_manager
            start_dashboard
            echo ""
            show_status
            echo ""
            print_status "Log validation suite started successfully!"
            print_status "Dashboard: http://localhost:$DASHBOARD_PORT"
            print_status "Logs directory: $LOGS_DIR"
            print_status ""
            print_status "To stop all services, run: $0 stop"
            ;;
        "stop")
            stop_services
            cleanup "stop"
            ;;
        "restart")
            print_status "Restarting log validation suite..."
            stop_services
            sleep 2
            main "start"
            ;;
        "status")
            show_status
            ;;
        "analyze")
            run_analysis
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"