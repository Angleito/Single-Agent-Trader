#!/bin/bash
#
# monitor-deployment.sh - Real-time deployment monitoring
#
# This script provides real-time monitoring during zero-downtime deployments,
# showing container status, resource usage, and log streams.
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
REFRESH_INTERVAL=2
LOG_LINES=5

# Clear screen and move cursor to top
clear_screen() {
    printf '\033[2J\033[H'
}

# Print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Get deployment colors
get_deployments() {
    local blue_count=$(docker ps --filter "label=deployment.color=blue" -q 2>/dev/null | wc -l)
    local green_count=$(docker ps --filter "label=deployment.color=green" -q 2>/dev/null | wc -l)

    local deployments=""
    if [ $blue_count -gt 0 ]; then
        deployments="${deployments}blue "
    fi
    if [ $green_count -gt 0 ]; then
        deployments="${deployments}green "
    fi

    echo "$deployments"
}

# Show container status
show_containers() {
    local deployment=$1

    print_color "$YELLOW" "Containers (${deployment}):"
    docker ps --filter "label=deployment.color=${deployment}" \
        --format "table {{.Names}}\t{{.Status}}\t{{.State}}" 2>/dev/null | sed 's/^/  /'
}

# Show resource usage
show_resources() {
    local deployment=$1

    print_color "$YELLOW" "Resource Usage (${deployment}):"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
        $(docker ps --filter "label=deployment.color=${deployment}" -q) 2>/dev/null | sed 's/^/  /'
}

# Show recent logs
show_logs() {
    local deployment=$1

    print_color "$YELLOW" "Recent Logs (${deployment}):"
    docker logs --tail ${LOG_LINES} --since "5s" \
        "ai-trading-bot-${deployment}" 2>&1 | sed 's/^/  /' || echo "  No recent logs"
}

# Show health status
show_health() {
    local deployment=$1

    print_color "$YELLOW" "Health Status (${deployment}):"

    local services=(
        "ai-trading-bot-${deployment}"
        "bluefin-service-${deployment}"
        "mcp-memory-${deployment}"
        "dashboard-backend-${deployment}"
    )

    for service in "${services[@]}"; do
        local health=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "unknown")
        local running=$(docker inspect --format='{{.State.Running}}' "$service" 2>/dev/null || echo "false")

        if [ "$health" = "healthy" ]; then
            echo -e "  ${service}: ${GREEN}✓ healthy${NC}"
        elif [ "$running" = "true" ]; then
            echo -e "  ${service}: ${YELLOW}⚡ running${NC}"
        else
            echo -e "  ${service}: ${RED}✗ down${NC}"
        fi
    done
}

# Main monitoring loop
main() {
    print_color "$BLUE" "=== DEPLOYMENT MONITOR ==="
    print_color "$CYAN" "Press Ctrl+C to exit"
    echo

    while true; do
        clear_screen

        # Header
        print_color "$BLUE" "=== DEPLOYMENT MONITOR ==="
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo

        # Get active deployments
        local deployments=$(get_deployments)

        if [ -z "$deployments" ]; then
            print_color "$RED" "No active deployments found!"
        else
            # Show info for each deployment
            for deployment in $deployments; do
                print_color "$CYAN" "━━━ ${deployment^^} DEPLOYMENT ━━━"
                echo

                show_health "$deployment"
                echo

                show_containers "$deployment"
                echo

                show_resources "$deployment"
                echo

                show_logs "$deployment"
                echo
            done
        fi

        # Show transition status if both deployments are active
        if [[ "$deployments" == *"blue"* ]] && [[ "$deployments" == *"green"* ]]; then
            print_color "$YELLOW" "⚠️  TRANSITION IN PROGRESS - Both deployments active"
        fi

        sleep $REFRESH_INTERVAL
    done
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${GREEN}Monitoring stopped${NC}"; exit 0' INT

# Run monitor
main
