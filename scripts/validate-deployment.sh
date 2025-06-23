#!/bin/bash
#
# validate-deployment.sh - Deployment validation for zero-downtime deployments
#
# This script validates that a deployment is ready and healthy before
# marking it as successful. Used as part of the zero-downtime deployment process.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Validation checks
CHECKS_PASSED=0
CHECKS_FAILED=0

# Print colored output
print_status() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Check function
run_check() {
    local check_name=$1
    local check_command=$2

    echo -n "Checking ${check_name}... "

    if eval "$check_command" &>/dev/null; then
        print_status "$GREEN" "PASS"
        ((CHECKS_PASSED++))
        return 0
    else
        print_status "$RED" "FAIL"
        ((CHECKS_FAILED++))
        return 1
    fi
}

# Get deployment color
get_deployment_color() {
    local color="${1:-}"
    if [ -z "$color" ]; then
        # Try to detect from running containers
        if docker ps --filter "label=deployment.color=blue" -q | grep -q .; then
            echo "blue"
        elif docker ps --filter "label=deployment.color=green" -q | grep -q .; then
            echo "green"
        else
            echo "unknown"
        fi
    else
        echo "$color"
    fi
}

# Main validation
main() {
    local deployment_color=$(get_deployment_color "${1:-}")

    if [ "$deployment_color" = "unknown" ]; then
        print_status "$RED" "No deployment found to validate"
        exit 1
    fi

    print_status "$BLUE" "=== VALIDATING ${deployment_color} DEPLOYMENT ==="
    echo

    # Container checks
    print_status "$YELLOW" "Container Status Checks:"
    run_check "Trading Bot Container" "docker ps --filter name=ai-trading-bot-${deployment_color} --filter status=running -q | grep -q ."
    run_check "Bluefin Service" "docker ps --filter name=bluefin-service-${deployment_color} --filter status=running -q | grep -q ."
    run_check "MCP Memory" "docker ps --filter name=mcp-memory-${deployment_color} --filter status=running -q | grep -q ."
    run_check "MCP OmniSearch" "docker ps --filter name=mcp-omnisearch-${deployment_color} --filter status=running -q | grep -q ."
    run_check "Dashboard Backend" "docker ps --filter name=dashboard-backend-${deployment_color} --filter status=running -q | grep -q ."
    run_check "Dashboard Frontend" "docker ps --filter name=dashboard-frontend-${deployment_color} --filter status=running -q | grep -q ."
    echo

    # Health checks
    print_status "$YELLOW" "Health Checks:"
    run_check "Trading Bot Health" "docker exec ai-trading-bot-${deployment_color} python -c 'import sys; sys.exit(0)' 2>/dev/null"
    run_check "Bluefin API Health" "curl -sf http://localhost:8081/health"
    run_check "MCP Memory Health" "curl -sf http://localhost:8765/health"
    run_check "Dashboard API Health" "curl -sf http://localhost:8000/health"
    echo

    # Resource checks
    print_status "$YELLOW" "Resource Checks:"
    run_check "Memory Usage" "[ \$(docker stats --no-stream --format '{{.MemPerc}}' ai-trading-bot-${deployment_color} | cut -d'%' -f1 | cut -d'.' -f1) -lt 80 ]"
    run_check "CPU Usage" "[ \$(docker stats --no-stream --format '{{.CPUPerc}}' ai-trading-bot-${deployment_color} | cut -d'%' -f1 | cut -d'.' -f1) -lt 80 ]"
    echo

    # Log checks
    print_status "$YELLOW" "Log Analysis:"
    run_check "No Critical Errors" "! docker logs ai-trading-bot-${deployment_color} 2>&1 | tail -100 | grep -i 'critical'"
    run_check "No Exceptions" "! docker logs ai-trading-bot-${deployment_color} 2>&1 | tail -100 | grep -i 'exception'"
    run_check "Recent Activity" "docker logs ai-trading-bot-${deployment_color} 2>&1 | tail -10 | grep -q ."
    echo

    # Network checks
    print_status "$YELLOW" "Network Connectivity:"
    run_check "Internal Network" "docker exec ai-trading-bot-${deployment_color} ping -c 1 mcp-memory-${deployment_color}"
    run_check "External Network" "docker exec ai-trading-bot-${deployment_color} ping -c 1 8.8.8.8"
    echo

    # Configuration checks
    print_status "$YELLOW" "Configuration Validation:"
    run_check "Environment Variables" "docker exec ai-trading-bot-${deployment_color} env | grep -q 'DEPLOYMENT_COLOR=${deployment_color}'"
    run_check "Config Files" "docker exec ai-trading-bot-${deployment_color} test -f /app/config/development.json"
    echo

    # Summary
    print_status "$BLUE" "=== VALIDATION SUMMARY ==="
    print_status "$GREEN" "Passed: ${CHECKS_PASSED}"
    print_status "$RED" "Failed: ${CHECKS_FAILED}"
    echo

    if [ $CHECKS_FAILED -eq 0 ]; then
        print_status "$GREEN" "✓ Deployment validation PASSED"
        exit 0
    else
        print_status "$RED" "✗ Deployment validation FAILED"
        exit 1
    fi
}

# Run validation
main "$@"
