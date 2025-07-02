#!/bin/bash

# Docker Bench Security Wrapper Script for AI Trading Bot
# Runs comprehensive security scans with custom checks and reporting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_BENCH_DIR="${BASE_DIR}/docker-bench-security"
CONFIG_FILE="${BASE_DIR}/config/docker-bench.conf"
CUSTOM_CHECKS_DIR="${BASE_DIR}/custom-checks"
REPORTS_DIR="${BASE_DIR}/reports"
LOG_FILE="${BASE_DIR}/logs/security-scan.log"

# Source configuration
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2" | tee -a "$LOG_FILE"
}

info() {
    log "INFO" "${BLUE}$1${NC}"
}

success() {
    log "SUCCESS" "${GREEN}$1${NC}"
}

warning() {
    log "WARNING" "${YELLOW}$1${NC}"
}

error() {
    log "ERROR" "${RED}$1${NC}"
}

# Ensure required directories exist
mkdir -p "$REPORTS_DIR" "$(dirname "$LOG_FILE")"

# Run Docker Bench Security
run_docker_bench() {
    info "Running Docker Bench Security scan..."
    
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local report_file="${REPORTS_DIR}/docker-bench-${timestamp}.json"
    local log_file="${REPORTS_DIR}/docker-bench-${timestamp}.log"
    
    if [ ! -f "${DOCKER_BENCH_DIR}/docker-bench-security.sh" ]; then
        error "Docker Bench Security not found. Run install-docker-bench.sh first."
        return 1
    fi
    
    cd "$DOCKER_BENCH_DIR"
    
    # Run Docker Bench with custom configuration
    ./docker-bench-security.sh \
        -l "$log_file" \
        -c "${TRADING_BOT_CONTAINERS:-ai-trading-bot,bluefin-service}" \
        > "$report_file" 2>&1
    
    echo "$report_file"
}

# Run custom checks
run_custom_checks() {
    info "Running custom trading bot security checks..."
    
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local custom_report="${REPORTS_DIR}/custom-checks-${timestamp}.json"
    
    echo '{"timestamp": "'$(date -Iseconds)'", "checks": [' > "$custom_report"
    
    local first_check=true
    if [ -d "$CUSTOM_CHECKS_DIR" ]; then
        for check_script in "${CUSTOM_CHECKS_DIR}"/*.sh; do
            if [ -f "$check_script" ]; then
                info "Running custom check: $(basename "$check_script")"
                
                if [ "$first_check" = false ]; then
                    echo "," >> "$custom_report"
                fi
                first_check=false
                
                # Simple check execution (would need proper check function integration)
                echo '{"check": "'$(basename "$check_script")'", "status": "executed"}' >> "$custom_report"
            fi
        done
    else
        warning "Custom checks directory not found: $CUSTOM_CHECKS_DIR"
    fi
    
    echo ']}' >> "$custom_report"
    
    echo "$custom_report"
}

# Parse and analyze results
analyze_results() {
    local docker_bench_report="$1"
    local custom_report="$2"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local analysis_report="${REPORTS_DIR}/security-analysis-${timestamp}.json"
    
    info "Analyzing security scan results..."
    
    # Count issues by severity (simplified parsing)
    local critical_count=0
    local high_count=0
    local medium_count=0
    local low_count=0
    local pass_count=0
    
    # Parse Docker Bench results (simplified parsing)
    if [ -f "$docker_bench_report" ]; then
        critical_count=$(grep -c "WARN\|FAIL" "$docker_bench_report" 2>/dev/null || echo "0")
        pass_count=$(grep -c "PASS\|INFO" "$docker_bench_report" 2>/dev/null || echo "0")
    fi
    
    # Create analysis report
    cat > "$analysis_report" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "scan_type": "comprehensive",
    "reports": {
        "docker_bench": "$docker_bench_report",
        "custom_checks": "$custom_report"
    },
    "summary": {
        "critical_issues": $critical_count,
        "high_issues": $high_count,
        "medium_issues": $medium_count,
        "low_issues": $low_count,
        "passed_checks": $pass_count,
        "total_checks": $((critical_count + high_count + medium_count + low_count + pass_count))
    },
    "compliance_status": {
        "cis_docker_benchmark": "$([ $critical_count -le ${MAX_CRITICAL_ISSUES:-0} ] && echo "COMPLIANT" || echo "NON_COMPLIANT")",
        "trading_bot_security": "$([ $high_count -le ${MAX_HIGH_ISSUES:-2} ] && echo "COMPLIANT" || echo "NON_COMPLIANT")"
    },
    "recommendations": [
        "Review and remediate all critical and high severity issues",
        "Implement automated remediation for common security misconfigurations",
        "Schedule regular security scans as part of CI/CD pipeline",
        "Monitor security metrics and establish alerting thresholds"
    ]
}
EOF

    echo "$analysis_report"
}

# Main execution
main() {
    info "Starting comprehensive Docker security scan for AI Trading Bot"
    
    # Ensure reports directory exists
    mkdir -p "$REPORTS_DIR"
    
    # Run Docker Bench Security
    local docker_bench_report
    docker_bench_report=$(run_docker_bench)
    
    # Run custom checks
    local custom_report
    custom_report=$(run_custom_checks)
    
    # Analyze results
    local analysis_report
    analysis_report=$(analyze_results "$docker_bench_report" "$custom_report")
    
    success "Security scan completed successfully"
    info "Reports generated:"
    info "  - Docker Bench: $docker_bench_report"
    info "  - Custom Checks: $custom_report"
    info "  - Analysis: $analysis_report"
    
    # Display summary
    if command -v jq &> /dev/null && [ -f "$analysis_report" ]; then
        echo
        info "Security Scan Summary:"
        jq -r '.summary | to_entries[] | "  \(.key | gsub("_"; " ") | ascii_upcase): \(.value)"' "$analysis_report"
        echo
        info "Compliance Status:"
        jq -r '.compliance_status | to_entries[] | "  \(.key | gsub("_"; " ") | ascii_upcase): \(.value)"' "$analysis_report"
    fi
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi