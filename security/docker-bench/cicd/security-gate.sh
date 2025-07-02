#!/bin/bash

# CI/CD Security Gate for AI Trading Bot
# Integrated security scanning and policy enforcement for deployment pipelines

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$BASE_DIR")")")"
SECURITY_CONFIG="${BASE_DIR}/config/security-gate.conf"
LOGS_DIR="${BASE_DIR}/logs"

# Load configuration
if [ -f "$SECURITY_CONFIG" ]; then
    source "$SECURITY_CONFIG"
fi

# Default security gate configuration
GATE_MODE=${GATE_MODE:-"enforcing"}  # enforcing, permissive, disabled
FAIL_ON_CRITICAL=${FAIL_ON_CRITICAL:-true}
FAIL_ON_HIGH=${FAIL_ON_HIGH:-false}
MAX_CRITICAL_ISSUES=${MAX_CRITICAL_ISSUES:-0}
MAX_HIGH_ISSUES=${MAX_HIGH_ISSUES:-2}
MAX_MEDIUM_ISSUES=${MAX_MEDIUM_ISSUES:-5}
ENABLE_AUTO_REMEDIATION=${ENABLE_AUTO_REMEDIATION:-true}
REMEDIATION_TIMEOUT=${REMEDIATION_TIMEOUT:-300}
SECURITY_SCAN_TIMEOUT=${SECURITY_SCAN_TIMEOUT:-600}

# CI/CD Integration settings
CI_ENVIRONMENT=${CI_ENVIRONMENT:-"unknown"}
BUILD_ID=${BUILD_ID:-"local-$(date +%s)"}
COMMIT_SHA=${COMMIT_SHA:-"$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"}
BRANCH_NAME=${BRANCH_NAME:-"$(git branch --show-current 2>/dev/null || echo 'unknown')"}
PIPELINE_URL=${PIPELINE_URL:-""}

# Notification settings
SECURITY_WEBHOOK_URL=${SECURITY_WEBHOOK_URL:-""}
SLACK_SECURITY_CHANNEL=${SLACK_SECURITY_CHANNEL:-""}
TEAMS_WEBHOOK_URL=${TEAMS_WEBHOOK_URL:-""}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOGS_DIR}/security-gate.log"
}

info() { log "INFO" "${BLUE}$1${NC}"; }
success() { log "SUCCESS" "${GREEN}$1${NC}"; }
warning() { log "WARNING" "${YELLOW}$1${NC}"; }
error() { log "ERROR" "${RED}$1${NC}"; }
debug() { log "DEBUG" "${PURPLE}$1${NC}"; }

# Initialize logging
mkdir -p "$LOGS_DIR"

# Security gate status tracking
GATE_STATUS="UNKNOWN"
GATE_DETAILS=""
SECURITY_ISSUES_FOUND=0
CRITICAL_ISSUES=0
HIGH_ISSUES=0
MEDIUM_ISSUES=0
LOW_ISSUES=0

# Create security gate report
create_gate_report() {
    local gate_result="$1"
    local scan_report="$2"
    local remediation_report="$3"
    
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local gate_report="${LOGS_DIR}/security-gate-${BUILD_ID}-${timestamp}.json"
    
    cat > "$gate_report" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "gate_result": "$gate_result",
    "build_info": {
        "build_id": "$BUILD_ID",
        "commit_sha": "$COMMIT_SHA",
        "branch": "$BRANCH_NAME",
        "ci_environment": "$CI_ENVIRONMENT",
        "pipeline_url": "$PIPELINE_URL"
    },
    "security_scan": {
        "report_file": "$scan_report",
        "issues_found": {
            "critical": $CRITICAL_ISSUES,
            "high": $HIGH_ISSUES,
            "medium": $MEDIUM_ISSUES,
            "low": $LOW_ISSUES,
            "total": $SECURITY_ISSUES_FOUND
        }
    },
    "gate_configuration": {
        "mode": "$GATE_MODE",
        "fail_on_critical": $FAIL_ON_CRITICAL,
        "fail_on_high": $FAIL_ON_HIGH,
        "max_critical_issues": $MAX_CRITICAL_ISSUES,
        "max_high_issues": $MAX_HIGH_ISSUES,
        "max_medium_issues": $MAX_MEDIUM_ISSUES
    },
    "remediation": {
        "enabled": $ENABLE_AUTO_REMEDIATION,
        "report_file": "$remediation_report",
        "timeout": $REMEDIATION_TIMEOUT
    },
    "recommendations": [
        $([ $CRITICAL_ISSUES -gt 0 ] && echo '"Immediately address all critical security issues before deployment",' || echo '')
        $([ $HIGH_ISSUES -gt $MAX_HIGH_ISSUES ] && echo '"Reduce high-severity issues to acceptable levels",' || echo '')
        $([ "$gate_result" = "FAILED" ] && echo '"Security gate failed - deployment blocked for security reasons",' || echo '')
        "Implement continuous security monitoring in production environment"
    ]
}
EOF

    echo "$gate_report"
}

# Send security gate notifications
send_security_notification() {
    local gate_result="$1"
    local gate_report="$2"
    
    local notification_title="Security Gate $gate_result"
    local notification_color
    local notification_priority
    
    case "$gate_result" in
        "PASSED")
            notification_color="good"
            notification_priority="low"
            ;;
        "WARNING")
            notification_color="warning"
            notification_priority="medium"
            ;;
        "FAILED")
            notification_color="danger"
            notification_priority="high"
            ;;
        *)
            notification_color="warning"
            notification_priority="medium"
            ;;
    esac
    
    local message="Build: $BUILD_ID | Branch: $BRANCH_NAME | Critical: $CRITICAL_ISSUES | High: $HIGH_ISSUES"
    
    # Send to Slack
    if [ -n "$SLACK_SECURITY_CHANNEL" ]; then
        local slack_payload=$(cat << EOF
{
    "channel": "$SLACK_SECURITY_CHANNEL",
    "username": "Security Gate Bot",
    "icon_emoji": ":lock:",
    "attachments": [
        {
            "color": "$notification_color",
            "title": "🔒 $notification_title - AI Trading Bot",
            "text": "$message",
            "fields": [
                {
                    "title": "Build ID",
                    "value": "$BUILD_ID",
                    "short": true
                },
                {
                    "title": "Branch",
                    "value": "$BRANCH_NAME",
                    "short": true
                },
                {
                    "title": "Critical Issues",
                    "value": "$CRITICAL_ISSUES",
                    "short": true
                },
                {
                    "title": "High Issues",
                    "value": "$HIGH_ISSUES",
                    "short": true
                },
                {
                    "title": "Gate Mode",
                    "value": "$GATE_MODE",
                    "short": true
                },
                {
                    "title": "CI Environment",
                    "value": "$CI_ENVIRONMENT",
                    "short": true
                }
            ],
            "actions": [
                {
                    "type": "button",
                    "text": "View Report",
                    "url": "$PIPELINE_URL"
                }
            ]
        }
    ]
}
EOF
)
        
        if [ -n "$SECURITY_WEBHOOK_URL" ]; then
            curl -s -X POST "$SECURITY_WEBHOOK_URL" \
                -H "Content-Type: application/json" \
                -d "$slack_payload" || warning "Failed to send Slack notification"
        fi
    fi
    
    # Send to Microsoft Teams
    if [ -n "$TEAMS_WEBHOOK_URL" ]; then
        local teams_payload=$(cat << EOF
{
    "@type": "MessageCard",
    "@context": "https://schema.org/extensions",
    "summary": "$notification_title",
    "themeColor": "$([ "$notification_color" = "danger" ] && echo "FF0000" || [ "$notification_color" = "warning" ] && echo "FFA500" || echo "008000")",
    "sections": [
        {
            "activityTitle": "🔒 $notification_title",
            "activitySubtitle": "AI Trading Bot Deployment Pipeline",
            "facts": [
                {
                    "name": "Build ID",
                    "value": "$BUILD_ID"
                },
                {
                    "name": "Branch",
                    "value": "$BRANCH_NAME"
                },
                {
                    "name": "Critical Issues",
                    "value": "$CRITICAL_ISSUES"
                },
                {
                    "name": "High Issues",
                    "value": "$HIGH_ISSUES"
                },
                {
                    "name": "Gate Result",
                    "value": "$gate_result"
                }
            ]
        }
    ],
    "potentialAction": [
        {
            "@type": "OpenUri",
            "name": "View Pipeline",
            "targets": [
                {
                    "os": "default",
                    "uri": "$PIPELINE_URL"
                }
            ]
        }
    ]
}
EOF
)
        
        curl -s -X POST "$TEAMS_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "$teams_payload" || warning "Failed to send Teams notification"
    fi
    
    info "Security gate notification sent: $gate_result"
}

# Pre-deployment security checks
run_predeploy_checks() {
    info "Running pre-deployment security checks..."
    
    local checks_passed=0
    local checks_failed=0
    
    # Check 1: Ensure Docker is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        checks_failed=$((checks_failed + 1))
    else
        success "Docker daemon is accessible"
        checks_passed=$((checks_passed + 1))
    fi
    
    # Check 2: Verify Docker Bench Security is available
    if [ ! -f "${BASE_DIR}/docker-bench-security/docker-bench-security.sh" ]; then
        error "Docker Bench Security not found"
        checks_failed=$((checks_failed + 1))
    else
        success "Docker Bench Security is available"
        checks_passed=$((checks_passed + 1))
    fi
    
    # Check 3: Validate configuration files
    local compose_files=("${PROJECT_ROOT}/docker-compose.yml")
    for compose_file in "${compose_files[@]}"; do
        if [ -f "$compose_file" ]; then
            if docker-compose -f "$compose_file" config &> /dev/null; then
                success "Docker Compose configuration valid: $(basename "$compose_file")"
                checks_passed=$((checks_passed + 1))
            else
                error "Docker Compose configuration invalid: $(basename "$compose_file")"
                checks_failed=$((checks_failed + 1))
            fi
        else
            warning "Docker Compose file not found: $(basename "$compose_file")"
        fi
    done
    
    # Check 4: Verify security policies exist
    if [ -f "${BASE_DIR}/config/docker-bench.conf" ]; then
        success "Security policies configuration found"
        checks_passed=$((checks_passed + 1))
    else
        warning "Security policies not configured"
    fi
    
    info "Pre-deployment checks completed: $checks_passed passed, $checks_failed failed"
    
    if [ $checks_failed -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# Run comprehensive security scan
run_security_scan() {
    info "Running comprehensive security scan..."
    
    local scan_start_time=$(date +%s)
    local scan_report=""
    
    # Set timeout for security scan
    timeout $SECURITY_SCAN_TIMEOUT "${BASE_DIR}/scripts/run-security-scan.sh" > /dev/null 2>&1 || {
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            error "Security scan timed out after ${SECURITY_SCAN_TIMEOUT} seconds"
        else
            error "Security scan failed with exit code: $exit_code"
        fi
        return 1
    }
    
    # Get the latest scan report
    scan_report=$(ls -t "${BASE_DIR}/reports"/security-analysis-*.json 2>/dev/null | head -1)
    
    if [ -z "$scan_report" ] || [ ! -f "$scan_report" ]; then
        error "Security scan report not found"
        return 1
    fi
    
    # Parse scan results
    if command -v jq &> /dev/null; then
        CRITICAL_ISSUES=$(jq -r '.summary.critical_issues // 0' "$scan_report")
        HIGH_ISSUES=$(jq -r '.summary.high_issues // 0' "$scan_report")
        MEDIUM_ISSUES=$(jq -r '.summary.medium_issues // 0' "$scan_report")
        LOW_ISSUES=$(jq -r '.summary.low_issues // 0' "$scan_report")
        SECURITY_ISSUES_FOUND=$((CRITICAL_ISSUES + HIGH_ISSUES + MEDIUM_ISSUES + LOW_ISSUES))
    fi
    
    local scan_duration=$(($(date +%s) - scan_start_time))
    success "Security scan completed in ${scan_duration}s - Issues: Critical=$CRITICAL_ISSUES, High=$HIGH_ISSUES, Medium=$MEDIUM_ISSUES, Low=$LOW_ISSUES"
    
    echo "$scan_report"
}

# Apply automated remediation
run_automated_remediation() {
    local scan_report="$1"
    
    info "Running automated remediation..."
    
    if [ "$ENABLE_AUTO_REMEDIATION" != "true" ]; then
        info "Automated remediation disabled"
        return 0
    fi
    
    if [ $CRITICAL_ISSUES -eq 0 ] && [ $HIGH_ISSUES -eq 0 ]; then
        info "No critical or high issues found - skipping remediation"
        return 0
    fi
    
    local remediation_start_time=$(date +%s)
    local remediation_report=""
    
    # Run remediation with timeout
    if timeout $REMEDIATION_TIMEOUT "${BASE_DIR}/remediation/auto-remediate.sh" "$scan_report" > /dev/null 2>&1; then
        local remediation_duration=$(($(date +%s) - remediation_start_time))
        success "Automated remediation completed in ${remediation_duration}s"
        
        # Get remediation report
        remediation_report=$(ls -t "${BASE_DIR}/logs"/remediation.log 2>/dev/null | head -1)
        
        # Re-run security scan to verify fixes
        info "Re-running security scan to verify remediation..."
        local post_remediation_scan
        post_remediation_scan=$(run_security_scan)
        
        if [ -n "$post_remediation_scan" ]; then
            info "Post-remediation scan completed"
        fi
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            warning "Automated remediation timed out after ${REMEDIATION_TIMEOUT} seconds"
        else
            warning "Automated remediation failed with exit code: $exit_code"
        fi
    fi
    
    echo "$remediation_report"
}

# Evaluate security gate
evaluate_security_gate() {
    local scan_report="$1"
    
    info "Evaluating security gate with mode: $GATE_MODE"
    
    # In disabled mode, always pass
    if [ "$GATE_MODE" = "disabled" ]; then
        GATE_STATUS="BYPASSED"
        GATE_DETAILS="Security gate disabled by configuration"
        success "Security gate bypassed (disabled mode)"
        return 0
    fi
    
    # Evaluate critical issues
    if [ "$FAIL_ON_CRITICAL" = "true" ] && [ $CRITICAL_ISSUES -gt $MAX_CRITICAL_ISSUES ]; then
        GATE_STATUS="FAILED"
        GATE_DETAILS="Critical security issues found: $CRITICAL_ISSUES (max allowed: $MAX_CRITICAL_ISSUES)"
        error "$GATE_DETAILS"
        
        if [ "$GATE_MODE" = "enforcing" ]; then
            return 1
        fi
    fi
    
    # Evaluate high issues
    if [ "$FAIL_ON_HIGH" = "true" ] && [ $HIGH_ISSUES -gt $MAX_HIGH_ISSUES ]; then
        if [ "$GATE_STATUS" != "FAILED" ]; then
            GATE_STATUS="FAILED"
            GATE_DETAILS="High severity security issues found: $HIGH_ISSUES (max allowed: $MAX_HIGH_ISSUES)"
        fi
        error "High severity security issues found: $HIGH_ISSUES (max allowed: $MAX_HIGH_ISSUES)"
        
        if [ "$GATE_MODE" = "enforcing" ]; then
            return 1
        fi
    fi
    
    # Evaluate medium issues (warning only)
    if [ $MEDIUM_ISSUES -gt $MAX_MEDIUM_ISSUES ]; then
        if [ "$GATE_STATUS" != "FAILED" ]; then
            GATE_STATUS="WARNING"
            GATE_DETAILS="Medium severity security issues found: $MEDIUM_ISSUES (recommended max: $MAX_MEDIUM_ISSUES)"
        fi
        warning "Medium severity security issues found: $MEDIUM_ISSUES (recommended max: $MAX_MEDIUM_ISSUES)"
    fi
    
    # If no issues found, gate passes
    if [ "$GATE_STATUS" = "UNKNOWN" ]; then
        GATE_STATUS="PASSED"
        GATE_DETAILS="All security checks passed"
        success "Security gate passed - deployment approved"
    elif [ "$GATE_STATUS" = "WARNING" ]; then
        warning "Security gate passed with warnings"
    fi
    
    return 0
}

# Main security gate execution
main() {
    local gate_start_time=$(date +%s)
    
    info "Starting security gate for AI Trading Bot"
    info "Build ID: $BUILD_ID | Branch: $BRANCH_NAME | Mode: $GATE_MODE"
    
    # Run pre-deployment checks
    if ! run_predeploy_checks; then
        error "Pre-deployment checks failed"
        exit 1
    fi
    
    # Run security scan
    local scan_report
    if ! scan_report=$(run_security_scan); then
        error "Security scan failed"
        GATE_STATUS="FAILED"
        GATE_DETAILS="Security scan execution failed"
    else
        # Run automated remediation if needed
        local remediation_report=""
        if [ $CRITICAL_ISSUES -gt 0 ] || [ $HIGH_ISSUES -gt 0 ]; then
            remediation_report=$(run_automated_remediation "$scan_report")
        fi
        
        # Evaluate security gate
        if ! evaluate_security_gate "$scan_report"; then
            if [ "$GATE_MODE" = "enforcing" ]; then
                error "Security gate evaluation failed in enforcing mode"
            else
                warning "Security gate evaluation failed in permissive mode"
            fi
        fi
        
        # Create gate report
        local gate_report
        gate_report=$(create_gate_report "$GATE_STATUS" "$scan_report" "$remediation_report")
        
        # Send notifications
        send_security_notification "$GATE_STATUS" "$gate_report"
    fi
    
    local gate_duration=$(($(date +%s) - gate_start_time))
    
    # Final gate decision
    case "$GATE_STATUS" in
        "PASSED")
            success "✅ Security gate PASSED in ${gate_duration}s - Deployment approved"
            exit 0
            ;;
        "WARNING")
            warning "⚠️  Security gate PASSED with warnings in ${gate_duration}s"
            exit 0
            ;;
        "FAILED")
            error "❌ Security gate FAILED in ${gate_duration}s - Deployment blocked"
            error "$GATE_DETAILS"
            exit 1
            ;;
        "BYPASSED")
            warning "⏭️  Security gate BYPASSED in ${gate_duration}s"
            exit 0
            ;;
        *)
            error "❓ Security gate status unknown - Deployment blocked"
            exit 1
            ;;
    esac
}

# Command line interface
case "${1:-run}" in
    "run")
        main
        ;;
    "check-config")
        info "Security gate configuration:"
        echo "  Mode: $GATE_MODE"
        echo "  Fail on critical: $FAIL_ON_CRITICAL"
        echo "  Fail on high: $FAIL_ON_HIGH"
        echo "  Max critical: $MAX_CRITICAL_ISSUES"
        echo "  Max high: $MAX_HIGH_ISSUES"
        echo "  Max medium: $MAX_MEDIUM_ISSUES"
        echo "  Auto remediation: $ENABLE_AUTO_REMEDIATION"
        ;;
    "test")
        info "Running security gate test..."
        GATE_MODE="permissive"
        main
        ;;
    *)
        echo "Usage: $0 {run|check-config|test}"
        echo "  run           - Execute security gate (default)"
        echo "  check-config  - Display current configuration"
        echo "  test          - Run in test mode (permissive)"
        exit 1
        ;;
esac