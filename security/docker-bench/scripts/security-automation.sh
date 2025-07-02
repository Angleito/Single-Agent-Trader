#!/bin/bash

# Docker Security Automation Framework
# Advanced automated security scanning with scheduling, monitoring, and alerting

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${BASE_DIR}/config"
REPORTS_DIR="${BASE_DIR}/reports"
LOGS_DIR="${BASE_DIR}/logs"
REMEDIATION_DIR="${BASE_DIR}/remediation"

# Load configuration
CONFIG_FILE="${CONFIG_DIR}/security-automation.conf"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Default configuration values
SCAN_INTERVAL_HOURS=${SCAN_INTERVAL_HOURS:-6}
REMEDIATION_ENABLED=${REMEDIATION_ENABLED:-true}
ALERT_WEBHOOK_URL=${ALERT_WEBHOOK_URL:-""}
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-""}
EMAIL_ALERTS_ENABLED=${EMAIL_ALERTS_ENABLED:-false}
EMAIL_RECIPIENTS=${EMAIL_RECIPIENTS:-""}
METRICS_ENABLED=${METRICS_ENABLED:-true}
METRICS_PORT=${METRICS_PORT:-9095}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOGS_DIR}/security-automation.log"
}

info() { log "INFO" "${BLUE}$1${NC}"; }
success() { log "SUCCESS" "${GREEN}$1${NC}"; }
warning() { log "WARNING" "${YELLOW}$1${NC}"; }
error() { log "ERROR" "${RED}$1${NC}"; }
debug() { log "DEBUG" "${PURPLE}$1${NC}"; }

# Create PID file management
create_pid_file() {
    local pid_file="${BASE_DIR}/security-automation.pid"
    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            error "Security automation is already running with PID $old_pid"
            exit 1
        else
            warning "Removing stale PID file"
            rm -f "$pid_file"
        fi
    fi
    echo $$ > "$pid_file"
}

cleanup() {
    local pid_file="${BASE_DIR}/security-automation.pid"
    rm -f "$pid_file"
    info "Security automation stopped"
}

trap cleanup EXIT

# Metrics collection
init_metrics() {
    if [ "$METRICS_ENABLED" = "true" ]; then
        local metrics_file="${BASE_DIR}/metrics/security-metrics.prom"
        mkdir -p "$(dirname "$metrics_file")"
        
        cat > "$metrics_file" << 'EOF'
# HELP docker_security_scan_total Total number of security scans performed
# TYPE docker_security_scan_total counter
docker_security_scan_total 0

# HELP docker_security_issues_total Total number of security issues found
# TYPE docker_security_issues_total gauge
docker_security_issues_total{severity="critical"} 0
docker_security_issues_total{severity="high"} 0
docker_security_issues_total{severity="medium"} 0
docker_security_issues_total{severity="low"} 0

# HELP docker_security_compliance_score Security compliance score (0-100)
# TYPE docker_security_compliance_score gauge
docker_security_compliance_score 100

# HELP docker_security_last_scan_timestamp Timestamp of last security scan
# TYPE docker_security_last_scan_timestamp gauge
docker_security_last_scan_timestamp 0

# HELP docker_security_remediation_total Total number of issues automatically remediated
# TYPE docker_security_remediation_total counter
docker_security_remediation_total 0
EOF
        
        info "Metrics initialized at $metrics_file"
    fi
}

update_metrics() {
    if [ "$METRICS_ENABLED" = "true" ]; then
        local metrics_file="${BASE_DIR}/metrics/security-metrics.prom"
        local critical_count="$1"
        local high_count="$2"
        local medium_count="$3"
        local low_count="$4"
        local compliance_score="$5"
        local scan_timestamp="$6"
        
        # Update metrics file
        sed -i "s/docker_security_scan_total .*/docker_security_scan_total $(($(grep -o 'docker_security_scan_total [0-9]*' "$metrics_file" | awk '{print $2}') + 1))/" "$metrics_file"
        sed -i "s/docker_security_issues_total{severity=\"critical\"} .*/docker_security_issues_total{severity=\"critical\"} $critical_count/" "$metrics_file"
        sed -i "s/docker_security_issues_total{severity=\"high\"} .*/docker_security_issues_total{severity=\"high\"} $high_count/" "$metrics_file"
        sed -i "s/docker_security_issues_total{severity=\"medium\"} .*/docker_security_issues_total{severity=\"medium\"} $medium_count/" "$metrics_file"
        sed -i "s/docker_security_issues_total{severity=\"low\"} .*/docker_security_issues_total{severity=\"low\"} $low_count/" "$metrics_file"
        sed -i "s/docker_security_compliance_score .*/docker_security_compliance_score $compliance_score/" "$metrics_file"
        sed -i "s/docker_security_last_scan_timestamp .*/docker_security_last_scan_timestamp $scan_timestamp/" "$metrics_file"
    fi
}

# Notification system
send_alert() {
    local severity="$1"
    local title="$2"
    local message="$3"
    local report_file="$4"
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local hostname=$(hostname)
    
    # Create alert payload
    local alert_payload=$(cat << EOF
{
    "timestamp": "$timestamp",
    "hostname": "$hostname",
    "service": "ai-trading-bot-security",
    "severity": "$severity",
    "title": "$title",
    "message": "$message",
    "report_file": "$report_file",
    "environment": "production"
}
EOF
)
    
    # Send to webhook if configured
    if [ -n "$ALERT_WEBHOOK_URL" ]; then
        curl -s -X POST "$ALERT_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "$alert_payload" || warning "Failed to send webhook alert"
    fi
    
    # Send to Slack if configured
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        local slack_color
        case "$severity" in
            "critical") slack_color="danger" ;;
            "high") slack_color="warning" ;;
            "medium") slack_color="good" ;;
            *) slack_color="good" ;;
        esac
        
        local slack_payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$slack_color",
            "title": "🚨 $title",
            "text": "$message",
            "fields": [
                {
                    "title": "Hostname",
                    "value": "$hostname",
                    "short": true
                },
                {
                    "title": "Severity",
                    "value": "$severity",
                    "short": true
                },
                {
                    "title": "Timestamp",
                    "value": "$timestamp",
                    "short": false
                }
            ]
        }
    ]
}
EOF
)
        
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "$slack_payload" || warning "Failed to send Slack alert"
    fi
    
    # Send email if configured
    if [ "$EMAIL_ALERTS_ENABLED" = "true" ] && [ -n "$EMAIL_RECIPIENTS" ]; then
        echo -e "Subject: [SECURITY ALERT] $title\n\n$message\n\nTimestamp: $timestamp\nHostname: $hostname\nReport: $report_file" | \
            sendmail "$EMAIL_RECIPIENTS" || warning "Failed to send email alert"
    fi
    
    info "Alert sent: $severity - $title"
}

# Security scan orchestrator
run_comprehensive_scan() {
    info "Starting comprehensive security scan..."
    
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local scan_start_time=$(date +%s)
    
    # Run the security scan
    local scan_report
    if ! scan_report=$("${SCRIPT_DIR}/run-security-scan.sh"); then
        error "Security scan failed"
        send_alert "critical" "Security Scan Failed" "The automated security scan failed to complete" ""
        return 1
    fi
    
    # Parse scan results
    local analysis_file="${REPORTS_DIR}/security-analysis-${timestamp}.json"
    if [ ! -f "$analysis_file" ]; then
        error "Analysis report not found: $analysis_file"
        return 1
    fi
    
    # Extract metrics from analysis
    local critical_count=0
    local high_count=0
    local medium_count=0
    local low_count=0
    local compliance_score=100
    
    if command -v jq &> /dev/null; then
        critical_count=$(jq -r '.summary.critical_issues // 0' "$analysis_file")
        high_count=$(jq -r '.summary.high_issues // 0' "$analysis_file")
        medium_count=$(jq -r '.summary.medium_issues // 0' "$analysis_file")
        low_count=$(jq -r '.summary.low_issues // 0' "$analysis_file")
        
        # Calculate compliance score
        local total_issues=$((critical_count + high_count + medium_count + low_count))
        local total_checks=$(jq -r '.summary.total_checks // 1' "$analysis_file")
        compliance_score=$(( (total_checks - total_issues) * 100 / total_checks ))
    fi
    
    # Update metrics
    update_metrics "$critical_count" "$high_count" "$medium_count" "$low_count" "$compliance_score" "$scan_start_time"
    
    # Check for alert conditions
    check_alert_conditions "$critical_count" "$high_count" "$medium_count" "$low_count" "$analysis_file"
    
    # Log scan completion
    local scan_duration=$(($(date +%s) - scan_start_time))
    success "Security scan completed in ${scan_duration}s - Critical: $critical_count, High: $high_count, Medium: $medium_count, Low: $low_count"
    
    # Trigger remediation if enabled
    if [ "$REMEDIATION_ENABLED" = "true" ] && [ $((critical_count + high_count)) -gt 0 ]; then
        info "Triggering automated remediation..."
        run_automated_remediation "$analysis_file"
    fi
    
    return 0
}

# Alert condition checker
check_alert_conditions() {
    local critical_count="$1"
    local high_count="$2"
    local medium_count="$3"
    local low_count="$4"
    local analysis_file="$5"
    
    # Critical issues always trigger alerts
    if [ "$critical_count" -gt 0 ]; then
        send_alert "critical" "Critical Security Issues Found" \
            "Found $critical_count critical security issues in Docker containers. Immediate action required." \
            "$analysis_file"
    fi
    
    # High issues trigger alerts if above threshold
    local high_threshold=${ALERT_HIGH_THRESHOLD:-1}
    if [ "$high_count" -gt "$high_threshold" ]; then
        send_alert "high" "High Severity Security Issues" \
            "Found $high_count high severity security issues (threshold: $high_threshold)." \
            "$analysis_file"
    fi
    
    # Medium issues trigger alerts if above threshold
    local medium_threshold=${ALERT_MEDIUM_THRESHOLD:-5}
    if [ "$medium_count" -gt "$medium_threshold" ]; then
        send_alert "medium" "Medium Severity Security Issues" \
            "Found $medium_count medium severity security issues (threshold: $medium_threshold)." \
            "$analysis_file"
    fi
    
    # Check compliance status
    local compliance_threshold=${COMPLIANCE_THRESHOLD:-80}
    local total_issues=$((critical_count + high_count + medium_count + low_count))
    if [ "$total_issues" -gt 0 ]; then
        local total_checks=$(jq -r '.summary.total_checks // 1' "$analysis_file" 2>/dev/null || echo "1")
        local compliance_score=$(( (total_checks - total_issues) * 100 / total_checks ))
        
        if [ "$compliance_score" -lt "$compliance_threshold" ]; then
            send_alert "high" "Security Compliance Below Threshold" \
                "Security compliance score is ${compliance_score}% (threshold: ${compliance_threshold}%)." \
                "$analysis_file"
        fi
    fi
}

# Automated remediation
run_automated_remediation() {
    local analysis_file="$1"
    
    info "Running automated remediation..."
    
    # Check if remediation script exists
    local remediation_script="${REMEDIATION_DIR}/auto-remediate.sh"
    if [ ! -f "$remediation_script" ]; then
        warning "Automated remediation script not found: $remediation_script"
        return 1
    fi
    
    # Run remediation
    if bash "$remediation_script" "$analysis_file"; then
        success "Automated remediation completed successfully"
        
        # Update remediation metrics
        if [ "$METRICS_ENABLED" = "true" ]; then
            local metrics_file="${BASE_DIR}/metrics/security-metrics.prom"
            local current_count=$(grep -o 'docker_security_remediation_total [0-9]*' "$metrics_file" | awk '{print $2}' || echo "0")
            sed -i "s/docker_security_remediation_total .*/docker_security_remediation_total $((current_count + 1))/" "$metrics_file"
        fi
        
        # Send success notification
        send_alert "medium" "Automated Remediation Completed" \
            "Automated remediation has been applied to resolve security issues." \
            "$analysis_file"
    else
        error "Automated remediation failed"
        send_alert "high" "Automated Remediation Failed" \
            "Automated remediation failed to complete. Manual intervention required." \
            "$analysis_file"
        return 1
    fi
}

# Health check for the automation service
health_check() {
    local status="healthy"
    local issues=()
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        status="unhealthy"
        issues+=("Docker daemon not accessible")
    fi
    
    # Check required directories
    local required_dirs=("$REPORTS_DIR" "$LOGS_DIR" "$CONFIG_DIR")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            status="unhealthy"
            issues+=("Required directory missing: $dir")
        fi
    done
    
    # Check disk space
    local disk_usage=$(df "$BASE_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        status="unhealthy"
        issues+=("Disk usage high: ${disk_usage}%")
    fi
    
    # Check last scan age
    local last_scan_file=$(ls -t "$REPORTS_DIR"/security-analysis-*.json 2>/dev/null | head -1)
    if [ -n "$last_scan_file" ]; then
        local last_scan_age=$(( ($(date +%s) - $(stat -c %Y "$last_scan_file")) / 3600 ))
        local max_age_hours=${MAX_SCAN_AGE_HOURS:-24}
        if [ "$last_scan_age" -gt "$max_age_hours" ]; then
            status="unhealthy"
            issues+=("Last scan too old: ${last_scan_age}h")
        fi
    fi
    
    # Report health status
    if [ "$status" = "healthy" ]; then
        info "Health check passed - all systems operational"
        return 0
    else
        error "Health check failed: ${issues[*]}"
        return 1
    fi
}

# Cleanup old reports
cleanup_old_reports() {
    local retention_days=${REPORT_RETENTION_DAYS:-30}
    
    info "Cleaning up reports older than $retention_days days..."
    
    find "$REPORTS_DIR" -name "*.json" -mtime +$retention_days -delete
    find "$REPORTS_DIR" -name "*.log" -mtime +$retention_days -delete
    find "$LOGS_DIR" -name "*.log" -mtime +$retention_days -delete
    
    success "Cleanup completed"
}

# Main daemon loop
run_daemon() {
    info "Starting security automation daemon (PID: $$)"
    info "Scan interval: ${SCAN_INTERVAL_HOURS} hours"
    info "Remediation enabled: $REMEDIATION_ENABLED"
    
    create_pid_file
    init_metrics
    
    # Initial health check
    if ! health_check; then
        error "Initial health check failed - exiting"
        exit 1
    fi
    
    # Run initial scan
    run_comprehensive_scan
    
    # Main loop
    while true; do
        info "Sleeping for ${SCAN_INTERVAL_HOURS} hours until next scan..."
        sleep $((SCAN_INTERVAL_HOURS * 3600))
        
        # Periodic health check
        if ! health_check; then
            warning "Health check failed - continuing with caution"
        fi
        
        # Run scheduled scan
        run_comprehensive_scan
        
        # Cleanup old reports
        cleanup_old_reports
    done
}

# Command line interface
case "${1:-daemon}" in
    "daemon")
        run_daemon
        ;;
    "scan")
        run_comprehensive_scan
        ;;
    "health")
        health_check
        exit $?
        ;;
    "cleanup")
        cleanup_old_reports
        ;;
    "test-alert")
        send_alert "medium" "Test Alert" "This is a test alert from the security automation system" ""
        ;;
    *)
        echo "Usage: $0 {daemon|scan|health|cleanup|test-alert}"
        exit 1
        ;;
esac