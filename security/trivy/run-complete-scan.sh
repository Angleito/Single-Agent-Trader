#!/bin/bash

# Complete Security Scan Pipeline for AI Trading Bot
# This master script runs all Trivy security scans and generates comprehensive reports

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default configuration
SCAN_IMAGES=true
SCAN_FILESYSTEM=true
GENERATE_DASHBOARD=true
GENERATE_REMEDIATION=true
RUN_SECURITY_GATE=true
CLEANUP_OLD_REPORTS=true
NOTIFY_ON_COMPLETION=false

# Output configuration
OUTPUT_FORMATS="table,json,sarif"
SEVERITY_LEVELS="CRITICAL,HIGH,MEDIUM"
EXIT_ON_FAILURE=false
SAVE_ARTIFACTS=true

# Notification settings
SLACK_WEBHOOK=""
EMAIL_NOTIFICATION=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

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

log_header() {
    echo -e "${BOLD}${CYAN}=== $1 ===${NC}"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Function to display banner
show_banner() {
    echo -e "${BOLD}${CYAN}"
    cat << "EOF"
   _____ _________   ____________  ____  ____  _____ __  ________
  / ___// ____/ /  / / __ \  _/ / / /  \/ / / / ___// / / / ____/
  \__ \/ __/ / /  / / /_/ // // / / /|  / / /  \__ \/ /_/ / __/   
 ___/ / /___/ /__/ / _, _// // /_/ / /|  / /_/ ___/ / __  / /___   
/____/_____/_____/_/ |_/___/\____/_/ |_/\____/____/_/ /_/_____/   
                                                                  
EOF
    echo -e "${NC}"
    echo -e "${BOLD}AI Trading Bot - Complete Security Scan Pipeline${NC}"
    echo -e "Comprehensive vulnerability scanning and security analysis"
    echo ""
}

# Function to check prerequisites
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    local missing_deps=()
    
    # Check for required tools
    if ! command -v trivy >/dev/null 2>&1; then
        missing_deps+=("trivy")
    fi
    
    if ! command -v docker >/dev/null 2>&1; then
        missing_deps+=("docker")
    fi
    
    if ! command -v jq >/dev/null 2>&1; then
        log_warning "jq not found - some features will be limited"
    fi
    
    if [[ "$GENERATE_DASHBOARD" == true ]]; then
        if ! python3 -c "import matplotlib, pandas, seaborn, jinja2" 2>/dev/null; then
            log_warning "Python dashboard dependencies missing - install with: pip install matplotlib pandas seaborn jinja2"
        fi
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Install missing dependencies and try again"
        log_info "Run: ./install-trivy.sh to install Trivy"
        exit 1
    fi
    
    # Check if scripts are executable
    for script in scan-images.sh scan-filesystem.sh ci-cd-security-gate.sh security-dashboard.py remediation-automation.sh; do
        if [[ ! -x "$SCRIPT_DIR/$script" ]]; then
            chmod +x "$SCRIPT_DIR/$script"
            log_info "Made $script executable"
        fi
    done
    
    log_success "All prerequisites satisfied"
}

# Function to setup environment
setup_environment() {
    log_header "Setting Up Environment"
    
    # Create reports directory structure
    mkdir -p "$SCRIPT_DIR/reports"/{images,filesystem,secrets,configs,licenses,json,html,sarif,sbom,ci,dashboard,remediation,archive}
    
    # Set environment variables
    export TRIVY_CACHE_DIR="$SCRIPT_DIR/reports/.cache"
    export TRIVY_CONFIG="$SCRIPT_DIR/trivy-config.yaml"
    
    # Create scan session directory
    export SCAN_SESSION_DIR="$SCRIPT_DIR/reports/archive/session_$TIMESTAMP"
    mkdir -p "$SCAN_SESSION_DIR"
    
    log_success "Environment setup completed"
    echo "Scan session: $TIMESTAMP"
    echo "Reports directory: $SCRIPT_DIR/reports"
}

# Function to update Trivy database
update_trivy_database() {
    log_header "Updating Trivy Database"
    
    log_info "Downloading latest vulnerability database..."
    if trivy image --download-db-only; then
        log_success "Trivy database updated successfully"
    else
        log_warning "Failed to update Trivy database - continuing with existing database"
    fi
}

# Function to scan Docker images
run_image_scan() {
    if [[ "$SCAN_IMAGES" != true ]]; then
        return
    fi
    
    log_header "Docker Image Security Scan"
    
    log_step "Scanning Docker images for vulnerabilities..."
    
    local scan_args=()
    scan_args+=("--all")
    scan_args+=("--format" "$OUTPUT_FORMATS")
    scan_args+=("--severity" "$SEVERITY_LEVELS")
    
    if [[ "$SAVE_ARTIFACTS" == true ]]; then
        scan_args+=("--save-reports")
    fi
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        scan_args+=("--slack-webhook" "$SLACK_WEBHOOK")
    fi
    
    if [[ "$EXIT_ON_FAILURE" == true ]]; then
        scan_args+=("--exit-on-vuln")
    fi
    
    if "$SCRIPT_DIR/scan-images.sh" "${scan_args[@]}"; then
        log_success "Docker image scan completed successfully"
        return 0
    else
        log_error "Docker image scan failed or found critical issues"
        return 1
    fi
}

# Function to scan filesystem
run_filesystem_scan() {
    if [[ "$SCAN_FILESYSTEM" != true ]]; then
        return
    fi
    
    log_header "Filesystem Security Scan"
    
    log_step "Scanning filesystem, dependencies, and source code..."
    
    local scan_args=()
    scan_args+=("--format" "$OUTPUT_FORMATS")
    scan_args+=("--severity" "$SEVERITY_LEVELS")
    
    if [[ "$SAVE_ARTIFACTS" == true ]]; then
        scan_args+=("--save-reports")
    fi
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        scan_args+=("--slack-webhook" "$SLACK_WEBHOOK")
    fi
    
    if [[ "$EXIT_ON_FAILURE" == true ]]; then
        scan_args+=("--exit-on-vuln")
    fi
    
    if "$SCRIPT_DIR/scan-filesystem.sh" "${scan_args[@]}"; then
        log_success "Filesystem scan completed successfully"
        return 0
    else
        log_error "Filesystem scan failed or found critical issues"
        return 1
    fi
}

# Function to run security gate analysis
run_security_gate() {
    if [[ "$RUN_SECURITY_GATE" != true ]]; then
        return
    fi
    
    log_header "Security Gate Analysis"
    
    log_step "Running security gate policies and compliance checks..."
    
    local gate_args=()
    gate_args+=("--max-critical" "0")
    gate_args+=("--max-high" "5")
    gate_args+=("--max-medium" "20")
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        gate_args+=("--slack-webhook" "$SLACK_WEBHOOK")
    fi
    
    if "$SCRIPT_DIR/ci-cd-security-gate.sh" "${gate_args[@]}"; then
        log_success "Security gate analysis completed - ALL CHECKS PASSED"
        return 0
    else
        log_error "Security gate analysis failed - SECURITY ISSUES FOUND"
        return 1
    fi
}

# Function to generate security dashboard
generate_dashboard() {
    if [[ "$GENERATE_DASHBOARD" != true ]]; then
        return
    fi
    
    log_header "Generating Security Dashboard"
    
    log_step "Creating comprehensive security dashboard and metrics..."
    
    if python3 "$SCRIPT_DIR/security-dashboard.py" \
        --reports-dir "$SCRIPT_DIR/reports" \
        --output-dir "$SCRIPT_DIR/reports/dashboard"; then
        log_success "Security dashboard generated successfully"
        return 0
    else
        log_warning "Dashboard generation failed - continuing without dashboard"
        return 1
    fi
}

# Function to generate remediation suggestions
generate_remediation() {
    if [[ "$GENERATE_REMEDIATION" != true ]]; then
        return
    fi
    
    log_header "Generating Remediation Suggestions"
    
    log_step "Analyzing results and creating remediation guidance..."
    
    local remediation_args=()
    remediation_args+=("--generate-patches")
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        remediation_args+=("--slack-webhook" "$SLACK_WEBHOOK")
    fi
    
    if "$SCRIPT_DIR/remediation-automation.sh" "${remediation_args[@]}"; then
        log_success "Remediation analysis completed"
        return 0
    else
        log_warning "Remediation analysis failed - continuing without remediation"
        return 1
    fi
}

# Function to generate comprehensive report
generate_final_report() {
    log_header "Generating Final Report"
    
    local report_file="$SCAN_SESSION_DIR/security_scan_summary_$TIMESTAMP.md"
    
    cat > "$report_file" <<EOF
# AI Trading Bot - Security Scan Summary

**Scan Date**: $(date)
**Scan Session**: $TIMESTAMP
**Pipeline Version**: Complete Security Scan v1.0

## Scan Configuration

- **Image Scanning**: $SCAN_IMAGES
- **Filesystem Scanning**: $SCAN_FILESYSTEM
- **Security Gate**: $RUN_SECURITY_GATE
- **Dashboard Generation**: $GENERATE_DASHBOARD
- **Remediation Analysis**: $GENERATE_REMEDIATION

## Results Summary

EOF
    
    # Count results if possible
    local total_vulns=0
    local critical_vulns=0
    local high_vulns=0
    local secrets_found=0
    
    if command -v jq >/dev/null 2>&1; then
        for json_file in "$SCRIPT_DIR/reports/json"/*.json; do
            if [[ -f "$json_file" ]]; then
                local file_critical=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL") | .VulnerabilityID' "$json_file" 2>/dev/null | wc -l)
                local file_high=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH") | .VulnerabilityID' "$json_file" 2>/dev/null | wc -l)
                local file_secrets=$(jq -r '.Results[]?.Secrets[]? | .RuleID' "$json_file" 2>/dev/null | wc -l)
                
                critical_vulns=$((critical_vulns + file_critical))
                high_vulns=$((high_vulns + file_high))
                secrets_found=$((secrets_found + file_secrets))
            fi
        done
        total_vulns=$((critical_vulns + high_vulns))
    fi
    
    cat >> "$report_file" <<EOF
### Vulnerability Summary

- **Total Critical**: $critical_vulns
- **Total High**: $high_vulns
- **Total Secrets**: $secrets_found

### Security Status

EOF
    
    if [[ $critical_vulns -eq 0 && $secrets_found -eq 0 ]]; then
        echo "✅ **SECURITY STATUS**: PASSED - No critical security issues found" >> "$report_file"
    else
        echo "❌ **SECURITY STATUS**: FAILED - Critical security issues require attention" >> "$report_file"
    fi
    
    cat >> "$report_file" <<EOF

## Generated Artifacts

### Reports
- Text reports: \`security/trivy/reports/images/\`, \`security/trivy/reports/filesystem/\`
- JSON reports: \`security/trivy/reports/json/\`
- SARIF reports: \`security/trivy/reports/sarif/\`
- HTML reports: \`security/trivy/reports/html/\`

### Analysis
- Security dashboard: \`security/trivy/reports/dashboard/security_dashboard.html\`
- Remediation guidance: \`security/trivy/reports/remediation/\`
- SBOM files: \`security/trivy/reports/sbom/\`

### Session Archive
- Complete session: \`security/trivy/reports/archive/session_$TIMESTAMP/\`

## Next Steps

1. **Review Results**: Examine detailed reports in the reports directory
2. **Address Critical Issues**: Fix critical vulnerabilities and remove secrets
3. **Apply Remediations**: Use generated scripts and patches
4. **Update Dependencies**: Apply security patches
5. **Rerun Scans**: Verify fixes with another complete scan

## Resources

- [Security Dashboard](./reports/dashboard/security_dashboard.html)
- [Remediation Scripts](./reports/remediation/scripts/)
- [Security Hardening Guide](../../SECURITY_HARDENING_GUIDE.md)

---
*Generated by AI Trading Bot Security Pipeline*
EOF
    
    # Copy important files to session archive
    cp -r "$SCRIPT_DIR/reports/json" "$SCAN_SESSION_DIR/" 2>/dev/null || true
    cp -r "$SCRIPT_DIR/reports/dashboard" "$SCAN_SESSION_DIR/" 2>/dev/null || true
    cp -r "$SCRIPT_DIR/reports/remediation" "$SCAN_SESSION_DIR/" 2>/dev/null || true
    
    log_success "Final report generated: $report_file"
    
    # Display summary
    echo ""
    log_header "SCAN RESULTS SUMMARY"
    echo "Critical Vulnerabilities: $critical_vulns"
    echo "High Vulnerabilities: $high_vulns"
    echo "Secrets Found: $secrets_found"
    
    if [[ $critical_vulns -eq 0 && $secrets_found -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✅ SECURITY STATUS: PASSED${NC}"
    else
        echo -e "${RED}${BOLD}❌ SECURITY STATUS: FAILED${NC}"
    fi
}

# Function to cleanup old reports
cleanup_old_reports() {
    if [[ "$CLEANUP_OLD_REPORTS" != true ]]; then
        return
    fi
    
    log_header "Cleaning Up Old Reports"
    
    # Keep last 10 scan sessions
    find "$SCRIPT_DIR/reports/archive" -name "session_*" -type d | sort | head -n -10 | xargs rm -rf 2>/dev/null || true
    
    # Clean temporary files older than 7 days
    find "$SCRIPT_DIR/reports" -name "*.tmp" -mtime +7 -delete 2>/dev/null || true
    
    # Compress old JSON files
    find "$SCRIPT_DIR/reports/json" -name "*.json" -mtime +7 -exec gzip {} \; 2>/dev/null || true
    
    log_success "Old reports cleaned up"
}

# Function to send completion notification
send_completion_notification() {
    if [[ "$NOTIFY_ON_COMPLETION" != true ]]; then
        return
    fi
    
    log_info "Sending completion notification..."
    
    local status_emoji="✅"
    local status_text="COMPLETED"
    local color="good"
    
    if [[ $? -ne 0 ]]; then
        status_emoji="❌"
        status_text="FAILED"
        color="danger"
    fi
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local slack_payload=$(cat <<EOF
{
  "attachments": [
    {
      "color": "$color",
      "title": "$status_emoji Security Scan Pipeline $status_text",
      "fields": [
        {
          "title": "Scan Session",
          "value": "$TIMESTAMP",
          "short": true
        },
        {
          "title": "Duration",
          "value": "$(date -d@$(($(date +%s) - scan_start_time)) -u +%H:%M:%S)",
          "short": true
        }
      ],
      "footer": "AI Trading Bot Security Pipeline",
      "ts": $(date +%s)
    }
  ]
}
EOF
)
        
        curl -X POST -H 'Content-type: application/json' \
            --data "$slack_payload" \
            "$SLACK_WEBHOOK" || log_warning "Failed to send Slack notification"
    fi
}

# Function to display help
show_help() {
    cat <<EOF
Complete Security Scan Pipeline for AI Trading Bot

Usage: $0 [OPTIONS]

Options:
    --skip-images           Skip Docker image scanning
    --skip-filesystem       Skip filesystem scanning
    --skip-dashboard        Skip dashboard generation
    --skip-remediation      Skip remediation analysis
    --skip-security-gate    Skip security gate enforcement
    --no-cleanup           Don't cleanup old reports
    --exit-on-failure      Exit pipeline on any scan failure
    --format FORMATS       Output formats: table,json,sarif,html (default: table,json,sarif)
    --severity LEVELS      Severity levels: CRITICAL,HIGH,MEDIUM,LOW (default: CRITICAL,HIGH,MEDIUM)
    --slack-webhook URL    Send notifications to Slack
    --notify-completion    Send notification when pipeline completes
    --help, -h             Show this help message

Examples:
    $0                                    # Run complete security scan
    $0 --skip-dashboard --skip-remediation # Quick scan without analysis
    $0 --exit-on-failure                 # Fail fast on security issues
    $0 --format json --severity CRITICAL,HIGH # Custom output and severity
    $0 --slack-webhook https://hooks.slack.com/... --notify-completion

Pipeline Steps:
1. Prerequisites check and environment setup
2. Trivy database update
3. Docker image vulnerability scanning
4. Filesystem and source code scanning
5. Security gate policy enforcement
6. Security dashboard generation
7. Remediation suggestions and automation
8. Final report generation and cleanup

Output Locations:
- Reports: security/trivy/reports/
- Dashboard: security/trivy/reports/dashboard/security_dashboard.html
- Session: security/trivy/reports/archive/session_<timestamp>/
EOF
}

# Main function
main() {
    local scan_start_time=$(date +%s)
    
    # Show banner
    show_banner
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-images)
                SCAN_IMAGES=false
                shift
                ;;
            --skip-filesystem)
                SCAN_FILESYSTEM=false
                shift
                ;;
            --skip-dashboard)
                GENERATE_DASHBOARD=false
                shift
                ;;
            --skip-remediation)
                GENERATE_REMEDIATION=false
                shift
                ;;
            --skip-security-gate)
                RUN_SECURITY_GATE=false
                shift
                ;;
            --no-cleanup)
                CLEANUP_OLD_REPORTS=false
                shift
                ;;
            --exit-on-failure)
                EXIT_ON_FAILURE=true
                shift
                ;;
            --format)
                OUTPUT_FORMATS="$2"
                shift 2
                ;;
            --severity)
                SEVERITY_LEVELS="$2"
                shift 2
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
                NOTIFY_ON_COMPLETION=true
                shift 2
                ;;
            --notify-completion)
                NOTIFY_ON_COMPLETION=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Initialize pipeline
    check_prerequisites
    setup_environment
    update_trivy_database
    
    # Track failures
    local failed_steps=()
    
    # Run scans
    if ! run_image_scan; then
        failed_steps+=("image-scan")
        [[ "$EXIT_ON_FAILURE" == true ]] && exit 1
    fi
    
    if ! run_filesystem_scan; then
        failed_steps+=("filesystem-scan")
        [[ "$EXIT_ON_FAILURE" == true ]] && exit 1
    fi
    
    if ! run_security_gate; then
        failed_steps+=("security-gate")
        [[ "$EXIT_ON_FAILURE" == true ]] && exit 1
    fi
    
    # Generate analysis
    if ! generate_dashboard; then
        failed_steps+=("dashboard")
    fi
    
    if ! generate_remediation; then
        failed_steps+=("remediation")
    fi
    
    # Finalize
    generate_final_report
    cleanup_old_reports
    send_completion_notification
    
    # Final status
    local scan_duration=$(($(date +%s) - scan_start_time))
    echo ""
    log_header "PIPELINE COMPLETED"
    echo "Duration: $(date -d@$scan_duration -u +%H:%M:%S)"
    echo "Session: $TIMESTAMP"
    echo "Reports: $SCRIPT_DIR/reports/"
    
    if [[ ${#failed_steps[@]} -gt 0 ]]; then
        echo -e "${RED}Failed steps: ${failed_steps[*]}${NC}"
        exit 1
    else
        echo -e "${GREEN}All steps completed successfully!${NC}"
        exit 0
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi