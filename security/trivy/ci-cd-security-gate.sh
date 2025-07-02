#!/bin/bash

# CI/CD Security Gate with Trivy Integration for AI Trading Bot
# This script provides automated security scanning in CI/CD pipelines with configurable gates

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/security/trivy/reports/ci"
CONFIG_FILE="$SCRIPT_DIR/trivy-config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# CI/CD Environment Detection
CI_SYSTEM="unknown"
BRANCH_NAME="${GITHUB_REF_NAME:-${GITLAB_CI_COMMIT_REF_NAME:-${BRANCH_NAME:-unknown}}}"
COMMIT_SHA="${GITHUB_SHA:-${CI_COMMIT_SHA:-${COMMIT_SHA:-unknown}}}"
BUILD_NUMBER="${GITHUB_RUN_NUMBER:-${CI_PIPELINE_ID:-${BUILD_NUMBER:-unknown}}}"

# Security Gate Configuration
FAIL_ON_CRITICAL=true
FAIL_ON_HIGH=true
FAIL_ON_SECRETS=true
FAIL_ON_MISCONFIG=false
MAX_CRITICAL=0
MAX_HIGH=5
MAX_MEDIUM=20
SCAN_TIMEOUT=1200  # 20 minutes

# Notification settings
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
TEAMS_WEBHOOK="${TEAMS_WEBHOOK:-}"
EMAIL_NOTIFICATIONS="${EMAIL_NOTIFICATIONS:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$REPORTS_DIR/ci_log_${TIMESTAMP}.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$REPORTS_DIR/ci_log_${TIMESTAMP}.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$REPORTS_DIR/ci_log_${TIMESTAMP}.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$REPORTS_DIR/ci_log_${TIMESTAMP}.log"
}

log_gate() {
    echo -e "${PURPLE}[GATE]${NC} $1" | tee -a "$REPORTS_DIR/ci_log_${TIMESTAMP}.log"
}

# Function to detect CI system
detect_ci_system() {
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        CI_SYSTEM="github-actions"
    elif [[ -n "${GITLAB_CI:-}" ]]; then
        CI_SYSTEM="gitlab-ci"
    elif [[ -n "${JENKINS_URL:-}" ]]; then
        CI_SYSTEM="jenkins"
    elif [[ -n "${CIRCLECI:-}" ]]; then
        CI_SYSTEM="circleci"
    elif [[ -n "${TRAVIS:-}" ]]; then
        CI_SYSTEM="travis"
    elif [[ -n "${AZURE_PIPELINES:-}" ]]; then
        CI_SYSTEM="azure-pipelines"
    fi

    log_info "Detected CI system: $CI_SYSTEM"
}

# Function to setup directories
setup_directories() {
    log_info "Setting up CI/CD report directories..."

    mkdir -p "$REPORTS_DIR"/{images,filesystem,sarif,badges,artifacts}
    mkdir -p "$REPORTS_DIR/archive/$TIMESTAMP"

    log_success "CI/CD directories created"
}

# Function to run security scans
run_security_scans() {
    log_info "Running comprehensive security scans..."

    local scan_failed=false

    # Image scanning
    if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "ai-trading-bot"; then
        log_info "Running Docker image security scan..."
        if ! timeout $SCAN_TIMEOUT "$SCRIPT_DIR/scan-images.sh" \
            --format sarif \
            --severity CRITICAL,HIGH,MEDIUM \
            --exit-on-vuln; then
            scan_failed=true
            log_error "Docker image scan failed or found vulnerabilities"
        fi
    else
        log_warning "No Docker images found to scan"
    fi

    # Filesystem scanning
    log_info "Running filesystem security scan..."
    if ! timeout $SCAN_TIMEOUT "$SCRIPT_DIR/scan-filesystem.sh" \
        --format sarif \
        --severity CRITICAL,HIGH,MEDIUM \
        --exit-on-vuln; then
        scan_failed=true
        log_error "Filesystem scan failed or found vulnerabilities"
    fi

    return $([[ "$scan_failed" == true ]] && echo 1 || echo 0)
}

# Function to analyze scan results
analyze_results() {
    log_info "Analyzing security scan results..."

    local critical_count=0
    local high_count=0
    local medium_count=0
    local secret_count=0
    local config_count=0
    local gate_failed=false

    # Count vulnerabilities from JSON reports
    if command -v jq >/dev/null 2>&1; then
        for json_file in "$PROJECT_ROOT/security/trivy/reports/json"/*_${TIMESTAMP}*.json; do
            if [[ -f "$json_file" ]]; then
                local file_critical=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL") | .VulnerabilityID' "$json_file" 2>/dev/null | wc -l)
                local file_high=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH") | .VulnerabilityID' "$json_file" 2>/dev/null | wc -l)
                local file_medium=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="MEDIUM") | .VulnerabilityID' "$json_file" 2>/dev/null | wc -l)
                local file_secrets=$(jq -r '.Results[]?.Secrets[]? | .RuleID' "$json_file" 2>/dev/null | wc -l)
                local file_configs=$(jq -r '.Results[]?.Misconfigurations[]? | .ID' "$json_file" 2>/dev/null | wc -l)

                critical_count=$((critical_count + file_critical))
                high_count=$((high_count + file_high))
                medium_count=$((medium_count + file_medium))
                secret_count=$((secret_count + file_secrets))
                config_count=$((config_count + file_configs))
            fi
        done
    else
        log_warning "jq not available, using text analysis"
        # Fallback to text analysis
        for txt_file in "$PROJECT_ROOT/security/trivy/reports"/*/*.txt; do
            if [[ -f "$txt_file" ]]; then
                critical_count=$((critical_count + $(grep -c "CRITICAL" "$txt_file" 2>/dev/null || echo 0)))
                high_count=$((high_count + $(grep -c "HIGH" "$txt_file" 2>/dev/null || echo 0)))
                medium_count=$((medium_count + $(grep -c "MEDIUM" "$txt_file" 2>/dev/null || echo 0)))
            fi
        done
    fi

    # Generate results summary
    cat > "$REPORTS_DIR/security_gate_results_${TIMESTAMP}.md" <<EOF
# Security Gate Results

**Build**: $BUILD_NUMBER
**Branch**: $BRANCH_NAME
**Commit**: $COMMIT_SHA
**Timestamp**: $(date)

## Vulnerability Summary

| Severity | Count | Threshold | Status |
|----------|-------|-----------|--------|
| Critical | $critical_count | $MAX_CRITICAL | $([ $critical_count -le $MAX_CRITICAL ] && echo "✅ PASS" || echo "❌ FAIL") |
| High     | $high_count | $MAX_HIGH | $([ $high_count -le $MAX_HIGH ] && echo "✅ PASS" || echo "❌ FAIL") |
| Medium   | $medium_count | $MAX_MEDIUM | $([ $medium_count -le $MAX_MEDIUM ] && echo "✅ PASS" || echo "❌ FAIL") |

## Security Issues

| Type | Count | Policy | Status |
|------|-------|--------|--------|
| Secrets | $secret_count | Zero tolerance | $([ $secret_count -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL") |
| Misconfigurations | $config_count | Advisory | $([ $FAIL_ON_MISCONFIG == "false" ] && echo "⚠️ INFO" || ([ $config_count -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")) |

EOF

    # Check security gates
    log_gate "Evaluating security gates..."

    # Critical vulnerabilities gate
    if [[ $critical_count -gt $MAX_CRITICAL ]]; then
        log_error "GATE FAILED: Critical vulnerabilities ($critical_count) exceed threshold ($MAX_CRITICAL)"
        gate_failed=true
    else
        log_success "GATE PASSED: Critical vulnerabilities within threshold"
    fi

    # High vulnerabilities gate
    if [[ $high_count -gt $MAX_HIGH ]]; then
        log_error "GATE FAILED: High vulnerabilities ($high_count) exceed threshold ($MAX_HIGH)"
        gate_failed=true
    else
        log_success "GATE PASSED: High vulnerabilities within threshold"
    fi

    # Secrets gate
    if [[ $secret_count -gt 0 && "$FAIL_ON_SECRETS" == true ]]; then
        log_error "GATE FAILED: Secrets found in codebase ($secret_count)"
        gate_failed=true
    else
        log_success "GATE PASSED: No secrets found"
    fi

    # Misconfigurations gate
    if [[ $config_count -gt 0 && "$FAIL_ON_MISCONFIG" == true ]]; then
        log_error "GATE FAILED: Misconfigurations found ($config_count)"
        gate_failed=true
    else
        log_success "GATE PASSED: Misconfigurations check"
    fi

    # Generate security badge
    generate_security_badge "$critical_count" "$high_count" "$secret_count" "$gate_failed"

    return $([[ "$gate_failed" == true ]] && echo 1 || echo 0)
}

# Function to generate security badge
generate_security_badge() {
    local critical=$1
    local high=$2
    local secrets=$3
    local failed=$4

    local badge_color="brightgreen"
    local badge_message="secure"

    if [[ "$failed" == true ]]; then
        badge_color="red"
        badge_message="vulnerable"
    elif [[ $critical -gt 0 || $secrets -gt 0 ]]; then
        badge_color="red"
        badge_message="critical-issues"
    elif [[ $high -gt 0 ]]; then
        badge_color="orange"
        badge_message="high-issues"
    fi

    # Generate badge URL
    local badge_url="https://img.shields.io/badge/security-${badge_message}-${badge_color}"

    cat > "$REPORTS_DIR/badges/security_badge_${TIMESTAMP}.md" <<EOF
# Security Badge

![Security Status]($badge_url)

**Status**: $badge_message
**Critical**: $critical
**High**: $high
**Secrets**: $secrets
EOF

    log_info "Security badge generated: $badge_message"
}

# Function to integrate with CI system
integrate_ci_system() {
    log_info "Integrating with CI system: $CI_SYSTEM"

    case $CI_SYSTEM in
        "github-actions")
            integrate_github_actions
            ;;
        "gitlab-ci")
            integrate_gitlab_ci
            ;;
        "jenkins")
            integrate_jenkins
            ;;
        *)
            log_info "Generic CI integration"
            ;;
    esac
}

# Function to integrate with GitHub Actions
integrate_github_actions() {
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        log_info "Setting up GitHub Actions integration..."

        # Set outputs
        echo "security_status=$([[ $? -eq 0 ]] && echo 'passed' || echo 'failed')" >> "$GITHUB_OUTPUT"
        echo "critical_count=$critical_count" >> "$GITHUB_OUTPUT"
        echo "high_count=$high_count" >> "$GITHUB_OUTPUT"
        echo "secret_count=$secret_count" >> "$GITHUB_OUTPUT"

        # Upload artifacts
        if [[ -d "$REPORTS_DIR" ]]; then
            echo "artifact_path=$REPORTS_DIR" >> "$GITHUB_OUTPUT"
        fi

        # Set step summary
        if [[ -f "$REPORTS_DIR/security_gate_results_${TIMESTAMP}.md" ]]; then
            cat "$REPORTS_DIR/security_gate_results_${TIMESTAMP}.md" >> "$GITHUB_STEP_SUMMARY"
        fi

        # Upload SARIF results
        find "$PROJECT_ROOT/security/trivy/reports/sarif" -name "*.sarif" -exec cp {} "$REPORTS_DIR/artifacts/" \;
    fi
}

# Function to integrate with GitLab CI
integrate_gitlab_ci() {
    if [[ -n "${GITLAB_CI:-}" ]]; then
        log_info "Setting up GitLab CI integration..."

        # Generate GitLab security report
        cat > "$REPORTS_DIR/gl-sast-report.json" <<EOF
{
  "version": "14.0.0",
  "vulnerabilities": [],
  "scan": {
    "scanner": {
      "id": "trivy",
      "name": "Trivy",
      "version": "$(trivy --version | head -n1 | cut -d' ' -f2)"
    },
    "type": "sast",
    "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "end_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "status": "success"
  }
}
EOF
    fi
}

# Function to integrate with Jenkins
integrate_jenkins() {
    if [[ -n "${JENKINS_URL:-}" ]]; then
        log_info "Setting up Jenkins integration..."

        # Create JUnit XML report for Jenkins
        cat > "$REPORTS_DIR/security-tests.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="Security Gate Tests" tests="4" failures="$([[ $? -eq 0 ]] && echo 0 || echo 1)">
    <testcase name="Critical Vulnerabilities" classname="security.critical">
      $([[ $critical_count -le $MAX_CRITICAL ]] || echo '<failure message="Too many critical vulnerabilities"/>')
    </testcase>
    <testcase name="High Vulnerabilities" classname="security.high">
      $([[ $high_count -le $MAX_HIGH ]] || echo '<failure message="Too many high vulnerabilities"/>')
    </testcase>
    <testcase name="Secrets Check" classname="security.secrets">
      $([[ $secret_count -eq 0 ]] || echo '<failure message="Secrets found in codebase"/>')
    </testcase>
    <testcase name="Configuration Check" classname="security.config">
      $([[ $FAIL_ON_MISCONFIG == "false" || $config_count -eq 0 ]] || echo '<failure message="Misconfigurations found"/>')
    </testcase>
  </testsuite>
</testsuites>
EOF
    fi
}

# Function to send notifications
send_notifications() {
    local status=$1
    local critical=$2
    local high=$3
    local secrets=$4

    local status_emoji
    local status_color

    if [[ "$status" == "passed" ]]; then
        status_emoji="✅"
        status_color="good"
    else
        status_emoji="❌"
        status_color="danger"
    fi

    # Slack notification
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        log_info "Sending Slack notification..."

        local slack_payload=$(cat <<EOF
{
  "attachments": [
    {
      "color": "$status_color",
      "title": "$status_emoji Security Gate - $status",
      "fields": [
        {
          "title": "Branch",
          "value": "$BRANCH_NAME",
          "short": true
        },
        {
          "title": "Build",
          "value": "$BUILD_NUMBER",
          "short": true
        },
        {
          "title": "Critical",
          "value": "$critical",
          "short": true
        },
        {
          "title": "High",
          "value": "$high",
          "short": true
        },
        {
          "title": "Secrets",
          "value": "$secrets",
          "short": true
        }
      ],
      "footer": "AI Trading Bot Security",
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

    # Teams notification
    if [[ -n "$TEAMS_WEBHOOK" ]]; then
        log_info "Sending Teams notification..."

        local teams_payload=$(cat <<EOF
{
  "@type": "MessageCard",
  "@context": "http://schema.org/extensions",
  "themeColor": "$([[ "$status" == "passed" ]] && echo "00FF00" || echo "FF0000")",
  "summary": "Security Gate $status",
  "sections": [{
    "activityTitle": "AI Trading Bot Security Gate",
    "activitySubtitle": "Branch: $BRANCH_NAME | Build: $BUILD_NUMBER",
    "facts": [{
      "name": "Status",
      "value": "$status_emoji $status"
    }, {
      "name": "Critical Vulnerabilities",
      "value": "$critical"
    }, {
      "name": "High Vulnerabilities",
      "value": "$high"
    }, {
      "name": "Secrets Found",
      "value": "$secrets"
    }]
  }]
}
EOF
)

        curl -X POST -H 'Content-type: application/json' \
            --data "$teams_payload" \
            "$TEAMS_WEBHOOK" || log_warning "Failed to send Teams notification"
    fi
}

# Function to cleanup old reports
cleanup_old_reports() {
    log_info "Cleaning up old CI reports..."

    # Keep only last 10 CI runs
    find "$REPORTS_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
    find "$REPORTS_DIR/archive" -type d -mtime +30 -exec rm -rf {} \; 2>/dev/null || true

    log_success "Cleanup completed"
}

# Function to display help
show_help() {
    cat <<EOF
CI/CD Security Gate with Trivy Integration for AI Trading Bot

Usage: $0 [OPTIONS]

Options:
    --max-critical NUM     Maximum critical vulnerabilities allowed (default: 0)
    --max-high NUM         Maximum high vulnerabilities allowed (default: 5)
    --max-medium NUM       Maximum medium vulnerabilities allowed (default: 20)
    --no-fail-critical     Don't fail on critical vulnerabilities
    --no-fail-high         Don't fail on high vulnerabilities
    --no-fail-secrets      Don't fail on secrets
    --fail-misconfig       Fail on misconfigurations
    --timeout SECONDS      Scan timeout in seconds (default: 1200)
    --slack-webhook URL    Slack webhook for notifications
    --teams-webhook URL    Teams webhook for notifications
    --help, -h             Show this help message

Environment Variables:
    GITHUB_ACTIONS         GitHub Actions environment
    GITLAB_CI              GitLab CI environment
    JENKINS_URL            Jenkins environment
    SLACK_WEBHOOK          Slack notification webhook
    TEAMS_WEBHOOK          Teams notification webhook

Examples:
    $0                                    # Run with default settings
    $0 --max-critical 2 --max-high 10   # Allow more vulnerabilities
    $0 --no-fail-secrets                # Don't fail on secrets
    $0 --fail-misconfig                 # Fail on misconfigurations
    $0 --timeout 1800                   # 30-minute timeout

Exit Codes:
    0 - Security gate passed
    1 - Security gate failed
    2 - Scan error or timeout

The script will:
1. Detect CI/CD environment
2. Run comprehensive security scans
3. Analyze results against security policies
4. Generate reports and badges
5. Integrate with CI/CD system
6. Send notifications on status
EOF
}

# Main function
main() {
    log_info "Starting CI/CD security gate..."

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --max-critical)
                MAX_CRITICAL="$2"
                shift 2
                ;;
            --max-high)
                MAX_HIGH="$2"
                shift 2
                ;;
            --max-medium)
                MAX_MEDIUM="$2"
                shift 2
                ;;
            --no-fail-critical)
                FAIL_ON_CRITICAL=false
                shift
                ;;
            --no-fail-high)
                FAIL_ON_HIGH=false
                shift
                ;;
            --no-fail-secrets)
                FAIL_ON_SECRETS=false
                shift
                ;;
            --fail-misconfig)
                FAIL_ON_MISCONFIG=true
                shift
                ;;
            --timeout)
                SCAN_TIMEOUT="$2"
                shift 2
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
                shift 2
                ;;
            --teams-webhook)
                TEAMS_WEBHOOK="$2"
                shift 2
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

    # Check if Trivy is installed
    if ! command -v trivy >/dev/null 2>&1; then
        log_error "Trivy is not installed. Install it first."
        exit 2
    fi

    # Detect CI system
    detect_ci_system

    # Setup directories
    setup_directories

    # Run security scans
    local scan_result=0
    if ! run_security_scans; then
        scan_result=1
    fi

    # Analyze results
    local gate_result=0
    if ! analyze_results; then
        gate_result=1
    fi

    # Integrate with CI system
    integrate_ci_system

    # Send notifications
    local final_status="passed"
    if [[ $scan_result -eq 1 || $gate_result -eq 1 ]]; then
        final_status="failed"
    fi

    send_notifications "$final_status" "$critical_count" "$high_count" "$secret_count"

    # Cleanup
    cleanup_old_reports

    # Final result
    if [[ "$final_status" == "passed" ]]; then
        log_success "Security gate PASSED - Build can proceed"
        exit 0
    else
        log_error "Security gate FAILED - Build should be blocked"
        exit 1
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
