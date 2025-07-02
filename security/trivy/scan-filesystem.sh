#!/bin/bash

# Filesystem and Configuration Vulnerability Scanner for AI Trading Bot
# This script scans the filesystem, configuration files, and source code for vulnerabilities and secrets

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/security/trivy/reports"
CONFIG_FILE="$SCRIPT_DIR/trivy-config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default values
SCAN_SECRETS=true
SCAN_CONFIGS=true
SCAN_LICENSES=true
SCAN_DEPENDENCIES=true
OUTPUT_FORMAT="table"
SEVERITY="CRITICAL,HIGH,MEDIUM"
EXIT_ON_VULN=false
SAVE_REPORTS=true
EXCLUDE_DIRS=".git,node_modules,__pycache__,.pytest_cache,venv,.venv,dist,build"
SLACK_WEBHOOK=""

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

log_scan() {
    echo -e "${PURPLE}[SCAN]${NC} $1"
}

log_report() {
    echo -e "${CYAN}[REPORT]${NC} $1"
}

# Function to setup directories
setup_directories() {
    log_info "Setting up report directories..."
    
    mkdir -p "$REPORTS_DIR"/{filesystem,secrets,configs,licenses,dependencies,json,html,sarif}
    mkdir -p "$REPORTS_DIR/archive/$TIMESTAMP"
    
    log_success "Report directories created"
}

# Function to scan filesystem for vulnerabilities
scan_filesystem_vulnerabilities() {
    log_scan "Scanning filesystem for vulnerabilities..."
    
    local base_report="$REPORTS_DIR/filesystem/filesystem_vulns_${TIMESTAMP}"
    local json_report="$REPORTS_DIR/json/filesystem_vulns_${TIMESTAMP}.json"
    local html_report="$REPORTS_DIR/html/filesystem_vulns_${TIMESTAMP}.html"
    local sarif_report="$REPORTS_DIR/sarif/filesystem_vulns_${TIMESTAMP}.sarif"
    
    # Basic filesystem scan
    log_info "Running filesystem vulnerability scan..."
    trivy fs \
        --config "$CONFIG_FILE" \
        --format table \
        --severity "$SEVERITY" \
        --output "${base_report}.txt" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    # JSON format for processing
    log_info "Generating JSON report for filesystem..."
    trivy fs \
        --config "$CONFIG_FILE" \
        --format json \
        --severity "$SEVERITY" \
        --output "$json_report" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    # HTML format for viewing
    log_info "Generating HTML report for filesystem..."
    trivy fs \
        --config "$CONFIG_FILE" \
        --format template \
        --template "@contrib/html.tpl" \
        --severity "$SEVERITY" \
        --output "$html_report" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    # SARIF format for CI/CD integration
    log_info "Generating SARIF report for filesystem..."
    trivy fs \
        --config "$CONFIG_FILE" \
        --format sarif \
        --severity "$SEVERITY" \
        --output "$sarif_report" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    log_success "Filesystem vulnerability scan completed"
}

# Function to scan for secrets
scan_secrets() {
    if [[ "$SCAN_SECRETS" != true ]]; then
        return
    fi
    
    log_scan "Scanning for secrets in source code..."
    
    local base_report="$REPORTS_DIR/secrets/secrets_${TIMESTAMP}"
    local json_report="$REPORTS_DIR/json/secrets_${TIMESTAMP}.json"
    
    # Scan for secrets
    log_info "Running secret scan..."
    trivy fs \
        --scanners secret \
        --format table \
        --output "${base_report}.txt" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    # JSON format for processing
    trivy fs \
        --scanners secret \
        --format json \
        --output "$json_report" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    # Scan specific high-risk files
    log_info "Scanning high-risk files for secrets..."
    
    # Environment files
    find "$PROJECT_ROOT" -name "*.env*" -not -path "*/.git/*" -not -path "*/node_modules/*" | while read -r env_file; do
        if [[ -f "$env_file" ]]; then
            log_info "Scanning env file: $env_file"
            trivy fs \
                --scanners secret \
                --format table \
                --output "${base_report}_$(basename "$env_file").txt" \
                "$env_file" || true
        fi
    done
    
    # Configuration files
    find "$PROJECT_ROOT" -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" \
        | grep -v -E "(node_modules|\.git|__pycache__|\.pytest_cache)" \
        | while read -r config_file; do
        if [[ -f "$config_file" && $(basename "$config_file") =~ (config|secret|key|credential) ]]; then
            log_info "Scanning config file: $config_file"
            trivy fs \
                --scanners secret \
                --format table \
                --output "${base_report}_$(basename "$config_file" | tr '/' '_').txt" \
                "$config_file" || true
        fi
    done
    
    # Docker files
    find "$PROJECT_ROOT" -name "Dockerfile*" -o -name "docker-compose*.yml" | while read -r docker_file; do
        if [[ -f "$docker_file" ]]; then
            log_info "Scanning Docker file: $docker_file"
            trivy fs \
                --scanners secret \
                --format table \
                --output "${base_report}_$(basename "$docker_file" | tr '/' '_').txt" \
                "$docker_file" || true
        fi
    done
    
    log_success "Secret scan completed"
}

# Function to scan configurations
scan_configurations() {
    if [[ "$SCAN_CONFIGS" != true ]]; then
        return
    fi
    
    log_scan "Scanning configuration files for misconfigurations..."
    
    local base_report="$REPORTS_DIR/configs/configs_${TIMESTAMP}"
    local json_report="$REPORTS_DIR/json/configs_${TIMESTAMP}.json"
    
    # Scan for misconfigurations
    log_info "Running configuration scan..."
    trivy fs \
        --scanners config \
        --format table \
        --output "${base_report}.txt" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    # JSON format for processing
    trivy fs \
        --scanners config \
        --format json \
        --output "$json_report" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    # Specific configuration scans
    log_info "Scanning Docker configurations..."
    
    # Docker-compose files
    find "$PROJECT_ROOT" -name "docker-compose*.yml" | while read -r compose_file; do
        if [[ -f "$compose_file" ]]; then
            log_info "Scanning compose file: $compose_file"
            trivy config \
                --format table \
                --output "${base_report}_$(basename "$compose_file" .yml).txt" \
                "$compose_file" || true
        fi
    done
    
    # Dockerfiles
    find "$PROJECT_ROOT" -name "Dockerfile*" | while read -r dockerfile; do
        if [[ -f "$dockerfile" ]]; then
            log_info "Scanning Dockerfile: $dockerfile"
            trivy config \
                --format table \
                --output "${base_report}_$(basename "$dockerfile").txt" \
                "$dockerfile" || true
        fi
    done
    
    # Kubernetes manifests (if any)
    find "$PROJECT_ROOT" -name "*.yaml" -o -name "*.yml" | grep -E "(k8s|kubernetes|manifest)" | while read -r k8s_file; do
        if [[ -f "$k8s_file" ]]; then
            log_info "Scanning Kubernetes manifest: $k8s_file"
            trivy config \
                --format table \
                --output "${base_report}_$(basename "$k8s_file" | tr '/' '_').txt" \
                "$k8s_file" || true
        fi
    done
    
    # Security configurations
    find "$PROJECT_ROOT/security" -name "*.yaml" -o -name "*.yml" 2>/dev/null | while read -r security_file; do
        if [[ -f "$security_file" ]]; then
            log_info "Scanning security config: $security_file"
            trivy config \
                --format table \
                --output "${base_report}_$(basename "$security_file" | tr '/' '_').txt" \
                "$security_file" || true
        fi
    done
    
    log_success "Configuration scan completed"
}

# Function to scan licenses
scan_licenses() {
    if [[ "$SCAN_LICENSES" != true ]]; then
        return
    fi
    
    log_scan "Scanning for license compliance..."
    
    local base_report="$REPORTS_DIR/licenses/licenses_${TIMESTAMP}"
    local json_report="$REPORTS_DIR/json/licenses_${TIMESTAMP}.json"
    
    # Scan for licenses
    log_info "Running license scan..."
    trivy fs \
        --scanners license \
        --format table \
        --output "${base_report}.txt" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    # JSON format for processing
    trivy fs \
        --scanners license \
        --format json \
        --output "$json_report" \
        --skip-dirs "$EXCLUDE_DIRS" \
        "$PROJECT_ROOT" || true
    
    # Scan specific dependency files
    log_info "Scanning dependency files for licenses..."
    
    # Python dependencies
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log_info "Scanning Python dependencies (pyproject.toml)..."
        trivy fs \
            --scanners license \
            --format table \
            --output "${base_report}_python.txt" \
            "$PROJECT_ROOT/pyproject.toml" || true
    fi
    
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        log_info "Scanning Python dependencies (requirements.txt)..."
        trivy fs \
            --scanners license \
            --format table \
            --output "${base_report}_requirements.txt" \
            "$PROJECT_ROOT/requirements.txt" || true
    fi
    
    # Node.js dependencies
    if [[ -f "$PROJECT_ROOT/package.json" ]]; then
        log_info "Scanning Node.js dependencies..."
        trivy fs \
            --scanners license \
            --format table \
            --output "${base_report}_nodejs.txt" \
            "$PROJECT_ROOT/package.json" || true
    fi
    
    # Dashboard dependencies
    if [[ -f "$PROJECT_ROOT/dashboard/frontend/package.json" ]]; then
        log_info "Scanning dashboard frontend dependencies..."
        trivy fs \
            --scanners license \
            --format table \
            --output "${base_report}_dashboard_frontend.txt" \
            "$PROJECT_ROOT/dashboard/frontend/package.json" || true
    fi
    
    if [[ -f "$PROJECT_ROOT/dashboard/backend/requirements.txt" ]]; then
        log_info "Scanning dashboard backend dependencies..."
        trivy fs \
            --scanners license \
            --format table \
            --output "${base_report}_dashboard_backend.txt" \
            "$PROJECT_ROOT/dashboard/backend/requirements.txt" || true
    fi
    
    log_success "License scan completed"
}

# Function to scan dependencies
scan_dependencies() {
    if [[ "$SCAN_DEPENDENCIES" != true ]]; then
        return
    fi
    
    log_scan "Scanning dependencies for vulnerabilities..."
    
    local base_report="$REPORTS_DIR/dependencies/deps_${TIMESTAMP}"
    local json_report="$REPORTS_DIR/json/deps_${TIMESTAMP}.json"
    
    # Python dependencies
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log_info "Scanning Python dependencies (pyproject.toml)..."
        trivy fs \
            --config "$CONFIG_FILE" \
            --format table \
            --severity "$SEVERITY" \
            --output "${base_report}_python.txt" \
            "$PROJECT_ROOT/pyproject.toml" || true
        
        trivy fs \
            --config "$CONFIG_FILE" \
            --format json \
            --severity "$SEVERITY" \
            --output "${json_report/deps_/deps_python_}" \
            "$PROJECT_ROOT/pyproject.toml" || true
    fi
    
    # Node.js dependencies
    if [[ -f "$PROJECT_ROOT/package.json" ]]; then
        log_info "Scanning Node.js dependencies..."
        trivy fs \
            --config "$CONFIG_FILE" \
            --format table \
            --severity "$SEVERITY" \
            --output "${base_report}_nodejs.txt" \
            "$PROJECT_ROOT/package.json" || true
    fi
    
    # Dashboard dependencies
    if [[ -f "$PROJECT_ROOT/dashboard/frontend/package.json" ]]; then
        log_info "Scanning dashboard frontend dependencies..."
        trivy fs \
            --config "$CONFIG_FILE" \
            --format table \
            --severity "$SEVERITY" \
            --output "${base_report}_dashboard_frontend.txt" \
            "$PROJECT_ROOT/dashboard/frontend" || true
    fi
    
    if [[ -f "$PROJECT_ROOT/dashboard/backend/requirements.txt" ]]; then
        log_info "Scanning dashboard backend dependencies..."
        trivy fs \
            --config "$CONFIG_FILE" \
            --format table \
            --severity "$SEVERITY" \
            --output "${base_report}_dashboard_backend.txt" \
            "$PROJECT_ROOT/dashboard/backend" || true
    fi
    
    log_success "Dependency scan completed"
}

# Function to generate summary report
generate_summary() {
    log_info "Generating filesystem scan summary report..."
    
    local summary_file="$REPORTS_DIR/filesystem_summary_${TIMESTAMP}.md"
    
    cat > "$summary_file" <<EOF
# Filesystem and Configuration Vulnerability Scan Summary

**Scan Date**: $(date)
**Scan ID**: $TIMESTAMP
**Scanned Path**: $PROJECT_ROOT

## Scan Results

EOF
    
    # Count findings by type
    local total_vulns=0
    local total_secrets=0
    local total_configs=0
    local total_licenses=0
    
    # Vulnerability counts
    if [[ -f "$REPORTS_DIR/json/filesystem_vulns_${TIMESTAMP}.json" ]]; then
        echo "### Vulnerabilities" >> "$summary_file"
        if command -v jq >/dev/null 2>&1; then
            local critical=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL") | .VulnerabilityID' "$REPORTS_DIR/json/filesystem_vulns_${TIMESTAMP}.json" 2>/dev/null | wc -l)
            local high=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH") | .VulnerabilityID' "$REPORTS_DIR/json/filesystem_vulns_${TIMESTAMP}.json" 2>/dev/null | wc -l)
            local medium=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="MEDIUM") | .VulnerabilityID' "$REPORTS_DIR/json/filesystem_vulns_${TIMESTAMP}.json" 2>/dev/null | wc -l)
            local low=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="LOW") | .VulnerabilityID' "$REPORTS_DIR/json/filesystem_vulns_${TIMESTAMP}.json" 2>/dev/null | wc -l)
            
            echo "- **Critical**: $critical" >> "$summary_file"
            echo "- **High**: $high" >> "$summary_file"
            echo "- **Medium**: $medium" >> "$summary_file"
            echo "- **Low**: $low" >> "$summary_file"
            
            total_vulns=$((critical + high + medium + low))
        fi
        echo "" >> "$summary_file"
    fi
    
    # Secret counts
    if [[ -f "$REPORTS_DIR/json/secrets_${TIMESTAMP}.json" ]]; then
        echo "### Secrets Found" >> "$summary_file"
        if command -v jq >/dev/null 2>&1; then
            total_secrets=$(jq -r '.Results[]?.Secrets[]? | .RuleID' "$REPORTS_DIR/json/secrets_${TIMESTAMP}.json" 2>/dev/null | wc -l)
            echo "- **Total Secrets**: $total_secrets" >> "$summary_file"
            
            # List secret types
            jq -r '.Results[]?.Secrets[]? | .Category' "$REPORTS_DIR/json/secrets_${TIMESTAMP}.json" 2>/dev/null | sort | uniq -c | while read -r count category; do
                echo "  - $category: $count" >> "$summary_file"
            done 2>/dev/null || true
        fi
        echo "" >> "$summary_file"
    fi
    
    # Configuration issues
    if [[ -f "$REPORTS_DIR/json/configs_${TIMESTAMP}.json" ]]; then
        echo "### Configuration Issues" >> "$summary_file"
        if command -v jq >/dev/null 2>&1; then
            total_configs=$(jq -r '.Results[]?.Misconfigurations[]? | .ID' "$REPORTS_DIR/json/configs_${TIMESTAMP}.json" 2>/dev/null | wc -l)
            echo "- **Total Issues**: $total_configs" >> "$summary_file"
        fi
        echo "" >> "$summary_file"
    fi
    
    # License issues
    if [[ -f "$REPORTS_DIR/json/licenses_${TIMESTAMP}.json" ]]; then
        echo "### License Issues" >> "$summary_file"
        if command -v jq >/dev/null 2>&1; then
            total_licenses=$(jq -r '.Results[]?.Licenses[]? | .Name' "$REPORTS_DIR/json/licenses_${TIMESTAMP}.json" 2>/dev/null | wc -l)
            echo "- **Total Licenses**: $total_licenses" >> "$summary_file"
        fi
        echo "" >> "$summary_file"
    fi
    
    # Add recommendations
    cat >> "$summary_file" <<EOF

## Security Recommendations

### Immediate Actions
1. **Review and rotate any exposed secrets**
2. **Fix critical and high severity vulnerabilities**
3. **Update dependencies to latest secure versions**
4. **Review configuration misconfigurations**

### Security Best Practices
1. **Implement secrets management** (HashiCorp Vault, AWS Secrets Manager)
2. **Enable pre-commit hooks** to prevent secret commits
3. **Regular dependency updates** and vulnerability scanning
4. **Configuration validation** in CI/CD pipeline

### Docker Security
1. **Use minimal base images** (Alpine, distroless)
2. **Run containers as non-root users**
3. **Implement multi-stage builds**
4. **Regular image scanning**

### Monitoring and Alerting
1. **Set up automated scanning** in CI/CD pipeline
2. **Implement security monitoring**
3. **Configure vulnerability alerts**
4. **Regular security audits**

## Report Files

- **Vulnerabilities**: \`security/trivy/reports/filesystem/\`
- **Secrets**: \`security/trivy/reports/secrets/\`
- **Configurations**: \`security/trivy/reports/configs/\`
- **Licenses**: \`security/trivy/reports/licenses/\`
- **Dependencies**: \`security/trivy/reports/dependencies/\`

## Next Steps

1. Prioritize fixes based on severity and exposure
2. Implement automated scanning in development workflow
3. Set up security monitoring and alerting
4. Regular security training for development team

EOF
    
    log_success "Summary report generated: $summary_file"
    
    # Display summary
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo ""
        log_report "=== FILESYSTEM SCAN SUMMARY ==="
        echo "Total Vulnerabilities: $total_vulns"
        echo "Total Secrets: $total_secrets"
        echo "Total Config Issues: $total_configs"
        echo "Total Licenses: $total_licenses"
        echo ""
        
        if [[ $total_secrets -gt 0 ]]; then
            log_error "Secrets found in codebase! Review immediately."
            if [[ "$EXIT_ON_VULN" == true ]]; then
                exit 1
            fi
        fi
        
        if [[ $total_vulns -gt 0 ]]; then
            log_warning "Vulnerabilities found in dependencies."
        fi
        
        if [[ $total_secrets -eq 0 && $total_vulns -eq 0 ]]; then
            log_success "No critical security issues found"
        fi
    fi
}

# Function to send notifications
send_notification() {
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        log_info "Sending notification to Slack..."
        
        local message="Filesystem security scan completed for AI Trading Bot. Check reports in security/trivy/reports/"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK" || log_warning "Failed to send Slack notification"
    fi
}

# Function to archive reports
archive_reports() {
    if [[ "$SAVE_REPORTS" == true ]]; then
        log_info "Archiving reports..."
        
        cp -r "$REPORTS_DIR"/filesystem/*_${TIMESTAMP}* "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        cp -r "$REPORTS_DIR"/secrets/*_${TIMESTAMP}* "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        cp -r "$REPORTS_DIR"/configs/*_${TIMESTAMP}* "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        cp -r "$REPORTS_DIR"/licenses/*_${TIMESTAMP}* "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        cp -r "$REPORTS_DIR"/dependencies/*_${TIMESTAMP}* "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        cp "$REPORTS_DIR/filesystem_summary_${TIMESTAMP}.md" "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        
        log_success "Reports archived to: $REPORTS_DIR/archive/$TIMESTAMP/"
    fi
}

# Function to display help
show_help() {
    cat <<EOF
Filesystem and Configuration Vulnerability Scanner for AI Trading Bot

Usage: $0 [OPTIONS]

Options:
    --no-secrets        Skip secret scanning
    --no-configs        Skip configuration scanning
    --no-licenses       Skip license scanning
    --no-dependencies   Skip dependency scanning
    --format FORMAT     Output format: table, json, sarif (default: table)
    --severity LEVELS   Severity levels: CRITICAL,HIGH,MEDIUM,LOW (default: CRITICAL,HIGH,MEDIUM)
    --exit-on-vuln     Exit with error code if vulnerabilities found
    --no-save          Don't save reports to files
    --exclude-dirs DIRS Comma-separated directories to exclude (default: .git,node_modules,__pycache__)
    --slack-webhook URL Send notifications to Slack webhook
    --help, -h         Show this help message

Examples:
    $0                                      # Full filesystem scan
    $0 --no-licenses                       # Skip license scanning
    $0 --severity CRITICAL,HIGH            # Scan only for critical and high issues
    $0 --format json --exit-on-vuln        # Generate JSON output and exit on vulnerabilities
    $0 --exclude-dirs ".git,venv,dist"     # Custom exclusions

The script will:
1. Scan filesystem for vulnerabilities in dependencies
2. Search for secrets and sensitive data in source code
3. Check configuration files for misconfigurations
4. Analyze license compliance
5. Generate detailed reports and remediation recommendations

Reports are saved to:
- Filesystem: security/trivy/reports/filesystem/
- Secrets: security/trivy/reports/secrets/
- Configs: security/trivy/reports/configs/
- Licenses: security/trivy/reports/licenses/
EOF
}

# Main function
main() {
    log_info "Starting filesystem and configuration vulnerability scan..."
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-secrets)
                SCAN_SECRETS=false
                shift
                ;;
            --no-configs)
                SCAN_CONFIGS=false
                shift
                ;;
            --no-licenses)
                SCAN_LICENSES=false
                shift
                ;;
            --no-dependencies)
                SCAN_DEPENDENCIES=false
                shift
                ;;
            --format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --severity)
                SEVERITY="$2"
                shift 2
                ;;
            --exit-on-vuln)
                EXIT_ON_VULN=true
                shift
                ;;
            --no-save)
                SAVE_REPORTS=false
                shift
                ;;
            --exclude-dirs)
                EXCLUDE_DIRS="$2"
                shift 2
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
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
        log_error "Trivy is not installed. Run ./install-trivy.sh first."
        exit 1
    fi
    
    # Setup directories
    setup_directories
    
    # Update Trivy database
    log_info "Updating Trivy database..."
    trivy image --download-db-only || log_warning "Failed to update database"
    
    # Run scans
    scan_filesystem_vulnerabilities
    scan_secrets
    scan_configurations
    scan_licenses
    scan_dependencies
    
    # Generate summary
    generate_summary
    
    # Archive reports
    archive_reports
    
    # Send notifications
    send_notification
    
    log_success "Filesystem vulnerability scan completed!"
    log_info "Reports saved to: $REPORTS_DIR"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi