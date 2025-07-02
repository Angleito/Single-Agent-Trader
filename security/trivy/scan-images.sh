#!/bin/bash

# Docker Image Vulnerability Scanner for AI Trading Bot
# This script scans Docker images for vulnerabilities, generates SBOMs, and creates detailed reports

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/security/trivy/reports"
CONFIG_FILE="$SCRIPT_DIR/trivy-config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default values
SCAN_ALL=false
GENERATE_SBOM=true
OUTPUT_FORMAT="table"
SEVERITY="CRITICAL,HIGH,MEDIUM"
EXIT_ON_VULN=false
SAVE_REPORTS=true
UPLOAD_REPORTS=false
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
    
    mkdir -p "$REPORTS_DIR"/{images,sbom,json,html,sarif}
    mkdir -p "$REPORTS_DIR/archive/$TIMESTAMP"
    
    log_success "Report directories created"
}

# Function to get trading bot images
get_trading_bot_images() {
    local images=()
    
    # Get images from docker-compose
    if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        # Extract image names from docker-compose.yml
        while IFS= read -r line; do
            if [[ $line =~ image:.*ai-trading-bot ]]; then
                image=$(echo "$line" | sed 's/.*image: *//g' | sed 's/["'"'"']//g')
                images+=("$image")
            fi
        done < "$PROJECT_ROOT/docker-compose.yml"
    fi
    
    # Add common trading bot images
    images+=(
        "ai-trading-bot:latest"
        "ai-trading-bot:bluefin-latest"
        "ai-trading-bot:coinbase-latest"
        "bluefin-sdk-service:latest"
        "mcp-memory-server:latest"
        "mcp-omnisearch-server:latest"
    )
    
    # Check which images actually exist
    local existing_images=()
    for image in "${images[@]}"; do
        if docker image inspect "$image" >/dev/null 2>&1; then
            existing_images+=("$image")
        fi
    done
    
    # If no trading bot images found, get all local images
    if [[ ${#existing_images[@]} -eq 0 ]]; then
        log_warning "No trading bot images found, scanning all local images"
        mapfile -t existing_images < <(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>")
    fi
    
    printf '%s\n' "${existing_images[@]}"
}

# Function to scan single image
scan_image() {
    local image="$1"
    local image_safe="${image//[\/:]/_}"
    
    log_scan "Scanning image: $image"
    
    # Create report files
    local base_report="$REPORTS_DIR/images/${image_safe}_${TIMESTAMP}"
    local json_report="$REPORTS_DIR/json/${image_safe}_${TIMESTAMP}.json"
    local html_report="$REPORTS_DIR/html/${image_safe}_${TIMESTAMP}.html"
    local sarif_report="$REPORTS_DIR/sarif/${image_safe}_${TIMESTAMP}.sarif"
    local sbom_report="$REPORTS_DIR/sbom/${image_safe}_${TIMESTAMP}.json"
    
    # Basic vulnerability scan
    log_info "Running vulnerability scan for $image..."
    trivy image \
        --config "$CONFIG_FILE" \
        --format table \
        --severity "$SEVERITY" \
        --output "${base_report}.txt" \
        "$image" || true
    
    # JSON format for processing
    log_info "Generating JSON report for $image..."
    trivy image \
        --config "$CONFIG_FILE" \
        --format json \
        --severity "$SEVERITY" \
        --output "$json_report" \
        "$image" || true
    
    # HTML format for viewing
    log_info "Generating HTML report for $image..."
    trivy image \
        --config "$CONFIG_FILE" \
        --format template \
        --template "@contrib/html.tpl" \
        --severity "$SEVERITY" \
        --output "$html_report" \
        "$image" || true
    
    # SARIF format for CI/CD integration
    log_info "Generating SARIF report for $image..."
    trivy image \
        --config "$CONFIG_FILE" \
        --format sarif \
        --severity "$SEVERITY" \
        --output "$sarif_report" \
        "$image" || true
    
    # Generate SBOM
    if [[ "$GENERATE_SBOM" == true ]]; then
        log_info "Generating SBOM for $image..."
        trivy image \
            --format cyclonedx \
            --output "$sbom_report" \
            "$image" || true
    fi
    
    # Check for secrets
    log_info "Scanning for secrets in $image..."
    trivy image \
        --config "$CONFIG_FILE" \
        --scanners secret \
        --format table \
        --output "${base_report}_secrets.txt" \
        "$image" || true
    
    # Configuration scan
    log_info "Scanning configurations in $image..."
    trivy image \
        --config "$CONFIG_FILE" \
        --scanners config \
        --format table \
        --output "${base_report}_config.txt" \
        "$image" || true
    
    # License scan
    log_info "Scanning licenses in $image..."
    trivy image \
        --config "$CONFIG_FILE" \
        --scanners license \
        --format table \
        --output "${base_report}_licenses.txt" \
        "$image" || true
    
    log_success "Scan completed for $image"
}

# Function to generate summary report
generate_summary() {
    log_info "Generating summary report..."
    
    local summary_file="$REPORTS_DIR/summary_${TIMESTAMP}.md"
    
    cat > "$summary_file" <<EOF
# Docker Image Vulnerability Scan Summary

**Scan Date**: $(date)
**Scan ID**: $TIMESTAMP

## Scanned Images

EOF
    
    # Count vulnerabilities by severity
    local total_critical=0
    local total_high=0
    local total_medium=0
    local total_low=0
    
    for json_file in "$REPORTS_DIR/json"/*_${TIMESTAMP}.json; do
        if [[ -f "$json_file" ]]; then
            local image_name=$(basename "$json_file" _${TIMESTAMP}.json)
            image_name=${image_name//_/\/}
            
            echo "### $image_name" >> "$summary_file"
            echo "" >> "$summary_file"
            
            # Parse JSON for vulnerability counts
            if command -v jq >/dev/null 2>&1; then
                local critical=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL") | .VulnerabilityID' "$json_file" 2>/dev/null | wc -l)
                local high=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH") | .VulnerabilityID' "$json_file" 2>/dev/null | wc -l)
                local medium=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="MEDIUM") | .VulnerabilityID' "$json_file" 2>/dev/null | wc -l)
                local low=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="LOW") | .VulnerabilityID' "$json_file" 2>/dev/null | wc -l)
                
                echo "- **Critical**: $critical" >> "$summary_file"
                echo "- **High**: $high" >> "$summary_file"
                echo "- **Medium**: $medium" >> "$summary_file"
                echo "- **Low**: $low" >> "$summary_file"
                echo "" >> "$summary_file"
                
                total_critical=$((total_critical + critical))
                total_high=$((total_high + high))
                total_medium=$((total_medium + medium))
                total_low=$((total_low + low))
            else
                echo "- Install 'jq' for detailed vulnerability counts" >> "$summary_file"
                echo "" >> "$summary_file"
            fi
        fi
    done
    
    # Add total summary
    cat >> "$summary_file" <<EOF

## Total Vulnerabilities

- **Critical**: $total_critical
- **High**: $total_high
- **Medium**: $total_medium
- **Low**: $total_low

## Recommendations

### Immediate Actions (Critical/High)
1. Update base images to latest versions
2. Update package dependencies
3. Review and rotate any exposed secrets
4. Apply security patches

### Medium Priority
1. Review container configurations
2. Implement security policies
3. Add runtime security monitoring

### Reports Location
- Text reports: \`security/trivy/reports/images/\`
- JSON reports: \`security/trivy/reports/json/\`
- HTML reports: \`security/trivy/reports/html/\`
- SBOM reports: \`security/trivy/reports/sbom/\`

## Next Steps
1. Review detailed reports for each image
2. Prioritize fixes based on severity and exploitability
3. Implement automated scanning in CI/CD pipeline
4. Set up monitoring for new vulnerabilities

EOF
    
    log_success "Summary report generated: $summary_file"
    
    # Display summary
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo ""
        log_report "=== SCAN SUMMARY ==="
        echo "Total Critical: $total_critical"
        echo "Total High: $total_high"  
        echo "Total Medium: $total_medium"
        echo "Total Low: $total_low"
        echo ""
        
        if [[ $total_critical -gt 0 || $total_high -gt 0 ]]; then
            log_error "Critical or high severity vulnerabilities found!"
            if [[ "$EXIT_ON_VULN" == true ]]; then
                exit 1
            fi
        else
            log_success "No critical or high severity vulnerabilities found"
        fi
    fi
}

# Function to send notifications
send_notification() {
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        log_info "Sending notification to Slack..."
        
        local message="Docker image scan completed for AI Trading Bot. Check reports in security/trivy/reports/"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK" || log_warning "Failed to send Slack notification"
    fi
}

# Function to archive reports
archive_reports() {
    if [[ "$SAVE_REPORTS" == true ]]; then
        log_info "Archiving reports..."
        
        cp -r "$REPORTS_DIR/images"/*_${TIMESTAMP}* "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        cp -r "$REPORTS_DIR/json"/*_${TIMESTAMP}* "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        cp -r "$REPORTS_DIR/html"/*_${TIMESTAMP}* "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        cp -r "$REPORTS_DIR/sbom"/*_${TIMESTAMP}* "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        cp "$REPORTS_DIR/summary_${TIMESTAMP}.md" "$REPORTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
        
        log_success "Reports archived to: $REPORTS_DIR/archive/$TIMESTAMP/"
    fi
}

# Function to display help
show_help() {
    cat <<EOF
Docker Image Vulnerability Scanner for AI Trading Bot

Usage: $0 [OPTIONS] [IMAGE...]

Options:
    --all               Scan all trading bot images
    --format FORMAT     Output format: table, json, sarif (default: table)
    --severity LEVELS   Severity levels: CRITICAL,HIGH,MEDIUM,LOW (default: CRITICAL,HIGH,MEDIUM)
    --no-sbom          Skip SBOM generation
    --exit-on-vuln     Exit with error code if vulnerabilities found
    --no-save          Don't save reports to files
    --slack-webhook URL Send notifications to Slack webhook
    --help, -h         Show this help message

Examples:
    $0                                    # Scan all trading bot images
    $0 --all                             # Scan all trading bot images
    $0 ai-trading-bot:latest             # Scan specific image
    $0 --severity CRITICAL,HIGH          # Scan only for critical and high vulnerabilities
    $0 --format json --exit-on-vuln      # Generate JSON output and exit on vulnerabilities
    $0 --slack-webhook https://hooks.slack.com/... # Send notifications

The script will:
1. Scan Docker images for vulnerabilities
2. Generate Software Bill of Materials (SBOM)
3. Check for secrets and misconfigurations
4. Create detailed reports in multiple formats
5. Generate summary with remediation recommendations

Reports are saved to:
- Text: security/trivy/reports/images/
- JSON: security/trivy/reports/json/
- HTML: security/trivy/reports/html/
- SBOM: security/trivy/reports/sbom/
EOF
}

# Main function
main() {
    log_info "Starting Docker image vulnerability scan..."
    
    # Parse arguments
    local images_to_scan=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                SCAN_ALL=true
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
            --no-sbom)
                GENERATE_SBOM=false
                shift
                ;;
            --exit-on-vuln)
                EXIT_ON_VULN=true
                shift
                ;;
            --no-save)
                SAVE_REPORTS=false
                shift
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                images_to_scan+=("$1")
                shift
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
    
    # Get images to scan
    if [[ ${#images_to_scan[@]} -eq 0 ]] || [[ "$SCAN_ALL" == true ]]; then
        log_info "Getting trading bot images..."
        mapfile -t images_to_scan < <(get_trading_bot_images)
    fi
    
    if [[ ${#images_to_scan[@]} -eq 0 ]]; then
        log_error "No images found to scan"
        exit 1
    fi
    
    log_info "Found ${#images_to_scan[@]} images to scan"
    
    # Update Trivy database
    log_info "Updating Trivy database..."
    trivy image --download-db-only || log_warning "Failed to update database"
    
    # Scan each image
    for image in "${images_to_scan[@]}"; do
        scan_image "$image"
    done
    
    # Generate summary
    generate_summary
    
    # Archive reports
    archive_reports
    
    # Send notifications
    send_notification
    
    log_success "Docker image vulnerability scan completed!"
    log_info "Reports saved to: $REPORTS_DIR"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi